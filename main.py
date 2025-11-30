import os
import json
import requests
import tempfile
import re
import pytesseract
from pytesseract import Output
import subprocess
import gc
import asyncio
import time
import math
import uuid
import numpy as np
import cv2
import torch
import json_repair
import io

from typing import List, Optional, Any, Dict, Tuple
from decimal import Decimal, ROUND_HALF_UP
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageOps
from openai import AsyncOpenAI

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SAVE_DEBUG_IMAGES = True 
DEBUG_DIR = "debug_processed_images"

if SAVE_DEBUG_IMAGES:
    os.makedirs(DEBUG_DIR, exist_ok=True)

# --- TUNING ---
TARGET_IMG_SIZE = 2200 
OCR_BATCH_SIZE = 4     
PDF_DPI = 250           
PREPROCESS_WORKERS = 4  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from deskew import determine_skew
from skimage.transform import rotate

# --- 1. SETUP GOOGLE GEMINI ---
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

client = AsyncOpenAI(
    base_url="http://172.18.40.139:8000/v1",
    api_key="my-local-secret"
)

# --- 2. LOAD SURYA ---
print("🚀 Loading Surya OCR predictors...")
foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)
detection_predictor = DetectionPredictor()
print("✅ Surya OCR Ready.")

# --- 3. LOAD UNITABLE ---
UNITABLE_ENABLED = False
UNITABLE = None
UniTable = None  

try:
    from unitable.unitable_loader import UniTable
except ImportError:
    try:
        from unitable_loader import UniTable
    except:
        pass

WEIGHTS_PATH = "unitable/experiments/unitable_weights/unitable_large_structure.pt"
VOCAB_PATH = "unitable/vocab/vocab_html.json"

if UniTable: 
    try:
        if os.path.exists(WEIGHTS_PATH) and os.path.exists(VOCAB_PATH):
            gc.collect()
            torch.cuda.empty_cache()
            try:
                print("🚀 Loading UniTable...")
                UNITABLE = UniTable(weights_path=WEIGHTS_PATH, vocab_path=VOCAB_PATH, device=device)
                if device.type == 'cuda':
                    UNITABLE.model.half()
                UNITABLE_ENABLED = True
                print(f"✅ UniTable loaded successfully.")
            except Exception as e:
                print(f"⚠️ UniTable init failed: {e}")
    except Exception as e:
        pass
else:
    print("ℹ️ UniTable not found. Printing processing will lack HTML structure hints.")

app = FastAPI(title="Hybrid Gemini-Surya-UniTable Extractor")

# --- Models ---
class BillItem(BaseModel):
    item_name: str = Field(..., description="Name")
    item_amount: Decimal = Field(..., description="Amount")
    item_rate: Decimal = Field(Decimal("0.00"), description="Rate")
    item_quantity: Decimal = Field(Decimal("0.00"), description="Qty")

    @field_validator("item_amount", "item_rate", "item_quantity", mode="before")
    def parse_decimal(cls, v):
        if v is None or v == "": return Decimal("0.00")
        try: 
            return Decimal(str(v).replace(",", "").replace("$", "").replace("₹", "").strip())
        except: return Decimal("0.00")

class PageData(BaseModel):
    page_no: str
    page_type: str = Field(..., description="Bill Detail | Final Bill | Pharmacy")
    bill_items: List[BillItem]

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class FinalData(BaseModel):
    pagewise_line_items: List[PageData]
    total_item_count: int

class APIResponse(BaseModel):
    is_success: bool
    token_usage: Optional[TokenUsage] = None
    data: Optional[FinalData] = None
    error: Optional[str] = None

class DocumentRequest(BaseModel):
    document: str

# --- 4. CLASSIFIER (Handwritten vs Printed) ---

def classify_image_content(pil_img: Image.Image) -> str:
    try:
        # Resize for speed
        img_small = pil_img.resize((1000, int(1000 * pil_img.height / pil_img.width)))
        data = pytesseract.image_to_data(img_small, config='--psm 6', output_type=Output.DICT)
        confidences = [int(c) for i, c in enumerate(data['conf']) if int(c) != -1 and data['text'][i].strip()]
        
        if not confidences: return "handwritten"
        
        avg_conf = sum(confidences) / len(confidences)
        print(f"    📊 OCR Confidence: {avg_conf:.2f}")
        
        if avg_conf > 75: return "printed"
        if avg_conf < 45: return "handwritten"
        
        # Tie Breaker: Line Detection
        cv_img = np.array(img_small)
        if cv_img.shape[-1] == 3: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        line_count = 0 if lines is None else len(lines)
        if line_count > 5: return "printed"
        else: return "handwritten"

    except: return "printed"

# --- 5. PREPROCESSING (SPLIT & OPTIMIZE) ---

def split_side_by_side_bills(pil_img: Image.Image) -> List[Image.Image]:
    try:
        img = np.array(pil_img.convert("L"))
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        height, width = thresh.shape
        start_x = int(width * 0.40); end_x = int(width * 0.60)
        center_strip = thresh[:, start_x:end_x]
        if center_strip.shape[1] <= 0: return [pil_img] 

        vertical_projection = np.sum(center_strip, axis=0)
        density_profile = vertical_projection / (height * 255)
        gap_indices = np.where(density_profile < 0.005)[0] 
        
        if len(gap_indices) == 0: return [pil_img]

        max_gap_width = 0; best_split_index = -1; current_gap_width = 0
        for i in range(len(gap_indices) - 1):
            if gap_indices[i+1] == gap_indices[i] + 1: current_gap_width += 1
            else:
                if current_gap_width > max_gap_width:
                    max_gap_width = current_gap_width; best_split_index = gap_indices[i] - (current_gap_width // 2)
                current_gap_width = 0
        if current_gap_width > max_gap_width:
            max_gap_width = current_gap_width; best_split_index = gap_indices[-1] - (current_gap_width // 2)

        if max_gap_width < 10: return [pil_img]

        split_x = start_x + best_split_index
        
        left_sum = np.sum(thresh[:, 0:split_x]); right_sum = np.sum(thresh[:, split_x:width])
        total_sum = left_sum + right_sum
        
        if total_sum == 0: return [pil_img]
        if (left_sum/total_sum < 0.15) or (right_sum/total_sum < 0.15): return [pil_img]

        print(f"    ✂️ Valid Split at x={split_x}")
        return [pil_img.crop((0, 0, split_x, pil_img.height)), pil_img.crop((split_x, 0, pil_img.width, pil_img.height))]
    except: return [pil_img]

def fast_image_prep(pil_img: Image.Image) -> List[Image.Image]:
    try:
        # Confidence-based Rotation
        try:
            osd_data = pytesseract.image_to_osd(pil_img, output_type=Output.DICT)
            rotation = osd_data['rotate']
            conf = osd_data['orientation_conf']
            if rotation != 0 and rotation != 360 and conf > 5.0:
                pil_img = pil_img.rotate(-rotation, expand=True)
        except: pass

        images_to_process = split_side_by_side_bills(pil_img)

        final_segments = []
        for img in images_to_process:
            cv_img = np.array(img)
            if cv_img.shape[-1] == 3: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            angle = determine_skew(gray)
            if angle and abs(angle) > 0.5:
                cv_img = (rotate(cv_img, angle, resize=True, cval=1.0) * 255).astype(np.uint8)

            h_s, w_s = cv_img.shape[:2]
            if max(h_s, w_s) < TARGET_IMG_SIZE:
                scale = TARGET_IMG_SIZE / max(h_s, w_s)
                cv_img = cv2.resize(cv_img, (int(w_s * scale), int(h_s * scale)), interpolation=cv2.INTER_LANCZOS4)
            
            final_segments.append(Image.fromarray(cv_img))
        return final_segments
    except: return [pil_img]
    
def apply_enhancement_for_printed_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Only applies to PRINTED bills going to Surya/Tesseract.
    Applies moderate CLAHE and gentle darkening.
    """
    try:
        cv_img = np.array(pil_img)
        if cv_img.ndim == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Moderate CLAHE (Pop the text, but don't burn it)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Gentle Darkening (alpha=1.2, beta=-30)
            cl = cv2.convertScaleAbs(cl, alpha=1.2, beta=-30)
            
            limg = cv2.merge((cl, a, b))
            cv_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        else:
            # Grayscale fallback
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enh = clahe.apply(cv_img)
            img_enh = cv2.convertScaleAbs(img_enh, alpha=1.2, beta=-30)
            return Image.fromarray(img_enh)
    except:
        return pil_img

def batch_smart_deskew(images: List[Image.Image]) -> List[Image.Image]:
    if not images: return []
    deskewed_images = []
    try:
        for i in range(0, len(images), OCR_BATCH_SIZE):
            batch = images[i : i + OCR_BATCH_SIZE]
            predictions = detection_predictor(batch) 
            for j, pred in enumerate(predictions):
                angles = []
                for polygon in pred.bboxes:
                    poly = polygon.polygon 
                    x1, y1 = poly[0]; x2, y2 = poly[1]
                    if x2 - x1 == 0: continue
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    angles.append(angle)
                img_to_rotate = batch[j]
                if angles:
                    median_angle = np.median(angles)
                    if 0.1 < abs(median_angle) < 45:
                        img_to_rotate = img_to_rotate.rotate(median_angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255))
                deskewed_images.append(img_to_rotate)
        return deskewed_images
    except: return images

def process_images_parallel(file_path: str) -> List[Tuple[int, Image.Image]]:
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf": original_images = convert_from_path(file_path, fmt="jpeg", dpi=PDF_DPI)
    else: original_images = [Image.open(file_path).convert("RGB")]

    all_segments_flat = [] 
    segment_map = []       

    print("    Running Parallel CPU Preprocessing...")
    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
        results = list(executor.map(fast_image_prep, original_images))
        for pg_idx, segments in enumerate(results):
            for seg in segments:
                all_segments_flat.append(seg)
                segment_map.append(pg_idx + 1)

    print(f"    Running GPU Batch Deskew on {len(all_segments_flat)} segments...")
    final_images = batch_smart_deskew(all_segments_flat)
    
    mapped_output = []
    for i, img in enumerate(final_images):
        mapped_output.append((segment_map[i], img))
    return mapped_output

# --- 6. OCR + UNITABLE (Printed) ---

def run_unitable_optimized(pil_image: Image.Image) -> str:
    """
    SPEED OPTIMIZATION:
    UniTable detects STRUCTURE. It does not need 2200px images. 
    Resizing to 1024px significantly speeds up the Transformer inference 
    without losing table layout information.
    """
    if not UNITABLE_ENABLED: return ""
    try: 
        # Create a smaller copy for UniTable (fast inference)
        img_fast = pil_image.copy()
        img_fast.thumbnail((1024, 1024))
        
        return UNITABLE.run(img_fast)
    except: return ""

def smart_linearize(text_lines: List[Any]) -> str:
    if not text_lines: return ""
    lines = sorted(text_lines, key=lambda x: x.bbox[1])
    text_lines_out = []
    if lines:
        curr_y = lines[0].bbox[1]; curr_row = []
        for line in lines:
            if abs(line.bbox[1] - curr_y) < 15: curr_row.append(line)
            else:
                curr_row.sort(key=lambda x: x.bbox[0])
                text_lines_out.append(" | ".join([l.text for l in curr_row]))
                curr_row = [line]; curr_y = line.bbox[1]
        curr_row.sort(key=lambda x: x.bbox[0])
        text_lines_out.append(" | ".join([l.text for l in curr_row]))
    return "\n".join(text_lines_out)

def run_ocr_batch_with_unitable(images: List[Image.Image], start_idx: int, request_id: str) -> List[Tuple[str, str]]:
    try:
        # 1. Run Surya OCR (GPU Batch - Fast)
        predictions = recognition_predictor(images, det_predictor=detection_predictor)
        outputs = []
        
        for idx, pred in enumerate(predictions):
            surya_text = smart_linearize(pred.text_lines)
            
            # 2. Run UniTable (CPU/GPU Sequential)
            # Optimized by resizing input image inside `run_unitable_optimized`
            unitable_hint = ""
            if UNITABLE_ENABLED:
                try: 
    
                    
                    unitable_hint = run_unitable_optimized(images[idx])
                except Exception as e: 
                    print(f"UniTable Error: {e}")
            
            outputs.append((surya_text, unitable_hint))
            
        return outputs
    except Exception as e:
        print(f"OCR Batch Error: {e}")
        return [("", "")] * len(images)

async def process_page_llm_printed(ocr_text: str, unitable_hint: str, page_idx: int) -> tuple:
    structured_hint = ""
    if unitable_hint:
        structured_hint = f"\nTABLE_STRUCTURE (HTML Hint):\n'''{unitable_hint}'''\n"
        
    prompt = f"""
    You are an expert Data Entry Clerk. Analyze the OCR text from Segment {page_idx}.
    OCR TEXT: '''{ocr_text}'''
    {structured_hint}
    
    TASK 1: CLASSIFY PAGE TYPE
    - "Pharmacy": If drug names, batch nos, doctor name found.
    - "Final Bill": If "Grand Total", "Room Charges", "Discharge Summary" found.
    - "Bill Detail": Standard itemized invoice.
    
    TASK 2: EXTRACT ITEMS
    - Ignore Cgst, Sgst, Total lines.
    - **Missing Columns**: If Rate or Quantity is not explicitly written, **OUTPUT 0.0**.
    
    RETURN JSON ONLY: 
    {{ 
      "page_type": "Bill Detail", 
      "bill_items": [ {{ "item_name": "Desc", "item_amount": 100.00, "item_rate": 0.0, "item_quantity": 0.0 }} ] 
    }}
    """
    usage_stats = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
    try:
        response = await client.chat.completions.create(
            model="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=4096,
        )
        if response.usage:
            usage_stats["total_tokens"] = response.usage.total_tokens
            usage_stats["input_tokens"] = response.usage.prompt_tokens
            usage_stats["output_tokens"] = response.usage.completion_tokens
        raw = response.choices[0].message.content
        data = json_repair.loads(raw)
        return (page_idx, data if isinstance(data, dict) else {}, usage_stats)
    except: return (page_idx, {}, usage_stats)

# --- 7. GEMINI (Handwritten) ---

async def process_with_gemini(pil_img: Image.Image, page_idx: int) -> tuple:
    print(f"    ✨ Sending Page {page_idx} (Handwritten) to Gemini...")
    prompt = """
    You are an expert Pharmacist. Extract items from this HANDWRITTEN bill into JSON.
    
    TASK 1: CLASSIFY PAGE TYPE
    - "Bill Detail": Standard itemized invoice.
    - "Pharmacy": If drug names, batch nos, doctor name found.
    - "Final Bill": If "Grand Total", "Room Charges", "Discharge Summary" found.
    
    
    TASK 2: EXTRACT ITEMS
    1. Autocorrect drug names (e.g. Shekcl->Shelcal). 
    2. **Missing Data**: If 'Rate' or 'Quantity' is NOT visible, set them to **0.0**.
    3. Ignore Totals/Signatures.
    4. Use your best vision abilities to get quantity, rate, amount, name
    
    Output: 
    { 
      "page_type": "Pharmacy",
      "bill_items": [ { "item_name": "Drug", "item_amount": 0.0, "item_rate": 0.0, "item_quantity": 0.0 } ] 
    }
    """
    try:
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        response = await asyncio.to_thread(gemini_model.generate_content, [prompt, {'mime_type': 'image/jpeg', 'data': img_byte_arr.getvalue()}])
        data = json_repair.loads(response.text)
        return (page_idx, data if isinstance(data, dict) else {}, {"total_tokens":0,"input_tokens":0,"output_tokens":0})
    except: return (page_idx, {}, {"total_tokens":0,"input_tokens":0,"output_tokens":0})

def normalize_llm_item(raw_item: Any) -> Optional[Dict]:
    try:
        item: Dict = {}
        if isinstance(raw_item, dict): item = raw_item.copy()
        elif isinstance(raw_item, list) and all(isinstance(x, dict) for x in raw_item):
             for d in raw_item: item.update(d)
        else: return None
        new_item = {}
        if "item_name" in item and item["item_name"]: new_item["item_name"] = str(item["item_name"]).strip()
        else: return None
        def clean_float(val):
            try: return float(str(val).replace(',', '').replace(' ', ''))
            except: return 0.0
        amount = clean_float(item.get("item_amount", 0.0))
        rate = clean_float(item.get("item_rate", 0.0))
        qty = clean_float(item.get("item_quantity", 0.0))
        
        # STRICT: No calculation logic.
        if rate == 0 and qty == 0:
            pass # Do not infer math.
            
        # 2. If Rate exists but Qty is missing/zero -> Calculate Qty
        elif amount > 0 and rate > 0:
            calculated_qty = amount / rate
            # Round to integer
            qty = int(Decimal(calculated_qty).quantize(Decimal("1.0"), rounding=ROUND_HALF_UP))
            
        new_item["item_amount"] = amount
        new_item["item_rate"] = rate
        new_item["item_quantity"] = int(qty)
        
        return new_item
    except: return None

# --- MAIN ---

@app.post("/extract-bill-data")
async def extract_bill_data(request: DocumentRequest):
    print(f"\n📄 Processing: {request.document}")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_file_path = download_file(request.document)
            unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
            print("🔹 Phase 1: Parallel Preprocessing...")
            mapped_data = await asyncio.to_thread(process_images_parallel, temp_file_path)
            
            printed_batch = []; handwritten_tasks = [] 
            
            for idx, (orig_pg, img) in enumerate(mapped_data):
                doc_type = classify_image_content(img)
                if doc_type == "handwritten":
                    print(f"    ⚠️ Seg {idx+1}: HANDWRITTEN -> Gemini.")
                    handwritten_tasks.append((idx, img))
                else:
                    print(f"    🖨️ Seg {idx+1}: PRINTED -> Surya+UniTable.")
                    enhanced_img = apply_enhancement_for_printed_ocr(img)
                    printed_batch.append((idx, enhanced_img))
            
            if SAVE_DEBUG_IMAGES:
                for idx, (orig_pg, img) in enumerate(mapped_data):
                    try: img.save(os.path.join(DEBUG_DIR, f"{unique_id}_seg_{idx}.jpg"))
                    except: pass
            
            llm_tasks = []
            
            # A. Printed (Batch)
            if printed_batch:
                p_imgs = [x[1] for x in printed_batch]; p_idxs = [x[0] for x in printed_batch]
                for i in range(0, len(p_imgs), OCR_BATCH_SIZE):
                    batch_imgs = p_imgs[i: i + OCR_BATCH_SIZE]
                    batch_indices = p_idxs[i: i + OCR_BATCH_SIZE]
                    
                    # Run OCR + Optimized UniTable
                    ocr_results = await asyncio.to_thread(run_ocr_batch_with_unitable, batch_imgs, i, unique_id)
                    
                    for k, (text, hint) in enumerate(ocr_results):
                        true_idx = batch_indices[k]
                        task = asyncio.create_task(process_page_llm_printed(text, hint, true_idx+1))
                        llm_tasks.append(task)
                    gc.collect(); torch.cuda.empty_cache()

            # B. Handwritten (Gemini)
            for idx, img in handwritten_tasks:
                task = asyncio.create_task(process_with_gemini(img, idx+1))
                llm_tasks.append(task)

            print(f"🔹 Waiting for {len(llm_tasks)} tasks...")
            results = await asyncio.gather(*llm_tasks)
            results.sort(key=lambda x: x[0]) 
            
            grouped_items = {}; page_types = {}; agg_tokens = {"total_tokens":0,"input_tokens":0,"output_tokens":0}
            original_page_nums = [x[0] for x in mapped_data]

            for idx, data, usage in results:
                agg_tokens["total_tokens"] += usage.get("total_tokens", 0)
                agg_tokens["input_tokens"] += usage.get("input_tokens", 0)
                agg_tokens["output_tokens"] += usage.get("output_tokens", 0)
                
                # Identify Original Page
                l_idx = idx - 1
                if 0 <= l_idx < len(original_page_nums): r_pg = str(original_page_nums[l_idx])
                else: r_pg = "unknown"
                if r_pg not in grouped_items: grouped_items[r_pg] = []

                # Extract Items
                items = []
                raw = data.get("bill_items", []) if isinstance(data, dict) else []
                if isinstance(raw, list):
                    for it in raw:
                        norm = normalize_llm_item(it)
                        if norm: items.append(BillItem(**norm))
                
                # Extract Page Type (Default to "Bill Detail" if missing)
                p_type = data.get("page_type", "Bill Detail")
                # Priority Logic: If any segment on this page is "Pharmacy", mark page as Pharmacy
                if r_pg not in page_types: page_types[r_pg] = p_type
                elif p_type == "Pharmacy": page_types[r_pg] = "Pharmacy" # Upgrade priority
                elif p_type == "Final Bill" and page_types[r_pg] != "Pharmacy": page_types[r_pg] = "Final Bill"

                print(f"    ✅ Seg {idx} (Pg {r_pg}): {len(items)} items. Type: {p_type}")
                grouped_items[r_pg].extend(items)

            final_pages = []
            for p in sorted(grouped_items.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                final_pages.append(PageData(
                    page_no=p, 
                    page_type=page_types.get(p, "Bill Detail"), 
                    bill_items=grouped_items[p]
                ))

            return APIResponse(is_success=True, token_usage=TokenUsage(**agg_tokens), data=FinalData(pagewise_line_items=final_pages, total_item_count=sum(len(p.bill_items) for p in final_pages)))

        except Exception as e:
            print(f"❌ Error: {e}")
            return APIResponse(is_success=False, error=str(e))
        finally:
            if 'temp_file_path' in locals():
                try: os.remove(temp_file_path)
                except: pass
            gc.collect()

def download_file(url: str) -> str:
    try:
        resp = requests.get(url, stream=True); resp.raise_for_status()
        ext = ".pdf" if ".pdf" in url.lower() else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in resp.iter_content(16384): tmp.write(chunk)
            return tmp.name
    except Exception as e: raise HTTPException(400, f"DL Failed: {e}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Hybrid API Server...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
