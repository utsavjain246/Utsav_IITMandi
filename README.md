# 🧠 Intelligent Hybrid Pharmacy Bill Extractor  
### _A High-Accuracy Multimodal System for Extracting Structured Line Items from Medical & Pharmacy Bills_

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange.svg)
![SuryaOCR](https://img.shields.io/badge/OCR-Surya-red.svg)
![Gemini](https://img.shields.io/badge/Google-Gemini_1.5_Flash-purple)
![UniTable](https://img.shields.io/badge/Model-UniTable_2.0-yellow)
![Qwen](https://img.shields.io/badge/LocalLLM-Qwen_2.5_14B-blueviolet)

---

# 📘 Overview

This project is a **highly accurate, production-grade Pharmacy Bill Extraction API** designed to process scanned invoices, hand-written prescriptions, medical bills, and multi-page PDFs.

It uses a **Hybrid AI Pipeline** that automatically decides:

| Document Type | Processing Path |
|--------------|------------------|
| 📝 Handwritten Bills | → Google Gemini Flash (Vision-Language Model) |
| 🖨️ Printed Bills | → Surya OCR + UniTable + Local LLM (Qwen 2.5) |

The system is **fault-tolerant**, **GPU accelerated**, **highly parallelized**, and uses **strict post-processing validation** rules to guarantee reliable structured output.

---

# 🚀 Key Features

### 🔍 1. Intelligent Document Routing  
Automatically classifies each page as **Handwritten** or **Printed** using:
- OCR Confidence (Tesseract)
- Hough Line Transform (Table/Grid detection)
- Color Saturation (Detect colored paper)
- Text density heuristics

### 🖼️ 2. Advanced Preprocessing Pipeline  
- Auto deskew  
- Auto rotation (OSD)
- Gutter detection → Split side-by-side scanned bills
- CLAHE contrast enhancement  
- Content-area cropping  
- Smart batch processing

### 🤖 3. Hybrid Extraction Engine  
#### **Handwritten Path (Generative Vision)**
- Google Gemini Flash Model Api
- Pharmacist persona prompt  
- Spelling correction for drug names  
- No hallucinated totals

#### **Printed Path (Structured AI)**
- Surya OCR for high-speed text recognition  
- UniTable for HTML structure extraction  
- Qwen 2.5 14B (local) for table-to-JSON transformation

### 🔒 4. Strong Normalization Layer  
- Cleans floats, currency, and inconsistent formatting  
- Rate/Quantity defaults to `0.0` when missing  
- Validated using Pydantic schemas

### 🛡️ 5. API-Ready & Production Optimized  
- FastAPI backend  
- Async IO everywhere  
- CUDA memory-safe  
- Automatic batch inference  
- Supports large PDFs  

---

# 🏗️ System Architecture

```mermaid
flowchart TD
  %% --- STYLING ---
  classDef base fill:#fff,stroke:#333,stroke-width:1px,color:#000;
  classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000;
  classDef ai fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
  classDef gemini fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000;
  classDef finish fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000;

  %% --- INPUT ---
  Start([User Upload]) --> Preprocess[Preprocessing Engine]:::base

  %% --- PREPROCESSING ---
  subgraph Prep [Phase 1: Intelligent Preprocessing]
    direction TB
    Preprocess --> Split{"Gutter Detect"}:::decision
    Split -- Yes --> SplitImg[Split Pages]:::base
    Split -- No --> SingleImg[Keep Page]:::base
    SplitImg & SingleImg --> Deskew["Tesseract Rotate &<br/>Surya Deskew"]:::base
    Deskew --> Classify{"Is Handwritten?"}:::decision
  end

  %% --- PARALLEL PIPELINES ---
  
  %% Left Path: Handwritten
  Classify -- YES --> GemNode["Google Gemini 1.5 Flash<br/>(Vision + Pharmacist Persona)"]:::gemini
  
  %% Right Path: Printed
  Classify -- NO --> Enhance[Contrast Enhancement]:::base
  
  subgraph PrintedStack [Phase 2: Printed Pipeline]
    direction TB
    Enhance --> ParStart(( ))
    ParStart --> Surya["Surya OCR<br/>(Text Extraction)"]:::ai
    ParStart --> Uni["UniTable<br/>(HTML Structure)"]:::ai
    Surya & Uni --> LocalLLM["Local LLM<br/>(Data Structuring)"]:::ai
  end

  %% --- CONVERGENCE ---
  GemNode --> Merge[Data Aggregation]:::base
  LocalLLM --> Merge

  %% --- POST PROCESSING ---
  subgraph Post [Phase 3: Validation]
    direction TB
    Merge --> Logic["Business Logic<br/>(Set Missing Rate/Qty to 0)"]:::base
    Logic --> Schema[Pydantic Validation]:::base
  end

  Schema --> End([Final JSON Output]):::finish

  %% --- LINKS & STYLING ---
  linkStyle default stroke:#333,stroke-width:1.5px;
```

---

# 🔄 Data Flow Journey

```mermaid
sequenceDiagram
    autonumber
    participant User as User/Client
    participant API as FastAPI Server
    participant Pre as Preprocessor (CPU)
    participant Class as Classifier Logic
    participant GPU as GPU Worker (Surya/UniTable)
    participant Gem as Gemini 1.5 Flash
    participant LLM as Local LLM (Qwen)

    Note over User, API: Step 1: Ingestion Phase
    User->>API: POST /extract-bill-data (File URL)
    activate API
    API->>API: Download File & Generate UUID
    
    Note over API, Pre: Step 2: Parallel Preprocessing
    API->>Pre: Send Raw PDF/Image
    activate Pre
    Pre->>Pre: Convert to RGB Images
    Pre->>Pre: Detect Side-by-Side Split (OpenCV)
    Pre->>Pre: Tesseract Orientation Fix
    Pre-->>API: Return Optimized Image Segments
    deactivate Pre

    loop For Each Segment
        Note over API, Class: Step 3: Classification Routing
        API->>Class: Analyze Image Features
        activate Class
        Class->>Class: Check Grid Lines & OCR Confidence
        
        alt is HANDWRITTEN (Low Conf/Colored Paper)
            Class-->>API: Route: Handwritten
            deactivate Class
            
            Note over API, Gem: Step 4A: Generative Extraction
            API->>Gem: Send Raw Image + Pharmacist Prompt
            activate Gem
            Gem-->>API: Return JSON (Autocorrected Drugs)
            deactivate Gem
            
        else is PRINTED (High Conf/Tables)
            Class-->>API: Route: Printed
            
            Note over API, LLM: Step 4B: Structured Extraction
            par Parallel Inference
                API->>GPU: Run Surya OCR (Batch Mode)
                activate GPU
                GPU-->>API: Return Linearized Text
                deactivate GPU
            and Structure Detection
                API->>GPU: Run UniTable (Resized 1024px)
                activate GPU
                GPU-->>API: Return HTML Structure Hint
                deactivate GPU
            end
            
            API->>LLM: Send Text + HTML Hint
            activate LLM
            LLM-->>API: Return Structured JSON
            deactivate LLM
        end
    end

    Note over API, User: Step 5: Final Aggregation
    API->>API: Merge Segments by Page ID
    API->>API: Apply Business Logic (Missing Rate=0.0)
    API->>API: Validate against Pydantic Schema
    API-->>User: Return Final JSON Response
    deactivate API
```

---

# 📦 Prerequisites

### 🖥️ System Requirements
| Component | Requirement |
|----------|-------------|
| OS | Ubuntu / Windows / Mac |
| Python | 3.10+ |
| GPU | Recommended (NVIDIA 4GB+ VRAM) |
| Poppler | Required for PDF → Image |
| Tesseract | Required for OSD |


### 🔧 Install System Dependencies

#### **Ubuntu**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

#### **Windows**
Install:
- Tesseract OCR  
- Poppler for Windows  
(Add both to PATH)

---

# ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/bill-extractor.git
cd bill-extractor
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn requests python-multipart pydantic pdf2image Pillow numpy opencv-python-headless json-repair
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install surya-ocr deskew scikit-image pytesseract
pip install transformers google-generativeai openai
```

### 4. Install UniTable (Local)
Put model weights here:

```
/unitable/experiments/unitable_weights/unitable_large_structure.pt
/unitable/experiments/unitable_weights/unitable_large_bbox.pt
/unitable/experiments/unitable_weights/unitable_large_content.pt
```

---

# 🔧 Configuration

Edit your `main.py`:

```python
GEMINI_API_KEY = "YOUR_KEY"
OCR_BATCH_SIZE = 4
PREPROCESS_WORKERS = 4
LOCAL_LLM_URL = "http://localhost:8000/v1"
```

---

# 🏃‍♂️ Running the Server

Start FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

---

# 📤 Example Request

```bash
curl -X POST "http://localhost:5000/extract-bill-data" \
     -H "Content-Type: application/json" \
     -d '{"document": "https://example.com/bill.pdf"}'
```

---

# 📦 API Response Structure

```json
{
  "is_success": true,
  "time_taken": 2.83,
  "data": {
    "pagewise": [
      {
        "page_no": "1",
        "page_type": "Bill",
        "bill_items": [
          {
            "item_name": "Shelcal 500",
            "item_amount": 95.0,
            "item_rate": 9.5,
            "item_quantity": 10
          }
        ]
      }
    ],
    "total_count": 1
  }
}
```

---

# 👨‍💻 Author
**Utsav**
