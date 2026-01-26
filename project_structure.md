# Receipt Processing Project Structure

## 1. Project Overview
This project is an automated system based on **CrewAI** designed to process Korean receipt images. It employs a multi-agent pipeline to classify, optimize, and extract structured data from receipt images.

The core goal is to take a raw image (scanned or photographed) and reliably extract 5 key fields:
- Business Number (사업자등록번호)
- Store Name (상호명)
- Total Amount (총합계)
- Transaction Date/Time (거래일시)
- VAT Amount (부가세)

## 2. Directory Structure (`d:\ocr_convert`)

```
d:\ocr_convert
├── src/                                # Source code directory
│   ├── config/                         # Configuration files for agents and tasks
│   │   ├── agents.yaml                 # Definitions of agent roles, goals, and backstories
│   │   └── tasks.yaml                  # Definitions of task descriptions and expected outputs
│   ├── tools/                          # Custom tools for agents
│   │   ├── image_optimization_tool.py  # Image enhancement (deskew, denoise, sharpen)
│   │   ├── receipt_sam_tool.py         # Segmentation & cropping using SAM (Segment Anything Model)
│   │   ├── vision_classification_tool.py # Visual classification (Scan vs. Photo)
│   │   └── vision_extraction_tool.py   # OCR & Data extraction
│   ├── crew.py                         # Main orchestration logic (CrewAI setup)
│   └── main.py                         # Application entry point
├── scan_sample/                        # Sample images (Scanned receipts)
├── receipt_sample/                     # Sample images (General receipts)
├── photo_sample/                       # Sample images (Phone camera photos)
├── requirements.txt                    # Python dependencies
└── .env                                # Environment variables (API Keys)
```

## 3. Architecture & Agent Pipeline

The system operates in a two-stage process managed by `src/crew.py`:

### Stage 1: Pre-processing (Classification & Optimization)
1.  **Receipt Classification Specialist (`receipt_classifier`)**
    -   **Input**: Raw Image Path
    -   **Action**: Analyzes visual cues (lighting, shadows, contrast) to determine if the image is a 'Scanned Receipt' or a 'Phone Photograph Receipt'.
    -   **Tool**: `VisionClassificationTool`

2.  **Image Preprocessing & Optimization Expert (`image_quality_optimizer`)**
    -   **Input**: Classification Result & Raw Image
    -   **Action**:
        -   If *Phone Photo*: Uses `ReceiptSAMTool` to crop the receipt from the background, then enhances.
        -   If *Scanned*: Applies `ImageQualityOptimizationTool` directly (deskew, sharpen).
    -   **Output**: Path to the optimized (cropped/cleaned) image.

### Stage 2: Data Extraction
3.  **Receipt Extraction Specialist (`receipt_extraction_specialist`)**
    -   **Input**: Optimized Image Path
    -   **Action**: Extracts the 5 specific fields using VLM (Vision Language Model).
    -   **Mechanism**: Includes a retry loop (max 3 attempts) that validates the JSON output format and field constraints (e.g., business number format, date format).
    -   **Output**: Final JSON object.

## 4. Key Configuration Files

### `src/config/agents.yaml`
Defines the three agents with their specific personas:
-   **Classifier**: Focuses on "subtle visual cues" and "background contrast".
-   **Optimizer**: "Digital image restoration" expert.
-   **Extractor**: "Korean receipt format expert" with strict rules about not generating hallucinated data (returning `null` instead).

### `src/config/tasks.yaml`
Detailed instructions for each step:
-   **classify_receipt_task**: Criteria for distinguishing scans from photos.
-   **optimize_receipt_image_task**: Logic for when to use SAM cropping vs. standard optimization.
-   **extract_receipt_data_task**: 
    -   Defines the partial schema for the 5 target fields.
    -   Sets output formatting rules (integers for money, hyphens for IDs).

## 5. Execution Flow (`src/main.py`)
1.  Checks for `OPENAI_API_KEY`.
2.  Accepts an image path as a command-line argument.
3.  Initializes `ReceiptProcessingCrew`.
4.  Runs the pipeline.
5.  Prints the final JSON result to the console.
