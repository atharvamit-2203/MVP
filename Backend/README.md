# Backend

FastAPI backend for P&ID upload, preprocessing, and local detection with OpenCV + PaddleOCR.

## Environment

The local pipeline does not require an API key. Optional knobs in `.env.example`:

- `PADDLEOCR_LANG`
- `PADDLEOCR_USE_GPU`
- `OCR_MIN_TEXT_CONFIDENCE`
- `OCR_MIN_COMPONENT_AREA`

## Run

```powershell
cd c:\Users\aadeshpande\Desktop\Sarla-Project\Backend
uvicorn Backend.main:app --reload
```

## Smoke Test

```powershell
cd c:\Users\aadeshpande\Desktop\Sarla-Project\Backend
python Backend\test_ocr.py
```
