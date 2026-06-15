# New Pipeline Integration Guide

## Overview
The new 4-phase pipeline has been integrated with the existing backend without modifying any existing files. This guide explains how to test and use the new pipeline.

## Architecture

### New Pipeline Components
- **Phase 1**: Reference Image Processing (Florence-2 → Grounding DINO → OpenCV → Gemini)
- **Phase 2**: P&ID Image Analysis (Parallel: OpenCV, PaddleOCR, Florence-2, Grounding DINO + Heuristic Engine)
- **Phase 3**: Subtype Matching (OpenCV Template Matching → Gemini Fallback)
- **Phase 4**: Final Output Generation

### Key Features
- **Pipe Detection**: Enhanced line detection with multiple thresholds for better pipe identification
- **Component Detection**: Uses Florence-2 and Grounding DINO for accurate symbol detection
- **Text Recognition**: PaddleOCR for instrument tags and equipment names
- **ISA 5.1 Compliance**: Validates tags and naming conventions
- **Gemini Verification**: Uses Gemini 2.5 Flash for subtype verification

## Testing the New Pipeline

### Option 1: Run Separate API Server (Recommended for Testing)

The new pipeline has its own API server that runs on port 8001:

```bash
cd Backend
python Backend/new_pipeline_server.py
```

This will start a server at `http://127.0.0.1:8001` with the following endpoints:

- `GET /health` - Health check
- `POST /analyze_new_pipeline` - Analyze P&ID (frontend-compatible format)
- `POST /analyze_new_pipeline_detailed` - Analyze with detailed results

### Option 2: Test with Frontend (Temporary Modification)

To test with the existing frontend, temporarily modify the API URL in the frontend:

1. Open `Frontend/src/screens/Generator.tsx`
2. Find line 5: `const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';`
3. Change it to: `const API_BASE_URL = 'http://127.0.0.1:8001';`
4. Start the new pipeline server: `python Backend/new_pipeline_server.py`
5. Start the frontend: `cd Frontend && npm run dev`
6. Test the upload and analysis

**Note**: Remember to revert the change after testing.

### Option 3: Direct API Testing

Use curl or Postman to test the endpoint directly:

```bash
# Health check
curl http://127.0.0.1:8001/health

# Analyze P&ID
curl -X POST http://127.0.0.1:8001/analyze_new_pipeline \
  -F "file=@path/to/your/pid_image.png"
```

## Pipe Detection

The new pipeline includes enhanced pipe detection:

- **Multi-threshold Line Detection**: Uses multiple Canny edge detection thresholds
- **Pipe Classification**: Identifies pipes based on line length (>50 pixels)
- **Junction Detection**: Finds intersections between pipes
- **Line Type Detection**: Distinguishes between solid and dashed lines

### Pipe Detection Results

Pipe detection results are included in:
- `opencv_results.total_pipes` - Total number of pipes detected
- `opencv_results.lines` - Array of all detected lines with `is_pipe` flag
- `pipeline_info.pipe_count` - Summary pipe count in API response

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Florence-2
FLORENCE_MODEL_PATH=microsoft/Florence-2-large
FLORENCE_DEVICE=cuda  # or cpu

# Grounding DINO
GROUNDING_DINO_CONFIG_PATH=GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
GROUNDING_DINO_CHECKPOINT_PATH=groundingdino_swint_ogc.pth
GROUNDING_DINO_BOX_THRESHOLD=0.35
GROUNDING_DINO_TEXT_THRESHOLD=0.25

# Gemini
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.1

# PaddleOCR
PADDLEOCR_LANG=en
PADDLEOCR_USE_GPU=false
OCR_MIN_TEXT_CONFIDENCE=0.5

# OpenCV
TEMPLATE_MATCH_THRESHOLD=0.8
MIN_COMPONENT_AREA=100
```

### Model Requirements

The pipeline requires:
- Florence-2 model (auto-downloaded from HuggingFace)
- Grounding DINO checkpoint (download from official repo)
- PaddleOCR models (auto-downloaded)
- Gemini API key (required)

## Response Format

The new pipeline returns data in the same format as the existing backend to ensure frontend compatibility:

```json
{
  "detection": {
    "filename": "uploaded-file",
    "pages": [
      {
        "counts": {
          "motor": 0,
          "pump": 5,
          "tank": 2,
          "valve": 8,
          "instrument": 3,
          "other": 1
        }
      }
    ]
  },
  "coordinates": {
    "root": {
      "children": [...],
      "meta": {"name": "P&ID Components"}
    }
  },
  "pipeline_info": {
    "pipeline": "new_4_phase",
    "pipe_count": 15,
    "total_components": 19,
    "opencv_lines": 25,
    "ocr_text_count": 8
  }
}
```

## Troubleshooting

### Model Loading Issues
- Ensure sufficient GPU memory if using CUDA
- Check model paths in configuration
- Verify internet connection for initial model downloads

### Pipe Detection Not Working
- Adjust line detection thresholds in `phase2_pid_analyzer.py`
- Check image quality and resolution
- Try different Canny edge detection parameters

### API Server Won't Start
- Check if port 8001 is already in use
- Verify all dependencies are installed
- Check Python version compatibility

### Frontend Integration Issues
- Ensure CORS is enabled (it is by default)
- Check API URL configuration
- Verify response format matches expectations

## Performance

- **Phase 1**: One-time setup (builds reference library)
- **Phase 2**: ~10-30 seconds per image (parallel processing)
- **Phase 3**: ~5-15 seconds per image (template matching + Gemini)
- **Phase 4**: <1 second (output generation)

## Next Steps

1. Test the new pipeline with sample P&ID images
2. Verify pipe detection accuracy
3. Compare results with existing pipeline
4. Adjust configuration parameters as needed
5. Integrate permanently if satisfied with results

## File Structure

```
Backend/
├── Backend/
│   ├── main.py                          # Existing backend (unchanged)
│   ├── new_pipeline_api.py             # New pipeline integration
│   └── new_pipeline_server.py          # New API server
├── new_pipeline/
│   ├── __init__.py
│   ├── config.py                       # Configuration
│   ├── utils.py                        # Utilities
│   ├── phase1_reference_processor.py    # Phase 1
│   ├── phase2_pid_analyzer.py          # Phase 2 (with pipe detection)
│   ├── phase3_subtype_matcher.py       # Phase 3
│   ├── phase4_output_generator.py      # Phase 4
│   ├── pipeline_orchestrator.py        # Orchestrator
│   ├── example_usage.py                # Examples
│   └── README.md                       # Documentation
└── .env                                # Environment variables
```

## Support

For issues or questions:
1. Check the logs in the terminal
2. Verify configuration in `.env`
3. Review the troubleshooting section above
4. Check model availability and API keys
