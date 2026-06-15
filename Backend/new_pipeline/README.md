# New 4-Phase P&ID Analysis Pipeline

A comprehensive pipeline for analyzing Piping and Instrumentation Diagrams (P&IDs) using Florence-2, Grounding DINO, and Gemini 2.5 Flash.

## Architecture

### Phase 1: Reference Image Processing
- **Input**: Reference images (pumps, valves, vessels, motors, pipes)
- **Florence-2**: Dense Region Caption to find and label all symbol regions
- **Grounding DINO**: Uses Florence labels as prompts to get precise bounding boxes
- **OpenCV**: Crops each bounding box and saves individual symbols
- **Gemini 2.5 Flash**: Verifies and confirms each subtype, corrects wrong labels
- **Output**: Reference library organized by category/subtype

### Phase 2: P&ID Image Analysis
- **Input**: P&ID image to analyze
- **Parallel Processing**:
  - **OpenCV**: Detect pipes/lines, line types, junctions/fittings, component bounding boxes
  - **PaddleOCR**: Read text labels, instrument tags (PT, LT, FT), equipment names
  - **Florence-2**: Symbol detection, region captions, bounding boxes
  - **Grounding DINO**: Object detection, confidence scores, precise bounding boxes
- **Heuristic Engine**: Applies ISA 5.1 rules, validates tags, filters non P&ID text, connects components
- **Output**: Validated components with metadata

### Phase 3: Subtype Matching
- **Input**: Detected components from Phase 2
- **OpenCV Template Matching**: Compare against reference library
  - Confidence > 80%: Subtype assigned automatically
  - Confidence < 80%: Pass to Gemini
- **Gemini 2.5 Flash**: Receives P&ID image, reference images, and all pipeline outputs
  - Returns confirmed subtype with reasoning
- **Output**: Components with matched subtypes and confidence scores

### Phase 4: Final Output
- **Input**: Results from all previous phases
- **Output Generation**:
  - Component inventory with categories and subtypes
  - Connection map showing component relationships
  - Text annotations (instrument tags, equipment names)
  - Statistics and quality metrics
  - Human-readable summary report
- **Output**: JSON file and text report

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Additional Dependencies
The pipeline requires:
- Florence-2 model (microsoft/Florence-2-large)
- Grounding DINO model and checkpoint
- PaddleOCR
- Google Generative AI (Gemini 2.5 Flash)
- OpenCV
- PyTorch

### Configuration
Set up your `.env` file with the following variables:

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

## Usage

### Basic Usage

```python
from new_pipeline.pipeline_orchestrator import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator()

# Define reference images
reference_images = {
    'pumps': ['path/to/pump1.png', 'path/to/pump2.png'],
    'valves': ['path/to/valve1.png', 'path/to/valve2.png'],
    'vessels': ['path/to/vessel1.png']
}

# Run full pipeline
results = orchestrator.run_full_pipeline(
    reference_images=reference_images,
    pid_image_path='path/to/pid_diagram.png',
    output_filename='my_analysis.json'
)
```

### Analysis Only (Reference Library Already Exists)

```python
from new_pipeline.pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()

# Run only Phases 2-4
results = orchestrator.run_analysis_only(
    pid_image_path='path/to/pid_diagram.png',
    output_filename='analysis_results.json'
)
```

### Build Reference Library Only

```python
from new_pipeline.pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()
orchestrator.initialize_phase1()

reference_images = {
    'pumps': ['path/to/pump1.png'],
    'valves': ['path/to/valve1.png']
}

phase1_results = orchestrator.run_phase1(reference_images)
```

## Output Structure

The pipeline generates two output files:

### JSON Output (`pipeline_outputs/pid_analysis_*.json`)
```json
{
  "metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "pipeline_version": "1.0",
    "image_path": "...",
    "image_name": "..."
  },
  "reference_library": {
    "total_symbols": 15,
    "categories": {...}
  },
  "component_inventory": {
    "total_components": 25,
    "by_category": {...},
    "by_subtype": {...},
    "components": [...]
  },
  "connection_map": {
    "total_connections": 18,
    "connections": [...]
  },
  "text_annotations": {
    "total_text": 12,
    "instrument_tags": [...],
    "equipment_names": [...]
  },
  "statistics": {...},
  "quality_metrics": {...}
}
```

### Summary Report (`pipeline_outputs/pid_summary_*.txt`)
Human-readable summary with:
- Component inventory breakdown
- Connection information
- Text annotations
- Statistics
- Quality metrics

## Module Structure

```
new_pipeline/
├── __init__.py
├── config.py                      # Configuration settings
├── utils.py                       # Utility functions
├── phase1_reference_processor.py  # Phase 1 implementation
├── phase2_pid_analyzer.py         # Phase 2 implementation
├── phase3_subtype_matcher.py      # Phase 3 implementation
├── phase4_output_generator.py     # Phase 4 implementation
├── pipeline_orchestrator.py       # Main orchestrator
├── example_usage.py               # Usage examples
└── README.md                      # This file
```

## ISA 5.1 Compliance

The pipeline follows ISA 5.1 standards for:
- Instrument tag validation (PT, LT, FT, TT, AT, etc.)
- Equipment naming conventions
- P&ID symbol recognition
- Connection and relationship mapping

## Performance Considerations

- **Phase 1**: One-time setup to build reference library
- **Phase 2**: Parallel processing for faster analysis
- **Phase 3**: Template matching is fast; Gemini fallback is slower but more accurate
- **Phase 4**: Lightweight output generation

## Troubleshooting

### Model Loading Issues
- Ensure sufficient GPU memory if using CUDA
- Check model paths in configuration
- Verify internet connection for initial model downloads

### Low Detection Accuracy
- Adjust `GROUNDING_DINO_BOX_THRESHOLD` and `GROUNDING_DINO_TEXT_THRESHOLD`
- Improve reference image quality
- Increase reference library diversity

### Slow Performance
- Use GPU for Florence-2 and Grounding DINO
- Enable PaddleOCR GPU acceleration
- Reduce image resolution for faster processing

## Example Workflow

See `example_usage.py` for complete examples of:
- Building reference library only
- Running analysis only
- Complete pipeline execution
- Custom workflows

## License

This pipeline is part of the Sarla P&ID Detector project.
