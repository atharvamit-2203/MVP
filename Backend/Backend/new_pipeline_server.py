"""
New Pipeline API Server
Separate FastAPI server for the new 4-phase pipeline
Runs alongside the existing backend
"""
import os
# Disable oneDNN and force CPU mode to avoid PaddleOCR compatibility issues
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_cudnn'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FLAGS_cudnn_deterministic'] = '0'
os.environ['FLAGS_cudnn_exhaustive_search'] = '0'
os.environ['FLAGS_enable_mkldnn'] = 'false'
# Force legacy executor to avoid oneDNN compatibility issues
os.environ['FLAGS_use_new_executor'] = '0'

import asyncio
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import fitz
import io
import base64

# Import the new pipeline analyzer
from new_pipeline_api import new_pipeline_analyzer, analyze_with_new_pipeline

# Import existing backend types for compatibility
import sys
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from main import (
    CategoryCounts,
    DetectionResponse,
    CoordinateDetectionResponse,
    PageDetectionResult,
    ModelDetectionResult,
    ComponentMatch,
    ComponentPosition,
    ComponentMeta,
    ComponentChild,
    RootMeta,
    Root,
    ComponentVerificationRequest,
    ComponentVerificationResponse,
    BatchComponentVerificationRequest,
    BatchComponentVerificationResponse,
    BatchComponentVerificationItem,
    _verify_component_industry_item,
    FeedbackRequest,
    ComponentMatchingRequest,
    ComponentMatchingResponse,
    Stage1Response,
    build_stage1_response,
    build_detection_response,
    build_coordinate_response,
    active_learning
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Create FastAPI app
app = FastAPI(
    title="New Pipeline API",
    version="1.0.0",
    description="4-phase P&ID analysis pipeline using Florence-2, Grounding DINO, and Gemini"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_image_frames(file_bytes: bytes, filename: str | None, content_type: str | None) -> tuple[list[Image.Image], str]:
    """Load image frames from file bytes (supports PDF and images)"""
    def is_pdf(filename: str | None, content_type: str | None) -> bool:
        if content_type == "application/pdf":
            return True
        return bool(filename and filename.lower().endswith(".pdf"))
    
    if is_pdf(filename, content_type):
        document = fitz.open(stream=file_bytes, filetype="pdf")
        frames: list[Image.Image] = []
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            frames.append(image)
        document.close()
        if not frames:
            raise ValueError("The uploaded PDF does not contain any renderable pages.")
        return frames, "pdf"

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
        return [image.convert("RGB")], "image"
    except Exception as exc:
        raise ValueError("Unsupported file type. Upload a valid image or PDF.") from exc


def counts_to_model(counts: dict[str, int]) -> CategoryCounts:
    """Convert counts dict to CategoryCounts model"""
    return CategoryCounts(
        motor=int(counts.get("motor", 0)),
        pump=int(counts.get("pump", 0)),
        tank=int(counts.get("tank", 0)),
        valve=int(counts.get("valve", 0)),
        instrument=int(counts.get("instrument", 0)),
        other=int(counts.get("other", 0)),
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline": "new_4_phase",
        "models": ["florence-2", "grounding-dino", "gemini-2.5-flash", "paddleocr", "opencv"]
    }


@app.post("/analyze_new_pipeline")
async def analyze_with_new_pipeline_endpoint(file: UploadFile = File(...), industry: str = None, components_json: str = None):
    """
    Analyze P&ID image using the new 4-phase pipeline
    
    Returns results in format compatible with the frontend
    """
    logger.info(f"Analyzing file with new pipeline: {file.filename}")
    
    try:
        # Load image
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        frames, source_type = load_image_frames(file_bytes, file.filename, file.content_type)
        
        if not frames:
            raise HTTPException(status_code=400, detail="No valid frames found in file.")
        
        # Analyze first frame with new pipeline
        first_frame = frames[0]
        # Reset file position so it can be read again by analyze_with_new_pipeline
        await file.seek(0)
        results = await analyze_with_new_pipeline(file)
        
        # Convert to frontend-compatible format
        counts = results.get('counts', {})
        coordinates = results.get('coordinates', {})
        
        # Build detection response
        category_counts = counts_to_model(counts)
        
        model_results = [
            ModelDetectionResult(
                page_index=1,
                model="florence-2+grounding-dino+gemini",
                role="detection",
                counts=category_counts
            ),
            ModelDetectionResult(
                page_index=1,
                model="opencv+paddleocr",
                role="verification",
                counts=category_counts
            )
        ]
        
        page_result = PageDetectionResult(
            page_index=1,
            counts=category_counts,
            model_results=model_results
        )
        
        detection_response = DetectionResponse(
            filename=file.filename or "uploaded-file",
            content_type=file.content_type,
            source_type=source_type,
            page_count=len(frames),
            models_used=["florence-2", "grounding-dino", "gemini-2.5-flash", "paddleocr", "opencv"],
            pages=[page_result],
            industry=None,
            industry_warnings={"component": [], "pid": []},
            component_matches=[]
        )
        
        # Convert to dict and inject pipe count directly for frontend consumption
        detection_dict = detection_response.dict()
        if detection_dict.get("pages") and len(detection_dict["pages"]) > 0:
            detection_dict["pages"][0]["counts"]["pipe"] = int(counts.get("pipe", 0))
            for mr in detection_dict["pages"][0].get("model_results", []):
                mr["counts"]["pipe"] = int(counts.get("pipe", 0))
        
        # Build coordinate response
        try:
            coordinate_response = CoordinateDetectionResponse(**coordinates)
        except Exception as e:
            logger.warning(f"Could not build coordinate response: {e}")
            # Return default coordinate structure
            coordinate_response = CoordinateDetectionResponse(
                custom={},
                params={},
                props={},
                root=Root(
                    children=[],
                    meta=RootMeta(name="P&ID Components"),
                    type="ia.cloud",
                    position={},
                    props={}
                )
            )
        
        # Return in format expected by frontend
        return {
            "detection": detection_dict,
            "coordinates": coordinate_response,
            "pipeline_info": {
                "pipeline": "new_4_phase",
                "pipe_count": counts.get('pipe', 0),
                "total_components": sum(counts.values()),
                "opencv_lines": results.get('opencv_results', {}).get('total_lines', 0),
                "ocr_text_count": results.get('ocr_results', {}).get('total_text', 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze_new_pipeline_detailed")
async def analyze_detailed(file: UploadFile = File(...)):
    """
    Analyze P&ID image with detailed pipeline information
    
    Returns full pipeline results including intermediate steps
    """
    logger.info(f"Detailed analysis with new pipeline: {file.filename}")
    
    try:
        # Load image
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        frames, source_type = load_image_frames(file_bytes, file.filename, file.content_type)
        
        if not frames:
            raise HTTPException(status_code=400, detail="No valid frames found in file.")
        
        # Analyze with new pipeline
        results = await analyze_with_new_pipeline(file)
        
        # Return detailed results
        return {
            "status": "success",
            "pipeline": "new_4_phase",
            "source_type": source_type,
            "page_count": len(frames),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detailed analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/verify_component_industry", response_model=ComponentVerificationResponse)
async def verify_component_industry(request: ComponentVerificationRequest) -> ComponentVerificationResponse:
    """Verify component against industry using keyword matching (delegates to existing backend logic)"""
    try:
        return _verify_component_industry_item(request.name, request.industry)
    except Exception as exc:
        return ComponentVerificationResponse(
            matches=False,
            detected_industry=None,
            message=f"Verification failed: {str(exc)}"
        )


@app.post("/verify_component_industries", response_model=BatchComponentVerificationResponse)
async def verify_component_industries(request: BatchComponentVerificationRequest) -> BatchComponentVerificationResponse:
    """Verify multiple components against a selected industry in one request"""
    results = [_verify_component_industry_item(component.name, request.industry) for component in request.components]
    return BatchComponentVerificationResponse(
        industry=request.industry,
        results=[
            BatchComponentVerificationItem(
                name=request.components[index].name,
                matches=result.matches,
                detected_industry=result.detected_industry,
                message=result.message,
            )
            for index, result in enumerate(results)
        ],
    )


@app.post("/retrain")
async def retrain_model() -> dict[str, Any]:
    """Retrain the active learning model from collected annotations."""
    try:
        await asyncio.to_thread(active_learning.train_model)
        return {"status": "success", "message": "Model retrained successfully"}
    except Exception as exc:
        logger.error(f"Retrain failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest) -> dict[str, str]:
    """Append human-corrected counts to a feedback file for future fine-tuning."""
    try:
        feedback_dir = BACKEND_ROOT / "feedback"
        feedback_dir.mkdir(exist_ok=True)
        feedback_file = feedback_dir / "feedback.jsonl"
        
        entry = {
            "timestamp": asyncio.get_event_loop().time(),
            "filename": feedback.filename,
            "page_index": feedback.page_index,
            "counts": feedback.counts,
            "industry": feedback.industry,
            "notes": feedback.notes,
        }
        
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        return {"status": "saved"}
    except Exception as exc:
        logger.error(f"Feedback submission failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/match_components", response_model=ComponentMatchingResponse)
async def match_components(request: ComponentMatchingRequest, file: UploadFile = File(...)) -> ComponentMatchingResponse:
    """Match detected components from P&ID diagram against the component library."""
    try:
        # Delegate to existing backend logic
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        frames, _ = load_image_frames(file_bytes, file.filename, file.content_type)
        if not frames:
            raise HTTPException(status_code=400, detail="No valid frames found in file.")
        
        # Use existing matching logic
        from main import match_components as main_match_components
        return await main_match_components(request, file)
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Component matching failed: {exc}")
        # Return empty matches on error
        return ComponentMatchingResponse(matches=[])


@app.post("/upload", response_model=Stage1Response)
async def upload_stage1(file: UploadFile = File(...)) -> Stage1Response:
    """Stage 1 upload - preprocess and return image info"""
    return await build_stage1_response(file)


@app.post("/detect", response_model=DetectionResponse)
async def detect_components(
    file: UploadFile = File(...),
    industry: str = Form(None),
    components_json: str = Form(None)
) -> DetectionResponse:
    """Detect components in P&ID image"""
    # Parse components if provided
    components = None
    if components_json:
        try:
            from main import ComponentData
            components = [ComponentData(**comp) for comp in json.loads(components_json) if (comp or {}).get("name")]
        except Exception:
            components = None
    
    return await build_detection_response(file, industry, components)


@app.post("/analyze_fast")
async def analyze_fast(
    file: UploadFile = File(...),
    industry: str = Form(None),
    components_json: str = Form(None)
):
    """
    Fast analysis endpoint - delegates to new pipeline
    This is the main endpoint the frontend uses
    """
    # Use the new pipeline for analysis
    return await analyze_with_new_pipeline_endpoint(file, industry, components_json)


@app.post("/coordinates", response_model=CoordinateDetectionResponse)
async def detect_component_coordinates(file: UploadFile = File(...)) -> CoordinateDetectionResponse:
    """Detect component coordinates"""
    return await build_coordinate_response(file)


@app.post("/diagnostics")
async def diagnostics(file: UploadFile = File(...)) -> dict[str, Any]:
    """Return detailed detection outputs for debugging"""
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        frames, _ = load_image_frames(file_bytes, file.filename, file.content_type)
        if not frames:
            raise HTTPException(status_code=400, detail="No valid frames found in file.")
        
        # Use new pipeline for detailed analysis
        results = await analyze_with_new_pipeline(file)
        
        return {
            "pipeline": "new_4_phase",
            "opencv_results": results.get("opencv_results", {}),
            "ocr_results": results.get("ocr_results", {}),
            "heuristic_results": results.get("heuristic_results", {}),
            "detections": results.get("detections", [])
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Diagnostics failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
