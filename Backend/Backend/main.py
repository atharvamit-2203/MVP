from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal
import json
from datetime import datetime

import cv2
import fitz
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

if __package__:
	from . import local_detection
	from .local_detection import analyze_pid_image, detect_coordinates, analyze_pid_image_async, detect_coordinates_async
	from . import active_learning
else:
	import sys

	current_dir = Path(__file__).resolve().parent
	if str(current_dir) not in sys.path:
		sys.path.insert(0, str(current_dir))
	import local_detection
	from local_detection import analyze_pid_image, detect_coordinates, analyze_pid_image_async, detect_coordinates_async
	import active_learning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")

ML_COUNT_MIN_CONFIDENCE = float(os.getenv("ML_COUNT_MIN_CONFIDENCE", "0.50"))
FAST_ANALYSIS_TIMEOUT_SECONDS = float(os.getenv("FAST_ANALYSIS_TIMEOUT_SECONDS", "35"))
FAST_ANALYSIS_MAX_PAGES = max(1, int(os.getenv("FAST_ANALYSIS_MAX_PAGES", "1")))
FAST_ANALYSIS_IMAGE_MAX_EDGE = max(512, int(os.getenv("FAST_ANALYSIS_IMAGE_MAX_EDGE", "1024")))
FAST_ANALYSIS_MODE = os.getenv("FAST_ANALYSIS_MODE", "full").strip().lower()
COMPONENT_LIBRARY_AUTO_TRAIN = os.getenv("COMPONENT_LIBRARY_AUTO_TRAIN", "true").strip().lower() in {"1", "true", "yes", "on"}
DISABLE_OLLAMA_VERIFICATION = os.getenv("DISABLE_OLLAMA_VERIFICATION", "true").strip().lower() in {"1", "true", "yes", "on"}

_component_training_task: asyncio.Task | None = None
_analysis_warm: bool = False

app = FastAPI(title="Sarla P&ID Backend", version="0.3.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


async def _warmup_analysis_models() -> None:
	global _analysis_warm
	if _analysis_warm:
		return

	try:
		# Force OCR engine/model initialization up-front so first analyze request is not penalized.
		await asyncio.wait_for(asyncio.to_thread(local_detection.get_ocr_engine), timeout=90)
		# Build CV kernels/caches once on a tiny image.
		dummy = np.zeros((96, 96, 3), dtype=np.uint8)
		await asyncio.wait_for(asyncio.to_thread(local_detection.detect_shape_components, dummy, []), timeout=15)
		# Warm model cache if available.
		await asyncio.to_thread(_active_model_blob)
		_analysis_warm = True
		logger.info("Analysis models warmed successfully.")
	except Exception as exc:  # noqa: BLE001
		logger.warning(f"Model warmup incomplete: {exc}")


@app.on_event("startup")
async def on_startup() -> None:
	await _warmup_analysis_models()


class FrameSummary(BaseModel):
	page_index: int = Field(..., ge=1)
	width: int = Field(..., ge=1)
	height: int = Field(..., ge=1)
	mode: str
	preprocessed_width: int = Field(..., ge=1)
	preprocessed_height: int = Field(..., ge=1)
	preview_png_base64: str


class Stage1Response(BaseModel):
	filename: str
	content_type: str | None
	source_type: Literal["image", "pdf"]
	page_count: int = Field(..., ge=1)
	frames: list[FrameSummary]


class CategoryCounts(BaseModel):
	motor: int = Field(0, ge=0)
	pump: int = Field(0, ge=0)
	tank: int = Field(0, ge=0)
	valve: int = Field(0, ge=0)


class ModelDetectionResult(BaseModel):
	page_index: int = Field(..., ge=1)
	model: str
	role: Literal["detection", "verification"]
	counts: CategoryCounts


class PageDetectionResult(BaseModel):
	page_index: int = Field(..., ge=1)
	counts: CategoryCounts
	model_results: list[ModelDetectionResult]


class ComponentVerificationRequest(BaseModel):
	name: str
	file_data: str  # base64 encoded image
	industry: str


class ComponentVerificationResponse(BaseModel):
	matches: bool
	detected_industry: str | None = None
	message: str


class BatchComponentVerificationRequest(BaseModel):
	industry: str
	components: list[ComponentMeta]


class BatchComponentVerificationItem(BaseModel):
	name: str
	matches: bool
	detected_industry: str | None = None
	message: str


class BatchComponentVerificationResponse(BaseModel):
	industry: str
	results: list[BatchComponentVerificationItem]


class ComponentData(BaseModel):
	name: str
	file_data: str  # base64 encoded image


class DetectionRequest(BaseModel):
	industry: str
	components: list[ComponentData]


class DetectionResponse(BaseModel):
	filename: str
	content_type: str | None
	source_type: Literal["image", "pdf"]
	page_count: int = Field(..., ge=1)
	models_used: list[str]
	pages: list[PageDetectionResult]
	industry: str | None = None
	industry_warnings: dict[str, list[str]] = Field(default_factory=dict)


class ComponentPosition(BaseModel):
	x: int = Field(..., ge=0)
	y: int = Field(..., ge=0)
	width: int = Field(..., ge=0)
	height: int = Field(..., ge=0)


class ComponentMeta(BaseModel):
	name: str


class ComponentChild(BaseModel):
	meta: ComponentMeta
	position: ComponentPosition
	type: str


class RootMeta(BaseModel):
	name: str


class Root(BaseModel):
	children: list[ComponentChild]
	meta: RootMeta
	type: str


class CoordinateDetectionResponse(BaseModel):
	custom: dict[str, Any]
	params: dict[str, Any]
	props: dict[str, Any]
	root: Root


class BatchStage1Response(BaseModel):
	files: list[Stage1Response]


class BatchDetectionResponse(BaseModel):
	files: list[DetectionResponse]


class BatchCoordinateDetectionResponse(BaseModel):
	files: list[CoordinateDetectionResponse]


class BatchAnalysisItem(BaseModel):
	filename: str
	content_type: str | None
	source_type: Literal["image", "pdf"]
	page_count: int = Field(..., ge=1)
	result: Stage1Response | None = None
	detection: DetectionResponse | None = None
	coordinates: CoordinateDetectionResponse | None = None
	error: str | None = None


class BatchAnalysisResponse(BaseModel):
	files: list[BatchAnalysisItem]


class FeedbackRequest(BaseModel):
	filename: str | None = None
	page_index: int | None = None
	counts: dict[str, int]
	industry: str | None = None
	notes: str | None = None


def is_pdf(filename: str | None, content_type: str | None) -> bool:
	if content_type == "application/pdf":
		return True
	return bool(filename and filename.lower().endswith(".pdf"))


@app.get("/annotator", response_class=HTMLResponse)
async def annotator_ui() -> HTMLResponse:
	html_path = BACKEND_ROOT / "static" / "annotator.html"
	if not html_path.exists():
		raise HTTPException(status_code=404, detail="Annotator UI not found")
	return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/annotate")
async def annotate(file: UploadFile = File(...), annotations: str = Form(...), page_index: int = Form(1)) -> dict[str, str]:
	"""Save uploaded image and annotations (simple JSON lines archive)."""
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	try:
		save_dir = BACKEND_ROOT / "annotations"
		save_dir.mkdir(parents=True, exist_ok=True)
		image_path = save_dir / file.filename
		with open(image_path, "wb") as fh:
			fh.write(file_bytes)

		parsed = json.loads(annotations)
		entry = {
			"timestamp": datetime.utcnow().isoformat() + "Z",
			"image": str(image_path.name),
			"page_index": page_index,
			"annotations": parsed,
		}
		ann_path = save_dir / "annotations.jsonl"
		with open(ann_path, "a", encoding="utf-8") as fh:
			fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

		return {"status": "saved", "image": str(image_path)}
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=500, detail=str(exc)) from exc


def load_image_frames(file_bytes: bytes, filename: str | None, content_type: str | None) -> tuple[list[Image.Image], str]:
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
	except Exception as exc:  # noqa: BLE001
		raise ValueError("Unsupported file type. Upload a valid image or PDF.") from exc


def preprocess_image(image: Image.Image) -> Image.Image:
	rgb_array = np.array(image.convert("RGB"))
	gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced = clahe.apply(blurred)
	return Image.fromarray(enhanced)


def image_to_base64_png(image: Image.Image) -> str:
	buffer = io.BytesIO()
	image.save(buffer, format="PNG")
	return base64.b64encode(buffer.getvalue()).decode("ascii")


def counts_to_model(counts: dict[str, int]) -> CategoryCounts:
	return CategoryCounts(
		motor=int(counts.get("motor", 0)),
		pump=int(counts.get("pump", 0)),
		tank=int(counts.get("tank", 0)),
		valve=int(counts.get("valve", 0)),
	)


def model_counts_to_dict(counts: CategoryCounts) -> dict[str, int]:
	return {
		"motor": int(counts.motor),
		"pump": int(counts.pump),
		"tank": int(counts.tank),
		"valve": int(counts.valve),
	}


def _active_model_blob() -> dict[str, Any] | None:
	return active_learning.load_model_cached()


def _predict_trained_model_counts(frame: Image.Image, detections: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    model_blob = _active_model_blob()
    if model_blob is None or not detections:
        logger.info(f"No trained model available (model_blob={'None' if model_blob is None else 'loaded'}) or no detections ({len(detections)}), using zero counts")
        return {key: 0 for key in local_detection.COUNT_KEYS}, []

    logger.info(f"Using trained model for {len(detections)} detections")
    image_array = np.array(frame.convert("RGB"))
    scored = active_learning.predict_candidates(image_array, detections, model_blob=model_blob)
    counts = {key: 0 for key in local_detection.COUNT_KEYS}
    high_conf_count = 0
    for candidate in scored:
        predicted = candidate.get("predicted")
        prob = float(candidate.get("prob", 0.0) or 0.0)
        fallback_category = candidate.get("category")
        fallback_conf = float(candidate.get("confidence", 0.0) or 0.0)
        # Use a moderate confidence threshold for pumps/valves
        if predicted in counts and prob >= 0.45:
            counts[predicted] += 1
            if prob >= 0.45:
                high_conf_count += 1
        elif fallback_category in counts and fallback_conf >= local_detection.CONF_THRESH.get(fallback_category, 0.0):
            counts[fallback_category] += 1
    logger.info(f"Trained model predictions: {counts}, high confidence: {high_conf_count}/{len(scored)}")
    return counts, scored


def _merge_count_dicts(*count_dicts: dict[str, int]) -> dict[str, int]:
	merged = {key: 0 for key in local_detection.COUNT_KEYS}
	for counts in count_dicts:
		for key in local_detection.COUNT_KEYS:
			merged[key] = max(merged[key], int(counts.get(key, 0)))
	return merged


def _default_coordinate_payload() -> dict[str, Any]:
	return {
		"custom": {},
		"params": {},
		"props": {},
		"root": {
			"children": [],
			"meta": {"name": "P&ID Components"},
			"type": "ia.cloud",
		},
	}


def _scale_coordinates_to_original(
	coordinates: dict[str, Any],
	scale_x: float,
	scale_y: float,
	original_width: int,
	original_height: int,
) -> dict[str, Any]:
	if "root" not in coordinates or "children" not in coordinates["root"]:
		return coordinates

	for child in coordinates["root"]["children"]:
		position = child.get("position")
		if not isinstance(position, dict):
			continue
		try:
			x = int(float(position.get("x", 0)) * scale_x)
			y = int(float(position.get("y", 0)) * scale_y)
			width = int(float(position.get("width", 0)) * scale_x)
			height = int(float(position.get("height", 0)) * scale_y)
		except (TypeError, ValueError):
			continue

		if x < 0 or y < 0 or width <= 0 or height <= 0:
			continue
		if x + width > original_width or y + height > original_height:
			continue
		child["position"] = {"x": x, "y": y, "width": width, "height": height}

	return coordinates


def _normalize_component_label(label_name: str) -> str | None:
	name = label_name.strip().lower()
	if not name:
		return None

	if any(token in name for token in ("pump", "centrifugal", "reciprocating", "rotary", "screw", "submersible", "triplex", "sump")):
		return "pump"
	if any(token in name for token in ("valve", "gate", "globe", "ball", "butterfly", "check", "control", "plug", "pcv", "fcv", "lcv", "psv", "xv", "cv", "hv")):
		return "valve"
	if any(token in name for token in ("tank", "vessel", "drum", "reactor", "column")):
		return "tank"
	if any(token in name for token in ("motor", "drive")):
		return "motor"
	return None


def _extract_component_labels(raw_name: str) -> list[str]:
	parts = [part.strip() for part in re.split(r"[,;/|]+|\band\b", raw_name, flags=re.IGNORECASE) if part.strip()]
	if not parts:
		parts = [raw_name.strip()]

	labels: list[str] = []
	for part in parts:
		normalized = _normalize_component_label(part)
		if normalized:
			labels.append(normalized)

	if not labels:
		normalized_full = _normalize_component_label(raw_name)
		if normalized_full:
			labels.append(normalized_full)

	# Preserve ordering while removing duplicates.
	return list(dict.fromkeys(labels))


def _persist_component_library_samples(components: list[ComponentData]) -> int:
	active_learning.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
	ann_path = active_learning.ANNOTATIONS_DIR / "annotations.jsonl"
	saved = 0

	for component in components:
		name = (component.name or "").strip()
		if not name or not component.file_data:
			continue

		try:
			image_bytes = base64.b64decode(component.file_data)
		except Exception:
			continue

		image_array = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
		if image_array is None:
			continue

		height, width = image_array.shape[:2]
		stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
		safe_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "component"
		image_name = f"{stamp}_{safe_name}.png"
		image_path = active_learning.ANNOTATIONS_DIR / image_name

		with open(image_path, "wb") as fh:
			fh.write(image_bytes)

		labels = _extract_component_labels(name)
		if not labels:
			labels = ["other"]  # Will be filtered out later

		# Detect individual components within the uploaded image
		try:
			# Convert to RGB for detection
			image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
			# Detect shape components in the image
			detected_components = local_detection.detect_shape_components(image_rgb, [])
			
			if detected_components and len(detected_components) > 0:
				# Create annotations for each detected component
				annotations = []
				# If user provided multiple labels, distribute them across detected components
				# If user provided single label, use it for all detected components
				if len(labels) == 1:
					# Single label - apply to all detected components
					for det in detected_components:
						bbox = det.get("bbox", [0, 0, width, height])
						annotations.append({
							"label": labels[0],
							"bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
						})
				else:
					# Multiple labels - try to match detected categories with user labels
					for det in detected_components:
						bbox = det.get("bbox", [0, 0, width, height])
						category = det.get("category", "other")
						# Prioritize user-provided labels that match detected category
						if category in labels:
							component_label = category
						else:
							# Use the first label as default if no match
							component_label = labels[0]
						
						annotations.append({
							"label": component_label,
							"bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
						})
				
				if annotations:
					entry = {
						"timestamp": datetime.utcnow().isoformat() + "Z",
						"image": image_name,
						"annotations": annotations,
					}
					with open(ann_path, "a", encoding="utf-8") as fh:
						fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
					saved += 1
					logger.info(f"Saved {len(annotations)} component detections from image {image_name} with labels {labels}")
				else:
					# Fallback to full image annotation if no valid detections
					entry = {
						"timestamp": datetime.utcnow().isoformat() + "Z",
						"image": image_name,
						"annotations": [{"label": labels[0], "bbox": [0, 0, int(width), int(height)]}],
					}
					with open(ann_path, "a", encoding="utf-8") as fh:
						fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
					saved += 1
					logger.info(f"Saved full image annotation for {image_name} (no components detected)")
			else:
				# Fallback to full image annotation if no detections
				entry = {
					"timestamp": datetime.utcnow().isoformat() + "Z",
					"image": image_name,
					"annotations": [{"label": labels[0], "bbox": [0, 0, int(width), int(height)]}],
				}
				with open(ann_path, "a", encoding="utf-8") as fh:
					fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
				saved += 1
				logger.info(f"Saved full image annotation for {image_name} (no components detected)")
		except Exception as exc:
			logger.warning(f"Component detection failed for {image_name}: {exc}, using full image annotation")
			# Fallback to full image annotation on error
			entry = {
				"timestamp": datetime.utcnow().isoformat() + "Z",
				"image": image_name,
				"annotations": [{"label": labels[0], "bbox": [0, 0, int(width), int(height)]}],
			}
			with open(ann_path, "a", encoding="utf-8") as fh:
				fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
			saved += 1

	return saved


def _schedule_component_library_training() -> None:
	global _component_training_task
	if _component_training_task is not None and not _component_training_task.done():
		return

	async def _train_task() -> None:
		try:
			await asyncio.to_thread(active_learning.train_model)
		except Exception as exc:  # noqa: BLE001
			logger.warning(f"Background component-library training failed: {exc}")

	_component_training_task = asyncio.create_task(_train_task())


async def build_stage1_response(file: UploadFile) -> Stage1Response:
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	try:
		frames, source_type = load_image_frames(file_bytes, file.filename, file.content_type)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	frame_summaries: list[FrameSummary] = []
	for page_index, frame in enumerate(frames, start=1):
		preprocessed = preprocess_image(frame)
		frame_summaries.append(
			FrameSummary(
				page_index=page_index,
				width=frame.width,
				height=frame.height,
				mode=frame.mode,
				preprocessed_width=preprocessed.width,
				preprocessed_height=preprocessed.height,
				preview_png_base64=image_to_base64_png(preprocessed),
			),
		)

	return Stage1Response(
		filename=file.filename or "uploaded-file",
		content_type=file.content_type,
		source_type=source_type,
		page_count=len(frames),
		frames=frame_summaries,
	)


async def build_detection_response(file: UploadFile, industry: str | None = None, components: list[ComponentData] | None = None) -> DetectionResponse:
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	try:
		frames, source_type = load_image_frames(file_bytes, file.filename, file.content_type)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	try:
		pages = await detect_pages(frames)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=502, detail=str(exc)) from exc

	detected_industry = None
	models_used = ["opencv+paddleocr", "opencv+heuristics"]
	if _active_model_blob() is not None:
		models_used.append("active_learning/random_forest")

	# Industry validation disabled
	industry_warnings: dict[str, list[str]] = {"component": [], "pid": []}

	return DetectionResponse(
		filename=file.filename or "uploaded-file",
		content_type=file.content_type,
		source_type=source_type,
		page_count=len(frames),
		models_used=models_used,
		pages=pages,
		industry=detected_industry,
		industry_warnings=industry_warnings,
	)


async def build_detection_and_coordinate_response(
	file: UploadFile,
	industry: str | None = None,
	components: list[ComponentData] | None = None
) -> tuple[DetectionResponse, CoordinateDetectionResponse]:
	"""Build detection and coordinate responses with a bounded-time fast path."""
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	try:
		frames, source_type = load_image_frames(file_bytes, file.filename, file.content_type)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	frames_to_analyze = frames[: min(len(frames), FAST_ANALYSIS_MAX_PAGES)]
	if not frames_to_analyze:
		raise HTTPException(status_code=400, detail="No pages available for analysis.")

	first_frame = frames_to_analyze[0]
	analysis = await analyze_pid_image_async(first_frame, fast_mode=True)
	scale_x = 1.0
	scale_y = 1.0

	trained_counts, _scored = _predict_trained_model_counts(first_frame, list(analysis.get("detections", [])))
	analysis["trained_counts"] = trained_counts

	pages = [counts_from_analysis_page(analysis, page_index=1)]
	
	detected_industry = None
	models_used = ["opencv+paddleocr", "opencv+heuristics"]
	if _active_model_blob() is not None:
		models_used.append("active_learning/random_forest")

	# Industry validation disabled
	industry_warnings: dict[str, list[str]] = {"component": [], "pid": []}

	detection_response = DetectionResponse(
		filename=file.filename or "uploaded-file",
		content_type=file.content_type,
		source_type=source_type,
		page_count=len(frames_to_analyze),
		models_used=models_used,
		pages=pages,
		industry=detected_industry,
		industry_warnings=industry_warnings,
	)

	coordinate_payload = analysis.get("coordinates") or _default_coordinate_payload()
	coordinate_payload = _scale_coordinates_to_original(
		coordinate_payload,
		scale_x=scale_x,
		scale_y=scale_y,
		original_width=first_frame.width,
		original_height=first_frame.height,
	)
	coordinate_response = CoordinateDetectionResponse(**coordinate_payload)
	
	return detection_response, coordinate_response


def _infer_component_industry(component_name: str) -> str | None:
	"""Industry detection disabled - returns None"""
	return None


@app.post("/verify_component_industry", response_model=ComponentVerificationResponse)
async def verify_component_industry(request: ComponentVerificationRequest) -> ComponentVerificationResponse:
	"""Fast component industry verification using keyword matching."""
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
	"""Verify multiple components against a selected industry in one request."""
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


def _verify_component_industry_item(component_name: str, industry: str) -> ComponentVerificationResponse:
	detected_industry = _infer_component_industry(component_name)

	# Normalize industry names for comparison
	selected_industry_normalized = industry.lower()
	detected_industry_normalized = detected_industry.lower() if detected_industry else ""

	# Check if industries match
	matches = detected_industry is not None and (
		detected_industry == "Common P&ID Components" or
		selected_industry_normalized == detected_industry_normalized or
		selected_industry_normalized in detected_industry_normalized or
		detected_industry_normalized in selected_industry_normalized
	)

	message = ""
	if matches:
		if detected_industry == "Common P&ID Components":
			message = (
				f"Component '{component_name}' is a common P&ID symbol and applies to every industry, "
				f"including '{industry}'."
			)
		else:
			message = f"Component '{component_name}' matches the selected industry '{industry}'."
	elif detected_industry:
		message = f"Component '{component_name}' appears to belong to '{detected_industry}', not '{industry}'."
	else:
		message = (
			f"Could not determine industry for component '{component_name}'. "
			f"Checked against selected industry '{industry}'."
		)

	return ComponentVerificationResponse(
		matches=matches,
		detected_industry=detected_industry,
		message=message
	)


async def build_coordinate_response(file: UploadFile) -> CoordinateDetectionResponse:
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	try:
		frames, _source_type = load_image_frames(file_bytes, file.filename, file.content_type)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	try:
		local_result = await analyze_pid_image_async(frames[0])
		local_coordinates = local_result.get("coordinates", {})
		local_count = len(local_coordinates.get("root", {}).get("children", []))
		coordinates = local_coordinates
		logger.info(f"Using local detection coordinates with {local_count} components")

	except Exception as exc:  # noqa: BLE001
		logger.error(f"Coordinate detection failed: {exc}")
		try:
			coordinates = await detect_coordinates_async(frames[0])
			logger.info("Fallback to local detection successful")
		except Exception as fallback_exc:  # noqa: BLE001
			logger.error(f"Fallback detection also failed: {fallback_exc}")
			raise HTTPException(status_code=502, detail=str(exc)) from exc

	return CoordinateDetectionResponse(**coordinates)


def counts_from_analysis_page(analysis: dict[str, Any], page_index: int) -> PageDetectionResult:
	ocr_counts = counts_to_model(analysis["ocr_counts"])
	vision_counts = counts_to_model(analysis["vision_counts"])
	phi3_counts_raw = analysis.get("phi3_counts")
	phi3_counts = counts_to_model(phi3_counts_raw) if isinstance(phi3_counts_raw, dict) else None
	trained_counts_raw = analysis.get("trained_counts")
	trained_counts = counts_to_model(trained_counts_raw) if isinstance(trained_counts_raw, dict) else None
	# Use shape detection counts directly - simpler and more reliable
	final_counts = counts_to_model(analysis["counts"])
	model_results = [
		ModelDetectionResult(
			page_index=page_index,
			model="opencv+paddleocr",
			role="detection",
			counts=ocr_counts,
		),
		ModelDetectionResult(
			page_index=page_index,
			model="opencv+heuristics",
			role="verification",
			counts=vision_counts,
		),
	]
	if trained_counts is not None:
		model_results.append(
			ModelDetectionResult(
				page_index=page_index,
				model="active_learning/random_forest",
				role="detection",
				counts=trained_counts,
			),
		)
	if phi3_counts is not None:
		model_results.append(
			ModelDetectionResult(
				page_index=page_index,
				model="ollama/phi3",
				role="verification",
				counts=phi3_counts,
			),
		)
	return PageDetectionResult(page_index=page_index, counts=final_counts, model_results=model_results)


async def build_batch_analysis_item(file: UploadFile) -> BatchAnalysisItem:
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	try:
		frames, source_type = load_image_frames(file_bytes, file.filename, file.content_type)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	try:
		stage1_frames: list[FrameSummary] = []
		for page_index, frame in enumerate(frames, start=1):
			preprocessed = preprocess_image(frame)
			stage1_frames.append(
				FrameSummary(
					page_index=page_index,
					width=frame.width,
					height=frame.height,
					mode=frame.mode,
					preprocessed_width=preprocessed.width,
					preprocessed_height=preprocessed.height,
					preview_png_base64=image_to_base64_png(preprocessed),
				),
			)

		stage1 = Stage1Response(
			filename=file.filename or "uploaded-file",
			content_type=file.content_type,
			source_type=source_type,
			page_count=len(frames),
			frames=stage1_frames,
		)

		page_results = await detect_pages(frames)
		page_detection = DetectionResponse(
			filename=file.filename or "uploaded-file",
			content_type=file.content_type,
			source_type=source_type,
			page_count=len(frames),
			models_used=["opencv+paddleocr", "opencv+heuristics"],
			pages=page_results,
			industry=None,
		)
		coordinates = CoordinateDetectionResponse(**first_analysis.get("coordinates", {}))
		return BatchAnalysisItem(
			filename=file.filename or "uploaded-file",
			content_type=file.content_type,
			source_type=source_type,
			page_count=len(frames),
			result=stage1,
			detection=page_detection,
			coordinates=coordinates,
			error=None,
		)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=502, detail=str(exc)) from exc


async def detect_pages(frames: list[Image.Image]) -> list[PageDetectionResult]:
	concurrency = max(1, int(os.getenv("PAGE_ANALYZE_CONCURRENCY", "4")))
	semaphore = asyncio.Semaphore(concurrency)

	async def analyze_page(page_index: int, frame: Image.Image) -> PageDetectionResult:
		async with semaphore:
			analysis = await asyncio.to_thread(analyze_pid_image, frame)
			trained_counts, _scored = _predict_trained_model_counts(frame, list(analysis.get("detections", [])))
			analysis["trained_counts"] = trained_counts
			return counts_from_analysis_page(analysis, page_index)

	results = await asyncio.gather(*(analyze_page(page_index, frame) for page_index, frame in enumerate(frames, start=1)))
	return list(results)


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok", "stage": "local-opencv-paddleocr"}


@app.post("/upload", response_model=Stage1Response)
async def upload_stage1(file: UploadFile = File(...)) -> Stage1Response:
	return await build_stage1_response(file)


@app.post("/upload_batch", response_model=BatchStage1Response)
async def upload_stage1_batch(files: list[UploadFile] = File(...)) -> BatchStage1Response:
	results = await asyncio.gather(*(build_stage1_response(file) for file in files))
	return BatchStage1Response(files=list(results))


@app.post("/detect", response_model=DetectionResponse)
async def detect_components(
	file: UploadFile = File(...),
	industry: str = Form(None),
	components_json: str = Form(None)
) -> DetectionResponse:
	components: list[ComponentData] | None = None
	if components_json:
		try:
			components = [ComponentData(**comp) for comp in json.loads(components_json)]
		except Exception:
			components = None
	return await build_detection_response(file, industry, components)


@app.post("/analyze_fast", response_model=dict[str, DetectionResponse | CoordinateDetectionResponse])
async def analyze_fast(
	file: UploadFile = File(...),
	industry: str = Form(None),
	components_json: str = Form(None)
) -> dict[str, DetectionResponse | CoordinateDetectionResponse]:
	"""Fast analysis endpoint that runs detection and coordinate calculation in parallel with 35-second timeout."""
	components: list[ComponentData] | None = None
	if components_json:
		try:
			components = [ComponentData(**comp) for comp in json.loads(components_json) if (comp or {}).get("name")]
		except Exception:
			components = None

	if components:
		saved_samples = await asyncio.to_thread(_persist_component_library_samples, components)
		if saved_samples > 0 and COMPONENT_LIBRARY_AUTO_TRAIN:
			# Train synchronously when components are provided to ensure model uses new annotations
			try:
				await asyncio.wait_for(asyncio.to_thread(active_learning.train_model), timeout=20.0)
				# Invalidate cache to ensure new model is loaded
				active_learning.invalidate_model_cache()
				logger.info(f"Synchronously trained model on {saved_samples} new component samples")
			except asyncio.TimeoutError:
				logger.warning(f"Synchronous training timed out after 20s, proceeding with existing model")
			except Exception as exc:
				logger.warning(f"Synchronous training failed: {exc}")
		else:
			logger.info(f"Saved {saved_samples} component samples but auto-train is disabled")
	
	# Run detection with timeout
	try:
		detection_response, coordinate_response = await asyncio.wait_for(
			build_detection_and_coordinate_response(file, industry, components),
			timeout=FAST_ANALYSIS_TIMEOUT_SECONDS
		)
	except asyncio.TimeoutError:
		logger.error(f"Analysis timed out after {FAST_ANALYSIS_TIMEOUT_SECONDS}s")
		raise HTTPException(status_code=504, detail=f"Analysis timed out after {FAST_ANALYSIS_TIMEOUT_SECONDS} seconds")
	
	return {
		"detection": detection_response,
		"coordinates": coordinate_response
	}


@app.post("/detect_batch", response_model=BatchDetectionResponse)
async def detect_components_batch(files: list[UploadFile] = File(...)) -> BatchDetectionResponse:
	results = await asyncio.gather(*(build_detection_response(file) for file in files))
	return BatchDetectionResponse(files=list(results))


@app.post("/coordinates", response_model=CoordinateDetectionResponse)
async def detect_component_coordinates(file: UploadFile = File(...)) -> CoordinateDetectionResponse:
	return await build_coordinate_response(file)


@app.post("/coordinates_batch", response_model=BatchCoordinateDetectionResponse)
async def detect_component_coordinates_batch(files: list[UploadFile] = File(...)) -> BatchCoordinateDetectionResponse:
	results = await asyncio.gather(*(build_coordinate_response(file) for file in files))
	return BatchCoordinateDetectionResponse(files=list(results))


@app.post("/analyze_batch", response_model=BatchAnalysisResponse)
async def analyze_batch(files: list[UploadFile] = File(...)) -> BatchAnalysisResponse:
	concurrency = max(1, int(os.getenv("FILE_ANALYZE_CONCURRENCY", "4")))
	semaphore = asyncio.Semaphore(concurrency)

	async def analyze_file(file: UploadFile) -> BatchAnalysisItem:
		async with semaphore:
			try:
				return await build_batch_analysis_item(file)
			except HTTPException as exc:
				return BatchAnalysisItem(
					filename=file.filename or "uploaded-file",
					content_type=file.content_type,
					source_type="image" if not is_pdf(file.filename, file.content_type) else "pdf",
					page_count=1,
					error=str(exc.detail),
				)
			except Exception as exc:  # noqa: BLE001
				return BatchAnalysisItem(
					filename=file.filename or "uploaded-file",
					content_type=file.content_type,
					source_type="image" if not is_pdf(file.filename, file.content_type) else "pdf",
					page_count=1,
					error=str(exc),
				)

	results = await asyncio.gather(*(analyze_file(file) for file in files))
	return BatchAnalysisResponse(files=list(results))


@app.post("/active_candidates")
async def active_candidates(file: UploadFile = File(...), max_results: int = 25) -> dict[str, Any]:
	"""Return candidate detections ranked by uncertainty for human annotation."""
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	try:
		frames, _ = load_image_frames(file_bytes, file.filename, file.content_type)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	results = []
	for page_index, frame in enumerate(frames, start=1):
		try:
			analysis = await asyncio.to_thread(analyze_pid_image, frame)
			detections = analysis.get("detections", [])
			image_array = np.array(frame.convert("RGB"))
			candidates = []
			for d in detections:
				c = {
					"bbox": d.get("bbox"),
					"confidence": d.get("confidence"),
					"name": d.get("name"),
					"vertex_count": d.get("vertex_count", 0),
				}
				candidates.append(c)
			scored = active_learning.predict_candidates(image_array, candidates)
			scored_sorted = sorted(scored, key=lambda x: x.get("uncertainty", 1.0), reverse=True)[:max_results]
			results.append({"page_index": page_index, "candidates": scored_sorted})
		except Exception as exc:  # noqa: BLE001
			results.append({"page_index": page_index, "error": str(exc)})

	return {"filename": file.filename or "uploaded-file", "pages": results}


@app.post("/retrain")
async def retrain_model() -> dict[str, Any]:
	"""Retrain the active learning model from collected annotations."""
	try:
		result = active_learning.train_model()
		return result
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest) -> dict[str, str]:
	"""Append human-corrected counts to a feedback file for future fine-tuning."""
	try:
		feedback_path = BACKEND_ROOT / "feedback_corrections.jsonl"
		entry = {
			"timestamp": datetime.utcnow().isoformat() + "Z",
			"filename": feedback.filename,
			"page_index": feedback.page_index,
			"counts": feedback.counts,
			"industry": feedback.industry,
			"notes": feedback.notes,
		}
		feedback_path.parent.mkdir(parents=True, exist_ok=True)
		with open(feedback_path, "a", encoding="utf-8") as fh:
			fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
		return {"status": "saved"}
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/diagnostics")
async def diagnostics(file: UploadFile = File(...)) -> dict[str, Any]:
	"""Return detailed detection outputs (OCR, visual detections, LLM verifier outputs) for debugging."""
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	try:
		frames, source_type = load_image_frames(file_bytes, file.filename, file.content_type)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	analyses: list[dict[str, Any]] = []
	for page_index, frame in enumerate(frames, start=1):
		try:
			analysis = await asyncio.to_thread(analyze_pid_image, frame)
		except Exception as exc:  # noqa: BLE001
			analysis = {"error": str(exc)}
		analyses.append({"page_index": page_index, "analysis": analysis})

	return {
		"filename": file.filename or "uploaded-file",
		"content_type": file.content_type,
		"source_type": source_type,
		"page_count": len(frames),
		"analyses": analyses,
	}
