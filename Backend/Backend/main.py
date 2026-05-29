from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
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

ML_COUNT_MIN_CONFIDENCE = float(os.getenv("ML_COUNT_MIN_CONFIDENCE", "0.55"))

app = FastAPI(title="Sarla P&ID Backend", version="0.3.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


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


class DetectionResponse(BaseModel):
	filename: str
	content_type: str | None
	source_type: Literal["image", "pdf"]
	page_count: int = Field(..., ge=1)
	models_used: list[str]
	pages: list[PageDetectionResult]
	industry: str | None = None


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
		return {key: 0 for key in local_detection.COUNT_KEYS}, []

	image_array = np.array(frame.convert("RGB"))
	scored = active_learning.predict_candidates(image_array, detections, model_blob=model_blob)
	counts = {key: 0 for key in local_detection.COUNT_KEYS}
	for candidate in scored:
		predicted = candidate.get("predicted")
		prob = float(candidate.get("prob", 0.0) or 0.0)
		fallback_category = candidate.get("category")
		fallback_conf = float(candidate.get("confidence", 0.0) or 0.0)
		if predicted in counts and prob >= ML_COUNT_MIN_CONFIDENCE:
			counts[predicted] += 1
		elif fallback_category in counts and fallback_conf >= local_detection.CONF_THRESH.get(fallback_category, 0.0):
			counts[fallback_category] += 1
	return counts, scored


def _merge_count_dicts(*count_dicts: dict[str, int]) -> dict[str, int]:
	merged = {key: 0 for key in local_detection.COUNT_KEYS}
	for counts in count_dicts:
		for key in local_detection.COUNT_KEYS:
			merged[key] = max(merged[key], int(counts.get(key, 0)))
	return merged


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


async def build_detection_response(file: UploadFile) -> DetectionResponse:
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

	industry = "Unknown"
	models_used = ["opencv+paddleocr", "opencv+heuristics"]
	if _active_model_blob() is not None:
		models_used.append("active_learning/random_forest")
	if frames:
		try:
			analysis = await asyncio.to_thread(analyze_pid_image, frames[0])
			industry = analysis["industry"]
			if bool(analysis.get("used_ollama")):
				models_used.append("ollama/phi3")
		except Exception:  # noqa: BLE001
			industry = "Unknown"

	return DetectionResponse(
		filename=file.filename or "uploaded-file",
		content_type=file.content_type,
		source_type=source_type,
		page_count=len(frames),
		models_used=models_used,
		pages=pages,
		industry=industry,
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
	final_counts = counts_to_model(analysis["counts"])
	phi3_counts_raw = analysis.get("phi3_counts")
	phi3_counts = counts_to_model(phi3_counts_raw) if isinstance(phi3_counts_raw, dict) else None
	trained_counts_raw = analysis.get("trained_counts")
	trained_counts = counts_to_model(trained_counts_raw) if isinstance(trained_counts_raw, dict) else None
	if trained_counts is not None:
		final_counts = counts_to_model(_merge_count_dicts(model_counts_to_dict(final_counts), trained_counts_raw))
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
		first_analysis = await asyncio.to_thread(analyze_pid_image, frames[0])
		page_detection = DetectionResponse(
			filename=file.filename or "uploaded-file",
			content_type=file.content_type,
			source_type=source_type,
			page_count=len(frames),
			models_used=["opencv+paddleocr", "opencv+heuristics"] + (["ollama/phi3"] if bool(first_analysis.get("used_ollama")) else []),
			pages=page_results,
			industry=first_analysis.get("industry") or "Unknown",
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
async def detect_components(file: UploadFile = File(...)) -> DetectionResponse:
	return await build_detection_response(file)


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
