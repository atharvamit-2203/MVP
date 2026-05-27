from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import cv2
import fitz
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field


BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")

app = FastAPI(title="Sarla P&ID Backend", version="0.2.0")

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
	text: int = Field(0, ge=0)
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


CategoryCounts.model_rebuild()
ModelDetectionResult.model_rebuild()
PageDetectionResult.model_rebuild()
DetectionResponse.model_rebuild()


def is_pdf(filename: str | None, content_type: str | None) -> bool:
	if content_type == "application/pdf":
		return True
	return bool(filename and filename.lower().endswith(".pdf"))


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


def resize_for_model(image: Image.Image, max_edge: int = 1600) -> Image.Image:
	prepared = image.convert("RGB")
	prepared.thumbnail((max_edge, max_edge))
	return prepared


def get_openrouter_config() -> dict[str, str]:
	api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
	if not api_key:
		raise HTTPException(
			status_code=500,
			detail="OPENROUTER_API_KEY is missing. Add it to Backend/.env before using detection.",
		)

	return {
		"api_key": api_key,
		"base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
		"qwen_model": os.getenv("OPENROUTER_QWEN_MODEL", "qwen/qwen2.5-vl-72b-instruct"),
		"claude_model": os.getenv("OPENROUTER_CLAUDE_MODEL", "anthropic/claude-sonnet-4"),
		"site_url": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
		"app_name": os.getenv("OPENROUTER_APP_NAME", "Sarla P&ID Detector"),
		"qwen_max_tokens": os.getenv("OPENROUTER_QWEN_MAX_TOKENS", "800"),
		"claude_max_tokens": os.getenv("OPENROUTER_CLAUDE_MAX_TOKENS", "400"),
	}


def parse_json_payload(raw_content: str) -> dict[str, Any]:
	trimmed = raw_content.strip()
	for candidate in (
		trimmed,
		re.sub(r"^```(?:json)?\s*|\s*```$", "", trimmed, flags=re.IGNORECASE | re.DOTALL),
	):
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, dict):
				if all(key in parsed for key in ("text", "motor", "pump", "tank", "valve")):
					return parsed
				if any(key in parsed for key in ("components", "detections", "items")):
					return parsed
				if any(key in parsed for key in ("category", "label", "name")):
					return {"components": [parsed]}
				return parsed
			if isinstance(parsed, list):
				return {"components": parsed}
		except json.JSONDecodeError:
			continue

	json_start_tokens = ["[", "{"]
	for start_token in json_start_tokens:
		start_index = trimmed.find(start_token)
		if start_index == -1:
			continue
		end_token = "}" if start_token == "{" else "]"
		end_index = trimmed.rfind(end_token)
		if end_index != -1 and end_index > start_index:
			fragment = trimmed[start_index : end_index + 1]
			try:
				parsed = json.loads(fragment)
				if isinstance(parsed, dict):
					if all(key in parsed for key in ("text", "motor", "pump", "tank", "valve")):
						return parsed
					if any(key in parsed for key in ("components", "detections", "items")):
						return parsed
					if any(key in parsed for key in ("category", "label", "name")):
						return {"components": [parsed]}
					return parsed
				if isinstance(parsed, list):
					return {"components": parsed}
			except json.JSONDecodeError:
				continue

	start_index = trimmed.find("{")
	end_index = trimmed.rfind("}")
	if start_index != -1 and end_index != -1 and end_index > start_index:
		fragment = trimmed[start_index : end_index + 1]
		try:
			parsed = json.loads(fragment)
			if isinstance(parsed, dict):
				if all(key in parsed for key in ("text", "motor", "pump", "tank", "valve")):
					return parsed
				if any(key in parsed for key in ("components", "detections", "items")):
					return parsed
				if any(key in parsed for key in ("category", "label", "name")):
					return {"components": [parsed]}
				return parsed
			if isinstance(parsed, list):
				return {"components": parsed}
		except json.JSONDecodeError as exc:
			raise ValueError(f"Model response did not contain valid JSON. Raw output: {raw_content[:1000]}") from exc

	raise ValueError(f"Model response did not contain valid JSON. Raw output: {raw_content[:1000]}")


def coerce_message_content(raw_content: Any) -> str:
	if isinstance(raw_content, str):
		return raw_content

	if isinstance(raw_content, dict):
		if "content" in raw_content:
			return coerce_message_content(raw_content["content"])
		return json.dumps(raw_content)

	if isinstance(raw_content, list):
		parts: list[str] = []
		for part in raw_content:
			if isinstance(part, str):
				parts.append(part)
			elif isinstance(part, dict):
				if isinstance(part.get("text"), str):
					parts.append(part["text"])
				elif isinstance(part.get("content"), str):
					parts.append(part["content"])
		return "\n".join(parts)

	return str(raw_content)


def empty_counts() -> CategoryCounts:
	return CategoryCounts(text=0, motor=0, pump=0, tank=0, valve=0)


def add_counts(left: CategoryCounts, right: CategoryCounts) -> CategoryCounts:
	return CategoryCounts(
		text=left.text + right.text,
		motor=left.motor + right.motor,
		pump=left.pump + right.pump,
		tank=left.tank + right.tank,
		valve=left.valve + right.valve,
	)


def parse_counts_from_pipe_text(raw_content: str) -> CategoryCounts:
	counts = empty_counts()
	for line in raw_content.splitlines():
		line = line.strip().lower()
		if not line:
			continue
		for category in ("text", "motor", "pump", "tank", "valve"):
			if category in line:
				match = re.search(r"(\d+)", line)
				if match:
					value = int(match.group(1))
					setattr(counts, category, value)
				break
	return counts


def normalize_counts_payload(raw_payload: Any) -> CategoryCounts:
	if isinstance(raw_payload, dict):
		if "counts" in raw_payload and isinstance(raw_payload["counts"], dict):
			raw_payload = raw_payload["counts"]
		elif any(key in raw_payload for key in ("text", "motor", "pump", "tank", "valve")):
			raw_payload = raw_payload
		elif "components" in raw_payload and isinstance(raw_payload["components"], list):
			counts = empty_counts()
			for item in raw_payload["components"]:
				if not isinstance(item, dict):
					continue
				category = str(item.get("category", "")).strip().lower()
				if category in {"text", "motor", "pump", "tank", "valve"}:
					setattr(counts, category, getattr(counts, category) + 1)
			return counts
		else:
			return empty_counts()

	if not isinstance(raw_payload, dict):
		return empty_counts()

	counts = empty_counts()
	for category in ("text", "motor", "pump", "tank", "valve"):
		value = raw_payload.get(category, 0)
		try:
			setattr(counts, category, max(0, int(value)))
		except (TypeError, ValueError):
			setattr(counts, category, 0)
	return counts


def build_detection_prompt() -> str:
	return (
		"Analyze the P&ID image and return strict JSON only. "
		"Count the visible components in these categories: text, motor, pump, tank, valve. "
		"Return exactly one JSON object with integer keys: text, motor, pump, tank, valve. "
		"Do not include labels, coordinates, notes, or any extra fields. "
		"If a category is absent, use 0."
	)


def build_verification_prompt(candidate_json: str) -> str:
	return (
		"You are verifying candidate P&ID counts from another model. "
		"Review the image and the candidate JSON. "
		"Validate and correct the counts for these categories: text, motor, pump, tank, valve. "
		"Return strict JSON only with integer keys: text, motor, pump, tank, valve. "
		f"Candidate JSON: {candidate_json}"
	)


def build_response_format() -> dict[str, Any]:
	return {
		"type": "json_schema",
		"json_schema": {
			"name": "pid_counts",
			"strict": True,
			"schema": {
				"type": "object",
				"properties": {
					"text": {"type": "integer"},
					"motor": {"type": "integer"},
					"pump": {"type": "integer"},
					"tank": {"type": "integer"},
					"valve": {"type": "integer"},
				},
				"required": ["text", "motor", "pump", "tank", "valve"],
				"additionalProperties": False,
			},
		},
	}


def extract_message_text(message: Any) -> str:
	if isinstance(message, dict):
		for key in ("content", "output_text", "parsed", "text"):
			if key in message:
				candidate = message[key]
				if isinstance(candidate, str):
					return candidate
				if isinstance(candidate, (dict, list)):
					return coerce_message_content(candidate)

		for tool_call_key in ("tool_calls", "function_call"):
			if tool_call_key in message:
				candidate = message[tool_call_key]
				if isinstance(candidate, list):
					for tool_call in candidate:
						if not isinstance(tool_call, dict):
							continue
						function_block = tool_call.get("function")
						if isinstance(function_block, dict):
							for nested_key in ("arguments", "content", "text"):
								if isinstance(function_block.get(nested_key), str):
									return function_block[nested_key]
						for nested_key in ("arguments", "content", "text"):
							if isinstance(tool_call.get(nested_key), str):
								return tool_call[nested_key]
				elif isinstance(candidate, dict):
					function_block = candidate.get("function")
					if isinstance(function_block, dict):
						for nested_key in ("arguments", "content", "text"):
							if isinstance(function_block.get(nested_key), str):
								return function_block[nested_key]
					for nested_key in ("arguments", "content", "text"):
						if isinstance(candidate.get(nested_key), str):
							return candidate[nested_key]

	return coerce_message_content(message)


def call_openrouter_model(
	image: Image.Image,
	model_name: str,
	config: dict[str, str],
	page_index: int,
	role: Literal["detection", "verification"],
	candidate_json: str | None = None,
) -> ModelDetectionResult:
	prepared_image = resize_for_model(image)
	encoded_image = image_to_base64_png(prepared_image)
	if role == "detection":
		user_text = build_detection_prompt()
	else:
		user_text = build_verification_prompt(candidate_json or "{\"components\": []}")
	payload = {
		"model": model_name,
		"max_tokens": int(config["qwen_max_tokens"] if role == "detection" else config["claude_max_tokens"]),
		"response_format": build_response_format(),
		"messages": [
			{
				"role": "system",
				"content": (
					"You are a precise multimodal P&ID component detector. "
					"When role is detection, follow the format instructions precisely. "
					"When role is verification, output JSON only."
				),
			},
			{
				"role": "user",
				"content": [
					{"type": "text", "text": user_text},
					{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
				],
			},
		],
		"temperature": 0,
	}

	headers = {
		"Authorization": f"Bearer {config['api_key']}",
		"Content-Type": "application/json",
		"HTTP-Referer": config["site_url"],
		"X-Title": config["app_name"],
	}

	response = requests.post(
		f"{config['base_url']}/chat/completions",
		headers=headers,
		json=payload,
		timeout=180,
	)
	if response.status_code >= 400:
		raise ValueError(f"OpenRouter model {model_name} failed: {response.status_code} {response.text}")

	response_json = response.json()
	choices = response_json.get("choices") or []
	if not choices:
		raise ValueError(f"OpenRouter model {model_name} returned no choices.")

	message = choices[0].get("message", {})
	raw_content = extract_message_text(message)

	try:
		parsed = parse_json_payload(raw_content)
		counts = normalize_counts_payload(parsed)
	except ValueError:
		counts = parse_counts_from_pipe_text(raw_content)

	return ModelDetectionResult(page_index=page_index, model=model_name, role=role, counts=counts)


def merge_model_results(model_results: list[ModelDetectionResult]) -> list[Any]:
	return []


async def detect_pages(frames: list[Image.Image]) -> list[PageDetectionResult]:
	config = get_openrouter_config()
	qwen_model = config["qwen_model"]
	claude_model = config["claude_model"]
	pages: list[PageDetectionResult] = []

	for page_index, frame in enumerate(frames, start=1):
		qwen_result = await asyncio.to_thread(call_openrouter_model, frame, qwen_model, config, page_index, "detection")
		candidate_json = json.dumps(qwen_result.counts.model_dump())
		claude_result = await asyncio.to_thread(
			call_openrouter_model,
			frame,
			claude_model,
			config,
			page_index,
			"verification",
			candidate_json,
		)
		model_results = [qwen_result, claude_result]
		pages.append(
			PageDetectionResult(
				page_index=page_index,
				counts=claude_result.counts,
				model_results=model_results,
			),
		)

	return pages


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok", "stage": "stage-1-image-input"}


@app.post("/upload", response_model=Stage1Response)
async def upload_stage1(file: UploadFile = File(...)) -> Stage1Response:
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
			)
		)

	return Stage1Response(
		filename=file.filename or "uploaded-file",
		content_type=file.content_type,
		source_type=source_type,
		page_count=len(frames),
		frames=frame_summaries,
	)


@app.post("/detect", response_model=DetectionResponse)
async def detect_components(file: UploadFile = File(...)) -> DetectionResponse:
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	try:
		frames, source_type = load_image_frames(file_bytes, file.filename, file.content_type)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	try:
		pages = await detect_pages(frames)
	except HTTPException:
		raise
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=502, detail=str(exc)) from exc

	config = get_openrouter_config()
	return DetectionResponse(
		filename=file.filename or "uploaded-file",
		content_type=file.content_type,
		source_type=source_type,
		page_count=len(frames),
		models_used=[f"{config['qwen_model']} (detection)", f"{config['claude_model']} (verification)"],
		pages=pages,
	)
