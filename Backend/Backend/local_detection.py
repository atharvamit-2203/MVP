from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import json
from pathlib import Path
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Work around Paddle oneDNN/PIR executor crashes seen on Windows CPU builds.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_pir_apply_inplace_pass"] = "0"

try:
	import easyocr
except Exception as exc:  # noqa: BLE001
	easyocr = None
	EASYOCR_IMPORT_ERROR = exc
else:
	EASYOCR_IMPORT_ERROR = None


COUNT_KEYS = ("motor", "pump", "tank", "valve")
CATEGORY_TO_TYPE = {
	"text": "ia.symbol.text",
	"motor": "ia.symbol.motor",
	"pump": "ia.symbol.pump",
	"tank": "ia.symbol.tank",
	"valve": "ia.symbol.valve",
	"other": "ia.symbol.other",
}

INDUSTRY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
	("Water Treatment", ("water treatment", "wastewater", "effluent", "sewage", "clarifier", "sludge")),
	("Oil & Gas", ("oil and gas", "oil & gas", "refinery", "crude", "pipeline", "gas")),
	("Chemical Processing", ("chemical", "acid", "alkali", "solvent", "reactor", "distillation")),
	("Pharmaceutical", ("pharma", "pharmaceutical", "sterile", "tablet", "bioreactor")),
	("Food & Beverage", ("food", "beverage", "dairy", "brew", "syrup", "juice")),
	("Power Generation", ("power", "boiler", "steam", "turbine", "generator")),
	("Manufacturing", ("manufacturing", "plant", "process", "production")),
]

TEXT_CATEGORY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
	("motor", ("mtr", "m-", "mo-", "motor-")),
	("pump", ("p-", "pu-", "pmp", "pump-")),
	("tank", ("tk-", "t-", "vessel-", "tank-", "column")),
	("valve", ("xv", "cv", "hv", "lv", "sv", "pv", "tv", "gv", "bv", "wv", "pcv", "fcv", "lcv", "tcv", "psv", "nrv", "sdv", "mov", "sov")),
]

# Simple initial-letter mapping for compact P&ID tags (e.g. 'm123' -> motor)
INITIAL_PREFIX_MAP: dict[str, str] = {
    "m": "motor",
    "p": "pump",
    "t": "tank",
    "v": "valve",
}

CATEGORY_REGEX_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
	(
		"valve",
		(
			r"\b(?:check\s*valve|gate\s*valve|globe\s*valve|ball\s*valve|butterfly\s*valve|plug\s*valve)\b",
			r"\b(?:pcv|fcv|lcv|tcv|psv|nrv|sdv|xv|hv|lv|fv|sv|cv|tv|pv|mov|sov|bv|gv|wv)\b",
			r"\b(?:v|xv|cv|hv|lv|sv|pv|tv|gv|bv|wv)-?\d{1,5}[a-z]?\b",
			r"\b(?:v|xv|cv|hv|lv|sv|pv|tv|gv|bv|wv)\d{1,5}[a-z]?\b",
			r"\bvalve\b",
		),
	),
	("pump", (r"\bp-?\d{2,5}[a-z]?\b", r"\bpu-?\d{2,5}[a-z]?\b", r"\bpmp-?\d{1,5}[a-z]?\b", r"\bpump\b")),
	("motor", (r"\bm-?\d{2,5}[a-z]?\b", r"\bmo-?\d{2,5}[a-z]?\b", r"\bmtr-?\d{1,5}[a-z]?\b", r"\bmotor\b")),
	("tank", (r"\b(?:tk|t|v)-?\d{2,5}[a-z]?\b", r"\btk-?\d{1,5}[a-z]?\b", r"\btank\b", r"\bvessel\b", r"\bcolumn\b")),
]

OCR_MIN_TEXT_CONFIDENCE = float(os.getenv("OCR_MIN_TEXT_CONFIDENCE", "0.2"))
OCR_MIN_COMPONENT_AREA = int(os.getenv("OCR_MIN_COMPONENT_AREA", "100"))
PADDLEOCR_LANG = os.getenv("PADDLEOCR_LANG", "en")
PADDLEOCR_USE_GPU = os.getenv("PADDLEOCR_USE_GPU", "false").strip().lower() in {"1", "true", "yes", "on"}
# Force disable Ollama to prevent timeout issues - override any env settings
OLLAMA_ENABLED = False
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
# Comma-separated list of Ollama models to consult (e.g. "phi3,llama2")
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", OLLAMA_MODEL)
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "25"))
# Shorter timeout used for the fast analysis path.
OLLAMA_FAST_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_FAST_TIMEOUT_SECONDS", "8"))
# When false, Ollama is skipped in the count path for speed and determinism.
OLLAMA_USE_FOR_COUNTS = os.getenv("OLLAMA_USE_FOR_COUNTS", "true").strip().lower() in {"1", "true", "yes", "on"}

# Minimum confidence required to count a visual detection for each category.
# Minimum confidence required to count a visual detection for each category.
CONF_THRESH: dict[str, float] = {
    "motor": 0.65,
    "pump": 0.70,
    "tank": 0.55,
    "valve": 0.30,
}


def empty_counts() -> dict[str, int]:
	return {key: 0 for key in COUNT_KEYS}


def clamp(value: int, lower: int, upper: int) -> int:
	return max(lower, min(upper, value))


def normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip().lower()


def bbox_from_points(points: Any) -> tuple[int, int, int, int]:
	array = np.array(points, dtype=np.float32)
	if array.ndim != 2 or array.shape[0] < 4:
		raise ValueError("Invalid OCR polygon.")
	x, y, width, height = cv2.boundingRect(array.astype(np.int32))
	return int(x), int(y), int(width), int(height)


def bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
	x, y, width, height = box
	return x + width / 2.0, y + height / 2.0


def bbox_area(box: tuple[int, int, int, int]) -> int:
	return max(0, box[2]) * max(0, box[3])


def prepare_ocr_image(image_array: np.ndarray, fast_mode: bool = False) -> np.ndarray:
	"""Upscale and enhance the image before OCR to improve small tag recall."""
	h, w = image_array.shape[:2]
	max_edge = max(h, w)
	prepared = image_array
	target_edge = 1600 if fast_mode else 2400
	if max_edge < target_edge:
		scale = float(target_edge) / max_edge
		prepared = cv2.resize(image_array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(prepared, cv2.COLOR_RGB2GRAY)
	
	# Denoise for bad quality images
	gray = cv2.fastNlMeansDenoising(gray, h=10)
	
	clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
	boosted = clahe.apply(gray)
	
	# Unsharp masking for clearer text
	gaussian = cv2.GaussianBlur(boosted, (0, 0), 2.0)
	sharpened = cv2.addWeighted(boosted, 1.5, gaussian, -0.5, 0)
	
	return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
	ax1, ay1, aw, ah = box_a
	bx1, by1, bw, bh = box_b
	ax2, ay2 = ax1 + aw, ay1 + ah
	bx2, by2 = bx1 + bw, by1 + bh

	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)
	if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
		return 0.0

	inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
	area_a = aw * ah
	area_b = bw * bh
	denominator = area_a + area_b - inter_area
	if denominator <= 0:
		return 0.0
	return inter_area / denominator


@lru_cache(maxsize=1)
def get_ocr_engine() -> Any:
	if easyocr is None:
		raise RuntimeError(
			"easyocr is not available. Install easyocr in the current Python environment."
		) from EASYOCR_IMPORT_ERROR
	return easyocr.Reader([PADDLEOCR_LANG], gpu=False, verbose=False)


@lru_cache(maxsize=1)
def get_easyocr_engine() -> Any:
	if easyocr is None:
		raise RuntimeError(
			"easyocr is not available. Install easyocr in the current Python environment."
		) from EASYOCR_IMPORT_ERROR
	return easyocr.Reader([PADDLEOCR_LANG], gpu=False, verbose=False)


@lru_cache(maxsize=1)
def get_available_ollama_models() -> list[str]:
	"""Query the Ollama server for available models and return a list of model names.

	Returns an empty list on failure or if the server returns no models.
	"""
	try:
		resp = requests.get(f"{OLLAMA_BASE_URL}/api/models", timeout=5)
		resp.raise_for_status()
		body = resp.json()
		models: list[str] = []
		if isinstance(body, list):
			for item in body:
				if isinstance(item, str):
					models.append(item)
				elif isinstance(item, dict):
					name = item.get("name") or item.get("model") or item.get("id")
					if name:
						models.append(str(name))
		elif isinstance(body, dict):
			candidates = body.get("models") or body.get("results") or []
			for item in candidates:
				if isinstance(item, str):
					models.append(item)
				elif isinstance(item, dict):
					name = item.get("name") or item.get("model") or item.get("id")
					if name:
						models.append(str(name))
		return models
	except Exception:
		return []


def run_ocr(engine: Any, image_array: np.ndarray) -> Any:
	"""Run OCR using the active OCR engine."""
	return engine.readtext(image_array)


def flatten_ocr_result(raw_result: Any) -> list[tuple[Any, str, float]]:
	"""Normalize PaddleOCR outputs to a flat list of (box, text, confidence)."""
	entries: list[tuple[Any, str, float]] = []
	if not raw_result:
		return entries

	if isinstance(raw_result, tuple):
		raw_result = raw_result[0]

	# Common old format: [[ [box, (text, conf)], ... ]]
	if isinstance(raw_result, list):
		if raw_result and isinstance(raw_result[0], list) and raw_result[0] and isinstance(raw_result[0][0], (list, tuple)):
			candidates = raw_result[0]
		else:
			candidates = raw_result

		for item in candidates:
			# Newer predict-style item can be dict-like with polygons/text arrays.
			if isinstance(item, dict):
				polys = item.get("dt_polys") or item.get("rec_polys") or []
				texts = item.get("rec_texts") or []
				scores = item.get("rec_scores") or []
				for index, text in enumerate(texts):
					box = polys[index] if index < len(polys) else None
					if box is None:
						continue
					try:
						confidence = float(scores[index]) if index < len(scores) else 0.0
					except (TypeError, ValueError):
						confidence = 0.0
					entries.append((box, str(text), confidence))
				continue

			# Old ocr format entry: [box, (text, conf)]
			if isinstance(item, (list, tuple)) and len(item) >= 2:
				box = item[0]
				info = item[1]
				if isinstance(info, (list, tuple)) and info:
					text = str(info[0]) if info[0] is not None else ""
					try:
						confidence = float(info[1]) if len(info) > 1 else 0.0
					except (TypeError, ValueError):
						confidence = 0.0
				else:
					text = str(info)
					confidence = 0.0
				entries.append((box, text, confidence))

	return entries


def run_easyocr(image_array: np.ndarray) -> list[tuple[Any, str, float]]:
	"""Fallback OCR path used only when PaddleOCR fails at runtime."""
	reader = get_easyocr_engine()
	results = reader.readtext(image_array)
	entries: list[tuple[Any, str, float]] = []
	for item in results:
		if not isinstance(item, (list, tuple)) or len(item) < 3:
			continue
		box = item[0]
		text = str(item[1])
		try:
			confidence = float(item[2])
		except (TypeError, ValueError):
			confidence = 0.0
		entries.append((box, text, confidence))
	return entries


def merge_candidates(*candidate_groups: list[tuple[Any, str, float]]) -> list[tuple[Any, str, float]]:
	merged: list[tuple[Any, str, float]] = []
	seen: set[tuple[str, str]] = set()
	for group in candidate_groups:
		for box, text, confidence in group:
			try:
				bbox = bbox_from_points(box)
			except ValueError:
				continue
			key = (normalize_text(text), f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}")
			if key in seen:
				continue
			seen.add(key)
			merged.append((box, text, confidence))
	return merged


def extract_ocr_detections(image_array: np.ndarray, fast_mode: bool = False) -> list[dict[str, Any]]:
	engine = get_ocr_engine()
	primary_image = prepare_ocr_image(image_array, fast_mode=fast_mode)
	primary_raw = run_ocr(engine, primary_image)
	primary_candidates = flatten_ocr_result(primary_raw)
	secondary_candidates: list[tuple[Any, str, float]] = []
	if not fast_mode:
		# A second pass on inverted contrast often recovers faint tags and small valve labels.
		inverted = 255 - primary_image
		try:
			secondary_raw = run_ocr(engine, inverted)
			secondary_candidates = flatten_ocr_result(secondary_raw)
		except Exception:
			secondary_candidates = []
	candidates = merge_candidates(primary_candidates, secondary_candidates)

	detections: list[dict[str, Any]] = []
	for box, text, confidence in candidates:
		try:
			bbox = bbox_from_points(box)
		except ValueError:
			continue
		clean_text = text.strip()
		if not clean_text:
			continue
		if confidence < OCR_MIN_TEXT_CONFIDENCE:
			continue
		detections.append(
			{
				"text": clean_text,
				"normalized_text": normalize_text(clean_text),
				"confidence": confidence,
				"bbox": bbox,
				"center": bbox_center(bbox),
			},
		)
	return detections


def classify_text_label(text: str) -> str | None:
	normalized = normalize_text(text)
	if not normalized:
		return None
	for category, patterns in CATEGORY_REGEX_PATTERNS:
		if any(re.search(pattern, normalized) for pattern in patterns):
			return category
	for category, patterns in TEXT_CATEGORY_PATTERNS:
		if any(pattern in normalized for pattern in patterns):
			return category
	# Fallback: check for single-letter initial tags like 'm-123', 'p123', 't 45'
	initial_candidate = _infer_category_from_initial(normalized)
	if initial_candidate:
		return initial_candidate
	return None


def _infer_category_from_initial(text: str) -> str | None:
	if not text:
		return None
	# take first alpha char
	first = None
	for ch in text:
		if ch.isalpha():
			first = ch
			break
		if ch.isdigit():
			break
	if not first:
		return None
	first = first.lower()
	mapped = INITIAL_PREFIX_MAP.get(first)
	if not mapped:
		return None
	# Accept if the text is short or starts with letter+digit/hyphen (common P&ID tags)
	if len(text) <= 4 or (len(text) > 1 and (text[1].isdigit() or text[1] in "-_")):
		return mapped
	return None


def infer_industry_from_text(text_blob: str) -> str:
	normalized = normalize_text(text_blob)
	for industry, patterns in INDUSTRY_PATTERNS:
		if any(pattern in normalized for pattern in patterns):
			return industry
	return "Unknown"


def extract_counts_from_text(text_blob: str) -> dict[str, int]:
	"""Deterministic token-based extraction from OCR text to suggest counts.

	This is conservative: it searches for common P&ID tokens and returns
	minimal counts derived from explicit tokens (e.g., P-123 -> pump).
	"""
	text = normalize_text(text_blob or "")
	counts = empty_counts()

	# Pumps: explicit tags only (avoid counting plain text labels like "pump")
	# Require at least 2 digits to avoid false matches
	pumps = re.findall(r"\bp-?\d{2,5}[a-z]?\b", text)
	pumps += re.findall(r"\bpu-?\d{2,5}[a-z]?\b", text)
	pumps += re.findall(r"\bpmp-?\d{2,5}[a-z]?\b", text)
	# Filter out matches that are part of longer words
	pumps = [p for p in pumps if len(p) <= 10]
	counts["pump"] = max(counts["pump"], len(set(pumps)))

	# Motors: explicit tags only
	motors = re.findall(r"\bm-?\d{2,5}[a-z]?\b", text)
	motors += re.findall(r"\bmo-?\d{2,5}[a-z]?\b", text)
	motors += re.findall(r"\bmtr-?\d{2,5}[a-z]?\b", text)
	# Filter out matches that are part of longer words
	motors = [m for m in motors if len(m) <= 10]
	counts["motor"] = max(counts["motor"], len(set(motors)))

	# Tanks: explicit tags only
	tanks = re.findall(r"\bt-?\d{2,5}[a-z]?\b", text)
	tanks += re.findall(r"\btk-?\d{2,5}[a-z]?\b", text)
	tanks += re.findall(r"\bv-?\d{2,5}[a-z]?\b", text)
	# Filter out matches that are part of longer words
	tanks = [t for t in tanks if len(t) <= 10]
	counts["tank"] = max(counts["tank"], len(set(tanks)))

	# Valves: explicit tags only
	valves = re.findall(r"\b(?:xv|cv|hv|lv|sv|pv|tv|gv|bv|wv|pcv|fcv|lcv|tcv|psv|nrv|sdv|mov|sov)-?\d{2,5}[a-z]?\b", text)
	valves += re.findall(r"\b(?:v|xv|cv|hv|lv|sv|pv|tv|gv|bv|wv)-?\d{2,5}[a-z]?\b", text)
	# Filter out matches that are part of longer words
	valves = [v for v in valves if len(v) <= 10]
	counts["valve"] = max(counts["valve"], len(set(valves)))

	return counts


def preprocess_for_shapes(image_array: np.ndarray) -> np.ndarray:
	gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
	
	# Denoise for bad quality images while preserving edges
	gray = cv2.bilateralFilter(gray, 9, 75, 75)
	
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	adaptive = cv2.adaptiveThreshold(
		blurred,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		41,
		10,
	)
	# Close small interior gaps to preserve hollow shapes like tanks
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
	# Remove small noise
	kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2, iterations=1)
	return cleaned


def detect_text_driven_components(ocr_detections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int], str]:
	counts = empty_counts()
	text_blob_parts: list[str] = []
	components: list[dict[str, Any]] = []

	for detection in ocr_detections:
		text = detection["text"]
		text_blob_parts.append(text)
		category = classify_text_label(text)
		if category is None:
			continue
		# Only count if confidence is reasonable
		if detection.get("confidence", 0.0) < 0.3:
			continue
		counts[category] += 1
		components.append(
			{
				"name": text,
				"category": category,
				"bbox": detection["bbox"],
				"confidence": detection["confidence"],
			},
		)

	return components, counts, infer_industry_from_text(" ".join(text_blob_parts))


def nearby_ocr_texts(
	candidate_box: tuple[int, int, int, int],
	ocr_detections: list[dict[str, Any]],
	padding_ratio: float = 0.35,
) -> list[dict[str, Any]]:
	x, y, width, height = candidate_box
	padding_x = max(12, int(width * padding_ratio))
	padding_y = max(12, int(height * padding_ratio))
	expanded = (
		max(0, x - padding_x),
		max(0, y - padding_y),
		width + padding_x * 2,
		height + padding_y * 2,
	)
	matches: list[dict[str, Any]] = []
	for detection in ocr_detections:
		if iou(expanded, detection["bbox"]) > 0.0:
			matches.append(detection)
	return matches


def classify_visual_candidate(
	candidate_box: tuple[int, int, int, int],
	ocr_detections: list[dict[str, Any]],
	area: float,
	circularity: float,
	aspect_ratio: float,
	vertex_count: int,
	extent: float,
	solidity: float,
	image_area: float | None = None,
) -> tuple[str | None, str, float]:
	nearby = nearby_ocr_texts(candidate_box, ocr_detections)
	nearby_blob = " ".join(item["normalized_text"] for item in nearby)
	nearby_text = " ".join(item["text"] for item in nearby).strip()
	confidence = 0.0
	
	# Reject shape candidates that are essentially just unclassified text blobs
	for det in nearby:
		if iou(candidate_box, det["bbox"]) > 0.6:
			text_cat = classify_text_label(det.get("normalized_text", ""))
			if not text_cat:
				return None, "", 0.0

	valve_like_geometry = (
		3 <= vertex_count <= 10
		and 0.35 <= aspect_ratio <= 4.5
		and 0.08 <= circularity <= 0.75
		and 0.10 <= extent <= 0.82
		and solidity <= 0.85
	)

	if nearby_blob:
		nearby_category = classify_text_label(nearby_blob)
		if nearby_category is not None:
			# If the text is nearby (within the 35% padded box), we can trust it to classify the shape.
			# We relax the strict 20% IoU overlap requirement because P&ID labels are often adjacent.
			confidence = 0.9
			return nearby_category, nearby_text or nearby_category.title(), confidence

		# If no nearby_category, try compact-initial mapping on individual OCR tokens
		if not nearby_category:
			for det in nearby:
				try:
					initial_map = _infer_category_from_initial(det.get("normalized_text", ""))
				except Exception:
					initial_map = None
				if initial_map:
					confidence = 0.88
					return initial_map, det.get("text") or initial_map.title(), confidence

	# Strong geometry-first valve fallback.
	# Compact bow-tie / diamond / check-valve symbols often have a low solidity,
	# lower circularity, and a near-square footprint even when vertex counting is noisy.
	if (
		3 <= vertex_count <= 12
		and 0.65 <= aspect_ratio <= 1.6
		and 0.05 <= circularity <= 0.78
		and 0.08 <= extent <= 0.80
		and solidity <= 0.92
		and area <= 8000
	):
		confidence = min(0.9, 0.4 + (0.2 * (1.0 - min(abs(1.0 - aspect_ratio), 1.0))) + (0.2 if vertex_count >= 5 else 0.0) + (0.1 if solidity <= 0.7 else 0.0))
		return "valve", nearby_text or "Valve", confidence

	# Additional valve-only promotion when the text looks like a common tag.
	if re.search(r"\b(?:xv|cv|hv|lv|sv|pv|tv|gv|bv|wv|pcv|fcv|lcv|tcv|psv|nrv|sdv|mov|sov)\b", nearby_blob) and len(nearby) > 0:
		# Reject promotion for candidates that are very low-extent/low-solidity (likely a text region).
		if extent < 0.12 or solidity < 0.20:
			pass
		else:
			confidence = 0.85
			return "valve", nearby_text or "Valve", confidence

	# Geometry-first valve promotion: compact bow-tie / diamond / triangle-like symbols
	# should resolve to valve before the looser pump/motor heuristics below.
	if valve_like_geometry and area <= 6000:
		confidence = min(0.88, 0.35 + (extent * 0.25) + (solidity * 0.25) + (0.25 if vertex_count >= 5 else 0.0))
		return "valve", nearby_text or "Valve", confidence

	# Valves are often compact symbols (diamond/triangle/bow-tie) with specific geometry.
	# Tighten thresholds to reduce false positives from other small shapes.
	if 3 <= vertex_count <= 8 and 0.4 <= aspect_ratio <= 2.0 and 0.12 <= circularity <= 0.6 and 0.20 <= extent <= 0.70:
		if area <= 2000:
			confidence = min(0.8, (extent + solidity) / 2.0)
			return "valve", nearby_text or "Valve", confidence

	# Pumps are often elongated machine symbols with moderate circularity
	# Improved thresholds for pump detection
	if 1.1 <= aspect_ratio <= 6.0 and area >= 200 and extent >= 0.2 and solidity >= 0.35 and not valve_like_geometry:
		confidence = min(0.75, (extent + solidity + (1.0 / aspect_ratio)) / 3.0)
		return "pump", nearby_text or "Pump", confidence
	# 1. Explicit geometry-based checks for motors and pumps
	# Motors: typically perfect circles, moderate area
	if 0.85 <= circularity <= 1.0 and 8 <= vertex_count <= 16 and area >= 100:
		confidence = min(0.9, 0.5 + (0.4 * circularity))
		return "motor", nearby_text or "Motor", confidence

	# Pumps: often circular main body with a small attached triangle/polygon, leading to slightly lower circularity but high solidity.
	if 0.50 <= circularity <= 0.88 and 0.80 <= solidity <= 1.0 and 5 <= vertex_count <= 14 and area >= 150:
		confidence = min(0.85, 0.4 + (0.3 * circularity) + (0.2 * solidity))
		return "pump", nearby_text or "Pump", confidence

	# 2. Tanks are larger vessels — evaluate before valves to prevent large rectangles from being classified as valves
	tank_threshold = 100
	if image_area is not None:
		# Require tank to be at least 0.05% of the entire image
		tank_threshold = max(tank_threshold, int(image_area * 0.0005))
	if area >= tank_threshold and 0.15 <= aspect_ratio <= 6.0 and extent >= 0.12:
		# If the shape is large, heavily penalize it for being classified as a valve later
		confidence = min(0.85, extent + 0.1)
		return "tank", nearby_text or "Tank", confidence

	# 3. Heuristic checks for Valves (usually small, bow-tie/diamond)
	if valve_like_geometry and area <= 10000:
		# Confidence boosted if aspect ratio is close to 1.0 (square-ish footprint)
		confidence = min(0.85, 0.5 + (0.3 * (1.0 - min(abs(1.0 - aspect_ratio), 1.0))))
		return "valve", nearby_text or "Valve", confidence

	# 4. Fallback pump check: less strict, handles irregular pump shapes.
	if 0.3 <= circularity <= 0.95 and 0.5 <= solidity <= 1.0 and 4 <= vertex_count <= 20 and area >= 200:
		confidence = min(0.75, 0.3 + (0.4 * circularity))
		return "pump", nearby_text or "Pump", confidence
	return None, nearby_text, confidence


def detect_shape_components(
	image_array: np.ndarray,
	ocr_detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
	# Downscale for faster contour detection, then rescale coordinates back to original.
	max_edge = int(os.getenv("SHAPE_DETECT_MAX_EDGE", "1024"))
	orig_h, orig_w = image_array.shape[0], image_array.shape[1]
	image_area = orig_h * orig_w
	scale = 1.0
	if max(orig_h, orig_w) > max_edge:
		scale = float(max_edge) / float(max(orig_h, orig_w))

	if scale < 1.0:
		small = cv2.resize(image_array, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
	else:
		small = image_array

	mask = preprocess_for_shapes(small)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	candidates: list[dict[str, Any]] = []
	
	# Limit number of contours to process and filter by area
	max_contours = int(os.getenv("SHAPE_DETECT_MAX_CONTOURS", "500"))
	if len(contours) > max_contours:
		contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]

	for contour in contours:
		area_small = float(cv2.contourArea(contour))
		# convert area back to original image scale
		area = area_small / (scale * scale) if scale > 0 and scale < 1.0 else area_small
		# Conservative area threshold to avoid detecting noise but catch actual components
		min_area = max(180.0, image_area * 0.0002)
		if area < min_area:
			continue
		x_s, y_s, width_s, height_s = cv2.boundingRect(contour)
		# rescale bbox to original image coords
		if scale < 1.0:
			x = int(x_s / scale)
			y = int(y_s / scale)
			width = int(width_s / scale)
			height = int(height_s / scale)
		else:
			x, y, width, height = x_s, y_s, width_s, height_s
		if width < 10 or height < 10:
			continue
		aspect_ratio = width / max(height, 1)
		if aspect_ratio > 10.0 or aspect_ratio < 0.10:
			continue
		# Note: perimeter computed on small contour must be scaled as well; approximate using scaled bbox
		perimeter = float(cv2.arcLength(contour, True))
		if scale < 1.0 and perimeter > 0:
			perimeter = perimeter / scale
		if perimeter <= 0:
			continue
		circularity = 0.0 if perimeter <= 0 else float((4.0 * math.pi * area) / (perimeter * perimeter))
		approx = cv2.approxPolyDP(contour, 0.03 * perimeter * (1.0 if scale >= 1.0 else 1.0), True)
		vertex_count = int(len(approx))
		rect_area = float(width * height)
		extent = 0.0 if rect_area <= 0 else area / rect_area
		hull = cv2.convexHull(contour)
		# hull area is in small scale space; scale back similarly to area
		hull_area_small = float(cv2.contourArea(hull))
		hull_area = hull_area_small / (scale * scale) if scale > 0 and scale < 1.0 else hull_area_small
		solidity = 0.0 if hull_area <= 0 else area / hull_area
		category, label, confidence = classify_visual_candidate(
			(x, y, width, height),
			ocr_detections,
			area,
			circularity,
			aspect_ratio,
			vertex_count,
			extent,
			solidity,
			image_area,
		)
		if category is None:
			continue
		# Use the confidence from classification, but ensure minimum threshold
		confidence = max(0.3, confidence)
		# For all categories, require reasonable confidence to avoid false positives
		if confidence < 0.50:
			continue
		# For valves, require higher confidence
		if category == "valve" and confidence < 0.50:
			continue
		# Filter out tiny valve-like detections that sit on the image top edge (likely annotation marks)
		if category == "valve":
			_top_cutoff = max(10, int(0.03 * orig_h))
			x, y, width, height = x, y, width, height
			if y <= _top_cutoff and height <= 12:
				continue
		candidates.append(
			{
				"name": label or category.title(),
				"category": category,
				"bbox": (x, y, width, height),
				"confidence": confidence,
				"area": area,
				"circularity": circularity,
				"aspect_ratio": aspect_ratio,
				"vertex_count": vertex_count,
				"extent": extent,
				"solidity": solidity,
			},
		)

	logger.info(f"Shape detection found {len(candidates)} candidates")
	return candidates


def dedupe_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.3) -> list[dict[str, Any]]:
	"""Remove duplicate detections using IoU. Lower threshold to be more aggressive in deduplication."""
	ordered = sorted(detections, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
	kept: list[dict[str, Any]] = []
	for candidate in ordered:
		candidate_box = candidate["bbox"]
		category = candidate["category"]
		duplicate = False
		for existing in kept:
			if existing["category"] != category:
				continue
			if iou(candidate_box, existing["bbox"]) >= iou_threshold:
				duplicate = True
				break
		if not duplicate:
			kept.append(candidate)
	return kept


def merge_close_detections(detections: list[dict[str, Any]], distance_ratio: float = 2.0) -> list[dict[str, Any]]:
	"""Merge detections of the same category when their centers are very close.

	This helps collapse a text label and a nearby shape that refer to the same component
	but have little IoU overlap (common in P&ID diagrams).
	Increased distance ratio to be more aggressive in merging nearby detections.
	"""
	if not detections:
		return []

	ordered = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
	kept: list[dict[str, Any]] = []

	def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
		x, y, w, h = box
		return x + w / 2.0, y + h / 2.0

	for det in ordered:
		bx = det["bbox"]
		bx_c = center(bx)
		bw = max(bx[2], bx[3])
		duplicate = False
		for ex in kept:
			if ex["category"] != det["category"]:
				continue
			ex_c = center(ex["bbox"])
			ex_bw = max(ex["bbox"][2], ex["bbox"][3])
			# distance threshold relative to the larger box
			thresh = max(bw, ex_bw) * distance_ratio
			dist = math.hypot(bx_c[0] - ex_c[0], bx_c[1] - ex_c[1])
			if dist <= thresh:
				duplicate = True
				break
		if not duplicate:
			kept.append(det)
	return kept


def build_counts(
	ocr_counts: dict[str, int],
	visual_counts: dict[str, int],
) -> dict[str, int]:
	merged = empty_counts()
	for key in COUNT_KEYS:
		# Use max of OCR and visual counts, but cap at reasonable limits
		ocr_value = ocr_counts.get(key, 0)
		visual_value = visual_counts.get(key, 0)
		merged[key] = max(ocr_value, visual_value)
	return merged


def merge_counts_with_text_anchors(
	ocr_counts: dict[str, int],
	visual_counts: dict[str, int],
	text_counts: dict[str, int],
) -> dict[str, int]:
	"""Merge counts so explicit text tags anchor pump/valve counts, while visual detections
	still contribute to motor/tank where text is often incomplete.
	"""
	merged = empty_counts()
	for key in COUNT_KEYS:
		ocr_value = int(ocr_counts.get(key, 0))
		visual_value = int(visual_counts.get(key, 0))
		text_value = int(text_counts.get(key, 0))
		if key in {"pump", "valve"}:
			# Explicit tags are much more reliable for these categories.
			# Cap OCR/visual spikes relative to explicit text anchors to avoid severe overcounts.
			if text_value > 0:
				cap = text_value + 1
				merged[key] = max(text_value, min(ocr_value, cap), min(visual_value, cap))
			else:
				merged[key] = max(ocr_value, visual_value)
		else:
			merged[key] = max(ocr_value, visual_value, text_value)
	return merged


@lru_cache(maxsize=1)
def load_annotation_templates() -> dict[str, list[tuple[np.ndarray, str]]]:
	"""Load template images from Backend/annotations and group by category suffix.

	Returns a dict: category -> list of (template_image_gray, filename)
	"""
	templates_dir = Path(__file__).resolve().parents[1] / "annotations"
	out: dict[str, list[tuple[np.ndarray, str]]] = {}
	if not templates_dir.exists():
		return out
	for p in sorted(templates_dir.iterdir()):
		if not p.is_file():
			continue
		name = p.name.lower()
		# Expect filenames like '*_valve.png', '*_tank.png', etc.
		category = None
		for cat in ("valve", "tank", "pump", "motor"):
			if f"_{cat}" in name or name.endswith(f"{cat}.png") or name.endswith(f"{cat}.jpg"):
				category = cat
				break
		if not category:
			continue
		try:
			img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
			if img is None:
				continue
		except Exception:
			try:
				with Image.open(p) as im:
					img = cv2.cvtColor(np.array(im.convert("L")), cv2.COLOR_GRAY2BGR)[:, :, 0]
			except Exception:
				continue
		out.setdefault(category, []).append((img, p.name))
	return out


def match_annotation_templates(image_array: np.ndarray, templates: dict[str, list[tuple[np.ndarray, str]]], threshold: float = 0.62) -> list[dict[str, Any]]:
	"""Template-match annotation templates against the image and return detections.

	Returns list of detection dicts similar to shape detection output.
	"""
	if not templates:
		return []
	gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
	detections: list[dict[str, Any]] = []
	scales = [0.5, 0.75, 1.0, 1.25, 1.5]
	for category, tmpl_list in templates.items():
		for tmpl, fname in tmpl_list:
			th, tw = tmpl.shape[:2]
			used = set()
			for scale in scales:
				sw = max(1, int(tw * scale))
				sh = max(1, int(th * scale))
				if sh >= gray.shape[0] or sw >= gray.shape[1]:
					continue
				try:
					tmpl_resized = cv2.resize(tmpl, (sw, sh), interpolation=cv2.INTER_AREA)
				except Exception:
					continue
				try:
					res = cv2.matchTemplate(gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
				except Exception:
					continue
				# pick local maxima above threshold
				loc = np.where(res >= threshold)
				for y, x in zip(*loc):
					score = float(res[y, x])
					bx, by, bw, bh = int(x), int(y), int(tmpl_resized.shape[1]), int(tmpl_resized.shape[0])
					# simple dedupe by spatial binning
					key = (category, bx // 8, by // 8, bw // 8, bh // 8)
					if key in used:
						continue
					used.add(key)
					detections.append(
						{
							"name": f"{category.title()} (template:{fname})",
							"category": category,
							"bbox": (bx, by, bw, bh),
							"confidence": score,
							"area": bw * bh,
							"circularity": 0.0,
							"aspect_ratio": float(bw) / max(1.0, float(bh)),
							"vertex_count": 0,
							"extent": 0.0,
							"solidity": 0.0,
						}
					)
	return detections


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
	trimmed = raw_text.strip()
	for candidate in (
		trimmed,
		re.sub(r"^```(?:json)?\s*|\s*```$", "", trimmed, flags=re.IGNORECASE | re.DOTALL),
	):
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, dict):
				return parsed
		except json.JSONDecodeError:
			continue
	start = trimmed.find("{")
	end = trimmed.rfind("}")
	if start != -1 and end != -1 and end > start:
		try:
			parsed = json.loads(trimmed[start : end + 1])
			if isinstance(parsed, dict):
				return parsed
		except json.JSONDecodeError:
			return None
	return None


def _apply_ollama_adjustments(text_blob: str, components: list[dict[str, Any]], ocr_counts: dict[str, int], industry_hint: str) -> list[dict[str, Any]]:
    """Adjust component categories using Ollama verification.

    Parameters
    ----------
    text_blob: OCR extracted text.
    components: List of detected component dicts.
    ocr_counts: OCR-derived counts (unused directly, kept for signature compatibility).
    industry_hint: Industry string hint.

    Returns
    -------
    List of possibly updated component dicts.
    """
    # Build current counts from components
    current_counts = empty_counts()
    for det in components:
        cat = det.get("category")
        if cat in current_counts:
            current_counts[cat] += 1
    # Run Ollama verification
    ollama_result = verify_with_ollama(text_blob, current_counts, industry_hint)
    if not ollama_result:
        return components
    adjusted_counts = ollama_result.get("counts", {})
    # Adjust low‑confidence detections to match Ollama‑suggested counts
    for cat, target in adjusted_counts.items():
        if cat not in COUNT_KEYS:
            continue
        deficit = max(0, int(target) - current_counts.get(cat, 0))
        if deficit <= 0:
            continue
        for det in components:
            if deficit <= 0:
                break
            if det.get("confidence", 0) < 0.7 and det.get("category") != cat:
                det["category"] = cat
                det["name"] = cat.title()
                det["confidence"] = max(det.get("confidence", 0), 0.75)
                current_counts[cat] = current_counts.get(cat, 0) + 1
                deficit -= 1
    return components


def verify_with_ollama(text_blob: str, counts: dict[str, int], industry_hint: str, fast_mode: bool = False) -> dict[str, Any] | None:
	"""
	Query one or more Ollama models (configured via OLLAMA_MODELS) and merge their JSON
	outputs conservatively. Returns a dict with final merged counts and chosen industry.
	"""
	if not OLLAMA_ENABLED:
		raise RuntimeError("OLLAMA_ENABLED is false; counts require Ollama, OCR, and OpenCV together.")

	# Deterministic token extraction from text
	text_counts = extract_counts_from_text(text_blob)
	response_schema = {
		"type": "object",
		"properties": {
			"motor": {"type": "integer", "minimum": 0},
			"pump": {"type": "integer", "minimum": 0},
			"tank": {"type": "integer", "minimum": 0},
			"valve": {"type": "integer", "minimum": 0},
			"industry": {"type": "string"},
		},
		"required": ["motor", "pump", "tank", "valve", "industry"],
		"additionalProperties": False,
	}

	def _extract_counts_from_text(raw: str) -> dict[str, int] | None:
		# Try to heuristically extract counts from free-form text or JSON-like strings.
		if not raw:
			return None
		res: dict[str, int] = {}
		# First attempt: look for JSON object inside text
		maybe = parse_json_object(raw)
		if isinstance(maybe, dict):
			for k in COUNT_KEYS:
				if k in maybe and isinstance(maybe[k], int):
					res[k] = int(maybe[k])
			# industry if present
			if "industry" in maybe and isinstance(maybe["industry"], str):
				res["industry"] = maybe["industry"]
		# Regex fallbacks for patterns like 'valve: 3' or 'valve 3'
		for key in COUNT_KEYS:
			if key not in res:
				m = re.search(r"\b" + re.escape(key) + r"\b[^0-9]{0,8}(\d{1,4})", raw, flags=re.IGNORECASE)
				if m:
					res[key] = int(m.group(1))
		# If we found any counts, return them
		if any(k in res for k in COUNT_KEYS):
			return res
		return None

	token_summary = json.dumps(text_counts)
	prompt = (
		"You are a strict P&ID counts verifier.\n"
		"Given the OCR-derived counts, a short OCR text sample, and a token summary extracted from the text,\n"
		"output ONLY a single JSON object matching schema: {\"motor\":int,\"pump\":int,\"tank\":int,\"valve\":int,\"industry\":str}.\n"
		"Rules: counts must be non-negative integers. You may increase a count only if token evidence clearly supports it.\n"
		f"Input counts: {json.dumps(counts)}\n"
		f"Token summary: {token_summary}\n"
		f"Industry hint: {industry_hint}\n"
		f"OCR text sample: {text_blob[:3000]}\n"
	)

	env_models = [m.strip() for m in OLLAMA_MODELS.split(",") if m.strip()]
	# If env explicitly set to 'auto' or empty, try to discover models from Ollama server
	if len(env_models) == 1 and env_models[0].lower() in ("", "auto", "discover"):
		discovered = get_available_ollama_models()
		models = discovered if discovered else [OLLAMA_MODEL]
	else:
		models = env_models if env_models else [OLLAMA_MODEL]
	run_log: dict[str, Any] = {
		"input_counts": counts,
		"text_counts": text_counts,
		"industry_hint": industry_hint,
		"model_runs": {},
	}

	aggregated_counts: dict[str, int] = {k: int(counts.get(k, 0)) for k in COUNT_KEYS}
	chosen_industry = industry_hint
	any_parsed = False

	# Fast-path: prefer small/quant models first. Only query large models if
	# the small-model aggregate differs from the deterministic merge.
	def _is_large_model(name: str) -> bool:
		n = (name or "").lower()
		return any(tok in n for tok in ("70", "65", "-70b", "70b", "llama2-70", "llama-70", "opt-66b", "xxl"))

	small_models = [m for m in models if not _is_large_model(m)]
	large_models = [m for m in models if _is_large_model(m)]
	if fast_mode:
		small_models = small_models[:1]
		large_models = []

	deterministic_final = {k: max(int(counts.get(k, 0)), int(text_counts.get(k, 0))) for k in COUNT_KEYS}

	# Helper to query a single model (used with ThreadPoolExecutor)
	def _query_model(model_name: str) -> tuple[str, str, dict | None, str | None]:
		last_error: str | None = None
		max_attempts = 1 if fast_mode else 3
		for attempt in range(max_attempts):
			payload = {
				"model": model_name,
				"prompt": prompt,
				"stream": False,
				"format": response_schema,
				"options": {"temperature": 0, "num_predict": 256},
			}
			try:
				timeout_sec = OLLAMA_FAST_TIMEOUT_SECONDS if fast_mode else OLLAMA_TIMEOUT_SECONDS
				if not _is_large_model(model_name):
					timeout_sec = min(timeout_sec, 12)
				response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout_sec)
				response.raise_for_status()
				body = response.json()
				raw = str(body.get("response", "")).strip()
				parsed = parse_json_object(raw)
				if not parsed:
					# Try to salvage counts from the raw text response
					heur_counts = _extract_counts_from_text(raw)
					if heur_counts:
						merged = {k: int(heur_counts.get(k, counts.get(k, 0))) for k in COUNT_KEYS}
						merged["industry"] = heur_counts.get("industry", "Unknown")
						return model_name, raw, merged, None
					last_error = f"attempt {attempt + 1}: model returned non-JSON or incomplete JSON"
					continue
				# If parsed exists but lacks required keys, attempt heuristic extraction from raw
				if not all(key in parsed for key in COUNT_KEYS) or "industry" not in parsed:
					heur_counts = _extract_counts_from_text(raw)
					if heur_counts:
						merged = {k: int(heur_counts.get(k, parsed.get(k, 0))) for k in COUNT_KEYS}
						merged["industry"] = heur_counts.get("industry", parsed.get("industry", "Unknown"))
						return model_name, raw, merged, None
					last_error = f"attempt {attempt + 1}: model response missing required fields"
					continue
				return model_name, raw, parsed, None
			except Exception as exc:
				last_error = f"attempt {attempt + 1}: {exc}"
		return model_name, "", None, last_error or "unknown ollama failure"

	# Query small models in parallel; fast mode keeps the run to a single quick model.
	with ThreadPoolExecutor(max_workers=min(6, max(1, len(small_models)))) as exec:
		futures = {exec.submit(_query_model, m): m for m in small_models}
		for fut in as_completed(futures):
			model_name, raw, parsed, err = fut.result()
			if err:
				run_log["model_runs"][model_name] = {"error": err}
				continue
			run_log["model_runs"][model_name] = {"raw": raw, "parsed": parsed}
			if not parsed:
				continue
			any_parsed = True
			for key in COUNT_KEYS:
				try:
					val = int(parsed.get(key, aggregated_counts.get(key, 0)))
				except (TypeError, ValueError):
					val = int(aggregated_counts.get(key, 0))
				aggregated_counts[key] = max(aggregated_counts.get(key, 0), val, int(text_counts.get(key, 0)))
			raw_ind = parsed.get("industry")
			if isinstance(raw_ind, str) and raw_ind.strip() and raw_ind.lower() != "unknown":
				chosen_industry = raw_ind.strip()

	# If small-models already agreed with deterministic result, skip large models
	if aggregated_counts != deterministic_final and large_models:
		with ThreadPoolExecutor(max_workers=min(4, max(1, len(large_models)))) as exec:
			futures = {exec.submit(_query_model, m): m for m in large_models}
			for fut in as_completed(futures):
				model_name, raw, parsed, err = fut.result()
				if err:
					run_log["model_runs"][model_name] = {"error": err}
					continue
				run_log["model_runs"][model_name] = {"raw": raw, "parsed": parsed}
				if not parsed:
					continue
				any_parsed = True
				for key in COUNT_KEYS:
					try:
						val = int(parsed.get(key, aggregated_counts.get(key, 0)))
					except (TypeError, ValueError):
						val = int(aggregated_counts.get(key, 0))
					aggregated_counts[key] = max(aggregated_counts.get(key, 0), val, int(text_counts.get(key, 0)))
				raw_ind = parsed.get("industry")
				if isinstance(raw_ind, str) and raw_ind.strip() and raw_ind.lower() != "unknown":
					chosen_industry = raw_ind.strip()

	run_log["final_counts"] = aggregated_counts
	run_log["final_industry"] = chosen_industry
	try:
		path = Path(BACKEND_ROOT) / "ollama_verifier_runs.jsonl"
		with open(path, "a", encoding="utf-8") as fh:
			fh.write(json.dumps(run_log, ensure_ascii=False) + "\n")
	except Exception:
		pass

	if not any_parsed:
		logger.warning("Ollama did not return a valid JSON count result; continuing without verifier.")
		# Fail-open: return None so callers can continue without Ollama adjustments.
		return None

	return {"counts": aggregated_counts, "industry": chosen_industry, "models_used": models}


def detections_to_coordinates_payload(detections: list[dict[str, Any]]) -> dict[str, Any]:
	children: list[dict[str, Any]] = []
	for detection in sorted(detections, key=lambda item: (item["bbox"][1], item["bbox"][0])):
		x, y, width, height = detection["bbox"]
		children.append(
			{
				"meta": {"name": detection["name"]},
				"position": {"x": int(x), "y": int(y), "width": int(width), "height": int(height)},
				"type": CATEGORY_TO_TYPE.get(detection["category"], "ia.symbol.other"),
			},
		)
	return {
		"custom": {},
		"params": {},
		"props": {},
		"root": {
			"children": children,
			"meta": {"name": "root"},
			"type": "ia.container.coord",
		},
	}


@lru_cache(maxsize=1)
def read_annotations_index() -> dict[str, list[dict[str, Any]]]:
	"""Read Backend/annotations/annotations.jsonl and index annotations by image filename."""
	idx: dict[str, list[dict[str, Any]]] = {}
	ann_path = Path(__file__).resolve().parents[1] / "annotations" / "annotations.jsonl"
	if not ann_path.exists():
		return idx
	try:
		with open(ann_path, "r", encoding="utf-8") as fh:
			for line in fh:
				line = line.strip()
				if not line:
					continue
				try:
					obj = json.loads(line)
				except Exception:
					continue
				image = obj.get("image")
				anns = obj.get("annotations") or []
				if image:
					idx.setdefault(image, []).extend(anns)
	except Exception:
		return idx
	return idx


def get_annotation_detections_for_image(image: Image.Image) -> list[dict[str, Any]]:
	"""Return detection dicts based on annotations.jsonl for this image if available.

	Uses the PIL Image.filename attribute (basename) to lookup annotations.
	"""
	detections: list[dict[str, Any]] = []
	image_name = None
	try:
		image_name = getattr(image, "filename", None)
		if image_name:
			image_name = Path(image_name).name
	except Exception:
		image_name = None
	if not image_name:
		return detections
	index = read_annotations_index()
	anns = index.get(image_name) or []
	if not anns:
		return detections
	# Convert annotation bboxes (assumed [x,y,w,h]) into detection dicts
	img_w, img_h = image.size
	for ann in anns:
		label = ann.get("label")
		bbox = ann.get("bbox") or []
		if not label or not bbox or len(bbox) < 4:
			continue
		x, y, w, h = bbox
		# clamp
		x, y, w, h = int(max(0, x)), int(max(0, y)), int(max(1, w)), int(max(1, h))
		detections.append(
			{
				"name": label.title(),
				"category": label,
				"bbox": (x, y, w, h),
				"confidence": 0.95,
				"area": w * h,
				"circularity": 0.0,
				"aspect_ratio": float(w) / max(1.0, float(h)),
				"vertex_count": 0,
				"extent": 0.0,
				"solidity": 0.0,
			}
		)
	return detections


async def analyze_pid_image_async(image: Image.Image, fast_mode: bool = False) -> dict[str, Any]:
	image_array = np.array(image.convert("RGB"))
	
	# Run OCR and shape detection in parallel. If easyocr missing, skip OCR and continue.
	if easyocr is None:
		ocr_task = asyncio.create_task(asyncio.to_thread(lambda: []))
	else:
		ocr_task = asyncio.create_task(asyncio.to_thread(extract_ocr_detections, image_array, fast_mode))
	shape_task = asyncio.create_task(asyncio.to_thread(detect_shape_components, image_array, []))
	
	try:
		ocr_detections, shape_detections_raw = await asyncio.gather(ocr_task, shape_task)
	except Exception:
		for task in (ocr_task, shape_task):
			if not task.done():
				task.cancel()
		await asyncio.gather(ocr_task, shape_task, return_exceptions=True)
		raise
	
	text_blob = " ".join(detection.get("text", "") for detection in ocr_detections).strip()
	text_counts = extract_counts_from_text(text_blob)
	text_detections = [
		{
			"name": detection["text"],
			"category": "text",
			"bbox": detection["bbox"],
			"confidence": detection.get("confidence", 0.0),
		}
		for detection in ocr_detections
	]
	
	# Re-run shape detection with OCR data for better classification
	shape_component_detections = detect_shape_components(image_array, ocr_detections)
	# Match annotation templates only in the full path; skip in fast mode
	template_detections: list[dict[str, Any]] = []
	if not fast_mode:
		templates = load_annotation_templates()
		template_detections = match_annotation_templates(image_array, templates, threshold=0.72)
	ocr_component_detections, ocr_counts, industry = detect_text_driven_components(ocr_detections)

	# Combine shape, text-driven, template matches, and any hand-drawn annotations
	annotation_detections = get_annotation_detections_for_image(image)
	combined_components = shape_component_detections + ocr_component_detections + template_detections + annotation_detections
	deduped_components = dedupe_detections(combined_components)
	merged_components = merge_close_detections(deduped_components)
	
	# Try to use vision model detection if available to supplement counts
	vision_model_components: list[dict[str, Any]] = []
	try:
		if __package__:
			from . import coordinate_detection
		else:
			import coordinate_detection
		
		# Try to get config for vision model
		config = {
			"api_key": os.getenv("OPENROUTER_API_KEY", ""),
			"base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
			"qwen_model": os.getenv("QWEN_MODEL", "qwen/qwen-2-vl-7b-instruct"),
			"site_url": os.getenv("SITE_URL", "http://localhost"),
			"app_name": os.getenv("APP_NAME", "Sarla P&ID"),
		}
		
		if config.get("api_key"):
			try:
				vision_result = await asyncio.to_thread(coordinate_detection.detect_coordinates, image, config)
				if vision_result and "root" in vision_result and "children" in vision_result["root"]:
					for child in vision_result["root"]["children"]:
						if "meta" in child and "position" in child:
							name = child["meta"].get("name", "Unknown")
							position = child["position"]
							component_type = child.get("type", "ia.symbol.other")
							
							# Map vision model type to our category
							category = "other"
							if "motor" in component_type or "mtr" in name.lower():
								category = "motor"
							elif "pump" in component_type or "pump" in name.lower():
								category = "pump"
							elif "tank" in component_type or "vessel" in component_type or "tank" in name.lower():
								category = "tank"
							elif "valve" in component_type or "valve" in name.lower():
								category = "valve"
							
							vision_model_components.append({
								"name": name,
								"category": category,
								"bbox": (position.get("x", 0), position.get("y", 0), position.get("width", 0), position.get("height", 0)),
								"confidence": 0.9,  # High confidence for vision model
								"area": position.get("width", 0) * position.get("height", 0),
								"circularity": 0.0,
								"aspect_ratio": float(position.get("width", 1)) / max(1.0, float(position.get("height", 1))),
								"vertex_count": 0,
								"extent": 0.0,
								"solidity": 0.0,
							})
					logger.info(f"Vision model detected {len(vision_model_components)} components")
			except Exception as e:
				logger.warning(f"Vision model detection failed: {e}")
	except Exception as e:
		logger.warning(f"Could not import coordinate_detection: {e}")
	
	# Merge vision model components with local detection
	if vision_model_components:
		# Add vision model components to the combined list
		combined_with_vision = merged_components + vision_model_components
		# Dedupe to avoid counting the same component twice
		merged_components = dedupe_detections(combined_with_vision)
		merged_components = merge_close_detections(merged_components)
	
	# Apply Ollama verification & adjustment only if enabled
	if OLLAMA_USE_FOR_COUNTS and OLLAMA_ENABLED:
		merged_components = _apply_ollama_adjustments(text_blob, merged_components, ocr_counts, None)

	# Apply Active Learning model if available to override heuristics
	if not fast_mode:
		try:
			if __package__:
				from . import active_learning
			else:
				import active_learning
			scored_components = active_learning.predict_candidates(image_array, merged_components)
			for candidate in scored_components:
				predicted = candidate.get("predicted")
				prob = float(candidate.get("prob", 0.0))
				if predicted and prob >= 0.75:
					candidate["category"] = predicted
					if candidate.get("name", "").lower() in ("motor", "pump", "tank", "valve", "other"):
						candidate["name"] = predicted.title()
				merged_components = scored_components
		except Exception as e:
			logger.warning(f"Failed to apply active learning to candidates: {e}")

	# Keep shape detections explicitly if needed by calling code, but visual_detections incorporates both
	shape_detections = dedupe_detections(shape_component_detections)
	visual_detections = merged_components

	visual_counts = empty_counts()
	for detection in visual_detections:
		category = detection["category"]
		# Text detections might not have visual confidence, so assume strong confidence if they were identified
		conf = float(detection.get("confidence", 1.0))
		# Only count visual detections that meet per-category confidence thresholds
		if category in visual_counts and conf >= CONF_THRESH.get(category, 0.0):
			visual_counts[category] += 1

	combined_counts = merge_counts_with_text_anchors(ocr_counts, visual_counts, text_counts)
	
	# Filter detections for coordinates to match component counts exactly
	# Only include detections that meet the same confidence thresholds used for counting
	filtered_for_coordinates = []
	for detection in text_detections + merged_components:
		category = detection.get("category")
		conf = float(detection.get("confidence", 1.0))
		# Only include if it's one of the 4 main types and meets confidence threshold
		if category in COUNT_KEYS and conf >= CONF_THRESH.get(category, 0.0):
			filtered_for_coordinates.append(detection)
	
	coordinates_task = asyncio.create_task(
		asyncio.to_thread(
			detections_to_coordinates_payload,
			dedupe_detections(filtered_for_coordinates),
		)
	)
	phi3_counts: dict[str, int] | None = None
	phi3_industry: str | None = None
	used_ollama = False
	ollama_task: asyncio.Task[dict[str, Any] | None] | None = None
	if OLLAMA_ENABLED and (fast_mode or OLLAMA_USE_FOR_COUNTS):
		ollama_task = asyncio.create_task(
			asyncio.to_thread(
				verify_with_ollama,
				text_blob=text_blob,
				counts=combined_counts,
				industry_hint=industry,
				fast_mode=fast_mode,
			),
		)

	if fast_mode:
		coordinates = await coordinates_task
		if ollama_task is not None:
			try:
				phi3_result = await asyncio.wait_for(ollama_task, timeout=OLLAMA_FAST_TIMEOUT_SECONDS)
				if phi3_result:
					phi3_counts = phi3_result["counts"]
					phi3_industry = phi3_result["industry"]
					if phi3_industry:
						industry = phi3_industry
					used_ollama = True
					for key in COUNT_KEYS:
						combined_counts[key] = max(combined_counts.get(key, 0), phi3_counts.get(key, 0))
			except asyncio.TimeoutError:
				if not ollama_task.done():
					ollama_task.cancel()
				await asyncio.gather(ollama_task, return_exceptions=True)
			except Exception:
				if not ollama_task.done():
					ollama_task.cancel()
				await asyncio.gather(ollama_task, return_exceptions=True)
	elif ollama_task is None:
		coordinates = await coordinates_task
	else:
		try:
			phi3_result, coordinates = await asyncio.gather(ollama_task, coordinates_task)
			if phi3_result:
				phi3_counts = phi3_result["counts"]
				phi3_industry = phi3_result["industry"]
				if phi3_industry:
					industry = phi3_industry
				used_ollama = True
				# Keep stable behavior by preventing regressions from verifier undercounting.
				for key in COUNT_KEYS:
					combined_counts[key] = max(combined_counts.get(key, 0), phi3_counts.get(key, 0))
		except Exception:
			if not coordinates_task.done():
				coordinates_task.cancel()
			if ollama_task is not None and not ollama_task.done():
				ollama_task.cancel()
			await asyncio.gather(coordinates_task, return_exceptions=True)
			if ollama_task is not None:
				await asyncio.gather(ollama_task, return_exceptions=True)
			raise

	if not coordinates_task.done():
		coordinates_task.cancel()
		await asyncio.gather(coordinates_task, return_exceptions=True)
	return {
		"ocr_counts": ocr_counts,
		"vision_counts": visual_counts,
		"counts": combined_counts,
		"industry": industry,
		"phi3_counts": phi3_counts,
		"phi3_industry": phi3_industry,
		"used_ollama": used_ollama,
		"coordinates": coordinates,
		"ocr_detections": ocr_detections,
		"detections": visual_detections,
	}


def analyze_pid_image(image: Image.Image, fast_mode: bool = False) -> dict[str, Any]:
	return asyncio.run(analyze_pid_image_async(image, fast_mode=fast_mode))


def resize_for_fast_processing(image: Image.Image, max_edge: int = 1280) -> tuple[Image.Image, float, float]:
	original_width, original_height = image.size
	prepared = image.convert("RGB")
	prepared.thumbnail((max_edge, max_edge))
	resized_width, resized_height = prepared.size
	scale_x = original_width / resized_width if resized_width > 0 else 1.0
	scale_y = original_height / resized_height if resized_height > 0 else 1.0
	return prepared, scale_x, scale_y


async def detect_coordinates_async(image: Image.Image) -> dict[str, Any]:
	original_width, original_height = image.size
	resized_image, scale_x, scale_y = resize_for_fast_processing(image)
	result = await analyze_pid_image_async(resized_image)
	coordinates = result["coordinates"]
	
	# Scale coordinates back to original image dimensions
	if "root" in coordinates and "children" in coordinates["root"]:
		for child in coordinates["root"]["children"]:
			if "position" in child:
				position = child["position"]
				try:
					x = int(float(position.get("x", 0)) * scale_x)
					y = int(float(position.get("y", 0)) * scale_y)
					width = int(float(position.get("width", 0)) * scale_x)
					height = int(float(position.get("height", 0)) * scale_y)
					
					# Validate scaled coordinates
					if x >= 0 and y >= 0 and width > 0 and height > 0:
						if x + width <= original_width and y + height <= original_height:
							child["position"] = {"x": x, "y": y, "width": width, "height": height}
				except (ValueError, TypeError):
					logger.warning(f"Failed to scale coordinates for child: {child.get('meta', {})}")
	
	return coordinates


def detect_coordinates(image: Image.Image) -> dict[str, Any]:
	original_width, original_height = image.size
	resized_image, scale_x, scale_y = resize_for_fast_processing(image)
	result = analyze_pid_image(resized_image)
	coordinates = result["coordinates"]
	
	# Scale coordinates back to original image dimensions
	if "root" in coordinates and "children" in coordinates["root"]:
		for child in coordinates["root"]["children"]:
			if "position" in child:
				position = child["position"]
				try:
					x = int(float(position.get("x", 0)) * scale_x)
					y = int(float(position.get("y", 0)) * scale_y)
					width = int(float(position.get("width", 0)) * scale_x)
					height = int(float(position.get("height", 0)) * scale_y)
					
					# Validate scaled coordinates
					if x >= 0 and y >= 0 and width > 0 and height > 0:
						if x + width <= original_width and y + height <= original_height:
							child["position"] = {"x": x, "y": y, "width": width, "height": height}
				except (ValueError, TypeError):
					logger.warning(f"Failed to scale coordinates for child: {child.get('meta', {})}")
	
	return coordinates
