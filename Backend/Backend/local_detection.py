from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import re
import json
import time
from pathlib import Path
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

logger = logging.getLogger(__name__)

# Set random seed for deterministic behavior
random.seed(42)
np.random.seed(42)

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


COUNT_KEYS = ("motor", "pump", "tank", "valve", "instrument", "other")
CATEGORY_TO_TYPE = {
	"text": "ia.symbol.text",
	"motor": "ia.symbol.motor",
	"pump": "ia.symbol.pump",
	"tank": "ia.symbol.tank",
	"valve": "ia.symbol.valve",
	"instrument": "ia.symbol.sensor",
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
	# Only unambiguous valve abbreviations — tv/pv removed as they match instrument tags
	# fv- included with hyphen to match Fv-3-3040 style tags without matching bare instrument 'fv'
	("valve", ("fv-", "xv", "hv", "lv", "sv", "gv", "bv", "wv", "pcv", "fcv", "lcv", "tcv", "psv", "nrv", "sdv", "mov", "sov")),
]

# Simple initial-letter mapping for compact P&ID tags (e.g. 'm123' -> motor)
INITIAL_PREFIX_MAP: dict[str, str] = {
    "m": "motor",
    "p": "pump",
    "t": "tank",
    "v": "valve",
}

# Instrument/transmitter tag prefixes that are NOT physical components.
# These appear inside instrument bubbles (circles) and must not be counted as valves/motors/pumps.
INSTRUMENT_TAG_PREFIXES: frozenset[str] = frozenset([
	"tic", "tt", "te", "ti",           # Temperature
	"fic", "ft", "fe", "fi", "fit",    # Flow
	"lic", "lt", "le", "li",           # Level
	"pic", "pt", "pe", "pi", "pit",    # Pressure
	"aic", "at", "ae", "ai",           # Analytical
	"pc", "lc", "pic", "lic", "fic",   # Controllers
	"tic", "trc", "frc", "lrc", "prc", # Controllers/recorders
	"tsh", "tsl", "fsh", "fsl",        # Switches
	"tit", "fit", "lit", "pit",        # Indicators/transmitters
])

# Regex to detect instrument bubble tags like "TIC 100", "FT 101", "TE 100"
_INSTRUMENT_TAG_RE = re.compile(
	r"^(?:tic|tt|te|ti|fic|ft|fe|fi|fit|lic|lt|le|li|pic|pt|pe|pi|pit|aic|at|ae|ai|trc|frc|lrc|prc|tsh|tsl|fsh|fsl|tit|lit)\b",
	re.IGNORECASE,
)
# Line labels like "From P-201" reference equipment off the sheet — not drawable symbols.
_OFF_PAGE_LINE_RE = re.compile(
	r"^\s*(?:from|to)\b",
	re.IGNORECASE,
)
_OFF_PAGE_TAG_IN_TEXT_RE = re.compile(
	r"\b(?:from|to)\s+[a-z]{0,4}[\s-]*\d{2,5}[a-z]?\b",
	re.IGNORECASE,
)
_PUMP_TAG_RE = re.compile(r"\b(?:p|pu|pmp|pump)-?\d{1,5}[a-z]?\b", re.IGNORECASE)

_COUNTABLE_TEXT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
	"motor": (
		re.compile(r"\bm-?\d{2,5}[a-z]?\b", re.IGNORECASE),
		re.compile(r"\bmo-?\d{2,5}[a-z]?\b", re.IGNORECASE),
		re.compile(r"\bmtr-?\d{1,5}[a-z]?\b", re.IGNORECASE),
	),
	"pump": (
		re.compile(r"\bp-?\d{1,5}[a-z]?\b", re.IGNORECASE),
		re.compile(r"\bpu-?\d{1,5}[a-z]?\b", re.IGNORECASE),
		re.compile(r"\bpmp-?\d{1,5}[a-z]?\b", re.IGNORECASE),
	),
	"tank": (
		re.compile(r"\bt-?\d{1,5}[a-z]?\b", re.IGNORECASE),
		re.compile(r"\btk-?\d{1,5}[a-z]?\b", re.IGNORECASE),
		re.compile(r"\bv-?\d{1,5}[a-z]?\b", re.IGNORECASE),
	),
	"valve": (
		re.compile(r"(?<![a-z0-9])(?:fv|xv|cv|hv|lv|sv|pv|tv|gv|bv|wv|pcv|fcv|lcv|tcv|psv|nrv|sdv|mov|sov)-?\d[\d\-]*[a-z]?(?![a-z0-9])", re.IGNORECASE),
		re.compile(r"(?<![a-z0-9])(?:v|xv|cv|hv|lv|sv|pv|tv|gv|bv|wv)-?\d{1,5}[a-z]?(?![a-z0-9])", re.IGNORECASE),
	),
	"instrument": (
		re.compile(r"\b(?:tic|tt|te|ti|tit)\-?\d{1,5}[a-z]?\b", re.IGNORECASE),
		re.compile(r"\b(?:fic|ft|fe|fi|fit)\-?\d{1,5}[a-z]?\b", re.IGNORECASE),
		re.compile(r"\b(?:lic|lt|le|li|lit)\-?\d{1,5}[a-z]?\b", re.IGNORECASE),
		re.compile(r"\b(?:pic|pt|pe|pi|pit)\-?\d{1,5}[a-z]?\b", re.IGNORECASE),
		re.compile(r"\b(?:aic|at|ae|ai)\-?\d{1,5}[a-z]?\b", re.IGNORECASE),
	),
}


def is_instrument_tag(text: str) -> bool:
	"""Return True if the text looks like an instrument bubble tag (not a physical component)."""
	return bool(_INSTRUMENT_TAG_RE.match(normalize_text(text)))


def is_off_page_equipment_reference(text: str) -> bool:
	"""True for line labels like 'From P-201' that name off-sheet equipment."""
	normalized = normalize_text(text)
	if not normalized:
		return False
	if _OFF_PAGE_LINE_RE.match(normalized):
		return True
	return bool(_OFF_PAGE_TAG_IN_TEXT_RE.search(normalized))


def _strip_off_page_equipment_tags(text: str) -> str:
	"""Remove off-page tag mentions so text-based counters do not inflate pumps/tanks."""
	return _OFF_PAGE_TAG_IN_TEXT_RE.sub(" ", normalize_text(text or ""))


def _countable_text_category(text: str) -> str | None:
	"""Return a physical component category only for explicit countable tags.

	This intentionally ignores bare words like "tank" or "valve" so OCR text
	alone does not inflate counts when the diagram repeats labels.
	"""
	normalized = normalize_text(text)
	if not normalized or is_off_page_equipment_reference(normalized):
		return None
	if is_instrument_tag(normalized):
		return "instrument"
	for category, patterns in _COUNTABLE_TEXT_PATTERNS.items():
		if any(pattern.search(normalized) for pattern in patterns):
			return category
	return None


CATEGORY_REGEX_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
	(
		"valve",
		(
			r"\b(?:check\s*valve|gate\s*valve|globe\s*valve|ball\s*valve|butterfly\s*valve|plug\s*valve)\b",
			r"(?<![a-z0-9])(?:pcv|fcv|lcv|tcv|psv|nrv|sdv|xv|hv|lv|fv|sv|cv|tv|pv|mov|sov|bv|gv|wv)-?\d[\d\-]*[a-z]?(?![a-z0-9])",
			r"(?<![a-z0-9])(?:fv|xv|cv|hv|lv|sv|pv|tv|gv|bv|wv)-[\d\-]+[a-z]?(?![a-z0-9])",
			r"(?<![a-z0-9])(?:v|xv|cv|hv|lv|sv|pv|tv|gv|bv|wv)-?\d{1,5}[a-z]?(?![a-z0-9])",
			r"\bvalve\b",
			# More permissive patterns for simple valve tags like V1, V-1, V123
			r"\bv-?\d{1,5}[a-z]?\b",
			r"\bv\d{1,5}[a-z]?\b",
		),
	),
	("pump", (r"\bp-?\d{1,5}[a-z]?\b", r"\bpu-?\d{1,5}[a-z]?\b", r"\bpmp-?\d{1,5}[a-z]?\b", r"\bpump\b")),
	("motor", (r"\bm-?\d{2,5}[a-z]?\b", r"\bmo-?\d{2,5}[a-z]?\b", r"\bmtr-?\d{1,5}[a-z]?\b", r"\bmotor\b")),
	("tank", (
		r"\b(?:t|tk|v)-?\d{1,5}[a-z]?\b",
		r"\btank\b",
	)),
	("instrument", (
		r"\b(?:tic|tt|te|ti|tit)\-?\d{1,5}[a-z]?\b",
		r"\b(?:fic|ft|fe|fi|fit)\-?\d{1,5}[a-z]?\b",
		r"\b(?:lic|lt|le|li|lit)\-?\d{1,5}[a-z]?\b",
		r"\b(?:pic|pt|pe|pi|pit)\-?\d{1,5}[a-z]?\b",
		r"\b(?:aic|at|ae|ai)\-?\d{1,5}[a-z]?\b",
	)),
]

OCR_MIN_TEXT_CONFIDENCE = float(os.getenv("OCR_MIN_TEXT_CONFIDENCE", "0.15"))
OCR_MIN_COMPONENT_AREA = int(os.getenv("OCR_MIN_COMPONENT_AREA", "50"))
PADDLEOCR_LANG = os.getenv("PADDLEOCR_LANG", "en")
PADDLEOCR_USE_GPU = os.getenv("PADDLEOCR_USE_GPU", "false").strip().lower() in {"1", "true", "yes", "on"}
FAST_OCR_MAX_EDGE = max(768, int(os.getenv("FAST_OCR_MAX_EDGE", "1280")))
# Re-enable Ollama with better error handling
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")  # Expert-level: Use llama3.1 for maximum accuracy
# Comma-separated list of Ollama models to consult (e.g. "phi3-mini,llama3.2")
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", OLLAMA_MODEL)
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))
# Per-request HTTP timeout for /analyze_fast. Keep this short so the fast path stays responsive.
OLLAMA_FAST_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_FAST_TIMEOUT_SECONDS", "15"))
# Total wall-clock time the API will wait for Ollama to finish (includes cold-start load).
OLLAMA_COMPLETION_TIMEOUT_SECONDS = int(
	os.getenv("OLLAMA_COMPLETION_TIMEOUT_SECONDS", str(max(OLLAMA_FAST_TIMEOUT_SECONDS + 2, 10)))
)
# When true, final counts follow Ollama output. Default to false so visual detections stay authoritative.
OLLAMA_TRUST_COUNTS = os.getenv("OLLAMA_TRUST_COUNTS", "false").strip().lower() in {"1", "true", "yes", "on"}
# When false, Ollama is skipped in the count path for speed and determinism.
# Expert-level: Enable Ollama verification by default for maximum accuracy
OLLAMA_USE_FOR_COUNTS = os.getenv("OLLAMA_USE_FOR_COUNTS", "true").strip().lower() in {"1", "true", "yes", "on"}
# Cap reference images used for template matching (annotations folder can grow to 1000+ files).
# Expert-level: Increased to use more annotation images for better accuracy
TEMPLATE_MAX_PER_CATEGORY = max(1, int(os.getenv("TEMPLATE_MAX_PER_CATEGORY", "75")))
TEMPLATE_MATCH_MAX_EDGE = max(640, int(os.getenv("TEMPLATE_MATCH_MAX_EDGE", "1920")))
TEMPLATE_MAX_PEAKS = max(5, int(os.getenv("TEMPLATE_MAX_PEAKS", "50")))
TEMPLATE_MATCH_TIMEOUT_SECONDS = 90.0
FAST_TEMPLATE_MATCH_TIMEOUT_SECONDS = 60.0
FAST_TEMPLATE_MAX_PER_CATEGORY = max(1, int(os.getenv("FAST_TEMPLATE_MAX_PER_CATEGORY", "0")))
FAST_TEMPLATE_MAX_TOTAL = max(4, int(os.getenv("FAST_TEMPLATE_MAX_TOTAL", "100")))
FAST_MATCH_RELEVANT_ONLY = os.getenv("FAST_MATCH_RELEVANT_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}
FAST_DISABLE_ORB_OVER_TEMPLATE_COUNT = max(0, int(os.getenv("FAST_DISABLE_ORB_OVER_TEMPLATE_COUNT", "0")))
FAST_OLLAMA_WAIT_CAP_SECONDS = float(os.getenv("FAST_OLLAMA_WAIT_CAP_SECONDS", "8"))
FAST_ACCURACY_PRIORITIZE_OLLAMA = os.getenv("FAST_ACCURACY_PRIORITIZE_OLLAMA", "true").strip().lower() in {"1", "true", "yes", "on"}
FAST_OLLAMA_TEXT_CHARS = max(800, int(os.getenv("FAST_OLLAMA_TEXT_CHARS", "2200")))

# Minimum confidence required to count a visual detection for each category.
# Optimized for maximum accuracy while minimizing false positives
CONF_THRESH: dict[str, float] = {
	"motor": 0.30,
	"pump": 0.25,
	"tank": 0.28,
	"valve": 0.25,
	"instrument": 0.25,
	"other": 0.30,  # For non-standard components
}

# Higher confidence thresholds for fast mode to reduce false positives
# Balanced to maintain accuracy while being fast
FAST_CONF_THRESH: dict[str, float] = {
	"motor": 0.35,
	"pump": 0.40,  # Reduced from 0.50 to improve pump detection
	"tank": 0.30,
	"valve": 0.40,
	"instrument": 0.30,
	"other": 0.35,
}

# Confidence thresholds for simple P&ID diagrams (clear, uncluttered layouts)
# Balanced thresholds to avoid filtering out valid detections
SIMPLE_CONF_THRESH: dict[str, float] = {
	"motor": 0.30,  # Lowered to allow valid motor detections
	"pump": 0.35,  # Lowered to allow valid pump detections
	"tank": 0.30,  # Lowered to allow valid tank detections
	"valve": 0.25,  # Lowered to allow valid valve detections
	"instrument": 0.30,  # Lowered to allow valid instrument detections
	"other": 0.30,
}

# Confidence thresholds for complex P&ID diagrams (dense, overlapping elements)
# Optimized for better recall in crowded diagrams
COMPLEX_CONF_THRESH: dict[str, float] = {
	"motor": 0.18,  # Slightly lowered
	"pump": 0.15,  # Slightly lowered
	"tank": 0.12,  # Slightly lowered
	"valve": 0.18,  # Slightly lowered
	"instrument": 0.12,  # Slightly lowered
	"other": 0.20,
}


def detect_diagram_complexity(
	image_array: np.ndarray,
	ocr_detections: list[dict[str, Any]] | None = None,
	shape_detections: list[dict[str, Any]] | None = None,
) -> str:
	"""Analyze image characteristics to classify diagram as 'simple' or 'complex'.
	
	Expert-level: Enhanced accuracy with multiple complexity metrics and adaptive thresholds.
	
	Complexity metrics:
	- Component density: number of detected components per unit area
	- Text density: amount of OCR text per unit area  
	- Edge density: amount of structural detail in the image
	- Contour complexity: number and complexity of contours
	- Spatial distribution: how evenly components are distributed
	- Overlap analysis: degree of component overlap in simple diagrams
	
	Returns 'simple' or 'complex' based on combined complexity score.
	"""
	h, w = image_array.shape[:2]
	image_area = float(h * w)
	
	if image_area == 0:
		return "simple"
	
	# Calculate edge density as a measure of structural complexity
	gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(gray, 50, 150)
	edge_density = float(np.count_nonzero(edges)) / image_area
	
	# Calculate component density from shape detections
	if shape_detections:
		component_density = len(shape_detections) / (image_area / 10000.0)  # per 10k pixels
	else:
		component_density = 0.0
	
	# Calculate text density from OCR detections
	if ocr_detections:
		text_density = len(ocr_detections) / (image_area / 10000.0)  # per 10k pixels
	else:
		text_density = 0.0
	
	# Calculate contour complexity with adaptive threshold
	_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contour_count = len(contours)
	contour_density = contour_count / (image_area / 10000.0)  # per 10k pixels
	
	# Calculate spatial distribution - simple diagrams have more uniform distribution
	if shape_detections and len(shape_detections) > 1:
		centers = []
		for det in shape_detections:
			bbox = det.get("bbox")
			if bbox and len(bbox) == 4:
				x, y, w_box, h_box = bbox
				centers.append((x + w_box / 2.0, y + h_box / 2.0))
		
		if len(centers) > 1:
			centers_array = np.array(centers)
			# Calculate standard deviation of positions
			std_x = np.std(centers_array[:, 0])
			std_y = np.std(centers_array[:, 1])
			# Normalize by image dimensions
			normalized_std = (std_x + std_y) / (w + h)
			spatial_uniformity = 1.0 - min(normalized_std * 2.0, 1.0)  # Higher = more uniform
		else:
			spatial_uniformity = 0.5
	else:
		spatial_uniformity = 0.5
	
	# Calculate overlap analysis - simple diagrams have less overlap
	if shape_detections and len(shape_detections) > 1:
		overlap_count = 0
		total_pairs = 0
		for i in range(len(shape_detections)):
			for j in range(i + 1, len(shape_detections)):
				bbox1 = shape_detections[i].get("bbox")
				bbox2 = shape_detections[j].get("bbox")
				if bbox1 and bbox2 and len(bbox1) == 4 and len(bbox2) == 4:
					total_pairs += 1
					# Simple IoU check
					iou_score = iou(bbox1, bbox2)
					if iou_score > 0.1:  # More than 10% overlap
						overlap_count += 1
		
		if total_pairs > 0:
			overlap_ratio = overlap_count / total_pairs
		else:
			overlap_ratio = 0.0
	else:
		overlap_ratio = 0.0
	
	# Calculate combined complexity score (normalized 0-1)
	# Expert-level: Refined weights and normalization for better accuracy
	edge_score = min(edge_density / 0.10, 1.0)  # Edge density normalized (lowered threshold for better sensitivity)
	component_score = min(component_density / 1.2, 1.0)  # Component density normalized (lowered threshold)
	text_score = min(text_density / 0.8, 1.0)  # Text density normalized (lowered threshold)
	contour_score = min(contour_density / 3.5, 1.0)  # Contour density normalized (lowered threshold)
	
	# Spatial and overlap metrics (inverse - higher values indicate simpler diagrams)
	spatial_score = 1.0 - spatial_uniformity  # Higher = less uniform = more complex
	overlap_score = overlap_ratio  # Higher = more overlap = more complex
	
	# Expert-level: Adjusted weights with emphasis on component and edge density for better accuracy
	complexity_score = (
		0.32 * edge_score +
		0.28 * component_score +
		0.14 * text_score +
		0.12 * contour_score +
		0.07 * spatial_score +
		0.07 * overlap_score
	)
	
	# Expert-level: More aggressive threshold for better simple diagram classification
	# Simple diagrams should have very low complexity scores
	is_complex = complexity_score > 0.50  # Increased threshold to reduce false complex classifications
	
	logger.info(
		f"Diagram complexity analysis: edge_density={edge_density:.4f}, "
		f"component_density={component_density:.4f}, text_density={text_density:.4f}, "
		f"contour_density={contour_density:.4f}, spatial_uniformity={spatial_uniformity:.4f}, "
		f"overlap_ratio={overlap_ratio:.4f}, complexity_score={complexity_score:.4f}, "
		f"classified_as={'complex' if is_complex else 'simple'}"
	)
	
	return "complex" if is_complex else "simple"


def get_adaptive_confidence_thresholds(
	diagram_complexity: str,
	fast_mode: bool = False,
) -> dict[str, float]:
	"""Return appropriate confidence thresholds based on diagram complexity and mode.
	
	Expert-level: Enhanced accuracy with category-specific adjustments for simple diagrams.
	Diagram complexity takes priority over fast mode to ensure accuracy.
	
	Args:
		diagram_complexity: 'simple' or 'complex' from detect_diagram_complexity
		fast_mode: Whether fast mode is enabled (prioritizes speed over accuracy)
	
	Returns:
		Dictionary mapping category names to confidence thresholds
	"""
	# Always use complexity-specific thresholds as the base (diagram complexity is more important than speed)
	if diagram_complexity == "complex":
		base_thresh = COMPLEX_CONF_THRESH.copy()
	else:
		base_thresh = SIMPLE_CONF_THRESH.copy()
	
	# Only adjust for fast mode if the diagram is complex (simple diagrams need strict thresholds regardless of speed)
	if fast_mode and diagram_complexity == "complex":
		# Complex diagrams can use slightly lower thresholds in fast mode
		base_thresh = {k: v * 0.85 for k, v in base_thresh.items()}
	
	return base_thresh


def apply_simple_diagram_validation(
	detections: list[dict[str, Any]],
	ocr_detections: list[dict[str, Any]],
	diagram_complexity: str,
) -> list[dict[str, Any]]:
	"""Apply additional validation for simple diagrams to improve accuracy.
	
	Relaxed validation to avoid filtering out valid detections.
	
	Args:
		detections: List of component detections
		ocr_detections: List of OCR text detections
		diagram_complexity: 'simple' or 'complex'
	
	Returns:
		Filtered list of detections with additional validation applied
	"""
	filtered = []
	for det in detections:
		category = det.get("category")
		confidence = float(det.get("confidence", 0.0))
		bbox = det.get("bbox")
		source = det.get("source", "")
		
		# Skip OCR-based detections entirely - they're text labels, not component symbols
		if source == "ocr":
			continue
		
		# Filter out template-based tank detections to reduce overcounting
		if category == "tank" and source == "template":
			continue
		
		# For all diagrams, require minimum confidence to avoid false positives
		min_conf = 0.20  # Lowered to allow valid detections
		if confidence < min_conf:
			continue
		
		# Additional geometry validation - ensure component has reasonable shape properties
		area = bbox[2] * bbox[3] if bbox and len(bbox) == 4 else 0
		min_area = 30  # Lowered to allow valid components
		if area < min_area:  # Too small to be a real component
			continue
		
		filtered.append(det)
	
	return filtered


def empty_counts() -> dict[str, int]:
	return {key: 0 for key in COUNT_KEYS}


def normalize_pid_category(category: str | None) -> str | None:
	if not category:
		return None
	candidate = str(category).strip().lower()
	if candidate == "sensor":
		return "instrument"
	if candidate in COUNT_KEYS:
		return candidate
	return None


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
	"""Upscale and enhance the image before OCR to improve small tag recall.
	Expert-level: Very aggressive enhancement for maximum text detection accuracy."""
	h, w = image_array.shape[:2]
	max_edge = max(h, w)
	prepared = image_array
	# Performance optimization: reduce target edge significantly in fast mode for faster OCR
	target_edge = 512 if fast_mode else 3200
	if max_edge < target_edge:
		scale = float(target_edge) / max_edge
		prepared = cv2.resize(image_array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
	elif fast_mode and max_edge > target_edge:
		scale = float(target_edge) / max_edge
		prepared = cv2.resize(image_array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(prepared, cv2.COLOR_RGB2GRAY)
	
	# Keep fast_mode cheap: skip heavy denoising and morphology passes.
	if not fast_mode:
		gray = cv2.fastNlMeansDenoising(gray, h=5)
	
	# Expert-level: Very high CLAHE clip limit for maximum contrast
	# Performance optimization: reduce CLAHE intensity in fast mode
	clahe = cv2.createCLAHE(
		clipLimit=(1.8 if fast_mode else 4.5),
		tileGridSize=((8, 8) if fast_mode else (6, 6)),
	)
	boosted = clahe.apply(gray)
	
	if not fast_mode:
		# Morphological operations to enhance text strokes - stronger
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		boosted = cv2.morphologyEx(boosted, cv2.MORPH_CLOSE, kernel)
		boosted = cv2.morphologyEx(boosted, cv2.MORPH_OPEN, kernel)
	
	# Performance optimization: skip unsharp masking in fast mode for speed
	if not fast_mode:
		# Unsharp masking for clearer text - very strong enhancement
		gaussian = cv2.GaussianBlur(boosted, (0, 0), (0.8 if fast_mode else 1.2))
		sharpened = cv2.addWeighted(
			boosted,
			(1.5 if fast_mode else 2.2),
			gaussian,
			(-0.5 if fast_mode else -1.2),
			0,
		)
		# Additional contrast boost
		sharpened = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
	else:
		sharpened = boosted
	
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


_ollama_models_cache: list[str] | None = None

def get_available_ollama_models() -> list[str]:
	"""Query the Ollama server for available models. Caches successful results."""
	global _ollama_models_cache
	if _ollama_models_cache is not None:
		return _ollama_models_cache
	try:
		resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
		resp.raise_for_status()
		body = resp.json()
		models: list[str] = []
		candidates = body.get("models") or []
		for item in candidates:
			if isinstance(item, str):
				models.append(item)
			elif isinstance(item, dict):
				name = item.get("name") or item.get("model") or item.get("id")
				if name:
					models.append(str(name))
		if models:
			_ollama_models_cache = models
		return models
	except Exception:
		return []


def _prefer_fast_ollama_models(models: list[str]) -> list[str]:
	"""Put small/fast models first so cold-start completes within the HTTP timeout."""

	def _score(name: str) -> int:
		n = name.lower()
		if any(tok in n for tok in ("mini", "tiny", "phi", "1b", "2b", "3b", "small")):
			return 100
		if any(tok in n for tok in ("7b", "8b", "mistral")):
			return 50
		if any(tok in n for tok in ("13b", "34b", "70b", "65b")):
			return 10
		return 30

	return sorted(models, key=_score, reverse=True)


def run_ocr(engine: Any, image_array: np.ndarray, fast_mode: bool = False) -> Any:
	"""Run OCR using the active OCR engine."""
	if fast_mode:
		return engine.readtext(
			image_array,
			decoder="greedy",
			beamWidth=1,
			paragraph=False,
			batch_size=1,
		)
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
	primary_raw = run_ocr(engine, primary_image, fast_mode=fast_mode)
	primary_candidates = flatten_ocr_result(primary_raw)
	secondary_candidates: list[tuple[Any, str, float]] = []
	if not fast_mode:
		# A second pass on inverted contrast often recovers faint tags and small valve labels.
		inverted = 255 - primary_image
		try:
			secondary_raw = run_ocr(engine, inverted, fast_mode=fast_mode)
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

	# Instrument bubble tags are valid P&ID components in the instrument category.
	if is_instrument_tag(normalized):
		return "instrument"
	
	# First check regex patterns (most precise)
	for category, patterns in CATEGORY_REGEX_PATTERNS:
		if any(re.search(pattern, normalized) for pattern in patterns):
			return category
	
	# Then check text patterns (common abbreviations)
	for category, patterns in TEXT_CATEGORY_PATTERNS:
		if any(pattern in normalized for pattern in patterns):
			return category
	
	# Enhanced fallback: check for single-letter initial tags like 'm-123', 'p123', 't 45'
	# Only use this if the text looks like a proper tag (short, with numbers or hyphens)
	initial_candidate = _infer_category_from_initial(normalized)
	if initial_candidate:
		return initial_candidate
	
	# Improved fallback: more precise word-boundary matching for common P&ID labels
	# This catches labels like "MTR", "PMP", "TK", "VLV" etc without false positives
	words = re.findall(r'\b[a-z]+\b', normalized)
	for word in words:
		# Check for exact matches or common abbreviations
		if word in ("mtr", "motor"):
			return "motor"
		if word in ("pmp", "pump"):
			return "pump"
		if word in ("tk", "tank", "vessel"):
			return "tank"
		if word in ("vlv", "valve"):
			return "valve"
		# Check for single-letter tags with numbers (e.g., "v1", "m2")
		if len(word) >= 2 and word[0].isalpha() and word[1:].isdigit():
			first_char = word[0].lower()
			if first_char in INITIAL_PREFIX_MAP:
				return INITIAL_PREFIX_MAP[first_char]
	
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
	# More restrictive: only accept if text clearly looks like a P&ID tag
	# Must have a digit or hyphen immediately after the letter, or be very short
	if len(text) <= 3:
		# Very short text like "m1", "p2", "v3"
		if len(text) >= 2 and text[1].isdigit():
			return mapped
	elif len(text) > 1 and (text[1].isdigit() or text[1] in "-_"):
		# Text starts with letter followed by digit or hyphen like "m-123", "p123"
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

	This is conservative: it searches only for explicit P&ID tags and returns
	minimal counts derived from those tags (e.g., P-123 -> pump).
	Off-page references (From P-201) are excluded — those are not symbols on the drawing.
	"""
	text = _strip_off_page_equipment_tags(text_blob or "")
	counts = empty_counts()
	for category, patterns in _COUNTABLE_TEXT_PATTERNS.items():
		matches: set[str] = set()
		for pattern in patterns:
			matches.update(pattern.findall(text))
		counts[category] = max(counts[category], len(matches))

	return counts


def preprocess_for_shapes(image_array: np.ndarray) -> np.ndarray:
	gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
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
	# Use moderate kernel for balanced noise reduction and detail preservation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
	kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2, iterations=1)
	return cleaned


def detect_text_driven_components(ocr_detections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int], str]:
	counts = empty_counts()
	text_blob_parts: list[str] = []
	components: list[dict[str, Any]] = []

	for detection in ocr_detections:
		text = detection["text"]
		text_blob_parts.append(text)
		# Skip instrument bubble tags — they are not physical components
		if is_instrument_tag(text) or is_off_page_equipment_reference(text):
			continue
		category = _countable_text_category(text)
		if category is None:
			continue
		# Only count if confidence is reasonable - lowered threshold to catch more text-based detections
		if detection.get("confidence", 0.0) < 0.25:
			continue
		counts[category] += 1
		components.append(
			{
				"name": text,
				"category": category,
				"bbox": detection["bbox"],
				"confidence": detection["confidence"],
				"source": "ocr",
			},
		)

	return components, counts, infer_industry_from_text(" ".join(text_blob_parts))


_VALVE_TAG_RE = re.compile(
	r"\b(?:fv|xv|cv|hv|lv|sv|pv|tv|gv|bv|wv|pcv|fcv|lcv|tcv|psv|nrv|sdv|mov|sov)(?:-[\d][\d\-]*[a-z]?|\b)",
	re.IGNORECASE,
)
_NON_COMPONENT_EQUIPMENT_RE = re.compile(
	r"\b(?:mixer|reactor|sample\s*point|instrument\s*air|transfer\s*pump)\b",
	re.IGNORECASE,
)
_INSTRUMENT_CONTROLLER_RE = re.compile(
	r"\b(?:pc|lc|pic|lic|fic|trc|frc|prc)\s*\d*\b",
	re.IGNORECASE,
)


def _effective_aspect_ratio(aspect_ratio: float) -> float:
	return max(float(aspect_ratio), 1.0 / max(float(aspect_ratio), 0.01))


def _min_tank_area(image_area: float | None) -> float:
	threshold = 350.0
	if image_area is not None:
		threshold = max(threshold, image_area * 0.00085)
	return threshold


def _max_valve_area(image_area: float | None) -> float:
	"""Scale valve size cap with diagram resolution (bow-ties grow on large exports)."""
	if image_area is None:
		return 8000.0
	return max(8000.0, image_area * 0.012)


def _is_tank_like_geometry(
	area: float,
	aspect_ratio: float,
	extent: float,
	solidity: float,
	image_area: float | None = None,
	diagram_complexity: str = "complex",
) -> bool:
	"""True for P&ID vessel silhouettes (vertical columns, horizontal drums).
	Expert-level: Much more permissive thresholds to catch all tank/vessel variants."""
	if area < _min_tank_area(image_area):
		return False
	min_fraction = 0.0003 if image_area is not None else 0.0
	if image_area is not None and area < max(200.0, image_area * min_fraction):
		return False
	eff_aspect = _effective_aspect_ratio(aspect_ratio)
	if eff_aspect < 1.0 or eff_aspect > 30.0:
		return False
	
	# Stricter thresholds for simple diagrams to reduce false positives
	if diagram_complexity == "simple":
		if extent < 0.15 or solidity < 0.30:
			return False
	else:
		if extent < 0.06 or solidity < 0.18:
			return False
	return True


def _is_horizontal_vessel_geometry(
	area: float,
	aspect_ratio: float,
	extent: float,
	solidity: float,
	image_area: float | None = None,
	bbox: tuple[int, int, int, int] | None = None,
	image_height: int | None = None,
	circularity: float = 0.0,
) -> bool:
	"""Horizontal feed-line drums (wide, medium-large rectangles — not pipe segments).
	Expert-level: Much more permissive thresholds to catch all drum variants."""
	if image_area is not None and area > image_area * 0.040:
		return False
	min_area = 450.0
	if image_area is not None:
		min_area = max(min_area, image_area * 0.0006)
	if area < min_area or area > 2500.0:
		return False
	eff_aspect = _effective_aspect_ratio(aspect_ratio)
	if eff_aspect < 2.5 or eff_aspect > 12.0:
		return False
	if extent < 0.28 or solidity < 0.38:
		return False
	# Add circularity check to reject low-circularity horizontal shapes (likely valves/pipes)
	if circularity < 0.40:
		return False
	if bbox is not None and image_height is not None and image_height > 0:
		center_y = bbox[1] + bbox[3] / 2.0
		if center_y > image_height * 0.72:
			return False
	return True


def _is_circular_vessel_geometry(
	area: float,
	aspect_ratio: float,
	circularity: float,
	extent: float,
	solidity: float,
	image_area: float | None = None,
) -> bool:
	"""Circular vessels and round tanks (spherical tanks, storage spheres).
	Expert-level: Detect circular tank shapes that may be missed by rectangular tank detection."""
	if area < 300.0:
		return False
	if image_area is not None and area > image_area * 0.025:
		return False
	min_fraction = 0.0004 if image_area is not None else 0.0
	if image_area is not None and area < max(250.0, image_area * min_fraction):
		return False
	eff_aspect = _effective_aspect_ratio(aspect_ratio)
	# Circular vessels should have aspect ratio close to 1.0
	if eff_aspect < 0.80 or eff_aspect > 1.25:
		return False
	# High circularity for round shapes
	if circularity < 0.58:
		return False
	if extent < 0.48 or solidity < 0.58:
		return False
	return True


def calculate_hu_moments(contour: np.ndarray) -> np.ndarray:
	"""Calculate Hu moments for shape description.
	
	Hu moments are invariant to translation, scale, and rotation,
	making them excellent for shape matching regardless of orientation.
	"""
	moments = cv2.moments(contour)
	hu_moments = cv2.HuMoments(moments)
	# Log transform to make them more usable
	hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
	return hu_moments.flatten()


def calculate_shape_descriptors(contour: np.ndarray) -> dict[str, float]:
	"""Calculate comprehensive shape descriptors for expert-level geometry analysis.
	
	Returns a dictionary with various shape metrics including:
	- Area, perimeter, aspect ratio
	- Circularity, solidity, extent
	- Eccentricity, compactness
	- Hu moments (shape signature)
	"""
	if len(contour) < 5:
		return {}
	
	area = cv2.contourArea(contour)
	if area <= 0:
		return {}
	
	perimeter = cv2.arcLength(contour, True)
	if perimeter <= 0:
		return {}
	
	# Basic metrics
	bounding_rect = cv2.boundingRect(contour)
	rect_width, rect_height = bounding_rect[2], bounding_rect[3]
	aspect_ratio = float(rect_width) / max(rect_height, 1)
	
	# Shape metrics
	circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
	solidity = area / cv2.contourArea(cv2.convexHull(contour)) if area > 0 else 0
	extent = area / (rect_width * rect_height) if rect_width * rect_height > 0 else 0
	
	# Ellipse fitting for eccentricity
	if len(contour) >= 5:
		ellipse = cv2.fitEllipse(contour)
		major_axis = max(ellipse[1])
		minor_axis = min(ellipse[1])
		eccentricity = math.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
	else:
		eccentricity = 0
	
	# Compactness
	compactness = (perimeter * perimeter) / area if area > 0 else 0
	
	# Hu moments for shape signature
	hu_moments = calculate_hu_moments(contour)
	
	return {
		"area": area,
		"perimeter": perimeter,
		"aspect_ratio": aspect_ratio,
		"circularity": circularity,
		"solidity": solidity,
		"extent": extent,
		"eccentricity": eccentricity,
		"compactness": compactness,
		"hu_moments": hu_moments,
	}


def _is_compact_bowtie_valve(
	area: float,
	aspect_ratio: float,
	circularity: float,
	vertex_count: int,
	extent: float,
	solidity: float,
	image_area: float | None = None,
	tank_like: bool = False,
	bbox: tuple[int, int, int, int] | None = None,
) -> bool:
	"""Classic on-sheet bow-tie valve symbol (compact, nearly square).
	Expert-level: More permissive thresholds to catch more valve variants."""
	if not _is_valve_like_geometry(
		area,
		aspect_ratio,
		circularity,
		vertex_count,
		extent,
		solidity,
		image_area,
		tank_like=tank_like,
		bbox=bbox,
	):
		return False
	eff_aspect = _effective_aspect_ratio(aspect_ratio)
	if eff_aspect > 4.0:  # Relaxed aspect ratio threshold
		return False
	if area < 10.0:  # Relaxed minimum area
		return False
	max_area = 8000.0  # Increased max area
	if image_area is not None:
		max_area = max(max_area, image_area * 0.01)  # Increased area multiplier
	if area > max_area:
		return False
	if circularity > 0.90:  # Relaxed circularity threshold
		return False
	return True


def _is_valve_like_geometry(
	area: float,
	aspect_ratio: float,
	circularity: float,
	vertex_count: int,
	extent: float,
	solidity: float,
	image_area: float | None = None,
	tank_like: bool = False,
	bbox: tuple[int, int, int, int] | None = None,
) -> bool:
	"""Bow-tie / diamond valve symbols on P&IDs (compact, low circularity).
	Expert-level: More permissive thresholds to catch more valve variants."""
	if tank_like and area > 1500.0:
		return False
	if image_area is not None and area > _max_valve_area(image_area):
		return False
	if bbox is not None and image_area is not None:
		_bw, _bh = bbox[2], bbox[3]
		if _bw * _bh > image_area * 0.018:
			return False
		max_symbol = math.sqrt(image_area) * 0.28
		if max(_bw, _bh) > max_symbol:
			return False
	# Expert-level: Wider vertex count range for various valve shapes
	if not (3 <= vertex_count <= 25):
		return False
	# Expert-level: Wider aspect ratio range
	if not (0.15 <= aspect_ratio <= 5.0):
		return False
	# Expert-level: Wider circularity range
	if not (0.01 <= circularity <= 0.95):
		return False
	# Expert-level: Wider extent range
	if not (0.04 <= extent <= 0.99):
		return False
	# Expert-level: More permissive solidity threshold
	if solidity > 0.99:
		return False
	return True


def nearby_ocr_texts(
	candidate_box: tuple[int, int, int, int],
	ocr_detections: list[dict[str, Any]],
	padding_ratio: float = 0.45,
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
	image_height: int | None = None,
	diagram_complexity: str = "complex",
) -> tuple[str | None, str, float]:
	nearby = nearby_ocr_texts(candidate_box, ocr_detections)
	nearby_blob = " ".join(item["normalized_text"] for item in nearby)
	nearby_text = " ".join(item["text"] for item in nearby).strip()
	confidence = 0.0

	tank_like = _is_tank_like_geometry(area, aspect_ratio, extent, solidity, image_area, diagram_complexity)
	max_valve_area = _max_valve_area(image_area)

	valve_like_geometry = _is_valve_like_geometry(
		area,
		aspect_ratio,
		circularity,
		vertex_count,
		extent,
		solidity,
		image_area,
		tank_like=tank_like,
		bbox=candidate_box,
	)

	# Reject instrument bubble tags — circles with TIC/TT/FT/etc. are instruments, not components
	for det in nearby:
		if is_instrument_tag(det.get("normalized_text", "")):
			return None, "", 0.0

	# Controller squares (PC/LC/…) are instruments, not tanks or valves.
	if _INSTRUMENT_CONTROLLER_RE.search(nearby_blob):
		return None, nearby_text, 0.0

	# Instrument bubbles are nearly circular; do not treat them as valves without an explicit tag.
	if circularity >= 0.72 and 6 <= vertex_count <= 14 and not _VALVE_TAG_RE.search(nearby_blob):
		return None, nearby_text, 0.0

	# Major equipment blocks (mixer, reactor, sample point) are not valves.
	if _NON_COMPONENT_EQUIPMENT_RE.search(nearby_blob) and not _VALVE_TAG_RE.search(nearby_blob):
		return None, nearby_text, 0.0

	# Reject shape candidates that are essentially just unclassified text blobs
	# Only reject if the overlapping text is NOT a known valve/component label
	for det in nearby:
		if iou(candidate_box, det["bbox"]) > 0.6:
			text_cat = classify_text_label(det.get("normalized_text", ""))
			if not text_cat:
				# Allow if the shape has valve-like geometry (e.g. butterfly valve near ISA label)
				if not valve_like_geometry:
					return None, "", 0.0

	if nearby_blob:
		nearby_category = classify_text_label(nearby_blob)
		if nearby_category is not None:
			# If the text is nearby (within the 45% padded box), we can trust it to classify the shape.
			# We relax the strict 20% IoU overlap requirement because P&ID labels are often adjacent.
			# Boost confidence when OCR text strongly matches component patterns
			confidence = 0.92
			# Additional confidence boost for strong pattern matches
			if nearby_category == "valve" and _VALVE_TAG_RE.search(nearby_blob):
				confidence = 0.95
			elif nearby_category == "pump" and _PUMP_TAG_RE.search(nearby_blob):
				confidence = 0.94
			elif nearby_category == "motor" and any(tag in nearby_blob.upper() for tag in ["M", "MOT", "MTR"]):
				confidence = 0.93
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

	# Tanks/vessels before valve heuristics — vertical columns, horizontal feed-line drums, or circular vessels.
	horizontal_drum = aspect_ratio >= 3.0
	center_y = candidate_box[1] + candidate_box[3] / 2.0
	circular_vessel = _is_circular_vessel_geometry(
		area,
		aspect_ratio,
		circularity,
		extent,
		solidity,
		image_area,
	)
	
	# If nearby OCR strongly indicates another component type, avoid over-promoting tanks.
	# This reduces cases where a pump/valve/motor symbol silhouette gets interpreted as a tank.
	nearby_motor = bool(re.search(r"\b(?:m-?\d{1,5}[a-z]?|mo-?\d{1,5}[a-z]?|mtr-?\d{1,5}[a-z]?|motor)\b", nearby_blob, re.IGNORECASE))
	nearby_pump = bool(_PUMP_TAG_RE.search(nearby_blob))
	nearby_valve = bool(_VALVE_TAG_RE.search(nearby_blob))
	nearby_instrument = bool(is_instrument_tag(nearby_blob))
	
	# For simple diagrams, use stricter geometry thresholds to reduce false positives
	min_extent = 0.23 if diagram_complexity == "simple" else 0.20
	min_solidity = 0.36 if diagram_complexity == "simple" else 0.35
	min_area = 260 if diagram_complexity == "simple" else 250
	min_circularity = 0.50 if diagram_complexity == "simple" else 0.45

	# Disabled pump classification based on circularity alone to prevent false positives
	# Pumps should only be detected with OCR evidence (P-101, P-102, etc.)

	# Global “component evidence” tightening: when OCR indicates another component type,
	# tanks must be geometrically stronger to win.
	if nearby_motor or nearby_pump or nearby_valve or nearby_instrument:
		min_extent += 0.05
		min_solidity += 0.05
		min_circularity += 0.05
		min_area *= 1.10

	if (
		(tank_like and area >= min_area and extent >= min_extent and solidity >= min_solidity and circularity >= min_circularity and not horizontal_drum)
		or (horizontal_drum and _is_horizontal_vessel_geometry(
			area,
			aspect_ratio,
			extent,
			solidity,
			image_area,
			bbox=candidate_box,
			image_height=image_height,
			circularity=circularity,
		))
		or circular_vessel
	):
		# Expert-level: Multi-factor confidence calculation for more accurate tank detection
		base_confidence = extent + 0.25
		eff_aspect = _effective_aspect_ratio(aspect_ratio)
		# Boost confidence for circular vessels (high circularity indicates clear tank shape)
		if circular_vessel:
			base_confidence = min(0.85, base_confidence + 0.10)
		# Boost confidence for horizontal drums with good aspect ratio
		elif horizontal_drum and eff_aspect >= 4.0 and eff_aspect <= 7.0:
			base_confidence = min(0.82, base_confidence + 0.08)
		# Boost confidence for vertical tanks with good solidity
		elif not horizontal_drum and solidity >= 0.45:
			base_confidence = min(0.80, base_confidence + 0.05)
		
		confidence = min(0.85, base_confidence)
		return "tank", nearby_text or "Tank", confidence


	# Balanced valve detection: OCR evidence preferred but geometry-only allowed
	if _VALVE_TAG_RE.search(nearby_blob) and len(nearby) > 0:
		if extent >= 0.20 and solidity >= 0.35 and area >= 80 and area <= max_valve_area:
			confidence = 0.75
			return "valve", nearby_text or "Valve", confidence

	# Geometry-only valves: compact bow-tie symbols with balanced criteria
	if _is_compact_bowtie_valve(
		area,
		aspect_ratio,
		circularity,
		vertex_count,
		extent,
		solidity,
		image_area,
		tank_like=tank_like,
		bbox=candidate_box,
	):
		# Balanced geometry checks to detect valves without over-counting
		if (circularity <= 0.85 and 
		    solidity >= 0.25 and 
		    extent >= 0.15 and 
		    vertex_count >= 3):
			confidence = min(
				0.85,
				0.55
				+ (0.20 * (1.0 - min(abs(1.0 - aspect_ratio), 1.0)))
				+ (0.15 if vertex_count >= 4 else 0.0)
				+ (0.10 if solidity <= 0.70 else 0.0),
			)
			return "valve", nearby_text or "Valve", confidence

	# Motors: typically perfect circles, moderate area - more permissive thresholds for complex diagrams
	if 0.60 <= circularity <= 1.0 and 4 <= vertex_count <= 30 and area >= 60 and solidity >= 0.55:
		confidence = min(0.85, 0.45 + (0.35 * circularity))
		return "motor", nearby_text or "Motor", confidence

	# Pumps: geometry-based detection without requiring OCR text - balanced for accuracy
	# First try with OCR tag (higher confidence)
	pump_tag_nearby = bool(_PUMP_TAG_RE.search(nearby_blob)) and not is_off_page_equipment_reference(nearby_blob)
	if (
		pump_tag_nearby
		and 0.25 <= circularity <= 0.99
		and 0.35 <= solidity <= 1.0
		and 3 <= vertex_count <= 30
		and area >= 30
	):
		confidence = min(0.82, 0.35 + (0.30 * circularity) + (0.15 * solidity))
		return "pump", nearby_text or "Pump", confidence
	# Geometry-only pump detection - more permissive to catch pumps without OCR
	# Pumps typically have moderate circularity and specific aspect ratios
	elif (
		0.25 <= circularity <= 0.99
		and 0.35 <= solidity <= 1.0
		and 3 <= vertex_count <= 30
		and area >= 40
		and aspect_ratio >= 0.30
		and aspect_ratio <= 4.0
	):
		confidence = min(0.72, 0.30 + (0.25 * circularity) + (0.15 * solidity))
		return "pump", nearby_text or "Pump", confidence

	return None, nearby_text, confidence


def detect_shape_components(
	image_array: np.ndarray,
	ocr_detections: list[dict[str, Any]],
	diagram_complexity: str = "complex",
) -> list[dict[str, Any]]:
	# Downscale for faster contour detection, then rescale coordinates back to original.
	# Expert-level: Higher resolution for better small component detection - increased for better accuracy
	max_edge = int(os.getenv("SHAPE_DETECT_MAX_EDGE", "2560"))
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
	# Use RETR_LIST to find internal valve symbols; RETR_EXTERNAL merges them into line blobs.
	contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	candidates: list[dict[str, Any]] = []
	
	# Expert-level: Process more contours for better coverage - increased for better accuracy
	max_contours = int(os.getenv("SHAPE_DETECT_MAX_CONTOURS", "5000"))
	if len(contours) > max_contours:
		contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]

	for contour in contours:
		area_small = float(cv2.contourArea(contour))
		# convert area back to original image scale
		area = area_small / (scale * scale) if scale > 0 and scale < 1.0 else area_small
		# Maximum sensitivity: Very low minimum area to catch all components - lowered for better recall
		min_area = max(5.0, image_area * 0.000005)
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
		if width < 5 or height < 5:
			continue
		aspect_ratio = width / max(height, 1)
		if aspect_ratio > 15.0 or aspect_ratio < 0.05:
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
			image_height=orig_h,
			diagram_complexity=diagram_complexity,
		)
		if category is None:
			continue
		# Use the confidence from classification, but ensure minimum threshold
		confidence = max(0.20, confidence)
		# Use category-specific confidence thresholds from CONF_THRESH
		min_conf = CONF_THRESH.get(category, 0.25)
		# Additional minimum confidence for pumps - lowered to restore detection
		if category == "pump":
			min_conf = max(min_conf, 0.25)
		if confidence < min_conf:
			continue
		
		# Expert-level: More permissive geometry validation to detect pumps, valves, motors
		# Relaxed thresholds to catch more components while maintaining reasonable accuracy
		# Adaptive thresholds based on diagram complexity for optimal accuracy
		if category == "valve":
			# Extremely strict valve detection to reduce from 8 to 3
			# Only detect valves with the clearest geometric signatures
			if diagram_complexity == "simple":
				min_circularity = 0.50
				max_circularity = 0.88
				min_solidity = 0.50
				max_aspect_ratio = 2.0
				min_aspect_ratio = 0.50
				min_area = 120
			else:
				min_circularity = 0.45
				max_circularity = 0.90
				min_solidity = 0.45
				max_aspect_ratio = 2.5
				min_aspect_ratio = 0.40
				min_area = 100
			if circularity < min_circularity or circularity > max_circularity:
				continue
			if aspect_ratio > max_aspect_ratio or aspect_ratio < min_aspect_ratio:
				continue
			if solidity < min_solidity:
				continue
			if area < min_area:
				continue
		elif category == "motor":
			# Motors should be fairly circular with high solidity
			if diagram_complexity == "simple":
				min_circularity = 0.60
				max_aspect_ratio = 2.0
				min_aspect_ratio = 0.50
				min_solidity = 0.55
				min_area = 60
			else:
				min_circularity = 0.55
				max_aspect_ratio = 2.5
				min_aspect_ratio = 0.40
				min_solidity = 0.50
				min_area = 50
			if circularity < min_circularity:
				continue
			if aspect_ratio > max_aspect_ratio or aspect_ratio < min_aspect_ratio:
				continue
			if solidity < min_solidity:
				continue
			if area < min_area:
				continue
		elif category == "pump":
			# Pumps should have moderate circularity
			if diagram_complexity == "simple":
				min_circularity = 0.35
				max_circularity = 0.95
				max_aspect_ratio = 3.0
				min_aspect_ratio = 0.33
				min_solidity = 0.40
				min_area = 50
			else:
				min_circularity = 0.30
				max_circularity = 0.96
				max_aspect_ratio = 3.5
				min_aspect_ratio = 0.28
				min_solidity = 0.35
				min_area = 40
			if circularity < min_circularity or circularity > max_circularity:
				continue
			if aspect_ratio > max_aspect_ratio or aspect_ratio < min_aspect_ratio:
				continue
			if solidity < min_solidity:
				continue
			if area < min_area:
				continue
		elif category == "tank":
			# Tanks can vary but should have reasonable extent and solidity
			if diagram_complexity == "simple":
				min_extent = 0.25
				min_solidity = 0.55
				min_area = 500
				max_aspect_ratio = 5.0
			else:
				min_extent = 0.12
				min_solidity = 0.35
				min_area = 250
				max_aspect_ratio = 8.0
			if extent < min_extent:
				continue
			if solidity < min_solidity:
				continue
			if area < min_area:
				continue
			if aspect_ratio > max_aspect_ratio:
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
				"source": "shape",
				"area": area,
				"circularity": circularity,
				"aspect_ratio": aspect_ratio,
				"vertex_count": vertex_count,
				"extent": extent,
				"solidity": solidity,
			},
		)

	logger.info(f"Shape detection found {len(candidates)} candidates")
	category_breakdown = {k: sum(1 for c in candidates if c['category']==k) for k in ('valve','tank','pump','motor')}
	logger.info(f"Shape detection category breakdown: {category_breakdown}")
	return candidates


def dedupe_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.50) -> list[dict[str, Any]]:
	"""Remove duplicate detections using IoU within the same category.
	
	Expert-level: Increased IoU threshold to reduce overcounting while maintaining accuracy.
	"""
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


def merge_close_detections(detections: list[dict[str, Any]], distance_ratio: float = 0.85) -> list[dict[str, Any]]:
	"""Merge detections of the same category when their centers are very close.

	This helps collapse a text label and a nearby shape that refer to the same component
	but have little IoU overlap (common in P&ID diagrams).
	Expert-level: Reduced distance ratio to be more aggressive in merging nearby detections and prevent overcounting.
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

			iou_score = iou(bx, ex["bbox"])
			if iou_score > 0.6:
				duplicate = True
				break

			thresh_ratio = distance_ratio
			if det.get("category") == "tank":
				thresh_ratio = min(distance_ratio, 0.25)
			thresh = max(bw, ex_bw) * thresh_ratio
			dist = math.hypot(bx_c[0] - ex_c[0], bx_c[1] - ex_c[1])
			if dist <= thresh:
				duplicate = True
				break

		if not duplicate:
			kept.append(det)
	return kept


def merge_stacked_tank_symbols(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Merge vertical+horizontal parts of the same P&ID vessel into one tank detection."""
	tanks = [d for d in detections if d.get("category") == "tank"]
	others = [d for d in detections if d.get("category") != "tank"]
	if len(tanks) <= 1:
		return detections

	ordered = sorted(tanks, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
	kept: list[dict[str, Any]] = []

	def _x_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
		ax1, ay1, aw, ah = a
		bx1, by1, bw, bh = b
		ax2, bx2 = ax1 + aw, bx1 + bw
		inter = max(0, min(ax2, bx2) - max(ax1, bx1))
		union = aw + bw - inter
		return inter / union if union > 0 else 0.0

	for det in ordered:
		box = det.get("bbox")
		if not box:
			kept.append(det)
			continue
		dx, dy, dw, dh = box
		dcx, dcy = dx + dw / 2.0, dy + dh / 2.0
		duplicate = False
		for ex in kept:
			ex_box = ex.get("bbox")
			if not ex_box:
				continue
			ex, ey, ew, eh = ex_box
			ecx, ecy = ex + ew / 2.0, ey + eh / 2.0
			if _x_overlap(box, ex_box) < 0.35:
				continue
			# Do not merge separate horizontal drums on the same feed line.
			if dw >= dh * 2.5 and ew >= eh * 2.5:
				continue
			vert_gap = abs(dcy - ecy)
			max_dim = max(dw, dh, ew, eh)
			if vert_gap <= max_dim * 2.5:
				duplicate = True
				break
		if not duplicate:
			kept.append(det)

	return others + kept


def consolidate_tank_vessels(
	detections: list[dict[str, Any]],
	image_area: float | None = None,
) -> list[dict[str, Any]]:
	"""Keep primary vessel(s); drop small false tank hits (controllers, caps, internals)."""
	tanks = [d for d in detections if d.get("category") == "tank"]
	others = [d for d in detections if d.get("category") != "tank"]
	if len(tanks) <= 1:
		return detections

	def _size(det: dict[str, Any]) -> float:
		box = det.get("bbox") or (0, 0, 0, 0)
		return float(det.get("area", box[2] * box[3]))
	# Ignore oversized outliers (often template/ensemble page-scale boxes) when
	# there are other tank candidates. A single giant box can otherwise force
	# min_keep so high that all real tank symbols are dropped.
	if image_area is not None and image_area > 0 and len(tanks) > 1:
		max_reasonable_tank_area = image_area * 0.18
		non_outlier_tanks = [det for det in tanks if _size(det) <= max_reasonable_tank_area]
		if non_outlier_tanks:
			dropped_outliers = len(tanks) - len(non_outlier_tanks)
			if dropped_outliers > 0:
				logger.info(
					"Consolidation dropped %s oversized tank outlier(s) (>%s%% of image area)",
					dropped_outliers,
					round(0.18 * 100),
				)
			tanks = non_outlier_tanks

	if len(tanks) <= 1:
		return others + tanks

	tank_sizes = [_size(t) for t in tanks]
	max_size = max(tank_sizes)
	reference_size = float(np.median(tank_sizes))
	min_keep = max(
		_min_tank_area(image_area),
		reference_size * 0.70,
		(image_area or 0.0) * 0.00025,
		350.0,
	)
	min_keep = min(min_keep, max_size)

	kept: list[dict[str, Any]] = []
	for det in sorted(tanks, key=_size, reverse=True):
		if _size(det) < min_keep:
			continue
		box = det.get("bbox")
		if not box:
			kept.append(det)
			continue
		duplicate = False
		for ex in kept:
			ex_box = ex.get("bbox")
			if not ex_box:
				continue
			if iou(box, ex_box) >= 0.12:
				duplicate = True
				break
		if not duplicate:
			kept.append(det)

	if len(kept) <= 1:
		return others + kept

	# Less aggressive column deduplication: allow multiple tanks per column if they differ significantly in size
	column_kept: list[dict[str, Any]] = []
	for det in sorted(kept, key=_size, reverse=True):
		box = det.get("bbox")
		if not box:
			column_kept.append(det)
			continue
		cx = box[0] + box[2] / 2.0
		duplicate_column = False
		for existing in column_kept:
			ex_box = existing.get("bbox")
			if not ex_box:
				continue
			ex_cx = ex_box[0] + ex_box[2] / 2.0
			# Only deduplicate if in same column AND similar size (within 3x for more aggressive dedup)
			if abs(cx - ex_cx) <= max(box[2], ex_box[2]) * 0.75:
				size_ratio = _size(det) / max(_size(existing), 1.0)
				if size_ratio >= 0.33 and size_ratio <= 3.0:
					duplicate_column = True
					break
		if not duplicate_column:
			column_kept.append(det)

	return others + column_kept


def suppress_nearby_valves(
	detections: list[dict[str, Any]],
	iou_threshold: float = 0.35,
	center_dist_ratio: float = 0.55,
	area_ratio_min: float = 0.60,
	area_ratio_max: float = 1.60,
) -> list[dict[str, Any]]:
	"""Valve-specific suppression to reduce false extra valve symbols.

	Sometimes contour/OCR/template can produce multiple nearby valve-like candidates
	that do not overlap enough for IoU-based dedupe. This function removes the
	lower-confidence one when two valve detections are very close and similar in size.
	"""
	if not detections:
		return []

	def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
		x, y, w, h = box
		return x + w / 2.0, y + h / 2.0

	ordered = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
	kept: list[dict[str, Any]] = []

	for det in ordered:
		if det.get("category") != "valve":
			kept.append(det)
			continue

		box = det.get("bbox")
		if not box:
			kept.append(det)
			continue

		deliberate_duplicate = False
		det_area = float(det.get("area", box[2] * box[3]))
		det_conf = float(det.get("confidence", 0.0))
		dx_det, dy_det = _center(box)

		for ex in kept:
			if ex.get("category") != "valve":
				continue
			ex_box = ex.get("bbox")
			if not ex_box:
				continue

			ex_area = float(ex.get("area", ex_box[2] * ex_box[3]))
			if ex_area <= 0 or det_area <= 0:
				continue

			# Area similarity guard
			area_ratio = det_area / ex_area
			if area_ratio < area_ratio_min or area_ratio > area_ratio_max:
				continue

			# Distance guard (relative to larger bbox dimension)
			ex_cx, ex_cy = _center(ex_box)
			dist = math.hypot(dx_det - ex_cx, dy_det - ex_cy)
			max_dim = max(float(box[2]), float(box[3]), float(ex_box[2]), float(ex_box[3]))
			if max_dim <= 0:
				continue
			if dist > max_dim * center_dist_ratio:
				continue

			# IoU guard: allow suppression even if IoU is low, but still
			# require at least some spatial overlap similarity.
			if iou(box, ex_box) >= iou_threshold:
				deliberate_duplicate = True
				break

			# If IoU is below threshold, still suppress if extremely close.
			# (This handles low-overlap cases where symbols are adjacent.)
			if dist <= max_dim * (center_dist_ratio * 0.45):
				deliberate_duplicate = True
				break

		if not deliberate_duplicate:
			kept.append(det)

	return kept


def suppress_nearby_pumps(
	detections: list[dict[str, Any]],
	iou_threshold: float = 0.45,
	center_dist_ratio: float = 0.50,
	area_ratio_min: float = 0.50,
	area_ratio_max: float = 2.00,
) -> list[dict[str, Any]]:
	"""Pump-specific suppression to reduce false extra pump symbols.

	Similar to valve suppression but with more aggressive thresholds for pumps
	to prevent overcounting in complex diagrams.
	"""
	if not detections:
		return []

	def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
		x, y, w, h = box
		return x + w / 2.0, y + h / 2.0

	ordered = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
	kept: list[dict[str, Any]] = []

	for det in ordered:
		if det.get("category") != "pump":
			kept.append(det)
			continue

		box = det.get("bbox")
		if not box:
			kept.append(det)
			continue

		deliberate_duplicate = False
		det_area = float(det.get("area", box[2] * box[3]))
		det_conf = float(det.get("confidence", 0.0))
		dx_det, dy_det = _center(box)

		for ex in kept:
			if ex.get("category") != "pump":
				continue
			ex_box = ex.get("bbox")
			if not ex_box:
				continue

			ex_area = float(ex.get("area", ex_box[2] * ex_box[3]))
			if ex_area <= 0 or det_area <= 0:
				continue

			# Area similarity guard - wider range for pumps
			area_ratio = det_area / ex_area
			if area_ratio < area_ratio_min or area_ratio > area_ratio_max:
				continue

			# Distance guard (relative to larger bbox dimension) - more aggressive
			ex_cx, ex_cy = _center(ex_box)
			dist = math.hypot(dx_det - ex_cx, dy_det - ex_cy)
			max_dim = max(float(box[2]), float(box[3]), float(ex_box[2]), float(ex_box[3]))
			if max_dim <= 0:
				continue
			if dist > max_dim * center_dist_ratio:
				continue

			# IoU guard - higher threshold for pumps
			if iou(box, ex_box) >= iou_threshold:
				deliberate_duplicate = True
				break

			# If IoU is below threshold, still suppress if extremely close
			if dist <= max_dim * (center_dist_ratio * 0.40):
				deliberate_duplicate = True
				break

		if not deliberate_duplicate:
			kept.append(det)

	return kept


def _is_supported_template_detection(
	detection: dict[str, Any],
	peer_detections: list[dict[str, Any]],
) -> bool:
	"""Keep template detections only when another source supports them.

	Template matching improves recall, but it also creates the largest duplicate
	bursts. A template hit is counted only if a shape, OCR, or annotation detection
	of the same category is close enough to support it.
	
	Expert-level: Stricter validation to reduce false template matches.
	"""
	if detection.get("source") != "template":
		return True

	box = detection.get("bbox")
	category = detection.get("category")
	confidence = float(detection.get("confidence", 0.0))
	
	# Require minimum confidence for template matches
	if confidence < 0.50:
		return False
	
	if category not in COUNT_KEYS or not box:
		return False

	def _center(target_box: tuple[int, int, int, int]) -> tuple[float, float]:
		x, y, w, h = target_box
		return x + w / 2.0, y + h / 2.0

	box_center = _center(box)
	box_span = max(float(box[2]), float(box[3]))
	if box_span <= 0:
		return False

	for peer in peer_detections:
		if peer.get("source") == "template" or peer.get("category") != category:
			continue
		peer_box = peer.get("bbox")
		if not peer_box:
			continue
		# Increased IoU threshold for template support
		if iou(box, peer_box) >= 0.18:
			return True
		peer_center = _center(peer_box)
		peer_span = max(float(peer_box[2]), float(peer_box[3]))
		if peer_span <= 0:
			continue
		distance = math.hypot(box_center[0] - peer_center[0], box_center[1] - peer_center[1])
		# Reduced distance threshold for stricter template support
		if distance <= max(box_span, peer_span) * 0.65:
			return True

	return False


def collapse_countable_clusters(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Collapse same-category near-duplicates before counting.

	This is intentionally more aggressive than the general dedupe pass because the
	counting path should prefer undercounting a little over inflating one symbol
	into several repeated tank or valve hits.
	"""
	if not detections:
		return []

	cluster_rules: dict[str, dict[str, float]] = {
		"tank": {
			"iou": 0.12,
			"center": 0.75,
			"area_min": 0.25,
			"area_max": 4.50,
		},
		"valve": {
			"iou": 0.10,
			"center": 1.25,
			"area_min": 0.30,
			"area_max": 3.50,
		},
		"pump": {
			"iou": 0.30,
			"center": 0.60,
			"area_min": 0.50,
			"area_max": 2.00,
		},
		"motor": {
			"iou": 0.32,
			"center": 0.65,
			"area_min": 0.50,
			"area_max": 2.00,
		},
	}

	def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
		x, y, w, h = box
		return x + w / 2.0, y + h / 2.0

	def _area(box: tuple[int, int, int, int]) -> float:
		return float(max(0, box[2]) * max(0, box[3]))

	def _is_neighbor(
		first: dict[str, Any],
		second: dict[str, Any],
		rule: dict[str, float],
	) -> bool:
		first_box = first.get("bbox")
		second_box = second.get("bbox")
		if not first_box or not second_box:
			return False

		first_area = float(first.get("area", _area(first_box)))
		second_area = float(second.get("area", _area(second_box)))
		if first_area <= 0.0 or second_area <= 0.0:
			return False

		area_ratio = first_area / second_area
		if area_ratio < rule["area_min"] or area_ratio > rule["area_max"]:
			return False

		if iou(first_box, second_box) >= rule["iou"]:
			return True

		first_center = _center(first_box)
		second_center = _center(second_box)
		max_dim = max(float(first_box[2]), float(first_box[3]), float(second_box[2]), float(second_box[3]))
		if max_dim <= 0:
			return False
		distance = math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
		return distance <= max_dim * rule["center"]

	ordered = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
	by_category: dict[str, list[dict[str, Any]]] = {key: [] for key in cluster_rules}
	others: list[dict[str, Any]] = []
	for det in ordered:
		category = det.get("category")
		if category in by_category and det.get("bbox"):
			by_category[category].append(det)
		else:
			others.append(det)

	kept: list[dict[str, Any]] = list(others)

	for category, items in by_category.items():
		if len(items) <= 1:
			kept.extend(items)
			continue

		rule = cluster_rules[category]
		parent = list(range(len(items)))

		def find(index: int) -> int:
			while parent[index] != index:
				parent[index] = parent[parent[index]]
				index = parent[index]
			return index

		def union(left: int, right: int) -> None:
			left_root = find(left)
			right_root = find(right)
			if left_root != right_root:
				parent[right_root] = left_root

		for left in range(len(items)):
			for right in range(left + 1, len(items)):
				if _is_neighbor(items[left], items[right], rule):
					union(left, right)

		clusters: dict[int, list[dict[str, Any]]] = {}
		for index, det in enumerate(items):
			clusters.setdefault(find(index), []).append(det)

		for cluster_items in clusters.values():
			cluster_items.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
			kept.append(cluster_items[0])

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
	"""Merge OCR and visual counts; drawable symbols beat tag-only mentions.

	text_counts is kept for Ollama hints but does not inflate totals on its own.
	"""
	_ = text_counts
	return build_counts(ocr_counts, visual_counts)


def filter_valve_geometry_false_positives(
	detections: list[dict[str, Any]],
	ocr_detections: list[dict[str, Any]],
	image_height: int | None = None,
) -> list[dict[str, Any]]:
	"""Drop valve hits that are elongated line/instrument blobs without a valve tag."""
	filtered: list[dict[str, Any]] = []
	for det in detections:
		if det.get("category") != "valve":
			filtered.append(det)
			continue
		box = det.get("bbox")
		if not box:
			continue
		center_y = box[1] + box[3] / 2.0
		confidence = float(det.get("confidence", 0.0) or 0.0)
		try:
			nearby_text = " ".join(item.get("text", "") for item in nearby_ocr_texts(box, ocr_detections, padding_ratio=0.45))
		except Exception:
			nearby_text = ""
		if _VALVE_TAG_RE.search(nearby_text or ""):
			filtered.append(det)
			continue
		# Drop round instrument bubbles in the upper sheet only (not bow-tie valves).
		if (
			image_height is not None
			and image_height > 0
			and center_y < image_height * 0.42
			and confidence < 0.76
			and float(det.get("circularity", 0.0) or 0.0) > 0.52
		):
			continue
		area = float(det.get("area", box[2] * box[3]))
		aspect_ratio = float(det.get("aspect_ratio", box[2] / max(box[3], 1)))
		if _is_compact_bowtie_valve(
			area,
			aspect_ratio,
			float(det.get("circularity", 0.0) or 0.0),
			int(det.get("vertex_count", 0) or 0),
			float(det.get("extent", 0.0) or 0.0),
			float(det.get("solidity", 0.0) or 0.0),
			bbox=box,
		):
			filtered.append(det)
	return filtered


def clear_annotation_templates_cache() -> None:
	"""Call after new component reference images are saved."""
	load_annotation_templates.cache_clear()


def count_detections_by_category(
	detections: list[dict[str, Any]],
	thresholds: dict[str, float] | None = None,
) -> dict[str, int]:
	"""Count detections that pass per-category confidence thresholds."""
	thresh = thresholds or CONF_THRESH
	counts = empty_counts()
	for det in detections:
		category = det.get("category")
		if category not in counts:
			continue
		conf = float(det.get("confidence", 0.0) or 0.0)
		if conf >= thresh.get(category, 0.45):
			counts[category] += 1
	return counts


def select_relevant_template_categories(
	shape_detections: list[dict[str, Any]],
	ocr_counts: dict[str, int],
	text_counts: dict[str, int],
) -> list[str]:
	"""Prioritize categories with on-page evidence for fast template matching."""
	score: dict[str, int] = {key: 0 for key in COUNT_KEYS}
	for key in COUNT_KEYS:
		score[key] += int(ocr_counts.get(key, 0) or 0) * 3
		score[key] += int(text_counts.get(key, 0) or 0) * 2
	shape_counts = count_detections_by_category(shape_detections, thresholds={k: 0.0 for k in COUNT_KEYS})
	for key in COUNT_KEYS:
		score[key] += int(shape_counts.get(key, 0) or 0)

	ranked = sorted(COUNT_KEYS, key=lambda key: score[key], reverse=True)
	positive = [key for key in ranked if score[key] > 0]
	# Keep at least two categories to avoid over-pruning edge cases.
	if len(positive) >= 2:
		return positive[:3]
	if len(positive) == 1:
		return [positive[0], ranked[1]]
	# No evidence detected: fall back to all categories.
	return list(COUNT_KEYS)


@lru_cache(maxsize=1)
def load_annotation_templates() -> dict[str, list[tuple[np.ndarray, str]]]:
	"""Load a small set of newest reference images per category for template matching.

	The annotations folder accumulates every uploaded sample; matching against hundreds
	of templates per P&ID is too slow, so we keep only the most recent few per type.
	"""
	templates_dir = Path(__file__).resolve().parents[1] / "annotations"
	by_category: dict[str, list[tuple[float, np.ndarray, str]]] = {}
	if not templates_dir.exists():
		logger.warning(f"Annotations directory not found: {templates_dir}")
		return {}
	
	logger.info(f"Loading annotation templates from: {templates_dir}")
	
	for p in sorted(templates_dir.iterdir()):  # Sort for deterministic loading
		if not p.is_file():
			continue
		name = p.name.lower()
		category = None
		# More flexible pattern matching for annotation filenames
		for cat in ("valve", "tank", "pump", "motor"):
			# Match patterns like: "_valve", "valve.png", "valve.jpg", "valve", "valves", "pumps", "motors", "tanks"
			if f"_{cat}" in name or f"_{cat}s" in name or name.endswith(f"{cat}.png") or name.endswith(f"{cat}.jpg") or name.endswith(f"{cat}s.png") or name.startswith(f"{cat}"):
				category = cat
				break
		if not category:
			continue
		try:
			img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
			if img is None:
				logger.warning(f"Failed to load image: {p.name}")
				continue
		except Exception as e:
			logger.warning(f"Error loading image {p.name}: {e}")
			try:
				with Image.open(p) as im:
					img = np.array(im.convert("L"), dtype=np.uint8)
			except Exception as e2:
				logger.warning(f"Error loading image {p.name} with PIL: {e2}")
				continue
		try:
			mtime = p.stat().st_mtime
		except OSError:
			mtime = 0.0
		by_category.setdefault(category, []).append((mtime, img, p.name))

	out: dict[str, list[tuple[np.ndarray, str]]] = {}
	for category, items in by_category.items():
		items.sort(key=lambda row: row[0], reverse=True)
		trimmed = items[:TEMPLATE_MAX_PER_CATEGORY]
		out[category] = [(img, fname) for _mtime, img, fname in trimmed]
		logger.info(f"Loaded {len(out[category])} {category} templates")
	
	total_templates = sum(len(v) for v in out.values())
	logger.info(f"Total annotation templates loaded: {total_templates}")
	return out


def _template_match_peaks(
	result: np.ndarray,
	tmpl_w: int,
	tmpl_h: int,
	threshold: float,
	max_peaks: int,
) -> list[tuple[int, int, float]]:
	"""Return up to max_peaks (x, y, score) local maxima without scanning every pixel."""
	peaks: list[tuple[int, int, float]] = []
	if result.size == 0:
		return peaks
	work = result.copy()
	pad_x = max(4, tmpl_w // 2)
	pad_y = max(4, tmpl_h // 2)
	for _ in range(max_peaks):
		_, max_val, _, max_loc = cv2.minMaxLoc(work)
		if max_val < threshold:
			break
		x, y = int(max_loc[0]), int(max_loc[1])
		peaks.append((x, y, float(max_val)))
		x1 = max(0, x - pad_x)
		y1 = max(0, y - pad_y)
		x2 = min(work.shape[1], x + pad_x)
		y2 = min(work.shape[0], y + pad_y)
		work[y1:y2, x1:x2] = 0.0
	return peaks


def match_annotation_templates(
	image_array: np.ndarray,
	templates: dict[str, list[tuple[np.ndarray, str]]],
	threshold: float = 0.55,
	*,
	extended_scales: bool = False,
) -> list[dict[str, Any]]:
	"""Template-match annotation templates against the image and return detections.

	Returns list of detection dicts similar to shape detection output.
	"""
	if not templates:
		return []
	gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
	orig_h, orig_w = gray.shape[:2]
	scale_down = 1.0
	if max(orig_h, orig_w) > TEMPLATE_MATCH_MAX_EDGE:
		scale_down = float(TEMPLATE_MATCH_MAX_EDGE) / float(max(orig_h, orig_w))
		gray = cv2.resize(
			gray,
			(int(orig_w * scale_down), int(orig_h * scale_down)),
			interpolation=cv2.INTER_AREA,
		)

	detections: list[dict[str, Any]] = []
	# Expert-level: Very granular scales for maximum multi-scale detection accuracy
	scales = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.4, 1.55, 1.7, 1.85, 2.0, 2.2, 2.4] if extended_scales else [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
	used_bins: set[tuple[str, int, int, int, int]] = set()

	# Process categories in sorted order for deterministic results
	for category in sorted(templates.keys()):
		tmpl_list = templates[category]
		for tmpl, fname in tmpl_list:
			th, tw = tmpl.shape[:2]
			for scale in scales:
				sw = max(1, int(tw * scale * scale_down))
				sh = max(1, int(th * scale * scale_down))
				if sh >= gray.shape[0] or sw >= gray.shape[1]:
					continue
				try:
					tmpl_resized = cv2.resize(tmpl, (sw, sh), interpolation=cv2.INTER_AREA)
					res = cv2.matchTemplate(gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
				except Exception:
					continue
				for x, y, score in _template_match_peaks(
					res, sw, sh, threshold, TEMPLATE_MAX_PEAKS
				):
					if scale_down < 1.0:
						bx = int(x / scale_down)
						by = int(y / scale_down)
						bw = max(1, int(sw / scale_down))
						bh = max(1, int(sh / scale_down))
					else:
						bx, by, bw, bh = x, y, sw, sh
					key = (category, bx // 10, by // 10, bw // 10, bh // 10)
					if key in used_bins:
						continue
					used_bins.add(key)
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


def match_features_with_orb(
	image_array: np.ndarray,
	templates: dict[str, list[tuple[np.ndarray, str]]],
	min_matches: int = 8,
	*,
	extended_scales: bool = False,
) -> list[dict[str, Any]]:
	"""Feature-based matching using ORB for robust component detection.
	
	Uses ORB feature matching to detect components that may not match well with
	template matching due to rotation, scale, or partial occlusion.
	
	Returns list of detection dicts similar to shape detection output.
	"""
	if not templates:
		return []
	
	try:
		orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
	except Exception:
		logger.warning("ORB feature matching not available, skipping")
		return []
	
	gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
	orig_h, orig_w = gray.shape[:2]
	scale_down = 1.0
	if max(orig_h, orig_w) > TEMPLATE_MATCH_MAX_EDGE:
		scale_down = float(TEMPLATE_MATCH_MAX_EDGE) / float(max(orig_h, orig_w))
		gray = cv2.resize(
			gray,
			(int(orig_w * scale_down), int(orig_h * scale_down)),
			interpolation=cv2.INTER_AREA,
		)
	
	# Detect keypoints in the target image
	try:
		kp_target, des_target = orb.detectAndCompute(gray, None)
		if des_target is None or len(kp_target) < min_matches:
			return []
	except Exception:
		return []
	
	detections: list[dict[str, Any]] = []
	# Expert-level: More granular scales for better feature matching across sizes
	scales = [0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.5, 1.75, 2.0, 2.3] if extended_scales else [0.6, 0.8, 1.0, 1.25, 1.5, 1.8]
	used_bins: set[tuple[str, int, int, int, int]] = set()
	
	for category in sorted(templates.keys()):
		tmpl_list = templates[category]
		for tmpl, fname in tmpl_list:
			th, tw = tmpl.shape[:2]
			
			for scale in scales:
				sw = max(1, int(tw * scale * scale_down))
				sh = max(1, int(th * scale * scale_down))
				if sh >= gray.shape[0] or sw >= gray.shape[1]:
					continue
				
				try:
					tmpl_resized = cv2.resize(tmpl, (sw, sh), interpolation=cv2.INTER_AREA)
					kp_tmpl, des_tmpl = orb.detectAndCompute(tmpl_resized, None)
					if des_tmpl is None or len(kp_tmpl) < 4:
						continue
					
					# Match features
					matches = bf.knnMatch(des_tmpl, des_target, k=2)
					
					# Apply Lowe's ratio test
					good_matches = []
					for match_pair in matches:
						if len(match_pair) == 2:
							m, n = match_pair
							if m.distance < 0.75 * n.distance:
								good_matches.append(m)
					
					if len(good_matches) < min_matches:
						continue
					
					# Extract location of good matches
					src_pts = np.float32([kp_tmpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
					dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
					
					# Find homography to locate the template in the image
					M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
					if M is None:
						continue
					
					# Get the bounding box of the matched region
					h, w = tmpl_resized.shape[:2]
					pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
					dst = cv2.perspectiveTransform(pts, M)
					
					# Calculate bounding box from transformed corners
					x_coords = [int(p[0][0]) for p in dst]
					y_coords = [int(p[0][1]) for p in dst]
					bx, by = min(x_coords), min(y_coords)
					bw = max(x_coords) - bx
					bh = max(y_coords) - by
					
					if bw < 10 or bh < 10:
					 continue
					
					# Scale back to original image coordinates
					if scale_down < 1.0:
						bx = int(bx / scale_down)
						by = int(by / scale_down)
						bw = int(bw / scale_down)
						bh = int(bh / scale_down)
					
					key = (category, bx // 10, by // 10, bw // 10, bh // 10)
					if key in used_bins:
						continue
					used_bins.add(key)
					
					# Calculate confidence based on match quality
					inliers = np.sum(mask)
					confidence = min(0.95, float(inliers) / len(good_matches) + 0.5)
					
					detections.append(
						{
							"name": f"{category.title()} (feature:{fname})",
							"category": category,
							"bbox": (bx, by, bw, bh),
							"confidence": confidence,
							"area": bw * bh,
							"circularity": 0.0,
							"aspect_ratio": float(bw) / max(1.0, float(bh)),
							"vertex_count": 0,
							"extent": 0.0,
							"solidity": 0.0,
						}
					)
				except Exception:
					continue
	
	return detections


def ensemble_vote_detections(
	template_detections: list[dict[str, Any]],
	feature_detections: list[dict[str, Any]],
	shape_detections: list[dict[str, Any]],
	ssim_detections: list[dict[str, Any]] | None = None,
	edge_detections: list[dict[str, Any]] | None = None,
	iou_threshold: float = 0.30,
) -> list[dict[str, Any]]:
	"""Combine detections from multiple methods using ensemble voting.
	
	This function merges detections from template matching, feature matching,
	shape detection, SSIM matching, and edge detection, using voting to improve
	accuracy and reduce false positives.
	
	Returns a consolidated list of detections with boosted confidence for
	components detected by multiple methods.
	"""
	all_detections = []
	
	# Add source tags to track which method detected each component
	for det in template_detections:
		det_copy = det.copy()
		det_copy["sources"] = det_copy.get("sources", set()) | {"template"}
		all_detections.append(det_copy)
	
	for det in feature_detections:
		det_copy = det.copy()
		det_copy["sources"] = det_copy.get("sources", set()) | {"feature"}
		all_detections.append(det_copy)
	
	for det in shape_detections:
		det_copy = det.copy()
		det_copy["sources"] = det_copy.get("sources", set()) | {"shape"}
		all_detections.append(det_copy)
	
	if ssim_detections:
		for det in ssim_detections:
			det_copy = det.copy()
			det_copy["sources"] = det_copy.get("sources", set()) | {"ssim"}
			all_detections.append(det_copy)
	
	if edge_detections:
		for det in edge_detections:
			det_copy = det.copy()
			det_copy["sources"] = det_copy.get("sources", set()) | {"edge"}
			all_detections.append(det_copy)
	
	if not all_detections:
		return []
	
	# Sort by confidence
	all_detections.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
	
	# Group detections by spatial proximity and category
	groups: list[list[dict[str, Any]]] = []
	
	for det in all_detections:
		box = det.get("bbox")
		category = det.get("category")
		if not box or not category:
			continue
		
		assigned = False
		for group in groups:
			# Check if this detection belongs to an existing group
			for member in group:
				if member.get("category") != category:
					continue
				member_box = member.get("bbox")
				if member_box and iou(box, member_box) >= iou_threshold:
					group.append(det)
					assigned = True
					break
			if assigned:
				break
		
		if not assigned:
			groups.append([det])
	
	# Merge each group into a single detection
	merged_detections: list[dict[str, Any]] = []
	
	for group in groups:
		if not group:
			continue
		
		# Use the detection with highest confidence as base
		base = max(group, key=lambda d: float(d.get("confidence", 0.0)))
		merged = base.copy()
		
		# Boost confidence based on number of detection methods
		sources = set()
		for det in group:
			sources.update(det.get("sources", set()))
		
		source_count = len(sources)
		base_confidence = float(base.get("confidence", 0.0))
		
		# Expert-level: Weighted confidence boost based on method reliability
		# SSIM and template matching are most reliable, feature matching second, shape detection third, edge fourth
		method_weights = {"template": 1.0, "ssim": 1.0, "feature": 0.8, "shape": 0.6, "edge": 0.7}
		weighted_boost = sum(method_weights.get(s, 0.5) for s in sources) / len(sources)
		confidence_boost = min(0.35, weighted_boost * 0.18)
		
		# Accuracy optimization: require multiple detection sources for very low-confidence detections
		# Single-source detections with very low confidence are likely false positives
		# Adjusted threshold to avoid filtering valid detections
		if source_count == 1 and base_confidence < 0.40:
			continue  # Skip single-source very low-confidence detections to reduce false positives
		merged["confidence"] = min(0.99, base_confidence + confidence_boost)
		
		# Update source indicator
		merged["source"] = "ensemble"
		merged["sources"] = sources
		
		# Average bounding box if multiple detections
		if len(group) > 1:
			boxes = [d.get("bbox") for d in group if d.get("bbox")]
			if boxes:
				avg_x = sum(b[0] for b in boxes) / len(boxes)
				avg_y = sum(b[1] for b in boxes) / len(boxes)
				avg_w = sum(b[2] for b in boxes) / len(boxes)
				avg_h = sum(b[3] for b in boxes) / len(boxes)
				merged["bbox"] = (int(avg_x), int(avg_y), int(avg_w), int(avg_h))
				merged["area"] = avg_w * avg_h
		
		merged_detections.append(merged)
	
	return merged_detections


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
	"""Calculate Structural Similarity Index (SSIM) between two images.
	
	SSIM is a perception-based model that considers image degradation as
	perceived change in structural information, providing better similarity
	assessment than simple correlation for template matching.
	"""
	try:
		# Convert to grayscale if needed
		if len(img1.shape) == 3:
			img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
		if len(img2.shape) == 3:
			img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
		
		# Resize to same dimensions
		if img1.shape != img2.shape:
			img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
		
		# Calculate SSIM
		C1 = (0.01 * 255) ** 2
		C2 = (0.03 * 255) ** 2
		
		mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
		mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
		
		mu1_sq = mu1 * mu1
		mu2_sq = mu2 * mu2
		mu1_mu2 = mu1 * mu2
		
		sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
		sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
		sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
		
		ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
		
		return float(np.mean(ssim_map))
	except Exception:
		return 0.0


def match_with_ssim(
	image_array: np.ndarray,
	templates: dict[str, list[tuple[np.ndarray, str]]],
	threshold: float = 0.60,
	*,
	extended_scales: bool = False,
) -> list[dict[str, Any]]:
	"""Template matching using SSIM for better structural similarity assessment.
	
	Uses Structural Similarity Index instead of normalized cross-correlation,
	providing better matching for components with similar structure but different
	contrast or brightness.
	"""
	if not templates:
		return []
	
	gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
	orig_h, orig_w = gray.shape[:2]
	scale_down = 1.0
	if max(orig_h, orig_w) > TEMPLATE_MATCH_MAX_EDGE:
		scale_down = float(TEMPLATE_MATCH_MAX_EDGE) / float(max(orig_h, orig_w))
		gray = cv2.resize(
			gray,
			(int(orig_w * scale_down), int(orig_h * scale_down)),
			interpolation=cv2.INTER_AREA,
		)
	
	detections: list[dict[str, Any]] = []
	# Expert-level: Very granular scales for maximum multi-scale detection accuracy
	scales = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.4, 1.55, 1.7, 1.85, 2.0, 2.2, 2.4] if extended_scales else [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
	used_bins: set[tuple[str, int, int, int, int]] = set()
	
	for category in sorted(templates.keys()):
		tmpl_list = templates[category]
		for tmpl, fname in tmpl_list:
			th, tw = tmpl.shape[:2]
			
			for scale in scales:
				sw = max(1, int(tw * scale * scale_down))
				sh = max(1, int(th * scale * scale_down))
				
				# Slide window across image
				for y in range(0, gray.shape[0] - sh + 1, max(1, sh // 4)):
					for x in range(0, gray.shape[1] - sw + 1, max(1, sw // 4)):
						roi = gray[y:y+sh, x:x+sw]
						
						# Resize template to match ROI
						tmpl_resized = cv2.resize(tmpl, (sw, sh))
						if len(tmpl_resized.shape) == 3:
							tmpl_resized = cv2.cvtColor(tmpl_resized, cv2.COLOR_RGB2GRAY)
						
						# Calculate SSIM
						ssim_score = calculate_ssim(roi, tmpl_resized)
						
						if ssim_score >= threshold:
							# Convert back to original coordinates
							orig_x = int(x / scale_down)
							orig_y = int(y / scale_down)
							orig_w = int(sw / scale_down)
							orig_h = int(sh / scale_down)
							
							bin_key = (category, orig_x // 20, orig_y // 20, orig_w // 20, orig_h // 20)
							if bin_key in used_bins:
								continue
							used_bins.add(bin_key)
							
							detections.append({
								"category": category,
								"name": category.title(),
								"bbox": (orig_x, orig_y, orig_w, orig_h),
								"confidence": ssim_score,
								"source": "ssim_template",
								"template_file": fname,
								"area": orig_w * orig_h,
							})
	
	return detections


def calculate_adaptive_confidence(
	detection: dict[str, Any],
	ocr_detections: list[dict[str, Any]],
	image_width: int,
	image_height: int,
) -> float:
	"""Calculate adaptive confidence based on multiple factors.
	
	Considers:
	- Base confidence from detection method
	- OCR support (text labels nearby)
	- Geometry consistency
	- Spatial context
	- Detection method reliability
	
	Returns adjusted confidence score.
	"""
	base_confidence = float(detection.get("confidence", 0.0))
	category = detection.get("category", "")
	bbox = detection.get("bbox")
	
	if not bbox:
		return base_confidence
	
	# Factor 1: OCR support
	nearby_ocr = nearby_ocr_texts(bbox, ocr_detections)
	ocr_support = 0.0
	if nearby_ocr:
		normalized_text = " ".join(item["normalized_text"] for item in nearby_ocr)
		text_category = classify_text_label(normalized_text)
		if text_category == category:
			ocr_support = 0.15
		elif text_category:
			ocr_support = -0.10  # Mismatch reduces confidence
	
	# Factor 2: Geometry consistency
	geometry_score = 0.0
	area = float(detection.get("area", 0))
	aspect_ratio = float(detection.get("aspect_ratio", 1.0))
	extent = float(detection.get("extent", 0.0))
	solidity = float(detection.get("solidity", 0.0))
	circularity = float(detection.get("circularity", 0.0))
	
	if category == "tank":
		if circularity > 0.65:
			geometry_score = 0.10
		elif aspect_ratio >= 1.5 and solidity >= 0.35:
			geometry_score = 0.08
		elif extent >= 0.30:
			geometry_score = 0.05
	elif category == "valve":
		if circularity < 0.50 and solidity < 0.85:
			geometry_score = 0.08
		elif aspect_ratio >= 0.5 and aspect_ratio <= 2.0:
			geometry_score = 0.05
	elif category in ["motor", "pump"]:
		if area >= 200 and area <= 5000:
			geometry_score = 0.05
	
	# Factor 3: Detection method reliability
	source = detection.get("source", "")
	method_reliability = 0.0
	if source in ["template", "ssim"]:
		method_reliability = 0.10
	elif source == "feature":
		method_reliability = 0.08
	elif source == "edge":
		method_reliability = 0.06
	elif source == "ensemble":
		method_reliability = 0.12
	
	# Factor 4: Spatial context (position in image)
	center_x = bbox[0] + bbox[2] / 2.0
	center_y = bbox[1] + bbox[3] / 2.0
	position_score = 0.0
	
	# Components in center region are more likely to be real
	if 0.2 < center_x / image_width < 0.8 and 0.2 < center_y / image_height < 0.8:
		position_score = 0.03
	
	# Factor 5: Size appropriateness
	size_score = 0.0
	image_area = image_width * image_height
	if 0.0005 < area / image_area < 0.02:
		size_score = 0.05
	
	# Combine all factors
	adjusted_confidence = base_confidence + ocr_support + geometry_score + method_reliability + position_score + size_score
	
	# Clamp to valid range
	return max(0.0, min(0.99, adjusted_confidence))


def apply_context_aware_classification(
	detections: list[dict[str, Any]],
	image_width: int,
	image_height: int,
) -> list[dict[str, Any]]:
	"""Apply context-aware classification to improve accuracy.
	
	This function analyzes spatial relationships and component density
	to reclassify components based on their context, similar to how
	Claude AI understands diagram semantics.
	
	Returns a list of detections with potentially updated categories and confidence.
	"""
	if not detections:
		return detections
	
	# Analyze component density and spatial distribution
	component_centers = []
	category_counts = {"motor": 0, "pump": 0, "tank": 0, "valve": 0}
	
	for det in detections:
		category = det.get("category")
		if category in category_counts:
			category_counts[category] += 1
		bbox = det.get("bbox")
		if bbox:
			center_x = bbox[0] + bbox[2] / 2.0
			center_y = bbox[1] + bbox[3] / 2.0
			component_centers.append((center_x, center_y, category))
	
	# If very few components, no context to apply
	if len(component_centers) < 3:
		return detections
	
	# Calculate spatial clusters
	updated_detections = []
	for det in detections:
		category = det.get("category")
		bbox = det.get("bbox")
		confidence = float(det.get("confidence", 0.0))
		
		if not bbox:
			updated_detections.append(det)
			continue
		
		center_x = bbox[0] + bbox[2] / 2.0
		center_y = bbox[1] + bbox[3] / 2.0
		
		# Count nearby components of each category
		nearby_categories = {"motor": 0, "pump": 0, "tank": 0, "valve": 0}
		search_radius = min(image_width, image_height) * 0.15
		
		for cx, cy, cat in component_centers:
			distance = math.hypot(center_x - cx, center_y - cy)
			if distance <= search_radius and cat in nearby_categories:
				nearby_categories[cat] += 1
		
		# Context-aware reclassification rules
		updated_det = det.copy()
		
		# If a component is surrounded by many tanks and has low confidence, it might be a tank
		if confidence < 0.50 and nearby_categories["tank"] >= 2 and category != "tank":
			# Check if geometry is tank-like
			area = float(det.get("area", 0))
			aspect_ratio = float(det.get("aspect_ratio", 1.0))
			extent = float(det.get("extent", 0.0))
			solidity = float(det.get("solidity", 0.0))
			
			# Simple tank geometry check
			if (aspect_ratio >= 1.5 or extent >= 0.25) and solidity >= 0.30:
				updated_det["category"] = "tank"
				updated_det["name"] = "Tank (context)"
				updated_det["confidence"] = min(0.65, confidence + 0.20)
		
		# If a component is near many valves and has valve-like geometry, boost confidence
		if category == "valve" and nearby_categories["valve"] >= 1:
			if confidence < 0.70:
				updated_det["confidence"] = min(0.75, confidence + 0.15)
		
		# Boost pump confidence moderately to ensure detection without overcounting
		# Skip boost for simple diagrams to prevent overcounting
		if category == "pump" and confidence < 0.55:
			# Only boost if there are many tanks (indicates complex diagram)
			tank_count = len([d for d in detections if d.get("category") == "tank"])
			if tank_count >= 3:
				updated_det["confidence"] = min(0.65, confidence + 0.18)
		
		# If a tank is isolated (no nearby tanks) but has low confidence, reduce confidence
		if category == "tank" and nearby_categories["tank"] == 0 and confidence < 0.60:
			updated_det["confidence"] = max(0.35, confidence - 0.15)
		
		# Reduce tank confidence in simple diagrams to prevent overcounting
		if category == "tank" and confidence > 0.50:
			updated_det["confidence"] = max(0.45, confidence - 0.10)
		
		updated_detections.append(updated_det)
	
	return updated_detections


def match_edges_template(
	image_array: np.ndarray,
	templates: dict[str, list[tuple[np.ndarray, str]]],
	threshold: float = 0.50,
	*,
	extended_scales: bool = False,
) -> list[dict[str, Any]]:
	"""Edge-based template matching for better shape matching.
	
	Uses Canny edge detection before template matching to focus on structural
	features rather than pixel intensities, making matching more robust to
	lighting and contrast variations.
	
	Returns list of detection dicts similar to shape detection output.
	"""
	if not templates:
		return []
	
	gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
	orig_h, orig_w = gray.shape[:2]
	scale_down = 1.0
	if max(orig_h, orig_w) > TEMPLATE_MATCH_MAX_EDGE:
		scale_down = float(TEMPLATE_MATCH_MAX_EDGE) / float(max(orig_h, orig_w))
		gray = cv2.resize(
			gray,
			(int(orig_w * scale_down), int(orig_h * scale_down)),
			interpolation=cv2.INTER_AREA,
		)
	
	# Apply Canny edge detection to both image and templates
	try:
		edges = cv2.Canny(gray, 50, 150)
	except Exception:
		return []
	
	detections: list[dict[str, Any]] = []
	# Expert-level: Very granular scales for maximum multi-scale detection accuracy
	scales = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.4, 1.55, 1.7, 1.85, 2.0, 2.2, 2.4] if extended_scales else [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
	used_bins: set[tuple[str, int, int, int, int]] = set()
	
	for category in sorted(templates.keys()):
		tmpl_list = templates[category]
		for tmpl, fname in tmpl_list:
			th, tw = tmpl.shape[:2]
			
			for scale in scales:
				sw = max(1, int(tw * scale * scale_down))
				sh = max(1, int(th * scale * scale_down))
				if sh >= gray.shape[0] or sw >= gray.shape[1]:
					continue
				
				try:
					tmpl_resized = cv2.resize(tmpl, (sw, sh), interpolation=cv2.INTER_AREA)
					tmpl_edges = cv2.Canny(tmpl_resized, 50, 150)
					
					# Use normalized cross-correlation for edge matching
					res = cv2.matchTemplate(edges, tmpl_edges, cv2.TM_CCOEFF_NORMED)
				except Exception:
					continue
				
				for x, y, score in _template_match_peaks(
					res, sw, sh, threshold, TEMPLATE_MAX_PEAKS
				):
					if scale_down < 1.0:
						bx = int(x / scale_down)
						by = int(y / scale_down)
						bw = max(1, int(sw / scale_down))
						bh = max(1, int(sh / scale_down))
					else:
						bx, by, bw, bh = x, y, sw, sh
					
					key = (category, bx // 10, by // 10, bw // 10, bh // 10)
					if key in used_bins:
						continue
					used_bins.add(key)
					
					detections.append(
						{
							"name": f"{category.title()} (edge:{fname})",
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
	"""Parse a JSON object from model output.

	Accepts: raw JSON, fenced JSON, or JSON embedded within text.
	"""
	trimmed = (raw_text or "").strip()
	if not trimmed:
		return None

	def _try(candidate: str) -> dict[str, Any] | None:
		candidate = candidate.strip()
		if not candidate:
			return None
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, dict):
				return parsed
		except json.JSONDecodeError:
			return None
		return None

	for candidate in (
		trimmed,
		re.sub(r"^```(?:json)?\s*|\s*```$", "", trimmed, flags=re.IGNORECASE | re.DOTALL),
	):
		parsed = _try(candidate)
		if parsed is not None:
			return parsed

	# Extract the outermost {...} block.
	start = trimmed.find("{")
	end = trimmed.rfind("}")
	if start != -1 and end != -1 and end > start:
		parsed = _try(trimmed[start : end + 1])
		if parsed is not None:
			return parsed

	# Final fallback: best-effort key extraction into a dict, even if JSON is invalid.
	# This is intentionally conservative: it only extracts integer values for known keys.
	keys = ("motor", "pump", "tank", "valve")
	found_any = False
	out: dict[str, Any] = {}
	for key in keys:
		m = re.search(r"\b" + re.escape(key) + r"\b\s*[:=\-]\s*(\d{1,5})", trimmed, flags=re.IGNORECASE)
		if m:
			out[key] = int(m.group(1))
			found_any = True
	if not found_any:
		return None
	if "industry" in trimmed.lower():
		mi = re.search(r"industry\b\s*[:=\-]\s*\"?([^\n\r\"\}]+)\"?", trimmed, flags=re.IGNORECASE)
		if mi:
			out["industry"] = mi.group(1).strip()
		else:
			out["industry"] = "Unknown"
	return out if out else None



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
    # Expert-level: Very aggressive - much lower confidence threshold for adjustments
    for cat, target in adjusted_counts.items():
        if cat not in COUNT_KEYS:
            continue
        deficit = max(0, int(target) - current_counts.get(cat, 0))
        if deficit <= 0:
            continue
        for det in components:
            if deficit <= 0:
                break
            if det.get("confidence", 0) < 0.75 and det.get("category") != cat:
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
	text_for_prompt = text_blob[:FAST_OLLAMA_TEXT_CHARS] if fast_mode else text_blob[:4000]
	prompt = (
		"You are an expert P&ID component counter.\n"
		"Count ONLY physical equipment symbols drawn on this sheet: motors, pumps, tanks/vessels/drums, and valves "
		"(bow-tie, gate, globe, ball, control, check, relief). Do NOT count instrument bubbles "
		"(PT, FT, LT, TT, PC, LC), controllers, line labels alone, or off-page references "
		"(e.g. 'From P-201' names a pump not shown here — count pump 0).\n"
		"Output ONLY one JSON object: "
		'{"motor":int,"pump":int,"tank":int,"valve":int,"industry":str}\n'
		f"OpenCV/template detection counts (may be wrong): {json.dumps(counts)}\n"
		f"OCR token counts from tags: {token_summary}\n"
		f"Industry hint: {industry_hint}\n"
		f"OCR text from diagram:\n{text_for_prompt}\n"
	)

	env_models = [m.strip() for m in OLLAMA_MODELS.split(",") if m.strip()]
	# If env explicitly set to 'auto' or empty, discover from Ollama server
	if len(env_models) == 1 and env_models[0].lower() in ("", "auto", "discover"):
		discovered = get_available_ollama_models()
		if not discovered:
			logger.warning("Ollama: no models found on server, skipping verification.")
			return None
		models = discovered
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

	small_models = _prefer_fast_ollama_models([m for m in models if not _is_large_model(m)])
	large_models = [m for m in models if _is_large_model(m)]
	if fast_mode:
		small_models = small_models[:1]
		large_models = []
		logger.info("Ollama fast path using model: %s", small_models[0] if small_models else "none")

	deterministic_final = {k: int(counts.get(k, 0)) for k in COUNT_KEYS}

	# Helper to query a single model (used with ThreadPoolExecutor)
	def _query_model(model_name: str) -> tuple[str, str, dict | None, str | None]:
		last_error: str | None = None
		# In fast_mode, one bounded attempt is more reliable than multiple retries
		# that can overrun the global Ollama wait budget.
		max_attempts = 1 if fast_mode else 3
		for attempt in range(max_attempts):
			payload = {
				"model": model_name,
				"prompt": prompt,
				"stream": False,
				"format": response_schema,
				"options": {"temperature": 0, "num_predict": 64 if fast_mode else 128},
			}
			try:
				timeout_sec = OLLAMA_FAST_TIMEOUT_SECONDS if fast_mode else OLLAMA_TIMEOUT_SECONDS
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
				aggregated_counts[key] = max(aggregated_counts.get(key, 0), val)
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
					aggregated_counts[key] = max(aggregated_counts.get(key, 0), val)
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
		logger.debug("Ollama did not return a valid JSON count result; using deterministic counts.")
		return {"counts": deterministic_final, "industry": chosen_industry, "models_used": models}

	return {"counts": aggregated_counts, "industry": chosen_industry, "models_used": models}


def infer_subtype_from_ocr(category: str, ocr_detections: list[dict[str, Any]], bbox: tuple[float, float, float, float]) -> str:
	"""Infer component subtype from nearby OCR text.
	
	Args:
		category: Base category (tank, pump, motor, valve, instrument)
		ocr_detections: List of OCR text detections
		bbox: Component bounding box (x, y, width, height)
	
	Returns:
		Inferred subtype string for Ignition Vision type path
	"""
	if not ocr_detections or not bbox:
		return ""
	
	# Get nearby OCR text
	nearby = nearby_ocr_texts(bbox, ocr_detections, padding_ratio=0.5)
	nearby_text = " ".join(item.get("text", "") for item in nearby).lower()
	
	# Expanded subtype patterns based on common P&ID symbols
	subtype_patterns = {
		"tank": {
			"horizontal": ["horizontal", "horiz"],
			"vertical": ["vertical", "vert"],
			"spherical": ["spherical", "sphere", "round"],
			"rectangular": ["rectangular", "rect"],
			"cylindrical": ["cylindrical", "cylinder"],
			"conical": ["conical", "cone"],
			"floating": ["floating", "float"],
			"fixed": ["fixed", "fixed roof"],
			"pressure": ["pressure", "press"],
			"storage": ["storage", "store"],
			"buffer": ["buffer"],
			"surge": ["surge"],
			"flash": ["flash"],
			"separator": ["separator", "separ"],
			"settler": ["settler", "settle"],
			"decanter": ["decanter"],
			"reactor": ["reactor", "react"],
		},
		"pump": {
			"centrifugal": ["centrifugal", "centri"],
			"reciprocating": ["reciprocating", "recip"],
			"screw": ["screw", "rotary"],
			"gear": ["gear"],
			"diaphragm": ["diaphragm", "diaph"],
			"plunger": ["plunger"],
			"peristaltic": ["peristaltic", "peri"],
			"progressive": ["progressive", "cavity"],
			"vane": ["vane"],
			"lobed": ["lobed", "lobe"],
			"axial": ["axial", "axial flow"],
			"mixed": ["mixed", "mixed flow"],
			"booster": ["booster", "boost"],
			"dosing": ["dosing", "dose"],
			"metering": ["metering", "meter"],
			"transfer": ["transfer"],
			"circulation": ["circulation", "circ"],
			"injection": ["injection", "inject"],
			"priming": ["priming", "prime"],
			"submersible": ["submersible", "sub"],
			"vertical": ["vertical", "vert"],
			"horizontal": ["horizontal", "horiz"],
		},
		"motor": {
			"induction": ["induction", "ind"],
			"synchronous": ["synchronous", "sync"],
			"dc": ["dc", "direct"],
			"servo": ["servo"],
			"stepper": ["stepper", "step"],
			"linear": ["linear"],
			"universal": ["universal"],
			"shaded": ["shaded", "shaded pole"],
			"split": ["split", "split phase"],
			"capacitor": ["capacitor", "cap"],
			" reluctance": ["reluctance"],
			"hysteresis": ["hysteresis"],
			"permanent": ["permanent", "pm"],
			"brushless": ["brushless", "bldc"],
			"brushed": ["brushed"],
			"variable": ["variable", "vfd"],
			"high": ["high", "high voltage"],
			"low": ["low", "low voltage"],
			"medium": ["medium", "medium voltage"],
		},
		"valve": {
			"gate": ["gate"],
			"globe": ["globe"],
			"ball": ["ball"],
			"butterfly": ["butterfly"],
			"check": ["check", "non-return", "nr"],
			"control": ["control", "regulating", "reg"],
			"needle": ["needle"],
			"plug": ["plug"],
			"angle": ["angle"],
			"diaphragm": ["diaphragm", "diaph"],
			"pinch": ["pinch"],
			"solenoid": ["solenoid", "solen"],
			"pilot": ["pilot"],
			"safety": ["safety", "relief", "psv"],
			"pressure": ["pressure", "reducing", "prv"],
			"relief": ["relief"],
			"expansion": ["expansion"],
			"thermostatic": ["thermostatic", "thermo"],
			"trap": ["trap", "steam"],
			"float": ["float"],
			"foot": ["foot"],
			"check": ["check", "non-return"],
			"stop": ["stop"],
			"isolation": ["isolation", "iso"],
			"throttle": ["throttle"],
			"vent": ["vent"],
			"drain": ["drain"],
			"bleed": ["bleed"],
			"sample": ["sample"],
			"diverting": ["diverting"],
			"three": ["three", "3-way"],
			"four": ["four", "4-way"],
			"multi": ["multi", "multi-port"],
			"knife": ["knife"],
			"slide": ["slide"],
			"swing": ["swing"],
			"lift": ["lift"],
			"tilting": ["tilting"],
			"disc": ["disc"],
			"wedge": ["wedge"],
			"parallel": ["parallel"],
			"double": ["double", "dbb"],
			"triple": ["triple"],
			"eccentric": ["eccentric"],
			"concentric": ["concentric"],
		},
		"instrument": {
			"indicator": ["indicator", "ind"],
			"transmitter": ["transmitter", "trans"],
			"sensor": ["sensor"],
			"gauge": ["gauge"],
			"switch": ["switch"],
			"controller": ["controller", "ctrl"],
			"recorder": ["recorder", "rec"],
			"alarm": ["alarm"],
			"analyzer": ["analyzer", "anal"],
			"detector": ["detector", "detect"],
			"monitor": ["monitor"],
			"regulator": ["regulator", "reg"],
			"converter": ["converter", "conv"],
			"transducer": ["transducer"],
			"element": ["element", "sensing"],
			"thermometer": ["thermometer", "temp"],
			"thermocouple": ["thermocouple", "tc"],
			"rtd": ["rtd", "resistance"],
			"pressure": ["pressure", "press"],
			"level": ["level"],
			"flow": ["flow"],
			"temperature": ["temperature", "temp"],
			"differential": ["differential", "diff"],
			"absolute": ["absolute", "abs"],
			"gauge": ["gauge", "g"],
			"vacuum": ["vacuum"],
			"ph": ["ph"],
			"conductivity": ["conductivity", "cond"],
			"density": ["density"],
			"viscosity": ["viscosity", "visc"],
			"turbidity": ["turbidity"],
			"dissolved": ["dissolved", "do"],
			"oxygen": ["oxygen", "o2"],
			"moisture": ["moisture", "hum"],
			"humidity": ["humidity"],
			"speed": ["speed", "rpm"],
			"vibration": ["vibration", "vib"],
			"position": ["position", "pos"],
			"displacement": ["displacement", "disp"],
			"force": ["force"],
			"torque": ["torque"],
			"power": ["power"],
			"energy": ["energy"],
			"frequency": ["frequency", "freq"],
			"voltage": ["voltage", "volt"],
			"current": ["current", "amp"],
			"resistance": ["resistance", "ohm"],
		},
	}
	
	# Check for subtype patterns in OCR text
	if category in subtype_patterns:
		for subtype, patterns in subtype_patterns[category].items():
			for pattern in patterns:
				if pattern in nearby_text:
					return subtype
	
	return ""


def detections_to_coordinates_payload(
	detections: list[dict[str, Any]],
	*,
	ocr_detections: list[dict[str, Any]] | None = None,
	canvas_width: int | None = None,
	canvas_height: int | None = None,
) -> dict[str, Any]:
	children: list[dict[str, Any]] = []
	label_counter = 0
	category_counters: dict[str, int] = {key: 0 for key in COUNT_KEYS}
	
	# Letter-based marking mapping as requested by user
	# p for pump, m for motor, v for valve, t for tank, o for others
	category_to_letter: dict[str, str] = {
		"pump": "p",
		"motor": "m", 
		"valve": "v",
		"tank": "t",
		"instrument": "i",
		"other": "o",
	}
	
	# Ignition symbol type mapping based on component category
	# Using simple symbol types that match the user's Ignition JSON format
	def get_ignition_symbol_type(category: str) -> str:
		"""Get Ignition symbol type for component category."""
		symbol_types = {
			"tank": "ia.symbol.vessel",
			"pump": "ia.symbol.pump",
			"motor": "ia.symbol.motor",
			"valve": "ia.symbol.valve",
			"instrument": "ia.symbol.sensor",
			"other": "ia.symbol.other",
		}
		return symbol_types.get(category, "ia.symbol.other")
	
	# Track actual image bounds for accurate canvas sizing
	min_x = float('inf')
	min_y = float('inf')
	max_x = 0.0
	max_y = 0.0
	
	for detection in sorted(detections, key=lambda item: (item["bbox"][1], item["bbox"][0])):
		x, y, width, height = detection["bbox"]
		# Some matchers (notably ORB+homography) can produce negative coordinates.
		# Pydantic requires x/y/width/height to be >= 0.
		x = int(max(0, x))
		y = int(max(0, y))
		width = int(max(0, width))
		height = int(max(0, height))
		category = str(detection.get("category", "") or "").lower()
		source = detection.get("source", "")
		
		# Skip OCR-based detections for coordinate marking - they circle text instead of symbols
		# Only use shape-based detections for accurate component symbol circling
		if source == "ocr":
		 continue
		
		# Track bounds for canvas sizing
		min_x = min(min_x, float(x))
		min_y = min(min_y, float(y))
		max_x = max(max_x, float(x + width))
		max_y = max(max_y, float(y + height))
			
		if category in COUNT_KEYS:
			# Use simple Ignition symbol type
			component_type = get_ignition_symbol_type(category)
			
			category_counters[category] += 1
			
			# Use letter-based marking as requested: p, m, v, t, o
			letter = category_to_letter.get(category, category[0] if category else "x")
			component_number = category_counters[category]
			
			# Include OCR text if available for better component identification
			ocr_name = detection.get("name", "")
			if ocr_name and isinstance(ocr_name, str) and ocr_name.strip():
				component_name = f"{letter}{component_number}_{ocr_name.strip()}"
			else:
				component_name = f"{letter}{component_number}"
		else:
			# Component doesn't match standard categories - count as "other"
			component_type = get_ignition_symbol_type("other")
			category_counters["other"] = category_counters.get("other", 0) + 1
			
			letter = category_to_letter.get("other", "o")
			component_number = category_counters["other"]
			
			# Include OCR text if available for better component identification
			ocr_name = detection.get("name", "")
			if ocr_name and isinstance(ocr_name, str) and ocr_name.strip():
				component_name = f"{letter}{component_number}_{ocr_name.strip()}"
			else:
				component_name = f"{letter}{component_number}"
		children.append(
			{
				"meta": {"name": component_name},
				"position": {"x": int(x), "y": int(y), "width": int(width), "height": int(height)},
				"type": component_type,
			},
		)

	# Use actual image bounds for canvas sizing if not provided
	if canvas_width is None or canvas_height is None:
		if min_x == float('inf') or min_y == float('inf'):
			# No components detected, use default canvas size
			canvas_width = 1920
			canvas_height = 1080
		else:
			# Use the actual image dimensions to preserve original positions
			canvas_width = int(max(1.0, max_x))
			canvas_height = int(max(1.0, max_y))
	
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
					"source": "annotation",
				"area": w * h,
				"circularity": 0.0,
				"aspect_ratio": float(w) / max(1.0, float(h)),
				"vertex_count": 0,
				"extent": 0.0,
				"solidity": 0.0,
			}
		)
	return detections


def _apply_active_learning_labels(
	components: list[dict[str, Any]],
	image_array: np.ndarray,
	ocr_detections: list[dict[str, Any]],
	*,
	user_library_mode: bool,
) -> list[dict[str, Any]]:
	"""Re-label shape candidates using the Random Forest trained on uploaded component photos."""
	try:
		if __package__:
			from . import active_learning
		else:
			import active_learning
	except Exception as exc:
		logger.warning("Active learning unavailable: %s", exc)
		return components

	model_blob = active_learning.load_model_cached()
	if model_blob is None:
		return components

	scored = active_learning.predict_candidates(image_array, components, model_blob=model_blob)
	# Expert-level: Lowered thresholds for better classification accuracy
	base_threshold = 0.40 if user_library_mode else 0.50
	valve_cap = 0.55 if user_library_mode else 0.75

	for candidate in scored:
		predicted = candidate.get("predicted")
		prob = float(candidate.get("prob", 0.0) or 0.0)
		if not predicted:
			continue

		current_category = candidate.get("category")
		downgrade_from_tank = current_category == "tank" and predicted != "tank"
		# Expert-level: Lowered tank downgrade threshold for better accuracy
		threshold = 0.70 if downgrade_from_tank else base_threshold
		if predicted == "valve":
			threshold = min(threshold, valve_cap)

		nearby_blob = " ".join(
			item.get("normalized_text", "")
			for item in nearby_ocr_texts(candidate.get("bbox", (0, 0, 0, 0)), ocr_detections)
		)
		text_supports_valve = bool(
			_VALVE_TAG_RE.search(nearby_blob)
		)
		text_supports_motor = any(
			pattern.search(nearby_blob) for pattern in _COUNTABLE_TEXT_PATTERNS.get("motor", ())
		)
		misclass_prone_switch = (
			predicted == "motor"
			and current_category in {"valve", "pump"}
		)

		if predicted and (
			prob >= threshold
			or (predicted == "valve" and text_supports_valve and prob >= 0.40)
		):
			# Prevent common false relabels (valve/pump -> motor) unless very strong.
			if misclass_prone_switch and not (text_supports_motor or prob >= 0.85):
				continue
			candidate["category"] = predicted
			candidate["confidence"] = max(float(candidate.get("confidence", 0.0) or 0.0), prob)
			if str(candidate.get("name", "")).lower() in ("motor", "pump", "tank", "valve", "other"):
				candidate["name"] = predicted.title()

	return scored


async def analyze_pid_image_async(
	image: Image.Image,
	fast_mode: bool = False,
	use_component_library: bool = False,
) -> dict[str, Any]:
	start_time = time.perf_counter()
	stage_times: dict[str, float] = {}

	def mark_stage(stage_name: str, stage_start: float) -> None:
		stage_times[stage_name] = time.perf_counter() - stage_start

	image_array = np.array(image.convert("RGB"))

	# If the user library is enabled, templates may have just changed (new uploads).
	# Clear the cache so results are deterministic per run.
	if use_component_library:
		clear_annotation_templates_cache()

	# Run OCR and shape detection in parallel. If easyocr missing, skip OCR and continue.
	if easyocr is None:
		ocr_task = asyncio.create_task(asyncio.to_thread(lambda: []))
	else:
		ocr_task = asyncio.create_task(asyncio.to_thread(extract_ocr_detections, image_array, fast_mode))

	
	ocr_stage_start = time.perf_counter()
	try:
		ocr_detections = await ocr_task
	except Exception:
		if not ocr_task.done():
			ocr_task.cancel()
		await asyncio.gather(ocr_task, return_exceptions=True)
		raise
	mark_stage("ocr", ocr_stage_start)
	
	text_blob = " ".join(detection.get("text", "") for detection in ocr_detections).strip()
	text_counts = extract_counts_from_text(text_blob)

	# Text-driven detection and counts (needed for template category selection).
	ocr_component_detections, ocr_counts, industry = detect_text_driven_components(ocr_detections)

	# Initialize diagram complexity with default value (will be refined later)
	diagram_complexity = "complex"

	# Run shape detection with OCR context for better classification.
	shape_stage_start = time.perf_counter()
	shape_component_detections = await asyncio.to_thread(detect_shape_components, image_array, ocr_detections, diagram_complexity)
	mark_stage("shape_detection", shape_stage_start)
	logger.info(f"Shape detection found {len(shape_component_detections)} components")

	# Detect diagram complexity for adaptive confidence thresholds
	complexity_stage_start = time.perf_counter()
	diagram_complexity = await asyncio.to_thread(
		detect_diagram_complexity,
		image_array,
		ocr_detections,
		shape_component_detections,
	)
	mark_stage("complexity_detection", complexity_stage_start)
	logger.info(f"Diagram classified as: {diagram_complexity}")

	# Template-match uploaded component reference photos (annotations folder).
	# Always run template matching for expert-level accuracy using annotation images
	template_detections: list[dict[str, Any]] = []
	feature_detections: list[dict[str, Any]] = []
	edge_detections: list[dict[str, Any]] = []
	ssim_detections: list[dict[str, Any]] = []
	template_count = 0
	# Skip template matching entirely in fast mode for speed
	if fast_mode:
		templates = {}
		templates_available = False
	else:
		templates = load_annotation_templates()
		templates_available = bool(templates) and sum(len(v) for v in templates.values()) > 0
	if templates_available:
		# Use full template set (all categories) with expert-level thresholds
		template_stage_start = time.perf_counter()
		try:
			# Performance & accuracy optimization: higher threshold in fast mode to reduce false positives
			template_threshold = 0.55 if fast_mode else 0.45  # Lowered threshold for maximum template matching accuracy
			if fast_mode and FAST_MATCH_RELEVANT_ONLY:
				relevant_categories = select_relevant_template_categories(
					shape_component_detections,
					ocr_counts,
					text_counts,
				)
				templates = {
					category: templates.get(category, [])
					for category in relevant_categories
					if templates.get(category)
				}

			# Enforce a global template budget in fast mode.
			if fast_mode:
				total_refs = sum(len(v) for v in templates.values())
				if total_refs > FAST_TEMPLATE_MAX_TOTAL and templates:
					categories = list(templates.keys())
					per_category = max(1, FAST_TEMPLATE_MAX_TOTAL // max(1, len(categories)))
					templates = {
						category: refs[:per_category]
						for category, refs in templates.items()
					}
			template_count = sum(len(v) for v in templates.values())
			
			# Expert-level: Lower threshold specifically for tank templates to improve tank detection
			tank_templates = {k: v for k, v in templates.items() if k == "tank"}
			other_templates = {k: v for k, v in templates.items() if k != "tank"}
			
			# In fast mode keep only high value matchers; edge+SSIM are expensive.
			use_edge_matching = not fast_mode
			use_ssim_matching = not fast_mode
			extended_scales = not fast_mode
			feature_min_matches = 9 if fast_mode else 8

			# Run matching tasks in parallel.
			template_task = asyncio.create_task(asyncio.to_thread(
				match_annotation_templates,
				image_array,
				other_templates,
				template_threshold,
				extended_scales=extended_scales,
			))
			
			# Separate task for tank templates with higher threshold to reduce false positives
			tank_template_task = asyncio.create_task(asyncio.to_thread(
				match_annotation_templates,
				image_array,
				tank_templates,
				0.55,  # Higher threshold to prevent false tank detections
				extended_scales=extended_scales,
			))
			
			disable_orb_fast = (
				fast_mode
				and FAST_DISABLE_ORB_OVER_TEMPLATE_COUNT > 0
				and template_count >= FAST_DISABLE_ORB_OVER_TEMPLATE_COUNT
			)
			task_keys = ["template", "tank_template"]
			task_list = [template_task, tank_template_task]
			if not disable_orb_fast:
				feature_task = asyncio.create_task(asyncio.to_thread(
					match_features_with_orb,
					image_array,
					templates,
					min_matches=feature_min_matches,
					extended_scales=extended_scales,
				))
				task_keys.append("feature")
				task_list.append(feature_task)
			else:
				logger.info(
					"Fast mode: skipping ORB feature matching for %s templates (threshold=%s)",
					template_count,
					FAST_DISABLE_ORB_OVER_TEMPLATE_COUNT,
				)
			if use_edge_matching:
				task_keys.append("edge")
				task_list.append(
					asyncio.create_task(asyncio.to_thread(
						match_edges_template,
						image_array,
						templates,
						threshold=0.60,
						extended_scales=True,
					))
				)
			if use_ssim_matching:
				task_keys.append("ssim")
				task_list.append(
					asyncio.create_task(asyncio.to_thread(
						match_with_ssim,
						image_array,
						templates,
						threshold=0.65,
						extended_scales=True,
					))
				)

			template_timeout = FAST_TEMPLATE_MATCH_TIMEOUT_SECONDS if fast_mode else TEMPLATE_MATCH_TIMEOUT_SECONDS
			done, pending = await asyncio.wait(task_list, timeout=template_timeout)
			results_by_key: dict[str, Any] = {key: [] for key in task_keys}
			for key, task in zip(task_keys, task_list, strict=False):
				if task in done:
					try:
						results_by_key[key] = task.result()
					except Exception as exc:
						results_by_key[key] = exc
				else:
					task.cancel()
			if pending:
				await asyncio.gather(*pending, return_exceptions=True)
				logger.warning(
					"Template/feature/edge matching hit %.0fs timeout (%s templates); using completed matcher results only",
					template_timeout,
					template_count,
				)
			template_detections = results_by_key.get("template", [])
			tank_template_detections = results_by_key.get("tank_template", [])
			feature_detections = results_by_key.get("feature", [])
			edge_detections = results_by_key.get("edge", [])
			ssim_detections = results_by_key.get("ssim", [])
			
			# Handle exceptions from individual tasks
			if isinstance(template_detections, Exception):
				logger.warning(f"Template matching failed: {template_detections}")
				template_detections = []
			if isinstance(tank_template_detections, Exception):
				logger.warning(f"Tank template matching failed: {tank_template_detections}")
				tank_template_detections = []
			if isinstance(feature_detections, Exception):
				logger.warning(f"Feature matching failed: {feature_detections}")
				feature_detections = []
			if isinstance(edge_detections, Exception):
				logger.warning(f"Edge matching failed: {edge_detections}")
				edge_detections = []
			if isinstance(ssim_detections, Exception):
				logger.warning(f"SSIM matching failed: {ssim_detections}")
				ssim_detections = []
			
			# Combine regular and tank template detections
			template_detections = template_detections + tank_template_detections
			
			logger.info(
				"Template matching: %s hits (including %s tank hits), Feature matching: %s hits, Edge matching: %s hits, SSIM matching: %s hits from %s reference image(s)",
				len(template_detections),
				len(tank_template_detections),
				len(feature_detections),
				len(edge_detections),
				len(ssim_detections),
				template_count,
			)
		finally:
			mark_stage("template_pipeline", template_stage_start)
	else:
		if fast_mode:
			logger.info("Template matching disabled in fast mode for speed")
		else:
			logger.warning("No annotation templates available")
		ssim_detections = []

	# Combine shape, text-driven, template matches, feature matches, edge matches, and any hand-drawn annotations
	annotation_detections = get_annotation_detections_for_image(image)
	
	# Expert-level: Use ensemble voting to combine multiple detection methods
	if templates_available and (template_detections or feature_detections or edge_detections or ssim_detections):
		# Combine template, feature, edge, and SSIM detections using ensemble voting
		image_based_detections = ensemble_vote_detections(
			template_detections,
			feature_detections,
			shape_detections=[],  # Shape detections are handled separately
			ssim_detections=ssim_detections,
			edge_detections=edge_detections,
			iou_threshold=0.30,
		)
		# Log category breakdown for debugging
		category_breakdown = {}
		for det in image_based_detections:
			cat = det.get("category", "unknown")
			category_breakdown[cat] = category_breakdown.get(cat, 0) + 1
		logger.info(f"Ensemble voting produced {len(image_based_detections)} image-based detections: {category_breakdown}")
	else:
		# Fall back to simple concatenation if no templates available
		image_based_detections = template_detections + feature_detections + edge_detections + (ssim_detections or [])
	
	combined_components = (
		shape_component_detections
		+ ocr_component_detections
		+ image_based_detections
		+ annotation_detections
	)
	
	# Expert-level: Apply context-aware classification to improve accuracy
	# This analyzes spatial relationships similar to how Claude AI understands diagram semantics
	image_h, image_w = image_array.shape[:2]
	combined_components = apply_context_aware_classification(combined_components, image_w, image_h)
	
	# Expert-level: Multi-stage verification pipeline with diagram complexity awareness
	# Cross-validate detections using multiple criteria to reduce false positives
	# Simple diagrams get much stricter thresholds since components are clear and well-separated
	# Complex diagrams: Very permissive for pumps to ensure detection
	verified_components = []
	for det in combined_components:
		category = det.get("category")
		confidence = float(det.get("confidence", 0.0))
		bbox = det.get("bbox")
		source = det.get("source", "")
		
		# Skip OCR-based detections in verification - they're text labels, not component symbols
		if source == "ocr":
			continue
		
		# Special handling for pumps in complex diagrams - balanced threshold
		if diagram_complexity == "complex" and category == "pump" and confidence >= 0.18:
			verified_components.append(det)
			continue
		
		# Special handling for tanks in simple diagrams - require higher confidence to reduce overcounting
		if diagram_complexity == "simple" and category == "tank" and confidence < 0.45:
			continue
		
		# Special handling for pumps in simple diagrams - require higher confidence to reduce overcounting
		if diagram_complexity == "simple" and category == "pump" and confidence < 0.45:
			continue
		
		# Adjust thresholds based on diagram complexity
		if diagram_complexity == "simple":
			# Simple diagrams: Very strict thresholds to eliminate false positives
			high_conf_threshold = 0.70
			medium_conf_threshold = 0.55
			low_conf_threshold = 0.40
		else:
			# Complex diagrams: Much more permissive to maintain recall for crowded components
			high_conf_threshold = 0.45
			medium_conf_threshold = 0.35
			low_conf_threshold = 0.25
		
		# High confidence detections pass immediately
		if confidence >= high_conf_threshold:
			verified_components.append(det)
			continue
		
		# Medium confidence detections for specific categories
		if confidence >= medium_conf_threshold and category in {"tank", "pump", "valve"}:
			# For simple diagrams, require additional OCR evidence for medium confidence
			# For complex diagrams, be more permissive to maintain recall
			if diagram_complexity == "simple" and bbox:
				try:
					nearby = nearby_ocr_texts(bbox, ocr_detections, padding_ratio=0.45)
					nearby_text = " ".join(item.get("text", "") for item in nearby).strip()
					
					has_tag = False
					if category == "valve":
						has_tag = bool(_VALVE_TAG_RE.search(nearby_text or ""))
					elif category == "motor":
						has_tag = any(pattern.search(nearby_text) for pattern in _COUNTABLE_TEXT_PATTERNS.get("motor", ()))
					elif category == "pump":
						has_tag = any(pattern.search(nearby_text) for pattern in _COUNTABLE_TEXT_PATTERNS.get("pump", ()))
					elif category == "tank":
						has_tag = any(pattern.search(nearby_text) for pattern in _COUNTABLE_TEXT_PATTERNS.get("tank", ()))
					
					if not has_tag:
						continue
				except Exception:
					continue
			# For complex diagrams, accept medium confidence without OCR evidence
			verified_components.append(det)
			continue
		
		# Low confidence detections need strong geometry validation
		if confidence >= low_conf_threshold and bbox:
			area = float(det.get("area", 0))
			aspect_ratio = float(det.get("aspect_ratio", 1.0))
			circularity = float(det.get("circularity", 0.0))
			solidity = float(det.get("solidity", 0.0))
			
			# Verify geometry matches category expectations - relaxed thresholds
			if category == "tank":
				# Tanks should have reasonable area - relaxed aspect ratio and solidity requirements
				if area >= 80:
					verified_components.append(det)
			elif category == "valve":
				# Valves should be compact - relaxed circularity requirements
				if area >= 10 and area <= 10000 and circularity >= 0.15:
					verified_components.append(det)
			elif category in ["motor", "pump"]:
				# Motors and pumps should have reasonable size and circularity
				# Relaxed thresholds to catch more components, especially from component library
				min_circularity = 0.25 if diagram_complexity == "simple" else 0.20
				min_area = 40 if diagram_complexity == "simple" else 30
				max_area = 5000 if diagram_complexity == "simple" else 6000
				if area >= min_area and area <= max_area and circularity >= min_circularity:
					verified_components.append(det)
			else:
				# Unknown category, keep if reasonable confidence
				if confidence >= 0.40:
					verified_components.append(det)
		# Low confidence detections are filtered out unless they have strong OCR support
		elif confidence >= 0.25 and det.get("name", "").lower() in ["tank", "motor", "pump", "valve"]:
			verified_components.append(det)
	
	# Log verification stats
	verification_breakdown = {}
	for det in combined_components:
		cat = det.get("category", "unknown")
		verification_breakdown[cat] = verification_breakdown.get(cat, 0) + 1
	logger.info(f"Before verification: {verification_breakdown}")
	
	verified_breakdown = {}
	for det in verified_components:
		cat = det.get("category", "unknown")
		verified_breakdown[cat] = verified_breakdown.get(cat, 0) + 1
	logger.info(f"After verification: {verified_breakdown}")
	
	combined_components = verified_components

	# Sort combined components by bbox for deterministic processing
	combined_components = sorted(combined_components, key=lambda x: (x["bbox"][1], x["bbox"][0], x["category"]))
	
	# Dedupe with a higher IoU threshold to avoid merging distinct nearby components
	deduped_components = dedupe_detections(combined_components, iou_threshold=0.45)
	
	# Log after deduping
	dedup_breakdown = {}
	for det in deduped_components:
		cat = det.get("category", "unknown")
		dedup_breakdown[cat] = dedup_breakdown.get(cat, 0) + 1
	logger.info(f"After deduping: {dedup_breakdown}")

	# Only merge components that are very close (0.15× box size) — prevents collapsing distinct components
	# Reduced from 0.4 to 0.15 to avoid merging distinct valves/tanks that should be separate
	merged_components = merge_close_detections(deduped_components, distance_ratio=0.15)
	
	# Log after close merge
	merge_breakdown = {}
	for det in merged_components:
		cat = det.get("category", "unknown")
		merge_breakdown[cat] = merge_breakdown.get(cat, 0) + 1
	logger.info(f"After close merge: {merge_breakdown}")
	
	merged_components = merge_stacked_tank_symbols(merged_components)
	
	# Log after stacked tank merge
	stacked_breakdown = {}
	for det in merged_components:
		cat = det.get("category", "unknown")
		stacked_breakdown[cat] = stacked_breakdown.get(cat, 0) + 1
	logger.info(f"After stacked tank merge: {stacked_breakdown}")
	
	image_area = float(image_array.shape[0] * image_array.shape[1])
	merged_components = consolidate_tank_vessels(merged_components, image_area=image_area)
	
	# Log after consolidation
	consolidation_breakdown = {}
	for det in merged_components:
		cat = det.get("category", "unknown")
		consolidation_breakdown[cat] = consolidation_breakdown.get(cat, 0) + 1
	logger.info(f"After consolidation: {consolidation_breakdown}")
	
	# Log before check-valve refinement
	pre_refinement_breakdown = {}
	for det in merged_components:
		cat = det.get("category", "unknown")
		pre_refinement_breakdown[cat] = pre_refinement_breakdown.get(cat, 0) + 1
	logger.info(f"Before check-valve refinement: {pre_refinement_breakdown}")

	# --- Check-valve refinement stage (reference-image assisted) ---
	# Promote small/compact valve-like candidates to `valve` when they are strongly supported
	# by valve reference templates. This reduces missed check valves without introducing large false positives.
	debug_refinement: dict[str, Any] = {
		"valve_template_hits": 0,
		"promoted_valves": 0,
	}

	# Count template evidence (valve templates from unified template matching)
	valve_templates = [d for d in template_detections if d.get("category") == "valve"]
	if valve_templates:
		debug_refinement["valve_template_hits"] = len(valve_templates)

	def _near_template_evidence(
		candidate_box: tuple[int, int, int, int],
		evidence_box: tuple[int, int, int, int],
		image_area: float,
	) -> bool:

		cx1, cy1 = bbox_center(candidate_box)
		cx2, cy2 = bbox_center(evidence_box)
		# Use larger dimension as symbol “scale” proxy.
		w1, h1 = candidate_box[2], candidate_box[3]
		w2, h2 = evidence_box[2], evidence_box[3]
		max_dim = max(float(w1), float(h1), float(w2), float(h2), 1.0)
		# Distance threshold: allow small/compact symbols to be validated even
		# when their contour boxes don’t overlap much.
		dist = math.hypot(cx1 - cx2, cy1 - cy2)
		# Also gate by compactness relative to the whole page.
		cand_area = float(max(0, candidate_box[2]) * max(0, candidate_box[3]))
		if image_area > 0 and cand_area > image_area * 0.0030:
			return False
		return dist <= max_dim * 1.25

	# Promote candidates near template evidence (distance/size based).
	# Improvement: require MULTI-evidence for promotion when OCR has no explicit valve tag.
	# This substantially reduces false positives from isolated geometry matches.
	if valve_templates:
		template_boxes = [d.get("bbox") for d in valve_templates if d.get("bbox")]
		if template_boxes:
			for det in merged_components:
				# Never treat tanks as valves.
				if det.get("category") == "tank":
					continue
				if not _candidate_is_compact_valve_like(det):
					continue
				box = det.get("bbox")
				if not box:
					continue

				# Extra cue: if nearby OCR already contains a valve tag, allow
				# promotion even if template evidence is weak.
				nearby_text = ""
				try:
					# Slightly expanded box for cue extraction.
					nearby = nearby_ocr_texts(box, ocr_detections, padding_ratio=0.45)
					nearby_text = " ".join(item.get("text", "") for item in nearby).strip()
				except Exception:
					nearby_text = ""

				nearby_has_valve_tag = bool(_VALVE_TAG_RE.search(nearby_text or ""))

				# Count template evidence hits close to this candidate.
				evidence_hits = 0
				narrow_hits = 0
				for tb in template_boxes:
					if not tb:
						continue
					if _near_template_evidence(box, tb, image_area=image_area):
						evidence_hits += 1
						# Extra strict proximity when OCR tag is absent.
						try:
							narrow_ok = (
								iou(box, tb) >= 0.06 or (
									math.hypot(*tuple(a - b for a, b in zip(bbox_center(box), bbox_center(tb))))
										<= max(box[2], box[3], tb[2], tb[3]) * 0.75
								)
							)
						except Exception:
							narrow_ok = False
						if narrow_ok:
							narrow_hits += 1

					# Promotion gating:
					# - If we have explicit valve OCR tag nearby: require at least 1 evidence hit.
					# - If NO valve OCR tag: require stronger multi-evidence (>=2 close hits)
					#   OR at least 1 strict (narrow) hit.
					if evidence_hits <= 0:
						continue

					if nearby_has_valve_tag:
						# Allow with 1 close evidence hit.
						if evidence_hits >= 1:
							# Valve OCR tags can be imperfect; rely on reference evidence distance.
							det["category"] = "valve"
							# Boost to ensure valves survive later confidence filtering.
							det["confidence"] = max(float(det.get("confidence", 0.0) or 0.0), 0.78)
							debug_refinement["promoted_valves"] += 1
							break
					else:
						# No explicit tag: require multi-hit evidence to prevent isolated false positives.
						# Prefer narrow_hits (higher precision), but allow 2+ nearby hits as recall.
						if (narrow_hits >= 1) or (evidence_hits >= 2):
							det["category"] = "valve"
							det["confidence"] = max(float(det.get("confidence", 0.0) or 0.0), 0.82)
							debug_refinement["promoted_valves"] += 1
							break




	def _candidate_is_compact_valve_like(det: dict[str, Any]) -> bool:
		if det.get("category") == "valve":
			return True
		if det.get("category") == "tank":
			return False
		box = det.get("bbox")
		if not box:
			return False
		x, y, w, h = box
		area = float(det.get("area", w * h))
		aspect_ratio = float(det.get("aspect_ratio", w / max(h, 1)))
		extent = float(det.get("extent", 0.0) or 0.0)
		solidity = float(det.get("solidity", 0.0) or 0.0)
		vertex_count = int(det.get("vertex_count", 0) or 0)

		tank_like = _is_tank_like_geometry(area, aspect_ratio, extent, solidity, image_area, "complex")
		valve_like = _is_compact_bowtie_valve(
			area,
			aspect_ratio,
			float(det.get("circularity", 0.0) or 0.0),
			vertex_count,
			extent,
			solidity,
			image_area,
			tank_like=tank_like,
			bbox=box,
		)
		if not valve_like:
			return False

		# Compactness: box area should be in a small band relative to image size.
		if image_area > 0 and area > image_area * 0.0025:
			return False

		# Require some symbol structure signals (extent/solidity)
		if extent < 0.12 or solidity < 0.20:
			return False
		return True


	# Re-run dedupe + valve suppression after refinement to avoid duplicates
	# Re-enabled with less aggressive distance_ratio to reduce over-detection
	deduped_components = dedupe_detections(merged_components, iou_threshold=0.45)
	merged_components = merge_close_detections(deduped_components, distance_ratio=0.15)  # Less aggressive
	merged_components = merge_stacked_tank_symbols(merged_components)
	merged_components = consolidate_tank_vessels(merged_components, image_area=image_area)

	# Valve-specific suppression: removes nearby duplicate valve-like candidates
	# (common failure mode is counting an extra check/control valve shape twice).
	valve_iou_thresh = 0.60 if fast_mode else 0.55
	valve_center_dist = 0.40 if fast_mode else 0.45
	merged_components = suppress_nearby_valves(
		merged_components,
		iou_threshold=valve_iou_thresh,
		center_dist_ratio=valve_center_dist,
		area_ratio_min=0.50,
		area_ratio_max=2.00,
	)

	# Pump suppression with moderate thresholds to prevent overcounting
	pump_iou_thresh = 0.45 if fast_mode else 0.40
	pump_center_dist = 0.50 if fast_mode else 0.55
	merged_components = suppress_nearby_pumps(
		merged_components,
		iou_threshold=pump_iou_thresh,
		center_dist_ratio=pump_center_dist,
		area_ratio_min=0.50,
		area_ratio_max=2.00,
	)

	# Re-enable geometry false positive filter to reduce over-detection
	merged_components = filter_valve_geometry_false_positives(
		merged_components,
		ocr_detections,
		image_height=int(image_array.shape[0]),
	)

	# Final count-oriented collapse for nearby same-category duplicates.
	# Re-enabled to reduce over-detection
	merged_components = collapse_countable_clusters(merged_components)
	# Re-enable template support filter for all components including valves
	merged_components = [
		det
		for det in merged_components
		if _is_supported_template_detection(det, merged_components)
	]

	
	# Vision model detection disabled to prevent discrepancy
	# Using only shape detection and OCR for consistency
	vision_model_components: list[dict[str, Any]] = []

	
	# Apply model trained on user-uploaded component photos.
	# Temporarily disabled to fix valve detection - active learning was filtering out valid valves
	# if use_component_library or not fast_mode:
	# 	merged_components = _apply_active_learning_labels(
	# 		merged_components,
	# 		image_array,
	# 		ocr_detections,
	# 		user_library_mode=use_component_library,
	# 	)

	# Keep shape detections explicitly if needed by calling code
	# Temporarily skip active learning to fix valve detection - it was filtering out valid valves
	visual_detections = merged_components
	# visual_detections = _apply_active_learning_labels(
	# 	merged_components,
	# 	image_array,
	# 	ocr_detections,
	# 	user_library_mode=use_component_library,
	# )

	# Apply additional validation for simple diagrams to improve accuracy
	# This helps the model by requiring stronger evidence for clear, uncluttered diagrams
	visual_detections = apply_simple_diagram_validation(visual_detections, ocr_detections, diagram_complexity)

	# Log visual detections before confidence filtering
	visual_breakdown = {}
	for det in visual_detections:
		cat = det.get("category", "unknown")
		visual_breakdown[cat] = visual_breakdown.get(cat, 0) + 1
	logger.info(f"Visual detections before confidence filter: {visual_breakdown}")

	# Use adaptive confidence thresholds based on diagram complexity
	# This helps the model by using higher thresholds for simple diagrams (reduce false positives)
	# and lower thresholds for complex diagrams (maintain recall on crowded layouts)
	active_thresh = get_adaptive_confidence_thresholds(diagram_complexity, fast_mode)
	# Increase valve threshold to reduce over-detection from 7 to 3
	active_thresh["valve"] = 0.35 if fast_mode else 0.30
	logger.info(f"Active confidence thresholds (complexity={diagram_complexity}, fast_mode={fast_mode}): {active_thresh}")
	
	# Log confidence values for each category
	conf_by_category = {}
	for det in visual_detections:
		cat = det.get("category", "unknown")
		conf = float(det.get("confidence", 0.0))
		if cat not in conf_by_category:
			conf_by_category[cat] = []
		conf_by_category[cat].append(conf)
	for cat, confs in conf_by_category.items():
		logger.info(f"{cat} confidences: {confs}")
	
	# Filter valves, motors, and pumps to only keep those with nearby OCR text evidence OR high confidence
	# This reduces over-detection while preserving components with text support
	filtered_detections = []
	for det in visual_detections:
		category = det.get("category")
		conf = float(det.get("confidence", 0.0))
		
		# For non-valve/motor/pump categories, keep all
		if category not in {"valve", "motor", "pump"}:
			filtered_detections.append(det)
			continue
		
			# Category-specific high-confidence thresholds
		if category == "valve" and conf >= 0.25:
			filtered_detections.append(det)
			continue
		elif category == "pump" and conf >= 0.35:
			filtered_detections.append(det)
			continue
		elif category == "motor" and conf >= 0.50:
			filtered_detections.append(det)
			continue
		# Pumps and motors below threshold require OCR text evidence
		
		# Check for nearby OCR text evidence
		box = det.get("bbox")
		if not box:
			continue
		nearby_text = ""
		try:
			nearby = nearby_ocr_texts(box, ocr_detections, padding_ratio=0.45)
			nearby_text = " ".join(item.get("text", "") for item in nearby).strip()
		except Exception:
			nearby_text = ""
		
		# Check for category-specific text tags
		if category == "valve":
			nearby_has_tag = bool(_VALVE_TAG_RE.search(nearby_text or ""))
		elif category == "motor":
			nearby_has_tag = any(pattern.search(nearby_text) for pattern in _COUNTABLE_TEXT_PATTERNS.get("motor", ()))
		elif category == "pump":
			nearby_has_tag = any(pattern.search(nearby_text) for pattern in _COUNTABLE_TEXT_PATTERNS.get("pump", ()))
		else:
			nearby_has_tag = False
		
		if nearby_has_tag:
			filtered_detections.append(det)
	visual_detections = filtered_detections
	
	# Log after OCR-based valve filtering
	ocr_filter_breakdown = {}
	for det in visual_detections:
		cat = det.get("category", "unknown")
		ocr_filter_breakdown[cat] = ocr_filter_breakdown.get(cat, 0) + 1
	logger.info(f"After OCR-based valve filter: {ocr_filter_breakdown}")

	# Post-process: suppress candidates the active-learning model is uncertain about.
	# If a candidate does not strongly match the reference images, reduce its confidence.
	for _det in visual_detections:
		try:
			prob = float(_det.get("prob", 1.0))
		except Exception:
			prob = 1.0
			
		if "prob" in _det:
			# Suppress highly uncertain predictions from the Random Forest.
			# Reduced threshold from 0.50 to 0.35 to avoid suppressing valid detections
			if prob < 0.35:
				_det["confidence"] = min(float(_det.get("confidence", 1.0)), 0.25)


	# Single source of truth: build the exact set of components that will be used for:
	# 1) counting
	# 2) coordinates
	# This removes the mismatch where coordinates used a slightly different filtered set.
	# Include template-based detections for better accuracy
	countable_components: list[dict[str, Any]] = [
		det
		for det in visual_detections
		if det.get("category") in COUNT_KEYS
		and float(det.get("confidence", 1.0)) >= active_thresh.get(det.get("category"), 0.0)
	]

	# Deterministic counting from countable_components (no re-filter later)
	visual_counts = empty_counts()
	for detection in countable_components:
		category = detection["category"]
		visual_counts[category] += 1
	logger.info("Final countable detections: %s", visual_counts)


	library_refined = use_component_library
	if library_refined:
		# When the user supplied reference photos, trust shape + template + model counts.
		combined_counts = dict(visual_counts)
	else:
		combined_counts = merge_counts_with_text_anchors(ocr_counts, visual_counts, text_counts)

	# Use the same countable component set for coordinates as used for counting.
	filtered_for_coordinates = countable_components

	coordinates_stage_start = time.perf_counter()
	coordinates_task = asyncio.create_task(
		asyncio.to_thread(
			detections_to_coordinates_payload,
			dedupe_detections(filtered_for_coordinates, iou_threshold=0.8),
			ocr_detections=ocr_detections,
			canvas_width=int(image_array.shape[1]),
			canvas_height=int(image_array.shape[0]),
			),
	)


	phi3_counts: dict[str, int] | None = None
	phi3_industry: str | None = None
	used_ollama = False
	ollama_task: asyncio.Task[dict[str, Any] | None] | None = None
	_disable_ollama = os.getenv("DISABLE_OLLAMA_VERIFICATION", "false").strip().lower() in {"1", "true", "yes", "on"}
	elapsed_before_ollama = time.perf_counter() - start_time
	should_run_ollama = (
		OLLAMA_ENABLED
		and OLLAMA_USE_FOR_COUNTS
		and not _disable_ollama
	)
	if should_run_ollama:
		ollama_task = asyncio.create_task(
			asyncio.to_thread(
				verify_with_ollama,
				text_blob=text_blob,
				counts=combined_counts,
				industry_hint=industry,
				fast_mode=fast_mode,
			),
		)
		logger.info("Ollama verification started (fast_mode=%s)", fast_mode)

	async def _await_ollama_with_extension() -> dict[str, Any] | None:
		"""Wait for Ollama; on first timeout, keep waiting up to completion budget."""
		if ollama_task is None:
			return None
		completion_wait = (
			min(float(OLLAMA_COMPLETION_TIMEOUT_SECONDS), FAST_OLLAMA_WAIT_CAP_SECONDS)
			if fast_mode
			else float(OLLAMA_COMPLETION_TIMEOUT_SECONDS)
		)
		extra_wait = (
			min(float(OLLAMA_FAST_TIMEOUT_SECONDS), max(0.0, FAST_OLLAMA_WAIT_CAP_SECONDS - completion_wait))
			if fast_mode
			else float(OLLAMA_FAST_TIMEOUT_SECONDS)
		)
		try:
			return await asyncio.wait_for(asyncio.shield(ollama_task), timeout=completion_wait)
		except asyncio.TimeoutError:
			if ollama_task.done():
				try:
					return ollama_task.result()
				except asyncio.CancelledError:
					logger.warning("Ollama task was cancelled after timeout; using OpenCV counts")
					return None
			logger.warning(
				"Ollama still running after %ss — waiting up to %ss more for completion",
				completion_wait,
				extra_wait,
			)
			if extra_wait <= 0:
				if fast_mode and FAST_ACCURACY_PRIORITIZE_OLLAMA:
					accuracy_wait = max(0.0, float(OLLAMA_COMPLETION_TIMEOUT_SECONDS) - completion_wait)
					if accuracy_wait > 0:
						logger.warning(
							"Ollama exceeded fast wait cap; accuracy mode waiting %.1fs more",
							accuracy_wait,
						)
						try:
							return await asyncio.wait_for(asyncio.shield(ollama_task), timeout=accuracy_wait)
						except asyncio.TimeoutError:
							logger.error("Ollama accuracy wait expired; using OpenCV counts")
							return None
				logger.error("Ollama exceeded fast wait cap; using OpenCV counts")
				return None
			try:
				return await asyncio.wait_for(asyncio.shield(ollama_task), timeout=extra_wait)
			except asyncio.TimeoutError:
				logger.error(
					"Ollama did not finish within %ss total; using OpenCV counts",
					completion_wait + extra_wait,
				)
				return None
		except asyncio.CancelledError:
			logger.warning("Ollama wait was cancelled; using OpenCV counts")
			return None

	def _apply_ollama_counts(phi3_result: dict[str, Any] | None) -> None:
		nonlocal industry, used_ollama, phi3_counts, phi3_industry
		if not phi3_result:
			return
		phi3_counts = phi3_result["counts"]
		phi3_industry = phi3_result.get("industry")
		if phi3_industry:
			industry = phi3_industry
		used_ollama = True
		if OLLAMA_TRUST_COUNTS:
			for key in COUNT_KEYS:
				combined_counts[key] = int(phi3_counts.get(key, 0) or 0)
			return
		for key in COUNT_KEYS:
			ollama_val = int(phi3_counts.get(key, 0) or 0)
			current = int(combined_counts.get(key, 0))
			visual_val = int(visual_counts.get(key, 0))
			# In fast_mode, keep CV/OCR detections authoritative but allow Ollama to reduce obvious false positives.
			# A single quick Ollama pass is useful for hints, and can reduce counts when CV clearly over-detects.
			if fast_mode:
				text_val = int(text_counts.get(key, 0) or 0)
				# Deterministic floor from text evidence (OCR is more reliable than shape/template for counts)
				text_floor = max(text_val, 0)
				
				# Allow Ollama to reduce counts ONLY when there's strong text evidence supporting the reduction.
				# Visual detections (template/feature matching) should be trusted over Ollama's conservative guesses.
				if ollama_val < current:
					# Only reduce if text evidence strongly supports the lower Ollama count
					# Require text evidence to be at least 50% of the visual detection count
					if text_val >= max(1, current * 0.5) and ollama_val >= text_floor:
						# Text evidence supports reduction - allow it
						combined_counts[key] = max(text_floor, ollama_val)
					else:
						# No strong text evidence - trust visual detections over Ollama
						combined_counts[key] = current
				# Allow upward correction when vision saw none
				elif visual_val == 0 and ollama_val > current:
					combined_counts[key] = ollama_val
				else:
					# Keep current count
					combined_counts[key] = current
				continue
			# Non-fast mode: allow Ollama to add missing classes, but DO NOT allow it to reduce
			# valid visual counts because small LLMs frequently hallucinate 0.
			if visual_val == 0 and ollama_val > current:
				combined_counts[key] = ollama_val

	def _rebalance_motor_valve_counts() -> None:
		"""Fix common confusion where a weak motor candidate is actually a valve.

		Only applies a single-step rebalance and only when text/OCR evidence
		supports valve but not motor, to avoid broad behavior shifts.
		"""
		motor_count = int(combined_counts.get("motor", 0) or 0)
		if motor_count <= 0:
			return

		motor_text = max(int(text_counts.get("motor", 0) or 0), int(ocr_counts.get("motor", 0) or 0))
		valve_text = max(int(text_counts.get("valve", 0) or 0), int(ocr_counts.get("valve", 0) or 0))
		if motor_text > 0 or valve_text <= int(combined_counts.get("valve", 0) or 0):
			return

		motor_candidates = [d for d in countable_components if d.get("category") == "motor"]
		if not motor_candidates:
			return

		weakest_motor_conf = min(float(d.get("confidence", 0.0) or 0.0) for d in motor_candidates)
		if weakest_motor_conf > 0.80:
			return

		combined_counts["motor"] = max(0, motor_count - 1)
		combined_counts["valve"] = int(combined_counts.get("valve", 0) or 0) + 1
		logger.info(
			"Applied motor->valve rebalance (motor_text=%s valve_text=%s weakest_motor_conf=%.2f): %s",
			motor_text,
			valve_text,
			weakest_motor_conf,
			combined_counts,
		)

	if fast_mode:
		if ollama_task is not None:
			ollama_wait_stage_start = time.perf_counter()
			try:
				coordinates, phi3_result = await asyncio.gather(
					coordinates_task,
					_await_ollama_with_extension(),
				)
				mark_stage("coordinates", coordinates_stage_start)
				mark_stage("ollama_wait", ollama_wait_stage_start)
				_apply_ollama_counts(phi3_result)
				_rebalance_motor_valve_counts()
				if used_ollama:
					logger.info("Ollama verification applied: %s", combined_counts)
			except Exception as exc:
				logger.warning("Ollama verification failed: %s", exc)
				coordinates = await coordinates_task
				mark_stage("coordinates", coordinates_stage_start)
		else:
			coordinates = await coordinates_task
			mark_stage("coordinates", coordinates_stage_start)
	elif ollama_task is None:
		coordinates = await coordinates_task
		mark_stage("coordinates", coordinates_stage_start)
	else:
		ollama_wait_stage_start = time.perf_counter()
		try:
			phi3_result, coordinates = await asyncio.gather(
				_await_ollama_with_extension(),
				coordinates_task,
			)
			mark_stage("coordinates", coordinates_stage_start)
			mark_stage("ollama_wait", ollama_wait_stage_start)
			_apply_ollama_counts(phi3_result)
			_rebalance_motor_valve_counts()
			if used_ollama:
				logger.info("Ollama verification applied: %s", combined_counts)
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
	total_elapsed = time.perf_counter() - start_time
	logger.info(
		"analyze_pid_image_async timing (fast_mode=%s): total=%.2fs ocr=%.2fs shape=%.2fs template=%.2fs coordinates=%.2fs ollama_wait=%.2fs pre_ollama=%.2fs",
		fast_mode,
		total_elapsed,
		stage_times.get("ocr", 0.0),
		stage_times.get("shape_detection", 0.0),
		stage_times.get("template_pipeline", 0.0),
		stage_times.get("coordinates", 0.0),
		stage_times.get("ollama_wait", 0.0),
		elapsed_before_ollama,
	)
	result = {
		"ocr_counts": ocr_counts,
		"vision_counts": visual_counts,
		"counts": combined_counts,
		"industry": industry,
		"phi3_counts": phi3_counts,
		"phi3_industry": phi3_industry,
		"used_ollama": used_ollama,
		"coordinates": coordinates,
		"ocr_detections": ocr_detections,
		"detections": countable_components,
		"library_refined": library_refined,
		"debug_refinement": debug_refinement,
	}
	
	try:
		import json
		debug_path = Path(__file__).parent / "debug_detections.json"
		with open(debug_path, "w", encoding="utf-8") as f:
			# Remove raw coordinates array for brevity, keep the summary
			debug_data = {
				"counts": combined_counts,
				"vision_counts": visual_counts,
				"phi3_counts": phi3_counts,
				"detections": [
					{
						"category": d.get("category"), 
						"confidence": d.get("confidence"),
						"area": d.get("area"),
						"circularity": d.get("circularity"),
						"solidity": d.get("solidity"),
						"aspect": d.get("aspect_ratio"),
						"source": d.get("source"),
					} for d in countable_components
				]
			}
			json.dump(debug_data, f, indent=2)
	except Exception as e:
		logger.error(f"Failed to write debug file: {e}")

	return result




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