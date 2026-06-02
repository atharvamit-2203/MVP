from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

logger = logging.getLogger(__name__)

if __package__:
    from .local_detection import detect_shape_components, bbox_area, bbox_center
else:
    from local_detection import detect_shape_components, bbox_area, bbox_center

BACKEND_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = BACKEND_ROOT / "annotations"
MODEL_DIR = BACKEND_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "active_model.joblib"

TOP_LEVEL_LABELS = {"motor", "pump", "tank", "valve"}
LABEL_ALIASES = {
    "check_valve": "valve",
    "control_valve": "valve",
    "gate_valve": "valve",
    "globe_valve": "valve",
    "ball_valve": "valve",
    "butterfly_valve": "valve",
    "plug_valve": "valve",
    "pump_centrifugal": "pump",
    "pump_gear": "pump",
    "pump_positive_displacement": "pump",
    "pump_submersible": "pump",
    "pump_diaphragm": "pump",
    "motor_electric": "motor",
    "motor_drive": "motor",
    "tank_vessel": "tank",
    "reactor": "tank",
    "drum": "tank",
    "vessel": "tank",
}

_MODEL_CACHE: dict[str, Any] | None = None
_MODEL_CACHE_MTIME_NS: int | None = None


def _annotations_signature() -> tuple[int, int]:
    """Return a stable signature for the annotations file so feature caches refresh on edits."""
    path = ANNOTATIONS_DIR / "annotations.jsonl"
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _normalize_label(label: str) -> str | None:
    cleaned = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not cleaned:
        return None
    if cleaned in TOP_LEVEL_LABELS:
        return cleaned
    return LABEL_ALIASES.get(cleaned)


def _extract_features_from_box(image_array: np.ndarray, bbox: Tuple[int, int, int, int], vertex_count: int = 0) -> Dict[str, Any]:
    x, y, w, h = bbox
    h_img, w_img = image_array.shape[:2]
    pad_x = max(2, int(w * 0.08))
    pad_y = max(2, int(h * 0.08))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)
    roi = image_array[y1:y2, x1:x2]
    if roi.size == 0:
        roi = image_array[max(0, y):min(h_img, y + max(1, h)), max(0, x):min(w_img, x + max(1, w))]
    if roi.size == 0:
        roi = image_array
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # Reduced bilateral filter parameters for faster processing
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    blurred_full = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh_full = cv2.adaptiveThreshold(
        blurred_full, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 10
    )
    thresh = cv2.resize(thresh_full, (96, 96), interpolation=cv2.INTER_NEAREST)
    resized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    foreground_ratio = float(np.count_nonzero(thresh) / max(1, thresh.size))
    mean_int = float(np.mean(resized))
    std_int = float(np.std(resized))
    edges = cv2.Canny(resized, 50, 150)
    edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
    aspect = float(w / max(h, 1))
    
    # Additional color-based features for better discrimination
    if len(image_array.shape) == 3:
        # Convert ROI to RGB if it's BGR
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if roi.shape[2] == 3 else roi
        # Calculate color statistics in different channels
        mean_r = float(np.mean(roi_rgb[:, :, 0]))
        mean_g = float(np.mean(roi_rgb[:, :, 1]))
        mean_b = float(np.mean(roi_rgb[:, :, 2]))
        std_r = float(np.std(roi_rgb[:, :, 0]))
        std_g = float(np.std(roi_rgb[:, :, 1]))
        std_b = float(np.std(roi_rgb[:, :, 2]))
        # Add color ratios for better discrimination
        total_color = mean_r + mean_g + mean_b + 1e-10
        r_ratio = float(mean_r / total_color)
        g_ratio = float(mean_g / total_color)
        b_ratio = float(mean_b / total_color)
    else:
        mean_r = mean_g = mean_b = mean_int
        std_r = std_g = std_b = std_int
        r_ratio = g_ratio = b_ratio = 0.33
    
    features: Dict[str, Any] = {
        "area": float(max(1, w * h)),
        "aspect": aspect,
        "mean_intensity": mean_int,
        "std_intensity": std_int,
        "edge_density": edge_density,
        "foreground_ratio": foreground_ratio,
        "vertex_count": int(vertex_count or 0),
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "std_r": std_r,
        "std_g": std_g,
        "std_b": std_b,
        "r_ratio": r_ratio,
        "g_ratio": g_ratio,
        "b_ratio": b_ratio,
    }

    if largest is not None:
        contour_area = float(cv2.contourArea(largest))
        perimeter = float(cv2.arcLength(largest, True))
        hull = cv2.convexHull(largest)
        hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else 0.0
        moments = cv2.moments(largest)
        features.update(
            {
                "contour_area": contour_area,
                "perimeter": perimeter,
                "extent": float(contour_area / max(1.0, float(w * h))),
                "solidity": float(contour_area / hull_area) if hull_area > 0 else 0.0,
                "circularity": float((4.0 * math.pi * contour_area) / max(1.0, perimeter * perimeter)),
                "contour_count": float(len(contours)),
            }
        )
        if moments.get("m00"):
            hu = cv2.HuMoments(moments).flatten()
            for index, value in enumerate(hu, start=1):
                features[f"hu_{index}"] = float(-math.copysign(1.0, value) * math.log10(abs(value) + 1e-12))

    left_right = np.mean(np.abs(resized[:, :48].astype(np.float32) - np.fliplr(resized[:, 48:]).astype(np.float32)))
    top_bottom = np.mean(np.abs(resized[:48, :].astype(np.float32) - np.flipud(resized[48:, :]).astype(np.float32)))
    features["symmetry_lr"] = float(left_right / 255.0)
    features["symmetry_tb"] = float(top_bottom / 255.0)

    # Optimized HOG with smaller block size for faster computation
    hog = cv2.HOGDescriptor(
        (32, 32),
        (8, 8),
        (4, 4),
        (4, 4),
        9,
    )
    hog_vector = hog.compute(cv2.resize(resized, (32, 32), interpolation=cv2.INTER_AREA)).flatten()
    for index, value in enumerate(hog_vector):
        features[f"hog_{index}"] = float(value)

    return features


@lru_cache(maxsize=2)
def _load_annotation_lines_cached(signature: tuple[int, int]) -> List[Dict[str, Any]]:
    path = ANNOTATIONS_DIR / "annotations.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _load_annotation_lines() -> List[Dict[str, Any]]:
    return _load_annotation_lines_cached(_annotations_signature())


@lru_cache(maxsize=2)
def _build_library_feature_bank(signature: tuple[int, int]) -> Dict[str, List[Dict[str, Any]]]:
    """Precompute library features once so per-component matching is fast."""
    bank: Dict[str, List[Dict[str, Any]]] = {label: [] for label in TOP_LEVEL_LABELS}
    anns = _load_annotation_lines_cached(signature)
    for entry in anns:
        image_name = entry.get("image")
        image_path = ANNOTATIONS_DIR / image_name
        if not image_name or not image_path.exists():
            continue
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        for ann in entry.get("annotations", []):
            label = _normalize_label(ann.get("label", ""))
            if label not in TOP_LEVEL_LABELS:
                continue
            bbox = ann.get("bbox", [0, 0, img.shape[1], img.shape[0]])
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            bank[label].append(
                {
                    "image": image_name,
                    "label": label,
                    "bbox": bbox,
                    "features": _extract_fast_features(img, (x, y, w, h)),
                }
            )
    return bank


def compute_feature_similarity(features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
    """Compute similarity score between two feature dictionaries using cosine similarity."""
    # Get common feature keys
    common_keys = set(features1.keys()) & set(features2.keys())
    if not common_keys:
        return 0.0
    
    # Extract feature vectors
    vec1 = np.array([features1[k] for k in common_keys])
    vec2 = np.array([features2[k] for k in common_keys])
    
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity
    similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
    return max(0.0, similarity)  # Ensure non-negative


def match_component_to_library(
    component_features: Dict[str, Any],
    component_category: str | None = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Match a component against the component library based on visual features."""
    matches = []

    bank = _build_library_feature_bank(_annotations_signature())
    categories = [component_category] if component_category in bank else list(bank.keys())
    if component_category and component_category not in bank:
        categories = list(bank.keys())

    for label in categories:
        for item in bank.get(label, []):
            similarity = compute_feature_similarity(component_features, item["features"])
            matches.append({
                "image": item["image"],
                "label": item["label"],
                "bbox": item["bbox"],
                "similarity": similarity,
            })
    
    # Sort by similarity and return top matches
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches[:top_k]


def _extract_fast_features(image_array: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """Lightweight feature extraction for training — no HOG, no bilateral filter."""
    x, y, w, h = bbox
    h_img, w_img = image_array.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    roi = image_array[y1:y2, x1:x2]
    if roi.size == 0:
        roi = image_array
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(resized, 50, 150)
    aspect = float(w / max(h, 1))
    mean_int = float(np.mean(resized))
    std_int = float(np.std(resized))
    edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
    thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    foreground_ratio = float(np.count_nonzero(thresh) / max(1, thresh.size))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    feats: Dict[str, Any] = {
        "area": float(max(1, w * h)),
        "aspect": aspect,
        "mean_intensity": mean_int,
        "std_intensity": std_int,
        "edge_density": edge_density,
        "foreground_ratio": foreground_ratio,
    }
    if largest is not None:
        ca = float(cv2.contourArea(largest))
        perim = float(cv2.arcLength(largest, True))
        hull = cv2.convexHull(largest)
        hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else 0.0
        feats["circularity"] = float((4.0 * math.pi * ca) / max(1.0, perim * perim))
        feats["solidity"] = float(ca / hull_area) if hull_area > 0 else 0.0
        feats["extent"] = float(ca / max(1.0, float(w * h)))
        
        # Add expert shape features (Hu Moments) for scale/rotation invariant shape matching
        moments = cv2.moments(largest)
        if moments.get("m00"):
            hu = cv2.HuMoments(moments).flatten()
            for index, value in enumerate(hu, start=1):
                feats[f"hu_{index}"] = float(-math.copysign(1.0, value) * math.log10(abs(value) + 1e-12))
    else:
        feats["circularity"] = 0.0
        feats["solidity"] = 0.0
        feats["extent"] = 0.0
        for i in range(1, 8):
            feats[f"hu_{i}"] = 0.0

    # Add expert texture/gradient features (HOG) to distinguish internal details (e.g. mixer blades vs empty tank)
    try:
        hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
        hog_vector = hog.compute(resized).flatten()
        for index, value in enumerate(hog_vector[:36]): # Take first 36 bins to keep it fast
            feats[f"hog_{index}"] = float(value)
    except Exception:
        pass

    return feats


def build_training_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    rows = []
    seen: set[str] = set()
    anns = _load_annotation_lines()
    for entry in anns:
        image_name = entry.get("image")
        image_path = ANNOTATIONS_DIR / image_name
        if not image_path.exists():
            continue
        img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        for ann in entry.get("annotations", []):
            label = _normalize_label(ann.get("label", ""))
            if label not in TOP_LEVEL_LABELS:
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            # Deduplicate: same image + same bbox + same label seen before → skip
            dedup_key = f"{image_name}|{bbox}|{label}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            feats = _extract_fast_features(img, tuple(bbox))
            feats["label"] = label
            rows.append(feats)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int)

    df = pd.DataFrame(rows)
    y = df.pop("label")
    return df, y


def train_model() -> Dict[str, Any]:
    df, y = build_training_dataset()
    if df.empty:
        return {"status": "no_data"}
    if len(df) < 4:
        logger.warning(
            "Only %s training sample(s) — upload more component photos (2+ per type) for reliable P&ID counts.",
            len(df),
        )
    # encode labels
    labels = sorted(y.unique())
    label_to_int = {lab: i for i, lab in enumerate(labels)}
    y_int = y.map(label_to_int)

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
    )
    clf.fit(df.values, y_int.values)
    joblib.dump({"model": clf, "labels": labels, "columns": df.columns.tolist(), "feature_version": FEATURE_VERSION}, MODEL_PATH)
    invalidate_model_cache()
    return {"status": "trained", "rows": len(df), "labels": labels}


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


FEATURE_VERSION = 4  # increment when fast-feature schema changes


def load_model_cached():
    global _MODEL_CACHE, _MODEL_CACHE_MTIME_NS
    if not MODEL_PATH.exists():
        invalidate_model_cache()
        return None
    current_mtime = MODEL_PATH.stat().st_mtime_ns
    if _MODEL_CACHE is not None and _MODEL_CACHE_MTIME_NS == current_mtime:
        return _MODEL_CACHE
    blob = load_model()
    # Reject stale models trained with a different feature schema
    if blob is not None and blob.get("feature_version") != FEATURE_VERSION:
        logger.warning("Stale model (wrong feature_version), deleting and retraining.")
        MODEL_PATH.unlink(missing_ok=True)
        invalidate_model_cache()
        try:
            train_model()
            blob = load_model()
        except Exception as exc:
            logger.warning(f"Auto-retrain after stale model failed: {exc}")
            return None
    _MODEL_CACHE = blob
    _MODEL_CACHE_MTIME_NS = current_mtime if blob is not None else None
    return _MODEL_CACHE


def invalidate_model_cache() -> None:
    global _MODEL_CACHE, _MODEL_CACHE_MTIME_NS
    _MODEL_CACHE = None
    _MODEL_CACHE_MTIME_NS = None


def predict_candidates(
    image_array: np.ndarray,
    candidates: List[Dict[str, Any]],
    model_blob: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Given image and candidate detections (with bbox and vertex_count), return with uncertainty scores."""
    if model_blob is None:
        model_blob = load_model_cached()
    
    if model_blob is None:
        # no model: return candidates with uncertainty 1.0
        results = []
        for c in candidates:
            results.append({**c, "uncertainty": 1.0})
        return results

    clf = model_blob["model"]
    cols = model_blob["columns"]
    labels = model_blob["labels"]
    feats_list = []
    for c in candidates:
        bbox = tuple(c.get("bbox", (0, 0, 0, 0)))
        feats = _extract_fast_features(image_array, bbox)
        feats_list.append([feats.get(col, 0) for col in cols])

    probs = clf.predict_proba(feats_list)
    results = []
    for c, p in zip(candidates, probs):
        maxp = float(max(p))
        label_idx = int(p.argmax())
        predicted = labels[label_idx]
        uncertainty = 1.0 - maxp
        results.append({**c, "predicted": predicted, "uncertainty": uncertainty, "prob": maxp})
    return results
