from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

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
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
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
    features: Dict[str, Any] = {
        "area": float(max(1, w * h)),
        "aspect": aspect,
        "mean_intensity": mean_int,
        "std_intensity": std_int,
        "edge_density": edge_density,
        "foreground_ratio": foreground_ratio,
        "vertex_count": int(vertex_count or 0),
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

    hog = cv2.HOGDescriptor(
        (64, 64),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    )
    hog_vector = hog.compute(cv2.resize(resized, (64, 64), interpolation=cv2.INTER_AREA)).flatten()
    for index, value in enumerate(hog_vector):
        features[f"hog_{index}"] = float(value)

    return features


def _load_annotation_lines() -> List[Dict[str, Any]]:
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


def build_training_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    rows = []
    anns = _load_annotation_lines()
    for entry in anns:
        image_name = entry.get("image")
        image_path = ANNOTATIONS_DIR / image_name
        if not image_path.exists():
            continue
        img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        for ann in entry.get("annotations", []):
            label = _normalize_label(ann.get("label", "other")) or "other"
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            # approximate vertex_count as 0 (could be improved)
            feats = _extract_features_from_box(img, tuple(bbox), vertex_count=0)
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
    # encode labels
    labels = sorted(y.unique())
    label_to_int = {lab: i for i, lab in enumerate(labels)}
    y_int = y.map(label_to_int)

    clf = RandomForestClassifier(
        n_estimators=300, 
        random_state=42, 
        class_weight="balanced_subsample",
        max_depth=5,
        min_samples_leaf=2
    )
    clf.fit(df.values, y_int.values)
    joblib.dump({"model": clf, "labels": labels, "columns": df.columns.tolist(), "feature_version": 2}, MODEL_PATH)
    invalidate_model_cache()
    return {"status": "trained", "rows": len(df), "labels": labels}


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_model_cached():
    global _MODEL_CACHE, _MODEL_CACHE_MTIME_NS
    if not MODEL_PATH.exists():
        invalidate_model_cache()
        return None
    current_mtime = MODEL_PATH.stat().st_mtime_ns
    if _MODEL_CACHE is not None and _MODEL_CACHE_MTIME_NS == current_mtime:
        return _MODEL_CACHE
    _MODEL_CACHE = load_model()
    _MODEL_CACHE_MTIME_NS = current_mtime if _MODEL_CACHE is not None else None
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
    model_blob = model_blob or load_model_cached()
    results = []
    if model_blob is None:
        # no model: return candidates with uncertainty 1.0
        for c in candidates:
            results.append({**c, "uncertainty": 1.0})
        return results

    clf = model_blob["model"]
    cols = model_blob["columns"]
    labels = model_blob["labels"]
    feats_list = []
    for c in candidates:
        bbox = tuple(c.get("bbox", (0, 0, 0, 0)))
        vc = c.get("vertex_count", 0)
        feats = _extract_features_from_box(image_array, bbox, vertex_count=vc)
        feats_list.append([feats.get(col, 0) for col in cols])

    probs = clf.predict_proba(feats_list)
    for c, p in zip(candidates, probs):
        maxp = float(max(p))
        label_idx = int(p.argmax())
        predicted = labels[label_idx]
        fallback_category = str(c.get("category", "")).strip().lower()
        fallback_conf = float(c.get("confidence", 0.0) or 0.0)
        if fallback_category == "valve" and predicted != "valve" and fallback_conf >= 0.45:
            predicted = "valve"
            maxp = max(maxp, fallback_conf)
        uncertainty = 1.0 - maxp
        results.append({**c, "predicted": predicted, "uncertainty": uncertainty, "prob": maxp})
    return results
