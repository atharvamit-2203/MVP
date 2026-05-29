from __future__ import annotations

import json
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


def _extract_features_from_box(image_array: np.ndarray, bbox: Tuple[int, int, int, int], vertex_count: int = 0) -> Dict[str, Any]:
    x, y, w, h = bbox
    h_img, w_img = image_array.shape[:2]
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    roi = image_array[y:y2, x:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    area = float(max(1, w * h))
    mean_int = float(np.mean(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / area)
    contour_area = float(np.count_nonzero(gray < 250))
    aspect = float(w / max(h, 1))
    features = {
        "area": area,
        "aspect": aspect,
        "mean_intensity": mean_int,
        "edge_density": edge_density,
        "contour_area_ratio": contour_area / area,
        "vertex_count": int(vertex_count or 0),
    }
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
            label = ann.get("label", "other")
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

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(df.values, y_int.values)
    joblib.dump({"model": clf, "labels": labels, "columns": df.columns.tolist()}, MODEL_PATH)
    return {"status": "trained", "rows": len(df), "labels": labels}


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_candidates(image_array: np.ndarray, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Given image and candidate detections (with bbox and vertex_count), return with uncertainty scores."""
    model_blob = load_model()
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
        uncertainty = 1.0 - maxp
        results.append({**c, "predicted": predicted, "uncertainty": uncertainty, "prob": maxp})
    return results
