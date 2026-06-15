"""
Utility functions for the new pipeline
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import base64
from PIL import Image
import io


def load_image(image_path: str) -> np.ndarray:
    """Load image from path"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return img


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image to path"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


def crop_image(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """Crop image using bounding box [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def resize_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h))


def image_to_base64(image: np.ndarray) -> str:
    """Convert image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to image"""
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def draw_bounding_boxes(image: np.ndarray, bboxes: List[List[int]], 
                        labels: List[str] = None, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes on image"""
    img_copy = image.copy()
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        if labels and i < len(labels):
            cv2.putText(img_copy, labels[i], (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_copy


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Calculate Intersection over Union for two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(bboxes: List[List[int]], scores: List[float], 
                       iou_threshold: float = 0.5) -> List[int]:
    """Apply Non-Maximum Suppression to bounding boxes"""
    if len(bboxes) == 0:
        return []
    
    # Sort by score
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while indices:
        current = indices.pop(0)
        keep.append(current)
        
        # Remove overlapping boxes
        remaining = []
        for idx in indices:
            if calculate_iou(bboxes[current], bboxes[idx]) < iou_threshold:
                remaining.append(idx)
        indices = remaining
    
    return keep


def save_json(data: Any, output_path: str) -> None:
    """Save data to JSON file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(input_path: str) -> Any:
    """Load data from JSON file"""
    with open(input_path, 'r') as f:
        return json.load(f)


def merge_overlapping_bboxes(bboxes: List[List[int]], labels: List[str], 
                            iou_threshold: float = 0.7) -> Tuple[List[List[int]], List[str]]:
    """Merge overlapping bounding boxes with same labels"""
    if len(bboxes) == 0:
        return [], []
    
    merged_bboxes = []
    merged_labels = []
    used = [False] * len(bboxes)
    
    for i in range(len(bboxes)):
        if used[i]:
            continue
        
        current_bbox = bboxes[i]
        current_label = labels[i]
        used[i] = True
        
        # Find overlapping boxes with same label
        for j in range(i + 1, len(bboxes)):
            if used[j] or labels[j] != current_label:
                continue
            
            if calculate_iou(current_bbox, bboxes[j]) >= iou_threshold:
                # Merge boxes
                x1 = min(current_bbox[0], bboxes[j][0])
                y1 = min(current_bbox[1], bboxes[j][1])
                x2 = max(current_bbox[2], bboxes[j][2])
                y2 = max(current_bbox[3], bboxes[j][3])
                current_bbox = [x1, y1, x2, y2]
                used[j] = True
        
        merged_bboxes.append(current_bbox)
        merged_labels.append(current_label)
    
    return merged_bboxes, merged_labels
