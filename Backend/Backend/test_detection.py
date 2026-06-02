#!/usr/bin/env python3
"""Test script to verify component detection accuracy without OCR."""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from local_detection import (
    detect_shape_components,
    empty_counts,
    merge_close_detections,
    dedupe_detections,
)

def test_detection(image_path: Path):
    """Test detection on a single image without OCR."""
    print(f"\nTesting detection on: {image_path.name}")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Shape-based component detection (without OCR)
    print("Running shape-based component detection...")
    shape_components = detect_shape_components(image_rgb, [])
    print(f"Found {len(shape_components)} shape components")
    
    # Print shape component details
    for comp in shape_components[:15]:  # Show first 15
        print(f"  - {comp['category']}: {comp['name']} (confidence: {comp['confidence']:.2f}, area: {comp['area']:.0f})")
    
    # Dedupe
    print("\nDeduplicating detections...")
    deduped = dedupe_detections(shape_components, iou_threshold=0.35)
    print(f"After deduplication: {len(deduped)} components")
    
    # Merge close detections
    print("\nMerging close detections...")
    merged = merge_close_detections(deduped, distance_ratio=1.0)
    print(f"After merging: {len(merged)} components")
    
    # Calculate final counts
    final_counts = empty_counts()
    for comp in merged:
        category = comp.get("category")
        if category in final_counts:
            final_counts[category] += 1
    
    print(f"\nFinal counts: {final_counts}")
    print("=" * 60)
    
    return final_counts

if __name__ == "__main__":
    # Test with the specific image provided by user
    test_image = Path(r"C:\Users\aadeshpande\Downloads\70642920-a5c7-41df-8e90-26d479e7b4b3.png")
    
    if test_image.exists():
        try:
            test_detection(test_image)
        except Exception as e:
            print(f"Error testing {test_image.name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Image not found: {test_image}")
