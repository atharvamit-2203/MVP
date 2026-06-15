"""
New Pipeline API Integration
Integrates the 4-phase pipeline with the existing backend API structure
"""
import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np
from fastapi import File, HTTPException, UploadFile
from PIL import Image

import sys
current_dir = Path(__file__).resolve().parent
backend_root = current_dir.parent

# Add both Backend and new_pipeline to path
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from new_pipeline.pipeline_orchestrator import PipelineOrchestrator
from new_pipeline.phase2_pid_analyzer import PIDImageAnalyzer
from new_pipeline.utils import load_image, image_to_base64

# Import existing backend utilities
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import local_detection

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class NewPipelineAnalyzer:
    """Analyzer using the new 4-phase pipeline"""
    
    def __init__(self):
        self.phase2_analyzer = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the pipeline (lazy loading)"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing new pipeline analyzer...")
            self.phase2_analyzer = PIDImageAnalyzer()
            self._initialized = True
            logger.info("New pipeline analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize new pipeline: {e}")
            self._initialized = False
    
    async def analyze_pid_image(self, image: Image.Image) -> dict[str, Any]:
        """Analyze P&ID image using the new pipeline"""
        if not self._initialized:
            self.initialize()
        
        if not self._initialized:
            logger.error("New pipeline initialization failed - cannot proceed without it")
            raise HTTPException(status_code=500, detail="New pipeline initialization failed. Please check model paths and dependencies.")
        
        try:
            # Convert PIL to OpenCV format
            image_array = np.array(image.convert("RGB"))
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Save temporarily for the pipeline
            temp_path = BACKEND_ROOT / "temp_analysis.png"
            cv2.imwrite(str(temp_path), image_bgr)
            
            # Run Phase 2 analysis (P&ID Image Analysis)
            results = self.phase2_analyzer.analyze_pid_image(str(temp_path))
            
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            
            # Convert to format compatible with frontend
            return self._convert_to_frontend_format(results, image)
            
        except Exception as e:
            logger.error(f"New pipeline analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"New pipeline analysis failed: {str(e)}")
    
    async def _fallback_analysis(self, image: Image.Image) -> dict[str, Any]:
        """Fallback to existing local detection"""
        try:
            # Use existing local_detection module
            result = await asyncio.to_thread(local_detection.analyze_pid_image, image)
            
            # Convert to new pipeline format
            return {
                'detections': result.get('detections', []),
                'counts': result.get('counts', {}),
                'ocr_counts': result.get('ocr_counts', {}),
                'vision_counts': result.get('vision_counts', {}),
                'coordinates': result.get('coordinates', {}),
                'opencv_results': {
                    'total_pipes': result.get('counts', {}).get('pipe', 0),
                    'total_lines': 0,
                    'lines': []
                },
                'ocr_results': {
                    'instrument_tags': [],
                    'total_text': len(result.get('ocr_detections', []))
                },
                'heuristic_results': {
                    'validated_components': result.get('detections', [])
                }
            }
        except Exception as e:
            logger.error(f"Fallback analysis also failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    def _convert_to_frontend_format(self, pipeline_results: dict, original_image: Image.Image) -> dict[str, Any]:
        """Convert new pipeline results to frontend-compatible format"""
        opencv_results = pipeline_results.get('opencv_results', {})
        ocr_results = pipeline_results.get('ocr_results', {})
        heuristic_results = pipeline_results.get('heuristic_results', {})
        
        # Extract component counts
        validated_components = heuristic_results.get('validated_components', [])
        
        # Debug logging
        print(f"DEBUG: Validated components count: {len(validated_components)}")
        if validated_components:
            print(f"DEBUG: Sample component: {validated_components[0]}")
        
        counts = {
            'motor': 0,
            'pump': 0,
            'tank': 0,
            'valve': 0,
            'instrument': 0,
            'other': 0,
            'pipe': opencv_results.get('total_pipes', 0)  # Add pipe count
        }
        
        # Count components by type
        for component in validated_components:
            label = component.get('label', '').lower()
            print(f"DEBUG: Processing component with label: {label}")
            if 'pump' in label:
                counts['pump'] += 1
            elif 'valve' in label:
                counts['valve'] += 1
            elif 'tank' in label or 'vessel' in label:
                counts['tank'] += 1
            elif 'motor' in label:
                counts['motor'] += 1
            elif 'instrument' in label or 'sensor' in label or 'tag' in label:
                counts['instrument'] += 1
            else:
                counts['other'] += 1
        
        print(f"DEBUG: Final counts: {counts}")
        
        # Add OCR-detected instruments
        for tag in ocr_results.get('instrument_tags', []):
            counts['instrument'] += 1
        
        # Build detections list
        detections = []
        for component in validated_components:
            bbox = component.get('bbox', [0, 0, 0, 0])
            if len(bbox) == 4:
                x, y, x2, y2 = bbox
                detections.append({
                    'name': component.get('label', 'Unknown'),
                    'category': self._infer_category(component.get('label', '')),
                    'bbox': [int(x), int(y), int(x2 - x), int(y2 - y)],
                    'confidence': component.get('confidence', 0.5)
                })
        
        # Add OCR text as components
        for tag in ocr_results.get('instrument_tags', []):
            bbox = tag.get('bbox', [])
            if len(bbox) >= 2:
                # Convert polygon bbox to rectangle
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x, y = min(x_coords), min(y_coords)
                w, h = max(x_coords) - x, max(y_coords) - y
                detections.append({
                    'name': tag.get('text', ''),
                    'category': 'instrument',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': tag.get('confidence', 0.5)
                })
        
        # Build coordinates in frontend format
        coordinates = self._build_coordinates(detections, original_image, opencv_results)
        
        # Return in expected format with pipe count and connections
        return {
            'detections': detections,
            'counts': counts,
            'ocr_counts': counts.copy(),  # Same for now
            'vision_counts': counts.copy(),  # Same for now
            'coordinates': coordinates,
            'opencv_results': {
                'total_pipes': opencv_results.get('total_pipes', 0),
                'total_lines': opencv_results.get('total_lines', 0),
                'lines': opencv_results.get('lines', []),
                'junctions': opencv_results.get('junctions', [])
            },
            'ocr_results': ocr_results,
            'heuristic_results': heuristic_results,
            'pipeline_info': {
                'pipe_count': opencv_results.get('total_pipes', 0),
                'line_count': opencv_results.get('total_lines', 0),
                'junction_count': opencv_results.get('total_junctions', 0),
                'component_count': len(validated_components)
            }
        }
    
    def _infer_category(self, label: str) -> str:
        """Infer category from label"""
        label_lower = label.lower()
        
        if any(kw in label_lower for kw in ['pump', 'centrifugal', 'gear']):
            return 'pump'
        elif any(kw in label_lower for kw in ['valve', 'gate', 'globe', 'ball']):
            return 'valve'
        elif any(kw in label_lower for kw in ['tank', 'vessel', 'reactor']):
            return 'tank'
        elif any(kw in label_lower for kw in ['motor', 'drive']):
            return 'motor'
        elif any(kw in label_lower for kw in ['instrument', 'sensor', 'tag', 'pt', 'lt', 'ft']):
            return 'instrument'
        else:
            return 'other'
    
    def _build_coordinates(self, detections: list, image: Image.Image, opencv_results: dict) -> dict:
        """Build coordinate structure in frontend format"""
        width, height = image.size
        
        children = []
        for i, detection in enumerate(detections):
            bbox = detection.get('bbox', [0, 0, 0, 0])
            if len(bbox) == 4:
                x, y, w, h = bbox
                
                # Clamp values to be non-negative
                w = max(0, float(w))
                h = max(0, float(h))
                
                # Convert to percentage or keep as pixels
                # Frontend expects pixels in the current implementation
                children.append({
                    'type': 'ia.symbol.basic',
                    'meta': {
                        'name': detection.get('name', f'Component_{i}')
                    },
                    'position': {
                        'x': float(x),
                        'y': float(y),
                        'width': w,
                        'height': h
                    },
                    'props': {}
                })
        
        # Add pipe lines as connections
        lines = opencv_results.get('lines', [])
        for line in lines:
            if line.get('is_pipe', False):
                start = line.get('start', [0, 0])
                end = line.get('end', [0, 0])
                line_width = max(0, float(end[0] - start[0]))
                line_height = max(0, float(end[1] - start[1]))
                children.append({
                    'type': 'ia.shape.line',
                    'meta': {
                        'name': f'Pipe_{len(children)}'
                    },
                    'position': {
                        'x': float(start[0]),
                        'y': float(start[1]),
                        'width': line_width,
                        'height': line_height
                    },
                    'props': {
                        'stroke': '#000000',
                        'strokeWidth': 2
                    }
                })
        
        return {
            'root': {
                'type': 'ia.cloud',
                'meta': {
                    'name': 'P&ID Components'
                },
                'children': children,
                'position': {},
                'props': {}
            },
            'custom': {},
            'params': {},
            'props': {}
        }


# Global analyzer instance
new_pipeline_analyzer = NewPipelineAnalyzer()


async def analyze_with_new_pipeline(file: UploadFile) -> dict[str, Any]:
    """
    Analyze P&ID image using the new 4-phase pipeline
    
    Returns results in format compatible with existing frontend
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    
    try:
        # Load image
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Analyze with new pipeline
        results = await new_pipeline_analyzer.analyze_pid_image(image)
        
        return results
        
    except Exception as e:
        logger.error(f"New pipeline analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
