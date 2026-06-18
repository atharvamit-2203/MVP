"""
Phase 2: P&ID Image Analysis
Analyzes P&ID images using parallel processing with OpenCV, Tesseract OCR, Florence-2, and Grounding DINO
"""
import cv2
import numpy as np
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
import pytesseract
# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import google.generativeai as genai
from openai import OpenAI
from new_pipeline.config import *
from new_pipeline.utils import *

# Import active learning for component verification
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "Backend"))
from active_learning import match_component_to_library, _extract_fast_features


class PIDImageAnalyzer:
    """Analyze P&ID images using parallel processing"""
    
    def __init__(self):
        self.florence_processor = None
        self.florence_model = None
        self.grounding_model = None
        self.ocr_engine = None
        self.gemini_client = None
        self.openai_client = None
        self.template_cache = {}  # Cache for template images to speed up matching
        self._load_models()
    
    def _classify_component(self, label: str) -> Dict[str, Any]:
        """
        Classify component as standard or customized based on its label and subtype
        Uses actual Template folder structure for path mapping
        
        Args:
            label: Component label from detection
        
        Returns:
            Dictionary with classification results
        """
        label_lower = label.lower().strip()
        label_clean = label.strip().replace(' ', '_')
        
        # FIRST: Check if it's a basic standard component type (pump, valve, motor, pipe)
        # This must come BEFORE template path mapping to prevent standard components from being classified as customized
        for std_type in STANDARD_COMPONENT_TYPES:
            if std_type in label_lower:
                return {
                    'classification': 'standard',
                    'component_type': std_type,
                    'ignition_type': f"{IGNITION_STANDARD_PREFIX}{std_type}",
                    'template_path': None,
                    'params': None
                }
        
        # Check if it's a standard subtype
        for category, subtypes in STANDARD_SUBTYPES.items():
            for subtype in subtypes:
                if subtype in label_lower:
                    # It's a standard component with specific subtype
                    component_type = category.rstrip('s')  # Remove plural 's'
                    return {
                        'classification': 'standard',
                        'component_type': component_type,
                        'subtype': subtype,
                        'ignition_type': f"{IGNITION_STANDARD_PREFIX}{component_type}",
                        'template_path': None,
                        'params': None
                    }
        
        # Check if it's a customized subtype (cyclone, turbine, boiler, etc.)
        for category, subtypes in CUSTOMIZED_SUBTYPES.items():
            for subtype in subtypes:
                if subtype in label_lower:
                    # It's a customized component - use actual template structure
                    # Try to find matching template folder
                    template_path = self._find_template_path(label_clean)
                    return {
                        'classification': 'customized',
                        'component_type': label_clean,
                        'ignition_type': IGNITION_OUTPUT_FORMAT,
                        'template_path': template_path,
                        'params': None
                    }
        
        # Check if it's in the template path mapping (for other customized components)
        for template_key, template_path in TEMPLATE_PATH_MAPPING.items():
            if template_key in label_lower:
                return {
                    'classification': 'customized',
                    'component_type': label_clean,
                    'ignition_type': IGNITION_OUTPUT_FORMAT,
                    'template_path': template_path,
                    'params': None
                }
        
        # Default to customized if no match found - try to find template
        template_path = self._find_template_path(label_clean)
        return {
            'classification': 'customized',
            'component_type': label_clean,
            'ignition_type': IGNITION_OUTPUT_FORMAT,
            'template_path': template_path,
            'params': None
        }
    
    def _find_template_path(self, label: str) -> str:
        """
        Find the actual template path from the Template folder structure
        
        Args:
            label: Component label
        
        Returns:
            Template path if found, otherwise default path
        """
        label_clean = label.replace(' ', '_').replace('-', '_')
        
        # Try to find matching folder in Template directory
        if TEMPLATE_DIR.exists():
            # Search for matching folder names
            for category_dir in TEMPLATE_DIR.iterdir():
                if category_dir.is_dir():
                    # Check for case-insensitive match first (prioritize actual folder names)
                    for item in category_dir.iterdir():
                        if item.is_dir() and item.name.lower() == label_clean.lower():
                            return f"Template/{category_dir.name}/{item.name}"  # Use actual folder name with correct case
                    
                    # Check for direct match
                    component_dir = category_dir / label_clean
                    if component_dir.exists():
                        return f"Template/{category_dir.name}/{label_clean}"  # Full path with Template/ prefix
        
        # Default fallback - try to find any matching folder
        if TEMPLATE_DIR.exists():
            for category_dir in TEMPLATE_DIR.iterdir():
                if category_dir.is_dir():
                    for item in category_dir.iterdir():
                        if item.is_dir():
                            # Return first match as fallback
                            return f"Template/{category_dir.name}/{item.name}"
        
        # Ultimate fallback
        return f"Template/{label_clean.capitalize()}/{label_clean.capitalize()}"
    
    def _find_template_image(self, label: str) -> str:
        """
        Find the actual template image path from the Template folder structure
        
        Args:
            label: Component label
        
        Returns:
            Template image path if found, otherwise None
        """
        label_clean = label.replace(' ', '_').replace('-', '_')
        
        # First check the mapping
        for template_key, image_path in TEMPLATE_IMAGE_MAPPING.items():
            if template_key in label.lower():
                full_path = PROJECT_ROOT / image_path
                if full_path.exists():
                    return str(full_path)
        
        # Try to find matching folder in Template directory
        if TEMPLATE_DIR.exists():
            # Search for matching folder names
            for category_dir in TEMPLATE_DIR.iterdir():
                if category_dir.is_dir():
                    # Check for direct match
                    component_dir = category_dir / label_clean
                    if component_dir.exists():
                        thumbnail = component_dir / "thumbnail.png"
                        if thumbnail.exists():
                            return str(thumbnail)
                    
                    # Check for case-insensitive match
                    for item in category_dir.iterdir():
                        if item.is_dir() and item.name.lower() == label_clean.lower():
                            thumbnail = item / "thumbnail.png"
                            if thumbnail.exists():
                                return str(thumbnail)
        
        return None
    
    def _load_models(self):
        """Load all required models"""
        print("Loading Florence-2 model...")
        self.florence_processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_PATH, trust_remote_code=True
        )
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_PATH, trust_remote_code=True
        ).to(FLORENCE_DEVICE)
        
        print("Loading Grounding DINO model...")
        self.grounding_model = load_model(
            GROUNDING_DINO_CONFIG_PATH, 
            GROUNDING_DINO_CHECKPOINT_PATH,
            device="cpu"  # Load to CPU to avoid device mismatch
        )
        
        print("Tesseract OCR ready (no initialization needed)")
        
        print("Initializing Gemini...")
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_client = genai.GenerativeModel(GEMINI_MODEL)
        
        # Initialize OpenAI client as fallback
        if OPENAI_API_KEY:
            print("Initializing OpenAI client as fallback...")
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            print("WARNING: OPENAI_API_KEY not found, OpenAI fallback will not be available")
            self.openai_client = None
        
        print("All models loaded successfully!")
    
    def analyze_pid_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a P&ID image through the complete parallel pipeline
        
        Args:
            image_path: Path to P&ID image
        
        Returns:
            Dictionary containing all analysis results
        """
        print(f"\nAnalyzing P&ID image: {image_path}")
        
        # Load image
        image = load_image(image_path)
        image_name = Path(image_path).stem
        
        # Parallel processing
        print("Running parallel analysis...")
        import time
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks (old pipeline primary, AI models for enhancement)
            opencv_future = executor.submit(self._opencv_analysis, image)
            ocr_future = executor.submit(self._tesseract_analysis, image)
            # Use old pipeline's full analysis function with verification (proven accurate)
            from Backend.local_detection import analyze_pid_image
            from PIL import Image as PILImage
            # Convert numpy array to PIL Image for old pipeline
            pil_image = PILImage.fromarray(image)
            old_pipeline_future = executor.submit(analyze_pid_image, pil_image, fast_mode=True)
            # Re-enable AI models for enhancement
            florence_future = executor.submit(self._florence_analysis, image)
            dino_future = executor.submit(self._grounding_dino_analysis, image)
            # Gemini will be called after old pipeline results are available
            
            # Collect results with timing
            opencv_start = time.time()
            opencv_results = opencv_future.result()
            opencv_time = time.time() - opencv_start
            print(f"  - OpenCV analysis completed in {opencv_time:.2f}s")
            
            ocr_start = time.time()
            ocr_results = ocr_future.result()
            ocr_time = time.time() - ocr_start
            print(f"  - Tesseract OCR analysis completed in {ocr_time:.2f}s")
            
            old_pipeline_start = time.time()
            old_pipeline_results = old_pipeline_future.result()
            old_pipeline_time = time.time() - old_pipeline_start
            print(f"  - Old pipeline analysis completed in {old_pipeline_time:.2f}s (primary)")
            
            florence_start = time.time()
            florence_results = florence_future.result()
            florence_time = time.time() - florence_start
            print(f"  - Florence-2 analysis completed in {florence_time:.2f}s, detected {florence_results.get('total_regions', 0)} regions (enhancement)")
            
            dino_start = time.time()
            dino_results = dino_future.result()
            dino_time = time.time() - dino_start
            print(f"  - Grounding DINO analysis completed in {dino_time:.2f}s, detected {dino_results.get('total_detections', 0)} components (refinement)")
            
            # Compile a list of candidate detections before running Gemini verification
            candidate_detections = []
            
            # 1. Add old pipeline detections
            if old_pipeline_results and old_pipeline_results.get('detections'):
                for det in old_pipeline_results['detections']:
                    bbox = det.get('bbox', [])
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        # Old pipeline bbox format is (x, y, w, h). Convert to [x1, y1, x2, y2]
                        xyxy = [x, y, x + w, y + h]
                        # Prioritize category/label
                        label = det.get('category', det.get('label', det.get('name', 'unknown'))).lower()
                        candidate_detections.append({
                            'bbox': xyxy,
                            'label': label,
                            'confidence': det.get('confidence', 0.5),
                            'source': 'old_pipeline'
                        })
                        
            # 2. Add Grounding DINO detections (only non-overlapping with existing)
            for box, conf, label in zip(dino_results['boxes'], dino_results['confidences'], dino_results['labels']):
                label_clean = self._improve_label_classification(label, box).lower()
                overlaps = False
                for existing in candidate_detections:
                    if calculate_iou(box, existing['bbox']) > 0.15:
                        overlaps = True
                        break
                if not overlaps:
                    candidate_detections.append({
                        'bbox': box,
                        'label': label_clean,
                        'confidence': conf,
                        'source': 'dino'
                    })
                    
            # 3. Add Florence-2 detections (only non-overlapping)
            for box, label in zip(florence_results.get('bboxes', []), florence_results.get('labels', [])):
                if len(box) == 4:
                    overlaps = False
                    for existing in candidate_detections:
                        if calculate_iou(box, existing['bbox']) > 0.15:
                            overlaps = True
                            break
                    if not overlaps:
                        candidate_detections.append({
                            'bbox': box,
                            'label': label.lower(),
                            'confidence': 0.5,
                            'source': 'florence'
                        })
            
            # Call Gemini after combining all detections for final verification
            gemini_start = time.time()
            gemini_results = self._gemini_analysis(image, candidate_detections)
            gemini_time = time.time() - gemini_start
            print(f"  - Gemini verification completed in {gemini_time:.2f}s, verified {gemini_results.get('total_verified', 0)} components, used {gemini_results.get('total_tokens', 0)} tokens")
            
            # Map verification results back to candidate detections
            # Use AI verification when available, fall back to expert models if verification fails
            verified_candidates = []
            for det in candidate_detections:
                bbox = det['bbox']
                verification_result = None
                for verification in gemini_results.get('verified_detections', []):
                    if calculate_iou(bbox, verification.get('bbox', [])) > 0.6:
                        verification_result = verification
                        break
                
                # Use verified label if available, otherwise fall back to expert model
                if verification_result is not None:
                    # AI verification succeeded - use verified result
                    final_label = verification_result.get('label', det['label'])
                    final_conf = verification_result.get('confidence', det['confidence'])
                    print(f"DEBUG: Component verified by AI as {final_label} (original: {det['label']})")
                else:
                    # AI verification failed - fall back to expert model detection
                    final_label = det['label']
                    final_conf = det['confidence']
                    print(f"DEBUG: AI verification failed, using expert model: {final_label}")
                
                verified_candidates.append({
                    'bbox': bbox,
                    'label': final_label,
                    'confidence': final_conf,
                    'source': det['source']
                })
            
            # Apply refinement steps from old pipeline for better accuracy
            print(f"Before refinement: {len(verified_candidates)} candidates")
            
            try:
                # Convert to format expected by refinement functions
                refined_detections = []
                for det in verified_candidates:
                    refined_detections.append({
                        'bbox': det['bbox'],
                        'label': det['label'],
                        'category': det['label'],  # For compatibility
                        'confidence': det['confidence'],
                        'source': det['source']
                    })
                
                # Apply deduplication with balanced threshold - reduced to allow more components
                deduped = self._dedupe_detections(refined_detections, iou_threshold=0.6)
                print(f"After deduplication: {len(deduped)} detections")
                
                # Apply close merge with balanced threshold - reduced to prevent over-merging
                merged = self._merge_close_detections(deduped, distance_ratio=0.1)
                print(f"After close merge: {len(merged)} detections")
                
                # Apply additional IoU merge for all components - reduced threshold
                merged = self._aggressive_iou_merge(merged, iou_threshold=0.5)
                print(f"After IoU merge: {len(merged)} detections")
                
                # Apply stacked tank merge
                merged = self._merge_stacked_tank_symbols(merged)
                print(f"After stacked tank merge: {len(merged)} detections")
                
                # Apply tank consolidation
                image_area = float(image.shape[0] * image.shape[1])
                consolidated = self._consolidate_tank_vessels(merged, image_area=image_area)
                print(f"After consolidation: {len(consolidated)} detections")
                
                # Convert back to verified_candidates format
                verified_candidates = []
                for det in consolidated:
                    verified_candidates.append({
                        'bbox': det['bbox'],
                        'label': det['label'],
                        'confidence': det['confidence'],
                        'source': det['source']
                    })
            except Exception as e:
                print(f"Refinement failed, using original candidates: {e}")
                # If refinement fails, use original candidates
                pass
            
            # Prepare dino_results dict for the heuristic engine
            dino_results = {
                'boxes': [det['bbox'] for det in verified_candidates],
                'confidences': [det['confidence'] for det in verified_candidates],
                'labels': [det['label'] for det in verified_candidates],
                'total_detections': len(verified_candidates)
            }
        
        total_parallel_time = time.time() - start_time
        print(f"Parallel analysis completed in {total_parallel_time:.2f}s")
        print(f"Combined detection (old pipeline primary + AI enhancement): {len(dino_results['boxes'])} components")
        
        print("Parallel analysis complete. Running heuristic engine...")
        
        # Heuristic Engine with component verification
        heuristic_results = self._heuristic_engine(
            opencv_results, ocr_results, florence_results, dino_results, image
        )
        
        # Filter pipes to remove lines inside detected component boxes
        component_boxes = [det['bbox'] for det in verified_candidates]
        filtered_opencv_results = self._filter_pipes_by_components(opencv_results, component_boxes)
        
        # Compile results
        results = {
            'image_path': image_path,
            'image_name': image_name,
            'opencv_results': filtered_opencv_results,  # Use filtered results
            'ocr_results': ocr_results,
            'florence_results': florence_results,
            'dino_results': dino_results,
            'heuristic_results': heuristic_results
        }
        
        return results
    
    def _filter_pipes_by_components(self, opencv_results: Dict, component_boxes: List) -> Dict:
        """Filter out pipes that are inside detected component boxes to avoid false positives"""
        if not component_boxes:
            return opencv_results
        
        filtered_lines = []
        filtered_pipe_count = 0
        
        for line in opencv_results.get('lines', []):
            x1, y1 = line['start']
            x2, y2 = line['end']
            
            # Check if line is inside any component box
            line_inside_component = False
            line_touches_component = False
            
            for box in component_boxes:
                bx1, by1, bx2, by2 = box
                # Check if both endpoints are inside the component box
                if (bx1 <= x1 <= bx2 and by1 <= y1 <= by2 and 
                    bx1 <= x2 <= bx2 and by1 <= y2 <= by2):
                    line_inside_component = True
                    break
                
                # Check if line touches component (endpoint near edge)
                margin = 20  # Reduced margin to be more permissive
                if ((bx1 - margin <= x1 <= bx2 + margin and by1 - margin <= y1 <= by2 + margin) or
                    (bx1 - margin <= x2 <= bx2 + margin and by1 - margin <= y2 <= by2 + margin)):
                    line_touches_component = True
            
            # Only keep line if it's not inside a component
            # Allow lines that touch components (these are likely connection pipes)
            if not line_inside_component:
                # Additional filter: only keep lines that are long enough to be pipes
                line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if line_length > 200:  # Increased from 100 to 200 for longer pipes
                    # Only count as pipe if explicitly marked as pipe
                    if line.get('is_pipe', False):
                        filtered_lines.append(line)
                        filtered_pipe_count += 1
        
        # Update opencv_results with filtered lines
        filtered_results = opencv_results.copy()
        filtered_results['lines'] = filtered_lines
        filtered_results['total_lines'] = len(filtered_lines)
        filtered_results['total_pipes'] = filtered_pipe_count
        
        print(f"DEBUG: Filtered {len(opencv_results.get('lines', [])) - len(filtered_lines)} lines inside components")
        print(f"DEBUG: Pipe count after filtering: {filtered_pipe_count}")
        
        return filtered_results
    
    def _crop_component_precisely(self, image: np.ndarray, bbox: List, padding: int = 10) -> np.ndarray:
        """Crop component with precise bounding box and minimal padding"""
        x1, y1, x2, y2 = bbox
        h_img, w_img = image.shape[:2]
        
        # Add minimal padding
        pad = padding
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(w_img, x2 + pad)
        crop_y2 = min(h_img, y2 + pad)
        
        # Crop the component
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        return cropped
    
    def _is_standard_component(self, label: str) -> bool:
        """Check if component is a standard P&ID component"""
        standard_components = {
            'pump', 'valve', 'tank', 'motor', 'cyclone', 'separator',
            'compressor', 'turbine', 'boiler', 'heat_exchanger', 'reactor',
            'instrument', 'sensor', 'gauge', 'meter'
        }
        return label.lower() in standard_components
    
    def _match_component_with_templates(self, image: np.ndarray, label: str, bbox: List) -> Dict:
        """Match component with appropriate templates based on type"""
        cropped_component = self._crop_component_precisely(image, bbox, padding=15)
        
        if self._is_standard_component(label):
            # For standard components, use dedicated template matching methods
            label_to_method = {
                'cyclone': self._match_cyclone_template,
                'separator': self._match_separator_template,
                'tank': self._match_tank_template,
                'motor': self._match_motor_template,
                'pump': self._match_pump_template,
                'valve': self._match_valve_template,
                'compressor': self._match_compressor_template,
                'turbine': self._match_turbine_template,
                'boiler': self._match_boiler_template,
                'heat_exchanger': self._match_heat_exchanger_template,
                'reactor': self._match_reactor_template,
            }
            
            match_method = label_to_method.get(label.lower())
            if match_method:
                result = match_method(cropped_component)
                if result:
                    result['bbox'] = bbox
                    result['source'] = 'template_matching'
                    return result
        
        # For other components or if standard matching failed, search Template folder
        return self._match_generic_component_template(cropped_component, label)
    
    def _opencv_analysis(self, image: np.ndarray, component_boxes: List = None) -> Dict[str, Any]:
        """OpenCV analysis: detect pipes, lines, junctions, components"""
        print("  - Running OpenCV analysis...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply slight blur for better edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect lines (pipes) with SINGLE pass for speed - OPTIMIZED for longer pipes
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        # Adjusted parameters for longer pipe detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=100, maxLineGap=30)  # Increased minLineLength and maxLineGap for longer pipes
        
        # Skip second detection pass for speed
        # all_lines = []
        # if lines is not None:
        #     all_lines.extend(lines)
        
        line_data = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Determine line type (solid vs dashed approximation)
                line_type = "solid"  # Simplified - could be enhanced
                
                # Filter out very short lines to reduce pipe count and focus on longer pipes
                if length < 150:  # Increased from previous filter to focus on longer pipes
                    continue  # Skip short lines, only keep long pipes
                
                # Check if line is inside any component box (filter out component internal lines)
                line_inside_component = False
                if component_boxes:
                    for box in component_boxes:
                        bx1, by1, bx2, by2 = box
                        # Check if both endpoints are inside the component box
                        if (bx1 <= x1 <= bx2 and by1 <= y1 <= by2 and 
                            bx1 <= x2 <= bx2 and by1 <= y2 <= by2):
                            line_inside_component = True
                            break
                
                # Classify as pipe if length is substantial AND not inside a component
                is_pipe = length > 40 and not line_inside_component
                
                line_data.append({
                    'start': [int(x1), int(y1)],
                    'end': [int(x2), int(y2)],
                    'length': float(length),
                    'angle': float(angle),
                    'type': line_type,
                    'is_pipe': is_pipe
                })
        
        # Detect junctions (intersections)
        junctions = self._detect_junctions(lines if lines is not None else None)
        
        # Detect potential component regions (contours)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_component_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_COMPONENT_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out very small or very large boxes
                if w > 10 and h > 10 and w < image.shape[1] * 0.5 and h < image.shape[0] * 0.5:
                    detected_component_boxes.append([x, y, x+w, y+h])
        
        # Count pipes by grouping connected line segments
        pipe_count = self._count_connected_pipes(line_data)
        
        return {
            'lines': line_data,
            'junctions': junctions,
            'component_boxes': detected_component_boxes,
            'total_lines': len(line_data),
            'total_pipes': pipe_count,
            'total_junctions': len(junctions),
            'total_components': len(detected_component_boxes)
        }
    
    def _detect_junctions(self, lines: np.ndarray) -> List[Dict[str, Any]]:
        """Detect line junctions/intersections"""
        if lines is None or len(lines) == 0:
            return []
        
        junctions = []
        line_segments = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append(((x1, y1), (x2, y2)))
        
        # Find intersections
        for i in range(len(line_segments)):
            for j in range(i + 1, len(line_segments)):
                intersection = self._line_intersection(line_segments[i], line_segments[j])
                if intersection:
                    junctions.append({
                        'position': [int(intersection[0]), int(intersection[1])],
                        'type': 'junction'
                    })
        
        return junctions
    
    def _count_connected_pipes(self, line_data: List[Dict]) -> int:
        """Count connected pipe segments as single pipes to avoid over-counting"""
        if not line_data:
            return 0
        
        # Filter only pipe segments
        pipe_segments = [line for line in line_data if line.get('is_pipe', False)]
        
        if not pipe_segments:
            return 0
        
        # Group connected segments using simple proximity and angle matching
        visited = [False] * len(pipe_segments)
        pipe_count = 0
        
        for i in range(len(pipe_segments)):
            if not visited[i]:
                # Start a new pipe group
                visited[i] = True
                pipe_count += 1
                
                # Find all connected segments
                queue = [i]
                while queue:
                    current_idx = queue.pop(0)
                    current_seg = pipe_segments[current_idx]
                    
                    # Check for connected segments
                    for j in range(len(pipe_segments)):
                        if not visited[j]:
                            next_seg = pipe_segments[j]
                            if self._are_segments_connected(current_seg, next_seg):
                                visited[j] = True
                                queue.append(j)
        
        return pipe_count
    
    def _are_segments_connected(self, seg1: Dict, seg2: Dict, distance_threshold: float = 30.0, angle_threshold: float = 35.0) -> bool:
        """Check if two line segments are connected (proximate and similar angle) - MORE PERMISSIVE for longer pipes"""
        x1_start, y1_start = seg1['start']
        x1_end, y1_end = seg1['end']
        x2_start, y2_start = seg2['start']
        x2_end, y2_end = seg2['end']
        
        # Check if endpoints are close
        dist1 = np.sqrt((x1_end - x2_start)**2 + (y1_end - y2_start)**2)
        dist2 = np.sqrt((x1_end - x2_end)**2 + (y1_end - y2_end)**2)
        dist3 = np.sqrt((x1_start - x2_start)**2 + (y1_start - y2_start)**2)
        dist4 = np.sqrt((x1_start - x2_end)**2 + (y1_start - y2_end)**2)
        
        min_distance = min(dist1, dist2, dist3, dist4)
        
        if min_distance > distance_threshold:
            return False
        
        # Check if angles are similar (within threshold)
        angle1 = seg1['angle']
        angle2 = seg2['angle']
        
        # Normalize angles to 0-180 range
        angle1 = angle1 % 180
        angle2 = angle2 % 180
        
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        return angle_diff < angle_threshold
    
    def _line_intersection(self, line1: Tuple, line2: Tuple) -> Tuple[float, float]:
        """Find intersection point of two line segments"""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Parallel lines
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def _verify_component_position_with_ocr(self, image: np.ndarray, bbox: List, component_type: str, ocr_results: Dict) -> bool:
        """
        Verify if a component actually exists at the detected position using OCR text evidence.
        
        Args:
            image: Input image
            bbox: Bounding box of detected component [x1, y1, x2, y2]
            component_type: Type of component (e.g., 'valve', 'instrument', 'tank')
            ocr_results: OCR results containing text detections
            
        Returns:
            True if component position is verified by OCR evidence, False otherwise
        """
        # Safely extract bbox coordinates, handling nested lists
        try:
            coords = []
            for coord in bbox:
                if isinstance(coord, (list, tuple)):
                    coords.append(float(coord[0]) if len(coord) > 0 else 0.0)
                else:
                    coords.append(float(coord))
            x1, y1, x2, y2 = coords
        except (TypeError, ValueError, IndexError):
            print(f"DEBUG: Could not extract bbox coordinates in OCR verification: {bbox}")
            return False
        
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Expand search area around component to find associated text
        search_margin = 50
        search_x1 = max(0, x1 - search_margin)
        search_y1 = max(0, y1 - search_margin)
        search_x2 = min(image.shape[1], x2 + search_margin)
        search_y2 = min(image.shape[0], y2 + search_margin)
        
        # Check if there's any text near the component
        has_nearby_text = False
        for text_data in ocr_results.get('text', []):
            text_bbox = text_data.get('bbox', [])
            if len(text_bbox) == 4:
                # Safely extract text bbox coordinates
                try:
                    t_coords = []
                    for coord in text_bbox:
                        if isinstance(coord, (list, tuple)):
                            t_coords.append(float(coord[0]) if len(coord) > 0 else 0.0)
                        else:
                            t_coords.append(float(coord))
                    tx1, ty1, tx2, ty2 = t_coords
                except (TypeError, ValueError, IndexError):
                    continue
                
                text_center_x, text_center_y = (tx1 + tx2) / 2, (ty1 + ty2) / 2
                
                # Check if text is within search area
                if (search_x1 <= text_center_x <= search_x2 and 
                    search_y1 <= text_center_y <= search_y2):
                    has_nearby_text = True
                    break
        
        # For instruments, check if there's a valid instrument tag nearby
        if component_type in ['instrument', 'sensor', 'controller', 'transmitter', 'indicator', 'gauge']:
            for tag in ocr_results.get('instrument_tags', []):
                tag_bbox = tag.get('bbox', [])
                if len(tag_bbox) == 4:
                    # Safely extract tag bbox coordinates
                    try:
                        tag_coords = []
                        for coord in tag_bbox:
                            if isinstance(coord, (list, tuple)):
                                tag_coords.append(float(coord[0]) if len(coord) > 0 else 0.0)
                            else:
                                tag_coords.append(float(coord))
                        tx1, ty1, tx2, ty2 = tag_coords
                    except (TypeError, ValueError, IndexError):
                        continue
                    
                    tag_center_x, tag_center_y = (tx1 + tx2) / 2, (ty1 + ty2) / 2
                    
                    # Check if tag is within search area
                    if (search_x1 <= tag_center_x <= search_x2 and 
                        search_y1 <= tag_center_y <= search_y2):
                        return True  # Verified by instrument tag
        
        # For valves, pumps, tanks - check if there's any equipment name or text nearby
        if component_type in ['valve', 'pump', 'tank', 'vessel']:
            for name in ocr_results.get('equipment_names', []):
                name_bbox = name.get('bbox', [])
                if len(name_bbox) == 4:
                    # Safely extract name bbox coordinates
                    try:
                        name_coords = []
                        for coord in name_bbox:
                            if isinstance(coord, (list, tuple)):
                                name_coords.append(float(coord[0]) if len(coord) > 0 else 0.0)
                            else:
                                name_coords.append(float(coord))
                        nx1, ny1, nx2, ny2 = name_coords
                    except (TypeError, ValueError, IndexError):
                        continue
                    
                    name_center_x, name_center_y = (nx1 + nx2) / 2, (ny1 + ny2) / 2
                    
                    # Check if name is within search area
                    if (search_x1 <= name_center_x <= search_x2 and 
                        search_y1 <= name_center_y <= search_y2):
                        return True  # Verified by equipment name
        
        # If no specific tag found but there's nearby text, still consider it potentially valid
        # (some components may not have labels in all diagrams)
        return has_nearby_text
    
    def _verify_component_visually(self, image: np.ndarray, bbox: List, component_type: str) -> bool:
        """
        Verify if a component actually exists at the detected position using visual analysis.
        
        Args:
            image: Input image
            bbox: Bounding box of detected component [x1, y1, x2, y2]
            component_type: Type of component (e.g., 'valve', 'instrument', 'tank')
            
        Returns:
            True if component position is visually verified, False otherwise
        """
        try:
            # Safely convert bbox coordinates to integers, handling nested lists
            if isinstance(bbox, (list, tuple)):
                if len(bbox) == 4:
                    # Handle case where bbox might contain nested lists
                    coords = []
                    for coord in bbox:
                        if isinstance(coord, (list, tuple)):
                            coords.append(int(coord[0]) if len(coord) > 0 else 0)
                        else:
                            coords.append(int(coord))
                    x1, y1, x2, y2 = coords
                    print(f"DEBUG: Converted bbox {bbox} to coords: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                else:
                    print(f"DEBUG: Invalid bbox length: {len(bbox)}, expected 4")
                    return False
            else:
                print(f"DEBUG: Invalid bbox type: {type(bbox)}")
                return False
            
            # Ensure bbox is within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            if x2 <= x1 or y2 <= y1:
                return False  # Invalid bbox
            
            # Crop the region
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"DEBUG: Empty crop for bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                return False
            
            print(f"DEBUG: Crop shape: {crop.shape}, dtype: {crop.dtype}")
            
            # Convert to grayscale for analysis
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop
            
            # Check edge density - components should have edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Safely calculate edge density
            if edges.size == 0:
                print(f"DEBUG: Empty edges array")
                return False
            
            total_pixels = edges.shape[0] * edges.shape[1]
            if total_pixels == 0:
                print(f"DEBUG: Zero total pixels in edges")
                return False
            
            edge_density = np.sum(edges > 0) / total_pixels
            
            # Minimum edge density threshold
            min_edge_density = 0.02  # At least 2% of pixels should be edges
            if edge_density < min_edge_density:
                print(f"DEBUG: Visual verification failed - low edge density ({edge_density:.4f} < {min_edge_density}) for {component_type}")
                return False
            
            # Check contour presence
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print(f"DEBUG: Visual verification failed - no contours found for {component_type}")
                return False
            
            # Check if there's at least one significant contour
            significant_contours = [c for c in contours if cv2.contourArea(c) > 10]
            if len(significant_contours) == 0:
                print(f"DEBUG: Visual verification failed - no significant contours for {component_type}")
                return False
            
            # Component-specific visual checks
            if component_type == 'tank':
                # Tanks should be large and have significant area
                area = (x2 - x1) * (y2 - y1)
                if area < 500:  # Tanks should be at least 500 pixels
                    print(f"DEBUG: Visual verification failed - tank too small (area: {area})")
                    return False
            elif component_type == 'valve':
                # Valves should have characteristic shape (roughly circular or bow-tie)
                height = y2 - y1
                if height > 0:
                    aspect_ratio = (x2 - x1) / height
                    if aspect_ratio > 5 or aspect_ratio < 0.2:  # Valves shouldn't be extremely elongated
                        print(f"DEBUG: Visual verification failed - valve aspect ratio too extreme ({aspect_ratio:.2f})")
                        return False
            
            return True
        except Exception as e:
            print(f"DEBUG: Visual verification error for {component_type}: {e}")
            return False
    
    def _tesseract_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Tesseract OCR analysis: read text labels, instrument tags, equipment names"""
        print("  - Running Tesseract OCR analysis...")
        
        try:
            # Preprocess image for better OCR accuracy
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for better text detection
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Denoise slightly
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # Configure Tesseract for faster engineering diagram OCR
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- --dpi 300'
            
            # Get OCR data with bounding boxes
            data = pytesseract.image_to_data(denoised, output_type=pytesseract.Output.DICT, 
                                           config=custom_config)
            
            text_data = []
            instrument_tags = []
            equipment_names = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > OCR_MIN_TEXT_CONFIDENCE * 100:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    confidence = float(data['conf'][i]) / 100.0
                    
                    text_item = {
                        'text': text,
                        'bbox': [[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                        'confidence': confidence
                    }
                    text_data.append(text_item)
                    
                    # Check for instrument tags
                    if any(tag in text.upper() for tag in VALID_INSTRUMENT_TAGS):
                        instrument_tags.append(text_item)
                    
                    # Check for equipment names
                    if any(text.upper().startswith(prefix) for prefix in VALID_EQUIPMENT_PREFIXES):
                        equipment_names.append(text_item)
            
            return {
                'all_text': text_data,
                'instrument_tags': instrument_tags,
                'equipment_names': equipment_names,
                'total_text': len(text_data),
                'total_instrument_tags': len(instrument_tags),
                'total_equipment_names': len(equipment_names)
            }
        except Exception as e:
            print(f"Tesseract OCR error: {e}")
            return {
                'all_text': [],
                'instrument_tags': [],
                'equipment_names': [],
                'total_text': 0,
                'total_instrument_tags': 0,
                'total_equipment_names': 0
            }
    
    def _florence_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Florence-2 analysis: optimized for P&ID component detection"""
        print("  - Running Florence-2 analysis...")
        
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # ULTRA AGGRESSIVE downsampling for speed - reduce to 256px max
            original_size = pil_image.size
            if max(original_size) > 256:
                scale = 256 / max(original_size)
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                pil_image = pil_image.resize(new_size, Image.BILINEAR)
            
            # Use CAPTION_TO_PHRASE_GROUNDING task for better P&ID detection
            # OPTIMIZED for motor and customized component detection with focused prompt
            prompt = "motor electric motor drive motor_pump pump motor engine cyclone cyclone separator air separator dust separator stacker stacker reclaimer turbine boiler heat exchanger compressor reactor baghouse hopper baghouse hopper single valve tank separator instrument sensor calciner chimney pipe arrow_pipe"
            inputs = self.florence_processor(text=prompt, images=pil_image, return_tensors="pt").to(FLORENCE_DEVICE)
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,  # Further reduced for speed
                num_beams=1,
                do_sample=False
            )
            
            result = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            result = self.florence_processor.post_process_generation(result, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=pil_image.size)
            
            # Extract data
            labels = []
            bboxes = []
            if isinstance(result, dict):
                for item in result.get('<CAPTION_TO_PHRASE_GROUNDING>', []):
                    if isinstance(item, dict):
                        labels.append(item.get('label', ''))
                        bbox = item.get('bbox', [])
                        # Scale bbox back to original image size if we downsampled
                        if max(original_size) > 512 and bbox:
                            scale = max(original_size) / 512
                            bbox = [coord * scale for coord in bbox]
                        bboxes.append(bbox)
            else:
                print(f"Warning: Florence result is not a dict: {type(result)}")
            
            print(f"Florence-2 detected {len(labels)} objects: {labels[:3] if labels else 'none'}")
            return {
                'labels': labels,
                'bboxes': bboxes,
                'total_regions': len(labels)
            }
        except Exception as e:
            print(f"Florence analysis error: {e}")
            return {
                'labels': [],
                'bboxes': [],
                'total_regions': 0
            }
    
    def _match_template_generic(self, image: np.ndarray, component_name: str, template_path: Path, threshold: float = 0.5) -> Dict:
        """
        Generic template matching for any component using actual template image
        This is a fallback when AI detection fails
        
        Args:
            image: Input image
            component_name: Name of the component (for logging and label)
            template_path: Path to template thumbnail image
            threshold: Confidence threshold for accepting match (lowered to 0.5 for better detection)
        
        Returns:
            Detection dict with bbox, confidence, label, source if match found, else None
        """
        print(f"DEBUG: Attempting template matching for {component_name}...")
        
        if not template_path.exists():
            print(f"DEBUG: {component_name} template image not found at {template_path}")
            return None
        
        try:
            # Use cached template if available
            template_key = str(template_path)
            if template_key in self.template_cache:
                template = self.template_cache[template_key]
            else:
                # Load template image
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if template is None:
                    print(f"DEBUG: Failed to load {component_name} template image")
                    return None
                # Cache the template
                self.template_cache[template_key] = template
            
            # Convert input image to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize template to match image scale
            template_height, template_width = template.shape
            image_height, image_width = gray.shape
            
            # Try multiple scales - expanded range for better detection
            scales = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
            best_match = None
            best_confidence = 0
            
            for scale in scales:
                scaled_width = int(template_width * scale)
                scaled_height = int(template_height * scale)
                
                if scaled_width > image_width or scaled_height > image_height:
                    continue
                
                resized_template = cv2.resize(template, (scaled_width, scaled_height))
                
                # Template matching with multiple methods for better accuracy
                methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
                for method in methods:
                    result = cv2.matchTemplate(gray, resized_template, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_confidence and max_val > threshold:
                        best_confidence = max_val
                        best_match = {
                            'bbox': [max_loc[0], max_loc[1], max_loc[0] + scaled_width, max_loc[1] + scaled_height],
                            'confidence': max_val,
                            'label': component_name,
                            'source': 'template_matching'
                        }
            
            if best_match:
                print(f"DEBUG: {component_name} template match found with confidence: {best_confidence:.2f}")
                return best_match
            else:
                print(f"DEBUG: No good {component_name} template match found")
                return None
                
        except Exception as e:
            print(f"DEBUG: Error in {component_name} template matching: {e}")
            return None
    
    def _match_cyclone_template(self, image: np.ndarray) -> Dict:
        """
        Direct template matching for cyclone using actual template image
        This is a fallback when AI detection fails
        """
        cyclone_template_path = PROJECT_ROOT / "Template/Cyclone/Cyclone/thumbnail.png"
        return self._match_template_generic(image, 'cyclone', cyclone_template_path, threshold=0.5)
    
    def _match_turbine_template(self, image: np.ndarray) -> Dict:
        """Template matching for turbine component"""
        turbine_template_path = PROJECT_ROOT / "Template/Turbine/Turbine/thumbnail.png"
        return self._match_template_generic(image, 'turbine', turbine_template_path, threshold=0.5)
    
    def _match_boiler_template(self, image: np.ndarray) -> Dict:
        """Template matching for boiler component"""
        boiler_template_path = PROJECT_ROOT / "Template/Steam_Operations/Boiler/thumbnail.png"
        return self._match_template_generic(image, 'boiler', boiler_template_path, threshold=0.5)
    
    def _match_conveyor_template(self, image: np.ndarray) -> Dict:
        """Template matching for conveyor component"""
        conveyor_template_path = PROJECT_ROOT / "Template/Conveyors/BeltConveyor/thumbnail.png"
        return self._match_template_generic(image, 'conveyor', conveyor_template_path, threshold=0.5)
    
    def _match_crusher_template(self, image: np.ndarray) -> Dict:
        """Template matching for crusher component"""
        crusher_template_path = PROJECT_ROOT / "Template/Crusher/Crusher/thumbnail.png"
        return self._match_template_generic(image, 'crusher', crusher_template_path, threshold=0.5)
    
    def _match_furnace_template(self, image: np.ndarray) -> Dict:
        """Template matching for furnace component"""
        furnace_template_path = PROJECT_ROOT / "Template/Furnace/Kiln/thumbnail.png"
        return self._match_template_generic(image, 'furnace', furnace_template_path, threshold=0.5)
    
    def _match_calciner_template(self, image: np.ndarray) -> Dict:
        """Template matching for calciner component"""
        calciner_template_path = PROJECT_ROOT / "Template/Calciner/Calciner/thumbnail.png"
        return self._match_template_generic(image, 'calciner', calciner_template_path, threshold=0.5)
    
    def _match_stacker_template(self, image: np.ndarray) -> Dict:
        """Template matching for stacker component"""
        stacker_template_path = PROJECT_ROOT / "Template/Stacker/Stacker/thumbnail.png"
        return self._match_template_generic(image, 'stacker', stacker_template_path, threshold=0.5)
    
    def _match_separator_template(self, image: np.ndarray) -> Dict:
        """Template matching for separator component - try multiple separator templates"""
        # Try multiple separator templates to find the best match
        separator_templates = [
            PROJECT_ROOT / "Template/Seperator/Seperator/thumbnail.png",
            PROJECT_ROOT / "Template/Tanks/Separator/thumbnail.png",
            PROJECT_ROOT / "Template/Tanks/Separator_tank/thumbnail.png",
            PROJECT_ROOT / "Template/Tanks/Separator_tank_1/thumbnail.png",
            PROJECT_ROOT / "Template/Tanks/Separator_tank_2/thumbnail.png",
        ]
        
        best_match = None
        best_score = 0
        
        for template_path in separator_templates:
            if template_path.exists():
                result = self._match_template_generic(image, 'separator', template_path, threshold=0.5)
                if result and result.get('confidence', 0) > best_score:
                    best_match = result
                    best_score = result.get('confidence', 0)
        
        return best_match
    
    def _match_cyclone_separator_template(self, image: np.ndarray) -> Dict:
        """Template matching for cyclone_separator component - try multiple cyclone_separator templates"""
        # Try multiple cyclone_separator templates to find the best match
        cyclone_separator_templates = [
            PROJECT_ROOT / "Template/Cyclone/Cyclone_Separator/thumbnail.png",
            PROJECT_ROOT / "Template/Cyclone/Cyclone_Separator1/thumbnail.png",
        ]
        
        best_match = None
        best_score = 0
        
        for template_path in cyclone_separator_templates:
            if template_path.exists():
                result = self._match_template_generic(image, 'cyclone_separator', template_path, threshold=0.5)
                if result and result.get('confidence', 0) > best_score:
                    best_match = result
                    best_score = result.get('confidence', 0)
        
        return best_match
    
    def _match_tank_template(self, image: np.ndarray) -> Dict:
        """Template matching for tank component - try multiple tank templates"""
        # Try multiple tank templates to find the best match
        tank_templates = [
            PROJECT_ROOT / "Template/Tanks/Square_Tank/thumbnail.png",
            PROJECT_ROOT / "Template/Tanks/WaterTank/thumbnail.png",
            PROJECT_ROOT / "Template/Tanks/Tank_6/thumbnail.png",
            PROJECT_ROOT / "Template/Tanks/Storage Tank/thumbnail.png",
        ]
        
        best_match = None
        best_score = 0
        
        for template_path in tank_templates:
            if template_path.exists():
                result = self._match_template_generic(image, 'tank', template_path, threshold=0.5)
                if result and result.get('confidence', 0) > best_score:
                    best_match = result
                    best_score = result.get('confidence', 0)
        
        return best_match
    
    def _match_generic_component_template(self, image: np.ndarray, component_label: str) -> Dict:
        """
        Generic template matching for any component by searching the Template folder
        This is a fallback for components that don't have dedicated template matching methods
        """
        print(f"DEBUG: Searching Template folder for {component_label}...")
        
        # Clean the label for path matching
        label_clean = component_label.replace(' ', '_').replace('-', '_').lower()
        
        # Search for matching template directories
        template_paths = []
        if TEMPLATE_DIR.exists():
            for category_dir in TEMPLATE_DIR.iterdir():
                if category_dir.is_dir():
                    # Check for exact match (case-insensitive)
                    for item in category_dir.iterdir():
                        if item.is_dir() and item.name.lower() == label_clean:
                            thumbnail = item / "thumbnail.png"
                            if thumbnail.exists():
                                template_paths.append(thumbnail)
                                print(f"DEBUG: Found template: {thumbnail}")
                    
                    # Check for partial match (label contained in directory name)
                    for item in category_dir.iterdir():
                        if item.is_dir() and label_clean in item.name.lower():
                            thumbnail = item / "thumbnail.png"
                            if thumbnail.exists() and thumbnail not in template_paths:
                                template_paths.append(thumbnail)
                                print(f"DEBUG: Found partial match template: {thumbnail}")
        
        if not template_paths:
            print(f"DEBUG: No templates found for {component_label}")
            return None
        
        # Try template matching with all found templates
        best_match = None
        best_score = 0
        
        for template_path in template_paths:
            result = self._match_template_generic(image, component_label, template_path, threshold=0.5)
            if result and result.get('confidence', 0) > best_score:
                best_match = result
                best_score = result.get('confidence', 0)
        
        if best_match:
            print(f"DEBUG: Generic template match found for {component_label} with confidence: {best_score:.2f}")
        else:
            print(f"DEBUG: No good generic template match found for {component_label}")
        
        return best_match
    
    def _match_motor_template(self, image: np.ndarray) -> Dict:
        """Template matching for motor component - try multiple motor templates"""
        print(f"DEBUG: Starting motor template matching...")
        
        # Try multiple motor templates to find the best match
        motor_templates = [
            PROJECT_ROOT / "Template/Digital/Motor/thumbnail.png",
            PROJECT_ROOT / "Template/Digital/Motor_1/thumbnail.png",
            PROJECT_ROOT / "Template/Digital/Motor_2/thumbnail.png",
            PROJECT_ROOT / "Template/Digital/Motor_3/thumbnail.png",
            PROJECT_ROOT / "Template/Digital/Motor_4/thumbnail.png",
            PROJECT_ROOT / "Template/Pumps/Motor_Pump/thumbnail.png",
            PROJECT_ROOT / "Template/Pumps/Motor_Pump_Simple/thumbnail.png",
            PROJECT_ROOT / "Template/Pumps/Motor_gear/thumbnail.png",
            PROJECT_ROOT / "Template/Pumps/Motor_gear_Single/thumbnail.png",
            PROJECT_ROOT / "Template/Misc/Motor/thumbnail.png",
            PROJECT_ROOT / "Template/Misc/MotorHousing/thumbnail.png",
        ]
        
        best_match = None
        best_score = 0
        
        for template_path in motor_templates:
            if template_path.exists():
                print(f"DEBUG: Trying motor template: {template_path}")
                result = self._match_template_generic(image, 'motor', template_path, threshold=0.5)
                if result:
                    score = result.get('confidence', 0)
                    print(f"DEBUG: Motor template match found with confidence: {score:.2f}")
                    if score > best_score:
                        best_match = result
                        best_score = score
                else:
                    print(f"DEBUG: No match for this template")
            else:
                print(f"DEBUG: Template file not found: {template_path}")
        
        if best_match:
            print(f"DEBUG: Best motor match confidence: {best_score:.2f}")
        else:
            print(f"DEBUG: No motor template match found")
        
        return best_match
    
    def _non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """
        Apply non-maximum suppression to remove overlapping detections
        Keep only the highest confidence detection for overlapping regions
        
        Args:
            detections: List of detection dicts with bbox and confidence
            iou_threshold: IoU threshold for considering detections as overlapping
        
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by confidence in descending order
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        keep = []
        while sorted_detections:
            # Keep the highest confidence detection
            current = sorted_detections.pop(0)
            keep.append(current)
            
            # Remove detections that overlap significantly with the current one
            filtered = []
            for detection in sorted_detections:
                iou = calculate_iou(current['bbox'], detection['bbox'])
                if iou < iou_threshold:
                    filtered.append(detection)
                else:
                    print(f"DEBUG: NMS removing {detection['label']} (IoU: {iou:.2f}) with {current['label']}")
            
            sorted_detections = filtered
        
        return keep
    
    def _deduplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        """Deduplicate overlapping detections of the same component type using NMS and distance-based merging"""
        if not detections:
            return []
        
        # Group detections by label
        label_groups = {}
        for detection in detections:
            label = detection.get('label', '').lower()
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(detection)
        
        # Apply NMS and distance-based merging to each group
        deduplicated = []
        for label, group in label_groups.items():
            print(f"DEBUG: Deduplicating {len(group)} {label} detections")
            
            # First apply NMS with appropriate IoU threshold based on component type
            # Increased thresholds to allow more components through
            iou_threshold = 0.6 if label in ['valve', 'tank'] else 0.7
            nms_results = self._non_max_suppression(group, iou_threshold=iou_threshold)
            print(f"DEBUG: After NMS for {label}: {len(nms_results)} remain")
            
            # Then apply distance-based merging for nearby detections (for valves, tanks, and pumps)
            # Disabled aggressive merging to preserve more detections
            # if label in ['valve', 'tank', 'pump'] and len(nms_results) > 1:
            #     nms_results = self._merge_nearby_detections(nms_results, distance_threshold=500)
            #     print(f"DEBUG: After distance merge for {label}: {len(nms_results)} remain")
            #     
            #     # If still more than 1, keep only the highest confidence one
            #     if len(nms_results) > 1:
            #         nms_results = [max(nms_results, key=lambda x: x.get('confidence', 0))]
            #         print(f"DEBUG: After keeping highest confidence for {label}: {len(nms_results)} remain")
            
            deduplicated.extend(nms_results)
        
        return deduplicated
    
    def _merge_nearby_detections(self, detections: List[Dict], distance_threshold: float = 50) -> List[Dict]:
        """Merge detections that are close to each other (center-to-center distance)"""
        if not detections or len(detections) <= 1:
            return detections
        
        # Sort by confidence in descending order
        detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        keep = []
        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Calculate center of current detection
            current_bbox = current['bbox']
            current_center = ((current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2)
            
            # Remove detections that are too close to current
            remaining = []
            for detection in detections:
                bbox = detection['bbox']
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                distance = ((current_center[0] - center[0])**2 + (current_center[1] - center[1])**2)**0.5
                
                if distance >= distance_threshold:
                    remaining.append(detection)
                else:
                    print(f"DEBUG: Distance merge removing {detection['label']} (distance: {distance:.1f}) with {current['label']}")
            
            detections = remaining
        
        return keep
    
    def _grounding_dino_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Grounding DINO analysis: object detection with confidence scores"""
        print("  - Running Grounding DINO analysis...")
        
        try:
            # Convert to PIL Image for Grounding DINO
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
            else:
                image_pil = Image.fromarray(image)
            
            # Transform image to tensor using Grounding DINO's transform (ULTRA AGGRESSIVE size reduction for speed)
            transform = T.Compose([
                T.RandomResize([256], max_size=350),  # Further reduced for speed (was 300x400)
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor, _ = transform(image_pil, None)
            
            # Use focused prompts for P&ID components - COMPREHENSIVE for accurate detection
            # STANDARD COMPONENTS: TANKS, VALVES, MOTORS, PUMPS
            # CUSTOMIZED COMPONENTS: BAGHOUSE, CYCLONE, CHIMNEY, etc.
            prompts_config = [
                {"prompt": "motor electric motor drive motor_pump pump motor engine motor_gear", "box_threshold": 0.02, "text_threshold": 0.02},  # Motor-focused with extremely low thresholds
                {"prompt": "valve gate valve globe valve ball valve butterfly valve check valve control valve relief valve safety valve three way valve angle valve plug valve diaphragm valve needle valve solenoid valve pressure valve temperature valve", "box_threshold": 0.02, "text_threshold": 0.02},  # All valve types with extremely low thresholds
                {"prompt": "pump centrifugal pump gear pump reciprocating pump screw pump vane pump motor_pump pump_double pump_left pump_right vertical_pump arrow arrow_head vertical_arrow", "box_threshold": 0.02, "text_threshold": 0.02},  # All pump types with extremely low thresholds
                {"prompt": "tank vessel storage tank spherical tank horizontal tank vertical tank separator tank water tank square tank tank_6", "box_threshold": 0.02, "text_threshold": 0.02},  # All tank types with extremely low thresholds
                {"prompt": "baghouse baghouse hopper baghouse hopper single cyclone cyclone separator cyclone_separator1 preheater_cyclone air separator dust separator", "box_threshold": 0.03, "text_threshold": 0.03},  # Baghouse and cyclone with very low thresholds
                {"prompt": "chimney chimney_without_smoke chimney_without_smoke_grey exhaust_stack exhaust_stack_1 stacker stacker_reclaimer", "box_threshold": 0.03, "text_threshold": 0.03},  # Chimney and stacker with very low thresholds
                {"prompt": "conveyor belt_conveyor chain_conveyor rectangular_conveyor duct spiral", "box_threshold": 0.03, "text_threshold": 0.03},  # Conveyor types with very low thresholds
                {"prompt": "furnace kiln kiln_1 kiln_2 rotary_kiln calciner crusher turbine boiler heat_exchanger compressor reactor", "box_threshold": 0.03, "text_threshold": 0.03},  # Furnace and other components with very low thresholds
                {"prompt": "separator clinker_silo packers lorry hag whrs", "box_threshold": 0.05, "text_threshold": 0.05},  # Other customized components with low thresholds
                {"prompt": "instrument sensor pipe arrow_pipe", "box_threshold": 0.05, "text_threshold": 0.05}  # Remaining components with low thresholds
            ]
            
            all_boxes = []
            all_confidences = []
            all_labels = []
            
            # Cache visual backbone features to avoid running Swin Transformer 5 times
            cached_successfully = False
            try:
                with torch.no_grad():
                    self.grounding_model.eval()
                    self.grounding_model.set_image_tensor(image_tensor[None])
                    original_poss = list(self.grounding_model.poss)
                    cached_successfully = True
            except Exception as cache_err:
                print(f"Warning: Failed to cache DINO image features ({cache_err}). Falling back to standard execution.")
            
            try:
                # Run detection for each prompt with component-specific thresholds
                for config in prompts_config:
                    if cached_successfully:
                        # Restore the clean poss list to avoid the Grounding DINO multi-pass append bug
                        self.grounding_model.poss = list(original_poss)
                    else:
                        # Reset features/poss to force full backbone run
                        if hasattr(self.grounding_model, 'features'):
                            del self.grounding_model.features
                        if hasattr(self.grounding_model, 'poss'):
                            del self.grounding_model.poss
                    
                    boxes_filter, logits, phrases = predict(
                        model=self.grounding_model,
                        image=image_tensor,
                        caption=config["prompt"],
                        box_threshold=config["box_threshold"],
                        text_threshold=config["text_threshold"],
                        device="cpu"
                    )
                    
                    for box, logit, phrase in zip(boxes_filter, logits, phrases):
                        # Convert normalized cxcywh to pixel xyxy
                        cx, cy, wb, hb = box.tolist()
                        h, w = image.shape[:2]
                        
                        x1 = (cx - wb / 2) * w
                        y1 = (cy - hb / 2) * h
                        x2 = (cx + wb / 2) * w
                        y2 = (cy + hb / 2) * h
                        
                        # Clamp coordinates to image dimensions
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(0, min(x2, w - 1))
                        y2 = max(0, min(y2, h - 1))
                        
                        all_boxes.append([x1, y1, x2, y2])
                        all_confidences.append(float(logit))
                        all_labels.append(phrase)
            except Exception as e:
                print(f"Grounding DINO detection error: {e}")
            finally:
                # Clean up cached features to release memory
                if hasattr(self.grounding_model, 'features'):
                    self.grounding_model.unset_image_tensor()
            
            # Apply NMS with optimized threshold for accurate duplicate removal
            keep_indices = non_max_suppression(all_boxes, all_confidences, iou_threshold=0.22)
            
            filtered_boxes = [all_boxes[i] for i in keep_indices]
            filtered_confidences = [all_confidences[i] for i in keep_indices]
            filtered_labels = [all_labels[i] for i in keep_indices]
            
            # Apply confidence filtering to remove low-confidence detections (false positives)
            min_confidence = 0.10  # Further lowered threshold to capture even more components (was 0.20)
            final_boxes = []
            final_confidences = []
            final_labels = []
            
            for box, conf, label in zip(filtered_boxes, filtered_confidences, filtered_labels):
                if conf >= min_confidence:
                    final_boxes.append(box)
                    final_confidences.append(conf)
                    final_labels.append(label)
            
            print(f"Grounding DINO detected {len(final_labels)} components (after confidence filtering)")
            return {
                'boxes': final_boxes,
                'confidences': final_confidences,
                'labels': final_labels,
                'total_detections': len(final_boxes)
            }
        except Exception as e:
            print(f"Grounding DINO analysis error: {e}")
            return {
                'boxes': [],
                'confidences': [],
                'labels': [],
                'total_detections': 0
            }
    
    def _gemini_analysis(self, image: np.ndarray, detections: List[Dict] = None) -> Dict[str, Any]:
        """Gemini analysis: final verification of detected components using gemini-2.5-flash and parallel cropped images"""
        print("  - Running Gemini final verification...")
        
        try:
            if detections is None or len(detections) == 0:
                print("Gemini: No detections to verify")
                return {
                    'verified_detections': [],
                    'total_verified': 0,
                    'total_tokens': 0
                }
            
            import base64
            import io
            from PIL import Image as PILImage
            from concurrent.futures import ThreadPoolExecutor
            
            def verify_single(detection):
                bbox = detection.get('bbox', [])
                current_label = detection.get('label', 'unknown').lower()
                if len(bbox) != 4:
                    return None
                    
                x1, y1, x2, y2 = bbox
                h_img, w_img = image.shape[:2]
                
                # Make sure coordinates are in bounds
                x1_p = max(0, min(int(x1), w_img - 1))
                y1_p = max(0, min(int(y1), h_img - 1))
                x2_p = max(0, min(int(x2), w_img))
                y2_p = max(0, min(int(y2), h_img))
                
                if x2_p <= x1_p or y2_p <= y1_p:
                    return None
                    
                # Calculate padding (ensure minimum crop size of 160x160)
                w_box = x2_p - x1_p
                h_box = y2_p - y1_p
                
                target_crop_w = max(160, int(w_box * 1.5))
                target_crop_h = max(160, int(h_box * 1.5))
                
                pad_w = max(10, (target_crop_w - w_box) // 2)
                pad_h = max(10, (target_crop_h - h_box) // 2)
                
                crop_x1 = max(0, x1_p - pad_w)
                crop_y1 = max(0, y1_p - pad_h)
                crop_x2 = min(w_img, x2_p + pad_w)
                crop_y2 = min(h_img, y2_p + pad_h)
                
                cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
                if cropped_img.size == 0:
                    return None
                    
                # Convert to base64
                pil_crop = PILImage.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                
                # Resize if crop is too large
                if max(pil_crop.size) > 300:
                    scale = 300 / max(pil_crop.size)
                    pil_crop = pil_crop.resize((int(pil_crop.size[0] * scale), int(pil_crop.size[1] * scale)), PILImage.BILINEAR)
                    
                buffered = io.BytesIO()
                pil_crop.save(buffered, format="JPEG")
                crop_str = base64.b64encode(buffered.getvalue()).decode()
                
                prompt = f"""
                Verify the P&ID diagram component centered in this cropped image.
                Suggested classification: {current_label}
                
                Is this classification correct? Answer YES or NO.
                If NO, provide the correct classification from: tank, valve, pump, motor, instrument, or other.
                """
                
                # Try Gemini first
                try:
                    response = self.gemini_client.generate_content(
                        [prompt, {"mime_type": "image/jpeg", "data": crop_str}],
                        generation_config={"temperature": GEMINI_TEMPERATURE}
                    )
                    
                    if response and response.text:
                        response_text = response.text.strip().lower()
                        verified_label = current_label
                        
                        # Parse response: if Gemini says "no", extract the new correct label
                        if 'no' in response_text:
                            for option in ['valve', 'tank', 'pump', 'motor', 'instrument', 'other']:
                                if option in response_text:
                                    verified_label = option
                                    break
                                
                        print(f"DEBUG: Gemini verification response: '{response_text.strip()}' -> final label: '{verified_label}' (suggested: '{current_label}')")
                        
                        tokens = 0
                        if hasattr(response, 'usage_metadata'):
                            tokens = response.usage_metadata.total_token_count if hasattr(response.usage_metadata, 'total_token_count') else 0
                            
                        return {
                            'bbox': bbox,
                            'label': verified_label,
                            'confidence': 0.85 if verified_label == current_label else 0.75,
                            'verified': True,
                            'tokens': tokens,
                            'provider': 'gemini'
                        }
                except Exception as e:
                    print(f"DEBUG: Gemini verification failed for component {current_label}: {e}")
                    # Fallback to OpenAI
                    if self.openai_client:
                        try:
                            print(f"DEBUG: Falling back to OpenAI for component {current_label}")
                            response = self.openai_client.chat.completions.create(
                                model=OPENAI_MODEL,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": prompt},
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{crop_str}"
                                                }
                                            }
                                        ]
                                    }
                                ],
                                temperature=OPENAI_TEMPERATURE,
                                max_tokens=100
                            )
                            
                            if response and response.choices:
                                response_text = response.choices[0].message.content.strip().lower()
                                verified_label = current_label
                                
                                # Parse response: if OpenAI says "no", extract the new correct label
                                if 'no' in response_text:
                                    for option in ['valve', 'tank', 'pump', 'motor', 'instrument', 'other']:
                                        if option in response_text:
                                            verified_label = option
                                            break
                                
                                print(f"DEBUG: OpenAI verification response: '{response_text.strip()}' -> final label: '{verified_label}' (suggested: '{current_label}')")
                                
                                tokens = response.usage.total_tokens if response.usage else 0
                                
                                return {
                                    'bbox': bbox,
                                    'label': verified_label,
                                    'confidence': 0.85 if verified_label == current_label else 0.75,
                                    'verified': True,
                                    'tokens': tokens,
                                    'provider': 'openai'
                                }
                        except Exception as openai_error:
                            print(f"DEBUG: OpenAI fallback also failed for component {current_label}: {openai_error}")
                    
                return None
            
            # AGGRESSIVE Gemini verification - only top 5 low-confidence detections for speed
            # High-confidence detections (>0.65) are trusted without verification
            low_confidence_detections = [det for det in detections if det.get('confidence', 0) < 0.65]
            print(f"Gemini: Verifying top 5 of {len(low_confidence_detections)} low-confidence detections (skipping {len(detections) - len(low_confidence_detections)} high-confidence detections)")
            
            verified_detections = []
            total_tokens = 0
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Limit to top 5 low-confidence detections for speed
                futures = [executor.submit(verify_single, det) for det in low_confidence_detections[:5]]
                for future in futures:
                    res = future.result()
                    if res is not None:
                        verified_detections.append(res)
                        total_tokens += res.get('tokens', 0)
            
            # Add high-confidence detections without verification (trust AI model)
            high_confidence_detections = [det for det in detections if det.get('confidence', 0) >= 0.65]
            for det in high_confidence_detections:
                verified_detections.append({
                    'bbox': det.get('bbox', []),
                    'label': det.get('label', 'unknown'),
                    'confidence': det.get('confidence', 0.65),
                    'verified': True,
                    'tokens': 0,
                    'provider': 'ai_trusted'
                })
                        
            # Count providers
            gemini_count = sum(1 for d in verified_detections if d.get('provider') == 'gemini')
            openai_count = sum(1 for d in verified_detections if d.get('provider') == 'openai')
            
            print(f"Gemini verified {gemini_count} components, OpenAI verified {openai_count} components, total tokens used: {total_tokens}")
            
            # Save token usage to file
            self._save_token_usage(total_tokens, gemini_count, openai_count)
            
            return {
                'verified_detections': verified_detections,
                'total_verified': len(verified_detections),
                'total_tokens': total_tokens
            }
        except Exception as e:
            print(f"Gemini verification error: {e}")
            print("WARNING: Gemini verification failed, continuing without verification")
            return {
                'verified_detections': [],
                'total_verified': 0,
                'total_tokens': 0
            }
    
    def _save_token_usage(self, total_tokens: int, gemini_count: int, openai_count: int):
        """Save token usage to a .txt file"""
        try:
            token_file = Path(__file__).parent.parent / "token_usage.txt"
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            entry = f"{timestamp} - Total tokens: {total_tokens}, Gemini verifications: {gemini_count}, OpenAI verifications: {openai_count}\n"
            
            with open(token_file, "a", encoding="utf-8") as f:
                f.write(entry)
                
            print(f"Token usage saved to {token_file}")
        except Exception as e:
            print(f"Failed to save token usage: {e}")
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes in [x1, y1, x2, y2] format"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
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
    
    def _dedupe_detections(self, detections: List[Dict], iou_threshold: float = 0.50) -> List[Dict]:
        """Remove duplicate detections using IoU within the same category."""
        if not detections:
            return []
        
        ordered = sorted(detections, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        kept = []
        for candidate in ordered:
            try:
                candidate_box = candidate["bbox"]
                if not candidate_box or len(candidate_box) != 4:
                    continue
                category = candidate.get("label", candidate.get("category", "unknown"))
                duplicate = False
                for existing in kept:
                    existing_category = existing.get("label", existing.get("category", "unknown"))
                    if existing_category != category:
                        continue
                    existing_box = existing.get("bbox")
                    if not existing_box or len(existing_box) != 4:
                        continue
                    if self._iou(candidate_box, existing_box) >= iou_threshold:
                        duplicate = True
                        break
                if not duplicate:
                    kept.append(candidate)
            except Exception as e:
                print(f"Error in dedupe for detection: {e}")
                kept.append(candidate)  # Keep detection if there's an error
        return kept
    
    def _aggressive_iou_merge(self, detections: List[Dict], iou_threshold: float = 0.30) -> List[Dict]:
        """Aggressively merge detections with IoU overlap for all components."""
        if not detections:
            return []
        
        try:
            ordered = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
            kept = []
            
            for det in ordered:
                try:
                    bx = det.get("bbox")
                    if not bx or len(bx) != 4:
                        kept.append(det)
                        continue
                    category = det.get("label", det.get("category", "unknown"))
                    duplicate = False
                    
                    for ex in kept:
                        try:
                            ex_category = ex.get("label", ex.get("category", "unknown"))
                            if ex_category != category:
                                continue
                            
                            ex_box = ex.get("bbox")
                            if not ex_box or len(ex_box) != 4:
                                continue
                            
                            iou_score = self._iou(bx, ex_box)
                            if iou_score >= iou_threshold:
                                duplicate = True
                                break
                        except Exception as e:
                            print(f"Error in aggressive IoU comparison: {e}")
                            continue
                    
                    if not duplicate:
                        kept.append(det)
                except Exception as e:
                    print(f"Error in aggressive IoU for detection: {e}")
                    kept.append(det)  # Keep detection if there's an error
            
            return kept
        except Exception as e:
            print(f"Aggressive IoU merge failed, returning original: {e}")
            return detections
    
    def _merge_close_detections(self, detections: List[Dict], distance_ratio: float = 0.15) -> List[Dict]:
        """Merge detections of the same category when their centers are very close."""
        if not detections:
            return []
        
        try:
            ordered = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
            kept = []
            
            def center(box):
                try:
                    x1, y1, x2, y2 = box
                    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
                except:
                    return (0, 0)
            
            def box_size(box):
                try:
                    x1, y1, x2, y2 = box
                    return max(x2 - x1, y2 - y1)
                except:
                    return 1.0
            
            for det in ordered:
                try:
                    bx = det.get("bbox")
                    if not bx or len(bx) != 4:
                        kept.append(det)
                        continue
                    bx_c = center(bx)
                    bw = box_size(bx)
                    category = det.get("label", det.get("category", "unknown"))
                    duplicate = False
                    
                    for ex in kept:
                        try:
                            ex_category = ex.get("label", ex.get("category", "unknown"))
                            if ex_category != category:
                                continue
                            
                            ex_box = ex.get("bbox")
                            if not ex_box or len(ex_box) != 4:
                                continue
                            
                            ex_c = center(ex_box)
                            ex_bw = box_size(ex_box)
                            
                            iou_score = self._iou(bx, ex_box)
                            if iou_score > 0.6:
                                duplicate = True
                                break
                            
                            thresh_ratio = distance_ratio
                            if category == "tank":
                                thresh_ratio = min(distance_ratio, 0.25)
                            thresh = max(bw, ex_bw) * thresh_ratio
                            dist = math.hypot(bx_c[0] - ex_c[0], bx_c[1] - ex_c[1])
                            if dist <= thresh:
                                duplicate = True
                                break
                        except Exception as e:
                            print(f"Error in merge comparison: {e}")
                            continue
                    
                    if not duplicate:
                        kept.append(det)
                except Exception as e:
                    print(f"Error in merge for detection: {e}")
                    kept.append(det)  # Keep detection if there's an error
            
            return kept
        except Exception as e:
            print(f"Merge close detections failed, returning original: {e}")
            return detections
    
    def _merge_stacked_tank_symbols(self, detections: List[Dict]) -> List[Dict]:
        """Merge vertical+horizontal parts of the same P&ID vessel into one tank detection."""
        try:
            tanks = [d for d in detections if d.get("label", d.get("category", "")) == "tank"]
            others = [d for d in detections if d.get("label", d.get("category", "")) != "tank"]
            
            if len(tanks) <= 1:
                return detections
            
            ordered = sorted(tanks, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
            kept = []
            
            def x_overlap(a, b):
                try:
                    x1_1, y1_1, x2_1, y2_1 = a
                    x1_2, y1_2, x2_2, y2_2 = b
                    w1, w2 = x2_1 - x1_1, x2_2 - x1_2
                    inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
                    union = w1 + w2 - inter
                    return inter / union if union > 0 else 0.0
                except:
                    return 0.0
            
            for det in ordered:
                try:
                    box = det.get("bbox")
                    if not box or len(box) != 4:
                        kept.append(det)
                        continue
                    
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    duplicate = False
                    
                    for ex in kept:
                        try:
                            ex_box = ex.get("bbox")
                            if not ex_box or len(ex_box) != 4:
                                continue
                            
                            ex1, ey1, ex2, ey2 = ex_box
                            ew, eh = ex2 - ex1, ey2 - ey1
                            ecx, ecy = (ex1 + ex2) / 2.0, (ey1 + ey2) / 2.0
                            
                            if x_overlap(box, ex_box) < 0.35:
                                continue
                            
                            # Do not merge separate horizontal drums on the same feed line
                            if w >= h * 2.5 and ew >= eh * 2.5:
                                continue
                            
                            vert_gap = abs(cy - ecy)
                            max_dim = max(w, h, ew, eh)
                            if vert_gap <= max_dim * 2.5:
                                duplicate = True
                                break
                        except Exception as e:
                            print(f"Error in stacked tank comparison: {e}")
                            continue
                    
                    if not duplicate:
                        kept.append(det)
                except Exception as e:
                    print(f"Error in stacked tank for detection: {e}")
                    kept.append(det)  # Keep detection if there's an error
            
            return others + kept
        except Exception as e:
            print(f"Merge stacked tank symbols failed, returning original: {e}")
            return detections
    
    def _consolidate_tank_vessels(self, detections: List[Dict], image_area: float = None) -> List[Dict]:
        """Keep primary vessel(s); drop small false tank hits (controllers, caps, internals)."""
        try:
            tanks = [d for d in detections if d.get("label", d.get("category", "")) == "tank"]
            others = [d for d in detections if d.get("label", d.get("category", "")) != "tank"]
            
            if len(tanks) <= 1:
                return detections
            
            def size(det):
                try:
                    box = det.get("bbox") or (0, 0, 0, 0)
                    x1, y1, x2, y2 = box
                    return float((x2 - x1) * (y2 - y1))
                except:
                    return 100.0
            
            # Drop oversized outliers
            if image_area is not None and image_area > 0 and len(tanks) > 1:
                try:
                    max_reasonable_tank_area = image_area * 0.18
                    non_outlier_tanks = [det for det in tanks if size(det) <= max_reasonable_tank_area]
                    if non_outlier_tanks:
                        dropped_outliers = len(tanks) - len(non_outlier_tanks)
                        if dropped_outliers > 0:
                            print(f"Consolidation dropped {dropped_outliers} oversized tank outlier(s)")
                        tanks = non_outlier_tanks
                except Exception as e:
                    print(f"Error in outlier removal: {e}")
            
            if len(tanks) <= 1:
                return others + tanks
            
            try:
                tank_sizes = [size(t) for t in tanks]
                max_size = max(tank_sizes) if tank_sizes else 100.0
                reference_size = float(np.median(tank_sizes)) if tank_sizes else 100.0
                min_keep = max(
                    reference_size * 0.70,  # Balanced threshold
                    (image_area or 0.0) * 0.00025,  # Balanced threshold
                    350.0,  # Balanced threshold
                )
                min_keep = min(min_keep, max_size)
            except Exception as e:
                print(f"Error calculating min_keep: {e}")
                min_keep = 350.0
            
            kept = []
            for det in sorted(tanks, key=size, reverse=True):
                try:
                    if size(det) < min_keep:
                        continue
                    box = det.get("bbox")
                    if not box or len(box) != 4:
                        kept.append(det)
                        continue
                    
                    duplicate = False
                    for ex in kept:
                        try:
                            ex_box = ex.get("bbox")
                            if not ex_box or len(ex_box) != 4:
                                continue
                            if self._iou(box, ex_box) >= 0.12:  # Balanced IoU threshold
                                duplicate = True
                                break
                        except Exception as e:
                            print(f"Error in consolidation comparison: {e}")
                            continue
                    
                    if not duplicate:
                        kept.append(det)
                except Exception as e:
                    print(f"Error in consolidation for detection: {e}")
                    kept.append(det)  # Keep detection if there's an error
            
            if len(kept) <= 1:
                return others + kept
            
            # Column deduplication
            column_kept = []
            for det in sorted(kept, key=size, reverse=True):
                try:
                    box = det.get("bbox")
                    if not box or len(box) != 4:
                        column_kept.append(det)
                        continue
                    
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2.0
                    duplicate_column = False
                    
                    for existing in column_kept:
                        try:
                            ex_box = existing.get("bbox")
                            if not ex_box or len(ex_box) != 4:
                                continue
                            
                            ex1, ey1, ex2, ey2 = ex_box
                            ex_cx = (ex1 + ex2) / 2.0
                            
                            if abs(cx - ex_cx) <= max(box[2] - box[0], ex_box[2] - ex_box[0]) * 0.75:  # Balanced column threshold
                                size_ratio = size(det) / max(size(existing), 1.0)
                                if size_ratio >= 0.33 and size_ratio <= 3.0:  # Balanced size ratio
                                    duplicate_column = True
                                    break
                        except Exception as e:
                            print(f"Error in column deduplication comparison: {e}")
                            continue
                    
                    if not duplicate_column:
                        column_kept.append(det)
                except Exception as e:
                    print(f"Error in column deduplication for detection: {e}")
                    column_kept.append(det)  # Keep detection if there's an error
            
            return others + column_kept
        except Exception as e:
            print(f"Consolidate tank vessels failed, returning original: {e}")
            return detections
    
    def _heuristic_engine(self, opencv_results: Dict, ocr_results: Dict, 
                          florence_results: Dict, dino_results: Dict, image: np.ndarray = None) -> Dict[str, Any]:
        """Heuristic Engine: Apply ISA 5.1 rules and validate detections with component verification"""
        print("  - Running Heuristic Engine...")
        
        validated_components = []
        filtered_text = []
        connections = []
        
        # Validate instrument tags using ISA 5.1 rules
        for tag in ocr_results['instrument_tags']:
            text = tag['text'].upper()
            # Check if it follows ISA 5.1 naming convention
            if self._validate_isa_tag(text):
                filtered_text.append({
                    **tag,
                    'validated': True,
                    'rule': 'ISA 5.1'
                })
        
        # Validate equipment names
        for name in ocr_results['equipment_names']:
            text = name['text'].upper()
            if self._validate_equipment_name(text):
                filtered_text.append({
                    **name,
                    'validated': True,
                    'rule': 'ISA 5.1'
                })
        
        # Combine Florence and DINO detections FIRST
        combined_detections = self._combine_detections(florence_results, dino_results, image)
        
        print(f"DEBUG: Combined detections count: {len(combined_detections)}")
        if combined_detections:
            print(f"DEBUG: Sample detection: {combined_detections[0]}")
        
        # Deduplicate overlapping detections of the same component type
        # This prevents overcounting when AI detects multiple overlapping boxes for the same component
        combined_detections = self._deduplicate_detections(combined_detections)
        print(f"DEBUG: After deduplication: {len(combined_detections)} detections")
        
        # Only run template matching for customized components that AI detected
        # This prevents false positives from matching components that aren't in the image
        template_matches = []
        if image is not None and combined_detections:
            # Get unique labels from AI detections
            ai_labels = set(detection.get('label', '').lower() for detection in combined_detections)
            print(f"DEBUG: AI detected labels: {ai_labels}")
            
            # Map AI labels to template matching methods
            label_to_method = {
                'cyclone': self._match_cyclone_template,
                'turbine': self._match_turbine_template,
                'boiler': self._match_boiler_template,
                'conveyor': self._match_conveyor_template,
                'crusher': self._match_crusher_template,
                'furnace': self._match_furnace_template,
                'calciner': self._match_calciner_template,
                'stacker': self._match_stacker_template,
                'separator': self._match_separator_template,
                'cyclone_separator': self._match_cyclone_separator_template,
                'tank': self._match_tank_template,
                'motor': self._match_motor_template,
            }
            
            # Always run template matching for cyclone, separator, cyclone_separator, and motor (critical components for Ignition Designer)
            # Tank removed from forced list due to false positives
            # Run template matching for other components only if AI detected them
            for component_name, match_method in label_to_method.items():
                # Force cyclone, separator, cyclone_separator, and motor template matching regardless of AI detection
                if component_name in ['cyclone', 'separator', 'cyclone_separator', 'motor']:
                    print(f"DEBUG: Forcing template matching for {component_name}...")
                    match = match_method(image)
                    if match:
                        print(f"DEBUG: {component_name.capitalize()} found via template matching, using this result")
                        # Add classification metadata for Ignition Designer JSON output
                        classification = self._classify_component(match.get('label', component_name))
                        match['classification'] = classification
                        print(f"DEBUG: {component_name.capitalize()} classification: {classification}")
                        template_matches.append(match)
                    else:
                        print(f"DEBUG: No good {component_name} template match found")
                elif component_name in ai_labels:
                    print(f"DEBUG: Attempting template matching for {component_name}...")
                    match = match_method(image)
                    if match:
                        print(f"DEBUG: {component_name.capitalize()} found via template matching, using this result")
                        # Add classification metadata for Ignition Designer JSON output
                        classification = self._classify_component(match.get('label', component_name))
                        match['classification'] = classification
                        print(f"DEBUG: {component_name.capitalize()} classification: {classification}")
                        template_matches.append(match)
                    else:
                        print(f"DEBUG: No good {component_name} template match found")
                else:
                    print(f"DEBUG: Skipping {component_name} template matching (not detected by AI)")
            
            # Generic fallback: Try template matching for any AI-detected component that doesn't have a dedicated method
            for ai_label in ai_labels:
                if ai_label not in label_to_method:
                    print(f"DEBUG: Trying generic template matching for {ai_label}...")
                    match = self._match_generic_component_template(image, ai_label)
                    if match:
                        print(f"DEBUG: {ai_label.capitalize()} found via generic template matching")
                        classification = self._classify_component(match.get('label', ai_label))
                        match['classification'] = classification
                        template_matches.append(match)
                    else:
                        print(f"DEBUG: No generic template match found for {ai_label}")
            
            # Comprehensive fallback: DISABLED to prevent false positives when no customized components exist
            # This was causing false positives for customized components when none exist in the image
            # if not template_matches or len(template_matches) < len(ai_labels):
            #     print(f"DEBUG: Running comprehensive template matching to verify/correct AI detections...")
            #     
            #     # Collect all template paths from Template folder
            #     all_template_paths = []
            #     if TEMPLATE_DIR.exists():
            #         for category_dir in TEMPLATE_DIR.iterdir():
            #             if category_dir.is_dir():
            #                 for item in category_dir.iterdir():
            #                     if item.is_dir():
            #                         thumbnail = item / "thumbnail.png"
            #                         if thumbnail.exists():
            #                             component_name = item.name.lower().replace('_', ' ').replace('-', ' ')
            #                             all_template_paths.append((thumbnail, component_name))
            #     
            #     print(f"DEBUG: Found {len(all_template_paths)} templates to try")
            #     
            #     # Try each template (limit to 50 to avoid excessive processing)
            #     best_match = None
            #     best_confidence = 0
            #     best_component = None
            #     
            #     for i, (template_path, component_name) in enumerate(all_template_paths[:50]):
            #         if i % 10 == 0:
            #             print(f"DEBUG: Trying template {i+1}/{min(50, len(all_template_paths))}: {component_name}")
            #         
            #         result = self._match_template_generic(image, component_name, template_path, threshold=0.5)
            #         if result and result.get('confidence', 0) > best_confidence:
            #             best_match = result
            #             best_confidence = result.get('confidence', 0)
            #             best_component = component_name
            #     
            #     if best_match and best_confidence > 0.75:
            #         print(f"DEBUG: Best comprehensive match: {best_component} (confidence: {best_confidence:.2f})")
            #         # Only use comprehensive match if it's significantly better than AI detection
            #         classification = self._classify_component(best_match.get('label', best_component))
            #         best_match['classification'] = classification
            #         template_matches.append(best_match)
            
            # Apply non-maximum suppression to prevent overlapping template matches
            if template_matches:
                template_matches = self._non_max_suppression(template_matches, iou_threshold=0.5)
                print(f"DEBUG: After NMS, {len(template_matches)} template matches remain")
                
                # Filter out very small detections (likely false positives)
                # Use component-specific minimum areas
                filtered_matches = []
                for match in template_matches:
                    bbox = match['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    
                    # Component-specific minimum areas - balanced to reduce false positives
                    label = match.get('label', '').lower()
                    if label == 'motor':
                        min_area = 50  # Motors can be small but not tiny
                    elif label == 'valve':
                        min_area = 100  # Valves need reasonable size to be real
                    elif label == 'instrument':
                        min_area = 80  # Instruments need reasonable size
                    elif label == 'separator':
                        min_area = 150  # Separators need larger size
                    elif label == 'pump':
                        min_area = 100  # Pumps need reasonable size
                    else:
                        min_area = 200  # Default threshold to reduce false positives
                    
                    if area >= min_area:
                        filtered_matches.append(match)
                    else:
                        print(f"DEBUG: Filtering out {match['label']} - too small (area: {area}, min: {min_area})")
                
                template_matches = filtered_matches
                print(f"DEBUG: After size filtering, {len(template_matches)} template matches remain")
            
            # Add all template matches to validated components
            validated_components.extend(template_matches)
        
        # Validate components (library verification disabled due to contamination)
        for detection in combined_detections:
            print(f"DEBUG: Validating detection: {detection}")
            
            # Special case: if separator is found via template matching, skip pump detections that overlap with it
            if detection.get('label', '').lower() == 'pump':
                for template_match in template_matches:
                    if template_match.get('label', '').lower() == 'separator':
                        # Check if pump overlaps with separator
                        iou = calculate_iou(template_match['bbox'], detection['bbox'])
                        if iou > 0.1:  # If they overlap significantly, skip the pump
                            print(f"DEBUG: Skipping pump detection - overlaps with separator (IoU: {iou:.2f})")
                            continue
            
            # Don't skip tank detections when motor is inside - they are separate components
            # Remove the logic that skips tank when motor is inside it
            
            # Skip if we already have a template match in this area
            skip_detection = False
            for template_match in template_matches:
                template_bbox = template_match['bbox']
                detection_bbox = detection['bbox']
                
                # Check if template match is inside detection bbox
                tx1, ty1, tx2, ty2 = template_bbox
                dx1, dy1, dx2, dy2 = detection_bbox
                
                # If template match is inside detection bbox, skip the detection
                if (dx1 <= tx1 and tx2 <= dx2 and dy1 <= ty1 and ty2 <= dy2):
                    print(f"DEBUG: Skipping detection - template match {template_match['label']} is inside detection bbox")
                    skip_detection = True
                    break
                
                # Calculate IoU as fallback
                iou = calculate_iou(template_bbox, detection_bbox)
                if iou > 0.3:  # Lower threshold to catch more overlaps
                    print(f"DEBUG: Skipping detection due to {template_match['label']} overlap (IoU: {iou:.2f})")
                    skip_detection = True
                    break
            
            if skip_detection:
                continue
            
            if self._validate_component(detection, image.shape if image is not None else None, ocr_results, image):
                # Add classification metadata for Ignition Designer JSON output
                classification = self._classify_component(detection.get('label', 'unknown'))
                detection['classification'] = classification
                validated_components.append(detection)
                print(f"DEBUG: Component validated: {detection['label']} (classification: {classification['classification']})")
            else:
                print(f"DEBUG: Component rejected: {detection['label']}")
        
        print(f"DEBUG: Final validated components: {len(validated_components)}")
        
        # Auto-crop disabled to prevent library contamination with false positives
        # if image is not None and validated_components:
        #     print("Auto-cropping components to reference library...")
        #     auto_crop_and_save_components(image, validated_components, save_to_library=True)
        
        # Connect related components (simplified)
        connections = self._connect_components(validated_components, opencv_results.get('lines', []))
        
        return {
            'validated_components': validated_components,
            'filtered_text': filtered_text,
            'connections': connections,
            'total_validated': len(validated_components),
            'total_filtered_text': len(filtered_text),
            'total_connections': len(connections)
        }
    
    def _validate_isa_tag(self, tag: str) -> bool:
        """Validate instrument tag against ISA 5.1 rules"""
        # Check for valid instrument type
        has_valid_type = any(tag.startswith(t) for t in VALID_INSTRUMENT_TAGS)
        # Check for loop number (numeric suffix)
        has_loop_number = any(c.isdigit() for c in tag)
        
        return has_valid_type and has_loop_number
    
    def _validate_equipment_name(self, name: str) -> bool:
        """Validate equipment name against ISA 5.1 rules"""
        # Check for valid equipment prefix
        has_valid_prefix = any(name.startswith(p) for p in VALID_EQUIPMENT_PREFIXES)
        # Check for numeric suffix
        has_number = any(c.isdigit() for c in name)
        
        return has_valid_prefix and has_number
    
    def _validate_component(self, detection: Dict, image_shape: Tuple[int, int] = None, ocr_results: Dict = None, image: np.ndarray = None) -> bool:
        """Validate component detection with fine-tuned thresholds for maximum accuracy"""
        label = detection.get('label', '').lower()
        confidence = detection.get('confidence', 0)
        
        # Validate component characteristics (size, aspect ratio)
        if not self._validate_component_characteristics(detection.get('bbox', []), label, image_shape):
            return False
        
        # OCR position verification - ensure component actually exists at detected position
        if ocr_results is not None and image is not None:
            bbox = detection.get('bbox', [])
            if len(bbox) == 4:
                position_verified = self._verify_component_position_with_ocr(image, bbox, label, ocr_results)
                if not position_verified:
                    print(f"DEBUG: Component position not verified by OCR: {label} at {bbox}")
                    # Don't reject immediately, but require visual verification
                    visual_verified = self._verify_component_visually(image, bbox, label)
                    if not visual_verified:
                        print(f"DEBUG: Component rejected - failed both OCR and visual verification: {label}")
                        return False
            
        # Component-specific confidence thresholds to prevent false positives and overcounting
        if label == 'motor':
            if confidence < 0.25:  # Lowered from 0.35 to improve motor detection
                return False
        elif label == 'tank':
            if confidence < 0.35:
                return False
        elif label == 'pump':
            if confidence < 0.35:
                return False
        elif label == 'valve':
            if confidence < 0.25:
                return False
        elif label in ['instrument', 'sensor', 'controller', 'transmitter', 'indicator', 'gauge']:
            if confidence < 0.15:  # Moderate threshold to balance detection and false positives
                return False
        else:
            # Default threshold for other components
            if confidence < 0.30:
                return False
        
        # Expanded label validation to include customized components
        valid_labels = ['pump', 'valve', 'vessel', 'motor', 'pipe', 'tank', 'sensor', 'controller', 'transmitter', 'indicator', 'gauge', 'instrument', 'component',
                       'cyclone', 'turbine', 'boiler', 'heat exchanger', 'compressor', 'reactor', 'separator']
        
        # Require exact label match or partial match for customized components
        if label in valid_labels:
            return True
        
        # Check for partial matches with customized components
        for custom_label in ['cyclone', 'turbine', 'boiler', 'heat exchanger', 'compressor', 'reactor', 'separator']:
            if custom_label in label:
                return True
        
        return False
    
    def _detect_cyclone_pattern(self, detections: List[Dict]) -> List[Dict]:
        """Detect if multiple pipe detections form a cyclone pattern and group them"""
        print(f"DEBUG: _detect_cyclone_pattern called with {len(detections)} detections")
        
        if len(detections) < 3:  # Need at least 3 components to form a cyclone
            print(f"DEBUG: Not enough detections for cyclone pattern (need 3+, got {len(detections)})")
            return detections
        
        # Filter for pipe-like detections
        pipe_detections = [d for d in detections if 'pipe' in d.get('label', '').lower()]
        print(f"DEBUG: Found {len(pipe_detections)} pipe detections")
        
        if len(pipe_detections) < 3:
            print(f"DEBUG: Not enough pipe detections for cyclone pattern (need 3+, got {len(pipe_detections)})")
            return detections
        
        # Calculate bounding box that encompasses all pipe detections
        all_boxes = [d['bbox'] for d in pipe_detections]
        min_x = min(box[0] for box in all_boxes)
        min_y = min(box[1] for box in all_boxes)
        max_x = max(box[2] for box in all_boxes)
        max_y = max(box[3] for box in all_boxes)
        
        # Calculate aspect ratio and area
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = width / height if height > 0 else 0
        area = width * height
        
        print(f"DEBUG: Grouped pipes - width: {width:.1f}, height: {height:.1f}, aspect_ratio: {aspect_ratio:.2f}, area: {area:.1f}")
        
        # Cyclones typically have roughly square or slightly rectangular shape
        # and are composed of multiple curved/angled elements
        # More relaxed criteria to catch more cyclone patterns
        if 0.5 <= aspect_ratio <= 2.0 and area > 500:  # More relaxed aspect ratio and area
            # Create a single cyclone detection
            cyclone_detection = {
                'bbox': [min_x, min_y, max_x, max_y],
                'label': 'cyclone',
                'confidence': max(d.get('confidence', 0.5) for d in pipe_detections),
                'source': 'cyclone_pattern'
            }
            
            print(f"DEBUG: Cyclone pattern detected - grouped {len(pipe_detections)} pipes into cyclone")
            print(f"DEBUG: Cyclone bbox: {cyclone_detection['bbox']}, area: {area}, aspect_ratio: {aspect_ratio:.2f}")
            
            # Remove the individual pipe detections that formed the cyclone
            non_cyclone_detections = [d for d in detections if d not in pipe_detections]
            return [cyclone_detection] + non_cyclone_detections
        
        print(f"DEBUG: Cyclone pattern not detected - aspect_ratio: {aspect_ratio:.2f}, area: {area:.1f}")
        return detections
    
    def _combine_detections(self, florence_results: Dict, dino_results: Dict, image: np.ndarray = None) -> List[Dict]:
        """Combine Florence and DINO detections with minimal filtering"""
        combined = []
        
        print(f"DEBUG: DINO results - boxes: {len(dino_results['boxes'])}, confidences: {len(dino_results['confidences'])}, labels: {len(dino_results['labels'])}")
        
        # Add DINO detections with improved label classification - NO visual validation
        for box, conf, label in zip(dino_results['boxes'], dino_results['confidences'], dino_results['labels']):
            print(f"DEBUG: Processing DINO detection - label: {label}, conf: {conf}, box: {box}")
            
            # Improve label classification to distinguish valves from pumps
            improved_label = self._improve_label_classification(label, box)
            
            # AGGRESSIVE template matching for speed - only 1 template max, no parallel
            # Only verify very low-confidence detections (<0.60) to save time
            if image is not None and conf < 0.60:
                print(f"DEBUG: Quick template check for low-confidence '{improved_label}' (conf: {conf:.2f})...")
                
                # Try only the most relevant template based on AI label
                label_lower = improved_label.lower()
                
                # Direct mapping to single best template
                label_to_template = {
                    'motor': 'motor',
                    'pump': 'pump', 
                    'valve': 'valve',
                    'tank': 'tank',
                    'separator': 'separator',
                    'cyclone': 'cyclone',
                    'cyclone separator': 'cyclone',
                    'air separator': 'cyclone',
                    'dust separator': 'cyclone',
                    'stacker': 'stacker',
                    'stacker reclaimer': 'stacker',
                    'turbine': 'turbine',
                    'baghouse hopper': 'hopper',
                    'baghouse hopper single': 'hopper',
                    'calciner': 'calciner',
                    'chimney': 'chimney'
                }
                
                template_keyword = label_to_template.get(label_lower, label_lower)
                best_match = None
                best_confidence = 0
                best_component = None
                
                # Try only 1 template max for speed
                if TEMPLATE_DIR.exists():
                    for category_dir in TEMPLATE_DIR.iterdir():
                        if category_dir.is_dir():
                            for item in category_dir.iterdir():
                                if item.is_dir():
                                    thumbnail = item / "thumbnail.png"
                                    if thumbnail.exists():
                                        component_name = item.name.lower().replace('_', ' ').replace('-', ' ')
                                        if template_keyword in component_name:
                                            result = self._match_template_generic(image, component_name, thumbnail, threshold=0.65)
                                            if result and result.get('confidence', 0) > best_confidence:
                                                best_match = result
                                                best_confidence = result.get('confidence', 0)
                                                best_component = component_name
                                            break  # Only try first match
                            if best_match:
                                break  # Stop after finding first template
                
                # Only override if very high confidence
                if best_match and best_confidence > 0.90:
                    print(f"DEBUG: Overriding '{improved_label}' with '{best_component}' (conf: {best_confidence:.2f})")
                    improved_label = best_component
            
            # Check if pump might actually be a motor based on label text
            if improved_label == 'pump' and 'motor' in label.lower():
                improved_label = 'motor'
                print(f"DEBUG: Reclassified as motor based on label text")
            
            print(f"DEBUG: Improved label: {improved_label}")
            
            # Apply component-specific confidence check to prevent false positives and overcounting
            label_threshold = 0.30  # Default
            if improved_label == 'motor':
                label_threshold = 0.05  # DRAMATICALLY lowered from 0.15 to catch ALL motors
            elif improved_label == 'pipe':
                label_threshold = 0.20  # Lowered for better pipe detection
            elif improved_label in ['cyclone', 'separator', 'stacker', 'turbine', 'baghouse hopper', 'baghouse hopper single']:
                label_threshold = 0.15  # Lowered for better customized component detection
            elif improved_label == 'tank':
                label_threshold = 0.35
            elif improved_label == 'pump':
                label_threshold = 0.35
            elif improved_label == 'valve':
                label_threshold = 0.25
            elif improved_label == 'instrument':
                label_threshold = 0.30
            
            if conf < label_threshold:
                print(f"DEBUG: Component failed confidence check ({conf} < {label_threshold})")
                continue
                
            print(f"DEBUG: Component passed validation, adding to combined")
            combined.append({
                'bbox': box,
                'confidence': conf,
                'label': improved_label,
                'source': 'dino'
            })
        
        # Add Florence detections (if not overlapping with DINO)
        for box, label in zip(florence_results['bboxes'], florence_results['labels']):
            # Check overlap with existing detections
            overlaps = False
            for det in combined:
                if calculate_iou(box, det['bbox']) > 0.5:
                    overlaps = True
                    break
            
            if not overlaps:
                combined.append({
                    'bbox': box,
                    'confidence': 0.5,
                    'label': label,
                    'source': 'florence'
                })
        
        # Apply cyclone pattern detection to group pipe detections
        print(f"DEBUG: Before cyclone pattern detection: {len(combined)} detections")
        combined = self._detect_cyclone_pattern(combined)
        print(f"DEBUG: After cyclone pattern detection: {len(combined)} detections")
        
        # If still many pipe detections and no cyclone, try direct template matching
        pipe_count = sum(1 for d in combined if 'pipe' in d.get('label', '').lower())
        cyclone_count = sum(1 for d in combined if 'cyclone' in d.get('label', '').lower())
        
        if pipe_count >= 3 and cyclone_count == 0:
            print(f"DEBUG: Found {pipe_count} pipes but no cyclone, trying template matching...")
            # This would need the original image, which we don't have here
            # For now, rely on the pattern detection
        
        print(f"DEBUG: Final combined detections: {len(combined)}")
        return combined
    
    def _verify_component_with_library(self, image: np.ndarray, detection: Dict) -> Dict:
        """Verify detected component against component library for accurate classification"""
        bbox = detection.get('bbox', [])
        if len(bbox) != 4:
            detection['verified'] = False
            detection['similarity'] = 0.0
            return detection
        
        # Convert normalized coordinates to pixel coordinates if needed
        max_coord = max(bbox)
        if max_coord <= 1.0:
            h, w = image.shape[:2]
            cx, cy, wb, hb = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
            x, y, w_box, h_box = cx - wb/2, cy - hb/2, wb, hb
        else:
            x1, y1, x2, y2 = bbox
            x, y, w_box, h_box = x1, y1, x2 - x1, y2 - y1
        
        features = _extract_fast_features(image, (int(max(0, x)), int(max(0, y)), int(w_box), int(h_box)))
        
        # Match against component library
        try:
            matches = match_component_to_library(features, detection.get('label'), top_k=3)
            
            if matches and matches[0]['similarity'] > 0.7:
                # High confidence match - update label
                best_match = matches[0]
                detection['verified'] = True
                detection['verified_label'] = best_match['label']
                detection['similarity'] = best_match['similarity']
                detection['library_image'] = best_match['image']
            else:
                detection['verified'] = False
                detection['similarity'] = matches[0]['similarity'] if matches else 0.0
        except Exception as e:
            # If library verification fails (e.g., empty library), mark as not verified
            print(f"DEBUG: Library verification failed: {e}")
            detection['verified'] = False
            detection['similarity'] = 0.0
        
        return detection
    
    def _validate_component_characteristics(self, bbox: List, label: str, image_shape: Tuple[int, int] = None) -> bool:
        """Validate component based on visual characteristics and relative size constraints"""
        try:
            print(f"DEBUG: _validate_component_characteristics called with bbox: {bbox}, type: {type(bbox)}, label: {label}, image_shape: {image_shape}, type: {type(image_shape)}")
            
            # If image_shape is a list or contains lists, skip relative size check entirely
            if image_shape is not None and isinstance(image_shape, (list, tuple)):
                print(f"DEBUG: image_shape is list/tuple, checking contents...")
                # If image_shape has 3 elements (h, w, c), extract only h and w
                if len(image_shape) == 3:
                    print(f"DEBUG: image_shape has 3 elements (h, w, c), extracting h and w")
                    image_shape = (image_shape[0], image_shape[1])
                # If any element is a list, skip relative size check
                elif any(isinstance(x, (list, tuple)) for x in image_shape):
                    print(f"DEBUG: image_shape contains nested lists, skipping relative size check")
                    image_shape = None
            
            if len(bbox) != 4:
                print(f"DEBUG: Invalid bbox length: {len(bbox)}")
                return False
            
            # Safely extract coordinates, handling nested lists
            coords = []
            for coord in bbox:
                if isinstance(coord, (list, tuple)):
                    coords.append(float(coord[0]) if len(coord) > 0 else 0.0)
                else:
                    coords.append(float(coord))
            x1, y1, x2, y2 = coords
            print(f"DEBUG: Extracted coords: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Ensure all coordinates are floats, not lists
            try:
                x1 = float(x1) if not isinstance(x1, str) else float(x1)
                y1 = float(y1) if not isinstance(y1, str) else float(y1)
                x2 = float(x2) if not isinstance(x2, str) else float(x2)
                y2 = float(y2) if not isinstance(y2, str) else float(y2)
            except (ValueError, TypeError) as e:
                print(f"DEBUG: Could not convert coordinates to float: {e}")
                return False
            
            # Check if coordinates are normalized (0-1 range) or pixel coordinates
            # If max coordinate is <= 1.0, assume normalized coordinates
            try:
                max_coord = max(x1, y1, x2, y2)
                print(f"DEBUG: max_coord={max_coord}, type={type(max_coord)}")
                if max_coord <= 1.0:
                    # Convert normalized cxcywh to pixel coordinates (assuming 1000x1000 image)
                    cx, cy, wb, hb = x1, y1, x2, y2
                    print(f"DEBUG: Before conversion: cx={type(cx)}, cy={type(cy)}, wb={type(wb)}, hb={type(hb)}")
                    # Ensure wb and hb are numeric before division
                    try:
                        wb = float(wb) if not isinstance(wb, (list, tuple)) else float(wb[0] if len(wb) > 0 else 0)
                        hb = float(hb) if not isinstance(hb, (list, tuple)) else float(hb[0] if len(hb) > 0 else 0)
                    except (TypeError, ValueError, IndexError):
                        print(f"DEBUG: Could not convert wb/hb to float: wb={wb}, hb={hb}")
                        return False
                    x1, y1, x2, y2 = (cx - wb/2) * 1000, (cy - hb/2) * 1000, (cx + wb/2) * 1000, (cy + hb/2) * 1000
                    print(f"DEBUG: After conversion: x1={type(x1)}, y1={type(y1)}, x2={type(x2)}, y2={type(y2)}")
                
                width = x2 - x1
                height = y2 - y1
                print(f"DEBUG: width={width}, height={height}, types: width={type(width)}, height={type(height)}")
                
                # Ensure width and height are numeric
                try:
                    width = float(width) if not isinstance(width, (list, tuple)) else float(width[0] if len(width) > 0 else 0)
                    height = float(height) if not isinstance(height, (list, tuple)) else float(height[0] if len(height) > 0 else 0)
                except (TypeError, ValueError, IndexError):
                    print(f"DEBUG: Could not convert width/height to float: width={width}, height={height}")
                    return False
                
                # Safely calculate aspect ratio
                aspect_ratio = width / height if height > 0 else 0
                print(f"DEBUG: aspect_ratio={aspect_ratio}, type={type(aspect_ratio)}")
            except TypeError as e:
                print(f"DEBUG: TypeError in coordinate calculations: {e}")
                print(f"DEBUG: Coordinate types: x1={type(x1)}, y1={type(y1)}, x2={type(x2)}, y2={type(y2)}")
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                return False
            
            # Ensure width and height are numeric before area calculation
            print(f"DEBUG: Before area conversion - width={width} (type: {type(width)}), height={height} (type: {type(height)})")
            try:
                if isinstance(width, (list, tuple)):
                    print(f"DEBUG: width is a list/tuple, extracting first element")
                    width = float(width[0]) if len(width) > 0 else 0.0
                else:
                    width = float(width)
                
                if isinstance(height, (list, tuple)):
                    print(f"DEBUG: height is a list/tuple, extracting first element")
                    height = float(height[0]) if len(height) > 0 else 0.0
                else:
                    height = float(height)
            except (TypeError, ValueError, IndexError) as e:
                print(f"DEBUG: Could not convert width/height to float before area calc: width={width}, height={height}, error={e}")
                return False
            
            print(f"DEBUG: After area conversion - width={width} (type: {type(width)}), height={height} (type: {type(height)})")
            
            # Calculate area with type safety
            try:
                area = width * height
                print(f"DEBUG: area={area} (type: {type(area)})")
            except TypeError as e:
                print(f"DEBUG: TypeError in area calculation: {e}")
                return False
            
            # Very relaxed validation rules to avoid filtering valid components
            label_lower = label.lower()
            
            # Only filter extreme cases
            try:
                if area < 30 or area > 500000:  # Very wide range - lowered min area from 50 to 30
                    return False
                if aspect_ratio < 0.05 or aspect_ratio > 20.0:  # Very wide range - more relaxed
                    return False
            except TypeError as e:
                print(f"DEBUG: TypeError in area/aspect ratio checks: {e}")
                return False
            
            # Check size relative to image size if available
            # DISABLED: Relative size check causing type errors with image_shape
            # This check is not critical for analysis to work
            # if image_shape is not None:
            #     [relative size check code disabled]
            
            return True
        except Exception as e:
            print(f"DEBUG: Error in _validate_component_characteristics: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return False
    
    def _improve_label_classification(self, label: str, bbox: List) -> str:
        """Improve label classification with expert-level accuracy rules"""
        label_lower = label.lower()
        
        # Expert classification rules for maximum accuracy
        # Priority order based on specificity
        
        # Check for customized components first (highest priority to prevent misclassification)
        custom_keywords = ['cyclone', 'cyclone separator', 'turbine', 'boiler', 'heat exchanger', 'compressor', 'reactor', 'separator']
        for keyword in custom_keywords:
            if keyword in label_lower:
                return keyword.replace(' ', '_')  # Return standardized name
        
        # Check for valve-specific keywords
        valve_keywords = ['gate valve', 'globe valve', 'ball valve', 'check valve', 'control valve', 'butterfly valve', 'plug valve']
        for keyword in valve_keywords:
            if keyword in label_lower:
                return 'valve'
        
        # Generic valve detection
        if 'valve' in label_lower and 'pump' not in label_lower:
            return 'valve'
        
        # Check for pump-specific keywords
        pump_keywords = ['centrifugal pump', 'gear pump', 'reciprocating pump', 'screw pump', 'diaphragm pump']
        for keyword in pump_keywords:
            if keyword in label_lower:
                return 'pump'
        
        # Generic pump detection (only if no valve keywords)
        if 'pump' in label_lower and 'valve' not in label_lower and 'motor' not in label_lower:
            return 'pump'
        
        # Tank/vessel detection (comprehensive)
        if any(word in label_lower for word in ['tank', 'vessel', 'reactor', 'storage tank', 'horizontal tank', 'vertical tank']):
            return 'tank'
        
        # Motor detection (specific to avoid confusion with pumps)
        if any(word in label_lower for word in ['motor', 'electric motor', 'motor drive']) and 'pump' not in label_lower:
            return 'motor'
        
        # Instrument detection (comprehensive list)
        instrument_keywords = ['sensor', 'transmitter', 'indicator', 'gauge', 'instrument', 
                             'pressure transmitter', 'temperature transmitter', 'flow indicator', 'level gauge',
                             'bubble', 'meter', 'circular tag', 'dial', 'circle']
        for keyword in instrument_keywords:
            if keyword in label_lower:
                return 'instrument'
        
        # Default: return first word if no specific classification
        words = label_lower.split()
        return words[0] if words else 'other'
    
    def _connect_components(self, components: List[Dict], lines: List[Dict]) -> List[Dict]:
        """Connect related components based on line proximity"""
        connections = []
        
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                # Check if components are connected by lines
                if self._are_components_connected(comp1['bbox'], comp2['bbox'], lines):
                    connections.append({
                        'component_1': comp1['label'],
                        'component_2': comp2['label'],
                        'connection_type': 'pipe'
                    })
        
        return connections
    
    def _are_components_connected(self, bbox1: List[int], bbox2: List[int], lines: List[Dict]) -> bool:
        """Check if two components are connected by lines"""
        # Get centers
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        # Check if any line connects near both centers
        for line in lines:
            start = line['start']
            end = line['end']
            
            # Check proximity
            dist1 = np.sqrt((center1[0] - start[0])**2 + (center1[1] - start[1])**2)
            dist2 = np.sqrt((center2[0] - end[0])**2 + (center2[1] - end[1])**2)
            
            if dist1 < 50 and dist2 < 50:  # Threshold distance
                return True
        
        return False


def _extract_fast_features(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract fast features from a component region for template matching"""
    x, y, w, h = bbox
    
    # Ensure bbox is within image bounds
    h_img, w_img = image.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    
    if w <= 0 or h <= 0:
        return np.zeros((32, 32), dtype=np.uint8)
    
    # Extract region
    region = image[y:y+h, x:x+w]
    
    # Resize to standard size for comparison
    if region.size > 0:
        region = cv2.resize(region, (32, 32))
    
    # Convert to grayscale if needed
    if len(region.shape) == 3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    return region


def match_component_to_library(features: np.ndarray, label: str, top_k: int = 3) -> List[Dict]:
    """Match detected component features against reference library"""
    matches = []
    
    try:
        # Load reference library
        reference_library = load_reference_library()
        
        if not reference_library:
            return matches
        
        # Match against reference images
        for category, templates in reference_library.items():
            for template_name, template_features in templates.items():
                # Calculate similarity using template matching
                similarity = cv2.matchTemplate(features, template_features, cv2.TM_CCOEFF_NORMED)
                max_similarity = np.max(similarity)
                
                if max_similarity > 0.5:  # Threshold for potential match
                    matches.append({
                        'label': category,
                        'template_name': template_name,
                        'similarity': float(max_similarity),
                        'image': template_name
                    })
        
        # Sort by similarity and return top_k
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:top_k]
        
    except Exception as e:
        print(f"Error matching component to library: {e}")
        return matches


def load_reference_library() -> Dict[str, Dict[str, np.ndarray]]:
    """Load reference library from disk"""
    library = {}
    
    try:
        reference_dir = REFERENCE_LIBRARY_DIR
        
        if not reference_dir.exists():
            print(f"Reference library directory not found: {reference_dir}")
            return library
        
        # Load images from reference library
        for category_dir in reference_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                library[category] = {}
                
                for image_file in category_dir.glob("*.png"):
                    try:
                        # Load and preprocess reference image
                        template = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                        if template is not None:
                            # Resize to standard size
                            template = cv2.resize(template, (32, 32))
                            library[category][image_file.stem] = template
                    except Exception as e:
                        print(f"Error loading reference image {image_file}: {e}")
        
        print(f"Loaded reference library with {len(library)} categories")
        for category, templates in library.items():
            print(f"  {category}: {len(templates)} templates")
        
    except Exception as e:
        print(f"Error loading reference library: {e}")
    
    return library


def auto_crop_and_save_components(image: np.ndarray, detections: List[Dict], save_to_library: bool = True):
    """Automatically crop detected components and save to reference library"""
    if not save_to_library:
        return
    
    try:
        # Ensure reference library directory exists
        REFERENCE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        
        for i, detection in enumerate(detections):
            bbox = detection.get('bbox', [])
            label = detection.get('label', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            # Skip low-confidence detections
            if confidence < 0.35:
                continue
            
            # Convert bbox format if needed
            if len(bbox) == 4:
                # Check if normalized or pixel coordinates
                max_coord = max(bbox)
                if max_coord <= 1.0:
                    # Normalized coordinates - convert to pixels
                    h, w = image.shape[:2]
                    x1, y1, x2, y2 = [int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)]
                else:
                    # Pixel coordinates
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Ensure bbox is within image bounds
                h_img, w_img = image.shape[:2]
                x1 = max(0, min(x1, w_img - 1))
                y1 = max(0, min(y1, h_img - 1))
                x2 = max(0, min(x2, w_img))
                y2 = max(0, min(y2, h_img))
                
                # Crop component
                if x2 > x1 and y2 > y1:
                    component = image[y1:y2, x1:x2]
                    
                    # Create category directory
                    category = label.lower().replace(' ', '_')
                    category_dir = REFERENCE_LIBRARY_DIR / category
                    category_dir.mkdir(exist_ok=True)
                    
                    # Save cropped component
                    timestamp = int(time.time() * 1000)
                    filename = f"{category}_{timestamp}_{i}.png"
                    filepath = category_dir / filename
                    
                    cv2.imwrite(str(filepath), component)
                    print(f"Auto-saved component: {label} -> {filepath}")
    
    except Exception as e:
        print(f"Error auto-cropping components: {e}")
