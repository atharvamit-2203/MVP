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
        self._load_models()
    
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
                
                # Apply deduplication with balanced threshold
                deduped = self._dedupe_detections(refined_detections, iou_threshold=0.45)
                print(f"After deduplication: {len(deduped)} detections")
                
                # Apply close merge with balanced threshold
                merged = self._merge_close_detections(deduped, distance_ratio=0.20)
                print(f"After close merge: {len(merged)} detections")
                
                # Apply additional IoU merge for all components
                merged = self._aggressive_iou_merge(merged, iou_threshold=0.35)
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
        
        # Compile results
        results = {
            'image_path': image_path,
            'image_name': image_name,
            'opencv_results': opencv_results,
            'ocr_results': ocr_results,
            'florence_results': florence_results,
            'dino_results': dino_results,
            'heuristic_results': heuristic_results
        }
        
        return results
    
    def _opencv_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """OpenCV analysis: detect pipes, lines, junctions, components"""
        print("  - Running OpenCV analysis...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply slight blur for better edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect lines (pipes) with optimized thresholds for better detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=70, 
                               minLineLength=40, maxLineGap=10)
        
        # Also try with different parameters for horizontal/vertical lines
        edges2 = cv2.Canny(blurred, 30, 100, apertureSize=3)
        lines2 = cv2.HoughLinesP(edges2, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=15)
        
        # Merge lines from both detections
        all_lines = []
        if lines is not None:
            all_lines.extend(lines)
        if lines2 is not None:
            all_lines.extend(lines2)
        
        line_data = []
        if all_lines:
            for line in all_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Determine line type (solid vs dashed approximation)
                line_type = "solid"  # Simplified - could be enhanced
                
                # Classify as pipe if length is substantial
                is_pipe = length > 40  # Reduced threshold for better pipe detection
                
                line_data.append({
                    'start': [int(x1), int(y1)],
                    'end': [int(x2), int(y2)],
                    'length': float(length),
                    'angle': float(angle),
                    'type': line_type,
                    'is_pipe': is_pipe
                })
        
        # Detect junctions (intersections)
        junctions = self._detect_junctions(all_lines if all_lines else None)
        
        # Detect potential component regions (contours) with both edge maps
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Merge contours from both detections
        all_contours = []
        if contours is not None:
            all_contours.extend(contours)
        if contours2 is not None:
            all_contours.extend(contours2)
        
        component_boxes = []
        for contour in all_contours:
            area = cv2.contourArea(contour)
            if area > MIN_COMPONENT_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out very small or very large boxes
                if w > 10 and h > 10 and w < image.shape[1] * 0.5 and h < image.shape[0] * 0.5:
                    component_boxes.append([x, y, x+w, y+h])
        
        # Count pipes by grouping connected line segments
        pipe_count = self._count_connected_pipes(line_data)
        
        return {
            'lines': line_data,
            'junctions': junctions,
            'component_boxes': component_boxes,
            'total_lines': len(line_data),
            'total_pipes': pipe_count,
            'total_junctions': len(junctions),
            'total_components': len(component_boxes)
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
    
    def _are_segments_connected(self, seg1: Dict, seg2: Dict, distance_threshold: float = 15.0, angle_threshold: float = 25.0) -> bool:
        """Check if two line segments are connected (proximate and similar angle)"""
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
            
            # Moderate downsampling for balance of speed and accuracy
            original_size = pil_image.size
            if max(original_size) > 512:
                scale = 512 / max(original_size)
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                pil_image = pil_image.resize(new_size, Image.BILINEAR)  # BILINEAR for better quality
            
            # Use CAPTION_TO_PHRASE_GROUNDING task for better P&ID detection
            prompt = "tank valve pump instrument"
            inputs = self.florence_processor(text=prompt, images=pil_image, return_tensors="pt").to(FLORENCE_DEVICE)
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,  # Increased for better detection
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
            
            # Transform image to tensor using Grounding DINO's transform (reduced size for speed)
            transform = T.Compose([
                T.RandomResize([400], max_size=600),  # Reduced for speed (was 800x1333)
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor, _ = transform(image_pil, None)
            
            # Use focused prompts for P&ID components with optimized thresholds for maximum recall
            prompts_config = [
                {"prompt": "tank", "box_threshold": 0.30, "text_threshold": 0.25},
                {"prompt": "valve", "box_threshold": 0.25, "text_threshold": 0.20},
                {"prompt": "pump", "box_threshold": 0.25, "text_threshold": 0.20},
                {"prompt": "instrument sensor bubble gauge meter circular tag", "box_threshold": 0.22, "text_threshold": 0.18}
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
            min_confidence = 0.35  # Filter out detections below this confidence
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
            
            verified_detections = []
            total_tokens = 0
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(verify_single, det) for det in detections[:35]]
                for future in futures:
                    res = future.result()
                    if res is not None:
                        verified_detections.append(res)
                        total_tokens += res.get('tokens', 0)
                        
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
        
        # Combine Florence and DINO detections
        combined_detections = self._combine_detections(florence_results, dino_results)
        
        print(f"DEBUG: Combined detections count: {len(combined_detections)}")
        if combined_detections:
            print(f"DEBUG: Sample detection: {combined_detections[0]}")
        
        # Validate components (library verification disabled due to contamination)
        for detection in combined_detections:
            print(f"DEBUG: Validating detection: {detection}")
            
            # Library verification disabled - reference library contaminated with false positives
            # if image is not None:
            #     detection = self._verify_component_with_library(image, detection)
            #     if detection.get('verified'):
            #         print(f"DEBUG: Component verified as {detection['verified_label']} (similarity: {detection['similarity']:.2f})")
            #         if detection['similarity'] > 0.6:
            #             detection['label'] = detection['verified_label']
            #             validated_components.append(detection)
            #             print(f"DEBUG: Component validated via library: {detection['label']}")
            #             continue
            
            if self._validate_component(detection, image.shape if image is not None else None):
                validated_components.append(detection)
                print(f"DEBUG: Component validated: {detection['label']}")
            else:
                print(f"DEBUG: Component rejected: {detection['label']}")
        
        print(f"DEBUG: Final validated components: {len(validated_components)}")
        
        # Auto-crop disabled to prevent library contamination with false positives
        # if image is not None and validated_components:
        #     print("Auto-cropping components to reference library...")
        #     auto_crop_and_save_components(image, validated_components, save_to_library=True)
        
        # Connect related components (simplified)
        connections = self._connect_components(validated_components, opencv_results['lines'])
        
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
    
    def _validate_component(self, detection: Dict, image_shape: Tuple[int, int] = None) -> bool:
        """Validate component detection with fine-tuned thresholds for maximum accuracy"""
        label = detection.get('label', '').lower()
        confidence = detection.get('confidence', 0)
        
        # Validate component characteristics (size, aspect ratio)
        if not self._validate_component_characteristics(detection.get('bbox', []), label, image_shape):
            return False
            
        # Component-specific confidence thresholds to prevent false positives and overcounting
        if label == 'motor':
            if confidence < 0.35:
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
            if confidence < 0.30:
                return False
        else:
            # Default threshold for other components
            if confidence < 0.30:
                return False
        
        # Strict label validation - expanded to include more instrument types
        valid_labels = ['pump', 'valve', 'vessel', 'motor', 'pipe', 'tank', 'sensor', 'controller', 'transmitter', 'indicator', 'gauge', 'instrument', 'component']
        
        # Require exact label match
        if label in valid_labels:
            return True
        
        return False
    
    def _combine_detections(self, florence_results: Dict, dino_results: Dict) -> List[Dict]:
        """Combine Florence and DINO detections with minimal filtering"""
        combined = []
        
        print(f"DEBUG: DINO results - boxes: {len(dino_results['boxes'])}, confidences: {len(dino_results['confidences'])}, labels: {len(dino_results['labels'])}")
        
        # Add DINO detections with improved label classification - NO visual validation
        for box, conf, label in zip(dino_results['boxes'], dino_results['confidences'], dino_results['labels']):
            print(f"DEBUG: Processing DINO detection - label: {label}, conf: {conf}, box: {box}")
            
            # Improve label classification to distinguish valves from pumps
            improved_label = self._improve_label_classification(label, box)
            print(f"DEBUG: Improved label: {improved_label}")
            
            # Apply component-specific confidence check to prevent false positives and overcounting
            label_threshold = 0.30  # Default
            if improved_label == 'motor':
                label_threshold = 0.35
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
                    'confidence': 0.5,  # Default confidence for Florence
                    'label': label,
                    'source': 'florence'
                })
        
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
        if len(bbox) != 4:
            return False
            
        x1, y1, x2, y2 = bbox
        
        # Check if coordinates are normalized (0-1 range) or pixel coordinates
        # If max coordinate is <= 1.0, assume normalized coordinates
        max_coord = max(x1, y1, x2, y2)
        if max_coord <= 1.0:
            # Convert normalized cxcywh to pixel coordinates (assuming 1000x1000 image)
            cx, cy, wb, hb = x1, y1, x2, y2
            x1, y1, x2, y2 = (cx - wb/2) * 1000, (cy - hb/2) * 1000, (cx + wb/2) * 1000, (cy + hb/2) * 1000
        
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        area = width * height
        
        # Very relaxed validation rules to avoid filtering valid components
        label_lower = label.lower()
        
        # Only filter extreme cases
        if area < 50 or area > 500000:  # Very wide range
            return False
        if aspect_ratio < 0.1 or aspect_ratio > 10.0:  # Very wide range
            return False
            
        # Check size relative to image size if available
        if image_shape is not None:
            img_h, img_w = image_shape[:2]
            rel_w = width / img_w
            rel_h = height / img_h
            
            # Instruments and valves must be small
            if any(k in label_lower for k in ['instrument', 'sensor', 'tag', 'controller', 'transmitter', 'indicator', 'gauge']):
                if rel_w > 0.18 or rel_h > 0.18:
                    return False
            elif 'valve' in label_lower:
                if rel_w > 0.18 or rel_h > 0.18:
                    return False
            # Pumps and motors must be small to medium
            elif any(k in label_lower for k in ['pump', 'motor']):
                if rel_w > 0.30 or rel_h > 0.30:
                    return False
            # Vessels/tanks can be larger, but should not span almost the entire image
            elif any(k in label_lower for k in ['tank', 'vessel', 'reactor']):
                if rel_w > 0.90 or rel_h > 0.90:
                    return False
            
        return True
    
    def _improve_label_classification(self, label: str, bbox: List) -> str:
        """Improve label classification with expert-level accuracy rules"""
        label_lower = label.lower()
        
        # Expert classification rules for maximum accuracy
        # Priority order based on specificity
        
        # Check for valve-specific keywords first (highest priority)
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
