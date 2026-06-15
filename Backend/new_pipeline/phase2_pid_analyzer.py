"""
Phase 2: P&ID Image Analysis
Analyzes P&ID images using parallel processing with OpenCV, Tesseract OCR, Florence-2, and Grounding DINO
"""
import cv2
import numpy as np
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
        
        total_parallel_time = time.time() - start_time
        print(f"Parallel analysis completed in {total_parallel_time:.2f}s")
        
        # Use old pipeline's accurate results as primary, AI models for enhancement
        if old_pipeline_results:
            print(f"Old pipeline detected components: {old_pipeline_results.get('counts', {})}")
            print(f"DEBUG: Old pipeline result keys: {old_pipeline_results.keys()}")
            print(f"DEBUG: Old pipeline detections structure: {old_pipeline_results.get('detections', [])[:2] if old_pipeline_results.get('detections') else 'No detections'}")
            # Convert old pipeline results to expected format
            dino_boxes = []
            dino_confidences = []
            dino_labels = []
            
            for component in old_pipeline_results.get('detections', []):
                bbox = component.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    dino_boxes.append([x, y, x+w, y+h])  # Convert to xyxy format
                    dino_confidences.append(component.get('confidence', 0.5))
                    # Try multiple possible label fields
                    label = component.get('label', component.get('category', component.get('type', 'unknown')))
                    dino_labels.append(label)
                    print(f"DEBUG: Extracted component - label: {label}, confidence: {component.get('confidence', 0.5)}, bbox: {bbox}")
            
            # Enhance with Grounding DINO results (only non-overlapping high-confidence)
            for box, conf, label in zip(dino_results['boxes'], dino_results['confidences'], dino_results['labels']):
                overlaps = False
                for existing_box in dino_boxes:
                    iou = calculate_iou(box, existing_box)
                    if iou > 0.3:
                        overlaps = True
                        break
                if not overlaps and conf > 0.5:
                    dino_boxes.append(box)
                    dino_confidences.append(conf)
                    dino_labels.append(label)
            
            # Enhance with Florence-2 results (only non-overlapping)
            for box, label in zip(florence_results.get('bboxes', []), florence_results.get('labels', [])):
                if len(box) == 4:
                    overlaps = False
                    for existing_box in dino_boxes:
                        iou = calculate_iou(box, existing_box)
                        if iou > 0.3:
                            overlaps = True
                            break
                    if not overlaps:
                        dino_boxes.append(box)
                        dino_confidences.append(0.5)
                        dino_labels.append(label)
            
            dino_results = {
                'boxes': dino_boxes,
                'confidences': dino_confidences,
                'labels': dino_labels,
                'total_detections': len(dino_boxes)
            }
        else:
            # Fallback to AI models only
            dino_results = {
                'boxes': dino_results['boxes'],
                'confidences': dino_results['confidences'],
                'labels': dino_results['labels'],
                'total_detections': dino_results['total_detections']
            }
        
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
            prompt = "tank valve pump"
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
            
            # Use focused prompts for P&ID components only (tank, valve, pump)
            prompts_config = [
                {"prompt": "tank", "box_threshold": 0.30, "text_threshold": 0.25},
                {"prompt": "valve", "box_threshold": 0.30, "text_threshold": 0.25},
                {"prompt": "pump", "box_threshold": 0.30, "text_threshold": 0.25}
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
            
            if self._validate_component(detection):
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
    
    def _validate_component(self, detection: Dict) -> bool:
        """Validate component detection with fine-tuned thresholds for maximum accuracy"""
        label = detection.get('label', '').lower()
        confidence = detection.get('confidence', 0)
        
        # Fine-tuned component-specific confidence thresholds for maximum recall
        if label == 'motor':
            # Lowered threshold for motors
            if confidence < 0.25:
                return False
        elif label == 'tank':
            # Lowered threshold for tanks
            if confidence < 0.25:
                return False
        elif label == 'pump':
            # Lowered threshold for pumps
            if confidence < 0.25:
                return False
        elif label == 'valve':
            # Lowered threshold for valves
            if confidence < 0.15:
                return False
        elif label == 'instrument':
            # Lowered threshold for instruments
            if confidence < 0.20:
                return False
        else:
            # Default threshold for other components
            if confidence < 0.20:
                return False
        
        # Strict label validation
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
            
            # Apply component-specific confidence check (lowered for recall)
            label_threshold = 0.20  # Default
            if improved_label == 'motor':
                label_threshold = 0.25  # Lowered for motors
            elif improved_label == 'tank':
                label_threshold = 0.20  # Lowered for tanks
            elif improved_label == 'pump':
                label_threshold = 0.25  # Lowered for pumps
            elif improved_label == 'valve':
                label_threshold = 0.15  # Lowered for valves
            elif improved_label == 'instrument':
                label_threshold = 0.20  # Lowered for instruments
            
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
    
    def _validate_component_characteristics(self, bbox: List, label: str) -> bool:
        """Validate component based on visual characteristics (handles normalized coordinates)"""
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
        instrument_keywords = ['sensor', 'transmitter', 'indicator', 'gauge', 'controller', 'instrument', 
                             'pressure transmitter', 'temperature transmitter', 'flow indicator', 'level gauge']
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
