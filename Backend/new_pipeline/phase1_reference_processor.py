"""
Phase 1: Reference Image Processing
Processes reference images to build a symbol library using Florence-2, Grounding DINO, and Gemini
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import json
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
import google.generativeai as genai
from new_pipeline.config import *
from new_pipeline.utils import *


class ReferenceImageProcessor:
    """Process reference images to build symbol library"""
    
    def __init__(self):
        self.florence_processor = None
        self.florence_model = None
        self.grounding_model = None
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
            device=FLORENCE_DEVICE
        )
        
        print("Initializing Gemini...")
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_client = genai.GenerativeModel(GEMINI_MODEL)
        
        print("All models loaded successfully!")
    
    def process_reference_image(self, image_path: str, category: str) -> Dict[str, Any]:
        """
        Process a single reference image through the complete pipeline
        
        Args:
            image_path: Path to reference image
            category: Category (pumps, valves, vessels, motors, pipes)
        
        Returns:
            Dictionary containing processed symbols and metadata
        """
        print(f"\nProcessing reference image: {image_path}")
        print(f"Category: {category}")
        
        # Load image
        image = load_image(image_path)
        image_name = Path(image_path).stem
        
        # Step 1: Florence-2 Dense Region Caption
        print("Step 1: Running Florence-2 dense region caption...")
        florence_results = self._florence_dense_region_caption(image)
        
        # Step 2: Grounding DINO for precise bounding boxes
        print("Step 2: Running Grounding DINO for precise bounding boxes...")
        dino_results = self._grounding_dino_detection(image, florence_results['labels'])
        
        # Step 3: OpenCV crop and save symbols
        print("Step 3: Cropping and saving individual symbols...")
        cropped_symbols = self._crop_and_save_symbols(
            image, dino_results['boxes'], dino_results['labels'], 
            category, image_name
        )
        
        # Step 4: Gemini verification
        print("Step 4: Running Gemini verification...")
        verified_symbols = self._gemini_verify_symbols(cropped_symbols, category)
        
        # Save results
        results = {
            'image_path': image_path,
            'category': category,
            'florence_results': florence_results,
            'dino_results': dino_results,
            'verified_symbols': verified_symbols,
            'total_symbols': len(verified_symbols)
        }
        
        return results
    
    def _florence_dense_region_caption(self, image: np.ndarray) -> Dict[str, Any]:
        """Run Florence-2 dense region caption"""
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare prompt
        prompt = "<DENSE_REGION_CAPTION>"
        
        # Process
        inputs = self.florence_processor(text=prompt, images=pil_image, return_tensors="pt").to(FLORENCE_DEVICE)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        # Parse results
        result = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result = self.florence_processor.post_process_generation(result, task="<DENSE_REGION_CAPTION>", image_size=pil_image.size)
        
        # Extract labels and bboxes
        labels = []
        bboxes = []
        for item in result.get('<DENSE_REGION_CAPTION>', []):
            labels.append(item.get('caption', ''))
            bboxes.append(item.get('bbox', []))
        
        return {
            'labels': labels,
            'bboxes': bboxes,
            'raw_result': result
        }
    
    def _grounding_dino_detection(self, image: np.ndarray, labels: List[str]) -> Dict[str, Any]:
        """Run Grounding DINO with Florence labels as prompts"""
        boxes = []
        confidences = []
        detected_labels = []
        
        # Process each label
        for label in labels:
            if not label or label.strip() == "":
                continue
            
            # Run detection
            boxes_filter, logits, phrases = predict(
                model=self.grounding_model,
                image=image,
                caption=label,
                box_threshold=GROUNDING_DINO_BOX_THRESHOLD,
                text_threshold=GROUNDING_DINO_TEXT_THRESHOLD,
                device=FLORENCE_DEVICE
            )
            
            # Add results
            for box, logit, phrase in zip(boxes_filter, logits, phrases):
                boxes.append(box.tolist())
                confidences.append(float(logit))
                detected_labels.append(phrase)
        
        return {
            'boxes': boxes,
            'confidences': confidences,
            'labels': detected_labels
        }
    
    def _crop_and_save_symbols(self, image: np.ndarray, boxes: List[List[int]], 
                               labels: List[str], category: str, image_name: str) -> List[Dict[str, Any]]:
        """Crop and save individual symbols"""
        symbols = []
        
        # Create category directory
        category_dir = REFERENCE_LIBRARY_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            # Crop
            cropped = crop_image(image, box)
            
            # Generate filename
            safe_label = label.replace(' ', '_').replace('/', '_').lower()
            filename = f"{image_name}_{i}_{safe_label}.png"
            output_path = category_dir / filename
            
            # Save
            save_image(cropped, str(output_path))
            
            symbols.append({
                'label': label,
                'bbox': box,
                'image_path': str(output_path),
                'category': category
            })
        
        return symbols
    
    def _gemini_verify_symbols(self, symbols: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
        """Verify and correct symbol labels using Gemini"""
        verified_symbols = []
        
        for symbol in symbols:
            # Load cropped image
            cropped_image = load_image(symbol['image_path'])
            
            # Prepare prompt
            prompt = f"""
            Analyze this P&ID symbol image and verify its classification.
            
            Current label: {symbol['label']}
            Category: {category}
            
            Please:
            1. Confirm if the label is correct
            2. If incorrect, provide the correct subtype
            3. Provide confidence score (0-1)
            
            Valid subtypes for {category}: {REFERENCE_CATEGORIES.get(category, [])}
            
            Return your answer in this exact JSON format:
            {{
                "verified_label": "correct_subtype",
                "is_correct": true/false,
                "confidence": 0.95,
                "reasoning": "brief explanation"
            }}
            """
            
            try:
                # Convert image to base64
                base64_image = image_to_base64(cropped_image)
                
                # Call Gemini
                response = self.gemini_client.generate_content([
                    prompt,
                    {"mime_type": "image/png", "data": base64_image}
                ])
                
                # Parse response
                result_text = response.text
                # Extract JSON from response
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    result_json = json.loads(result_text[json_start:json_end])
                    
                    # Update symbol
                    verified_symbol = symbol.copy()
                    verified_symbol['verified_label'] = result_json.get('verified_label', symbol['label'])
                    verified_symbol['is_correct'] = result_json.get('is_correct', True)
                    verified_symbol['confidence'] = result_json.get('confidence', 1.0)
                    verified_symbol['reasoning'] = result_json.get('reasoning', '')
                    
                    verified_symbols.append(verified_symbol)
                else:
                    # If JSON parsing fails, keep original
                    verified_symbol = symbol.copy()
                    verified_symbol['verified_label'] = symbol['label']
                    verified_symbol['is_correct'] = True
                    verified_symbol['confidence'] = 0.5
                    verified_symbol['reasoning'] = 'JSON parsing failed'
                    verified_symbols.append(verified_symbol)
                    
            except Exception as e:
                print(f"Error verifying symbol {symbol['label']}: {e}")
                # Keep original on error
                verified_symbol = symbol.copy()
                verified_symbol['verified_label'] = symbol['label']
                verified_symbol['is_correct'] = True
                verified_symbol['confidence'] = 0.5
                verified_symbol['reasoning'] = f'Error: {str(e)}'
                verified_symbols.append(verified_symbol)
        
        return verified_symbols
    
    def process_batch(self, image_paths: List[str], category: str) -> List[Dict[str, Any]]:
        """Process multiple reference images"""
        results = []
        for image_path in image_paths:
            result = self.process_reference_image(image_path, category)
            results.append(result)
        return results
    
    def save_reference_library_manifest(self, results: List[Dict[str, Any]]) -> str:
        """Save manifest of reference library"""
        manifest = {
            'total_images': len(results),
            'total_symbols': sum(r['total_symbols'] for r in results),
            'categories': {},
            'symbols': []
        }
        
        for result in results:
            category = result['category']
            if category not in manifest['categories']:
                manifest['categories'][category] = 0
            manifest['categories'][category] += result['total_symbols']
            manifest['symbols'].extend(result['verified_symbols'])
        
        manifest_path = REFERENCE_LIBRARY_DIR / "manifest.json"
        save_json(manifest, str(manifest_path))
        
        return str(manifest_path)
