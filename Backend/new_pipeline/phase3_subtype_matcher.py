"""
Phase 3: Subtype Matching
Matches detected components to reference library using OpenCV template matching and Gemini fallback
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import json
import google.generativeai as genai
from new_pipeline.config import *
from new_pipeline.utils import *


class SubtypeMatcher:
    """Match detected components to reference library subtypes"""
    
    def __init__(self):
        self.gemini_client = None
        self.reference_library = {}
        self._load_gemini()
        self._load_reference_library()
    
    def _load_gemini(self):
        """Initialize Gemini client"""
        print("Initializing Gemini...")
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_client = genai.GenerativeModel(GEMINI_MODEL)
    
    def _load_reference_library(self):
        """Load reference library manifest and images"""
        manifest_path = REFERENCE_LIBRARY_DIR / "manifest.json"
        
        if manifest_path.exists():
            manifest = load_json(str(manifest_path))
            print(f"Loaded reference library with {manifest['total_symbols']} symbols")
            
            # Group symbols by category
            for symbol in manifest['symbols']:
                category = symbol.get('category', 'unknown')
                if category not in self.reference_library:
                    self.reference_library[category] = []
                self.reference_library[category].append(symbol)
        else:
            print("Warning: Reference library manifest not found. Please run Phase 1 first.")
    
    def match_components(self, pid_image: np.ndarray, detected_components: List[Dict]) -> List[Dict]:
        """
        Match detected components to reference library
        
        Args:
            pid_image: Original P&ID image
            detected_components: List of detected components from Phase 2
        
        Returns:
            List of components with matched subtypes
        """
        print(f"\nMatching {len(detected_components)} components to reference library...")
        
        matched_components = []
        
        for component in detected_components:
            # Crop component from P&ID image
            bbox = component['bbox']
            cropped_component = crop_image(pid_image, bbox)
            
            # Try template matching first
            template_match_result = self._template_matching(cropped_component, component['label'])
            
            if template_match_result['confidence'] >= TEMPLATE_MATCH_THRESHOLD:
                # High confidence match
                matched_component = component.copy()
                matched_component['matched_subtype'] = template_match_result['subtype']
                matched_component['match_confidence'] = template_match_result['confidence']
                matched_component['match_method'] = 'template_matching'
                matched_component['reference_image'] = template_match_result['reference_image']
                matched_components.append(matched_component)
                print(f"  - {component['label']} -> {template_match_result['subtype']} (confidence: {template_match_result['confidence']:.2f})")
            else:
                # Low confidence - pass to Gemini
                print(f"  - {component['label']} -> Low template confidence, using Gemini...")
                gemini_match_result = self._gemini_matching(
                    pid_image, cropped_component, component, template_match_result
                )
                
                matched_component = component.copy()
                matched_component['matched_subtype'] = gemini_match_result['subtype']
                matched_component['match_confidence'] = gemini_match_result['confidence']
                matched_component['match_method'] = 'gemini'
                matched_component['gemini_reasoning'] = gemini_match_result['reasoning']
                matched_components.append(matched_component)
        
        return matched_components
    
    def _template_matching(self, component_image: np.ndarray, label: str) -> Dict[str, Any]:
        """
        Match component using OpenCV template matching against reference library
        
        Returns:
            Dictionary with best match info
        """
        best_match = {
            'subtype': label,
            'confidence': 0.0,
            'reference_image': None
        }
        
        # Determine category from label
        category = self._infer_category(label)
        
        if category not in self.reference_library:
            return best_match
        
        # Try matching against all reference symbols in category
        for symbol in self.reference_library[category]:
            reference_path = symbol['image_path']
            
            if not Path(reference_path).exists():
                continue
            
            # Load reference image
            reference_image = load_image(reference_path)
            
            # Resize to match component
            if reference_image.shape != component_image.shape:
                reference_image = cv2.resize(reference_image, 
                                           (component_image.shape[1], component_image.shape[0]))
            
            # Perform template matching
            result = cv2.matchTemplate(component_image, reference_image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_match['confidence']:
                best_match = {
                    'subtype': symbol['verified_label'],
                    'confidence': float(max_val),
                    'reference_image': reference_path
                }
        
        return best_match
    
    def _infer_category(self, label: str) -> str:
        """Infer category from label"""
        label_lower = label.lower()
        
        category_keywords = {
            'pumps': ['pump', 'centrifugal', 'gear', 'reciprocating'],
            'valves': ['valve', 'gate', 'globe', 'ball', 'butterfly', 'check'],
            'vessels': ['vessel', 'tank', 'reactor', 'separator', 'vertical', 'horizontal'],
            'motors': ['motor', 'induction', 'synchronous', 'servo'],
            'pipes': ['pipe', 'process', 'utility', 'instrument']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in label_lower for keyword in keywords):
                return category
        
        return 'unknown'
    
    def _gemini_matching(self, pid_image: np.ndarray, component_image: np.ndarray,
                        component: Dict, template_result: Dict) -> Dict[str, Any]:
        """
        Use Gemini to determine subtype when template matching confidence is low
        
        Receives:
        - P&ID image
        - Component cropped image
        - Component metadata
        - Template matching results
        """
        # Prepare context
        category = self._infer_category(component['label'])
        valid_subtypes = REFERENCE_CATEGORIES.get(category, [])
        
        # Load reference images for context
        reference_context = []
        if category in self.reference_library:
            for symbol in self.reference_library[category][:5]:  # Limit to 5 for context
                if Path(symbol['image_path']).exists():
                    ref_img = load_image(symbol['image_path'])
                    ref_base64 = image_to_base64(ref_img)
                    reference_context.append({
                        'subtype': symbol['verified_label'],
                        'image': ref_base64
                    })
        
        # Prepare prompt
        prompt = f"""
        Analyze this P&ID component and determine its exact subtype.
        
        Component Information:
        - Detected Label: {component['label']}
        - Category: {category}
        - Template Matching Confidence: {template_result['confidence']:.2f}
        - Template Matched Subtype: {template_result['subtype']}
        
        Valid Subtypes for {category}: {', '.join(valid_subtypes)}
        
        Pipeline Context:
        - This component was detected in a P&ID diagram
        - Template matching had low confidence (< {TEMPLATE_MATCH_THRESHOLD})
        - Please analyze the component shape and features
        - Compare with reference symbols provided below
        
        Reference Symbols:
        """
        
        for ref in reference_context:
            prompt += f"\n- {ref['subtype']}: [reference image]"
        
        prompt += """
        
        Please provide:
        1. The most likely subtype
        2. Confidence score (0-1)
        3. Reasoning for your choice
        
        Return your answer in this exact JSON format:
        {
            "subtype": "exact_subtype",
            "confidence": 0.85,
            "reasoning": "detailed explanation"
        }
        """
        
        try:
            # Convert images to base64
            component_base64 = image_to_base64(component_image)
            pid_base64 = image_to_base64(pid_image)
            
            # Prepare content
            content = [prompt]
            
            # Add component image
            content.append({
                "mime_type": "image/png",
                "data": component_base64
            })
            
            # Add reference images
            for ref in reference_context:
                content.append({
                    "mime_type": "image/png",
                    "data": ref['image']
                })
            
            # Call Gemini
            response = self.gemini_client.generate_content(content)
            
            # Parse response
            result_text = response.text
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                result_json = json.loads(result_text[json_start:json_end])
                
                return {
                    'subtype': result_json.get('subtype', component['label']),
                    'confidence': result_json.get('confidence', template_result['confidence']),
                    'reasoning': result_json.get('reasoning', 'No reasoning provided')
                }
            else:
                # Fallback to template result
                return {
                    'subtype': template_result['subtype'],
                    'confidence': template_result['confidence'],
                    'reasoning': 'JSON parsing failed, using template match result'
                }
                
        except Exception as e:
            print(f"Error in Gemini matching: {e}")
            # Fallback to template result
            return {
                'subtype': template_result['subtype'],
                'confidence': template_result['confidence'],
                'reasoning': f'Error: {str(e)}'
            }
    
    def batch_match(self, pid_image: np.ndarray, detected_components: List[Dict]) -> List[Dict]:
        """Batch match multiple components (same as match_components but for clarity)"""
        return self.match_components(pid_image, detected_components)
