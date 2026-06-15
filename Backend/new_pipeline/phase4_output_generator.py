"""
Phase 4: Final Output
Generates structured output from the complete pipeline analysis
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from new_pipeline.config import *
from new_pipeline.utils import *


class OutputGenerator:
    """Generate final structured output from pipeline analysis"""
    
    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_final_output(self, phase1_results: Dict, phase2_results: Dict, 
                             phase3_results: List[Dict]) -> Dict[str, Any]:
        """
        Generate final structured output from all phases
        
        Args:
            phase1_results: Results from reference image processing
            phase2_results: Results from P&ID image analysis
            phase3_results: Results from subtype matching
        
        Returns:
            Complete structured output
        """
        print("\nGenerating final output...")
        
        timestamp = datetime.now().isoformat()
        
        # Build component inventory
        component_inventory = self._build_component_inventory(phase3_results)
        
        # Build connection map
        connection_map = self._build_connection_map(phase2_results, phase3_results)
        
        # Build text annotations
        text_annotations = self._build_text_annotations(phase2_results)
        
        # Build statistics
        statistics = self._build_statistics(phase2_results, phase3_results)
        
        # Build quality metrics
        quality_metrics = self._build_quality_metrics(phase2_results, phase3_results)
        
        # Compile final output
        final_output = {
            'metadata': {
                'timestamp': timestamp,
                'pipeline_version': '1.0',
                'image_path': phase2_results.get('image_path', ''),
                'image_name': phase2_results.get('image_name', '')
            },
            'reference_library': {
                'total_symbols': phase1_results.get('total_symbols', 0),
                'categories': phase1_results.get('categories', {})
            },
            'component_inventory': component_inventory,
            'connection_map': connection_map,
            'text_annotations': text_annotations,
            'statistics': statistics,
            'quality_metrics': quality_metrics,
            'pipeline_results': {
                'phase1': self._summarize_phase1(phase1_results),
                'phase2': self._summarize_phase2(phase2_results),
                'phase3': self._summarize_phase3(phase3_results)
            }
        }
        
        return final_output
    
    def _build_component_inventory(self, phase3_results: List[Dict]) -> Dict[str, Any]:
        """Build component inventory from matched components"""
        inventory = {
            'total_components': len(phase3_results),
            'by_category': {},
            'by_subtype': {},
            'components': []
        }
        
        for component in phase3_results:
            category = component.get('category', self._infer_category(component['label']))
            subtype = component.get('matched_subtype', component['label'])
            
            # Add to category count
            if category not in inventory['by_category']:
                inventory['by_category'][category] = 0
            inventory['by_category'][category] += 1
            
            # Add to subtype count
            if subtype not in inventory['by_subtype']:
                inventory['by_subtype'][subtype] = 0
            inventory['by_subtype'][subtype] += 1
            
            # Add component details
            component_entry = {
                'id': f"{category}_{subtype}_{len(inventory['components'])}",
                'label': component['label'],
                'category': category,
                'subtype': subtype,
                'confidence': component.get('match_confidence', 0.0),
                'match_method': component.get('match_method', 'unknown'),
                'bbox': component['bbox'],
                'position': {
                    'center': [
                        int((component['bbox'][0] + component['bbox'][2]) / 2),
                        int((component['bbox'][1] + component['bbox'][3]) / 2)
                    ],
                    'size': [
                        component['bbox'][2] - component['bbox'][0],
                        component['bbox'][3] - component['bbox'][1]
                    ]
                }
            }
            
            # Add reference image if available
            if 'reference_image' in component:
                component_entry['reference_image'] = component['reference_image']
            
            # Add Gemini reasoning if available
            if 'gemini_reasoning' in component:
                component_entry['gemini_reasoning'] = component['gemini_reasoning']
            
            inventory['components'].append(component_entry)
        
        return inventory
    
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
    
    def _build_connection_map(self, phase2_results: Dict, phase3_results: List[Dict]) -> Dict[str, Any]:
        """Build connection map from heuristic results"""
        heuristic_results = phase2_results.get('heuristic_results', {})
        connections = heuristic_results.get('connections', [])
        
        connection_map = {
            'total_connections': len(connections),
            'connections': []
        }
        
        # Map component labels to IDs
        component_id_map = {}
        for i, component in enumerate(phase3_results):
            label = component['label']
            component_id_map[label] = f"component_{i}"
        
        for connection in connections:
            connection_entry = {
                'id': f"connection_{len(connection_map['connections'])}",
                'source': connection.get('component_1', ''),
                'target': connection.get('component_2', ''),
                'connection_type': connection.get('connection_type', 'pipe'),
                'source_id': component_id_map.get(connection.get('component_1', ''), ''),
                'target_id': component_id_map.get(connection.get('component_2', ''), '')
            }
            connection_map['connections'].append(connection_entry)
        
        return connection_map
    
    def _build_text_annotations(self, phase2_results: Dict) -> Dict[str, Any]:
        """Build text annotations from OCR results"""
        ocr_results = phase2_results.get('ocr_results', {})
        heuristic_results = phase2_results.get('heuristic_results', {})
        
        text_annotations = {
            'total_text': ocr_results.get('total_text', 0),
            'instrument_tags': [],
            'equipment_names': [],
            'other_text': []
        }
        
        # Add instrument tags
        for tag in ocr_results.get('instrument_tags', []):
            text_annotations['instrument_tags'].append({
                'text': tag['text'],
                'bbox': tag['bbox'],
                'confidence': tag['confidence'],
                'validated': True
            })
        
        # Add equipment names
        for name in ocr_results.get('equipment_names', []):
            text_annotations['equipment_names'].append({
                'text': name['text'],
                'bbox': name['bbox'],
                'confidence': name['confidence'],
                'validated': True
            })
        
        # Add other validated text
        for text_item in heuristic_results.get('filtered_text', []):
            if text_item not in text_annotations['instrument_tags'] and text_item not in text_annotations['equipment_names']:
                text_annotations['other_text'].append({
                    'text': text_item['text'],
                    'bbox': text_item['bbox'],
                    'confidence': text_item['confidence'],
                    'validated': text_item.get('validated', False)
                })
        
        return text_annotations
    
    def _build_statistics(self, phase2_results: Dict, phase3_results: List[Dict]) -> Dict[str, Any]:
        """Build statistics summary"""
        opencv_results = phase2_results.get('opencv_results', {})
        dino_results = phase2_results.get('dino_results', {})
        
        statistics = {
            'detection': {
                'total_lines': opencv_results.get('total_lines', 0),
                'total_junctions': opencv_results.get('total_junctions', 0),
                'total_detections': dino_results.get('total_detections', 0),
                'total_validated': phase2_results.get('heuristic_results', {}).get('total_validated', 0)
            },
            'matching': {
                'total_matched': len(phase3_results),
                'template_matched': sum(1 for c in phase3_results if c.get('match_method') == 'template_matching'),
                'gemini_matched': sum(1 for c in phase3_results if c.get('match_method') == 'gemini'),
                'average_confidence': sum(c.get('match_confidence', 0) for c in phase3_results) / len(phase3_results) if phase3_results else 0
            }
        }
        
        return statistics
    
    def _build_quality_metrics(self, phase2_results: Dict, phase3_results: List[Dict]) -> Dict[str, Any]:
        """Build quality metrics"""
        quality_metrics = {
            'detection_quality': {
                'validation_rate': 0.0,
                'average_detection_confidence': 0.0
            },
            'matching_quality': {
                'high_confidence_matches': sum(1 for c in phase3_results if c.get('match_confidence', 0) >= 0.8),
                'medium_confidence_matches': sum(1 for c in phase3_results if 0.5 <= c.get('match_confidence', 0) < 0.8),
                'low_confidence_matches': sum(1 for c in phase3_results if c.get('match_confidence', 0) < 0.5)
            },
            'pipeline_efficiency': {
                'template_match_rate': 0.0,
                'gemini_fallback_rate': 0.0
            }
        }
        
        # Calculate validation rate
        total_detections = phase2_results.get('dino_results', {}).get('total_detections', 0)
        total_validated = phase2_results.get('heuristic_results', {}).get('total_validated', 0)
        if total_detections > 0:
            quality_metrics['detection_quality']['validation_rate'] = total_validated / total_detections
        
        # Calculate average detection confidence
        dino_confidences = phase2_results.get('dino_results', {}).get('confidences', [])
        if dino_confidences:
            quality_metrics['detection_quality']['average_detection_confidence'] = sum(dino_confidences) / len(dino_confidences)
        
        # Calculate match rates
        if phase3_results:
            total_matches = len(phase3_results)
            template_matches = sum(1 for c in phase3_results if c.get('match_method') == 'template_matching')
            gemini_matches = sum(1 for c in phase3_results if c.get('match_method') == 'gemini')
            
            quality_metrics['pipeline_efficiency']['template_match_rate'] = template_matches / total_matches
            quality_metrics['pipeline_efficiency']['gemini_fallback_rate'] = gemini_matches / total_matches
        
        return quality_metrics
    
    def _summarize_phase1(self, phase1_results: Dict) -> Dict[str, Any]:
        """Summarize Phase 1 results"""
        return {
            'status': 'completed',
            'total_symbols': phase1_results.get('total_symbols', 0),
            'categories': phase1_results.get('categories', {})
        }
    
    def _summarize_phase2(self, phase2_results: Dict) -> Dict[str, Any]:
        """Summarize Phase 2 results"""
        return {
            'status': 'completed',
            'opencv_features': phase2_results.get('opencv_results', {}).get('total_components', 0),
            'ocr_text': phase2_results.get('ocr_results', {}).get('total_text', 0),
            'florence_regions': phase2_results.get('florence_results', {}).get('total_regions', 0),
            'dino_detections': phase2_results.get('dino_results', {}).get('total_detections', 0),
            'validated_components': phase2_results.get('heuristic_results', {}).get('total_validated', 0)
        }
    
    def _summarize_phase3(self, phase3_results: List[Dict]) -> Dict[str, Any]:
        """Summarize Phase 3 results"""
        return {
            'status': 'completed',
            'total_matched': len(phase3_results),
            'template_matches': sum(1 for c in phase3_results if c.get('match_method') == 'template_matching'),
            'gemini_matches': sum(1 for c in phase3_results if c.get('match_method') == 'gemini'),
            'average_confidence': sum(c.get('match_confidence', 0) for c in phase3_results) / len(phase3_results) if phase3_results else 0
        }
    
    def save_output(self, final_output: Dict, output_filename: str = None) -> str:
        """Save final output to JSON file"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"pid_analysis_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        save_json(final_output, str(output_path))
        
        print(f"Final output saved to: {output_path}")
        return str(output_path)
    
    def generate_summary_report(self, final_output: Dict) -> str:
        """Generate human-readable summary report"""
        report = []
        report.append("=" * 80)
        report.append("P&ID ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Metadata
        report.append("METADATA")
        report.append("-" * 40)
        report.append(f"Timestamp: {final_output['metadata']['timestamp']}")
        report.append(f"Image: {final_output['metadata']['image_name']}")
        report.append("")
        
        # Component Inventory
        report.append("COMPONENT INVENTORY")
        report.append("-" * 40)
        report.append(f"Total Components: {final_output['component_inventory']['total_components']}")
        report.append("")
        report.append("By Category:")
        for category, count in final_output['component_inventory']['by_category'].items():
            report.append(f"  - {category}: {count}")
        report.append("")
        report.append("By Subtype:")
        for subtype, count in final_output['component_inventory']['by_subtype'].items():
            report.append(f"  - {subtype}: {count}")
        report.append("")
        
        # Connections
        report.append("CONNECTIONS")
        report.append("-" * 40)
        report.append(f"Total Connections: {final_output['connection_map']['total_connections']}")
        report.append("")
        
        # Text Annotations
        report.append("TEXT ANNOTATIONS")
        report.append("-" * 40)
        report.append(f"Total Text: {final_output['text_annotations']['total_text']}")
        report.append(f"Instrument Tags: {len(final_output['text_annotations']['instrument_tags'])}")
        report.append(f"Equipment Names: {len(final_output['text_annotations']['equipment_names'])}")
        report.append("")
        
        # Statistics
        report.append("STATISTICS")
        report.append("-" * 40)
        report.append(f"Lines Detected: {final_output['statistics']['detection']['total_lines']}")
        report.append(f"Junctions: {final_output['statistics']['detection']['total_junctions']}")
        report.append(f"Validated Components: {final_output['statistics']['detection']['total_validated']}")
        report.append(f"Average Match Confidence: {final_output['statistics']['matching']['average_confidence']:.2f}")
        report.append("")
        
        # Quality Metrics
        report.append("QUALITY METRICS")
        report.append("-" * 40)
        report.append(f"Validation Rate: {final_output['quality_metrics']['detection_quality']['validation_rate']:.2%}")
        report.append(f"Template Match Rate: {final_output['quality_metrics']['pipeline_efficiency']['template_match_rate']:.2%}")
        report.append(f"Gemini Fallback Rate: {final_output['quality_metrics']['pipeline_efficiency']['gemini_fallback_rate']:.2%}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_summary_report(self, final_output: Dict, output_filename: str = None) -> str:
        """Save summary report to text file"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"pid_summary_{timestamp}.txt"
        
        output_path = self.output_dir / output_filename
        report = self.generate_summary_report(final_output)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to: {output_path}")
        return str(output_path)
