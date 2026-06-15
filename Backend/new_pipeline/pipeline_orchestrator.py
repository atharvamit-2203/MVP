"""
Main Pipeline Orchestrator
Coordinates all 4 phases of the P&ID analysis pipeline
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from new_pipeline.config import *
from new_pipeline.utils import *
from new_pipeline.phase1_reference_processor import ReferenceImageProcessor
from new_pipeline.phase2_pid_analyzer import PIDImageAnalyzer
from new_pipeline.phase3_subtype_matcher import SubtypeMatcher
from new_pipeline.phase4_output_generator import OutputGenerator


class PipelineOrchestrator:
    """Main orchestrator for the 4-phase P&ID analysis pipeline"""
    
    def __init__(self):
        self.phase1_processor = None
        self.phase2_analyzer = None
        self.phase3_matcher = None
        self.phase4_generator = None
        self.reference_library_loaded = False
    
    def initialize_phase1(self):
        """Initialize Phase 1 processor"""
        print("Initializing Phase 1: Reference Image Processor...")
        self.phase1_processor = ReferenceImageProcessor()
    
    def initialize_phase2(self):
        """Initialize Phase 2 analyzer"""
        print("Initializing Phase 2: P&ID Image Analyzer...")
        self.phase2_analyzer = PIDImageAnalyzer()
    
    def initialize_phase3(self):
        """Initialize Phase 3 matcher"""
        print("Initializing Phase 3: Subtype Matcher...")
        self.phase3_matcher = SubtypeMatcher()
        self.reference_library_loaded = True
    
    def initialize_phase4(self):
        """Initialize Phase 4 generator"""
        print("Initializing Phase 4: Output Generator...")
        self.phase4_generator = OutputGenerator()
    
    def initialize_all(self):
        """Initialize all phases"""
        self.initialize_phase1()
        self.initialize_phase2()
        self.initialize_phase3()
        self.initialize_phase4()
        print("All phases initialized successfully!")
    
    def run_phase1(self, reference_images: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Run Phase 1: Reference Image Processing
        
        Args:
            reference_images: Dictionary mapping categories to lists of image paths
                e.g., {'pumps': ['path1.png', 'path2.png'], 'valves': ['path3.png']}
        
        Returns:
            Phase 1 results summary
        """
        print("\n" + "=" * 80)
        print("PHASE 1: REFERENCE IMAGE PROCESSING")
        print("=" * 80)
        
        if self.phase1_processor is None:
            self.initialize_phase1()
        
        all_results = []
        
        for category, image_paths in reference_images.items():
            print(f"\nProcessing category: {category}")
            results = self.phase1_processor.process_batch(image_paths, category)
            all_results.extend(results)
        
        # Save manifest
        manifest_path = self.phase1_processor.save_reference_library_manifest(all_results)
        
        # Build summary
        summary = {
            'status': 'completed',
            'total_images_processed': len(all_results),
            'total_symbols': sum(r['total_symbols'] for r in all_results),
            'categories': {category: sum(r['total_symbols'] for r in all_results if r['category'] == category) 
                          for category in reference_images.keys()},
            'manifest_path': manifest_path
        }
        
        print(f"\nPhase 1 Complete!")
        print(f"  - Images processed: {summary['total_images_processed']}")
        print(f"  - Total symbols extracted: {summary['total_symbols']}")
        print(f"  - Manifest saved to: {manifest_path}")
        
        return summary
    
    def run_phase2(self, pid_image_path: str) -> Dict[str, Any]:
        """
        Run Phase 2: P&ID Image Analysis
        
        Args:
            pid_image_path: Path to P&ID image to analyze
        
        Returns:
            Phase 2 results
        """
        print("\n" + "=" * 80)
        print("PHASE 2: P&ID IMAGE ANALYSIS")
        print("=" * 80)
        
        if self.phase2_analyzer is None:
            self.initialize_phase2()
        
        results = self.phase2_analyzer.analyze_pid_image(pid_image_path)
        
        print(f"\nPhase 2 Complete!")
        print(f"  - Lines detected: {results['opencv_results']['total_lines']}")
        print(f"  - Junctions found: {results['opencv_results']['total_junctions']}")
        print(f"  - Text detected: {results['ocr_results']['total_text']}")
        print(f"  - Instrument tags: {results['ocr_results']['total_instrument_tags']}")
        print(f"  - Florence regions: {results['florence_results']['total_regions']}")
        print(f"  - DINO detections: {results['dino_results']['total_detections']}")
        print(f"  - Validated components: {results['heuristic_results']['total_validated']}")
        
        return results
    
    def run_phase3(self, pid_image_path: str, phase2_results: Dict[str, Any]) -> List[Dict]:
        """
        Run Phase 3: Subtype Matching
        
        Args:
            pid_image_path: Path to P&ID image
            phase2_results: Results from Phase 2
        
        Returns:
            List of matched components with subtypes
        """
        print("\n" + "=" * 80)
        print("PHASE 3: SUBTYPE MATCHING")
        print("=" * 80)
        
        if self.phase3_matcher is None:
            self.initialize_phase3()
        
        # Load P&ID image
        pid_image = load_image(pid_image_path)
        
        # Get validated components from Phase 2
        detected_components = phase2_results['heuristic_results']['validated_components']
        
        if not detected_components:
            print("No components to match. Skipping Phase 3.")
            return []
        
        # Run matching
        matched_components = self.phase3_matcher.match_components(pid_image, detected_components)
        
        print(f"\nPhase 3 Complete!")
        print(f"  - Components matched: {len(matched_components)}")
        
        # Count match methods
        template_matches = sum(1 for c in matched_components if c.get('match_method') == 'template_matching')
        gemini_matches = sum(1 for c in matched_components if c.get('match_method') == 'gemini')
        
        print(f"  - Template matches: {template_matches}")
        print(f"  - Gemini matches: {gemini_matches}")
        
        return matched_components
    
    def run_phase4(self, phase1_summary: Dict, phase2_results: Dict, 
                  phase3_results: List[Dict], output_filename: str = None) -> Dict[str, Any]:
        """
        Run Phase 4: Final Output Generation
        
        Args:
            phase1_summary: Summary from Phase 1
            phase2_results: Results from Phase 2
            phase3_results: Results from Phase 3
            output_filename: Optional custom output filename
        
        Returns:
            Final structured output
        """
        print("\n" + "=" * 80)
        print("PHASE 4: FINAL OUTPUT GENERATION")
        print("=" * 80)
        
        if self.phase4_generator is None:
            self.initialize_phase4()
        
        # Generate final output
        final_output = self.phase4_generator.generate_final_output(
            phase1_summary, phase2_results, phase3_results
        )
        
        # Save output
        output_path = self.phase4_generator.save_output(final_output, output_filename)
        
        # Generate and save summary report
        report_path = self.phase4_generator.save_summary_report(final_output)
        
        print(f"\nPhase 4 Complete!")
        print(f"  - JSON output saved to: {output_path}")
        print(f"  - Summary report saved to: {report_path}")
        
        return final_output
    
    def run_full_pipeline(self, reference_images: Dict[str, List[str]], 
                         pid_image_path: str, output_filename: str = None) -> Dict[str, Any]:
        """
        Run the complete 4-phase pipeline
        
        Args:
            reference_images: Dictionary mapping categories to lists of reference image paths
            pid_image_path: Path to P&ID image to analyze
            output_filename: Optional custom output filename
        
        Returns:
            Complete pipeline results
        """
        print("\n" + "=" * 80)
        print("STARTING COMPLETE 4-PHASE PIPELINE")
        print("=" * 80)
        
        # Initialize all phases
        self.initialize_all()
        
        # Phase 1: Reference Image Processing
        phase1_summary = self.run_phase1(reference_images)
        
        # Phase 2: P&ID Image Analysis
        phase2_results = self.run_phase2(pid_image_path)
        
        # Phase 3: Subtype Matching
        phase3_results = self.run_phase3(pid_image_path, phase2_results)
        
        # Phase 4: Final Output
        final_output = self.run_phase4(phase1_summary, phase2_results, phase3_results, output_filename)
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 80)
        
        return final_output
    
    def run_analysis_only(self, pid_image_path: str, output_filename: str = None) -> Dict[str, Any]:
        """
        Run only Phases 2-4 (assuming Phase 1 reference library already exists)
        
        Args:
            pid_image_path: Path to P&ID image to analyze
            output_filename: Optional custom output filename
        
        Returns:
            Complete pipeline results
        """
        print("\n" + "=" * 80)
        print("RUNNING ANALYSIS ONLY (PHASES 2-4)")
        print("=" * 80)
        
        # Initialize phases 2-4
        self.initialize_phase2()
        self.initialize_phase3()
        self.initialize_phase4()
        
        # Load reference library summary
        manifest_path = REFERENCE_LIBRARY_DIR / "manifest.json"
        if manifest_path.exists():
            phase1_summary = load_json(str(manifest_path))
            phase1_summary['status'] = 'loaded_from_cache'
        else:
            print("Warning: Reference library not found. Results may be incomplete.")
            phase1_summary = {
                'status': 'not_available',
                'total_symbols': 0,
                'categories': {}
            }
        
        # Phase 2: P&ID Image Analysis
        phase2_results = self.run_phase2(pid_image_path)
        
        # Phase 3: Subtype Matching
        phase3_results = self.run_phase3(pid_image_path, phase2_results)
        
        # Phase 4: Final Output
        final_output = self.run_phase4(phase1_summary, phase2_results, phase3_results, output_filename)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        return final_output


def main():
    """Example usage of the pipeline orchestrator"""
    # Example: Run full pipeline
    orchestrator = PipelineOrchestrator()
    
    # Define reference images
    reference_images = {
        'pumps': ['path/to/pump1.png', 'path/to/pump2.png'],
        'valves': ['path/to/valve1.png', 'path/to/valve2.png'],
        'vessels': ['path/to/vessel1.png']
    }
    
    # Run full pipeline
    results = orchestrator.run_full_pipeline(
        reference_images=reference_images,
        pid_image_path='path/to/pid_image.png',
        output_filename='my_analysis.json'
    )
    
    print("\nPipeline execution complete!")
    print(f"Results saved to pipeline_outputs/")


if __name__ == "__main__":
    main()
