"""
Example usage script for the new 4-phase P&ID analysis pipeline
"""
from pathlib import Path
from new_pipeline.pipeline_orchestrator import PipelineOrchestrator


def example_phase1_only():
    """Example: Run only Phase 1 to build reference library"""
    print("Example: Building Reference Library (Phase 1 only)")
    
    orchestrator = PipelineOrchestrator()
    orchestrator.initialize_phase1()
    
    # Define your reference images
    reference_images = {
        'pumps': [
            'reference_images/pumps/centrifugal.png',
            'reference_images/pumps/gear.png'
        ],
        'valves': [
            'reference_images/valves/gate.png',
            'reference_images/valves/globe.png',
            'reference_images/valves/ball.png'
        ],
        'vessels': [
            'reference_images/vessels/vertical.png',
            'reference_images/vessels/horizontal.png'
        ]
    }
    
    # Run Phase 1
    phase1_results = orchestrator.run_phase1(reference_images)
    
    print(f"\nReference library built with {phase1_results['total_symbols']} symbols")


def example_analysis_only():
    """Example: Run analysis only (Phases 2-4) assuming reference library exists"""
    print("Example: P&ID Analysis (Phases 2-4 only)")
    
    orchestrator = PipelineOrchestrator()
    
    # Run analysis on a P&ID image
    pid_image_path = 'test_images/pid_diagram.png'
    
    results = orchestrator.run_analysis_only(
        pid_image_path=pid_image_path,
        output_filename='pid_analysis_results.json'
    )
    
    print(f"\nAnalysis complete!")
    print(f"Components detected: {results['component_inventory']['total_components']}")


def example_full_pipeline():
    """Example: Run complete pipeline from scratch"""
    print("Example: Complete Pipeline (Phases 1-4)")
    
    orchestrator = PipelineOrchestrator()
    
    # Define reference images
    reference_images = {
        'pumps': [
            'reference_images/pumps/centrifugal.png',
            'reference_images/pumps/gear.png'
        ],
        'valves': [
            'reference_images/valves/gate.png',
            'reference_images/valves/globe.png'
        ]
    }
    
    # Run full pipeline
    results = orchestrator.run_full_pipeline(
        reference_images=reference_images,
        pid_image_path='test_images/pid_diagram.png',
        output_filename='complete_pipeline_results.json'
    )
    
    print(f"\nPipeline complete!")
    print(f"Total components: {results['component_inventory']['total_components']}")
    print(f"Average confidence: {results['statistics']['matching']['average_confidence']:.2f}")


def example_custom_workflow():
    """Example: Custom workflow with individual phase control"""
    print("Example: Custom Workflow")
    
    orchestrator = PipelineOrchestrator()
    
    # Initialize only what you need
    orchestrator.initialize_phase1()
    orchestrator.initialize_phase2()
    
    # Build reference library
    reference_images = {
        'pumps': ['reference_images/pumps/centrifugal.png']
    }
    phase1_results = orchestrator.run_phase1(reference_images)
    
    # Analyze multiple P&ID images
    pid_images = [
        'test_images/diagram1.png',
        'test_images/diagram2.png'
    ]
    
    for pid_image in pid_images:
        print(f"\nAnalyzing {pid_image}...")
        phase2_results = orchestrator.run_phase2(pid_image)
        
        # Initialize Phase 3 and 4 for each image
        orchestrator.initialize_phase3()
        orchestrator.initialize_phase4()
        
        phase3_results = orchestrator.run_phase3(pid_image, phase2_results)
        
        final_output = orchestrator.run_phase4(
            phase1_results, phase2_results, phase3_results,
            output_filename=f"analysis_{Path(pid_image).stem}.json"
        )


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # example_phase1_only()
    # example_analysis_only()
    # example_full_pipeline()
    # example_custom_workflow()
    
    print("\nPlease uncomment the example function you want to run in the script.")
