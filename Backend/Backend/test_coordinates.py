"""Test coordinate generation for Ignition Vision Client compatibility."""
import sys
from pathlib import Path
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_detection

# Test with the P&ID image
image_path = Path(__file__).resolve().parents[2] / "70642920-a5c7-41df-8e90-26d479e7b4b3.png"

if not image_path.exists():
    print(f"Error: Test image not found at {image_path}")
    sys.exit(1)

print(f"Testing coordinate generation with: {image_path}")
print("=" * 80)

# Load image
image = Image.open(image_path).convert("RGB")
print(f"Image size: {image.size}")
print()

# Run detection (not fast mode to use proper diagram complexity detection)
result = local_detection.analyze_pid_image(image, fast_mode=False)

print(f"Counts: {result['counts']}")
print(f"Industry: {result['industry']}")
print(f"Number of detections: {len(result['detections'])}")
print()

# Get coordinates
coordinates = result['coordinates']
print(f"Canvas size: {coordinates['props']['defaultSize']}")
print(f"Number of components in coordinates: {len(coordinates['root']['children'])}")
print()

# Show first few components with their coordinates
print("Sample components with coordinates:")
print("-" * 80)
for i, child in enumerate(coordinates['root']['children'][:5]):
    print(f"Component {i+1}:")
    print(f"  Name: {child['meta']['name']}")
    print(f"  Type: {child['type']}")
    print(f"  Position: x={child['position']['x']:.2f}, y={child['position']['y']:.2f}, "
          f"width={child['position']['width']:.2f}, height={child['position']['height']:.2f}")
    print()

# Save coordinates to JSON for inspection
import json
output_path = Path(__file__).parent / "test_coordinates_output.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(coordinates, f, indent=2)

print(f"Full coordinates saved to: {output_path}")
print("=" * 80)
print("Test completed successfully!")
