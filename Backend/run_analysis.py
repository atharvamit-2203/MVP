import sys
import json
from pathlib import Path
# Usage: python run_analysis.py <image_path>
if len(sys.argv) < 2:
    print(json.dumps({"error": "missing_image_path"}))
    sys.exit(2)
img_path = sys.argv[1]
try:
    from PIL import Image
    repo_root = Path('.').resolve()
    backend_dir = repo_root / 'Backend' / 'Backend'
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    import importlib
    import local_detection
    importlib.reload(local_detection)
    img = Image.open(img_path).convert('RGB')
    result = local_detection.analyze_pid_image(img)
    print(json.dumps(result, ensure_ascii=False))
except Exception as e:
    import traceback
    print(json.dumps({"error": str(e), "trace": traceback.format_exc()}))
    sys.exit(1)
