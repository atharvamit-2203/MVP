import sys
import json
import os
from PIL import Image
import traceback

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\aadeshpande\Downloads\f046bdfb-350c-49e6-80bd-c629d4983682.png"

try:
    from local_detection import analyze_pid_image
except Exception:
    # try package import
    from .local_detection import analyze_pid_image  # type: ignore

img = Image.open(IMAGE_PATH)
print('=== RUN_SAMPLE_DETECT START ===')
try:
    fast_mode = os.getenv("FAST_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
    result = analyze_pid_image(img, fast_mode=fast_mode)
    print(json.dumps(result, indent=2))
except Exception:
    print('Exception during analyze_pid_image:')
    traceback.print_exc()
    raise
finally:
    print('=== RUN_SAMPLE_DETECT END ===')
