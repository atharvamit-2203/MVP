import sys
sys.path.insert(0, 'c:\\Users\\aadeshpande\\Desktop\\Sarla-Project\\Backend')
from Backend.local_detection import analyze_pid_image_async
from PIL import Image
import asyncio

# Test with a motor image
img = Image.open('annotations/20260601_043512_065940_motor.png')
result = asyncio.run(analyze_pid_image_async(img))
print('Detection result:', result.get('counts', {}))
