import requests
import base64
import os
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

# Test different models
models = [
    "dima806/deepfake_vs_real_image_detection",  # Most popular
    "prithivMLmods/deepfake-detector-model-v1",  # Current one
    "prithivMLmods/Deep-Fake-Detector-v2-Model",  # v2 model
]

# Get token
token = os.getenv("HF_API_TOKEN")
if not token:
    print("HF_API_TOKEN not found!")
    exit(1)

# Create a simple test image
img = Image.new('RGB', (224, 224), color='red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

# Test each model
for model in models:
    print(f"\nTesting model: {model}")
    print("-" * 60)
    
    # Method 1: Direct binary data (most common for image models)
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.post(url, headers=headers, data=img_byte_arr)
    print(f"Binary data response: {response.status_code}")
    if response.status_code == 200:
        print(f"Result: {response.json()}")
    else:
        print(f"Error: {response.text[:200]}")
