import os
import sys

# Set your HF token here for testing (DO NOT COMMIT THIS FILE!)
# Get a free token from: https://huggingface.co/settings/tokens
HF_TOKEN = "hf_YOUR_TOKEN_HERE"  # Replace with your actual token

if HF_TOKEN == "hf_YOUR_TOKEN_HERE":
    print("❌ Please set your actual Hugging Face token in this file!")
    print("Get one free from: https://huggingface.co/settings/tokens")
    sys.exit(1)

# Set the token in environment
os.environ["HF_API_TOKEN"] = HF_TOKEN

# Now test the deepfake detector
from deepfake_analyzer_light import get_detector
from PIL import Image
import io

# Create test image
img = Image.new('RGB', (224, 224), color='blue')
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')

print("Testing with HF Token...")
print("-" * 40)
detector = get_detector()
result = detector.analyze_image(img_bytes.getvalue())

print(f"✅ Result:")
print(f"  Is deepfake: {result['is_deepfake']}")
print(f"  Confidence: {result['confidence']:.2%}")
print(f"  Model: {result.get('details', {}).get('model', 'fallback')}")

if 'all_predictions' in result.get('details', {}):
    print(f"  Raw predictions: {result['details']['all_predictions']}")
