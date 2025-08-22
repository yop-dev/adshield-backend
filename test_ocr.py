#!/usr/bin/env python3
"""
Test OCR functionality
"""

import asyncio
import httpx
import os

async def test_ocr():
    """Test the OCR endpoint"""
    
    # Create a simple test image with text (if no sample image exists)
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple image with text
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        
        # Add some scam-like text
        text = "URGENT: Your account\nhas been suspended!\nClick here to verify\nyour identity NOW!"
        
        # Try to use a basic font
        try:
            d.text((10, 10), text, fill='red')
        except:
            # If font fails, use default
            d.text((10, 10), text, fill='red')
        
        # Save the test image
        test_image_path = "test_scam_image.png"
        img.save(test_image_path)
        print(f"✓ Created test image: {test_image_path}")
        
    except ImportError:
        print("! PIL not available, skipping test image creation")
        test_image_path = None
    
    # Test the OCR endpoint
    url = "http://localhost:8000/api/v1/text/extract"
    
    if test_image_path and os.path.exists(test_image_path):
        # Test with created image
        async with httpx.AsyncClient() as client:
            with open(test_image_path, 'rb') as f:
                files = {'file': ('test_scam.png', f, 'image/png')}
                response = await client.post(url, files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"\n✓ OCR API Response:")
                    print(f"  Extracted Text: {result.get('text', '')[:200]}...")
                else:
                    print(f"✗ OCR API failed: {response.status_code}")
                    print(f"  Response: {response.text}")
    else:
        print("! No test image available, OCR test skipped")
    
    # Check what OCR methods are available
    print("\n📋 OCR Methods Available:")
    
    # Check for Tesseract
    try:
        import pytesseract
        print("  ✓ pytesseract installed")
        # Check if tesseract binary is available
        try:
            import subprocess
            result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("  ✓ Tesseract binary found")
            else:
                print("  ✗ Tesseract binary not found - install from:")
                print("    https://github.com/UB-Mannheim/tesseract/wiki")
        except FileNotFoundError:
            print("  ✗ Tesseract binary not found - install from:")
            print("    https://github.com/UB-Mannheim/tesseract/wiki")
    except ImportError:
        print("  ✗ pytesseract not installed")
    
    # Check for EasyOCR
    try:
        import easyocr
        print("  ✓ easyocr installed")
    except ImportError:
        print("  ✗ easyocr not installed")
    
    print("\n💡 To enable better OCR:")
    print("  1. Install Tesseract binary from GitHub")
    print("  2. Or run: pip install easyocr")

if __name__ == "__main__":
    asyncio.run(test_ocr())
