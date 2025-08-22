#!/usr/bin/env python3
"""
Test the API endpoints
"""

import asyncio
import httpx
import json

async def test_text_analysis():
    """Test text analysis endpoint"""
    url = "http://localhost:8000/api/v1/text/analyze"
    
    test_text = "URGENT: Your account has been suspended. Click here immediately to verify your identity."
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"text": test_text},
            timeout=30
        )
        
        print(f"Text Analysis Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")

async def test_ocr():
    """Test OCR endpoint"""
    url = "http://localhost:8000/api/v1/text/extract"
    
    # Create a simple test image
    try:
        from PIL import Image, ImageDraw
        import io
        
        # Create test image with text
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 10), "URGENT: Verify your account NOW!", fill='red')
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        async with httpx.AsyncClient() as client:
            files = {'file': ('test.png', img_bytes, 'image/png')}
            response = await client.post(url, files=files, timeout=30)
            
            print(f"\nOCR Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Extracted Text: {result.get('text', 'No text')[:200]}...")
            else:
                print(f"Error: {response.text}")
    except ImportError:
        print("\nOCR test skipped (PIL not available)")

async def main():
    print("Testing AdShield AI Backend...\n")
    await test_text_analysis()
    await test_ocr()

if __name__ == "__main__":
    asyncio.run(main())
