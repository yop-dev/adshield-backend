"""
Test script to verify backend setup
"""
import sys
import os

print("=" * 60)
print("AdShield AI Backend Setup Test")
print("=" * 60)

# Test imports
try:
    import fastapi
    print("✓ FastAPI installed")
except ImportError:
    print("✗ FastAPI not installed")
    
try:
    import uvicorn
    print("✓ Uvicorn installed")
except ImportError:
    print("✗ Uvicorn not installed")
    
try:
    import httpx
    print("✓ HTTPX installed")
except ImportError:
    print("✗ HTTPX not installed")

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Python-dotenv installed and loaded")
except ImportError:
    print("✗ Python-dotenv not installed")

# Test config
try:
    from config import settings
    print("✓ Config loaded successfully")
    
    if settings.hf_api_token:
        print(f"✓ HF Token configured (starts with: {settings.hf_api_token[:10]}...)")
    else:
        print("⚠ HF Token NOT configured - API will use mock data")
        print("  To add token: Edit .env file and add HF_API_TOKEN=your_token")
        
except Exception as e:
    print(f"✗ Config error: {e}")

# Test if main app loads
try:
    from main import app
    print("✓ FastAPI app loads successfully")
except Exception as e:
    print(f"✗ Main app error: {e}")

print("=" * 60)
print("Setup test complete!")
print("=" * 60)

if not os.path.exists(".env"):
    print("\n⚠ No .env file found! Creating from .env.example...")
    try:
        import shutil
        shutil.copy(".env.example", ".env")
        print("✓ Created .env file - please add your HF token")
    except:
        print("✗ Could not create .env file")
        
print("\nTo run the backend:")
print("  python main.py")
print("\nOr:")
print("  uvicorn main:app --reload")
