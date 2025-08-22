#!/usr/bin/env python3
"""Check deployment readiness"""

import os
import re

print('CHECKING FOR DEPLOYMENT ISSUES:')
print('=' * 60)

# Check for hardcoded localhost references
files_to_check = ['main.py', 'config.py', 'deepfake_analyzer_light.py']
localhost_pattern = re.compile(r'localhost|127\.0\.0\.1')

print('\n1. CHECKING FOR HARDCODED LOCALHOST:')
for filename in files_to_check:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
            matches = localhost_pattern.findall(content)
            if matches:
                print(f'   ⚠️  {filename} contains localhost references')
            else:
                print(f'   ✅ {filename} no hardcoded localhost')

# Check config for production settings
print('\n2. PRODUCTION CONFIG CHECK:')
from config import settings
print(f'   API Host: {settings.api_host} {"✅ Good for deployment" if settings.api_host == "0.0.0.0" else "❌ Should be 0.0.0.0"}')
print(f'   Debug Mode: {settings.debug} {"❌ Should be False for production" if settings.debug else "✅ Good for production"}')
print(f'   Frontend URL: {settings.frontend_url} {"⚠️  Update for production" if "localhost" in settings.frontend_url else "✅"}')

# Check imports
print('\n3. IMPORT CHECK:')
try:
    import fastapi
    print('   ✅ FastAPI imports')
except:
    print('   ❌ FastAPI import failed')
    
try:
    from deepfake_analyzer_light import get_detector
    print('   ✅ Deepfake analyzer imports')
except Exception as e:
    print(f'   ❌ Deepfake analyzer import failed: {e}')

try:
    from services.huggingface_service import hf_service
    print('   ✅ HuggingFace service imports')
except Exception as e:
    print(f'   ❌ HuggingFace service import failed: {e}')

print('\n4. DEPLOYMENT FILES:')
deployment_files = ['Procfile', 'runtime.txt', '.gitignore']
for file in deployment_files:
    exists = os.path.exists(file)
    if exists:
        print(f'   ✅ {file} exists')
    else:
        print(f'   ⚠️  {file} not found (may be needed for some platforms)')

print('\n5. CORS CONFIGURATION:')
# Check main.py for CORS settings
with open('main.py', 'r') as f:
    main_content = f.read()
    if 'localhost:5173' in main_content:
        print('   ✅ Development CORS configured')
    if 'allow_origins' in main_content:
        print('   ✅ CORS middleware configured')
    if 'settings.frontend_url' in main_content:
        print('   ✅ Dynamic frontend URL from settings')

print('\n' + '=' * 60)
print('DEPLOYMENT RECOMMENDATIONS:')
print('=' * 60)
print("""
1. For Railway/Render deployment:
   - Set environment variable: PORT (will be provided by platform)
   - Set environment variable: HF_API_TOKEN (your Hugging Face token)
   - Set environment variable: FRONTEND_URL (your deployed frontend URL)
   
2. The application will work with:
   ✅ Text scam detection (fully functional)
   ⚠️  Deepfake detection (basic/demo mode)
   ⚠️  OCR (returns informative message)
   
3. Before deploying:
   - Update FRONTEND_URL in production environment
   - Ensure HF_API_TOKEN is set in production
   - Set DEBUG=False in production
""")
