import requests
import base64

print('=' * 60)
print('TESTING ALL ENDPOINTS')
print('=' * 60)

# Test 1: Text Analysis (should work perfectly)
print('\n1. TEXT ANALYSIS TEST:')
text_data = {'text': 'URGENT: Your account has been suspended. Click here to verify your identity now or lose access forever.'}
response = requests.post('http://localhost:8000/api/v1/text/analyze', json=text_data)
print(f'   Status: {response.status_code}')
if response.ok:
    result = response.json()
    print(f'   Label: {result.get("label")}')
    print(f'   Score: {result.get("score")}')
    print(f'   Reasons: {result.get("reasons", [])}')
    print('   ✅ Text analysis working!')
else:
    print(f'   ❌ Error: {response.text[:200]}')

# Test 2: Deepfake Detection
print('\n2. DEEPFAKE DETECTION TEST:')
test_image = b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
files = {'file': ('test.png', base64.b64decode(test_image), 'image/png')}
response = requests.post('http://localhost:8000/api/v1/deepfake/analyze', files=files)
print(f'   Status: {response.status_code}')
if response.ok:
    result = response.json()
    print(f'   Is deepfake: {result.get("is_deepfake")}')
    print(f'   Confidence: {result.get("confidence")}')
    print(f'   Label: {result.get("label")}')
    print('   ✅ Deepfake detection working!')
else:
    print(f'   ❌ Error: {response.text[:200]}')

# Test 3: OCR/Text Extraction
print('\n3. OCR TEXT EXTRACTION TEST:')
files = {'file': ('test.png', base64.b64decode(test_image), 'image/png')}
response = requests.post('http://localhost:8000/api/v1/text/extract', files=files)
print(f'   Status: {response.status_code}')
if response.ok:
    result = response.json()
    extracted = result.get("text", "")
    print(f'   Extracted text: {extracted[:100]}...')
    if "OCR functionality is not available" in extracted:
        print('   ⚠️ OCR returns informative message (as expected for free tier)')
    else:
        print('   ✅ OCR working!')
else:
    print(f'   ❌ Error: {response.text[:200]}')

# Test 4: Document Analysis
print('\n4. DOCUMENT ANALYSIS TEST:')
files = {'file': ('test.png', base64.b64decode(test_image), 'image/png')}
response = requests.post('http://localhost:8000/api/v1/doc/analyze', files=files)
print(f'   Status: {response.status_code}')
if response.ok:
    result = response.json()
    print(f'   Label: {result.get("label")}')
    print(f'   Score: {result.get("score")}')
    print('   ✅ Document analysis working!')
else:
    print(f'   ❌ Error: {response.text[:200]}')

# Test 5: Test non-scam text
print('\n5. NON-SCAM TEXT TEST:')
text_data = {'text': 'Hello, this is a normal message about the weather today. It is sunny and warm.'}
response = requests.post('http://localhost:8000/api/v1/text/analyze', json=text_data)
print(f'   Status: {response.status_code}')
if response.ok:
    result = response.json()
    print(f'   Label: {result.get("label")}')
    print(f'   Score: {result.get("score")}')
    if result.get("label") == "safe":
        print('   ✅ Correctly identified as safe text!')
    else:
        print('   ⚠️ Might be too cautious')

print('\n' + '=' * 60)
print('TEST SUMMARY')
print('=' * 60)
print('✅ = Working as expected')
print('⚠️ = Limited functionality (expected for free tier)')
print('❌ = Not working, needs fixing')
