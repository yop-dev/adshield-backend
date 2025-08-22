# AdShield AI Backend

AI-powered scam detection API using Hugging Face models.

## Setup Instructions

### 1. Get Hugging Face API Token

1. Go to https://huggingface.co/settings/tokens
2. Sign up or log in
3. Click "New token"
4. Give it a name (e.g., "AdShield AI")
5. Select "read" role
6. Copy the token (starts with `hf_...`)

### 2. Configure Environment

Create a `.env` file in the backend directory:

```bash
# Copy from .env.example
cp .env.example .env
```

Edit `.env` and add your Hugging Face token:
```
HF_API_TOKEN=hf_YOUR_ACTUAL_TOKEN_HERE
```

### 3. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3.1 OCR Setup (Optional but Recommended for Image Text Extraction)

For accurate text extraction from screenshots/images, install one of these:

**Option A: Tesseract OCR (Recommended - Most Accurate)**
```bash
# Install Python package
pip install pytesseract pillow

# Install Tesseract binary:
# Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
# Mac: brew install tesseract
# Linux: sudo apt-get install tesseract-ocr
```

**Option B: EasyOCR (Alternative - No External Dependencies)**
```bash
pip install easyocr
# Note: Downloads ~64MB model on first use
```

**Note**: Without OCR setup, the system will try Hugging Face API or provide instructions for manual text entry.

### 4. Run the Backend

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

## How It Works

### With Hugging Face Token:
- **Text Analysis**: Uses real phishing/spam detection models
- **OCR Text Extraction**: Extracts text from images (works better with local OCR)
- **Document Analysis**: Analyzes documents for fraud patterns
- **Audio Analysis**: Transcribes audio and checks for scam content

### Without Hugging Face Token:
- Works with mock data for testing
- Returns simulated results based on keywords
- Good for development and UI testing

## Models Used

The backend uses these Hugging Face models:

1. **Text/Email Scam Detection**:
   - `ealvaradob/bert-finetuned-phishing` - Phishing detection
   - `mrm8488/bert-tiny-finetuned-sms-spam-detection` - Spam detection

2. **Audio Processing**:
   - `openai/whisper-base` - Audio transcription
   - Then analyzes transcript for scam content

3. **Document Analysis**:
   - Currently uses mock analysis
   - Future: OCR + fraud pattern detection

## API Endpoints

### Text Analysis
```bash
POST /api/v1/text/analyze
Content-Type: application/json

{
  "text": "Your account has been suspended. Click here to verify..."
}
```

### Text Extraction from Images (OCR)
```bash
POST /api/v1/text/extract
Content-Type: multipart/form-data

file: [image file (PNG/JPG/JPEG)]
```
Returns extracted text that can be analyzed for scams.

### Document Analysis
```bash
POST /api/v1/doc/analyze
Content-Type: multipart/form-data

file: [upload file]
question: (optional)
```

### Audio Analysis
```bash
POST /api/v1/audio/analyze
Content-Type: multipart/form-data

file: [upload audio file]
```

## Testing the API

### Using curl:

```bash
# Test text analysis
curl -X POST http://localhost:8000/api/v1/text/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Urgent: Your account will be suspended. Click here now!"}'

# Test with the frontend
# Just navigate to http://localhost:5173 and use the UI
```

## Troubleshooting

### "HF Token not configured" warning:
- Make sure you created the `.env` file
- Check that the token starts with `hf_`
- Restart the backend after adding the token

### OCR Not Extracting Text from Images:
- Install Tesseract OCR for best results (see section 3.1)
- Make sure image has clear, readable text
- Try preprocessing image (good lighting, contrast)
- If all else fails, manually type the text

### Model loading errors:
- Some models might be gated or require authentication
- The backend will fall back to mock data if models fail
- Check console output for specific errors

### CORS errors:
- Make sure frontend is running on port 5173 or 5174
- Check that backend is running on port 8000

## Rate Limits

Hugging Face free tier limits:
- ~30,000 requests per month
- Some models have individual limits
- Consider upgrading for production use
