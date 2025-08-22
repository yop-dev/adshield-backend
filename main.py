from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from config import settings
from services.huggingface_service import hf_service
# Try to import the lightweight version first (for deployment)
try:
    from deepfake_analyzer_light import get_detector
except ImportError:
    # Fall back to full version if available
    try:
        from deepfake_analyzer import get_detector
    except ImportError:
        # If neither is available, create a mock detector
        class MockDetector:
            def analyze_image(self, image_bytes):
                return {
                    "is_deepfake": False,
                    "confidence": 0.5,
                    "label": "unknown",
                    "risk_score": 0.5,
                    "risk_level": "medium",
                    "explanations": ["Deepfake detection not available"],
                    "recommendations": ["Please configure the detection service"],
                    "details": {}
                }
        def get_detector():
            return MockDetector()
import os

# Create FastAPI app
app = FastAPI(
    title="AdShield AI API",
    description="AI-powered scam detection API",
    version="1.0.0"
)

# Configure CORS
allowed_origins = [
    settings.frontend_url,
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:3000",  # Common React dev server
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]

# Add production URLs if they exist
if settings.frontend_url and settings.frontend_url.startswith("https://"):
    # Add www version if main domain doesn't have it
    if "www." not in settings.frontend_url:
        allowed_origins.append(settings.frontend_url.replace("https://", "https://www."))
    # Add non-www version if main domain has www
    else:
        allowed_origins.append(settings.frontend_url.replace("https://www.", "https://"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class TextAnalysisRequest(BaseModel):
    text: str

class TextAnalysisResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    label: str
    score: float
    highlights: List[Dict[str, Any]]
    reasons: List[str]
    model_version: str

class DocumentAnalysisResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    label: str
    score: float
    findings: List[Dict[str, Any]]
    extractedFields: Dict[str, Any]
    model_version: str

class DeepfakeAnalysisResponse(BaseModel):
    is_deepfake: bool
    confidence: float
    label: str
    risk_score: float
    risk_level: str
    explanations: List[str]
    recommendations: List[str]
    details: Dict[str, Any]
    error: Optional[str] = None

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "AdShield AI API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "hf_token_configured": bool(settings.hf_api_token),
        "models": {
            "text": settings.phishing_detection_model,
            "spam": settings.spam_detection_model,
            "deepfake": "prithivMLmods/deepfake-detector-model-v1"
        }
    }

# Text Extraction Endpoint (OCR)
@app.post("/api/v1/text/extract")
async def extract_text_from_image(file: UploadFile = File(...)):
    """
    Extract text from an image using OCR via Hugging Face
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} not supported"
            )
        
        # Check file size (max 5MB for OCR)
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Extract text using Hugging Face OCR
        extracted_text = await hf_service.extract_text_from_image(contents, file.filename)
        
        return {"text": extracted_text, "filename": file.filename}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in text extraction: {e}")
        # Return mock text for testing
        return {
            "text": "This is sample text extracted from the image. If you see this, OCR is not fully configured yet.",
            "filename": file.filename
        }

# Text Analysis Endpoint
@app.post("/api/v1/text/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text for phishing/scam content
    """
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text content is required")
        
        # Check text size limit
        if len(request.text) > settings.max_text_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Text too large")
        
        # Analyze with Hugging Face
        result = await hf_service.analyze_text_for_scam(request.text)
        
        return TextAnalysisResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in text analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document Analysis Endpoint
@app.post("/api/v1/doc/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None)
):
    """
    Analyze document for fraud indicators
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/gif", "application/pdf"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} not supported"
            )
        
        # Check file size
        contents = await file.read()
        if len(contents) > settings.max_document_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Analyze with Hugging Face
        result = await hf_service.analyze_document(contents, file.filename)
        
        return DocumentAnalysisResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in document analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Deepfake Image Detection Endpoint
@app.post("/api/v1/deepfake/analyze", response_model=DeepfakeAnalysisResponse)
async def analyze_deepfake(file: UploadFile = File(...)):
    """
    Analyze image for deepfake detection
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} not supported. Please upload JPEG, PNG, GIF, or WebP images."
            )
        
        # Check file size (max 10MB for deepfake detection)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
        
        # Analyze with deepfake detector
        detector = get_detector()
        result = detector.analyze_image(contents)
        
        return DeepfakeAnalysisResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in deepfake analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Optional: History endpoints (for future implementation)
@app.post("/api/v1/history")
async def save_history(summary: Dict[str, Any]):
    """Save analysis to history (placeholder)"""
    return {"id": "mock-id-123"}

@app.get("/api/v1/history")
async def get_history(limit: int = 10, offset: int = 0):
    """Get analysis history (placeholder)"""
    return []

# Run the server
if __name__ == "__main__":
    print("\n" + "="*60)
    print("                   AdShield AI Backend")
    print("="*60)
    print(f"  API running at: http://{settings.api_host}:{settings.api_port}")
    print(f"  Frontend URL: {settings.frontend_url}")
    print(f"  HF Token: {'Configured' if settings.hf_api_token else 'Not configured'}")
    print("\n  Endpoints:")
    print("  - POST /api/v1/text/analyze")
    print("  - POST /api/v1/doc/analyze")
    print("  - POST /api/v1/deepfake/analyze")
    print("="*60 + "\n")
    
    if not settings.hf_api_token:
        print("WARNING: Hugging Face API token not configured!")
        print("   The API will work with mock data only.")
        print("   To enable real AI analysis:")
        print("   1. Get token from https://huggingface.co/settings/tokens")
        print("   2. Create .env file with: HF_API_TOKEN=your_token_here")
        print()
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
