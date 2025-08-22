from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # Hugging Face
    hf_api_token: str = os.getenv("HF_API_TOKEN", "")
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # CORS
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
    
    # Model names - Using publicly available models that work with HF Inference API
    text_model: str = os.getenv("TEXT_MODEL", "unitary/toxic-bert")  # For scam detection
    document_model: str = os.getenv("DOCUMENT_MODEL", "nateraw/vit-base-beans")  # Placeholder
    audio_model: str = os.getenv("AUDIO_MODEL", "openai/whisper-base")  # For transcription
    
    # For better text classification, we'll use these:
    spam_detection_model: str = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    phishing_detection_model: str = "ealvaradob/bert-finetuned-phishing"
    
    # File limits
    max_text_size_mb: float = float(os.getenv("MAX_TEXT_SIZE_MB", "1"))
    max_document_size_mb: float = float(os.getenv("MAX_DOCUMENT_SIZE_MB", "10"))
    max_audio_size_mb: float = float(os.getenv("MAX_AUDIO_SIZE_MB", "50"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields in .env

settings = Settings()
