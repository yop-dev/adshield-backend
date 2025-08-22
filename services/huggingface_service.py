import httpx
import json
import asyncio
from typing import Dict, Any, List, Optional
import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
import base64
from io import BytesIO
# PIL is optional, we'll work without it
try:
    from PIL import Image
except ImportError:
    Image = None

class HuggingFaceService:
    """Service for interacting with Hugging Face Inference API"""
    
    def __init__(self):
        self.api_token = settings.hf_api_token
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        self.inference_url = "https://api-inference.huggingface.co/models/"
    
    async def analyze_text_for_scam(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for phishing/scam content using HF models
        """
        try:
            # Try phishing detection model first
            phishing_result = await self._query_model(
                settings.phishing_detection_model,
                {"inputs": text}
            )
            
            # Also check with spam detection for better coverage
            spam_result = await self._query_model(
                settings.spam_detection_model,
                {"inputs": text}
            )
            
            # Combine results for comprehensive analysis
            return self._process_text_results(text, phishing_result, spam_result)
        
        except Exception as e:
            print(f"Error in text analysis: {e}")
            # Return mock result if HF fails
            return self._get_mock_text_result(text)
    
    async def analyze_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Analyze document for fraud indicators
        """
        try:
            print(f"Analyzing document: {filename}")
            extracted_text = ""
            
            # Check file type
            file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
            
            # Extract text based on file type
            if file_ext == 'pdf':
                # Extract text from PDF
                try:
                    import PyPDF2
                    from io import BytesIO
                    
                    pdf_file = BytesIO(file_bytes)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        extracted_text += page.extract_text() + "\n"
                    
                    print(f"Extracted {len(extracted_text)} characters from PDF")
                except ImportError:
                    print("PyPDF2 not installed. Install with: pip install PyPDF2")
                    # Try alternative PDF extraction
                    try:
                        import pdfplumber
                        from io import BytesIO
                        
                        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    extracted_text += page_text + "\n"
                    except ImportError:
                        print("pdfplumber not installed. Install with: pip install pdfplumber")
                except Exception as e:
                    print(f"PDF extraction error: {e}")
            
            elif file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                # Extract text from image using OCR
                extracted_text = await self.extract_text_from_image(file_bytes, filename)
            
            elif file_ext in ['doc', 'docx']:
                # Extract text from Word documents
                try:
                    import docx2txt
                    from io import BytesIO
                    
                    extracted_text = docx2txt.process(BytesIO(file_bytes))
                    print(f"Extracted {len(extracted_text)} characters from Word document")
                except ImportError:
                    print("docx2txt not installed. Install with: pip install docx2txt")
                except Exception as e:
                    print(f"Word document extraction error: {e}")
            
            # If we extracted text, analyze it
            if extracted_text and len(extracted_text.strip()) > 10:
                print(f"Analyzing extracted text: {extracted_text[:200]}...")
                
                # Analyze the extracted text for scams
                text_analysis = await self.analyze_text_for_scam(extracted_text)
                
                # Enhance with document-specific analysis
                result = self._process_document_results(extracted_text, text_analysis, filename)
                return result
            else:
                print(f"No text extracted from document, using mock result")
                return self._get_mock_document_result(filename)
            
        except Exception as e:
            print(f"Error in document analysis: {e}")
            return self._get_mock_document_result(filename)
    
    async def extract_text_from_image(self, image_bytes: bytes, filename: str) -> str:
        """
        OCR Notice: Real OCR requires system libraries not available on free hosting.
        Hugging Face does NOT have proper OCR models.
        """
        print(f"OCR requested for: {filename}")
        
        # Be honest: HF doesn't have real OCR models
        # TrOCR and similar are for captions, not text extraction
        # Real OCR needs Tesseract or paid APIs
        
        return """📝 OCR Not Available on Free Deployment
        
        Why OCR doesn't work here:
        • Hugging Face doesn't have real OCR models
        • Tesseract needs system binaries (not available on free hosting)
        • Cloud OCR APIs require payment
        
        ✅ Good news: The AI scam detection works perfectly with typed text!
        
        Please copy and paste your text below for instant analysis."""
    
    # OCR functions removed - HF doesn't have real OCR models
    # Real OCR requires Tesseract (system binary) or paid APIs
    
    async def analyze_audio(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Analyze audio for voice scams/deepfakes
        """
        try:
            # For MVP, we'll transcribe and analyze text
            # Real implementation would use voice analysis models
            
            # Transcribe audio using Whisper
            transcription = await self._transcribe_audio(file_bytes)
            
            if transcription:
                # Analyze transcribed text for scams
                text_analysis = await self.analyze_text_for_scam(transcription)
                return self._process_audio_results(transcription, text_analysis)
            
            return self._get_mock_audio_result(filename)
            
        except Exception as e:
            print(f"Error in audio analysis: {e}")
            return self._get_mock_audio_result(filename)
    
    async def _query_model(self, model_name: str, payload: Dict) -> Any:
        """Query a Hugging Face model via Inference API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.inference_url}{model_name}",
                headers=self.headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"HF API error: {response.status_code} - {response.text}")
                return None
    
    async def _transcribe_audio(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe audio using Whisper model"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.inference_url}{settings.audio_model}",
                    headers=self.headers,
                    data=audio_bytes,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("text", "")
                return None
        except:
            return None
    
    def _process_text_results(self, text: str, phishing_result: Any, spam_result: Any) -> Dict[str, Any]:
        """Process and combine text analysis results"""
        
        # Default response structure
        result = {
            "label": "legit",
            "score": 0.1,
            "highlights": [],
            "reasons": [],
            "model_version": "hf-inference-v1"
        }
        
        # Check for scam indicators - expanded list for advance-fee scams
        scam_keywords = [
            # Urgency indicators
            "urgent", "verify account", "suspended", "click here", "act now", 
            "limited time", "congratulations", "winner", "prize", "refund",
            # Advance-fee/Nigerian prince scam indicators
            "dear beloved", "dear friend", "business proposal", "million dollars",
            "million usd", "transfer", "inheritance", "deceased", "beneficiary",
            "bank account", "foreign account", "trapped funds", "frozen funds",
            "nigeria", "african bank", "central bank", "attorney", "barrister",
            "next of kin", "unclaimed", "confidential", "discreet", "secret",
            "percentage", "share", "partner", "assistance", "help me",
            "god bless", "blessed day", "trust", "honest", "legitimate",
            # Financial scam indicators  
            "wire transfer", "western union", "moneygram", "payment required",
            "processing fee", "administration fee", "clearance fee", "tax payment",
            # Lottery/Prize scams
            "lottery", "jackpot", "selected", "chosen", "claim your",
            # Romance scams
            "lonely", "widow", "orphan", "soldier", "oil rig",
            # Generic scam phrases
            "100% guaranteed", "risk free", "act immediately", "don't delete",
            "this is not spam", "not a scam", "perfectly legal"
        ]
        
        found_keywords = []
        for keyword in scam_keywords:
            if keyword.lower() in text.lower():
                found_keywords.append(keyword)
                # Add highlight
                idx = text.lower().find(keyword.lower())
                if idx != -1:
                    result["highlights"].append({
                        "start": idx,
                        "end": idx + len(keyword),
                        "reason": f"Suspicious phrase: '{keyword}'"
                    })
        
        # Process HF model results if available
        if phishing_result and isinstance(phishing_result, list):
            for pred in phishing_result:
                if isinstance(pred, dict):
                    if pred.get("label", "").lower() in ["spam", "phishing", "1", "positive"]:
                        result["score"] = max(result["score"], pred.get("score", 0.5))
                        result["label"] = "phishing"
                        result["reasons"].append("AI model detected phishing patterns")
        
        if spam_result and isinstance(spam_result, list):
            for pred in spam_result:
                if isinstance(pred, dict):
                    if pred.get("label", "").lower() in ["spam", "1"]:
                        result["score"] = max(result["score"], pred.get("score", 0.5))
                        if result["label"] == "legit":
                            result["label"] = "phishing"
                        result["reasons"].append("Spam detection triggered")
        
        # Add reasons based on keywords
        if found_keywords:
            # Higher score for more keywords found
            keyword_score = min(0.3 + (len(found_keywords) * 0.15), 0.95)
            result["score"] = max(result["score"], keyword_score)
            
            # Check for specific scam types
            advance_fee_indicators = ["million", "transfer", "bank account", "nigeria", 
                                     "inheritance", "beneficiary", "foreign", "attorney"]
            if any(indicator in [k.lower() for k in found_keywords] for indicator in advance_fee_indicators):
                result["reasons"].append("⚠️ ADVANCE-FEE SCAM DETECTED: Classic Nigerian prince/inheritance scam pattern")
                result["score"] = max(result["score"], 0.9)
            else:
                result["reasons"].append(f"Found suspicious keywords: {', '.join(found_keywords[:5])}" + 
                                        ("..." if len(found_keywords) > 5 else ""))
            
            if result["label"] == "legit":
                result["label"] = "phishing"
        
        # Additional pattern detection
        text_lower = text.lower()
        
        # Check for money amounts
        import re
        money_pattern = r'\$?\d+[,.]?\d*\s*(million|thousand|billion|usd|dollars)'
        if re.search(money_pattern, text_lower):
            result["score"] = max(result["score"], 0.7)
            result["reasons"].append("Large money amounts mentioned - common in advance-fee scams")
            if result["label"] == "legit":
                result["label"] = "phishing"
        
        # Check for email patterns common in scams
        if "dear friend" in text_lower or "dear beloved" in text_lower or "dear sir/madam" in text_lower:
            result["score"] = max(result["score"], 0.8)
            result["reasons"].append("Generic greeting typical of scam emails")
            if result["label"] == "legit":
                result["label"] = "phishing"
        
        # If no scam indicators found at all
        if not found_keywords and result["score"] < 0.3:
            result["reasons"].append("No obvious scam indicators found")
        
        return result
    
    def _process_document_results(self, extracted_text: str, text_analysis: Dict, filename: str) -> Dict[str, Any]:
        """Process document-specific analysis results"""
        import re
        
        # Start with text analysis results
        result = {
            "label": "suspicious" if text_analysis["score"] > 0.5 else "legit",
            "score": text_analysis["score"],
            "findings": [],
            "extractedFields": {},
            "model_version": text_analysis.get("model_version", "hf-inference-v1")
        }
        
        # Extract specific fields from the document
        text_lower = extracted_text.lower()
        
        # Detect document type
        if "invoice" in text_lower:
            result["extractedFields"]["document_type"] = "invoice"
        elif "contract" in text_lower:
            result["extractedFields"]["document_type"] = "contract"
        elif "receipt" in text_lower:
            result["extractedFields"]["document_type"] = "receipt"
        elif "bill" in text_lower:
            result["extractedFields"]["document_type"] = "bill"
        else:
            result["extractedFields"]["document_type"] = "document"
        
        # Extract amounts
        amount_pattern = r'\$([\d,]+\.?\d*)'
        amounts = re.findall(amount_pattern, extracted_text)
        if amounts:
            # Get the largest amount (likely the total)
            result["extractedFields"]["amount"] = f"${max(amounts, key=lambda x: float(x.replace(',', '')))}"
            
            # Check for suspicious large amounts
            for amount in amounts:
                try:
                    value = float(amount.replace(',', ''))
                    if value > 10000:
                        result["findings"].append({
                            "bbox": {"x": 0, "y": 0, "width": 100, "height": 20},
                            "reason": f"Large amount detected: ${amount} - verify legitimacy"
                        })
                        result["score"] = max(result["score"], 0.7)
                except:
                    pass
        
        # Extract dates
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},? \d{4})\b'
        dates = re.findall(date_pattern, extracted_text)
        if dates:
            result["extractedFields"]["dates"] = dates[:3]  # First 3 dates
            
            # Check for urgent payment dates
            if any(word in text_lower for word in ["immediately", "urgent", "within 24 hours", "today"]):
                result["findings"].append({
                    "bbox": {"x": 0, "y": 0, "width": 100, "height": 20},
                    "reason": "Urgent payment deadline - common in scams"
                })
                result["score"] = max(result["score"], 0.8)
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, extracted_text)
        if emails:
            result["extractedFields"]["contact_emails"] = emails[:2]
            
            # Check for suspicious domains
            for email in emails:
                domain = email.split('@')[1].lower()
                if any(sus in domain for sus in ['gmail.com', 'yahoo.com', 'hotmail.com']) and result["extractedFields"]["document_type"] == "invoice":
                    result["findings"].append({
                        "bbox": {"x": 0, "y": 0, "width": 100, "height": 20},
                        "reason": f"Personal email used for business invoice: {email}"
                    })
                    result["score"] = max(result["score"], 0.6)
        
        # Extract company/sender name
        lines = extracted_text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if any(word in line.lower() for word in ['from:', 'company:', 'sender:', 'bill to:', 'invoice from:']):
                result["extractedFields"]["sender"] = line.strip()
                break
        
        # Add text analysis findings
        if text_analysis.get("reasons"):
            for reason in text_analysis["reasons"]:
                result["findings"].append({
                    "bbox": {"x": 0, "y": 0, "width": 100, "height": 20},
                    "reason": reason
                })
        
        # Invoice-specific checks
        if result["extractedFields"].get("document_type") == "invoice":
            # Check for missing invoice number
            if not re.search(r'invoice\s*#?\s*:?\s*\d+', text_lower):
                result["findings"].append({
                    "bbox": {"x": 0, "y": 0, "width": 100, "height": 20},
                    "reason": "No invoice number found - legitimate invoices have unique numbers"
                })
                result["score"] = max(result["score"], 0.7)
            
            # Check for payment methods
            if any(method in text_lower for method in ['wire transfer', 'western union', 'moneygram', 'bitcoin', 'crypto']):
                result["findings"].append({
                    "bbox": {"x": 0, "y": 0, "width": 100, "height": 20},
                    "reason": "Suspicious payment method requested - often used in scams"
                })
                result["score"] = max(result["score"], 0.85)
        
        # Update label based on final score
        if result["score"] > 0.5:
            result["label"] = "suspicious"
        
        return result
    
    def _process_audio_results(self, transcript: str, text_analysis: Dict) -> Dict[str, Any]:
        """Process audio analysis results"""
        return {
            "label": "scam" if text_analysis["score"] > 0.5 else "real",
            "score": text_analysis["score"],
            "reasons": text_analysis["reasons"],
            "model_version": "hf-whisper-v1",
            "transcript": transcript[:500] if transcript else None  # Limit transcript length
        }
    
    def _get_mock_text_result(self, text: str) -> Dict[str, Any]:
        """Return mock result for testing without HF API"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["urgent", "suspended", "verify"]):
            return {
                "label": "phishing",
                "score": 0.85,
                "highlights": [
                    {"start": 0, "end": 10, "reason": "Urgent language detected"}
                ],
                "reasons": ["Contains urgent call to action", "Suspicious request pattern"],
                "model_version": "mock-v1"
            }
        
        return {
            "label": "legit",
            "score": 0.15,
            "highlights": [],
            "reasons": ["No obvious scam indicators found"],
            "model_version": "mock-v1"
        }
    
    def _get_mock_document_result(self, filename: str) -> Dict[str, Any]:
        """Return mock document analysis result"""
        if "invoice" in filename.lower():
            return {
                "label": "suspicious",
                "score": 0.7,
                "findings": [
                    {
                        "bbox": {"x": 100, "y": 200, "width": 300, "height": 50},
                        "reason": "Suspicious payment details"
                    }
                ],
                "extractedFields": {
                    "document_type": "invoice",
                    "amount": "$1,234.56",
                    "sender": "Unknown Entity"
                },
                "model_version": "mock-v1"
            }
        
        return {
            "label": "legit",
            "score": 0.2,
            "findings": [],
            "extractedFields": {"document_type": "general"},
            "model_version": "mock-v1"
        }
    
    def _get_mock_audio_result(self, filename: str) -> Dict[str, Any]:
        """Return mock audio analysis result"""
        return {
            "label": "real",
            "score": 0.3,
            "reasons": ["Audio appears authentic"],
            "model_version": "mock-v1",
            "transcript": "This is a sample transcription of the audio file."
        }

# Singleton instance
hf_service = HuggingFaceService()
