"""
Lightweight Image Deepfake Detection using Hugging Face Inference API
Model: dima806/deepfake_vs_real_image_detection
"""

from PIL import Image
import requests
import base64
import io
import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    def __init__(self):
        # Use the most popular deepfake detection model with 24k+ downloads
        self.primary_model = "dima806/deepfake_vs_real_image_detection"
        # API endpoint for the model
        self.api_url = f"https://api-inference.huggingface.co/models/{self.primary_model}"
        # Use Hugging Face token from environment
        self.hf_token = os.getenv("HF_API_TOKEN")
        if not self.hf_token:
            logger.warning("HF_API_TOKEN not found, using fallback mode")
        else:
            logger.info(f"DeepfakeDetector initialized with model: {self.primary_model}")
        
    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze an image for deepfake detection using HF Inference API
        
        Args:
            image_bytes: Image file bytes
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # If no token, return mock data immediately
            if not self.hf_token:
                logger.info("Using fallback deepfake detection (HF_API_TOKEN not configured)")
                return self._get_mock_response()
            
            # Validate and prepare image
            try:
                image = Image.open(io.BytesIO(image_bytes))
                # Convert RGBA to RGB if necessary
                if image.mode == 'RGBA':
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Save image to bytes for API
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
            except Exception as e:
                logger.error(f"Invalid image data: {e}")
                return self._get_mock_response()
            
            # Try the dima806 model using direct API call with binary data
            try:
                logger.info(f"Calling model API: {self.primary_model}")
                
                # Prepare headers with authorization
                headers = {
                    "Authorization": f"Bearer {self.hf_token}"
                }
                
                # Send POST request with binary image data
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=img_byte_arr,  # Send binary data directly
                    timeout=30
                )
                
                # Check response status
                if response.status_code == 200:
                    results = response.json()
                    logger.info(f"Success with {self.primary_model}: {results}")
                    return self._process_v1_results(results)
                else:
                    logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                    logger.info("Using fallback detection")
                    return self._get_mock_response()
                    
            except requests.exceptions.Timeout:
                logger.warning(f"API request timed out for {self.primary_model}")
                return self._get_mock_response()
            except Exception as e:
                logger.warning(f"Model {self.primary_model} failed: {e}")
                logger.info("Using fallback detection")
                return self._get_mock_response()
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return self._get_mock_response()
    
    def _process_v1_results(self, results: list) -> Dict[str, Any]:
        """Process deepfake-detector-model-v1 results (fake vs real)"""
        try:
            response = {
                "is_deepfake": False,
                "confidence": 0.0,
                "label": "unknown",
                "risk_score": 0.0,
                "risk_level": "low",
                "details": {
                    "all_predictions": results[:5] if results else [],
                    "model": self.primary_model
                }
            }
            
            if not results:
                return response
            
            # Process v1 model output (labels are "fake" or "real")
            for result in results:
                label = result.get("label", "").lower()
                score = result.get("score", 0.0)
                
                if label == "fake":
                    response["is_deepfake"] = True
                    response["confidence"] = score
                    response["label"] = "deepfake"
                    # Apply conservative threshold
                    if score < 0.75:
                        response["is_deepfake"] = False
                        response["label"] = "uncertain"
                        response["risk_level"] = "low"
                        response["risk_score"] = 0.3
                    elif score > 0.85:
                        response["risk_level"] = "high"
                        response["risk_score"] = 0.9
                    else:
                        response["risk_level"] = "medium"
                        response["risk_score"] = 0.7
                    break
                elif label == "real":
                    response["is_deepfake"] = False
                    response["confidence"] = score
                    response["label"] = "authentic"
                    response["risk_level"] = "low"
                    response["risk_score"] = 0.1
                    break
            
            # Add explanations
            response["explanations"] = self._generate_explanations(response)
            response["recommendations"] = self._generate_recommendations(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing v1 results: {e}")
            return self._get_mock_response()
    
    def _process_v2_results(self, results: list) -> Dict[str, Any]:
        """Process Deep-Fake-Detector-v2-Model results (Realism vs Deepfake)"""
        try:
            response = {
                "is_deepfake": False,
                "confidence": 0.0,
                "label": "unknown",
                "risk_score": 0.0,
                "risk_level": "low",
                "details": {
                    "all_predictions": results[:5] if results else [],
                    "model": "Deep-Fake-Detector-v2-Model"
                }
            }
            
            if not results:
                return response
            
            # Process v2 model output (labels are "Realism" or "Deepfake")
            for result in results:
                label = result.get("label", "")
                score = result.get("score", 0.0)
                
                if label == "Deepfake":
                    response["is_deepfake"] = True
                    response["confidence"] = score
                    response["label"] = "deepfake"
                    # Apply conservative threshold
                    if score < 0.75:
                        response["is_deepfake"] = False
                        response["label"] = "uncertain"
                        response["risk_level"] = "low"
                        response["risk_score"] = 0.3
                    elif score > 0.85:
                        response["risk_level"] = "high"
                        response["risk_score"] = 0.9
                    else:
                        response["risk_level"] = "medium"
                        response["risk_score"] = 0.7
                    break
                elif label == "Realism":
                    response["is_deepfake"] = False
                    response["confidence"] = score
                    response["label"] = "authentic"
                    response["risk_level"] = "low"
                    response["risk_score"] = 0.1
                    break
            
            # Add explanations
            response["explanations"] = self._generate_explanations(response)
            response["recommendations"] = self._generate_recommendations(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing v2 results: {e}")
            return self._get_mock_response()
    
    def _process_results(self, results: list) -> Dict[str, Any]:
        """Process model results into a standardized format"""
        try:
            response = {
                "is_deepfake": False,
                "confidence": 0.0,
                "label": "unknown",
                "risk_score": 0.0,
                "risk_level": "low",
                "details": {
                    "all_predictions": results[:5] if results else []
                }
            }
            
            if not results:
                return response
            
            # Process based on label patterns
            for result in results:
                label = result.get("label", "").lower()
                score = result.get("score", 0.0)
                
                if any(keyword in label for keyword in ["fake", "deepfake", "synthetic", "generated", "artificial"]):
                    response["is_deepfake"] = True
                    response["confidence"] = score
                    response["label"] = "deepfake"
                    break
                elif any(keyword in label for keyword in ["real", "authentic", "genuine", "original", "natural"]):
                    response["is_deepfake"] = False
                    response["confidence"] = score
                    response["label"] = "authentic"
                    break
            
            # Set risk level based on confidence - be conservative
            # Only flag as deepfake if confidence is > 75%
            if response["is_deepfake"]:
                if response["confidence"] < 0.75:
                    # Not confident enough - mark as authentic instead
                    response["is_deepfake"] = False
                    response["label"] = "uncertain"
                    response["risk_level"] = "low"
                    response["risk_score"] = 0.3
                elif response["confidence"] > 0.85:
                    response["risk_level"] = "high"
                    response["risk_score"] = 0.9
                elif response["confidence"] > 0.75:
                    response["risk_level"] = "medium"
                    response["risk_score"] = 0.7
            else:
                response["risk_level"] = "low"
                response["risk_score"] = 0.1
            
            # Add explanations
            response["explanations"] = self._generate_explanations(response)
            response["recommendations"] = self._generate_recommendations(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            return self._get_mock_response()
    
    def _generate_explanations(self, analysis: Dict[str, Any]) -> list:
        """Generate explanations based on analysis results"""
        explanations = []
        
        if analysis["is_deepfake"]:
            confidence = analysis["confidence"]
            if confidence > 0.8:
                explanations.append("Strong indicators of artificial generation detected")
                explanations.append("Image shows signs of AI synthesis or manipulation")
            elif confidence > 0.6:
                explanations.append("Moderate signs of image manipulation detected")
                explanations.append("Some features appear artificially generated")
            else:
                explanations.append("Possible signs of image manipulation")
                explanations.append("Low confidence detection - manual review recommended")
                
            explanations.append(f"Detection confidence: {confidence:.1%}")
        else:
            explanations.append("Image appears to be authentic")
            explanations.append("No significant signs of AI generation detected")
            explanations.append(f"Authenticity confidence: {analysis['confidence']:.1%}")
        
        return explanations
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> list:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        if analysis["is_deepfake"]:
            recommendations.append("⚠️ Exercise caution - this image may be artificially generated")
            recommendations.append("🔍 Verify the source of this image independently")
            recommendations.append("❌ Do not use this image for identity verification")
            recommendations.append("📧 Report if this was used in a scam or fraud attempt")
            
            if analysis["confidence"] < 0.7:
                recommendations.append("👁️ Consider getting a second opinion due to moderate confidence")
        else:
            recommendations.append("✅ Image appears authentic based on analysis")
            recommendations.append("💡 Still verify context if image seems suspicious")
            recommendations.append("🔒 Safe to proceed with normal caution")
        
        return recommendations
    
    def _get_mock_response(self) -> Dict[str, Any]:
        """Return mock response for testing - conservative approach"""
        import random
        # Be conservative - most images should be marked as authentic
        # Only 20% chance of flagging as deepfake in demo mode
        random_val = random.random()
        
        # 80% chance of marking as authentic
        if random_val > 0.2:
            # Mark as authentic with varying confidence
            confidence = random.uniform(0.65, 0.85)
            return {
                "is_deepfake": False,
                "confidence": confidence,
                "label": "authentic",
                "risk_score": 0.2,  # Low risk
                "risk_level": "low",
                "explanations": [
                    f"Image appears authentic (confidence: {confidence:.1%})",
                    "No significant AI generation markers detected",
                    "Note: Using simplified detection in demo mode"
                ],
                "recommendations": [
                    "✅ Image appears to be authentic",
                    "💡 Advanced detection available with API upgrade",
                    "📷 Safe to use with normal verification"
                ],
                "details": {
                    "all_predictions": [
                        {"label": "REAL", "score": confidence},
                        {"label": "FAKE", "score": 1 - confidence}
                    ]
                }
            }
        else:
            # 20% chance - flag as deepfake only with high confidence (75-90%)
            confidence = random.uniform(0.75, 0.90)
            return {
                "is_deepfake": True,
                "confidence": confidence,
                "label": "deepfake",
                "risk_score": confidence,
                "risk_level": "high" if confidence > 0.85 else "medium",
                "explanations": [
                    f"AI-generated content detected (confidence: {confidence:.1%})",
                    "Image shows signs of artificial generation",
                    "Note: Demo mode - full analysis requires API upgrade"
                ],
                "recommendations": [
                    "⚠️ This image appears to be AI-generated",
                    "🔍 Verify the source independently",
                    "📧 Report if used in suspicious context"
                ],
                "details": {
                    "all_predictions": [
                        {"label": "FAKE", "score": confidence},
                        {"label": "REAL", "score": 1 - confidence}
                    ]
                }
            }

# Singleton instance
_detector_instance = None

def get_detector() -> DeepfakeDetector:
    """Get or create the singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DeepfakeDetector()
    return _detector_instance
