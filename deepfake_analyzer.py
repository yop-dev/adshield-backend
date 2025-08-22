"""
Image Deepfake Detection using Hugging Face model
Model: prithivMLmods/deepfake-detector-model-v1
"""

import torch
from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import io
import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    def __init__(self):
        self.model_name = "prithivMLmods/deepfake-detector-model-v1"
        self.pipeline = None
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"DeepfakeDetector initialized with device: {self.device}")
        
    def load_model(self):
        """Load the deepfake detection model from Hugging Face"""
        try:
            logger.info(f"Loading deepfake detection model: {self.model_name}")
            
            # Try to use pipeline first (simpler approach)
            try:
                self.pipeline = pipeline(
                    "image-classification",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Model loaded successfully using pipeline")
            except Exception as e:
                logger.warning(f"Pipeline loading failed: {e}, trying direct model loading")
                # Fallback to direct model loading
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                self.model.eval()
                logger.info("Model loaded successfully using direct loading")
                
        except Exception as e:
            logger.error(f"Failed to load deepfake detection model: {e}")
            raise
    
    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze an image for deepfake detection
        
        Args:
            image_bytes: Image file bytes
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load model if not already loaded
            if self.pipeline is None and self.model is None:
                self.load_model()
            
            # Open and prepare image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform inference
            if self.pipeline:
                results = self.pipeline(image)
                logger.info(f"Pipeline results: {results}")
            else:
                # Manual inference with processor and model
                inputs = self.processor(images=image, return_tensors="pt")
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Get probabilities
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get labels and scores
                results = []
                for idx, prob in enumerate(probabilities[0]):
                    label = self.model.config.id2label.get(idx, f"Class_{idx}")
                    results.append({
                        "label": label,
                        "score": float(prob.cpu().numpy())
                    })
                
                # Sort by score
                results = sorted(results, key=lambda x: x["score"], reverse=True)
                logger.info(f"Model results: {results}")
            
            # Process results
            return self._process_results(results)
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "error": str(e),
                "details": {}
            }
    
    def _process_results(self, results: list) -> Dict[str, Any]:
        """
        Process model results into a standardized format
        
        Args:
            results: Raw model results
            
        Returns:
            Processed analysis results
        """
        try:
            # Initialize default response
            response = {
                "is_deepfake": False,
                "confidence": 0.0,
                "label": "unknown",
                "details": {
                    "all_predictions": results[:5] if results else []  # Top 5 predictions
                }
            }
            
            if not results:
                return response
            
            # Get the top prediction
            top_result = results[0]
            label = top_result.get("label", "").lower()
            score = top_result.get("score", 0.0)
            
            # Determine if it's a deepfake based on label
            deepfake_labels = ["fake", "deepfake", "synthetic", "generated", "manipulated", "artificial"]
            real_labels = ["real", "authentic", "genuine", "original", "natural"]
            
            is_deepfake = any(keyword in label for keyword in deepfake_labels)
            is_real = any(keyword in label for keyword in real_labels)
            
            if is_deepfake:
                response["is_deepfake"] = True
                response["confidence"] = score
                response["label"] = "deepfake"
            elif is_real:
                response["is_deepfake"] = False
                response["confidence"] = score
                response["label"] = "real"
            else:
                # If labels don't clearly indicate fake/real, use score threshold
                # Assuming the first class is "fake" and second is "real" (adjust based on actual model)
                if len(results) >= 2:
                    fake_score = results[0]["score"] if "fake" in results[0]["label"].lower() else results[1]["score"]
                    real_score = results[1]["score"] if "real" in results[1]["label"].lower() else results[0]["score"]
                    
                    response["is_deepfake"] = fake_score > real_score
                    response["confidence"] = max(fake_score, real_score)
                    response["label"] = "deepfake" if response["is_deepfake"] else "real"
                else:
                    response["confidence"] = score
                    response["label"] = label
            
            # Add risk assessment
            if response["is_deepfake"]:
                if response["confidence"] > 0.8:
                    response["risk_level"] = "high"
                    response["risk_score"] = 0.9
                elif response["confidence"] > 0.6:
                    response["risk_level"] = "medium"
                    response["risk_score"] = 0.6
                else:
                    response["risk_level"] = "low"
                    response["risk_score"] = 0.3
            else:
                response["risk_level"] = "low"
                response["risk_score"] = 0.1
            
            # Add explanations
            response["explanations"] = self._generate_explanations(response)
            response["recommendations"] = self._generate_recommendations(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "error": str(e),
                "details": {}
            }
    
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

# Singleton instance
_detector_instance = None

def get_detector() -> DeepfakeDetector:
    """Get or create the singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DeepfakeDetector()
    return _detector_instance
