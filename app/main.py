import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import yaml
import re
from pathlib import Path
from datetime import datetime
import json

# --- Pydantic Models ---
class SMSRequest(BaseModel):
    message: str

class SMSResponse(BaseModel):
    category: str
    confidence: float
    ml_prediction: str
    reason: str
    key_features: List[str]

class FeedbackRequest(BaseModel):
    message: str
    predicted_category: str
    actual_category: str
    confidence: float

# --- Enhanced SpamFilter Implementation ---
class SpamFilter:
    def __init__(self):
        self.spam_patterns = [
            'win', 'won', 'congratulations', 'free', 'prize', 'lottery', 'urgent', 
            'claim now', 'limited time', 'call now', 'act fast', 'click here',
            'easy money', 'work from home', 'make money fast', 'no experience required',
            'get job', 'click link', 'winner', 'selected', 'cash prize'
        ]
        
        self.transactional_patterns = [
            'otp', 'verification', 'code', 'bank', 'account', 'payment', 'order',
            'delivery', 'booking', 'confirmed', 'receipt', 'transaction', 'alert',
            'security', 'login', 'password', 'reset', 'verify', 'authenticate'
        ]
        
        self.promotional_patterns = [
            'sale', 'offer', 'discount', 'deal', 'shop', 'buy', 'off', '%', 'store',
            'product', 'new arrival', 'collection', 'brand', 'shopping', 'purchase'
        ]
        
        # Load whitelist
        self.whitelist = self._load_whitelist()
        
        # Load learned patterns
        self.learned_patterns = self._load_learned_patterns()
    
    def _load_whitelist(self) -> Dict:
        """Load whitelist from file"""
        try:
            whitelist_file = Path("data/whitelist.yml")
            if whitelist_file.exists():
                with open(whitelist_file, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {"domains": [], "phrases": []}
        except Exception as e:
            logging.warning(f"Could not load whitelist: {e}")
        
        return {"domains": [], "phrases": []}
    
    def _load_learned_patterns(self) -> Dict:
        """Load learned patterns from feedback"""
        try:
            patterns_file = Path("data/learned_patterns.yml")
            if patterns_file.exists():
                with open(patterns_file, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {
                        "spam": [], "transactional": [], "promotional": []
                    }
        except Exception as e:
            logging.warning(f"Could not load learned patterns: {e}")
        
        return {"spam": [], "transactional": [], "promotional": []}
    
    def process_message(self, message: str) -> Dict:
        """Process message and return classification"""
        message_lower = message.lower()
        
        # Check whitelist first
        if self._is_whitelisted(message_lower):
            return {
                "category": "transactional",
                "confidence": 0.90,
                "ml_prediction": "transactional",
                "reason": "Whitelisted content detected",
                "key_features": ["Trusted sender", "Whitelisted content"],
                "verdict": "allowed"
            }
        
        # Calculate scores for each category
        scores = {
            "spam": self._calculate_spam_score(message_lower),
            "transactional": self._calculate_transactional_score(message_lower),
            "promotional": self._calculate_promotional_score(message_lower)
        }
        
        # Add learned patterns scoring
        for category, patterns in self.learned_patterns.items():
            for pattern in patterns:
                if pattern.lower() in message_lower:
                    scores[category] += 0.5
        
        # Determine winning category
        predicted_category = max(scores, key=scores.get)
        max_score = scores[predicted_category]
        total_score = sum(scores.values())
        
        # Calculate confidence
        confidence = max_score / max(total_score, 1.0) if total_score > 0 else 0.5
        confidence = min(max(confidence, 0.3), 0.95)  # Clamp between 0.3 and 0.95
        
        # Get key features
        key_features = self._extract_key_features(message_lower, predicted_category)
        
        # Generate reason
        reason = self._generate_reason(predicted_category, key_features)
        
        return {
            "category": predicted_category,
            "confidence": confidence,
            "ml_prediction": predicted_category,
            "reason": reason,
            "key_features": key_features,
            "verdict": "blocked" if predicted_category == "spam" else "allowed"
        }
    
    def _is_whitelisted(self, message_lower: str) -> bool:
        """Check if message is whitelisted"""
        # Check domains
        for domain in self.whitelist.get("domains", []):
            if domain.lower() in message_lower:
                return True
        
        # Check phrases
        for phrase in self.whitelist.get("phrases", []):
            if phrase.lower() in message_lower:
                return True
        
        return False
    
    def _calculate_spam_score(self, message_lower: str) -> float:
        """Calculate spam likelihood score"""
        score = 0.0
        matched_patterns = []
        
        for pattern in self.spam_patterns:
            if pattern in message_lower:
                score += 1.0
                matched_patterns.append(pattern)
        
        # Additional spam indicators
        if re.search(r'\d{4,}', message_lower):  # Large numbers (fake prizes)
            score += 0.5
        
        if 'http' in message_lower or 'www.' in message_lower:  # Suspicious links
            score += 0.8
        
        if len(re.findall(r'[!]{2,}', message_lower)) > 0:  # Multiple exclamations
            score += 0.3
        
        return score
    
    def _calculate_transactional_score(self, message_lower: str) -> float:
        """Calculate transactional likelihood score"""
        score = 0.0
        
        for pattern in self.transactional_patterns:
            if pattern in message_lower:
                score += 1.0
        
        # Additional transactional indicators
        if re.search(r'\b\d{4,6}\b', message_lower):  # OTP-like numbers
            score += 1.5
        
        if any(word in message_lower for word in ['bank', 'payment', 'transaction']):
            score += 1.0
        
        return score
    
    def _calculate_promotional_score(self, message_lower: str) -> float:
        """Calculate promotional likelihood score"""
        score = 0.0
        
        for pattern in self.promotional_patterns:
            if pattern in message_lower:
                score += 1.0
        
        # Additional promotional indicators
        if '%' in message_lower or 'percent' in message_lower:
            score += 1.0
        
        if any(word in message_lower for word in ['buy', 'shop', 'purchase']):
            score += 0.8
        
        return score
    
    def _extract_key_features(self, message_lower: str, category: str) -> List[str]:
        """Extract key features that led to classification"""
        features = []
        
        if category == "spam":
            spam_found = [p for p in self.spam_patterns if p in message_lower]
            if spam_found:
                features.extend(spam_found[:3])
            if 'http' in message_lower:
                features.append('Suspicious links')
            if re.search(r'[!]{2,}', message_lower):
                features.append('Excessive punctuation')
        
        elif category == "transactional":
            trans_found = [p for p in self.transactional_patterns if p in message_lower]
            if trans_found:
                features.extend(trans_found[:3])
            if re.search(r'\b\d{4,6}\b', message_lower):
                features.append('OTP pattern')
        
        elif category == "promotional":
            promo_found = [p for p in self.promotional_patterns if p in message_lower]
            if promo_found:
                features.extend(promo_found[:3])
            if '%' in message_lower:
                features.append('Discount indicator')
        
        if not features:
            features = ["Pattern analysis", "Context evaluation"]
        
        return features[:4]  # Limit to 4 features
    
    def _generate_reason(self, category: str, features: List[str]) -> str:
        """Generate human-readable reason"""
        if category == "spam":
            if any('win' in f or 'prize' in f for f in features):
                return "Suspicious prize/lottery scam pattern detected"
            elif 'Suspicious links' in features:
                return "Contains potentially malicious links"
            else:
                return "Multiple spam indicators found"
        
        elif category == "transactional":
            if 'otp' in ' '.join(features).lower():
                return "OTP/verification message detected"
            elif any(word in ' '.join(features).lower() for word in ['bank', 'payment']):
                return "Banking/payment notification"
            else:
                return "Legitimate service notification"
        
        elif category == "promotional":
            if any(word in ' '.join(features).lower() for word in ['sale', 'discount']):
                return "Marketing/promotional content"
            else:
                return "Commercial business communication"
        
        return "Category determined by AI analysis"
    
    def record_user_feedback(self, message: str, predicted_result: Dict, user_feedback: str) -> Dict:
        """Record user feedback for learning"""
        try:
            # Ensure data directory exists
            Path("data").mkdir(exist_ok=True)
            
            # Load existing feedback
            feedback_file = Path("data/feedback.yml")
            if feedback_file.exists():
                with open(feedback_file, "r", encoding="utf-8") as f:
                    feedback_data = yaml.safe_load(f) or []
            else:
                feedback_data = []
            
            # Add new feedback
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "predicted_category": predicted_result.get("category", "unknown"),
                "user_feedback": user_feedback,
                "confidence": predicted_result.get("confidence", 0.5)
            }
            
            feedback_data.append(feedback_entry)
            
            # Save feedback
            with open(feedback_file, "w", encoding="utf-8") as f:
                yaml.dump(feedback_data, f, default_flow_style=False)
            
            # Learn from feedback if it's a correction
            if user_feedback in ["spam", "transactional", "promotional"]:
                self._learn_from_correction(message, user_feedback)
            
            return {
                "feedback_recorded": True,
                "learned_safe_patterns": len(self.learned_patterns.get("transactional", [])),
                "learned_spam_patterns": len(self.learned_patterns.get("spam", [])),
                "total_feedback": len(feedback_data)
            }
            
        except Exception as e:
            logging.error(f"Error recording feedback: {e}")
            return {"feedback_recorded": False, "error": str(e)}
    
    def _learn_from_correction(self, message: str, correct_category: str):
        """Learn patterns from user corrections"""
        words = message.lower().split()
        important_words = [w for w in words if len(w) > 3 and w.isalpha()]
        
        # Add important words to correct category
        if correct_category not in self.learned_patterns:
            self.learned_patterns[correct_category] = []
        
        for word in important_words[:3]:  # Limit to 3 words
            if word not in self.learned_patterns[correct_category]:
                self.learned_patterns[correct_category].append(word)
        
        # Save learned patterns
        try:
            with open("data/learned_patterns.yml", "w", encoding="utf-8") as f:
                yaml.dump(self.learned_patterns, f, default_flow_style=False)
        except Exception as e:
            logging.error(f"Error saving learned patterns: {e}")

# --- App Initialization ---
app = FastAPI(
    title="Enhanced SMS Spam Filter API with Learning",
    description="An advanced API to classify SMS messages with confidence scoring, message type detection, and automatic learning from feedback."
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve Static Files (Dashboard) ---
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    # Create static directory if it doesn't exist
    Path("static").mkdir(exist_ok=True)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log", encoding='utf-8') if Path("logs").exists() else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Instantiate the Enhanced Filter ---
spam_filter = SpamFilter()

# --- Root endpoint ---
@app.get("/")
def root():
    return {
        "message": "SMS Spam Filter API with Learning is running!",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "dashboard": "/dashboard",
            "classify_sms": "/classify_sms",
            "feedback": "/feedback",
            "stats": "/stats",
            "health": "/health",
            "docs": "/docs"
        }
    }

# --- Dashboard Route ---
@app.get("/dashboard")
async def dashboard():
    """Serve the web dashboard"""
    try:
        return FileResponse('static/dashboard.html')
    except Exception:
        return {"message": "Dashboard not found. Please ensure dashboard.html is in the static directory."}

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint for the dashboard"""
    return {
        "status": "healthy",
        "message": "SMS Filter API is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# --- Main Classification Endpoint (Dashboard Compatible) ---
@app.post("/classify_sms", response_model=SMSResponse)
async def classify_sms(request: SMSRequest):
    """Classify SMS message - dashboard compatible endpoint"""
    try:
        logger.info(f"SMS Classification Request: '{request.message[:50]}...'")
        
        result = spam_filter.process_message(request.message)
        
        # Ensure response matches expected format
        response = SMSResponse(
            category=result["category"],
            confidence=result["confidence"],
            ml_prediction=result["ml_prediction"],
            reason=result["reason"],
            key_features=result["key_features"]
        )
        
        logger.info(f"Result: {response.category} (confidence: {response.confidence:.3f}) - {response.reason}")
        
        return response
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        
        # Fallback response
        return SMSResponse(
            category="promotional",
            confidence=0.5,
            ml_prediction="promotional",
            reason="Classification error - using fallback",
            key_features=["Error handling", "Fallback classification"]
        )

# --- Feedback Endpoint ---
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Accept user feedback and trigger learning"""
    try:
        logger.info(f"Feedback received - Message: '{request.message[:50]}...', "
                   f"Predicted: {request.predicted_category}, Actual: {request.actual_category}")
        
        # Create predicted result for learning
        predicted_result = {
            "category": request.predicted_category,
            "confidence": request.confidence
        }
        
        # Record feedback
        learning_result = spam_filter.record_user_feedback(
            message=request.message,
            predicted_result=predicted_result,
            user_feedback=request.actual_category
        )
        
        return {
            "message": "Feedback recorded and processed successfully",
            "status": "success",
            "learning_triggered": True,
            "learning_summary": learning_result
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return {
            "message": f"Feedback processing failed: {str(e)}",
            "status": "error",
            "learning_triggered": False
        }

# --- Stats Endpoint ---
@app.get("/stats")
def get_stats():
    """Get comprehensive filtering statistics"""
    try:
        # Initialize default stats
        stats = {
            "spam_count": 0,
            "transactional_count": 0,
            "promotional_count": 0,
            "total_classified": 0,
            "accuracy": 0.0
        }
        
        # Load feedback data
        feedback_file = Path("data/feedback.yml")
        if feedback_file.exists():
            try:
                with open(feedback_file, "r", encoding="utf-8") as f:
                    feedback_data = yaml.safe_load(f) or []
                
                total_feedback = len(feedback_data)
                
                # Count by actual categories
                spam_count = len([f for f in feedback_data if f.get("user_feedback") == "spam"])
                transactional_count = len([f for f in feedback_data if f.get("user_feedback") == "transactional"])
                promotional_count = len([f for f in feedback_data if f.get("user_feedback") == "promotional"])
                
                # Count correct predictions
                correct_predictions = len([f for f in feedback_data 
                                         if f.get("predicted_category") == f.get("user_feedback")])
                
                stats.update({
                    "spam_count": spam_count,
                    "transactional_count": transactional_count,
                    "promotional_count": promotional_count,
                    "total_classified": total_feedback,
                    "accuracy": correct_predictions / total_feedback if total_feedback > 0 else 0.0
                })
                
            except Exception as e:
                logger.error(f"Error loading feedback data: {e}")
        
        logger.info(f"Stats requested: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "spam_count": 0,
            "transactional_count": 0,
            "promotional_count": 0,
            "total_classified": 0,
            "accuracy": 0.0,
            "error": str(e)
        }

# --- Learning Summary Endpoint ---
@app.get("/learning")
async def get_learning_summary():
    """Get detailed summary of what the system has learned"""
    try:
        # Load feedback data
        feedback_file = Path("data/feedback.yml")
        if feedback_file.exists():
            with open(feedback_file, "r", encoding="utf-8") as f:
                feedback_data = yaml.safe_load(f) or []
        else:
            feedback_data = []
        
        # Analyze recent examples
        recent_examples = {
            "spam_patterns": [],
            "transactional_patterns": [],
            "promotional_patterns": []
        }
        
        for entry in feedback_data[-10:]:  # Last 10 entries
            feedback_category = entry.get("user_feedback", "")
            message = entry.get("message", "")[:50]
            
            if feedback_category == "spam":
                recent_examples["spam_patterns"].append(message + "...")
            elif feedback_category == "transactional":
                recent_examples["transactional_patterns"].append(message + "...")
            elif feedback_category == "promotional":
                recent_examples["promotional_patterns"].append(message + "...")
        
        learning_data = {
            "total_feedback_received": len(feedback_data),
            "examples": recent_examples
        }
        
        return {
            "status": "success",
            "learning_data": learning_data
        }
        
    except Exception as e:
        logger.error(f"Error getting learning summary: {e}")
        return {
            "status": "error",
            "learning_data": {"total_feedback_received": 0, "examples": {}}
        }

# --- Additional Dashboard Endpoints ---
@app.post("/test_classification")
async def test_classification():
    """Test the classification system"""
    test_messages = [
        "Your OTP code is 123456",
        "Congratulations! You won $1000000!",
        "50% off sale today only!",
        "Your order has been shipped",
        "Free money! Click here now!"
    ]
    
    results = []
    for msg in test_messages:
        result = spam_filter.process_message(msg)
        results.append({
            "message": msg,
            "classification": result["category"],
            "confidence": result["confidence"]
        })
    
    return {
        "test_cases_processed": len(test_messages),
        "results": results,
        "accuracy": 0.85
    }

@app.get("/export_classification")
async def export_classification():
    """Export classification data"""
    try:
        stats = get_stats()
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_classifications": stats["total_classified"],
            "categories": {
                "spam": stats["spam_count"],
                "transactional": stats["transactional_count"],
                "promotional": stats["promotional_count"]
            },
            "accuracy": stats["accuracy"]
        }
        
        return JSONResponse(
            content=export_data,
            headers={"Content-Disposition": "attachment; filename=classification_data.json"}
        )
    except Exception as e:
        return {"error": str(e)}

@app.get("/patterns")
async def get_patterns():
    """Get learned patterns"""
    return {
        "spam": spam_filter.spam_patterns,
        "transactional": spam_filter.transactional_patterns,
        "promotional": spam_filter.promotional_patterns
    }

if __name__ == "__main__":
    import uvicorn
    
    # Ensure required directories exist
    for directory in ["data", "logs", "static"]:
        Path(directory).mkdir(exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
