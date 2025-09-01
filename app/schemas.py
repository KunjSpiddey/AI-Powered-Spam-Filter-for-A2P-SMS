from pydantic import BaseModel
from typing import Optional

class SMSRequest(BaseModel):
    message: str

class SMSResponse(BaseModel):
    verdict: str  # "allowed" or "blocked"
    reason: str
    confidence: float  # NEW: Confidence score (0.0 to 1.0)
    message_type: str  # NEW: Detected message type (transactional, promotional, spam)
    ml_prediction: str  # NEW: Raw ML model prediction
    ml_confidence: float  # NEW: ML model confidence

class FeedbackRequest(BaseModel):
    message: str
    predicted_result: dict
    user_feedback: str  # "correct", "false_positive", "false_negative"