import joblib
import yaml
import re
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import math
from .whitelist_manager import load_whitelist, add_domain, add_phrase, save_whitelist

# --- Load Configurations ---
with open("config/settings.yml", "r") as f:
    settings = yaml.safe_load(f)

# --- Load ML Model & Vectorizer ---
try:
    model = joblib.load(settings['model']['model_path'])
    vectorizer = joblib.load(settings['model']['vectorizer_path'])
    print("✅ Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("🔴 Error: Model or vectorizer file not found. Place your .pkl files in the /models directory.")
    model, vectorizer = None, None

# --- Enhanced Message Type Configurations with Adaptive Thresholds ---
MESSAGE_TYPE_RULES = {
    "otp": {
        "keywords": ["otp", "verification", "code", "verify", "authenticate", "login", "pin"],
        "confidence_threshold": 0.2,  # Very low threshold for OTP
        "always_allow": True,
        "regex_patterns": [
            r"\b\d{4,8}\b.*(?:otp|code|verification|pin)",
            r"your.*(?:code|otp|pin).*is.*\d+",
            r"verification.*code.*\d+",
            r"login.*code.*\d+"
        ],
        "adaptive_threshold": True,
        "min_threshold": 0.1,
        "max_threshold": 0.4
    },
    "transactional": {
        "keywords": ["receipt", "invoice", "payment", "order", "confirmation", "booking", "delivered", "shipped", "account", "balance", "transaction", "upi", "received"],
        "confidence_threshold": 0.4,
        "always_allow": False,
        "regex_patterns": [
            r"order\s*#?\s*\d+",
            r"booking\s*#?\s*\d+",
            r"transaction.*(?:successful|completed|failed)",
            r"payment.*(?:received|processed|failed)",
            r"account.*balance",
            r"received.*rs\.?\s*\d+.*(?:via|from|upi)",
            r"pnr\s*:?\s*[a-z0-9]+",
            r"bill.*rs\.?\s*\d+"
        ],
        "adaptive_threshold": True,
        "min_threshold": 0.2,
        "max_threshold": 0.7
    },
    "promotional": {
        "keywords": ["offer", "discount", "sale", "deal", "free", "limited time", "cashback", "reward", "bonus", "off", "shop", "buy"],
        "confidence_threshold": 0.6,
        "always_allow": False,
        "regex_patterns": [
            r"\d+%\s*off",
            r"free\s+\w+",
            r"limited.*time",
            r"special.*offer",
            r"save.*\$?\d+",
            r"get.*\d+%.*off"
        ],
        "adaptive_threshold": True,
        "min_threshold": 0.4,
        "max_threshold": 0.8
    }
}

class AdvancedFeedbackManager:
    """Advanced feedback manager with sophisticated learning capabilities."""
    
    def __init__(self):
        self.feedback_file = Path("data/feedback.yml")
        self.learning_rules_file = Path("data/learning_rules.yml")
        self.pattern_weights_file = Path("data/pattern_weights.yml")
        self.adaptive_thresholds_file = Path("data/adaptive_thresholds.yml")
        self.feedback_file.parent.mkdir(exist_ok=True)
        
        # Enhanced learning thresholds
        self.auto_whitelist_threshold = 3  # Auto-whitelist after 3 false positives
        self.pattern_learning_threshold = 2  # Learn new patterns after 2 instances
        self.confidence_adjustment_threshold = 5  # Adjust confidence after 5 samples
        self.pattern_decay_days = 30  # Decay old patterns after 30 days
        
        # Initialize learning components
        self._initialize_learning_files()
        self.pattern_weights = self._load_pattern_weights()
        self.adaptive_thresholds = self._load_adaptive_thresholds()
        
    def _initialize_learning_files(self):
        """Initialize all learning-related files if they don't exist."""
        default_rules = {
            "learned_safe_patterns": [],
            "learned_spam_patterns": [],
            "auto_whitelisted_domains": [],
            "auto_whitelisted_phrases": [],
            "pattern_confidence_scores": {},
            "context_patterns": {},
            "temporal_patterns": {},
            "learning_stats": {
                "total_feedback": 0,
                "patterns_learned": 0,
                "auto_whitelists_added": 0,
                "accuracy_improvements": 0,
                "last_updated": datetime.now().isoformat()
            }
        }
        
        if not self.learning_rules_file.exists():
            with open(self.learning_rules_file, "w") as f:
                yaml.dump(default_rules, f)
        
        if not self.pattern_weights_file.exists():
            with open(self.pattern_weights_file, "w") as f:
                yaml.dump({"word_weights": {}, "ngram_weights": {}}, f)
                
        if not self.adaptive_thresholds_file.exists():
            default_thresholds = {msg_type: rules["confidence_threshold"] 
                                for msg_type, rules in MESSAGE_TYPE_RULES.items()}
            with open(self.adaptive_thresholds_file, "w") as f:
                yaml.dump(default_thresholds, f)
    
    def _load_pattern_weights(self) -> Dict:
        """Load learned pattern weights."""
        try:
            with open(self.pattern_weights_file, "r") as f:
                return yaml.safe_load(f) or {"word_weights": {}, "ngram_weights": {}}
        except FileNotFoundError:
            return {"word_weights": {}, "ngram_weights": {}}
    
    def _save_pattern_weights(self):
        """Save pattern weights to file."""
        with open(self.pattern_weights_file, "w") as f:
            yaml.dump(self.pattern_weights, f)
    
    def _load_adaptive_thresholds(self) -> Dict:
        """Load adaptive confidence thresholds."""
        try:
            with open(self.adaptive_thresholds_file, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}
    
    def _save_adaptive_thresholds(self):
        """Save adaptive thresholds to file."""
        with open(self.adaptive_thresholds_file, "w") as f:
            yaml.dump(self.adaptive_thresholds, f)
    
    def record_feedback(self, message: str, predicted_label: str, 
                       user_feedback: str, confidence: float):
        """Enhanced feedback recording with comprehensive learning."""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "message_hash": hash(message) % 100000,
            "predicted": predicted_label,
            "user_feedback": user_feedback,
            "confidence": confidence,
            "message_snippet": message[:100],
            "message_length": len(message),
            "has_urls": bool(re.search(r'http[s]?://|www\.', message)),
            "has_numbers": bool(re.search(r'\d', message)),
            "word_count": len(message.split()),
            "has_currency": bool(re.search(r'rs\.?\s*\d+|\$\d+|₹\d+', message.lower())),
            "has_phone": bool(re.search(r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', message)),
            "urgency_words": len(re.findall(r'\b(?:urgent|immediate|now|quick|fast|limited|hurry)\b', message.lower())),
            "capitalization_ratio": sum(1 for c in message if c.isupper()) / max(len(message), 1)
        }
        
        # Load and save feedback
        feedback_data = self._load_feedback_data()
        feedback_data.append(feedback_entry)
        
        with open(self.feedback_file, "w") as f:
            yaml.dump(feedback_data, f)
        
        print(f"📝 Advanced feedback recorded: {user_feedback} for {predicted_label}")
        
        # Trigger comprehensive learning processes
        self._advanced_pattern_learning(message, user_feedback, predicted_label, confidence)
        self._contextual_learning(message, user_feedback, predicted_label)
        self._confidence_calibration(user_feedback, predicted_label, confidence)
        self._adaptive_threshold_adjustment(user_feedback, predicted_label, confidence)
        self._temporal_pattern_learning(message, user_feedback)
        self._auto_update_whitelist(message, user_feedback)
        self._update_learning_stats()
    
    def _load_feedback_data(self) -> List[Dict]:
        """Load existing feedback data."""
        try:
            with open(self.feedback_file, "r") as f:
                return yaml.safe_load(f) or []
        except FileNotFoundError:
            return []
    
    def _load_learning_rules(self) -> Dict:
        """Load learning rules."""
        with open(self.learning_rules_file, "r") as f:
            return yaml.safe_load(f) or {}
    
    def _save_learning_rules(self, rules: Dict):
        """Save learning rules."""
        with open(self.learning_rules_file, "w") as f:
            yaml.dump(rules, f)
    
    def _advanced_pattern_learning(self, message: str, user_feedback: str, 
                                 predicted_label: str, confidence: float):
        """Advanced pattern learning with weighted features."""
        message_lower = message.lower()
        words = message_lower.split()
        
        # Extract n-grams (1-3 words)
        ngrams = []
        for n in range(1, 4):
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i:i+n]))
        
        # Determine correct label
        correct_label = self._determine_correct_label(user_feedback, predicted_label)
        if not correct_label:
            return
        
        # Calculate learning weight based on confidence and feedback type
        learning_weight = self._calculate_learning_weight(user_feedback, confidence)
        
        # Update word and n-gram weights
        for ngram in ngrams:
            if ngram not in self.pattern_weights["ngram_weights"]:
                self.pattern_weights["ngram_weights"][ngram] = {
                    "spam": 0, "transactional": 0, "promotional": 0
                }
            
            # Increase weight for correct label
            self.pattern_weights["ngram_weights"][ngram][correct_label] += learning_weight
            
            # Decrease weight for incorrect labels if this was a misclassification
            if user_feedback in ["false_positive", "false_negative"]:
                for label in ["spam", "transactional", "promotional"]:
                    if label != correct_label:
                        self.pattern_weights["ngram_weights"][ngram][label] *= 0.95
        
        # Learn specific patterns
        if user_feedback == "false_positive":
            self._learn_safe_patterns(message, correct_label)
        elif user_feedback == "false_negative":
            self._learn_spam_patterns(message)
        
        self._save_pattern_weights()
        print(f"🧠 Advanced pattern learning: Updated weights for {len(ngrams)} patterns")
    
    def _calculate_learning_weight(self, user_feedback: str, confidence: float) -> float:
        """Calculate learning weight based on feedback type and confidence."""
        base_weights = {
            "false_positive": 1.5,  # High weight for false positives
            "false_negative": 1.3,  # High weight for false negatives
            "correct": 0.5,         # Lower weight for confirmations
            "spam": 1.2,
            "transactional": 1.2,
            "promotional": 1.2
        }
        
        base_weight = base_weights.get(user_feedback, 1.0)
        
        # Adjust based on confidence - learn more from high-confidence mistakes
        if user_feedback in ["false_positive", "false_negative"]:
            confidence_multiplier = 1.0 + confidence  # 1.0 to 2.0
        else:
            confidence_multiplier = 1.0
        
        return base_weight * confidence_multiplier
    
    def _determine_correct_label(self, user_feedback: str, predicted_label: str) -> Optional[str]:
        """Determine the correct label from user feedback."""
        if user_feedback in ["spam", "transactional", "promotional"]:
            return user_feedback
        elif user_feedback == "false_positive":
            # Model said spam but it's not spam
            if predicted_label == "spam":
                return "transactional"  # Default for false positive spam
        elif user_feedback == "false_negative":
            # Model missed spam
            return "spam"
        elif user_feedback == "correct":
            return predicted_label
        
        return None
    
    def _contextual_learning(self, message: str, user_feedback: str, predicted_label: str):
        """Learn contextual patterns (combinations of features)."""
        rules = self._load_learning_rules()
        
        # Extract contextual features
        context = {
            "has_numbers": bool(re.search(r'\d', message)),
            "has_currency": bool(re.search(r'rs\.?\s*\d+|\$\d+|₹\d+', message.lower())),
            "has_urgency": bool(re.search(r'\b(?:urgent|immediate|now|quick|limited)\b', message.lower())),
            "message_length": "short" if len(message) < 50 else "medium" if len(message) < 150 else "long",
            "has_call_to_action": bool(re.search(r'\b(?:click|call|reply|visit|download)\b', message.lower()))
        }
        
        # Create context signature
        context_signature = "&".join([f"{k}={v}" for k, v in context.items() if v])
        
        if "context_patterns" not in rules:
            rules["context_patterns"] = {}
        
        correct_label = self._determine_correct_label(user_feedback, predicted_label)
        if correct_label and context_signature:
            if context_signature not in rules["context_patterns"]:
                rules["context_patterns"][context_signature] = {
                    "spam": 0, "transactional": 0, "promotional": 0, "count": 0
                }
            
            rules["context_patterns"][context_signature][correct_label] += 1
            rules["context_patterns"][context_signature]["count"] += 1
        
        self._save_learning_rules(rules)
        print(f"📊 Context learning: Updated pattern '{context_signature[:50]}...'")
    
    def _confidence_calibration(self, user_feedback: str, predicted_label: str, confidence: float):
        """Calibrate confidence scores based on feedback accuracy."""
        # This would ideally retrain the model, but for now we track patterns
        rules = self._load_learning_rules()
        
        if "confidence_calibration" not in rules:
            rules["confidence_calibration"] = {"high": [], "medium": [], "low": []}
        
        confidence_bucket = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        is_correct = user_feedback == "correct" or user_feedback == predicted_label
        
        rules["confidence_calibration"][confidence_bucket].append({
            "confidence": confidence,
            "correct": is_correct,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent calibration data (last 100 entries per bucket)
        for bucket in rules["confidence_calibration"]:
            rules["confidence_calibration"][bucket] = rules["confidence_calibration"][bucket][-100:]
        
        self._save_learning_rules(rules)
    
    def _adaptive_threshold_adjustment(self, user_feedback: str, predicted_label: str, confidence: float):
        """Adjust confidence thresholds based on feedback patterns."""
        feedback_data = self._load_feedback_data()
        
        # Analyze recent performance for each category
        recent_feedback = [f for f in feedback_data 
                          if datetime.fromisoformat(f.get("timestamp", "2000-01-01")) > 
                          datetime.now() - timedelta(days=7)]
        
        # Calculate accuracy for each predicted category
        for category in ["spam", "transactional", "promotional"]:
            category_feedback = [f for f in recent_feedback 
                               if f.get("predicted") == category]
            
            if len(category_feedback) >= 5:  # Need sufficient data
                correct_predictions = len([f for f in category_feedback 
                                         if f.get("user_feedback") in ["correct", category]])
                accuracy = correct_predictions / len(category_feedback)
                
                # Adjust threshold based on accuracy
                current_threshold = self.adaptive_thresholds.get(category, MESSAGE_TYPE_RULES[category]["confidence_threshold"])
                
                if accuracy < 0.7:  # Too many mistakes
                    # Increase threshold to be more conservative
                    new_threshold = min(current_threshold * 1.1, 
                                      MESSAGE_TYPE_RULES[category]["max_threshold"])
                elif accuracy > 0.9:  # Very accurate
                    # Decrease threshold to catch more cases
                    new_threshold = max(current_threshold * 0.95, 
                                      MESSAGE_TYPE_RULES[category]["min_threshold"])
                else:
                    new_threshold = current_threshold
                
                if abs(new_threshold - current_threshold) > 0.05:
                    self.adaptive_thresholds[category] = new_threshold
                    print(f"🎯 Adjusted {category} threshold: {current_threshold:.3f} → {new_threshold:.3f}")
        
        self._save_adaptive_thresholds()
    
    def _temporal_pattern_learning(self, message: str, user_feedback: str):
        """Learn temporal patterns in spam/legitimate messages."""
        rules = self._load_learning_rules()
        
        current_time = datetime.now()
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()
        
        if "temporal_patterns" not in rules:
            rules["temporal_patterns"] = {"hourly": {}, "daily": {}}
        
        # Track patterns by hour and day
        hour_key = str(hour_of_day)
        day_key = str(day_of_week)
        
        for time_type, time_key in [("hourly", hour_key), ("daily", day_key)]:
            if time_key not in rules["temporal_patterns"][time_type]:
                rules["temporal_patterns"][time_type][time_key] = {
                    "spam": 0, "legitimate": 0
                }
            
            if user_feedback in ["spam", "false_negative"]:
                rules["temporal_patterns"][time_type][time_key]["spam"] += 1
            else:
                rules["temporal_patterns"][time_type][time_key]["legitimate"] += 1
        
        self._save_learning_rules(rules)
    
    def _learn_safe_patterns(self, message: str, correct_label: str):
        """Learn patterns that indicate safe messages."""
        rules = self._load_learning_rules()
        message_lower = message.lower()
        
        safe_indicators = []
        
        # Enhanced pattern detection
        if re.search(r'\b\d{4,8}\b.*(?:otp|code|verification|pin)', message_lower):
            safe_indicators.append("otp_pattern")
        
        if re.search(r'received.*rs\.?\s*\d+.*(?:via|from|upi)', message_lower):
            safe_indicators.append("upi_transaction")
        
        if re.search(r'(?:order|booking).*(?:#|number).*[a-z0-9]+', message_lower):
            safe_indicators.append("order_booking_pattern")
        
        if re.search(r'bill.*rs\.?\s*\d+.*(?:due|generated)', message_lower):
            safe_indicators.append("bill_notification")
        
        if re.search(r'pnr\s*:?\s*[a-z0-9]+', message_lower):
            safe_indicators.append("flight_pnr")
        
        # Add to learned patterns with category specificity
        for indicator in safe_indicators:
            pattern_key = f"{correct_label}_{indicator}"
            if pattern_key not in rules["learned_safe_patterns"]:
                rules["learned_safe_patterns"].append(pattern_key)
                print(f"✅ Learned new safe pattern: {pattern_key}")
        
        self._save_learning_rules(rules)
    
    def _learn_spam_patterns(self, message: str):
        """Learn patterns that indicate spam messages."""
        rules = self._load_learning_rules()
        message_lower = message.lower()
        
        spam_indicators = []
        
        if re.search(r'(?:win|won|winner).*(?:\$|\d+|prize|money)', message_lower):
            spam_indicators.append("fake_winner_pattern")
        
        if re.search(r'(?:free|get).*(?:instantly|immediately|now)', message_lower):
            spam_indicators.append("instant_reward_spam")
        
        if re.search(r'click.*(?:link|here).*(?:claim|get|win)', message_lower):
            spam_indicators.append("malicious_link_pattern")
        
        if re.search(r'reply.*(?:yes|stop|details).*(?:claim|get|win)', message_lower):
            spam_indicators.append("reply_scam_pattern")
        
        if len(re.findall(r'[!]{2,}', message)) > 0:
            spam_indicators.append("excessive_punctuation")
        
        for indicator in spam_indicators:
            if indicator not in rules["learned_spam_patterns"]:
                rules["learned_spam_patterns"].append(indicator)
                print(f"❌ Learned new spam pattern: {indicator}")
        
        self._save_learning_rules(rules)
    
    def _auto_update_whitelist(self, message: str, user_feedback: str):
        """Enhanced auto-whitelist with better pattern recognition."""
        if user_feedback not in ["false_positive", "transactional"]:
            return
        
        feedback_data = self._load_feedback_data()
        
        # Extract domains with better regex
        domains = re.findall(r'\b[a-zA-Z0-9][-a-zA-Z0-9]*[a-zA-Z0-9]*\.(?:com|org|net|in|co\.in|gov|edu)\b', message.lower())
        
        for domain in domains:
            domain_false_positives = [
                entry for entry in feedback_data 
                if (entry.get("user_feedback") in ["false_positive", "transactional"] and 
                    domain in entry.get("message_snippet", "").lower())
            ]
            
            if len(domain_false_positives) >= self.auto_whitelist_threshold:
                current_whitelist = load_whitelist()
                if domain not in current_whitelist["domains"]:
                    add_domain(domain)
                    
                    rules = self._load_learning_rules()
                    if domain not in rules["auto_whitelisted_domains"]:
                        rules["auto_whitelisted_domains"].append(domain)
                        self._save_learning_rules(rules)
                    
                    print(f"🤖 AUTO-WHITELISTED domain: {domain}")
        
        # Auto-whitelist high-confidence safe phrases
        words = message.lower().split()
        if len(words) >= 4:
            for i in range(len(words) - 3):
                phrase = " ".join(words[i:i+4])
                if len(phrase) > 15 and all(w.isalnum() or w in ['rs', 'upi', 'via'] for w in phrase.split()):
                    phrase_safe_count = len([
                        entry for entry in feedback_data 
                        if (entry.get("user_feedback") in ["false_positive", "transactional"] and 
                            phrase in entry.get("message_snippet", "").lower())
                    ])
                    
                    if phrase_safe_count >= self.auto_whitelist_threshold:
                        current_whitelist = load_whitelist()
                        if phrase not in current_whitelist["phrases"]:
                            add_phrase(phrase)
                            print(f"🤖 AUTO-WHITELISTED phrase: '{phrase}'")
    
    def _update_learning_stats(self):
        """Update comprehensive learning statistics."""
        rules = self._load_learning_rules()
        feedback_data = self._load_feedback_data()
        
        # Calculate accuracy metrics
        recent_feedback = [f for f in feedback_data if f.get("user_feedback") in ["correct", "false_positive", "false_negative"]]
        total_accuracy = 0
        if recent_feedback:
            correct_count = len([f for f in recent_feedback if f.get("user_feedback") == "correct"])
            total_accuracy = correct_count / len(recent_feedback)
        
        rules["learning_stats"] = {
            "total_feedback": len(feedback_data),
            "patterns_learned": len(rules.get("learned_safe_patterns", [])) + len(rules.get("learned_spam_patterns", [])),
            "auto_whitelists_added": len(rules.get("auto_whitelisted_domains", [])) + len(rules.get("auto_whitelisted_phrases", [])),
            "context_patterns": len(rules.get("context_patterns", {})),
            "weighted_patterns": len(self.pattern_weights.get("ngram_weights", {})),
            "accuracy": total_accuracy,
            "adaptive_thresholds_count": len(self.adaptive_thresholds),
            "last_updated": datetime.now().isoformat()
        }
        
        self._save_learning_rules(rules)
    
    def get_enhanced_prediction_score(self, message: str, base_category: str, base_confidence: float) -> Tuple[str, float]:
        """Get enhanced prediction using learned patterns."""
        message_lower = message.lower()
        words = message_lower.split()
        
        # Calculate learned pattern scores
        category_scores = {"spam": 0, "transactional": 0, "promotional": 0}
        
        # Apply n-gram weights
        for n in range(1, 4):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                if ngram in self.pattern_weights.get("ngram_weights", {}):
                    weights = self.pattern_weights["ngram_weights"][ngram]
                    for category in category_scores:
                        category_scores[category] += weights.get(category, 0)
        
        # Apply contextual patterns
        rules = self._load_learning_rules()
        context = {
            "has_numbers": bool(re.search(r'\d', message)),
            "has_currency": bool(re.search(r'rs\.?\s*\d+|\$\d+|₹\d+', message_lower)),
            "has_urgency": bool(re.search(r'\b(?:urgent|immediate|now|quick|limited)\b', message_lower)),
            "message_length": "short" if len(message) < 50 else "medium" if len(message) < 150 else "long",
            "has_call_to_action": bool(re.search(r'\b(?:click|call|reply|visit|download)\b', message_lower))
        }
        
        context_signature = "&".join([f"{k}={v}" for k, v in context.items() if v])
        if context_signature in rules.get("context_patterns", {}):
            context_weights = rules["context_patterns"][context_signature]
            for category in category_scores:
                category_scores[category] += context_weights.get(category, 0) * 0.5
        
        # Combine with base prediction
        if max(category_scores.values()) > 0:
            learned_category = max(category_scores, key=category_scores.get)
            learned_confidence = min(max(category_scores[learned_category] * 0.1, 0.0), 0.9)
            
            # Blend with base prediction
            if learned_category == base_category:
                enhanced_confidence = min((base_confidence + learned_confidence) / 2 + 0.1, 0.95)
                return base_category, enhanced_confidence
            elif learned_confidence > base_confidence * 1.5:  # Strong learned signal
                return learned_category, learned_confidence
        
        return base_category, base_confidence
    
    def get_learning_summary(self) -> Dict:
        """Get comprehensive learning summary."""
        rules = self._load_learning_rules()
        feedback_data = self._load_feedback_data()
        
        # Calculate advanced metrics
        recent_feedback = feedback_data[-50:] if len(feedback_data) > 50 else feedback_data
        accuracy_by_category = {}
        
        for category in ["spam", "transactional", "promotional"]:
            category_feedback = [f for f in recent_feedback if f.get("predicted") == category]
            if category_feedback:
                correct = len([f for f in category_feedback if f.get("user_feedback") in ["correct", category]])
                accuracy_by_category[category] = correct / len(category_feedback)
            else:
                accuracy_by_category[category] = 0
        
        return {
            "total_feedback_received": len(feedback_data),
            "learned_safe_patterns": len(rules.get("learned_safe_patterns", [])),
            "learned_spam_patterns": len(rules.get("learned_spam_patterns", [])),
            "auto_whitelisted_domains": len(rules.get("auto_whitelisted_domains", [])),
            "auto_whitelisted_phrases": len(rules.get("auto_whitelisted_phrases", [])),
            "context_patterns_learned": len(rules.get("context_patterns", {})),
            "weighted_ngrams": len(self.pattern_weights.get("ngram_weights", {})),
            "adaptive_thresholds": dict(self.adaptive_thresholds),
            "accuracy_by_category": accuracy_by_category,
            "recent_learning": rules.get("learning_stats", {}),
            "temporal_insights": self._get_temporal_insights(rules.get("temporal_patterns", {})),
            "examples": {
                "safe_patterns": rules.get("learned_safe_patterns", [])[-5:],
                "spam_patterns": rules.get("learned_spam_patterns", [])[-5:],
                "whitelisted_domains": rules.get("auto_whitelisted_domains", [])[-3:],
                "whitelisted_phrases": rules.get("auto_whitelisted_phrases", [])[-3:],
                "top_weighted_patterns": self._get_top_weighted_patterns(5)
            }
        }
    
    def _get_temporal_insights(self, temporal_patterns: Dict) -> Dict:
        """Get insights from temporal patterns."""
        insights = {"peak_spam_hours": [], "safe_hours": []}
        
        if "hourly" in temporal_patterns:
            hourly_data = temporal_patterns["hourly"]
            for hour, data in hourly_data.items():
                total = data.get("spam", 0) + data.get("legitimate", 0)
                if total > 5:  # Need sufficient data
                    spam_ratio = data.get("spam", 0) / total
                    if spam_ratio > 0.7:
                        insights["peak_spam_hours"].append(int(hour))
                    elif spam_ratio < 0.2:
                        insights["safe_hours"].append(int(hour))
        
        return insights
    
    def _get_top_weighted_patterns(self, limit: int) -> List[Dict]:
        """Get top weighted patterns for insight."""
        patterns = []
        
        for ngram, weights in self.pattern_weights.get("ngram_weights", {}).items():
            max_weight = max(weights.values())
            dominant_category = max(weights, key=weights.get)
            
            if max_weight > 0.5:  # Only significant patterns
                patterns.append({
                    "pattern": ngram,
                    "category": dominant_category,
                    "weight": max_weight
                })
        
        patterns.sort(key=lambda x: x["weight"], reverse=True)
        return patterns[:limit]

def clean_message(message: str) -> str:
    """Cleans message to match the preprocessing done during training."""
    message = message.lower()
    message = re.sub(r'\s+', ' ', message)
    message = re.sub(r'[^\w\s]', '', message)
    return message.strip()

def get_whitelist_regex():
    """Load whitelist dynamically and compile regex."""
    wl = load_whitelist()
    phrases = wl.get("phrases", [])
    domains = wl.get("domains", [])

    phrase_regex = re.compile("|".join(map(re.escape, phrases)), re.IGNORECASE) if phrases else None
    domain_regex = re.compile("|".join(map(re.escape, domains)), re.IGNORECASE) if domains else None
    return phrase_regex, domain_regex

def detect_message_type(message: str, feedback_manager: AdvancedFeedbackManager) -> Tuple[str, float]:
    """Enhanced message type detection with comprehensive learned patterns."""
    message_lower = message.lower()
    type_scores = {}
    
    # Load learned rules and patterns
    try:
        learning_rules_file = Path("data/learning_rules.yml")
        if learning_rules_file.exists():
            with open(learning_rules_file, "r") as f:
                rules = yaml.safe_load(f) or {}
            
            # Apply learned safe patterns
            for pattern in rules.get("learned_safe_patterns", []):
                if "otp" in pattern and re.search(r'\b\d{4,8}\b.*(?:otp|code|verification)', message_lower):
                    category = pattern.split("_")[0]
                    type_scores[category] = type_scores.get(category, 0) + 1.2
                elif "upi_transaction" in pattern and re.search(r'received.*rs\.?\s*\d+.*(?:via|from|upi)', message_lower):
                    category = pattern.split("_")[0]
                    type_scores[category] = type_scores.get(category, 0) + 1.1
                elif "order_booking" in pattern and re.search(r'(?:order|booking).*(?:#|number)', message_lower):
                    category = pattern.split("_")[0]
                    type_scores[category] = type_scores.get(category, 0) + 1.0
                elif "bill_notification" in pattern and re.search(r'bill.*rs\.?\s*\d+', message_lower):
                    category = pattern.split("_")[0]
                    type_scores[category] = type_scores.get(category, 0) + 1.0
            
            # Apply learned spam patterns
            for pattern in rules.get("learned_spam_patterns", []):
                if "fake_winner" in pattern and re.search(r'(?:win|won|winner).*(?:\$|\d+|prize)', message_lower):
                    type_scores["spam"] = type_scores.get("spam", 0) + 1.5
                elif "instant_reward" in pattern and re.search(r'(?:free|get).*(?:instantly|immediately)', message_lower):
                    type_scores["spam"] = type_scores.get("spam", 0) + 1.3
                elif "malicious_link" in pattern and re.search(r'click.*(?:link|here).*(?:claim|get)', message_lower):
                    type_scores["spam"] = type_scores.get("spam", 0) + 1.4
        
        # Apply weighted n-gram patterns
        enhanced_category, enhanced_confidence = feedback_manager.get_enhanced_prediction_score(
            message, "unknown", 0.0
        )
        if enhanced_category != "unknown":
            type_scores[enhanced_category] = type_scores.get(enhanced_category, 0) + enhanced_confidence * 2
                    
    except Exception as e:
        print(f"Error loading enhanced patterns: {e}")
    
    # Apply original rules with adaptive thresholds
    for msg_type, rules in MESSAGE_TYPE_RULES.items():
        base_score = type_scores.get(msg_type, 0)
        
        # Check keywords with learned weights
        for keyword in rules["keywords"]:
            if keyword in message_lower:
                # Use learned weight if available
                keyword_weight = feedback_manager.pattern_weights.get("ngram_weights", {}).get(keyword, {}).get(msg_type, 0.3)
                base_score += max(keyword_weight, 0.3)
        
        # Check regex patterns
        pattern_matches = sum(1 for pattern in rules["regex_patterns"] 
                             if re.search(pattern, message_lower))
        base_score += pattern_matches * 0.6
        
        type_scores[msg_type] = base_score
    
    # Return type with highest score
    if not type_scores or max(type_scores.values()) == 0:
        return "unknown", 0.0
    
    best_type = max(type_scores, key=type_scores.get)
    confidence = min(type_scores[best_type] / max(sum(type_scores.values()), 1.0), 1.0)
    
    return best_type, confidence

class SpamFilter:
    """Enhanced SMS filter with advanced learning system."""

    LABEL_MAP = {
        0: "transactional",
        1: "promotional", 
        2: "spam"
    }

    def __init__(self):
        self.feedback_manager = AdvancedFeedbackManager()

    def check_whitelist(self, message: str) -> Optional[str]:
        """Checks for whitelisted domains and phrases."""
        phrase_regex, domain_regex = get_whitelist_regex()
        lower_message = message.lower()
        
        if phrase_regex and phrase_regex.search(lower_message):
            return "whitelisted_phrase"
        if domain_regex and domain_regex.search(lower_message):
            return "whitelisted_domain"
        return None

    def get_prediction_confidence(self, message: str) -> Tuple[str, float]:
        """Get ML model prediction with confidence score, enhanced by learning."""
        if not model or not vectorizer:
            return "ai_model_not_loaded", 0.0

        cleaned_msg = clean_message(message)
        if not cleaned_msg:
            return "unclassified_empty_input", 0.0

        vectorized_msg = vectorizer.transform([cleaned_msg])
        
        # Get base prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(vectorized_msg)[0]
            prediction_raw = np.argmax(probabilities)
            base_confidence = float(np.max(probabilities))
        else:
            prediction_raw = model.predict(vectorized_msg)[0] 
            base_confidence = 0.8
        
        base_prediction = self.LABEL_MAP.get(prediction_raw, "unclassified")
        
        # Enhance with learned patterns
        enhanced_prediction, enhanced_confidence = self.feedback_manager.get_enhanced_prediction_score(
            message, base_prediction, base_confidence
        )
        
        print(f"🔍 ML Analysis: {prediction_raw} → {base_prediction} (base: {base_confidence:.3f})")
        if enhanced_prediction != base_prediction or abs(enhanced_confidence - base_confidence) > 0.1:
            print(f"🧠 Enhanced: {enhanced_prediction} (confidence: {enhanced_confidence:.3f})")
        
        return enhanced_prediction, enhanced_confidence

    def apply_message_type_rules(self, message: str, ml_prediction: str, 
                                ml_confidence: float) -> Dict[str, any]:
        """Apply enhanced message-type-specific filtering rules with adaptive thresholds."""
        
        # Detect message type with all enhancements
        detected_type, type_confidence = detect_message_type(message, self.feedback_manager)
        
        print(f"📋 Message type: {detected_type} (confidence: {type_confidence:.3f})")
        
        # Get adaptive threshold for this message type
        adaptive_threshold = self.feedback_manager.adaptive_thresholds.get(
            detected_type, 
            MESSAGE_TYPE_RULES.get(detected_type, {}).get("confidence_threshold", 0.5)
        )
        
        # Get rules for detected type
        if detected_type in MESSAGE_TYPE_RULES:
            rules = MESSAGE_TYPE_RULES[detected_type]
            
            # Always allow certain message types (like OTP)
            if rules.get("always_allow", False):
                return {
                    "verdict": "allowed",
                    "reason": f"always_allow_{detected_type}",
                    "confidence": type_confidence,
                    "message_type": detected_type,
                    "ml_prediction": ml_prediction,
                    "ml_confidence": ml_confidence,
                    "adaptive_threshold": adaptive_threshold,
                    "learning_enhanced": True
                }
            
            # Apply adaptive confidence threshold
            if ml_prediction == "spam" and ml_confidence >= adaptive_threshold:
                return {
                    "verdict": "blocked",
                    "reason": f"ai_spam_detection_{detected_type}",
                    "confidence": ml_confidence,
                    "message_type": detected_type,
                    "ml_prediction": ml_prediction,
                    "ml_confidence": ml_confidence,
                    "adaptive_threshold": adaptive_threshold,
                    "learning_enhanced": True
                }
        
        # Enhanced ML-based decision with temporal context
        current_hour = datetime.now().hour
        rules = self.feedback_manager._load_learning_rules()
        temporal_patterns = rules.get("temporal_patterns", {})
        
        # Check if current hour has high spam activity
        hour_data = temporal_patterns.get("hourly", {}).get(str(current_hour), {})
        if hour_data:
            total_msgs = hour_data.get("spam", 0) + hour_data.get("legitimate", 0)
            if total_msgs > 5:
                spam_ratio = hour_data.get("spam", 0) / total_msgs
                if spam_ratio > 0.7:  # High spam hour
                    ml_confidence *= 1.1  # Be more aggressive
                    print(f"⏰ High spam hour detected ({current_hour}:00), increasing vigilance")
        
        if ml_prediction == "spam":
            return {
                "verdict": "blocked", 
                "reason": "ai_spam_detection_enhanced",
                "confidence": ml_confidence,
                "message_type": detected_type,
                "ml_prediction": ml_prediction,
                "ml_confidence": ml_confidence,
                "adaptive_threshold": adaptive_threshold,
                "learning_enhanced": True
            }
        elif ml_prediction in ["transactional", "promotional"]:
            return {
                "verdict": "allowed",
                "reason": f"ai_classified_{ml_prediction}_enhanced",
                "confidence": ml_confidence,
                "message_type": detected_type,
                "ml_prediction": ml_prediction,
                "ml_confidence": ml_confidence,
                "adaptive_threshold": adaptive_threshold,
                "learning_enhanced": True
            }
        else:
            return {
                "verdict": "allowed",
                "reason": "ai_unclassified_default_allow",
                "confidence": 0.3,
                "message_type": detected_type,
                "ml_prediction": ml_prediction,
                "ml_confidence": ml_confidence,
                "adaptive_threshold": adaptive_threshold,
                "learning_enhanced": True
            }

    def process_message(self, message: str) -> Dict[str, any]:
        """Enhanced message processing with comprehensive learned patterns."""
        
        print(f"🔄 Processing with advanced learning: '{message[:50]}{'...' if len(message) > 50 else ''}'")
        
        # Check whitelist first (including auto-learned entries)
        whitelist_reason = self.check_whitelist(message)
        if whitelist_reason:
            detected_type, type_confidence = detect_message_type(message, self.feedback_manager)
            print(f"✅ WHITELISTED: {whitelist_reason}")
            return {
                "verdict": "allowed",
                "reason": whitelist_reason,
                "confidence": 1.0,
                "message_type": detected_type,
                "ml_prediction": "not_applicable",
                "ml_confidence": 0.0,
                "learning_enhanced": True
            }

        # Get enhanced ML prediction
        ml_prediction, ml_confidence = self.get_prediction_confidence(message)
        
        # Apply enhanced message-type-specific rules
        result = self.apply_message_type_rules(message, ml_prediction, ml_confidence)
        
        print(f"🎯 ENHANCED VERDICT: {result['verdict'].upper()} ({result['reason']})")
        return result
    
    def record_user_feedback(self, message: str, predicted_result: Dict[str, any], 
                           user_feedback: str):
        """Record user feedback and trigger comprehensive learning."""
        print(f"📨 Recording enhanced feedback: {user_feedback}")
        
        self.feedback_manager.record_feedback(
            message=message,
            predicted_label=predicted_result.get("ml_prediction", "unknown"),
            user_feedback=user_feedback,
            confidence=predicted_result.get("ml_confidence", 0.0)
        )
        
        # Return comprehensive learning summary
        return self.feedback_manager.get_learning_summary()
    
    def get_learning_summary(self) -> Dict:
        """Get comprehensive learning summary."""
        return self.feedback_manager.get_learning_summary()