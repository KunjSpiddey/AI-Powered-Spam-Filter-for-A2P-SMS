# SMS Spam Filter API with Learning Dashboard

An advanced, containerized API to classify SMS messages as transactional, promotional, or spam with real-time learning capabilities and a comprehensive web dashboard.

## ✨ Features

### Core Classification
- **3-Category Classification**: Differentiates between `Transactional`, `Promotional`, and `Spam`
- **Advanced Pattern Matching**: Uses keyword patterns, regex matching, and context analysis
- **Whitelisting Layer**: Immediately allows messages from trusted domains or containing safe phrases
- **Confidence Scoring**: Provides confidence levels (0.3-0.95) for each classification

### Learning System
- **Feedback Integration**: Learn from user corrections to improve accuracy
- **Pattern Learning**: Automatically discovers new spam/legitimate patterns
- **Adaptive Filtering**: Updates classification rules based on real-world feedback
- **Memory Persistence**: Stores learned patterns in YAML files

### Web Dashboard
- **Real-time Classification**: Test messages instantly through the web interface
- **Performance Analytics**: View classification statistics and accuracy metrics
- **Learning Insights**: Track what patterns the system has learned
- **Feedback System**: Correct classifications and train the model
- **Export Functionality**: Download classification data and statistics

### API Features
- **FastAPI Framework**: High-performance async endpoints
- **CORS Support**: Ready for frontend integration
- **Comprehensive Logging**: Track all classifications and errors
- **Health Monitoring**: Built-in health check endpoints

---

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine
- Python 3.9+ (for local development)

### 1. Project Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd sms_filter_project
   ```

2. **Create required directories:**
   ```bash
   mkdir -p data logs static models config
   ```

3. **Add the dashboard (optional):**
   Place the `dashboard.html` file in the `static/` directory for the web interface.

### 2. Build and Run with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t sms-filter-api .
   ```

2. **Run the Docker container:**
   ```bash
   docker run -p 8000:8000 -v $(pwd)/data:/app/data sms-filter-api
   ```

   The `-v` flag mounts the local data directory to persist learned patterns and feedback.

### 3. Local Development Setup

1. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn pyyaml python-multipart
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

The API will be running at `http://localhost:8000` with the dashboard at `http://localhost:8000/dashboard`.

---

## 🔧 Configuration

### Whitelist Configuration

Create `data/whitelist.yml` to define trusted senders and content:

```yaml
domains:
  - "icicibank.com"
  - "trip.com"
  - "amazon.com"
  - "google.com"

phrases:
  - "otp verification"
  - "transaction alert"
  - "order confirmation"
  - "booking confirmed"
```

### Learning Patterns

The system automatically creates and updates `data/learned_patterns.yml` based on user feedback:

```yaml
spam:
  - "congratulations"
  - "winner"
  - "claim"
  
transactional:
  - "verification"
  - "alert"
  - "confirmed"
  
promotional:
  - "discount"
  - "offer" 
  - "sale"
```

---

## ⚙️ API Usage

### Primary Classification Endpoint

**Endpoint**: `POST /classify_sms`
**Content-Type**: `application/json`

#### Example Request

```bash
curl -X POST "http://localhost:8000/classify_sms" \
-H "Content-Type: application/json" \
-d '{
    "message": "Your OTP for ICICI Bank transaction is 482651. Valid for 10 minutes."
}'
```

#### Example Response

```json
{
  "category": "transactional",
  "confidence": 0.92,
  "ml_prediction": "transactional",
  "reason": "OTP/verification message detected",
  "key_features": ["otp", "bank", "verification", "transaction"]
}
```

### Feedback Endpoint

**Endpoint**: `POST /feedback`

Provide feedback to improve the model:

```bash
curl -X POST "http://localhost:8000/feedback" \
-H "Content-Type: application/json" \
-d '{
    "message": "Win iPhone now! Click here",
    "predicted_category": "promotional", 
    "actual_category": "spam",
    "confidence": 0.65
}'
```

### Statistics Endpoint

**Endpoint**: `GET /stats`

Get classification performance metrics:

```json
{
  "spam_count": 23,
  "transactional_count": 145, 
  "promotional_count": 67,
  "total_classified": 235,
  "accuracy": 0.923
}
```

### Additional Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check for monitoring
- `GET /dashboard` - Web dashboard interface
- `GET /learning` - Learning insights and patterns
- `GET /patterns` - View current classification patterns
- `POST /test_classification` - Run system tests
- `GET /export_classification` - Export classification data

---

## 🎛️ Web Dashboard

Access the comprehensive web dashboard at `http://localhost:8000/dashboard`:

### Features:
- **Message Testing**: Classify messages in real-time
- **Performance Metrics**: View accuracy and classification statistics
- **Learning Progress**: Track system improvement over time
- **Feedback Interface**: Correct classifications to train the model
- **Category Distribution**: Visual breakdown of message types
- **Export Tools**: Download classification data and reports

### Dashboard Sections:
1. **Statistics Overview**: Real-time classification counts and accuracy
2. **Message Classification**: Test individual messages
3. **Learning Insights**: View learned patterns and examples
4. **Quick Actions**: Test system, view patterns, export data

---

## 🔍 Classification Logic

### Pattern-Based Scoring

The system uses multiple scoring mechanisms:

1. **Spam Indicators**: 'win', 'free', 'congratulations', 'prize', suspicious links
2. **Transactional Indicators**: 'otp', 'verification', 'bank', 'payment', numeric codes
3. **Promotional Indicators**: 'sale', 'offer', 'discount', '%', 'buy'

### Confidence Calculation

Confidence scores are calculated based on:
- Number of matching patterns
- Pattern strength and reliability
- Historical feedback accuracy
- Message context and structure

### Learning Mechanism

The system learns from feedback by:
- Extracting keywords from misclassified messages
- Adding patterns to correct categories
- Removing conflicting patterns from wrong categories
- Updating confidence thresholds based on accuracy

---

## 📊 Example Classifications

### Transactional (High Confidence)
```
"Your OTP for SBI account is 123456. Valid for 5 minutes."
→ Category: transactional, Confidence: 0.94
```

### Promotional (Medium Confidence)
```
"Flat 50% OFF on all electronics. Shop now at Amazon!"
→ Category: promotional, Confidence: 0.78
```

### Spam (High Confidence)
```
"CONGRATULATIONS! You won $1,000,000! Click http://suspicious-link.com"
→ Category: spam, Confidence: 0.89
```

---

## 🛠️ Development

### Project Structure

```
sms_filter_project/
├── main.py                 # Main FastAPI application
├── static/
│   └── dashboard.html      # Web dashboard
├── data/
│   ├── whitelist.yml       # Trusted senders/content
│   ├── learned_patterns.yml # AI-learned patterns
│   └── feedback.yml        # User feedback history
├── logs/
│   └── app.log            # Application logs
├── Dockerfile             # Container configuration
└── README.md              # This file
```

### Adding New Features

1. **Custom Patterns**: Add to the `spam_patterns`, `transactional_patterns`, or `promotional_patterns` lists in `SpamFilter` class
2. **New Endpoints**: Add FastAPI routes in `main.py`
3. **Dashboard Updates**: Modify `dashboard.html` for new UI features
4. **Learning Logic**: Extend the `_learn_from_correction` method

### Testing

The system includes built-in test functionality:
- Access `/test_classification` endpoint for automated tests
- Use the dashboard's "Test Classification" feature
- Monitor logs for classification accuracy

---

## 🐳 Docker Configuration

### Dockerfile Features

- Python 3.9 slim base image
- Automatic directory creation
- Port 8000 exposure
- Volume mounting for data persistence
- Uvicorn server with auto-reload

### Environment Variables

Set these environment variables if needed:
- `LOG_LEVEL`: Set logging level (default: INFO)
- `API_HOST`: Set host binding (default: 0.0.0.0)  
- `API_PORT`: Set port (default: 8000)

---

## 📈 Performance & Scalability

### Expected Performance
- **Latency**: < 50ms average response time
- **Throughput**: 1000+ requests per second
- **Accuracy**: 90%+ with sufficient training data
- **Memory Usage**: < 100MB base footprint

### Scaling Considerations
- Use Redis for shared learning patterns in multi-instance deployments
- Database integration for large-scale feedback storage
- Load balancing for high-traffic scenarios
- Caching layer for frequently classified patterns

---

## 🔒 Security Notes

- Input validation on all message content
- Rate limiting recommended for production
- CORS configured for frontend integration
- No persistent storage of sensitive message content
- Logging excludes full message content by default

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.