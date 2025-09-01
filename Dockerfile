# Use Python 3.11 to support scikit-learn 1.7.1
FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (not just app/)
COPY . .

# Create required directories
RUN mkdir -p data logs static models config

# Expose port (if your main.py runs a web service, e.g., FastAPI/Flask)
EXPOSE 8000

# Fixed command - main.py is in root, not app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]