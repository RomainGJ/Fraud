from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
from datetime import datetime
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import mlflow
import mlflow.sklearn

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('fraud_predictions_total', 'Total fraud predictions', ['outcome'])
PREDICTION_LATENCY = Histogram('fraud_prediction_duration_seconds', 'Prediction latency')
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])

# Pydantic models for request/response validation
class TransactionFeatures(BaseModel):
    transaction_amount: float = Field(..., description="Transaction amount", gt=0)
    account_age_days: int = Field(..., description="Account age in days", ge=0)
    merchant_category: str = Field(..., description="Merchant category")
    time_of_day: int = Field(..., description="Hour of transaction (0-23)", ge=0, le=23)
    day_of_week: int = Field(..., description="Day of week (0-6)", ge=0, le=6)
    transaction_count_last_hour: int = Field(..., description="Transactions in last hour", ge=0)
    average_transaction_amount: float = Field(..., description="User's average transaction amount", gt=0)
    location_risk_score: float = Field(..., description="Location risk score (0-1)", ge=0, le=1)
    payment_method: str = Field(..., description="Payment method")
    transaction_id: Optional[str] = Field(None, description="Optional transaction ID")

class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionFeatures]

class FraudPrediction(BaseModel):
    transaction_id: Optional[str]
    is_fraud: bool
    fraud_probability: float
    is_anomaly: bool
    risk_level: str
    confidence: float
    prediction_time: datetime
    model_version: Optional[str]

class BatchFraudResponse(BaseModel):
    results: List[FraudPrediction]
    total_processed: int
    successful_predictions: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float
    total_predictions: int

# FastAPI app initialization
app = FastAPI(
    title="FraudGuard API",
    description="Advanced fraud detection system with ML pipeline",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
fraud_detector = None
data_processor = None
model_version = None
start_time = time.time()

class ModelManager:
    def __init__(self):
        self.model = None
        self.processor = None
        self.version = None
        self.last_update = None

    async def load_model(self, model_path: str = None):
        """Load model from MLflow or local path"""
        try:
            if model_path and os.path.exists(model_path):
                # Load from local path
                self.model = joblib.load(model_path)
                self.version = "local"
                logger.info(f"Model loaded from local path: {model_path}")
            else:
                # Try to load from MLflow
                try:
                    mlflow.set_tracking_uri("http://mlflow:5000")  # MLflow server
                    model_name = "fraud-detection-model"
                    stage = "Production"

                    model_version_info = mlflow.pyfunc.load_model(
                        model_uri=f"models:/{model_name}/{stage}"
                    )
                    self.model = model_version_info
                    self.version = stage
                    logger.info(f"Model loaded from MLflow: {model_name}/{stage}")
                except Exception as mlflow_error:
                    logger.warning(f"MLflow model load failed: {mlflow_error}")
                    # Fallback to local model
                    fallback_path = "models/fraud_detector.pkl"
                    if os.path.exists(fallback_path):
                        self.model = joblib.load(fallback_path)
                        self.version = "fallback"
                        logger.info("Loaded fallback local model")
                    else:
                        raise Exception("No model available")

            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def is_healthy(self) -> bool:
        return self.model is not None

    async def predict_single(self, features: TransactionFeatures) -> Dict:
        """Make prediction for single transaction"""
        if not self.model:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Convert to dict for model prediction
            feature_dict = features.dict()
            feature_dict.pop('transaction_id', None)

            # Use the model's predict_single_transaction method
            result = self.model.predict_single_transaction(feature_dict)
            result['model_version'] = self.version
            result['prediction_time'] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize model manager
model_manager = ModelManager()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting FraudGuard API...")

    try:
        await model_manager.load_model()
        logger.info("FraudGuard API started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    API_REQUESTS.labels(method="GET", endpoint="/health").inc()

    return HealthResponse(
        status="healthy" if model_manager.is_healthy() else "unhealthy",
        model_loaded=model_manager.is_healthy(),
        model_version=model_manager.version,
        uptime_seconds=time.time() - start_time,
        total_predictions=int(PREDICTION_COUNTER._value.sum())
    )

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Single prediction endpoint
@app.post("/api/v1/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionFeatures):
    """Predict fraud for a single transaction"""
    API_REQUESTS.labels(method="POST", endpoint="/api/v1/predict").inc()

    with PREDICTION_LATENCY.time():
        try:
            result = await model_manager.predict_single(transaction)

            # Update metrics
            outcome = "fraud" if result['is_fraud'] else "legitimate"
            PREDICTION_COUNTER.labels(outcome=outcome).inc()

            # Create response
            prediction = FraudPrediction(
                transaction_id=transaction.transaction_id,
                is_fraud=result['is_fraud'],
                fraud_probability=result['fraud_probability'],
                is_anomaly=result['is_anomaly'],
                risk_level=result['risk_level'],
                confidence=result['confidence'],
                prediction_time=result['prediction_time'],
                model_version=result['model_version']
            )

            logger.info(f"Prediction made: fraud={result['is_fraud']}, risk={result['risk_level']}")
            return prediction

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/api/v1/predict/batch", response_model=BatchFraudResponse)
async def predict_fraud_batch(batch_request: BatchTransactionRequest):
    """Predict fraud for multiple transactions"""
    API_REQUESTS.labels(method="POST", endpoint="/api/v1/predict/batch").inc()

    start_time_batch = time.time()
    results = []
    successful_predictions = 0

    for transaction in batch_request.transactions:
        try:
            result = await model_manager.predict_single(transaction)

            prediction = FraudPrediction(
                transaction_id=transaction.transaction_id,
                is_fraud=result['is_fraud'],
                fraud_probability=result['fraud_probability'],
                is_anomaly=result['is_anomaly'],
                risk_level=result['risk_level'],
                confidence=result['confidence'],
                prediction_time=result['prediction_time'],
                model_version=result['model_version']
            )

            results.append(prediction)
            successful_predictions += 1

            # Update metrics
            outcome = "fraud" if result['is_fraud'] else "legitimate"
            PREDICTION_COUNTER.labels(outcome=outcome).inc()

        except Exception as e:
            logger.error(f"Batch prediction error for transaction: {str(e)}")
            # You might want to include failed predictions in response
            continue

    processing_time = (time.time() - start_time_batch) * 1000  # Convert to ms

    return BatchFraudResponse(
        results=results,
        total_processed=len(batch_request.transactions),
        successful_predictions=successful_predictions,
        processing_time_ms=processing_time
    )

# Model management endpoints
@app.post("/api/v1/model/reload")
async def reload_model():
    """Reload the fraud detection model"""
    API_REQUESTS.labels(method="POST", endpoint="/api/v1/model/reload").inc()

    try:
        await model_manager.load_model()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.get("/api/v1/model/info")
async def model_info():
    """Get information about the current model"""
    API_REQUESTS.labels(method="GET", endpoint="/api/v1/model/info").inc()

    if not model_manager.is_healthy():
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_version": model_manager.version,
        "last_update": model_manager.last_update,
        "status": "healthy",
        "total_predictions": int(PREDICTION_COUNTER._value.sum())
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Main function for running the app
def main():
    """Run the FastAPI application"""
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )

if __name__ == "__main__":
    main()