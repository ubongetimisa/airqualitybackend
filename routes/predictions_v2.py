"""
Production-Ready Predictions API v2.0

Comprehensive prediction endpoint with:
- Full model loader integration
- Real feature engineering (173 features)
- Ensemble stacking predictions
- Health impact assessment
- Confidence scoring
- Database persistence
- Detailed logging

Architecture:
1. Accept raw air quality data + location + date
2. Engineer 173 features (temporal, lag, rolling, interactions, ratios, encoding)
3. Scale features using StandardScaler
4. Get predictions from:
   - 9 traditional ML base models (SVM, Ridge, Lasso, etc.)
   - Ensemble meta-model (Ridge regressor on stacked predictions)
5. Calculate confidence scores and health impacts
6. Save to MongoDB and return results

Author: Ubong Isaiah Eka
Date: 2025
"""

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv

# Import model loader
from utils.model_loader import ModelLoader

load_dotenv()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2/predictions", tags=["predictions_v2"])

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "")
DB_NAME = "air_quality_db"
PREDICTIONS_COLLECTION = "predictions"

# ==================== Pydantic Models ====================

class AirQualityInput(BaseModel):
    """Raw air quality measurement data"""
    pm10: float = Field(..., ge=0, le=500, description="PM10 concentration (µg/m³)")
    no2: float = Field(..., ge=0, le=500, description="NO2 concentration (ppb)")
    so2: float = Field(..., ge=0, le=500, description="SO2 concentration (ppb)")
    co: float = Field(..., ge=0, le=50, description="CO concentration (ppm)")
    o3: float = Field(..., ge=0, le=500, description="O3 concentration (ppb)")
    temperature: float = Field(..., ge=-50, le=60, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative Humidity (%)")
    wind_speed: float = Field(..., ge=0, le=50, description="Wind Speed (m/s)")
    
    @validator("humidity")
    def validate_humidity(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Humidity must be between 0 and 100%")
        return v


class LocationData(BaseModel):
    """Location information"""
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")
    date: str = Field(..., description="Date in format YYYY-MM-DD or ISO format")
    
    @validator("date")
    def validate_date(cls, v):
        try:
            pd.to_datetime(v)
            return v
        except:
            raise ValueError("Invalid date format. Use YYYY-MM-DD or ISO format")


class PredictionRequestV2(BaseModel):
    """Complete prediction request"""
    location: LocationData
    air_quality: AirQualityInput
    model: Optional[str] = Field("ensemble", description="Model to use: 'ensemble', specific model name, or 'all'")
    use_historical_data: bool = Field(False, description="Whether to retrieve historical data for lag features")
    confidence_threshold: Optional[float] = Field(None, description="Only return if confidence >= threshold")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    return_feature_importance: bool = Field(False, description="Return feature importance scores")


class HealthImpactAssessment(BaseModel):
    """Health impact assessment"""
    risk_level: str
    aqi_category: str
    health_implications: List[str]
    recommended_actions: List[str]
    affected_groups: List[str]


class PredictionMetadata(BaseModel):
    """Prediction metadata"""
    features_engineered: int
    features_used: int
    model_type: str
    model_performance: Optional[Dict[str, float]] = None
    confidence_score: float
    processing_time_ms: float


class PredictionResultV2(BaseModel):
    """Complete prediction result"""
    prediction_id: str
    pm25_predicted: float
    unit: str = "µg/m³"
    status: str
    timestamp: str
    
    # Input data
    input_location: LocationData
    input_measurements: Dict[str, float]
    
    # Prediction details
    model_used: str
    base_model_predictions: Optional[Dict[str, float]] = None
    
    # Health & Impact
    health_impact: HealthImpactAssessment
    
    # Quality metrics
    metadata: PredictionMetadata
    
    # Optional
    feature_importance: Optional[Dict[str, float]] = None
    error: Optional[str] = None


# ==================== Global Model Loader ====================

global_model_loader: Optional[ModelLoader] = None


def initialize_model_loader():
    """Initialize global model loader"""
    global global_model_loader
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(current_dir)
        artifacts_dir = os.path.join(backend_dir, 'trained_model')
        
        global_model_loader = ModelLoader(artifacts_dir)
        global_model_loader.load_all()
        
        if global_model_loader.is_ready:
            logger.info(f"✓ Model loader initialized with {len(global_model_loader.models)} models")
            return True
        else:
            logger.error("❌ Models failed to load")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing model loader: {e}")
        global_model_loader = None
        return False


# ==================== Health Assessment ====================

def assess_health_impact(pm25: float) -> HealthImpactAssessment:
    """
    Assess health impact based on PM2.5 level using WHO/US EPA standards.
    
    Args:
        pm25: PM2.5 concentration in µg/m³
        
    Returns:
        HealthImpactAssessment object
    """
    # WHO/EPA AQI categories
    if pm25 <= 12:
        return HealthImpactAssessment(
            risk_level="Good",
            aqi_category="0-50",
            health_implications=[
                "Air quality is satisfactory",
                "No health risks expected"
            ],
            recommended_actions=["Enjoy outdoor activities"],
            affected_groups=["None"]
        )
    
    elif pm25 <= 35.4:
        return HealthImpactAssessment(
            risk_level="Moderate",
            aqi_category="51-100",
            health_implications=[
                "Air quality is acceptable",
                "Sensitive groups may experience minor respiratory effects"
            ],
            recommended_actions=[
                "Sensitive groups may limit prolonged outdoor exertion",
                "Consider air quality when planning activities"
            ],
            affected_groups=["Children", "Elderly", "People with respiratory/cardiac disease"]
        )
    
    elif pm25 <= 55.4:
        return HealthImpactAssessment(
            risk_level="Unhealthy for Sensitive Groups",
            aqi_category="101-150",
            health_implications=[
                "Members of sensitive groups may experience health effects",
                "Increased aggravation of existing respiratory or heart conditions",
                "General population beginning to be affected"
            ],
            recommended_actions=[
                "Sensitive groups should reduce prolonged outdoor exertion",
                "Use N95 masks for outdoor activities",
                "Keep windows closed"
            ],
            affected_groups=["Children", "Elderly", "People with asthma", "People with heart disease"]
        )
    
    elif pm25 <= 150.4:
        return HealthImpactAssessment(
            risk_level="Unhealthy",
            aqi_category="151-200",
            health_implications=[
                "Everyone may begin to experience health effects",
                "Increased respiratory and cardiovascular effects",
                "Increased risk of hospitalizations"
            ],
            recommended_actions=[
                "Avoid prolonged outdoor activities",
                "Use N95 masks for any outdoor activity",
                "Spend time in air-filtered environments",
                "Limit children's outdoor play"
            ],
            affected_groups=["All population groups"]
        )
    
    else:
        return HealthImpactAssessment(
            risk_level="Very Unhealthy/Hazardous",
            aqi_category=">200",
            health_implications=[
                "Health alert: everyone may experience serious health effects",
                "Significant aggravation of heart or lung disease",
                "Increased mortality risk"
            ],
            recommended_actions=[
                "Avoid all outdoor activity",
                "Remain indoors with air-conditioned/filtered environment",
                "Use HEPA air purifiers",
                "Seek medical attention if symptoms develop"
            ],
            affected_groups=["All population groups", "Healthcare workers"]
        )


def calculate_confidence_score(base_predictions: Dict[str, float], ensemble_pred: float) -> float:
    """
    Calculate prediction confidence based on:
    - Agreement between base models (lower std = higher confidence)
    - Value range (unrealistic values = lower confidence)
    
    Args:
        base_predictions: Dictionary of base model predictions
        ensemble_pred: Ensemble prediction value
        
    Returns:
        Confidence score between 0 and 1
    """
    try:
        if not base_predictions:
            return 0.5
        
        # Get prediction values
        pred_values = np.array(list(base_predictions.values()))
        
        # Model agreement (lower variance = higher confidence)
        pred_variance = np.var(pred_values)
        agreement_score = 1.0 / (1.0 + pred_variance / 100)  # Normalize
        
        # Value reasonableness (typical PM2.5 is 0-100 µg/m³)
        if 0 <= ensemble_pred <= 100:
            value_score = 1.0
        elif 0 <= ensemble_pred <= 200:
            value_score = 0.8
        elif 0 <= ensemble_pred <= 300:
            value_score = 0.6
        else:
            value_score = 0.3
        
        # Combined confidence
        confidence = (agreement_score * 0.4) + (value_score * 0.6)
        confidence = max(0.0, min(1.0, confidence))
        
        return float(confidence)
        
    except Exception as e:
        logger.warning(f"Error calculating confidence: {e}")
        return 0.5


def save_prediction_to_db(prediction_data: Dict[str, Any]) -> str:
    """
    Save prediction to MongoDB.
    
    Args:
        prediction_data: Complete prediction data
        
    Returns:
        Prediction ID (MongoDB ObjectId as string)
    """
    try:
        if not MONGODB_URI:
            logger.warning("MongoDB not configured, skipping DB save")
            return str(ObjectId())
        
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        
        # Log what we're about to save
        user_id = prediction_data.get('user_id', 'UNKNOWN')
        logger.info(f"💾 About to insert prediction with user_id='{user_id}' (type: {type(user_id).__name__})")
        logger.debug(f"Prediction data keys: {list(prediction_data.keys())}")
        
        # Save prediction
        result = db[PREDICTIONS_COLLECTION].insert_one(prediction_data)
        logger.info(f"✓ Prediction inserted to MongoDB with _id={result.inserted_id}, user_id='{user_id}'")
        
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {e}", exc_info=True)
        return str(ObjectId())


# ==================== API Endpoints ====================

@router.on_event("startup")
async def startup():
    """Initialize models on startup"""
    logger.info("Initializing prediction models...")
    initialize_model_loader()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check health of prediction service"""
    if global_model_loader is None or not global_model_loader.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not initialized"
        )
    
    return {
        "status": "healthy",
        "models_loaded": len(global_model_loader.models),
        "features_count": len(global_model_loader.feature_names) if global_model_loader.feature_names else 0,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """List all available models"""
    if global_model_loader is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    return {
        "models": list(global_model_loader.models.keys()),
        "count": len(global_model_loader.models),
        "default": "ensemble",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/predict", response_model=PredictionResultV2)
async def predict_pm25(request: PredictionRequestV2, background_tasks: BackgroundTasks) -> PredictionResultV2:
    """
    Make a PM2.5 prediction from raw air quality measurements.
    
    **Input Requirements:**
    - location: City, Country, Date (YYYY-MM-DD)
    - air_quality: PM10, NO2, SO2, CO, O3, Temperature, Humidity, Wind Speed
    
    **Processing:**
    1. Validate all inputs (ranges, types)
    2. Engineer 173 features from raw measurements
    3. Scale features using trained StandardScaler
    4. Get predictions from ensemble and base models
    5. Calculate confidence score
    6. Assess health impact
    7. Return comprehensive prediction
    
    **Output:**
    - PM2.5 prediction in µg/m³
    - Confidence score (0-1)
    - Health impact assessment
    - Model details and metadata
    """
    
    start_time = datetime.now()
    
    # Log the request user_id
    logger.info(f"📨 Prediction request received with user_id: {request.user_id}")
    
    try:
        # Validate models are loaded
        if global_model_loader is None or not global_model_loader.is_ready:
            logger.error("Models not ready for prediction")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction models not initialized"
            )
        
        logger.info(f"Prediction request from {request.location.city}, {request.location.country}")
        
        # ========== STEP 1: Prepare raw input ==========
        raw_input = {
            'City': request.location.city,
            'Country': request.location.country,
            'Date': request.location.date,
            'PM10': request.air_quality.pm10,
            'NO2': request.air_quality.no2,
            'SO2': request.air_quality.so2,
            'CO': request.air_quality.co,
            'O3': request.air_quality.o3,
            'Temperature': request.air_quality.temperature,
            'Humidity': request.air_quality.humidity,
            'Wind Speed': request.air_quality.wind_speed
        }
        
        logger.debug(f"Raw input: {raw_input}")
        
        # ========== STEP 2: Engineer features ==========
        X_scaled, error = global_model_loader.engineer_features_for_prediction(raw_input)
        
        if X_scaled is None:
            logger.error(f"Feature engineering failed: {error}")
            raise HTTPException(status_code=400, detail=f"Feature engineering failed: {error}")
        
        logger.info(f"Features engineered: shape {X_scaled.shape}")
        
        # ========== STEP 3: Make predictions ==========
        
        # Determine which models to use
        if request.model == "all":
            models_to_use = list(global_model_loader.models.keys())
        elif request.model:
            models_to_use = [request.model]
        else:
            models_to_use = ["ensemble"]
        
        # Get base model predictions (for ensemble understanding)
        base_model_predictions = {}
        base_models_to_query = ['svm_linear', 'lasso_regression', 'ridge_regression', 'linear_regression']
        
        for base_model_name in base_models_to_query:
            if base_model_name in global_model_loader.models:
                try:
                    result = global_model_loader.predict(X_scaled, base_model_name)
                    if result.get('prediction') is not None:
                        base_model_predictions[base_model_name] = result['prediction']
                        logger.debug(f"  {base_model_name}: {result['prediction']:.2f}")
                except Exception as e:
                    logger.debug(f"Failed to get {base_model_name} prediction: {e}")
        
        # Get ensemble prediction (uses stacking)
        ensemble_result = global_model_loader.predict(X_scaled, "ensemble")
        
        if ensemble_result.get('error'):
            logger.error(f"Ensemble prediction failed: {ensemble_result['error']}")
            raise HTTPException(status_code=400, detail="Prediction failed")
        
        ensemble_pred = ensemble_result['prediction']
        logger.info(f"Ensemble prediction: {ensemble_pred:.2f} µg/m³")
        
        # ========== STEP 4: Calculate confidence ==========
        confidence_score = calculate_confidence_score(base_model_predictions, ensemble_pred)
        logger.debug(f"Confidence score: {confidence_score:.3f}")
        
        # Check against threshold if provided
        if request.confidence_threshold and confidence_score < request.confidence_threshold:
            logger.warning(f"Confidence below threshold: {confidence_score:.3f} < {request.confidence_threshold}")
            raise HTTPException(
                status_code=400,
                detail=f"Confidence score {confidence_score:.3f} below threshold {request.confidence_threshold}"
            )
        
        # ========== STEP 5: Assess health impact ==========
        health_impact = assess_health_impact(ensemble_pred)
        logger.info(f"Health impact: {health_impact.risk_level}")
        
        # ========== STEP 6: Calculate processing time ==========
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # ========== STEP 7: Prepare response ==========
        prediction_id = str(ObjectId())
        
        response = PredictionResultV2(
            prediction_id=prediction_id,
            pm25_predicted=float(ensemble_pred),
            status="success",
            timestamp=datetime.utcnow().isoformat(),
            
            input_location=request.location,
            input_measurements=request.air_quality.dict(),
            
            model_used="ensemble",
            base_model_predictions=base_model_predictions if base_model_predictions else None,
            
            health_impact=health_impact,
            
            metadata=PredictionMetadata(
                features_engineered=173,
                features_used=X_scaled.shape[1],
                model_type="Stacking Ensemble with Ridge Meta-Learner",
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms
            )
        )
        
        # ========== STEP 8: Save to database in background ==========
        db_record = {
            '_id': ObjectId(prediction_id),
            'user_id': request.user_id or 'anonymous',
            'city': request.location.city,
            'country': request.location.country,
            'date': request.location.date,
            'pm25_predicted': float(ensemble_pred),
            'predicted_pm25': float(ensemble_pred),  # Keep both for compatibility
            'confidence_score': confidence_score,
            'health_impact': {
                'risk_level': health_impact.risk_level,
                'aqi_category': health_impact.aqi_category,
                'health_implications': health_impact.health_implications
            },
            'health_risk_level': health_impact.risk_level,  # Keep for compatibility
            'base_predictions': base_model_predictions,
            'input_measurements': request.air_quality.dict(),
            'processing_time_ms': processing_time_ms,
            'created_at': datetime.utcnow(),
            'metadata': {
                'features_engineered': 173,
                'features_used': X_scaled.shape[1],
                'model_type': 'Stacking Ensemble with Ridge Meta-Learner',
                'confidence_score': confidence_score,
                'processing_time_ms': processing_time_ms
            }
        }
        
        logger.info(f"💾 Saving prediction to DB with user_id='{db_record['user_id']}' and prediction_id='{prediction_id}'")
        background_tasks.add_task(save_prediction_to_db, db_record)
        
        logger.info(f"✓ Prediction complete: {prediction_id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict-batch")
async def predict_batch(requests: List[PredictionRequestV2]) -> Dict[str, Any]:
    """
    Make multiple predictions in batch.
    
    Returns a list of predictions with summary statistics.
    """
    if global_model_loader is None or not global_model_loader.is_ready:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    results = []
    successes = 0
    failures = 0
    
    logger.info(f"Processing batch of {len(requests)} predictions")
    
    for i, request in enumerate(requests):
        try:
            logger.debug(f"Batch prediction {i+1}/{len(requests)}")
            
            # Process each prediction
            # (Using same logic as single prediction)
            raw_input = {
                'City': request.location.city,
                'Country': request.location.country,
                'Date': request.location.date,
                'PM10': request.air_quality.pm10,
                'NO2': request.air_quality.no2,
                'SO2': request.air_quality.so2,
                'CO': request.air_quality.co,
                'O3': request.air_quality.o3,
                'Temperature': request.air_quality.temperature,
                'Humidity': request.air_quality.humidity,
                'Wind Speed': request.air_quality.wind_speed
            }
            
            X_scaled, error = global_model_loader.engineer_features_for_prediction(raw_input)
            
            if X_scaled is None:
                results.append({
                    "city": request.location.city,
                    "error": error,
                    "prediction": None
                })
                failures += 1
                continue
            
            ensemble_result = global_model_loader.predict(X_scaled, "ensemble")
            
            if ensemble_result.get('error'):
                results.append({
                    "city": request.location.city,
                    "error": ensemble_result['error'],
                    "prediction": None
                })
                failures += 1
            else:
                results.append({
                    "city": request.location.city,
                    "prediction": ensemble_result['prediction'],
                    "health_risk": assess_health_impact(ensemble_result['prediction']).risk_level
                })
                successes += 1
                
        except Exception as e:
            logger.warning(f"Batch prediction error for {request.location.city}: {e}")
            results.append({
                "city": request.location.city,
                "error": str(e),
                "prediction": None
            })
            failures += 1
    
    logger.info(f"Batch complete: {successes} successes, {failures} failures")
    
    return {
        "predictions": results,
        "summary": {
            "total": len(requests),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / len(requests) if requests else 0
        }
    }


@router.get("/feature-info")
async def get_feature_info() -> Dict[str, Any]:
    """Get information about engineered features"""
    if global_model_loader is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    return {
        "total_features": len(global_model_loader.feature_names) if global_model_loader.feature_names else 0,
        "feature_categories": {
            "input": 8,  # PM10, NO2, SO2, CO, O3, Temp, Humidity, Wind
            "temporal": 14,  # Year, month, day, etc + cyclical encodings
            "interactions": 6,  # Meteorological interactions
            "lag": 54,  # 9 features × 6 lag periods
            "rolling": 81,  # 9 features × 3 windows × 3 stats
            "ratios": 4,  # Pollutant ratios
            "categorical": 6  # City and Country one-hot encoding (approx)
        },
        "feature_names_sample": global_model_loader.feature_names[:10] if global_model_loader.feature_names else []
    }


@router.get("/health-scale")
async def get_health_scale() -> Dict[str, Any]:
    """Get WHO/EPA health impact scale"""
    return {
        "scale": [
            {
                "level": "Good",
                "aqi_range": "0-50",
                "pm25_range": "0-12 µg/m³",
                "health_effects": "None",
                "recommended_actions": ["Enjoy outdoor activities"]
            },
            {
                "level": "Moderate",
                "aqi_range": "51-100",
                "pm25_range": "12-35.4 µg/m³",
                "health_effects": "Sensitive groups may experience minor respiratory effects",
                "recommended_actions": ["Sensitive groups may limit prolonged outdoor exertion"]
            },
            {
                "level": "Unhealthy for Sensitive Groups",
                "aqi_range": "101-150",
                "pm25_range": "35.5-55.4 µg/m³",
                "health_effects": "Increased respiratory effects in sensitive groups",
                "recommended_actions": ["Sensitive groups avoid outdoor activities", "Use N95 masks"]
            },
            {
                "level": "Unhealthy",
                "aqi_range": "151-200",
                "pm25_range": "55.5-150.4 µg/m³",
                "health_effects": "Increased respiratory and cardiovascular effects in all groups",
                "recommended_actions": ["Avoid prolonged outdoor activities", "Use air-conditioned environments"]
            },
            {
                "level": "Very Unhealthy",
                "aqi_range": "201-300",
                "pm25_range": "150.5+ µg/m³",
                "health_effects": "Significant health effects for all population",
                "recommended_actions": ["Avoid all outdoor activity", "Use HEPA air purifiers"]
            }
        ]
    }
