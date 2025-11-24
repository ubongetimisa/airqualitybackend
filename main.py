# backend/main.py
from fastapi import FastAPI, HTTPException, Depends, status, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import jwt #type:ignore
from passlib.context import CryptContext
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv
import logging
import json
from utils.logger_config import get_logger, app_logger, error_logger, api_logger

load_dotenv()

# Initialize logger with a fixed name (not __name__ to avoid uvicorn reload issues)
logger = get_logger("air_quality_api")

# Initialize FastAPI app
app = FastAPI(
    title="Air Quality Prediction API",
    description="Post Graduate Research Grade Air Quality Prediction and Analysis Platform",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
# Import and register v2 predictions router (primary active router)
from routes import predictions_v2
app.include_router(predictions_v2.router)

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    error_msg = "❌ MONGODB_URI not found in .env file. Please set it before running the application."
    logger.critical(error_msg)
    raise ValueError(error_msg)

logger.info("✓ Connecting to MongoDB Atlas...")
client = MongoClient(MONGODB_URI)
db = client.air_quality_db

# Verify connection
try:
    client.admin.command('ping')
    logger.info("✓ Successfully connected to MongoDB Atlas")
except Exception as e:
    logger.error(f"❌ Failed to connect to MongoDB: {e}", exc_info=True)
    raise

# ==================== Startup Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=" * 80)
    logger.info("🚀 APPLICATION STARTUP")
    logger.info("=" * 80)
    
    # Initialize prediction models
    try:
        from utils.model_loader import initialize_models
        if initialize_models():  # Auto-detects path
            logger.info("✓ Prediction models initialized successfully")
        else:
            logger.warning("⚠ Some prediction models failed to initialize")
    except Exception as e:
        logger.warning(f"⚠ Could not initialize prediction models: {e}")
    
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 APPLICATION SHUTDOWN")
    try:
        client.close()
        logger.info("✓ MongoDB connection closed")
    except:
        pass

# Security
# Configure CryptContext with explicit bcrypt settings to avoid version warnings
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Explicitly set rounds to avoid version detection
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    affiliation: Optional[str] = ""
    research_interests: Optional[List[str]] = None

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    affiliation: Optional[str] = ""
    research_interests: List[str] = []
    created_at: datetime

class FeedbackCreate(BaseModel):
    prediction_id: str
    rating: int
    comments: str
    model_accuracy: int

class FeedbackResponse(BaseModel):
    id: str
    user_id: str
    prediction_id: str
    rating: int
    comments: str
    model_accuracy: int
    created_at: datetime

# Utility functions
def verify_password(plain_password, hashed_password):
    """
    Verify password against hash. BCrypt limit is 72 bytes, so we truncate if needed.
    """
    # Truncate password to 72 bytes (same as in get_password_hash)
    truncated_password = plain_password[:72] if isinstance(plain_password, str) else plain_password.decode('utf-8')[:72]
    return pwd_context.verify(truncated_password, hashed_password)

def get_password_hash(password):
    """
    Hash password with bcrypt. BCrypt has a 72-byte limit, so we truncate if needed.
    """
    # Truncate password to 72 bytes (bcrypt limit)
    truncated_password = password[:72] if isinstance(password, str) else password.decode('utf-8')[:72]
    return pwd_context.hash(truncated_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = db.users.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user

# backend/routes.py (continued from main.py)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Air Quality Prediction API"
    }

@app.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate):
    """Register a new research user"""
    logger.debug(f"Registration request received: {user.dict()}")
    logger.info(f"User registration attempt: {user.email}")
    
    # Check if user already exists
    if db.users.find_one({"email": user.email}):
        logger.warning(f"Registration failed: Email already registered - {user.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    try:
        # Create user document
        user_dict = user.dict()
        user_dict["hashed_password"] = get_password_hash(user.password)
        user_dict["created_at"] = datetime.utcnow()
        user_dict.pop("password")  # Remove plain password
        
        # Ensure research_interests is a list
        if user_dict.get("research_interests") is None:
            user_dict["research_interests"] = []
        
        # Insert user
        result = db.users.insert_one(user_dict)
        
        logger.info(f"✓ User registered successfully: {user.email} (ID: {result.inserted_id})")
        
        # Build and return user response with only required fields
        return UserResponse(
            id=str(result.inserted_id),
            email=user_dict["email"],
            full_name=user_dict["full_name"],
            affiliation=user_dict.get("affiliation", ""),
            research_interests=user_dict.get("research_interests", []),
            created_at=user_dict["created_at"]
        )
    except Exception as e:
        logger.error(f"Error registering user: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/token")
async def login_for_access_token(username: str = Form(...), password: str = Form(...)):
    """Login and get access token"""
    logger.info(f"Login attempt: {username}")
    
    user = db.users.find_one({"email": username})
    if not user or not verify_password(password, user["hashed_password"]):
        logger.warning(f"Failed login attempt: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    logger.info(f"✓ Token generated for user: {username}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": str(user["_id"]),
        "user_email": user["email"]
    }

@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    current_user["id"] = str(current_user["_id"])
    return UserResponse(**current_user)

@app.get("/predictions")
async def get_user_predictions(
    limit: int = 10,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get user's prediction history from v2 API"""
    user_id = str(current_user["_id"])
    logger.info(f"🔍 GET /predictions - user_id='{user_id}' (type: {type(user_id).__name__}, len: {len(user_id)})")
    
    try:
        # Query predictions for current user
        logger.info(f"📊 Querying db.predictions.find({{'user_id': '{user_id}'}})")
        predictions_list = list(db.predictions.find(
            {"user_id": user_id}
        ).sort("created_at", -1).skip(offset).limit(limit))
        
        logger.info(f"✓ Retrieved {len(predictions_list)} predictions for user {user_id}")
        
        # If no predictions found with user_id, also try to find anonymous predictions
        if len(predictions_list) == 0:
            logger.debug(f"No predictions found for user {user_id}, checking for anonymous predictions...")
            # This is for backward compatibility with old predictions
            anonymous_predictions = list(db.predictions.find(
                {"user_id": "anonymous"}
            ).sort("created_at", -1).limit(5))
            if len(anonymous_predictions) > 0:
                logger.debug(f"Found {len(anonymous_predictions)} anonymous predictions")
        
        # Convert to JSON-serializable format
        result = []
        for pred in predictions_list:
            pred["prediction_id"] = str(pred["_id"])
            pred["id"] = str(pred["_id"])
            # Remove _id to avoid duplication
            del pred["_id"]
            # Convert datetime to ISO format
            if "created_at" in pred and isinstance(pred["created_at"], datetime):
                pred["created_at"] = pred["created_at"].isoformat()
            
            # Fix old predictions that have city as JSON string
            if isinstance(pred.get("city"), str):
                try:
                    import json
                    if pred["city"].startswith("{"):  # It's a JSON string
                        city_data = json.loads(pred["city"])
                        pred["city"] = city_data.get("city", "Unknown")
                except:
                    pass  # Keep original value if parsing fails
            
            result.append(pred)
        
        return result
    except Exception as e:
        logger.error(f"Error fetching predictions for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch predictions")

@app.get("/public/predictions")
async def get_public_predictions(limit: int = 20):
    """Get recent public predictions (anonymous)"""
    try:
        # Query recent predictions from database - get all fields
        predictions_list = list(db.predictions.find().sort("created_at", -1).limit(limit))
        
        logger.debug(f"✓ Retrieved {len(predictions_list)} public predictions")
        
        # Convert to JSON-serializable format
        result = []
        for pred in predictions_list:
            # Extract city from JSON string if necessary (handle old predictions)
            city = pred.get("city", "Unknown")
            if isinstance(city, str) and city.startswith("{"):
                try:
                    import json
                    city_data = json.loads(city)
                    city = city_data.get("city", "Unknown")
                except:
                    pass  # Keep original if parsing fails
            
            # Transform database format to frontend format
            formatted_pred = {
                "prediction_id": str(pred["_id"]),
                "city": city,
                "country": pred.get("country", "Unknown"),
                "date": pred.get("date", ""),
                "pm25_predicted": pred.get("pm25_predicted", pred.get("predicted_pm25", 0)),
                "health_impact": {
                    "risk_level": pred.get("health_risk_level", "Unknown")
                },
                "metadata": {
                    "confidence_score": pred.get("confidence_score", 0),
                    "processing_time_ms": pred.get("processing_time_ms", 0)
                },
                "created_at": pred.get("created_at", datetime.utcnow()).isoformat() if isinstance(pred.get("created_at"), datetime) else pred.get("created_at", "")
            }
            result.append(formatted_pred)
        
        return result
    except Exception as e:
        logger.error(f"Error fetching public predictions: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch predictions")

@app.post("/feedback")
async def submit_feedback(
    feedback: FeedbackCreate,
    current_user: dict = Depends(get_current_user)
):
    """Submit feedback for a prediction"""
    user_id = str(current_user["_id"])
    user_name = current_user.get("full_name", "Anonymous")
    logger.info(f"Feedback submitted by user {user_id} for prediction {feedback.prediction_id} (rating: {feedback.rating})")
    
    try:
        feedback_doc = feedback.dict()
        feedback_doc["user_id"] = user_id
        feedback_doc["user_name"] = user_name
        feedback_doc["created_at"] = datetime.utcnow()
        
        result = db.feedback.insert_one(feedback_doc)
        feedback_doc["id"] = str(result.inserted_id)
        
        logger.debug(f"✓ Feedback stored: {result.inserted_id}")
        
        return feedback_doc
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to submit feedback")

@app.get("/feedback/recent")
async def get_recent_feedback(limit: int = 10):
    """Get recent feedback from all users"""
    try:
        logger.debug(f"Fetching recent feedback (limit: {limit})")
        
        # Map for accuracy labels
        accuracy_labels = {
            1: "Very Poor",
            2: "Poor",
            3: "Average",
            4: "Good",
            5: "Excellent"
        }
        
        # Query recent feedback
        feedback_list = list(db.feedback.find().sort("created_at", -1).limit(limit))
        
        result = []
        for item in feedback_list:
            # Convert datetime to ISO format
            created_at = item.get("created_at", datetime.utcnow())
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            
            result.append({
                "id": str(item["_id"]),
                "user_name": item.get("user_name", "Anonymous"),
                "rating": item.get("rating", 0),
                "comments": item.get("comments", ""),
                "model_accuracy": item.get("model_accuracy", 0),
                "model_accuracy_label": accuracy_labels.get(item.get("model_accuracy", 0), "N/A"),
                "created_at": created_at
            })
        
        logger.debug(f"✓ Retrieved {len(result)} feedback items")
        return result
    except Exception as e:
        logger.error(f"Error fetching feedback: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch feedback")

@app.get("/analytics/models")
async def get_model_analytics():
    """Get model performance analytics from v2 API"""
    logger.debug("Generating model analytics report")
    
    try:
        from routes.predictions_v2 import global_model_loader
        
        # Calculate model performance metrics from feedback
        pipeline = [
            {"$group": {
                "_id": "$model_accuracy",
                "count": {"$sum": 1},
                "avg_rating": {"$avg": "$rating"}
            }}
        ]
        
        feedback_stats = list(db.feedback.aggregate(pipeline))
        
        total_predictions = db.predictions.count_documents({})
        total_users = db.users.count_documents({})
        
        # Get available models from v2 model loader
        available_models = []
        if global_model_loader and global_model_loader.models:
            available_models = list(global_model_loader.models.keys())
        
        logger.info(f"✓ Analytics report generated: {total_predictions} predictions, {total_users} users")
        
        return {
            "total_predictions": total_predictions,
            "total_users": total_users,
            "feedback_stats": feedback_stats,
            "available_models": available_models
        }
    except Exception as e:
        logger.error(f"Error generating analytics: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate analytics")

@app.get("/data/training")
async def get_training_data(limit: int = 100):
    """Get sample training data"""
    try:
        logger.debug(f"Fetching training data (limit: {limit})")
        
        # Query training data from MongoDB
        data = list(db.training_data.find().limit(limit))
        logger.info(f"✓ Retrieved {len(data)} training records")
        
        # Convert MongoDB documents to JSON-serializable format
        result = []
        for item in data:
            # Convert ObjectId to string
            item["id"] = str(item["_id"])
            # Remove the _id field to avoid confusion
            del item["_id"]
            
            # Convert datetime objects to ISO format strings
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()
            
            result.append(item)
        
        return result
    except Exception as e:
        logger.error(f"Error fetching training data: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch training data")

@app.get("/cities/geodata")
async def get_cities_geodata():
    """Get geographical data for cities"""
    logger.debug("Fetching city geographical data")
    
    try:
        # This would return city coordinates and pollution data
        pipeline = [
            {"$group": {
                "_id": {"city": "$input_data.city", "country": "$input_data.country"},
                "avg_pm25": {"$avg": "$model_predictions.ensemble"},
                "prediction_count": {"$sum": 1},
                "latest_prediction": {"$last": "$created_at"}
            }},
            {"$limit": 100}
        ]
        
        city_data = list(db.predictions.aggregate(pipeline))
        logger.debug(f"Retrieved data for {len(city_data)} cities")
        
        # Convert to JSON-serializable format
        result = []
        for city in city_data:
            # Extract city and country info
            city_info = city.get("_id", {})
            city_name = city_info.get("city", "Unknown")
            country_name = city_info.get("country", "Unknown")
            
            # Add coordinates
            coordinates = get_city_coordinates(city_name, country_name)
            
            # Build result object
            result_item = {
                "city": city_name,
                "country": country_name,
                "avg_pm25": city.get("avg_pm25", 0),
                "prediction_count": city.get("prediction_count", 0),
                "latest_prediction": city.get("latest_prediction", "").isoformat() if city.get("latest_prediction") else None,
                "coordinates": coordinates
            }
            result.append(result_item)
        
        logger.info(f"✓ City geodata retrieved for {len(result)} locations")
        return result
    except Exception as e:
        logger.error(f"Error fetching city geodata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch city geodata")

def get_city_coordinates(city: str, country: str) -> Dict[str, float]:
    """Mock function to get city coordinates"""
    # In production, use a geocoding service
    coordinates_db = {
        "London": {"lat": 51.5074, "lng": -0.1278},
        "New York": {"lat": 40.7128, "lng": -74.0060},
        "Tokyo": {"lat": 35.6762, "lng": 139.6503},
        "Delhi": {"lat": 28.7041, "lng": 77.1025},
        "Beijing": {"lat": 39.9042, "lng": 116.4074},
    }
    return coordinates_db.get(city, {"lat": 0, "lng": 0})

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 80)
    logger.info("🚀 Starting Air Quality Prediction API v2.0")
    logger.info("=" * 80)
    logger.info(f"📝 Logs directory: {os.path.abspath('logs')}")
    logger.info("📊 Models: Loaded via v2 Predictions Router (/api/v2/predictions)")
    logger.info("🔧 Features: 173 engineered features (temporal, lag, rolling, interactions)")
    logger.info("=" * 80)
    logger.info("✅ Server is ready - listening on http://0.0.0.0:8000")
    logger.info("📍 API v2 Endpoints: http://0.0.0.0:8000/api/v2/predictions/*")
    logger.info("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8000)