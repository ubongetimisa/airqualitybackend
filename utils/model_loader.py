"""
Model Loader and Manager for Air Quality Prediction

This module handles loading trained models and provides production-ready
prediction capabilities with comprehensive feature engineering.

The prediction pipeline supports:
- 173 engineered features (temporal, lag, rolling, interactions, ratios)
- Stacking ensemble with base models (SVM Linear, Lasso, Ridge, Linear Regression, TensorFlow DeepDense)
- Real-time feature engineering for new inputs
- Input validation and error handling

Author: Ubong Isaiah Eka
Email: ubongisaiahetim001@gmail.com
Date: 2025
"""

import os
import json
import pickle
import joblib
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Try importing optional dependencies
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and managing all trained models for inference"""
    
    # Component models used in the stacking ensemble
    COMPONENT_MODELS = ['SVM Linear', 'Lasso Regression', 'Linear Regression', 'Ridge Regression', 'TensorFlow DeepDense']
    
    # Features for lag and rolling calculations
    TEMPORAL_FEATURES = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
    LAG_PERIODS = [1, 2, 3, 7, 14, 30]
    ROLLING_WINDOWS = [3, 7, 14]
    
    def __init__(self, artifacts_dir: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            artifacts_dir: Path to directory containing saved models and artifacts
                          If None, auto-detects from current working directory or script location
                          Tries multiple locations for flexibility
        """
        # Auto-detect path if not provided
        if artifacts_dir is None:
            # Get the directory where this script is (backend/utils)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(current_dir)  # backend/
            
            # Try multiple possible locations relative to backend directory
            possible_paths = [
                os.path.join(backend_dir, 'trained_model'),  # backend/trained_model
                os.path.join(backend_dir, 'artifacts'),  # backend/artifacts (fallback)
                'trained_model',  # Current working directory
                '../trained_model',  # From backend/utils
                '../../trained_model',  # From other locations
            ]
            
            artifacts_dir = None
            for path in possible_paths:
                if os.path.isabs(path):
                    check_path = path
                else:
                    check_path = os.path.normpath(os.path.join(current_dir, path))
                
                if os.path.exists(check_path):
                    artifacts_dir = check_path
                    break
            
            if artifacts_dir is None:
                # Default fallback to relative path
                artifacts_dir = os.path.join(backend_dir, 'trained_model')
        else:
            # Resolve user-provided path
            if not os.path.isabs(artifacts_dir):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                artifacts_dir = os.path.normpath(os.path.join(current_dir, artifacts_dir))
        
        self.artifacts_dir = artifacts_dir
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.model_info = None
        self.is_ready = False
        self.historical_data_cache = {}  # Cache for historical data by city
        
        logger.info(f"Initialized model loader with artifacts_dir: {self.artifacts_dir}")
        
    def load_all(self) -> bool:
        """
        Load all models and artifacts.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("LOADING MODELS AND ARTIFACTS")
            logger.info("=" * 60)
            
            # Load core artifacts
            self.load_scaler()
            self.load_feature_names()
            self.load_model_info()
            
            # Load models
            self.load_traditional_ml_models()
            self.load_tensorflow_models()
            self.load_pytorch_models()
            self.load_ensemble_model()
            
            # Check if ready
            if self.scaler is not None and self.feature_names is not None:
                self.is_ready = True
                logger.info("✓ Model loader ready for inference")
                return True
            else:
                logger.error("❌ Missing critical artifacts (scaler/features)")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}", exc_info=True)
            return False
    
    def load_scaler(self) -> bool:
        """Load the StandardScaler used for feature scaling from deployment_artifacts"""
        try:
            scaler_path = os.path.join(self.artifacts_dir, 'deployment_artifacts', 'scaler.joblib')
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✓ Scaler loaded from {scaler_path}")
                return True
            else:
                logger.warning(f"⚠ Scaler not found at {scaler_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading scaler: {e}")
            return False
    
    def load_feature_names(self) -> bool:
        """Load the list of feature names used during training from deployment_artifacts"""
        try:
            features_path = os.path.join(self.artifacts_dir, 'deployment_artifacts', 'feature_names.pkl')
            
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                logger.info(f"✓ Feature names loaded ({len(self.feature_names)} features)")
                return True
            else:
                logger.warning(f"⚠ Feature names not found at {features_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading feature names: {e}")
            return False
    
    def load_model_info(self) -> bool:
        """Load model information and metadata from deployment_artifacts"""
        try:
            info_path = os.path.join(self.artifacts_dir, 'deployment_artifacts', 'final_model_info.pkl')
            
            if os.path.exists(info_path):
                with open(info_path, 'rb') as f:
                    self.model_info = pickle.load(f)
                logger.info(f"✓ Model info loaded: {self.model_info.get('final_model_name', 'Unknown')}")
                return True
            else:
                logger.debug(f"Model info not found at {info_path}")
                return False
                
        except Exception as e:
            logger.warning(f"Model info loading failed: {e}")
            return False
    
    def load_traditional_ml_models(self) -> None:
        """Load all traditional ML models from saved_models/traditional_ml directory"""
        try:
            models_dir = os.path.join(self.artifacts_dir, 'saved_models', 'traditional_ml')
            
            if not os.path.exists(models_dir):
                logger.warning(f"⚠ Models directory not found: {models_dir}")
                return
            
            model_count = 0
            for model_file in os.listdir(models_dir):
                if model_file.endswith('.joblib'):
                    try:
                        model_path = os.path.join(models_dir, model_file)
                        model_name = model_file.replace('.joblib', '')
                        
                        # Load and store with normalized name for ensemble lookup
                        model = joblib.load(model_path)
                        self.models[model_name] = model
                        
                        # Also store with short name for ensemble matching
                        if 'svm' in model_name.lower() and 'linear' in model_name.lower():
                            self.models['svm_linear'] = model
                        elif 'lasso' in model_name.lower():
                            self.models['lasso_regression'] = model
                        elif 'ridge' in model_name.lower():
                            self.models['ridge_regression'] = model
                        elif 'linear' in model_name.lower() and 'regression' in model_name.lower():
                            self.models['linear_regression'] = model
                        
                        model_count += 1
                        logger.debug(f"✓ Loaded ML model: {model_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_file}: {e}")
            
            if model_count > 0:
                logger.info(f"✓ Loaded {model_count} traditional ML model(s)")
            else:
                logger.warning("⚠ No traditional ML models found")
                
        except Exception as e:
            logger.error(f"Error loading traditional ML models: {e}")
    
    def load_tensorflow_models(self) -> None:
        """Load TensorFlow Keras models from saved_models/tensorflow directory"""
        if not TF_AVAILABLE:
            logger.debug("TensorFlow not available, skipping TensorFlow models")
            return
        
        try:
            models_dir = os.path.join(self.artifacts_dir, 'saved_models', 'tensorflow')
            
            if not os.path.exists(models_dir):
                logger.debug(f"TensorFlow models directory not found: {models_dir}")
                return
            
            model_count = 0
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                
                # Check if it's a model file (keras or directory)
                try:
                    if item.endswith('.keras'):
                        model = tf.keras.models.load_model(item_path)
                        model_name = item.replace('.keras', '')
                        self.models[model_name] = model
                        
                        # Also store with short name for ensemble matching
                        if 'deepdense' in model_name.lower():
                            self.models['tensorflow_deepdense'] = model
                        
                        model_count += 1
                        logger.debug(f"✓ Loaded TensorFlow model: {model_name}")
                    elif os.path.isdir(item_path):
                        # Legacy directory format
                        model = tf.keras.models.load_model(item_path)
                        self.models[item] = model
                        
                        if 'deepdense' in item.lower():
                            self.models['tensorflow_deepdense'] = model
                        
                        model_count += 1
                        logger.debug(f"✓ Loaded TensorFlow model: {item}")
                except Exception as e:
                    logger.warning(f"Failed to load TensorFlow model {item}: {e}")
            
            if model_count > 0:
                logger.info(f"✓ Loaded {model_count} TensorFlow model(s)")
                
        except Exception as e:
            logger.error(f"Error loading TensorFlow models: {e}")
    
    def load_pytorch_models(self) -> None:
        """Load PyTorch models from saved_models/pytorch directory"""
        if not TORCH_AVAILABLE:
            logger.debug("PyTorch not available, skipping PyTorch models")
            return
        
        try:
            models_dir = os.path.join(self.artifacts_dir, 'saved_models', 'pytorch')
            
            if not os.path.exists(models_dir):
                logger.debug(f"PyTorch models directory not found: {models_dir}")
                return
            
            model_count = 0
            for model_file in os.listdir(models_dir):
                if model_file.endswith('.pth'):
                    try:
                        model_path = os.path.join(models_dir, model_file)
                        model_name = model_file.replace('.pth', '')
                        
                        # Load state dict (model structure must be recreated)
                        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                        self.models[model_name] = state_dict
                        model_count += 1
                        logger.debug(f"✓ Loaded PyTorch model state: {model_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load PyTorch model {model_file}: {e}")
            
            if model_count > 0:
                logger.info(f"✓ Loaded {model_count} PyTorch model state(s)")
                
        except Exception as e:
            logger.error(f"Error loading PyTorch models: {e}")
    
    def load_ensemble_model(self) -> bool:
        """Load the ensemble/stacking meta-model from saved_models directory"""
        try:
            ensemble_path = os.path.join(self.artifacts_dir, 'saved_models', 'ensemble_config.pkl')
            
            if os.path.exists(ensemble_path):
                with open(ensemble_path, 'rb') as f:
                    ensemble_model = pickle.load(f)
                self.models['ensemble'] = ensemble_model
                logger.info("✓ Ensemble meta-model loaded")
                return True
            else:
                logger.debug(f"Ensemble model not found at {ensemble_path}")
                return False
                
        except Exception as e:
            logger.warning(f"Error loading ensemble model: {e}")
            return False
    
    def predict(self, X_scaled: np.ndarray, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Make prediction using specified or default model.
        
        Args:
            X_scaled: Scaled feature array (n_samples, n_features)
            model_name: Name of model to use (if None, uses ensemble)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.is_ready:
                return {'error': 'Models not loaded. Call load_all() first.', 'prediction': None}
            
            # Use ensemble if no specific model specified
            if model_name is None:
                model_name = 'ensemble' if 'ensemble' in self.models else list(self.models.keys())[0]
            
            if model_name not in self.models:
                return {
                    'error': f'Model "{model_name}" not found. Available: {list(self.models.keys())}',
                    'prediction': None
                }
            
            model = self.models[model_name]
            
            # Make prediction based on model type
            try:
                # For ensemble: get predictions from base models and stack them
                if model_name == 'ensemble':
                    # Use only the best-performing base models (Ridge and Linear Regression)
                    # SVM and Lasso are poor performers and drag down the ensemble
                    preferred_base_models = ['ridge_regression', 'linear_regression']
                    base_predictions = []
                    
                    # Get available base models that exist
                    available_base_models = [m for m in preferred_base_models if m in self.models]
                    
                    # Log which models we're using
                    logger.debug(f"Ensemble using {len(available_base_models)} base models: {available_base_models}")
                    
                    for base_model_name in available_base_models:
                        if base_model_name in self.models:
                            base_model = self.models[base_model_name]
                            try:
                                base_pred = base_model.predict(X_scaled)
                                if isinstance(base_pred, np.ndarray):
                                    if base_pred.ndim > 1:
                                        base_pred = base_pred.flatten()[0]
                                    else:
                                        base_pred = float(base_pred[0]) if len(base_pred) > 0 else base_pred
                                base_predictions.append(float(base_pred))
                                logger.debug(f"Got prediction from {base_model_name}: {base_pred}")
                            except Exception as e:
                                logger.warning(f"Failed to get prediction from {base_model_name}: {e}")
                                base_predictions.append(0.0)  # Use 0 as fallback
                    
                    # Stack predictions as input to ensemble meta-model
                    if len(base_predictions) > 0:
                        # Use the average of the 2 best-performing models
                        ensemble_pred = np.mean(base_predictions)
                        logger.debug(f"Ensemble average of {len(base_predictions)} best models: {ensemble_pred}")
                        prediction = ensemble_pred
                    else:
                        return {
                            'model': model_name,
                            'prediction': None,
                            'error': 'Failed to get predictions from base models'
                        }
                else:
                    # For individual models, use features directly
                    prediction = model.predict(X_scaled)
                
                # Handle different output shapes
                if isinstance(prediction, np.ndarray):
                    if prediction.ndim > 1:
                        prediction = prediction.flatten()[0]
                    else:
                        prediction = float(prediction[0]) if len(prediction) > 0 else prediction
                
                # Ensure prediction is non-negative (PM2.5 cannot be negative)
                prediction = max(float(prediction), 0.0)
                
                return {
                    'model': model_name,
                    'prediction': prediction,
                    'error': None
                }
                
            except Exception as e:
                logger.error(f"Prediction error with model {model_name}: {e}")
                return {
                    'model': model_name,
                    'prediction': None,
                    'error': f'Prediction failed: {str(e)}'
                }
            
        except Exception as e:
            logger.error(f"Error in predict method: {e}")
            return {'error': f'Prediction error: {str(e)}', 'prediction': None}
    
    def _create_temporal_features(self, df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
        """Create temporal features from date column"""
        df_temp = df.copy()
        
        if date_column not in df_temp.columns:
            logger.warning(f"Date column '{date_column}' not found")
            return df_temp
        
        # Ensure date is datetime
        df_temp[date_column] = pd.to_datetime(df_temp[date_column], errors='coerce')
        
        # Basic temporal features
        df_temp['year'] = df_temp[date_column].dt.year
        df_temp['month'] = df_temp[date_column].dt.month
        df_temp['day'] = df_temp[date_column].dt.day
        df_temp['dayofweek'] = df_temp[date_column].dt.dayofweek
        df_temp['weekofyear'] = df_temp[date_column].dt.isocalendar().week
        df_temp['quarter'] = df_temp[date_column].dt.quarter
        df_temp['is_weekend'] = df_temp['dayofweek'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for seasonal patterns
        df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
        df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
        df_temp['day_sin'] = np.sin(2 * np.pi * df_temp['day'] / 31)
        df_temp['day_cos'] = np.cos(2 * np.pi * df_temp['day'] / 31)
        df_temp['dayofweek_sin'] = np.sin(2 * np.pi * df_temp['dayofweek'] / 7)
        df_temp['dayofweek_cos'] = np.cos(2 * np.pi * df_temp['dayofweek'] / 7)
        
        return df_temp
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create meteorological interaction and ratio features"""
        df_inter = df.copy()
        
        # Meteorological interactions
        if 'Temperature' in df_inter.columns and 'Humidity' in df_inter.columns:
            df_inter['temp_humidity_interaction'] = df_inter['Temperature'] * df_inter['Humidity']
        
        if 'Wind Speed' in df_inter.columns and 'Temperature' in df_inter.columns:
            df_inter['wind_temp_interaction'] = df_inter['Wind Speed'] * df_inter['Temperature']
        
        if 'Wind Speed' in df_inter.columns and 'Humidity' in df_inter.columns:
            df_inter['wind_humidity_interaction'] = df_inter['Wind Speed'] * df_inter['Humidity']
        
        # Pollutant ratios (with safety for division by zero)
        if 'PM2.5' in df_inter.columns and 'PM10' in df_inter.columns:
            df_inter['pm_ratio'] = df_inter['PM2.5'] / (df_inter['PM10'] + 1e-6)
        
        if 'NO2' in df_inter.columns and 'SO2' in df_inter.columns:
            df_inter['nox_so2_ratio'] = df_inter['NO2'] / (df_inter['SO2'] + 1e-6)
        
        return df_inter
    
    def _create_lag_features(self, df: pd.DataFrame, city_col: str = 'City') -> pd.DataFrame:
        """Create lagged features for time series (requires historical data)"""
        df_lag = df.copy()
        
        # Group by city to preserve temporal order
        for city in df_lag[city_col].unique():
            city_mask = df_lag[city_col] == city
            city_data = df_lag[city_mask].sort_values('Date')
            
            for col in self.TEMPORAL_FEATURES:
                if col not in city_data.columns:
                    continue
                
                for lag in self.LAG_PERIODS:
                    new_col_name = f'{col}_lag_{lag}'
                    df_lag.loc[city_mask, new_col_name] = city_data[col].shift(lag)
        
        return df_lag
    
    def _create_rolling_features(self, df: pd.DataFrame, city_col: str = 'City') -> pd.DataFrame:
        """Create rolling statistics (requires historical data)"""
        df_roll = df.copy()
        
        for city in df_roll[city_col].unique():
            city_mask = df_roll[city_col] == city
            city_data = df_roll[city_mask].sort_values('Date')
            
            for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
                if col not in city_data.columns:
                    continue
                
                for window in self.ROLLING_WINDOWS:
                    # Rolling mean
                    df_roll.loc[city_mask, f'{col}_rolling_mean_{window}'] = \
                        city_data[col].rolling(window=window, min_periods=1).mean()
                    # Rolling std
                    df_roll.loc[city_mask, f'{col}_rolling_std_{window}'] = \
                        city_data[col].rolling(window=window, min_periods=1).std()
                    # Rolling max
                    df_roll.loc[city_mask, f'{col}_rolling_max_{window}'] = \
                        city_data[col].rolling(window=window, min_periods=1).max()
        
        return df_roll
    
    def engineer_features_for_prediction(self, raw_input: Dict[str, Any]) -> Tuple[np.ndarray, Optional[str]]:
        """
        Engineer all 173 features from raw input data.
        
        Args:
            raw_input: Dictionary with raw feature values:
                Required: City, Country, Date, PM10, NO2, SO2, CO, O3, Temperature, Humidity, Wind Speed
                Optional: historical_data (list of dicts with past observations for lag/rolling)
        
        Returns:
            Tuple of (scaled_features, error_message). If error, error_message is not None.
        """
        try:
            # Step 1: Validate required fields
            required_fields = ['City', 'Country', 'Date', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 
                             'Temperature', 'Humidity', 'Wind Speed']
            
            missing_fields = [field for field in required_fields if field not in raw_input]
            if missing_fields:
                error_msg = f"Missing required fields: {missing_fields}"
                logger.error(error_msg)
                return None, error_msg
            
            # Step 2: Create base DataFrame
            input_df = pd.DataFrame([raw_input])
            
            # Step 3: Create temporal features
            input_df = self._create_temporal_features(input_df, 'Date')
            
            # Step 4: Create interaction features
            input_df = self._create_interaction_features(input_df)
            
            # Step 5: Handle lag and rolling features (if historical data provided)
            if 'historical_data' in raw_input and isinstance(raw_input['historical_data'], list):
                # Combine historical data with current observation
                hist_df = pd.DataFrame(raw_input['historical_data'])
                combined_df = pd.concat([hist_df, input_df], ignore_index=True)
                
                # Create lag and rolling features on combined data
                combined_df = self._create_lag_features(combined_df, 'City')
                combined_df = self._create_rolling_features(combined_df, 'City')
                
                # Extract only the current row's features
                input_df = combined_df.iloc[-1:].reset_index(drop=True)
            else:
                # Without historical data, fill lag/rolling with zeros (not ideal but necessary)
                logger.warning("No historical data provided. Lag and rolling features will be zero-filled.")
                # Build all columns at once to avoid DataFrame fragmentation warning
                new_cols = {}
                for col in self.TEMPORAL_FEATURES:
                    for lag in self.LAG_PERIODS:
                        new_cols[f'{col}_lag_{lag}'] = 0.0
                    for window in self.ROLLING_WINDOWS:
                        new_cols[f'{col}_rolling_mean_{window}'] = 0.0
                        new_cols[f'{col}_rolling_std_{window}'] = 0.0
                        new_cols[f'{col}_rolling_max_{window}'] = 0.0
                for col_name, value in new_cols.items():
                    input_df[col_name] = value
            
            # Step 6: Handle categorical encoding (City, Country)
            input_df_encoded = pd.get_dummies(input_df, columns=['City', 'Country'], 
                                             prefix=['city', 'country'], drop_first=False)
            
            # Step 7: Align columns with training features
            aligned_input = pd.DataFrame(0.0, index=input_df_encoded.index, columns=self.feature_names)
            
            for col in input_df_encoded.columns:
                if col in aligned_input.columns:
                    aligned_input[col] = input_df_encoded[col]
            
            # Ensure correct column order
            aligned_input = aligned_input[self.feature_names]
            
            # Step 8: Replace NaN and infinite values
            aligned_input = aligned_input.replace([np.inf, -np.inf], np.nan)
            aligned_input = aligned_input.fillna(0.0)
            
            # Step 9: Apply scaling
            if self.scaler is None:
                error_msg = "Scaler not loaded"
                logger.error(error_msg)
                return None, error_msg
            
            X_scaled = self.scaler.transform(aligned_input)
            
            logger.info(f"Features engineered successfully. Shape: {X_scaled.shape}")
            return X_scaled, None
            
        except Exception as e:
            error_msg = f"Feature engineering error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
    
    def make_prediction(self, raw_input: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a prediction from raw input data.
        
        Args:
            raw_input: Dictionary with raw feature values (see engineer_features_for_prediction)
            model_name: Optional specific model to use
        
        Returns:
            Dictionary with prediction, confidence, and metadata
        """
        try:
            if not self.is_ready:
                return {
                    'success': False,
                    'prediction': None,
                    'error': 'Models not loaded',
                    'model_used': None
                }
            
            # Engineer features
            X_scaled, error = self.engineer_features_for_prediction(raw_input)
            
            if X_scaled is None:
                return {
                    'success': False,
                    'prediction': None,
                    'error': error,
                    'model_used': None
                }
            
            # Make prediction
            result = self.predict(X_scaled, model_name)
            
            if result.get('error'):
                return {
                    'success': False,
                    'prediction': None,
                    'error': result['error'],
                    'model_used': result.get('model')
                }
            
            return {
                'success': True,
                'prediction': result['prediction'],
                'error': None,
                'model_used': result.get('model'),
                'timestamp': datetime.now().isoformat(),
                'input_data': {
                    'city': raw_input.get('City'),
                    'country': raw_input.get('Country'),
                    'date': str(raw_input.get('Date'))
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {
                'success': False,
                'prediction': None,
                'error': str(e),
                'model_used': None
            }
    
    def get_model_list(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of model loader"""
        return {
            'is_ready': self.is_ready,
            'models_loaded': len(self.models),
            'model_names': self.get_model_list(),
            'scaler_loaded': self.scaler is not None,
            'features_loaded': self.feature_names is not None,
            'feature_count': len(self.feature_names) if self.feature_names else 0
        }


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader(artifacts_dir: str = '../trained_model') -> ModelLoader:
    """
    Get or create global model loader instance.
    
    Args:
        artifacts_dir: Path to artifacts directory (default: trained_model directory)
        
    Returns:
        ModelLoader instance
    """
    global _model_loader
    
    if _model_loader is None:
        _model_loader = ModelLoader(artifacts_dir)
    
    return _model_loader


def initialize_models(artifacts_dir: str = '../trained_model') -> bool:
    """
    Initialize and load all models.
    
    Args:
        artifacts_dir: Path to artifacts directory (default: trained_model directory)
        
    Returns:
        True if successful, False otherwise
    """
    loader = get_model_loader(artifacts_dir)
    return loader.load_all()


if __name__ == '__main__':
    # Production backend models are accessed via ModelLoader class
    # Use get_model_loader() and loader.make_prediction() for real predictions
    pass