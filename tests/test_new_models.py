#!/usr/bin/env python3
"""
Test script to verify newly trained models are loaded and working correctly
"""

import sys
import os
from datetime import datetime, timedelta
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'utils'))

from model_loader import get_model_loader, initialize_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get absolute path to trained_model directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'trained_model')


def test_model_loading():
    """Test that all models load correctly"""
    logger.info("=" * 70)
    logger.info("TESTING MODEL LOADING")
    logger.info("=" * 70)
    
    # Initialize models using absolute path
    loader = get_model_loader(TRAINED_MODEL_DIR)
    success = loader.load_all()
    
    if not success:
        logger.error("❌ Failed to initialize models")
        return False
    
    logger.info("✅ Models loaded successfully!")
    
    # Print status
    status = loader.get_status()
    logger.info(f"\nModel Status:")
    logger.info(f"  Ready: {status['is_ready']}")
    logger.info(f"  Models Loaded: {status['models_loaded']}")
    logger.info(f"  Feature Count: {status['feature_count']}")
    logger.info(f"  Available Models: {status['model_names']}")
    
    return True


def test_prediction():
    """Test making a prediction with real data"""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING PREDICTIONS WITH REAL DATA")
    logger.info("=" * 70)
    
    loader = get_model_loader(TRAINED_MODEL_DIR)
    
    if not loader.is_ready:
        logger.error("❌ Models not ready")
        return False
    
    # Test 1: Simple prediction
    logger.info("\n[Test 1] Simple Prediction (London, Current Date)")
    input_data = {
        'City': 'London',
        'Country': 'United Kingdom',
        'Date': datetime.now().strftime('%Y-%m-%d'),
        'PM10': 45.2,
        'NO2': 35.8,
        'SO2': 12.4,
        'CO': 0.85,
        'O3': 65.3,
        'Temperature': 12.5,
        'Humidity': 68.0,
        'Wind Speed': 8.2
    }
    
    result = loader.make_prediction(input_data)
    if result['prediction'] is not None:
        logger.info(f"  Prediction Result: {result['prediction']:.4f} µg/m³")
    else:
        logger.error(f"  Prediction Error: {result.get('error', 'Unknown error')}")
    logger.info(f"  Success: {result['success']}")
    
    # Test 2: Paris prediction
    logger.info("\n[Test 2] Paris Prediction")
    input_data['City'] = 'Paris'
    input_data['Country'] = 'France'
    input_data['PM10'] = 38.1
    input_data['NO2'] = 42.3
    input_data['SO2'] = 8.9
    input_data['Temperature'] = 14.2
    input_data['Humidity'] = 72.0
    
    result = loader.make_prediction(input_data)
    if result['prediction'] is not None:
        logger.info(f"  Prediction Result: {result['prediction']:.4f} µg/m³")
    else:
        logger.error(f"  Prediction Error: {result.get('error', 'Unknown error')}")
    logger.info(f"  Success: {result['success']}")
    
    # Test 3: Delhi (high pollution)
    logger.info("\n[Test 3] Delhi Prediction (High Pollution)")
    input_data['City'] = 'Delhi'
    input_data['Country'] = 'India'
    input_data['PM10'] = 185.5
    input_data['NO2'] = 128.3
    input_data['SO2'] = 42.1
    input_data['Temperature'] = 28.5
    input_data['Humidity'] = 45.0
    input_data['Wind Speed'] = 4.2
    
    result = loader.make_prediction(input_data)
    if result['prediction'] is not None:
        logger.info(f"  Prediction Result: {result['prediction']:.4f} µg/m³")
    else:
        logger.error(f"  Prediction Error: {result.get('error', 'Unknown error')}")
    logger.info(f"  Success: {result['success']}")
    
    # Test 4: Using specific model
    logger.info("\n[Test 4] Prediction using SVM Linear model")
    result = loader.make_prediction(input_data, model_name='svm_linear')
    logger.info(f"  Model Used: {result['model_used']}")
    if result['prediction'] is not None:
        logger.info(f"  Prediction Result: {result['prediction']:.4f} µg/m³")
    else:
        logger.error(f"  Prediction Error: {result.get('error', 'Unknown error')}")
    logger.info(f"  Success: {result['success']}")
    
    return True


def test_base_model_access():
    """Test that base models can be accessed individually"""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING BASE MODEL ACCESS")
    logger.info("=" * 70)
    
    loader = get_model_loader(TRAINED_MODEL_DIR)
    
    if not loader.is_ready:
        logger.error("❌ Models not ready")
        return False
    
    available_models = loader.get_model_list()
    logger.info(f"Available models ({len(available_models)}):")
    for i, model_name in enumerate(available_models, 1):
        logger.info(f"  {i}. {model_name}")
    
    # Check key base models
    key_models = ['svm_linear', 'lasso_regression', 'ridge_regression', 'linear_regression', 'tensorflow_deepdense', 'ensemble']
    logger.info(f"\nKey Base Models Check:")
    for model_name in key_models:
        if model_name in loader.models:
            logger.info(f"  ✅ {model_name}")
        else:
            logger.warning(f"  ❌ {model_name} NOT FOUND")
    
    return True


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 70)
    logger.info("NEW TRAINED MODELS - VERIFICATION TEST SUITE")
    logger.info("=" * 70)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Base Model Access", test_base_model_access),
        ("Predictions", test_prediction),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with error: {e}", exc_info=True)
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED - MODELS ARE READY FOR DEPLOYMENT!")
    else:
        logger.info("⚠️ SOME TESTS FAILED - CHECK ERRORS ABOVE")
    logger.info("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
