#!/usr/bin/env python
"""
Setup Verification Script for Air Quality Prediction Backend

This script verifies that all models, artifacts, and dependencies are correctly set up.

Usage:
    python verify_setup.py
    
Author: Ubong Isaiah Eka
Email: ubongisaiahetim001@gmail.com
Date: 2025
"""

import os
import sys
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_directory_structure():
    """Check if required directories exist"""
    logger.info("=" * 60)
    logger.info("CHECKING DIRECTORY STRUCTURE")
    logger.info("=" * 60)
    
    required_dirs = [
        'artifacts',
        'artifacts/saved_models',
        'logs',
        'logs/backend',
        'utils',
        'routes'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            logger.info(f"✓ {dir_path}/")
        else:
            logger.error(f"✗ {dir_path}/ NOT FOUND")
            all_exist = False
    
    return all_exist


def check_artifact_files():
    """Check if required artifact files exist"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING ARTIFACT FILES")
    logger.info("=" * 60)
    
    required_files = [
        'artifacts/scaler.joblib',
        'artifacts/feature_names.pkl',
        'artifacts/final_model_info.pkl',
        'artifacts/model_card.json'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            logger.info(f"✓ {file_path} ({size:,} bytes)")
        else:
            logger.error(f"✗ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist


def check_models():
    """Check if trained models exist"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING TRAINED MODELS")
    logger.info("=" * 60)
    
    models_dir = 'artifacts/saved_models'
    
    if not os.path.exists(models_dir):
        logger.error(f"✗ {models_dir} NOT FOUND")
        return False
    
    models = os.listdir(models_dir)
    
    if not models:
        logger.error(f"✗ No models found in {models_dir}")
        return False
    
    logger.info(f"Found {len(models)} model(s):")
    for model in sorted(models):
        model_path = os.path.join(models_dir, model)
        if os.path.isdir(model_path):
            logger.info(f"  📁 {model}/ (TensorFlow model)")
        else:
            size = os.path.getsize(model_path)
            logger.info(f"  📄 {model} ({size:,} bytes)")
    
    return True


def check_dependencies():
    """Check if required Python packages are installed"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING DEPENDENCIES")
    logger.info("=" * 60)
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'pymongo',
        'pydantic',
        'xgboost',
        'lightgbm'
    ]
    
    optional_packages = [
        'tensorflow',
        'torch'
    ]
    
    all_ok = True
    
    logger.info("\nRequired packages:")
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package} NOT INSTALLED")
            all_ok = False
    
    logger.info("\nOptional packages:")
    for package in optional_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.warning(f"⚠ {package} NOT INSTALLED (optional)")
    
    return all_ok


def check_python_modules():
    """Check if custom Python modules can be imported"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING CUSTOM MODULES")
    logger.info("=" * 60)
    
    sys.path.insert(0, os.getcwd())
    
    modules = [
        ('utils.model_loader', 'ModelLoader'),
        ('utils.health_assessment', 'Health Assessment'),
        ('routes.predictions_v2', 'Prediction Routes v2')
    ]
    
    all_ok = True
    for module_path, module_name in modules:
        try:
            module = __import__(module_path, fromlist=[''])
            logger.info(f"✓ {module_path} ({module_name})")
        except ImportError as e:
            logger.error(f"✗ {module_path} - {e}")
            all_ok = False
    
    return all_ok


def load_and_test_models():
    """Attempt to load models and test basic functionality"""
    logger.info("\n" + "=" * 60)
    logger.info("LOADING AND TESTING MODELS")
    logger.info("=" * 60)
    
    try:
        from utils.model_loader import ModelLoader
        
        loader = ModelLoader('artifacts')
        logger.info("Loading models...")
        
        if loader.load_all():
            logger.info("✓ All models loaded successfully")
            
            # Print status
            status = loader.get_status()
            logger.info(f"  - Models loaded: {status['models_loaded']}")
            logger.info(f"  - Features: {status['feature_count']}")
            logger.info(f"  - Ready: {status['is_ready']}")
            
            # List models
            if status['model_names']:
                logger.info(f"  - Available models: {', '.join(status['model_names'][:5])}")
                if len(status['model_names']) > 5:
                    logger.info(f"    ... and {len(status['model_names']) - 5} more")
            
            return True
        else:
            logger.error("✗ Failed to load models")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_env_file():
    """Check if .env file exists with required variables"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING ENVIRONMENT CONFIGURATION")
    logger.info("=" * 60)
    
    if os.path.exists('.env'):
        logger.info("✓ .env file found")
        
        # Check for required variables
        with open('.env', 'r') as f:
            env_content = f.read()
        
        required_vars = ['MONGODB_URI', 'SECRET_KEY']
        
        for var in required_vars:
            if var in env_content:
                logger.info(f"✓ {var} configured")
            else:
                logger.warning(f"⚠ {var} NOT configured")
        
        return True
    else:
        logger.warning("⚠ .env file NOT found")
        logger.info("  Please create .env with MONGODB_URI and SECRET_KEY")
        return False


def main():
    """Run all checks"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 10 + "AIR QUALITY BACKEND SETUP VERIFICATION" + " " * 10 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Artifact Files", check_artifact_files),
        ("Trained Models", check_models),
        ("Dependencies", check_dependencies),
        ("Custom Modules", check_python_modules),
        ("Environment Config", check_env_file),
        ("Model Loading", load_and_test_models),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"✗ {check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {check_name}")
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} checks passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("\n✓ ✓ ✓ SETUP VERIFICATION SUCCESSFUL! ✓ ✓ ✓")
        logger.info("\nYou can now start the backend with:")
        logger.info("  python main.py")
        return 0
    else:
        logger.error(f"\n✗ ✗ ✗ {total - passed} CHECK(S) FAILED ✗ ✗ ✗")
        logger.error("\nPlease fix the issues above before starting the backend.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
