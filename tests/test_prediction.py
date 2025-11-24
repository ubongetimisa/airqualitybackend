#!/usr/bin/env python3
"""
Test script to make a real prediction and save it to the database
"""
import requests
import json

API_URL = "http://localhost:8000"

# Test prediction data
prediction_data = {
    "location": {
        "city": "London",
        "country": "UK",
        "date": "2025-11-22"
    },
    "air_quality": {
        "pm10": 45.5,
        "no2": 25.3,
        "so2": 12.1,
        "co": 0.8,
        "o3": 35.2,
        "temperature": 15.5,
        "humidity": 65,
        "wind_speed": 3.2
    },
    "model": "ensemble",
    "user_id": "test_user_001"
}

print("📊 Making test prediction...")
print(f"📍 Location: {prediction_data['location']['city']}, {prediction_data['location']['country']}")

try:
    response = requests.post(
        f"{API_URL}/api/v2/predictions/predict",
        json=prediction_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Prediction successful!")
        print(f"  PM2.5 Predicted: {result.get('pm25_predicted', 'N/A')} µg/m³")
        print(f"  Confidence: {result.get('metadata', {}).get('confidence_score', 'N/A')}")
        print(f"  Health Risk: {result.get('health_impact', {}).get('risk_level', 'N/A')}")
        print(f"  Processing Time: {result.get('metadata', {}).get('processing_time_ms', 'N/A')}ms")
        print(f"  Prediction ID: {result.get('prediction_id', 'N/A')}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"✗ Error: {e}")
