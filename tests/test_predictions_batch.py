#!/usr/bin/env python3
"""
Test script with multiple realistic predictions
"""
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8000"

# Multiple realistic test predictions
test_predictions = [
    {
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
        "user_id": "user_001"
    },
    {
        "location": {
            "city": "New York",
            "country": "USA",
            "date": "2025-11-22"
        },
        "air_quality": {
            "pm10": 38.2,
            "no2": 32.5,
            "so2": 8.3,
            "co": 1.2,
            "o3": 42.1,
            "temperature": 12.3,
            "humidity": 58,
            "wind_speed": 4.5
        },
        "model": "ensemble",
        "user_id": "user_002"
    },
    {
        "location": {
            "city": "Delhi",
            "country": "India",
            "date": "2025-11-22"
        },
        "air_quality": {
            "pm10": 185.6,
            "no2": 85.2,
            "so2": 35.4,
            "co": 2.1,
            "o3": 62.3,
            "temperature": 28.5,
            "humidity": 42,
            "wind_speed": 2.1
        },
        "model": "ensemble",
        "user_id": "user_003"
    },
    {
        "location": {
            "city": "Tokyo",
            "country": "Japan",
            "date": "2025-11-22"
        },
        "air_quality": {
            "pm10": 32.1,
            "no2": 28.9,
            "so2": 6.2,
            "co": 0.6,
            "o3": 38.5,
            "temperature": 18.2,
            "humidity": 72,
            "wind_speed": 5.3
        },
        "model": "ensemble",
        "user_id": "user_004"
    },
    {
        "location": {
            "city": "Beijing",
            "country": "China",
            "date": "2025-11-22"
        },
        "air_quality": {
            "pm10": 125.4,
            "no2": 56.8,
            "so2": 22.1,
            "co": 1.5,
            "o3": 45.6,
            "temperature": 5.2,
            "humidity": 55,
            "wind_speed": 3.8
        },
        "model": "ensemble",
        "user_id": "user_005"
    }
]

print("=" * 80)
print("🧪 TESTING PREDICTION ENDPOINT")
print("=" * 80)

successful = 0
failed = 0

for i, pred_data in enumerate(test_predictions, 1):
    city = pred_data['location']['city']
    country = pred_data['location']['country']
    
    print(f"\n📍 Test {i}: {city}, {country}")
    print(f"   PM10: {pred_data['air_quality']['pm10']} µg/m³")
    print(f"   NO2: {pred_data['air_quality']['no2']} ppb")
    print(f"   Temperature: {pred_data['air_quality']['temperature']}°C")
    
    try:
        response = requests.post(
            f"{API_URL}/api/v2/predictions/predict",
            json=pred_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ SUCCESS")
            print(f"   → PM2.5 Predicted: {result.get('pm25_predicted', 'N/A'):.2f} µg/m³")
            print(f"   → Confidence: {result.get('metadata', {}).get('confidence_score', 'N/A'):.1%}")
            print(f"   → Health Risk: {result.get('health_impact', {}).get('risk_level', 'N/A')}")
            print(f"   → Processing Time: {result.get('metadata', {}).get('processing_time_ms', 'N/A'):.1f}ms")
            successful += 1
        else:
            print(f"   ✗ ERROR: {response.status_code}")
            print(f"   {response.text[:200]}")
            failed += 1
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)[:100]}")
        failed += 1

print("\n" + "=" * 80)
print(f"📊 RESULTS: {successful} successful, {failed} failed")
print("=" * 80)
print("\n✓ Check the database now to see if predictions were saved!")
