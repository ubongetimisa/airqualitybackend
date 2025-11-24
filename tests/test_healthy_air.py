#!/usr/bin/env python3
"""
Test script to make predictions with healthy/excellent air quality levels
"""
import requests
import json

API_URL = "http://localhost:8000"

# Test cases with clean air quality
test_cases = [
    {
        "name": "EXCELLENT - Zurich Clean Air",
        "data": {
            "location": {
                "city": "Zurich",
                "country": "Switzerland",
                "date": "2025-11-22"
            },
            "air_quality": {
                "pm10": 15.0,
                "no2": 12.0,
                "so2": 3.0,
                "co": 0.3,
                "o3": 45.0,
                "temperature": 8.0,
                "humidity": 55.0,
                "wind_speed": 5.5
            },
            "model": "ensemble",
            "user_id": "test_excellent_001"
        }
    },
    {
        "name": "EXCELLENT - Stockholm Fresh Air",
        "data": {
            "location": {
                "city": "Stockholm",
                "country": "Sweden",
                "date": "2025-11-22"
            },
            "air_quality": {
                "pm10": 12.0,
                "no2": 10.0,
                "so2": 2.0,
                "co": 0.2,
                "o3": 50.0,
                "temperature": 5.0,
                "humidity": 60.0,
                "wind_speed": 6.0
            },
            "model": "ensemble",
            "user_id": "test_excellent_002"
        }
    },
    {
        "name": "GOOD - Vancouver Clean",
        "data": {
            "location": {
                "city": "Vancouver",
                "country": "Canada",
                "date": "2025-11-22"
            },
            "air_quality": {
                "pm10": 25.0,
                "no2": 18.0,
                "so2": 5.0,
                "co": 0.5,
                "o3": 55.0,
                "temperature": 10.0,
                "humidity": 70.0,
                "wind_speed": 4.5
            },
            "model": "ensemble",
            "user_id": "test_good_001"
        }
    },
    {
        "name": "GOOD - Auckland Marine",
        "data": {
            "location": {
                "city": "Auckland",
                "country": "New Zealand",
                "date": "2025-11-22"
            },
            "air_quality": {
                "pm10": 22.0,
                "no2": 15.0,
                "so2": 4.0,
                "co": 0.4,
                "o3": 60.0,
                "temperature": 18.0,
                "humidity": 65.0,
                "wind_speed": 7.0
            },
            "model": "ensemble",
            "user_id": "test_good_002"
        }
    },
    {
        "name": "SATISFACTORY - Munich",
        "data": {
            "location": {
                "city": "Munich",
                "country": "Germany",
                "date": "2025-11-22"
            },
            "air_quality": {
                "pm10": 40.0,
                "no2": 35.0,
                "so2": 10.0,
                "co": 0.8,
                "o3": 50.0,
                "temperature": 7.0,
                "humidity": 60.0,
                "wind_speed": 3.5
            },
            "model": "ensemble",
            "user_id": "test_satisfactory_001"
        }
    }
]

print("=" * 80)
print("🧪 TESTING HEALTHY AIR QUALITY CONDITIONS")
print("=" * 80)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n📍 Test {i}: {test_case['name']}")
    print(f"   Location: {test_case['data']['location']['city']}, {test_case['data']['location']['country']}")
    print(f"   PM10: {test_case['data']['air_quality']['pm10']} µg/m³")
    print(f"   NO2: {test_case['data']['air_quality']['no2']} ppb")
    
    try:
        response = requests.post(
            f"{API_URL}/api/v2/predictions/predict",
            json=test_case['data'],
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ SUCCESS")
            print(f"   → PM2.5 Predicted: {result.get('pm25_predicted', 'N/A')} µg/m³")
            print(f"   → Health Risk: {result.get('health_impact', {}).get('risk_level', 'N/A')}")
            print(f"   → AQI Category: {result.get('health_impact', {}).get('aqi_category', 'N/A')}")
            print(f"   → Confidence: {result.get('metadata', {}).get('confidence_score', 'N/A')}")
            print(f"   → Processing Time: {result.get('metadata', {}).get('processing_time_ms', 'N/A')}ms")
        else:
            print(f"   ✗ Error: {response.status_code}")
            print(f"   {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("✓ All tests complete! Check Public Predictions tab to see the results")
print("=" * 80)
