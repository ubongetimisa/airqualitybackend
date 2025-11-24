#!/usr/bin/env python3
"""
Test script to make predictions with extreme pollution levels
"""
import requests
import json

API_URL = "http://localhost:8000"

# Test cases with different pollution levels
test_cases = [
    {
        "name": "HAZARDOUS - New Delhi Severe Smog",
        "data": {
            "location": {
                "city": "New Delhi",
                "country": "India",
                "date": "2025-11-22"
            },
            "air_quality": {
                "pm10": 500.0,
                "no2": 250.0,
                "so2": 150.0,
                "co": 15.0,
                "o3": 10.0,
                "temperature": 15.0,
                "humidity": 30.0,
                "wind_speed": 0.5
            },
            "model": "ensemble",
            "user_id": "test_hazardous_001"
        }
    },
    {
        "name": "VERY UNHEALTHY - Lahore Winter Pollution",
        "data": {
            "location": {
                "city": "Lahore",
                "country": "Pakistan",
                "date": "2025-11-22"
            },
            "air_quality": {
                "pm10": 350.0,
                "no2": 180.0,
                "so2": 95.0,
                "co": 8.5,
                "o3": 25.0,
                "temperature": 18.0,
                "humidity": 45.0,
                "wind_speed": 1.2
            },
            "model": "ensemble",
            "user_id": "test_very_unhealthy_001"
        }
    },
    {
        "name": "UNHEALTHY - Cairo Desert Dust",
        "data": {
            "location": {
                "city": "Cairo",
                "country": "Egypt",
                "date": "2025-11-22"
            },
            "air_quality": {
                "pm10": 250.0,
                "no2": 120.0,
                "so2": 60.0,
                "co": 5.0,
                "o3": 40.0,
                "temperature": 32.0,
                "humidity": 20.0,
                "wind_speed": 8.5
            },
            "model": "ensemble",
            "user_id": "test_unhealthy_001"
        }
    },
    {
        "name": "UNHEALTHY FOR SENSITIVE - Bangkok Traffic",
        "data": {
            "location": {
                "city": "Bangkok",
                "country": "Thailand",
                "date": "2025-11-22"
            },
            "air_quality": {
                "pm10": 120.0,
                "no2": 85.0,
                "so2": 35.0,
                "co": 3.5,
                "o3": 55.0,
                "temperature": 28.0,
                "humidity": 70.0,
                "wind_speed": 2.5
            },
            "model": "ensemble",
            "user_id": "test_sensitive_001"
        }
    },
    {
        "name": "MODERATE - Paris Urban",
        "data": {
            "location": {
                "city": "Paris",
                "country": "France",
                "date": "2025-11-22"
            },
            "air_quality": {
                "pm10": 55.0,
                "no2": 45.0,
                "so2": 15.0,
                "co": 1.2,
                "o3": 65.0,
                "temperature": 12.0,
                "humidity": 65.0,
                "wind_speed": 4.5
            },
            "model": "ensemble",
            "user_id": "test_moderate_001"
        }
    }
]

print("=" * 80)
print("🧪 TESTING EXTREME POLLUTION CONDITIONS")
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
