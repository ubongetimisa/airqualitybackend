#!/usr/bin/env python3
"""
Test the public predictions endpoint
"""
import requests
import json

API_URL = "http://localhost:8000"

print("📊 Fetching public predictions...")

try:
    response = requests.get(
        f"{API_URL}/public/predictions?limit=20",
        timeout=10
    )
    
    if response.status_code == 200:
        predictions = response.json()
        print(f"✓ Successfully retrieved {len(predictions)} predictions")
        
        if predictions:
            print(f"\n📋 First prediction:")
            first = predictions[0]
            print(f"  City: {first.get('city')}")
            print(f"  Country: {first.get('country')}")
            print(f"  Date: {first.get('date')}")
            print(f"  PM2.5: {first.get('pm25_predicted')} µg/m³")
            print(f"  Health Risk: {first.get('health_impact', {}).get('risk_level')}")
            print(f"  Confidence: {first.get('metadata', {}).get('confidence_score')}")
            print(f"  Processing Time: {first.get('metadata', {}).get('processing_time_ms')}ms")
            print(f"  Created: {first.get('created_at')}")
        else:
            print("⚠ No predictions found")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"✗ Error: {e}")
