#!/usr/bin/env python
"""Debug script to test registration endpoint"""
import requests
import json

BASE_URL = "http://localhost:8000"

test_data = {
    "email": "test_debug@example.com",
    "password": "TestPassword123!",
    "full_name": "Test User Debug",
    "affiliation": "Test University"
}

print(f"Sending registration request to {BASE_URL}/register")
print(f"Data: {json.dumps(test_data, indent=2)}")
print("-" * 50)

try:
    response = requests.post(
        f"{BASE_URL}/register",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body: {response.text}")
    
    if response.status_code == 200:
        print("\n✓ Registration successful!")
        print(json.dumps(response.json(), indent=2))
    else:
        print("\n✗ Registration failed!")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
            
except Exception as e:
    print(f"Error: {e}")
