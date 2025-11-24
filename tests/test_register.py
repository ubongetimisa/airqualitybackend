#!/usr/bin/env python3
"""
Test the /register endpoint with proper JSON
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# Test data
test_user = {
    "email": "alice.researcher@example.edu",
    "password": "S3cur3P@ssw0rd!",
    "full_name": "Alice M. Researcher",
    "affiliation": "Institute of Atmospheric Studies",
    "research_interests": ["Air Quality", "Epidemiology", "Time Series Analysis"]
}

print("=" * 80)
print("TESTING /register ENDPOINT")
print("=" * 80)
print(f"\nSending JSON payload:")
print(json.dumps(test_user, indent=2))

try:
    response = requests.post(
        f"{BASE_URL}/register",
        json=test_user,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"\nResponse Body:")
    print(json.dumps(response.json(), indent=2))
    
    if response.status_code == 200:
        print("\n✅ Registration successful!")
    else:
        print(f"\n❌ Registration failed with status {response.status_code}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")

print("=" * 80)
