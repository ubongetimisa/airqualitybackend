#!/bin/bash
# Quick API Testing Script
# Test all prediction endpoints with real data

BASE_URL="http://localhost:8000/api/v2/predictions"

echo "================================================"
echo "Air Quality Prediction API - Quick Test"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test 1: Health Check
echo -e "${BLUE}1. Health Check${NC}"
curl -s "$BASE_URL/health" | python -m json.tool
echo ""

# Test 2: List Models
echo -e "${BLUE}2. List Available Models${NC}"
curl -s "$BASE_URL/models" | python -m json.tool
echo ""

# Test 3: Feature Info
echo -e "${BLUE}3. Feature Engineering Info${NC}"
curl -s "$BASE_URL/feature-info" | python -m json.tool
echo ""

# Test 4: Health Scale
echo -e "${BLUE}4. Health Impact Scale${NC}"
curl -s "$BASE_URL/health-scale" | python -m json.tool | head -50
echo "..."
echo ""

# Test 5: Prediction - Good Air Quality (London)
echo -e "${BLUE}5. Prediction: London (Good Air Quality)${NC}"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "location": {
      "city": "London",
      "country": "United Kingdom",
      "date": "2025-11-21"
    },
    "air_quality": {
      "pm10": 45.5,
      "no2": 25.3,
      "so2": 12.1,
      "co": 0.8,
      "o3": 35.2,
      "temperature": 15.5,
      "humidity": 65.0,
      "wind_speed": 3.2
    }
  }' | python -m json.tool
echo ""

# Test 6: Prediction - Moderate Air Quality (Paris)
echo -e "${BLUE}6. Prediction: Paris (Moderate Air Quality)${NC}"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "location": {
      "city": "Paris",
      "country": "France",
      "date": "2025-11-21"
    },
    "air_quality": {
      "pm10": 55.0,
      "no2": 35.5,
      "so2": 18.0,
      "co": 1.2,
      "o3": 32.0,
      "temperature": 14.2,
      "humidity": 70.0,
      "wind_speed": 2.8
    }
  }' | python -m json.tool
echo ""

# Test 7: Prediction - Poor Air Quality (Delhi)
echo -e "${BLUE}7. Prediction: Delhi (Unhealthy Air Quality)${NC}"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "location": {
      "city": "Delhi",
      "country": "India",
      "date": "2025-11-21"
    },
    "air_quality": {
      "pm10": 250.0,
      "no2": 120.5,
      "so2": 45.2,
      "co": 2.5,
      "o3": 25.0,
      "temperature": 22.0,
      "humidity": 55.0,
      "wind_speed": 1.5
    }
  }' | python -m json.tool
echo ""

# Test 8: Batch Prediction
echo -e "${BLUE}8. Batch Predictions (3 cities)${NC}"
curl -s -X POST "$BASE_URL/predict-batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "location": {"city": "London", "country": "UK", "date": "2025-11-21"},
      "air_quality": {"pm10": 45.5, "no2": 25.3, "so2": 12.1, "co": 0.8, "o3": 35.2, "temperature": 15.5, "humidity": 65.0, "wind_speed": 3.2}
    },
    {
      "location": {"city": "Paris", "country": "France", "date": "2025-11-21"},
      "air_quality": {"pm10": 55.0, "no2": 35.5, "so2": 18.0, "co": 1.2, "o3": 32.0, "temperature": 14.2, "humidity": 70.0, "wind_speed": 2.8}
    },
    {
      "location": {"city": "Delhi", "country": "India", "date": "2025-11-21"},
      "air_quality": {"pm10": 250.0, "no2": 120.5, "so2": 45.2, "co": 2.5, "o3": 25.0, "temperature": 22.0, "humidity": 55.0, "wind_speed": 1.5}
    }
  ]' | python -m json.tool
echo ""

echo "================================================"
echo -e "${GREEN}✓ All tests completed!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Access Swagger UI: http://localhost:8000/docs"
echo "2. Check logs: backend/logs/"
echo "3. Read documentation: API_GUIDE_v2.md"
echo "4. Run Python tests: python test_api.py"
