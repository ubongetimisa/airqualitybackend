# Quick API Testing Script for Windows PowerShell
# Test all prediction endpoints with real data

$BASE_URL = "http://localhost:8000/api/v2/predictions"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Air Quality Prediction API - Quick Test" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check
Write-Host "1. Health Check" -ForegroundColor Blue
$response = Invoke-RestMethod -Uri "$BASE_URL/health" -Method Get
$response | ConvertTo-Json | Write-Host
Write-Host ""

# Test 2: List Models
Write-Host "2. List Available Models" -ForegroundColor Blue
$response = Invoke-RestMethod -Uri "$BASE_URL/models" -Method Get
$response | ConvertTo-Json | Write-Host
Write-Host ""

# Test 3: Feature Info
Write-Host "3. Feature Engineering Info" -ForegroundColor Blue
$response = Invoke-RestMethod -Uri "$BASE_URL/feature-info" -Method Get
$response | ConvertTo-Json | Write-Host
Write-Host ""

# Test 4: Health Scale
Write-Host "4. Health Impact Scale (First 50 entries)" -ForegroundColor Blue
$response = Invoke-RestMethod -Uri "$BASE_URL/health-scale" -Method Get
($response | ConvertTo-Json).Split("`n") | Select-Object -First 50 | Write-Host
Write-Host "..."
Write-Host ""

# Test 5: Prediction - Good Air Quality (London)
Write-Host "5. Prediction: London (Good Air Quality)" -ForegroundColor Blue
$body = @{
    location = @{
        city = "London"
        country = "United Kingdom"
        date = "2025-11-21"
    }
    air_quality = @{
        pm10 = 45.5
        no2 = 25.3
        so2 = 12.1
        co = 0.8
        o3 = 35.2
        temperature = 15.5
        humidity = 65.0
        wind_speed = 3.2
    }
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "$BASE_URL/predict" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
$response | ConvertTo-Json | Write-Host
Write-Host ""

# Test 6: Prediction - Moderate Air Quality (Paris)
Write-Host "6. Prediction: Paris (Moderate Air Quality)" -ForegroundColor Blue
$body = @{
    location = @{
        city = "Paris"
        country = "France"
        date = "2025-11-21"
    }
    air_quality = @{
        pm10 = 55.0
        no2 = 35.5
        so2 = 18.0
        co = 1.2
        o3 = 32.0
        temperature = 14.2
        humidity = 70.0
        wind_speed = 2.8
    }
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "$BASE_URL/predict" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
$response | ConvertTo-Json | Write-Host
Write-Host ""

# Test 7: Prediction - Poor Air Quality (Delhi)
Write-Host "7. Prediction: Delhi (Unhealthy Air Quality)" -ForegroundColor Blue
$body = @{
    location = @{
        city = "Delhi"
        country = "India"
        date = "2025-11-21"
    }
    air_quality = @{
        pm10 = 250.0
        no2 = 120.5
        so2 = 45.2
        co = 2.5
        o3 = 25.0
        temperature = 22.0
        humidity = 55.0
        wind_speed = 1.5
    }
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "$BASE_URL/predict" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
$response | ConvertTo-Json | Write-Host
Write-Host ""

# Test 8: Batch Prediction
Write-Host "8. Batch Predictions (3 cities)" -ForegroundColor Blue
$body = @(
    @{
        location = @{city = "London"; country = "UK"; date = "2025-11-21"}
        air_quality = @{pm10 = 45.5; no2 = 25.3; so2 = 12.1; co = 0.8; o3 = 35.2; temperature = 15.5; humidity = 65.0; wind_speed = 3.2}
    },
    @{
        location = @{city = "Paris"; country = "France"; date = "2025-11-21"}
        air_quality = @{pm10 = 55.0; no2 = 35.5; so2 = 18.0; co = 1.2; o3 = 32.0; temperature = 14.2; humidity = 70.0; wind_speed = 2.8}
    },
    @{
        location = @{city = "Delhi"; country = "India"; date = "2025-11-21"}
        air_quality = @{pm10 = 250.0; no2 = 120.5; so2 = 45.2; co = 2.5; o3 = 25.0; temperature = 22.0; humidity = 55.0; wind_speed = 1.5}
    }
) | ConvertTo-Json

$response = Invoke-RestMethod -Uri "$BASE_URL/predict-batch" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
$response | ConvertTo-Json -Depth 10 | Write-Host
Write-Host ""

Write-Host "================================================" -ForegroundColor Green
Write-Host "✓ All tests completed!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Access Swagger UI: http://localhost:8000/docs"
Write-Host "2. Check logs: backend/logs/"
Write-Host "3. Read documentation: API_GUIDE_v2.md"
Write-Host "4. View MongoDB data: Check your MongoDB instance"
