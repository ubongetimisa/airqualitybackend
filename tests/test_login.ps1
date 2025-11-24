#!/usr/bin/env pwsh
<#
Complete test suite for the /token (login) endpoint
Tests various scenarios for authentication
#>

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "COMPLETE /token ENDPOINT TEST SUITE (LOGIN)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$baseUri = "http://localhost:8000"
$testResults = @()

# Test credentials - using a known registered user
$validEmail = "grace.researcher@example.edu"
$validPassword = "S3cur3P@ssw0rd!"

# Test Case 1: Valid login
Write-Host "`n[TEST 1] Valid login with correct credentials" -ForegroundColor Yellow
Write-Host "Email: $validEmail" -ForegroundColor Gray
Write-Host "Password: [REDACTED]" -ForegroundColor Gray

$body1 = @{
    username = $validEmail
    password = $validPassword
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUri/token" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $body1
    
    Write-Host "✅ SUCCESS - HTTP 200 OK" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Green
    Write-Host $response | ConvertTo-Json
    
    # Store token for later use
    $validToken = $response.access_token
    
    $testResults += @{test = "Test 1"; result = "PASS"; message = "Valid login successful" }
} catch {
    Write-Host "❌ FAILED - $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response.StatusCode -eq 401) {
        Write-Host "Note: This user may not exist. Try registering first." -ForegroundColor Yellow
    }
    $testResults += @{test = "Test 1"; result = "FAIL"; message = $_.Exception.Message }
}

# Test Case 2: Invalid password
Write-Host "`n[TEST 2] Invalid password (should fail)" -ForegroundColor Yellow
Write-Host "Email: $validEmail" -ForegroundColor Gray
Write-Host "Password: WrongPassword123!" -ForegroundColor Gray

$body2 = @{
    username = $validEmail
    password = "WrongPassword123!"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUri/token" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $body2
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed" -ForegroundColor Red
    $testResults += @{test = "Test 2"; result = "FAIL"; message = "Invalid password was accepted" }
} catch {
    if ($_.Exception.Response.StatusCode -eq 401) {
        Write-Host "✅ CORRECTLY FAILED - HTTP 401 Unauthorized" -ForegroundColor Green
        Write-Host "Expected behavior: Wrong password rejected" -ForegroundColor Green
        $testResults += @{test = "Test 2"; result = "PASS"; message = "Invalid password correctly rejected" }
    } else {
        Write-Host "❌ FAILED with unexpected status - $($_.Exception.Response.StatusCode)" -ForegroundColor Red
        $testResults += @{test = "Test 2"; result = "FAIL"; message = "Unexpected error" }
    }
}

# Test Case 3: Non-existent user
Write-Host "`n[TEST 3] Non-existent user (should fail)" -ForegroundColor Yellow
Write-Host "Email: nonexistent@example.com" -ForegroundColor Gray

$body3 = @{
    username = "nonexistent@example.com"
    password = "SomePassword123!"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUri/token" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $body3
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed" -ForegroundColor Red
    $testResults += @{test = "Test 3"; result = "FAIL"; message = "Non-existent user was accepted" }
} catch {
    if ($_.Exception.Response.StatusCode -eq 401) {
        Write-Host "✅ CORRECTLY FAILED - HTTP 401 Unauthorized" -ForegroundColor Green
        Write-Host "Expected behavior: Non-existent user rejected" -ForegroundColor Green
        $testResults += @{test = "Test 3"; result = "PASS"; message = "Non-existent user correctly rejected" }
    } else {
        Write-Host "❌ FAILED with unexpected status - $($_.Exception.Response.StatusCode)" -ForegroundColor Red
        $testResults += @{test = "Test 3"; result = "FAIL"; message = "Unexpected error" }
    }
}

# Test Case 4: Empty password
Write-Host "`n[TEST 4] Empty password (should fail)" -ForegroundColor Yellow
Write-Host "Email: $validEmail" -ForegroundColor Gray
Write-Host "Password: [empty]" -ForegroundColor Gray

$body4 = @{
    username = $validEmail
    password = ""
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUri/token" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $body4
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed" -ForegroundColor Red
    $testResults += @{test = "Test 4"; result = "FAIL"; message = "Empty password was accepted" }
} catch {
    if ($_.Exception.Response.StatusCode -eq 401) {
        Write-Host "✅ CORRECTLY FAILED - HTTP 401 Unauthorized" -ForegroundColor Green
        Write-Host "Expected behavior: Empty password rejected" -ForegroundColor Green
        $testResults += @{test = "Test 4"; result = "PASS"; message = "Empty password correctly rejected" }
    } else {
        Write-Host "❌ FAILED with unexpected status" -ForegroundColor Red
        $testResults += @{test = "Test 4"; result = "FAIL"; message = "Unexpected error" }
    }
}

# Test Case 5: Missing username
Write-Host "`n[TEST 5] Missing username (should fail)" -ForegroundColor Yellow

$body5 = @{
    password = "SomePassword123!"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUri/token" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $body5
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed" -ForegroundColor Red
    $testResults += @{test = "Test 5"; result = "FAIL"; message = "Missing username was accepted" }
} catch {
    if ($_.Exception.Response.StatusCode -eq 422) {
        Write-Host "✅ CORRECTLY FAILED - HTTP 422 Unprocessable Entity" -ForegroundColor Green
        Write-Host "Expected behavior: Missing field rejected" -ForegroundColor Green
        $testResults += @{test = "Test 5"; result = "PASS"; message = "Missing field validation working" }
    } else {
        Write-Host "❌ FAILED with unexpected status - $($_.Exception.Response.StatusCode)" -ForegroundColor Red
        $testResults += @{test = "Test 5"; result = "FAIL"; message = "Unexpected error" }
    }
}

# Test Case 6: Test token structure if login succeeded
if ($validToken) {
    Write-Host "`n[TEST 6] Validate token structure" -ForegroundColor Yellow
    
    Write-Host "Token (first 50 chars): $($validToken.Substring(0, 50))..." -ForegroundColor Gray
    
    try {
        # JWT tokens have 3 parts separated by dots
        $parts = $validToken.Split('.')
        
        if ($parts.Count -eq 3) {
            Write-Host "✅ Token has valid JWT structure (3 parts)" -ForegroundColor Green
            Write-Host "  Part 1 (Header): $($parts[0].Substring(0, 20))..." -ForegroundColor Gray
            Write-Host "  Part 2 (Payload): $($parts[1].Substring(0, 20))..." -ForegroundColor Gray
            Write-Host "  Part 3 (Signature): $($parts[2].Substring(0, 20))..." -ForegroundColor Gray
            $testResults += @{test = "Test 6"; result = "PASS"; message = "Token structure valid" }
        } else {
            Write-Host "❌ Invalid JWT structure - expected 3 parts, got $($parts.Count)" -ForegroundColor Red
            $testResults += @{test = "Test 6"; result = "FAIL"; message = "Invalid token structure" }
        }
    } catch {
        Write-Host "❌ Error validating token - $($_.Exception.Message)" -ForegroundColor Red
        $testResults += @{test = "Test 6"; result = "FAIL"; message = "Token validation error" }
    }
}

# Display Summary
Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
Write-Host "TEST SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan

$passed = ($testResults | Where-Object {$_.result -eq "PASS"}).Count
$failed = ($testResults | Where-Object {$_.result -eq "FAIL"}).Count

foreach ($result in $testResults) {
    $color = if ($result.result -eq "PASS") { "Green" } else { "Red" }
    $symbol = if ($result.result -eq "PASS") { "✅" } else { "❌" }
    Write-Host "$symbol $($result.test): $($result.result) - $($result.message)" -ForegroundColor $color
}

Write-Host "`n" -NoNewline
Write-Host "Total: " -ForegroundColor Cyan -NoNewline
Write-Host "$passed Passed, " -ForegroundColor Green -NoNewline
Write-Host "$failed Failed" -ForegroundColor Red

# Schema Reference
Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
Write-Host "TOKEN ENDPOINT SCHEMA" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host @"
Endpoint: POST /token
Content-Type: application/x-www-form-urlencoded

Request Body (Form Data):
  username: string (required) - User email address
  password: string (required) - User password

Response (200 OK):
{
  "access_token": "string (JWT token)",
  "token_type": "bearer",
  "user_id": "string (MongoDB ObjectId)",
  "user_email": "string"
}

Error Responses:
  401 Unauthorized: Incorrect email or password
  422 Unprocessable Entity: Missing required fields

HTTP Status Codes:
  - 200: Login successful, token generated
  - 401: Invalid credentials
  - 422: Validation error (missing fields)

Token Details:
  - Type: JWT (JSON Web Token)
  - Format: header.payload.signature
  - Expiration: 30 minutes (configurable)
  - Algorithm: HS256
  - Payload includes: user email (sub), expiration (exp)

Usage:
  Include token in Authorization header for protected endpoints:
  Authorization: Bearer <token>
"@ -ForegroundColor White

Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "`nNote: Use 'grace.researcher@example.edu' for testing if previously registered." -ForegroundColor Yellow
Write-Host ("=" * 80) -ForegroundColor Cyan
