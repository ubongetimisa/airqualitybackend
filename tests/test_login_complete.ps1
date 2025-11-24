#!/usr/bin/env pwsh
<#
Complete Login & Authentication Test Suite
Tests token generation, usage, and protected endpoints
#>

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "LOGIN & AUTHENTICATION TEST SUITE" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$baseUri = "http://localhost:8000"
$testResults = @()

# Step 1: Get valid credentials
Write-Host "`n[SETUP] Using credentials:" -ForegroundColor Magenta
$email = "grace.researcher@example.edu"
$password = "S3cur3P@ssw0rd!"
Write-Host "  Email: $email" -ForegroundColor Gray
Write-Host "  Password: ***" -ForegroundColor Gray

# TEST 1: Valid login
Write-Host "`n[TEST 1] Valid login with correct credentials" -ForegroundColor Yellow
$loginBody = @{
    username = $email
    password = $password
} | ConvertTo-Json

Write-Host "Request Body:" -ForegroundColor Gray
Write-Host $loginBody

$token = $null
try {
    $response = Invoke-RestMethod -Uri "$baseUri/token" `
        -Method POST `
        -Headers @{"Content-Type" = "application/x-www-form-urlencoded"} `
        -Body "username=$email&password=$password"
    
    $token = $response.access_token
    Write-Host "✅ SUCCESS - Status: 200" -ForegroundColor Green
    Write-Host "Token Type: $($response.token_type)" -ForegroundColor Green
    Write-Host "User ID: $($response.user_id)" -ForegroundColor Green
    Write-Host "User Email: $($response.user_email)" -ForegroundColor Green
    Write-Host "Token (first 50 chars): $($token.Substring(0, 50))..." -ForegroundColor Green
    $testResults += @{test = "Test 1"; result = "PASS"; message = "Valid login successful" }
} catch {
    Write-Host "❌ FAILED - $($_.Exception.Message)" -ForegroundColor Red
    $testResults += @{test = "Test 1"; result = "FAIL"; message = $_.Exception.Message }
}

# TEST 2: Invalid password
Write-Host "`n[TEST 2] Login with wrong password (should fail)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUri/token" `
        -Method POST `
        -Headers @{"Content-Type" = "application/x-www-form-urlencoded"} `
        -Body "username=$email&password=WrongPassword123!"
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed" -ForegroundColor Red
    $testResults += @{test = "Test 2"; result = "FAIL"; message = "Invalid password accepted" }
} catch {
    Write-Host "✅ CORRECTLY FAILED - $($_.Exception.Message)" -ForegroundColor Green
    Write-Host "Expected behavior: Incorrect credentials rejected" -ForegroundColor Green
    $testResults += @{test = "Test 2"; result = "PASS"; message = "Password validation working" }
}

# TEST 3: Non-existent user
Write-Host "`n[TEST 3] Login with non-existent email (should fail)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUri/token" `
        -Method POST `
        -Headers @{"Content-Type" = "application/x-www-form-urlencoded"} `
        -Body "username=nonexistent@example.com&password=AnyPassword123!"
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed" -ForegroundColor Red
    $testResults += @{test = "Test 3"; result = "FAIL"; message = "Non-existent user accepted" }
} catch {
    Write-Host "✅ CORRECTLY FAILED - User not found" -ForegroundColor Green
    $testResults += @{test = "Test 3"; result = "PASS"; message = "User validation working" }
}

# TEST 4: Use token to access protected endpoint (/users/me)
if ($token) {
    Write-Host "`n[TEST 4] Access protected endpoint (/users/me) with valid token" -ForegroundColor Yellow
    Write-Host "Using token header: Authorization: Bearer $($token.Substring(0, 30))..." -ForegroundColor Gray
    
    try {
        $response = Invoke-RestMethod -Uri "$baseUri/users/me" `
            -Method GET `
            -Headers @{"Authorization" = "Bearer $token"}
        
        Write-Host "✅ SUCCESS - Retrieved user info" -ForegroundColor Green
        Write-Host "User ID: $($response.id)" -ForegroundColor Green
        Write-Host "Email: $($response.email)" -ForegroundColor Green
        Write-Host "Full Name: $($response.full_name)" -ForegroundColor Green
        Write-Host "Affiliation: $($response.affiliation)" -ForegroundColor Green
        Write-Host "Research Interests: $($response.research_interests -join ', ')" -ForegroundColor Green
        Write-Host "Created: $($response.created_at)" -ForegroundColor Green
        $testResults += @{test = "Test 4"; result = "PASS"; message = "Protected endpoint access working" }
    } catch {
        Write-Host "❌ FAILED - $($_.Exception.Message)" -ForegroundColor Red
        $testResults += @{test = "Test 4"; result = "FAIL"; message = $_.Exception.Message }
    }
}

# TEST 5: Access protected endpoint without token (should fail)
Write-Host "`n[TEST 5] Access protected endpoint WITHOUT token (should fail)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUri/users/me" -Method GET
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed without token" -ForegroundColor Red
    $testResults += @{test = "Test 5"; result = "FAIL"; message = "Missing token accepted" }
} catch {
    Write-Host "✅ CORRECTLY FAILED - Access denied without token" -ForegroundColor Green
    $testResults += @{test = "Test 5"; result = "PASS"; message = "Token requirement enforced" }
}

# TEST 6: Access protected endpoint with invalid token (should fail)
Write-Host "`n[TEST 6] Access protected endpoint with invalid token (should fail)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUri/users/me" `
        -Method GET `
        -Headers @{"Authorization" = "Bearer invalid_token_xyz"}
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed with invalid token" -ForegroundColor Red
    $testResults += @{test = "Test 6"; result = "FAIL"; message = "Invalid token accepted" }
} catch {
    Write-Host "✅ CORRECTLY FAILED - Invalid token rejected" -ForegroundColor Green
    $testResults += @{test = "Test 6"; result = "PASS"; message = "Token validation working" }
}

# TEST 7: Check token format
if ($token) {
    Write-Host "`n[TEST 7] Verify JWT token format" -ForegroundColor Yellow
    $parts = $token.Split('.')
    
    if ($parts.Length -eq 3) {
        Write-Host "✅ Token has correct JWT format (3 parts separated by dots)" -ForegroundColor Green
        Write-Host "  Header length: $($parts[0].Length) chars" -ForegroundColor Gray
        Write-Host "  Payload length: $($parts[1].Length) chars" -ForegroundColor Gray
        Write-Host "  Signature length: $($parts[2].Length) chars" -ForegroundColor Gray
        $testResults += @{test = "Test 7"; result = "PASS"; message = "JWT format correct" }
    } else {
        Write-Host "❌ Token does not have valid JWT format" -ForegroundColor Red
        $testResults += @{test = "Test 7"; result = "FAIL"; message = "Invalid JWT format" }
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

# API Schema Reference
Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
Write-Host "API SCHEMA REFERENCE" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host @"
LOGIN ENDPOINT: POST /token
Request Format: application/x-www-form-urlencoded
Body:
  username=email@example.com
  password=YourPassword123!

Response (200 OK):
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "507f1f77bcf86cd799439011",
  "user_email": "user@example.com"
}

ERROR RESPONSES:
- 401 Unauthorized: Incorrect email or password
- 422 Unprocessable Entity: Missing username or password

PROTECTED ENDPOINT: GET /users/me
Requires: Authorization header with Bearer token
Response (200 OK):
{
  "id": "507f1f77bcf86cd799439011",
  "email": "user@example.com",
  "full_name": "User Name",
  "affiliation": "Organization",
  "research_interests": ["Topic1", "Topic2"],
  "created_at": "2025-11-12T12:18:29"
}

ERROR RESPONSES:
- 401 Unauthorized: Missing or invalid token

COMMON ERRORS:
- 422 Unprocessable: Check Content-Type and form data encoding
- 401 Unauthorized: Token expired or invalid
- 400 Bad Request: User not found or password incorrect
"@ -ForegroundColor White

Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
Write-Host "💡 TIP: Save the access_token from login and use it for protected endpoints" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
