#!/usr/bin/env pwsh
<#
Complete test suite for the /register endpoint
Tests various scenarios and edge cases
#>

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "COMPLETE /register ENDPOINT TEST SUITE" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$baseUri = "http://localhost:8000"
$testResults = @()

# Test Case 1: Valid registration with all fields
Write-Host "`n[TEST 1] Valid registration with all fields" -ForegroundColor Yellow
$test1Body = @{
    email = "alice.researcher@example.edu"
    password = "S3cur3P@ssw0rd!"
    full_name = "Alice M. Researcher"
    affiliation = "Institute of Atmospheric Studies"
    research_interests = @("Air Quality", "Epidemiology", "Time Series Analysis")
} | ConvertTo-Json

Write-Host "Request Body:" -ForegroundColor Gray
Write-Host $test1Body

try {
    $response = Invoke-RestMethod -Uri "$baseUri/register" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $test1Body
    
    Write-Host "✅ SUCCESS - Status: 200" -ForegroundColor Green
    Write-Host "Response: $($response | ConvertTo-Json -Depth 5)" -ForegroundColor Green
    $testResults += @{test = "Test 1"; result = "PASS"; message = "Valid registration" }
} catch {
    Write-Host "❌ FAILED - $($_.Exception.Message)" -ForegroundColor Red
    $testResults += @{test = "Test 1"; result = "FAIL"; message = $_.Exception.Message }
}

# Test Case 2: Valid registration with minimal fields (no affiliation, empty interests)
Write-Host "`n[TEST 2] Valid registration with minimal fields" -ForegroundColor Yellow
$test2Body = @{
    email = "bob.smith@example.com"
    password = "TestPass123!"
    full_name = "Bob Smith"
    affiliation = $null
    research_interests = @()
} | ConvertTo-Json

Write-Host "Request Body:" -ForegroundColor Gray
Write-Host $test2Body

try {
    $response = Invoke-RestMethod -Uri "$baseUri/register" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $test2Body
    
    Write-Host "✅ SUCCESS - Status: 200" -ForegroundColor Green
    Write-Host "Response ID: $($response.id)" -ForegroundColor Green
    $testResults += @{test = "Test 2"; result = "PASS"; message = "Minimal fields registration" }
} catch {
    Write-Host "❌ FAILED - $($_.Exception.Message)" -ForegroundColor Red
    $testResults += @{test = "Test 2"; result = "FAIL"; message = $_.Exception.Message }
}

# Test Case 3: Invalid email format
Write-Host "`n[TEST 3] Invalid email format (should fail)" -ForegroundColor Yellow
$test3Body = @{
    email = "not-an-email"
    password = "TestPass123!"
    full_name = "Test User"
    affiliation = "Test Org"
    research_interests = @("Test")
} | ConvertTo-Json

Write-Host "Request Body:" -ForegroundColor Gray
Write-Host $test3Body

try {
    $response = Invoke-RestMethod -Uri "$baseUri/register" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $test3Body
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed with invalid email" -ForegroundColor Red
    $testResults += @{test = "Test 3"; result = "FAIL"; message = "Invalid email accepted" }
} catch {
    Write-Host "✅ CORRECTLY FAILED - $($_.Exception.Message)" -ForegroundColor Green
    Write-Host "Expected behavior: Email validation rejected" -ForegroundColor Green
    $testResults += @{test = "Test 3"; result = "PASS"; message = "Email validation working" }
}

# Test Case 4: Missing required field (email)
Write-Host "`n[TEST 4] Missing required field - email (should fail)" -ForegroundColor Yellow
$test4Body = @{
    password = "TestPass123!"
    full_name = "Test User"
    affiliation = "Test Org"
    research_interests = @()
} | ConvertTo-Json

Write-Host "Request Body:" -ForegroundColor Gray
Write-Host $test4Body

try {
    $response = Invoke-RestMethod -Uri "$baseUri/register" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $test4Body
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed with missing email" -ForegroundColor Red
    $testResults += @{test = "Test 4"; result = "FAIL"; message = "Missing field accepted" }
} catch {
    Write-Host "✅ CORRECTLY FAILED - $($_.Exception.Message)" -ForegroundColor Green
    Write-Host "Expected behavior: Missing required field rejected" -ForegroundColor Green
    $testResults += @{test = "Test 4"; result = "PASS"; message = "Required field validation working" }
}

# Test Case 5: Missing required field (password)
Write-Host "`n[TEST 5] Missing required field - password (should fail)" -ForegroundColor Yellow
$test5Body = @{
    email = "charlie@example.com"
    full_name = "Charlie Brown"
    affiliation = "Test Org"
    research_interests = @()
} | ConvertTo-Json

Write-Host "Request Body:" -ForegroundColor Gray
Write-Host $test5Body

try {
    $response = Invoke-RestMethod -Uri "$baseUri/register" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $test5Body
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed with missing password" -ForegroundColor Red
    $testResults += @{test = "Test 5"; result = "FAIL"; message = "Missing field accepted" }
} catch {
    Write-Host "✅ CORRECTLY FAILED - $($_.Exception.Message)" -ForegroundColor Green
    Write-Host "Expected behavior: Missing required field rejected" -ForegroundColor Green
    $testResults += @{test = "Test 5"; result = "PASS"; message = "Required field validation working" }
}

# Test Case 6: Missing required field (full_name)
Write-Host "`n[TEST 6] Missing required field - full_name (should fail)" -ForegroundColor Yellow
$test6Body = @{
    email = "diana@example.com"
    password = "TestPass123!"
    affiliation = "Test Org"
    research_interests = @()
} | ConvertTo-Json

Write-Host "Request Body:" -ForegroundColor Gray
Write-Host $test6Body

try {
    $response = Invoke-RestMethod -Uri "$baseUri/register" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $test6Body
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed with missing full_name" -ForegroundColor Red
    $testResults += @{test = "Test 6"; result = "FAIL"; message = "Missing field accepted" }
} catch {
    Write-Host "✅ CORRECTLY FAILED - $($_.Exception.Message)" -ForegroundColor Green
    Write-Host "Expected behavior: Missing required field rejected" -ForegroundColor Green
    $testResults += @{test = "Test 6"; result = "PASS"; message = "Required field validation working" }
}

# Test Case 7: Duplicate email (if first test succeeded)
Write-Host "`n[TEST 7] Duplicate email (should fail with 400)" -ForegroundColor Yellow
$test7Body = @{
    email = "alice.researcher@example.edu"
    password = "DifferentPass123!"
    full_name = "Different Name"
    affiliation = "Different Org"
    research_interests = @()
} | ConvertTo-Json

Write-Host "Request Body:" -ForegroundColor Gray
Write-Host $test7Body

try {
    $response = Invoke-RestMethod -Uri "$baseUri/register" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $test7Body
    
    Write-Host "❌ UNEXPECTED SUCCESS - Should have failed with duplicate email" -ForegroundColor Red
    $testResults += @{test = "Test 7"; result = "FAIL"; message = "Duplicate email accepted" }
} catch {
    if ($_.Exception.Response.StatusCode -eq 400) {
        Write-Host "✅ CORRECTLY FAILED - HTTP 400 Bad Request" -ForegroundColor Green
        Write-Host "Expected behavior: Duplicate email rejected" -ForegroundColor Green
        $testResults += @{test = "Test 7"; result = "PASS"; message = "Duplicate email validation working" }
    } else {
        Write-Host "❌ FAILED with unexpected error - $($_.Exception.Message)" -ForegroundColor Red
        $testResults += @{test = "Test 7"; result = "FAIL"; message = "Unexpected error" }
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

Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan

# Schema Reference
Write-Host "`nSCHEMA REFERENCE" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host @"
UserCreate Schema (Request Body):
{
  "email": "string (required, must be valid email)",
  "password": "string (required)",
  "full_name": "string (required)",
  "affiliation": "string (optional, can be null)",
  "research_interests": ["string", "string"] (optional, can be empty array)
}

UserResponse Schema (Response Body):
{
  "id": "string (MongoDB ObjectId as string)",
  "email": "string",
  "full_name": "string",
  "affiliation": "string or null",
  "research_interests": ["string"],
  "created_at": "datetime (ISO 8601 format)"
}

Status Codes:
- 200: Successful registration
- 400: Email already registered or validation error
- 422: Unprocessable Entity (invalid JSON schema)
- 500: Internal server error
"@ -ForegroundColor White

Write-Host ("=" * 80) -ForegroundColor Cyan
