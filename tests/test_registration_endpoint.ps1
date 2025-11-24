#!/usr/bin/env powershell
# Test registration from frontend

$baseURL = "http://localhost:8000"
$email = "frontend_test_$(Get-Random)@test.com"
$password = "TestPass123!"
$fullName = "Frontend Test User"
$affiliation = "Test Uni"

$payload = @{
    email = $email
    password = $password
    full_name = $fullName
    affiliation = $affiliation
} | ConvertTo-Json

Write-Host "Testing Registration Endpoint"
Write-Host "=============================="
Write-Host "URL: $baseURL/register"
Write-Host "Payload: $payload"
Write-Host ""

$response = Invoke-WebRequest -Uri "$baseURL/register" `
    -Method POST `
    -Headers @{"Content-Type" = "application/json"} `
    -Body $payload `
    -UseBasicParsing

Write-Host "Status Code: $($response.StatusCode)"
Write-Host "Response Body:"
$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
