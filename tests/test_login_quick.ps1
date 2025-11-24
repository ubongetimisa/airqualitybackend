#!/usr/bin/env pwsh
<#
Quick login test with token usage
#>

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "QUICK LOGIN TEST" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$baseUri = "http://localhost:8000"

# Use registered user credentials
$email = "grace.researcher@example.edu"
$password = "S3cur3P@ssw0rd!"

Write-Host "`nAttempting login..." -ForegroundColor Yellow
Write-Host "Email: $email" -ForegroundColor Gray

try {
    # Login and get token
    $body = @{
        username = $email
        password = $password
    } | ConvertTo-Json

    $response = Invoke-RestMethod -Uri "$baseUri/token" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $body
    
    Write-Host "`n✅ LOGIN SUCCESSFUL!" -ForegroundColor Green
    Write-Host "`nResponse Details:" -ForegroundColor Cyan
    Write-Host "  Token Type: $($response.token_type)" -ForegroundColor Green
    Write-Host "  User ID: $($response.user_id)" -ForegroundColor Green
    Write-Host "  User Email: $($response.user_email)" -ForegroundColor Green
    Write-Host "  Access Token: $($response.access_token.Substring(0, 50))..." -ForegroundColor Green
    
    $token = $response.access_token
    
    # Now test using the token to access a protected endpoint
    Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
    Write-Host "TESTING PROTECTED ENDPOINT WITH TOKEN" -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Cyan
    
    Write-Host "`nAttempting to access /users/me with token..." -ForegroundColor Yellow
    
    try {
        $userResponse = Invoke-RestMethod -Uri "$baseUri/users/me" `
            -Method GET `
            -Headers @{
                "Authorization" = "Bearer $token"
                "Content-Type" = "application/json"
            }
        
        Write-Host "`n✅ PROTECTED ENDPOINT ACCESS SUCCESSFUL!" -ForegroundColor Green
        Write-Host "`nUser Details:" -ForegroundColor Cyan
        Write-Host $userResponse | ConvertTo-Json
        
    } catch {
        Write-Host "`n❌ Failed to access protected endpoint" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    }
    
} catch {
    Write-Host "`n❌ LOGIN FAILED" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    
    if ($_.Exception.Response) {
        Write-Host "`nStatus Code: $($_.Exception.Response.StatusCode)" -ForegroundColor Red
        $stream = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        $errorBody = $reader.ReadToEnd()
        Write-Host "Response: $errorBody" -ForegroundColor Red
    }
}

Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
