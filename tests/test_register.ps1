#!/usr/bin/env pwsh
<#
PowerShell script to test the /register endpoint
#>

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Testing /register Endpoint" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$uri = "http://localhost:8000/register"

# Create the request body
$body = @{
    email = "alice.researcher@example.edu"
    password = "S3cur3P@ssw0rd!"
    full_name = "Alice M. Researcher"
    affiliation = "Institute of Atmospheric Studies"
    research_interests = @("Air Quality", "Epidemiology", "Time Series Analysis")
} | ConvertTo-Json

Write-Host "`nRequest URI: $uri" -ForegroundColor Yellow
Write-Host "`nRequest Body:" -ForegroundColor Yellow
Write-Host $body -ForegroundColor White

try {
    Write-Host "`nSending request..." -ForegroundColor Yellow
    
    $response = Invoke-RestMethod -Uri $uri `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $body
    
    Write-Host "`n✅ Success! Response:" -ForegroundColor Green
    Write-Host ($response | ConvertTo-Json -Depth 10)
    
} catch {
    Write-Host "`n❌ Error: " -ForegroundColor Red -NoNewline
    Write-Host $_.Exception.Message -ForegroundColor Red
    
    if ($_.Exception.Response) {
        $stream = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        $errorBody = $reader.ReadToEnd()
        Write-Host "`nError Details:" -ForegroundColor Red
        Write-Host $errorBody -ForegroundColor White
    }
}

Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
