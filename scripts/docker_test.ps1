# Nestling Docker smoke test (Windows PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Invoke-Cleanup {
    Write-Host "==> docker compose down"
    docker compose down
}

try {
    Write-Host "==> docker compose build"
    docker compose build
    if ($LASTEXITCODE -ne 0) { throw "build failed" }

    Write-Host "==> docker compose up -d"
    docker compose up -d
    if ($LASTEXITCODE -ne 0) { throw "up failed" }

    Write-Host "==> waiting for health"
    $ok = $false
    for ($i = 1; $i -le 60; $i++) {
        try {
            $health = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method Get -TimeoutSec 3
            $health | ConvertTo-Json -Compress
            Write-Host ""
            $ok = $true
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $ok) {
        docker compose logs --tail 80
        throw "Health check failed"
    }

    Write-Host "==> create child"
    $childBody = @{ name = "DockerBaby"; sex = "male"; gestational_age_weeks = 32 } | ConvertTo-Json
    $child = Invoke-RestMethod -Uri "http://localhost:8000/api/children" -Method Post `
        -ContentType "application/json" -Body $childBody
    $child | ConvertTo-Json -Compress
    Write-Host ""
    $childId = $child.child_id

    Write-Host "==> growth"
    $growthBody = @{
        child_id = $childId
        sex = "male"
        measure = "weight"
        weeks = 40
        value = 3.2
    } | ConvertTo-Json
    $growth = Invoke-RestMethod -Uri "http://localhost:8000/api/growth" -Method Post `
        -ContentType "application/json" -Body $growthBody
    $growth | ConvertTo-Json -Depth 6 -Compress
    Write-Host ""

    Write-Host "==> docker smoke tests passed"
}
finally {
    Invoke-Cleanup
}
