# PowerShell Service Launcher for Showroom Intelligence

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  STARTING ALL SHOWROOM INTELLIGENCE SERVICES" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
Set-Location $rootDir

Write-Host "[1/3] Starting FastAPI Backend (Port 8000)..." -ForegroundColor Green
Start-Process cmd.exe -ArgumentList '/k', "call `"$rootDir\.venv\Scripts\activate.bat`" && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload" -WorkingDirectory $rootDir

Write-Host "[2/3] Starting Auth Server (Port 5000)..." -ForegroundColor Green
Start-Process cmd.exe -ArgumentList '/k', "npm run server" -WorkingDirectory "$rootDir\dashboard"

Write-Host "[3/3] Starting Frontend Dashboard (Port 5173)..." -ForegroundColor Green
Start-Process cmd.exe -ArgumentList '/k', "npm run dev" -WorkingDirectory "$rootDir\dashboard"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  ALL 3 SERVICES LAUNCHED SUCCESSFULLY!" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  - Backend API Docs : http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "  - Auth API Health  : http://localhost:5000/health" -ForegroundColor Yellow
Write-Host "  - Web Dashboard    : http://localhost:5173" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Check the 3 opened terminal windows for real-time service logs."
Write-Host "  To stop all 3 services at any time, run: .\stop_services.bat"
Write-Host "======================================================================" -ForegroundColor Cyan
