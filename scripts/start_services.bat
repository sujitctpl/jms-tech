@echo off
title Capermint Showroom Intelligence - Service Launcher
cls
echo ======================================================================
echo   STARTING ALL SHOWROOM INTELLIGENCE SERVICES
echo ======================================================================
echo.

cd /d "%~dp0\.."

echo [1/3] Starting FastAPI Backend (Port 8000)...
if exist ".venv\Scripts\uvicorn.exe" (
    start "JMS - FastAPI Backend (8000)" cmd /k "call .\.venv\Scripts\activate.bat && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
) else (
    start "JMS - FastAPI Backend (8000)" cmd /k "uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
)

echo [2/3] Starting Auth Server (Port 5000)...
start "JMS - Auth Server (5000)" cmd /k "cd dashboard && npm run server"

echo [3/3] Starting Frontend Dashboard (Port 5173)...
start "JMS - Web Dashboard (5173)" cmd /k "cd dashboard && npm run dev"

echo.
echo ======================================================================
echo   ALL 3 SERVICES LAUNCHED IN SEPARATE WINDOWS!
echo ======================================================================
echo   - Backend API Docs : http://localhost:8000/docs
echo   - Auth API Health  : http://localhost:5000/health
echo   - Web Dashboard    : http://localhost:5173
echo.
echo   Check the 3 opened terminal windows for real-time service logs.
echo   To stop all 3 services at any time, run: .\stop_services.bat
echo ======================================================================
echo.
pause
