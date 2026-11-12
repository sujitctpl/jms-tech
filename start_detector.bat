@echo off
title Showroom CCTV AI Detector
echo ========================================================
echo   Capermint Showroom Edge AI Detector
echo   Loading active configuration from camera_config.json...
echo ========================================================
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
python local_detector.py %*
if errorlevel 1 (
    echo.
    echo [!] Detector stopped or encountered an error.
    pause
)

