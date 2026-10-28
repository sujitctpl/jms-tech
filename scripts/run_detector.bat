@echo off
title Capermint Showroom Intelligence - Local AI Edge Detector
cls
cd /d "%~dp0\.."

echo ======================================================================
echo   CAPERMINT SHOWROOM INTELLIGENCE - LOCAL AI DETECTOR
echo ======================================================================
echo.
echo Starting local YOLO person detection and tracking engine...
echo Video and detections are processed locally on this machine.
echo Data and visitor counts are synchronized with the Dashboard.
echo.

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe local_detector.py %*
) else if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe local_detector.py %*
) else (
    python local_detector.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Detector exited with code %ERRORLEVEL%.
    pause
)
