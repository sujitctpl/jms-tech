@echo off
title Capermint Showroom Intelligence - Multi-Camera Launcher
cls
cd /d "%~dp0\.."

echo ======================================================================
echo   STARTING ALL 4 CCTV AI DETECTOR CHANNELS
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe scripts\run_multi_camera.py %*
) else (
    python scripts\run_multi_camera.py %*
)

echo.
pause
