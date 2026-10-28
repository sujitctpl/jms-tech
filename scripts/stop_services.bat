@echo off
title Stop Showroom Intelligence Services
cls
echo ======================================================================
echo   STOPPING SHOWROOM INTELLIGENCE SERVICES (Ports 8000, 5000, 5173)
echo ======================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_services.ps1"

echo.
pause
