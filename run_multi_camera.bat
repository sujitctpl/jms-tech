@echo off
cd /d "%~dp0"
title Multi-Camera Showroom Edge Detectors
echo ========================================================
echo   Capermint Showroom Edge AI - Multi-Camera Launcher
echo ========================================================
echo.

if "%~1"=="" goto prompt
if "%~1"=="all" goto launch_all
if "%~1"=="both" goto launch_two

:custom_args
for %%c in (%*) do (
    echo [*] Starting GPU Edge Detector for Channel %%c...
    start "Showroom AI - Cam %%c" /d "%~dp0" cmd /k "call start_detector.bat --channel %%c --imgsz 640"
    ping -n 3 127.0.0.1 >nul
)
goto end

:prompt
echo Select camera configuration to launch:
echo   [1] Launch Camera 101 (Entrance) and Camera 201 (Floor)  [Recommended for 4GB GPU]
echo   [2] Launch All 4 Cameras (101, 201, 301, 401)
echo   [3] Exit
echo.
set /p opt="Enter choice [1, 2, or 3]: "

if "%opt%"=="1" goto launch_two
if "%opt%"=="2" goto launch_all
goto end

:launch_two
echo [*] Launching Channel 101 (Main Glass Entrance)...
start "Showroom AI - Cam 101 (Entrance)" /d "%~dp0" cmd /k "call start_detector.bat --channel 101 --imgsz 640"
ping -n 3 127.0.0.1 >nul

echo [*] Launching Channel 201 (Floor / Display Area)...
start "Showroom AI - Cam 201 (Floor)" /d "%~dp0" cmd /k "call start_detector.bat --channel 201 --imgsz 640"
echo.
echo [!] Both camera detectors are now running in separate windows.
goto end

:launch_all
echo [*] Launching Channel 101 (Entrance)...
start "Showroom AI - Cam 101" /d "%~dp0" cmd /k "call start_detector.bat --channel 101 --imgsz 640"
ping -n 3 127.0.0.1 >nul

echo [*] Launching Channel 201 (Floor)...
start "Showroom AI - Cam 201" /d "%~dp0" cmd /k "call start_detector.bat --channel 201 --imgsz 640"
ping -n 3 127.0.0.1 >nul

echo [*] Launching Channel 301 (Lounge/Billing)...
start "Showroom AI - Cam 301" /d "%~dp0" cmd /k "call start_detector.bat --channel 301 --imgsz 640"
ping -n 3 127.0.0.1 >nul

echo [*] Launching Channel 401 (Accounts/Admin)...
start "Showroom AI - Cam 401" /d "%~dp0" cmd /k "call start_detector.bat --channel 401 --imgsz 640"
echo.
echo [!] All 4 camera detectors are running in separate windows.
goto end

:end
echo.
