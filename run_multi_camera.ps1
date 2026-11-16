<#
.SYNOPSIS
    Multi-Camera Showroom Edge AI Launcher for PowerShell
#>
param(
    [string[]]$Channels = @("101", "201")
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Capermint Showroom Edge AI - Multi-Camera Launcher" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

foreach ($ch in $Channels) {
    Write-Host "[*] Launching Camera Channel $ch on GPU..." -ForegroundColor Green
    Start-Process cmd.exe -ArgumentList "/k", "cd /d `"$scriptDir`" && call start_detector.bat --channel $ch --imgsz 640"
    Start-Sleep -Seconds 3
}

Write-Host "`n[+] Launched $($Channels.Count) camera window(s)." -ForegroundColor Yellow
