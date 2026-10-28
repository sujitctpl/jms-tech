# Stop all processes on ports 8000, 5000, and 5173

$ports = @(8000, 5000, 5173)

foreach ($port in $ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connections) {
        $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $pids) {
            try {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                Write-Host "[+] Stopped process $procId on port $port" -ForegroundColor Green
            } catch {
                Write-Host "[!] Could not stop process $procId on port $port" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "[*] Port $port is already free." -ForegroundColor Gray
    }
}

Write-Host "`n[DONE] All Showroom Intelligence background ports are cleared." -ForegroundColor Cyan
