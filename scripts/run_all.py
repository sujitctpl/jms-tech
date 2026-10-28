"""
Run all Showroom Intelligence services in a SINGLE terminal window:
1. FastAPI Backend (port 8000)
2. Express Auth Server (port 5000)
3. Vite Frontend Dashboard (port 5173)

Press Ctrl+C to stop all services simultaneously.
"""

import os
import signal
import subprocess
import sys
import threading
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(ROOT_DIR, "dashboard")

# Find virtual environment python / uvicorn
VENV_PYTHON = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

processes = []
stop_event = threading.Event()


def stream_logs(pipe, prefix, color_code):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            clean_line = line.rstrip()
            if clean_line:
                # ANSI colors for readable terminal logs
                sys.stdout.write(f"\033[{color_code}m[{prefix}]\033[0m {clean_line}\n")
                sys.stdout.flush()
    except Exception:
        pass


def start_service(name, cmd, cwd, color_code):
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=True,
        )
        processes.append((name, proc))
        t = threading.Thread(target=stream_logs, args=(proc.stdout, name, color_code), daemon=True)
        t.start()
        return proc
    except Exception as e:
        print(f"\033[31m[ERROR] Failed to start {name}: {e}\033[0m")
        return None


def cleanup():
    print("\n\033[33m[*] Stopping all Showroom Intelligence services...\033[0m")
    for name, proc in processes:
        try:
            if proc.poll() is None:
                # On Windows, taskkill /T /F terminates the process tree (including npm/node/uvicorn children)
                subprocess.call(f"taskkill /F /T /PID {proc.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[+] Stopped {name}")
        except Exception:
            pass
    print("\033[32m[+] All services stopped successfully.\033[0m")


def main():
    print("=" * 70)
    print("  CAPERMINT SHOWROOM INTELLIGENCE - ALL-IN-ONE SERVICE RUNNER")
    print("=" * 70)
    print("  - Backend API Docs : http://localhost:8000/docs")
    print("  - Auth API Health  : http://localhost:5000/health")
    print("  - Web Dashboard    : http://localhost:5173")
    print("=" * 70)
    print("  Press Ctrl+C at any time to cleanly stop all services.\n")

    # 1. FastAPI Backend (Cyan: 36)
    backend_cmd = f'"{VENV_PYTHON}" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload'
    start_service("FASTAPI", backend_cmd, ROOT_DIR, "36")

    # 2. Auth Server (Magenta: 35)
    start_service("AUTH", "npm run server", DASHBOARD_DIR, "35")

    # 3. Vite Dashboard (Green: 32)
    start_service("VITE", "npm run dev", DASHBOARD_DIR, "32")

    try:
        while True:
            time.sleep(0.5)
            # Check if any process terminated unexpectedly
            for name, proc in processes:
                if proc.poll() is not None and not stop_event.is_set():
                    # Process died
                    pass
    except KeyboardInterrupt:
        stop_event.set()
        cleanup()


if __name__ == "__main__":
    main()
