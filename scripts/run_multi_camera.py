"""
Multi-Camera Launcher for Capermint Showroom Intelligence.
Runs multiple or all camera channels simultaneously in parallel.

Usage:
  python scripts/run_multi_camera.py              # Runs all 4 cameras (101, 201, 301, 401)
  python scripts/run_multi_camera.py 101 201      # Runs specific cameras
  python scripts/run_multi_camera.py --headless   # Runs all cameras without GUI windows
"""

import json
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

CONFIG_PATH = os.path.join(ROOT_DIR, "camera_config.json")


def load_channels():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                cams = data.get("cameras", {})
                active = [ch for ch, cfg in cams.items() if cfg.get("is_active", True)]
                if active:
                    return sorted(active)
        except Exception:
            pass
    return ["101", "201", "301", "401"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]

    channels = args if args else load_channels()
    flag_str = " ".join(flags)

    print("=" * 70)
    print("  CAPERMINT SHOWROOM INTELLIGENCE - MULTI-CAMERA LAUNCHER")
    print("=" * 70)
    print(f"  Target Channels : {', '.join(channels)}")
    print(f"  Python Bin      : {VENV_PYTHON}")
    print(f"  Extra Flags     : {flag_str if flag_str else 'None'}")
    print("=" * 70)

    for ch in channels:
        window_title = f"JMS Detector - Channel {ch}"
        cmd = f'start "{window_title}" cmd /k "call "{ROOT_DIR}\\.venv\\Scripts\\activate.bat" && python local_detector.py {ch} {flag_str}"'
        print(f"[*] Launching Channel {ch} in separate window...")
        subprocess.Popen(cmd, cwd=ROOT_DIR, shell=True)
        time.sleep(1.0)  # Stagger launch slightly to avoid RTSP connection spikes

    print("\n[+] All target cameras launched successfully!")


if __name__ == "__main__":
    main()
