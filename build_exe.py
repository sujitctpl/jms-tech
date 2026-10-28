"""
Build Script for compiling local_detector.py into a standalone Windows .exe
using PyInstaller.
"""

import os
import subprocess
import sys


def build():
    print("=" * 70)
    print("  BUILDING STANDALONE EXECUTABLE: CapermintLocalDetector.exe")
    print("=" * 70)

    # Check PyInstaller availability
    python_exe = sys.executable
    try:
        import PyInstaller
        print(f"[+] PyInstaller version {PyInstaller.__version__} detected.")
    except ImportError:
        print("[*] PyInstaller not detected in current environment.")
        print("[*] Attempting to install PyInstaller via pip...")
        res = subprocess.call([python_exe, "-m", "pip", "install", "pyinstaller"])
        if res != 0:
            print("[!] Could not automatically install PyInstaller.")
            print("[*] Please run: pip install pyinstaller")
            print("[*] Then re-run: python build_exe.py")
            return

    model_file = "yolo11n.pt"
    if not os.path.exists(model_file):
        print(f"[!] Warning: Model file '{model_file}' not found in current folder.")

    # PyInstaller command
    # Using --onedir by default because PyTorch/YOLO extract faster on startup
    cmd = [
        python_exe,
        "-m",
        "PyInstaller",
        "--name=CapermintLocalDetector",
        "--noconfirm",
        "--clean",
        "--add-data",
        f"{model_file};.",
        "local_detector.py",
    ]

    print(f"\n[*] Running: {' '.join(cmd)}\n")
    ret = subprocess.call(cmd)

    if ret == 0:
        print("\n" + "=" * 70)
        print("[SUCCESS] Build complete!")
        print("Your executable is located at:")
        print(os.path.abspath(os.path.join("dist", "CapermintLocalDetector", "CapermintLocalDetector.exe")))
        print("=" * 70)
    else:
        print(f"\n[!] Build exited with error code {ret}.")


if __name__ == "__main__":
    build()
