"""
Model Asset Download and Verification Utility for Showroom Visitor Intelligence.
Ensures YOLO, SCRFD, YuNet, and ArcFace models are available in the models/ directory.
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

MODEL_SOURCES = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "scrfd_10g_bnkps.onnx": "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
    "scrfd_2.5g_bnkps.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps.onnx",
    "w600k_r50.onnx": "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
}


def verify_or_download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("SHOWROOM VISITOR INTELLIGENCE - MODEL ASSET VERIFIER")
    print("=" * 60)

    for filename, url in MODEL_SOURCES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[+] {filename}: PRESENT ({size_mb:.2f} MB)")
        else:
            print(f"[*] {filename}: MISSING. Attempting download from {url}...")
            try:
                urllib.request.urlretrieve(url, dest)
                size_mb = os.path.getsize(dest) / (1024 * 1024)
                print(f"[+] {filename} downloaded successfully ({size_mb:.2f} MB)!")
            except Exception as e:
                print(f"[!] Warning: Could not download {filename}: {e}")


if __name__ == "__main__":
    verify_or_download_models()

