"""Snap an employee portrait from webcam and enroll their face into pgvector.

Usage:
    python scripts/enroll_employee_webcam.py --id EMP-101 --name "Pooja Patel" --dept "Sales"
"""

import argparse
import os
import sys
import cv2

# Ensure app is discoverable on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.enroll_employee_image import enroll_employee_photo


def capture_and_enroll(employee_id: str, name: str = None, department: str = "General", designation: str = "Staff", camera_idx: int = 0):
    print(f"[*] Opening webcam {camera_idx}...")
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[-] Error: Could not access webcam at index {camera_idx}.")
        return False

    window_name = f"Enroll Face: {name or employee_id} - Press [SPACE] to Capture, [ESC] to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("[*] Controls:")
    print("    [SPACEBAR] -> Capture photo and enroll")
    print("    [ESC]      -> Exit without saving")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        display = frame.copy()
        h, w = display.shape[:2]

        # Draw alignment oval guide
        center = (w // 2, h // 2)
        axes = (int(w * 0.18), int(h * 0.28))
        cv2.ellipse(display, center, axes, 0, 0, 360, (0, 255, 0), 2)
        cv2.putText(
            display,
            "Align face inside oval and press SPACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            captured_frame = frame
            break
        elif key == 27:  # ESC
            print("[*] Enrollment cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        save_path = f"employee_{employee_id}_portrait.jpg"
        cv2.imwrite(save_path, captured_frame)
        print(f"[+] Saved captured portrait to {save_path}")
        success = enroll_employee_photo(
            employee_id=employee_id,
            image_path=save_path,
            name=name,
            department=department,
            designation=designation,
        )
        return success
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture face from webcam and enroll into pgvector.")
    parser.add_argument("--id", required=True, help="Employee ID (e.g. EMP-101)")
    parser.add_argument("--name", default=None, help="Employee Full Name")
    parser.add_argument("--dept", default="General", help="Department")
    parser.add_argument("--designation", default="Staff", help="Designation")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index (default: 0)")

    args = parser.parse_args()
    capture_and_enroll(
        employee_id=args.id,
        name=args.name,
        department=args.dept,
        designation=args.designation,
        camera_idx=args.cam,
    )

