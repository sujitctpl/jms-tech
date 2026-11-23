import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.abspath("."))
from app.vision.body_embedder import BodyEmbedder
from app.vision.face_embedder import FaceEmbedder

be = BodyEmbedder()
fe = FaceEmbedder()

# Load raw pairs from static/uploads/visitors/
pairs = [
    ("VIS-2026-1044 (Tan shirt - TRUE MATCH)", "static/uploads/visitors/VIS-2026-1044_1795415033.jpg", "static/uploads/visitors/VIS-2026-1044_exit_1795415360.jpg", True),
    ("VIS-2026-1048 (Woman vs Plaid Man - FALSE MATCH)", "static/uploads/visitors/VIS-2026-1048_1795415210.jpg", "static/uploads/visitors/VIS-2026-1048_exit_1795415358.jpg", False),
    ("VIS-2026-1049 (Maroon polo - TRUE MATCH)", "static/uploads/visitors/VIS-2026-1049_1795415220.jpg", "static/uploads/visitors/VIS-2026-1049_exit_1795415239.jpg", True),
    ("VIS-2026-1045 (Maroon shirt vs Black T-shirt - FALSE MATCH)", "static/uploads/visitors/VIS-2026-1045_1795415038.jpg", "static/uploads/visitors/VIS-2026-1045_exit_1795415397.jpg", False),
]

for label, p1, p2, expected in pairs:
    im1 = cv2.imread(p1)
    im2 = cv2.imread(p2)
    un1 = be.unpad_canvas(im1)
    un2 = be.unpad_canvas(im2)
    t1 = be.estimate_clothing_appearance(im1)
    t2 = be.estimate_clothing_appearance(im2)
    reid_s, cloth_s, score, is_m = be.verify_appearance(im1, im2)
    print(f"\n[{label}]")
    print(f"  Shapes unpadded: im1={un1.shape if un1 is not None else None}, im2={un2.shape if un2 is not None else None}")
    print(f"  Aspect ratios (H/W): im1={un1.shape[0]/un1.shape[1]:.2f}, im2={un2.shape[0]/un2.shape[1]:.2f}")
    print(f"  Tags: im1='{t1}', im2='{t2}'")
    print(f"  ReID: {reid_s:.3f}, Clothing: {cloth_s:.3f}, Score: {score:.3f}, is_match={is_m} (Expected match: {expected})")
