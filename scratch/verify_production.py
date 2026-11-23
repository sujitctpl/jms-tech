import os, sys, cv2
sys.path.insert(0, os.path.abspath("."))
from app.vision.body_embedder import BodyEmbedder

be = BodyEmbedder()

test_cases = [
    # Label, Path1, Path2, Expected Match
    ("VIS-2026-1044 (Tan shirt - Test 1 & 2)", "static/uploads/visitors/VIS-2026-1044_1795415033.jpg", "static/uploads/visitors/VIS-2026-1044_exit_1795415360.jpg", True),
    ("VIS-2026-1048 (Woman vs Plaid Man - Test 3 & 4)", "static/uploads/visitors/VIS-2026-1048_1795415210.jpg", "static/uploads/visitors/VIS-2026-1048_exit_1795415358.jpg", False),
    ("VIS-2026-1049 (Maroon Polo - Test 5 & 6)", "static/uploads/visitors/VIS-2026-1049_1795415220.jpg", "static/uploads/visitors/VIS-2026-1049_exit_1795415239.jpg", True),
    ("VIS-2026-1045 (Maroon Shirt vs Black T-shirt - Test 7 & 8)", "static/uploads/visitors/VIS-2026-1045_1795415038.jpg", "static/uploads/visitors/VIS-2026-1045_exit_1795415397.jpg", False),
    ("VIS-2026-1046 (Maroon Sports vs Woman Brown - Test SS 1 & 2)", "scratch/1046_Entry_(Man_Maroon).jpg", "scratch/1046_Exit_(Woman_Brown).jpg", False),
    ("VIS-2026-1043 (Blue Striped vs Man White - Test SS 3 & 4)", "scratch/1043_Entry_(Man_Blue).jpg", "scratch/1043_Exit_(Man_White).jpg", False),
]

all_passed = True
total = len(test_cases)
correct = 0

print("=" * 70)
print(f"EVALUATING PRODUCTION BodyEmbedder.verify_appearance ON {total} REAL CASES")
print("=" * 70)

for label, p1, p2, expected in test_cases:
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print(f"Skipping {label}: missing files")
        continue
    im1 = cv2.imread(p1)
    im2 = cv2.imread(p2)
    tag1 = be.estimate_clothing_appearance(im1)
    tag2 = be.estimate_clothing_appearance(im2)
    reid_s, cloth_s, score, is_m = be.verify_appearance(im1, im2)
    
    passed = (is_m == expected)
    if passed:
        correct += 1
        status = "PASSED (CORRECT)"
    else:
        all_passed = False
        status = "FAILED (INCORRECT)"
        
    print(f"\n{label}")
    print(f"  Tag 1: '{tag1}' | Tag 2: '{tag2}'")
    print(f"  ReID: {reid_s:.3f} | Clothing Sim: {cloth_s:.3f} | Combined Score: {score:.3f}")
    print(f"  -> Decision: is_match={is_m} | Expected: {expected} => [{status}]")

print("\n" + "=" * 70)
acc = (correct / total) * 100
print(f"FINAL ACCURACY: {correct}/{total} ({acc:.1f}%)")
print("=" * 70)

if not all_passed:
    sys.exit(1)
