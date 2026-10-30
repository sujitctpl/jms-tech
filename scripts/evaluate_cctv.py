"""
Showroom CCTV Face Recognition Accuracy Evaluation Runner.
Loads labeled showroom CCTV probe datasets, benchmarks performance per camera ID
and across overhead pitch/lighting tiers, and outputs comprehensive accuracy reports.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.db.session import SessionLocal
from app.db.models.employee import Employee
from app.vision.cctv_evaluator import ShowroomCCTVEvaluator


def load_gallery_embeddings():
    session = SessionLocal()
    gallery = {}
    try:
        employees = session.query(Employee).filter(Employee.is_active.is_(True)).all()
        for emp in employees:
            embs = emp.get_gallery_embeddings()
            if embs:
                gallery[emp.employee_id] = embs
    finally:
        session.close()
    return gallery


def main():
    parser = argparse.ArgumentParser(description="Evaluate Showroom CCTV Recognition Accuracy")
    parser.add_argument("--manifest", type=str, default="data/cctv_eval/manifest.json", help="Path to manifest JSON/CSV")
    parser.add_argument("--base-dir", type=str, default="data/cctv_eval", help="Base directory for probe images")
    parser.add_argument("--threshold", type=float, default=settings.FACE_SIMILARITY_THRESHOLD, help="Cosine threshold")
    parser.add_argument("--margin", type=float, default=settings.FACE_MATCH_MARGIN, help="Top1 vs Top2 margin")
    args = parser.parse_args()

    print("=" * 80)
    print("SHOWROOM CCTV FACE RECOGNITION ACCURACY BENCHMARK")
    print("=" * 80)
    print(f"Operating Parameters:")
    print(f"  Similarity Threshold : {args.threshold}")
    print(f"  Match Margin (Top1-2): {args.margin}")
    print(f"  Manifest Path        : {args.manifest}")

    evaluator = ShowroomCCTVEvaluator(
        similarity_threshold=args.threshold,
        match_margin=args.margin,
    )

    gallery = load_gallery_embeddings()
    print(f"\n1. Loaded Reference Gallery:")
    print(f"   Active Employees   : {len(gallery)}")
    print(f"   Gallery Embeddings : {sum(len(v) for v in gallery.values())}")
    evaluator.enroll_gallery(gallery)

    if not os.path.exists(args.manifest):
        print(f"\n[STATUS] Labeled evaluation manifest not found at '{args.manifest}'.")
        print("\nDATASET MANIFEST FORMAT SPECIFICATION:")
        print("To evaluate on physical showroom CCTV footage, place clips/crops into data/cctv_eval/ and provide:")
        print("manifest.json format:")
        print("""[
  {
    "sample_id": "cctv_cam01_f00123",
    "employee_id": "EMP-101",
    "camera_id": "cam_entrance_01",
    "image_path": "cam01/f00123.jpg",
    "timestamp": "2026-10-30T10:15:30Z",
    "pitch": -22.5,
    "yaw": 12.0,
    "lighting_lux": 420.0,
    "distance_meters": 2.8
  }
]""")
        print("\nNo artificial evaluation metrics will be fabricated without labeled ground-truth files.")
        print("=" * 80)
        return

    # Load and run evaluation
    samples = evaluator.load_manifest(args.manifest)
    print(f"\n2. Loaded {len(samples)} Labeled CCTV Probes.")

    cam_summaries, overall, probe_results = evaluator.evaluate_probes(samples, base_dir=args.base_dir)

    print("\n" + "=" * 80)
    print("PER-CAMERA EVALUATION RESULTS")
    print("=" * 80)
    headers = ["Camera ID", "Probes", "Correct", "False ID", "False Rej", "Qual Fails", "Rank-1 Acc (%)", "Mean Sim"]
    row_fmt = "{:<18} | {:>6} | {:>7} | {:>8} | {:>9} | {:>10} | {:>14} | {:>8}"
    print(row_fmt.format(*headers))
    print("-" * 80)

    for cam_id, s in cam_summaries.items():
        print(
            row_fmt.format(
                cam_id,
                s.total_probes,
                s.correct_identifications,
                s.false_identifications,
                s.false_rejections,
                s.probes_rejected_quality,
                f"{s.rank1_accuracy * 100:.2f}%",
                f"{s.mean_similarity:.3f}",
            )
        )

    print("=" * 80)
    print("OVERALL SHOWROOM CCTV SUMMARY")
    print("=" * 80)
    print(f"Total Probe Images       : {overall.total_probes}")
    print(f"Correct Identifications  : {overall.correct_identifications}")
    print(f"False Identifications    : {overall.false_identifications}")
    print(f"False Rejections         : {overall.false_rejections}")
    print(f"Quality Filter Rejections: {overall.probes_rejected_quality}")
    print(f"Overall Rank-1 Accuracy  : {overall.rank1_accuracy * 100:.2f}%")
    print(f"Mean Similarity Score    : {overall.mean_similarity:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

