"""
Real Showroom CCTV Face Recognition Evaluation Workflow.
Evaluates recognition accuracy across camera IDs, overhead angles, distances,
and lighting conditions with strict gallery/probe separation to prevent data leakage.
"""
from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.vision.face_embedder import FaceEmbedder
from app.vision.vector_store import PgVectorStoreManager


@dataclass
class CCTVSampleMetadata:
    sample_id: str
    employee_id: str  # Ground truth employee ID (or "VISITOR" / "UNKNOWN")
    camera_id: str
    image_path: str
    timestamp: Optional[str] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    lighting_lux: Optional[float] = None
    distance_meters: Optional[float] = None
    occlusion_percent: Optional[float] = None


@dataclass
class CCTVProbeResult:
    sample: CCTVSampleMetadata
    predicted_id: Optional[str]
    similarity: float
    is_correct: bool
    quality_passed: bool
    quality_reason: str


@dataclass
class CameraEvaluationSummary:
    camera_id: str
    total_probes: int
    correct_identifications: int
    false_identifications: int
    false_rejections: int
    rank1_accuracy: float
    mean_similarity: float
    probes_rejected_quality: int


class ShowroomCCTVEvaluator:
    """
    Evaluates ArcFace + pgvector employee recognition against real showroom CCTV probe images.
    Enforces clean separation between gallery/enrollment vectors and CCTV probe frames.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.38,
        match_margin: float = 0.035,
        embedder: Optional[FaceEmbedder] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.match_margin = match_margin
        self.embedder = embedder or FaceEmbedder()
        self.vector_store = PgVectorStoreManager(similarity_threshold=self.similarity_threshold)

    def load_manifest(self, manifest_path: str) -> List[CCTVSampleMetadata]:
        """Loads labeled CCTV probe dataset manifest (JSON or CSV)."""
        samples: List[CCTVSampleMetadata] = []
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        if manifest_path.endswith(".json"):
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx, item in enumerate(data):
                samples.append(
                    CCTVSampleMetadata(
                        sample_id=item.get("sample_id", f"s_{idx}"),
                        employee_id=item["employee_id"],
                        camera_id=item.get("camera_id", "cam_default"),
                        image_path=item["image_path"],
                        timestamp=item.get("timestamp"),
                        yaw=item.get("yaw"),
                        pitch=item.get("pitch"),
                        roll=item.get("roll"),
                        lighting_lux=item.get("lighting_lux"),
                        distance_meters=item.get("distance_meters"),
                        occlusion_percent=item.get("occlusion_percent"),
                    )
                )
        elif manifest_path.endswith(".csv"):
            import csv
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    samples.append(
                        CCTVSampleMetadata(
                            sample_id=row.get("sample_id", f"s_{idx}"),
                            employee_id=row["employee_id"],
                            camera_id=row.get("camera_id", "cam_default"),
                            image_path=row["image_path"],
                            timestamp=row.get("timestamp"),
                            yaw=float(row["yaw"]) if row.get("yaw") else None,
                            pitch=float(row["pitch"]) if row.get("pitch") else None,
                            roll=float(row["roll"]) if row.get("roll") else None,
                            lighting_lux=float(row["lighting_lux"]) if row.get("lighting_lux") else None,
                            distance_meters=float(row["distance_meters"]) if row.get("distance_meters") else None,
                            occlusion_percent=float(row["occlusion_percent"]) if row.get("occlusion_percent") else None,
                        )
                    )
        return samples

    def enroll_gallery(self, gallery_dict: Dict[str, List[np.ndarray]]) -> None:
        """Enrolls reference gallery embeddings into evaluation vector store."""
        self.vector_store.clear()
        for emp_id, vectors in gallery_dict.items():
            for v in vectors:
                self.vector_store.add(emp_id, v, replace=False)

    def evaluate_probes(
        self,
        samples: List[CCTVSampleMetadata],
        base_dir: str = "",
    ) -> Tuple[Dict[str, CameraEvaluationSummary], CameraEvaluationSummary, List[CCTVProbeResult]]:
        """
        Executes recognition on all CCTV probe images and computes camera-specific and overall metrics.
        """
        probe_results: List[CCTVProbeResult] = []

        for sample in samples:
            full_path = os.path.join(base_dir, sample.image_path) if base_dir else sample.image_path
            if not os.path.exists(full_path):
                continue

            img = cv2.imread(full_path)
            if img is None:
                continue

            # Quality validation
            is_valid, reason, cropped, crop_lms, full_lms = self.embedder.validate_enrollment_quality(img)
            if not is_valid or cropped is None:
                probe_results.append(
                    CCTVProbeResult(
                        sample=sample,
                        predicted_id=None,
                        similarity=0.0,
                        is_correct=False,
                        quality_passed=False,
                        quality_reason=reason,
                    )
                )
                continue

            # ArcFace Feature Extraction
            try:
                emb = self.embedder.extract_embedding(cropped, crop_lms, full_image=img, raw_landmarks=full_lms)
            except Exception as e:
                probe_results.append(
                    CCTVProbeResult(
                        sample=sample,
                        predicted_id=None,
                        similarity=0.0,
                        is_correct=False,
                        quality_passed=False,
                        quality_reason=f"embedder_exception: {e}",
                    )
                )
                continue

            # Vector Store Query
            pred_id, sim = self.vector_store.find_best_employee_match(
                emb,
                similarity_threshold=self.similarity_threshold,
                match_margin=self.match_margin,
            )

            is_correct = (pred_id == sample.employee_id)
            probe_results.append(
                CCTVProbeResult(
                    sample=sample,
                    predicted_id=pred_id,
                    similarity=sim,
                    is_correct=is_correct,
                    quality_passed=True,
                    quality_reason="ok",
                )
            )

        # Aggregate per Camera
        by_cam: Dict[str, List[CCTVProbeResult]] = {}
        for res in probe_results:
            by_cam.setdefault(res.sample.camera_id, []).append(res)

        cam_summaries: Dict[str, CameraEvaluationSummary] = {}
        for cam_id, res_list in by_cam.items():
            cam_summaries[cam_id] = self._summarize_results(cam_id, res_list)

        overall_summary = self._summarize_results("OVERALL_SHOWROOM", probe_results)
        return cam_summaries, overall_summary, probe_results

    @staticmethod
    def _summarize_results(cam_id: str, results: List[CCTVProbeResult]) -> CameraEvaluationSummary:
        total = len(results)
        if total == 0:
            return CameraEvaluationSummary(
                camera_id=cam_id,
                total_probes=0,
                correct_identifications=0,
                false_identifications=0,
                false_rejections=0,
                rank1_accuracy=0.0,
                mean_similarity=0.0,
                probes_rejected_quality=0,
            )

        correct = sum(1 for r in results if r.is_correct)
        quality_fails = sum(1 for r in results if not r.quality_passed)
        false_ids = sum(1 for r in results if r.quality_passed and r.predicted_id is not None and not r.is_correct)
        false_rejs = sum(1 for r in results if r.quality_passed and r.predicted_id is None)
        mean_sim = float(np.mean([r.similarity for r in results if r.quality_passed])) if (total - quality_fails) > 0 else 0.0

        return CameraEvaluationSummary(
            camera_id=cam_id,
            total_probes=total,
            correct_identifications=correct,
            false_identifications=false_ids,
            false_rejections=false_rejs,
            rank1_accuracy=round(correct / total, 4),
            mean_similarity=round(mean_sim, 4),
            probes_rejected_quality=quality_fails,
        )

