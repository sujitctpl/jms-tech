"""
Threshold Calibration and Biometric Metric Evaluation Utility.
Evaluates face similarity thresholds on genuine and impostor face pairs according to
ISO/IEC 19795 biometric evaluation standards.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ThresholdMetrics:
    threshold: float
    far: float  # False Acceptance Rate (FP / total impostor pairs)
    frr: float  # False Rejection Rate (FN / total genuine pairs)
    tar: float  # True Acceptance Rate (1 - FRR)
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass
class BiometricEvaluationSummary:
    total_genuine_pairs: int
    total_impostor_pairs: int
    eer: float  # Equal Error Rate
    eer_threshold: float
    auc: float  # Area under ROC curve
    recommended_threshold: float
    recommended_metrics: ThresholdMetrics
    metrics_by_threshold: List[ThresholdMetrics]


class ThresholdEvaluator:
    """
    Evaluates ArcFace similarity thresholds across genuine and impostor comparison pairs.
    Computes ISO/IEC biometric performance rates (FAR, FRR, TAR, EER, AUC, Precision, Recall).
    """

    def __init__(self, step: float = 0.02, min_thresh: float = 0.20, max_thresh: float = 0.85):
        self.step = step
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh

    @staticmethod
    def compute_pairs_from_labeled_embeddings(
        labeled_embeddings: Dict[str, List[np.ndarray]],
        max_impostors: int = 50000,
        random_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates genuine and impostor cosine similarity score distributions from
        labeled embeddings: {identity_id: [embedding_1, embedding_2, ...]}.
        """
        rng = np.random.RandomState(random_seed)
        genuine_scores: List[float] = []
        all_items = list(labeled_embeddings.items())

        # 1. Genuine pairs (same identity)
        for _, embs in all_items:
            n = len(embs)
            for i in range(n):
                for j in range(i + 1, n):
                    e1 = embs[i].flatten()
                    e2 = embs[j].flatten()
                    norm1 = np.linalg.norm(e1)
                    norm2 = np.linalg.norm(e2)
                    if norm1 > 0 and norm2 > 0:
                        sim = float(np.dot(e1 / norm1, e2 / norm2))
                        genuine_scores.append(sim)

        # 2. Impostor pairs (different identities)
        impostor_scores: List[float] = []
        m = len(all_items)
        if m >= 2:
            candidate_pairs = []
            for i in range(m):
                id_i, embs_i = all_items[i]
                for j in range(i + 1, m):
                    id_j, embs_j = all_items[j]
                    for e_i in embs_i:
                        for e_j in embs_j:
                            candidate_pairs.append((e_i, e_j))

            if len(candidate_pairs) > max_impostors:
                indices = rng.choice(len(candidate_pairs), size=max_impostors, replace=False)
                sampled_pairs = [candidate_pairs[idx] for idx in indices]
            else:
                sampled_pairs = candidate_pairs

            for e1, e2 in sampled_pairs:
                e1 = e1.flatten()
                e2 = e2.flatten()
                norm1 = np.linalg.norm(e1)
                norm2 = np.linalg.norm(e2)
                if norm1 > 0 and norm2 > 0:
                    sim = float(np.dot(e1 / norm1, e2 / norm2))
                    impostor_scores.append(sim)

        return np.array(genuine_scores, dtype=np.float32), np.array(impostor_scores, dtype=np.float32)

    def evaluate(
        self,
        genuine_scores: Union[List[float], np.ndarray],
        impostor_scores: Union[List[float], np.ndarray],
        target_far: float = 0.01,
    ) -> BiometricEvaluationSummary:
        """
        Evaluates metrics across thresholds from min_thresh to max_thresh.
        """
        gen = np.asarray(genuine_scores, dtype=np.float32)
        imp = np.asarray(impostor_scores, dtype=np.float32)

        total_gen = len(gen)
        total_imp = len(imp)

        if total_gen == 0 or total_imp == 0:
            raise ValueError(f"Insufficient evaluation pairs: {total_gen} genuine, {total_imp} impostor.")

        thresholds = np.arange(self.min_thresh, self.max_thresh + 1e-5, self.step)
        metrics_list: List[ThresholdMetrics] = []

        best_eer_diff = float("inf")
        eer_val = 0.0
        eer_thresh = self.min_thresh

        rec_thresh = 0.38
        rec_metrics: Optional[ThresholdMetrics] = None

        for t in thresholds:
            t = float(round(t, 4))
            tp = int(np.sum(gen >= t))
            fn = int(np.sum(gen < t))
            fp = int(np.sum(imp >= t))
            tn = int(np.sum(imp < t))

            far = fp / total_imp if total_imp > 0 else 0.0
            frr = fn / total_gen if total_gen > 0 else 0.0
            tar = tp / total_gen if total_gen > 0 else 0.0

            prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
            rec = tar
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            m = ThresholdMetrics(
                threshold=t,
                far=round(far, 5),
                frr=round(frr, 5),
                tar=round(tar, 5),
                precision=round(prec, 5),
                recall=round(rec, 5),
                f1_score=round(f1, 5),
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn,
            )
            metrics_list.append(m)

            # Find Equal Error Rate (EER) where |FAR - FRR| is minimal
            diff = abs(far - frr)
            if diff < best_eer_diff:
                best_eer_diff = diff
                eer_val = float((far + frr) / 2.0)
                eer_thresh = t

            # Operating threshold selection: highest F1-score with FAR <= target_far
            if far <= target_far:
                if rec_metrics is None or m.f1_score > rec_metrics.f1_score:
                    rec_metrics = m
                    rec_thresh = t

        if rec_metrics is None:
            # Fallback to EER threshold if no threshold achieved target FAR
            rec_thresh = eer_thresh
            rec_metrics = next(m for m in metrics_list if m.threshold == eer_thresh)

        # Calculate Area under ROC curve (AUC) via trapezoidal numerical integration
        # ROC points: x = FAR, y = TAR (sorted by ascending FAR)
        roc_pts = sorted([(m.far, m.tar) for m in metrics_list], key=lambda p: p[0])
        # Add boundary points (0, 0) and (1, 1)
        roc_pts = [(0.0, 0.0)] + roc_pts + [(1.0, 1.0)]
        # Remove consecutive duplicate x points
        unique_pts = []
        for p in roc_pts:
            if not unique_pts or unique_pts[-1][0] != p[0]:
                unique_pts.append(p)
            else:
                unique_pts[-1] = (p[0], max(unique_pts[-1][1], p[1]))

        x_far = np.asarray([p[0] for p in unique_pts], dtype=np.float32)
        y_tar = np.asarray([p[1] for p in unique_pts], dtype=np.float32)
        if hasattr(np, "trapezoid"):
            auc_val = float(np.trapezoid(y_tar, x_far))
        else:
            auc_val = float(np.sum(0.5 * (y_tar[:-1] + y_tar[1:]) * np.diff(x_far)))
        auc_val = max(0.0, min(1.0, auc_val))

        return BiometricEvaluationSummary(
            total_genuine_pairs=total_gen,
            total_impostor_pairs=total_imp,
            eer=round(eer_val, 4),
            eer_threshold=eer_thresh,
            auc=round(auc_val, 4),
            recommended_threshold=rec_thresh,
            recommended_metrics=rec_metrics,
            metrics_by_threshold=metrics_list,
        )
