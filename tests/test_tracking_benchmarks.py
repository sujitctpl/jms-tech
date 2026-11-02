import math
import os
import cv2
import numpy as np
import pytest
import yaml

from local_detector import RobustFallbackTracker, TrackSmoother, solve_linear_assignment


def test_botsort_yaml_configuration():
    """Verifies that botsort.yaml exists and is valid BoT-SORT configuration."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "botsort.yaml")
    assert os.path.exists(cfg_path), "botsort.yaml must exist in project root"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("tracker_type") == "botsort"
    assert "gmc_method" in cfg
    assert cfg.get("track_buffer", 0) >= 30
    assert "match_thresh" in cfg


def test_robust_fallback_tracker_crossing_paths_no_id_switches():
    """
    Simulates two persons walking toward each other, crossing paths in the center,
    and walking away.
    Verifies that RobustFallbackTracker uses velocity projection and appearance
    to avoid identity swaps (ID switches).
    """
    tracker = RobustFallbackTracker(max_disappeared=20, max_distance=120.0)

    # Person 1 (Blue): Starts at x=50, moves +15 px/frame to right
    # Person 2 (Red): Starts at x=350, moves -15 px/frame to left
    # Create synthetic frames with color distinguishing upper body
    frame_h, frame_w = 400, 500

    def make_frame_and_boxes(p1_x: float, p2_x: float):
        frame = np.full((frame_h, frame_w, 3), 50, dtype=np.uint8)
        b1 = (p1_x, 150.0, p1_x + 50.0, 300.0)
        b2 = (p2_x, 150.0, p2_x + 50.0, 300.0)

        # Color Person 1 in blue
        x1_i, x2_i = int(b1[0]), int(b1[2])
        cv2.rectangle(frame, (x1_i, 150), (x2_i, 300), (230, 80, 20), -1)

        # Color Person 2 in yellow/red
        x1_j, x2_j = int(b2[0]), int(b2[2])
        cv2.rectangle(frame, (x1_j, 150), (x2_j, 300), (30, 210, 240), -1)

        return frame, [b1, b2]

    # Frame 1: Initial detection
    f1, boxes1 = make_frame_and_boxes(50.0, 350.0)
    tracked1 = tracker.update(boxes1, frame=f1)
    assert len(tracked1) == 2
    # Find which ID belongs to Person 1 (left) and Person 2 (right)
    id_p1 = [tid for tid, box in tracked1.items() if box[0] < 200.0][0]
    id_p2 = [tid for tid, box in tracked1.items() if box[0] > 200.0][0]
    assert id_p1 != id_p2

    # Run for 15 frames as they approach and cross
    for step in range(1, 16):
        cur_p1_x = 50.0 + step * 15.0
        cur_p2_x = 350.0 - step * 15.0
        frame, boxes = make_frame_and_boxes(cur_p1_x, cur_p2_x)
        tracked = tracker.update(boxes, frame=frame)
        assert len(tracked) == 2

        # Check identities after crossing (step > 11, cur_p1_x > cur_p2_x)
        if step > 11:
            # Person 1 is now on the right (x > 215)
            # Person 2 is now on the left (x < 185)
            right_id = [tid for tid, box in tracked.items() if box[0] > 200.0][0]
            left_id = [tid for tid, box in tracked.items() if box[0] < 200.0][0]

            # Critical check: Identity MUST NOT swap!
            assert right_id == id_p1, f"Person 1 swapped identity at step {step}"
            assert left_id == id_p2, f"Person 2 swapped identity at step {step}"


def test_velocity_projection_prediction():
    """Verifies that predict_box correctly extrapolates position using learned velocity."""
    tracker = RobustFallbackTracker()

    # Step 1: Object at (100, 100, 150, 200)
    b1 = (100.0, 100.0, 150.0, 200.0)
    tracker.update([b1])
    tid = list(tracker.objects.keys())[0]

    # Step 2: Object moves to (110, 105, 160, 205) -> velocity is dx=+10, dy=+5
    b2 = (110.0, 105.0, 160.0, 205.0)
    tracker.update([b2])

    pred = tracker.predict_box(tid)
    # Predicted x should be greater than 110 (extrapolating rightward)
    assert pred[0] > 110.0
    assert pred[1] > 105.0


def test_appearance_feature_extraction():
    """Verifies appearance signature extraction and cosine similarity distance."""
    tracker = RobustFallbackTracker()

    img1 = np.zeros((200, 100, 3), dtype=np.uint8)
    img1[:100, :] = (255, 0, 0)  # Blue upper body
    img1[100:, :] = (0, 0, 0)

    img2 = np.zeros((200, 100, 3), dtype=np.uint8)
    img2[:100, :] = (0, 0, 255)  # Red upper body
    img2[100:, :] = (0, 0, 0)

    emb1 = tracker.extract_appearance(img1, (0.0, 0.0, 100.0, 200.0))
    emb2 = tracker.extract_appearance(img2, (0.0, 0.0, 100.0, 200.0))
    emb1_same = tracker.extract_appearance(img1, (0.0, 0.0, 100.0, 200.0))

    assert emb1 is not None and emb2 is not None
    assert emb1.shape == (32,)

    # Same image must have cosine similarity close to 1.0
    sim_same = float(np.dot(emb1, emb1_same))
    assert sim_same > 0.98

    # Completely different colors must have lower similarity
    sim_diff = float(np.dot(emb1, emb2))
    assert sim_diff < sim_same


def test_linear_assignment_with_gating():
    """Verifies solve_linear_assignment handles valid matches and rejects above threshold."""
    cost_matrix = np.array([
        [0.10, 0.90],
        [0.85, 0.15],
    ], dtype=np.float32)

    matches, unmatched_a, unmatched_b = solve_linear_assignment(cost_matrix, thresh=0.60)
    assert matches == [(0, 0), (1, 1)]
    assert unmatched_a == []
    assert unmatched_b == []

    # High cost exceeding threshold
    cost_matrix_high = np.array([
        [0.80, 0.95],
        [0.90, 0.85],
    ], dtype=np.float32)

    matches_gated, unmatched_a_gated, unmatched_b_gated = solve_linear_assignment(cost_matrix_high, thresh=0.60)
    assert matches_gated == []
    assert len(unmatched_a_gated) == 2
    assert len(unmatched_b_gated) == 2


def test_smoother_velocity_and_coasting():
    """Verifies that TrackSmoother calculates velocities and persists tracks across drops."""
    smoother = TrackSmoother(alpha=0.70, deadband_px=3.5, body_persistence=10, enabled=True)

    # Frame 1: Person at (100, 100, 150, 250)
    tracked_f1 = {1: (100.0, 100.0, 150.0, 250.0)}
    res1 = smoother.smooth_body_boxes(tracked_f1)
    assert 1 in res1

    # Frame 2: Person moves to (110, 100, 160, 250)
    tracked_f2 = {1: (110.0, 100.0, 160.0, 250.0)}
    res2 = smoother.smooth_body_boxes(tracked_f2)
    assert 1 in res2
    # Verify velocity was captured
    vx, vy = smoother.velocities.get(1, (0.0, 0.0))
    assert vx > 0.0, "Smoother should estimate positive X velocity"

    # Frame 3: Person temporarily dropped from tracker -> coasting kicks in
    tracked_f3 = {}
    res3 = smoother.smooth_body_boxes(tracked_f3)
    assert 1 in res3, "Smoother must coast track during brief drop"

    # Frame 4: Detections recover missing track
    unmatched = [(130.0, 100.0, 180.0, 250.0)]
    recovered = smoother.recover_missing_tracks(unmatched)
    assert 1 in recovered, "Smoother should re-associate detection using projected velocity"

