import unittest
import math
from typing import Dict, List, Tuple
from local_detector import LocalTracker, TrackSmoother, ZoneManager


class TestByteTrackTrackingAndFallback(unittest.TestCase):
    def setUp(self):
        self.smoother = TrackSmoother(
            alpha=0.7,
            deadband_px=3.5,
            face_persistence=8,
            body_persistence=40,
            enabled=True,
        )
        self.local_tracker = LocalTracker(max_disappeared=25, max_distance=90.0)
        
        # Virtual entrance line from (100, 200) to (500, 200), direction top_to_bottom
        entry_line_cfg = {
            "enabled": True,
            "p1": [100, 200],
            "p2": [500, 200],
            "direction": "top_to_bottom",
        }
        self.zone_manager = ZoneManager(
            config_path="camera_config.json",
            camera_channel="101",
            entry_line_cfg=entry_line_cfg,
            frame_width=640,
            frame_height=480,
        )

    def test_local_tracker_sync_and_no_collision(self):
        """Test that LocalTracker synchronizes with ByteTrack tracks and avoids ID collision."""
        # ByteTrack has tracks 1, 2, 3
        bytetrack_objects = {
            1: (150.0, 100.0, 200.0, 250.0),
            2: (300.0, 120.0, 350.0, 270.0),
            3: (400.0, 110.0, 450.0, 260.0),
        }
        self.local_tracker.sync_active_tracks(bytetrack_objects)
        
        self.assertEqual(len(self.local_tracker.objects), 3)
        self.assertGreater(self.local_tracker.next_id, 3)
        self.assertGreaterEqual(self.local_tracker.next_id, 10001)

        # In fallback mode, a new person appears at (50.0, 50.0, 90.0, 150.0)
        detections = [
            (152.0, 105.0, 202.0, 255.0), # Person 1 moving
            (301.0, 122.0, 351.0, 272.0), # Person 2 moving
            (401.0, 112.0, 451.0, 262.0), # Person 3 moving
            (50.0, 50.0, 90.0, 150.0),    # New Person 4 in fallback mode
        ]
        fallback_tracks = self.local_tracker.update(detections)
        self.assertEqual(len(fallback_tracks), 4)
        # Verify Person 1, 2, 3 retained their IDs
        self.assertIn(1, fallback_tracks)
        self.assertIn(2, fallback_tracks)
        self.assertIn(3, fallback_tracks)
        # Verify the new person received an ID > 3 without colliding with ByteTrack IDs
        new_ids = [k for k in fallback_tracks.keys() if k not in (1, 2, 3)]
        self.assertEqual(len(new_ids), 1)
        self.assertGreaterEqual(new_ids[0], 10001)

    def test_recover_missing_tracks_during_occlusion(self):
        """Test that unmatched detections can recover recently missing tracks before ID switches occur."""
        # Frame 1: Person 1 is actively tracked
        tracked = {1: (200.0, 100.0, 260.0, 250.0)}
        smoothed = self.smoother.smooth_body_boxes(tracked)
        self.assertIn(1, smoothed)

        # Frame 2: Person 1 detection momentarily missed by ByteTrack
        # ByteTrack outputs {} (missed)
        self.smoother.smooth_body_boxes({})
        self.assertEqual(self.smoother.disappeared.get(1), 1)

        # Frame 3: YOLO detects a box at Person 1's position, but ByteTrack didn't assign an ID
        unmatched_boxes = [(202.0, 104.0, 262.0, 254.0)]
        recovered = self.smoother.recover_missing_tracks(unmatched_boxes)
        
        self.assertIn(1, recovered)
        self.assertEqual(self.smoother.disappeared.get(1), 0)

        # Feed back to smoother
        smoothed_recovered = self.smoother.smooth_body_boxes(recovered)
        self.assertIn(1, smoothed_recovered)

    def test_persistent_id_aliasing_prevents_id_switching(self):
        """Test that when ByteTrack re-spawns an occluded track with a new ID, persistent aliasing preserves original ID."""
        # Person 1 is tracked for 5 frames
        for y in range(100, 150, 10):
            self.smoother.smooth_body_boxes({1: (200.0, float(y), 260.0, float(y + 150))})

        # Person 1 is occluded for 3 frames (disappeared count = 3)
        for _ in range(3):
            self.smoother.smooth_body_boxes({})
        self.assertEqual(self.smoother.disappeared.get(1), 3)

        # Person 1 reappears, but ByteTrack re-assigns them a new track ID (ID 88)
        reappeared = {88: (205.0, 160.0, 265.0, 310.0)}
        res_frame1 = self.smoother.smooth_body_boxes(reappeared)
        # Should be remapped to ID 1
        self.assertIn(1, res_frame1)
        self.assertNotIn(88, res_frame1)
        self.assertEqual(self.smoother.id_aliases.get(88), 1)

        # Next frame: ByteTrack CONTINUES outputting ID 88
        next_frame = {88: (207.0, 170.0, 267.0, 320.0)}
        res_frame2 = self.smoother.smooth_body_boxes(next_frame)
        # Persistent aliasing MUST keep it as ID 1 (preventing ID flip back to 88!)
        self.assertIn(1, res_frame2)
        self.assertNotIn(88, res_frame2)

    def test_end_to_end_cctv_simulation_no_duplicate_attendance(self):
        """
        Simulates two overlapping people walking through a CCTV entrance:
        - Person 1 walks across the entry line (top to bottom).
        - Person 2 walks close behind Person 1, gets occluded by Person 1, drops for 3 frames,
          then reappears with a ByteTrack ID switch.
        - Verify total attendance counted is EXACTLY 2 (zero duplicates).
        """
        attendance_events = []

        # Person 1: starts at y=150 (feet at 300), line is at y=200
        # Box format: (x1, y1, x2, y2), feet point is ((x1+x2)/2, y2)
        # Entry line is y = 200, so feet moving from y2=180 to y2=230 crosses the line!

        # 30 frames simulation
        for f in range(30):
            p1_y2 = 150.0 + f * 5.0 # crosses 200 around frame 10
            p1_box = (150.0, p1_y2 - 100.0, 210.0, p1_y2)

            p2_y2 = 130.0 + f * 5.0 # crosses 200 around frame 14
            p2_box = (170.0, p2_y2 - 100.0, 230.0, p2_y2)

            tracked_objects = {}
            unmatched_boxes = []

            # Person 1 is consistently tracked
            tracked_objects[1] = p1_box

            # Person 2:
            # Frames 0-8: tracked as ID 2
            # Frames 9-12: heavily occluded behind Person 1 (missed detection / no ByteTrack ID)
            # Frame 13: YOLO detects box, but ByteTrack hasn't confirmed ID yet (unmatched detection)
            # Frames 14-29: ByteTrack outputs Person 2 under new ID 99 (ID switch)
            if f <= 8:
                tracked_objects[2] = p2_box
            elif 9 <= f <= 12:
                # Occluded, no detection
                pass
            elif f == 13:
                # YOLO detects box, but ByteTrack has no ID (unmatched box)
                unmatched_boxes.append(p2_box)
            else:
                # ByteTrack outputs new ID 99
                tracked_objects[99] = p2_box

            # 1. Unmatched detection handling
            if unmatched_boxes:
                novel_unmatched = []
                for ubox in unmatched_boxes:
                    is_dup = any(
                        self.smoother._compute_iou(ubox, tbox) > 0.45
                        for tbox in tracked_objects.values()
                    )
                    if not is_dup:
                        novel_unmatched.append(ubox)
                if novel_unmatched:
                    recovered = self.smoother.recover_missing_tracks(novel_unmatched)
                    tracked_objects.update(recovered)

            self.local_tracker.sync_active_tracks(tracked_objects)

            # 2. Smoothing & persistent aliasing
            smoothed = self.smoother.smooth_body_boxes(tracked_objects)

            # 3. ZoneManager entry line crossing evaluation
            new_attendance, _ = self.zone_manager.process(smoothed)
            for crossed_id in new_attendance:
                attendance_events.append((f, crossed_id))

        # Check total attendance
        self.assertEqual(self.zone_manager.attendance_total, 2, "Attendance count must be exactly 2 for 2 people")
        self.assertEqual(len(attendance_events), 2, "There must be exactly 2 crossing events")
        counted_ids = {evt[1] for evt in attendance_events}
        self.assertEqual(counted_ids, {1, 2}, "Both Person 1 and Person 2 must be counted under original IDs without duplicates")


if __name__ == "__main__":
    unittest.main()
