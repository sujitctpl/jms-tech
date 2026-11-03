import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np

from app.vision.detector import (
    Detection,
    PersonDetector,
    RFDETRDetector,
    RTDETRDetector,
    YOLODetector,
    create_detector,
)
from app.vision.tracker import PersonTracker, TrackedPerson


class TestRFDETRPipeline(unittest.TestCase):
    def test_factory_creates_correct_detectors(self):
        """Test that create_detector instantiates the requested detector type."""
        d_yolo = create_detector(model_type="yolo")
        self.assertIsInstance(d_yolo, YOLODetector)

        d_rf = create_detector(model_type="rf-detr", rf_detr_variant="small")
        self.assertIsInstance(d_rf, RFDETRDetector)
        self.assertEqual(d_rf.variant, "small")

        d_rt = create_detector(model_type="rtdetr")
        self.assertIsInstance(d_rt, RTDETRDetector)

        # Auto-detect from model name
        d_auto_rf = create_detector(model_type="auto", model_path="weights/rf-detr-checkpoint.pth")
        self.assertIsInstance(d_auto_rf, RFDETRDetector)

    def test_rf_detr_empty_and_invalid_frames(self):
        """Test handling of None and empty frames in RFDETRDetector."""
        detector = RFDETRDetector(confidence_threshold=0.5)
        self.assertEqual(detector.detect(None), [])
        self.assertEqual(detector.detect(np.array([])), [])

    def test_rf_detr_inference_parsing_and_filtering(self):
        """Test that RFDETRDetector correctly extracts and filters supervision.Detections."""
        detector = RFDETRDetector(confidence_threshold=0.40, person_class_id=0)

        # Mock supervision.Detections object returned by rfdetr model.predict
        mock_detections = SimpleNamespace(
            xyxy=np.array([
                [100.0, 150.0, 200.0, 350.0],  # Person (conf=0.88, cls=0) -> KEEP
                [300.0, 200.0, 400.0, 450.0],  # Person (conf=0.25, cls=0) -> REJECT (< conf)
                [50.0, 50.0, 120.0, 130.0],    # Car/Backpack (conf=0.91, cls=2) -> REJECT (cls!=0)
                [450.0, 100.0, 550.0, 300.0],  # Person (conf=0.65, cls=0) -> KEEP
            ]),
            confidence=np.array([0.88, 0.25, 0.91, 0.65]),
            class_id=np.array([0, 0, 2, 0]),
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = mock_detections
        detector._model = mock_model

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(dummy_frame)

        self.assertEqual(len(results), 2)
        # Verify first valid detection
        self.assertEqual(results[0].box, (100.0, 150.0, 200.0, 350.0))
        self.assertAlmostEqual(results[0].confidence, 0.88, places=2)
        self.assertEqual(results[0].class_id, 0)

        # Verify second valid detection
        self.assertEqual(results[1].box, (450.0, 100.0, 550.0, 300.0))
        self.assertAlmostEqual(results[1].confidence, 0.65, places=2)
        self.assertEqual(results[1].class_id, 0)

    def test_person_tracker_with_rf_detr_detector(self):
        """Test multi-frame object tracking continuity when using an RF-DETR detector backend."""
        mock_rf_detector = MagicMock(spec=RFDETRDetector)
        tracker = PersonTracker(detector=mock_rf_detector)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1: Single person detected
        mock_rf_detector.detect.return_value = [
            Detection(box=(100.0, 100.0, 200.0, 300.0), confidence=0.90, class_id=0)
        ]
        tracks_f1 = tracker.track_frame(dummy_frame)
        self.assertEqual(len(tracks_f1), 1)
        first_id = tracks_f1[0].tracker_id

        # Verify bottom_center & center calculation
        self.assertEqual(tracks_f1[0].bottom_center, (150.0, 300.0))
        self.assertEqual(tracks_f1[0].center, (150.0, 200.0))

        # Frame 2: Person moves slightly
        mock_rf_detector.detect.return_value = [
            Detection(box=(105.0, 108.0, 205.0, 308.0), confidence=0.89, class_id=0)
        ]
        tracks_f2 = tracker.track_frame(dummy_frame)
        self.assertEqual(len(tracks_f2), 1)
        # Tracking continuity: ID must be preserved!
        self.assertEqual(tracks_f2[0].tracker_id, first_id)

        # Frame 3: Second person enters at a different position
        mock_rf_detector.detect.return_value = [
            Detection(box=(110.0, 115.0, 210.0, 315.0), confidence=0.87, class_id=0),
            Detection(box=(400.0, 200.0, 500.0, 450.0), confidence=0.93, class_id=0),
        ]
        tracks_f3 = tracker.track_frame(dummy_frame)
        self.assertEqual(len(tracks_f3), 2)
        tracked_ids = {t.tracker_id for t in tracks_f3}
        self.assertIn(first_id, tracked_ids)
        self.assertEqual(len(tracked_ids), 2)

    def test_legacy_person_detector_backward_compatibility(self):
        """Test that PersonDetector acts as a compatible facade."""
        # Default YOLO
        legacy = PersonDetector(confidence_threshold=0.45)
        self.assertEqual(legacy.confidence_threshold, 0.45)
        self.assertEqual(legacy.detect(None), [])

        # Configured for RF-DETR
        rf_facade = PersonDetector(model_type="rf-detr", rf_detr_variant="nano")
        self.assertIsInstance(rf_facade.detector, RFDETRDetector)
        self.assertEqual(rf_facade.detect(None), [])


if __name__ == "__main__":
    unittest.main()
