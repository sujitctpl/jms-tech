import numpy as np
from app.vision.detector import Detection, PersonDetector
from app.vision.stream_reader import StreamReader
from app.vision.tracker import PersonTracker, TrackedPerson


def test_person_detector_invalid_and_empty_frames():
    detector = PersonDetector(confidence_threshold=0.5, person_class_id=0)

    # Test None frame
    assert detector.detect(None) == []

    # Test empty ndarray
    empty_frame = np.array([])
    assert detector.detect(empty_frame) == []


def test_stream_reader_unavailable_stream():
    # Test unavailable video stream/source
    reader = StreamReader("nonexistent_stream_path.mp4", reconnect_interval=0, max_reconnect_attempts=1)
    connected = reader.connect()
    assert connected is False

    success, frame = reader.read_frame()
    assert success is False
    assert frame is None
    reader.release()


def test_tracking_continuity_with_controlled_detections():
    tracker = PersonTracker()

    # Frame 1: Detection at (100, 100, 200, 200)
    det1 = Detection(box=(100.0, 100.0, 200.0, 200.0), confidence=0.9, class_id=0)
    tracked_f1 = tracker.track_detections_mock([det1])
    assert len(tracked_f1) == 1
    track_id_1 = tracked_f1[0].tracker_id

    # Frame 2: Person moves slightly to (105, 105, 205, 205)
    det2 = Detection(box=(105.0, 105.0, 205.0, 205.0), confidence=0.88, class_id=0)
    tracked_f2 = tracker.track_detections_mock([det2])
    assert len(tracked_f2) == 1
    # Tracking continuity: ID must be retained across frames!
    assert tracked_f2[0].tracker_id == track_id_1

    # Frame 3: Another new person appears far away (500, 500, 600, 600)
    det3 = Detection(box=(500.0, 500.0, 600.0, 600.0), confidence=0.85, class_id=0)
    tracked_f3 = tracker.track_detections_mock([det2, det3])
    assert len(tracked_f3) == 2
    ids = {t.tracker_id for t in tracked_f3}
    assert track_id_1 in ids
    # New person gets distinct track ID
    assert len(ids) == 2


def test_tracking_person_attributes():
    tp = TrackedPerson(
        tracker_id="t_1",
        box=(50.0, 100.0, 150.0, 300.0),
        confidence=0.92,
        class_id=0,
    )
    assert tp.tracker_id == "t_1"
    assert tp.bottom_center == (100.0, 300.0)
    assert tp.center == (100.0, 200.0)


def test_body_embedder_portrait_and_fullbody_adaptive_matching():
    from app.vision.body_embedder import BodyEmbedder

    embedder = BodyEmbedder()

    # Create full body crop: 300 height x 120 width (aspect ratio 2.5) with blue shirt
    full_body = np.zeros((300, 120, 3), dtype=np.uint8)
    full_body[50:150, 30:90] = [180, 50, 20]  # Blue shirt (torso)
    full_body[170:270, 30:90] = [30, 30, 30]  # Dark pants

    # Create portrait bust crop: 100 height x 100 width (aspect ratio 1.0) with matching blue shirt
    portrait_bust = np.zeros((100, 100, 3), dtype=np.uint8)
    portrait_bust[40:90, 25:75] = [180, 50, 20]  # Blue shirt (chest)

    # Synthetic identical embeddings
    emb1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
    emb2 = np.ones(512, dtype=np.float32) / np.sqrt(512)

    reid_s, clothing_s, app_s, is_m = embedder.verify_appearance(
        crop1=full_body,
        crop2=portrait_bust,
        emb1=emb1,
        emb2=emb2,
    )

    # Adaptive matching must recognize the shared shirt without penalizing missing legs in the portrait!
    t1 = embedder.extract_body_part_hist(full_body, 0.18, 0.52)
    t2 = embedder.extract_body_part_hist(portrait_bust, 0.38, 0.90)
    torso_sim = embedder.compute_color_similarity(t1, t2)
    assert torso_sim > 0.50
    assert clothing_s >= 0.35
    assert is_m is True
