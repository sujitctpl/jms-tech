import time
import uuid
from app.db.models.camera import Camera
from app.workers.video_worker import VideoWorkerManager


def test_worker_manager_start_and_stop():
    manager = VideoWorkerManager()

    cam = Camera(
        id=uuid.uuid4(),
        name="Test Worker Cam",
        stream_url="nonexistent.mp4",
        entry_line_config={"line": [[10, 20], [30, 40]], "direction": "top_to_bottom"},
        is_active=True,
    )

    # Start worker
    started = manager.start_worker(cam)
    assert started is True

    # Duplicate worker prevention
    dup_started = manager.start_worker(cam)
    assert dup_started is False

    # Check status
    status = manager.get_status(cam.id)
    assert "is_processing" in status
    assert "frames_processed" in status

    # Stop worker
    stopped = manager.stop_worker(cam.id)
    assert stopped is True

    # Allow thread to terminate
    time.sleep(0.2)
    stopped_again = manager.stop_worker(cam.id)
    assert stopped_again is False
