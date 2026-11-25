"""Edge -> backend sync must survive a backend restart (uvicorn --reload, crash) without losing visitors.

At 13:00-13:02 on 2026-11-25 every send failed with "connection refused" during a backend reload and the
detector dropped those entries for good (a visitor's entry never appeared on the dashboard).
"""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import local_detector
from local_detector import DashboardSyncClient


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_backend(port: int, status: int, received: list) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append((self.path, json.loads(body)))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"person_id": "VIS-2026-1099"}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _wait(cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_entry_is_held_and_delivered_in_order_after_backend_restart(monkeypatch):
    monkeypatch.setattr(local_detector, "log_exit_debug", lambda event: None)
    port = _free_port()
    registered = []
    client = DashboardSyncClient(f"http://127.0.0.1:{port}", "cam-test",
                                 on_visitor_registered=lambda t, p: registered.append((t, p)))
    try:
        client.send_visitor_detect(tracker_id="81", appearance="Black / Dark Attire")
        client.send_visitor_zone_update({"tracker_id": "81", "exit_time": "2026-11-25T07:32:00Z"})
        time.sleep(1.5)  # backend is down: first attempts are refused
        received: list = []
        server = _start_backend(port, 200, received)
        try:
            assert _wait(lambda: len(received) == 2)
            assert [path for path, _ in received] == ["/api/v1/visitors/detect", "/api/v1/visitors/update-zone"]
            assert _wait(lambda: registered == [("81", "VIS-2026-1099")])
        finally:
            server.shutdown()
            server.server_close()
    finally:
        client.stop()


def test_backend_answer_is_not_retried(monkeypatch):
    monkeypatch.setattr(local_detector, "log_exit_debug", lambda event: None)
    port = _free_port()
    received: list = []
    server = _start_backend(port, 404, received)  # e.g. "no matching visitor" for an exit
    client = DashboardSyncClient(f"http://127.0.0.1:{port}", "cam-test")
    try:
        client.send_visitor_zone_update({"tracker_id": "12", "exit_time": "2026-11-25T07:25:37Z"})
        assert _wait(lambda: len(received) == 1)
        time.sleep(2.0)
        assert len(received) == 1
    finally:
        client.stop()
        server.shutdown()
        server.server_close()


def test_heartbeats_are_not_held_while_backend_is_down():
    port = _free_port()
    client = DashboardSyncClient(f"http://127.0.0.1:{port}", "cam-test")
    try:
        client.send_heartbeat(fps=10.0, frames_processed=1, current_persons=0, total_entries=0)
        assert _wait(lambda: client.queue.unfinished_tasks == 0, timeout=5.0)
    finally:
        client.stop()


def test_unreachable_classification():
    import urllib.error

    assert local_detector._is_unreachable_error(urllib.error.URLError(ConnectionRefusedError(10061, "refused")))
    assert local_detector._is_unreachable_error(urllib.error.URLError(
        "[WinError 10061] No connection could be made because the target machine actively refused it"))
    assert not local_detector._is_unreachable_error(TimeoutError("timed out"))  # may have been processed
