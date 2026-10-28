import uuid
from fastapi import status
from app.api.routes import (
    CAMERA_HEARTBEAT,
    CAMERA_STATUS,
    GET_CAMERA,
    LIST_CAMERAS,
    RECORD_CAMERA_FOOTFALL,
    REGISTER_CAMERAS,
    START_CAMERA,
    STOP_CAMERA,
    UPDATE_CAMERA,
)


def test_register_camera(client, sample_camera_payload):
    response = client.post(REGISTER_CAMERAS, json=sample_camera_payload)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()

    assert data["name"] == sample_camera_payload["name"]
    assert data["location"] == sample_camera_payload["location"]
    assert data["is_active"] is True
    assert "id" in data

    # Verify credentials masking requirement
    assert "secret123" not in data["stream_url"]
    assert "***:***" in data["stream_url"]


def test_list_cameras(client, sample_camera_payload):
    # Register camera
    client.post(REGISTER_CAMERAS, json=sample_camera_payload)

    response = client.get(LIST_CAMERAS)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["name"] == sample_camera_payload["name"]


def test_get_camera_details(client, sample_camera_payload):
    reg = client.post(REGISTER_CAMERAS, json=sample_camera_payload).json()
    camera_id = reg["id"]

    url = GET_CAMERA.format(camera_id=camera_id)
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == camera_id
    assert data["name"] == sample_camera_payload["name"]


def test_get_camera_not_found(client):
    url = GET_CAMERA.format(camera_id=uuid.uuid4())
    response = client.get(url)
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_update_camera_settings(client, sample_camera_payload):
    reg = client.post(REGISTER_CAMERAS, json=sample_camera_payload).json()
    camera_id = reg["id"]

    update_payload = {
        "name": "Updated Entrance Cam",
        "location": "North Wing",
        "is_active": False,  # soft deactivation
    }
    url = UPDATE_CAMERA.format(camera_id=camera_id)
    response = client.patch(url, json=update_payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "Updated Entrance Cam"
    assert data["location"] == "North Wing"
    assert data["is_active"] is False


def test_camera_configuration_validation(client):
    # Invalid line: identical points
    invalid_payload = {
        "name": "Invalid Cam",
        "stream_url": "rtsp://localhost/test",
        "entry_line_config": {
            "line": [[100.0, 100.0], [100.0, 100.0]],
            "direction": "top_to_bottom",
        },
    }
    response = client.post(REGISTER_CAMERAS, json=invalid_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Invalid line: negative coordinates
    invalid_payload2 = {
        "name": "Invalid Cam 2",
        "stream_url": "rtsp://localhost/test",
        "entry_line_config": {
            "line": [[-10.0, 100.0], [100.0, 100.0]],
            "direction": "top_to_bottom",
        },
    }
    response2 = client.post(REGISTER_CAMERAS, json=invalid_payload2)
    assert response2.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_start_camera_inactive_rejection(client, sample_camera_payload):
    reg = client.post(REGISTER_CAMERAS, json=sample_camera_payload).json()
    camera_id = reg["id"]

    # Deactivate camera
    client.patch(UPDATE_CAMERA.format(camera_id=camera_id), json={"is_active": False})

    # Attempt to start inactive camera
    start_url = START_CAMERA.format(camera_id=camera_id)
    response = client.post(start_url)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "inactive" in response.json()["detail"].lower()


def test_start_camera_without_entry_line_config(client):
    # Camera with no entry_line_config
    payload = {
        "name": "Unconfigured Camera",
        "stream_url": "rtsp://localhost/live",
    }
    reg = client.post(REGISTER_CAMERAS, json=payload).json()
    camera_id = reg["id"]

    start_url = START_CAMERA.format(camera_id=camera_id)
    response = client.post(start_url)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "entry line configuration is required" in response.json()["detail"].lower()


def test_camera_status_and_lifecycle(client, sample_camera_payload):
    reg = client.post(REGISTER_CAMERAS, json=sample_camera_payload).json()
    camera_id = reg["id"]

    # Initial status
    status_url = CAMERA_STATUS.format(camera_id=camera_id)
    status_resp = client.get(status_url)
    assert status_resp.status_code == status.HTTP_200_OK
    assert status_resp.json()["is_processing"] is False

    # Start camera
    start_url = START_CAMERA.format(camera_id=camera_id)
    start_resp = client.post(start_url)
    assert start_resp.status_code == status.HTTP_200_OK
    assert start_resp.json()["is_processing"] is True

    # Duplicate start check
    dup_resp = client.post(start_url)
    assert dup_resp.status_code == status.HTTP_200_OK
    assert "already active" in dup_resp.json()["message"].lower()

    # Stop camera
    stop_url = STOP_CAMERA.format(camera_id=camera_id)
    stop_resp = client.post(stop_url)
    assert stop_resp.status_code == status.HTTP_200_OK
    assert stop_resp.json()["is_processing"] is False


def test_edge_footfall_and_heartbeat(client, sample_camera_payload):
    reg = client.post(REGISTER_CAMERAS, json=sample_camera_payload).json()
    camera_id = reg["id"]

    footfall_url = RECORD_CAMERA_FOOTFALL.format(camera_id=camera_id)
    payload = {
        "tracker_id": "edge_track_01",
        "confidence": 0.94,
    }
    resp = client.post(footfall_url, json=payload)
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["success"] is True
    assert data["total_camera_footfall"] >= 1

    # Heartbeat
    hb_url = CAMERA_HEARTBEAT.format(camera_id=camera_id)
    hb_resp = client.post(hb_url, json={"is_processing": True, "fps": 29.5, "frames_processed": 100, "total_entries": 1})
    assert hb_resp.status_code == status.HTTP_200_OK

