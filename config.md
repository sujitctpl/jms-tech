# Showroom Visitor Intelligence (JMS) — Configuration Reference Guide

This document is a comprehensive configuration reference for all services, edge engines, computer vision pipelines, tracker algorithms, biometrics, and web applications in the platform.

---

## 1. Backend Environment Configuration (`.env`)

The central FastAPI backend uses `pydantic-settings` (defined in [`app/core/config.py`](file:///c:/Users/capermint/Desktop/Abhishek/jms/app/core/config.py)) to parse and validate settings loaded from the root `.env` file or host environment variables.

### 1.1 Application & Database Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `APP_NAME` | string | `"Showroom Visitor Intelligence - Phase 1"` | Descriptive name for API docs and log prefixes. |
| `APP_ENV` | string | `"development"` | Environment name (`"development"`, `"testing"`, `"production"`). |
| `DEBUG` | boolean | `true` | Enables verbose debug logs and hot reload. |
| `API_V1_STR` | string | `"/api/v1"` | Base prefix for all versioned REST endpoints. |
| `DATABASE_URL` | string | `"postgresql+psycopg://postgres:postgres@localhost:5432/jms_live"` | PostgreSQL connection string using `psycopg` 3. |
| `TEST_DATABASE_URL` | string | `"sqlite+pysqlite:///:memory:"` | Ephemeral test database for isolated pytest execution. |
| `DB_POOL_SIZE` | integer | `10` | SQLAlchemy connection pool size. |
| `DB_MAX_OVERFLOW` | integer | `20` | Max overflow connections allowed beyond pool size. |
| `DB_POOL_RECYCLE` | integer | `1800` | Connection recycling timeout in seconds (30 minutes). |
| `LOG_LEVEL` | string | `"INFO"` | Minimum logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

### 1.2 Computer Vision & Inference Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `DETECTION_MODEL_TYPE` | string | `"yolo"` | Detector engine: `"yolo"`, `"rf-detr"`, `"rtmdet"`, or `"auto"`. |
| `YOLO_MODEL_PATH` | string | `"models/yolo26s.pt"` | Checkpoint weights file for YOLO detection models. |
| `RTMDet_MODEL_PATH` | string | `"models/rtmdet-m-person.onnx"` | Checkpoint weights file for RTMDet ONNX model. |
| `RF_DETR_VARIANT` | string | `"small"` | Roboflow RF-DETR model size: `"nano"`, `"small"`, `"medium"`, `"base"`, `"large"`. |
| `RF_DETR_WEIGHTS_PATH` | string / null | `null` | Optional local checkpoint path for fine-tuned RF-DETR weights. |
| `DETECTION_CONFIDENCE_THRESHOLD` | float | `0.31` | Minimum confidence score to register a detected person. |
| `DETECTION_IMGSZ` | integer | `1024` | Input resolution in pixels fed into detector inference. |
| `PERSON_CLASS_ID` | integer | `0` | COCO class ID corresponding to human persons. |
| `DEVICE` | string | `"cuda:0"` | Target execution hardware: `"cuda:0"`, `"cuda"`, or `"cpu"`. |
| `TRACKER_TYPE` | string | `"bytetrack.yaml"` | Tracker configuration file (`"bytetrack.yaml"` or `"botsort.yaml"`). |

---

### 1.3 Server Background Video Worker Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `SAVE_DETECTION_EVENTS` | boolean | `false` | If `true`, saves every frame's raw bounding box coordinates to the DB. |
| `FRAME_SKIP` | integer | `0` | Number of video frames to skip between inference cycles (0 = process all frames). |
| `RECONNECT_INTERVAL_SECONDS` | integer | `5` | Time to wait in seconds before attempting an RTSP reconnect after stream loss. |
| `MAX_RECONNECT_ATTEMPTS` | integer | `5` | Maximum sequential reconnect attempts before reporting an ingestion error. |

---

### 1.4 Biometric Face Recognition Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `ARCFACE_MODEL_PATH` | string | `"models/w600k_r50.onnx"` | Path to ArcFace ResNet-50 512-D embedding model. |
| `FACE_SIMILARITY_THRESHOLD` | float | `0.32` | Cosine similarity threshold for staff identification. |
| `FACE_MATCH_MARGIN` | float | `0.030` | Minimum gap required between Top-1 and Top-2 candidate matches. |
| `FACE_MIN_SIZE` | integer | `16` | Minimum face width/height in pixels (calibrated for seated or distant staff). |
| `FACE_MIN_CONFIDENCE` | float | `0.36` | Minimum face detector confidence to reject non-face artifacts. |
| `FACE_BLUR_THRESHOLD` | float | `5.0` | Minimum Laplacian variance score to reject blurry face crops. |
| `FACE_MAX_YAW` | float | `65.0` | Maximum face yaw in degrees (tolerates walking angles). |
| `FACE_MAX_PITCH` | float | `50.0` | Maximum face pitch in degrees (accommodates 35°–45° overhead CCTV angles). |
| `FACE_MAX_ROLL` | float | `40.0` | Maximum head roll in degrees. |
| `FACE_MIN_EYE_DIST` | float | `2.5` | Minimum distance in pixels between eye landmarks. |
| `FACE_MIN_ILLUMINATION` | float | `18.0` | Minimum mean grayscale pixel brightness (rejects dark silhouettes). |
| `FACE_MAX_ILLUMINATION` | float | `245.0` | Maximum mean grayscale pixel brightness (rejects blown-out glare). |
| `EMPLOYEE_MIN_CONFIRM_FRAMES` | integer | `1` | Minimum confirmed observations before permanently tagging track as staff. |
| `EMPLOYEE_VOTE_WINDOW` | integer | `5` | Temporal sliding window size for employee voting buffer. |
| `FACE_RECOGNITION_COOLDOWN_FRAMES` | integer | `5` | Frames to wait between successive recognition attempts on unresolved tracks. |
| `FACE_MAX_RECOGNITION_ATTEMPTS` | integer | `5` | Maximum recognition attempts per track to avoid redundant compute. |

---

### 1.5 Biometric Privacy & 14-Day Rolling Retention

| Variable | Type | Default | Description |
|---|---|---|---|
| `FACE_RETENTION_DAYS` | integer | `14` | Rolling retention period in days for unconfirmed visitor embeddings. |
| `RETENTION_CLEANUP_INTERVAL_HOURS` | integer | `24` | Frequency in hours at which the scheduled background cleanup job executes. |
| `RETENTION_AUTO_CLEANUP_ENABLED` | boolean | `true` | Enables automatic background deletion of expired visitor biometrics. |

---

## 2. Camera & Edge Detector Configuration (`camera_config.json`)

Created from `camera_config.example.json`, this file defines the global defaults and channel-specific RTSP stream paths, virtual entry lines, and showroom department zones.

```json
{
  "default_channel": "101",
  "api_url": "http://127.0.0.1:8000",
  "model_type": "yolo",
  "rf_detr_variant": "small",
  "model": "models/yolo26s.pt",
  "confidence": 0.31,
  "imgsz": 1024,
  "face_model": "models/scrfd_2.5g_bnkps.onnx",
  "face_confidence": 0.36,
  "show_entry_line": true,
  "face_scale": 1.05,
  "allow_standalone_faces": false,
  "show_head_fallback": false,
  "box_smoothing": true,
  "smoothing_alpha": 0.35,
  "face_persistence": 8,
  "body_persistence": 6,
  "deadband_px": 6.0,
  "min_feet_y": 0.20,
  "line_y": 0.55,
  "direction": "top_to_bottom",
  "cameras": {
    "101": {
      "name": "Showroom Main Entrance Camera",
      "camera_id": "88970b31-d885-4fd5-a24a-bdc969c96240",
      "source": "rtsp://admin:password@192.168.1.101:554/Streaming/Channels/101",
      "entry_line": {
        "enabled": true,
        "p1": [389, 250],
        "p2": [551, 113],
        "direction": "top_to_bottom"
      },
      "sections": [
        {
          "id": "sales",
          "name": "Sales Section",
          "color": [0, 200, 255],
          "box": [17, 283, 398, 609]
        },
        {
          "id": "accounts",
          "name": "Accounts Section",
          "color": [255, 140, 0],
          "box": [756, 60, 1035, 434]
        }
      ]
    }
  }
}
```

### Parameter Reference
* `default_channel`: Camera channel used when running `python local_detector.py` without `--channel`.
* `source`: RTSP URL formatted as `rtsp://<user>:<password>@<ip>:<port>/<stream_path>`. Can also be a local video file (`"video.mp4"`) or webcam index (`"0"`).
* `entry_line.p1` & `p2`: 2D pixel coordinates `[x, y]` defining the virtual entry tripwire.
* `entry_line.direction`: Crossing direction (`"top_to_bottom"`, `"bottom_to_top"`, `"left_to_right"`, `"right_to_left"`).
* `sections[].box`: Polygon coordinates `[x1, y1, x2, y2]` denoting showroom zones for dwell-time tracking.

---

## 3. Multi-Object Tracker Configurations

### 3.1 ByteTrack Configuration ([`bytetrack.yaml`](file:///c:/Users/capermint/Desktop/Abhishek/jms/bytetrack.yaml))

ByteTrack associates both high-score and low-score detection bounding boxes to preserve tracks under partial motion blur and occlusion.

```yaml
tracker_type: bytetrack  # Backend tracker type
track_high_thresh: 0.20  # First-stage match threshold for confident detections
track_low_thresh: 0.10   # Second-stage threshold for low-score detections
new_track_thresh: 0.20   # Minimum score required to initiate a new tracklet
track_buffer: 150        # Number of frames to keep lost tracks alive (~5-7s)
match_thresh: 0.50       # IoU cost threshold for matching
fuse_score: True         # Fuses detection score with motion IoU
```

### 3.2 BoT-SORT Configuration ([`botsort.yaml`](file:///c:/Users/capermint/Desktop/Abhishek/jms/botsort.yaml))

BoT-SORT incorporates Global Motion Compensation (GMC) to mitigate camera shake, pole sway, and PTZ movement.

```yaml
tracker_type: botsort    # Backend tracker type
track_high_thresh: 0.25  # First-stage match threshold
track_low_thresh: 0.10   # Second-stage threshold for low-score detections
new_track_thresh: 0.25   # Minimum score to initiate a new tracklet
track_buffer: 120        # Frames to keep lost tracks alive (~4-5s)
match_thresh: 0.70       # IoU cost threshold
fuse_score: True         # Fuses score with motion
gmc_method: sparseOptFlow # Motion compensation: sparseOptFlow | orb | sift | ecc | none
proximity_thresh: 0.5    # Minimum IoU to consider tracks proximate
appearance_thresh: 0.80  # Appearance similarity threshold
with_reid: False         # Set to True when ReID feature model is active
model: auto              # ReID model identifier
```

---

## 4. Dashboard Frontend Environment ([`dashboard/.env`](file:///c:/Users/capermint/Desktop/Abhishek/jms/dashboard/.env.example))

The React 19 web dashboard configures its API connection endpoints via `dashboard/.env`:

```ini
# Single environment switch variable: 'LOCAL' or 'PROD'
VITE_APP_ENV=LOCAL

# Local Backend Settings (FastAPI Video Ingestion, Worker & Analytics)
VITE_LOCAL_API_URL=http://localhost:8000

# Local Authentication Server (Express PostgreSQL Auth Server)
VITE_LOCAL_AUTH_URL=http://localhost:5000

# Production API URL (Used when VITE_APP_ENV=PROD)
VITE_PROD_API_URL=https://production-gateway.example.com
```

---

## 5. Local Detector CLI Flags Reference (`local_detector.py`)

Any value in `camera_config.json` can be dynamically overridden from the command line:

```bash
python local_detector.py [OPTIONS]
```

| Flag | Argument | Description | Default |
|---|---|---|---|
| `-c`, `--channel` | `string` | Camera channel key in `camera_config.json` | `"101"` |
| `-s`, `--source` | `string` | Direct RTSP URL, video file, or webcam device index | Defined in JSON |
| `--v2` | *flag* | Use Roboflow RF-DETR Detection Transformer | `False` |
| `--v1` | *flag* | Use Ultralytics YOLO CNN detector | `True` |
| `--model-type` | `choice` | Architecture: `'yolo'`, `'rtmdet'`, `'rf-detr'`, `'rtdetr'`, or `'auto'` | `'yolo'` |
| `--rf-detr-variant`| `choice`| Variant: `'nano'`, `'small'`, `'medium'`, `'base'`, `'large'` | `'small'` |
| `-m`, `--model` | `string` | Model checkpoint file path | `'models/yolo26s.pt'` |
| `--conf` | `float` | Detection confidence threshold | `0.31` |
| `--imgsz` | `integer`| Input resolution in pixels | `1024` |
| `--tracker` | `string` | Tracker config: `'bytetrack'`, `'botsort'`, or path to YAML | `bytetrack.yaml` |
| `--face-model` | `string` | Face detector weights file path | `'models/scrfd_2.5g_bnkps.onnx'` |
| `--face-conf` | `float` | Face detector confidence threshold | `0.36` |
| `--face-scale` | `float` | Face bounding box scale multiplier | `1.05` |
| `--sim-thresh` | `float` | Cosine similarity threshold for employee identification | `0.32` |
| `--show-line` | *flag* | Display and calculate virtual entry line crossings | `True` |
| `--line-y` | `float` | Fallback entry line Y position as fraction of frame height | `0.55` |
| `--direction` | `choice` | Entry direction: `'top_to_bottom'` or `'bottom_to_top'` | `'top_to_bottom'` |
| `--deadband` | `float` | Pixel deadband filter to eliminate stationary box jitter | `6.0` |
| `--smoothing-alpha`| `float`| EMA box coordinate smoothing factor (0.1 to 1.0) | `0.35` |
| `--face-persistence`| `integer`| Frame buffer size for face detection continuity | `8` |
| `--body-persistence`| `integer`| Frame timeout before purging inactive tracks | `6` |
| `--standalone-faces`| *flag* | Allow face detection behind cubicle partitions without full body | `False` |
| `--head-fallback` | *flag* | Enable anatomical head box estimation when face is turned away | `False` |
| `--no-head-fallback`| *flag*| Disable anatomical head box estimation | Default |
| `--no-smoothing` | *flag* | Disable anti-jitter bounding box smoothing | `False` |
| `--headless` | *flag* | Run as background service without local OpenCV window | `False` |
| `--api-url` | `string` | Central FastAPI backend base URL | `'http://127.0.0.1:8000'` |

