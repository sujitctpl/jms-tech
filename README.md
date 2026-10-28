# Showroom Visitor Intelligence - Phase 1: AI-Based Visitor Detection & Entry Counting

An AI-powered Showroom Visitor Intelligence backend service utilizing existing CCTV infrastructure to detect visitors, maintain continuity tracking across video frames, detect directional virtual entry-line crossings, and calculate footfall analytics while strictly preventing duplicate counts.

---

## 1. Features (Phase 1 Scope)

- **Person Detection:** Powered by Ultralytics YOLO with configurable confidence thresholds and person class isolation (`class_id=0`).
- **Visitor Tracking:** Integrated ByteTrack tracking preserving temporary tracking IDs across frames, handling occlusions without creating spurious tracks.
- **Virtual Entry-Line Crossing:** Geometric 2D line segment crossing detection using bottom-center (foot/ground contact) reference points.
- **Strict Duplicate Prevention:** Enforced at both the in-memory application level and database constraint level (`UNIQUE` on `visitor_track_id` in `footfall_events`).
- **Security & Privacy:** Automatic masking of RTSP credentials in API responses and operational logs.
- **Centralized Endpoint Paths:** Dedicated `app/api/routes.py` path constants consumed across all routers.
- **Background Video Ingestion:** Multi-threaded asynchronous video ingestion worker with auto-reconnection and clean lifecycle management (`/start`, `/stop`, `/status`).
- **Footfall Analytics:** Total and daily entry counts aggregated over customizable time ranges and camera filters.

---

## 2. Technology Stack

- **Backend Framework:** FastAPI
- **Database:** PostgreSQL (with `psycopg` 3)
- **ORM:** SQLAlchemy 2.x
- **Database Migrations:** Alembic
- **Data Validation:** Pydantic v2 & `pydantic-settings`
- **Video Processing:** OpenCV
- **Computer Vision:** Ultralytics YOLO & ByteTrack
- **Testing:** pytest & HTTPX

---

## 3. Project Structure

```text
showroom-visitor-intelligence/
├── app/
│   ├── main.py                     # FastAPI application entrypoint & lifespan
│   ├── core/
│   │   ├── config.py               # Pydantic BaseSettings environment config
│   │   └── logging.py              # Log sanitization (masks RTSP credentials)
│   ├── api/
│   │   ├── routes.py               # Centralized endpoint path constants
│   │   └── v1/
│   │       ├── cameras.py          # Camera management & worker controls
│   │       └── analytics.py        # Footfall analytics endpoints
│   ├── db/
│   │   ├── session.py              # Database engine & session generator
│   │   ├── base.py                 # Declarative Base
│   │   └── models/
│   │       ├── camera.py           # cameras table model
│   │       ├── processing_run.py   # processing_runs table model
│   │       ├── visitor_track.py    # visitor_tracks table model
│   │       ├── detection_event.py  # detection_events table model
│   │       └── footfall_event.py   # footfall_events table model
│   ├── schemas/
│   │   ├── camera.py               # Pydantic schemas for cameras & configs
│   │   └── analytics.py            # Pydantic schemas for footfall results
│   ├── repositories/
│   │   ├── camera_repo.py
│   │   ├── processing_run_repo.py
│   │   ├── visitor_track_repo.py
│   │   ├── detection_event_repo.py
│   │   └── footfall_event_repo.py
│   ├── services/
│   │   ├── camera_service.py       # Camera business logic & validations
│   │   └── analytics_service.py    # Date-range footfall aggregations
│   ├── vision/
│   │   ├── detector.py             # YOLO person detector
│   │   ├── tracker.py              # ByteTrack visitor tracker
│   │   ├── stream_reader.py        # OpenCV RTSP video stream reader
│   │   └── entry_counter.py        # Virtual line crossing detector
│   └── workers/
│       └── video_worker.py         # Background worker & WorkerManager
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 0001_initial_phase1.py   # Initial database migration
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_cameras.py
│   ├── test_analytics.py
│   ├── test_vision.py
│   ├── test_entry_counter.py
│   ├── test_video_worker.py
│   └── test_database_models.py
├── alembic.ini
├── .env.example
├── requirements.txt
└── README.md
```

---

## 4. Setup and Installation

### Prerequisites
- Python 3.11+
- PostgreSQL (or SQLite for local unit testing)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env` and configure your database connection:
```bash
cp .env.example .env
```
Key configuration settings in `.env`:
```ini
DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/showroom_db"
YOLO_MODEL_PATH="yolov8n.pt"
DETECTION_CONFIDENCE_THRESHOLD=0.45
PERSON_CLASS_ID=0
DEVICE="cpu"  # Or "cuda" / "0" for GPU acceleration
SAVE_DETECTION_EVENTS=false
RECONNECT_INTERVAL_SECONDS=5
MAX_RECONNECT_ATTEMPTS=5
```

### Step 3: Run Database Migrations
Apply Alembic migrations to set up the database schema:
```bash
alembic upgrade head
```

### Step 4: Start the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at: `http://localhost:8000/docs`.

---

## 5. API Reference

All 10 endpoints are registered using path constants defined in `app/api/routes.py`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check API service health |
| `POST` | `/api/v1/cameras` | Register a new CCTV camera |
| `GET` | `/api/v1/cameras` | List configured cameras |
| `GET` | `/api/v1/cameras/{camera_id}` | Retrieve camera details |
| `PATCH` | `/api/v1/cameras/{camera_id}` | Update camera settings / soft deactivation |
| `POST` | `/api/v1/cameras/{camera_id}/start` | Start background video processing |
| `POST` | `/api/v1/cameras/{camera_id}/stop` | Stop background video processing |
| `GET` | `/api/v1/cameras/{camera_id}/status` | Retrieve camera processing status |
| `GET` | `/api/v1/analytics/footfall` | Retrieve total entry counts |
| `GET` | `/api/v1/analytics/footfall/daily` | Retrieve daily entry counts |

### Example 1: Register a Camera
```bash
curl -X POST "http://localhost:8000/api/v1/cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Main Showroom Entrance",
    "stream_url": "rtsp://user:adminpass@192.168.1.50:554/live",
    "location": "North Wing Entrance",
    "entry_line_config": {
      "line": [[200.0, 400.0], [800.0, 400.0]],
      "direction": "top_to_bottom"
    }
  }'
```
*Note: The response will automatically mask credentials:*
```json
{
  "id": "c1388653-ff8f-4bc4-b7c1-7a66fbc4b25b",
  "name": "Main Showroom Entrance",
  "stream_url": "rtsp://***:***@192.168.1.50:554/live",
  "location": "North Wing Entrance",
  "is_active": true,
  "entry_line_config": {
    "line": [[200.0, 400.0], [800.0, 400.0]],
    "direction": "top_to_bottom"
  },
  "created_at": "2026-10-27T12:00:00Z",
  "updated_at": "2026-10-27T12:00:00Z"
}
```

### Example 2: Start Video Processing
```bash
curl -X POST "http://localhost:8000/api/v1/cameras/c1388653-ff8f-4bc4-b7c1-7a66fbc4b25b/start"
```
*Response:*
```json
{
  "message": "Video processing started successfully.",
  "camera_id": "c1388653-ff8f-4bc4-b7c1-7a66fbc4b25b",
  "is_processing": true
}
```

### Example 3: Get Processing Status
```bash
curl -X GET "http://localhost:8000/api/v1/cameras/c1388653-ff8f-4bc4-b7c1-7a66fbc4b25b/status"
```
*Response:*
```json
{
  "camera_id": "c1388653-ff8f-4bc4-b7c1-7a66fbc4b25b",
  "name": "Main Showroom Entrance",
  "is_active": true,
  "is_processing": true,
  "active_run_id": "d9818815-1815-4602-9e90-c11df5fb1390",
  "frames_processed": 1420,
  "last_error": null
}
```

### Example 4: Retrieve Total Footfall
```bash
curl -X GET "http://localhost:8000/api/v1/analytics/footfall?start_time=2026-10-20T00:00:00Z&end_time=2026-10-27T23:59:59Z"
```
*Response:*
```json
{
  "total_entries": 148,
  "camera_id": null,
  "start_time": "2026-10-20T00:00:00Z",
  "end_time": "2026-10-27T23:59:59Z"
}
```

### Example 5: Retrieve Daily Footfall
```bash
curl -X GET "http://localhost:8000/api/v1/analytics/footfall/daily?start_date=2026-10-25&end_date=2026-10-27"
```
*Response:*
```json
{
  "daily_footfall": [
    {"date": "2026-10-25", "entries": 42},
    {"date": "2026-10-26", "entries": 55},
    {"date": "2026-10-27", "entries": 51}
  ],
  "total_entries": 148,
  "camera_id": null,
  "start_date": "2026-10-25",
  "end_date": "2026-10-27"
}
```

---

## 6. Virtual Entry-Line Crossing & Duplicate Prevention

### Reference Point Calculation
The system calculates the **bottom-center** of the detected bounding box:
$$\text{Point} = \left(\frac{x_1 + x_2}{2}, y_2\right)$$
This corresponds to the visitor's foot position on the ground plane, eliminating false crossings caused by torso or head tilt.

### Crossing Geometry & Direction Validation
1. Segment intersection test between previous/current frame trajectory and the configured entry line $P_1P_2$.
2. Orientation check: Ensures the crossing vector corresponds strictly to the designated entry direction (`top_to_bottom`, `bottom_to_top`, `left_to_right`, `right_to_left`, or `positive_to_negative`/`inward`).

### Duplicate Prevention Architecture
- **Application Level:** Tracked persons maintain an in-memory set of counted `tracker_id`s. Once a track crosses the line, subsequent frames or hovering movements near the line will never trigger another entry event.
- **Database Level:** The `footfall_events` table enforces a `UNIQUE (visitor_track_id)` constraint and a check constraint `event_type = 'entry'`. Any attempted duplicate insertion is safely handled and discarded.

---

## 7. Running Automated Tests

Run the complete test suite using `pytest`:
```bash
python -m pytest tests -v
```

The tests cover:
- Health check endpoint
- Camera CRUD, credential masking, and validation
- Pre-processing camera configuration validation
- Database models, foreign keys, and unique/check constraints
- Person detection edge cases (invalid frames, empty arrays, confidence thresholds)
- Tracking continuity across frames using ByteTrack logic
- Virtual entry-line crossing in designated directions & wrong-direction rejections
- Duplicate entry-count prevention
- Background worker start, stop, duplicate prevention, and status retrieval
- Footfall analytics (total & daily aggregations, date-range validation)
