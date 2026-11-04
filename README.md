# Showroom Visitor Intelligence & Footfall Analytics Platform

An enterprise-grade, edge-AI hybrid visitor intelligence and showroom analytics platform. The system leverages existing CCTV / RTSP camera infrastructure to deliver real-time person detection powered by **Roboflow RF-DETR** (Real-Time Detection Transformer with DINOv2 backbone, with Ultralytics YOLO fallback), multi-object tracking (**ByteTrack** and **BoT-SORT** with motion compensation), biometric face recognition (**ArcFace** 512-D embeddings with **SCRFD** / **YuNet** face detection), staff vs. visitor segregation, automated employee attendance logging, dynamic department zone management, directional virtual entry-line crossing, dwell-time analytics, and an interactive **React 19** web dashboard.

---

## 1. System Overview & Architecture

The platform operates on a high-efficiency **Edge-AI Hybrid Architecture**:

```mermaid
graph TD
    subgraph Edge Layer [Showroom Edge / Local Machine]
        CCTV["CCTV / IP Cameras (RTSP TCP)"]
        Detector["Edge AI Engine (local_detector.py)\n• RF-DETR Transformer / YOLO Detection\n• SCRFD / YuNet Face Tracking\n• ArcFace 512-D Biometric Extraction\n• ByteTrack / BoT-SORT Tracking\n• Entry/Exit Line & Department Zones\n• Local MJPEG Stream Server"]
        SyncQueue["Background Sync Client (Memory Queue)"]
    end

    subgraph Backend Layer [FastAPI & Express Backend]
        API["FastAPI Analytics Backend (:8000)\n• Camera Management & Streaming\n• Footfall & Daily Analytics\n• Dynamic Department Management\n• Employee Registration & Attendance\n• Biometric Rolling Retention Worker"]
        AuthSrv["Express Auth Server (:5000)\n• User & Admin Authentication\n• PostgreSQL Session Pool"]
        DB[("PostgreSQL Database (:5432)\n• cameras & departments\n• employees & employee_attendance\n• person_identities (pgvector 512)\n• visitor_tracks & footfall_events")]
    end

    subgraph Client Layer [Modern Web UI]
        Dashboard["React 19 Dashboard (:5173)\n• Live CCTV Preview & AI Overlays\n• Real-Time Footfall & Peak Hours\n• Dynamic Department Management\n• Staff Attendance & Visitor Tagging\n• Section Dwell Time & Reports"]
    end

    CCTV -->|RTSP TCP Stream| Detector
    Detector -->|Events, Punches & Heartbeat| SyncQueue
    SyncQueue -->|REST API JSON| API
    API --> DB
    AuthSrv --> DB
    Dashboard -->|Analytics, Cameras & Staff| API
    Dashboard -->|Auth| AuthSrv
    Dashboard -.->|MJPEG Live Stream| Detector
```

Detailed architectural flows, mathematical models, and sequence diagrams are documented in [flow.md](file:///c:/Users/capermint/Desktop/Abhishek/jms/flow.md). Full configuration reference is in [config.md](file:///c:/Users/capermint/Desktop/Abhishek/jms/config.md).

---

## 2. Key Features

- **Multi-Model Edge AI Inference (`local_detector.py`)**:
  - **Roboflow RF-DETR (`--v2`)**: Transformer-based object detection delivering state-of-the-art accuracy with DINOv2 backbone.
  - **Ultralytics YOLO (`--v1`)**: Ultra-fast CNN detection fallback (`yolo26s.pt` / `yolov8n.pt` / `yolo11n.pt`).
  - **RTMDet Support**: High-performance ONNX detection engine.
- **RTSP TCP Buffer Management**: Forces TCP transport globally (`rtsp_transport;tcp`) and employs a non-blocking threaded frame grabber (`ThreadedRTSPReader`) to prevent frame delay and macroblock artifacts.
- **Pluggable Multi-Object Tracking**:
  - **ByteTrack** (`bytetrack.yaml`): Low-latency matching using both high- and low-score detection boxes.
  - **BoT-SORT** (`botsort.yaml`): Includes Global Motion Compensation (GMC via sparse optical flow) and Kalman state estimation to eliminate camera shake/drift.
  - **Robust Fallback Centroid & Kalman Tracker**: Velocity-prediction fallback ensuring tracking continuity across frames.
- **Biometric Face Recognition & Staff vs. Visitor Segregation**:
  - Neural face detection via **SCRFD** (`scrfd_2.5g_bnkps.onnx`) or **YuNet**.
  - **ArcFace** (`w600k_r50.onnx`) 512-dimensional feature embedding extraction.
  - Real-time cosine similarity matching against registered employee vector gallery.
  - Temporal voting buffers (`EMPLOYEE_VOTE_WINDOW = 5`) ensuring zero false-positive visitor misclassifications.
- **Automated Employee Attendance System**:
  - Real-time check-in and check-out punches logged automatically when staff enter camera field of view.
  - Multi-photo gallery enrollment per employee to improve recognition across diverse lighting and angles.
  - Daily aggregated reporting calculating first seen, last seen, total hours worked, late arrival, and early departure flags.
- **Dynamic Department Management**:
  - Showroom zones/departments can be created, updated, and color-coded dynamically via API (`/api/v1/departments`) or Dashboard UI.
  - Associate specific departments with cameras or configure global showroom sections.
- **14-Day Rolling Biometric Retention (GDPR / Privacy Compliance)**:
  - Automated daily background scheduler (`retention_scheduler`) purging visitor facial embeddings and tracks inactive for > 14 days while preserving aggregated footfall numbers.
- **Virtual Entry-Line Crossing**:
  - 2D ground-plane foot contact point calculation ($\frac{x_1 + x_2}{2}, y_2$) with strict directional validation (`top_to_bottom`, `bottom_to_top`, `left_to_right`, `right_to_left`).
- **Strict Multi-Tier Duplicate Prevention**:
  - In-memory tracker lock (`counted_track_ids: Set[str]`) combined with database-level uniqueness (`UNIQUE` constraint on `visitor_track_id` in `footfall_events`).
- **Live Video Streaming & Overlays**:
  - Built-in HTTP MJPEG streaming server for low-latency live preview with AI bounding boxes and tracks directly in the web dashboard.
- **Modern Web Dashboard (`dashboard/`)**:
  - Built with React 19, Vite, TypeScript, and Tailwind CSS.
  - Live CCTV preview, dynamic department configuration, employee attendance management, peak hour heatmaps, visitor tagging, and exportable reports.
- **Single-Command Orchestration**:
  - Concurrently launch all 3 services (FastAPI, Express Auth, and Vite Dashboard) via `python scripts/run_all.py` or Windows `.bat` / `.ps1` scripts.
- **Desktop Executable Packaging**:
  - PyInstaller builder script (`build_exe.py`) to compile the edge detector into a standalone Windows `.exe`.

---

## 3. Technology Stack

| Component | Technologies |
|---|---|
| **Edge AI Engine** | Python 3.11+, Roboflow RF-DETR, Ultralytics YOLO, ByteTrack, BoT-SORT, OpenCV, ONNX Runtime, NumPy |
| **Biometrics & Vision** | InsightFace ArcFace (512-D), SCRFD, YuNet Face Detector, pgvector / Cosine Similarity |
| **Backend API** | FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2, PostgreSQL (`psycopg` 3), `pgvector` |
| **Auth Service** | Node.js, Express 5, TypeScript (`tsx`), `pg`, bcryptjs |
| **Frontend Web** | React 19, TypeScript, Vite, Tailwind CSS v4, Recharts, Lucide Icons |
| **Packaging & Scripts** | PyInstaller, Python subprocess, Windows Batch, PowerShell |

---

## 4. Project Structure

```text
jms/
├── app/                                 # Central FastAPI Backend Service
│   ├── main.py                          # FastAPI entrypoint, lifespan & static mount
│   ├── core/
│   │   ├── config.py                    # Pydantic BaseSettings environment config
│   │   └── logging.py                   # Masked logging (sanitizes RTSP credentials)
│   ├── api/
│   │   ├── routes.py                    # Centralized endpoint path constants
│   │   └── v1/
│   │       ├── analytics.py             # Footfall analytics endpoints
│   │       ├── cameras.py               # Camera CRUD, streaming & worker controls
│   │       ├── departments.py           # Dynamic department management endpoints
│   │       ├── employees.py             # Employee registration, gallery & attendance
│   │       └── identity.py              # Biometric retention status & cleanup
│   ├── db/
│   │   ├── session.py                   # Database engine & session generator
│   │   ├── base.py                      # Declarative Base
│   │   └── models/                      # SQLAlchemy models
│   │       ├── camera.py                # Camera entity & entry-line config
│   │       ├── department.py            # Dynamic showroom department entity
│   │       ├── employee.py              # Employee entity & gallery embeddings
│   │       ├── employee_attendance.py   # Raw punches & daily attendance roll
│   │       ├── person_identity.py       # 14-day rolling biometric identity (pgvector)
│   │       ├── visitor_track.py         # Visitor tracklets
│   │       ├── footfall_event.py        # Footfall events (UNIQUE constraint)
│   │       ├── detection_event.py       # Frame-by-frame detections
│   │       └── processing_run.py        # Edge worker session telemetry
│   ├── repositories/                    # Database repositories
│   ├── schemas/                         # Pydantic validation schemas
│   ├── services/                        # Business logic (camera, analytics, employee, retention)
│   ├── vision/                          # Server-side vision pipeline & stream readers
│   │   ├── detector.py                  # RF-DETR & YOLO model wrapper
│   │   ├── tracker.py                   # Multi-object tracker integration
│   │   ├── face_embedder.py             # ArcFace 512-D embedding extractor
│   │   ├── employee_recognizer.py       # Biometric matching & voting manager
│   │   └── vector_store.py              # pgvector search manager
│   └── workers/
│       └── video_worker.py              # Server-side background ingestion worker
├── dashboard/                           # Modern React 19 Web Dashboard
│   ├── src/
│   │   ├── components/                  # Live preview, overview, visitors, attendance, settings
│   │   ├── context/                     # ShowroomContext & AuthContext state managers
│   │   ├── services/                    # Camera, department, employee & analytics API clients
│   │   └── types/                       # TypeScript interfaces
│   ├── server/
│   │   ├── index.ts                     # Express authentication backend (:5000)
│   │   └── db.ts                        # PostgreSQL client connection pool
│   ├── package.json
│   └── vite.config.ts
├── scripts/                             # Multi-Service Orchestration & Benchmarking
│   ├── run_all.py                       # Single-command runner for FastAPI, Auth, and Dashboard
│   ├── run_detector.bat                 # Quick-launch batch script for local_detector.py
│   ├── benchmark_trackers.py            # ByteTrack vs BoT-SORT performance benchmark
│   ├── download_models.py               # Automated weights downloader
│   ├── start_services.bat / .ps1        # Multi-window startup scripts for Windows
│   └── stop_services.bat / .ps1         # Safe shutdown scripts for all background services
├── static/                              # Static file uploads (employee profile pictures)
├── alembic/                             # Database migration scripts
├── local_detector.py                    # Standalone Local Edge AI Video Processing Engine
├── build_exe.py                         # PyInstaller standalone EXE compiler
├── bytetrack.yaml                       # ByteTrack tracker configuration
├── botsort.yaml                         # BoT-SORT tracker configuration
├── benchmark_results.md                 # Tracker benchmark evaluation results
├── camera_config.example.json           # Camera stream, entry line & section configuration template
├── config.md                            # Comprehensive configuration reference guide
├── flow.md                              # Detailed architectural & data flow documentation
├── requirements.txt                     # Python dependencies
└── README.md                            # Main project documentation
```

---

## 5. Quick Start Guide

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** & npm
- **PostgreSQL 14+** with `pgvector` extension installed

---

### Option A: One-Command Full Stack Startup (Recommended)

To launch the **FastAPI Backend (:8000)**, **Auth Server (:5000)**, and **Vite Dashboard (:5173)** in unified terminal output:

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Dashboard dependencies
cd dashboard && npm install && cd ..

# 3. Launch all services simultaneously
python scripts/run_all.py
```

*Or on Windows, simply double-click [scripts/start_services.bat](file:///c:/Users/capermint/Desktop/Abhishek/jms/scripts/start_services.bat) (and use [scripts/stop_services.bat](file:///c:/Users/capermint/Desktop/Abhishek/jms/scripts/stop_services.bat) to stop).*

---

### Option B: Manual Step-by-Step Setup

#### Step 1: Environment Configuration
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Key settings:
```ini
DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/jms_live"
DETECTION_MODEL_TYPE="yolo"
YOLO_MODEL_PATH="models/yolo26s.pt"
DETECTION_CONFIDENCE_THRESHOLD=0.31
DETECTION_IMGSZ=1024
DEVICE="cuda:0"  # Or "cpu" for non-GPU environments
TRACKER_TYPE="bytetrack.yaml"

# Biometrics & 14-Day Rolling Retention
ARCFACE_MODEL_PATH="models/w600k_r50.onnx"
FACE_SIMILARITY_THRESHOLD=0.32
FACE_RETENTION_DAYS=14
RETENTION_AUTO_CLEANUP_ENABLED=true
```

Configure `dashboard/.env`:
```bash
cp dashboard/.env.example dashboard/.env
```

#### Step 2: Database Migration
```bash
alembic upgrade head
```

#### Step 3: Run the Services
**Terminal 1 — FastAPI Backend:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- OpenAPI Documentation: `http://localhost:8000/docs`

**Terminal 2 — Auth Server:**
```bash
cd dashboard
npm run server
```

**Terminal 3 — React Dashboard:**
```bash
cd dashboard
npm run dev
```
- Web Dashboard: `http://localhost:5173`

---

## 6. Running the Local Edge AI Detector

The Edge Detector ([`local_detector.py`](file:///c:/Users/capermint/Desktop/Abhishek/jms/local_detector.py)) processes camera video feeds locally, performs transformer or CNN person detection, extracts face embeddings, classifies staff vs. visitors, validates line crossings, and synchronizes footfall events to the backend.

### Camera Configuration
Copy `camera_config.example.json` to `camera_config.json` and configure your RTSP camera channels, entry lines, and zones:
```json
{
  "default_channel": "101",
  "cameras": {
    "101": {
      "name": "Main Entrance Camera",
      "source": "rtsp://admin:password@192.168.1.101:554/Streaming/Channels/101",
      "camera_id": "88970b31-d885-4fd5-a24a-bdc969c96240",
      "entry_line": {
        "enabled": true,
        "p1": [389, 250],
        "p2": [551, 113],
        "direction": "top_to_bottom"
      }
    }
  }
}
```

### Starting the Detector
Run the detector with the default YOLO configuration:
```bash
python local_detector.py --channel 101
```

Run in **RF-DETR v2 mode** (Transformer Detection):
```bash
python local_detector.py --channel 101 --v2 --rf-detr-variant small
```

Run headless in the background (no desktop GUI window):
```bash
python local_detector.py --channel 101 --headless
```

Select a specific tracker (ByteTrack vs BoT-SORT):
```bash
python local_detector.py --channel 101 --tracker botsort
```

### Detector CLI Arguments Reference
| Flag | Description | Default |
|---|---|---|
| `--channel`, `-c` | Camera channel key defined in `camera_config.json` | `101` |
| `--source`, `-s` | Direct RTSP URL, video file path, or webcam index (`0`) | None |
| `--v2` | Enable Roboflow RF-DETR Detection Transformer mode | `False` |
| `--v1` | Enable YOLO CNN mode | `True` |
| `--model-type` | Detector architecture: `'yolo'`, `'rtmdet'`, `'rf-detr'`, `'rtdetr'`, or `'auto'` | `'yolo'` |
| `--rf-detr-variant`| RF-DETR model variant: `'nano'`, `'small'`, `'medium'`, `'base'`, `'large'` | `'small'` |
| `--model`, `-m` | Path to detector weights checkpoint | `'models/yolo26s.pt'` |
| `--conf` | Person detection confidence threshold | `0.31` |
| `--imgsz` | Inference input resolution in pixels | `1024` |
| `--tracker` | Multi-object tracker: `'bytetrack'`, `'botsort'`, or path to YAML | `bytetrack.yaml` |
| `--face-model` | Path to SCRFD or YuNet face model | `'models/scrfd_2.5g_bnkps.onnx'` |
| `--face-conf` | Neural face detection confidence threshold | `0.36` |
| `--face-scale` | Face bounding box scale multiplier | `1.05` |
| `--sim-thresh` | Biometric face cosine similarity threshold for staff identification | `0.32` |
| `--show-line` | Display virtual entry line & calculate crossings | `True` |
| `--deadband` | Pixel displacement deadband to eliminate stationary jitter | `6.0` |
| `--smoothing-alpha`| EMA coordinate smoothing factor (0.1 to 1.0) | `0.35` |
| `--face-persistence`| Frames to buffer face detections across temporary drops | `8` |
| `--body-persistence`| Missing frames before an un-detected person track is purged | `6` |
| `--standalone-faces`| Enable detection of standalone faces behind cubicle partitions | `False` |
| `--head-fallback` | Enable synthetic anatomical head box fallback when face is turned away | `False` |
| `--headless` | Run as background service without local OpenCV window | `False` |
| `--api-url` | Base URL of FastAPI backend for syncing events | `'http://127.0.0.1:8000'` |

---

## 7. Packaging into Standalone Windows Executable

To compile `local_detector.py` into a portable Windows executable using PyInstaller:

```bash
python build_exe.py
```

The resulting executable will be placed in:
```text
dist/CapermintLocalDetector/CapermintLocalDetector.exe
```

---

## 8. API Reference Summary

All registered endpoints exposed by the FastAPI backend:

### Health & System
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API health check and service status |

### CCTV Cameras & Edge Synchronization
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/cameras` | Register a new CCTV camera (masks stream credentials) |
| `GET` | `/api/v1/cameras` | List all configured cameras |
| `GET` | `/api/v1/cameras/{camera_id}` | Retrieve single camera details |
| `PATCH` | `/api/v1/cameras/{camera_id}` | Update camera settings / active status |
| `POST` | `/api/v1/cameras/{camera_id}/start` | Start server-side background video ingestion worker |
| `POST` | `/api/v1/cameras/{camera_id}/stop` | Stop server-side background video ingestion worker |
| `GET` | `/api/v1/cameras/{camera_id}/status` | Check processing status, FPS, and frame counters |
| `GET` | `/api/v1/cameras/{camera_id}/frame` | Retrieve latest JPEG frame snapshot from camera |
| `GET` | `/api/v1/cameras/{camera_id}/stream` | Stream live MJPEG video with AI annotations |
| `POST` | `/api/v1/cameras/{camera_id}/footfall` | Synchronize footfall entry events from edge detector |
| `POST` | `/api/v1/cameras/{camera_id}/heartbeat`| Synchronize periodic runtime telemetry and occupancy |

### Footfall Analytics
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/analytics/footfall` | Retrieve total footfall counts with date/camera filters |
| `GET` | `/api/v1/analytics/footfall/daily` | Retrieve daily aggregated footfall entries |

### Dynamic Departments & Zones
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/departments` | List all showroom detection departments / sections |
| `POST` | `/api/v1/departments` | Create a new detection department |
| `PATCH` | `/api/v1/departments/{department_id}` | Update department configuration (name, color, polygon) |
| `DELETE` | `/api/v1/departments/{department_id}` | Delete a detection department |

### Employee Biometrics & Management
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/employees` | List registered employees with department filters |
| `POST` | `/api/v1/employees` | Register a new employee |
| `GET` | `/api/v1/employees/{employee_id}` | Get employee profile details |
| `PATCH` | `/api/v1/employees/{employee_id}` | Update employee profile or active status |
| `POST` | `/api/v1/employees/{employee_id}/face` | Enroll biometric embedding via Base64 or vector |
| `POST` | `/api/v1/employees/{employee_id}/face/upload` | Upload employee face photos (JPG/PNG) |
| `POST` | `/api/v1/employees/{employee_id}/photos` | Add multi-angle gallery photos to strengthen recognition |

### Employee Attendance
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/attendance/punch` | Record employee check-in, check-out, or presence punch |
| `GET` | `/api/v1/attendance/daily` | Retrieve daily employee attendance roll and hours worked |

### Biometric Privacy & Data Retention
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/identity/cleanup` | Execute 14-day biometric rolling retention cleanup |
| `GET` | `/api/v1/identity/status` | Check retention policy status and active face counts |

---

## 9. Automated Testing & Benchmarking

### Running Unit & Integration Tests
Execute the comprehensive test suite with `pytest`:
```bash
python -m pytest tests -v
```

The test suite validates:
- API routing, health checks, and camera CRUD
- Credential masking and sanitization across logs and responses
- Person detection edge cases and confidence thresholding
- ByteTrack and BoT-SORT temporal tracking continuity across frames
- Directional virtual entry-line crossing geometry & wrong-direction rejections
- Duplicate footfall event suppression (in-memory and database levels)
- Employee face embedding extraction and similarity thresholding
- 14-day rolling retention cleanup logic

### Multi-Object Tracker Benchmarking
Benchmark ByteTrack vs BoT-SORT latency, ID switching, and FPS:
```bash
python scripts/benchmark_trackers.py
```
Benchmark results and deployment recommendations are detailed in [benchmark_results.md](file:///c:/Users/capermint/Desktop/Abhishek/jms/benchmark_results.md).
