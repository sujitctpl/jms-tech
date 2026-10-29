# Showroom Visitor Intelligence & Footfall Analytics Platform

An enterprise-grade, edge-AI hybrid visitor intelligence and showroom analytics platform. The system leverages existing CCTV / RTSP camera infrastructure to deliver real-time person detection (YOLOv11), persistent multi-object tracking (ByteTrack), face/head detection fallbacks (SCRFD / YuNet), directional virtual entry-line crossing, dwell-time analytics, showroom zone occupancy, and an interactive React 19 web dashboard.

---

## 1. System Overview & Architecture

The platform operates on a high-efficiency **Edge-AI Hybrid Architecture**:

```mermaid
graph TD
    subgraph Edge Layer [Showroom Edge / Local Machine]
        CCTV["CCTV / IP Cameras (RTSP TCP)"]
        Detector["Edge AI Engine (local_detector.py)\n• YOLOv11 Person Detection\n• SCRFD / YuNet Face Tracking\n• ByteTrack & Bounding Box Smoothing\n• Entry/Exit Line & Zone Dwell Timers\n• Local MJPEG Stream Server"]
        SyncQueue["Background Sync Client (Queue)"]
    end

    subgraph Backend Layer [FastAPI & Express Backend]
        API["FastAPI Analytics Backend (:8000)\n• Camera Management & Masked RTSP\n• Footfall & Daily Analytics\n• Telemetry & Worker Health"]
        AuthSrv["Express Auth Server (:5000)\n• User & Admin Authentication\n• PostgreSQL Session Pool"]
        DB[("PostgreSQL Database (:5432)\n• cameras\n• visitor_tracks\n• footfall_events\n• detection_events")]
    end

    subgraph Client Layer [Modern Web UI]
        Dashboard["React 19 Dashboard (:5173)\n• Live CCTV Preview & AI Overlays\n• Real-Time Footfall & Peak Hours\n• Attendance & Visitor Tagging\n• Section & Dwell Time Analytics"]
    end

    CCTV -->|RTSP TCP Stream| Detector
    Detector -->|Events & Telemetry| SyncQueue
    SyncQueue -->|REST API JSON| API
    API --> DB
    AuthSrv --> DB
    Dashboard -->|Analytics & Cameras| API
    Dashboard -->|Auth| AuthSrv
    Dashboard -.->|MJPEG Live Stream| Detector
```

Detailed architectural flows, algorithms, and math models are documented in [flow.md](file:///c:/Users/capermint/Desktop/Abhishek/jms/flow.md).

---

## 2. Key Features

- **Local Edge AI Inference (`local_detector.py`)**: Runs YOLOv11 person detection directly on local hardware without sending raw video feeds over the cloud.
- **RTSP TCP Buffer Management**: Forces TCP transport globally (`rtsp_transport;tcp`) and uses a non-blocking threaded frame grabber (`ThreadedRTSPReader`) to prevent frame delay and macroblock artifacts.
- **Persistent Tracking & Smoothing**: Powered by ByteTrack and Euclidean centroid tracking with EMA (Exponential Moving Average) bounding box smoothing and deadband jitter suppression.
- **Face & Torso Alignment**: Integrated SCRFD / YuNet face detection constrained within upper-torso regions to prevent false face detections on posters or clothing graphics.
- **Virtual Entry-Line Crossing**: 2D ground-plane foot contact point calculation ($\frac{x_1 + x_2}{2}, y_2$) with strict directional validation (`top_to_bottom`, `bottom_to_top`, `left_to_right`, `right_to_left`).
- **Strict Duplicate Prevention**: Multi-tier duplicate suppression at the in-memory tracker level and database level (`UNIQUE` on `visitor_track_id` in `footfall_events`).
- **Showroom Zone Occupancy & Dwell Timers**: Configurable polygon sections (e.g. Diamond, Gold, Silver counters) tracking real-time visitor dwell duration.
- **Live Video Streaming & Overlays**: Built-in HTTP MJPEG streaming server for low-latency live preview with AI bounding boxes directly in the web dashboard.
- **Modern Web Dashboard (`dashboard/`)**: Built with React 19, Vite, TypeScript, and Tailwind CSS. Provides live previews, visitor tagging, quick add visitor, attendance management, peak hour heatmaps, and exportable reports.
- **Single-Command Orchestration**: Launch all 3 services (FastAPI, Express Auth, and Vite Dashboard) concurrently via `scripts/run_all.py` or Windows `.bat` / `.ps1` scripts.
- **Desktop Executable Packaging**: PyInstaller builder script (`build_exe.py`) to compile the edge detector into a standalone Windows `.exe`.

---

## 3. Technology Stack

| Component | Technologies |
|---|---|
| **Edge AI Engine** | Python 3.11+, Ultralytics YOLOv11, ByteTrack, OpenCV, NumPy |
| **Backend API** | FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2, PostgreSQL (`psycopg` 3) |
| **Auth Service** | Node.js, Express 5, TypeScript (`tsx`), `pg`, bcryptjs |
| **Frontend Web** | React 19, TypeScript, Vite, Tailwind CSS v4, Recharts, Lucide Icons |
| **Packaging & Scripts** | PyInstaller, Python subprocess, Windows Batch, PowerShell |

---

## 4. Project Structure

```text
jms/
├── app/                            # Central FastAPI Backend Service
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
│   │   └── models/                 # SQLAlchemy models (Camera, VisitorTrack, FootfallEvent, etc.)
│   ├── schemas/                    # Pydantic validation schemas
│   ├── repositories/               # Database repositories
│   ├── services/                   # Business logic (camera_service, analytics_service)
│   ├── vision/                     # Server-side vision pipeline & stream readers
│   └── workers/
│       └── video_worker.py         # Server-side background ingestion worker
├── dashboard/                      # Modern React 19 Web Dashboard
│   ├── src/
│   │   ├── components/             # Live preview, overview, visitors, attendance, peak hours, reports
│   │   ├── context/                # ShowroomContext & AuthContext state managers
│   │   ├── services/               # Camera and analytics API services
│   │   └── types/                  # TypeScript interface definitions
│   ├── server/
│   │   ├── index.ts                # Express authentication backend (:5000)
│   │   └── db.ts                   # PostgreSQL client connection pool
│   ├── package.json
│   └── vite.config.ts
├── scripts/                        # Multi-Service Orchestration Scripts
│   ├── run_all.py                  # Single-command runner for FastAPI, Auth, and Dashboard
│   ├── run_detector.bat            # Quick-launch batch script for local_detector.py
│   ├── start_services.bat / .ps1   # Multi-window startup scripts for Windows
│   └── stop_services.bat / .ps1    # Safe shutdown scripts for all background services
├── alembic/                        # Database migration scripts
├── local_detector.py               # Standalone Local Edge AI Video Processing Engine
├── build_exe.py                    # PyInstaller standalone EXE compiler
├── camera_config.example.json      # Camera stream, entry line & section configuration template
├── flow.md                         # Detailed architectural & data flow documentation
├── requirements.txt                # Python backend & detector dependencies
└── README.md                       # Main project documentation
```

---

## 5. Quick Start Guide

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** & npm
- **PostgreSQL 14+** running locally or remotely

---

### Option A: One-Command Full Stack Startup (Recommended)

To launch the **FastAPI Backend (8000)**, **Auth Server (5000)**, and **Vite Dashboard (5173)** in unified output:

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
DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/showroom_db"
YOLO_MODEL_PATH="yolov8n.pt"
DETECTION_CONFIDENCE_THRESHOLD=0.45
DEVICE="cpu"  # Or "cuda" / "0" for GPU acceleration
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

The Edge Detector (`local_detector.py`) processes camera video streams, computes tracks and line crossings, and sends count events to the FastAPI backend.

### Camera Configuration
Copy `camera_config.example.json` to `camera_config.json` and configure your RTSP camera channels, entry lines, and zones:
```json
{
  "default_channel": "201",
  "cameras": {
    "201": {
      "name": "Main Entrance Camera",
      "source": "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/201",
      "camera_id": "88970b31-d885-4fd5-a24a-bdc969c96240",
      "entry_line": {
        "line": [[200.0, 450.0], [900.0, 450.0]],
        "direction": "top_to_bottom"
      }
    }
  }
}
```

### Starting the Detector
Run the detector for a specific channel:
```bash
python local_detector.py --channel 201
```

Or run headless in the background (no desktop GUI window):
```bash
python local_detector.py --channel 201 --headless
```

*Or on Windows, launch [scripts/run_detector.bat](file:///c:/Users/capermint/Desktop/Abhishek/jms/scripts/run_detector.bat).*

### Useful Detector CLI Arguments
| Flag | Description | Default |
|---|---|---|
| `--channel`, `-c` | Camera channel key defined in `camera_config.json` | `201` |
| `--source`, `-s` | Direct RTSP URL, video file path, or webcam index (`0`) | None |
| `--conf` | Person detection confidence threshold | `0.35` |
| `--model`, `-m` | YOLO model weights (`yolo11n.pt`, `yolov8n.pt`, etc.) | `yolo11n.pt` |
| `--headless` | Run as background service without local OpenCV window | `False` |
| `--api-url` | Base URL of FastAPI backend for syncing events | `http://localhost:8000` |
| `--face-scale` | Multiplier for face detection scanning scale | `1.0` |
| `--deadband` | Pixel deadband filter to eliminate stationary jitter | `3.5` |

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

Registered routes defined centrally in `app/api/routes.py`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API health status check |
| `POST` | `/api/v1/cameras` | Register a new CCTV camera (masks stream credentials) |
| `GET` | `/api/v1/cameras` | List all configured cameras |
| `GET` | `/api/v1/cameras/{camera_id}` | Retrieve single camera details |
| `PATCH` | `/api/v1/cameras/{camera_id}` | Update camera settings / active status |
| `POST` | `/api/v1/cameras/{camera_id}/start` | Start server-side background video ingestion |
| `POST` | `/api/v1/cameras/{camera_id}/stop` | Stop server-side background video ingestion |
| `GET` | `/api/v1/cameras/{camera_id}/status` | Check processing run status & frame counters |
| `GET` | `/api/v1/analytics/footfall` | Retrieve total footfall counts with date/camera filters |
| `GET` | `/api/v1/analytics/footfall/daily` | Retrieve daily aggregated footfall entries |

---

## 9. Automated Testing

Run the test suite via `pytest`:
```bash
python -m pytest tests -v
```

The test suite validates:
- Health check and API routing
- Camera CRUD, credential masking, and validation
- Pre-processing camera configuration constraints
- Database models, foreign keys, and unique constraints
- Person detection edge cases and confidence thresholding
- ByteTrack temporal tracking continuity across frames
- Virtual entry-line crossing geometry & wrong-direction rejections
- Duplicate footfall event prevention
- Background worker lifecycle and status tracking
- Total and daily footfall analytics aggregations
