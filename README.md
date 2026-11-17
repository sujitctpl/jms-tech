# Showroom Visitor Intelligence & Footfall Analytics Platform

An enterprise-grade, edge-AI hybrid visitor intelligence and showroom analytics platform. The system leverages existing CCTV / RTSP camera infrastructure to deliver real-time person detection powered by **Roboflow RF-DETR** (Real-Time Detection Transformer with DINOv2 backbone, with Ultralytics YOLO fallback), multi-object tracking (**ByteTrack** and **BoT-SORT** with motion compensation), biometric face recognition (**ArcFace** 512-D embeddings with **SCRFD** / **YuNet** face detection), staff vs. visitor segregation, automated employee attendance logging, dynamic department zone management, visitor journey & dwell tracking, directional virtual entry-line crossing, visitor departure detection, and an interactive **React 19** web dashboard.

---

## 1. System Overview & Architecture

The platform operates on a high-efficiency **Edge-AI Hybrid Architecture**:

```mermaid
graph TD
    subgraph Edge Layer [Showroom Edge / Local Machine]
        CCTV["CCTV / IP Cameras (RTSP TCP)"]
        Detector["Edge AI Engine (local_detector.py)\n• RF-DETR Transformer / YOLO Detection\n• SCRFD / YuNet Face Tracking\n• ArcFace 512-D Biometric Extraction\n• ByteTrack / BoT-SORT Tracking\n• Multi-Frame Best-Shot Photo Selection\n• Visitor Zone Dwell & Journey Tracking\n• Directional Entry & Exit Line Crossing\n• Local MJPEG Stream Server\n• Zero-Downtime Hot-Reloading"]
        SyncQueue["Background Sync Client (Memory Queue)"]
    end

    subgraph Backend Layer [FastAPI & Express Backend]
        API["FastAPI Analytics Backend (:8000)\n• Dynamic Camera Configuration & Stream Probing\n• Footfall Analytics & Real-Time Telemetry\n• Visitor Ingestion, Tagging & Zone Updates\n• Dynamic Department Management\n• Employee Registration & Attendance\n• Biometric Rolling Retention Worker"]
        AuthSrv["Express Auth Server (:5000)\n• User & Admin Authentication\n• PostgreSQL Session Pool"]
        DB[("PostgreSQL Database (:5432)\n• cameras & departments\n• employees & employee_attendance\n• person_identities (pgvector 512)\n• visitor_tracks & footfall_events")]
    end

    subgraph Client Layer [Modern Web UI]
        Dashboard["React 19 Dashboard (:5173)\n• Live CCTV Preview & AI Overlays\n• Dynamic Camera & AI Settings Panel\n• Visit Purpose Tagging & High-Res Lightbox\n• Real-Time Zone Dwell & Customer Journey\n• Staff Attendance & Daily Rolls\n• Section Dwell Time & Exportable Reports"]
    end

    CCTV -->|RTSP TCP Stream| Detector
    Detector -->|Events, Snapshots, Dwell & Punches| SyncQueue
    SyncQueue -->|REST API JSON| API
    API --> DB
    AuthSrv --> DB
    Dashboard -->|Analytics, Cameras, Visitors & Staff| API
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
  - **Robust Fallback Centroid & Kalman Tracker**: Velocity-prediction fallback ensuring tracking continuity across temporary drops.
- **Biometric Face Recognition & Staff vs. Visitor Segregation**:
  - Neural face detection via **SCRFD** (`scrfd_2.5g_bnkps.onnx`) or **YuNet**.
  - **ArcFace** (`w600k_r50.onnx`) 512-dimensional feature embedding extraction.
  - Real-time cosine similarity matching against registered employee vector gallery.
  - Temporal voting buffers (`EMPLOYEE_VOTE_WINDOW = 5`) ensuring zero false-positive visitor misclassifications.
- **Optimized Visitor Photo Capture & Manual Tagging Pipeline**:
  - **Multi-Frame Best-Shot Selection**: Evaluates Laplacian sharpness, pixel area, pose frontality (eye horizontal symmetry and nose centering), and illumination balance into a unified quality score $Q$.
  - **Dynamic Progressive Upgrades**: Dispatches an instant snapshot on first frame for zero-latency UI updates, then automatically upgrades to a higher-resolution, frontal shot if quality improves by $\ge 25\%$ during the first 3.5s walk-in window.
  - **Adaptive Bust & Torso Framing**: Expanded portrait crop capturing head, hair, collar, and upper-chest context at $320\times320\text{ px}$ (JPEG Q95) for effortless visual recognition by showroom staff.
  - **CLAHE Backlight Compensation**: Corrects severe silhouettes caused by sunny showroom glass facades via adaptive histogram equalization on the LAB luminance channel.
  - **Automatic Clothing Appearance Tagging**: Evaluates upper-body HSV color distribution to generate human-readable attire badges (e.g. *Blue / Navy Top*, *White Shirt*, *Black Attire*).
  - **High-Res Lightbox & Modal Tagging UX**: Integrated zoom lightbox in `VisitPurposeView` and full visitor portrait card in `TagVisitorModal` for rapid 1-click visit purpose assignment.
- **Real-Time Visitor Journey & Zone Dwell Tracking**:
  - **`VisitorZoneTracker`**: Continuously tracks customer movement and accumulates dwell time across custom departments (e.g. Sales Section, Luxury Lounge, Diamond Counter) and the general showroom floor (`Main Showroom`).
  - **Live Dwell Formatting & Periodic Sync**: Synchronizes formatted durations (e.g. `12m 45s`, `35s`) with FastAPI backend (`POST /api/v1/visitors/update-zone`) every 4.0 seconds.
  - **Directional Reverse Exit Detection**: Crossing the entrance line in reverse detects visitor departure in Indian Standard Time (IST, UTC+05:30), logs final dwell breakdown across all visited zones, and persists departure time to CRM.
- **Visitor Waiting Timer & Unattended SLA Breach Monitoring**:
  - **Perspective-Calibrated Staff Proximity Filter (`is_staff_near_box`)**: Dynamically measures ground-plane distance in meters between visitor feet and staff feet using bounding box height scaling ($\text{Scale}_{\text{px/m}} = \frac{\bar{h}}{1.70}$). A visitor within $2.2\text{ m}$ is considered actively attended.
  - **2-Parameter Unattended Logic**: Customer waiting time accumulates only when the visitor is **unassisted** (no staff within $2.2\text{ m}$) **and outside any designated department counter box**.
  - **Three-Tier Visual HUD Alerting**:
    - **Normal Waiting ($< 5\text{ mins}$)**: Neon Emerald Green bounding box `(0, 240, 110)` with `⏱️ WAITING [MMm SSs]`.
    - **Amber Warning ($5\text{ to }15\text{ mins}$)**: Amber border `(0, 140, 255)` with `⚠️ WAITING [MMm SSs]` to alert floor staff.
    - **Critical SLA Breach ($> 15\text{ mins}$ / $900\text{s}$)**: Bright Red alert box `(0, 0, 255)` and red badge `🚨 UNATTENDED > 15m [MMm SSs]` with immediate console alarm.
  - **Dashboard SLA Breach & Waiting Badges**: Live counters, waiting duration pills, and unattended alerts surfaced across `OverviewView`, `VisitorsView`, and `VisitPurposeView`.
- **Dynamic Multi-Camera Handover & Cross-Camera Re-Identification**:
  - **CrossCameraTracker & Global Identity Synchronization**: Coordinates tracks between entrance cameras (`is_entrance: true`) and interior section cameras (`is_entrance: false`) via `.cross_camera_state.json`.
  - **ArcFace Cosine Matching ($\ge 0.55$)**: Re-identifies customers moving from entrance to floor sections without creating duplicate footfall counts.
  - **Isolated GPU VRAM Allocation**: Automatic memory fraction limit ($42\%$) ensures multi-camera instances run concurrently on 4GB consumer GPUs without OOM errors.
- **Dynamic Camera Configuration & Edge Hot-Reloading**:
  - **Full Web UI Management**: Configure RTSP feeds, camera names, confidence thresholds, and entry lines directly in the React Dashboard via `CameraSettingsSection.tsx`.
  - **Pre-Flight Stream Testing**: Dedicated `/api/v1/cameras/test-stream` endpoint opens OpenCV streams on demand to verify connectivity, resolution ($W\times H$), and framerate (FPS) before deployment.
  - **Entrance vs. Floor Camera Role Separation**: Flags camera channel as `is_entrance: true/false`. Entrance cameras count walk-in footfall and detect exits; in-store floor cameras track zone occupancy and dwell times without double-counting store footfall.
  - **Zero-Downtime Hot-Reloading**: `local_detector.py` watches `camera_config.json` timestamp and reloads lines, exclusion zones, and departments without dropping active RTSP connections.
  - **Profile Archiving**: Deleting a camera channel automatically archives its calibrated geometric boundaries and entry line to `archived_camera_profiles` for seamless restoration.
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
  - Live CCTV preview, dynamic camera settings, dynamic department configuration, employee attendance management, peak hour heatmaps, visitor tagging, and exportable reports.
- **Single-Command & Batch Orchestration**:
  - Concurrently launch all 3 services (FastAPI, Express Auth, and Vite Dashboard) via `python scripts/run_all.py` or Windows `.bat` / `.ps1` scripts.
  - 1-Click edge detector launch via [`start_detector.bat`](file:///c:/Users/capermint/Desktop/Abhishek/jms/start_detector.bat).
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
│   │       ├── cameras.py               # Camera CRUD, config sync, stream testing & worker controls
│   │       ├── departments.py           # Dynamic department management endpoints
│   │       ├── employees.py             # Employee registration, gallery & attendance
│   │       ├── identity.py              # Biometric retention status & cleanup
│   │       └── visitors.py              # Visitor ingestion, photo updates, tagging & zone dwell
│   ├── db/
│   │   ├── session.py                   # Database engine & session generator
│   │   ├── base.py                      # Declarative Base
│   │   └── models/                      # SQLAlchemy models
│   │       ├── camera.py                # Camera entity & entry-line config
│   │       ├── department.py            # Dynamic showroom department entity
│   │       ├── employee.py              # Employee entity & gallery embeddings
│   │       ├── employee_attendance.py   # Raw punches & daily attendance roll
│   │       ├── person_identity.py       # 14-day rolling biometric identity & visitor profile (pgvector)
│   │       ├── visitor_track.py         # Visitor tracklets
│   │       ├── footfall_event.py        # Footfall events (UNIQUE constraint)
│   │       ├── detection_event.py       # Frame-by-frame detections
│   │       └── processing_run.py        # Edge worker session telemetry
│   ├── repositories/                    # Database repositories
│   ├── schemas/                         # Pydantic validation schemas (camera, visitor, employee, etc.)
│   ├── services/                        # Business logic (camera, visitor, analytics, employee, retention)
│   ├── vision/                          # Server-side vision pipeline & stream readers
│   │   ├── detector.py                  # RF-DETR & YOLO model wrapper
│   │   ├── tracker.py                   # Multi-object tracker integration
│   │   ├── cross_camera_tracker.py      # Cross-camera entity coordination & global ID state
│   │   ├── face_embedder.py             # ArcFace 512-D embedding extractor
│   │   ├── employee_recognizer.py       # Biometric matching & voting manager
│   │   └── vector_store.py              # pgvector search manager
│   └── workers/
│       └── video_worker.py              # Server-side background ingestion worker
├── dashboard/                           # Modern React 19 Web Dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── cameras/                 # Live camera feeds, grids & overlays
│   │   │   ├── modals/                  # TagVisitorModal (portrait card, dwell, tagging)
│   │   │   ├── settings/                # CameraSettingsSection (RTSP configuration & hot reload)
│   │   │   ├── visit-purpose/           # VisitPurposeView (visitor table, zoom lightbox, filters)
│   │   │   ├── visitors/                # VisitorsView (demographics, journey & history)
│   │   │   └── attendance/              # Staff attendance roll & live punch logs
│   │   ├── context/                     # ShowroomContext & AuthContext state managers
│   │   ├── services/                    # Camera, visitor, department, employee & analytics API clients
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
├── static/                              # Static file uploads (employee and visitor photos)
│   └── uploads/
│       ├── employees/                   # Registered employee gallery photos
│       └── visitors/                    # 320x320 retina visitor portrait crops
├── tests/                               # Comprehensive Automated Test Suites
│   ├── test_waiting_timer.py            # Unit tests for waiting timer & SLA breach
│   ├── test_cross_camera_tracker.py     # Cross-camera tracking & re-ID tests
│   ├── test_returning_visitor.py        # Returning visitor recognition tests
│   └── test_identity_and_retention.py   # 14-day retention & biometric purge tests
├── .cross_camera_state.json             # Cross-camera persistent state registry
├── alembic/                             # Database migration scripts
├── local_detector.py                    # Standalone Local Edge AI Video Processing Engine
├── start_detector.bat                   # 1-Click Windows Batch Launcher for Single-Camera Detector
├── run_multi_camera.bat                 # Concurrent Multi-Camera Launcher for Windows CMD
├── run_multi_camera.ps1                 # Concurrent Multi-Camera Launcher for PowerShell
├── build_exe.py                         # PyInstaller standalone EXE compiler
├── bytetrack.yaml                       # ByteTrack tracker configuration
├── botsort.yaml                         # BoT-SORT tracker configuration
├── benchmark_results.md                 # Tracker benchmark evaluation results
├── camera_config.example.json           # Camera stream, entry line & section configuration template
├── camera_config.json                   # Active local camera & AI configuration
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

The Edge Detector ([`local_detector.py`](file:///c:/Users/capermint/Desktop/Abhishek/jms/local_detector.py)) processes camera video feeds locally, performs transformer or CNN person detection, extracts face embeddings, classifies staff vs. visitors, validates line crossings, manages visitor photo capture, tracks zone dwell times, and synchronizes footfall events to the backend.

### Camera Configuration
Copy `camera_config.example.json` to `camera_config.json` (or configure via the Dashboard Settings):
```json
{
  "default_channel": "101",
  "confidence": 0.31,
  "imgsz": 960,
  "cameras": {
    "101": {
      "name": "Main Entrance Camera",
      "source": "rtsp://admin:password@192.168.1.101:554/Streaming/Channels/101",
      "camera_id": "88970b31-d885-4fd5-a24a-bdc969c96240",
      "is_entrance": true,
      "confidence": 0.31,
      "min_feet_y": 0.08,
      "entry_line": {
        "enabled": true,
        "p1": [389, 250],
        "p2": [551, 113],
        "direction": "top_to_bottom"
      }
    },
    "201": {
      "name": "Sales Section Floor Camera",
      "source": "rtsp://admin:password@192.168.1.102:554/Streaming/Channels/101",
      "camera_id": "a1b2c3d4-e5f6-7890-1234-56789abcdef0",
      "is_entrance": false,
      "confidence": 0.31
    }
  }
}
```

### Starting the Detector (Single Camera)
On Windows, you can start the detector for your default channel with a single click using:
```bash
start_detector.bat
```
*(Or pass arguments: `start_detector.bat --channel 201`)*

Alternatively, launch using Python:
```bash
# Default YOLO mode on configured default channel
python local_detector.py --channel 101

# RF-DETR v2 mode (Transformer Detection)
python local_detector.py --channel 101 --v2 --rf-detr-variant small

# Run headless in the background (no desktop GUI window)
python local_detector.py --channel 101 --headless

# Select BoT-SORT tracker with Global Motion Compensation
python local_detector.py --channel 101 --tracker botsort
```

---

### Running Multiple Cameras Simultaneously

To process multiple camera feeds concurrently with independent preview windows on your desktop:

#### Option 1: Via Windows Batch Launcher
```cmd
# Launch Camera 101 (Entrance) and Camera 201 (Floor) concurrently:
run_multi_camera.bat 101 201

# Or run interactively to choose presets:
run_multi_camera.bat
```

#### Option 2: Via PowerShell Launcher
```powershell
# Launch Channels 101 and 201 in parallel:
.\run_multi_camera.ps1 101, 201

# Launch all 4 channels:
.\run_multi_camera.ps1 101, 201, 301, 401
```

#### Option 3: Via Separate Terminal Tabs
You can open separate terminal windows and run:
```powershell
# Terminal 1:
python local_detector.py --channel 101 --imgsz 640

# Terminal 2:
python local_detector.py --channel 201 --imgsz 640
```

Each process creates a dedicated desktop window:
- `Capermint Edge AI Detector - Cam 101`
- `Capermint Edge AI Detector - Cam 201`

---

### Multi-Camera Person & Visitor Re-Identification Flow

The platform features seamless cross-camera handover between entrance and interior floor cameras:

```mermaid
flowchart TD
    subgraph Entrance ["Camera 101 (Main Glass Entrance - is_entrance: true)"]
        E1["Person crosses Entrance Line"] --> E2["SCRFD Face Detection + ArcFace 512-D Embedding"]
        E2 --> E3{"Staff or Visitor?"}
        E3 -->|Employee| E4["Staff Attendance Checked In"]
        E3 -->|Visitor| E5["Mint Persistent ID (e.g. VIS-2026-1043)"]
        E5 --> E6["Save Face Vector & Portrait to PostgreSQL `person_identities`"]
    end

    subgraph CentralDB ["Central Database & Vector Store"]
        E6 --> DB[("PostgreSQL + pgvector Gallery")]
    end

    subgraph Interior ["Camera 201 (Floor / Sections - is_entrance: false)"]
        I1["Same person enters Camera 201 view"] --> I2["Detect Face & Extract 512-D Embedding"]
        I2 --> I3{"Cosine Match against Database?"}
        DB -.->|Query Active Embeddings| I3
        I3 -->|Staff Match| I4["Tagged as STAFF [Name] #ID<br/>Logs Section Dwell Time"]
        I3 -->|Visitor Match >= 0.55| I5["Matched to VIS-2026-1043<br/>Links Track & Updates Zone Dwell"]
        I5 --> I6["POST /api/v1/visitors/update-zone<br/>Footfall count is NOT duplicated"]
    end
```

1. **Entrance Camera (Channel 101, `is_entrance: true`)**:
   - Manages the virtual entry line crossing.
   - When a new customer enters, their 512-D ArcFace embedding and full-body portrait are registered in PostgreSQL (`person_identities` table) under a persistent ID (e.g., `VIS-2026-1043`).
   - If an employee enters, their attendance check-in is punched and visitor registration is suppressed.
2. **Interior Floor Camera (Channel 201, `is_entrance: false`)**:
   - Runs neural detection and face embedding extraction locally.
   - Queries the active showroom visitor gallery via `visitor_vector_store.find_best_visitor_match()`.
   - When cosine similarity is $\ge 0.55$, the track is automatically linked to `VIS-2026-1043`.
   - The HUD label updates to `Person #ID | VIS-2026-1043 | [Zone Name]`.
   - `VisitorZoneTracker` sends periodic zone dwell updates to `/api/v1/visitors/update-zone` without incrementing walk-in footfall counts.

---

### Multi-Process GPU & Runtime Optimizations

- **VRAM Memory Capping (`torch.cuda.set_per_process_memory_fraction`)**: Automatically restricts each detector process to $42\%$ of available VRAM (~$1.7\text{ GB}$ on a 4GB GTX 1650), preventing CUDA out-of-memory errors and PCIe paging thrashing when multiple cameras run in parallel.
- **Anti-Freeze Windows Message Pump**: Calls `cv2.waitKey(10)` during stream-wait cycles so Windows always services the GUI message queue, preventing windows from turning into "Not Responding" state.
- **Resilient RTSP Auto-Reconnection**: Detects consecutive dropped frames ($> 1.5\text{s}$) and automatically cycles the RTSP TCP socket without crashing or freezing.

### Detector CLI Arguments Reference
| Flag | Description | Default |
|---|---|---|
| `--channel`, `-c` | Camera channel key defined in `camera_config.json` | `101` |
| `--source`, `-s` | Direct RTSP URL, video file path, or webcam index (`0`) | Loaded from channel |
| `--v2` | Enable Roboflow RF-DETR Detection Transformer mode | `False` |
| `--v1` | Enable YOLO CNN mode | `True` |
| `--model-type` | Detector architecture: `'yolo'`, `'rtmdet'`, `'rf-detr'`, `'rtdetr'`, or `'auto'` | `'yolo'` |
| `--rf-detr-variant`| RF-DETR model variant: `'nano'`, `'small'`, `'medium'`, `'base'`, `'large'` | `'small'` |
| `--model`, `-m` | Path to detector weights checkpoint | `'models/yolo26s.pt'` |
| `--conf` | Person detection confidence threshold | `0.31` |
| `--imgsz` | Inference input resolution in pixels | `960` / `1024` |
| `--tracker` | Multi-object tracker: `'bytetrack'`, `'botsort'`, or path to YAML | `bytetrack.yaml` |
| `--face-model` | Path to SCRFD or YuNet face model | `'models/scrfd_2.5g_bnkps.onnx'` |
| `--face-conf` | Neural face detection confidence threshold | `0.36` |
| `--face-scale` | Face bounding box scale multiplier | `1.05` |
| `--sim-thresh` | Biometric face cosine similarity threshold for staff identification | `0.32` |
| `--show-line` | Display virtual entry line & calculate crossings | `True` |
| `--deadband` | Pixel displacement deadband to eliminate stationary jitter | `6.0` |
| `--smoothing-alpha`| EMA coordinate smoothing factor (0.1 to 1.0) | `0.35` |
| `--face-persistence`| Frames to buffer face detections across temporary drops | `8` |
| `--body-persistence`| Missing frames before an un-detected person track is purged | `26` |
| `--standalone-faces`| Enable detection of standalone faces behind cubicle partitions | `False` |
| `--head-fallback` | Enable synthetic anatomical head box fallback when face is turned away | `False` |
| `--headless` | Run as background service without local OpenCV window | `False` |
| `--api-url` | Base URL of FastAPI backend for syncing events | `'http://127.0.0.1:8000'` |
| `waiting_cutoff_seconds` | Configurable in `camera_config.json`: Cutoff SLA threshold in seconds (e.g. `900.0` for 15 mins) | `900.0` |

---

### Visitor Waiting Timer & SLA Breach Monitoring Pipeline

```mermaid
flowchart LR
    A["Customer in Showroom"] --> B{"Inside Calibrated<br/>Department Box?"}
    B -- Yes --> C["Dwell Accumulates to Zone<br/>(Attended / In Consultation)"]
    B -- No --> D{"Showroom Staff<br/>Within 2.2m Proximity?"}
    D -- Yes --> C
    D -- No --> E["Customer Marked Unattended<br/>(⏱️ Waiting Timer Starts)"]
    E --> F{"Waiting Duration?"}
    F -- "< 5 mins" --> G["🟢 Emerald Green Box<br/>⏱️ WAITING [MMm SSs]"]
    F -- "5 to 15 mins" --> H["🟠 Amber Warning Box<br/>⚠️ WAITING [MMm SSs]"]
    F -- "> 15 mins (SLA Breach)" --> I["🚨 Bright Red Alert Box<br/>🚨 UNATTENDED > 15m"]
    I --> J["Console Alert Alarm +<br/>CRM & Dashboard SLA Breach Badge"]
```

- **Perspective Proximity Math**:
  $$\text{dist\_m} = \frac{\sqrt{(x_v - x_s)^2 + (y_v - y_s)^2}}{\max(1.0, \; \bar{h} / 1.70)}$$
- **Automatic Reset**: As soon as a registered employee steps within $2.2\text{ m}$ of the customer or the customer enters a department zone, the active waiting state pauses and transitions back to attended.

---

### Automated Testing Suite

The repository includes end-to-end automated test suites for all major vision and intelligence subsystems:

```bash
# Run all tests
pytest tests/ -v

# Run Waiting Timer & SLA Breach unit test
pytest tests/test_waiting_timer.py -v

# Run Cross-Camera Tracking test
pytest tests/test_cross_camera_tracker.py -v

# Run Returning Visitor identification test
pytest tests/test_returning_visitor.py -v

# Run Biometric Retention & Purge test
pytest tests/test_identity_and_retention.py -v
```

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

### CCTV Cameras & Dynamic Configuration
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/cameras` | List all configured cameras in the system |
| `POST` | `/api/v1/cameras` | Register a new CCTV camera (masks stream credentials) |
| `GET` | `/api/v1/cameras/config` | Retrieve dynamic camera & AI settings configuration from `camera_config.json` |
| `POST` | `/api/v1/cameras/config` | Save dynamic settings to `camera_config.json` and hot-reload edge detector |
| `POST` | `/api/v1/cameras/test-stream` | Test RTSP stream reachability, resolution ($W\times H$), and framerate (FPS) |
| `DELETE` | `/api/v1/cameras/channels/{channel_or_id}` | Delete camera channel from configuration and archive its calibrated profile |
| `GET` | `/api/v1/cameras/{camera_id}` | Retrieve single camera details |
| `PATCH` | `/api/v1/cameras/{camera_id}` | Update camera settings / active status |
| `POST` | `/api/v1/cameras/{camera_id}/start` | Start server-side background video ingestion worker |
| `POST` | `/api/v1/cameras/{camera_id}/stop` | Stop server-side background video ingestion worker |
| `GET` | `/api/v1/cameras/{camera_id}/status` | Check processing status, FPS, and frame counters |
| `GET` | `/api/v1/cameras/{camera_id}/frame` | Retrieve latest JPEG frame snapshot from camera |
| `GET` | `/api/v1/cameras/{camera_id}/stream` | Stream live MJPEG video with AI annotations |
| `POST` | `/api/v1/cameras/{camera_id}/footfall` | Synchronize footfall entry events from edge detector |
| `POST` | `/api/v1/cameras/{camera_id}/heartbeat`| Synchronize periodic runtime telemetry and occupancy |

### Visitors & Visit Purpose Tagging
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/visitors` | List detected visitors with photos, timestamps, dwell times, and tagging statuses |
| `POST` | `/api/v1/visitors/detect` | Ingest edge face detection with best-shot snapshot and clothing appearance |
| `PATCH` | `/api/v1/visitors/{person_id}/tag` | Tag visitor with visit purpose (Test Ride, Vehicle Enquiry, Booking, etc.), salesperson, and notes |
| `POST` | `/api/v1/visitors/update-zone` | Update live visitor zone dwell times, departure time, and primary department |
| `POST` | `/api/v1/visitors/walk-in` | Manually register a walk-in visitor from showroom reception |
| `POST` | `/api/v1/visitors/clear` | Clear all visitor records and purge stored visitor snapshot images |
| `DELETE` | `/api/v1/visitors` | Reset all visitor records and purge image cache |

### Dynamic Departments & Zones
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/departments` | List all showroom detection departments / sections |
| `POST` | `/api/v1/departments` | Create a new detection department |
| `PATCH` | `/api/v1/departments/{department_id}` | Update department configuration (name, color, polygon) |
| `DELETE` | `/api/v1/departments/{department_id}` | Delete a detection department |

### Footfall Analytics
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/analytics/footfall` | Retrieve total footfall counts with date/camera filters |
| `GET` | `/api/v1/analytics/footfall/daily` | Retrieve daily aggregated footfall entries |

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
