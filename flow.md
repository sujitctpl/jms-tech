# Showroom Visitor Intelligence (JMS) — Architecture & Data Flow Specification

This document details the end-to-end system architecture, operational data flows, communication protocols, and core computer vision algorithms powering the **Showroom Visitor Intelligence** platform.

---

## 1. High-Level Architecture Overview

The system is engineered as an **Edge-AI Hybrid Architecture**. Video ingestion and deep-learning inference take place locally near the camera network, while analytics aggregation, administration, and reporting run via a centralized backend and modern web dashboard.

```mermaid
graph TD
    subgraph Edge Layer [Showroom Local Network / Edge Device]
        CCTV["CCTV / IP Camera\n(RTSP H.264/H.265 via TCP)"]
        Edge["Local AI Engine (local_detector.py / EXE)\n• Threaded RTSP Reader\n• YOLOv11 Person Detection\n• SCRFD / YuNet Face Detection\n• ByteTrack / Centroid Tracking\n• Virtual Entry Line & Zone Engine"]
        Queue["Background Sync Client\n(Non-blocking Memory Queue)"]
    end

    subgraph Central Backend [FastAPI & Express Backend]
        API["FastAPI Backend (:8000)\n• /api/v1/cameras\n• /api/v1/cameras/{id}/footfall\n• /api/v1/cameras/{id}/heartbeat\n• /api/v1/analytics"]
        AuthSrv["Express Auth Server (:5000)\n• /api/auth/login\n• /api/auth/register"]
        Worker["Optional Server VideoWorker\n(app/workers/video_worker.py)"]
    end

    subgraph Persistence [PostgreSQL Database]
        DB[("PostgreSQL Database (:5432)\n• cameras\n• processing_runs\n• visitor_tracks\n• detection_events\n• footfall_events\n• admins")]
    end

    subgraph Client Layer [Web Browser]
        UI["React 19 Dashboard (:5173)\n(Vite + Tailwind CSS + Recharts)\n• Live Telemetry Polling\n• Footfall & Peak Hours Analytics\n• Multi-Zone Occupancy"]
    end

    CCTV -->|RTSP TCP Stream| Edge
    Edge -->|Events & Heartbeat| Queue
    Queue -->|HTTP REST JSON| API
    CCTV -.->|Optional Direct RTSP| Worker
    Worker -->|Direct DB Ingestion| DB
    API -->|SQLAlchemy 2.x ORM| DB
    AuthSrv -->|pg Pool / bcrypt| DB
    UI -->|REST API Requests| API
    UI -->|Auth Requests| AuthSrv
```

---

## 2. Core Concepts & Algorithmic Mechanics

### 2.1 RTSP Stream Ingestion & Buffer Management
* **TCP Transport (`rtsp_transport;tcp`)**: UDP RTSP feeds commonly suffer from packet drops leading to H.264 macroblock artifacts and gray frames. The system forces TCP globally via `OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"`.
* **Threaded Reader & Buffer Draining (`ThreadedRTSPReader`)**: OpenCV's default `cv2.VideoCapture` maintains an internal multi-frame buffer. In standard loops, inference processing delays cause frame lag (seconds behind real time). `ThreadedRTSPReader` continuously calls `grab()` / `retrieve()` in a background daemon thread, discarding stale buffer frames and always providing the latest frame (`read_latest()`) to keep latency under 50ms.

### 2.2 Deep Learning Vision Pipeline
1. **Person Detection (YOLOv11)**:
   - Uses PyTorch / ONNX models (e.g. `yolo11m.pt`, `yolo11n.pt`).
   - Class filter strictly targets `class_id = 0` (Person).
   - High-resolution inference (`imgsz = 960` or `640`) with configurable confidence thresholds (`0.25 - 0.45`).
2. **Face & Head Detection (SCRFD & OpenCV YuNet)**:
   - Dual-engine fallback: High-speed SCRFD (`scrfd_2.5g_bnkps.onnx`) or YuNet (`face_detection_yunet_2023mar.onnx`).
   - Face bounding boxes are spatially mapped and constrained inside the upper torso of the detected person bounding box to eliminate false positives from clothing graphics or posters.
3. **Temporal Tracking (ByteTrack & Local Euclidean Tracker)**:
   - Associates bounding boxes across successive frames to maintain a persistent `tracker_id`.
   - Handles temporary occlusions (e.g., when a visitor passes behind a pillar or another person) for up to `max_disappeared` frames without generating duplicate IDs.

### 2.3 Track Stabilization (`TrackSmoother`)
Raw bounding box detections exhibit high-frequency jitter due to lighting and camera sensor noise. The system applies:
* **Exponential Moving Average (EMA)** smoothing on box coordinates:
  $$\text{Box}_{t} = \alpha \cdot \text{Box}_{\text{detected}} + (1 - \alpha) \cdot \text{Box}_{t-1}$$
  where $\alpha = 0.70$.
* **Deadband Filter**: Box coordinate shifts smaller than 3.5 pixels are suppressed, preventing micro-jitters when visitors are stationary.

---

## 3. Virtual Entry-Line Crossing & Spatial Zones

### 3.1 Feet Reference Point Calculation
To ensure immunity to head tilt, body swaying, or carrying bags, the reference point for line intersection is strictly pinned to the **bottom-center** of the detected bounding box (ground contact point):

$$\text{Point}_{\text{feet}} = \left( \frac{x_1 + x_2}{2}, \; y_2 \right)$$

```
     (x1, y1) +----------------+
              |     [FACE]     |
              |                |
              |     TORSO      |
              |                |
              +--------+-------+ (x2, y2)
                       |
               Point_feet = ((x1+x2)/2, y2)
```

### 3.2 2D Segment Intersection & Direction Validation
Given a trajectory vector from previous frame position $A(x_A, y_A)$ to current position $B(x_B, y_B)$ and a calibrated virtual entry line segment $P_1(x_1, y_1) \to P_2(x_2, y_2)$:

1. **Intersection Test**:
   The segments $AB$ and $P_1P_2$ intersect if and only if points $A$ and $B$ lie on opposite sides of line $P_1P_2$, and points $P_1$ and $P_2$ lie on opposite sides of line $AB$. This is evaluated using counter-clockwise orientation checks (cross products):
   $$\text{ccw}(A, B, C) = (C_y - A_y) \cdot (B_x - A_x) > (B_y - A_y) \cdot (C_x - A_x)$$
2. **Direction Validation**:
   - `top_to_bottom`: Requires $y_B > y_A$ across the line.
   - `bottom_to_top`: Requires $y_B < y_A$ across the line.
   - `left_to_right`: Requires $x_B > x_A$.
   - `right_to_left`: Requires $x_B < x_A$.
   Persons moving in the exit direction or loitering along the line are ignored.

### 3.3 Multi-Zone Occupancy & Section Analytics
Showroom sub-zones (e.g. "Sales Section", "Accounts Section", "Customer Lounge") are defined as 2D polygonal / bounding boxes:
* Each tracked person's feet reference point is tested against the zone boundary:
  $$x_{\min} \le \text{Point}_x \le x_{\max} \quad \text{and} \quad y_{\min} \le \text{Point}_y \le y_{\max}$$
* First entry into a zone triggers a section visit event.
* Continuous presence tracks live section occupancy without double counting.

### 3.4 Multi-Tier Duplicate Prevention Guarantee
Double-counting prevention is enforced at two independent architectural levels:

| Tier | Layer | Mechanism |
|---|---|---|
| **Tier 1 (In-Memory)** | Edge Engine (`EntryLineCounter` / `ZoneManager`) | Maintains `counted_track_ids: Set[str]`. Once a `tracker_id` triggers an entry event, it is permanently locked for the lifetime of that track. |
| **Tier 2 (Database)** | PostgreSQL (`footfall_events` table) | `visitor_tracks` maps `(camera_id, processing_run_id, tracker_id)`. The `footfall_events` table enforces a `UNIQUE(visitor_track_id)` constraint. Duplicate insertions raise a database conflict and are safely discarded. |

---

## 4. Data Transfer & Communication Protocols

### 4.1 Zero Video Transmission Over WAN
Traditional video streaming architectures send high-bandwidth RTSP/WebRTC streams to cloud servers, consuming massive network bandwidth (5–10 Mbps per camera). 

In this system:
* **Video frames never leave the local LAN / edge machine.**
* Only lightweight JSON payloads (approx. 200–500 bytes per event) are transmitted over HTTP/HTTPS.

```mermaid
sequenceDiagram
    autonumber
    participant Cam as CCTV Camera
    participant Edge as Local Edge Engine
    participant Client as DashboardSyncClient (Queue)
    participant API as FastAPI Backend (:8000)
    participant DB as PostgreSQL DB
    participant UI as React Dashboard

    Cam->>Edge: High-FPS RTSP stream (1080p H.264 over TCP)
    Note over Edge: YOLOv11 + ByteTrack Inference<br/>Line crossing detected for Track #42
    Edge->>Client: Enqueue event ("entry", payload)
    
    rect rgb(240, 248, 255)
    Note over Client,API: Non-blocking Async HTTP POST
    Client->>API: POST /api/v1/cameras/{id}/footfall<br/>{"tracker_id": "42", "confidence": 0.92, ...}
    API->>DB: Get or create visitor_track (ID: 42)
    API->>DB: INSERT INTO footfall_events (ON CONFLICT DO NOTHING)
    API-->>Client: 201 Created {"success": true, "total_camera_footfall": 128}
    end

    loop Every 2 to 3 Seconds
        Edge->>Client: Enqueue telemetry heartbeat
        Client->>API: POST /api/v1/cameras/{id}/heartbeat<br/>{"fps": 28.5, "current_persons": 3, "section_counts": {...}}
        API-->>Client: 200 OK
    end

    loop Every 5 Seconds
        UI->>API: GET /api/v1/cameras/{id}/status
        API-->>UI: Return live telemetry, FPS & occupancy
        UI->>API: GET /api/v1/analytics/footfall?start_time=...
        API->>DB: SELECT COUNT(*) FROM footfall_events
        DB-->>API: Aggregate numbers
        API-->>UI: Return total footfall & daily trends
    end
```

### 4.2 Edge-to-Backend HTTP Payloads

#### 1. Footfall Entry Event
* **Endpoint**: `POST /api/v1/cameras/{camera_id}/footfall`
* **Frequency**: Triggered on-demand when a person crosses the virtual entry line.
* **Payload**:
```json
{
  "tracker_id": "42",
  "confidence": 0.94,
  "section_id": "sales",
  "section_name": "Sales Section",
  "occurred_at": "2026-10-28T12:04:15.123456Z"
}
```

#### 2. Periodic Telemetry Heartbeat
* **Endpoint**: `POST /api/v1/cameras/{camera_id}/heartbeat`
* **Frequency**: Emitted every 2.0 – 3.0 seconds.
* **Payload**:
```json
{
  "is_processing": true,
  "fps": 29.4,
  "frames_processed": 18450,
  "current_persons": 4,
  "total_entries": 142,
  "section_counts": {
    "sales": 89,
    "accounts": 53
  },
  "section_occupancy": {
    "sales": 3,
    "accounts": 1
  }
}
```

---

## 5. Database Schema & Relational Mapping

The database schema is managed via **Alembic** migrations and **SQLAlchemy 2.x**.

```mermaid
erDiagram
    cameras ||--o{ processing_runs : "has"
    cameras ||--o{ visitor_tracks : "records"
    cameras ||--o{ footfall_events : "generates"
    processing_runs ||--o{ visitor_tracks : "contains"
    visitor_tracks ||--o{ detection_events : "tracks"
    visitor_tracks ||--o| footfall_events : "confirms (1:1 UNIQUE)"

    cameras {
        uuid id PK
        varchar name
        varchar stream_url
        varchar location
        jsonb entry_line_config
        boolean is_active
        timestamp created_at
        timestamp updated_at
    }

    processing_runs {
        uuid id PK
        uuid camera_id FK
        timestamp start_time
        timestamp end_time
        varchar status
        integer frames_processed
        text error_message
    }

    visitor_tracks {
        uuid id PK
        uuid camera_id FK
        uuid processing_run_id FK
        varchar tracker_id
        timestamp first_seen_at
        timestamp last_seen_at
    }

    detection_events {
        uuid id PK
        uuid camera_id FK
        uuid processing_run_id FK
        uuid visitor_track_id FK
        integer frame_number
        float confidence
        float bbox_x1
        float bbox_y1
        float bbox_x2
        float bbox_y2
        timestamp detected_at
    }

    footfall_events {
        uuid id PK
        uuid camera_id FK
        uuid visitor_track_id FK "UNIQUE"
        varchar event_type "CHECK (= 'entry')"
        timestamp occurred_at
    }

    admins {
        serial id PK
        varchar name
        varchar email "UNIQUE"
        varchar password_hash
        varchar role
        timestamp created_at
    }
```

### Table Lifecycles:
* `cameras`: Master registry of CCTV cameras, RTSP configurations, and calibrated virtual lines.
* `processing_runs`: Records execution sessions (started by the edge client or server background worker) to track uptime, processed frames, and operational errors.
* `visitor_tracks`: Temporal tracks mapping real-world individuals seen by ByteTrack/LocalTracker.
* `footfall_events`: Confirmed visitor counts. The `UNIQUE(visitor_track_id)` database constraint guarantees zero duplicate entries even under concurrent network retries.
* `admins`: Stores web console administrator credentials hashed via `bcrypt`.

---

## 6. Two Supported Deployment Modes

| Feature | Mode A: Distributed Local Edge (Recommended) | Mode B: Centralized Server Worker |
|---|---|---|
| **Executable / Script** | `local_detector.py` or compiled standalone `.exe` | `app/workers/video_worker.py` via FastAPI API |
| **Inference Location** | On-premise PC / Edge device inside showroom LAN | Central Server / Cloud GPU VM |
| **Network Consumption** | Tiny (HTTP JSON telemetry only, zero video) | High (Continuous RTSP stream routed to server) |
| **GUI & Interactive Setup** | Native OpenCV window with interactive mouse line calibration | Headless background thread controlled via `/start` and `/stop` API |
| **Hardware Required** | Local mini-PC / laptop with CPU or Nvidia GPU | High-bandwidth server with GPU multi-stream decoding |

---

## 7. Security, Privacy & Reliability

1. **RTSP Credential Sanitization**:
   - RTSP URLs typically include sensitive embedded credentials (e.g. `rtsp://admin:secret123@192.168.1.100:554`).
   - The logging and API serialization layers pass all URLs through regex masks (`sanitize_url()`), guaranteeing that credentials are substituted with `***:***` across logs and HTTP responses.
2. **Resilient Threading & Auto-Reconnect**:
   - `ThreadedRTSPReader` detects socket timeout or network drops and automatically executes exponential backoff reconnection loops without terminating the application.
3. **Non-Blocking Network Client**:
   - Network dips or slow backend responses never freeze video inference. The `DashboardSyncClient` pushes events into an in-memory queue consumed by a separate daemon thread.
4. **Graceful Offline Mode**:
   - If the backend or PostgreSQL database is temporarily offline, the local detector displays overlay statistics locally without crashing, while the React dashboard gracefully degrades to standalone mode.

---

## 8. Summary: Lifecycle of a Visitor Entry

1. **Physical Ingress**: Visitor steps through the showroom doorway.
2. **Frame Capture**: CCTV streams 1080p RTSP packets via TCP $\to$ `ThreadedRTSPReader` delivers the newest frame buffer.
3. **Inference**: YOLOv11 locates the person; SCRFD locates the face; `LocalTracker`/ByteTrack assigns `tracker_id = 15`.
4. **Smoothing**: `TrackSmoother` applies EMA $\alpha=0.7$ and deadband suppression to prevent box jitter.
5. **Geometric Evaluation**: `ZoneManager` tracks the bottom-center feet point across consecutive frames. The trajectory vector intersects virtual line $P_1P_2$ in the `top_to_bottom` direction.
6. **In-Memory Check**: ID `15` is not in `counted_track_ids` $\implies$ add to set $\implies$ line flash animation rendered on local UI.
7. **Edge Dispatch**: `DashboardSyncClient` puts event into thread queue $\to$ non-blocking `POST /api/v1/cameras/{id}/footfall`.
8. **Persistence**: FastAPI validates request $\to$ creates or links `visitor_tracks` $\to$ inserts into `footfall_events` (enforced by `UNIQUE` constraint).
9. **UI Visualization**: The React dashboard polls `/status` and `/footfall` $\to$ increments the live footfall KPI tile, updates peak-hour distribution, and logs the entry timestamp.

