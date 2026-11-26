# Showroom Visitor Intelligence & Footfall Analytics Platform

An enterprise-grade, edge-AI hybrid visitor intelligence and showroom analytics platform. The system leverages existing CCTV / RTSP camera infrastructure to deliver real-time person detection powered by **Roboflow RF-DETR** (Real-Time Detection Transformer with DINOv2 backbone, with Ultralytics YOLO fallback), multi-object tracking (**ByteTrack** and **BoT-SORT** with motion compensation), biometric face recognition (**ArcFace** 512-D embeddings with **SCRFD** / **YuNet** face detection), deep full-body re-identification (**OSNet** 512-D embeddings with multi-part 3D HSV color histograms), staff vs. visitor segregation, automated employee attendance logging, dynamic department zone management, visitor journey & dwell tracking, directional virtual entry-line crossing, dual-factor departure verification with high-resolution exit photo capture, and an interactive **React 19** web dashboard.

---

## 1. System Overview & Architecture

The platform operates on a high-efficiency **Edge-AI Hybrid Architecture**:

```mermaid
graph TD
    subgraph Edge Layer [Showroom Edge / Local Machine]
        CCTV["CCTV / IP Cameras (RTSP TCP)"]
        Detector["Edge AI Engine (local_detector.py)\n• RF-DETR Transformer / YOLO Detection\n• SCRFD / YuNet Face Tracking (Anatomical Validation)\n• ArcFace 512-D Face & OSNet 512-D Body Re-ID\n• Pose-Located Garment Colour & Soft Lab Histograms\n• ByteTrack / BoT-SORT Tracking\n• Multi-Frame Best-Shot Photo Selection\n• Visitor Zone Dwell & Journey Tracking\n• Directional Entry & Exit Line Crossing\n• Departure Snapshot Capture (exit_avatar_url)\n• Local MJPEG Stream Server\n• Zero-Downtime Hot-Reloading"]
        SyncQueue["Background Sync Client (DashboardSyncClient)\n• Crash-Resilient In-Order FIFO Queue\n• Backend Auto-Reconnect & Retry"]
    end

    subgraph Backend Layer [FastAPI & Express Backend]
        API["FastAPI Analytics Backend (:8000)\n• Dynamic Camera Configuration & Stream Probing\n• Footfall Analytics & Real-Time Telemetry\n• Dual-Factor Visitor Ingestion & Zone Updates\n• Departure Verification & Weak Re-ID Band Matching\n• Dynamic Department Management\n• Employee Registration & Attendance\n• Biometric Rolling Retention Worker"]
        AuthSrv["Express Auth Server (:5000)\n• User & Admin Authentication\n• PostgreSQL Session Pool"]
        DB[("PostgreSQL Database (:5432)\n• cameras & departments\n• employees & employee_attendance\n• person_identities (pgvector 512 face + 512 body Re-ID)\n• visitor_tracks & footfall_events")]
    end

    subgraph Client Layer [Modern Web UI]
        Dashboard["React 19 Dashboard (:5173)\n• Live CCTV Preview & AI Overlays\n• Dynamic Camera & AI Settings Panel\n• Visit Purpose Tagging & High-Res Lightbox\n• Dual Entry vs Departure Photo Review\n• 1-Click Bounced Visitor Removal\n• Real-Time Zone Dwell & Customer Journey\n• Staff Attendance & Daily Rolls\n• Section Dwell Time & Exportable Reports"]
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
- **Pluggable Multi-Object Tracking & Spatial-Temporal Appearance Handoff**:
  - **ByteTrack** (`bytetrack.yaml`): Low-latency matching using both high- and low-score detection boxes.
  - **BoT-SORT** (`botsort.yaml`): Includes Global Motion Compensation (GMC via sparse optical flow) and Kalman state estimation to eliminate camera shake/drift.
  - **Spatial-Temporal Handoff Tracker (`SpatialTemporalHandoffTracker`)**: Multi-cue coasting engine (25s window) combining spatial distance, bounding box scaling, and **Upper-Torso HSV Color Histogram Correlation** ($\ge 0.55$). Seamlessly preserves identity, display aliases, and waiting timers across 180° turnarounds, counter occlusions, and brief tracker ID switches.
  - **Robust Fallback Centroid & Kalman Tracker**: Velocity-prediction fallback ensuring tracking continuity across temporary drops.
- **Entry-Line Gated Ingress, 7s Dwell Bounce Filter & Ghost Elimination**:
  - **Strict Entry-Line Gating**: Walk-in visitor creation, footfall counts, waiting timers, and photo capture are strictly gated by directional virtual entry-line crossing.
  - **7-Second Dwell Threshold (Bounce Filter)**: When a person crosses the entry line, the system waits for $\ge 7.0$ seconds of showroom dwell before confirming the visitor and incrementing footfall. If someone turns around or leaves in under 7 seconds (e.g. delivery persons, wrong-store turnarounds, or pacing near the doorway), the entry is discarded with zero ghost records.
  - **Reverse Line Departure & Staff Checkout**: Crossing the line in reverse triggers departure logging only after $\ge 7.0$ seconds of stay, and automatically logs `CHECK_OUT` attendance punches for recognized showroom employees.
  - **Ghost Visitor Suppression**: Persons visible in camera view who have not crossed the entry line are marked as *Unconfirmed* with a neutral slate border `(180, 180, 180)` and tracker ID only, completely preventing transient passersby, outdoor foot traffic, or glass facade reflections from skewing visitor analytics.
  - **Interior Camera Guard**: Non-entrance cameras (`is_entrance: false`) suppress minting new walk-ins (`IGNORED_INTERIOR`) unless verified via cross-camera biometric matching against an entered visitor.
- **Dual-Factor Re-Identification & Deep Body Embedding (`BodyEmbedder`)**:
  - **OSNet Deep Feature Extraction**: Full-body appearance modeling via Omni-Scale Network (`osnet_x1_0` / `osnet_x0_25`) generating 512-dimensional L2-normalized feature embeddings.
  - **Aspect-Ratio Adaptive Salient Torso Extraction (`extract_salient_torso_patch`)**: Differentiates bust/portrait crops ($h/w < 1.20$, sampling lower torso band $35\% - 85\%$) from full-body CCTV frames ($h/w \ge 1.20$, sampling upper torso band $18\% - 50\%$). Uses a sliding column window with chromatic saturation and darkness contrast weighting against white doorway walls and floor mats.
  - **Salient Lower-Body Legs Patch Extraction (`extract_salient_legs_patch`)**: Slides across lower-body band ($54\% - 90\%$) with center-prior weighting to isolate trousers, jeans, or skirts and reject background floor tiles.
  - **Pose-Located Clothing Colour** (`app/vision/clothing_color.py`): a YOLO pose model (`models/yolo11n-pose.pt`, auto-downloaded) finds the shoulder→hip torso and the hip→ankle legs, so samples exclude the head, arms and doorway wall. Colours are white-balanced against the wall and measured in CIE Lab with lightness *relative to the wall*, so a shirt reads the same lit and in shadow. Clothing similarity is the Bhattacharyya overlap of soft Lab histograms: $0.65 	imes \text{torso} + 0.35 	imes \text{legs}$.
  - **Clothing Vetoes**: incompatible upper-garment colours (lighting neighbours such as Gray↔White/Cream/Green are allowed; patterned tops are compatible with neutrals) unless $\text{ReID} \ge 0.82$; dark vs light trousers (relative lightness gap $\ge 0.40$) unless $\text{ReID} \ge 0.85$.
  - **Appearance Verification (`verify_appearance`)**: match only when $\text{ReID} \ge 0.70$ and torso similarity $\ge 0.40$; there is no low-ReID branch, because similar dark outfits caused wrong departures. See `flow.md` for the full rules.
  - **Weak Re-ID Band ($0.60 \le \text{ReID} < 0.70$) & Sleeve-Aware Departure Resolver**: In `visitor_service.update_visitor_zone`, departures that do not meet the primary $\ge 0.70$ threshold (e.g. frontal ingress vs overhead back-of-head egress) are resolved using weak-band verification: requires $\text{torso\_sim} \ge 0.80$, compatible color tags, sleeve length agreement (short = skin-toned forearms vs long = fabric sleeves), trouser agreement, and a $\ge 0.08$ score gap over every other active visitor (or $0.50$ stranger baseline). Non-matches are capped at $0.45$, safely below the $0.48$ exit eligibility threshold.
  - **15-Minute Ingress Turnaround Re-ID**: Automatically re-identifies showroom visitors who entered within the last 15 minutes and turned around or whose tracker IDs flipped under overhead CCTV occlusion. Requires a passing `verify_appearance` plus ($\text{ReID} \ge 0.65$ and $\text{clothing} \ge 0.28$) or $\text{ReID} \ge 0.74$, without requiring frontal facial biometrics.
- **Crash-Resilient Edge Event Delivery (`DashboardSyncClient`)**:
  - **Restart-Proof FIFO Sync Queue**: Detects when the FastAPI backend is temporarily down or reloading (e.g., `uvicorn --reload` or network disruption) and preserves critical visitor ingress (`/visitors/detect`) and departure (`/visitors/update-zone`) events in an in-memory queue.
  - **Strict In-Order Retry**: Queued events are dispatched in strict chronological order upon reconnect without dropping visitor records or corrupting session sequences.
  - **Smart Error Classification (`_is_unreachable_error`)**: Classifies connection errors (e.g. `ConnectionRefusedError`, `WinError 10061`) to trigger backoff retries, while standard HTTP client/server responses (400, 404, 500) are handled without redundant retry loops.
  - **Selective Heartbeat Dropping**: Non-critical runtime telemetry (heartbeats and frame FPS) is dropped during outages to prevent queue backlog from starving critical visitor data upon reconnect.
- **Biometric Face Recognition & Staff vs. Visitor Segregation**:
  - Neural face detection via **SCRFD** (`scrfd_2.5g_bnkps.onnx`) or **YuNet** with anatomical bounding box validation (aspect ratio $0.45$–$1.55$, vertical center $c_y \le 0.50 \cdot h$ on standing crops, eliminating false triggers on floor items, shoes, and dustbins).
  - **ArcFace** (`w600k_r50.onnx`) 512-dimensional feature embedding extraction.
  - Real-time cosine similarity matching against registered employee vector gallery with **calibrated operating threshold $\ge 0.65$** (`FACE_SIMILARITY_THRESHOLD = 0.65`) and Top-1 vs Top-2 candidate match margin $\ge 0.030$.
  - Temporal voting buffers (`EMPLOYEE_VOTE_WINDOW = 5`) ensuring zero false-positive visitor misclassifications.
- **Optimized Visitor Photo Capture, Departure Verification & Lightbox UX**:
  - **Multi-Frame Best-Shot Selection**: Evaluates Laplacian sharpness, pixel area, pose frontality (eye horizontal symmetry and nose centering), and illumination balance into a unified quality score $Q$.
  - **Dynamic Progressive Upgrades & Occlusion Recovery**: Dispatches an instant snapshot on first frame for zero-latency UI updates, then automatically upgrades if quality improves by $\ge 25\%$ during the first 3.5s walk-in window. Detects partially occluded doorway slivers (box aspect ratio $w/h < 0.30$) and seamlessly upgrades when an unoccluded, wider box appears ($w/h \ge 0.32$, width $\ge 50\text{ px}$).
  - **Adaptive Bust & Torso Framing**: Expanded portrait crop capturing head, hair, collar, and upper-chest context at $320\times320\text{ px}$ (JPEG Q95) for effortless visual recognition by showroom staff.
  - **CLAHE Backlight Compensation**: Corrects severe silhouettes caused by sunny showroom glass facades via adaptive histogram equalization on the LAB luminance channel.
  - **Automatic Clothing Appearance Tagging**: Pose-located torso colour generates human-readable attire badges (e.g. *Blue / Navy Top*, *White / Light Shirt*, *Green / Olive Top*, *Checked / Patterned Shirt*).
  - **High-Resolution Departure Snapshot (`exit_avatar_url`) & Turnaround Departure Forwarding**: Grabs an un-annotated high-resolution exit crop when visitors cross the entrance line in reverse (after dwell $\ge 7.0$s). If a visitor was assigned a fresh turnaround track ID ($< 7.0$s), it is forwarded to the backend appearance engine so active showroom visitors are properly departed.
  - **Exact Tracker Token Lookup & Edge Exclusion Propagation**: Resolves `tracker_id` with strict regex word boundaries against `Tracker #N` notes, ensuring `Tracker #4` never mistakenly matches `Tracker #42`. If the edge detector's exit face contradicts the local track ($< 0.40$), it drops local fallback and transmits `exclude_person_id` to prevent the backend from incorrectly re-attaching that candidate.
  - **Evidence-Gated Edge Exit IDs**: An exit only attaches to a visitor still inside today (not departed, not from a previous day). A `person_id` / tracker sent by the edge is trusted outright only when it is the visitor's own registration tracker (the first `Tracker #N` in notes). Any other edge-supplied ID (face re-ID, handoff) needs the exit photo to back it (body `is_match` or reliable face $\ge 0.52$); otherwise the exit goes to the normal search over all active visitors.
  - **Re-ID Override in Body Search**: Candidates rank by combined score, but an eligible visitor whose Re-ID beats every other eligible one by $\ge 0.08$ wins (clothing histograms are weak on the washed-out doorway camera).
  - **Stale Prior-Day Visit Auto-Closure**: Visitors with unclosed visits from previous calendar days are automatically marked as `"Timed out"` (`is_expired = True`) before matching today's exits, eliminating false departure attachments to legacy records.
  - **Safe Collision-Resistant Visitor ID Minting**: Sequential count-based ID minting (`VIS-2026-XXXX`) performs database uniqueness collision checks (`_mint_person_id`) to prevent silent ID collisions and record overwrites.
  - **Reliable Exit Face Biometric Gating**: Requires minimum eye distance ($eye\_d \ge 16.0\text{ px}$) and face width ($w \ge 42\text{ px}$) before face biometrics can veto an exit crop, preventing blurry or side-profile departure frames from rejecting valid departures.
  - **Entry-Exit Match Confidence (`exitMatchConfidence`) & Verification Badge**: Calculates verified match confidence (e.g. `82%`), persists `Exit Match: XX% (...)` in visitor notes for every attached exit (readers use the latest one) for instantaneous retrieval without re-computing embeddings on every poll, and surfaces an interactive **`ENTRY-EXIT MATCH [XX]%`** pill with live status pulses on visitor cards.
  - **Dual-Photo Lightbox UX**: Visitor cards display an intuitive **"+1 Exit Image"** badge, with interactive Entry Snapshot vs Departure Snapshot toggle tabs in `VisitorsView.tsx`, an AI matching & verification debug breakdown card, Escape key dismissal, click-outside modal close, and sticky scrollable headers.
- **Real-Time Visitor Journey & Zone Dwell Tracking**:
  - **`VisitorZoneTracker` & Dual Line Crossing**: Continuously tracks customer movement and accumulates dwell time across custom departments (e.g. Sales Section, Luxury Lounge, Diamond Counter) and the general showroom floor (`Main Showroom`).
  - **Live Dwell Formatting & Periodic Sync**: Synchronizes formatted durations (e.g. `12m 45s`, `35s`) with FastAPI backend (`POST /api/v1/visitors/update-zone`) every 4.0 seconds.
  - **Directional Reverse Exit Detection & 7s Premature Exit Suppression**: Crossing the entrance line in reverse detects visitor departure in Indian Standard Time (IST, UTC+05:30), logs final dwell breakdown across all visited zones, and persists departure time to CRM. Exits under 7 seconds are suppressed to eliminate doorway bounce jitter.
  - **Server-Side Video Worker Ingress & Egress (`VideoWorker`)**: Server-side video worker leverages `EntryLineCounter.process_events()` to clock staff `CHECK_OUT` attendance punches and record visitor exits with automated crop extraction.
  - **Bounced Visitor Removal (`DELETE /api/v1/visitors/{person_id}`)**: Single-click removal of bounced or cancelled visitors, purging linked footfall events, detection rows, and physical snapshot files on disk.
- **Visitor Waiting Timer & Unattended SLA Breach Monitoring**:
  - **Perspective-Calibrated Staff Proximity Filter (`is_staff_near_box`)**: Dynamically measures ground-plane distance in meters between visitor feet and staff feet using bounding box height scaling ($\text{Scale}_{\text{px/m}} = \frac{\bar{h}}{1.70}$). A visitor within $2.2\text{ m}$ is considered actively attended.
  - **2-Parameter Unattended Logic**: Customer waiting time accumulates only when the visitor is **unassisted** (no staff within $2.2\text{ m}$) **and outside any designated department counter box**.
  - **Monotonic Protection & Physical Dwell Bound**: Backend updates protect against clobbering higher accumulated `waiting_seconds` via monotonic max resolution, and the edge engine caps waiting time to true physical dwell ($\min(W, \text{now} - T_{\text{first}})$), preventing artificial spikes.
  - **Configurable SLA Cutoff**: SLA threshold configurable directly from the Dashboard Settings panel (`waiting_cutoff_seconds`, default $900\text{s}$ / $15\text{ mins}$, warning threshold $5\text{ mins}$).
  - **High-Resolution SLA Breach Snapshot**: Automatically grabs a clean full-frame snapshot when an unattended visitor crosses the 15-minute SLA cutoff for manager intervention.
  - **Three-Tier Visual HUD Alerting**:
    - **Normal Waiting ($< 5\text{ mins}$)**: Neon Emerald Green bounding box `(0, 240, 110)` with `⏱️ WAITING [MMm SSs]`.
    - **Amber Warning ($5\text{ to }15\text{ mins}$)**: Amber border `(0, 140, 255)` with `⚠️ WAITING [MMm SSs]` to alert floor staff.
    - **Critical SLA Breach ($> 15\text{ mins}$ / $900\text{s}$)**: Bright Red alert box `(0, 0, 255)` and red badge `🚨 UNATTENDED > 15m [MMm SSs]` with immediate console alarm.
  - **Dashboard SLA Breach & Waiting Badges**: Live counters, waiting duration pills, and unattended alerts surfaced across `OverviewView`, `VisitorsView`, and `VisitPurposeView`.
- **Dynamic Multi-Camera Handover & Dual-Factor Cross-Camera Re-Identification**:
  - **CrossCameraTracker & Global Identity Synchronization**: Coordinates tracks between entrance cameras (`is_entrance: true`) and interior section cameras (`is_entrance: false`) via `.cross_camera_state.json`.
  - **Dual-Factor Biometric Matching**: Re-identifies customers moving from entrance to floor sections using both ArcFace facial biometrics ($\ge 0.65$) and OSNet 512-D body Re-ID vectors ($\ge 0.65$) via `PgVectorStoreManager.find_best_visitor_body_match()`, preventing duplicate footfall counts.
  - **Active 15-Minute Re-ID Window**: Automatically merges tracker ID flips for visitors who entered in the last 15 minutes and are looking down or occluded under overhead cameras (passing `verify_appearance` plus $\text{ReID} \ge 0.65$ and $\text{clothing} \ge 0.28$, or $\text{ReID} \ge 0.74$).
  - **Synchronized Identity Linking Callback (`on_visitor_registered_callback`)**: Automatically updates local tracker IDs, zone manager, and snapshots as soon as the persistent ID is minted by the backend.
  - **Cross-Camera Waiting Sync**: Seamlessly syncs accumulated waiting seconds and breach flags between camera streams.
  - **Isolated GPU VRAM Allocation**: Automatic memory fraction limit ($42\%$) ensures multi-camera instances run concurrently on 4GB consumer GPUs without OOM errors.
- **Dynamic Camera Configuration & Edge Hot-Reloading**:
  - **Full Web UI Management**: Configure RTSP feeds, camera names, confidence thresholds, and entry lines directly in the React Dashboard via `CameraSettingsSection.tsx` and `SettingsView.tsx`.
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
- **7-Day Rolling Biometric Retention & Physical Snapshot Purge (GDPR / Privacy Compliance)**:
  - Automated daily background scheduler (`retention_scheduler`) purging visitor face embeddings, OSNet body Re-ID vectors, and entry/exit snapshots for identities inactive for > 7 days (`FACE_RETENTION_DAYS`, default `7`) while preserving aggregated footfall numbers.
  - Automatically deletes physical snapshot files (`static/uploads/visitors/`) and purges orphaned snapshot images older than retention period.
- **Virtual Entry-Line Crossing**:
  - 2D ground-plane foot contact point calculation ($\frac{x_1 + x_2}{2}, y_2$) with strict directional validation (`top_to_bottom`, `bottom_to_top`, `left_to_right`, `right_to_left`).
- **Strict Multi-Tier Duplicate Prevention**:
  - In-memory tracker lock (`counted_track_ids: Set[str]`) combined with database-level uniqueness (`UNIQUE` constraint on `visitor_track_id` in `footfall_events`).
- **Live Video Streaming & Overlays**:
  - Built-in HTTP MJPEG streaming server for low-latency live preview with AI bounding boxes and tracks directly in the web dashboard.
- **Modern Web Dashboard (`dashboard/`)**:
  - Built with React 19, Vite, TypeScript, and Tailwind CSS.
  - Live CCTV preview, dynamic camera settings, dynamic department configuration, waiting time & SLA alert configuration, employee attendance management, peak hour heatmaps, visitor tagging, and exportable reports.
- **Single-Command & Batch Multi-Camera Orchestration**:
  - Concurrently launch all 3 services (FastAPI, Express Auth, and Vite Dashboard) via `python scripts/run_all.py` or Windows `.bat` / `.ps1` scripts.
  - 1-Click edge detector launch via [`start_detector.bat`](file:///c:/Users/capermint/Desktop/Abhishek/jms/start_detector.bat).
  - Concurrent multi-camera launch for all configured channels via [`run_all_cameras.bat`](file:///c:/Users/capermint/Desktop/Abhishek/jms/run_all_cameras.bat) or `python scripts/run_multi_camera.py`.
- **Desktop Executable Packaging**:
  - PyInstaller builder script (`build_exe.py`) to compile the edge detector into a standalone Windows `.exe`.

---

## 3. Technology Stack

| Component | Technologies |
|---|---|
| **Edge AI Engine** | Python 3.11+, Roboflow RF-DETR, Ultralytics YOLO, ByteTrack, BoT-SORT, OpenCV, ONNX Runtime, NumPy |
| **Biometrics & Vision** | InsightFace ArcFace (512-D), OSNet Full-Body Re-ID (512-D), YOLO Pose (YOLO11n-Pose), SCRFD, YuNet Face Detector, CIE Lab Relative Lightness & Soft Histograms, 3D HSV Color Histograms, pgvector / Cosine Similarity |
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
│   │       ├── person_identity.py       # 7-day rolling biometric identity (pgvector 512 face + 512 body Re-ID, exit_avatar_url)
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
│   │   ├── body_embedder.py             # OSNet Re-ID embedding & appearance verification
│   │   ├── clothing_color.py            # Pose-located garment colour tags & clothing similarity
│   │   ├── employee_recognizer.py       # Biometric matching & voting manager
│   │   └── vector_store.py              # pgvector search manager
│   └── workers/
│       └── video_worker.py              # Server-side background ingestion worker
├── dashboard/                           # Modern React 19 Web Dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── camera/                  # Live camera feeds, grids & overlays
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
│   ├── run_multi_camera.py              # Parallel launcher for all configured camera feeds
│   ├── run_all_cameras.bat              # Batch script to launch all 4 channels concurrently
│   ├── run_detector.bat                 # Quick-launch batch script for local_detector.py
│   ├── benchmark_trackers.py            # ByteTrack vs BoT-SORT performance benchmark
│   ├── download_models.py               # Automated weights downloader
│   ├── start_services.bat / .ps1        # Multi-window startup scripts for Windows
│   └── stop_services.bat / .ps1         # Safe shutdown scripts for all background services
├── static/                              # Static file uploads (employee and visitor photos)
│   └── uploads/
│       ├── employees/                   # Registered employee gallery photos
│       └── visitors/                    # 320x320 retina visitor portrait crops & SLA breach captures
├── tests/                               # Comprehensive Automated Test Suites
│   ├── test_waiting_timer.py            # Unit tests for waiting timer, handoff, SLA breach & monotonic bounds
│   ├── test_cross_camera_tracker.py     # Cross-camera tracking & re-ID tests
│   ├── test_returning_visitor.py        # Returning visitor recognition tests
│   ├── test_identity_and_retention.py   # Rolling retention & biometric purge tests
│   ├── test_entry_exit_fixes.py         # Exit matching, visitor ID minting & body-only retention regressions
│   ├── test_exit_attribution.py         # Evidence-gated edge exit IDs, Re-ID override, latest exit score
│   ├── test_clothing_color.py           # Pose-located garment extraction, CIE Lab soft histograms & sleeves
│   ├── test_sync_retry.py               # Crash-resilient DashboardSyncClient FIFO retry & reconnect tests
│   ├── test_entry_counter.py            # Directional line crossing geometry & duplicate prevention
│   └── test_vision.py                   # Model wrapper, detector & tracker integration tests
├── .cross_camera_state.json             # Cross-camera persistent state registry
├── CLAUDE.md                            # Claude Code project guidelines and operational notes
├── alembic/                             # Database migration scripts
├── local_detector.py                    # Standalone Local Edge AI Video Processing Engine
├── start_detector.bat                   # 1-Click Windows Batch Launcher for Single-Camera Detector
├── run_all_cameras.bat                  # 1-Click Root Batch Launcher for All Configured CCTV Channels
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

# Biometrics, Deep Re-ID & 7-Day Rolling Retention
ARCFACE_MODEL_PATH="models/w600k_r50.onnx"
FACE_SIMILARITY_THRESHOLD=0.65
VISITOR_REID_THRESHOLD=0.65
FACE_RETENTION_DAYS=7
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

#### Option 1: 1-Click Launch All Cameras (Recommended)
Launch all configured, active camera channels (101, 201, 301, 401) concurrently in separate terminal windows with staggered startup:
```cmd
# Double click or run from terminal:
run_all_cameras.bat

# Or run via Python orchestrator:
python scripts/run_multi_camera.py
```

#### Option 2: Launch Specific Cameras via Batch or Python
```cmd
# Launch specific camera subset (e.g. Channel 101 and Channel 201):
run_multi_camera.bat 101 201

# Or via Python script directly:
python scripts/run_multi_camera.py 101 201

# Launch all cameras in headless background mode:
python scripts/run_multi_camera.py --headless
```

#### Option 3: Via PowerShell Launcher
```powershell
# Launch Channels 101 and 201 in parallel:
.\run_multi_camera.ps1 101, 201

# Launch all 4 channels:
.\run_multi_camera.ps1 101, 201, 301, 401
```

#### Option 4: Via Separate Terminal Tabs
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

### Entry-Line Gated Ingress & Ghost Visitor Elimination

Showrooms with glass facades or open frontages often face challenges with transient passersby, outdoor foot traffic, or glass reflections triggering false detections. The platform implements a rigorous multi-tier gating architecture:

1. **Gated Walk-in Ingress**: A person detected in front of the camera is considered an **Unconfirmed Person** until they physically cross the calibrated directional entry line.
2. **Neutral Slate Visualization**: Unconfirmed individuals are rendered with neutral slate/silver bounding boxes `(180, 180, 180)` and labeled with their tracker ID only (`Person #ID (Unconfirmed)`). They **do not** trigger footfall events, create database visitor records, initiate waiting timers, or fire photo captures.
3. **Ghost Visitor Elimination**: Passersby on the sidewalk or glass facade reflections are completely prevented from polluting visitor analytics or creating ghost profiles in the CRM.
4. **Interior Camera Ingress Guard**: In-store section cameras (`is_entrance: false`) suppress minting new walk-ins (`IGNORED_INTERIOR`) unless verified via cross-camera biometric matching against an already-entered visitor.

---

### Multi-Cue Spatial-Temporal & Torso Handoff Tracking

When visitors turn 180° away from the camera, bend over jewelry showcases, or navigate behind display counters, facial recognition is temporarily occluded and raw tracker IDs can switch. The edge engine deploys `SpatialTemporalHandoffTracker`:

- **25-Second Spatial-Temporal Coasting Pool**: When a track is lost, its geometric state, last known position, display alias, and accumulated waiting metrics are held in a coasting pool for up to $25.0\text{s}$.
- **Upper-Torso HSV Histogram Correlation**: When a new track appears, the engine computes normalized HSV color histograms of the upper-torso region and compares them via histogram correlation:
  $$\text{Corr}(H_1, H_2) \ge 0.55$$
- **Seamless Timer & Identity Restoration**: Upon a correlation match within physical distance tolerance, the new track inherits the original `first_seen_ts`, `person_id`, display alias, accumulated `waiting_seconds`, and `waiting_breach` status (`Timer & Waiting unbroken!`).

---

### Multi-Camera Person & Visitor Re-Identification Flow

The platform features seamless cross-camera handover between entrance and interior floor cameras:

```mermaid
flowchart TD
    subgraph Entrance ["Camera 101 (Main Glass Entrance - is_entrance: true)"]
        E1["Person crosses Directional Entry Line"] --> E2["SCRFD Face Detection + ArcFace 512-D Embedding"]
        E2 --> E3{"Staff or Visitor?"}
        E3 -->|Employee| E4["Staff Attendance Checked In (Gated)"]
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
        I3 -->|Visitor Match: edge >= 0.55, backend >= 0.65| I5["Matched to VIS-2026-1043<br/>Links Track & Updates Zone Dwell"]
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
   - The edge identity manager links the track locally at cosine similarity $\ge 0.55$; the backend (`/visitors/detect`) confirms cross-camera visitor matches at $\ge 0.65$ (`VISITOR_REID_THRESHOLD`) before the track is linked to `VIS-2026-1043`.
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
| `--sim-thresh` | Biometric face cosine similarity threshold for staff identification | `0.65` |
| `--show-line` | Display virtual entry line & calculate crossings | `True` |
| `--deadband` | Pixel displacement deadband to eliminate stationary jitter | `6.0` |
| `--smoothing-alpha`| EMA coordinate smoothing factor (0.1 to 1.0) | `0.35` |
| `--face-persistence`| Frames to buffer face detections across temporary drops | `8` |
| `--body-persistence`| Missing frames before an un-detected person track is purged | `26` |
| `--standalone-faces`| Enable detection of standalone faces behind cubicle partitions | `False` |
| `--head-fallback` | Enable synthetic anatomical head box fallback when face is turned away | `False` |
| `--headless` | Run as background service without local OpenCV window | `False` |
| `--api-url` | Base URL of FastAPI backend for syncing events | `'http://127.0.0.1:8000'` |
| `waiting_cutoff_seconds` | Configurable in `camera_config.json` & UI: Cutoff SLA threshold in seconds (e.g. `900.0` for 15 mins) | `900.0` |

---

### Visitor Waiting Timer & SLA Breach Monitoring Pipeline

```mermaid
flowchart LR
    A["Customer in Showroom"] --> B{"Confirmed Ingress<br/>(Crossed Entry Line)?"}
    B -- No --> C["Unconfirmed Bystander<br/>(Neutral Slate Border - No SLA)"]
    B -- Yes --> D{"Inside Calibrated<br/>Department Box?"}
    D -- Yes --> E["Dwell Accumulates to Zone<br/>(Attended / In Consultation)"]
    D -- No --> F{"Showroom Staff<br/>Within 2.2m Proximity?"}
    F -- Yes --> E
    F -- No --> G["Customer Marked Unattended<br/>(⏱️ Waiting Timer Starts)"]
    G --> H{"Waiting Duration?"}
    H -- "< 5 mins" --> I["🟢 Emerald Green Box<br/>⏱️ WAITING [MMm SSs]"]
    H -- "5 to 15 mins" --> J["🟠 Amber Warning Box<br/>⚠️ WAITING [MMm SSs]"]
    H -- "> 15 mins (SLA Breach)" --> K["🚨 Bright Red Alert Box<br/>🚨 UNATTENDED > 15m"]
    K --> L["Console Alert Alarm +<br/>SLA Breach Snapshot Saved +<br/>CRM & Dashboard SLA Breach Badge"]
```

- **Perspective Proximity Math**:
  $$\text{dist\_m} = \frac{\sqrt{(x_v - x_s)^2 + (y_v - y_s)^2}}{\max(1.0, \; \bar{h} / 1.70)}$$
- **Automatic Attended Reset**: As soon as a registered employee steps within $2.2\text{ m}$ of the customer or the customer enters a department zone, the active waiting state pauses and transitions back to attended.
- **Monotonic Protection & Physical Dwell Upper Bound**:
  - The FastAPI backend updates preserve highest accumulated waiting seconds (`existing_rec.waiting_seconds = max(...)`), ensuring secondary cameras do not clobber metrics.
  - The edge engine enforces real physical dwell limits: $\text{waiting\_seconds} \le \text{now\_ts} - \text{first\_seen\_ts}$, preventing clock skew or artificial spikes.
- **Configurable SLA Cutoff**: Configurable via the React Dashboard Settings tab ("Waiting Time & SLA Alerts") and synchronized to `camera_config.json` (`waiting_cutoff_seconds`, default $900\text{s}$ / $15\text{ mins}$).
- **Automated High-Resolution SLA Breach Snapshot**: When an unattended visitor breaches the cutoff threshold, the engine immediately grabs an un-annotated full-resolution frame, writes it to `static/uploads/visitors/`, and dispatches it for immediate floor management intervention.

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

# Run Entry/Exit fixes & Body-only Retention test
pytest tests/test_entry_exit_fixes.py -v

# Run Clothing Color, Pose & Garment Similarity test
pytest tests/test_clothing_color.py -v

# Run Edge Sync Reconnection & FIFO Retry test
pytest tests/test_sync_retry.py -v

# Run Directional Entry Line Counter test
pytest tests/test_entry_counter.py -v
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
| `DELETE` | `/api/v1/cameras/channel/{channel_or_id}` | Delete camera channel from configuration and archive its calibrated profile |
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
| `GET` | `/api/v1/visitors` | List detected visitors with photos (`avatarUrl`, `exitAvatarUrl`), timestamps, dwell times, `exitMatchConfidence`, and tagging statuses |
| `POST` | `/api/v1/visitors/detect` | Ingest edge face and body detection with best-shot snapshot, clothing appearance, and 512-D OSNet Re-ID vector |
| `PATCH` | `/api/v1/visitors/{person_id}/tag` | Tag visitor with visit purpose (Test Ride, Vehicle Enquiry, Booking, etc.), salesperson, and notes |
| `POST` | `/api/v1/visitors/update-zone` | Update live visitor zone dwell times, departure timestamp, exit portrait crop (`exit_image_base64`), multi-factor departure verification, and department |
| `POST` | `/api/v1/visitors/walk-in` | Manually register a walk-in visitor from showroom reception |
| `DELETE` | `/api/v1/visitors/{person_id}` | Delete a single bounced or cancelled visitor record and purge its stored portrait snapshot |
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
| `POST` | `/api/v1/identity/cleanup` | Execute rolling biometric retention cleanup (default 7 days, `?retention_days=` overrides) |
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
- Rolling retention cleanup logic (including body-only visitors)
- Entry-exit matching: exact tracker lookup, edge exclusion, and walk-in ID collision safety

### Multi-Object Tracker Benchmarking
Benchmark ByteTrack vs BoT-SORT latency, ID switching, and FPS:
```bash
python scripts/benchmark_trackers.py
```
Benchmark results and deployment recommendations are detailed in [benchmark_results.md](file:///c:/Users/capermint/Desktop/Abhishek/jms/benchmark_results.md).
