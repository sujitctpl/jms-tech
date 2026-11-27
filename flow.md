# Showroom Visitor Intelligence (JMS) — Architecture & Data Flow Specification

This document details the end-to-end system architecture, operational data flows, communication protocols, computer vision algorithms, biometric matching pipelines, and database schemas powering the **Showroom Visitor Intelligence & Footfall Analytics** platform.

---

## 1. High-Level Architecture Overview

The system is engineered as an **Edge-AI Hybrid Architecture**. Computationally intensive deep-learning inference (object detection, multi-object tracking, neural face detection, biometric embedding extraction, photo quality scoring, and zone dwell computation) runs locally at the showroom edge near the camera network. Analytics aggregation, visitor CRM tagging, dynamic camera configuration, employee management, session authentication, and data retention run on a centralized backend and modern web dashboard.

```mermaid
graph TD
    subgraph Edge Layer [Showroom Edge / Local Machine]
        CCTV["CCTV / IP Cameras\n(RTSP H.264/H.265 via TCP)"]
        
        subgraph Vision Pipeline [local_detector.py / Standalone EXE]
            RTSP["Threaded RTSP Reader\n(Zero-latency frame queue)"]
            Detector["Person Detector\n• Roboflow RF-DETR (v2)\n• Ultralytics YOLO26s (v1)\n• RTMDet (ONNX)"]
            FaceDet["Face Detector\n• SCRFD (2.5G ONNX)\n• OpenCV YuNet (Anatomical Validation)"]
            Embedder["Biometric Face Extractor\n• ArcFace ResNet50 (512-D)\n• Calibrated Cosine Operating Thresh (>= 0.65)\n• Landmark Affine Alignment"]
            BodyEmbedder["Body Re-ID & Appearance\n• OSNet Feature Extractor (512-D)\n• Pose-Located Torso & Legs (YOLO Pose)\n• Wall White-Balance & Relative Lightness\n• CIE Lab Soft Histograms & Sleeve Check\n• Multi-Factor Appearance Verification"]
            Tracker["Multi-Object Tracker\n• ByteTrack (bytetrack.yaml)\n• BoT-SORT + GMC (botsort.yaml)\n• Fallback Kalman Tracker"]
            Handoff["Spatial-Temporal Handoff Tracker\n• 25s Multi-Cue Coasting Pool\n• Torso HSV Color Correlation\n• Unbroken Waiting Timers"]
            StaffClassifier["Employee vs Visitor Classifier\n• Cosine Similarity (thresh = 0.65)\n• Temporal Voting Buffer (win = 5)"]
            EntryGate["Entry-Line Gating & Ghost Elimination\n• Confirmed vs Unconfirmed Passersby\n• Neutral Slate (180,180,180) Bboxes\n• 7s Dwell Bounce Filter\n• Interior Ingress Suppression"]
            PhotoPipeline["Visitor Photo & Exit Optimization\n• Multi-Frame Best-Shot Scoring\n• Occlusion-Aware Upgrades (3.5s)\n• CLAHE Backlight & HSV Attire\n• Departure Capture & Turnaround Forwarding"]
            ZoneTracker["Visitor Zone & Dwell Tracker\n• Live Zone Dwell Accumulator\n• Reverse Exit Line Detector\n• 2-Param Waiting SLA Monitor\n• Cross-Camera Waiting Sync"]
            LineZone["Spatial Analytics Engine\n• Directional Entry-Line Crossing\n• Dynamic Department Dwell Timers"]
            Preview["Local OpenCV Preview Window\n(omitted with --headless)"]
            HotReload["Config Hot-Reloader\n(Watches camera_config.json mtime)"]
        end
        
        SyncQueue["Background Sync Client (DashboardSyncClient)\n• In-Order FIFO Queue\n• Retry While Backend Unreachable (≤ 180 s)"]
    end

    subgraph Central Backend [FastAPI & Express Backend]
        API["FastAPI Backend (:8000)\n• /api/v1/cameras (Config, CRUD, Stream Testing)\n• /api/v1/visitors (Detect, Tag, Dwell, Weak-Band Exit)\n• /api/v1/departments (Dynamic CRUD)\n• /api/v1/employees (Profiles, Photos, Gallery)\n• /api/v1/attendance (Punch & Daily Report)\n• /api/v1/analytics (Footfall & Heatmaps)\n• /api/v1/identity (Retention & Cleanup)"]
        RetentionWorker["Scheduled Retention Worker\n(7-Day Rolling Biometric Purge\n& Physical Snapshot Cleanup)"]
        AuthSrv["Express Auth Server (:5000)\n• /api/auth/login & /register\n• Admin Role-Based Access"]
        ServerWorker["Optional Server VideoWorker\n(app/workers/video_worker.py)"]
    end

    subgraph Persistence [PostgreSQL Database with pgvector]
        DB[("PostgreSQL 14+ (:5432)\n• cameras & departments\n• employees & employee_attendance_logs\n• employee_daily_attendance\n• person_identities (pgvector 512 face + 512 body Re-ID)\n• visitor_tracks & footfall_events\n• detection_events & processing_runs\n• admins")]
    end

    subgraph Client Layer [Modern Web UI]
        UI["React 19 Dashboard (:5173)\n(Vite + TypeScript + Tailwind CSS)\n• Dynamic Camera & AI Settings Panel\n• Waiting Time & SLA Alert Settings\n• Visit Purpose Tagging & Zoom Lightbox\n• Dual Entry vs Departure Lightbox Review\n• Real-Time Zone Dwell & Journey Analytics\n• Staff Attendance & Daily Rolls\n• Section Dwell Time & Export Reports"]
    end

    CCTV -->|RTSP TCP Stream| RTSP
    RTSP --> Detector
    Detector --> FaceDet
    FaceDet --> Embedder
    Embedder --> BodyEmbedder
    BodyEmbedder --> Tracker
    Tracker --> Handoff
    Handoff --> StaffClassifier
    StaffClassifier --> EntryGate
    EntryGate --> PhotoPipeline
    EntryGate --> ZoneTracker
    ZoneTracker --> LineZone
    LineZone --> SyncQueue
    LineZone -.->|Annotated Frame| Preview
    HotReload -.->|Zero-Downtime Reload| LineZone

    SyncQueue -->|HTTP REST JSON| API
    CCTV -.->|Optional Direct RTSP| ServerWorker
    ServerWorker -->|Ingestion| DB
    API -->|SQLAlchemy 2.x ORM| DB
    RetentionWorker -->|Purge Expired Vectors & Physical Files| DB
    AuthSrv -->|pg Pool / bcrypt| DB
    UI -->|REST API Requests| API
    UI -->|Authentication Requests| AuthSrv
```

Browser live preview is currently disabled. `GET /api/v1/cameras/{id}/frame` and `/stream` return `410 Gone`, and `local_detector.py` has no stream server. Annotated video is shown only in each detector's OpenCV window.

---

## 2. Core Computer Vision & Biometric Pipeline

### 2.1 RTSP Stream Ingestion & Non-Blocking Draining
* **TCP Transport (`rtsp_transport;tcp`)**: UDP RTSP feeds drop packets over congested LANs, creating H.264 macroblock corruption and frozen frames. The system globally enforces TCP via OpenCV FFmpeg flags.
* **Threaded Frame Draining (`ThreadedRTSPReader`)**: Standard OpenCV `cv2.VideoCapture` maintains an internal multi-frame decode buffer. If deep inference takes 30ms, frames queue up and create massive lag. `ThreadedRTSPReader` spawns a dedicated daemon thread running `grab()` continuously, storing solely the latest decoded frame. Latency from physical camera lens to inference input is kept under **30–50ms**.

### 2.2 Multi-Model Person Detection Architecture
The edge engine supports dynamic runtime switching across detection backends:
1. **Roboflow RF-DETR (`--v2`)**: Real-Time Detection Transformer with DINOv2 self-supervised visual backbone. Delivers unmatched precision in crowded environments, eliminating false merges and detecting seated, partially occluded, or distant persons.
2. **Ultralytics YOLO (`--v1`)**: Ultra-fast convolutional detection engine (defaults to `models/yolo26s.pt`). Provides rapid inference (15–30ms per frame on modern GPUs or 50–70ms on modern multi-core CPUs).
3. **RTMDet Support**: ONNX runtime engine optimized for high-throughput edge devices.

Inference input resolution defaults to `960x960` or `1024x1024` pixels (`--imgsz 960`), ensuring small faces and distant visitors entering through showroom glass doorways are detected reliably.

### 2.3 Neural Face Detection & Quality Gating
Face detection operates under a hierarchical upper-torso spatial constraint and multi-stage anatomical quality verification:
* **Detector Models**: SCRFD (`scrfd_2.5g_bnkps.onnx`) or OpenCV YuNet (`face_detection_yunet_2023mar.onnx`).
* **Spatial Alignment**: Face boxes are strictly validated within the top 35% of the person's bounding box. This prevents false face triggers from clothing graphics, catalog prints, or posters.
* **Biometric Quality & Anatomical Filters**:
  * **Face Blur Filtering**: Laplacian variance must exceed `FACE_BLUR_THRESHOLD = 5.0` to reject fast-motion blur.
  * **Head Pose Limits**: Maximum yaw $\le 65^\circ$, pitch $\le 50^\circ$, and roll $\le 40^\circ$ (specifically accommodating $35^\circ$–$45^\circ$ overhead CCTV angles).
  * **Inter-Ocular Distance**: Minimum eye distance $\ge 2.5$ pixels to reject unresolvable distant faces.
  * **Illumination Gating**: Mean grayscale illumination between 18.0 and 245.0 to filter extreme backlight silhouettes or lens glare.
  * **Anatomical Bounding Box Constraints (YuNet `detect_and_crop_face`)**:
    - Minimum face dimensions: $w \ge 16\text{ px}$, $h \ge 18\text{ px}$ with confidence threshold $\ge 0.40$ (score threshold $0.35$).
    - Maximum face proportion: Face box cannot occupy $> 75\%$ of crop width or $> 65\%$ of crop height.
    - Aspect ratio check: $0.45 \le \frac{w}{h} \le 1.55$.
    - **Upper-Body Anatomical Restriction**: For standing person crops ($h > w \times 1.15$), the vertical face center must satisfy $c_y \le 0.50 \cdot h$. This rigorously rejects false face triggers on circular floor items such as waste bins, shoe heels, or floor bag buckles.

### 2.4 ArcFace Biometric Feature Extraction
For validated faces:
* **Canonical 5-Point Alignment**: Faces are geometrically aligned to canonical $112 \times 112$ ArcFace reference landmarks using affine similarity transformation.
* **Deep Feature Representation**: Passing through ArcFace ResNet50 (`models/w600k_r50.onnx`) outputs a 512-dimensional feature vector.
* **L2-Normalization**: Feature vectors are L2-normalized:
  $$\mathbf{e} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}, \quad \|\mathbf{e}\|_2 = 1.0$$
* **Distance Metric**: Similarity between two face embeddings $\mathbf{e}_1, \mathbf{e}_2$ is calculated as dot-product cosine similarity:
  $$\text{CosineSimilarity}(\mathbf{e}_1, \mathbf{e}_2) = \mathbf{e}_1 \cdot \mathbf{e}_2$$
* **Calibrated Operating Thresholds**:
  - Staff / Visitor Matching Threshold: Calibrated operating threshold **$\ge 0.65$** (`FACE_SIMILARITY_THRESHOLD = 0.65`) providing high biometric precision.
  - Candidate Match Margin: Minimum margin between Top-1 and Top-2 candidate matches $\ge 0.030$.

### 2.5 Deep Full-Body Re-Identification & Multi-Part Appearance Modeling (`BodyEmbedder`)
In overhead showroom CCTV feeds, visitors frequently turn 180° away from the entrance, bend over jewelry showcases, or wear hats/masks where facial biometrics are temporarily unavailable. The system deploys `BodyEmbedder` for deep full-body Re-ID combined with multi-part 3D color histograms, adaptive salient patch extraction, and hard contrast/polarity vetoes:

1. **OSNet Feature Extraction**:
   - Built on Omni-Scale Network (`osnet_x1_0` or `osnet_x0_25` ONNX/PyTorch models).
   - Generates a **512-dimensional L2-normalized deep appearance vector** capturing multi-scale bodily features, proportions, and clothing patterns.
2. **Aspect-Ratio Adaptive Salient Torso Extraction (`extract_salient_torso_patch`)**:
   - Evaluates bounding box aspect ratio $AR = h / w$:
     - For bust / portrait crops ($AR < 1.20$), torso fabric resides in the lower half: band $y \in [0.35 h, 0.85 h]$.
     - For full-body crops ($AR \ge 1.20$), torso fabric resides in the upper-body band: $y \in [0.18 h, 0.50 h]$.
   - Operates a horizontal sliding window ($w_{\text{win}} = \max(20, \text{int}(bw \times 0.35))$) across horizontal columns from $0.08 bw$ to $0.92 bw - w_{\text{win}}$.
   - Computes fabric prominence against neutral/white showroom walls using chromatic saturation $S$ and darkness contrast $V$:
     $$\text{prominence} = \bar{S} + \max(0.0, \; 140.0 - \bar{V}) \times 0.40$$
     $$\text{score} = \text{prominence} \times \left(1.0 - 0.40 \cdot \frac{|x + 0.5 w_{\text{win}} - c_x|}{0.5 bw}\right)$$
   - The column window with the highest score isolates the actual shirt/top fabric, eliminating background walls, door frames, or reflections.
3. **Salient Lower-Body Legs Patch Extraction (`extract_salient_legs_patch`)**:
   - Slides across the lower-body band ($y \in [0.54 h, 0.90 h]$) with center-prior weighting to isolate trousers, jeans, or skirts ($w_{\text{win}} = \max(20, \text{int}(bw \times 0.35))$).
   - Rejects non-clothing background floor tiles, door mats, and carpet patterns.
4. **Pose-Located Garment Regions (`app/vision/clothing_color.py`)**:
   - A YOLO pose model (`CLOTHING_POSE_MODEL_PATH`, default `models/yolo11n-pose.pt`, auto-downloaded on first use) finds shoulders, hips, knees and ankles in the person crop.
   - **Torso** = the shoulder→hip quad from 12% to 80% of the way down, shrunk 18% from each side (keeps out collar/head, arms and background). Hidden hips are extrapolated as $1.4\times$ shoulder width below the shoulders.
   - **Legs** = thick strokes hip→knee→ankle. Without a usable pose the code falls back to a fixed central box (bust crops: $y \in [0.45, 0.85]$; full body: $y \in [0.24, 0.46]$).
   - Fixed-fraction boxes were dropped because on the Main Gate camera they were mostly wall and the person's head: white shirts read Gray/Cream, and every person's torso histogram matched every other person's.
5. **Garment Colour (`estimate_clothing_appearance` → `clothing_color.describe`)**:
   - **White balance** against bright neutral background pixels; **lightness relative to the wall**: $L_{rel} = L^*_{\text{garment}} / P_{90}(L^*_{\text{background}})$, so the same shirt reads the same lit in the doorway and in shadow.
   - Dominant colour = largest of 3 k-means clusters in CIE Lab. **Patterned** if $\sigma_{L} \ge 19$ and $P_{90}-P_{10} \ge 50$.
   - Naming (chroma $C = \sqrt{a^2+b^2}$, hue $h$): $C \ge 12$ → Red / Yellow / Green / Blue / Purple by hue; navy if $b \le -6.4$ (dark garments carry a camera blue cast of $b \approx -4$); Pink, Brown / Rust, Cream / Khaki, Green / Olive (sage: $a \le -1.6$) for muted tones; otherwise Black ($L_{rel} < 0.33$), Gray ($< 0.62$), White.
   - Tags: *Black / Dark Attire*, *Gray Attire*, *White / Light Shirt*, *Cream / Khaki Shirt*, *Brown / Rust Top*, *Red / Maroon Top*, *Pink / Peach Top*, *Yellow / Mustard Top*, *Green / Olive Top*, *Blue / Navy Top*, *Purple / Violet Top*, *Checked / Patterned Shirt*.
   - Measured on 90 hand-labelled Main Gate snapshots: 62% exact / 89% acceptable (vs. 20% / 38% before). The thresholds were fitted on that same set.
6. **Verification (`verify_appearance`)**:
   - Similarity = Bhattacharyya coefficient of soft $(L_{rel}, a^*, b^*)$ histograms. $\text{clothing\_sim} = 0.65\,\text{torso} + 0.35\,\text{legs}$ (torso only if either crop lacks legs). Legs weigh less because most visitors wear dark trousers.
   - **Face veto** (unchanged): both faces frontal and sharp and ArcFace $< 0.40$ → reject.
   - **Incompatible colour veto**: torso tags not equal and not lighting neighbours (e.g. Gray↔White/Cream/Green/Black, Blue↔Black) and $\text{reid} < 0.82$ → reject. A patterned top is compatible with any neutral (prints are often front-only, and exits show the back).
   - **Trouser veto**: $|L_{rel,1} - L_{rel,2}| \ge 0.40$ with the darker $\le 0.35$ (dark trousers vs light jeans) and $\text{reid} < 0.85$ → reject. Applied only when both crops have *reliable* legs (both knees at confidence $\ge 0.75$ and the two legs within 0.25 of each other in lightness; side-on walkers get knees placed between the legs, which samples the floor) and never when $\text{reid} \ge 0.78 \land \text{torso\_sim} \ge 0.85$.
   - **Match** only if $\text{reid} \ge 0.70 \land \text{torso\_sim} \ge 0.40$ (score $0.6\,\text{reid} + 0.4\,\text{clothing}$). There is deliberately no low-reid branch: dark tops with jeans are too common, and a reid 0.62-0.70 \"clothing agrees\" branch departed the wrong visitor (a long-sleeve shirt matched to a short-sleeve tee). A missed exit is preferred to a wrong one.
   - **Weak Re-ID band (backend exit resolver, `visitor_service.update_visitor_zone`)**: front-in vs side/back-out views of the same person often score $0.60 \le \text{reid} < 0.70$. When no candidate passes the main rule, the top band candidate is accepted only if (a) $\text{torso\_sim} \ge 0.80$, colours are compatible, sleeves (short = warm skin-coloured forearms, long = fabric) do not contradict, and reliable trousers do not contradict (`clothing_color.weak_reid_support`); and (b) its reid beats every other in-store visitor by $\ge 0.08$ (with a single visitor, against a stranger baseline of 0.50). Logged in `exit_debug_backend.jsonl` under `weak_band`. Non-matches are capped at 0.45, below the backend's 0.48 exit-eligibility bar.
   - Known limit: two different people in the same plain colours (white shirt + dark trousers) are only separated by reid.

### 2.6 Edge Sync Queue with Reconnect Retry (`DashboardSyncClient`)
Edge-to-backend communication uses HTTP REST (`urllib`). Backend reloads (`uvicorn --reload`), restarts and network blips would otherwise lose footfall counts, visitor registrations or exits.

`DashboardSyncClient` (top of `local_detector.py`) works as follows:
1. **Dedicated Worker Thread**: A daemon thread drains a FIFO `queue.Queue(maxsize=1000)`, so video decoding never waits on network I/O. Events enqueued while the queue is full are dropped.
2. **Error Classification (`_is_unreachable_error`)**: an error counts as "unreachable" only when the request certainly never reached the backend: `ConnectionRefusedError` / `ConnectionAbortedError`, `actively refused`, `WinError 10061`, no route to host, network unreachable, `getaddrinfo failed`. **Timeouts are excluded**, because the backend may already have processed the request and a retry could duplicate it.
3. **In-Order Retry (`_send_with_retry`)**:
   - Retried event types (`_RETRY_EVENTS`): `entry`, `section_entry`, `employee_punch`, `visitor_detect`, `visitor_zone_update`, `delete_visitor`.
   - The worker retries the **same** event in place with backoff ($1\text{s} \to 2\text{s} \to 4\text{s}$, capped at $10\text{s}$). Nothing behind it is sent meanwhile, so `/visitors/detect` always reaches the backend before the visitor's `/visitors/update-zone` exit.
   - Retries stop after `_RETRY_MAX_WAIT_S = 180 s`. The event is then dropped and logged as `edge_send_dropped` in `logs/exit_debug_edge.jsonl`.
   - When a held `/visitors/detect` finally succeeds, the returned `person_id` is passed to `on_visitor_registered` as usual.
4. **Heartbeats are not retried**: a failed heartbeat is simply skipped, so stale telemetry never delays visitor events after reconnect.
5. **No 4xx/5xx Retries**: An HTTP error response (e.g. `404` for an exit on an already-departed visitor, `400`) is a final answer from the backend and is never retried.

---

## 3. Employee Recognition & Automated Attendance Flow

```mermaid
sequenceDiagram
    autonumber
    participant Cam as CCTV / RTSP Stream
    participant Edge as Edge AI Engine (local_detector.py)
    participant Classifier as EmployeeRecognitionManager
    participant Sync as Background Sync Client
    participant API as FastAPI Backend (:8000)
    participant DB as PostgreSQL (pgvector)
    participant UI as React Dashboard

    Cam->>Edge: Deliver latest frame
    Edge->>Edge: Detect Person + Extract ArcFace 512-D Embedding
    Edge->>Classifier: classify_face(track_id, embedding)
    
    rect rgb(240, 248, 255)
    Note over Classifier: Cosine Similarity Matching against Cached Employee Vectors
    Classifier->>Classifier: Compute Top-1 Score vs Top-2 Score
    alt Score >= 0.65 and (Top1 - Top2) >= 0.03
        Classifier->>Classifier: Record vote for Employee (EMP-101)
        Note over Classifier: Buffer temporal votes across sliding window (win = 5)
        alt Votes >= min_confirm_frames (e.g. 1-2 frames)
            Classifier-->>Edge: Identified as Employee: EMP-101 (Staff)
            Edge->>Edge: Lock track as "Employee". Exclude from Visitor Footfall!
            Edge->>Sync: Enqueue Attendance Punch Event ("CHECK_IN" or "PRESENCE")
        end
    else Below Similarity Threshold
        Classifier-->>Edge: Unidentified Face (Visitor)
    end
    end

    opt Attendance Punch Dispatch
        Sync->>API: POST /api/v1/attendance/punch<br/>{"employee_id": "EMP-101", "event_type": "CHECK_IN", ...}
        API->>DB: INSERT INTO employee_attendance_logs
        API->>DB: UPSERT INTO employee_daily_attendance<br/>(Calculate first_punch, last_punch, total_hours)
        API-->>Sync: 200 OK {"punch_recorded": true}
    end

    opt Visitor Crosses Entry Line (Entrance Camera)
        Edge->>Edge: Visitor Track #42 crosses virtual line
        Edge->>Sync: Enqueue Footfall Event ("entry", track_id: "42")
        Sync->>API: POST /api/v1/cameras/{id}/footfall
        API->>DB: INSERT INTO footfall_events (ON CONFLICT DO NOTHING)
        API-->>UI: Live Dashboard Real-Time Footfall Increment
    end
```

### 3.1 Multi-Frame Voting Buffer Guarantee
To guarantee zero false positives (ensuring a visitor is never mistakenly tagged as an employee):
1. **Calibrated Threshold**: Baseline cosine similarity requires $\ge 0.65$ (calibrated for high precision ArcFace biometric discrimination).
2. **Top-1 vs Top-2 Match Margin**: The gap between the best match and second-best candidate from a different employee must be $\ge 0.030$.
3. **Temporal Voting Buffer**: Successive observations over `EMPLOYEE_VOTE_WINDOW` frames must vote for the same employee identity before the track is tagged as staff. Once confirmed, the employee ID is bound to that track for its duration.

### 3.2 Daily Attendance Roll Computation
When punches are ingested:
* **First Seen**: Sets `first_punch` for the day. If `first_punch > shift_start_time + late_grace_minutes`, `is_late` flag is set to `True`.
* **Last Seen**: Progressively updates `last_punch`.
* **Working Hours**: Total work duration is continuously calculated:
  $$\text{WorkDuration} = \text{last\_punch} - \text{first\_punch}$$
* **Early Departure**: If last punch is before `shift_end_time - early_departure_grace_minutes`, `is_early_departure` flag is recorded.

---

## 4. Multi-Object Tracking & Motion Stabilization

### 4.1 Pluggable Trackers
The system provides two production tracker configurations:
* **ByteTrack (`bytetrack.yaml`)**:
  - Associates both high-confidence ($> 0.20$) and low-confidence ($> 0.10$) detection boxes across frames.
  - Maintains lost tracks for up to `track_buffer: 150` frames (~5–7 seconds), bridging temporary occlusions when visitors walk behind showroom display cases or pillars.
* **BoT-SORT (`botsort.yaml`)**:
  - Extends tracking with **Global Motion Compensation (GMC)** via sparse optical flow (`sparseOptFlow`).
  - Mitigates false track drift caused by camera pole vibration, wind sway, or manual PTZ adjustments.
  - Tracks direct bounding box width and height states $(x, y, w, h)$ for improved tracking across seated-to-standing posture shifts.

### 4.2 Track Smoothing & Jitter Deadband
Raw detections exhibit coordinate micro-jitter due to digital sensor noise. The edge engine applies:
* **Exponential Moving Average (EMA)** smoothing:
  $$\text{Box}_{t} = \alpha \cdot \text{Box}_{\text{detected}} + (1 - \alpha) \cdot \text{Box}_{t-1}, \quad \text{with } \alpha = 0.35$$
* **Stationary Deadband Filter**: Shifts smaller than `deadband = 6.0` pixels are suppressed, locking bounding boxes motionless when persons stand still.

### 4.3 Multi-Cue Spatial-Temporal & Torso Handoff Tracking (`SpatialTemporalHandoffTracker`)
Showroom environments frequently present tracking interruptions: visitors turn 180° away from the lens, bend down over glass display cases, or are occluded by structural pillars. Under conventional tracking, such events cause raw tracker IDs to reset, which resets dwell timers, fragments visitor profiles, and duplicates records. 

`SpatialTemporalHandoffTracker` solves this through a multi-cue appearance coasting engine:
1. **25-Second Coasting Pool**:
   When a track disappears from the active frame, its state (last bounding box, upper-torso HSV histogram, `person_id`, display alias, `first_seen_ts`, `waiting_seconds`, and `waiting_breach` flag) is moved to a coasting pool for up to $25.0\text{ seconds}$ (`max_coasting_sec = 25.0`).
2. **Upper-Torso HSV Color Histogram Extraction**:
   For each tracked person, an adaptive upper-torso region (from $20\%$ to $58\%$ of total box height, with $15\%$ horizontal margin) is extracted and converted to HSV color space. A 2D joint Hue-Saturation histogram (16 hue bins $\times$ 8 saturation bins) is calculated and L1-normalized:
   $$H(h, s) = \frac{\text{Hist}(h, s)}{\sum \text{Hist}(h, s)}$$
3. **Multi-Cue Candidate Matching**:
   When a new track ID appears, it is matched against the coasting pool using a composite criterion:
   - **Normalized Spatial Proximity**: Center displacement normalized by maximum box height:
     $$\text{norm\_dist} = \frac{\sqrt{(x_{\text{new}} - x_{\text{coast}})^2 + (y_{\text{new}} - y_{\text{coast}})^2}}{\max(h_{\text{new}}, h_{\text{coast}})} \le 1.6$$
   - **Bounding Box Scale Consistency**: Area ratio $\frac{\text{Area}_{\text{new}}}{\text{Area}_{\text{coast}}} \in [0.45, 2.2]$ ensures matches are posture-compatible.
   - **Torso Color Histogram Correlation**: Evaluated via Pearson correlation coefficient:
     $$\text{Corr}(H_{\text{new}}, H_{\text{coast}}) = \frac{\sum_i (H_{\text{new}}(i) - \bar{H}_{\text{new}})(H_{\text{coast}}(i) - \bar{H}_{\text{coast}})}{\sqrt{\sum_i (H_{\text{new}}(i) - \bar{H}_{\text{new}})^2 \sum_i (H_{\text{coast}}(i) - \bar{H}_{\text{coast}})^2}} \ge 0.55$$
4. **Seamless Identity & Timer Restoration**:
   When a match satisfies $\text{Corr} \ge 0.55$ within proximity limits, the new track inherits the original visitor metadata:
   $$\text{first\_seen\_ts}_{\text{new}} = \text{first\_seen\_ts}_{\text{coast}}$$
   $$\text{waiting\_seconds}_{\text{new}} = \text{waiting\_seconds}_{\text{coast}}, \quad \text{waiting\_breach}_{\text{new}} = \text{waiting\_breach}_{\text{coast}}$$
   $$\text{person\_id}_{\text{new}} = \text{person\_id}_{\text{coast}}, \quad \text{display\_id}_{\text{new}} = \text{display\_id}_{\text{coast}}$$
   This guarantees unbroken dwell accumulation, uninterrupted SLA monitoring, and zero duplicate customer records.

---

## 5. Visitor Photo Optimization & Purpose Tagging Pipeline

To ensure high-quality identification for showroom floor staff and front-desk receptionists tagging walk-in customers, the system employs an intelligent, multi-stage photo capture and progressive upgrade engine.

```mermaid
sequenceDiagram
    autonumber
    participant Cam as Entrance Camera (RTSP)
    participant Edge as Edge AI Engine (local_detector.py)
    participant Backend as FastAPI Backend (:8000)
    participant Storage as File Storage (static/uploads)
    participant DB as PostgreSQL (person_identities)
    participant UI as React 19 Dashboard (Visit Purpose)

    Cam->>Edge: Frame t=0: Visitor steps through glass door (distant, 48px face)
    Edge->>Edge: Initial Detection (conf >= 0.55, q_score = 0.38)
    Edge->>Edge: Apply CLAHE backlight lift + HSV clothing estimation ("Navy Blue Top")
    Edge->>Backend: POST /api/v1/visitors/detect (initial snapshot @ 320x320)
    Backend->>Storage: Save VIS-2026-1042.jpg
    Backend->>DB: INSERT PersonIdentity (person_id, avatar_url, notes: "Appearance: Navy Blue Top")
    Backend-->>UI: Real-Time Table Row: VIS-2026-1042 appears with initial photo & badge

    Note over Edge: Visitor walks 3-5 steps closer to counter (t=1.4s, 140px face, frontal)
    Cam->>Edge: Frame t=1.4s: Closer, crisp frontal frame
    Edge->>Edge: Compute Quality Score (q_score = 0.82 >= 1.25 * old_score)
    Edge->>Edge: Progressive Upgrade Triggered!
    Edge->>Backend: POST /api/v1/visitors/detect (updated crisp snapshot @ 320x320)
    Backend->>Storage: Overwrite VIS-2026-1042.jpg with retina quality crop (Q95)
    Backend-->>UI: Avatar updates automatically with sharp portrait!

    UI->>UI: Receptionist clicks photo thumbnail -> Opens High-Res Lightbox
    UI->>UI: Clicks "Tag This Visitor" -> TagVisitorModal displays portrait & appearance
    UI->>Backend: PATCH /api/v1/visitors/VIS-2026-1042/tag (purpose: "Vehicle Enquiry", staff: "Rajesh")
    Backend->>DB: UPDATE PersonIdentity (tagging_status: "Tagged", visit_purpose: "Vehicle Enquiry")
```

### 5.1 Multi-Frame Best-Shot Quality Scoring
Rather than committing to the very first frame where a face is detected (which is typically far away, motion-blurred, or looking sideways at the entrance door), the edge engine continuously scores every frame using a composite metric:

$$Q = 0.35 \cdot S_{\text{sharpness}} + 0.25 \cdot S_{\text{resolution}} + 0.20 \cdot S_{\text{pose}} + 0.10 \cdot S_{\text{illumination}} + 0.10 \cdot S_{\text{confidence}}$$

1. **Sharpness ($S_{\text{sharpness}}$)**: Evaluated via Laplacian variance $\sigma^2(\nabla^2 I)$ over the face crop. Frames with walking motion blur get low scores; crisp, stabilized frames receive maximum weight.
2. **Resolution ($S_{\text{resolution}}$)**: Scaled by face area relative to canonical $120 \times 120\text{ px}$. Denser crops with clear biometric features are heavily prioritized.
3. **Frontality & Pose Symmetry ($S_{\text{pose}}$)**: Evaluates roll tilt angle via eye landmarks ($\text{atan2}(\Delta y, \Delta x)$) and yaw offset from nose-to-eye horizontal center. Frontal poses receive scores close to $1.0$.
4. **Illumination Balance ($S_{\text{illumination}}$)**: Penalizes severe silhouettes ($\text{mean lum} < 85$) and camera lens glare ($\text{mean lum} > 175$).

### 5.2 Dynamic Progressive Upgrade & Occlusion Recovery Policy
* **Zero-Latency Ingress**: The first valid face observation is dispatched immediately to the dashboard to avoid any delay in displaying new walk-ins.
* **3.5-Second Walk-In Window**: During the visitor's first 3.5 seconds of trajectory, any frame with $Q \ge 1.25 \cdot Q_{\text{best}}$ (or $Q \ge Q_{\text{best}} + 0.15$) with at least $400\text{ ms}$ separation triggers a seamless progressive photo upgrade.
* **Occlusion Recovery Gating**: When visitors enter through doorway thresholds or turnstiles, initial frames often capture a narrow, partially occluded sliver of the body (aspect ratio $AR = w / h < 0.30$). The engine evaluates bounding box geometry:
  $$\text{is\_box\_unoccluded} = (AR_{\text{prev}} < 0.30) \land (AR_{\text{curr}} \ge 0.32) \land (w_{\text{curr}} \ge 50\text{ px})$$
  When an unoccluded box appears, an immediate progressive upgrade is fired even if raw facial scores are comparable, replacing doorway slivers with clean, full-width visitor portraits.

### 5.3 Contextual Bust Framing & CLAHE Backlight Compensation
* **Bust & Torso Framing**: To enable floor staff to instantly recognize customers from a distance, portrait crops extend beyond the face boundary:
  $$\text{Top} = y_1 - 0.45 \cdot f_h, \quad \text{Bottom} = y_2 + 0.85 \cdot f_h, \quad \text{Sides} = x_{1,2} \pm 0.40 \cdot f_w$$
  This captures hairstyle, collar, neck, and upper-chest attire.
* **CLAHE Equalization**: High dynamic range backlighting from showroom glass facades is corrected in LAB color space by applying Contrast Limited Adaptive Histogram Equalization (`clipLimit = 2.2`, `tileGridSize = 8x8`) exclusively to the luminance ($L$) channel, lifting facial shadows without bleaching background highlights.
* **Retina Standard**: Snapshots are standardized to $320 \times 320\text{ px}$ with Lanczos4 interpolation (`cv2.INTER_LANCZOS4`) and encoded at JPEG quality 95.

### 5.4 Attire Classification
`estimate_torso_appearance` (edge) and `BodyEmbedder.estimate_clothing_appearance` (backend) both call `clothing_color.estimate_tag`. That is the pose-located, wall-relative CIE Lab pipeline in §2.5 (items 4–5), not a fixed HSV box. The tag (e.g. *Blue / Navy Top*, *White / Light Shirt*, *Checked / Patterned Shirt*) is stored in the visitor's notes as `Appearance: <tag>` and shown as a badge in the dashboard. On `/visitors/detect` the backend re-estimates the tag from the portrait when the edge sent a weak one (`Casual Attire`, `Walk-in Visitor`, `Gray Attire`).

### 5.5 Departure Snapshot Capture, Turnaround Forwarding & Dual-Photo Lightbox UX
When a visitor completes their visit and crosses the entrance line in reverse:
* **High-Resolution Exit Capture**: The edge engine grabs an un-annotated high-resolution portrait crop of the departing person and dispatches it via `exit_image_base64` in the departure update payload.
* **Turnaround Track Departure Forwarding**: When a customer turns 180° around towards the exit door, ByteTrack or BoT-SORT may allocate a fresh local tracker ID with local dwell $< 7.0\text{s}$. Instead of discarding the departure, `local_detector.py` forwards the departure crop to the backend appearance engine so active showroom visitors who entered earlier can be matched and departed!
* **Reliable Exit Face Biometric Gating**: Departing visitors often walk with head tilted downward or face angled away. Biometric face checks on exit are strictly gated by reliability criteria:
  - In `local_detector.py`: Inter-ocular distance $eye\_d \ge 16.0\text{ px}$ and face crop width $w \ge 42\text{ px}$.
  - In `VisitorService`: Inter-ocular distance $eye\_d \ge 16.0\text{ px}$ and face crop width $w \ge 28\text{ px}$.
  - Face similarity rejection threshold is calibrated to $< 0.40$. If exit face is angled or non-frontal, biometric veto is bypassed and identity is resolved via deep body Re-ID and appearance matching.
* **Edge-Supplied Exit IDs Need Evidence**: An exit only attaches to a visitor still inside today. A `person_id` / tracker from the edge is trusted outright only when it equals the visitor's registration tracker (first `Tracker #N` in notes). Otherwise the exit photo must back it (body `is_match` or reliable face $\ge 0.52$), or the exit falls through to the search over all active visitors. In that search, an eligible visitor whose OSNet Re-ID beats every other eligible one by $\ge 0.08$ wins over the combined-score leader.
* **Entry-Exit Match Confidence Reporting (`exitMatchConfidence`)**: The backend parses matching confidence from the latest `Exit Match: XX%` in record notes (e.g. `Exit Match: 82.5%`) or computes live verification score using `BodyEmbedder.verify_appearance(entry_crop, exit_crop)`, exposing `exitMatchConfidence` (e.g. `0.825`) in `VisitorResponse`.
* **Storage & Persistence**: Saved to `static/uploads/visitors/` as `VIS-..._exit_<timestamp>.jpg` and stored in `person_identities.exit_avatar_url`.
* **Dashboard Dual-Photo Lightbox UX (`VisitorsView.tsx` & `OverviewView.tsx`)**:
  - Visitor cards display a **"+1 Exit Image"** pill.
  - Interactive **Entry Snapshot** vs **Departure Snapshot** toggle tabs.
  - Live animated pulse badge: `ENTRY-EXIT MATCH [XX]%` (or `AI MATCH [XX]%` for entry).
  - AI Matching & Verification Debug Card detailing verification method (e.g. *"Verified via Deep OSNet Re-ID & Multi-Part Clothing Color Correlation"*).
  - Keyboard accessibility with Escape key support, backdrop click-to-dismiss, and sticky modal header/footer with scrollable content.

---

## 6. Real-Time Visitor Journey, Zone Dwell & Exit Tracking

Beyond basic ingress counts, modern showroom management demands full visibility into customer browsing patterns, department-level interest, and total visit durations. The system implements a dedicated, non-blocking **`VisitorZoneTracker`** to monitor customer dwell and capture departure events in Indian Standard Time (IST, UTC+05:30).

```mermaid
sequenceDiagram
    autonumber
    participant Cam as CCTV Cameras (Entrance & Floor)
    participant Edge as Edge AI Engine (VisitorZoneTracker)
    participant Sync as Background Sync Client
    participant API as FastAPI Backend (:8000)
    participant Storage as File Storage (static/uploads)
    participant DB as PostgreSQL (person_identities)
    participant UI as React 19 Dashboard (Visitors & Purpose)

    Cam->>Edge: Visitor #42 enters showroom (t=0)
    Edge->>Edge: Initial Zone: "Main Showroom" (dwell = 0s)
    
    loop Every 4.0 Seconds (Active Trajectory)
        Edge->>Edge: Evaluate foot/torso contact with Section Polygons
        alt Inside "Sales Section"
            Edge->>Edge: Increment "Sales Section" dwell (+delta_t)
        else In General Area
            Edge->>Edge: Increment "Main Showroom" dwell (+delta_t)
        end
        Edge->>Sync: send_visitor_zone_update(person_id, zone_dwell, top_zone)
        Sync->>API: POST /api/v1/visitors/update-zone
        API->>DB: UPDATE person_identities SET zone_dwell = ..., zone_entered = ...
        API-->>UI: Real-Time Dwell Badges: "Sales Section: 4m 12s", "Main Showroom: 1m 30s"
    end

    Note over Cam,Edge: Visitor finishes consultation and walks toward exit door (dwell >= 7s)
    Cam->>Edge: Frame t=940s: Foot crosses entry line in REVERSE (Outbound)
    Edge->>Edge: Reverse Line Crossing Detected! (rev_crossed = True)
    Edge->>Edge: Format Departure Time: "04:45 PM IST" & Grab Exit Portrait Crop
    Edge->>Sync: send_visitor_zone_update(exit_time="04:45 PM", is_departure=True, exit_image_base64=...)
    Sync->>API: POST /api/v1/visitors/update-zone
    
    rect rgb(240, 248, 255)
    Note over API: Multi-Factor Exit Verification Pipeline
    API->>API: Extract exit_face_emb (Reliable YuNet + ArcFace) & exit_body_emb (OSNet)
    API->>API: Verify Reliable Face Sim >= 0.40 & Body Appearance (verify_appearance)
    alt Candidate matches Face & Clothing Appearance
        API->>Storage: Save VIS-2026-1042_exit_1795180767.jpg
        API->>DB: UPDATE person_identities SET exit_time = "04:45 PM", exit_avatar_url = "...", notes = "... (Exit Match: 82.5%)"
    else Tracker ID flipped on turnaround
        API->>DB: Fallback search across today's still-inside visitors (Reliable Face >= 0.52 + margin, or unambiguous Appearance match)
        API->>DB: Auto-close stale prior-day unclosed visits ("Timed out")
        API->>Storage: Save exit snapshot to matched visitor profile
        API->>DB: UPDATE person_identities SET exit_time = "04:45 PM", exit_avatar_url = "..."
    end
    end
    API-->>UI: Dashboard updates visitor row with departure stamp, dwell breakdown, "+1 Exit Image" badge & match %!
```

### 6.1 Dwell Accumulation & Duration Formatting
* **Dynamic Dwell Bucketing**: For every active track, time spent inside each calibrated department polygon is accumulated. Time spent outside specific polygons is assigned to `"Main Showroom"`.
* **Human-Readable Formatter**: Dwell seconds are formatted into concise strings:
  $$\text{format\_duration}(s) = \begin{cases} 
  \text{Xs}, & s < 60 \\
  \text{Y mins}, & s \ge 60 \text{ and } s \pmod{60} = 0 \\
  \text{Ym Zs}, & s \ge 60 \text{ and } s \pmod{60} \ne 0
  \end{cases}$$
* **Periodic Synchronization**: The edge client bundles the dwell breakdown and flushes updates every `sync_interval = 4.0` seconds to `/api/v1/visitors/update-zone`.

### 6.2 Reverse Line Crossing, Premature Exit Suppression & Departure Verification
* **Directional Reversal & `EntryLineCounter.process_events`**: An entrance line configured for ingress monitors opposing cross-product trajectory ($y_{\text{curr}} < y_{\text{prev}}$ or reverse vector orientation). `EntryLineCounter.process_events()` evaluates both ingress crossings and egress crossings in a single pass.
* **Server-Side Video Worker Ingress & Egress (`VideoWorker`)**:
  The background ingestion worker (`app/workers/video_worker.py`) handles both events:
  - For showroom employees: clocks `CHECK_OUT` punch via `EmployeeAttendanceRepository.record_attendance_event`.
  - For visitors: enforces $7.0\text{s}$ minimum dwell window (`track_arrival_times`), extracts JPEG Q95 exit crop, converts to Base64, and calls `VisitorService.update_visitor_zone()` with `VisitorZoneUpdateRequest(exit_time, exit_image_base64)`.
* **7.0-Second Premature Exit Suppression**:
  To eliminate tracker noise, doorway turnarounds, or delivery people pacing at the threshold from triggering premature exits, `record_visitor_exit` enforces a minimum physical dwell threshold ($\text{dwell} \ge 7.0\text{ seconds}$). If reverse crossing fires with dwell $< 7.0\text{s}$ for an existing active session, the exit update is suppressed and not dispatched to the backend.
* **Turnaround Forwarding to Backend Appearance Engine**:
  If a turnaround track was newly assigned on departure ($< 7.0\text{s}$ presence) and was not part of an existing local session, it is forwarded directly to the backend appearance matcher so that an active showroom visitor who entered earlier can be matched and departed!
* **Edge Reverse-Crossing Branches** (main loop, for each `new_exits` track):
  1. **Staff** → `CHECK_OUT` punch (at most once per employee per 300 s).
  2. **Entry still pending** (< 7 s since crossing) → entry cancelled locally, footfall not counted. No backend delete is sent: the record is only minted after the 7 s confirmation, and a `person_id` on this track belongs to an existing visitor re-identified by face.
  3. **Visitor departure** (entrance cameras only) → exit crop taken from the clean frame → `VisitorZoneTracker.record_visitor_exit`.
* **`record_visitor_exit` (edge)**:
  - Exit face is used only when reliable ($eye\_d \ge 16\text{ px}$ and $w \ge 42\text{ px}$).
  - Track with no local session: the reliable exit face is compared against the other active sessions' face embeddings at cosine $\ge 0.65$.
  - Session found but its face contradicts the exit face ($< 0.40$): the session is dropped and its visitor sent as `exclude_person_id`.
  - Session with dwell $< 7$ s: suppressed if it has a visitor ID. Without an ID it is a fresh turnaround track and is forwarded to the backend.
  - Forwarded exits carry `person_id` from `track_to_pid` / the identity manager (not when contradicted), else a vector-store face match $\ge 0.65$, else `null`.
* **Backend Resolution Order (`update_visitor_zone`, when `exit_time` is set)**:
  1. **Direct lookup**: `person_id` (unless it equals `exclude_person_id`), then exact `Tracker #N` token among not-departed visitors.
  2. **Inside-today gate**: a direct record that is expired, already departed, or created before today (IST) is dropped.
  3. **Direct vetoes**: reliable exit face $< 0.40$ against the record, or body contradiction ($\text{reid} < 0.45 \land \text{clothing} < 0.15$, front-to-back turnarounds lower clothing naturally). Either veto drops the record and excludes it from the search.
  4. **Evidence gate**: if the tracker is not the record's registration tracker (first `Tracker #N`), the exit photo must back it: body `is_match` or reliable face $\ge 0.52$ (`DIRECT_EXIT_FACE_EVIDENCE`). Otherwise the record is dropped, without a veto, and re-scored in the search like everyone else.
  5. **Fallback search** over today's not-departed visitors (after auto-closing prior-day ones): face path, then body path, then weak Re-ID band (details below). If nothing passes, the exit is not attached and the request returns `404`.
* **Turnaround Fallback Search & Stale Record Auto-Closure**:
  If tracker ID flipped on a 180° exit turnaround (e.g. #2 became #6):
  - Stale unclosed visitor records from prior calendar days are automatically closed (`exit_time = "Timed out"`, `is_expired = True`) to prevent contamination.
  - Only visitors created today who have **not yet departed** (`exit_time IS NULL`) are candidates, so a later stranger's exit can never overwrite an earlier departure.
  - **Face path**: used only when the exit face is reliable ($eye\_d \ge 16\text{ px}$ and $w \ge 28\text{ px}$). Match requires cosine $\ge 0.52$ and a Top-1 vs Top-2 margin $\ge 0.035$ (or a single candidate). The body must not clearly contradict ($\text{reid} < 0.40 \land \text{clothing} < 0.20$ rejects). Blurry or side-profile faces skip this path, because ArcFace scores on them are poorly separated.
  - **Body path**: `verify_appearance` must pass with combined score $\ge 0.48$ and time in store $\ge 7.0\text{s}$. The highest combined score wins, unless another eligible visitor's Re-ID beats every other eligible one by $\ge 0.08$ (`REID_OVERRIDE_MARGIN`, the **Re-ID override**). Without an override, a runner-up within $0.03$ of the best (two similarly dressed visitors) leaves the exit unattached rather than departing the wrong person. An ambiguous result also skips the weak band.
  - Any candidate whose reliable face contradicts the exit face ($< 0.40$, or $< 0.38$ on the body path) is vetoed, as is any `exclude_person_id` sent by the edge.
* **Exact Tracker Lookup**: When the payload has no `person_id`, the backend resolves `tracker_id` against the `Tracker #N` note with an exact token match (`Tracker #4` never matches `Tracker #42`).
* **Edge Face-Contradiction Exclusion**: If the exit face contradicts the local session ($< 0.40$), the edge drops that visitor (it no longer falls back to the track's own `person_id`) and sends `exclude_person_id` so the backend cannot re-select it.
* **Weak Re-ID Band ($0.60 \le \text{ReID} < 0.70$) Departure Resolver**:
  - When no candidate passes the primary appearance verification threshold ($\text{ReID} \ge 0.70 \land \text{torso\_sim} \ge 0.40$), the top candidate in the weak Re-ID band ($0.60 \le \text{ReID} < 0.70$) is evaluated.
  - Requires `clothing_color.weak_reid_support`: $\text{torso\_sim} \ge 0.80$, compatible attire tags, sleeve consistency (short skin-toned forearms vs long fabric sleeves), and trouser agreement.
  - Requires candidate to beat every other active showroom visitor by $\ge 0.08$ (or beat stranger baseline $0.50$ for solitary visitors).
  - Unmatched scores are capped at $0.45$, safely beneath the backend's $0.48$ exit eligibility threshold.
* **Safe Collision-Resistant Visitor ID Minting**:
  - `VisitorService._mint_person_id()` uses atomic count-based generation (`VIS-2026-XXXX`) with rigorous database uniqueness checks.
  - Guarantees that even if visitor rows are deleted or concurrent transactions run, new visitors never overwrite existing active identities.
* **Persisted Match Score**: Every attached exit appends `Exit Match: XX% (...)` to notes (face, body or weak-band score), so `exitMatchConfidence` is read from notes instead of recomputing OSNet on every dashboard poll. A visitor can carry several scores from exits that were later re-opened. Both `latest_exit_match_pct` (backend) and `VisitorsView.tsx` use the **last** one.
* **Exit Debug Logs**: one JSON line per decision.
  - `logs/exit_debug_edge.jsonl` (edge): `edge_entry_confirmed`, `edge_reverse_crossing` (branch `staff_checkout` / `bounce_cancelled_pending_entry` / `visitor_departure` / `ignored_not_entrance_camera`), `edge_exit_suppressed`, `edge_exit_resolved`, `edge_exit_sent`, `edge_send_dropped`.
  - `logs/exit_debug_backend.jsonl` (backend, `stage: backend_exit`): direct record and why it was rejected, per-candidate face/reid/clothing scores and eligibility, `reid_override`, `ambiguous`, `weak_band`, `resolved_to`. The crop of each unresolved exit is saved to `logs/exit_debug/`. Backend logging is off when `APP_ENV=testing`.
* **Indian Standard Time (IST) Clock**: When reverse crossing is confirmed, the departure timestamp is calculated in `Asia/Kolkata` time (`UTC+05:30`) and recorded:
  ```python
  exit_time = datetime.fromtimestamp(now_ts, tz=IST).strftime("%I:%M %p")
  ```
* **Bounced Visitor Removal Endpoint (`DELETE /api/v1/visitors/{person_id}`)**:
  If a visitor record needs to be removed (e.g. bounce cancellation or mistaken manual tag), the endpoint completely removes the `person_identities` record, linked `footfall_events`, `detection_events`, and unlinks image snapshots from disk.
* **Summary Telemetry**: The final zone dwell breakdown is emitted to system logs and persisted to the visitor's database record.

### 6.3 Staff Proximity Detection & Ground-Plane Proximity Engine
To distinguish whether a browsing customer is receiving assistance or waiting unattended, the edge vision pipeline computes real-time ground-plane spatial proximity between visitors and showroom staff (`is_staff_near_box`):
1. **Perspective-Calibrated Ground Contact**:
   The feet contact points of the visitor $(x_v, y_v)$ and employee $(x_s, y_s)$ are extracted from bottom-center coordinates:
   $$\mathbf{P}_{\text{feet}} = \left( \frac{x_1 + x_2}{2}, \; y_2 \right)$$
2. **Perspective Pixel-to-Meter Scaling**:
   Because camera perspective distorts pixel distances depending on distance from the lens, average bounding box height $\bar{h} = \frac{h_v + h_s}{2}$ serves as an adaptive reference scale calibrated to a standard human height ($1.70\text{ m}$):
   $$\text{Scale}_{\text{px/m}} = \max\left(1.0, \; \frac{\bar{h}}{1.70}\right)$$
3. **Proximity Radius Gating**:
   Euclidean ground distance in meters is calculated:
   $$d_m = \frac{\sqrt{(x_v - x_s)^2 + (y_v - y_s)^2}}{\text{Scale}_{\text{px/m}}}$$
   If $d_m \le \text{radius\_m}$ (default $2.2\text{ m}$), the visitor is marked as actively engaged (`is_employee_nearby = True`).

### 6.4 Visitor Waiting Timer & Configurable SLA Breach Pipeline
Showroom customer experience hinges on prompt staff engagement. The platform implements a strict **2-Parameter Unattended Waiting Logic** evaluated every frame:

$$\text{is\_waiting\_now} = (\neg \text{is\_employee\_nearby}) \land (\neg \text{is\_in\_dept})$$

* **Condition 1**: No registered employee is within the $2.2\text{ m}$ perspective radius.
* **Condition 2**: The visitor has not entered any designated department counter box (e.g., Sales Section, Consultation Desk).

When both conditions hold true, `waiting_seconds` accumulates continuously.

```mermaid
stateDiagram-v2
    [*] --> Attended: Enters Department Box OR Staff Within 2.2m
    Attended --> WaitingNormal: Leaves Dept AND No Staff Nearby (t < 5m)
    
    state WaitingNormal {
        [*] --> EmeraldGreen: 0s - 300s (0-5 mins)
        EmeraldGreen: Bounding Box: Neon Green (0, 240, 110)
        EmeraldGreen: Label: ⏱️ WAITING [MMm SSs] #ID
    }

    WaitingNormal --> AmberWarning: t >= 300s (5-15 mins)
    state AmberWarning {
        [*] --> Amber: 300s - 900s (5-15 mins)
        Amber: Bounding Box: Amber Warning (0, 140, 255)
        Amber: Label: ⚠️ WAITING [MMm SSs] #ID
    }

    AmberWarning --> SLABreach: t >= 900s (Cutoff > 15m Breached!)
    state SLABreach {
        [*] --> BrightRed: > 900s (> 15 mins)
        BrightRed: Bounding Box: High-Alert Bright Red (0, 0, 255)
        BrightRed: Label: 🚨 UNATTENDED > 15m [MMm SSs] #ID
        BrightRed: Header Badge: Solid Red (0, 0, 180)
        BrightRed: Console Alarm: [🚨 UNATTENDED SLA BREACH > 15m]
    }

    WaitingNormal --> Attended: Staff Approaches (< 2.2m) OR Enters Dept
    AmberWarning --> Attended: Staff Approaches (< 2.2m) OR Enters Dept
    SLABreach --> Attended: Staff Approaches (< 2.2m) OR Enters Dept
```

* **Three-Tier Visual HUD Alerts**:
  1. **Normal Waiting ($< 5\text{ mins}$)**: Neon Emerald Green bounding box `(0, 240, 110)` with `⏱️ WAITING [MMm SSs]` timer.
  2. **Elevated Warning ($5\text{ to }15\text{ mins}$)**: High-contrast Amber border `(0, 140, 255)` with `⚠️ WAITING [MMm SSs]` warning.
  3. **Critical SLA Breach ($> 15\text{ mins}$ / $900\text{s}$)**: Bright Red alert box `(0, 0, 255, thickness=3)` and red header badge `🚨 UNATTENDED > 15m [MMm SSs]`.
* **Monotonic Dwell Protection & Physical Elapsed Bounds**:
  - The backend `VisitorService.update_visitor_zone` enforces monotonic protection:
    $$\text{rec.waiting\_seconds} = \max(\text{rec.waiting\_seconds}, \; \text{req.waiting\_seconds})$$
    preventing any secondary camera or fresh track from rolling back waiting counts.
  - The edge engine enforces real physical dwell limits:
    $$\text{waiting\_seconds} \le \max(0.0, \; \text{now\_ts} - \text{first\_seen\_ts})$$
    guaranteeing that waiting timers cannot exceed physical time elapsed in the showroom.
* **Configurable SLA Cutoff**:
  The cutoff threshold is fully configurable in the React Dashboard ("Waiting Time & SLA Alerts" tab in Settings) and stored in `camera_config.json` (`waiting_cutoff_seconds`, default $900\text{s}$ / $15\text{ mins}$).
* **High-Resolution SLA Breach Snapshot Capture**:
  When a visitor transitions into SLA breach state, `local_detector.py` automatically captures an un-annotated full-resolution frame from the RTSP feed, saves it to `static/uploads/visitors/`, and dispatches it for immediate managerial action.
* **Telemetry Synchronization & Cross-Camera Waiting Sync**:
  `waiting_seconds`, `is_waiting`, and `waiting_breach` are synced every $4.0\text{s}$ via `/api/v1/visitors/update-zone` and stored in PostgreSQL `person_identities`. Furthermore, `CrossCameraTracker` synchronizes waiting metrics across all cameras via atomic state updates in `.cross_camera_state.json`.

### 6.5 Entry-Line Gated Ingress & Ghost Visitor Elimination
Showrooms with glass facades or open frontages often face challenges with transient passersby, outdoor foot traffic, or glass reflections triggering false detections. The platform implements a rigorous multi-tier gating architecture:

1. **Unconfirmed Bystander Segregation**:
   - A person detected in camera view who has not yet crossed the calibrated directional entry line is tagged `is_confirmed_visitor = False`.
   - The HUD renders an unconfirmed person with a neutral slate/silver bounding box `(180, 180, 180)` and simple tracker alias (`Person #ID (Unconfirmed)`).
   - Unconfirmed tracks **never** trigger walk-in visitor creation, do not increment footfall counters, do not accumulate waiting timers, and do not capture photos.
2. **Directional Virtual Entry-Line Crossing & 7-Second Dwell Threshold**:
   - When a person crosses the entry line in the configured ingress direction, they are held in a pending state with a **7.0-second minimum dwell window**.
   - If the person turns around or disappears from view in under 7 seconds, the crossing is discarded as a transient bounce (no footfall count added, no profile minted, no photo saved).
   - If the person is identified as a showroom employee, an attendance `CHECK_IN` punch is recorded immediately without waiting.
   - Once a non-staff person sustains showroom presence for $\ge 7.0\text{ seconds}$, they are confirmed as a genuine visitor (`is_confirmed_visitor = True`), footfall increments by +1, best-shot portrait crop is dispatched, and waiting SLA monitoring begins.
3. **Reverse Line Departure & Staff Checkout**:
   - When an active, confirmed visitor leaves in reverse after staying $\ge 7.0$ seconds, their departure timestamp is recorded and final dwell breakdown is persisted to the database. Reverse crossings under 7 seconds are ignored to eliminate jitter.
   - When a confirmed showroom employee crosses in reverse, their `CHECK_OUT` attendance punch is logged automatically.
4. **Interior Camera Ingress Guard (`IGNORED_INTERIOR`)**:
   - In-store floor and display cameras (`is_entrance: false`) suppress minting new walk-ins entirely.
   - If an interior camera detects a person who was not previously registered at an entrance camera or matched via ArcFace biometric re-identification, the backend logs:
     `Interior camera: suppressed new visitor creation (no entrance line crossing)`
     and responds with `status: "IGNORED_INTERIOR"`. This eliminates false walk-ins from staff entering via side doors or reflections.

---

## 7. Virtual Entry-Line Crossing & Spatial Departments

### 7.1 Ground-Plane Contact Calculation
To eliminate false triggers from head bobbing or gestures, line crossings are computed solely from the **bottom-center ground contact point**:
$$\text{Point}_{\text{feet}} = \left( \frac{x_1 + x_2}{2}, \; y_2 \right)$$

### 7.2 2D Intersection Geometry
A track moving from position $A(x_A, y_A)$ to $B(x_B, y_B)$ crosses calibrated line segment $P_1(x_1, y_1) \to P_2(x_2, y_2)$ if:
1. Segments $AB$ and $P_1P_2$ intersect (tested via orientation cross products):
   $$\text{ccw}(A, B, C) = (C_y - A_y) \cdot (B_x - A_x) > (B_y - A_y) \cdot (C_x - A_x)$$
2. Vector $A \to B$ matches the configured direction (`top_to_bottom`, `bottom_to_top`, `left_to_right`, or `right_to_left`).

### 7.3 Multi-Tier Duplicate Prevention
Double counting is strictly prevented at two architectural tiers:
* **Tier 1 (In-Memory)**: Edge engine maintains `counted_track_ids: Set[str]`. Once crossed, the tracker ID is permanently locked.
* **Tier 2 (PostgreSQL)**: The `footfall_events` table enforces `UNIQUE(visitor_track_id)`. Duplicate requests return HTTP 200 without inserting duplicate rows.

### 7.4 Entrance vs. In-Store Floor Camera Segregation
To support multi-camera showroom deployments without skewing aggregate analytics:
* **Entrance Cameras (`is_entrance = True`)**: Configured at showroom entry/exit doors. These cameras process walk-in footfall counts, dispatch initial visitor profile creation, and detect reverse-line departures.
* **In-Store Floor Cameras (`is_entrance = False`)**: Positioned over interior sales desks, jewelry showcases, or waiting lounges. These cameras continuously track customer trajectories, department dwell times, and staff interactions **without** emitting redundant walk-in footfall entries.

---

## 8. Dynamic Multi-Camera Management & Edge Hot-Reloading

The platform decouples camera administration from hardcoded code, providing dynamic CRUD and real-time synchronization between the React web UI, FastAPI backend, and edge detector engines.

```mermaid
sequenceDiagram
    autonumber
    participant Admin as Showroom Manager (Web UI)
    participant UI as CameraSettingsSection.tsx
    participant API as FastAPI Backend (:8000)
    participant File as camera_config.json
    participant Edge as Edge AI Engine (local_detector.py)

    Admin->>UI: Adds new RTSP Channel "102" (Floor Camera)
    UI->>API: POST /api/v1/cameras/test-stream {"stream_url": "rtsp://..."}
    API->>API: Test OpenCV VideoCapture (probe resolution & FPS)
    API-->>UI: {"online": true, "resolution": [1920, 1080], "fps": 25.0}

    Admin->>UI: Adjusts AI confidence slider to 0.35 & clicks "Save Settings"
    UI->>API: POST /api/v1/cameras/config {payload}
    API->>File: Write atomically to camera_config.json
    API->>API: Synchronize camera records in PostgreSQL
    API-->>UI: 200 OK {"success": true, "message": "Hot-reloaded"}

    Note over Edge: local_detector.py is actively processing video frames
    Edge->>File: Check os.path.getmtime(camera_config.json)
    Edge->>Edge: mtime > last_mtime: Trigger check_and_reload_config()
    Edge->>Edge: Hot-reload entry lines, exclusion zones & department boxes
    Note over Edge: New parameters applied instantly with ZERO frame drops!
```

### 8.1 Stream Pre-Flight Connectivity Verification
Before saving a camera stream to production, the manager can verify reachability via `/api/v1/cameras/test-stream`. The backend attempts an OpenCV connection, confirms H.264/H.265 decoding, and returns the native resolution ($W \times H$) and FPS.

### 8.2 Zero-Downtime Hot-Reloading
The edge engine's `check_and_reload_config()` checks the file modification timestamp (`os.path.getmtime`) of `camera_config.json`. When configuration changes occur:
* Updated department boundaries and entry line coordinates are re-parsed into memory.
* Toolbar button layouts and zone geometry are recalculated.
* Active video stream loops continue uninterrupted without needing process termination or restarts.

### 8.3 Profile Archiving on Deletion
When a camera channel is deleted via `DELETE /api/v1/cameras/channel/{channel_or_id}`:
* The camera configuration (including custom polygonal exclusion zones, floor filters, and calibrated line coordinates) is copied into `archived_camera_profiles`.
* The channel is deactivated in the database rather than destroyed, allowing one-click restoration if a camera is reconnected.

### 8.4 Multi-Camera Tracking, Dual-Factor Re-Identification & Global State Handover
Showroom installations span multiple cameras (e.g. Channel 101 at Main Entrance, Channel 201 at Sales & Display Lounge, Channel 301 at Service Desk). The platform combines `CrossCameraTracker`, atomic cross-process state (`.cross_camera_state.json`), and **Dual-Factor Biometric + Body Re-Identification** to guarantee seamless identity handover across independent detector processes:

```mermaid
sequenceDiagram
    autonumber
    participant CamEntrance as Camera 101 (Entrance)
    participant EdgeEnt as local_detector (Cam 101)
    participant DB as PostgreSQL + pgvector
    participant State as CrossCameraTracker (.cross_camera_state.json)
    participant EdgeFloor as local_detector (Cam 201)
    participant CamFloor as Camera 201 (Floor / Lounge)
    participant UI as React Dashboard

    CamEntrance->>EdgeEnt: Customer enters front door
    EdgeEnt->>EdgeEnt: Crosses entry line -> Extracts ArcFace 512-D Face + OSNet 512-D Body Vectors
    EdgeEnt->>DB: POST /api/v1/visitors/detect -> Mints VIS-2026-1043 (Saves Face + Body Vectors)
    EdgeEnt->>State: register_entity(entity_type="visitor", entity_id="VIS-2026-1043", camera="101")
    State-->>State: Assigns global_track_id=1, persists to .cross_camera_state.json
    EdgeEnt->>DB: Enqueue footfall count increment (Entrance count + 1)
    DB-->>UI: Footfall counter updates to 43

    Note over EdgeEnt,EdgeFloor: Visitor walks into Section / Display Area (Leaves Cam 101 view)
    CamFloor->>EdgeFloor: Frame: Same customer observed browsing display case
    EdgeFloor->>EdgeFloor: Detects Person & extracts ArcFace Face + OSNet Body Vectors
    EdgeFloor->>DB: Dual-Vector similarity search against active day visitors
    Note over EdgeFloor,DB: Biometric Face Match (>= 0.65) OR Dual-Factor Body Re-ID (>= 0.65 + clothing)
    DB-->>EdgeFloor: Match confirmed: VIS-2026-1043 (Unified Identity Preserved)
    EdgeFloor->>State: update_seen("visitor", "VIS-2026-1043", camera="201")
    State-->>EdgeFloor: Returns global_track_id=1
    EdgeFloor->>EdgeFloor: HUD displays: Person #Global-1 | VIS-2026-1043 | Lounge
    EdgeFloor->>DB: POST /api/v1/visitors/update-zone (Zone dwell accumulated)
    Note over EdgeFloor,DB: Footfall count is strictly PRESERVED (NO double counting!)
```

1. **Dual-Factor Ingress & Interior Re-Identification** (`VisitorService.process_detection`, in order):
   - **Edge-supplied `person_id`**: reused only if that visitor has not departed and the new crop passes `verify_appearance` against the stored entry photo. The edge links at face cosine $0.55$, so an unchecked false link would overwrite another visitor's entry photo.
   - **Same tracker**: an exact `Tracker #N` token on a not-departed visitor seen within the last 30 s.
   - **Staff check**: face against employee embeddings at $\ge \max(0.55, \text{FACE\_SIMILARITY\_THRESHOLD}) = 0.65$ → `EMPLOYEE_DETECTED` plus a `CHECK_IN` punch.
   - **Ongoing visit today** (face): cosine $\ge 0.55$ on interior cameras, $\ge 0.60$ on entrance cameras (entrance: visitor created within the last 4 h). An ongoing visit unseen for more than 2 h is closed, and the person treated as returning.
   - **History** (face): cosine $\ge 0.65$ (`VISITOR_REID_THRESHOLD`). A same-day candidate must also pass `verify_appearance` when both crops are full-body. A same-day departed visit, or an open one unseen for more than 90 s (entrance) or 30 min (interior), is closed at its last-seen time. At an entrance camera the **same record is then reopened** (same ID, new segment appended to `visit_log`, total duration = sum of segments). Interior cameras still continue an open visit, and a previous-calendar-day match still mints a new "Returning" ID.
   - **OSNet Body Appearance Vector**: Vector searches via `find_best_visitor_body_match()` in pgvector compare 512-D normalized body embeddings.
   - **Active 15-Minute Re-ID Window**: If a customer's face is looking down at jewelry counters or tilted away from high-mounted CCTV cameras, not-yet-departed visitors who entered (`created_at`) within the past 15 minutes are evaluated via `BodyEmbedder.verify_appearance`. If it passes and ($\text{ReID} \ge 0.65$ and $\text{clothing} \ge 0.28$) or $\text{ReID} \ge 0.74$, the track is automatically linked without waiting for a frontal face.
2. **Persistent Cross-Camera State (`.cross_camera_state.json`)**:
   Tracks global identities across processes with atomic lock-free JSON serialization, storing `entity_type` (`staff` or `visitor`), `entity_id`, `global_track_id`, `first_seen_ts`, `last_seen_ts`, `first_camera`, and `last_camera`.
3. **Synchronized Identity Linking Callback (`on_visitor_registered_callback`)**:
   When the FastAPI backend mints or matches a persistent `person_id` (e.g. `VIS-2026-1043`), `local_detector.py` dispatches the callback to immediately associate the tracker ID with the zone manager, visitor zone tracker, and snapshot buffer in memory.
4. **GPU Memory Isolation (`set_per_process_memory_fraction = 0.42`)**:
   When launching multi-camera pipelines via `run_multi_camera.bat`, `run_all_cameras.bat`, or `run_multi_camera.ps1`, each detector process is allocated a capped memory fraction ($42\%$), allowing simultaneous execution of Cam 101, Cam 201, Cam 301 on consumer GPUs without CUDA out-of-memory crashes.
5. **No Redundant Ingress Counters**:
   In-store floor cameras (`is_entrance: false`) suppress footfall creation and solely contribute to visitor dwell tracking, staff interaction analysis, and waiting status monitoring.

---

## 9. Biometric Privacy & 7-Day Rolling Retention

To comply with global data privacy regulations (GDPR and national biometric privacy principles):

```mermaid
flowchart TD
    Start["Daily Retention Scheduler (Runs every 24h)"] --> Query["Query person_identities where last_seen_at < NOW() - 7 Days\nAND (face OR body vector OR entry/exit photo still present)"]
    Check{"Found Expired Visitor Identities?"}
    Query --> Check
    Check -- No --> ScanOrphaned["Scan & Purge Orphaned Physical Snapshots on Disk"]
    ScanOrphaned --> Finish["Log: Retention cleanup completed"]
    Check -- Yes --> PurgeDB["Set face_embedding = NULL\nSet body_embedding = NULL\nSet avatar_url / exit_avatar_url = NULL\nSet is_expired = TRUE\nSet expired_at = NOW()"]
    PurgeDB --> PurgeVectors["Purge expired vectors from pgvector / HNSW index"]
    PurgeVectors --> DeleteFiles["Delete Physical Snapshot Files from static/uploads/visitors/"]
    DeleteFiles --> PurgeOrphaned["Purge unlinked orphaned snapshot images older than 7 days"]
    PurgeOrphaned --> KeepStats["PRESERVE visitor_tracks, footfall_events & visit purpose\n(Aggregate analytics & counts remain intact!)"]
    KeepStats --> ExemptStaff["EXEMPT: Registered employee embeddings remain permanently active"]
    ExemptStaff --> Report["Generate RetentionCleanupReport & Log Summary"]
```

* **Visitor Biometric Isolation**: Face and body embeddings extracted from unidentified visitors are assigned an automatic 7-day rolling expiration window (`FACE_RETENTION_DAYS`, default `7`). Body-only visitors (face never captured) expire too.
* **Dual Database & Physical Disk Cleanup**:
  - Sets `face_embedding = NULL`, `body_embedding = NULL`, `avatar_url = NULL`, `exit_avatar_url = NULL`, and `is_expired = TRUE` in PostgreSQL `person_identities`.
  - Physically unlinks and deletes corresponding JPEG portrait crops from `static/uploads/visitors/`.
  - Executes `_purge_orphaned_snapshots(directory, retention_days)` to clean up any orphaned or unlinked image files on disk exceeding the retention threshold.
* **Permanent Employee Exemption**: Registered staff embeddings stored in the `employees` table and their gallery photos in `static/uploads/employees/` are permanent and strictly exempt from the purge.
* **Analytics Preservation**: Historical footfall counts, peak-hour distributions, visit purpose tags, and dwell durations remain fully intact when biometric embeddings and photos are purged.

---

## 10. Relational Database Schema (PostgreSQL + pgvector)

```mermaid
erDiagram
    cameras ||--o{ processing_runs : "records"
    cameras ||--o{ visitor_tracks : "captures"
    cameras ||--o{ footfall_events : "generates"
    cameras ||--o{ departments : "may assign"
    cameras ||--o{ employees : "may assign workstation"
    cameras ||--o{ employee_attendance_logs : "captures"

    departments ||--o{ cameras : "contains"
    
    employees ||--o{ employee_attendance_logs : "punches"
    employees ||--o{ employee_daily_attendance : "aggregates"
    
    visitor_tracks ||--o| person_identities : "links (N:1)"
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

    departments {
        varchar id PK
        varchar name
        varchar color
        uuid camera_id FK
        json box
        boolean is_active
        timestamp created_at
        timestamp updated_at
    }

    employees {
        uuid id PK
        varchar employee_id UK
        varchar name
        varchar department
        varchar designation
        uuid assigned_camera_id FK
        time shift_start_time
        time shift_end_time
        integer late_grace_minutes
        vector face_embedding "Vector(512)"
        json face_gallery "Multi-angle embeddings"
        varchar photo_url
        boolean is_active
        timestamp created_at
    }

    employee_attendance_logs {
        uuid id PK
        uuid employee_id FK
        uuid camera_id FK
        varchar event_type "CHECK_IN / CHECK_OUT / PRESENCE"
        float confidence
        timestamp timestamp
        timestamp created_at
    }

    employee_daily_attendance {
        uuid id PK
        uuid employee_id FK
        date date "UK (employee_id, date)"
        timestamp first_punch
        timestamp last_punch
        float total_hours
        boolean is_late
        boolean is_early_departure
        varchar status "PRESENT / HALF_DAY / ABSENT"
        timestamp updated_at
    }

    person_identities {
        uuid id PK
        varchar person_id UK
        vector face_embedding "Vector(512) (ArcFace, Nulled on expiry)"
        vector body_embedding "Vector(512) (OSNet ReID, Nulled on expiry)"
        varchar avatar_url "Retina 320x320 portrait snapshot"
        varchar exit_avatar_url "Retina 320x320 departure snapshot"
        varchar visit_purpose "Vehicle Enquiry / Booking / Test Ride / Service"
        varchar assigned_salesperson "Staff name"
        varchar notes "CRM notes, appearance badge & Exit Match %"
        varchar visitor_type "New / Returning"
        varchar tagging_status "Tagged / Untagged"
        varchar gender_estimated
        varchar age_group_estimated
        varchar zone_entered "Primary showroom department"
        varchar camera_source "Entrance / Floor camera"
        varchar exit_time "Departure timestamp (e.g. 04:45 PM)"
        varchar zone_dwell "JSON-encoded duration breakdown"
        integer waiting_seconds "Accumulated unattended dwell (seconds)"
        boolean is_waiting "Active waiting flag (no staff within 2.2m & not in dept)"
        boolean waiting_breach "SLA breach flag (waiting > 15m / 900s)"
        boolean is_expired
        timestamp last_seen_at
        timestamp expired_at
        timestamp created_at
    }

    visitor_tracks {
        uuid id PK
        uuid camera_id FK
        uuid processing_run_id FK
        uuid person_identity_id FK
        varchar tracker_id
        timestamp first_seen_at
        timestamp last_seen_at
    }

    footfall_events {
        uuid id PK
        uuid camera_id FK
        uuid visitor_track_id FK "UNIQUE"
        varchar event_type "CHECK (= 'entry')"
        timestamp occurred_at
    }

    detection_events {
        uuid id PK
        uuid camera_id FK
        uuid visitor_track_id FK
        integer frame_number
        float confidence
        float bbox_x1
        float bbox_y1
        float bbox_x2
        float bbox_y2
        timestamp detected_at
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

    admins {
        serial id PK
        varchar name
        varchar email UK
        varchar password_hash
        varchar role
        timestamp created_at
    }
```

---

## 11. Summary: End-to-End Ingress & Customer Journey Lifecycle

```
[ Visitor / Staff steps through Showroom Entrance ]
                        │
                        ▼
            [ CCTV RTSP TCP Stream ]
                        │
                        ▼
      [ ThreadedRTSPReader (Zero Lag Frame Grab) ]
                        │
                        ▼
        [ Roboflow RF-DETR / YOLO Person Detection ]
                        │
                        ▼
  [ Multi-Cue Tracking & Spatial-Temporal Handoff Tracker ]
  ├─► ByteTrack / BoT-SORT Coordinate State Filtering
  └─► 25s Torso HSV Color Correlation (Unbroken Timers across Turnarounds/Occlusions)
                        │
                        ▼
        [ SCRFD / YuNet Face Detection + Quality Gating ]
                        │
                        ▼
   [ ArcFace 512-D Face + OSNet 512-D Body Embedding Extraction ]
                        │
                        ▼
   [ Cosine Similarity Matching vs Registered Staff Gallery ]
        │                                       │
        │ (Match >= 0.65 + Voting Lock)         │ (Unmatched / Showroom Customer)
        ▼                                       ▼
[ Identified as Employee ]             [ Tracked as Showroom Visitor ]
        │                                       │
        ├─► Exclude from footfall counts        ├─► Unconfirmed Passerby Filter: Neutral Slate Box (180,180,180)
        ├─► Log Attendance Punch (CHECK_IN)     │   (Ghost Visitor Elimination: No DB writes, no footfall count)
        └─► Upsert Daily Attendance Roll        │
                                                ├─► Ingress: Directional Entry-Line Crossing Verification
                                                │   └─► Upgrades track to Confirmed Visitor (is_confirmed_visitor=True)
                                                ├─► Multi-Frame Best-Shot Quality Scoring (Laplacian + Pose)
                                                ├─► Expanded Bust Framing (Hair + Collar + Upper Chest)
                                                ├─► CLAHE Backlight Compensation (Shadow lifting)
                                                ├─► Pose-Located Attire & Soft Lab Histograms (YOLO Pose + CIE Lab)
                                                ├─► High-DPI Snapshot Dispatch (@ 320x320 Q95) to /visitors/detect
                                                ├─► Progressive Upgrade & Occlusion Recovery (3.5s window, ar >= 0.32)
                                                ├─► Crash-Resilient Sync: DashboardSyncClient FIFO Queue (Retry on reload/drop)
                                                ├─► Sequential Collision-Safe Visitor ID Minting (_mint_person_id)
                                                │
                                                ├─► Store Interior: Real-time department dwell accumulation
                                                │   └─► Periodic HTTP POST to /visitors/update-zone (every 4s)
                                                ├─► Proximity & Waiting Engine: Ground-plane staff distance (2.2m)
                                                │   ├─► Unattended & outside dept: Accumulate waiting timer
                                                │   ├─► Monotonic Bound: waiting_sec <= elapsed physical presence
                                                │   ├─► HUD Alerts: Green (<5m) ➔ Amber (5-15m) ➔ Red (>15m)
                                                │   ├─► SLA Breach: 🚨 UNATTENDED > 15m (Cutoff configurable in UI)
                                                │   └─► SLA Breach Snapshot: Full-resolution clean frame saved to disk
                                                ├─► Dual-Factor Cross-Camera Re-ID (ArcFace Face + OSNet Body)
                                                │   ├─► In-Store cameras suppress redundant walk-in counts (IGNORED_INTERIOR)
                                                │   └─► 15m Active Window: Merges tracker flips without frontal face
                                                ├─► Receptionist UI: 1-click purpose & staff assignment
                                                │   ├─► HTTP PATCH to /visitors/{person_id}/tag
                                                │   └─► Dual-Photo Lightbox: Entry vs Exit photo toggle & match % badge
                                                └─► Egress: Reverse line crossing detection (Exit Door, dwell >= 7s)
                                                    ├─► Turnaround Track Departure Forwarding (< 7s fresh tracks)
                                                    ├─► Grab high-DPI departure portrait crop (exit_avatar_url)
                                                    ├─► Reliable Face Biometrics (eye_d >= 16px, w >= 42px) & Appearance Check
                                                    ├─► Evidence Gate for Edge IDs & Re-ID Override (margin >= 0.08)
                                                    ├─► Weak Re-ID Band (0.60 <= ReID < 0.70) & Sleeve Consistency Check
                                                    ├─► Exact Tracker Token Lookup & Edge Exclusion Propagation
                                                    ├─► Auto-close Stale Prior-Day Unclosed Visits ("Timed out")
                                                    ├─► Resolve Entry-Exit Match Confidence (exitMatchConfidence)
                                                    └─► HTTP POST to /visitors/update-zone (exit_time in IST)
```
