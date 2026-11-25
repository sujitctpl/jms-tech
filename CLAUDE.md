# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Windows project; the Python venv is `.venv\Scripts\python.exe`.

```bash
# Tests (pytest, in-memory SQLite via tests/conftest.py)
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m pytest tests/test_entry_exit_fixes.py -q
.venv\Scripts\python.exe -m pytest tests/test_waiting_timer.py::test_waiting_breach_detection -q

# Run backend + auth server + dashboard together (ports 8000 / 5000 / 5173)
python scripts/run_all.py
# or individually
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
cd dashboard && npm run server   # Express auth (tsx server/index.ts)
cd dashboard && npm run dev      # Vite dashboard

# Edge detector (one process per camera channel from camera_config.json)
python local_detector.py --channel 101            # add --headless, --v2 (RF-DETR), --tracker botsort
python scripts/run_multi_camera.py 101 201

# Dashboard lint / build
cd dashboard && npm run lint     # oxlint
cd dashboard && npm run build    # tsc -b && vite build

# DB migrations (PostgreSQL + pgvector only; the chain fails on SQLite)
alembic upgrade head
```

There is no Python linter/formatter configured.

## Architecture

Three tiers that communicate only over HTTP:

1. **Edge: `local_detector.py`** (~5k lines, standalone, one process per camera). It reads RTSP, detects and tracks people, does face/body recognition, applies the entry-line gating, and pushes events to the backend through a background queue (`BackgroundSyncClient`, top of file). Key in-file classes: `VisitorZoneTracker` (per-track sessions, zone dwell, waiting timer, `record_visitor_exit`), `ZoneManager` (entry-line crossing in `process()`, departments), `SpatialTemporalHandoffTracker` (25 s re-link across tracker ID switches), `ThreadedRTSPReader`. The main loop is the large function near the bottom; `argparse` setup is at the end. It hot-reloads `camera_config.json` by mtime.
2. **Backend: FastAPI `app/`** follows a routes → services → repositories → SQLAlchemy models layering. `app/api/v1/*` are thin; the real logic lives in `app/services/`. `visitor_service.py` is the core: `process_detection` (mints `VIS-2026-xxxx` IDs, staff filtering, re-ID, `IGNORED_INTERIOR` for non-entrance cameras) and `update_visitor_zone` (zone dwell, waiting, and **exit verification/matching**). Vision code in `app/vision/` (`face_embedder` ArcFace, `body_embedder` OSNet + `verify_appearance`, `vector_store` pgvector, `employee_recognizer`) is shared by the backend and imported by `local_detector.py`. `app/workers/video_worker.py` is an optional server-side alternative to the edge detector. `retention_service.retention_scheduler` starts in the FastAPI lifespan (`app/main.py`).
3. **Dashboard: `dashboard/`.** React 19 + Vite. `src/context/ShowroomContext.tsx` holds app state and calls FastAPI via `src/services/`; `server/index.ts` is a separate Express auth server (admins table, bcrypt) on :5000.

`camera_config.json` is shared state: the dashboard writes it via `POST /api/v1/cameras/config`, and each detector process watches it. `.cross_camera_state.json` shares global visitor/staff identity across detector processes (`app/vision/cross_camera_tracker.py`).

### Visitor lifecycle (spans edge and backend; read both sides before changing)

- A track becomes a visitor only after crossing the entry line in the configured direction and staying ≥ 7 s (`pending_line_crossings` in the main loop). Before confirmation nothing is sent to the backend. `/visitors/detect` is gated on `confirmed_visitor_tracks`, and the backend mints the ID and returns it via the `on_visitor_registered` callback.
- Staff are identified by ArcFace against employee embeddings (threshold `FACE_SIMILARITY_THRESHOLD`, margin, `EMPLOYEE_VOTE_WINDOW` in `app/core/config.py`) and produce attendance punches instead of visitors.
- Exits are reverse line crossings on entrance cameras → `VisitorZoneTracker.record_visitor_exit` → `POST /visitors/update-zone` with `exit_time` and `exit_image_base64`. The backend resolves the visitor by `person_id`, then by the exact `Tracker #N` token in `notes`, then by a fallback search over today's not-yet-departed visitors (reliable face first, then OSNet + clothing via `BodyEmbedder.verify_appearance`). The edge can send `exclude_person_id` to rule out a visitor. Match scores are persisted in `notes` as `Exit Match: XX%`, which `exitMatchConfidence` reads.
- At the entrance camera, leaving visitors usually have a new tracker ID and show their backs, so exits depend mostly on body matching. Treat changes to the thresholds in `visitor_service.update_visitor_zone` and `body_embedder.verify_appearance` with care, and validate them on real snapshots in `static/uploads/visitors/`.

## Gotchas

- **Schema drift:** `app/main.py` calls `Base.metadata.create_all()` on startup, which creates missing tables but never adds columns. When adding a model column, also add an Alembic migration (see `alembic/versions/0006_*` for the idempotent add-if-missing pattern).
- Tests use SQLite, which returns timezone-naive datetimes while PostgreSQL returns aware ones. Use `to_ist()` in `visitor_service.py` when doing arithmetic with DB timestamps.
- `tests/test_returning_visitor.py` writes snapshot files into the real `static/uploads/visitors/` with IDs like `VIS-2026-1041`, the same numbering as live data.
- Retention defaults to 7 days (`FACE_RETENTION_DAYS`, intentionally changed from 14). Expiry must null the face vector, body vector and both snapshot URLs.
- Visitor IDs come from `VisitorService._mint_person_id()` (count-based with a collision check). Don't build IDs from the row count alone, because `create_or_update` would silently overwrite an existing visitor.
- README.md and flow.md document thresholds in detail but have drifted from the code before. Trust the code, and update the docs when you change a threshold.
