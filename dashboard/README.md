# Showroom Visitor Intelligence — Web Dashboard & Auth Service

The web client for the Showroom Visitor Intelligence platform. Built with **React 19**, **TypeScript**, **Vite**, and **Tailwind CSS v4**, paired with an **Express 5** authentication service.

---

## Features

- **Live CCTV Feeds & AI Overlays**: Real-time camera feeds and detection bounding boxes streamed from the local edge detector.
- **Footfall & Executive Overview**: Total visitors, active visitors, peak occupancy, and hourly trend charts powered by Recharts.
- **Visitor Tagging & Quick Add**: Tag visitors with custom labels (e.g. VIP, returning, inquiry) and log visits directly.
- **Staff Attendance Tracking**: Real-time attendance monitoring and activity logs.
- **Peak Hours & Section Analytics**: Hourly footfall heatmaps and showroom counter/zone dwell duration analytics.
- **Exportable Reports**: Generate summary analytics and CSV/PDF exportable reporting views.
- **Authentication Service**: Dedicated Express + PostgreSQL auth service with bcrypt password hashing and session tokens.

---

## Directory Structure

```text
dashboard/
├── src/
│   ├── components/
│   │   ├── attendance/         # Attendance tracking views
│   │   ├── auth/               # Login & Registration views
│   │   ├── camera/             # Live RTSP/MJPEG preview with canvas overlays
│   │   ├── dashboard/          # Executive Overview & footfall metrics
│   │   ├── layout/             # Sidebar, Navbar & layout components
│   │   ├── modals/             # QuickAddVisitor, TagVisitor, LiveCctv modals
│   │   ├── peak-hours/         # Peak footfall charts & heatmaps
│   │   ├── reports/            # Exportable reporting views
│   │   ├── settings/           # System & camera configuration views
│   │   ├── visit-purpose/      # Purpose breakdown charts
│   │   └── visitors/           # Visitor lists & history tables
│   ├── config/                 # Environment & API URL configurations
│   ├── context/                # ShowroomContext (data) & AuthContext (auth state)
│   ├── data/                   # Seed & fallback dummy data
│   ├── services/               # Camera and analytics API clients
│   └── types/                  # TypeScript interface definitions
├── server/
│   ├── index.ts                # Express Auth Server (:5000)
│   └── db.ts                   # PostgreSQL client pool
├── package.json
└── vite.config.ts
```

---

## Getting Started

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Key environment variables:
```ini
VITE_API_URL=http://localhost:8000
VITE_AUTH_API_URL=http://localhost:5000
PORT=5000
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/showroom_db
```

### 3. Run Development Servers

**Run the Auth Server (Port 5000):**
```bash
npm run server
```

**Run the Vite Web Client (Port 5173):**
```bash
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## Production Build

```bash
# Type-check and build frontend
npm run build

# Preview production build
npm run preview
```
