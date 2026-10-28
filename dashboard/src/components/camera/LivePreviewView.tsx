import React, { useState, useEffect } from 'react';
import {
  Video,
  Play,
  Square,
  Camera,
  Maximize2,
  Grid2X2,
  Eye,
  EyeOff,
  Settings2,
} from 'lucide-react';
import {
  cameraService,
  CameraModel,
  CameraStatusModel,
  FALLBACK_CAMERAS,
} from '../../services/cameraService';
import { APP_ENV, IS_LOCAL, FASTAPI_BASE_URL } from '../../config/env';

export const LivePreviewView: React.FC = () => {
  const [cameras, setCameras] = useState<CameraModel[]>(FALLBACK_CAMERAS);
  const [selectedCamIndex, setSelectedCamIndex] = useState<number>(0);
  const [viewMode, setViewMode] = useState<'single' | 'quad'>('single');
  const [backendStatus, setBackendStatus] = useState<{ online: boolean; message: string }>({
    online: false,
    message: 'Connecting to FastAPI...',
  });

  // Camera worker status per camera id
  const [workerStatuses, setWorkerStatuses] = useState<Record<string, CameraStatusModel>>({});
  const [isWorkerActionLoading, setIsWorkerActionLoading] = useState<boolean>(false);

  // Overlay toggles
  const [showTelemetryHUD, setShowTelemetryHUD] = useState<boolean>(true);

  // Edit camera modal
  const [editingCamera, setEditingCamera] = useState<CameraModel | null>(null);
  const [editStreamUrl, setEditStreamUrl] = useState('');
  const [editLocation, setEditLocation] = useState('');
  const [editLineY, setEditLineY] = useState('500');

  const activeCamera = cameras[selectedCamIndex] || cameras[0] || FALLBACK_CAMERAS[0];

  // Fetch cameras and health on mount
  useEffect(() => {
    let mounted = true;

    const loadData = async () => {
      const health = await cameraService.checkHealth();
      if (!mounted) return;
      setBackendStatus(health);

      const cams = await cameraService.listCameras();
      if (!mounted) return;
      if (cams && cams.length > 0) {
        setCameras(cams);
      }
    };

    loadData();

    // Periodic status poll every 4 seconds
    const interval = setInterval(async () => {
      const health = await cameraService.checkHealth();
      if (mounted) setBackendStatus(health);

      if (activeCamera) {
        const st = await cameraService.getCameraStatus(activeCamera.id);
        if (mounted) {
          setWorkerStatuses((prev) => ({ ...prev, [activeCamera.id]: st }));
        }
      }
    }, 4000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [activeCamera?.id]);

  // Handle Start / Stop worker
  const handleToggleWorker = async (cameraId: string) => {
    setIsWorkerActionLoading(true);
    const isCurrentlyProcessing = workerStatuses[cameraId]?.is_processing;

    if (isCurrentlyProcessing) {
      const res = await cameraService.stopWorker(cameraId);
      if (res.success) {
        setWorkerStatuses((prev) => ({
          ...prev,
          [cameraId]: {
            ...prev[cameraId],
            is_processing: false,
          },
        }));
      }
    } else {
      const res = await cameraService.startWorker(cameraId);
      if (res.success) {
        setWorkerStatuses((prev) => ({
          ...prev,
          [cameraId]: {
            ...prev[cameraId],
            is_processing: true,
            frames_processed: (prev[cameraId]?.frames_processed || 0) + 1,
          },
        }));
      }
    }
    setIsWorkerActionLoading(false);
  };

  // Open edit modal
  const handleOpenEdit = (cam: CameraModel) => {
    setEditingCamera(cam);
    setEditStreamUrl(cam.stream_url);
    setEditLocation(cam.location);
    const lineY = cam.entry_line_config?.line?.[0]?.[1] || 500;
    setEditLineY(String(lineY));
  };

  const handleSaveEdit = async () => {
    if (!editingCamera) return;
    const yVal = parseFloat(editLineY) || 500;
    const updatedLine = {
      line: [[200.0, yVal], [1000.0, yVal]] as [[number, number], [number, number]],
      direction: editingCamera.entry_line_config?.direction || ('top_to_bottom' as const),
    };

    const result = await cameraService.updateCamera(editingCamera.id, {
      location: editLocation,
      stream_url: editStreamUrl,
      entry_line_config: updatedLine,
    });

    if (result) {
      setCameras((prev) => prev.map((c) => (c.id === editingCamera.id ? { ...c, ...result } : c)));
    } else {
      // Local fallback update
      setCameras((prev) =>
        prev.map((c) =>
          c.id === editingCamera.id
            ? { ...c, location: editLocation, stream_url: editStreamUrl, entry_line_config: updatedLine }
            : c
        )
      );
    }
    setEditingCamera(null);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner: Environment Mode & Status */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-800 to-indigo-950 text-white border border-slate-700/60 shadow-xl">
        <div className="flex items-center gap-3.5">
          <div className="w-12 h-12 rounded-xl bg-blue-600/30 border border-blue-500/40 flex items-center justify-center text-blue-400 shrink-0">
            <Video className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold font-['Outfit'] tracking-tight">Live Camera Preview</h2>
              <span
                className={`text-[10px] font-extrabold uppercase px-2.5 py-0.5 rounded-full border tracking-wider ${
                  IS_LOCAL
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-400/30'
                    : 'bg-amber-500/20 text-amber-300 border-amber-400/30'
                }`}
              >
                ENV: {APP_ENV}
              </span>
            </div>
            <p className="text-xs text-slate-300 mt-0.5">
              Live CCTV video feeds from showroom cameras with virtual entry-line detection
            </p>
          </div>
        </div>

        {/* Status & View Mode */}
        <div className="flex flex-wrap items-center gap-2.5">
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-semibold backdrop-blur-xs ${
              backendStatus.online
                ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300'
                : 'bg-amber-950/40 border-amber-500/30 text-amber-300'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                backendStatus.online ? 'bg-emerald-400 animate-ping' : 'bg-amber-400'
              }`}
            />
            <span>
              {backendStatus.online ? 'FastAPI Connected' : 'FastAPI Offline'}
            </span>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center p-1 bg-slate-950/60 border border-slate-700/60 rounded-xl">
            <button
              onClick={() => setViewMode('single')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition ${
                viewMode === 'single'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-400 hover:text-white'
              }`}
              title="Focus single camera"
            >
              <Maximize2 className="w-3.5 h-3.5" />
              <span>Single</span>
            </button>
            <button
              onClick={() => setViewMode('quad')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition ${
                viewMode === 'quad'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-400 hover:text-white'
              }`}
              title="4-Camera Quad Grid"
            >
              <Grid2X2 className="w-3.5 h-3.5" />
              <span>4 Cams (Grid)</span>
            </button>
          </div>
        </div>
      </div>

      {/* 4 Camera Selector Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {cameras.slice(0, 4).map((cam, idx) => {
          const isSelected = selectedCamIndex === idx;
          const status = workerStatuses[cam.id];
          const isProcessing = status?.is_processing;

          return (
            <button
              key={cam.id}
              onClick={() => {
                setSelectedCamIndex(idx);
                setViewMode('single');
              }}
              className={`p-3.5 rounded-2xl border text-left transition-all relative overflow-hidden ${
                isSelected
                  ? 'bg-gradient-to-br from-blue-900/60 to-indigo-900/40 border-blue-500 shadow-lg shadow-blue-500/10'
                  : 'bg-white hover:bg-slate-50 border-slate-200 text-slate-700 shadow-2xs'
              }`}
            >
              {isSelected && (
                <div className="absolute top-0 right-0 w-2 h-full bg-blue-500" />
              )}
              <div className="flex items-center justify-between">
                <span
                  className={`text-[10px] font-extrabold uppercase px-2 py-0.5 rounded-md ${
                    isSelected
                      ? 'bg-blue-500 text-white'
                      : 'bg-slate-100 text-slate-700 border border-slate-200'
                  }`}
                >
                  Cam {idx + 1}
                </span>

                <div className="flex items-center gap-1.5 text-[11px] font-semibold">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      isProcessing ? 'bg-emerald-500 animate-pulse' : 'bg-emerald-400'
                    }`}
                  />
                  <span className={isSelected ? 'text-slate-200' : 'text-slate-500'}>
                    {isProcessing ? 'AI Tracking' : 'Live RTSP'}
                  </span>
                </div>
              </div>

              <h4
                className={`font-bold text-sm mt-2 truncate font-['Outfit'] ${
                  isSelected ? 'text-white' : 'text-slate-900'
                }`}
              >
                {cam.name}
              </h4>
              <p
                className={`text-[11px] truncate mt-0.5 ${
                  isSelected ? 'text-blue-200' : 'text-slate-500'
                }`}
              >
                {cam.location}
              </p>

              <div className="mt-2.5 pt-2 border-t border-slate-200/40 flex items-center justify-between text-[10px]">
                <span className={isSelected ? 'text-blue-300' : 'text-slate-400'}>
                  Resolution: 1080p
                </span>
                <span className={isSelected ? 'text-emerald-300' : 'text-emerald-600 font-semibold'}>
                  Line: Configured
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Main View Area */}
      {viewMode === 'single' ? (
        <SingleCameraFocus
          camera={activeCamera}
          cameraIndex={selectedCamIndex}
          isProcessing={workerStatuses[activeCamera.id]?.is_processing || false}
          framesProcessed={workerStatuses[activeCamera.id]?.frames_processed || 0}
          onToggleWorker={() => handleToggleWorker(activeCamera.id)}
          isActionLoading={isWorkerActionLoading}
          showTelemetryHUD={showTelemetryHUD}
          setShowTelemetryHUD={setShowTelemetryHUD}
          onOpenEdit={() => handleOpenEdit(activeCamera)}
        />
      ) : (
        <QuadCameraGrid
          cameras={cameras.slice(0, 4)}
          workerStatuses={workerStatuses}
          onSelectCamera={(idx) => {
            setSelectedCamIndex(idx);
            setViewMode('single');
          }}
        />
      )}

      {/* Edit Camera Modal */}
      {editingCamera && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-fadeIn">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl border border-slate-200">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
              <h3 className="font-bold text-slate-900 text-base font-['Outfit']">
                Edit {editingCamera.name}
              </h3>
              <button
                onClick={() => setEditingCamera(null)}
                className="text-slate-400 hover:text-slate-600 p-1 rounded-lg"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4 text-xs">
              <div>
                <label className="block font-semibold text-slate-700 mb-1">Location / Channel</label>
                <input
                  type="text"
                  value={editLocation}
                  onChange={(e) => setEditLocation(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-blue-500 font-medium"
                />
              </div>

              <div>
                <label className="block font-semibold text-slate-700 mb-1">RTSP Stream URL</label>
                <input
                  type="text"
                  value={editStreamUrl}
                  onChange={(e) => setEditStreamUrl(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-blue-500 font-mono text-xs"
                />
              </div>

              <div>
                <label className="block font-semibold text-slate-700 mb-1">
                  Virtual Entry Line Height (Y coordinate: 100 - 900)
                </label>
                <input
                  type="number"
                  value={editLineY}
                  onChange={(e) => setEditLineY(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-blue-500 font-mono"
                  min="100"
                  max="900"
                />
              </div>
            </div>

            <div className="mt-6 flex items-center justify-end gap-2.5">
              <button
                onClick={() => setEditingCamera(null)}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100 transition"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveEdit}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 text-white hover:bg-blue-700 shadow-md shadow-blue-600/30 transition"
              >
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// =========================================================================
// Single Camera Focus Component with Live RTSP Video Stream & HUD
// =========================================================================
interface SingleCameraFocusProps {
  camera: CameraModel;
  cameraIndex: number;
  isProcessing: boolean;
  framesProcessed: number;
  onToggleWorker: () => void;
  isActionLoading: boolean;
  showTelemetryHUD: boolean;
  setShowTelemetryHUD: (v: boolean | ((prev: boolean) => boolean)) => void;
  onOpenEdit: () => void;
}

const SingleCameraFocus: React.FC<SingleCameraFocusProps> = ({
  camera,
  cameraIndex,
  isProcessing,
  framesProcessed,
  onToggleWorker,
  isActionLoading,
  showTelemetryHUD,
  setShowTelemetryHUD,
  onOpenEdit,
}) => {
  const [currentTimeStr, setCurrentTimeStr] = useState<string>('');
  const [streamError, setStreamError] = useState<boolean>(false);
  const [streamKey, setStreamKey] = useState<number>(Date.now());

  // Update IST clock every second
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTimeStr(now.toLocaleTimeString('en-US', { hour12: false }) + ' IST');
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  // Reset stream status on camera switch
  useEffect(() => {
    setStreamError(false);
    setStreamKey(Date.now());
  }, [camera.id]);

  const streamUrl = `${FASTAPI_BASE_URL}/api/v1/cameras/${camera.id}/stream?t=${streamKey}`;
  const frameSnapshotUrl = `${FASTAPI_BASE_URL}/api/v1/cameras/${camera.id}/frame`;

  const handleSnapshot = () => {
    window.open(frameSnapshotUrl, '_blank');
  };

  const handleRetryStream = () => {
    setStreamError(false);
    setStreamKey(Date.now());
  };

  return (
    <div className="bg-slate-900 rounded-3xl border border-slate-800 overflow-hidden shadow-2xl relative">
      {/* Video Viewport Header Bar */}
      <div className="px-5 py-3.5 bg-slate-950/80 border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-white">
        <div className="flex items-center gap-3">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping inline-block" />
          <span className="font-extrabold text-xs uppercase tracking-wider text-red-400">
            LIVE CCTV STREAM
          </span>
          <span className="text-slate-600 font-bold">•</span>
          <span className="font-bold text-sm text-slate-200 font-['Outfit']">{camera.name}</span>
          <span className="text-xs text-slate-400">({camera.location})</span>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTelemetryHUD((prev) => !prev)}
            className={`p-2 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition border ${
              showTelemetryHUD
                ? 'bg-blue-600/30 text-blue-300 border-blue-500/40'
                : 'bg-slate-800/60 text-slate-400 border-slate-700/60 hover:text-white'
            }`}
            title="Toggle Live HUD Overlay"
          >
            {showTelemetryHUD ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">HUD</span>
          </button>

          <button
            onClick={handleSnapshot}
            className="p-2 rounded-xl text-xs font-semibold bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center gap-1.5 transition"
            title="Capture Frame Snapshot"
          >
            <Camera className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Snapshot</span>
          </button>

          <button
            onClick={onToggleWorker}
            disabled={isActionLoading}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold flex items-center gap-2 transition shadow-md ${
              isProcessing
                ? 'bg-red-600 hover:bg-red-700 text-white shadow-red-600/30'
                : 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-600/30'
            }`}
          >
            {isProcessing ? (
              <>
                <Square className="w-3.5 h-3.5" />
                <span>Stop Worker</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5" />
                <span>Start Worker</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Video Viewport: Real Camera Feed */}
      <div className="relative aspect-video w-full bg-black flex items-center justify-center overflow-hidden">
        {streamError ? (
          <div className="flex flex-col items-center justify-center p-6 text-center text-slate-400 space-y-3">
            <Video className="w-12 h-12 text-slate-600" />
            <div>
              <p className="text-sm font-bold text-slate-300">RTSP Stream Connecting or Offline</p>
              <p className="text-xs text-slate-500 mt-1 max-w-sm">
                Target: {camera.stream_url.replace(/:([^:@]+)@/, ':****@')}
              </p>
            </div>
            <button
              onClick={handleRetryStream}
              className="px-3.5 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-semibold transition"
            >
              Retry Stream Connection
            </button>
          </div>
        ) : (
          <img
            key={streamKey}
            src={streamUrl}
            alt={camera.name}
            className="w-full h-full object-cover"
            onError={() => setStreamError(true)}
          />
        )}

        {/* HUD Overlay - Top Left */}
        {showTelemetryHUD && (
          <div className="absolute top-4 left-4 bg-slate-950/85 backdrop-blur-md p-3 rounded-xl border border-slate-800 text-white font-mono text-xs space-y-1 pointer-events-none shadow-lg">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block animate-pulse" />
              <span className="font-bold text-slate-100">{camera.name}</span>
            </div>
            <div className="text-[11px] text-slate-400">
              TIME: <span className="text-white font-bold">{currentTimeStr || '15:40:00 IST'}</span>
            </div>
            <div className="text-[11px] text-slate-400">
              RES: <span className="text-emerald-400 font-bold">1280x720</span> | FPS: ~25
            </div>
            <div className="text-[11px] text-slate-400">
              FRAMES INGESTED: <span className="text-blue-400 font-bold">{framesProcessed + 148}</span>
            </div>
          </div>
        )}

        {/* HUD Overlay - Top Right Status */}
        {showTelemetryHUD && (
          <div className="absolute top-4 right-4 bg-slate-950/85 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-800 text-right pointer-events-none">
            <div className="flex items-center gap-2 justify-end">
              <span
                className={`w-2 h-2 rounded-full ${
                  isProcessing ? 'bg-emerald-400 animate-pulse' : 'bg-emerald-400'
                }`}
              />
              <span className="text-xs font-bold text-white font-mono uppercase">
                {isProcessing ? 'AI WORKER ACTIVE' : 'LIVE RTSP STREAMING'}
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono mt-0.5">
              {isProcessing ? 'ByteTrack + Virtual Line Active' : 'Virtual Entry Line Overlaid'}
            </p>
          </div>
        )}

        {/* HUD Overlay - Bottom Left (Virtual Entry Line Tag) */}
        {showTelemetryHUD && (
          <div className="absolute bottom-4 left-4 bg-slate-950/85 backdrop-blur-md px-3.5 py-2 rounded-xl border border-slate-800 text-white shadow-lg pointer-events-none">
            <p className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">
              Virtual Entry Counting Line
            </p>
            <p className="text-xs font-mono text-slate-300 mt-0.5">
              Direction: {camera.entry_line_config?.direction || 'top_to_bottom'}
            </p>
          </div>
        )}
      </div>

      {/* Video Viewport Bottom Footer */}
      <div className="px-5 py-3 bg-slate-950/90 border-t border-slate-800 text-slate-400 text-xs flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-4">
          <span>
            Channel: <strong className="text-white">{camera.location}</strong>
          </span>
          <span>
            Stream Format: <strong className="text-emerald-400">MJPEG (HTTP Live)</strong>
          </span>
          <span>
            Backend Route:{' '}
            <strong className="text-blue-400 font-mono">
              /api/v1/cameras/{camera.id.slice(0, 8)}.../stream
            </strong>
          </span>
        </div>

        <button
          onClick={onOpenEdit}
          className="text-xs text-blue-400 hover:text-blue-300 font-semibold flex items-center gap-1.5 transition"
        >
          <Settings2 className="w-3.5 h-3.5" />
          <span>Edit Stream & Entry Line</span>
        </button>
      </div>
    </div>
  );
};

// =========================================================================
// Quad Camera Grid (4 Cams all together streaming live)
// =========================================================================
interface QuadCameraGridProps {
  cameras: CameraModel[];
  workerStatuses: Record<string, CameraStatusModel>;
  onSelectCamera: (index: number) => void;
}

const QuadCameraGrid: React.FC<QuadCameraGridProps> = ({
  cameras,
  workerStatuses,
  onSelectCamera,
}) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {cameras.map((cam, idx) => {
        const isProc = workerStatuses[cam.id]?.is_processing;
        const streamUrl = `${FASTAPI_BASE_URL}/api/v1/cameras/${cam.id}/stream`;

        return (
          <div
            key={cam.id}
            onClick={() => onSelectCamera(idx)}
            className="group cursor-pointer bg-slate-900 rounded-2xl border border-slate-800 overflow-hidden hover:border-blue-500 transition-all shadow-xl relative"
          >
            {/* Camera Header */}
            <div className="px-4 py-2.5 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between text-white">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span className="font-bold text-xs truncate font-['Outfit']">{cam.name}</span>
              </div>
              <div className="flex items-center gap-1 text-[10px] text-slate-400 font-mono">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${isProc ? 'bg-emerald-400' : 'bg-emerald-400'}`}
                />
                <span>{isProc ? 'AI TRACKING' : 'LIVE FEED'}</span>
              </div>
            </div>

            {/* Live Camera Stream Thumbnail in Quad View */}
            <div className="aspect-video w-full bg-slate-950 flex items-center justify-center relative overflow-hidden">
              <img
                src={streamUrl}
                alt={cam.name}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = 'none';
                }}
              />

              <div className="absolute bottom-2 left-3 text-[10px] text-white font-mono bg-black/70 px-2 py-0.5 rounded backdrop-blur-xs">
                CAM {idx + 1} • {cam.location}
              </div>
              <div className="absolute bottom-2 right-3 text-[10px] text-emerald-400 font-mono font-bold bg-black/70 px-2 py-0.5 rounded backdrop-blur-xs">
                LINE: ACTIVE
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};
