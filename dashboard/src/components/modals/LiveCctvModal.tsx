import React, { useState } from 'react';
import { X, Video, ShieldCheck, Sparkles, RefreshCw, Layers, CheckCircle } from 'lucide-react';
import { useShowroom } from '../../context/ShowroomContext';

interface LiveCctvModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const LiveCctvModal: React.FC<LiveCctvModalProps> = ({ isOpen, onClose }) => {
  const { showroomSettings, metrics, addNewVisitor } = useShowroom();
  const [activeCamId, setActiveCamId] = useState<string>('all');
  const [simulatedTriggerStatus, setSimulatedTriggerStatus] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleManualTriggerWalkin = () => {
    addNewVisitor({
      visitorType: Math.random() > 0.5 ? 'New' : 'Returning',
      zoneEntered: 'Display Area',
      notes: 'Live pedestrian bounding box crossed NVR-01 Virtual Detection Line',
    });
    setSimulatedTriggerStatus('Simulated AI Line-Cross Triggered! Visitor added to dashboard.');
    setTimeout(() => setSimulatedTriggerStatus(null), 3000);
  };

  return (
    <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fadeIn">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-5xl w-full p-6 text-white shadow-2xl relative max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-red-500/20 text-red-400 border border-red-500/30 flex items-center justify-center">
              <Video className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-bold text-white text-lg font-['Outfit']">
                  Showroom CCTV & AI Video Intelligence
                </h3>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  SIMULATION PROTOTYPE
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Connected to {showroomSettings.showroomName} • NVR-H.265 Edge Processor
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleManualTriggerWalkin}
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-semibold flex items-center gap-1.5 transition"
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-300" />
              <span>Simulate Line-Cross</span>
            </button>
            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Integration Notice & Privacy compliance bar */}
        <div className="my-3 p-3 bg-blue-950/40 border border-blue-800/60 rounded-xl flex items-center justify-between text-xs text-blue-200">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>
              <strong>Privacy Compliant:</strong> AI models perform edge bounding-box analysis without permanent facial storage.
              Footfall counters adhere to showroom visitor data compliance.
            </span>
          </div>
          <span className="text-[11px] font-mono text-emerald-400 font-bold shrink-0 ml-2">FPS: 25.0 Synced</span>
        </div>

        {simulatedTriggerStatus && (
          <div className="mb-3 p-2.5 bg-emerald-950/80 border border-emerald-600 rounded-xl text-xs text-emerald-300 flex items-center gap-2 animate-fadeIn">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            <span>{simulatedTriggerStatus}</span>
          </div>
        )}

        {/* 4 Camera Stream Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1 overflow-y-auto py-2">
          {showroomSettings.cameraStreams.map((cam, idx) => {
            const isEntryCam = idx === 0;
            return (
              <div
                key={cam.id}
                className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden relative group hover:border-blue-500/50 transition flex flex-col"
              >
                {/* Camera Video Simulation Canvas */}
                <div className="relative aspect-video bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 flex items-center justify-center overflow-hidden">
                  {/* Subtle Grid Lines to represent CCTV security monitor */}
                  <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b10_1px,transparent_1px),linear-gradient(to_bottom,#1e293b10_1px,transparent_1px)] bg-[size:24px_24px]" />

                  {/* Camera Details Overlay */}
                  <div className="absolute top-2 left-2 z-20 flex items-center gap-1.5 bg-slate-900/80 backdrop-blur-xs px-2 py-1 rounded text-[10px] font-mono text-slate-300 border border-slate-700/60">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
                    <span className="font-bold text-white">{cam.name}</span>
                  </div>

                  <div className="absolute top-2 right-2 z-20 bg-slate-900/80 backdrop-blur-xs px-2 py-1 rounded text-[10px] font-mono text-emerald-400 border border-slate-700/60">
                    25.0 FPS • 1080p
                  </div>

                  {/* AI Detection Visual Overlays (Bounding Boxes) */}
                  {isEntryCam ? (
                    <div className="relative w-full h-full p-6 flex flex-col justify-between">
                      {/* Detection Virtual Tripwire Line */}
                      <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-emerald-500/60 border-t border-dashed border-emerald-400 flex items-center justify-end px-3">
                        <span className="text-[9px] bg-emerald-900/80 text-emerald-300 px-1 rounded font-mono">
                          VIRTUAL ENTRY LINE #1
                        </span>
                      </div>

                      {/* Simulated Bounding Box for Visitor */}
                      <div className="absolute top-[28%] left-[35%] w-24 h-44 border-2 border-amber-400 rounded-xs bg-amber-400/10 flex flex-col justify-between p-1 animate-soft-pulse">
                        <div className="bg-amber-500 text-slate-950 text-[9px] font-mono font-bold px-1 rounded-xs inline-block">
                          Person 96% [New]
                        </div>
                        <div className="text-[8px] font-mono text-amber-200 bg-slate-950/80 px-1 rounded">
                          Track: #1057
                        </div>
                      </div>

                      {/* Simulated Bounding Box for Staff */}
                      <div className="absolute top-[20%] right-[22%] w-20 h-40 border-2 border-blue-400 rounded-xs bg-blue-400/10 flex flex-col justify-between p-1">
                        <div className="bg-blue-500 text-slate-950 text-[9px] font-mono font-bold px-1 rounded-xs inline-block">
                          Staff [Rajesh K.]
                        </div>
                        <div className="text-[8px] font-mono text-blue-200 bg-slate-950/80 px-1 rounded">
                          Badge: Verified
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="relative w-full h-full p-6 flex items-center justify-center">
                      <div className="text-center">
                        <Layers className="w-8 h-8 text-slate-600 mx-auto mb-1" />
                        <p className="text-xs font-semibold text-slate-400">{cam.zone}</p>
                        <p className="text-[10px] text-slate-500 font-mono mt-0.5">Edge AI Tracker Active</p>
                      </div>
                    </div>
                  )}

                  {/* Bottom Camera Metadata */}
                  <div className="absolute bottom-2 left-2 z-20 text-[10px] font-mono text-slate-400 bg-slate-900/80 px-2 py-0.5 rounded">
                    RTSP: {cam.rtspUrl}
                  </div>
                  <div className="absolute bottom-2 right-2 z-20 text-[10px] font-mono text-slate-400 bg-slate-900/80 px-2 py-0.5 rounded">
                    ZONE: {cam.zone}
                  </div>
                </div>

                {/* Sub-bar */}
                <div className="p-2.5 bg-slate-900 flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-medium">Status: <span className="text-emerald-400 font-semibold">Online & Tracking</span></span>
                  <span className="text-slate-400">Detections Today: <span className="text-white font-bold">{isEntryCam ? metrics.totalVisitors : Math.floor(metrics.totalVisitors * 0.7)}</span></span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer info */}
        <div className="pt-4 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-400">
          <p>
            Connected AI Models: <span className="text-slate-200 font-semibold">YOLOv8 Edge Footfall + DeepSORT Tracker</span>
          </p>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl font-semibold transition"
            >
              Close Feed
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
