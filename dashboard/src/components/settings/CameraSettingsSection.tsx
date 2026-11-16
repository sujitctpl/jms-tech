import React, { useState, useEffect } from 'react';
import {
  Video,
  Save,
  Plus,
  Trash2,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Eye,
  EyeOff,
  Crosshair,
  Activity,
  Check,
  Cpu,
} from 'lucide-react';
import {
  cameraService,
  DynamicCameraConfig,
  CameraChannelConfig,
  DEFAULT_CAMERA_CONFIG,
  StreamTestResult,
} from '../../services/cameraService';

const PRESET_STREAMS = [
  {
    label: 'Main Entrance (Ch 101)',
    url: 'rtsp://zakir:06nov2014@192.168.20.184:554/Streaming/Channels/101',
  },
  {
    label: 'Display Floor (Ch 201)',
    url: 'rtsp://zakir:06nov2014@192.168.20.184:554/Streaming/Channels/201',
  },
  {
    label: 'Customer Lounge (Ch 301)',
    url: 'rtsp://zakir:06nov2014@192.168.20.184:554/Streaming/Channels/301',
  },
  {
    label: 'Service Bay (Ch 401)',
    url: 'rtsp://zakir:06nov2014@192.168.20.184:554/Streaming/Channels/401',
  },
  {
    label: 'Local USB / Webcam (0)',
    url: '0',
  },
];

export const CameraSettingsSection: React.FC = () => {
  const [config, setConfig] = useState<DynamicCameraConfig>(DEFAULT_CAMERA_CONFIG);
  const [selectedChannel, setSelectedChannel] = useState<string>('101');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveToast, setSaveToast] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Stream Testing state
  const [isTestingStream, setIsTestingStream] = useState<boolean>(false);
  const [testResult, setTestResult] = useState<StreamTestResult | null>(null);
  const [showSecret, setShowSecret] = useState<boolean>(false);

  // Add Camera Modal
  const [isAddModalOpen, setIsAddModalOpen] = useState<boolean>(false);
  const [newChannelNumber, setNewChannelNumber] = useState<string>('');
  const [newCameraName, setNewCameraName] = useState<string>('');
  const [newCameraSource, setNewCameraSource] = useState<string>('');
  const [newCameraLocation, setNewCameraLocation] = useState<string>('');

  useEffect(() => {
    loadCameraConfig();
  }, []);

  const loadCameraConfig = async () => {
    setIsLoading(true);
    try {
      const data = await cameraService.getCameraConfig();
      if (data && data.cameras) {
        setConfig(data);
        const channels = Object.keys(data.cameras);
        if (channels.length > 0 && !data.cameras[selectedChannel]) {
          setSelectedChannel(data.default_channel || channels[0]);
        }
      }
    } catch (err) {
      console.error('Failed to load camera settings:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const currentCam: CameraChannelConfig =
    config.cameras[selectedChannel] || {
      name: `Showroom Camera ${selectedChannel}`,
      source: '',
      location: `Channel ${selectedChannel}`,
      confidence: config.confidence,
      min_feet_y: 0.08,
      allow_standalone_faces: false,
      is_active: true,
      entry_line: {
        enabled: true,
        p1: [200, 500],
        p2: [1000, 500],
        direction: 'both',
      },
    };

  const handleUpdateCurrentCam = (updates: Partial<CameraChannelConfig>) => {
    setConfig((prev) => ({
      ...prev,
      cameras: {
        ...prev.cameras,
        [selectedChannel]: {
          ...currentCam,
          ...updates,
        },
      },
    }));
  };

  const handleUpdateGlobal = <K extends keyof DynamicCameraConfig>(
    key: K,
    value: DynamicCameraConfig[K]
  ) => {
    setConfig((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleTestCurrentStream = async () => {
    if (!currentCam.source) {
      setTestResult({
        online: false,
        error: 'Please enter a stream URL or camera index before testing.',
      });
      return;
    }
    setIsTestingStream(true);
    setTestResult(null);
    try {
      const res = await cameraService.testStream(currentCam.source);
      setTestResult(res);
    } catch (err: any) {
      setTestResult({
        online: false,
        error: err?.message || 'Failed to ping stream tester.',
      });
    } finally {
      setIsTestingStream(false);
    }
  };

  const handleSaveAll = async () => {
    setIsSaving(true);
    try {
      const res = await cameraService.saveCameraConfig(config);
      if (res.success) {
        setSaveToast({
          type: 'success',
          text: res.message || 'Camera configuration synchronized with Edge AI detector.',
        });
        if (res.config) {
          setConfig(res.config);
        }
      } else {
        setSaveToast({
          type: 'error',
          text: res.message || 'Failed to sync camera configuration.',
        });
      }
    } catch (err: any) {
      setSaveToast({
        type: 'error',
        text: err?.message || 'Error saving camera configuration.',
      });
    } finally {
      setIsSaving(false);
      setTimeout(() => setSaveToast(null), 4000);
    }
  };

  const handleDeleteChannel = async (channelKey: string) => {
    if (Object.keys(config.cameras).length <= 1) {
      alert('You must have at least one camera channel configured.');
      return;
    }
    if (
      !window.confirm(
        `Are you sure you want to remove Camera Channel [${channelKey}]? This will remove it from camera_config.json.`
      )
    ) {
      return;
    }

    const newCams = { ...config.cameras };
    delete newCams[channelKey];
    const remainingChannels = Object.keys(newCams);
    const nextDefault =
      config.default_channel === channelKey ? remainingChannels[0] : config.default_channel;

    const updatedConfig: DynamicCameraConfig = {
      ...config,
      default_channel: nextDefault,
      cameras: newCams,
    };

    setConfig(updatedConfig);
    setSelectedChannel(remainingChannels[0]);
    await cameraService.deleteCameraChannel(channelKey);
    await cameraService.saveCameraConfig(updatedConfig);

    setSaveToast({
      type: 'success',
      text: `Channel [${channelKey}] removed. Calibration profile preserved in archive.`,
    });
    setTimeout(() => setSaveToast(null), 3500);
  };

  const handleAddCameraSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const ch = newChannelNumber.trim().replace(/[^a-zA-Z0-9_]/g, '');
    if (!ch) {
      alert('Please enter a valid channel identifier (e.g. 501 or bay_cam)');
      return;
    }
    if (config.cameras[ch]) {
      alert(`Channel [${ch}] already exists in active cameras.`);
      return;
    }

    // Backend-level automatic preservation: Archive -> Default Template -> Standard
    const archived = config.archived_camera_profiles?.[ch];
    const template = DEFAULT_CAMERA_CONFIG.cameras[ch];

    const newCam: CameraChannelConfig = {
      name: newCameraName.trim() || archived?.name || template?.name || `Showroom Camera ${ch}`,
      camera_id: archived?.camera_id || template?.camera_id,
      source: newCameraSource.trim() || archived?.source || template?.source || '',
      location: newCameraLocation.trim() || archived?.location || template?.location || `Channel ${ch}`,
      confidence: archived?.confidence ?? template?.confidence ?? config.confidence,
      min_feet_y: archived?.min_feet_y ?? template?.min_feet_y ?? 0.08,
      allow_standalone_faces: false,
      is_active: true,
      entry_line: archived?.entry_line || template?.entry_line || {
        enabled: true,
        p1: [200, 500],
        p2: [1000, 500],
        direction: 'both',
      },
      exclusion_zones: archived?.exclusion_zones || template?.exclusion_zones || [],
      sections: archived?.sections || template?.sections || [],
    };

    const updatedConfig: DynamicCameraConfig = {
      ...config,
      cameras: {
        ...config.cameras,
        [ch]: newCam,
      },
    };

    setConfig(updatedConfig);
    setSelectedChannel(ch);
    setIsAddModalOpen(false);
    setNewChannelNumber('');
    setNewCameraName('');
    setNewCameraSource('');
    setNewCameraLocation('');

    await cameraService.saveCameraConfig(updatedConfig);
    setSaveToast({
      type: 'success',
      text: `Channel [${ch}] added and synchronized!`,
    });
    setTimeout(() => setSaveToast(null), 3500);
  };

  const channelsList = Object.keys(config.cameras);

  return (
    <div className="space-y-6">
      {/* Toast Alert */}
      {saveToast && (
        <div
          className={`p-4 rounded-2xl text-xs flex items-center gap-3 border shadow-sm animate-fadeIn ${saveToast.type === 'success'
              ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
              : 'bg-rose-50 border-rose-200 text-rose-900'
            }`}
        >
          {saveToast.type === 'success' ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          ) : (
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          )}
          <span className="font-semibold">{saveToast.text}</span>
        </div>
      )}

      {/* Main Container */}
      <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] space-y-6">
        {/* Top Header Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-slate-100 gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-2xl bg-blue-50 text-blue-600 border border-blue-100 shadow-2xs">
              <Video className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-extrabold text-slate-900 text-lg font-['Outfit'] flex items-center gap-2">
                <span>CCTV Cameras & Edge AI Vision</span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-extrabold bg-blue-100 text-blue-800">
                  Dynamic Feed Engine
                </span>
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Configure dynamic RTSP camera streams, AI confidence thresholds, face recognition sensitivity, and entry counting lines.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={loadCameraConfig}
              title="Refresh from camera_config.json"
              className="p-2.5 text-slate-500 hover:text-slate-800 bg-slate-50 hover:bg-slate-100 rounded-xl border border-slate-200 transition cursor-pointer"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => setIsAddModalOpen(true)}
              className="px-4 py-2.5 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-bold flex items-center gap-2 transition active:scale-[0.98] cursor-pointer"
            >
              <Plus className="w-4 h-4" />
              <span>Add Camera</span>
            </button>
            <button
              onClick={handleSaveAll}
              disabled={isSaving}
              className="px-5 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-xs font-bold shadow-md shadow-blue-500/20 flex items-center gap-2 transition active:scale-[0.98] disabled:opacity-60 cursor-pointer"
            >
              <Save className="w-4 h-4" />
              <span>{isSaving ? 'Syncing...' : 'Save & Sync'}</span>
            </button>
          </div>
        </div>


        {/* Channel Switcher Pills */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-bold uppercase tracking-wider text-slate-600">
              Showroom Camera Channels ({channelsList.length})
            </label>
            <span className="text-[11px] text-slate-400">
              Default Startup Channel: <strong className="text-blue-600">[{config.default_channel}]</strong>
            </span>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {channelsList.map((ch) => {
              const cam = config.cameras[ch];
              const isSelected = selectedChannel === ch;
              const isDefault = config.default_channel === ch;

              return (
                <button
                  key={ch}
                  onClick={() => {
                    setSelectedChannel(ch);
                    setTestResult(null);
                  }}
                  className={`px-4 py-2.5 rounded-2xl text-xs font-bold transition flex items-center gap-2 cursor-pointer border ${isSelected
                      ? 'bg-blue-600 text-white border-blue-600 shadow-md shadow-blue-500/25 scale-[1.02]'
                      : 'bg-slate-50 hover:bg-slate-100 text-slate-700 border-slate-200'
                    }`}
                >
                  <Video className="w-3.5 h-3.5 opacity-80" />
                  <span>Channel [{ch}]</span>
                  <span className={`text-[10px] font-normal truncate max-w-[120px] ${isSelected ? 'text-blue-100' : 'text-slate-500'}`}>
                    {cam?.name || `Camera ${ch}`}
                  </span>
                  {isDefault && (
                    <span
                      className={`px-1.5 py-0.5 rounded text-[9px] font-extrabold uppercase ${isSelected ? 'bg-white/20 text-white' : 'bg-blue-100 text-blue-700'
                        }`}
                    >
                      Default
                    </span>
                  )}
                </button>
              );
            })}

            <button
              onClick={() => setIsAddModalOpen(true)}
              className="px-3.5 py-2.5 rounded-2xl text-xs font-bold text-slate-500 hover:text-slate-800 bg-slate-50 hover:bg-slate-100 border border-dashed border-slate-300 flex items-center gap-1.5 transition cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Add Channel</span>
            </button>
          </div>
        </div>

        {/* Selected Camera Details Card */}
        <div className="p-6 rounded-3xl bg-slate-50/70 border border-slate-200/90 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200/60 gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-1 rounded-lg text-xs font-mono font-extrabold bg-blue-600 text-white shadow-xs">
                  Channel {selectedChannel}
                </span>
                <h3 className="font-extrabold text-slate-900 text-base font-['Outfit']">
                  {currentCam.name || `Camera ${selectedChannel}`}
                </h3>
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Configure stream source URL, location, and dedicated entry-crossing trigger lines.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => handleUpdateGlobal('default_channel', selectedChannel)}
                disabled={config.default_channel === selectedChannel}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold transition flex items-center gap-1.5 cursor-pointer ${config.default_channel === selectedChannel
                    ? 'bg-emerald-100 text-emerald-800 border border-emerald-200 cursor-default'
                    : 'bg-white hover:bg-slate-100 text-slate-700 border border-slate-200'
                  }`}
              >
                {config.default_channel === selectedChannel ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Default Startup Channel</span>
                  </>
                ) : (
                  <span>Set as Default Channel</span>
                )}
              </button>

              <button
                type="button"
                onClick={() => handleDeleteChannel(selectedChannel)}
                title="Delete Camera Channel"
                className="p-2 rounded-xl text-slate-400 hover:text-rose-600 hover:bg-rose-50 border border-transparent hover:border-rose-200 transition cursor-pointer"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Form Fields: Name & Location */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                Camera Display Name *
              </label>
              <input
                type="text"
                value={currentCam.name}
                onChange={(e) => handleUpdateCurrentCam({ name: e.target.value })}
                placeholder="e.g. Showroom Camera 101 (Main Glass Entrance)"
                className="w-full px-4 py-2.5 text-sm bg-white border border-slate-200 rounded-xl focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition font-semibold text-slate-800"
              />
            </div>

            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                Showroom Location / Physical Zone
              </label>
              <input
                type="text"
                value={currentCam.location || ''}
                onChange={(e) => handleUpdateCurrentCam({ location: e.target.value })}
                placeholder="e.g. Channel 101 (Main Glass Entrance)"
                className="w-full px-4 py-2.5 text-sm bg-white border border-slate-200 rounded-xl focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition font-medium text-slate-800"
              />
            </div>
          </div>

          {/* Stream Source URL & Tester */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600">
                Stream Source URL (RTSP / USB / File) *
              </label>
              <button
                type="button"
                onClick={() => setShowSecret(!showSecret)}
                className="text-[11px] font-semibold text-blue-600 hover:text-blue-800 flex items-center gap-1 cursor-pointer"
              >
                {showSecret ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                <span>{showSecret ? 'Mask Credentials' : 'Show Credentials'}</span>
              </button>
            </div>

            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
              <div className="relative flex-1">
                <input
                  type={showSecret ? 'text' : 'password'}
                  value={currentCam.source}
                  onChange={(e) => handleUpdateCurrentCam({ source: e.target.value })}
                  placeholder="rtsp://username:password@192.168.1.101:554/Streaming/Channels/101"
                  className="w-full px-4 py-2.5 text-xs font-mono bg-white border border-slate-200 rounded-xl focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition text-slate-800"
                />
              </div>

              <button
                type="button"
                onClick={handleTestCurrentStream}
                disabled={isTestingStream}
                className="px-4 py-2.5 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-bold flex items-center justify-center gap-2 transition active:scale-[0.98] disabled:opacity-60 cursor-pointer shrink-0"
              >
                <Activity className={`w-3.5 h-3.5 ${isTestingStream ? 'animate-spin text-blue-400' : 'text-emerald-400'}`} />
                <span>{isTestingStream ? 'Testing Ping...' : 'Test Stream'}</span>
              </button>
            </div>

            {/* Quick Stream Preset Buttons */}
            <div className="mt-2.5 flex items-center gap-1.5 flex-wrap">
              <span className="text-[11px] text-slate-400 mr-1">Quick Presets:</span>
              {PRESET_STREAMS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => handleUpdateCurrentCam({ source: p.url })}
                  className="px-2.5 py-1 bg-white hover:bg-slate-100 border border-slate-200 rounded-lg text-[10px] font-semibold text-slate-600 transition cursor-pointer"
                >
                  {p.label}
                </button>
              ))}
            </div>

            {/* Stream Test Result Display */}
            {testResult && (
              <div
                className={`mt-3 p-3.5 rounded-2xl text-xs border flex items-start gap-2.5 animate-fadeIn ${testResult.online
                    ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                    : 'bg-amber-50 border-amber-200 text-amber-900'
                  }`}
              >
                {testResult.online ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                )}
                <div className="space-y-0.5">
                  <div className="font-bold">
                    {testResult.online ? 'Stream Connected & Verified' : 'Stream Connection Issue'}
                  </div>
                  <p className="text-[11px] opacity-90">
                    {testResult.message || testResult.error}
                  </p>
                  {testResult.resolution && (
                    <div className="font-mono text-[10px] text-emerald-800 pt-1">
                      Frame Dimensions: <strong>{testResult.resolution[0]} x {testResult.resolution[1]} px</strong> | Output: <strong>{testResult.fps} FPS</strong>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Virtual Entry Counting Line Section */}
          <div className="p-5 rounded-2xl bg-white border border-slate-200/90 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-3">
              <div className="flex items-center gap-2">
                <Crosshair className="w-4 h-4 text-blue-600" />
                <h4 className="font-bold text-slate-900 text-sm font-['Outfit']">
                  Virtual Entry-Crossing Trigger Line
                </h4>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-xs text-slate-600 font-semibold select-none cursor-pointer flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={currentCam.entry_line?.enabled ?? true}
                    onChange={(e) =>
                      handleUpdateCurrentCam({
                        entry_line: {
                          enabled: e.target.checked,
                          p1: currentCam.entry_line?.p1 || [200, 500],
                          p2: currentCam.entry_line?.p2 || [1000, 500],
                          direction: currentCam.entry_line?.direction || 'both',
                        },
                      })
                    }
                    className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500"
                  />
                  <span>Enable Entry Line Crossing Counting</span>
                </label>
              </div>
            </div>

            {currentCam.entry_line?.enabled && (
              <div className="space-y-4 pt-1">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs font-bold text-slate-600 uppercase tracking-wider mb-1">
                      Crossing Direction
                    </label>
                    <select
                      value={currentCam.entry_line?.direction || 'both'}
                      onChange={(e) =>
                        handleUpdateCurrentCam({
                          entry_line: {
                            ...currentCam.entry_line!,
                            direction: e.target.value,
                          },
                        })
                      }
                      className="w-full px-3.5 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-600 outline-none font-semibold text-slate-800"
                    >
                      <option value="both">Both Directions (In & Out)</option>
                      <option value="top_to_bottom">Top to Bottom (Entrance)</option>
                      <option value="bottom_to_top">Bottom to Top (Exit)</option>
                      <option value="left_to_right">Left to Right</option>
                      <option value="right_to_left">Right to Left</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-bold text-slate-600 uppercase tracking-wider mb-1">
                      Point 1 Coordinates [X1, Y1]
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        type="number"
                        value={currentCam.entry_line?.p1?.[0] ?? 200}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 0;
                          const p2 = currentCam.entry_line?.p2 || [1000, 500];
                          handleUpdateCurrentCam({
                            entry_line: {
                              ...currentCam.entry_line!,
                              p1: [val, currentCam.entry_line?.p1?.[1] ?? 500],
                              p2,
                            },
                          });
                        }}
                        className="w-full px-3 py-1.5 text-xs font-mono bg-slate-50 border border-slate-200 rounded-xl text-center"
                        placeholder="X1"
                      />
                      <input
                        type="number"
                        value={currentCam.entry_line?.p1?.[1] ?? 500}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 0;
                          const p2 = currentCam.entry_line?.p2 || [1000, 500];
                          handleUpdateCurrentCam({
                            entry_line: {
                              ...currentCam.entry_line!,
                              p1: [currentCam.entry_line?.p1?.[0] ?? 200, val],
                              p2,
                            },
                          });
                        }}
                        className="w-full px-3 py-1.5 text-xs font-mono bg-slate-50 border border-slate-200 rounded-xl text-center"
                        placeholder="Y1"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-bold text-slate-600 uppercase tracking-wider mb-1">
                      Point 2 Coordinates [X2, Y2]
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        type="number"
                        value={currentCam.entry_line?.p2?.[0] ?? 1000}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 0;
                          const p1 = currentCam.entry_line?.p1 || [200, 500];
                          handleUpdateCurrentCam({
                            entry_line: {
                              ...currentCam.entry_line!,
                              p1,
                              p2: [val, currentCam.entry_line?.p2?.[1] ?? 500],
                            },
                          });
                        }}
                        className="w-full px-3 py-1.5 text-xs font-mono bg-slate-50 border border-slate-200 rounded-xl text-center"
                        placeholder="X2"
                      />
                      <input
                        type="number"
                        value={currentCam.entry_line?.p2?.[1] ?? 500}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 0;
                          const p1 = currentCam.entry_line?.p1 || [200, 500];
                          handleUpdateCurrentCam({
                            entry_line: {
                              ...currentCam.entry_line!,
                              p1,
                              p2: [currentCam.entry_line?.p2?.[0] ?? 1000, val],
                            },
                          });
                        }}
                        className="w-full px-3 py-1.5 text-xs font-mono bg-slate-50 border border-slate-200 rounded-xl text-center"
                        placeholder="Y2"
                      />
                    </div>
                  </div>
                </div>

                {/* Line Presets */}
                <div className="flex items-center gap-2 pt-1 flex-wrap text-xs">
                  <span className="text-slate-400 text-[11px]">Preset Positions:</span>
                  <button
                    type="button"
                    onClick={() =>
                      handleUpdateCurrentCam({
                        entry_line: {
                          enabled: true,
                          p1: [200, 500],
                          p2: [1000, 500],
                          direction: 'both',
                        },
                      })
                    }
                    className="px-2.5 py-1 bg-slate-100 hover:bg-slate-200 rounded-lg text-[10px] font-semibold text-slate-700"
                  >
                    Horizontal Mid-Floor [200, 500] - [1000, 500]
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      handleUpdateCurrentCam({
                        entry_line: {
                          enabled: true,
                          p1: [547, 537],
                          p2: [975, 345],
                          direction: 'both',
                        },
                      })
                    }
                    className="px-2.5 py-1 bg-slate-100 hover:bg-slate-200 rounded-lg text-[10px] font-semibold text-slate-700"
                  >
                    Diagonal Entrance [547, 537] - [975, 345]
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      handleUpdateCurrentCam({
                        entry_line: {
                          enabled: true,
                          p1: [640, 150],
                          p2: [640, 680],
                          direction: 'left_to_right',
                        },
                      })
                    }
                    className="px-2.5 py-1 bg-slate-100 hover:bg-slate-200 rounded-lg text-[10px] font-semibold text-slate-700"
                  >
                    Vertical Center [640, 150] - [640, 680]
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* AI Person Detection Model & Resolution */}
        <div className="p-6 rounded-3xl bg-slate-50/70 border border-slate-200/90 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-200/60 gap-3">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-indigo-50 text-indigo-600 border border-indigo-100">
                <Cpu className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-extrabold text-slate-900 text-sm font-['Outfit']">
                  AI Person Detection Model
                </h3>
                <p className="text-xs text-slate-500">
                  Restart, after switching the model
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-lg text-[11px] font-mono font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">
                Active Resolution: {config.imgsz || 1024}px
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* YOLOv26 Small Option */}
            <div
              onClick={() => {
                handleUpdateGlobal('model', 'models/yolo26s.pt');
                handleUpdateGlobal('imgsz', 1024);
              }}
              className={`p-4 rounded-2xl border-2 transition cursor-pointer flex flex-col justify-between gap-3 ${(config.model || 'models/yolo26s.pt') === 'models/yolo26s.pt'
                  ? 'bg-blue-50/60 border-blue-600 shadow-xs'
                  : 'bg-white border-slate-200 hover:border-slate-300'
                }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-extrabold text-sm text-slate-900 font-['Outfit']">
                      YOLOv26 Small (yolo26s)
                    </span>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800">
                      Recommended Default
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Fast real-time inference (~25-30+ FPS) with optimized lightweight weights (~20.4 MB). Ideal for entrance cameras and standard showroom zones.
                  </p>
                </div>
                <div
                  className={`w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0 mt-0.5 ${(config.model || 'models/yolo26s.pt') === 'models/yolo26s.pt'
                      ? 'border-blue-600 bg-blue-600'
                      : 'border-slate-300'
                    }`}
                >
                  {(config.model || 'models/yolo26s.pt') === 'models/yolo26s.pt' && (
                    <span className="w-2 h-2 rounded-full bg-white" />
                  )}
                </div>
              </div>

              <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] font-medium text-slate-600">
                <span>Default Resolution: <strong>1024px</strong></span>
                <span className="text-emerald-700 font-semibold">High FPS / Low Latency</span>
              </div>
            </div>

            {/* YOLOv26 Medium Option */}
            <div
              onClick={() => {
                handleUpdateGlobal('model', 'models/yolo26m.pt');
                handleUpdateGlobal('imgsz', 1280);
              }}
              className={`p-4 rounded-2xl border-2 transition cursor-pointer flex flex-col justify-between gap-3 ${config.model === 'models/yolo26m.pt'
                  ? 'bg-indigo-50/60 border-indigo-600 shadow-xs'
                  : 'bg-white border-slate-200 hover:border-slate-300'
                }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-extrabold text-sm text-slate-900 font-['Outfit']">
                      YOLOv26 Medium (yolo26m)
                    </span>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-100 text-indigo-800">
                      High Precision
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Deeper neural network feature extraction (~44.2 MB). Excels at wide display floors, high ceiling mounts, and distant occlusions.
                  </p>
                </div>
                <div
                  className={`w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0 mt-0.5 ${config.model === 'models/yolo26m.pt'
                      ? 'border-indigo-600 bg-indigo-600'
                      : 'border-slate-300'
                    }`}
                >
                  {config.model === 'models/yolo26m.pt' && (
                    <span className="w-2 h-2 rounded-full bg-white" />
                  )}
                </div>
              </div>

              <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] font-medium text-slate-600">
                <span>Default Resolution: <strong>1280px (Ultra)</strong></span>
                <span className="text-indigo-700 font-semibold">Max Distance & Accuracy</span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Save Bar */}
        <div className="pt-4 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="text-xs text-slate-500 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>Changes save directly and take effect automatically across all showroom cameras.</span>
          </div>

          <button
            type="button"
            onClick={handleSaveAll}
            disabled={isSaving}
            className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-xs font-bold shadow-md shadow-blue-500/20 flex items-center gap-2 transition active:scale-[0.98] disabled:opacity-60 cursor-pointer"
          >
            <Save className="w-4 h-4" />
            <span>{isSaving ? 'Applying Changes...' : 'Save & Sync Camera Settings'}</span>
          </button>
        </div>
      </div>

      {/* ADD NEW CAMERA MODAL */}
      {isAddModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs animate-fadeIn">
          <div className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200 shadow-2xl max-w-md w-full relative">
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center shadow-xs">
                <Video className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-extrabold text-slate-900 text-lg font-['Outfit']">
                  Add Camera Channel
                </h3>
                <p className="text-xs text-slate-500">
                  Connect a new CCTV RTSP or video stream to Edge AI
                </p>
              </div>
            </div>

            <form onSubmit={handleAddCameraSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Channel Identifier (e.g. 101, 501, bay_cam) *
                </label>
                <input
                  type="text"
                  required
                  placeholder="101"
                  value={newChannelNumber}
                  onChange={(e) => setNewChannelNumber(e.target.value)}
                  className="w-full px-4 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none font-mono font-bold"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Camera Display Name *
                </label>
                <input
                  type="text"
                  required
                  placeholder="Showroom Camera 101 (Main Glass Entrance)"
                  value={newCameraName}
                  onChange={(e) => setNewCameraName(e.target.value)}
                  className="w-full px-4 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none font-semibold"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Stream Source (RTSP / USB / Path) *
                </label>
                <input
                  type="text"
                  required
                  placeholder="rtsp://username:password@192.168.1.101:554/Streaming/Channels/101"
                  value={newCameraSource}
                  onChange={(e) => setNewCameraSource(e.target.value)}
                  className="w-full px-4 py-2.5 text-xs font-mono bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none font-medium"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Showroom Location
                </label>
                <input
                  type="text"
                  placeholder="Channel 101 (Main Glass Entrance)"
                  value={newCameraLocation}
                  onChange={(e) => setNewCameraLocation(e.target.value)}
                  className="w-full px-4 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none font-medium"
                />
              </div>

              <div className="pt-3 flex items-center justify-end gap-3 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => setIsAddModalOpen(false)}
                  className="px-4 py-2.5 text-xs font-bold text-slate-600 hover:bg-slate-100 rounded-xl transition cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-xs font-bold shadow-md shadow-blue-500/20 flex items-center gap-2 transition active:scale-[0.98] cursor-pointer"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add & Connect Camera</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

