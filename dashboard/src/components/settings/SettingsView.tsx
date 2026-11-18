import React, { useState, useEffect } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import { useAuth } from '../../context/AuthContext';
import {
  Clock,
  CheckCircle2,
  Save,
  Layers,
  Plus,
  Trash2,
  Edit3,
  X,
  RefreshCw,
  AlertCircle,
  Video,
  Camera,
  Sparkles,
  ToggleLeft,
  ToggleRight,
  Timer,
  Sliders,
  Bell,
  Info,
  AlertTriangle,
} from 'lucide-react';
import {
  cameraService,
  DepartmentModel,
  DepartmentCreatePayload,
  CameraModel,
} from '../../services/cameraService';
import { CameraSettingsSection } from './CameraSettingsSection';

const PRESET_COLORS = [
  { name: 'Cyan', hex: '#00D7FF' },
  { name: 'Emerald', hex: '#10B981' },
  { name: 'Amber', hex: '#F59E0B' },
  { name: 'Purple', hex: '#8B5CF6' },
  { name: 'Rose', hex: '#F43F5E' },
  { name: 'Blue', hex: '#3B82F6' },
  { name: 'Orange', hex: '#FF8C00' },
  { name: 'Teal', hex: '#14B8A6' },
];

export const SettingsView: React.FC = () => {
  const { user } = useAuth();
  const {
    storeHours,
    updateStoreHours,
    isStoreCurrentlyOpen,
    refreshDepartments,
    waitingConfig,
    updateWaitingConfig,
  } = useShowroom();

  // Tab State
  const [activeTab, setActiveTab] = useState<'cameras' | 'detection' | 'schedule' | 'waiting'>('cameras');

  // Waiting Time Form State
  const [formSlaCutoff, setFormSlaCutoff] = useState(waitingConfig.slaCutoffMinutes);
  const [formWarningThreshold, setFormWarningThreshold] = useState(waitingConfig.warningThresholdMinutes);
  const [formStaffProximity, setFormStaffProximity] = useState(waitingConfig.staffProximityMeters);
  const [formSoundAlert, setFormSoundAlert] = useState(waitingConfig.soundAlertEnabled);
  const [formExcludeDepts, setFormExcludeDepts] = useState(waitingConfig.excludeDepartmentZones);
  const [waitingSaveSuccess, setWaitingSaveSuccess] = useState<string | null>(null);

  // Sync state when waitingConfig changes
  useEffect(() => {
    setFormSlaCutoff(waitingConfig.slaCutoffMinutes);
    setFormWarningThreshold(waitingConfig.warningThresholdMinutes);
    setFormStaffProximity(waitingConfig.staffProximityMeters);
    setFormSoundAlert(waitingConfig.soundAlertEnabled);
    setFormExcludeDepts(waitingConfig.excludeDepartmentZones);
  }, [waitingConfig]);

  const handleSaveWaitingConfig = async (e: React.FormEvent) => {
    e.preventDefault();
    const cutoffMins = Number(formSlaCutoff);
    updateWaitingConfig({
      slaCutoffMinutes: cutoffMins,
      warningThresholdMinutes: Number(formWarningThreshold),
      staffProximityMeters: Number(formStaffProximity),
      soundAlertEnabled: Boolean(formSoundAlert),
      excludeDepartmentZones: Boolean(formExcludeDepts),
    });
    try {
      await cameraService.saveCameraConfig({
        waiting_cutoff_seconds: cutoffMins * 60,
      } as any);
    } catch (err) {
      console.warn('Failed to sync SLA cutoff to camera_config.json:', err);
    }
    setWaitingSaveSuccess('Waiting time & SLA configuration saved successfully and synced with Vision AI.');
    setTimeout(() => setWaitingSaveSuccess(null), 4000);
  };

  const handleResetWaitingDefaults = async () => {
    setFormSlaCutoff(15);
    setFormWarningThreshold(5);
    setFormStaffProximity(2.2);
    setFormSoundAlert(true);
    setFormExcludeDepts(true);
    updateWaitingConfig({
      slaCutoffMinutes: 15,
      warningThresholdMinutes: 5,
      staffProximityMeters: 2.2,
      soundAlertEnabled: true,
      excludeDepartmentZones: true,
    });
    try {
      await cameraService.saveCameraConfig({
        waiting_cutoff_seconds: 900,
      } as any);
    } catch (_) {}
    setWaitingSaveSuccess('Reset to showroom default parameters (15m SLA, 5m Warning, 2.2m distance).');
    setTimeout(() => setWaitingSaveSuccess(null), 4000);
  };

  // Store Hours Form State
  const [openingTime, setOpeningTime] = useState(storeHours.openingTime);
  const [closingTime, setClosingTime] = useState(storeHours.closingTime);
  const [timezone, setTimezone] = useState(storeHours.timezone);
  const [isOpenSunday, setIsOpenSunday] = useState(storeHours.isOpenSunday);
  const [hoursNotes, setHoursNotes] = useState(storeHours.notes);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);

  // Departments (Edit Detection) State
  const [departments, setDepartments] = useState<DepartmentModel[]>([]);
  const [cameras, setCameras] = useState<CameraModel[]>([]);
  const [isLoadingDepts, setIsLoadingDepts] = useState<boolean>(true);
  const [isSavingDept, setIsSavingDept] = useState<boolean>(false);
  const [deptMessage, setDeptMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Add / Edit Modal State
  const [isDeptModalOpen, setIsDeptModalOpen] = useState<boolean>(false);
  const [editingDeptId, setEditingDeptId] = useState<string | null>(null);
  const [formDeptName, setFormDeptName] = useState<string>('');
  const [formDeptId, setFormDeptId] = useState<string>('');
  const [formDeptColor, setFormDeptColor] = useState<string>('#00D7FF');
  const [formCameraId, setFormCameraId] = useState<string>('');
  const [formIsActive, setFormIsActive] = useState<boolean>(true);

  // Load departments and cameras on mount
  useEffect(() => {
    loadDepartmentsAndCameras();
  }, []);

  const loadDepartmentsAndCameras = async () => {
    setIsLoadingDepts(true);
    try {
      const [deptsData, camsData] = await Promise.all([
        cameraService.listDepartments(),
        cameraService.listCameras(),
      ]);
      setDepartments(deptsData || []);
      setCameras(camsData || []);
    } catch (err) {
      console.error('Failed to load detection settings:', err);
    } finally {
      setIsLoadingDepts(false);
    }
  };

  const handleOpenAddModal = () => {
    setEditingDeptId(null);
    setFormDeptName('');
    setFormDeptId('');
    setFormDeptColor('#00D7FF');
    setFormCameraId('');
    setFormIsActive(true);
    setIsDeptModalOpen(true);
  };

  const handleOpenEditModal = (dept: DepartmentModel) => {
    setEditingDeptId(dept.id);
    setFormDeptName(dept.name);
    setFormDeptId(dept.id);
    setFormDeptColor(dept.color || '#00D7FF');
    setFormCameraId(dept.camera_id || '');
    setFormIsActive(dept.is_active);
    setIsDeptModalOpen(true);
  };

  const handleDeptNameChange = (name: string) => {
    setFormDeptName(name);
    if (!editingDeptId) {
      // Auto-generate slug ID for new departments
      const slug = name
        .toLowerCase()
        .trim()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '');
      setFormDeptId(slug);
    }
  };

  const handleSaveDepartment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formDeptName.trim()) {
      setDeptMessage({ type: 'error', text: 'Department name is required.' });
      return;
    }

    setIsSavingDept(true);
    try {
      if (editingDeptId) {
        // Update existing department
        const updated = await cameraService.updateDepartment(editingDeptId, {
          name: formDeptName.trim(),
          color: formDeptColor,
          camera_id: formCameraId ? formCameraId : null,
          is_active: formIsActive,
        });
        if (updated) {
          setDepartments((prev) =>
            prev.map((d) => (d.id === editingDeptId ? { ...d, ...updated } : d))
          );
          setDeptMessage({
            type: 'success',
            text: `Department "${formDeptName}" updated and synced with detection screen!`,
          });
        }
      } else {
        // Create new department
        const payload: DepartmentCreatePayload = {
          id: formDeptId.trim() || undefined,
          name: formDeptName.trim(),
          color: formDeptColor,
          camera_id: formCameraId ? formCameraId : null,
          is_active: formIsActive,
        };
        const created = await cameraService.createDepartment(payload);
        if (created) {
          setDepartments((prev) => [...prev, created]);
          refreshDepartments();
          setDeptMessage({
            type: 'success',
            text: `New department "${created.name}" created! It now appears on the CCTV detection screen.`,
          });
        }
      }
      setIsDeptModalOpen(false);
      setTimeout(() => setDeptMessage(null), 4000);
    } catch (err: any) {
      setDeptMessage({ type: 'error', text: err?.message || 'Failed to save department.' });
    } finally {
      setIsSavingDept(false);
    }
  };

  const handleDeleteDepartment = async (dept: DepartmentModel) => {
    if (!window.confirm(`Are you sure you want to delete the "${dept.name}" detection department? This will remove its zone from the CCTV screen and database.`)) {
      return;
    }

    try {
      const ok = await cameraService.deleteDepartment(dept.id);
      if (ok) {
        setDepartments((prev) => prev.filter((d) => d.id !== dept.id));
        setDeptMessage({
          type: 'success',
          text: `Department "${dept.name}" removed from detection screen and database.`,
        });
        setTimeout(() => setDeptMessage(null), 3000);
      }
    } catch (err) {
      setDeptMessage({ type: 'error', text: 'Failed to delete department.' });
    }
  };

  const handleToggleActive = async (dept: DepartmentModel) => {
    try {
      const updated = await cameraService.updateDepartment(dept.id, {
        is_active: !dept.is_active,
      });
      if (updated) {
        setDepartments((prev) =>
          prev.map((d) => (d.id === dept.id ? { ...d, is_active: !d.is_active } : d))
        );
      }
    } catch (err) {
      console.error('Failed to toggle department:', err);
    }
  };

  const handleSaveStoreHours = (e: React.FormEvent) => {
    e.preventDefault();
    updateStoreHours({
      openingTime,
      closingTime,
      timezone,
      isOpenSunday,
      notes: hoursNotes,
    });
    setSaveSuccess('Showroom operating hours saved successfully.');
    setTimeout(() => setSaveSuccess(null), 3000);
  };

  return (
    <div className="space-y-6 pb-12 max-w-5xl">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-['Outfit'] text-slate-900">Showroom Settings & Detection</h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-blue-100 text-blue-800">
              Capermint Edge AI
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Configure dynamic department detection zones, CCTV camera parameters, and showroom operating schedule.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-100/90 rounded-2xl border border-slate-200">
          <button
            onClick={() => setActiveTab('cameras')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition flex items-center gap-2 cursor-pointer ${
              activeTab === 'cameras'
                ? 'bg-white text-blue-600 shadow-sm border border-slate-200/60'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            <span>Camera Settings</span>
          </button>
          <button
            onClick={() => setActiveTab('detection')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition flex items-center gap-2 cursor-pointer ${
              activeTab === 'detection'
                ? 'bg-white text-blue-600 shadow-sm border border-slate-200/60'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Edit Detection & Zones</span>
          </button>
          <button
            onClick={() => setActiveTab('schedule')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition flex items-center gap-2 cursor-pointer ${
              activeTab === 'schedule'
                ? 'bg-white text-blue-600 shadow-sm border border-slate-200/60'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Clock className="w-3.5 h-3.5" />
            <span>Store Schedule</span>
          </button>
          <button
            onClick={() => setActiveTab('waiting')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition flex items-center gap-2 cursor-pointer ${
              activeTab === 'waiting'
                ? 'bg-white text-blue-600 shadow-sm border border-slate-200/60'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Timer className="w-3.5 h-3.5" />
            <span>Waiting Time &amp; SLA</span>
          </button>
        </div>
      </div>

      {/* Global Alerts */}
      {deptMessage && (
        <div
          className={`p-3.5 rounded-2xl text-xs flex items-center gap-2.5 animate-fadeIn border ${
            deptMessage.type === 'success'
              ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
              : 'bg-rose-50 border-rose-200 text-rose-900'
          }`}
        >
          {deptMessage.type === 'success' ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          ) : (
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          )}
          <span className="font-medium">{deptMessage.text}</span>
        </div>
      )}

      {saveSuccess && (
        <div className="p-3 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-xl text-xs flex items-center gap-2 animate-fadeIn">
          <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          <span>{saveSuccess}</span>
        </div>
      )}

      {/* TAB 0: CAMERA SETTINGS */}
      {activeTab === 'cameras' && <CameraSettingsSection />}

      {/* TAB 1: EDIT DETECTION & DEPARTMENTS */}
      {activeTab === 'detection' && (
        <div className="space-y-6">
          {/* Main Card */}
          <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)]">
            {/* Top Toolbar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-slate-100 gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-2xl bg-blue-50 text-blue-600 border border-blue-100 shadow-2xs">
                  <Layers className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="font-extrabold text-slate-900 text-lg font-['Outfit']">
                    Showroom Detection Departments
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Add or remove departments. Each department dynamically appears on the CCTV Edge AI detector screen.
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={loadDepartmentsAndCameras}
                  title="Refresh from Database"
                  className="p-2.5 text-slate-500 hover:text-slate-800 bg-slate-50 hover:bg-slate-100 rounded-xl border border-slate-200 transition cursor-pointer"
                >
                  <RefreshCw className={`w-4 h-4 ${isLoadingDepts ? 'animate-spin' : ''}`} />
                </button>
                <button
                  onClick={handleOpenAddModal}
                  className="px-4 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-xs font-bold shadow-md shadow-blue-500/20 flex items-center gap-2 transition active:scale-[0.98] cursor-pointer"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add Department</span>
                </button>
              </div>
            </div>


            {/* Department List */}
            <div className="mt-6">
              {isLoadingDepts ? (
                <div className="py-12 flex flex-col items-center justify-center text-slate-400 gap-3">
                  <RefreshCw className="w-6 h-6 animate-spin text-blue-500" />
                  <span className="text-xs font-medium">Loading detection departments...</span>
                </div>
              ) : departments.length === 0 ? (
                <div className="py-12 border-2 border-dashed border-slate-200 rounded-2xl flex flex-col items-center justify-center text-center p-6">
                  <div className="w-12 h-12 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mb-3">
                    <Layers className="w-6 h-6" />
                  </div>
                  <h3 className="text-sm font-bold text-slate-800">No detection departments configured</h3>
                  <p className="text-xs text-slate-500 mt-1 max-w-sm">
                    Click "Add Department" above to create your first dynamic showroom zone (e.g. Sales, Service, VIP Lounge).
                  </p>
                  <button
                    onClick={handleOpenAddModal}
                    className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-xl text-xs font-bold hover:bg-blue-700 transition"
                  >
                    Create Department
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {departments.map((dept) => {
                    const assignedCam = cameras.find((c) => c.id === dept.camera_id);
                    const activeIndex = departments.filter((d) => d.is_active).findIndex((d) => d.id === dept.id);
                    const shortcutKey = activeIndex >= 0 ? activeIndex + 2 : null;

                    return (
                      <div
                        key={dept.id}
                        className={`p-5 rounded-2xl border transition relative overflow-hidden group ${
                          dept.is_active
                            ? 'bg-slate-50/70 border-slate-200/90 hover:border-blue-300 hover:shadow-md hover:bg-white'
                            : 'bg-slate-100/50 border-slate-200/50 opacity-60'
                        }`}
                      >
                        {/* Color accent left bar */}
                        <div
                          className="absolute left-0 top-0 bottom-0 w-1.5"
                          style={{ backgroundColor: dept.color || '#00D7FF' }}
                        />

                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-center gap-3">
                            <span
                              className="w-4 h-4 rounded-full border border-white shadow-xs shrink-0"
                              style={{ backgroundColor: dept.color || '#00D7FF' }}
                            />
                            <div>
                              <div className="flex items-center gap-2">
                                <h3 className="font-bold text-slate-900 text-sm font-['Outfit']">
                                  {dept.name}
                                </h3>
                                {shortcutKey ? (
                                  <span className="px-2 py-0.5 rounded-md text-[10px] font-extrabold bg-blue-50 text-blue-700 border border-blue-200/60 font-mono">
                                    Key [{shortcutKey}]
                                  </span>
                                ) : (
                                  <span className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-slate-100 text-slate-500 border border-slate-200 font-mono">
                                    Inactive
                                  </span>
                                )}
                              </div>
                              <p className="text-[11px] font-mono text-slate-500 mt-0.5">
                                id: <strong>{dept.id}</strong>
                              </p>
                            </div>
                          </div>

                          {/* Quick Actions */}
                          <div className="flex items-center gap-1 opacity-90 group-hover:opacity-100">
                            <button
                              onClick={() => handleToggleActive(dept)}
                              title={dept.is_active ? 'Deactivate department' : 'Activate department'}
                              className="p-1.5 text-slate-400 hover:text-slate-700 transition cursor-pointer"
                            >
                              {dept.is_active ? (
                                <ToggleRight className="w-5 h-5 text-emerald-500" />
                              ) : (
                                <ToggleLeft className="w-5 h-5 text-slate-400" />
                              )}
                            </button>
                            <button
                              onClick={() => handleOpenEditModal(dept)}
                              title="Edit Department"
                              className="p-1.5 text-slate-400 hover:text-blue-600 transition cursor-pointer"
                            >
                              <Edit3 className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => handleDeleteDepartment(dept)}
                              title="Delete Department"
                              className="p-1.5 text-slate-400 hover:text-rose-600 transition cursor-pointer"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </div>

                        {/* Metadata Tags */}
                        <div className="mt-4 pt-3 border-t border-slate-200/60 flex items-center justify-between text-[11px] text-slate-500">
                          <div className="flex items-center gap-1.5">
                            <Video className="w-3.5 h-3.5 text-slate-400" />
                            <span>{assignedCam ? assignedCam.name : 'All Showroom Cameras'}</span>
                          </div>

                          <span className="font-mono text-slate-600 bg-white px-2 py-0.5 rounded border border-slate-200 text-[10px]">
                            {dept.box && dept.box.some((b) => b > 0)
                              ? `Box: [${dept.box.join(', ')}]`
                              : 'Ready to Draw on Screen'}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Guidance Tip */}
            <div className="mt-8 p-4 rounded-2xl bg-amber-50/80 border border-amber-200/70 text-amber-900 text-xs flex items-start gap-3">
              <Sparkles className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <strong>How to draw the physical zone on CCTV:</strong>
                <p className="mt-0.5 text-amber-800 text-[11px] leading-relaxed">
                  On the local detector window, select your department's tool (e.g. click its button on the top bar or press its number key like <strong>[2]</strong>, <strong>[3]</strong>), then <strong>Click & Drag</strong> a box over the designated floor area. The coordinates save automatically to the database and config.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: STORE SCHEDULE */}
      {activeTab === 'schedule' && (
        <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)]">
          <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-6">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-2xl bg-purple-50 text-purple-600 border border-purple-100 shadow-2xs">
                <Clock className="w-5 h-5" />
              </div>
              <div>
                <h2 className="font-extrabold text-slate-900 text-lg font-['Outfit']">
                  Store Operating Schedule
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">Live open/closed status calculation & footfall tracking window</p>
              </div>
            </div>
          </div>

          {/* Current Real-time Store Status Banner */}
          <div
            className={`p-4 sm:p-5 rounded-2xl border mb-6 flex items-center justify-between text-xs transition shadow-2xs ${
              isStoreCurrentlyOpen
                ? 'bg-emerald-50/90 border-emerald-200 text-emerald-950'
                : 'bg-rose-50/90 border-rose-200 text-rose-950'
            }`}
          >
            <div className="flex items-center gap-3">
              <span
                className={`w-3 h-3 rounded-full ${
                  isStoreCurrentlyOpen ? 'bg-emerald-500 animate-ping' : 'bg-rose-500'
                }`}
              />
              <span className="font-extrabold text-sm font-['Outfit']">
                Current Live Status: <strong>{isStoreCurrentlyOpen ? 'Showroom is OPEN' : 'Showroom is CLOSED'}</strong>
              </span>
            </div>
            <span className="font-mono text-slate-700 bg-white/80 px-3 py-1.5 rounded-xl border border-slate-200 font-bold shadow-2xs">
              {storeHours.timezone}
            </span>
          </div>

          <form onSubmit={handleSaveStoreHours} className="space-y-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                  Opening Time (24h or AM/PM)
                </label>
                <input
                  type="time"
                  value={openingTime}
                  onChange={(e) => setOpeningTime(e.target.value)}
                  className="w-full px-4 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition font-semibold"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                  Closing Time (24h or AM/PM)
                </label>
                <input
                  type="time"
                  value={closingTime}
                  onChange={(e) => setClosingTime(e.target.value)}
                  className="w-full px-4 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition font-semibold"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                Local Showroom Timezone
              </label>
              <select
                value={timezone}
                onChange={(e) => setTimezone(e.target.value)}
                className="w-full px-4 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none transition font-semibold text-slate-700"
              >
                <option value="Asia/Kolkata (IST +5:30)">Asia/Kolkata (IST +5:30)</option>
                <option value="Asia/Dubai (GST +4:00)">Asia/Dubai (GST +4:00)</option>
                <option value="Asia/Singapore (SGT +8:00)">Asia/Singapore (SGT +8:00)</option>
              </select>
            </div>

            <div className="flex items-center gap-2.5 pt-1">
              <input
                id="sunday-open"
                type="checkbox"
                checked={isOpenSunday}
                onChange={(e) => setIsOpenSunday(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500"
              />
              <label htmlFor="sunday-open" className="text-xs text-slate-700 font-semibold select-none cursor-pointer">
                Showroom is open on Sundays for customer test rides & bookings
              </label>
            </div>

            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                Operating Notes & Special Timings
              </label>
              <textarea
                value={hoursNotes}
                onChange={(e) => setHoursNotes(e.target.value)}
                rows={3}
                placeholder="e.g. Workshop open 7 days a week. Service counter closes at 7:00 PM."
                className="w-full p-3.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition resize-none font-medium text-slate-800"
              />
            </div>

            <button
              type="submit"
              className="w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-sm font-bold border border-blue-400/30 shadow-md shadow-blue-500/20 flex items-center justify-center gap-2 transition active:scale-[0.98] cursor-pointer"
            >
              <Save className="w-4 h-4" />
              <span>Save Operating Hours</span>
            </button>
          </form>
        </div>
      )}

      {/* TAB 3: WAITING TIME & SERVICE SLA CONFIGURATION */}
      {activeTab === 'waiting' && (
        <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] space-y-6">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-3">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-2xl bg-amber-50 text-amber-600 border border-amber-200/80 shadow-2xs">
                <Timer className="w-5 h-5" />
              </div>
              <div>
                <h2 className="font-extrabold text-slate-900 text-lg font-['Outfit']">
                  Customer Waiting Time &amp; Service SLA Configuration
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Configure SLA breach cutoff times, attention alert thresholds, and staff attendance proximity rules.
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={handleResetWaitingDefaults}
              className="text-xs text-slate-500 hover:text-slate-800 font-semibold px-3 py-1.5 rounded-xl border border-slate-200 hover:bg-slate-50 transition self-start sm:self-auto cursor-pointer"
            >
              Reset to Defaults (15m SLA)
            </button>
          </div>

          {/* Success Banner */}
          {waitingSaveSuccess && (
            <div className="p-3.5 bg-emerald-50 border border-emerald-200 text-emerald-900 rounded-2xl text-xs flex items-center gap-2.5 animate-fadeIn">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span className="font-medium">{waitingSaveSuccess}</span>
            </div>
          )}

          {/* Current Live Status Banner */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-4 bg-red-50/80 rounded-2xl border border-red-200/80">
              <span className="text-[10px] font-bold uppercase tracking-wider text-red-700">SLA Breach Cutoff</span>
              <p className="text-2xl font-black font-['Outfit'] text-red-950 mt-1">
                {formSlaCutoff} <span className="text-xs font-semibold text-red-700">mins</span>
              </p>
              <span className="text-[10px] text-red-600 font-medium">Critical red alert cutoff</span>
            </div>

            <div className="p-4 bg-amber-50/80 rounded-2xl border border-amber-200/80">
              <span className="text-[10px] font-bold uppercase tracking-wider text-amber-800">Warning Threshold</span>
              <p className="text-2xl font-black font-['Outfit'] text-amber-950 mt-1">
                {formWarningThreshold} <span className="text-xs font-semibold text-amber-800">mins</span>
              </p>
              <span className="text-[10px] text-amber-700 font-medium">Early attention banner</span>
            </div>

            <div className="p-4 bg-blue-50/80 rounded-2xl border border-blue-200/80">
              <span className="text-[10px] font-bold uppercase tracking-wider text-blue-800">Staff Proximity</span>
              <p className="text-2xl font-black font-['Outfit'] text-blue-950 mt-1">
                {formStaffProximity} <span className="text-xs font-semibold text-blue-800">meters</span>
              </p>
              <span className="text-[10px] text-blue-700 font-medium">AI detection attendance radius</span>
            </div>

            <div className="p-4 bg-slate-50/90 rounded-2xl border border-slate-200">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-600">Department Exclusion</span>
              <p className="text-2xl font-black font-['Outfit'] text-slate-900 mt-1">
                {formExcludeDepts ? 'Active' : 'Off'}
              </p>
              <span className="text-[10px] text-slate-500 font-medium">Desk zones exempt from wait</span>
            </div>
          </div>

          <form onSubmit={handleSaveWaitingConfig} className="space-y-6 pt-2">
            {/* Setting 1: SLA Cutoff */}
            <div className="p-5 rounded-2xl bg-slate-50/70 border border-slate-200/80 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-1.5">
                    <span>1. Critical SLA Breach Cutoff Limit (Minutes)</span>
                    <span className="px-2 py-0.5 rounded-full text-[10px] bg-red-100 text-red-700 font-bold">High Priority</span>
                  </label>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    If an unassisted customer waits longer than this duration on the floor, the system triggers a critical SLA breach alert.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={1}
                    max={60}
                    value={formSlaCutoff}
                    onChange={(e) => {
                      const val = Math.max(1, parseInt(e.target.value, 10) || 1);
                      setFormSlaCutoff(val);
                      updateWaitingConfig({ slaCutoffMinutes: val });
                      cameraService.saveCameraConfig({ waiting_cutoff_seconds: val * 60 } as any).catch(() => {});
                    }}
                    className="w-24 px-3 py-2 text-sm bg-white border border-slate-300 rounded-xl font-bold text-center focus:border-blue-600 outline-none shadow-2xs"
                  />
                  <span className="text-xs font-bold text-slate-500">mins</span>
                </div>
              </div>

              {/* Quick Preset Pills */}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="text-[10px] font-bold text-slate-500 uppercase">Presets:</span>
                {[2, 5, 10, 15, 20, 25, 30].map((mins) => (
                  <button
                    key={mins}
                    type="button"
                    onClick={() => {
                      setFormSlaCutoff(mins);
                      updateWaitingConfig({ slaCutoffMinutes: mins });
                      cameraService.saveCameraConfig({ waiting_cutoff_seconds: mins * 60 } as any).catch(() => {});
                    }}
                    className={`px-3 py-1 rounded-xl text-xs font-bold transition border cursor-pointer ${
                      formSlaCutoff === mins
                        ? 'bg-red-600 text-white border-red-600 shadow-xs'
                        : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    {mins} mins {mins === 15 ? '(Default)' : mins === 2 ? '(Test 2m)' : mins === 5 ? '(Express)' : ''}
                  </button>
                ))}
              </div>
            </div>

            {/* Setting 2: Early Warning Threshold */}
            <div className="p-5 rounded-2xl bg-slate-50/70 border border-slate-200/80 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-1.5">
                    <span>2. Early Attention Warning Threshold (Minutes)</span>
                    <span className="px-2 py-0.5 rounded-full text-[10px] bg-amber-100 text-amber-800 font-bold">Proactive</span>
                  </label>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    Triggers the amber attention banner and highlights the visitor in the waiting queue before the SLA is breached.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={1}
                    max={formSlaCutoff - 1}
                    value={formWarningThreshold}
                    onChange={(e) => setFormWarningThreshold(Math.max(1, parseInt(e.target.value, 10) || 1))}
                    className="w-24 px-3 py-2 text-sm bg-white border border-slate-300 rounded-xl font-bold text-center focus:border-blue-600 outline-none shadow-2xs"
                  />
                  <span className="text-xs font-bold text-slate-500">mins</span>
                </div>
              </div>

              {/* Quick Preset Pills */}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="text-[10px] font-bold text-slate-500 uppercase">Presets:</span>
                {[2, 3, 5, 7, 10].map((mins) => (
                  <button
                    key={mins}
                    type="button"
                    onClick={() => setFormWarningThreshold(mins)}
                    className={`px-3 py-1 rounded-xl text-xs font-bold transition border cursor-pointer ${
                      formWarningThreshold === mins
                        ? 'bg-amber-600 text-white border-amber-600 shadow-xs'
                        : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    {mins} mins {mins === 5 ? '(Default)' : ''}
                  </button>
                ))}
              </div>
            </div>

            {/* Setting 3: Staff Attendance Proximity */}
            <div className="p-5 rounded-2xl bg-slate-50/70 border border-slate-200/80 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-1.5">
                    <span>3. Staff Proximity Distance Threshold (Meters)</span>
                  </label>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    The AI vision detector checks perspective ground distance. If a registered staff member is within this radius, the customer is counted as attended.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    step="0.1"
                    min={1.0}
                    max={5.0}
                    value={formStaffProximity}
                    onChange={(e) => setFormStaffProximity(parseFloat(e.target.value) || 2.2)}
                    className="w-24 px-3 py-2 text-sm bg-white border border-slate-300 rounded-xl font-bold text-center focus:border-blue-600 outline-none shadow-2xs"
                  />
                  <span className="text-xs font-bold text-slate-500">meters</span>
                </div>
              </div>

              {/* Quick Preset Pills */}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="text-[10px] font-bold text-slate-500 uppercase">Presets:</span>
                {[1.8, 2.0, 2.2, 2.5, 3.0].map((dist) => (
                  <button
                    key={dist}
                    type="button"
                    onClick={() => setFormStaffProximity(dist)}
                    className={`px-3 py-1 rounded-xl text-xs font-bold transition border cursor-pointer ${
                      formStaffProximity === dist
                        ? 'bg-blue-600 text-white border-blue-600 shadow-xs'
                        : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    {dist} m {dist === 2.2 ? '(Default)' : ''}
                  </button>
                ))}
              </div>
            </div>

            {/* Setting 4 & 5: Toggles for Department Desk Zones & Sound Alerts */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div
                onClick={() => setFormExcludeDepts(!formExcludeDepts)}
                className={`p-4 rounded-2xl border transition-all cursor-pointer flex items-start gap-3.5 select-none ${
                  formExcludeDepts
                    ? 'bg-blue-50/60 border-blue-200 shadow-2xs'
                    : 'bg-slate-50/80 border-slate-200 opacity-80'
                }`}
              >
                <div className="mt-0.5 text-blue-600">
                  {formExcludeDepts ? (
                    <ToggleRight className="w-6 h-6 text-blue-600" />
                  ) : (
                    <ToggleLeft className="w-6 h-6 text-slate-400" />
                  )}
                </div>
                <div>
                  <h4 className="font-bold text-xs text-slate-900">Exclude Department Desk Zones</h4>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    Customers seated or standing inside defined department zones (Sales Desk, Accounts, Delivery) are considered assisted.
                  </p>
                </div>
              </div>

              <div
                onClick={() => setFormSoundAlert(!formSoundAlert)}
                className={`p-4 rounded-2xl border transition-all cursor-pointer flex items-start gap-3.5 select-none ${
                  formSoundAlert
                    ? 'bg-amber-50/60 border-amber-200 shadow-2xs'
                    : 'bg-slate-50/80 border-slate-200 opacity-80'
                }`}
              >
                <div className="mt-0.5 text-amber-600">
                  {formSoundAlert ? (
                    <ToggleRight className="w-6 h-6 text-amber-600" />
                  ) : (
                    <ToggleLeft className="w-6 h-6 text-slate-400" />
                  )}
                </div>
                <div>
                  <h4 className="font-bold text-xs text-slate-900">Sound / Chime Alert on SLA Breach</h4>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    Plays an alert sound on the dashboard browser when a customer crosses the SLA cutoff without assistance.
                  </p>
                </div>
              </div>
            </div>

            {/* Save Button */}
            <div className="pt-2">
              <button
                type="submit"
                className="w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-sm font-bold border border-blue-400/30 shadow-md shadow-blue-500/20 flex items-center justify-center gap-2 transition active:scale-[0.98] cursor-pointer"
              >
                <Save className="w-4 h-4" />
                <span>Save Waiting Time &amp; SLA Configuration</span>
              </button>
            </div>
          </form>
        </div>
      )}

      {/* ADD / EDIT DEPARTMENT MODAL */}
      {isDeptModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs animate-fadeIn">
          <div className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200 shadow-2xl max-w-md w-full relative">
            <button
              onClick={() => setIsDeptModalOpen(false)}
              className="absolute right-5 top-5 p-2 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="flex items-center gap-3 mb-5">
              <div
                className="w-10 h-10 rounded-2xl flex items-center justify-center shadow-xs"
                style={{ backgroundColor: `${formDeptColor}20`, color: formDeptColor }}
              >
                <Layers className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-extrabold text-slate-900 text-lg font-['Outfit']">
                  {editingDeptId ? 'Edit Department Zone' : 'Add Detection Department'}
                </h3>
                <p className="text-xs text-slate-500">
                  {editingDeptId ? 'Update zone parameters & color' : 'Creates a new dynamic detection area'}
                </p>
              </div>
            </div>

            <form onSubmit={handleSaveDepartment} className="space-y-4">
              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Department Name *
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. VIP Lounge, Accessories, Billing"
                  value={formDeptName}
                  onChange={(e) => handleDeptNameChange(e.target.value)}
                  className="w-full px-4 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none transition font-semibold"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Identifier Code (Slug) *
                </label>
                <input
                  type="text"
                  required
                  disabled={!!editingDeptId}
                  placeholder="e.g. vip_lounge"
                  value={formDeptId}
                  onChange={(e) => setFormDeptId(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ''))}
                  className="w-full px-4 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none transition font-mono text-slate-700 font-medium disabled:opacity-60"
                />
                <span className="text-[10px] text-slate-400 mt-1 block">Used internally by CCTV tracker and footfall metrics</span>
              </div>

              {/* Color Selection */}
              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Zone Color Preview
                </label>
                <div className="flex items-center gap-2 mb-2">
                  <input
                    type="color"
                    value={formDeptColor}
                    onChange={(e) => setFormDeptColor(e.target.value)}
                    className="w-9 h-9 p-0.5 rounded-xl border border-slate-200 cursor-pointer bg-white"
                  />
                  <input
                    type="text"
                    value={formDeptColor}
                    onChange={(e) => setFormDeptColor(e.target.value)}
                    className="w-28 px-3 py-1.5 text-xs font-mono uppercase bg-slate-50 border border-slate-200 rounded-xl"
                  />
                </div>

                {/* Preset Swatches */}
                <div className="flex items-center gap-1.5 flex-wrap">
                  {PRESET_COLORS.map((c) => (
                    <button
                      key={c.hex}
                      type="button"
                      onClick={() => setFormDeptColor(c.hex)}
                      className={`w-6 h-6 rounded-lg transition-transform cursor-pointer border ${
                        formDeptColor.toLowerCase() === c.hex.toLowerCase()
                          ? 'scale-115 ring-2 ring-blue-500 border-white'
                          : 'border-slate-200 hover:scale-105'
                      }`}
                      style={{ backgroundColor: c.hex }}
                      title={c.name}
                    />
                  ))}
                </div>
              </div>

              {/* Assigned Camera */}
              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Target CCTV Camera
                </label>
                <select
                  value={formCameraId}
                  onChange={(e) => setFormCameraId(e.target.value)}
                  className="w-full px-4 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none transition font-medium text-slate-700"
                >
                  <option value="">Global (All Showroom Cameras)</option>
                  {cameras.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name} ({c.location || 'Showroom'})
                    </option>
                  ))}
                </select>
              </div>

              <div className="pt-3 flex items-center justify-end gap-3">
                <button
                  type="button"
                  onClick={() => setIsDeptModalOpen(false)}
                  className="px-4 py-2.5 text-xs font-bold text-slate-600 hover:bg-slate-100 rounded-xl transition cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSavingDept}
                  className="px-5 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-xs font-bold shadow-md shadow-blue-500/20 flex items-center gap-2 transition active:scale-[0.98] disabled:opacity-60 cursor-pointer"
                >
                  <Save className="w-4 h-4" />
                  <span>{isSavingDept ? 'Saving...' : editingDeptId ? 'Update & Sync' : 'Create & Sync'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
