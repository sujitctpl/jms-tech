import React, { useState, useRef, useEffect } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import { FASTAPI_BASE_URL } from '../../config/env';
import {
  X,
  UserPlus,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Camera,
  Trash2,
  Sparkles,
  Clock,
  Building2,
  ShieldCheck,
  Star,
  Plus,
} from 'lucide-react';

interface AddEmployeeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onEmployeeAdded?: (employeeName: string) => void;
}

interface EmployeePhoto {
  id: string;
  file?: File;
  preview: string;
  name: string;
  size?: number;
}

// Client-side image optimizer to keep memory and local storage lightweight
const compressImageFile = (
  file: File,
  maxDim = 640,
  quality = 0.85
): Promise<{ file: File; preview: string }> => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;
        if (width > maxDim || height > maxDim) {
          if (width > height) {
            height = Math.round((height * maxDim) / width);
            width = maxDim;
          } else {
            width = Math.round((width * maxDim) / height);
            height = maxDim;
          }
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(img, 0, 0, width, height);
          const dataUrl = canvas.toDataURL('image/jpeg', quality);
          canvas.toBlob(
            (blob) => {
              if (blob) {
                const compressedFile = new File(
                  [blob],
                  file.name.replace(/\.[^/.]+$/, '.jpg'),
                  { type: 'image/jpeg' }
                );
                resolve({ file: compressedFile, preview: dataUrl });
              } else {
                resolve({ file, preview: e.target?.result as string });
              }
            },
            'image/jpeg',
            quality
          );
        } else {
          resolve({ file, preview: e.target?.result as string });
        }
      };
      img.onerror = () => resolve({ file, preview: e.target?.result as string });
      img.src = e.target?.result as string;
    };
    reader.onerror = () => resolve({ file, preview: '' });
    reader.readAsDataURL(file);
  });
};

const DEPARTMENTS = [
  'Sales',
  'Service',
  'Finance & Insurance',
  'Customer Relations',
  'Accessories & Spares',
] as const;

export const AddEmployeeModal: React.FC<AddEmployeeModalProps> = ({
  isOpen,
  onClose,
  onEmployeeAdded,
}) => {
  const { employees, addEmployee, refreshEmployees } = useShowroom();

  const [employeeId, setEmployeeId] = useState('');
  const [name, setName] = useState('');
  const [department, setDepartment] = useState<typeof DEPARTMENTS[number]>('Sales');
  const [designation, setDesignation] = useState('Sales Executive');
  const [assignedLocation, setAssignedLocation] = useState('Main Showroom Floor');
  const [shiftStart, setShiftStart] = useState('09:30');
  const [shiftEnd, setShiftEnd] = useState('18:30');
  const [lateGrace, setLateGrace] = useState('15');

  // Multi-Photo / Webcam state
  const [photos, setPhotos] = useState<EmployeePhoto[]>([]);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  // Status state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Auto-generate next employee ID when opened by checking both local and backend IDs
  useEffect(() => {
    if (isOpen) {
      setErrorMsg(null);
      setSuccessMsg(null);

      const computeNextEmployeeId = async () => {
        let allIds = employees.map((e) => e.id);
        try {
          const res = await fetch(`${FASTAPI_BASE_URL}/api/v1/employees`);
          if (res.ok) {
            const data = await res.json();
            if (Array.isArray(data)) {
              allIds = Array.from(new Set([...allIds, ...data.map((d: any) => d.employee_id)]));
            }
          }
        } catch {
          // Backend might be offline; fallback to local state IDs
        }

        const numericIds = allIds
          .map((id) => {
            const match = id.match(/EMP-(\d+)/i) || id.match(/(\d+)/);
            return match ? parseInt(match[1], 10) : 0;
          })
          .filter((n) => n >= 100);

        const maxNum = numericIds.length > 0 ? Math.max(...numericIds) : 100;
        setEmployeeId(`EMP-${maxNum + 1}`);
      };

      computeNextEmployeeId();
    } else {
      stopWebcam();
    }
  }, [isOpen, employees]);

  const stopWebcam = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    setIsWebcamActive(false);
  };

  const startWebcam = async () => {
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 640 }, facingMode: 'user' },
      });
      mediaStreamRef.current = stream;
      setIsWebcamActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch {
      setErrorMsg('Could not access webcam. Please upload an image file instead.');
      setIsWebcamActive(false);
    }
  };

  const captureWebcamSnapshot = () => {
    if (!videoRef.current) return;
    const canvas = canvasRef.current || document.createElement('canvas');
    canvas.width = 480;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0, 480, 480);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

      // Convert data URL to File object for multipart upload
      canvas.toBlob((blob) => {
        if (blob) {
          const photoNum = photos.length + 1;
          const file = new File([blob], `${employeeId || 'employee'}-snap-${photoNum}.jpg`, { type: 'image/jpeg' });
          setPhotos((prev) => [
            ...prev,
            {
              id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
              file,
              preview: dataUrl,
              name: `Webcam Snapshot #${photoNum}`,
              size: blob.size,
            },
          ]);
        }
      }, 'image/jpeg', 0.92);
    }
    stopWebcam();
  };

  const handleFilesChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMsg(null);
    if (e.target.files && e.target.files.length > 0) {
      const fileList = Array.from(e.target.files);
      const validImages = fileList.filter((f) => f.type.startsWith('image/'));
      if (validImages.length === 0) {
        setErrorMsg('Please select valid image files (JPG or PNG).');
        return;
      }

      for (const file of validImages) {
        try {
          const { file: compressedFile, preview } = await compressImageFile(file);
          setPhotos((prev) => [
            ...prev,
            {
              id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
              file: compressedFile,
              preview,
              name: file.name,
              size: compressedFile.size,
            },
          ]);
        } catch {
          const reader = new FileReader();
          reader.onload = () => {
            setPhotos((prev) => [
              ...prev,
              {
                id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
                file,
                preview: reader.result as string,
                name: file.name,
                size: file.size,
              },
            ]);
          };
          reader.readAsDataURL(file);
        }
      }
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleRemovePhoto = (index: number) => {
    setPhotos((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSetPrimary = (index: number) => {
    if (index === 0) return;
    setPhotos((prev) => {
      const selected = prev[index];
      const remaining = prev.filter((_, i) => i !== index);
      return [selected, ...remaining];
    });
  };

  const handleClearAllPhotos = () => {
    setPhotos([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);

    const cleanId = employeeId.trim().toUpperCase();
    const cleanName = name.trim();

    if (!cleanId || !cleanName) {
      setErrorMsg('Employee ID and Full Name are required.');
      return;
    }

    setIsSubmitting(true);

    try {
      // 1. Prepare base64 face images if photos were snapped or uploaded
      const faceImagesBase64 = photos.map((p) => p.preview).filter(Boolean);

      // 2. Register employee profile with FastAPI backend (including face images so embeddings are created atomically)
      let backendSuccess = false;
      let registeredResponse: any = null;

      try {
        const createRes = await fetch(`${FASTAPI_BASE_URL}/api/v1/employees`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            employee_id: cleanId,
            name: cleanName,
            department: department,
            designation: designation.trim() || 'Floor Consultant',
            assigned_location: assignedLocation.trim() || undefined,
            shift_start_time: `${shiftStart}:00`,
            shift_end_time: `${shiftEnd}:00`,
            late_grace_minutes: Number(lateGrace) || 15,
            early_departure_grace_minutes: 15,
            face_images_base64: faceImagesBase64.length > 0 ? faceImagesBase64 : undefined,
          }),
        });

        if (createRes.ok) {
          registeredResponse = await createRes.json();
          backendSuccess = true;
        } else if (createRes.status === 409) {
          setErrorMsg(`Employee ID "${cleanId}" already exists in the system. Please pick another ID.`);
          setIsSubmitting(false);
          return;
        } else {
          const errData = await createRes.json().catch(() => ({}));
          setErrorMsg(errData.detail || `Server error during employee creation (${createRes.status}).`);
          setIsSubmitting(false);
          return;
        }
      } catch (networkErr: any) {
        console.warn('FastAPI backend connection note:', networkErr);
        // If backend completely offline, allow local dashboard addition with warning
        backendSuccess = false;
      }

      // 3. Check if face embedding was already registered in the creation response
      let faceEnrolled = registeredResponse?.has_face_registered || false;
      let backendPhotoUrl: string | undefined = registeredResponse?.photo_url
        ? (registeredResponse.photo_url.startsWith('http') || registeredResponse.photo_url.startsWith('data:')
            ? registeredResponse.photo_url
            : `${FASTAPI_BASE_URL}${registeredResponse.photo_url}`)
        : undefined;

      // 4. Fallback: If not enrolled via JSON and photos are present, upload via multipart
      if (!faceEnrolled && photos.length > 0 && backendSuccess) {
        try {
          const formData = new FormData();
          photos.forEach((p) => {
            if (p.file) {
              formData.append('files', p.file);
            }
          });

          const faceRes = await fetch(`${FASTAPI_BASE_URL}/api/v1/employees/${cleanId}/face/upload`, {
            method: 'POST',
            body: formData,
          });

          if (faceRes.ok) {
            faceEnrolled = true;
            const faceData = await faceRes.json();
            if (faceData.photo_url) {
              backendPhotoUrl = faceData.photo_url.startsWith('http') || faceData.photo_url.startsWith('data:')
                ? faceData.photo_url
                : `${FASTAPI_BASE_URL}${faceData.photo_url}`;
            }
          } else {
            const faceErr = await faceRes.json().catch(() => ({}));
            console.warn('Face embedding enrollment response:', faceErr);
            setErrorMsg(`Employee created, but face vector generation reported: ${faceErr.detail || 'Could not extract face'}.`);
            setIsSubmitting(false);
            return;
          }
        } catch (uploadErr: any) {
          console.warn('Could not upload face embedding to backend:', uploadErr);
          setErrorMsg(`Employee created, but face portrait failed to upload: ${uploadErr.message}`);
          setIsSubmitting(false);
          return;
        }
      }

      // 5. Update local showroom state: 1st photo is kept as profile picture!
      const defaultAvatars = [
        'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&auto=format&fit=crop&q=80',
      ];
      const fallbackAvatar = defaultAvatars[Math.floor(Math.random() * defaultAvatars.length)];

      // Explicitly keep the 1st photo as their profile picture (avatarUrl)
      const primaryProfilePic = photos.length > 0 ? (backendPhotoUrl || photos[0].preview) : fallbackAvatar;

      addEmployee({
        id: cleanId,
        name: cleanName,
        role: designation.trim() || 'Floor Consultant',
        department: department,
        phone: '+91 98765 43210',
        email: `${cleanId.toLowerCase().replace(/[^a-z0-9]/g, '')}@capermintdealership.com`,
        avatarUrl: primaryProfilePic,
        facePhotosCount: registeredResponse?.face_photos_count || (faceEnrolled ? photos.length : 0),
      });

      // Synchronize context state with backend database
      try {
        await refreshEmployees();
      } catch (e) {
        console.warn('Failed refreshing employees list:', e);
      }

      // Notify parent to trigger the on-screen popup notification message
      if (onEmployeeAdded) {
        onEmployeeAdded(cleanName);
      }

      setIsSubmitting(false);
      setSuccessMsg(null);
      handleClearAllPhotos();
      stopWebcam();
      onClose();
    } catch (err: any) {
      setErrorMsg(err.message || 'An unexpected error occurred while adding the employee.');
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fadeIn">
      <div className="bg-white rounded-3xl max-w-xl w-full p-6 sm:p-7 shadow-2xl border border-slate-200/80 relative max-h-[92vh] overflow-y-auto">
        <button
          onClick={() => {
            stopWebcam();
            onClose();
          }}
          className="absolute top-5 right-5 p-2 text-slate-400 hover:text-slate-600 rounded-xl hover:bg-slate-100 transition"
          type="button"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center gap-3 mb-5">
          <div className="p-2.5 rounded-2xl bg-indigo-50 text-indigo-600 border border-indigo-100 shadow-2xs">
            <UserPlus className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-extrabold text-slate-900 text-lg font-['Outfit']">Add New Employee</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Enroll staff profile and biometric face portrait for automated CCTV attendance
            </p>
          </div>
        </div>

        {/* Success View */}
        {successMsg ? (
          <div className="py-10 text-center space-y-3">
            <div className="w-14 h-14 bg-emerald-100 text-emerald-600 rounded-2xl flex items-center justify-center mx-auto shadow-inner animate-bounce">
              <CheckCircle2 className="w-8 h-8" />
            </div>
            <h4 className="font-bold text-slate-900 text-lg font-['Outfit']">Employee Enrolled!</h4>
            <p className="text-xs text-slate-600 max-w-sm mx-auto">{successMsg}</p>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-emerald-50 text-emerald-700 rounded-full text-xs font-semibold">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Biometric Vector Indexed in pgvector</span>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            {errorMsg && (
              <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl flex items-start gap-2.5 text-xs text-rose-800">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Basic Info: ID & Name */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1">
                  Employee ID <span className="text-rose-500">*</span>
                </label>
                <input
                  type="text"
                  value={employeeId}
                  onChange={(e) => setEmployeeId(e.target.value)}
                  placeholder="e.g. EMP-106"
                  required
                  className="w-full px-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-indigo-600 focus:ring-2 focus:ring-indigo-100 outline-none font-medium transition"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1">
                  Full Name <span className="text-rose-500">*</span>
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Priya Sharma"
                  required
                  className="w-full px-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-indigo-600 focus:ring-2 focus:ring-indigo-100 outline-none font-medium transition"
                />
              </div>
            </div>

            {/* Department & Designation */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1">
                  Department
                </label>
                <select
                  value={department}
                  onChange={(e) => setDepartment(e.target.value as any)}
                  className="w-full px-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-indigo-600 focus:ring-2 focus:ring-indigo-100 outline-none font-semibold text-slate-700 transition"
                >
                  {DEPARTMENTS.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1">
                  Role / Designation
                </label>
                <input
                  type="text"
                  value={designation}
                  onChange={(e) => setDesignation(e.target.value)}
                  placeholder="e.g. Senior Sales Consultant"
                  className="w-full px-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-indigo-600 focus:ring-2 focus:ring-indigo-100 outline-none font-medium transition"
                />
              </div>
            </div>

            {/* Assigned Location */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1">
                Assigned Location / Showroom Desk
              </label>
              <input
                type="text"
                value={assignedLocation}
                onChange={(e) => setAssignedLocation(e.target.value)}
                placeholder="e.g. Main Showroom Desk 02, Accounts Counter"
                className="w-full px-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-indigo-600 focus:ring-2 focus:ring-indigo-100 outline-none font-medium transition"
              />
            </div>

            {/* Shift Timings & Grace Period */}
            <div className="p-3.5 bg-slate-50/80 rounded-2xl border border-slate-200/70 space-y-2.5">
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-700">
                <Clock className="w-3.5 h-3.5 text-indigo-600" />
                <span>Shift Schedule & Attendance Grace</span>
              </div>
              <div className="grid grid-cols-3 gap-2.5">
                <div>
                  <label className="block text-[10px] font-semibold text-slate-500 mb-0.5">Shift Start</label>
                  <input
                    type="time"
                    value={shiftStart}
                    onChange={(e) => setShiftStart(e.target.value)}
                    className="w-full px-2.5 py-1.5 text-xs bg-white border border-slate-200 rounded-lg outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-semibold text-slate-500 mb-0.5">Shift End</label>
                  <input
                    type="time"
                    value={shiftEnd}
                    onChange={(e) => setShiftEnd(e.target.value)}
                    className="w-full px-2.5 py-1.5 text-xs bg-white border border-slate-200 rounded-lg outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-semibold text-slate-500 mb-0.5">Late Grace (mins)</label>
                  <input
                    type="number"
                    min="0"
                    max="60"
                    value={lateGrace}
                    onChange={(e) => setLateGrace(e.target.value)}
                    className="w-full px-2.5 py-1.5 text-xs bg-white border border-slate-200 rounded-lg outline-none focus:border-indigo-500"
                  />
                </div>
              </div>
            </div>

            {/* Biometric Face Portrait Enrollment */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-xs font-bold uppercase tracking-wider text-slate-700">
                    Employee Photos ({photos.length})
                  </label>
                  <p className="text-[11px] text-slate-500">
                    Upload 1 or more photos. <strong className="text-indigo-600 font-semibold">1st photo is kept as their profile picture</strong>.
                  </p>
                </div>
                <span className="text-[10px] font-semibold text-indigo-600 flex items-center gap-1 bg-indigo-50 px-2 py-0.5 rounded-full border border-indigo-100">
                  <Sparkles className="w-3 h-3" />
                  ArcFace 512D
                </span>
              </div>

              {/* Webcam Live Capture View */}
              {isWebcamActive ? (
                <div className="relative rounded-2xl overflow-hidden bg-slate-900 border border-slate-700 flex flex-col items-center p-3">
                  <video ref={videoRef} autoPlay playsInline className="w-56 h-56 object-cover rounded-xl mb-3" />
                  <canvas ref={canvasRef} className="hidden" />
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={captureWebcamSnapshot}
                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-bold shadow-md flex items-center gap-1.5 transition"
                    >
                      <Camera className="w-4 h-4" />
                      Take Snapshot
                    </button>
                    <button
                      type="button"
                      onClick={stopWebcam}
                      className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-semibold transition"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  {photos.length > 0 && (
                    <div className="space-y-2.5">
                      {/* Primary Profile Picture Card (1st photo) */}
                      <div className="p-3 bg-indigo-50/70 rounded-2xl border-2 border-indigo-200/90 shadow-2xs relative">
                        <div className="flex items-center gap-3">
                          <div className="relative shrink-0">
                            <img
                              src={photos[0].preview}
                              alt="1st Photo (Profile Picture)"
                              className="w-16 h-16 rounded-xl object-cover border-2 border-indigo-500 shadow-sm"
                            />
                            <div className="absolute -top-1.5 -left-1.5 bg-indigo-600 text-white p-1 rounded-full shadow-xs" title="1st Photo - Profile Picture">
                              <Star className="w-3 h-3 fill-white" />
                            </div>
                          </div>

                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1.5 flex-wrap">
                              <span className="text-[10px] font-extrabold uppercase tracking-wide bg-indigo-600 text-white px-2 py-0.5 rounded-full shadow-2xs">
                                1st Photo • Profile Picture
                              </span>
                              <span className="text-[10px] text-emerald-700 font-semibold flex items-center gap-0.5 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-100">
                                <CheckCircle2 className="w-3 h-3 text-emerald-600" /> Active Avatar
                              </span>
                            </div>
                            <p className="text-xs font-bold text-slate-900 truncate mt-1">
                              {photos[0].name}
                            </p>
                            <p className="text-[11px] text-slate-500">
                              {photos[0].size ? `${Math.round(photos[0].size / 1024)} KB • ` : ''}Main avatar for profile, duty roster & attendance
                            </p>
                          </div>

                          <button
                            type="button"
                            onClick={() => handleRemovePhoto(0)}
                            className="p-2 text-rose-500 hover:bg-rose-100/60 rounded-xl transition shrink-0"
                            title="Remove Photo"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>

                      {/* Additional Photos List */}
                      {photos.length > 1 && (
                        <div className="space-y-1.5">
                          <div className="flex items-center justify-between text-[11px] font-bold text-slate-600 uppercase tracking-wider">
                            <span>Additional Face Angles ({photos.length - 1})</span>
                            <span className="text-[10px] text-slate-400 font-normal">All photos enhance ArcFace recognition</span>
                          </div>

                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-40 overflow-y-auto pr-1">
                            {photos.slice(1).map((photo, index) => {
                              const actualIndex = index + 1;
                              return (
                                <div
                                  key={photo.id}
                                  className="p-2 bg-slate-50 hover:bg-slate-100/80 rounded-xl border border-slate-200 flex items-center justify-between gap-2 transition"
                                >
                                  <div className="flex items-center gap-2 min-w-0">
                                    <img
                                      src={photo.preview}
                                      alt={`Sample #${actualIndex + 1}`}
                                      className="w-9 h-9 rounded-lg object-cover border border-slate-200 shrink-0"
                                    />
                                    <div className="min-w-0">
                                      <p className="text-[11px] font-bold text-slate-800 truncate">
                                        Photo #{actualIndex + 1}
                                      </p>
                                      <p className="text-[10px] text-slate-400 truncate">
                                        {photo.name}
                                      </p>
                                    </div>
                                  </div>

                                  <div className="flex items-center gap-1 shrink-0">
                                    <button
                                      type="button"
                                      onClick={() => handleSetPrimary(actualIndex)}
                                      className="px-2 py-0.5 bg-white hover:bg-indigo-50 text-indigo-600 border border-indigo-200 rounded-lg text-[10px] font-semibold transition"
                                      title="Set this photo as 1st (Profile Picture)"
                                    >
                                      Set Profile
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => handleRemovePhoto(actualIndex)}
                                      className="p-1 text-slate-400 hover:text-rose-500 rounded-lg transition"
                                      title="Remove this photo"
                                    >
                                      <Trash2 className="w-3.5 h-3.5" />
                                    </button>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Upload / Snap Controls */}
                  <div className="grid grid-cols-2 gap-2.5">
                    <label className="border-2 border-dashed border-slate-200 hover:border-indigo-400 bg-slate-50 hover:bg-indigo-50/30 rounded-2xl p-3 text-center cursor-pointer transition flex flex-col items-center justify-center">
                      <UploadCloud className="w-5 h-5 text-indigo-500 mb-1" />
                      <span className="text-xs font-bold text-slate-700">
                        {photos.length > 0 ? '+ Add More Photos' : 'Upload Photos'}
                      </span>
                      <span className="text-[10px] text-slate-400 mt-0.5">Select 1 or more (JPG/PNG)</span>
                      <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept="image/jpeg,image/png,image/webp"
                        onChange={handleFilesChange}
                        className="hidden"
                      />
                    </label>

                    <button
                      type="button"
                      onClick={startWebcam}
                      className="border-2 border-dashed border-slate-200 hover:border-indigo-400 bg-slate-50 hover:bg-indigo-50/30 rounded-2xl p-3 text-center cursor-pointer transition flex flex-col items-center justify-center"
                    >
                      <Camera className="w-5 h-5 text-indigo-500 mb-1" />
                      <span className="text-xs font-bold text-slate-700">
                        {photos.length > 0 ? '+ Snap Another' : 'Use Webcam'}
                      </span>
                      <span className="text-[10px] text-slate-400 mt-0.5">Take snapshot live</span>
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex items-center gap-3 pt-3 border-t border-slate-100">
              <button
                type="button"
                onClick={() => {
                  stopWebcam();
                  onClose();
                }}
                disabled={isSubmitting}
                className="flex-1 py-2.5 border border-slate-200 hover:bg-slate-50 text-slate-700 font-semibold rounded-xl text-xs transition"
              >
                Cancel
              </button>

              <button
                type="submit"
                disabled={isSubmitting}
                className="flex-1 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white font-bold rounded-xl text-xs shadow-md shadow-indigo-600/25 flex items-center justify-center gap-2 transition"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Enrolling Face Vector...</span>
                  </>
                ) : (
                  <>
                    <UserPlus className="w-4 h-4" />
                    <span>Save & Enroll Employee</span>
                  </>
                )}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

