import React, { useState, useRef, useEffect } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import { FASTAPI_BASE_URL } from '../../config/env';
import {
  X,
  Camera,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Trash2,
  Sparkles,
  ShieldCheck,
  UserCheck,
  Plus,
  Image as ImageIcon,
  Check,
} from 'lucide-react';

interface AddEmployeePhotosModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialEmployeeId?: string | null;
  onPhotosAdded?: (employeeName: string, count: number) => void;
}

interface NewPhoto {
  id: string;
  file?: File;
  preview: string;
  name: string;
  size?: number;
}

// Client-side image optimizer to keep memory and transmission lightweight
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

export const AddEmployeePhotosModal: React.FC<AddEmployeePhotosModalProps> = ({
  isOpen,
  onClose,
  initialEmployeeId,
  onPhotosAdded,
}) => {
  const { employees, addEmployeePhotos, refreshEmployees } = useShowroom();

  const [selectedEmployeeId, setSelectedEmployeeId] = useState<string>('');
  const [photos, setPhotos] = useState<NewPhoto[]>([]);
  const [setAsPrimaryProfile, setSetAsPrimaryProfile] = useState(false);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Sync selected employee when modal opens or initialEmployeeId changes
  useEffect(() => {
    if (isOpen) {
      setErrorMsg(null);
      setSuccessMsg(null);
      setPhotos([]);
      setSetAsPrimaryProfile(false);

      if (initialEmployeeId && employees.some((e) => e.id === initialEmployeeId)) {
        setSelectedEmployeeId(initialEmployeeId);
      } else if (employees.length > 0) {
        setSelectedEmployeeId(employees[0].id);
      }
    } else {
      stopWebcam();
    }
  }, [isOpen, initialEmployeeId, employees]);

  const selectedEmployee = employees.find((e) => e.id === selectedEmployeeId);

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
      setErrorMsg('Could not access webcam. Please upload image files instead.');
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

      canvas.toBlob((blob) => {
        if (blob) {
          const photoNum = photos.length + 1;
          const file = new File([blob], `${selectedEmployeeId || 'emp'}-angle-${photoNum}.jpg`, { type: 'image/jpeg' });
          setPhotos((prev) => [
            ...prev,
            {
              id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
              file,
              preview: dataUrl,
              name: `Webcam Angle #${photoNum}`,
              size: blob.size,
            },
          ]);
        }
      }, 'image/jpeg', 0.92);
    }
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

  const handleClearAllPhotos = () => {
    setPhotos([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setSuccessMsg(null);

    if (!selectedEmployeeId) {
      setErrorMsg('Please select an employee.');
      return;
    }

    if (photos.length === 0) {
      setErrorMsg('Please upload or snap at least one photo to add.');
      return;
    }

    setIsSubmitting(true);

    try {
      const formData = new FormData();
      photos.forEach((p) => {
        if (p.file) {
          formData.append('files', p.file);
        }
      });

      const endpoint = `${FASTAPI_BASE_URL}/api/v1/employees/${selectedEmployeeId}/photos?set_as_profile=${setAsPrimaryProfile}`;
      const res = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned error ${res.status}`);
      }

      const updatedData = await res.json();
      const newAvatar = setAsPrimaryProfile && updatedData.photo_url
        ? (updatedData.photo_url.startsWith('http') || updatedData.photo_url.startsWith('data:') ? updatedData.photo_url : `${FASTAPI_BASE_URL}${updatedData.photo_url}`)
        : undefined;

      // Update local state in ShowroomContext
      addEmployeePhotos(selectedEmployeeId, photos.length, newAvatar);
      await refreshEmployees();

      const empName = selectedEmployee?.name || selectedEmployeeId;
      setSuccessMsg(`Successfully added ${photos.length} photo(s) to ${empName}'s biometric gallery!`);

      if (onPhotosAdded) {
        onPhotosAdded(empName, photos.length);
      }

      // Close modal after brief success feedback
      setTimeout(() => {
        stopWebcam();
        onClose();
      }, 1500);
    } catch (err: any) {
      console.warn('Could not add photos to backend:', err);
      // Fallback: allow updating local state if backend offline or mocked
      if (err.message && err.message.includes('Failed to fetch')) {
        const empName = selectedEmployee?.name || selectedEmployeeId;
        addEmployeePhotos(selectedEmployeeId, photos.length, setAsPrimaryProfile ? photos[0].preview : undefined);
        setSuccessMsg(`[Offline Mode] Added ${photos.length} photo(s) to ${empName}.`);
        if (onPhotosAdded) {
          onPhotosAdded(empName, photos.length);
        }
        setTimeout(() => {
          stopWebcam();
          onClose();
        }, 1500);
      } else {
        setErrorMsg(err.message || 'Failed to enroll additional photos. Make sure faces are clear and well lit.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-slate-900/60 backdrop-blur-xs overflow-y-auto animate-fade-in">
      <div className="relative w-full max-w-2xl bg-white rounded-3xl shadow-2xl border border-slate-200 overflow-hidden my-6">
        {/* Header */}
        <div className="px-6 py-5 bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-indigo-500/20 border border-indigo-400/30 flex items-center justify-center text-indigo-400">
              <Camera className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold font-['Outfit'] flex items-center gap-2">
                Add Employee Face Photos
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/30 text-indigo-200 border border-indigo-400/30">
                  Biometric Gallery
                </span>
              </h2>
              <p className="text-xs text-slate-300">
                Enhance CCTV face recognition accuracy by training additional angles and lighting conditions
              </p>
            </div>
          </div>
          <button
            onClick={() => {
              stopWebcam();
              onClose();
            }}
            className="p-2 text-slate-400 hover:text-white rounded-xl hover:bg-white/10 transition"
            title="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          {/* Messages */}
          {errorMsg && (
            <div className="p-3.5 bg-rose-50 border border-rose-200 text-rose-800 rounded-2xl text-xs flex items-start gap-2.5 animate-shake">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-600" />
              <div className="flex-1">{errorMsg}</div>
            </div>
          )}

          {successMsg && (
            <div className="p-3.5 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-2xl text-xs flex items-center gap-2.5">
              <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-600" />
              <div className="font-semibold">{successMsg}</div>
            </div>
          )}

          {/* Select Registered Employee */}
          <div>
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Select Showroom Employee
            </label>
            <select
              value={selectedEmployeeId}
              onChange={(e) => {
                setSelectedEmployeeId(e.target.value);
                setErrorMsg(null);
              }}
              className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs font-medium text-slate-800 focus:outline-hidden focus:ring-2 focus:ring-indigo-500 focus:bg-white transition"
            >
              {employees.map((emp) => (
                <option key={emp.id} value={emp.id}>
                  {emp.name} ({emp.id.toUpperCase()}) — {emp.role} • {emp.department}
                </option>
              ))}
            </select>
          </div>

          {/* Selected Employee Info Banner */}
          {selectedEmployee && (
            <div className="p-3.5 bg-slate-50 rounded-2xl border border-slate-200/80 flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <img
                  src={selectedEmployee.avatarUrl || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80'}
                  alt={selectedEmployee.name}
                  className="w-11 h-11 rounded-full object-cover border-2 border-indigo-200 shadow-2xs"
                />
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-slate-900 text-sm font-['Outfit']">
                      {selectedEmployee.name}
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-200/80 text-slate-700 rounded-md font-bold">
                      {selectedEmployee.id.toUpperCase()}
                    </span>
                  </div>
                  <span className="text-xs text-slate-500 block">
                    {selectedEmployee.role} • {selectedEmployee.department}
                  </span>
                </div>
              </div>

              <div className="text-right">
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">
                  <ShieldCheck className="w-3.5 h-3.5 text-indigo-600" />
                  {selectedEmployee.facePhotosCount && selectedEmployee.facePhotosCount > 0
                    ? `${selectedEmployee.facePhotosCount} Angle(s) Active`
                    : '1 Angle Enrolled'}
                </span>
                <p className="text-[10px] text-slate-400 mt-0.5">Multi-angle gallery active</p>
              </div>
            </div>
          )}

          {/* Capture / Upload Options */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                <ImageIcon className="w-4 h-4 text-indigo-600" />
                Capture or Upload New Photos
              </label>

              <div className="flex items-center gap-2">
                {!isWebcamActive ? (
                  <button
                    type="button"
                    onClick={startWebcam}
                    className="px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition active:scale-95"
                  >
                    <Camera className="w-3.5 h-3.5 text-indigo-600" />
                    <span>Open Webcam</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={stopWebcam}
                    className="px-3 py-1.5 bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition active:scale-95"
                  >
                    <X className="w-3.5 h-3.5 text-rose-600" />
                    <span>Stop Camera</span>
                  </button>
                )}

                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition active:scale-95"
                >
                  <UploadCloud className="w-3.5 h-3.5 text-slate-600" />
                  <span>Choose Files</span>
                </button>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFilesChange}
                  multiple
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                />
              </div>
            </div>

            {/* Webcam Live View */}
            {isWebcamActive && (
              <div className="relative rounded-2xl overflow-hidden border-2 border-indigo-500 bg-slate-950 p-2 text-center animate-fade-in shadow-inner">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-56 sm:h-64 object-cover rounded-xl mx-auto shadow-md transform -scale-x-100"
                />
                <canvas ref={canvasRef} className="hidden" />

                {/* Face positioning crosshair guide */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-36 h-48 border-2 border-dashed border-indigo-400/70 rounded-full flex items-center justify-center">
                    <span className="text-[10px] text-white/70 font-semibold bg-black/40 px-2 py-0.5 rounded-full">
                      Align Face
                    </span>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-center gap-3">
                  <button
                    type="button"
                    onClick={captureWebcamSnapshot}
                    className="px-5 py-2 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 text-white rounded-xl text-xs font-bold shadow-md flex items-center gap-2 transition active:scale-95"
                  >
                    <Camera className="w-4 h-4" />
                    <span>Snap Photo</span>
                  </button>
                  <span className="text-[11px] text-slate-400">
                    Snap multiple angles: Front, Turn 30° Left, Turn 30° Right
                  </span>
                </div>
              </div>
            )}

            {/* Drag & Drop Area if no webcam and no photos */}
            {!isWebcamActive && photos.length === 0 && (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-200 hover:border-indigo-400 rounded-2xl p-6 text-center cursor-pointer transition bg-slate-50/50 hover:bg-indigo-50/20 group"
              >
                <div className="w-12 h-12 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 mx-auto mb-2 group-hover:scale-110 transition">
                  <UploadCloud className="w-6 h-6" />
                </div>
                <p className="text-xs font-bold text-slate-800">
                  Click to select photos or drag & drop files here
                </p>
                <p className="text-[11px] text-slate-400 mt-1">
                  Upload 1 or more portrait photos (JPG, PNG). Optimal angles: Front, Left 30°, Right 30°.
                </p>
              </div>
            )}

            {/* Preview of Newly Added Photos */}
            {photos.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-500">
                  <span className="font-bold text-slate-700">
                    {photos.length} Photo{photos.length > 1 ? 's' : ''} Ready to Add
                  </span>
                  <button
                    type="button"
                    onClick={handleClearAllPhotos}
                    className="text-rose-600 hover:text-rose-700 text-[11px] font-semibold hover:underline"
                  >
                    Clear All
                  </button>
                </div>

                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2.5 max-h-48 overflow-y-auto p-1.5 bg-slate-50 rounded-2xl border border-slate-200">
                  {photos.map((photo, idx) => (
                    <div
                      key={photo.id}
                      className="relative group rounded-xl overflow-hidden border border-slate-200 bg-white aspect-square shadow-2xs"
                    >
                      <img
                        src={photo.preview}
                        alt={photo.name}
                        className="w-full h-full object-cover"
                      />
                      <span className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded-md text-[9px] font-bold bg-black/60 text-white font-mono">
                        #{idx + 1}
                      </span>
                      <button
                        type="button"
                        onClick={() => handleRemovePhoto(idx)}
                        className="absolute top-1 right-1 p-1 bg-rose-600 text-white rounded-md opacity-0 group-hover:opacity-100 transition shadow-xs hover:bg-rose-700"
                        title="Remove photo"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  ))}

                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-slate-300 hover:border-indigo-500 rounded-xl flex flex-col items-center justify-center p-2 text-slate-400 hover:text-indigo-600 transition aspect-square bg-white"
                    title="Add more photos"
                  >
                    <Plus className="w-5 h-5 mb-0.5" />
                    <span className="text-[10px] font-bold">Add More</span>
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Options: Set as Primary Profile Picture */}
          <div className="pt-2 border-t border-slate-100 flex items-center justify-between">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={setAsPrimaryProfile}
                onChange={(e) => setSetAsPrimaryProfile(e.target.checked)}
                className="w-4 h-4 rounded-md text-indigo-600 focus:ring-indigo-500 border-slate-300"
              />
              <span className="text-xs font-semibold text-slate-700">
                Set first newly added photo as primary profile avatar
              </span>
            </label>
          </div>

          {/* Best Practices Tip Banner */}
          <div className="p-3 bg-indigo-50/70 border border-indigo-100 rounded-2xl flex items-start gap-2.5 text-indigo-900 text-xs">
            <Sparkles className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
            <p className="text-[11px] leading-relaxed text-indigo-800">
              <span className="font-bold">Biometric Tip:</span> For maximum CCTV recognition fidelity, capture one frontal photo, one looking slightly to the left, and one to the right under natural showroom lighting.
            </p>
          </div>

          {/* Footer Actions */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-100">
            <button
              type="button"
              onClick={() => {
                stopWebcam();
                onClose();
              }}
              disabled={isSubmitting}
              className="px-4 py-2.5 rounded-xl text-xs font-bold text-slate-600 hover:bg-slate-100 transition disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || photos.length === 0}
              className="px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 text-white rounded-xl text-xs font-bold shadow-md shadow-indigo-600/30 flex items-center gap-2 transition active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-white" />
                  <span>Enrolling Biometrics...</span>
                </>
              ) : (
                <>
                  <UserCheck className="w-4 h-4" />
                  <span>Add {photos.length > 0 ? `${photos.length} Photo${photos.length > 1 ? 's' : ''}` : 'Photos'} to Gallery</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

