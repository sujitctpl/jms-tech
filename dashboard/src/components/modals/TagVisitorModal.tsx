import React, { useState, useEffect } from 'react';
import { VisitorRecord, VisitPurpose } from '../../types';
import { useShowroom } from '../../context/ShowroomContext';
import { X, Tag, UserCheck, FileText, CheckCircle2, Eye, ZoomIn, User } from 'lucide-react';
import { PURPOSE_COLORS } from '../../data/seedData';

interface TagVisitorModalProps {
  visitor: VisitorRecord | null;
  onClose: () => void;
}

const PURPOSES: VisitPurpose[] = [
  'Test Ride',
  'Vehicle Enquiry',
  'Booking',
  'Service',
  'Vehicle Delivery',
  'Spare Parts',
  'General Enquiry',
  'Other',
];

export const TagVisitorModal: React.FC<TagVisitorModalProps> = ({ visitor, onClose }) => {
  const { tagVisitor, employees } = useShowroom();

  const [selectedPurpose, setSelectedPurpose] = useState<VisitPurpose>('Vehicle Enquiry');
  const [selectedSalesperson, setSelectedSalesperson] = useState<string>('');
  const [notes, setNotes] = useState<string>('');
  const [isSaved, setIsSaved] = useState(false);
  const [isZoomed, setIsZoomed] = useState(false);

  const formatDate = (dateStr?: string) => {
    if (!dateStr) {
      return new Date().toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' });
    }
    try {
      const parts = dateStr.split('-');
      if (parts.length === 3) {
        const dateObj = new Date(parseInt(parts[0], 10), parseInt(parts[1], 10) - 1, parseInt(parts[2], 10));
        return dateObj.toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' });
      }
    } catch (_) {}
    return dateStr;
  };

  const zoneEntries: [string, string][] = visitor
    ? visitor.zoneDwell && Object.keys(visitor.zoneDwell).length > 0
      ? Object.entries(visitor.zoneDwell)
      : [[visitor.zoneEntered || 'Main Showroom', `${visitor.durationMinutes || 1} mins`]]
    : [];

  const departureTime = visitor?.exitTime || (
    visitor?.lastSeenTime && visitor.lastSeenTime !== visitor.firstSeenTime
      ? visitor.lastSeenTime
      : null
  );

  useEffect(() => {
    if (visitor) {
      setSelectedPurpose(visitor.visitPurpose || 'Vehicle Enquiry');
      setSelectedSalesperson(visitor.assignedSalesperson || employees[0]?.name || '');

      const currentZoneEntries = visitor.zoneDwell && Object.keys(visitor.zoneDwell).length > 0
        ? Object.entries(visitor.zoneDwell)
        : [[visitor.zoneEntered || 'Main Showroom', `${visitor.durationMinutes || 1} mins`]];
      const zSummary = currentZoneEntries.map(([z, t]) => `${z} (${t})`).join(', ');

      let initNotes = visitor.notes || '';
      if (!initNotes || initNotes.includes('via CCTV AI Detection Line') || initNotes.includes('AI Face Detection')) {
        initNotes = `Walk-in visitor tracked via CCTV AI Detection Line. Time spent in zone: ${zSummary}.`;
      }
      setNotes(initNotes);
      setIsSaved(false);
      setIsZoomed(false);
    }
  }, [visitor, employees]);

  if (!visitor) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    tagVisitor(visitor.id, selectedPurpose, selectedSalesperson, notes);
    setIsSaved(true);
    setTimeout(() => {
      onClose();
    }, 600);
  };

  const appearanceText = visitor.notes && visitor.notes.includes('Appearance:')
    ? visitor.notes.split('|')[0].replace('Appearance:', '').trim()
    : null;

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fadeIn">
      {/* High-res Image Zoom Lightbox */}
      {isZoomed && (
        <div 
          onClick={() => setIsZoomed(false)}
          className="fixed inset-0 bg-black/85 z-60 flex flex-col items-center justify-center p-4 cursor-zoom-out animate-fadeIn"
        >
          <div className="relative max-w-sm w-full bg-slate-900 rounded-3xl p-3 border border-slate-700 shadow-2xl">
            <button
              onClick={() => setIsZoomed(false)}
              className="absolute top-4 right-4 p-2 bg-black/60 hover:bg-black text-white rounded-full transition"
            >
              <X className="w-5 h-5" />
            </button>
            {visitor.avatarUrl ? (
              <img
                src={visitor.avatarUrl}
                alt={visitor.visitorId}
                className="w-full max-h-[75vh] object-contain rounded-2xl bg-black"
              />
            ) : (
              <div className="w-full h-80 flex flex-col items-center justify-center bg-slate-950 rounded-2xl text-slate-400">
                <User className="w-20 h-20 text-slate-600 mb-2" />
                <span className="text-xs">No body photo available</span>
              </div>
            )}
            <div className="mt-3 text-center text-white">
              <span className="font-mono font-bold text-base">{visitor.visitorId}</span>
              <p className="text-xs text-slate-400 mt-0.5">
                {visitor.cameraSource || 'Entrance Camera'} • {visitor.firstSeenTime}
              </p>
              {appearanceText && (
                <span className="mt-2 inline-block px-2.5 py-0.5 rounded-full text-xs font-bold bg-purple-900/80 text-purple-200 border border-purple-700">
                  👔 {appearanceText}
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-3xl max-w-lg w-full p-6 sm:p-7 shadow-2xl border border-slate-200/80 relative">
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2 text-slate-400 hover:text-slate-600 rounded-xl hover:bg-slate-100 transition"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Visitor Card Header with Crisp Portrait */}
        <div className="flex items-start gap-4 mb-5 p-3.5 bg-slate-50/80 rounded-2xl border border-slate-200/80">
          <div 
            onClick={() => visitor.avatarUrl && setIsZoomed(true)}
            title={visitor.avatarUrl ? "Click to zoom photo" : "No photo"}
            className={`relative group shrink-0 w-14 h-18 rounded-2xl overflow-hidden border border-slate-700/80 bg-slate-950 shadow-sm ${
              visitor.avatarUrl ? 'cursor-pointer' : 'cursor-default'
            }`}
          >
            {visitor.avatarUrl ? (
              <>
                <img
                  src={visitor.avatarUrl}
                  alt={visitor.visitorId}
                  className="w-full h-full object-cover object-top contrast-[1.05] brightness-95 group-hover:scale-110 transition duration-300"
                />
                <div className="absolute inset-0 bg-black/25 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
                  <ZoomIn className="w-4 h-4 text-white drop-shadow" />
                </div>
              </>
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-slate-800 text-slate-400">
                <User className="w-7 h-7 text-slate-400" />
              </div>
            )}
            <div className="absolute top-1 left-1 bg-black/80 px-1 py-0.2 rounded text-[7px] font-mono font-bold text-emerald-400 leading-tight">
              {visitor.cameraSource ? visitor.cameraSource.slice(0, 7) : 'CAM-01'}
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="font-extrabold text-slate-900 text-base font-['Outfit']">Tag Showroom Visitor</h3>
              <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-blue-100 text-blue-800">
                {visitor.visitorId}
              </span>
            </div>

            {/* Arrived, Departed, and Arrival Date line */}
            <div className="text-xs text-slate-500 mt-1 flex items-center gap-1.5 flex-wrap">
              <span>Date: <strong className="text-slate-800 font-semibold">{formatDate(visitor.date)}</strong></span>
              <span>•</span>
              <span>Arrived: <strong className="text-slate-800 font-semibold">{visitor.firstSeenTime}</strong></span>
              <span>•</span>
              <span>
                Departed:{' '}
                {departureTime ? (
                  <strong className="text-slate-800 font-semibold">{departureTime}</strong>
                ) : (
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                    In Showroom
                  </span>
                )}
              </span>
            </div>

            {/* Zone and Time Spent in Zone */}
            <div className="mt-1.5 flex items-center gap-1.5 flex-wrap">
              <span className="text-[11px] font-semibold text-slate-500">Zone Time:</span>
              {zoneEntries.map(([zName, zDuration]) => (
                <span
                  key={zName}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-amber-50 text-amber-900 border border-amber-200 shadow-2xs"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500"></span>
                  <span>{zName}:</span>
                  <span className="text-amber-700">{zDuration}</span>
                </span>
              ))}
            </div>

            {appearanceText && (
              <div className="mt-1.5 inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-bold bg-purple-50 text-purple-700 border border-purple-200">
                <span>👔</span>
                <span>{appearanceText}</span>
              </div>
            )}
          </div>
        </div>

        {isSaved ? (
          <div className="py-8 text-center">
            <CheckCircle2 className="w-12 h-12 text-emerald-500 mx-auto animate-bounce mb-2" />
            <h4 className="font-bold text-slate-900 text-base">Visit Purpose Assigned!</h4>
            <p className="text-xs text-slate-500">Visitor record updated and synced with showroom CRM.</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Quick Purpose Selection Chips */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-2">
                Select Visit Purpose
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {PURPOSES.map((purpose) => {
                  const isSelected = selectedPurpose === purpose;
                  const color = PURPOSE_COLORS[purpose];
                  return (
                    <button
                      key={purpose}
                      type="button"
                      onClick={() => setSelectedPurpose(purpose)}
                      className={`p-2.5 rounded-xl text-xs font-bold border text-center transition flex flex-col items-center justify-center gap-1.5 active:scale-95 ${
                        isSelected
                          ? 'border-blue-600 bg-blue-50/90 text-blue-900 shadow-xs ring-2 ring-blue-500/30'
                          : 'border-slate-200/80 bg-slate-50/70 text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: color }}
                      />
                      <span className="truncate">{purpose}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Assigned Salesperson */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                Assign Sales / Floor Consultant
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <UserCheck className="w-4 h-4" />
                </div>
                <select
                  value={selectedSalesperson}
                  onChange={(e) => setSelectedSalesperson(e.target.value)}
                  className="w-full pl-10 pr-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition font-semibold text-slate-800"
                >
                  {employees.length === 0 && <option value="">No registered staff available</option>}
                  {employees.map((emp) => (
                    <option key={emp.id} value={emp.name}>
                      {emp.name} — {emp.role} ({emp.department})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Visit Notes */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                Customer Notes / Model Inquired
              </label>
              <div className="relative">
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={3}
                  placeholder="e.g. Interested in Honda Activa 125 Disc. Requested loan quote with ₹20,000 down-payment."
                  className="w-full p-3 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition resize-none font-medium text-slate-800"
                />
              </div>
            </div>

            {/* Metadata Badges */}
            <div className="p-3 bg-slate-50/80 rounded-2xl border border-slate-200/80 text-xs text-slate-600 space-y-2">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-bold">Type</span>
                  <span className="font-bold text-slate-800">{visitor.visitorType} Visitor</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-bold">Date</span>
                  <span className="font-bold text-slate-800">{formatDate(visitor.date)}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-bold">Departure</span>
                  <span className={`font-bold ${departureTime ? 'text-slate-800' : 'text-emerald-700'}`}>
                    {departureTime || 'In Showroom'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-bold">Total Duration</span>
                  <span className="font-bold text-slate-800">{visitor.durationMinutes || 1} mins</span>
                </div>
              </div>
              <div className="pt-2 border-t border-slate-200/70 flex items-center gap-1.5 flex-wrap">
                <span className="text-slate-400 text-[10px] uppercase font-bold">Zone Spent:</span>
                {zoneEntries.map(([zName, zDuration]) => (
                  <span
                    key={zName}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-blue-50 text-blue-800 border border-blue-200"
                  >
                    <span>📍 {zName}:</span>
                    <span className="text-blue-600 font-semibold">{zDuration}</span>
                  </span>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2.5 pt-2">
              <button
                type="button"
                onClick={onClose}
                className="flex-1 py-2.5 bg-white hover:bg-slate-50 text-slate-700 font-bold rounded-xl text-xs border border-slate-300 transition shadow-2xs active:scale-95"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="flex-1 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold rounded-xl text-xs border border-blue-400/30 shadow-md shadow-blue-500/20 transition active:scale-95"
              >
                Save & Assign Purpose
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
