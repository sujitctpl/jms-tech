import React, { useState, useEffect } from 'react';
import { VisitorRecord, VisitPurpose } from '../../types';
import { useShowroom } from '../../context/ShowroomContext';
import { X, Tag, UserCheck, FileText, CheckCircle2 } from 'lucide-react';
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
  const [selectedSalesperson, setSelectedSalesperson] = useState<string>('Rajesh Kumar');
  const [notes, setNotes] = useState<string>('');
  const [isSaved, setIsSaved] = useState(false);

  useEffect(() => {
    if (visitor) {
      setSelectedPurpose(visitor.visitPurpose || 'Vehicle Enquiry');
      setSelectedSalesperson(visitor.assignedSalesperson || employees[0]?.name || 'Rajesh Kumar');
      setNotes(visitor.notes || '');
      setIsSaved(false);
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

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fadeIn">
      <div className="bg-white rounded-3xl max-w-lg w-full p-6 sm:p-7 shadow-2xl border border-slate-200/80 relative">
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2 text-slate-400 hover:text-slate-600 rounded-xl hover:bg-slate-100 transition"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 mb-5">
          <div className="p-2.5 rounded-2xl bg-blue-50 text-blue-600 border border-blue-100 shadow-2xs">
            <Tag className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-extrabold text-slate-900 text-lg font-['Outfit']">Tag Showroom Visit Purpose</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Visitor ID: <span className="font-mono font-bold text-blue-700">{visitor.visitorId}</span> ({visitor.firstSeenTime})
            </p>
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
            <div className="p-3 bg-slate-50/80 rounded-2xl border border-slate-200/80 flex flex-wrap gap-3 text-xs text-slate-600">
              <div>
                <span className="text-slate-400">Type:</span>{' '}
                <span className="font-bold text-slate-800">{visitor.visitorType} Visitor</span>
              </div>
              <div>
                <span className="text-slate-400">Zone:</span>{' '}
                <span className="font-bold text-slate-800">{visitor.zoneEntered || 'Main Showroom'}</span>
              </div>
              <div>
                <span className="text-slate-400">Duration:</span>{' '}
                <span className="font-bold text-slate-800">{visitor.durationMinutes} mins</span>
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
