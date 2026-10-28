import React, { useState } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import { VisitPurpose, VisitorType } from '../../types';
import { X, UserPlus, Sparkles, CheckCircle2 } from 'lucide-react';
import { PURPOSE_COLORS } from '../../data/seedData';

interface QuickAddVisitorModalProps {
  isOpen: boolean;
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

export const QuickAddVisitorModal: React.FC<QuickAddVisitorModalProps> = ({ isOpen, onClose }) => {
  const { addNewVisitor, employees } = useShowroom();

  const [visitorType, setVisitorType] = useState<VisitorType>('New');
  const [purpose, setPurpose] = useState<VisitPurpose | ''>('Vehicle Enquiry');
  const [salesperson, setSalesperson] = useState<string>('Rajesh Kumar');
  const [notes, setNotes] = useState<string>('');
  const [zone, setZone] = useState<'Main Showroom' | 'Display Area' | 'Test Track' | 'Customer Lounge' | 'Service Bay'>(
    'Main Showroom'
  );
  const [success, setSuccess] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    addNewVisitor({
      visitorType,
      visitPurpose: purpose ? (purpose as VisitPurpose) : null,
      assignedSalesperson: purpose ? salesperson : null,
      notes: notes || 'Direct walk-in customer logged by front desk',
      zoneEntered: zone,
    });

    setSuccess(true);
    setTimeout(() => {
      setSuccess(false);
      onClose();
    }, 600);
  };

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fadeIn">
      <div className="bg-white rounded-3xl max-w-md w-full p-6 sm:p-7 shadow-2xl border border-slate-200/80 relative">
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2 text-slate-400 hover:text-slate-600 rounded-xl hover:bg-slate-100 transition"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 mb-5">
          <div className="p-2.5 rounded-2xl bg-blue-50 text-blue-600 border border-blue-100 shadow-2xs">
            <UserPlus className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-extrabold text-slate-900 text-lg font-['Outfit']">Log Showroom Walk-in</h3>
            <p className="text-xs text-slate-500 mt-0.5">Record a new visitor entry or trigger AI camera simulation</p>
          </div>
        </div>

        {success ? (
          <div className="py-8 text-center">
            <CheckCircle2 className="w-12 h-12 text-emerald-500 mx-auto animate-bounce mb-2" />
            <h4 className="font-bold text-slate-900 text-base">Visitor Recorded!</h4>
            <p className="text-xs text-slate-500">Dashboard counters and analytics updated in real time.</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* New vs Returning */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                Visitor Classification
              </label>
              <div className="grid grid-cols-2 gap-2.5">
                <button
                  type="button"
                  onClick={() => setVisitorType('New')}
                  className={`py-2.5 px-3 rounded-xl text-xs font-bold border transition active:scale-95 ${
                    visitorType === 'New'
                      ? 'border-blue-600 bg-blue-50/90 text-blue-900 shadow-xs ring-2 ring-blue-500/30'
                      : 'border-slate-200/80 bg-slate-50/70 text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  New Visitor (First Time)
                </button>
                <button
                  type="button"
                  onClick={() => setVisitorType('Returning')}
                  className={`py-2.5 px-3 rounded-xl text-xs font-bold border transition active:scale-95 ${
                    visitorType === 'Returning'
                      ? 'border-blue-600 bg-blue-50/90 text-blue-900 shadow-xs ring-2 ring-blue-500/30'
                      : 'border-slate-200/80 bg-slate-50/70 text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  Returning Customer
                </button>
              </div>
            </div>

            {/* Purpose */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                Visit Purpose (Optional)
              </label>
              <select
                value={purpose}
                onChange={(e) => setPurpose(e.target.value as VisitPurpose)}
                className="w-full px-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition font-semibold text-slate-800"
              >
                <option value="">Leave Untagged (Tag Later)</option>
                {PURPOSES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>

            {/* Salesperson */}
            {purpose && (
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                  Assign Salesperson
                </label>
                <select
                  value={salesperson}
                  onChange={(e) => setSalesperson(e.target.value)}
                  className="w-full px-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition font-semibold text-slate-800"
                >
                  {employees.map((e) => (
                    <option key={e.id} value={e.name}>
                      {e.name} ({e.role})
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Zone */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                Initial Zone Entered
              </label>
              <select
                value={zone}
                onChange={(e) => setZone(e.target.value as any)}
                className="w-full px-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition font-semibold text-slate-800"
              >
                <option value="Main Showroom">Main Showroom</option>
                <option value="Display Area">Scooter & Bike Display Floor</option>
                <option value="Test Track">Test Ride Track / Staging Area</option>
                <option value="Customer Lounge">Customer Finance & Delivery Lounge</option>
                <option value="Service Bay">Service Reception & Workshop</option>
              </select>
            </div>

            {/* Notes */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1.5">
                Visitor Notes
              </label>
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="e.g. Inquired about Activa 6G anniversary edition"
                className="w-full px-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition font-medium text-slate-800"
              />
            </div>

            {/* Buttons */}
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
                className="flex-1 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold rounded-xl text-xs border border-blue-400/30 shadow-md shadow-blue-500/20 active:scale-95 transition"
              >
                Confirm Entry
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
