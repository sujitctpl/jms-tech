import React, { useState } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import { useAuth } from '../../context/AuthContext';
import { Clock, CheckCircle2, Save, Info, ShieldCheck } from 'lucide-react';

export const SettingsView: React.FC = () => {
  const { user } = useAuth();
  const { storeHours, updateStoreHours, isStoreCurrentlyOpen } = useShowroom();

  // Store Hours Form State
  const [openingTime, setOpeningTime] = useState(storeHours.openingTime);
  const [closingTime, setClosingTime] = useState(storeHours.closingTime);
  const [timezone, setTimezone] = useState(storeHours.timezone);
  const [isOpenSunday, setIsOpenSunday] = useState(storeHours.isOpenSunday);
  const [hoursNotes, setHoursNotes] = useState(storeHours.notes);

  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);

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
    <div className="space-y-6 pb-12 max-w-4xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-['Outfit'] text-slate-900">Showroom Operating Hours</h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-blue-100 text-blue-800">
              Capermint Intelligence
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Configure showroom opening and closing schedule, timezone, and operating status.
          </p>
        </div>
      </div>

      {saveSuccess && (
        <div className="p-3 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-xl text-xs flex items-center gap-2 animate-fadeIn">
          <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          <span>{saveSuccess}</span>
        </div>
      )}

      {/* Main Settings Card */}
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
            className="w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-sm font-bold border border-blue-400/30 shadow-md shadow-blue-500/20 flex items-center justify-center gap-2 transition active:scale-[0.98]"
          >
            <Save className="w-4 h-4" />
            <span>Save Operating Hours</span>
          </button>
        </form>

        <div className="mt-6 pt-4 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-500 gap-2 font-medium">
          <span>First visitor entry today: <strong className="text-slate-800 font-mono">{storeHours.actualFirstEntryToday || '09:34 AM'}</strong></span>
          <span>Latest exit recorded: <strong className="text-slate-800 font-mono">{storeHours.actualLastExitToday || 'Active'}</strong></span>
        </div>
      </div>
    </div>
  );
};
