import React, { useState, useMemo } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import { VisitorRecord, VisitPurpose, VisitorType, TaggingStatus } from '../../types';
import {
  Users,
  Search,
  Filter,
  FileSpreadsheet,
  PlusCircle,
  Tag,
  LogOut,
  Clock,
  Sparkles,
  ArrowUpDown,
  CheckCircle,
  Camera,
  X,
} from 'lucide-react';
import { PURPOSE_COLORS, INITIAL_EMPLOYEES } from '../../data/seedData';

interface VisitorsViewProps {
  onTagVisitor: (visitor: VisitorRecord) => void;
  onOpenQuickAdd: () => void;
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

export const VisitorsView: React.FC<VisitorsViewProps> = ({ onTagVisitor, onOpenQuickAdd }) => {
  const { visitors, checkoutVisitor, exportVisitorsCSV, metrics } = useShowroom();

  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [filterPurpose, setFilterPurpose] = useState<string>('all');
  const [filterTagging, setFilterTagging] = useState<string>('all');
  const [filterSalesperson, setFilterSalesperson] = useState<string>('all');
  const [checkoutNotification, setCheckoutNotification] = useState<string | null>(null);
  const [inspectedVisitor, setInspectedVisitor] = useState<VisitorRecord | null>(null);

  // Extract unique salespeople
  const salespeople = useMemo(() => {
    const set = new Set<string>();
    visitors.forEach((v) => {
      if (v.assignedSalesperson) set.add(v.assignedSalesperson);
    });
    return Array.from(set);
  }, [visitors]);

  // Filter visitors
  const filteredVisitors = useMemo(() => {
    return visitors.filter((v) => {
      // Search
      const q = searchQuery.toLowerCase();
      const matchesSearch =
        v.visitorId.toLowerCase().includes(q) ||
        (v.notes && v.notes.toLowerCase().includes(q)) ||
        (v.assignedSalesperson && v.assignedSalesperson.toLowerCase().includes(q)) ||
        (v.visitPurpose && v.visitPurpose.toLowerCase().includes(q));

      if (!matchesSearch) return false;

      // Filter by Type
      if (filterType !== 'all' && v.visitorType !== filterType) return false;

      // Filter by Purpose
      if (filterPurpose !== 'all') {
        if (filterPurpose === 'untagged' && v.visitPurpose !== null) return false;
        if (filterPurpose !== 'untagged' && v.visitPurpose !== filterPurpose) return false;
      }

      // Filter by Tagging status
      if (filterTagging !== 'all' && v.taggingStatus !== filterTagging) return false;

      // Filter by Salesperson
      if (filterSalesperson !== 'all' && v.assignedSalesperson !== filterSalesperson) return false;

      return true;
    });
  }, [visitors, searchQuery, filterType, filterPurpose, filterTagging, filterSalesperson]);

  const handleCheckout = (visitorId: string, customId: string) => {
    checkoutVisitor(visitorId);
    setCheckoutNotification(`Visitor ${customId} exit recorded successfully.`);
    setTimeout(() => setCheckoutNotification(null), 3000);
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-['Outfit'] text-slate-900">Showroom Visitors Registry</h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-blue-100 text-blue-800">
              {filteredVisitors.length} / {visitors.length} records
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Complete log of showroom entries, dwell durations, and assigned visit purposes.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={exportVisitorsCSV}
            className="px-3.5 py-2 bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 hover:border-slate-400 rounded-xl text-xs font-semibold shadow-2xs flex items-center gap-1.5 transition"
          >
            <FileSpreadsheet className="w-4 h-4 text-emerald-600" />
            <span>Export CSV</span>
          </button>
          <button
            onClick={onOpenQuickAdd}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-semibold border-2 border-blue-700 shadow-md shadow-blue-600/20 flex items-center gap-1.5 transition active:scale-95"
          >
            <PlusCircle className="w-4 h-4" />
            <span>Log Walk-in</span>
          </button>
        </div>
      </div>

      {checkoutNotification && (
        <div className="p-3 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-xl text-xs flex items-center gap-2 animate-fadeIn">
          <CheckCircle className="w-4 h-4 text-emerald-600" />
          <span>{checkoutNotification}</span>
        </div>
      )}

      {/* KPI Cards Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
        <div className="p-4 sm:p-5 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Total Logged</span>
            <div className="p-2 rounded-xl bg-slate-100 text-slate-700 group-hover:scale-110 transition-transform">
              <Users className="w-4 h-4" />
            </div>
          </div>
          <p className="text-3xl font-extrabold font-['Outfit'] text-slate-900 mt-1">{metrics.totalVisitors}</p>
          <span className="text-[11px] text-slate-400 mt-0.5 block font-medium">All recorded visits today</span>
        </div>

        <div className="p-4 sm:p-5 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold text-blue-700 uppercase tracking-wider">New Walk-ins</span>
            <div className="p-2 rounded-xl bg-blue-50 text-blue-600 group-hover:scale-110 transition-transform">
              <Sparkles className="w-4 h-4" />
            </div>
          </div>
          <p className="text-3xl font-extrabold font-['Outfit'] text-blue-900 mt-1">{metrics.newVisitors}</p>
          <span className="text-[11px] text-blue-700 mt-0.5 block font-semibold">{metrics.newVisitorPct}% of total footfall</span>
        </div>

        <div className="p-4 sm:p-5 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold text-indigo-700 uppercase tracking-wider">Returning</span>
            <div className="p-2 rounded-xl bg-indigo-50 text-indigo-600 group-hover:scale-110 transition-transform">
              <Users className="w-4 h-4" />
            </div>
          </div>
          <p className="text-3xl font-extrabold font-['Outfit'] text-indigo-900 mt-1">{metrics.returningVisitors}</p>
          <span className="text-[11px] text-indigo-700 mt-0.5 block font-semibold">{metrics.returningVisitorPct}% repeat engagement</span>
        </div>

        <div className="p-4 sm:p-5 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold text-amber-700 uppercase tracking-wider">Untagged Leads</span>
            <div className="p-2 rounded-xl bg-amber-50 text-amber-600 group-hover:scale-110 transition-transform">
              <Tag className="w-4 h-4" />
            </div>
          </div>
          <p className="text-3xl font-extrabold font-['Outfit'] text-amber-900 mt-1">{metrics.untaggedCount}</p>
          <span className="text-[11px] text-amber-700 mt-0.5 block font-semibold">Requires staff tagging</span>
        </div>
      </div>

      {/* Search and Filters Bar */}
      <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] space-y-3">
        <div className="flex flex-col md:flex-row gap-3">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by Visitor ID, notes, salesperson, or purpose..."
              className="w-full pl-10 pr-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition"
            />
          </div>

          {/* Quick Filters */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Type */}
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-600 outline-none font-semibold text-slate-700 shadow-2xs"
            >
              <option value="all">All Types (New & Ret)</option>
              <option value="New">New Visitors Only</option>
              <option value="Returning">Returning Customers Only</option>
            </select>

            {/* Purpose */}
            <select
              value={filterPurpose}
              onChange={(e) => setFilterPurpose(e.target.value)}
              className="px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-600 outline-none font-semibold text-slate-700 shadow-2xs"
            >
              <option value="all">All Purposes</option>
              <option value="untagged">⚠️ Untagged Only</option>
              {PURPOSES.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>

            {/* Salesperson */}
            <select
              value={filterSalesperson}
              onChange={(e) => setFilterSalesperson(e.target.value)}
              className="px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-600 outline-none font-semibold text-slate-700 shadow-2xs"
            >
              <option value="all">All Salespersons</option>
              {salespeople.map((sp) => (
                <option key={sp} value={sp}>
                  {sp}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Main Table */}
      <div className="bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50/90 border-b border-slate-200 text-slate-600 uppercase text-[10px] tracking-wider font-bold">
              <tr>
                <th className="py-3 px-4">Visitor & Detected Face</th>
                <th className="py-3 px-3">First Seen</th>
                <th className="py-3 px-3">Exit Time</th>
                <th className="py-3 px-3">Type</th>
                <th className="py-3 px-3">Visit Purpose</th>
                <th className="py-3 px-3">Assigned Salesperson</th>
                <th className="py-3 px-3">Duration</th>
                <th className="py-3 px-3">Zone</th>
                <th className="py-3 px-4">Notes</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {filteredVisitors.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-12 text-center text-slate-400">
                    <p className="text-sm font-semibold">No visitor records match your filters</p>
                    <p className="text-xs text-slate-500 mt-1">Try resetting search keywords or filters.</p>
                  </td>
                </tr>
              ) : (
                filteredVisitors.map((vis) => {
                  const isTagged = vis.taggingStatus === 'Tagged' && vis.visitPurpose;
                  const purposeColor = vis.visitPurpose ? PURPOSE_COLORS[vis.visitPurpose] : '#94a3b8';
                  const isCurrentlyInside = !vis.exitTime;
                  const assignedRep = INITIAL_EMPLOYEES.find((e) => e.name === vis.assignedSalesperson);

                  return (
                    <tr
                      key={vis.id}
                      className={`hover:bg-slate-50/80 transition-colors ${
                        !isTagged ? 'bg-amber-50/25' : ''
                      }`}
                    >
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          <div
                            onClick={() => setInspectedVisitor(vis)}
                            className="relative group/cctv shrink-0 cursor-pointer overflow-hidden rounded-xl border border-slate-700/80 bg-slate-950 shadow-xs hover:border-emerald-500 transition-all w-11 h-11"
                            title="Click to view full CCTV camera detection snapshot"
                          >
                            <img
                              src={vis.avatarUrl || 'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=160&auto=format&fit=crop&q=80'}
                              alt={vis.visitorId}
                              className="w-full h-full object-cover object-top contrast-[1.05] brightness-95 group-hover/cctv:scale-105 transition-transform"
                            />
                            {/* CCTV Reticle & Corner Box */}
                            <div className="absolute inset-0.5 border border-emerald-400/40 rounded-lg pointer-events-none" />
                            <div className="absolute top-0.5 left-1 bg-black/80 px-1 py-0.2 rounded text-[7px] font-mono font-bold text-emerald-400 leading-tight">
                              CAM-01
                            </div>
                            <div className="absolute bottom-0.5 right-1 flex items-center gap-0.5 bg-black/80 px-1 py-0.2 rounded leading-tight">
                              <span className="w-1 h-1 rounded-full bg-red-500 animate-ping" />
                              <span className="text-[7px] font-mono text-white/90 font-bold">REC</span>
                            </div>
                          </div>
                          <div>
                            <span className="font-mono font-bold text-slate-900 block">{vis.visitorId}</span>
                            <span className="text-[10px] text-slate-400 flex items-center gap-1 font-medium">
                              Main Gate NVR
                            </span>
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-3 font-mono text-slate-700 font-medium">{vis.firstSeenTime}</td>
                      <td className="py-3 px-3 font-mono">
                        {vis.exitTime ? (
                          <span className="text-slate-700 font-medium">{vis.exitTime}</span>
                        ) : (
                          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 animate-pulse border border-emerald-200">
                            Inside
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            vis.visitorType === 'New'
                              ? 'bg-blue-100 text-blue-800'
                              : 'bg-indigo-100 text-indigo-800'
                          }`}
                        >
                          {vis.visitorType}
                        </span>
                      </td>
                      <td className="py-3 px-3">
                        {vis.visitPurpose ? (
                          <span
                            className="px-2.5 py-0.5 rounded-full text-[10px] font-bold text-white inline-flex items-center gap-1 shadow-2xs"
                            style={{ backgroundColor: purposeColor }}
                          >
                            {vis.visitPurpose}
                          </span>
                        ) : (
                          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800 border border-amber-300 shadow-2xs">
                            Untagged
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-3">
                        {vis.assignedSalesperson ? (
                          <div className="flex items-center gap-2">
                            <img
                              src={assignedRep?.avatarUrl || 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&auto=format&fit=crop&q=80'}
                              alt={vis.assignedSalesperson}
                              className="w-6 h-6 rounded-full object-cover border border-slate-200 shrink-0 shadow-2xs"
                            />
                            <span className="font-semibold text-slate-800">{vis.assignedSalesperson}</span>
                          </div>
                        ) : (
                          <span className="text-slate-400 font-normal">Unassigned</span>
                        )}
                      </td>
                      <td className="py-3 px-3 font-mono font-semibold text-slate-700">{vis.durationMinutes}m</td>
                      <td className="py-3 px-3 text-slate-600">{vis.zoneEntered || 'Showroom'}</td>
                      <td className="py-3 px-4 text-slate-600 max-w-[220px] truncate" title={vis.notes}>
                        {vis.notes}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            onClick={() => onTagVisitor(vis)}
                            className="px-3 py-1.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-xs font-bold border border-blue-400/30 shadow-2xs transition active:scale-95"
                          >
                            {vis.visitPurpose ? 'Edit' : 'Tag'}
                          </button>
                          {isCurrentlyInside && (
                            <button
                              onClick={() => handleCheckout(vis.id, vis.visitorId)}
                              title="Mark customer exit"
                              className="px-2.5 py-1.5 bg-white hover:bg-slate-100 text-slate-700 rounded-xl text-xs font-bold border border-slate-300 shadow-2xs transition active:scale-95"
                            >
                              Exit
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* CCTV Camera Face Detection Snapshot Modal */}
      {inspectedVisitor && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/75 backdrop-blur-sm animate-fadeIn">
          <div className="bg-slate-900 text-white rounded-3xl max-w-lg w-full border border-slate-700 shadow-2xl overflow-hidden relative animate-scaleUp">
            {/* Header */}
            <div className="p-4 sm:p-5 border-b border-slate-800 flex items-center justify-between bg-slate-950">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-emerald-500/20 text-emerald-400 border border-emerald-400/30">
                  <Camera className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-extrabold text-sm sm:text-base font-['Outfit'] text-white flex items-center gap-2">
                    CCTV Camera Entry Snapshot
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  </h3>
                  <p className="text-[11px] text-slate-400 font-mono">
                    CH-01 Main Glass Entrance • NVR Stream
                  </p>
                </div>
              </div>
              <button
                onClick={() => setInspectedVisitor(null)}
                className="p-2 text-slate-400 hover:text-white rounded-xl hover:bg-slate-800 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-5 sm:p-6 space-y-4">
              {/* Clear CCTV Snapshot Frame */}
              <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-black flex items-center justify-center h-64 sm:h-72 group shadow-inner">
                <img
                  src={inspectedVisitor.avatarUrl || 'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=600&auto=format&fit=crop&q=80'}
                  alt={inspectedVisitor.visitorId}
                  className="w-full h-full object-cover object-top contrast-[1.05] brightness-95"
                />

                {/* CCTV Top Status Overlay */}
                <div className="absolute top-2.5 inset-x-3 flex items-center justify-between text-[10px] font-mono text-emerald-400 bg-black/75 px-2.5 py-1 rounded border border-emerald-400/20 pointer-events-none">
                  <span className="flex items-center gap-1.5 font-bold">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
                    LIVE • CAM 01 [MAIN ENTRANCE]
                  </span>
                  <span className="text-slate-300">1080P HD • 25 FPS</span>
                </div>

                {/* Face Detection Bounding Corners */}
                <div className="absolute inset-x-12 inset-y-8 border border-emerald-400/80 rounded-xl pointer-events-none shadow-[0_0_15px_rgba(16,185,129,0.4)] flex flex-col justify-between p-2">
                  <div className="flex justify-between items-start">
                    <span className="bg-emerald-600 text-slate-950 text-[9px] font-black px-1.5 py-0.5 rounded font-mono uppercase tracking-wider">
                      AI MATCH 98.8%
                    </span>
                    <span className="text-[8px] font-mono text-emerald-300 bg-black/80 px-1 py-0.5 rounded">
                      ID: {inspectedVisitor.visitorId}
                    </span>
                  </div>
                  <div className="flex justify-between items-end text-[8px] font-mono text-emerald-300/80">
                    <span>⌜ ROI_FACE_01 ⌝</span>
                    <span>CONFIRMED</span>
                  </div>
                </div>

                {/* CCTV Bottom Watermark */}
                <div className="absolute bottom-2.5 inset-x-3 flex items-center justify-between text-[9px] font-mono text-white/90 bg-black/75 px-2.5 py-1 rounded pointer-events-none">
                  <span>TIME: 2026-10-26 {inspectedVisitor.firstSeenTime} IST</span>
                  <span>BITRATE: 4096 KBPS</span>
                </div>
              </div>

              {/* Telemetry Grid */}
              <div className="grid grid-cols-2 gap-2.5 text-xs">
                <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/60">
                  <span className="text-slate-400 text-[10px] uppercase font-bold block">First Seen</span>
                  <p className="font-extrabold text-white font-mono mt-0.5">{inspectedVisitor.firstSeenTime}</p>
                </div>
                <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/60">
                  <span className="text-slate-400 text-[10px] uppercase font-bold block">Status & Dwell</span>
                  <p className="font-extrabold text-white font-mono mt-0.5">
                    {inspectedVisitor.exitTime ? `Exited (${inspectedVisitor.durationMinutes}m)` : `Inside Showroom (${inspectedVisitor.durationMinutes}m)`}
                  </p>
                </div>
                <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/60">
                  <span className="text-slate-400 text-[10px] uppercase font-bold block">Visit Intent</span>
                  <p className="font-extrabold text-cyan-400 mt-0.5">{inspectedVisitor.visitPurpose || 'Untagged'}</p>
                </div>
                <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/60">
                  <span className="text-slate-400 text-[10px] uppercase font-bold block">Assigned Staff</span>
                  <p className="font-extrabold text-white mt-0.5">{inspectedVisitor.assignedSalesperson || 'Unassigned'}</p>
                </div>
              </div>

              {inspectedVisitor.notes && (
                <div className="p-3 bg-slate-800/40 rounded-xl border border-slate-700/50 text-xs text-slate-300">
                  <span className="text-[10px] text-slate-400 uppercase font-bold block mb-1">Customer Session Notes</span>
                  {inspectedVisitor.notes}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="p-4 bg-slate-950 border-t border-slate-800 flex items-center justify-end gap-2.5">
              <button
                onClick={() => setInspectedVisitor(null)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-xl text-xs font-semibold transition"
              >
                Close
              </button>
              <button
                onClick={() => {
                  const target = inspectedVisitor;
                  setInspectedVisitor(null);
                  onTagVisitor(target);
                }}
                className="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl text-xs font-bold transition shadow-lg shadow-blue-600/30"
              >
                Assign / Edit Purpose
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
