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
  Trash2,
  User,
  AlertTriangle,
} from 'lucide-react';
import { PURPOSE_COLORS } from '../../data/seedData';

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
  const { visitors, checkoutVisitor, exportVisitorsCSV, clearAllVisitors, metrics, employees } = useShowroom();

  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [filterPurpose, setFilterPurpose] = useState<string>('all');
  const [filterTagging, setFilterTagging] = useState<string>('all');
  const [filterSalesperson, setFilterSalesperson] = useState<string>('all');
  const [filterWaiting, setFilterWaiting] = useState<string>('all');
  const [checkoutNotification, setCheckoutNotification] = useState<string | null>(null);
  const [inspectedVisitor, setInspectedVisitor] = useState<VisitorRecord | null>(null);
  const [isClearing, setIsClearing] = useState(false);

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

      // Filter by Waiting status
      if (filterWaiting === 'waiting' && (!v.isWaiting || v.exitTime || v.assignedSalesperson)) return false;
      if (filterWaiting === 'breached' && (!v.waitingBreach || !v.isWaiting || v.exitTime || v.assignedSalesperson)) return false;
      if (filterWaiting === 'attended' && (v.isWaiting && !v.exitTime && !v.assignedSalesperson)) return false;

      return true;
    });
  }, [visitors, searchQuery, filterType, filterPurpose, filterTagging, filterSalesperson, filterWaiting]);

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
          {visitors.length > 0 && (
            <button
              onClick={async () => {
                if (window.confirm('Are you sure you want to clear all visitor records and captured photos? This will reset the visitor list to start fresh.')) {
                  setIsClearing(true);
                  await clearAllVisitors();
                  setIsClearing(false);
                }
              }}
              disabled={isClearing}
              className="px-3.5 py-2 bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 rounded-xl text-xs font-semibold shadow-2xs flex items-center gap-1.5 transition active:scale-95 disabled:opacity-50"
            >
              <Trash2 className="w-4 h-4 text-red-600" />
              <span>{isClearing ? 'Clearing...' : 'Clear All Visitors'}</span>
            </button>
          )}
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

      {/* 🚨 CRITICAL UNATTENDED WAITING CUTOFF BREACH ALERT (> 15 MINS) */}
      {metrics.waitingBreachCount > 0 && (
        <div className="bg-gradient-to-r from-red-600 via-rose-600 to-red-700 text-white rounded-2xl p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-lg shadow-red-600/25 border-2 border-red-400 animate-pulse">
          <div className="flex items-start gap-3.5">
            <div className="p-2.5 bg-white/20 text-white rounded-xl shrink-0 mt-0.5 border border-white/30">
              <Users className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider bg-white text-red-700">
                  Critical SLA Breach (&gt; 15 Mins)
                </span>
                <span className="w-2.5 h-2.5 rounded-full bg-white animate-ping" />
              </div>
              <h2 className="text-base font-extrabold font-['Outfit'] mt-1">
                {metrics.waitingBreachCount} Visitor{metrics.waitingBreachCount > 1 ? 's' : ''} Left Unattended for &gt; 15 Mins!
              </h2>
              <p className="text-xs text-red-100 mt-0.5">
                These visitors are outside any department desk and no staff member has approached them.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* KPI Cards Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 sm:gap-4">
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

        {/* Waiting Unassisted */}
        <div
          onClick={() => setFilterWaiting(filterWaiting === 'waiting' ? 'all' : 'waiting')}
          className={`p-4 sm:p-5 bg-white rounded-3xl border ${
            filterWaiting === 'waiting' ? 'border-amber-500 ring-2 ring-amber-200' : 'border-slate-200/80'
          } shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition group cursor-pointer`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold text-amber-800 uppercase tracking-wider">Waiting Unassisted</span>
            <div className="p-2 rounded-xl bg-amber-50 text-amber-700 group-hover:scale-110 transition-transform">
              <Clock className="w-4 h-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <p className="text-3xl font-extrabold font-['Outfit'] text-amber-950 mt-1">{metrics.unattendedWaitingCount}</p>
            {metrics.waitingBreachCount > 0 && (
              <span className="text-[10px] font-black bg-red-100 text-red-700 px-2 py-0.5 rounded-full animate-pulse">
                {metrics.waitingBreachCount} breach!
              </span>
            )}
          </div>
          <span className="text-[11px] text-amber-700 mt-0.5 block font-semibold">
            {metrics.unattendedWaitingCount > 0 ? `Avg wait: ${metrics.avgWaitingFormatted}` : 'All currently attended'}
          </span>
        </div>
      </div>

      {/* Search and Filters Bar */}
      <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] space-y-3">
        {/* Quick Filter Pill Buttons */}
        <div className="flex items-center gap-2 overflow-x-auto pb-1 text-xs">
          <button
            onClick={() => { setFilterWaiting('all'); setFilterPurpose('all'); }}
            className={`px-3 py-1.5 rounded-xl font-bold transition whitespace-nowrap ${filterWaiting === 'all' && filterPurpose === 'all' ? 'bg-slate-900 text-white shadow-xs' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
          >
            All Visitors ({visitors.length})
          </button>
          <button
            onClick={() => setFilterWaiting(filterWaiting === 'waiting' ? 'all' : 'waiting')}
            className={`px-3 py-1.5 rounded-xl font-bold transition flex items-center gap-1.5 whitespace-nowrap ${filterWaiting === 'waiting' ? 'bg-amber-600 text-white shadow-xs' : 'bg-amber-50 text-amber-800 hover:bg-amber-100 border border-amber-200/80'}`}
          >
            <Clock className="w-3.5 h-3.5" />
            <span>Waiting Unassisted ({metrics.unattendedWaitingCount})</span>
          </button>
          {metrics.waitingBreachCount > 0 && (
            <button
              onClick={() => setFilterWaiting(filterWaiting === 'breached' ? 'all' : 'breached')}
              className={`px-3 py-1.5 rounded-xl font-bold transition flex items-center gap-1.5 whitespace-nowrap ${filterWaiting === 'breached' ? 'bg-red-600 text-white shadow-xs' : 'bg-red-50 text-red-700 hover:bg-red-100 border border-red-200 animate-pulse'}`}
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>SLA Breaches ({metrics.waitingBreachCount})</span>
            </button>
          )}
          <button
            onClick={() => setFilterPurpose(filterPurpose === 'untagged' ? 'all' : 'untagged')}
            className={`px-3 py-1.5 rounded-xl font-bold transition flex items-center gap-1.5 whitespace-nowrap ${filterPurpose === 'untagged' ? 'bg-blue-600 text-white shadow-xs' : 'bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200/80'}`}
          >
            <Tag className="w-3.5 h-3.5" />
            <span>Untagged Leads ({metrics.untaggedCount})</span>
          </button>
        </div>

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
            {/* Waiting / Service Status */}
            <select
              value={filterWaiting}
              onChange={(e) => setFilterWaiting(e.target.value)}
              className="px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-600 outline-none font-semibold text-slate-700 shadow-2xs"
            >
              <option value="all">All Service Statuses</option>
              <option value="waiting">⏳ Waiting Unattended ({metrics.unattendedWaitingCount})</option>
              <option value="breached">🚨 SLA Breaches ({metrics.waitingBreachCount})</option>
              <option value="attended">✅ Attended by Staff</option>
            </select>

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
                  const assignedRep = employees.find((e) => e.name === vis.assignedSalesperson);

                  return (
                    <tr
                      key={vis.id}
                      className={`hover:bg-slate-50/80 transition-colors ${
                        vis.waitingBreach && vis.isWaiting && !vis.exitTime && !vis.assignedSalesperson
                          ? 'bg-red-50/80 border-l-4 border-l-red-600'
                          : !isTagged
                          ? 'bg-amber-50/25'
                          : ''
                      }`}
                    >
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          <div
                            onClick={() => vis.avatarUrl && setInspectedVisitor(vis)}
                            className={`relative group/cctv shrink-0 overflow-hidden rounded-xl border border-slate-700/80 bg-slate-950 shadow-xs hover:border-emerald-500 transition-all w-12 h-16 ${
                              vis.avatarUrl ? 'cursor-pointer' : 'cursor-default'
                            }`}
                            title={vis.avatarUrl ? "Click to view full CCTV camera detection snapshot" : "No photo captured"}
                          >
                            {vis.avatarUrl ? (
                              <img
                                src={vis.avatarUrl}
                                alt={vis.visitorId}
                                className="w-full h-full object-cover object-top contrast-[1.05] brightness-95 group-hover/cctv:scale-105 transition-transform"
                              />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center bg-slate-800 text-slate-400">
                                <User className="w-5 h-5 text-slate-400" />
                              </div>
                            )}
                            {/* CCTV Reticle & Corner Box */}
                            <div className="absolute inset-0.5 border border-emerald-400/30 rounded-lg pointer-events-none" />
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
                      <td className="py-3 px-3 font-mono text-slate-700 font-medium">
                        <span>{vis.firstSeenTime}</span>
                        <span className="block text-[10px] text-slate-400 font-sans font-normal">{vis.date}</span>
                      </td>
                      <td className="py-3 px-3 font-mono">
                        {vis.exitTime ? (
                          <span className="text-slate-700 font-medium">{vis.exitTime}</span>
                        ) : (vis.waitingBreach && vis.isWaiting && !vis.assignedSalesperson) ? (
                          <span className="px-2.5 py-1 rounded-full text-[10px] font-black bg-red-100 text-red-700 border border-red-300 flex items-center gap-1.5 animate-pulse whitespace-nowrap">
                            <span className="w-1.5 h-1.5 rounded-full bg-red-600 animate-ping" />
                            Unattended {vis.waitingFormatted} (SLA Breach)
                          </span>
                        ) : (vis.isWaiting && !vis.assignedSalesperson) ? (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800 border border-amber-200 flex items-center gap-1 whitespace-nowrap">
                            <Clock className="w-3 h-3 text-amber-600" />
                            Waiting {vis.waitingFormatted}
                          </span>
                        ) : vis.waitingBreach ? (
                          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-rose-50 text-rose-700 border border-rose-200 whitespace-nowrap">
                            Attended (Waited {vis.waitingFormatted})
                          </span>
                        ) : (
                          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 animate-pulse border border-emerald-200">
                            Inside (Attended)
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
                            {assignedRep?.avatarUrl ? (
                              <img
                                src={assignedRep.avatarUrl}
                                alt={vis.assignedSalesperson}
                                className="w-6 h-6 rounded-full object-cover border border-slate-200 shrink-0 shadow-2xs"
                              />
                            ) : (
                              <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center text-[10px] font-bold border border-blue-200 shrink-0 shadow-2xs">
                                {vis.assignedSalesperson.charAt(0).toUpperCase()}
                              </div>
                            )}
                            <span className="font-semibold text-slate-800">{vis.assignedSalesperson}</span>
                          </div>
                        ) : (
                          <span className="text-slate-400 font-normal">Unassigned</span>
                        )}
                      </td>
                      <td className="py-3 px-3 font-mono font-semibold text-slate-700">
                        {vis.durationMinutes <= 0 ? '< 1m' : `${vis.durationMinutes}m`}
                      </td>
                      <td className="py-3 px-3 text-slate-600">
                        <span className="font-semibold text-slate-800 block">{vis.zoneEntered || 'Showroom'}</span>
                        {vis.zoneDwell && Object.keys(vis.zoneDwell).length > 0 && (
                          <span
                            className="text-[10px] text-amber-700 font-medium block truncate max-w-[130px]"
                            title={Object.entries(vis.zoneDwell).map(([z, t]) => `${z}: ${t}`).join(', ')}
                          >
                            {Object.entries(vis.zoneDwell).map(([z, t]) => `${z} (${t})`).join(', ')}
                          </span>
                        )}
                      </td>
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
              {/* Clear CCTV Snapshot Frame (Full Body Portrait) */}
              <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center h-80 sm:h-96 group shadow-inner">
                {inspectedVisitor.avatarUrl ? (
                  <img
                    src={inspectedVisitor.avatarUrl}
                    alt={inspectedVisitor.visitorId}
                    className="w-full h-full object-contain contrast-[1.05] brightness-95"
                  />
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center bg-slate-900 text-slate-400">
                    <User className="w-16 h-16 text-slate-500 mb-2" />
                    <span className="text-xs font-semibold">No body photo captured</span>
                  </div>
                )}

                {/* CCTV Top Status Overlay */}
                <div className="absolute top-2.5 inset-x-3 flex items-center justify-between text-[10px] font-mono text-emerald-400 bg-black/75 px-2.5 py-1 rounded border border-emerald-400/20 pointer-events-none">
                  <span className="flex items-center gap-1.5 font-bold">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
                    LIVE • CAM 01 [MAIN ENTRANCE]
                  </span>
                  <span className="text-emerald-400 font-bold bg-emerald-950/80 px-1.5 py-0.5 rounded border border-emerald-500/30">
                    BODY SNAPSHOT • {inspectedVisitor.visitorId}
                  </span>
                </div>

                {/* CCTV Bottom Watermark */}
                <div className="absolute bottom-2.5 inset-x-3 flex items-center justify-between text-[9px] font-mono text-white/90 bg-black/75 px-2.5 py-1 rounded pointer-events-none">
                  <span>TIME: {inspectedVisitor.date} {inspectedVisitor.firstSeenTime} IST</span>
                  <span className="text-slate-400">1080P HD • 4096 KBPS</span>
                </div>
              </div>

              {/* Telemetry Grid */}
              <div className="grid grid-cols-2 gap-2.5 text-xs">
                <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/60">
                  <span className="text-slate-400 text-[10px] uppercase font-bold block">Arrived Date & Time</span>
                  <p className="font-extrabold text-white font-mono mt-0.5">{inspectedVisitor.date} • {inspectedVisitor.firstSeenTime}</p>
                </div>
                <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/60">
                  <span className="text-slate-400 text-[10px] uppercase font-bold block">Departure & Status</span>
                  <p className="font-extrabold text-white font-mono mt-0.5">
                    {inspectedVisitor.exitTime ? `Departed at ${inspectedVisitor.exitTime}` : `Inside Showroom (${inspectedVisitor.durationMinutes}m)`}
                  </p>
                </div>
                <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/60">
                  <span className="text-slate-400 text-[10px] uppercase font-bold block">Visit Intent & Staff</span>
                  <p className="font-extrabold text-cyan-400 mt-0.5">{inspectedVisitor.visitPurpose || 'Untagged'}</p>
                  <p className="text-[11px] text-slate-300 mt-0.5 font-medium">{inspectedVisitor.assignedSalesperson || 'Unassigned Staff'}</p>
                </div>
                <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/60">
                  <span className="text-slate-400 text-[10px] uppercase font-bold block">Zone Dwell Time</span>
                  <p className="font-extrabold text-amber-400 mt-0.5">
                    {inspectedVisitor.zoneDwell && Object.keys(inspectedVisitor.zoneDwell).length > 0
                      ? Object.entries(inspectedVisitor.zoneDwell).map(([z, t]) => `${z} (${t})`).join(', ')
                      : `${inspectedVisitor.zoneEntered || 'Main Showroom'} (${inspectedVisitor.durationMinutes}m)`}
                  </p>
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
