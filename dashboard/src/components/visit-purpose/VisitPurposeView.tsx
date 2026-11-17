import React, { useState, useMemo } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import { VisitorRecord, VisitPurpose, TaggingStatus } from '../../types';
import {
  Tags,
  AlertTriangle,
  Search,
  Filter,
  CheckCircle,
  FileSpreadsheet,
  Clock,
  Sparkles,
  ArrowRight,
  Camera,
  X,
  ZoomIn,
  Trash2,
  User,
} from 'lucide-react';
import { PURPOSE_COLORS } from '../../data/seedData';

interface VisitPurposeViewProps {
  onTagVisitor: (visitor: VisitorRecord) => void;
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

export const VisitPurposeView: React.FC<VisitPurposeViewProps> = ({ onTagVisitor }) => {
  const { visitors, tagVisitor, clearAllVisitors, metrics, exportVisitorsCSV, employees } = useShowroom();

  const [activeTabFilter, setActiveTabFilter] = useState<'all' | 'untagged' | 'tagged'>('all');
  const [purposeFilter, setPurposeFilter] = useState<string>('all');
  const [salespersonFilter, setSalespersonFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [previewVisitor, setPreviewVisitor] = useState<VisitorRecord | null>(null);
  const [isClearing, setIsClearing] = useState(false);

  // Extract salespeople
  const salespeople = useMemo(() => {
    const list = new Set<string>();
    employees.forEach((e) => list.add(e.name));
    return Array.from(list);
  }, [employees]);

  // Filter visitors
  const filteredVisitors = useMemo(() => {
    return visitors.filter((v) => {
      const q = searchQuery.toLowerCase();
      const matchSearch =
        v.visitorId.toLowerCase().includes(q) ||
        (v.notes && v.notes.toLowerCase().includes(q)) ||
        (v.assignedSalesperson && v.assignedSalesperson.toLowerCase().includes(q));

      if (!matchSearch) return false;

      if (activeTabFilter === 'untagged' && v.taggingStatus !== 'Untagged') return false;
      if (activeTabFilter === 'tagged' && v.taggingStatus !== 'Tagged') return false;

      if (purposeFilter !== 'all' && v.visitPurpose !== purposeFilter) return false;
      if (salespersonFilter !== 'all' && v.assignedSalesperson !== salespersonFilter) return false;

      return true;
    });
  }, [visitors, searchQuery, activeTabFilter, purposeFilter, salespersonFilter]);

  // Inline quick tag helper
  const handleInlineQuickTag = (visitorId: string, p: VisitPurpose) => {
    const currentVis = visitors.find((v) => v.id === visitorId);
    tagVisitor(visitorId, p, currentVis?.assignedSalesperson || 'Rajesh Kumar', currentVis?.notes || '');
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-['Outfit'] text-slate-900">Visit Purpose Management & Tagging</h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-purple-100 text-purple-800">
              8 Core Categories
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Assign and track customer journey intent (Test Ride, Vehicle Enquiry, Booking, Service, Delivery).
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
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
            <span>Export Tagged CSV</span>
          </button>
        </div>
      </div>

      {/* Untagged Warning Banner */}
      {metrics.untaggedCount > 0 && (
        <div className="p-5 bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-transparent border border-amber-300/80 rounded-3xl flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-2xs">
          <div className="flex items-start gap-3.5">
            <div className="p-2.5 bg-amber-500 text-white rounded-2xl shrink-0 mt-0.5 shadow-md shadow-amber-500/20">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-extrabold text-amber-950 text-base font-['Outfit']">
                Action Required: {metrics.untaggedCount} Visitors Need Purpose Tagging
              </h3>
              <p className="text-xs text-amber-800 font-medium mt-0.5">
                Quickly tag untagged visitors to maintain accurate showroom conversion funnels and lead attribution.
              </p>
            </div>
          </div>
          <button
            onClick={() => setActiveTabFilter('untagged')}
            className="px-4 py-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 text-white font-bold text-xs rounded-xl shadow-md shadow-amber-600/20 transition shrink-0 active:scale-95"
          >
            Show Untagged Only
          </button>
        </div>
      )}

      {/* 8 Purpose Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
        {metrics.purposeDistribution.map((item) => (
          <div
            key={item.purpose}
            onClick={() => setPurposeFilter(purposeFilter === item.purpose ? 'all' : item.purpose)}
            className={`p-3.5 rounded-2xl border transition-all cursor-pointer text-center relative group ${
              purposeFilter === item.purpose
                ? 'bg-blue-50/90 border-blue-600 ring-2 ring-blue-500/30 shadow-md scale-[1.02]'
                : 'bg-white border-slate-200/80 hover:border-blue-300 shadow-2xs hover:shadow-xs hover:scale-[1.02]'
            }`}
          >
            <div className="flex items-center justify-center gap-1.5 mb-1.5">
              <span className="w-2.5 h-2.5 rounded-full shrink-0 group-hover:scale-125 transition-transform" style={{ backgroundColor: item.color }} />
              <span className="text-[11px] font-bold text-slate-800 truncate">{item.purpose}</span>
            </div>
            <p className="text-2xl font-black font-['Outfit'] text-slate-900">{item.count}</p>
            <span className="text-[10px] text-slate-500 font-semibold">{item.percentage}% share</span>
          </div>
        ))}
      </div>

      {/* Search and Tag State Tabs */}
      <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] space-y-4">
        <div className="flex flex-col md:flex-row justify-between gap-3">
          {/* Tag Status Filter Tabs */}
          <div className="flex bg-slate-100/90 p-1 rounded-2xl self-start border border-slate-200/80 shadow-2xs">
            <button
              onClick={() => setActiveTabFilter('all')}
              className={`px-3.5 py-1.5 text-xs font-bold rounded-xl transition ${
                activeTabFilter === 'all' ? 'bg-blue-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              All Visitors ({visitors.length})
            </button>
            <button
              onClick={() => setActiveTabFilter('untagged')}
              className={`px-3.5 py-1.5 text-xs font-bold rounded-xl transition flex items-center gap-1.5 ${
                activeTabFilter === 'untagged'
                  ? 'bg-amber-600 text-white shadow-xs'
                  : 'text-amber-800 hover:text-amber-950'
              }`}
            >
              <span>Untagged</span>
              <span className="px-1.5 py-0.2 bg-white/30 text-white text-[10px] rounded-full font-bold">
                {metrics.untaggedCount}
              </span>
            </button>
            <button
              onClick={() => setActiveTabFilter('tagged')}
              className={`px-3.5 py-1.5 text-xs font-bold rounded-xl transition ${
                activeTabFilter === 'tagged'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Tagged ({metrics.taggedCount})
            </button>
          </div>

          {/* Search Input */}
          <div className="relative min-w-[260px]">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search visitor ID, notes, rep..."
              className="w-full pl-10 pr-3.5 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition"
            />
          </div>
        </div>

        {/* Dropdowns */}
        <div className="flex flex-wrap items-center gap-2 pt-3 border-t border-slate-100 text-xs">
          <span className="text-slate-500 font-bold uppercase tracking-wider text-[10px]">Filter by:</span>
          <select
            value={purposeFilter}
            onChange={(e) => setPurposeFilter(e.target.value)}
            className="px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-700 font-semibold shadow-2xs outline-none focus:border-blue-600"
          >
            <option value="all">All Purpose Categories</option>
            {PURPOSES.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>

          <select
            value={salespersonFilter}
            onChange={(e) => setSalespersonFilter(e.target.value)}
            className="px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-700 font-semibold shadow-2xs outline-none focus:border-blue-600"
          >
            <option value="all">All Assigned Staff</option>
            {salespeople.map((sp) => (
              <option key={sp} value={sp}>
                {sp}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Visitors List / Table with Quick-Tag Pill bar */}
      <div className="bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50/90 border-b border-slate-200 text-slate-600 uppercase text-[10px] tracking-wider font-bold">
              <tr>
                <th className="py-3 px-4">Visitor & Detected Face</th>
                <th className="py-3 px-3">First Seen</th>
                <th className="py-3 px-3">Type</th>
                <th className="py-3 px-3">Current Tag</th>
                <th className="py-3 px-3">Quick-Tag Actions (1-Click)</th>
                <th className="py-3 px-3">Assigned Salesperson</th>
                <th className="py-3 px-4">Notes</th>
                <th className="py-3 px-4 text-right">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {filteredVisitors.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-400">
                    <div className="flex flex-col items-center justify-center gap-2">
                      <Camera className="w-8 h-8 text-slate-300 animate-pulse" />
                      <p className="text-sm font-semibold text-slate-600">No visitors recorded yet</p>
                      <p className="text-xs text-slate-400">
                        CCTV AI detection will automatically snapshot arriving customers and list them here for 1-click tagging.
                      </p>
                    </div>
                  </td>
                </tr>
              ) : (
                filteredVisitors.map((vis) => {
                const isTagged = vis.taggingStatus === 'Tagged' && vis.visitPurpose;
                const purposeColor = vis.visitPurpose ? PURPOSE_COLORS[vis.visitPurpose] : '#94a3b8';

                return (
                  <tr
                    key={vis.id}
                    className={`hover:bg-slate-50/80 transition-colors ${
                      vis.waitingBreach
                        ? 'bg-red-50/80 border-l-4 border-l-red-600'
                        : !isTagged
                        ? 'bg-amber-50/20'
                        : ''
                    }`}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-3">
                        <div 
                          onClick={() => vis.avatarUrl && setPreviewVisitor(vis)}
                          title={vis.avatarUrl ? "Click to zoom high-res CCTV photo" : "No photo captured"}
                          className={`relative group shrink-0 overflow-hidden rounded-xl border border-slate-700/80 bg-slate-950 shadow-xs w-11 h-11 ${
                            vis.avatarUrl ? 'cursor-pointer' : 'cursor-default'
                          }`}
                        >
                          {vis.avatarUrl ? (
                            <>
                              <img
                                src={vis.avatarUrl}
                                alt={vis.visitorId}
                                className="w-full h-full object-cover object-top contrast-[1.05] brightness-95 group-hover:scale-110 transition duration-200"
                              />
                              <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
                                <ZoomIn className="w-3.5 h-3.5 text-white drop-shadow" />
                              </div>
                            </>
                          ) : (
                            <div className="w-full h-full flex items-center justify-center bg-slate-800 text-slate-400">
                              <User className="w-5 h-5 text-slate-400" />
                            </div>
                          )}
                          <div className="absolute top-0.5 left-1 bg-black/80 px-1 py-0.2 rounded text-[7px] font-mono font-bold text-emerald-400 leading-tight">
                            {vis.cameraSource ? vis.cameraSource.slice(0, 7) : 'CAM-01'}
                          </div>
                        </div>
                        <div>
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="font-mono font-bold text-slate-900 block">{vis.visitorId}</span>
                            {vis.notes && vis.notes.includes("Appearance:") && (
                              <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-purple-50 text-purple-700 border border-purple-200">
                                👔 {vis.notes.split('|')[0].replace("Appearance:", "").trim()}
                              </span>
                            )}
                          </div>
                          <span className="text-[10px] text-slate-400 font-medium">{vis.cameraSource || 'Entrance Camera'}</span>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-700 font-medium">{vis.firstSeenTime}</td>
                    <td className="py-3 px-3">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          vis.visitorType === 'New' ? 'bg-blue-100 text-blue-800' : 'bg-indigo-100 text-indigo-800'
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
                        <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800 border border-amber-300 shadow-2xs animate-pulse">
                          Untagged
                        </span>
                      )}
                    </td>
                    {/* 1-Click Quick Tag Pill Buttons */}
                    <td className="py-3 px-3">
                      <div className="flex flex-wrap items-center gap-1.5 max-w-[300px]">
                        {(['Test Ride', 'Vehicle Enquiry', 'Booking', 'Service'] as VisitPurpose[]).map((p) => {
                          const isActive = vis.visitPurpose === p;
                          return (
                            <button
                              key={p}
                              onClick={() => handleInlineQuickTag(vis.id, p)}
                              className={`px-2 py-1 rounded-lg text-[10px] font-bold border transition active:scale-95 ${
                                isActive
                                  ? 'bg-blue-600 text-white border-blue-700 shadow-2xs'
                                  : 'bg-slate-100 text-slate-700 border-slate-200 hover:bg-slate-200'
                              }`}
                            >
                              {p}
                            </button>
                          );
                        })}
                      </div>
                    </td>
                    <td className="py-3 px-3 text-slate-800 font-semibold">
                      {vis.assignedSalesperson || <span className="text-slate-400 font-normal">Unassigned</span>}
                    </td>
                    <td className="py-3 px-4 text-slate-600 max-w-[200px] truncate" title={vis.notes}>
                      {vis.notes}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => onTagVisitor(vis)}
                        className="px-3 py-1.5 bg-white hover:bg-slate-100 text-slate-700 font-bold rounded-xl text-xs border border-slate-300 hover:border-slate-400 shadow-2xs transition active:scale-95"
                      >
                        Details
                      </button>
                    </td>
                  </tr>
                );
              }))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visitor Photo Lightbox Modal */}
      {previewVisitor && (
        <div 
          onClick={() => setPreviewVisitor(null)}
          className="fixed inset-0 bg-slate-900/75 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fadeIn"
        >
          <div 
            onClick={(e) => e.stopPropagation()}
            className="bg-white rounded-3xl max-w-sm w-full p-5 shadow-2xl border border-slate-200 relative animate-scaleUp"
          >
            <button
              onClick={() => setPreviewVisitor(null)}
              className="absolute top-4 right-4 p-2 text-slate-400 hover:text-slate-600 rounded-xl hover:bg-slate-100 transition"
            >
              <X className="w-5 h-5" />
            </button>
            <div className="relative rounded-2xl overflow-hidden border border-slate-700 bg-slate-950 aspect-square shadow-inner flex items-center justify-center">
              {previewVisitor.avatarUrl ? (
                <img
                  src={previewVisitor.avatarUrl}
                  alt={previewVisitor.visitorId}
                  className="w-full h-full object-cover object-top contrast-[1.05] brightness-95"
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center bg-slate-900 text-slate-400">
                  <User className="w-16 h-16 text-slate-500 mb-2" />
                  <span className="text-xs font-semibold">No face photo captured</span>
                </div>
              )}
              <div className="absolute top-2 left-2 bg-black/80 px-2 py-0.5 rounded text-[10px] font-mono font-bold text-emerald-400">
                {previewVisitor.cameraSource || 'Entrance Camera'}
              </div>
              <div className="absolute bottom-2 right-2 flex items-center gap-1 bg-black/80 px-2 py-0.5 rounded text-[10px] font-mono text-white font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping" />
                <span>AI CAPTURE</span>
              </div>
            </div>
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-mono font-extrabold text-slate-900 text-lg block">{previewVisitor.visitorId}</span>
                  <span className="text-xs text-slate-500">First seen: {previewVisitor.firstSeenTime} ({previewVisitor.durationMinutes} mins ago)</span>
                </div>
                <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-blue-100 text-blue-800">
                  {previewVisitor.visitorType}
                </span>
              </div>
              {previewVisitor.notes && previewVisitor.notes.includes("Appearance:") && (
                <div className="p-2.5 rounded-xl bg-purple-50 border border-purple-100 text-xs font-semibold text-purple-800 flex items-center gap-1.5">
                  <span>👔</span>
                  <span>{previewVisitor.notes.split('|')[0].replace("Appearance:", "").trim()}</span>
                </div>
              )}
              <div className="pt-2 flex gap-2">
                <button
                  onClick={() => {
                    const v = previewVisitor;
                    setPreviewVisitor(null);
                    onTagVisitor(v);
                  }}
                  className="flex-1 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl text-xs transition shadow-md shadow-blue-500/20 active:scale-95"
                >
                  Tag This Visitor
                </button>
                <button
                  onClick={() => setPreviewVisitor(null)}
                  className="px-4 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold rounded-xl text-xs transition"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
