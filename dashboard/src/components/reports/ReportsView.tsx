import React, { useState } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import {
  FileSpreadsheet,
  Download,
  Calendar,
  CheckCircle2,
  FileText,
  TrendingUp,
  Users,
  Clock,
  Sparkles,
} from 'lucide-react';
import { DateFilterOption } from '../../types';

export const ReportsView: React.FC = () => {
  const {
    exportVisitorsCSV,
    exportAttendanceCSV,
    metrics,
    dateFilter,
    setDateFilter,
    visitors,
    attendance,
  } = useShowroom();

  const [downloadSuccess, setDownloadSuccess] = useState<string | null>(null);

  const handleDownloadVisitors = () => {
    exportVisitorsCSV();
    setDownloadSuccess('Visitors CSV report generated and downloaded.');
    setTimeout(() => setDownloadSuccess(null), 3000);
  };

  const handleDownloadAttendance = () => {
    exportAttendanceCSV();
    setDownloadSuccess('Employee attendance CSV report generated and downloaded.');
    setTimeout(() => setDownloadSuccess(null), 3000);
  };

  const handleDownloadSummary = () => {
    // Generate Executive Summary CSV
    const headers = ['Metric', 'Value', 'Context'];
    const rows = [
      ['Report Date', new Date().toLocaleDateString(), 'Showroom Intelligence Summary'],
      ['Total Footfall Today', metrics.totalVisitors, 'Edge AI Detection'],
      ['New Visitors', `${metrics.newVisitors} (${metrics.newVisitorPct}%)`, 'First-time walk-ins'],
      ['Returning Visitors', `${metrics.returningVisitors} (${metrics.returningVisitorPct}%)`, 'Repeat customers'],
      ['Peak Hour', metrics.busiestHour, `${metrics.busiestHourCount} visitors in peak hour`],
      ['Average Visit Duration', `${metrics.avgDurationMinutes} mins`, 'Completed visits'],
      ['Median Visit Duration', `${metrics.medianDurationMinutes} mins`, 'Center of duration distribution'],
      ['Total Staff Present', `${metrics.employeesPresent} / ${metrics.employeesTotal}`, 'Floor staff on duty'],
      ['Untagged Leads', metrics.untaggedCount, 'Awaiting purpose assignment'],
    ];

    const csvContent =
      'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `Showroom_Executive_Summary_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setDownloadSuccess('Executive KPI Summary CSV downloaded.');
    setTimeout(() => setDownloadSuccess(null), 3000);
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold font-['Outfit'] text-slate-900">Showroom Analytics & CSV Reports</h1>
        <p className="text-xs text-slate-500 mt-1">
          Export dealership footfall logs, employee attendance audits, and executive conversion summaries.
        </p>
      </div>

      {downloadSuccess && (
        <div className="p-3 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-xl text-xs flex items-center gap-2 animate-fadeIn">
          <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          <span>{downloadSuccess}</span>
        </div>
      )}

      {/* Date Filter selector bar */}
      <div className="p-5 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-blue-50 text-blue-600">
            <Calendar className="w-4 h-4" />
          </div>
          <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">Report Scope:</span>
        </div>

        <div className="flex flex-wrap gap-1.5">
          {(['today', 'yesterday', 'last7', 'last30', 'custom'] as DateFilterOption[]).map((f) => (
            <button
              key={f}
              onClick={() => setDateFilter(f)}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold border transition ${
                dateFilter === f
                  ? 'bg-blue-600 text-white border-blue-700 shadow-xs'
                  : 'bg-slate-100 text-slate-700 border-slate-200 hover:bg-slate-200'
              }`}
            >
              {f === 'today'
                ? 'Today'
                : f === 'yesterday'
                ? 'Yesterday'
                : f === 'last7'
                ? 'Last 7 Days'
                : f === 'last30'
                ? 'Last 30 Days'
                : 'Custom Date Range'}
            </button>
          ))}
        </div>
      </div>

      {/* 3 Download Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Visitors Report Card */}
        <div className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] flex flex-col justify-between hover:shadow-md hover:border-blue-300 transition group">
          <div>
            <div className="w-12 h-12 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform shadow-2xs">
              <Users className="w-6 h-6" />
            </div>
            <h3 className="font-extrabold text-slate-900 text-lg font-['Outfit']">Visitor Session Registry</h3>
            <p className="text-xs text-slate-500 mt-1">
              Detailed entry timestamps, exit timestamps, dwell duration, test ride / booking tags, and notes for all{' '}
              {visitors.length} logged sessions.
            </p>
            <div className="mt-5 p-3.5 bg-slate-50/80 rounded-2xl border border-slate-200/80 text-xs text-slate-600 space-y-1.5">
              <div className="flex justify-between">
                <span>Total records:</span>
                <span className="font-extrabold text-slate-900 font-mono">{visitors.length}</span>
              </div>
              <div className="flex justify-between">
                <span>Tagged with purpose:</span>
                <span className="font-extrabold text-emerald-600 font-mono">{metrics.taggedCount}</span>
              </div>
              <div className="flex justify-between">
                <span>Untagged:</span>
                <span className="font-extrabold text-amber-600 font-mono">{metrics.untaggedCount}</span>
              </div>
            </div>
          </div>

          <button
            onClick={handleDownloadVisitors}
            className="mt-6 w-full py-2.5 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-xs font-bold border border-blue-400/30 shadow-md shadow-blue-500/20 flex items-center justify-center gap-2 transition active:scale-[0.98]"
          >
            <Download className="w-4 h-4" />
            <span>Download Visitors CSV</span>
          </button>
        </div>

        {/* Employee Attendance Report Card */}
        <div className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] flex flex-col justify-between hover:shadow-md hover:border-emerald-300 transition group">
          <div>
            <div className="w-12 h-12 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform shadow-2xs">
              <Clock className="w-6 h-6" />
            </div>
            <h3 className="font-extrabold text-slate-900 text-lg font-['Outfit']">Employee Attendance Report</h3>
            <p className="text-xs text-slate-500 mt-1">
              Daily staff check-ins, check-outs, shift durations, verification source (Biometric/Manual), and department allocations.
            </p>
            <div className="mt-5 p-3.5 bg-slate-50/80 rounded-2xl border border-slate-200/80 text-xs text-slate-600 space-y-1.5">
              <div className="flex justify-between">
                <span>Total employees:</span>
                <span className="font-extrabold text-slate-900 font-mono">{attendance.length}</span>
              </div>
              <div className="flex justify-between">
                <span>Present & Late:</span>
                <span className="font-extrabold text-emerald-600 font-mono">{metrics.employeesOnDuty}</span>
              </div>
              <div className="flex justify-between">
                <span>Absent / Leave:</span>
                <span className="font-extrabold text-rose-600 font-mono">{metrics.employeesAbsent + metrics.employeesOnLeave}</span>
              </div>
            </div>
          </div>

          <button
            onClick={handleDownloadAttendance}
            className="mt-6 w-full py-2.5 px-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white rounded-xl text-xs font-bold border border-emerald-400/30 shadow-md shadow-emerald-500/20 flex items-center justify-center gap-2 transition active:scale-[0.98]"
          >
            <Download className="w-4 h-4" />
            <span>Download Attendance CSV</span>
          </button>
        </div>

        {/* Executive KPI Summary */}
        <div className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] flex flex-col justify-between hover:shadow-md hover:border-purple-300 transition group">
          <div>
            <div className="w-12 h-12 rounded-2xl bg-purple-50 text-purple-600 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform shadow-2xs">
              <TrendingUp className="w-6 h-6" />
            </div>
            <h3 className="font-extrabold text-slate-900 text-lg font-['Outfit']">Executive Footfall Summary</h3>
            <p className="text-xs text-slate-500 mt-1">
              Executive high-level summary including peak hours footfall, average visit dwell time, and day-over-day growth.
            </p>
            <div className="mt-5 p-3.5 bg-slate-50/80 rounded-2xl border border-slate-200/80 text-xs text-slate-600 space-y-1.5">
              <div className="flex justify-between">
                <span>Peak Hour:</span>
                <span className="font-extrabold text-slate-900 font-mono">{metrics.busiestHour}</span>
              </div>
              <div className="flex justify-between">
                <span>Peak Count:</span>
                <span className="font-extrabold text-slate-900 font-mono">{metrics.busiestHourCount} visitors</span>
              </div>
              <div className="flex justify-between">
                <span>Avg Duration:</span>
                <span className="font-extrabold text-blue-700 font-mono">{metrics.avgDurationMinutes} mins</span>
              </div>
            </div>
          </div>

          <button
            onClick={handleDownloadSummary}
            className="mt-6 w-full py-2.5 px-4 bg-gradient-to-r from-slate-900 to-slate-800 hover:from-black hover:to-slate-900 text-white rounded-xl text-xs font-bold border border-slate-700/50 shadow-md shadow-slate-900/20 flex items-center justify-center gap-2 transition active:scale-[0.98]"
          >
            <Download className="w-4 h-4" />
            <span>Download Executive KPI CSV</span>
          </button>
        </div>
      </div>
    </div>
  );
};
