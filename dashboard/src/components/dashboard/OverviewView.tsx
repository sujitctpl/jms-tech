import React, { useState } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import { useAuth } from '../../context/AuthContext';
import {
  Users,
  UserCheck,
  TrendingUp,
  Clock,
  Tags,
  Timer,
  ArrowUpRight,
  Sparkles,
  AlertTriangle,
  Info,
  Calendar,
  ChevronRight,
  Flame,
  CheckCircle2,
  Camera,
  Maximize2,
  Eye,
  Activity,
  X,
  Layers,
  BarChart3,
  LineChart,
  ShieldCheck,
  PlusCircle,
} from 'lucide-react';
import { ActiveTab } from '../layout/Sidebar';
import { VisitorRecord } from '../../types';
import { PURPOSE_COLORS } from '../../data/seedData';

interface OverviewViewProps {
  onNavigateTab: (tab: ActiveTab) => void;
  onTagVisitor: (visitor: VisitorRecord) => void;
  onOpenQuickAdd: () => void;
}

const YESTERDAY_HOURLY = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const WEEK_AVG_HOURLY = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

// Catmull-Rom to Cubic Bezier curve for ultra-smooth neon wave SVG
const getSvgPath = (pts: { x: number; y: number }[]): string => {
  if (pts.length === 0) return '';
  if (pts.length === 1) return `M ${pts[0].x} ${pts[0].y}`;
  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i === 0 ? 0 : i - 1];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2 < pts.length ? i + 2 : i + 1];
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  return d;
};

export const OverviewView: React.FC<OverviewViewProps> = ({
  onNavigateTab,
  onTagVisitor,
  onOpenQuickAdd,
}) => {
  const { user } = useAuth();
  const {
    metrics,
    storeHours,
    isStoreCurrentlyOpen,
    visitors,
    attendance,
    quickCheckInEmployee,
    quickCheckOutEmployee,
  } = useShowroom();

  // Interactive Chart States (Neon Wave Curve / Dual Comparative Pillars / Stacked)
  const [selectedHourIndex, setSelectedHourIndex] = useState<number>(8); // Default peak (5-6 PM)
  const [hoveredHourIndex, setHoveredHourIndex] = useState<number | null>(null);
  const [chartStyle, setChartStyle] = useState<'wave' | 'dual' | 'stacked'>('wave');
  const [chartTimeframe, setChartTimeframe] = useState<'today' | 'yesterday' | 'week'>('today');
  const [chartViewMode, setChartViewMode] = useState<'combined' | 'new' | 'returning'>('combined');

  // AI Face Snapshot Inspection Modal
  const [inspectedVisitor, setInspectedVisitor] = useState<VisitorRecord | null>(null);

  // Average Time per Visitor Period Filter State
  const [durationPeriod, setDurationPeriod] = useState<'today' | 'prev_week' | 'prev_month'>('today');

  const selectedHourData = metrics.hourlyChartData[selectedHourIndex] || metrics.hourlyChartData[8];
  const maxHourlyVisitors = Math.max(...metrics.hourlyChartData.map((d) => d.visitors), 1);

  // Period-based average time metrics
  const durationStats = {
    today: {
      avg: metrics.avgDurationMinutes,
      median: metrics.medianDurationMinutes,
      delta: '0.0%',
      isPositive: true,
      comparisonText: 'Live real-time showroom average',
      finishedVisits: visitors.filter((v) => v.exitTime).length,
      longest: metrics.longestDurationMinutes,
      shortest: metrics.shortestDurationMinutes,
      statusText: metrics.totalVisitors > 0 ? 'Live customer engagement tracking' : "Awaiting today's showroom entries",
    },
    prev_week: {
      avg: 0,
      median: 0,
      delta: '0.0%',
      isPositive: false,
      comparisonText: 'Historical comparison',
      finishedVisits: 0,
      longest: 0,
      shortest: 0,
      statusText: 'No historical week records logged yet',
    },
    prev_month: {
      avg: 0,
      median: 0,
      delta: 'Baseline',
      isPositive: false,
      comparisonText: 'Historical baseline',
      finishedVisits: 0,
      longest: 0,
      shortest: 0,
      statusText: 'No historical month records logged yet',
    },
  }[durationPeriod];

  return (
    <div className="space-y-6 pb-12">
      {/* Top Welcome Banner (Clean, Executive Enterprise Styling) */}
      <div className="bg-slate-900 rounded-3xl p-6 sm:p-7 text-white shadow-md border border-slate-700/80 relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-5 relative z-10">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-3 py-1 rounded-full text-[11px] font-bold bg-slate-800 text-slate-200 border border-slate-700 flex items-center gap-2 shadow-2xs">
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                Capermint Showroom Operations Console
              </span>
              <span className="text-xs text-slate-400 font-medium">• Administrator Console</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold font-['Outfit'] tracking-tight text-white">
              Today&apos;s Showroom Footfall & Activity
            </h1>
            <p className="text-xs sm:text-sm text-slate-300 mt-1 max-w-2xl leading-relaxed">
              Real-time monitoring of customer footfall, floor attendance, peak visiting hours, dwell duration, and visit intents for{' '}
              <strong className="text-white font-semibold">Capermint Showroom</strong>.
            </p>
          </div>

          <div className="flex items-center gap-3 shrink-0 self-start md:self-center">
            <button
              onClick={() => onNavigateTab('reports')}
              className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-semibold border border-slate-700 hover:border-slate-600 shadow-xs transition-all active:scale-[0.98]"
            >
              Export CSV Reports
            </button>
            <button
              onClick={onOpenQuickAdd}
              className="px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-semibold border border-blue-500 shadow-md shadow-blue-600/20 transition-all flex items-center gap-2 active:scale-[0.98]"
            >
              <PlusCircle className="w-4 h-4 text-white" />
              <span>Log Walk-in</span>
            </button>
          </div>
        </div>
      </div>

      {/* Untagged Alert Banner if any exist */}
      {metrics.untaggedCount > 0 && (
        <div className="bg-gradient-to-r from-amber-50 to-orange-50/60 border border-amber-200/90 rounded-2xl p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-xs animate-fadeIn">
          <div className="flex items-start gap-3">
            <div className="p-2.5 bg-amber-100/80 text-amber-800 rounded-xl shrink-0 mt-0.5 border border-amber-200">
              <AlertTriangle className="w-5 h-5 text-amber-700" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-amber-950 font-['Outfit'] flex items-center gap-2">
                {metrics.untaggedCount} Untagged Visitors Awaiting Purpose Assignment
                <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
              </h2>
              <p className="text-xs text-amber-800 mt-0.5">
                Assign visit purposes (Test Ride, Vehicle Enquiry, Booking, Service) to ensure showroom leads are captured.
              </p>
            </div>
          </div>
          <button
            onClick={() => onNavigateTab('visit-purpose')}
            className="px-4 py-2 bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold rounded-xl border border-amber-700 shadow-xs transition-all shrink-0 active:scale-[0.98]"
          >
            Review & Tag Visitors
          </button>
        </div>
      )}

      {/* ========================================================================= */}
      {/* SECTION A: VISITORS COUNT WITH IMPROVED INTERACTIVE BAR GRAPH */}
      {/* ========================================================================= */}
      <section className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)]">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-3 mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-2xl bg-blue-50 text-blue-600 border border-blue-100 shadow-2xs">
              <Users className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-slate-900 font-['Outfit']">
                  A. Visitors Count
                </h2>
                <span className="text-[10px] font-bold bg-blue-100/80 text-blue-800 px-2.5 py-0.5 rounded-full border border-blue-200/60">
                  Real-time Tracking
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">
                Total footfall, new vs. returning breakdown, and interactive hourly distribution
              </p>
            </div>
          </div>
          <button
            onClick={() => onNavigateTab('visitors')}
            className="text-xs text-blue-600 hover:text-blue-800 font-bold flex items-center gap-1 self-start sm:self-auto px-3 py-1.5 rounded-xl hover:bg-blue-50 transition border border-transparent hover:border-blue-200"
          >
            <span>View Full Visitor Registry</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* 4 Elevated KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5 mb-6">
          {/* Total Visitors Today */}
          <div className="p-5 rounded-2xl bg-gradient-to-br from-blue-50/70 via-slate-50/40 to-white border border-blue-200/70 shadow-xs hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 group">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-blue-900 uppercase tracking-wider">Total Visitors Today</span>
              <div className="p-2 rounded-xl bg-blue-100/70 text-blue-700 border border-blue-200/60 group-hover:scale-110 transition-transform">
                <Users className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl sm:text-4xl font-extrabold text-slate-900 font-['Outfit'] tracking-tight">
                {metrics.totalVisitors}
              </span>
              <span className="text-xs font-bold text-slate-600">walk-ins</span>
            </div>
            <div className="mt-3 flex items-center text-xs text-emerald-700 font-bold gap-1 bg-emerald-50 px-2 py-0.5 rounded-lg border border-emerald-200/60 w-fit">
              <ArrowUpRight className="w-3.5 h-3.5" />
              <span>+{Math.abs(metrics.comparisonYesterdayPct)}% vs yesterday</span>
            </div>
          </div>

          {/* New Visitors */}
          <div className="p-5 rounded-2xl bg-gradient-to-br from-emerald-50/70 via-slate-50/40 to-white border border-emerald-200/70 shadow-xs hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 group">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-emerald-950 uppercase tracking-wider">New Visitors</span>
              <div className="p-2 rounded-xl bg-emerald-100/70 text-emerald-700 border border-emerald-200/60 group-hover:scale-110 transition-transform">
                <Sparkles className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl sm:text-4xl font-extrabold text-slate-900 font-['Outfit'] tracking-tight">
                {metrics.newVisitors}
              </span>
              <span className="text-xs font-bold text-emerald-800 bg-emerald-100/80 px-2 py-0.5 rounded-md border border-emerald-200/60">
                {metrics.newVisitorPct}% share
              </span>
            </div>
            <p className="text-[11px] text-slate-600 font-medium mt-3">First-time showroom walk-ins</p>
          </div>

          {/* Returning Visitors */}
          <div className="p-5 rounded-2xl bg-gradient-to-br from-indigo-50/70 via-slate-50/40 to-white border border-indigo-200/70 shadow-xs hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 group">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-indigo-950 uppercase tracking-wider">Returning Visitors</span>
              <div className="p-2 rounded-xl bg-indigo-100/70 text-indigo-700 border border-indigo-200/60 group-hover:scale-110 transition-transform">
                <UserCheck className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl sm:text-4xl font-extrabold text-slate-900 font-['Outfit'] tracking-tight">
                {metrics.returningVisitors}
              </span>
              <span className="text-xs font-bold text-indigo-800 bg-indigo-100/80 px-2 py-0.5 rounded-md border border-indigo-200/60">
                {metrics.returningVisitorPct}% share
              </span>
            </div>
            <p className="text-[11px] text-slate-600 font-medium mt-3">Repeat visits & test ride follow-ups</p>
          </div>

          {/* Comparison with Yesterday */}
          <div className="p-5 rounded-2xl bg-gradient-to-br from-amber-50/70 via-slate-50/40 to-white border border-amber-200/70 shadow-xs hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 group">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-amber-950 uppercase tracking-wider">Day-over-Day</span>
              <div className="p-2 rounded-xl bg-amber-100/70 text-amber-700 border border-amber-200/60 group-hover:scale-110 transition-transform">
                <TrendingUp className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl sm:text-4xl font-extrabold text-slate-900 font-['Outfit'] tracking-tight">
                {metrics.comparisonYesterdayPct >= 0 ? `+${metrics.comparisonYesterdayPct}%` : `${metrics.comparisonYesterdayPct}%`}
              </span>
              <span className="text-xs font-semibold text-slate-600">(0 y&apos;day)</span>
            </div>
            <p className="text-[11px] text-slate-600 font-medium mt-3">Steady day-over-day showroom growth</p>
          </div>
        </div>

        {/* Dynamic Multi-Zone Footfall & Section Analytics Banner */}
        <div className="mb-6 p-4 sm:p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-850 to-slate-900 border border-slate-700/80 shadow-md text-white">
          <div className="flex flex-col md:flex-row md:items-center justify-between pb-3.5 border-b border-slate-800 gap-3">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-blue-500/20 text-blue-400 border border-blue-500/30">
                <Layers className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm sm:text-base font-bold font-['Outfit'] text-white">
                    Multi-Zone Dynamic Footfall & Section Tracking
                  </h3>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 animate-pulse">
                    LIVE EDGE AI
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  Dynamic Entry Line triggers Attendance = 1. Moving into Sales or Accounts increments section metrics independently.
                </p>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 bg-slate-800/80 px-3 py-1.5 rounded-xl border border-slate-700 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span>Interactive Calibration: Press <kbd className="px-1.5 py-0.5 bg-slate-700 text-cyan-300 rounded font-mono text-[10px]">C</kbd> in detector window</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5 mt-4">
            {/* Main Entrance Attendance Card */}
            <div className="p-4 rounded-xl bg-slate-800/70 border border-slate-700/80 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span className="text-xs font-bold text-slate-300 uppercase tracking-wide">Main Entry Attendance</span>
                </div>
                <div className="text-2xl sm:text-3xl font-extrabold font-['Outfit'] text-white mt-1">
                  {metrics.totalVisitors}
                </div>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  Crossed Virtual Line
                </div>
              </div>
              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                <Users className="w-6 h-6" />
              </div>
            </div>

            {/* Sales Section Card */}
            <div className="p-4 rounded-xl bg-slate-800/70 border border-amber-500/30 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-amber-400" />
                  <span className="text-xs font-bold text-amber-300 uppercase tracking-wide">Sales Section</span>
                </div>
                <div className="text-2xl sm:text-3xl font-extrabold font-['Outfit'] text-amber-400 mt-1">
                  {metrics.sectionCounts?.sales || 0}
                </div>
                <div className="text-[11px] text-slate-400 mt-0.5 flex items-center gap-1.5">
                  <span>Cumulative Visits</span>
                  <span className="text-slate-600">•</span>
                  <span className="text-amber-300 font-semibold">{metrics.sectionOccupancy?.sales || 0} active now</span>
                </div>
              </div>
              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
                <Flame className="w-6 h-6" />
              </div>
            </div>

            {/* Accounts Section Card */}
            <div className="p-4 rounded-xl bg-slate-800/70 border border-blue-500/30 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-400" />
                  <span className="text-xs font-bold text-blue-300 uppercase tracking-wide">Accounts Section</span>
                </div>
                <div className="text-2xl sm:text-3xl font-extrabold font-['Outfit'] text-blue-400 mt-1">
                  {metrics.sectionCounts?.accounts || 0}
                </div>
                <div className="text-[11px] text-slate-400 mt-0.5 flex items-center gap-1.5">
                  <span>Cumulative Visits</span>
                  <span className="text-slate-600">•</span>
                  <span className="text-blue-300 font-semibold">{metrics.sectionOccupancy?.accounts || 0} active now</span>
                </div>
              </div>
              <div className="p-3 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400">
                <BarChart3 className="w-6 h-6" />
              </div>
            </div>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* OPTIMIZED CONTINUOUS HOURLY FOOTFALL HISTOGRAM & TELEMETRY */}
        {/* ========================================================================= */}
        <div className="bg-white rounded-2xl p-4 sm:p-6 border border-slate-200/90 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)]">
          {/* Histogram Header & Statistical Telemetry Ribbon */}
          <div className="flex flex-col lg:flex-row lg:items-center justify-between pb-4 border-b border-slate-100 gap-3">
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs font-extrabold uppercase tracking-wider text-slate-900 font-['Outfit'] flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-blue-600" />
                  Hourly Footfall Frequency Histogram
                </span>
                <span className="text-[10px] bg-blue-100/90 text-blue-800 px-2 py-0.5 rounded-full font-bold border border-blue-200/70">
                  Continuous Bins [09:00 - 20:00 IST]
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Distribution of showroom walk-in frequency across contiguous 60-minute time intervals.
              </p>
            </div>

            {/* Histogram View Mode Toggles */}
            <div className="flex items-center bg-slate-100/90 p-1 rounded-xl border border-slate-200 shadow-2xs self-start lg:self-center">
              <button
                type="button"
                onClick={() => setChartViewMode('combined')}
                className={`px-3 py-1.5 text-xs font-bold rounded-lg transition ${
                  chartViewMode === 'combined'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Density Histogram
              </button>
              <button
                type="button"
                onClick={() => setChartViewMode('new')}
                className={`px-3 py-1.5 text-xs font-bold rounded-lg transition ${
                  chartViewMode === 'new'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                New Walk-ins
              </button>
              <button
                type="button"
                onClick={() => setChartViewMode('returning')}
                className={`px-3 py-1.5 text-xs font-bold rounded-lg transition ${
                  chartViewMode === 'returning'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Returning Cohort
              </button>
            </div>
          </div>

          {/* Statistical Distribution Parameters Strip */}
          {(() => {
            const totalObs = metrics.hourlyChartData.reduce((acc, d) => acc + d.visitors, 0);
            const meanVal = (totalObs / (metrics.hourlyChartData.length || 1)).toFixed(1);
            const sortedVals = [...metrics.hourlyChartData.map((d) => d.visitors)].sort((a, b) => a - b);
            const medianVal = sortedVals[Math.floor(sortedVals.length / 2)] || 13;
            const variance =
              metrics.hourlyChartData.reduce((acc, d) => acc + Math.pow(d.visitors - Number(meanVal), 2), 0) /
              (metrics.hourlyChartData.length || 1);
            const stdDev = Math.sqrt(variance).toFixed(1);

            return (
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 my-4 p-2.5 bg-slate-50/80 rounded-xl border border-slate-200/70 text-xs">
                <div className="px-2 py-1 border-r border-slate-200/80">
                  <span className="text-[10px] font-bold uppercase text-slate-600 block">Sample Size (N)</span>
                  <span className="font-extrabold text-slate-900 text-sm font-['Outfit']">{totalObs} visitors</span>
                </div>
                <div className="px-2 py-1 border-r border-slate-200/80">
                  <span className="text-[10px] font-bold uppercase text-slate-600 block">Mean (μ)</span>
                  <span className="font-extrabold text-blue-700 text-sm font-['Outfit']">{meanVal} / hr</span>
                </div>
                <div className="px-2 py-1 border-r border-slate-200/80">
                  <span className="text-[10px] font-bold uppercase text-slate-600 block">Median (M)</span>
                  <span className="font-extrabold text-indigo-700 text-sm font-['Outfit']">{medianVal} / hr</span>
                </div>
                <div className="px-2 py-1 border-r border-slate-200/80">
                  <span className="text-[10px] font-bold uppercase text-slate-600 block">Std Dev (σ)</span>
                  <span className="font-extrabold text-slate-700 text-sm font-['Outfit]">±{stdDev}</span>
                </div>
                <div className="px-2 py-1 col-span-2 sm:col-span-1">
                  <span className="text-[10px] font-bold uppercase text-slate-600 block">Peak Mode Window</span>
                  <span className="font-extrabold text-red-600 text-sm font-['Outfit'] flex items-center gap-1">
                    <Flame className="w-3.5 h-3.5 text-red-500 animate-pulse" />
                    17:00 - 18:00 (f=28)
                  </span>
                </div>
              </div>
            );
          })()}

          {/* Interactive Inspection Card for Active Bin */}
          <div className="mb-4 p-3.5 bg-gradient-to-r from-blue-50/90 via-slate-50 to-indigo-50/70 rounded-2xl border border-blue-200 shadow-2xs flex flex-col md:flex-row md:items-center justify-between gap-3 transition-all">
            <div className="flex items-center gap-3">
              <div
                className={`w-11 h-11 rounded-xl flex items-center justify-center font-bold text-white shadow-xs shrink-0 ${
                  selectedHourData.visitors === metrics.busiestHourCount
                    ? 'bg-gradient-to-tr from-red-600 to-amber-500 shadow-red-500/20'
                    : 'bg-gradient-to-tr from-blue-600 to-indigo-600 shadow-blue-500/20'
                }`}
              >
                {selectedHourData.visitors === metrics.busiestHourCount ? (
                  <Flame className="w-5 h-5 text-white animate-pulse" />
                ) : (
                  <Clock className="w-5 h-5 text-white" />
                )}
              </div>
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-extrabold text-slate-900 text-sm sm:text-base font-['Outfit']">
                    Interval: {selectedHourData.hour}
                  </span>
                  {selectedHourData.visitors === metrics.busiestHourCount ? (
                    <span className="px-2 py-0.5 bg-red-100 text-red-700 text-[10px] font-extrabold rounded-full border border-red-200 flex items-center gap-1 shadow-2xs">
                      <Flame className="w-3 h-3 text-red-600" /> Peak Surge Mode Bin
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 bg-blue-100/80 text-blue-700 text-[10px] font-bold rounded-full border border-blue-200/60">
                      Standard Bin Interval
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-600 mt-0.5 flex-wrap">
                  <span><strong>{selectedHourData.visitors}</strong> Visitors (Frequency)</span>
                  <span>•</span>
                  <span className="text-blue-700 font-semibold">{selectedHourData.newVisitors} New ({selectedHourData.visitors > 0 ? Math.round((selectedHourData.newVisitors / selectedHourData.visitors) * 100) : 0}%)</span>
                  <span>•</span>
                  <span className="text-indigo-700 font-semibold">{selectedHourData.returningVisitors} Returning</span>
                  <span>•</span>
                  <span className="text-slate-500 font-medium">Share of Total: {Math.round((selectedHourData.visitors / Math.max(metrics.totalVisitors, 1)) * 100)}%</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4 bg-white p-2.5 px-4 rounded-xl border border-slate-200 shadow-2xs self-start md:self-center">
              <div>
                <span className="text-[10px] font-bold text-slate-500 uppercase block tracking-wider">Test Rides</span>
                <span className="text-xs sm:text-sm font-extrabold text-slate-900 font-['Outfit']">{selectedHourData.testRides} booked</span>
              </div>
              <div className="h-6 w-px bg-slate-200" />
              <div>
                <span className="text-[10px] font-bold text-slate-500 uppercase block tracking-wider">Deliveries</span>
                <span className="text-xs sm:text-sm font-extrabold text-slate-900 font-['Outfit']">{selectedHourData.deliveries} handovers</span>
              </div>
            </div>
          </div>

          {/* Continuous Histogram Canvas with Y-Axis and Density Polyline */}
          <div className="relative pt-4 pb-2">
            <div className="flex">
              {/* Left Y-Axis Scale */}
              <div className="w-8 shrink-0 flex flex-col justify-between text-[10px] font-bold text-slate-400 font-mono pr-2 h-60 select-none pb-7">
                <span>30</span>
                <span>20</span>
                <span>10</span>
                <span>0</span>
              </div>

              {/* Main Histogram Bin Canvas */}
              <div className="relative flex-1 h-60">
                {/* Horizontal Scale Gridlines */}
                <div className="absolute inset-x-0 top-1 border-b border-dashed border-slate-200/90" />
                <div className="absolute inset-x-0 top-[33%] border-b border-dashed border-slate-200/90" />
                <div className="absolute inset-x-0 top-[66%] border-b border-dashed border-slate-200/90" />
                <div className="absolute inset-x-0 bottom-7 border-b border-slate-300" />

                {/* Contiguous Histogram Bins */}
                <div className="h-full w-full flex items-end pb-7 relative z-10">
                  {metrics.hourlyChartData.map((slot, index) => {
                    const isSelected = selectedHourIndex === index;
                    const isPeak = slot.visitors === metrics.busiestHourCount;
                    const maxScale = Math.max(maxHourlyVisitors, 30);

                    const displayVal =
                      chartViewMode === 'new'
                        ? slot.newVisitors
                        : chartViewMode === 'returning'
                        ? slot.returningVisitors
                        : slot.visitors;

                    const heightPct = Math.round((displayVal / maxScale) * 100);
                    const newRatio = slot.visitors > 0 ? (slot.newVisitors / slot.visitors) * 100 : 50;

                    return (
                      <div
                        key={index}
                        onClick={() => setSelectedHourIndex(index)}
                        className={`flex-1 h-full flex flex-col items-stretch justify-end cursor-pointer group relative select-none border-r border-slate-200/90 last:border-r-0 transition ${
                          isSelected ? 'bg-blue-50/60' : 'hover:bg-slate-50/70'
                        }`}
                      >
                        {/* Top Frequency Badge */}
                        <div className="text-center mb-1 transition-all">
                          <span
                            className={`text-[11px] font-mono font-bold inline-block px-1 rounded ${
                              isSelected
                                ? 'bg-blue-600 text-white font-extrabold scale-110 shadow-xs'
                                : isPeak
                                ? 'bg-red-100 text-red-700 font-extrabold border border-red-200'
                                : 'text-slate-600 group-hover:text-slate-900 group-hover:font-extrabold'
                            }`}
                          >
                            {displayVal}
                          </span>
                        </div>

                        {/* Contiguous Solid Histogram Column */}
                        <div
                          className={`w-full overflow-hidden flex flex-col justify-end transition-all border-t-2 relative ${
                            isPeak
                              ? 'border-t-red-600 bg-gradient-to-t from-red-600/90 via-rose-500/80 to-amber-400/90 shadow-xs'
                              : isSelected
                              ? 'border-t-blue-700 bg-gradient-to-t from-blue-700 via-blue-600 to-sky-400 shadow-xs'
                              : 'border-t-blue-500 bg-gradient-to-t from-blue-600/85 via-blue-500/75 to-sky-400/70 group-hover:brightness-105'
                          }`}
                          style={{ height: `${Math.max(heightPct, 6)}%` }}
                        >
                          {chartViewMode === 'combined' && (
                            <>
                              {/* Stacked Segment Divider */}
                              <div
                                className="w-full bg-indigo-600/90 border-b border-white/20 transition-all"
                                style={{ height: `${100 - newRatio}%` }}
                                title={`Returning: ${slot.returningVisitors}`}
                              />
                              <div
                                className="w-full transition-all"
                                style={{ height: `${newRatio}%` }}
                                title={`New: ${slot.newVisitors}`}
                              />
                            </>
                          )}
                        </div>

                        {/* Active Crosshair Indicator */}
                        {isSelected && (
                          <div className="absolute top-0 bottom-7 left-1/2 -translate-x-1/2 w-0.5 bg-blue-600 pointer-events-none" />
                        )}

                        {/* Continuous Bin Interval Label at Bottom */}
                        <div className="absolute -bottom-7 inset-x-0 text-center">
                          <span
                            className={`text-[10px] font-mono font-semibold block truncate ${
                              isSelected ? 'text-blue-700 font-extrabold' : 'text-slate-500 group-hover:text-slate-900'
                            }`}
                          >
                            {slot.hour.split(' ')[0]}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Histogram Axis Title & Legend Footer */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pt-8 mt-2 border-t border-slate-200 text-xs text-slate-600 gap-2">
              <div className="flex items-center gap-4 flex-wrap">
                <span className="flex items-center gap-1.5 font-semibold text-blue-700">
                  <span className="w-3 h-3 rounded-xs bg-blue-600 inline-block shadow-2xs" />
                  New Walk-ins
                </span>
                <span className="flex items-center gap-1.5 font-semibold text-indigo-700">
                  <span className="w-3 h-3 rounded-xs bg-indigo-600 inline-block shadow-2xs" />
                  Returning Customers
                </span>
                <span className="flex items-center gap-1.5 font-semibold text-red-600">
                  <span className="w-3 h-3 rounded-xs bg-gradient-to-t from-red-600 to-amber-400 inline-block shadow-2xs" />
                  Peak Frequency Bin (17:00 - 18:00)
                </span>
              </div>
              <span className="text-[11px] text-slate-500 font-mono">
                Continuous Bin Width: 1 Hour (Δt = 60 min)
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}

      {/* ========================================================================= */}
      {/* 2-COLUMN GRID: SECTION B (ATTENDANCE) & SECTION C (PEAK HOURS) */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* SECTION B: EMPLOYEE ATTENDANCE TODAY */}
        <section className="lg:col-span-7 bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-5">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-2xl bg-emerald-50 text-emerald-600 border border-emerald-100 shadow-2xs">
                  <UserCheck className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base sm:text-lg font-bold text-slate-900 font-['Outfit']">
                    B. Employee Attendance Today
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">Floor staff on duty, check-in timestamps, and shift status</p>
                </div>
              </div>
              <button
                onClick={() => onNavigateTab('attendance')}
                className="text-xs text-blue-600 hover:text-blue-800 font-bold flex items-center gap-1 px-3 py-1.5 rounded-xl hover:bg-blue-50 transition border border-transparent hover:border-blue-200"
              >
                <span>Manage Attendance</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* Attendance Mini KPI Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
              <div className="p-3 bg-slate-50/80 rounded-2xl border border-slate-200 text-center">
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Total Staff</p>
                <p className="text-2xl font-extrabold text-slate-900 font-['Outfit'] mt-0.5">{metrics.employeesTotal}</p>
              </div>
              <div className="p-3 bg-emerald-50/70 rounded-2xl border border-emerald-200 text-center">
                <p className="text-[10px] font-bold uppercase tracking-wider text-emerald-800">Present</p>
                <p className="text-2xl font-extrabold text-emerald-900 font-['Outfit'] mt-0.5">{metrics.employeesPresent}</p>
              </div>
              <div className="p-3 bg-amber-50/70 rounded-2xl border border-amber-200 text-center">
                <p className="text-[10px] font-bold uppercase tracking-wider text-amber-800">Late</p>
                <p className="text-2xl font-extrabold text-amber-900 font-['Outfit'] mt-0.5">{metrics.employeesLate}</p>
              </div>
              <div className="p-3 bg-rose-50/70 rounded-2xl border border-rose-200 text-center">
                <p className="text-[10px] font-bold uppercase tracking-wider text-rose-800">Absent / Leave</p>
                <p className="text-2xl font-extrabold text-rose-900 font-['Outfit'] mt-0.5">
                  {metrics.employeesAbsent + metrics.employeesOnLeave}
                </p>
              </div>
            </div>

            {/* Attendance Table */}
            <div className="overflow-x-auto rounded-2xl border border-slate-200/80 shadow-2xs">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-600 uppercase text-[10px] tracking-wider bg-slate-50/80 font-bold">
                    <th className="py-3 px-3.5">Employee</th>
                    <th className="py-3 px-2">Role & Dept</th>
                    <th className="py-3 px-2">Check-in</th>
                    <th className="py-3 px-2">Status</th>
                    <th className="py-3 px-3.5 text-right">Quick Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white">
                  {attendance.slice(0, 5).map((att) => {
                    const isPresent = att.status === 'Present';
                    const isLate = att.status === 'Late';

                    return (
                      <tr key={att.id} className="hover:bg-slate-50/80 transition-colors">
                        <td className="py-3 px-3.5">
                          <div className="flex items-center gap-2.5">
                            <div className="relative shrink-0">
                              <img
                                src={att.avatarUrl || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80'}
                                alt={att.employeeName}
                                className="w-8 h-8 rounded-full object-cover border border-slate-200 shadow-2xs"
                              />
                              <span
                                className={`w-2 h-2 rounded-full absolute bottom-0 right-0 ring-1 ring-white ${
                                  isPresent ? 'bg-emerald-500' : isLate ? 'bg-amber-500' : 'bg-rose-500'
                                }`}
                              />
                            </div>
                            <span className="font-bold text-slate-900 font-['Outfit']">{att.employeeName}</span>
                          </div>
                        </td>
                        <td className="py-3 px-2">
                          <p className="text-slate-800 font-semibold truncate max-w-[140px]">{att.role}</p>
                          <span className="text-[10px] text-slate-400 font-medium">{att.department}</span>
                        </td>
                        <td className="py-3 px-2 font-mono text-[11px] text-slate-700 font-medium">
                          {att.checkInTime || '—'}
                        </td>
                        <td className="py-3 px-2">
                          <span
                            className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold inline-flex items-center gap-1.5 border shadow-2xs ${
                              isPresent
                                ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                                : isLate
                                ? 'bg-amber-50 text-amber-800 border-amber-200'
                                : 'bg-rose-50 text-rose-800 border-rose-200'
                            }`}
                          >
                            <span
                              className={`w-1.5 h-1.5 rounded-full ${
                                isPresent ? 'bg-emerald-500' : isLate ? 'bg-amber-500' : 'bg-rose-500'
                              }`}
                            />
                            {att.status}
                          </span>
                        </td>
                        <td className="py-3 px-3.5 text-right">
                          {att.checkInTime && !att.checkOutTime ? (
                            <button
                              onClick={() => quickCheckOutEmployee(att.employeeId)}
                              className="px-2.5 py-1 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-[11px] font-semibold border border-slate-300 shadow-2xs transition active:scale-95"
                            >
                              Check Out
                            </button>
                          ) : !att.checkInTime ? (
                            <button
                              onClick={() => quickCheckInEmployee(att.employeeId)}
                              className="px-2.5 py-1 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-[11px] font-semibold border border-emerald-700 shadow-2xs transition active:scale-95"
                            >
                              Check In
                            </button>
                          ) : (
                            <span className="text-[10px] text-slate-400 font-medium">Shift Done</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <p className="mt-4 text-[11px] text-slate-500 flex items-center gap-1.5 pt-3 border-t border-slate-100">
            <Info className="w-3.5 h-3.5 text-slate-400 shrink-0" />
            <span>Attendance records are synced from biometric & verified manual overrides.</span>
          </p>
        </section>

        {/* SECTION C: PEAK HOURS TODAY */}
        <section className="lg:col-span-5 bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-5">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-2xl bg-amber-50 text-amber-600 border border-amber-100 shadow-2xs">
                  <TrendingUp className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base sm:text-lg font-bold text-slate-900 font-['Outfit']">C. Peak Hours Today</h2>
                  <p className="text-xs text-slate-500 mt-0.5">Busiest footfall window & day-part breakdown</p>
                </div>
              </div>
              <button
                onClick={() => onNavigateTab('peak-hours')}
                className="text-xs text-blue-600 hover:text-blue-800 font-bold flex items-center gap-1 px-3 py-1.5 rounded-xl hover:bg-blue-50 transition border border-transparent hover:border-blue-200"
              >
                <span>Details</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* Busiest Hour Highlight Box */}
            <div className="p-5 bg-gradient-to-r from-red-600 via-rose-600 to-amber-500 rounded-2xl text-white shadow-md shadow-red-500/20 mb-5 relative overflow-hidden">
              <div className="absolute right-0 top-0 w-32 h-full bg-white/10 blur-xl pointer-events-none" />
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-white/90">Busiest Hour of the Day</span>
                <span className="text-[10px] bg-white/20 px-2.5 py-0.5 rounded-full font-mono font-extrabold border border-white/20">
                  PEAK DEMAND
                </span>
              </div>
              <div className="flex items-baseline justify-between mt-3">
                <div>
                  <h3 className="text-2xl font-black font-['Outfit']">{metrics.busiestHour}</h3>
                  <p className="text-xs text-white/90 mt-0.5">Evening commuter & test ride peak</p>
                </div>
                <div className="text-right">
                  <span className="text-3xl sm:text-4xl font-black font-['Outfit']">{metrics.busiestHourCount}</span>
                  <span className="text-xs block text-white/90 font-semibold">visitors</span>
                </div>
              </div>
            </div>

            {/* Day-Part Distribution */}
            <div className="space-y-3">
              <p className="text-xs font-bold uppercase tracking-wider text-slate-600">
                Day-Part Footfall Distribution
              </p>

              {/* Morning */}
              <div className="p-3 bg-slate-50/80 rounded-2xl border border-slate-200/80 flex items-center justify-between text-xs hover:bg-white hover:shadow-2xs transition">
                <div>
                  <span className="font-bold text-slate-800 font-['Outfit']">Morning (9:00 AM - 1:00 PM)</span>
                  <p className="text-[10px] text-slate-500 mt-0.5">Service drop-offs & early corporate visits</p>
                </div>
                <div className="text-right">
                  <span className="font-bold text-slate-900 font-mono text-sm">{metrics.morningTraffic}</span>
                  <span className="text-[10px] text-slate-500 block font-semibold">
                    {Math.round((metrics.morningTraffic / (metrics.morningTraffic + metrics.afternoonTraffic + metrics.eveningTraffic)) * 100)}%
                  </span>
                </div>
              </div>

              {/* Afternoon */}
              <div className="p-3 bg-slate-50/80 rounded-2xl border border-slate-200/80 flex items-center justify-between text-xs hover:bg-white hover:shadow-2xs transition">
                <div>
                  <span className="font-bold text-slate-800 font-['Outfit']">Afternoon (1:00 PM - 5:00 PM)</span>
                  <p className="text-[10px] text-slate-500 mt-0.5">Spares buyers & casual walk-ins</p>
                </div>
                <div className="text-right">
                  <span className="font-bold text-slate-900 font-mono text-sm">{metrics.afternoonTraffic}</span>
                  <span className="text-[10px] text-slate-500 block font-semibold">
                    {Math.round((metrics.afternoonTraffic / (metrics.morningTraffic + metrics.afternoonTraffic + metrics.eveningTraffic)) * 100)}%
                  </span>
                </div>
              </div>

              {/* Evening */}
              <div className="p-3 bg-red-50/70 rounded-2xl border border-red-200/80 flex items-center justify-between text-xs hover:bg-white hover:shadow-2xs transition">
                <div>
                  <span className="font-bold text-red-950 font-['Outfit'] flex items-center gap-1.5">
                    <span>Evening (5:00 PM - 8:00 PM)</span>
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  </span>
                  <p className="text-[10px] text-red-700 mt-0.5">Family visits, test rides & delivery ceremonies</p>
                </div>
                <div className="text-right">
                  <span className="font-bold text-red-950 font-mono text-sm">{metrics.eveningTraffic}</span>
                  <span className="text-[10px] text-red-700 block font-bold">
                    {Math.round((metrics.eveningTraffic / (metrics.morningTraffic + metrics.afternoonTraffic + metrics.eveningTraffic)) * 100)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          <p className="mt-4 text-[11px] text-slate-500 pt-3 border-t border-slate-100">
            Calculated from verified customer arrival timestamps. Peak floor coverage recommended 5-8 PM.
          </p>
        </section>
      </div>

      {/* ========================================================================= */}
      {/* 2-COLUMN GRID: SECTION D (AVERAGE TIME PER VISITOR) & SECTION E (STORE OPERATING HOURS) */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* SECTION D: STREAMLINED AVERAGE TIME PER VISITOR WITH PERIOD FILTERS (7 cols) */}
        <section className="lg:col-span-7 bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] flex flex-col justify-between">
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-3 mb-5">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-2xl bg-cyan-50 text-cyan-600 border border-cyan-100 shadow-2xs">
                  <Timer className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base sm:text-lg font-bold text-slate-900 font-['Outfit']">
                    D. Average Time per Visitor
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">Customer dwell duration & showroom engagement pattern</p>
                </div>
              </div>

              {/* Period Filter Tabs: Today | Prev Week | Prev Month */}
              <div className="flex items-center bg-slate-100/90 p-1 rounded-2xl self-start sm:self-auto border border-slate-200/90 shadow-2xs">
                <button
                  type="button"
                  onClick={() => setDurationPeriod('today')}
                  className={`px-3 py-1.5 text-xs font-bold rounded-xl transition ${
                    durationPeriod === 'today'
                      ? 'bg-blue-600 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Today
                </button>
                <button
                  type="button"
                  onClick={() => setDurationPeriod('prev_week')}
                  className={`px-3 py-1.5 text-xs font-bold rounded-xl transition ${
                    durationPeriod === 'prev_week'
                      ? 'bg-blue-600 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Prev Week
                </button>
                <button
                  type="button"
                  onClick={() => setDurationPeriod('prev_month')}
                  className={`px-3 py-1.5 text-xs font-bold rounded-xl transition ${
                    durationPeriod === 'prev_month'
                      ? 'bg-blue-600 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Prev Month
                </button>
              </div>
            </div>

            {/* Primary Hero Average Duration Display */}
            <div className="p-5 sm:p-6 bg-gradient-to-br from-blue-50/90 via-indigo-50/30 to-white rounded-2xl border border-blue-200/80 mb-5 relative overflow-hidden">
              <div className="absolute right-0 top-0 w-36 h-full bg-blue-400/5 blur-2xl pointer-events-none" />
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold uppercase tracking-wider text-blue-900">
                  {durationPeriod === 'today'
                    ? "Today's Average Duration"
                    : durationPeriod === 'prev_week'
                    ? 'Previous Week Benchmark'
                    : 'Previous Month Benchmark'}
                </span>
                <span className="text-[10px] bg-blue-100/80 text-blue-800 font-extrabold px-2.5 py-0.5 rounded-full border border-blue-200/60 shadow-2xs">
                  Verified Completed Visits
                </span>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-3 mt-3">
                <div className="flex items-baseline gap-2">
                  <span className="text-4xl sm:text-5xl font-black font-['Outfit'] text-slate-900">
                    {durationStats.avg}
                  </span>
                  <span className="text-base font-semibold text-slate-500">minutes / visitor</span>
                </div>

                <div className="flex items-center gap-1.5 text-xs font-extrabold text-emerald-800 bg-emerald-100/90 px-3 py-1 rounded-xl border border-emerald-200/80 self-start sm:self-auto shadow-2xs">
                  {durationStats.isPositive && <ArrowUpRight className="w-4 h-4 text-emerald-600" />}
                  <span>{durationStats.delta}</span>
                </div>
              </div>

              <p className="text-xs text-slate-600 mt-3 flex items-center gap-2 pt-2 border-t border-blue-100/80">
                <Info className="w-3.5 h-3.5 text-blue-600 shrink-0" />
                <span>{durationStats.comparisonText}</span>
              </p>
            </div>

            {/* 3 Secondary Clean Stat Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
              <div className="p-3.5 bg-slate-50/80 rounded-2xl border border-slate-200/80 text-center hover:bg-white transition shadow-2xs">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Median Duration</span>
                <p className="text-2xl font-extrabold font-['Outfit'] text-slate-900 mt-1">
                  {durationStats.median} <span className="text-xs font-medium text-slate-500">m</span>
                </p>
                <span className="text-[10px] text-slate-500 font-medium">50th percentile</span>
              </div>

              <div className="p-3.5 bg-blue-50/60 rounded-2xl border border-blue-200/70 text-center hover:bg-white transition shadow-2xs">
                <span className="text-[10px] font-bold uppercase tracking-wider text-blue-700">Completed Sessions</span>
                <p className="text-2xl font-extrabold font-['Outfit'] text-blue-900 mt-1">
                  {durationStats.finishedVisits}
                </p>
                <span className="text-[10px] text-blue-700 font-medium">verified exits</span>
              </div>

              <div className="p-3.5 bg-slate-50/80 rounded-2xl border border-slate-200/80 text-center hover:bg-white transition shadow-2xs">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Duration Range</span>
                <p className="text-2xl font-extrabold font-['Outfit'] text-slate-900 mt-1">
                  {durationStats.shortest}m – {durationStats.longest}m
                </p>
                <span className="text-[10px] text-slate-500 font-medium">min to max</span>
              </div>
            </div>

            <div className="p-3 bg-emerald-50/80 border border-emerald-200/80 rounded-2xl flex items-center gap-2 text-xs text-emerald-950 font-medium">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{durationStats.statusText}</span>
            </div>
          </div>

          <p className="mt-4 text-[11px] text-slate-500 pt-3 border-t border-slate-100 flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-slate-400 shrink-0" />
            <span>Calculated strictly from entry to exit timestamps. In-progress visitors inside the showroom are excluded from average.</span>
          </p>
        </section>

        {/* SECTION E: STORE OPERATING HOURS (5 cols) */}
        <section className="lg:col-span-5 bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-5">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-2xl bg-purple-50 text-purple-600 border border-purple-100 shadow-2xs">
                  <Clock className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base sm:text-lg font-bold text-slate-900 font-['Outfit']">E. Store Operating Hours</h2>
                  <p className="text-xs text-slate-500 mt-0.5">Configured hours vs. actual visitor activity</p>
                </div>
              </div>
              <button
                onClick={() => onNavigateTab('settings')}
                className="text-xs text-blue-600 hover:text-blue-800 font-bold flex items-center gap-1 px-3 py-1.5 rounded-xl hover:bg-blue-50 transition border border-transparent hover:border-blue-200"
              >
                <span>Edit Hours</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* Current Real-time Store Status Banner */}
            <div
              className={`p-4 sm:p-5 rounded-2xl border mb-5 flex items-center justify-between transition-all shadow-2xs ${
                isStoreCurrentlyOpen
                  ? 'bg-emerald-50/90 border-emerald-200 text-emerald-950'
                  : 'bg-rose-50/90 border-rose-200 text-rose-950'
              }`}
            >
              <div className="flex items-center gap-3.5">
                <div className="relative flex items-center justify-center">
                  <span
                    className={`w-3.5 h-3.5 rounded-full ${
                      isStoreCurrentlyOpen ? 'bg-emerald-500' : 'bg-rose-500'
                    }`}
                  />
                  {isStoreCurrentlyOpen && (
                    <span className="w-5 h-5 rounded-full bg-emerald-400 absolute animate-ping opacity-75" />
                  )}
                </div>
                <div>
                  <h3 className="font-extrabold text-sm sm:text-base font-['Outfit']">
                    Current Status: {isStoreCurrentlyOpen ? 'Showroom is OPEN' : 'Showroom is CLOSED'}
                  </h3>
                  <p className="text-xs opacity-80 mt-0.5">{storeHours.timezone}</p>
                </div>
              </div>
              <span className="text-xs font-mono font-bold bg-white/80 text-slate-900 px-3 py-1.5 rounded-xl border border-slate-200 shadow-2xs">
                11h 00m Total
              </span>
            </div>

            {/* Configured Scheduled Hours vs Actual Footfall Activity */}
            <div className="space-y-3.5">
              <div className="p-4 bg-slate-50/80 rounded-2xl border border-slate-200/80">
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Configured Schedule</p>
                <div className="grid grid-cols-2 gap-3 mt-2 text-xs">
                  <div>
                    <span className="text-slate-500 text-[11px]">Opening Time:</span>
                    <p className="font-extrabold text-slate-900 font-mono text-base mt-0.5">{storeHours.openingTime} AM</p>
                  </div>
                  <div>
                    <span className="text-slate-500 text-[11px]">Closing Time:</span>
                    <p className="font-extrabold text-slate-900 font-mono text-base mt-0.5">{storeHours.closingTime} PM</p>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-blue-50/60 rounded-2xl border border-blue-200/70">
                <p className="text-[10px] font-bold uppercase tracking-wider text-blue-900">Today Actual Footfall Timeline</p>
                <div className="grid grid-cols-2 gap-3 mt-2 text-xs">
                  <div>
                    <span className="text-blue-700 text-[11px]">First Recorded Entry:</span>
                    <p className="font-extrabold text-slate-900 font-mono text-base mt-0.5">{storeHours.actualFirstEntryToday || '09:34 AM'}</p>
                    <span className="text-[10px] text-emerald-700 font-bold bg-emerald-100/80 px-1.5 py-0.5 rounded mt-1 inline-block">+4m after opening</span>
                  </div>
                  <div>
                    <span className="text-blue-700 text-[11px]">Latest Recorded Exit:</span>
                    <p className="font-extrabold text-slate-900 font-mono text-base mt-0.5">{storeHours.actualLastExitToday || 'Active'}</p>
                    <span className="text-[10px] text-blue-700 font-bold bg-blue-100/80 px-1.5 py-0.5 rounded mt-1 inline-block">Tracked at gate</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <p className="mt-4 text-[11px] text-slate-500 pt-3 border-t border-slate-100">
            Showroom open 7 days a week. Indian Standard Time (IST).
          </p>
        </section>
      </div>

      {/* ========================================================================= */}
      {/* SECTION F: TAG CUSTOMERS / VISIT PURPOSE MANAGEMENT (12 cols) */}
      {/* ========================================================================= */}
      <section className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)]">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-3 mb-5">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-2xl bg-red-50 text-red-600 border border-red-100 shadow-2xs">
              <Tags className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base sm:text-lg font-bold text-slate-900 font-['Outfit']">
                F. Tag Customers / Visit Purpose Management
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Categorize showroom visitors by intent (Test Ride, Enquiry, Booking, Service, Delivery, Spares)
              </p>
            </div>
          </div>

          <button
            onClick={() => onNavigateTab('visit-purpose')}
            className="text-xs text-blue-600 hover:text-blue-800 font-bold flex items-center gap-1 px-3 py-1.5 rounded-xl hover:bg-blue-50 transition border border-transparent hover:border-blue-200 self-start sm:self-auto"
          >
            <span>Open Purpose Manager</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* 8 Purpose Distribution Pills */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5 mb-6">
          {metrics.purposeDistribution.map((item) => (
            <div
              key={item.purpose}
              className="p-3 rounded-2xl border border-slate-200/80 bg-slate-50/70 text-center hover:bg-white hover:border-blue-300 hover:shadow-xs transition group"
            >
              <div className="flex items-center justify-center gap-1.5 mb-1.5">
                <span className="w-2.5 h-2.5 rounded-full shrink-0 group-hover:scale-125 transition-transform" style={{ backgroundColor: item.color }} />
                <span className="text-[11px] font-bold text-slate-800 truncate">{item.purpose}</span>
              </div>
              <p className="text-xl font-extrabold text-slate-900 font-['Outfit']">{item.count}</p>
              <span className="text-[10px] text-slate-500 font-semibold">{item.percentage}% share</span>
            </div>
          ))}
        </div>

        {/* Recent Visitors Table with 1-Click Quick Tag */}
        <div>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 gap-1">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Recent Visitor Sessions ({visitors.length} total recorded)
            </h3>
            <span className="text-xs text-slate-500">Click &ldquo;Tag&rdquo; to assign customer intent and sales executive</span>
          </div>

          <div className="overflow-x-auto border border-slate-200/80 rounded-2xl shadow-2xs">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50/90 border-b border-slate-200 text-slate-600 uppercase text-[10px] tracking-wider font-bold">
                <tr>
                  <th className="py-3 px-3.5">Visitor ID</th>
                  <th className="py-3 px-2">First Seen</th>
                  <th className="py-3 px-2">Exit / Last Seen</th>
                  <th className="py-3 px-2">Type</th>
                  <th className="py-3 px-3">Visit Purpose</th>
                  <th className="py-3 px-2">Salesperson</th>
                  <th className="py-3 px-2">Duration</th>
                  <th className="py-3 px-3">Customer Notes</th>
                  <th className="py-3 px-3.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {visitors.slice(0, 6).map((vis) => {
                  const isTagged = vis.taggingStatus === 'Tagged' && vis.visitPurpose;
                  const purposeColor = vis.visitPurpose ? PURPOSE_COLORS[vis.visitPurpose] : '#94a3b8';

                  return (
                    <tr
                      key={vis.id}
                      className={`hover:bg-slate-50/80 transition-colors ${
                        !isTagged ? 'bg-amber-50/30' : ''
                      }`}
                    >
                      <td className="py-3 px-3.5">
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
                      <td className="py-3 px-2 font-mono text-slate-700 font-medium">{vis.firstSeenTime}</td>
                      <td className="py-3 px-2 font-mono text-slate-700">
                        {vis.exitTime ? (
                          vis.exitTime
                        ) : (
                          <span className="text-[10px] text-emerald-800 bg-emerald-100 font-extrabold px-2 py-0.5 rounded-full inline-block">
                            On Floor
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-2">
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
                          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800 border border-amber-300 inline-flex items-center gap-1 animate-pulse shadow-2xs">
                            <AlertTriangle className="w-3 h-3 text-amber-600" />
                            Untagged
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-2 text-slate-800 font-semibold">
                        {vis.assignedSalesperson || <span className="text-slate-400 font-normal">—</span>}
                      </td>
                      <td className="py-3 px-2 font-mono font-semibold text-slate-700">{vis.durationMinutes}m</td>
                      <td className="py-3 px-3 text-slate-600 max-w-[200px] truncate" title={vis.notes}>
                        {vis.notes}
                      </td>
                      <td className="py-3 px-3.5 text-right">
                        <button
                          onClick={() => onTagVisitor(vis)}
                          className="px-3 py-1.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-xs font-bold border border-blue-400/30 shadow-2xs transition active:scale-95"
                        >
                          {vis.visitPurpose ? 'Edit Tag' : 'Tag Now'}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>

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
                  <span className="text-slate-400 text-[10px] uppercase font-bold block">Arrival Timestamp</span>
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
