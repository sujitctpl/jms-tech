import React, { useState } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import { TrendingUp, Clock, Info, Flame, Users, Calendar, ArrowUpRight } from 'lucide-react';

export const PeakHoursView: React.FC = () => {
  const { metrics, dateFilter } = useShowroom();

  const [chartMode, setChartMode] = useState<'comparative' | 'stacked' | 'new_only'>('comparative');
  const [selectedHourIdx, setSelectedHourIdx] = useState<number>(8); // Default to 5:00 - 6:00 PM peak
  const TARGET_HOURLY = [8, 12, 18, 16, 12, 10, 15, 20, 25, 22, 12];
  const selectedSlot = metrics.hourlyChartData[selectedHourIdx] || metrics.hourlyChartData[8] || {
    hour: '05:00 - 06:00 PM',
    visitors: 28,
    newVisitors: 19,
    returningVisitors: 9,
    testRides: 6,
    deliveries: 2,
  };

  const maxVal = Math.max(...metrics.hourlyChartData.map((d) => d.visitors), 1);
  const totalDayTraffic = metrics.morningTraffic + metrics.afternoonTraffic + metrics.eveningTraffic;

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-['Outfit'] text-slate-900">Peak Hours & Footfall Analytics</h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-100 text-amber-800 flex items-center gap-1">
              <Flame className="w-3.5 h-3.5 text-amber-600" />
              Peak: {metrics.busiestHour}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Analyzing customer arrival patterns across morning, afternoon, and evening dealership peak hours.
          </p>
        </div>

        <div className="p-2 bg-slate-100 rounded-xl text-xs font-medium text-slate-600 flex items-center gap-1.5 self-start sm:self-auto">
          <Calendar className="w-3.5 h-3.5 text-slate-400" />
          <span>Showing Data: <strong>Today (Active Real-Time)</strong></span>
        </div>
      </div>

      {/* Hero Peak Hour Banner */}
      <div className="bg-gradient-to-r from-red-600 via-rose-600 to-amber-500 rounded-3xl p-6 sm:p-8 text-white shadow-xl shadow-red-500/20 relative overflow-hidden">
        <div className="absolute right-0 top-0 w-80 h-full bg-white/10 blur-3xl pointer-events-none" />
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 relative z-10">
          <div>
            <div className="flex items-center gap-2 mb-2.5">
              <span className="px-3 py-1 rounded-full text-[10px] font-extrabold bg-white/20 backdrop-blur-md text-white uppercase tracking-wider border border-white/25">
                Prime Footfall Window
              </span>
              <span className="text-xs text-white/90 font-medium">• Verified Arrival Peak</span>
            </div>
            <h2 className="text-3xl sm:text-4xl font-black font-['Outfit']">{metrics.busiestHour}</h2>
            <p className="text-xs sm:text-sm text-white/90 mt-1.5 max-w-xl leading-relaxed">
              During this window, customer footfall surges with working professionals and families seeking bike test rides and loan discussions.
            </p>
          </div>

          <div className="flex items-center gap-5 shrink-0 bg-white/15 backdrop-blur-md p-5 rounded-2xl border border-white/25 shadow-lg">
            <div>
              <span className="text-[11px] font-bold text-white/80 block uppercase tracking-wider">Peak Hour Footfall</span>
              <span className="text-3xl sm:text-4xl font-black font-['Outfit'] text-white">{metrics.busiestHourCount}</span>
              <span className="text-xs text-white/80 block font-medium">visitors / hour</span>
            </div>
            <div className="h-12 w-px bg-white/25" />
            <div>
              <span className="text-[11px] font-bold text-white/80 block uppercase tracking-wider">Share of Day</span>
              <span className="text-3xl sm:text-4xl font-black font-['Outfit'] text-white">
                {totalDayTraffic > 0 ? Math.round((metrics.busiestHourCount / totalDayTraffic) * 100) : 18}%
              </span>
              <span className="text-xs text-white/80 block font-medium">traffic concentration</span>
            </div>
          </div>
        </div>
      </div>

      {/* Day-Part Breakdown: Morning vs Afternoon vs Evening */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Morning */}
        <div className="p-6 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition relative overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Morning Shift</span>
            <span className="text-[10px] bg-slate-100 text-slate-700 px-2.5 py-1 rounded-lg font-mono font-bold border border-slate-200/80">
              09:00 AM - 01:00 PM
            </span>
          </div>
          <div className="flex items-baseline gap-2 mb-2">
            <span className="text-3xl sm:text-4xl font-extrabold text-slate-900 font-['Outfit']">{metrics.morningTraffic}</span>
            <span className="text-xs font-semibold text-slate-500">visitors</span>
          </div>
          <p className="text-xs text-slate-600 mb-4 leading-relaxed">
            Primary driver: Periodic service vehicle check-ins, early-bird test rides, and genuine parts inquiries.
          </p>
          <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 rounded-full transition-all"
              style={{
                width: `${totalDayTraffic > 0 ? Math.round((metrics.morningTraffic / totalDayTraffic) * 100) : 30}%`,
              }}
            />
          </div>
          <span className="text-[11px] font-semibold text-slate-500 block mt-2 text-right">
            {totalDayTraffic > 0 ? Math.round((metrics.morningTraffic / totalDayTraffic) * 100) : 30}% of total
          </span>
        </div>

        {/* Afternoon */}
        <div className="p-6 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition relative overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Afternoon Shift</span>
            <span className="text-[10px] bg-slate-100 text-slate-700 px-2.5 py-1 rounded-lg font-mono font-bold border border-slate-200/80">
              01:00 PM - 05:00 PM
            </span>
          </div>
          <div className="flex items-baseline gap-2 mb-2">
            <span className="text-3xl sm:text-4xl font-extrabold text-slate-900 font-['Outfit']">{metrics.afternoonTraffic}</span>
            <span className="text-xs font-semibold text-slate-500">visitors</span>
          </div>
          <p className="text-xs text-slate-600 mb-4 leading-relaxed">
            Primary driver: Steady walk-ins, financing documentation with bank representatives, and accessories sales.
          </p>
          <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-600 rounded-full transition-all"
              style={{
                width: `${totalDayTraffic > 0 ? Math.round((metrics.afternoonTraffic / totalDayTraffic) * 100) : 30}%`,
              }}
            />
          </div>
          <span className="text-[11px] font-semibold text-slate-500 block mt-2 text-right">
            {totalDayTraffic > 0 ? Math.round((metrics.afternoonTraffic / totalDayTraffic) * 100) : 30}% of total
          </span>
        </div>

        {/* Evening */}
        <div className="p-6 bg-gradient-to-br from-red-50/50 via-white to-amber-50/30 rounded-3xl border border-red-200/80 shadow-[0_4px_24px_-4px_rgba(239,68,68,0.1)] hover:shadow-md transition relative overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold uppercase tracking-wider text-red-950 flex items-center gap-1.5">
              <span>Evening Peak Shift</span>
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
            </span>
            <span className="text-[10px] bg-red-100/90 text-red-800 px-2.5 py-1 rounded-lg font-mono font-bold border border-red-200/60">
              05:00 PM - 08:00 PM
            </span>
          </div>
          <div className="flex items-baseline gap-2 mb-2">
            <span className="text-3xl sm:text-4xl font-extrabold text-red-950 font-['Outfit']">{metrics.eveningTraffic}</span>
            <span className="text-xs font-bold text-red-700 bg-red-100/80 px-2 py-0.5 rounded-full">Peak Density</span>
          </div>
          <p className="text-xs text-red-900/80 mb-4 leading-relaxed">
            Primary driver: High-conversion test rides, booking confirmations, delivery ceremonies, and key handovers.
          </p>
          <div className="w-full h-2.5 bg-red-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-red-600 rounded-full transition-all"
              style={{
                width: `${totalDayTraffic > 0 ? Math.round((metrics.eveningTraffic / totalDayTraffic) * 100) : 40}%`,
              }}
            />
          </div>
          <span className="text-[11px] font-bold text-red-700 block mt-2 text-right">
            {totalDayTraffic > 0 ? Math.round((metrics.eveningTraffic / totalDayTraffic) * 100) : 40}% of total
          </span>
        </div>
      </div>

      {/* Comprehensive Hour-by-Hour Footfall Analytics Chart (Optimized) */}
      <div className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)]">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between pb-5 border-b border-slate-100 gap-3 mb-5">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base sm:text-lg font-bold text-slate-900 font-['Outfit']">
                Complete Hour-by-Hour Footfall & Conversion Telemetry
              </h3>
              <span className="text-[10px] font-bold bg-blue-100 text-blue-800 px-2.5 py-0.5 rounded-full border border-blue-200">
                Optimized Analytics
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Continuous camera footfall tracking, hourly surge detection, and benchmark comparison.
            </p>
          </div>

          {/* View Mode Toggle Switch */}
          <div className="flex items-center bg-slate-100 p-1 rounded-xl border border-slate-200 shadow-2xs self-start lg:self-center">
            <button
              type="button"
              onClick={() => setChartMode('comparative')}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition ${
                chartMode === 'comparative'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Dual Comparison
            </button>
            <button
              type="button"
              onClick={() => setChartMode('stacked')}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition ${
                chartMode === 'stacked'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Stacked Flow
            </button>
            <button
              type="button"
              onClick={() => setChartMode('new_only')}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition ${
                chartMode === 'new_only'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              New Only
            </button>
          </div>
        </div>

        {/* Selected Hour Telemetry Inspection Card */}
        <div className="mb-5 p-4 bg-gradient-to-r from-blue-50/80 via-slate-50 to-indigo-50/60 rounded-2xl border border-blue-200/80 shadow-2xs flex flex-col md:flex-row md:items-center justify-between gap-4 transition-all">
          <div className="flex items-center gap-3.5">
            <div
              className={`w-11 h-11 rounded-2xl flex items-center justify-center font-bold text-white shadow-md shrink-0 ${
                selectedSlot.visitors === metrics.busiestHourCount
                  ? 'bg-gradient-to-tr from-red-600 to-amber-500 shadow-red-500/30'
                  : 'bg-gradient-to-tr from-blue-600 to-indigo-600 shadow-blue-500/30'
              }`}
            >
              {selectedSlot.visitors === metrics.busiestHourCount ? (
                <Flame className="w-5 h-5 text-white animate-pulse" />
              ) : (
                <Clock className="w-5 h-5 text-white" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-extrabold text-slate-900 text-base font-['Outfit']">
                  {selectedSlot.hour}
                </span>
                {selectedSlot.visitors === metrics.busiestHourCount ? (
                  <span className="px-2.5 py-0.5 bg-red-100 text-red-700 text-[10px] font-extrabold rounded-full border border-red-200 flex items-center gap-1 shadow-2xs">
                    <Flame className="w-3.5 h-3.5 text-red-600" /> Busiest Peak Traffic Window
                  </span>
                ) : (
                  <span className="px-2 py-0.5 bg-blue-100/80 text-blue-700 text-[10px] font-bold rounded-full border border-blue-200/60">
                    Standard Operational Slot
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2.5 text-xs text-slate-600 mt-1 flex-wrap">
                <span><strong>{selectedSlot.visitors}</strong> Total Visitors</span>
                <span>•</span>
                <span className="text-blue-700 font-semibold">{selectedSlot.newVisitors} New Walk-ins</span>
                <span>•</span>
                <span className="text-indigo-700 font-semibold">{selectedSlot.returningVisitors} Returning</span>
                <span>•</span>
                <span className="text-amber-700 font-semibold">Target: {TARGET_HOURLY[selectedHourIdx]}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 bg-white/90 p-2.5 px-4 rounded-xl border border-slate-200 shadow-2xs self-start md:self-center">
            <div>
              <span className="text-[10px] font-bold text-slate-500 uppercase block tracking-wider">Test Rides</span>
              <span className="text-sm font-extrabold text-slate-900 font-['Outfit']">{selectedSlot.testRides} booked</span>
            </div>
            <div className="h-6 w-px bg-slate-200" />
            <div>
              <span className="text-[10px] font-bold text-slate-500 uppercase block tracking-wider">Deliveries</span>
              <span className="text-sm font-extrabold text-slate-900 font-['Outfit']">{selectedSlot.deliveries} handovers</span>
            </div>
          </div>
        </div>

        {/* Detailed Modern Chart Canvas with Reference Grid */}
        <div className="relative pt-6 pb-2">
          {/* Soft Reference Gridlines */}
          <div className="absolute inset-x-0 top-10 border-b border-dashed border-slate-200" />
          <div className="absolute inset-x-0 top-24 border-b border-dashed border-slate-200" />
          <div className="absolute inset-x-0 top-38 border-b border-dashed border-slate-200" />

          <div className="h-64 w-full flex items-end gap-2 sm:gap-3 overflow-x-auto relative z-10 px-1">
            {metrics.hourlyChartData.map((slot, i) => {
              const isSelected = selectedHourIdx === i;
              const isPeak = slot.visitors === metrics.busiestHourCount;
              const targetVal = TARGET_HOURLY[i];
              const maxScale = Math.max(maxVal, 30);

              const actualHeightPct = Math.round((slot.visitors / maxScale) * 100);
              const targetHeightPct = Math.round((targetVal / maxScale) * 100);
              const newRatio = slot.visitors > 0 ? (slot.newVisitors / slot.visitors) * 100 : 50;

              return (
                <div
                  key={i}
                  onClick={() => setSelectedHourIdx(i)}
                  className={`flex-1 min-w-[62px] flex flex-col items-center h-full justify-end cursor-pointer group p-1 rounded-2xl transition select-none ${
                    isSelected ? 'bg-blue-50/70 ring-2 ring-blue-500/50 shadow-xs' : 'hover:bg-slate-50'
                  }`}
                >
                  {/* Top Visitor Count Label */}
                  <span
                    className={`text-[11px] font-bold mb-1.5 transition-all ${
                      isSelected
                        ? 'text-blue-700 font-extrabold scale-110'
                        : isPeak
                        ? 'text-red-600 font-bold'
                        : 'text-slate-600 group-hover:text-slate-900'
                    }`}
                  >
                    {slot.visitors}
                  </span>

                  {/* Bar Geometry Switch */}
                  {chartMode === 'comparative' ? (
                    /* Side-by-Side Dual Pillars: Today Actual vs Target */
                    <div className="w-full flex items-end justify-center gap-1.5 h-full pb-1">
                      {/* Actual Pillar */}
                      <div className="w-1/2 flex flex-col items-center justify-end h-full">
                        <div
                          className={`w-full rounded-t-lg bg-gradient-to-t ${
                            isPeak
                              ? 'from-red-600 to-rose-400 shadow-md shadow-red-500/30'
                              : 'from-blue-700 via-blue-600 to-cyan-400'
                          } transition-all ${
                            isSelected ? 'brightness-110 shadow-md' : 'group-hover:brightness-105'
                          }`}
                          style={{ height: `${Math.max(actualHeightPct, 12)}%` }}
                        />
                      </div>

                      {/* Target / Benchmark Pillar */}
                      <div className="w-1/2 flex flex-col items-center justify-end h-full">
                        <div
                          className="w-full rounded-t-lg bg-gradient-to-t from-amber-500 to-yellow-400 opacity-70 group-hover:opacity-100 transition-all"
                          style={{ height: `${Math.max(targetHeightPct, 10)}%` }}
                        />
                      </div>
                    </div>
                  ) : chartMode === 'stacked' ? (
                    /* Stacked Flow: New vs Returning */
                    <div
                      className={`w-full max-w-[40px] rounded-t-xl overflow-hidden flex flex-col justify-end transition-all ${
                        isSelected
                          ? 'ring-3 ring-blue-500 shadow-md scale-105'
                          : isPeak
                          ? 'ring-2 ring-red-400 shadow-xs'
                          : 'group-hover:opacity-90'
                      }`}
                      style={{ height: `${Math.max(actualHeightPct, 12)}%` }}
                    >
                      <div
                        className="w-full bg-gradient-to-t from-indigo-500 to-indigo-400"
                        style={{ height: `${100 - newRatio}%` }}
                      />
                      <div
                        className={`w-full ${
                          isPeak
                            ? 'bg-gradient-to-t from-red-600 to-rose-500'
                            : 'bg-gradient-to-t from-blue-700 to-blue-500'
                        }`}
                        style={{ height: `${newRatio}%` }}
                      />
                    </div>
                  ) : (
                    /* New Walk-ins Only */
                    <div
                      className="w-full max-w-[36px] rounded-t-xl bg-gradient-to-t from-blue-700 to-cyan-400 transition-all group-hover:brightness-105"
                      style={{ height: `${Math.max(Math.round((slot.newVisitors / maxScale) * 100), 10)}%` }}
                    />
                  )}

                  {/* Hour Label */}
                  <span
                    className={`text-[10px] mt-2 font-semibold text-center w-full truncate ${
                      isSelected ? 'text-blue-700 font-bold' : 'text-slate-500 group-hover:text-slate-900'
                    }`}
                  >
                    {slot.hour.replace(' AM', '').replace(' PM', '')}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Legend and Footer */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pt-4 mt-3 border-t border-slate-200 text-xs text-slate-600 gap-2">
            <div className="flex items-center gap-4 flex-wrap">
              {chartMode === 'comparative' ? (
                <>
                  <span className="flex items-center gap-1.5 font-semibold text-blue-700">
                    <span className="w-2.5 h-2.5 rounded-sm bg-gradient-to-t from-blue-600 to-cyan-400 inline-block shadow-2xs" />
                    Actual Walk-ins
                  </span>
                  <span className="flex items-center gap-1.5 font-semibold text-amber-700">
                    <span className="w-2.5 h-2.5 rounded-sm bg-gradient-to-t from-amber-500 to-yellow-400 inline-block shadow-2xs" />
                    Hourly Target Benchmark
                  </span>
                  <span className="flex items-center gap-1.5 font-semibold text-red-600">
                    <span className="w-2.5 h-2.5 rounded-sm bg-red-600 inline-block shadow-2xs" />
                    Peak Surge Hour
                  </span>
                </>
              ) : (
                <>
                  <span className="flex items-center gap-1.5 font-semibold text-blue-700">
                    <span className="w-2.5 h-2.5 rounded-sm bg-blue-600 inline-block" />
                    New First-time Walk-ins
                  </span>
                  <span className="flex items-center gap-1.5 font-semibold text-indigo-700">
                    <span className="w-2.5 h-2.5 rounded-sm bg-indigo-500 inline-block" />
                    Returning Customers
                  </span>
                  <span className="flex items-center gap-1.5 font-semibold text-red-600">
                    <span className="w-2.5 h-2.5 rounded-sm bg-red-600 inline-block" />
                    Peak Surge Hour
                  </span>
                </>
              )}
            </div>

            <span className="text-[11px] text-slate-500 font-medium">
              Click any bar to inspect conversion telemetry
            </span>
          </div>
        </div>

        {/* Staffing Recommendation Footer Box */}
        <div className="mt-4 p-4 bg-gradient-to-r from-blue-50/90 to-indigo-50/50 rounded-2xl border border-blue-200/90 flex flex-col sm:flex-row sm:items-center justify-between text-xs text-blue-950 gap-2 shadow-2xs">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-blue-600 shrink-0" />
            <span>
              <strong>Staffing Recommendation:</strong> Ensure 100% sales consultant floor presence between 4:30 PM and 7:30 PM for peak evening footfall.
            </span>
          </div>
          <span className="font-mono text-blue-800 text-[11px] font-semibold">IST Timezone (+5:30)</span>
        </div>
      </div>
    </div>
  );
};
