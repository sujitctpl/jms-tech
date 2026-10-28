import React, { useState } from 'react';
import { useShowroom } from '../../context/ShowroomContext';
import { useAuth } from '../../context/AuthContext';
import { AttendanceRecord, AttendanceStatus } from '../../types';
import {
  UserCheck,
  Search,
  Filter,
  FileSpreadsheet,
  Clock,
  CheckCircle,
  AlertCircle,
  Calendar,
  Shield,
  Edit,
} from 'lucide-react';

export const AttendanceView: React.FC = () => {
  const { user } = useAuth();
  const {
    attendance,
    markAttendance,
    quickCheckInEmployee,
    quickCheckOutEmployee,
    exportAttendanceCSV,
    metrics,
  } = useShowroom();

  const [searchQuery, setSearchQuery] = useState('');
  const [departmentFilter, setDepartmentFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');

  // Edit attendance modal
  const [editingRecord, setEditingRecord] = useState<AttendanceRecord | null>(null);
  const [editStatus, setEditStatus] = useState<AttendanceStatus>('Present');
  const [editCheckIn, setEditCheckIn] = useState('');
  const [editCheckOut, setEditCheckOut] = useState('');

  const isOwner = user?.role === 'owner';

  const departments = ['all', 'Sales', 'Service', 'Customer Relations', 'Finance & Insurance', 'Accessories & Spares'];

  const filteredAttendance = attendance.filter((a) => {
    const q = searchQuery.toLowerCase();
    const matchSearch =
      a.employeeName.toLowerCase().includes(q) ||
      a.role.toLowerCase().includes(q) ||
      a.department.toLowerCase().includes(q);

    if (!matchSearch) return false;
    if (departmentFilter !== 'all' && a.department !== departmentFilter) return false;
    if (statusFilter !== 'all' && a.status !== statusFilter) return false;
    return true;
  });

  const handleOpenEdit = (rec: AttendanceRecord) => {
    setEditingRecord(rec);
    setEditStatus(rec.status);
    setEditCheckIn(rec.checkInTime || '09:30 AM');
    setEditCheckOut(rec.checkOutTime || '');
  };

  const handleSaveEdit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingRecord) return;
    markAttendance(editingRecord.id, editStatus, editCheckIn, editCheckOut || undefined);
    setEditingRecord(null);
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold font-['Outfit'] text-slate-900">Employee Attendance & Floor Duty</h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800">
              {metrics.employeesOnDuty} / {metrics.employeesTotal} On Duty Today
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Real-time dealership staff check-ins, departmental shifts, and manual attendance overrides.
          </p>
        </div>

        <button
          onClick={exportAttendanceCSV}
          className="px-3.5 py-2 bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 rounded-xl text-xs font-semibold shadow-xs flex items-center gap-1.5 transition self-start sm:self-auto"
        >
          <FileSpreadsheet className="w-4 h-4 text-emerald-600" />
          <span>Export Attendance CSV</span>
        </button>
      </div>

      {/* 4 Stat KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
        <div className="p-4 sm:p-5 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Total Staff</span>
            <div className="p-2 rounded-xl bg-slate-100 text-slate-700 group-hover:scale-110 transition-transform">
              <UserCheck className="w-4 h-4" />
            </div>
          </div>
          <p className="text-3xl font-extrabold text-slate-900 font-['Outfit'] mt-1">{metrics.employeesTotal}</p>
          <span className="text-[11px] text-slate-400 mt-0.5 block font-medium">dealership employees</span>
        </div>

        <div className="p-4 sm:p-5 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold text-emerald-700 uppercase tracking-wider">Present & Active</span>
            <div className="p-2 rounded-xl bg-emerald-50 text-emerald-600 group-hover:scale-110 transition-transform">
              <CheckCircle className="w-4 h-4" />
            </div>
          </div>
          <p className="text-3xl font-extrabold text-emerald-900 font-['Outfit'] mt-1">{metrics.employeesPresent}</p>
          <span className="text-[11px] text-emerald-700 mt-0.5 block font-semibold">
            {Math.round((metrics.employeesPresent / metrics.employeesTotal) * 100)}% attendance rate
          </span>
        </div>

        <div className="p-4 sm:p-5 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold text-amber-700 uppercase tracking-wider">Late Arrivals</span>
            <div className="p-2 rounded-xl bg-amber-50 text-amber-600 group-hover:scale-110 transition-transform">
              <Clock className="w-4 h-4" />
            </div>
          </div>
          <p className="text-3xl font-extrabold text-amber-900 font-['Outfit'] mt-1">{metrics.employeesLate}</p>
          <span className="text-[11px] text-amber-700 mt-0.5 block font-semibold">Checked in &gt; 9:30 AM</span>
        </div>

        <div className="p-4 sm:p-5 bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] hover:shadow-md transition group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold text-rose-700 uppercase tracking-wider">Absent / On Leave</span>
            <div className="p-2 rounded-xl bg-rose-50 text-rose-600 group-hover:scale-110 transition-transform">
              <AlertCircle className="w-4 h-4" />
            </div>
          </div>
          <p className="text-3xl font-extrabold text-rose-900 font-['Outfit'] mt-1">
            {metrics.employeesAbsent + metrics.employeesOnLeave}
          </p>
          <span className="text-[11px] text-rose-700 mt-0.5 block font-semibold">1 Absent, 1 On Leave</span>
        </div>
      </div>

      {/* Integration Disclaimer Alert */}
      <div className="p-4 bg-gradient-to-r from-blue-50/90 via-indigo-50/40 to-slate-50 border border-blue-200/80 rounded-3xl flex items-start gap-3.5 shadow-2xs">
        <Shield className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
        <div className="text-xs text-blue-950">
          <p className="font-extrabold font-['Outfit'] text-sm">Attendance Synchronization & Security</p>
          <p className="mt-0.5 text-blue-800 font-medium">
            Showroom attendance is logged through showroom biometric readers and verified manual overrides. CCTV automated attendance
            is available as an add-on integration in Settings with staff consent and opt-in badge scanning.
          </p>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] flex flex-col sm:flex-row gap-3">
        <div className="flex-1 relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search employee by name, role, or department..."
            className="w-full pl-10 pr-3.5 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition"
          />
        </div>

        <div className="flex flex-wrap gap-2">
          <select
            value={departmentFilter}
            onChange={(e) => setDepartmentFilter(e.target.value)}
            className="px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-600 outline-none font-semibold text-slate-700 shadow-2xs"
          >
            {departments.map((d) => (
              <option key={d} value={d}>
                {d === 'all' ? 'All Departments' : d}
              </option>
            ))}
          </select>

          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-600 outline-none font-semibold text-slate-700 shadow-2xs"
          >
            <option value="all">All Statuses</option>
            <option value="Present">Present Only</option>
            <option value="Late">Late Only</option>
            <option value="On Leave">On Leave Only</option>
            <option value="Absent">Absent Only</option>
          </select>
        </div>
      </div>

      {/* Attendance Records Table */}
      <div className="bg-white rounded-3xl border border-slate-200/80 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.06)] overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50/90 border-b border-slate-200 text-slate-600 uppercase text-[10px] tracking-wider font-bold">
              <tr>
                <th className="py-3 px-4">Employee Name</th>
                <th className="py-3 px-3">Role & Department</th>
                <th className="py-3 px-3">Check-in Time</th>
                <th className="py-3 px-3">Check-out Time</th>
                <th className="py-3 px-3">Work Hours</th>
                <th className="py-3 px-3">Verification Source</th>
                <th className="py-3 px-3">Status</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {filteredAttendance.map((rec) => {
                const isPresent = rec.status === 'Present';
                const isLate = rec.status === 'Late';
                const isOnLeave = rec.status === 'On Leave';
                const isAbsent = rec.status === 'Absent';

                return (
                  <tr key={rec.id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-3">
                        <div className="relative shrink-0">
                          <img
                            src={rec.avatarUrl || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80'}
                            alt={rec.employeeName}
                            className="w-9 h-9 rounded-full object-cover border border-slate-200 shadow-2xs"
                          />
                          <span
                            className={`w-2.5 h-2.5 rounded-full absolute bottom-0 right-0 ring-2 ring-white ${
                              isPresent ? 'bg-emerald-500' : isLate ? 'bg-amber-500' : isOnLeave ? 'bg-blue-500' : 'bg-rose-500'
                            }`}
                          />
                        </div>
                        <div>
                          <span className="font-bold text-slate-900 font-['Outfit'] block text-xs sm:text-sm">{rec.employeeName}</span>
                          <span className="text-[10px] text-slate-400 font-mono">ID: {rec.employeeId.toUpperCase()}</span>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-3">
                      <p className="text-slate-800 font-semibold">{rec.role}</p>
                      <span className="text-[10px] text-slate-400 font-medium">{rec.department}</span>
                    </td>
                    <td className="py-3 px-3 font-mono font-medium text-slate-700">{rec.checkInTime || '—'}</td>
                    <td className="py-3 px-3 font-mono font-medium text-slate-700">{rec.checkOutTime || 'On Duty'}</td>
                    <td className="py-3 px-3 font-mono text-slate-700">{rec.workHours ? `${rec.workHours}h` : '0h'}</td>
                    <td className="py-3 px-3 text-[11px] text-slate-500">{rec.method}</td>
                    <td className="py-3 px-3">
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold inline-flex items-center gap-1.5 border shadow-2xs ${
                          isPresent
                            ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                            : isLate
                            ? 'bg-amber-50 text-amber-800 border-amber-200'
                            : isOnLeave
                            ? 'bg-blue-50 text-blue-800 border-blue-200'
                            : 'bg-rose-50 text-rose-800 border-rose-200'
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            isPresent
                              ? 'bg-emerald-500'
                              : isLate
                              ? 'bg-amber-500'
                              : isOnLeave
                              ? 'bg-blue-500'
                              : 'bg-rose-500'
                          }`}
                        />
                        {rec.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        {!rec.checkInTime ? (
                          <button
                            onClick={() => quickCheckInEmployee(rec.employeeId)}
                            className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold border border-emerald-700 shadow-2xs transition active:scale-95"
                          >
                            Check In
                          </button>
                        ) : !rec.checkOutTime ? (
                          <button
                            onClick={() => quickCheckOutEmployee(rec.employeeId)}
                            className="px-3 py-1.5 bg-white hover:bg-slate-100 text-slate-700 rounded-xl text-xs font-bold border border-slate-300 shadow-2xs transition active:scale-95"
                          >
                            Check Out
                          </button>
                        ) : null}

                        {isOwner && (
                          <button
                            onClick={() => handleOpenEdit(rec)}
                            title="Edit attendance record"
                            className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-xl transition border border-transparent hover:border-blue-200"
                          >
                            <Edit className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Edit Record Modal (For Authorized Users) */}
      {editingRecord && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-fadeIn">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl border border-slate-100">
            <h3 className="font-bold text-slate-900 text-base font-['Outfit']">
              Adjust Attendance: {editingRecord.employeeName}
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">Override status, check-in, or check-out timestamps</p>

            <form onSubmit={handleSaveEdit} className="mt-4 space-y-3">
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1">Status</label>
                <select
                  value={editStatus}
                  onChange={(e) => setEditStatus(e.target.value as AttendanceStatus)}
                  className="w-full px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-600 outline-none"
                >
                  <option value="Present">Present</option>
                  <option value="Late">Late</option>
                  <option value="Absent">Absent</option>
                  <option value="On Leave">On Leave</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1">
                  Check-in Time
                </label>
                <input
                  type="text"
                  value={editCheckIn}
                  onChange={(e) => setEditCheckIn(e.target.value)}
                  placeholder="e.g. 09:22 AM"
                  className="w-full px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-600 outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-1">
                  Check-out Time
                </label>
                <input
                  type="text"
                  value={editCheckOut}
                  onChange={(e) => setEditCheckOut(e.target.value)}
                  placeholder="e.g. 07:30 PM (or leave blank if on duty)"
                  className="w-full px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-600 outline-none"
                />
              </div>

              <div className="flex gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setEditingRecord(null)}
                  className="flex-1 py-2 border border-slate-200 text-slate-600 font-semibold rounded-xl text-xs hover:bg-slate-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl text-xs shadow-md shadow-blue-600/20"
                >
                  Save Override
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
