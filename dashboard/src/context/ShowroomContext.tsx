import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';
import {
  VisitorRecord,
  AttendanceRecord,
  Employee,
  StoreHoursConfig,
  ShowroomSettings,
  DateFilterOption,
  VisitPurpose,
  TaggingStatus,
  HourlyFootfall,
  DurationDistribution,
  PurposeDistribution,
  WaitingTimeConfig,
} from '../types';
import {
  INITIAL_VISITORS,
  INITIAL_EMPLOYEES,
  INITIAL_ATTENDANCE,
  INITIAL_STORE_HOURS,
  INITIAL_SHOWROOM_SETTINGS,
  TODAY_HOURLY_FOOTFALL,
  PURPOSE_COLORS,
  YESTERDAY_METRICS,
} from '../data/seedData';
import { cameraService, DepartmentModel } from '../services/cameraService';
import { FASTAPI_BASE_URL } from '../config/env';

interface ShowroomContextType {
  // Visitors
  visitors: VisitorRecord[];
  tagVisitor: (visitorId: string, purpose: VisitPurpose, salesperson: string, notes: string) => void;
  addNewVisitor: (record: Partial<VisitorRecord>) => void;
  checkoutVisitor: (visitorId: string) => void;
  refreshVisitors: () => Promise<void>;
  clearAllVisitors: () => Promise<void>;

  // Employees & Attendance
  employees: Employee[];
  attendance: AttendanceRecord[];
  addEmployee: (employee: Employee, attendance?: AttendanceRecord) => void;
  addEmployeePhotos: (employeeId: string, additionalPhotosCount: number, newAvatarUrl?: string) => void;
  refreshEmployees: () => Promise<void>;
  markAttendance: (recordId: string, status: AttendanceRecord['status'], checkInTime?: string, checkOutTime?: string) => void;
  quickCheckInEmployee: (employeeId: string) => void;
  quickCheckOutEmployee: (employeeId: string) => void;

  // Store Hours & Settings
  storeHours: StoreHoursConfig;
  updateStoreHours: (hours: Partial<StoreHoursConfig>) => void;
  isStoreCurrentlyOpen: boolean;
  showroomSettings: ShowroomSettings;
  updateShowroomSettings: (settings: Partial<ShowroomSettings>) => void;
  waitingConfig: WaitingTimeConfig;
  updateWaitingConfig: (config: Partial<WaitingTimeConfig>) => void;

  // Filters & State
  dateFilter: DateFilterOption;
  setDateFilter: (filter: DateFilterOption) => void;
  customDateRange: { start: string; end: string };
  setCustomDateRange: (range: { start: string; end: string }) => void;

  // Simulation & Real-time
  isLiveSimulationActive: boolean;
  toggleLiveSimulation: () => void;
  lastSimulatedEvent: string | null;

  // Computed Metrics
  metrics: {
    virtualLineCrossings: number;
    totalVisitors: number;
    newVisitors: number;
    returningVisitors: number;
    newVisitorPct: number;
    returningVisitorPct: number;
    comparisonYesterdayPct: number;
    yesterdayVisitors: number;
    employeesTotal: number;
    employeesPresent: number;
    employeesAbsent: number;
    employeesOnDuty: number;
    employeesLate: number;
    employeesOnLeave: number;
    busiestHour: string;
    busiestHourCount: number;
    morningTraffic: number;
    afternoonTraffic: number;
    eveningTraffic: number;
    avgDurationMinutes: number;
    medianDurationMinutes: number;
    shortestDurationMinutes: number;
    longestDurationMinutes: number;
    untaggedCount: number;
    taggedCount: number;
    hourlyChartData: HourlyFootfall[];
    durationDistribution: DurationDistribution[];
    purposeDistribution: PurposeDistribution[];
    sectionCounts: { sales: number; accounts: number; [key: string]: number };
    sectionOccupancy: { sales: number; accounts: number; [key: string]: number };
    unattendedWaitingCount: number;
    waitingBreachCount: number;
    breachVisitors: VisitorRecord[];
    avgWaitingSeconds: number;
    avgWaitingFormatted: string;
    longestWaitingSeconds: number;
    slaCompliancePct: number;
    currentlyWaitingVisitors: VisitorRecord[];
  };

  // Section Tracking
  sectionCounts: { sales: number; accounts: number; [key: string]: number };
  sectionOccupancy: { sales: number; accounts: number; [key: string]: number };
  staffDepartmentDwell: Record<string, {
    name: string;
    department_id: string;
    department_name: string;
    dwell_seconds: number;
    dwell_formatted: string;
    is_active?: boolean;
    total_seconds?: number;
    total_formatted?: string;
    visits?: number;
  }>;
  departments: DepartmentModel[];
  refreshDepartments: () => Promise<void>;
  lastSyncedFootfall: number | null;

  // CSV Export
  exportVisitorsCSV: () => void;
  exportAttendanceCSV: () => void;
}

const ShowroomContext = createContext<ShowroomContextType | undefined>(undefined);

export const ShowroomProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [visitors, setVisitors] = useState<VisitorRecord[]>(() => {
    // Purge legacy mock seed data so visitor list is populated strictly by real detections
    const saved = localStorage.getItem('capermint_visitors') || localStorage.getItem('jms_visitors');
    if (!saved) return [];
    try {
      const parsed: VisitorRecord[] = JSON.parse(saved);
      const todayStr = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(new Date());
      // Auto-purge any stale records from previous days, stock Unsplash photos, or mock test IDs
      const hasStaleOrMock = parsed.some(
        (v) =>
          (v.avatarUrl && v.avatarUrl.includes('unsplash.com')) ||
          (v.date && v.date !== todayStr) ||
          v.id.startsWith('vis-10') ||
          v.id.startsWith('vis-11')
      );
      if (hasStaleOrMock) {
        localStorage.removeItem('capermint_visitors');
        localStorage.removeItem('jms_visitors');
        return [];
      }
      return parsed;
    } catch {
      return [];
    }
  });

  const [employees, setEmployees] = useState<Employee[]>(() => {
    const saved = localStorage.getItem('capermint_employees') || localStorage.getItem('jms_employees');
    if (!saved) return INITIAL_EMPLOYEES;
    try {
      const parsed: Employee[] = JSON.parse(saved);
      // Strictly purge hardcoded legacy mock employee records (emp-1 through emp-8)
      return parsed.filter((e) => !/^emp-[1-8]$/i.test(e.id));
    } catch {
      return INITIAL_EMPLOYEES;
    }
  });

  const [attendance, setAttendance] = useState<AttendanceRecord[]>(() => {
    const saved = localStorage.getItem('capermint_attendance') || localStorage.getItem('jms_attendance');
    if (!saved) return INITIAL_ATTENDANCE;
    try {
      const parsed: AttendanceRecord[] = JSON.parse(saved);
      // Strictly purge hardcoded legacy mock attendance (emp-1 through emp-8)
      return parsed.filter((a) => !/^emp-[1-8]$/i.test(a.employeeId));
    } catch {
      return INITIAL_ATTENDANCE;
    }
  });

  // Synchronize registered employees & attendance from FastAPI backend
  useEffect(() => {
    let isMounted = true;
    const fetchBackendEmployees = async () => {
      try {
        const res = await fetch(`${FASTAPI_BASE_URL}/api/v1/employees`);
        if (res.ok) {
          const data = await res.json();
          if (isMounted && Array.isArray(data)) {
            const mapped: Employee[] = data.map((d: any) => {
              const photo = d.photo_url
                ? (d.photo_url.startsWith('http') || d.photo_url.startsWith('data:') ? d.photo_url : `${FASTAPI_BASE_URL}${d.photo_url}`)
                : 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80';
              return {
                id: d.employee_id,
                name: d.name,
                role: d.designation || 'Staff',
                department: d.department || 'Sales',
                phone: '+91 98450 12345',
                email: `${d.employee_id.toLowerCase().replace(/[^a-z0-9]/g, '')}@capermintdealership.com`,
                avatarUrl: photo,
                facePhotosCount: d.face_photos_count ?? (d.has_face_registered ? 1 : 0),
              };
            });
            setEmployees(mapped);
          }
        }
      } catch (err) {
        console.warn('Backend employees not reachable:', err);
      }
    };

    const fetchBackendAttendance = async () => {
      try {
        const res = await fetch(`${FASTAPI_BASE_URL}/api/v1/attendance/daily`);
        if (res.ok) {
          const data = await res.json();
          if (isMounted && data.records && Array.isArray(data.records)) {
            const todayStr = new Date().toISOString().split('T')[0];
            const mapped: AttendanceRecord[] = data.records.map((r: any, idx: number) => {
              // Employee has checked out only if last_check_out exists and is distinct from first_check_in
              const isSameAsCheckIn = r.first_check_in && r.last_check_out &&
                Math.abs(new Date(r.last_check_out).getTime() - new Date(r.first_check_in).getTime()) < 120000;
              const hasCheckedOut = Boolean(r.last_check_out && !isSameAsCheckIn);

              return {
                id: `att-${r.employee_id}-${r.date || todayStr}-${idx}`,
                employeeId: r.employee_id,
                employeeName: r.name,
                role: r.designation,
                department: r.department,
                avatarUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80',
                date: r.date || todayStr,
                checkInTime: r.first_check_in
                  ? new Date(r.first_check_in).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })
                  : null,
                checkOutTime: hasCheckedOut
                  ? new Date(r.last_check_out).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })
                  : null,
                status: r.status === 'PRESENT' ? 'Present' : r.status === 'LATE' ? 'Late' : r.status === 'HALF_DAY' ? 'Present' : 'Absent',
                workHours: r.total_work_hours || 0,
                method: 'CCTV AI Check-in',
              };
            });
            setAttendance(mapped);
          }
        }
      } catch (err) {
        console.warn('Backend daily attendance not reachable:', err);
      }
    };

    fetchBackendEmployees();
    fetchBackendAttendance();
    const attInterval = setInterval(fetchBackendAttendance, 5000);

    return () => {
      isMounted = false;
      clearInterval(attInterval);
    };
  }, []);

  const [storeHours, setStoreHours] = useState<StoreHoursConfig>(() => {
    const saved = localStorage.getItem('capermint_store_hours') || localStorage.getItem('jms_store_hours');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed.actualFirstEntryToday === '09:34 AM') parsed.actualFirstEntryToday = null;
        if (parsed.actualLastExitToday === '02:28 PM') parsed.actualLastExitToday = null;
        return parsed;
      } catch {}
    }
    return INITIAL_STORE_HOURS;
  });

  const [showroomSettings, setShowroomSettings] = useState<ShowroomSettings>(() => {
    const saved = localStorage.getItem('capermint_settings') || localStorage.getItem('jms_settings');
    return saved ? JSON.parse(saved) : INITIAL_SHOWROOM_SETTINGS;
  });

  const [dateFilter, setDateFilter] = useState<DateFilterOption>('today');
  const [customDateRange, setCustomDateRange] = useState({ start: '2026-10-20', end: '2026-10-26' });
  const [isLiveSimulationActive, setIsLiveSimulationActive] = useState<boolean>(false);
  const [lastSimulatedEvent, setLastSimulatedEvent] = useState<string | null>(null);

  // Sync state to LocalStorage safely (stripping oversized base64 to avoid browser QuotaExceededError)
  useEffect(() => {
    try {
      localStorage.setItem('capermint_visitors', JSON.stringify(visitors));
    } catch (e) {
      console.warn('LocalStorage save failed for visitors:', e);
    }
  }, [visitors]);

  useEffect(() => {
    try {
      const lightweightAttendance = attendance.map((a) => ({
        ...a,
        avatarUrl: a.avatarUrl && a.avatarUrl.startsWith('data:') && a.avatarUrl.length > 20000
          ? 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80'
          : a.avatarUrl,
      }));
      localStorage.setItem('capermint_attendance', JSON.stringify(lightweightAttendance));
    } catch (e) {
      console.warn('LocalStorage save failed for attendance:', e);
    }
  }, [attendance]);

  useEffect(() => {
    try {
      const lightweightEmployees = employees.map((e) => ({
        ...e,
        photoUrls: undefined,
        avatarUrl: e.avatarUrl && e.avatarUrl.startsWith('data:') && e.avatarUrl.length > 20000
          ? 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80'
          : e.avatarUrl,
      }));
      localStorage.setItem('capermint_employees', JSON.stringify(lightweightEmployees));
    } catch (e) {
      console.warn('LocalStorage save failed for employees:', e);
    }
  }, [employees]);

  useEffect(() => {
    try {
      localStorage.setItem('capermint_store_hours', JSON.stringify(storeHours));
    } catch (e) {
      console.warn('LocalStorage save failed for store hours:', e);
    }
  }, [storeHours]);

  // Synchronize actual first entry and last exit based on genuine visitor entries for today
  useEffect(() => {
    const todayStr = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(new Date());
    const todayVis = visitors.filter((v) => !v.date || v.date === todayStr);
    if (todayVis.length === 0) {
      setStoreHours((prev) => {
        if (prev.actualFirstEntryToday || prev.actualLastExitToday) {
          return { ...prev, actualFirstEntryToday: null, actualLastExitToday: null };
        }
        return prev;
      });
    } else {
      const sortedByTime = [...todayVis].filter((v) => v.firstSeenTime);
      const firstEntry = sortedByTime.length > 0 ? sortedByTime[sortedByTime.length - 1].firstSeenTime : null;
      const exited = todayVis.filter((v) => v.exitTime);
      const lastExit = exited.length > 0 ? exited[0].exitTime : null;
      setStoreHours((prev) => {
        if (prev.actualFirstEntryToday !== firstEntry || prev.actualLastExitToday !== lastExit) {
          return { ...prev, actualFirstEntryToday: firstEntry, actualLastExitToday: lastExit };
        }
        return prev;
      });
    }
  }, [visitors]);

  useEffect(() => {
    try {
      localStorage.setItem('capermint_settings', JSON.stringify(showroomSettings));
    } catch (e) {
      console.warn('LocalStorage save failed for settings:', e);
    }
  }, [showroomSettings]);

  // Synchronize dynamic waiting & camera settings from camera_config.json
  useEffect(() => {
    let isMounted = true;
    const fetchCameraConfig = async () => {
      try {
        const cfg = await cameraService.getCameraConfig();
        if (isMounted && cfg) {
          const cutoffMin = (cfg as any).waiting_cutoff_seconds
            ? Math.round((cfg as any).waiting_cutoff_seconds / 60)
            : 15;
          setShowroomSettings((prev) => ({
            ...prev,
            waitingConfig: {
              ...(prev.waitingConfig || {
                warningThresholdMinutes: 5,
                staffProximityMeters: 2.2,
                soundAlertEnabled: true,
                excludeDepartmentZones: true,
              }),
              slaCutoffMinutes: cutoffMin,
            },
          }));
        }
      } catch (err) {
        console.warn('Failed to sync camera config to ShowroomContext', err);
      }
    };
    fetchCameraConfig();
  }, []);

  // Determine if store is currently open based on local IST time
  const isStoreCurrentlyOpen = useMemo(() => {
    const now = new Date();
    const currentHours = now.getHours();
    const currentMinutes = now.getMinutes();
    const currentTimeInMinutes = currentHours * 60 + currentMinutes;

    const [openH, openM] = storeHours.openingTime.split(':').map(Number);
    const [closeH, closeM] = storeHours.closingTime.split(':').map(Number);

    const openTimeInMinutes = openH * 60 + openM;
    const closeTimeInMinutes = closeH * 60 + closeM;

    return currentTimeInMinutes >= openTimeInMinutes && currentTimeInMinutes < closeTimeInMinutes;
  }, [storeHours.openingTime, storeHours.closingTime]);

  // Tag visitor
  const tagVisitor = (visitorId: string, purpose: VisitPurpose, salesperson: string, notes: string) => {
    const target = visitors.find((v) => v.id === visitorId || v.visitorId === visitorId);
    const pidToTag = target?.visitorId || target?.id || visitorId;

    setVisitors((prev) =>
      prev.map((v) => {
        if (v.id === visitorId || v.visitorId === visitorId) {
          return {
            ...v,
            visitPurpose: purpose,
            assignedSalesperson: salesperson,
            notes: notes || v.notes,
            taggingStatus: 'Tagged' as TaggingStatus,
          };
        }
        return v;
      })
    );
    // Sync with backend API using canonical visitor ID
    cameraService.tagVisitor(pidToTag, purpose, salesperson, notes).catch((err) => {
      console.warn('Failed to persist visitor tag to server:', err);
    });
  };

  // Synchronize live visitors from backend (including AI detected face photos)
  const refreshVisitors = async () => {
    try {
      const data = await cameraService.listVisitors();
      if (data && Array.isArray(data.visitors)) {
        const mapped: VisitorRecord[] = data.visitors.map((v: any) => ({
          id: v.id,
          visitorId: v.visitorId,
          avatarUrl: v.avatarUrl
            ? (v.avatarUrl.startsWith('http') ? v.avatarUrl : `${FASTAPI_BASE_URL}${v.avatarUrl}`)
            : undefined,
          firstSeenTime: v.firstSeenTime,
          lastSeenTime: v.lastSeenTime,
          exitTime: v.exitTime,
          visitorType: (v.visitorType as 'New' | 'Returning') || 'New',
          visitPurpose: v.visitPurpose || null,
          assignedSalesperson: v.assignedSalesperson || null,
          durationMinutes: v.durationMinutes || 1,
          notes: v.notes || '',
          taggingStatus: (v.taggingStatus as 'Tagged' | 'Untagged') || (v.visitPurpose ? 'Tagged' : 'Untagged'),
          date: v.date,
          genderEstimated: v.genderEstimated || 'Male',
          ageGroupEstimated: v.ageGroupEstimated || '26-35',
          zoneEntered: v.zoneEntered || 'Main Showroom',
          confidenceScore: v.confidenceScore || 0.96,
          cameraSource: v.cameraSource || 'Entrance Camera',
          zoneDwell: v.zoneDwell || { [v.zoneEntered || 'Main Showroom']: `${v.durationMinutes || 1} mins` },
          waitingSeconds: v.waitingSeconds || 0,
          isWaiting: Boolean(v.isWaiting && !v.exitTime && !v.assignedSalesperson),
          waitingBreach: Boolean((v.waitingBreach || (v.waitingSeconds && v.waitingSeconds >= 900)) && !v.exitTime && !v.assignedSalesperson),
          waitingFormatted: v.waitingFormatted || (v.waitingSeconds ? `${Math.floor(v.waitingSeconds / 60)}m ${v.waitingSeconds % 60}s` : '0s'),
        }));

        setVisitors(mapped);
      }
    } catch (_) {}
  };

  const clearAllVisitors = async () => {
    try {
      await cameraService.clearAllVisitors();
    } catch (e) {
      console.warn('Failed to clear visitors on backend:', e);
    }
    setVisitors([]);
    setStoreHours((prev) => ({
      ...prev,
      actualFirstEntryToday: null,
      actualLastExitToday: null,
    }));
    setSectionCounts({ sales: 0, accounts: 0 });
    setSectionOccupancy({ sales: 0, accounts: 0 });
    setStaffDepartmentDwell({});
    setLastSyncedFootfall(0);
    localStorage.removeItem('capermint_visitors');
    localStorage.removeItem('jms_visitors');
  };

  useEffect(() => {
    refreshVisitors();
    const visInterval = setInterval(refreshVisitors, 2500);
    return () => clearInterval(visInterval);
  }, []);

  // Add new visitor
  const addNewVisitor = (record: Partial<VisitorRecord>) => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
    const count = visitors.length + 1;
    const newRecord: VisitorRecord = {
      id: `vis-${Date.now()}`,
      visitorId: `VIS-2026-${1040 + count}`,
      avatarUrl: record.avatarUrl || undefined,
      firstSeenTime: timeStr,
      lastSeenTime: null,
      exitTime: null,
      visitorType: (record.visitorType as 'New' | 'Returning') || 'New',
      visitPurpose: record.visitPurpose || null,
      assignedSalesperson: record.assignedSalesperson || null,
      durationMinutes: 1,
      notes: record.notes || 'Front desk manual walk-in entry',
      taggingStatus: record.visitPurpose ? 'Tagged' : 'Untagged',
      date: new Date().toISOString().split('T')[0],
      genderEstimated: record.genderEstimated || 'Male',
      ageGroupEstimated: record.ageGroupEstimated || '26-35',
      zoneEntered: record.zoneEntered || 'Main Showroom',
      confidenceScore: 0.96,
      cameraSource: record.cameraSource || 'Reception Front Desk',
    };

    setVisitors((prev) => [newRecord, ...prev]);

    // Persist to backend database as manual walk-in
    cameraService.createWalkIn({
      visit_purpose: record.visitPurpose || undefined,
      assigned_salesperson: record.assignedSalesperson || undefined,
      notes: record.notes || undefined,
    }).then(() => {
      refreshVisitors();
    }).catch(() => {});

    // Update actual last exit or first entry in store hours if needed
    if (!storeHours.actualFirstEntryToday) {
      setStoreHours((prev) => ({ ...prev, actualFirstEntryToday: timeStr }));
    }
  };

  // Checkout visitor
  const checkoutVisitor = (visitorId: string) => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });

    setVisitors((prev) =>
      prev.map((v) => {
        if (v.id === visitorId || v.visitorId === visitorId) {
          const duration = Math.max(v.durationMinutes, Math.floor(Math.random() * 25) + 15);
          return {
            ...v,
            lastSeenTime: timeStr,
            exitTime: timeStr,
            durationMinutes: duration,
            zoneDwell: v.zoneDwell || { [v.zoneEntered || 'Main Showroom']: `${duration} mins` },
          };
        }
        return v;
      })
    );

    const target = visitors.find((v) => v.id === visitorId || v.visitorId === visitorId);
    if (target) {
      cameraService.checkoutVisitor(target.visitorId || target.id, timeStr).catch(() => {});
    }

    setStoreHours((prev) => ({ ...prev, actualLastExitToday: timeStr }));
  };

  // Attendance & Employee management
  const addEmployee = (newEmployee: Employee, initialAttendance?: AttendanceRecord) => {
    setEmployees((prev) => {
      const filtered = prev.filter((e) => e.id !== newEmployee.id);
      return [newEmployee, ...filtered];
    });
    if (initialAttendance) {
      setAttendance((prev) => {
        const filtered = prev.filter((a) => a.employeeId !== newEmployee.id);
        return [initialAttendance, ...filtered];
      });
    } else {
      const todayStr = new Date().toISOString().split('T')[0];
      const newAttendanceRecord: AttendanceRecord = {
        id: `att-${Date.now()}`,
        employeeId: newEmployee.id,
        employeeName: newEmployee.name,
        role: newEmployee.role,
        department: newEmployee.department,
        avatarUrl: newEmployee.avatarUrl,
        date: todayStr,
        checkInTime: null,
        checkOutTime: null,
        status: 'Absent',
        workHours: 0,
        method: 'CCTV AI Check-in',
      };
      setAttendance((prev) => {
        const filtered = prev.filter((a) => a.employeeId !== newEmployee.id);
        return [newAttendanceRecord, ...filtered];
      });
    }
  };

  const addEmployeePhotos = (employeeId: string, additionalPhotosCount: number, newAvatarUrl?: string) => {
    setEmployees((prev) =>
      prev.map((e) => {
        if (e.id === employeeId) {
          const currentCount = e.facePhotosCount || 1;
          return {
            ...e,
            avatarUrl: newAvatarUrl || e.avatarUrl,
            facePhotosCount: currentCount + additionalPhotosCount,
          };
        }
        return e;
      })
    );
    setAttendance((prev) =>
      prev.map((a) => {
        if (a.employeeId === employeeId) {
          const currentCount = a.facePhotosCount || 1;
          return {
            ...a,
            avatarUrl: newAvatarUrl || a.avatarUrl,
            facePhotosCount: currentCount + additionalPhotosCount,
          };
        }
        return a;
      })
    );
  };

  const refreshEmployees = async () => {
    try {
      const res = await fetch(`${FASTAPI_BASE_URL}/api/v1/employees`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data)) {
          const mapped: Employee[] = data.map((d: any) => {
            const photo = d.photo_url
              ? (d.photo_url.startsWith('http') || d.photo_url.startsWith('data:') ? d.photo_url : `${FASTAPI_BASE_URL}${d.photo_url}`)
              : '';
            return {
              id: d.employee_id,
              name: d.name,
              role: d.designation || 'Staff',
              department: d.department || 'Sales',
              phone: '+91 98450 12345',
              email: `${d.employee_id.toLowerCase().replace(/[^a-z0-9]/g, '')}@capermintdealership.com`,
              avatarUrl: photo,
              facePhotosCount: d.face_photos_count ?? (d.has_face_registered ? 1 : 0),
            };
          });
          setEmployees(mapped);
        }
      }
    } catch (err) {
      console.warn('Backend refreshEmployees failed:', err);
    }
  };

  const markAttendance = (
    recordId: string,
    status: AttendanceRecord['status'],
    checkInTime?: string,
    checkOutTime?: string
  ) => {
    setAttendance((prev) =>
      prev.map((item) => {
        if (item.id === recordId) {
          return {
            ...item,
            status,
            checkInTime: checkInTime !== undefined ? checkInTime : item.checkInTime,
            checkOutTime: checkOutTime !== undefined ? checkOutTime : item.checkOutTime,
            method: 'Manual Override',
          };
        }
        return item;
      })
    );
  };

  const quickCheckInEmployee = (employeeId: string) => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
    setAttendance((prev) =>
      prev.map((item) => {
        if (item.employeeId === employeeId) {
          return {
            ...item,
            checkInTime: timeStr,
            status: 'Present',
            method: 'Biometric',
          };
        }
        return item;
      })
    );
  };

  const quickCheckOutEmployee = (employeeId: string) => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
    setAttendance((prev) =>
      prev.map((item) => {
        if (item.employeeId === employeeId) {
          return {
            ...item,
            checkOutTime: timeStr,
            method: 'Biometric',
          };
        }
        return item;
      })
    );
  };

  const updateStoreHours = (newHours: Partial<StoreHoursConfig>) => {
    setStoreHours((prev) => ({ ...prev, ...newHours }));
  };

  const updateShowroomSettings = (newSettings: Partial<ShowroomSettings>) => {
    setShowroomSettings((prev) => ({ ...prev, ...newSettings }));
  };

  const waitingConfig: WaitingTimeConfig = useMemo(() => {
    return {
      slaCutoffMinutes: showroomSettings.waitingConfig?.slaCutoffMinutes ?? 15,
      warningThresholdMinutes: showroomSettings.waitingConfig?.warningThresholdMinutes ?? 5,
      staffProximityMeters: showroomSettings.waitingConfig?.staffProximityMeters ?? 2.2,
      soundAlertEnabled: showroomSettings.waitingConfig?.soundAlertEnabled ?? true,
      excludeDepartmentZones: showroomSettings.waitingConfig?.excludeDepartmentZones ?? true,
    };
  }, [showroomSettings.waitingConfig]);

  const updateWaitingConfig = (config: Partial<WaitingTimeConfig>) => {
    setShowroomSettings((prev) => {
      const current = prev.waitingConfig || {
        slaCutoffMinutes: 15,
        warningThresholdMinutes: 5,
        staffProximityMeters: 2.2,
        soundAlertEnabled: true,
        excludeDepartmentZones: true,
      };
      return {
        ...prev,
        waitingConfig: {
          ...current,
          ...config,
        },
      };
    });
  };

  const toggleLiveSimulation = () => {
    setIsLiveSimulationActive((prev) => !prev);
  };

  // Live CCTV Event Simulator (Disabled: strictly real detections from edge camera)
  useEffect(() => {
    // No mock visitors are generated; visitor registry is strictly populated via local camera AI detections
  }, []);

  // Section tracking state
  const [sectionCounts, setSectionCounts] = useState<{ sales: number; accounts: number; [key: string]: number }>({
    sales: 0,
    accounts: 0,
  });
  const [sectionOccupancy, setSectionOccupancy] = useState<{ sales: number; accounts: number; [key: string]: number }>({
    sales: 0,
    accounts: 0,
  });
  const [staffDepartmentDwell, setStaffDepartmentDwell] = useState<Record<string, {
    name: string;
    department_id: string;
    department_name: string;
    dwell_seconds: number;
    dwell_formatted: string;
    is_active?: boolean;
    total_seconds?: number;
    total_formatted?: string;
    visits?: number;
  }>>({});

  const [departments, setDepartments] = useState<DepartmentModel[]>([]);

  const refreshDepartments = async () => {
    try {
      const data = await cameraService.listDepartments();
      if (data && Array.isArray(data)) {
        setDepartments(data);
        const validIds = new Set(data.filter((d) => d.is_active !== false).map((d) => d.id.toLowerCase()));
        setSectionCounts((prev) => {
          const next: Record<string, number> = {};
          for (const [k, v] of Object.entries(prev)) {
            if (validIds.has(k.toLowerCase())) {
              next[k] = v;
            }
          }
          return next as any;
        });
        setSectionOccupancy((prev) => {
          const next: Record<string, number> = {};
          for (const [k, v] of Object.entries(prev)) {
            if (validIds.has(k.toLowerCase())) {
              next[k] = v;
            }
          }
          return next as any;
        });
      }
    } catch (e) {
      console.warn('[ShowroomContext] Failed to load departments:', e);
    }
  };

  useEffect(() => {
    refreshDepartments();
    const deptInterval = setInterval(refreshDepartments, 3000);
    return () => clearInterval(deptInterval);
  }, []);

  // Real-time backend footfall polling (receives live entries from local AI detector / edge client)
  const [lastSyncedFootfall, setLastSyncedFootfall] = useState<number | null>(null);

  useEffect(() => {
    let isMounted = true;

    const pollFootfall = async () => {
      try {
        const data = await cameraService.getFootfall();
        if (!isMounted || !data) return;

        const activeIds = new Set(departments.filter((d) => d.is_active !== false).map((d) => d.id.toLowerCase()));

        if (data.section_counts) {
          setSectionCounts((prev) => {
            const next: Record<string, number> = {};
            const source = { ...prev, ...data.section_counts };
            for (const [k, v] of Object.entries(source)) {
              if (activeIds.size === 0 || activeIds.has(k.toLowerCase())) {
                next[k] = v;
              }
            }
            return next as any;
          });
        }
        if (data.section_occupancy) {
          setSectionOccupancy((prev) => {
            const next: Record<string, number> = {};
            const source = { ...prev, ...data.section_occupancy };
            for (const [k, v] of Object.entries(source)) {
              if (activeIds.size === 0 || activeIds.has(k.toLowerCase())) {
                next[k] = v;
              }
            }
            return next as any;
          });
        }
        if (data.staff_dwell) {
          setStaffDepartmentDwell(data.staff_dwell);
        }

        const backendTotal = typeof data.total_entries === 'number' ? data.total_entries : 0;

        // Sync edge detector line crossing count
        if (lastSyncedFootfall === null || lastSyncedFootfall !== backendTotal) {
          if (lastSyncedFootfall !== null && backendTotal > lastSyncedFootfall) {
            const newEntriesCount = backendTotal - lastSyncedFootfall;
            setLastSimulatedEvent(`Local AI Edge: Registered ${newEntriesCount} new visitor entry count from local camera.`);
          }
          setLastSyncedFootfall(backendTotal);
        }
      } catch {
        // Ignore connection hiccups
      }
    };

    pollFootfall();
    const interval = setInterval(pollFootfall, 3000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [lastSyncedFootfall]);

  // Dynamic analytics calculations
  const metrics = useMemo(() => {
    // Current IST date strings
    const todayStr = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(new Date());
    const yesterdayObj = new Date();
    yesterdayObj.setDate(yesterdayObj.getDate() - 1);
    const yesterdayStr = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(yesterdayObj);

    // Active visitors filtered by active dateFilter
    let activeVisitors = visitors;
    if (dateFilter === 'today') {
      activeVisitors = visitors.filter((v) => !v.date || v.date === todayStr);
    } else if (dateFilter === 'yesterday') {
      activeVisitors = visitors.filter((v) => v.date === yesterdayStr);
    }

    // Total visitors
    const totalVisitors = activeVisitors.length;
    const newVisitors = activeVisitors.filter((v) => v.visitorType === 'New').length;
    const returningVisitors = activeVisitors.filter((v) => v.visitorType === 'Returning').length;
    const newVisitorPct = totalVisitors > 0 ? Math.round((newVisitors / totalVisitors) * 100) : 0;
    const returningVisitorPct = totalVisitors > 0 ? Math.round((returningVisitors / totalVisitors) * 100) : 0;

    // Real yesterday count & comparison
    const yesterdayVisitors = visitors.filter((v) => v.date === yesterdayStr).length;
    let comparisonYesterdayPct = 0;
    if (yesterdayVisitors > 0) {
      comparisonYesterdayPct = Math.round(((totalVisitors - yesterdayVisitors) / yesterdayVisitors) * 100);
    } else if (totalVisitors > 0) {
      comparisonYesterdayPct = 100;
    } else {
      comparisonYesterdayPct = 0;
    }

    // Employees attendance
    const employeesTotal = attendance.length;
    const employeesPresent = attendance.filter((a) => a.status === 'Present').length;
    const employeesAbsent = attendance.filter((a) => a.status === 'Absent').length;
    const employeesLate = attendance.filter((a) => a.status === 'Late').length;
    const employeesOnLeave = attendance.filter((a) => a.status === 'On Leave').length;
    const employeesOnDuty = attendance.filter((a) => a.status === 'Present' || a.status === 'Late').length;

    // Real-time hourly footfall distribution across 11 contiguous bins [09:00 - 20:00]
    const BASE_HOURLY_SLOTS = [
      { hour: '09:00 - 10:00 AM', startH: 9, endH: 10 },
      { hour: '10:00 - 11:00 AM', startH: 10, endH: 11 },
      { hour: '11:00 - 12:00 PM', startH: 11, endH: 12 },
      { hour: '12:00 - 01:00 PM', startH: 12, endH: 13 },
      { hour: '01:00 - 02:00 PM', startH: 13, endH: 14 },
      { hour: '02:00 - 03:00 PM', startH: 14, endH: 15 },
      { hour: '03:00 - 04:00 PM', startH: 15, endH: 16 },
      { hour: '04:00 - 05:00 PM', startH: 16, endH: 17 },
      { hour: '05:00 - 06:00 PM', startH: 17, endH: 18 },
      { hour: '06:00 - 07:00 PM', startH: 18, endH: 19 },
      { hour: '07:00 - 08:00 PM', startH: 19, endH: 20 },
    ];

    const hourlyChartData: HourlyFootfall[] = BASE_HOURLY_SLOTS.map((s) => ({
      hour: s.hour,
      visitors: 0,
      newVisitors: 0,
      returningVisitors: 0,
      testRides: 0,
      deliveries: 0,
    }));

    activeVisitors.forEach((v) => {
      if (!v.firstSeenTime) return;
      let hour24 = -1;
      const match = v.firstSeenTime.match(/(\d+):(\d+)\s*(AM|PM)?/i);
      if (match) {
        let h = parseInt(match[1], 10);
        const ampm = match[3]?.toUpperCase();
        if (ampm === 'PM' && h < 12) h += 12;
        if (ampm === 'AM' && h === 12) h = 0;
        hour24 = h;
      }
      if (hour24 >= 0) {
        const slotIdx = BASE_HOURLY_SLOTS.findIndex((s) => hour24 >= s.startH && hour24 < s.endH);
        const targetIdx = slotIdx !== -1 ? slotIdx : (hour24 < 9 ? 0 : BASE_HOURLY_SLOTS.length - 1);
        hourlyChartData[targetIdx].visitors += 1;
        if (v.visitorType === 'Returning') {
          hourlyChartData[targetIdx].returningVisitors += 1;
        } else {
          hourlyChartData[targetIdx].newVisitors += 1;
        }
        if (v.visitPurpose === 'Test Ride') {
          hourlyChartData[targetIdx].testRides += 1;
        } else if (v.visitPurpose === 'Vehicle Delivery') {
          hourlyChartData[targetIdx].deliveries += 1;
        }
      }
    });

    let busiestHour = 'N/A';
    let busiestHourCount = 0;

    hourlyChartData.forEach((slot) => {
      if (slot.visitors > busiestHourCount) {
        busiestHour = slot.hour;
        busiestHourCount = slot.visitors;
      }
    });

    const morningTraffic = hourlyChartData.slice(0, 4).reduce((sum, s) => sum + s.visitors, 0);
    const afternoonTraffic = hourlyChartData.slice(4, 8).reduce((sum, s) => sum + s.visitors, 0);
    const eveningTraffic = hourlyChartData.slice(8, 11).reduce((sum, s) => sum + s.visitors, 0);

    // Duration calculation for completed visits
    const finishedVisits = activeVisitors.filter((v) => v.exitTime !== null && v.durationMinutes > 0);
    const durations = finishedVisits.map((v) => v.durationMinutes).sort((a, b) => a - b);

    const avgDurationMinutes =
      durations.length > 0 ? Math.round(durations.reduce((sum, d) => sum + d, 0) / durations.length) : 0;

    let medianDurationMinutes = 0;
    if (durations.length > 0) {
      const mid = Math.floor(durations.length / 2);
      medianDurationMinutes =
        durations.length % 2 !== 0 ? durations[mid] : Math.round((durations[mid - 1] + durations[mid]) / 2);
    }

    const shortestDurationMinutes = durations.length > 0 ? durations[0] : 0;
    const longestDurationMinutes = durations.length > 0 ? durations[durations.length - 1] : 0;

    // Duration distribution buckets
    const bucketCounts = {
      '< 10 mins': 0,
      '10 - 20 mins': 0,
      '20 - 40 mins': 0,
      '40 - 60 mins': 0,
      '60+ mins': 0,
    };

    finishedVisits.forEach((v) => {
      if (v.durationMinutes < 10) bucketCounts['< 10 mins']++;
      else if (v.durationMinutes <= 20) bucketCounts['10 - 20 mins']++;
      else if (v.durationMinutes <= 40) bucketCounts['20 - 40 mins']++;
      else if (v.durationMinutes <= 60) bucketCounts['40 - 60 mins']++;
      else bucketCounts['60+ mins']++;
    });

    const totalFinished = finishedVisits.length;
    const durationDistribution: DurationDistribution[] = [
      {
        range: '< 10 mins (Quick inquiry)',
        count: bucketCounts['< 10 mins'],
        percentage: totalFinished > 0 ? Math.round((bucketCounts['< 10 mins'] / totalFinished) * 100) : 0,
        avgConversion: 0,
      },
      {
        range: '10 - 20 mins (Product look)',
        count: bucketCounts['10 - 20 mins'],
        percentage: totalFinished > 0 ? Math.round((bucketCounts['10 - 20 mins'] / totalFinished) * 100) : 0,
        avgConversion: 0,
      },
      {
        range: '20 - 40 mins (Discussion/Demo)',
        count: bucketCounts['20 - 40 mins'],
        percentage: totalFinished > 0 ? Math.round((bucketCounts['20 - 40 mins'] / totalFinished) * 100) : 0,
        avgConversion: 0,
      },
      {
        range: '40 - 60 mins (Test ride/Finance)',
        count: bucketCounts['40 - 60 mins'],
        percentage: totalFinished > 0 ? Math.round((bucketCounts['40 - 60 mins'] / totalFinished) * 100) : 0,
        avgConversion: 0,
      },
      {
        range: '60+ mins (Booking/Delivery)',
        count: bucketCounts['60+ mins'],
        percentage: totalFinished > 0 ? Math.round((bucketCounts['60+ mins'] / totalFinished) * 100) : 0,
        avgConversion: 0,
      },
    ];

    // Visit purpose distribution
    const taggedVisits = activeVisitors.filter((v) => v.visitPurpose !== null);
    const untaggedCount = activeVisitors.filter((v) => !v.visitPurpose).length;
    const taggedCount = taggedVisits.length;

    const purposeCounts: Record<VisitPurpose, number> = {
      'Test Ride': 0,
      'Vehicle Enquiry': 0,
      Booking: 0,
      Service: 0,
      'Vehicle Delivery': 0,
      'Spare Parts': 0,
      'General Enquiry': 0,
      Other: 0,
    };

    taggedVisits.forEach((v) => {
      if (v.visitPurpose) {
        purposeCounts[v.visitPurpose] = (purposeCounts[v.visitPurpose] || 0) + 1;
      }
    });

    const purposeDistribution: PurposeDistribution[] = (Object.keys(purposeCounts) as VisitPurpose[]).map((p) => {
      const count = purposeCounts[p];
      return {
        purpose: p,
        count,
        percentage: totalVisitors > 0 ? Math.round((count / totalVisitors) * 100) : 0,
        color: PURPOSE_COLORS[p] || '#94a3b8',
      };
    });

    const cutoffSec = (waitingConfig.slaCutoffMinutes || 15) * 60;
    const unattendedWaitingCount = activeVisitors.filter((v) => v.isWaiting && !v.exitTime && !v.assignedSalesperson).length;
    const breachVisitors = activeVisitors
      .filter((v) => !v.exitTime && !v.assignedSalesperson && Boolean((v.waitingBreach && v.isWaiting) || ((v.waitingSeconds || 0) >= cutoffSec && v.isWaiting)))
      .sort((a, b) => (b.waitingSeconds || 0) - (a.waitingSeconds || 0));
    const waitingBreachCount = breachVisitors.length;
    const currentlyWaitingVisitors = activeVisitors
      .filter((v) => v.isWaiting && !v.exitTime && !v.assignedSalesperson)
      .sort((a, b) => (b.waitingSeconds || 0) - (a.waitingSeconds || 0));

    const waitingVisits = activeVisitors.filter((v) => (v.waitingSeconds || 0) > 0);
    const avgWaitingSeconds = waitingVisits.length > 0
      ? Math.round(waitingVisits.reduce((acc, v) => acc + (v.waitingSeconds || 0), 0) / waitingVisits.length)
      : 0;
    const wMins = Math.floor(avgWaitingSeconds / 60);
    const wSecs = avgWaitingSeconds % 60;
    const avgWaitingFormatted = wMins > 0 ? `${wMins}m ${wSecs < 10 ? '0' : ''}${wSecs}s` : `${wSecs}s`;

    const longestWaitingSeconds = activeVisitors.length > 0
      ? Math.max(...activeVisitors.map((v) => v.waitingSeconds || 0), 0)
      : 0;

    const slaCompliancePct = totalVisitors > 0
      ? Math.max(0, Math.round(((totalVisitors - waitingBreachCount) / totalVisitors) * 100))
      : 100;

    const virtualLineCrossings = lastSyncedFootfall ?? 0;

    return {
      virtualLineCrossings,
      totalVisitors,
      newVisitors,
      returningVisitors,
      newVisitorPct,
      returningVisitorPct,
      comparisonYesterdayPct,
      yesterdayVisitors,
      employeesTotal,
      employeesPresent,
      employeesAbsent,
      employeesOnDuty,
      employeesLate,
      employeesOnLeave,
      busiestHour,
      busiestHourCount,
      morningTraffic,
      afternoonTraffic,
      eveningTraffic,
      avgDurationMinutes,
      medianDurationMinutes,
      shortestDurationMinutes,
      longestDurationMinutes,
      untaggedCount,
      taggedCount,
      hourlyChartData,
      durationDistribution,
      purposeDistribution,
      sectionCounts,
      sectionOccupancy,
      unattendedWaitingCount,
      waitingBreachCount,
      breachVisitors,
      avgWaitingSeconds,
      avgWaitingFormatted,
      longestWaitingSeconds,
      slaCompliancePct,
      currentlyWaitingVisitors,
    };
  }, [visitors, attendance, sectionCounts, sectionOccupancy, dateFilter, waitingConfig, lastSyncedFootfall]);

  // Export functions
  const exportVisitorsCSV = () => {
    const headers = [
      'Visitor ID',
      'Date',
      'First Seen Time',
      'Last Seen Time',
      'Exit Time',
      'Type',
      'Visit Purpose',
      'Assigned Salesperson',
      'Duration (Mins)',
      'Wait Time',
      'Waiting SLA Status',
      'Tagging Status',
      'Zone',
      'Notes',
    ];

    const rows = visitors.map((v) => [
      v.visitorId,
      v.date,
      v.firstSeenTime,
      v.lastSeenTime || 'In Showroom',
      v.exitTime || 'Active',
      v.visitorType,
      v.visitPurpose || 'Untagged',
      v.assignedSalesperson || 'Unassigned',
      v.durationMinutes,
      v.waitingFormatted || `${v.waitingSeconds || 0}s`,
      v.waitingBreach ? 'SLA Breach (>15m)' : v.isWaiting ? 'Currently Waiting' : 'Attended',
      v.taggingStatus,
      v.zoneEntered || 'Showroom Floor',
      `"${(v.notes || '').replace(/"/g, '""')}"`,
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `Showroom_Visitors_Report_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportAttendanceCSV = () => {
    const headers = [
      'Employee ID',
      'Name',
      'Role',
      'Department',
      'Date',
      'Check-in Time',
      'Check-out Time',
      'Status',
      'Work Hours',
      'Method',
    ];

    const rows = attendance.map((a) => [
      a.employeeId,
      `"${a.employeeName}"`,
      `"${a.role}"`,
      `"${a.department}"`,
      a.date,
      a.checkInTime || 'N/A',
      a.checkOutTime || 'On Duty',
      a.status,
      a.workHours,
      a.method,
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `Showroom_Attendance_Report_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <ShowroomContext.Provider
      value={{
        visitors,
        tagVisitor,
        addNewVisitor,
        checkoutVisitor,
        refreshVisitors,
        clearAllVisitors,
        employees,
        attendance,
        addEmployee,
        addEmployeePhotos,
        refreshEmployees,
        markAttendance,
        quickCheckInEmployee,
        quickCheckOutEmployee,
        storeHours,
        updateStoreHours,
        isStoreCurrentlyOpen,
        showroomSettings,
        updateShowroomSettings,
        waitingConfig,
        updateWaitingConfig,
        dateFilter,
        setDateFilter,
        customDateRange,
        setCustomDateRange,
        isLiveSimulationActive,
        toggleLiveSimulation,
        lastSimulatedEvent,
        metrics,
        lastSyncedFootfall,
        sectionCounts,
        sectionOccupancy,
        staffDepartmentDwell,
        departments,
        refreshDepartments,
        exportVisitorsCSV,
        exportAttendanceCSV,
      }}
    >
      {children}
    </ShowroomContext.Provider>
  );
};

export const useShowroom = () => {
  const context = useContext(ShowroomContext);
  if (!context) {
    throw new Error('useShowroom must be used within a ShowroomProvider');
  }
  return context;
};
