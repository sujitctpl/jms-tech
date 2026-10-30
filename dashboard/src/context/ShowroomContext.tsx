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
import { cameraService } from '../services/cameraService';
import { FASTAPI_BASE_URL } from '../config/env';

interface ShowroomContextType {
  // Visitors
  visitors: VisitorRecord[];
  tagVisitor: (visitorId: string, purpose: VisitPurpose, salesperson: string, notes: string) => void;
  addNewVisitor: (record: Partial<VisitorRecord>) => void;
  checkoutVisitor: (visitorId: string) => void;

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
    totalVisitors: number;
    newVisitors: number;
    returningVisitors: number;
    newVisitorPct: number;
    returningVisitorPct: number;
    comparisonYesterdayPct: number;
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
  };

  // Section Tracking
  sectionCounts: { sales: number; accounts: number; [key: string]: number };
  sectionOccupancy: { sales: number; accounts: number; [key: string]: number };

  // CSV Export
  exportVisitorsCSV: () => void;
  exportAttendanceCSV: () => void;
}

const ShowroomContext = createContext<ShowroomContextType | undefined>(undefined);

export const ShowroomProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [visitors, setVisitors] = useState<VisitorRecord[]>(() => {
    const saved = localStorage.getItem('capermint_visitors') || localStorage.getItem('jms_visitors');
    if (!saved) return INITIAL_VISITORS;
    try {
      const parsed: VisitorRecord[] = JSON.parse(saved);
      // Auto-purge legacy mock seed data (ids starting with vis-10 or vis-11) so values start fresh at 0
      if (parsed.some((v) => v.id.startsWith('vis-10') || v.id.startsWith('vis-11'))) {
        localStorage.removeItem('capermint_visitors');
        localStorage.removeItem('jms_visitors');
        return [];
      }
      return parsed;
    } catch {
      return INITIAL_VISITORS;
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
            const mapped: AttendanceRecord[] = data.records.map((r: any, idx: number) => ({
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
              checkOutTime: r.last_check_out
                ? new Date(r.last_check_out).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })
                : null,
              status: r.status === 'PRESENT' ? 'Present' : r.status === 'LATE' ? 'Late' : r.status === 'HALF_DAY' ? 'Present' : 'Absent',
              workHours: r.total_work_hours || 0,
              method: 'CCTV AI Check-in',
            }));
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
    return saved ? JSON.parse(saved) : INITIAL_STORE_HOURS;
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

  useEffect(() => {
    try {
      localStorage.setItem('capermint_settings', JSON.stringify(showroomSettings));
    } catch (e) {
      console.warn('LocalStorage save failed for settings:', e);
    }
  }, [showroomSettings]);

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
  };

  // Add new visitor
  const addNewVisitor = (record: Partial<VisitorRecord>) => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
    const count = visitors.length + 1;
    const sampleFaces = [
      'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=150&auto=format&fit=crop&q=80',
      'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150&auto=format&fit=crop&q=80',
      'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80',
      'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150&auto=format&fit=crop&q=80',
      'https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=150&auto=format&fit=crop&q=80',
      'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&auto=format&fit=crop&q=80',
    ];
    const newRecord: VisitorRecord = {
      id: `vis-${Date.now()}`,
      visitorId: `VIS-2026-${1040 + count}`,
      avatarUrl: record.avatarUrl || sampleFaces[Math.floor(Math.random() * sampleFaces.length)],
      firstSeenTime: timeStr,
      lastSeenTime: null,
      exitTime: null,
      visitorType: (record.visitorType as 'New' | 'Returning') || 'New',
      visitPurpose: record.visitPurpose || null,
      assignedSalesperson: record.assignedSalesperson || null,
      durationMinutes: 1,
      notes: record.notes || 'Walk-in visitor tracked via CCTV AI Detection Line',
      taggingStatus: record.visitPurpose ? 'Tagged' : 'Untagged',
      date: '2026-10-26',
      genderEstimated: record.genderEstimated || 'Male',
      ageGroupEstimated: record.ageGroupEstimated || '26-35',
      zoneEntered: record.zoneEntered || 'Main Showroom',
      confidenceScore: 0.96,
      cameraSource: 'Main Glass Entrance NVR-01',
    };

    setVisitors((prev) => [newRecord, ...prev]);

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
          };
        }
        return v;
      })
    );

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

  const toggleLiveSimulation = () => {
    setIsLiveSimulationActive((prev) => !prev);
  };

  // Live CCTV Event Simulator (Periodically simulates walk-in detection)
  useEffect(() => {
    if (!isLiveSimulationActive) return;

    const interval = setInterval(() => {
      const actions = ['entry', 'tag', 'checkout'];
      const action = actions[Math.floor(Math.random() * actions.length)];

      if (action === 'entry') {
        const bikePurposes: VisitPurpose[] = ['Test Ride', 'Vehicle Enquiry', 'Booking', 'Service'];
        const randomPurpose = Math.random() > 0.4 ? bikePurposes[Math.floor(Math.random() * bikePurposes.length)] : null;
        const salespeople = employees.length > 0 ? employees.map((e) => e.name) : ['Floor Consultant'];
        const randomSalesperson = randomPurpose ? salespeople[Math.floor(Math.random() * salespeople.length)] : null;
        const isReturning = Math.random() > 0.65;

        addNewVisitor({
          visitorType: isReturning ? 'Returning' : 'New',
          visitPurpose: randomPurpose,
          assignedSalesperson: randomSalesperson,
          notes: randomPurpose
            ? `Customer interested in ${randomPurpose}. Approached floor executive.`
            : 'Unassisted walk-in visitor detected near main entrance standee.',
          genderEstimated: Math.random() > 0.35 ? 'Male' : 'Female',
          zoneEntered: 'Display Area',
        });

        setLastSimulatedEvent(`Live AI Detection: New ${isReturning ? 'Returning' : 'First-time'} visitor entered showroom.`);
      } else if (action === 'checkout') {
        // Find an active visitor
        const active = visitors.find((v) => !v.exitTime);
        if (active) {
          checkoutVisitor(active.id);
          setLastSimulatedEvent(`Live AI Detection: Visitor ${active.visitorId} exited via Main Gate.`);
        }
      }
    }, 12000);

    return () => clearInterval(interval);
  }, [isLiveSimulationActive, visitors]);

  // Section tracking state
  const [sectionCounts, setSectionCounts] = useState<{ sales: number; accounts: number; [key: string]: number }>({
    sales: 0,
    accounts: 0,
  });
  const [sectionOccupancy, setSectionOccupancy] = useState<{ sales: number; accounts: number; [key: string]: number }>({
    sales: 0,
    accounts: 0,
  });

  // Real-time backend footfall polling (receives live entries from local AI detector / edge client)
  const [lastSyncedFootfall, setLastSyncedFootfall] = useState<number | null>(null);

  useEffect(() => {
    let isMounted = true;

    const pollFootfall = async () => {
      try {
        const data = await cameraService.getFootfall();
        if (!isMounted || !data) return;

        if (data.section_counts) {
          setSectionCounts((prev) => ({ ...prev, ...data.section_counts }));
        }
        if (data.section_occupancy) {
          setSectionOccupancy((prev) => ({ ...prev, ...data.section_occupancy }));
        }

        const backendTotal = data.total_entries;

        // On first load, record baseline count
        if (lastSyncedFootfall === null) {
          setLastSyncedFootfall(backendTotal);
          return;
        }

        // When local detector detects new people crossing entry line:
        if (backendTotal > lastSyncedFootfall) {
          const newEntriesCount = backendTotal - lastSyncedFootfall;
          setLastSyncedFootfall(backendTotal);

          for (let i = 0; i < newEntriesCount; i++) {
            addNewVisitor({
              visitorType: Math.random() > 0.3 ? 'New' : 'Returning',
              notes: 'Walk-in visitor tracked via CCTV AI Detection Line (Local Edge Model)',
              cameraSource: 'Showroom Camera 101 (Local Edge)',
              zoneEntered: 'Main Showroom',
              confidenceScore: 0.94,
            });
          }

          setLastSimulatedEvent(`Local AI Edge: Registered ${newEntriesCount} new visitor entry count from local model!`);
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
    // Total visitors
    const totalVisitors = visitors.length;
    const newVisitors = visitors.filter((v) => v.visitorType === 'New').length;
    const returningVisitors = visitors.filter((v) => v.visitorType === 'Returning').length;
    const newVisitorPct = totalVisitors > 0 ? Math.round((newVisitors / totalVisitors) * 100) : 0;
    const returningVisitorPct = totalVisitors > 0 ? Math.round((returningVisitors / totalVisitors) * 100) : 0;

    // Comparison with yesterday (using seed baseline)
    const diff = totalVisitors - YESTERDAY_METRICS.totalVisitors;
    const comparisonYesterdayPct = Math.round((diff / YESTERDAY_METRICS.totalVisitors) * 100);

    // Employees attendance
    const employeesTotal = attendance.length;
    const employeesPresent = attendance.filter((a) => a.status === 'Present').length;
    const employeesAbsent = attendance.filter((a) => a.status === 'Absent').length;
    const employeesLate = attendance.filter((a) => a.status === 'Late').length;
    const employeesOnLeave = attendance.filter((a) => a.status === 'On Leave').length;
    const employeesOnDuty = attendance.filter((a) => a.status === 'Present' || a.status === 'Late').length;

    // Peak hours calculation
    const hourlyMap: Record<string, number> = {};
    TODAY_HOURLY_FOOTFALL.forEach((slot) => {
      hourlyMap[slot.hour] = slot.visitors;
    });

    let busiestHour = TODAY_HOURLY_FOOTFALL[8].hour; // default "05:00 - 06:00 PM"
    let busiestHourCount = TODAY_HOURLY_FOOTFALL[8].visitors;

    TODAY_HOURLY_FOOTFALL.forEach((slot) => {
      if (slot.visitors > busiestHourCount) {
        busiestHour = slot.hour;
        busiestHourCount = slot.visitors;
      }
    });

    // Morning, Afternoon, Evening breakdown
    // Morning: 9 AM - 1 PM (slots 0, 1, 2, 3)
    const morningTraffic = TODAY_HOURLY_FOOTFALL.slice(0, 4).reduce((sum, s) => sum + s.visitors, 0);
    // Afternoon: 1 PM - 5 PM (slots 4, 5, 6, 7)
    const afternoonTraffic = TODAY_HOURLY_FOOTFALL.slice(4, 8).reduce((sum, s) => sum + s.visitors, 0);
    // Evening: 5 PM - 8 PM (slots 8, 9, 10)
    const eveningTraffic = TODAY_HOURLY_FOOTFALL.slice(8, 11).reduce((sum, s) => sum + s.visitors, 0);

    // Duration calculation for completed visits
    const finishedVisits = visitors.filter((v) => v.exitTime !== null && v.durationMinutes > 0);
    const durations = finishedVisits.map((v) => v.durationMinutes).sort((a, b) => a - b);

    const avgDurationMinutes =
      durations.length > 0 ? Math.round(durations.reduce((sum, d) => sum + d, 0) / durations.length) : 26;

    let medianDurationMinutes = 22;
    if (durations.length > 0) {
      const mid = Math.floor(durations.length / 2);
      medianDurationMinutes =
        durations.length % 2 !== 0 ? durations[mid] : Math.round((durations[mid - 1] + durations[mid]) / 2);
    }

    const shortestDurationMinutes = durations.length > 0 ? durations[0] : 8;
    const longestDurationMinutes = durations.length > 0 ? durations[durations.length - 1] : 74;

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

    const totalFinished = Math.max(finishedVisits.length, 1);
    const durationDistribution: DurationDistribution[] = [
      {
        range: '< 10 mins (Quick inquiry)',
        count: bucketCounts['< 10 mins'],
        percentage: Math.round((bucketCounts['< 10 mins'] / totalFinished) * 100),
        avgConversion: 12,
      },
      {
        range: '10 - 20 mins (Product look)',
        count: bucketCounts['10 - 20 mins'],
        percentage: Math.round((bucketCounts['10 - 20 mins'] / totalFinished) * 100),
        avgConversion: 28,
      },
      {
        range: '20 - 40 mins (Discussion/Demo)',
        count: bucketCounts['20 - 40 mins'],
        percentage: Math.round((bucketCounts['20 - 40 mins'] / totalFinished) * 100),
        avgConversion: 54,
      },
      {
        range: '40 - 60 mins (Test ride/Finance)',
        count: bucketCounts['40 - 60 mins'],
        percentage: Math.round((bucketCounts['40 - 60 mins'] / totalFinished) * 100),
        avgConversion: 78,
      },
      {
        range: '60+ mins (Booking/Delivery)',
        count: bucketCounts['60+ mins'],
        percentage: Math.round((bucketCounts['60+ mins'] / totalFinished) * 100),
        avgConversion: 92,
      },
    ];

    // Visit purpose distribution
    const taggedVisits = visitors.filter((v) => v.visitPurpose !== null);
    const untaggedCount = visitors.filter((v) => !v.visitPurpose).length;
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

    return {
      totalVisitors,
      newVisitors,
      returningVisitors,
      newVisitorPct,
      returningVisitorPct,
      comparisonYesterdayPct,
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
      hourlyChartData: TODAY_HOURLY_FOOTFALL,
      durationDistribution,
      purposeDistribution,
      sectionCounts,
      sectionOccupancy,
    };
  }, [visitors, attendance, sectionCounts, sectionOccupancy]);

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
        dateFilter,
        setDateFilter,
        customDateRange,
        setCustomDateRange,
        isLiveSimulationActive,
        toggleLiveSimulation,
        lastSimulatedEvent,
        metrics,
        sectionCounts,
        sectionOccupancy,
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
