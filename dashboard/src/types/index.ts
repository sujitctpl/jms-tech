export type UserRole = 'owner' | 'manager' | 'employee';

export interface User {
  id: string;
  name: string;
  email: string;
  showroomName: string;
  role: UserRole;
  passwordHash?: string;
  avatarUrl?: string;
  createdAt: string;
}

export type VisitPurpose =
  | 'Test Ride'
  | 'Vehicle Enquiry'
  | 'Booking'
  | 'Service'
  | 'Vehicle Delivery'
  | 'Spare Parts'
  | 'General Enquiry'
  | 'Other';

export type TaggingStatus = 'Tagged' | 'Untagged';

export type VisitorType = 'New' | 'Returning';

export interface VisitorRecord {
  id: string;
  visitorId: string;
  avatarUrl?: string; // AI Detected face crop / camera snapshot
  exitAvatarUrl?: string; // AI Detected departure crop snapshot
  firstSeenTime: string; // ISO string or time string
  lastSeenTime: string | null; // null if currently in showroom
  exitTime: string | null;
  visitorType: VisitorType;
  visitPurpose: VisitPurpose | null;
  assignedSalesperson: string | null;
  durationMinutes: number; // calculated from entry to exit, or ongoing duration
  notes: string;
  taggingStatus: TaggingStatus;
  date: string; // YYYY-MM-DD
  genderEstimated?: 'Male' | 'Female' | 'Unspecified';
  ageGroupEstimated?: '18-25' | '26-35' | '36-50' | '50+';
  zoneEntered?: string;
  confidenceScore?: number;
  cameraSource?: string;
  zoneDwell?: Record<string, string>; // e.g. { "Sales Section": "8 mins", "Main Showroom": "2 mins" }
  waitingSeconds?: number;
  isWaiting?: boolean;
  waitingBreach?: boolean;
  waitingFormatted?: string;
}

export type AttendanceStatus = 'Present' | 'Absent' | 'Late' | 'On Leave';

export interface Employee {
  id: string;
  name: string;
  role: string;
  department: 'Sales' | 'Service' | 'Finance & Insurance' | 'Customer Relations' | 'Accessories & Spares';
  phone: string;
  email: string;
  avatarUrl: string;
  photoUrls?: string[];
  facePhotosCount?: number;
}

export interface AttendanceRecord {
  id: string;
  employeeId: string;
  employeeName: string;
  role: string;
  department: string;
  avatarUrl?: string;
  facePhotosCount?: number;
  date: string;
  checkInTime: string | null;
  checkOutTime: string | null;
  status: AttendanceStatus;
  workHours: number;
  method: 'Biometric' | 'CCTV AI Check-in' | 'Manual Override';
}

export interface StoreHoursConfig {
  openingTime: string; // "09:30" (24h format)
  closingTime: string; // "20:30" (24h format)
  timezone: string;
  isOpenSunday: boolean;
  notes: string;
  actualFirstEntryToday: string | null;
  actualLastExitToday: string | null;
}

export type DateFilterOption = 'today' | 'yesterday' | 'last7' | 'last30' | 'custom';

export interface DateRangeFilter {
  option: DateFilterOption;
  startDate: string; // YYYY-MM-DD
  endDate: string; // YYYY-MM-DD
}

export interface HourlyFootfall {
  hour: string; // "09:00", "10:00", etc.
  visitors: number;
  newVisitors: number;
  returningVisitors: number;
  testRides: number;
  deliveries: number;
}

export interface DurationDistribution {
  range: string;
  count: number;
  percentage: number;
  avgConversion: number;
}

export interface PurposeDistribution {
  purpose: VisitPurpose;
  count: number;
  percentage: number;
  color: string;
}

export interface WaitingTimeConfig {
  slaCutoffMinutes: number; // default: 15
  warningThresholdMinutes: number; // default: 5
  staffProximityMeters: number; // default: 2.2
  soundAlertEnabled: boolean; // default: true
  excludeDepartmentZones: boolean; // default: true
}

export interface ShowroomSettings {
  showroomName: string;
  brandDealership: string; // "Honda 2-Wheelers", "Hero MotoCorp", "TVS Motors", "Royal Enfield"
  address: string;
  phone: string;
  email: string;
  cameraStreams: {
    id: string;
    name: string;
    rtspUrl: string;
    zone: string;
    status: 'online' | 'degraded' | 'offline';
    fps: number;
  }[];
  cctvAiEnabled: boolean;
  privacyComplianceEnabled: boolean;
  autoTaggingAssistance: boolean;
  waitingConfig?: WaitingTimeConfig;
}
