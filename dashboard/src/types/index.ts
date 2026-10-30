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
  zoneEntered?: 'Main Showroom' | 'Display Area' | 'Test Track' | 'Customer Lounge' | 'Service Bay';
  confidenceScore?: number;
  cameraSource?: string;
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
}
