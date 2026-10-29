import { Employee, AttendanceRecord, VisitorRecord, StoreHoursConfig, ShowroomSettings, VisitPurpose } from '../types';

export const INITIAL_EMPLOYEES: Employee[] = [];

export const INITIAL_ATTENDANCE: AttendanceRecord[] = [];

export const INITIAL_STORE_HOURS: StoreHoursConfig = {
  openingTime: '09:30',
  closingTime: '20:30',
  timezone: 'Asia/Kolkata (IST +5:30)',
  isOpenSunday: true,
  notes: 'Showroom open 7 days a week. Service workshop closes at 7:00 PM on Sundays.',
  actualFirstEntryToday: '09:34 AM',
  actualLastExitToday: '02:28 PM', // Updated in real time as visitors leave
};

export const INITIAL_SHOWROOM_SETTINGS: ShowroomSettings = {
  showroomName: 'Capermint Showroom',
  brandDealership: 'Capermint Tech Intelligence',
  address: 'Plot 42, Commercial Zone, Tech Park Corridor',
  phone: '+91 11 4509 8800',
  email: 'admin@capermint.com',
  cameraStreams: [
    {
      id: 'cam-entry-1',
      name: 'Main Glass Entrance NVR-01',
      rtspUrl: 'rtsp://192.168.1.120:554/live/stream1',
      zone: 'Main Entrance & Footfall Line',
      status: 'online',
      fps: 25,
    },
    {
      id: 'cam-floor-2',
      name: 'Scooter & Bike Display Floor NVR-02',
      rtspUrl: 'rtsp://192.168.1.121:554/live/stream1',
      zone: 'Display Lounge',
      status: 'online',
      fps: 25,
    },
    {
      id: 'cam-delivery-3',
      name: 'New Vehicle Delivery Bay NVR-03',
      rtspUrl: 'rtsp://192.168.1.122:554/live/stream1',
      zone: 'Ceremony Delivery Area',
      status: 'online',
      fps: 20,
    },
    {
      id: 'cam-service-4',
      name: 'Quick Service Reception NVR-04',
      rtspUrl: 'rtsp://192.168.1.123:554/live/stream1',
      zone: 'Service Entry Gate',
      status: 'online',
      fps: 25,
    },
  ],
  cctvAiEnabled: true,
  privacyComplianceEnabled: true,
  autoTaggingAssistance: true,
};

export const INITIAL_VISITORS: VisitorRecord[] = [];

// Historical footfall for the day (9 AM to 8 PM) - initialized to 0
export const TODAY_HOURLY_FOOTFALL = [
  { hour: '09:00 - 10:00 AM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
  { hour: '10:00 - 11:00 AM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
  { hour: '11:00 - 12:00 PM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
  { hour: '12:00 - 01:00 PM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
  { hour: '01:00 - 02:00 PM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
  { hour: '02:00 - 03:00 PM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
  { hour: '03:00 - 04:00 PM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
  { hour: '04:00 - 05:00 PM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
  { hour: '05:00 - 06:00 PM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
  { hour: '06:00 - 07:00 PM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
  { hour: '07:00 - 08:00 PM', visitors: 0, newVisitors: 0, returningVisitors: 0, testRides: 0, deliveries: 0 },
];

export const YESTERDAY_METRICS = {
  totalVisitors: 0,
  newVisitors: 0,
  returningVisitors: 0,
  avgDuration: 0,
  peakHour: 'N/A',
  peakCount: 0,
};

export const PURPOSE_COLORS: Record<VisitPurpose, string> = {
  'Test Ride': '#ef4444', // Red (Capermint brand accent)
  'Vehicle Enquiry': '#2563eb', // Blue (Capermint brand primary)
  'Booking': '#10b981', // Emerald green
  'Service': '#8b5cf6', // Violet
  'Vehicle Delivery': '#f59e0b', // Amber (Capermint brand yellow accent)
  'Spare Parts': '#06b6d4', // Cyan
  'General Enquiry': '#64748b', // Slate
  'Other': '#94a3b8', // Gray
};
