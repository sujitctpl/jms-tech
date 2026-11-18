import { FASTAPI_BASE_URL, IS_LOCAL, APP_ENV } from '../config/env';

export interface EntryLineConfig {
  line: [[number, number], [number, number]];
  direction: 'top_to_bottom' | 'bottom_to_top' | 'left_to_right' | 'right_to_left';
}

export interface CameraModel {
  id: string;
  name: string;
  stream_url: string;
  location: string;
  entry_line_config: EntryLineConfig;
  is_active: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface CameraStatusModel {
  camera_id: string;
  is_processing: boolean;
  active_run_id: string | null;
  frames_processed: number;
  last_error: string | null;
  section_counts?: Record<string, number>;
  section_occupancy?: Record<string, number>;
}

export interface FootfallSummary {
  total_entries: number;
  camera_id: string | null;
  start_time: string | null;
  end_time: string | null;
  section_counts?: Record<string, number>;
  section_occupancy?: Record<string, number>;
  staff_dwell?: Record<string, {
    name: string;
    department_id: string;
    department_name: string;
    dwell_seconds: number;
    dwell_formatted: string;
  }>;
}

export interface DepartmentModel {
  id: string;
  name: string;
  color: string;
  camera_id?: string | null;
  box?: number[] | null;
  is_active: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface DepartmentCreatePayload {
  id?: string;
  name: string;
  color?: string;
  camera_id?: string | null;
  box?: number[] | null;
  is_active?: boolean;
}

export interface CameraChannelConfig {
  name: string;
  camera_id?: string;
  source: string;
  location?: string;
  confidence?: number;
  min_feet_y?: number;
  allow_standalone_faces?: boolean;
  is_active?: boolean;
  entry_line?: {
    enabled: boolean;
    p1: [number, number];
    p2: [number, number];
    direction: string;
  } | null;
  sections?: Array<{
    id: string;
    name: string;
    color: [number, number, number];
    box: [number, number, number, number];
  }>;
  exclusion_zones?: Array<{
    name: string;
    box: [number, number, number, number];
    enabled?: boolean;
  }>;
}

export interface DynamicCameraConfig {
  default_channel: string;
  api_url?: string;
  model_type: string;
  rf_detr_variant?: string;
  model?: string;
  confidence: number;
  imgsz: number;
  face_model?: string;
  face_confidence: number;
  face_similarity_threshold: number;
  face_scale: number;
  box_smoothing: boolean;
  smoothing_alpha: number;
  deadband_px?: number;
  body_persistence?: number;
  face_persistence?: number;
  show_entry_line: boolean;
  allow_standalone_faces: boolean;
  show_head_fallback: boolean;
  min_feet_y?: number;
  line_y?: number;
  direction?: string;
  cameras: Record<string, CameraChannelConfig>;
  archived_camera_profiles?: Record<string, CameraChannelConfig>;
}

export interface StreamTestResult {
  online: boolean;
  resolution?: [number, number];
  fps?: number;
  message?: string;
  error?: string;
}

export const DEFAULT_CAMERA_CONFIG: DynamicCameraConfig = {
  default_channel: '201',
  api_url: 'http://127.0.0.1:8000',
  model_type: 'yolo',
  model: 'models/yolo26s.pt',
  confidence: 0.31,
  imgsz: 1024,
  face_confidence: 0.36,
  face_similarity_threshold: 0.32,
  face_scale: 1.05,
  box_smoothing: true,
  smoothing_alpha: 0.35,
  deadband_px: 6.0,
  body_persistence: 26,
  face_persistence: 8,
  show_entry_line: true,
  allow_standalone_faces: false,
  show_head_fallback: false,
  cameras: {
    '101': {
      name: 'Showroom Camera 101 (Main Glass Entrance)',
      camera_id: '88970b31-d885-4fd5-a24a-bdc969c96240',
      source: 'rtsp://zakir:06nov2014@192.168.20.134:554/Streaming/Channels/101',
      location: 'Channel 101 (Main Glass Entrance)',
      confidence: 0.31,
      min_feet_y: 0.32,
      allow_standalone_faces: false,
      is_active: true,
      entry_line: {
        enabled: true,
        p1: [547, 537],
        p2: [975, 345],
        direction: 'both',
      },
      exclusion_zones: [
        {
          name: 'Ceiling Lamp',
          box: [40, 40, 200, 220],
        },
        {
          name: 'Wall Decor',
          box: [50, 430, 120, 520],
        },
        {
          name: 'Bottom Floor Edge',
          box: [550, 620, 800, 720],
        },
      ],
    },
    '201': {
      name: 'Showroom Camera 201 (Floor & Display Area)',
      camera_id: '0e7e29df-7375-47ce-8c3f-bf03c14d6e67',
      source: 'rtsp://zakir:06nov2014@192.168.20.134:554/Streaming/Channels/201',
      location: 'Channel 201 (Showroom Floor & Display Area)',
      confidence: 0.31,
      min_feet_y: 0.08,
      allow_standalone_faces: false,
      is_active: true,
      entry_line: {
        enabled: true,
        p1: [200, 500],
        p2: [1000, 500],
        direction: 'top_to_bottom',
      },
    },
    '301': {
      name: 'Showroom Camera 301 (Lounge & Billing Desk)',
      camera_id: '8da1af51-b4fb-4745-bc47-a5a5c0e4571e',
      source: 'rtsp://zakir:06nov2014@192.168.20.134:554/Streaming/Channels/301',
      location: 'Channel 301 (Customer Lounge & Billing Desk)',
      confidence: 0.31,
      min_feet_y: 0.08,
      allow_standalone_faces: false,
      is_active: true,
      entry_line: {
        enabled: true,
        p1: [200, 500],
        p2: [1000, 500],
        direction: 'top_to_bottom',
      },
    },
    '401': {
      name: 'Showroom Camera 401 (Delivery Bay & Service Gate)',
      camera_id: 'f959f6fc-eb56-4515-85c6-510f5c7fb27d',
      source: 'rtsp://zakir:06nov2014@192.168.20.134:554/Streaming/Channels/401',
      location: 'Channel 401 (Delivery Bay & Service Gate)',
      confidence: 0.31,
      min_feet_y: 0.08,
      allow_standalone_faces: false,
      is_active: true,
      entry_line: {
        enabled: true,
        p1: [200, 500],
        p2: [1000, 500],
        direction: 'top_to_bottom',
      },
    },
  },
};

// Built-in configured cameras matching backend database
export const FALLBACK_CAMERAS: CameraModel[] = [
  {
    id: '88970b31-d885-4fd5-a24a-bdc969c96240',
    name: 'Showroom Camera 101',
    location: 'Channel 101 (Main Glass Entrance)',
    stream_url: 'rtsp://username:password@192.168.1.101:554/Streaming/Channels/101',
    entry_line_config: {
      line: [[200.0, 500.0], [1000.0, 500.0]],
      direction: 'top_to_bottom',
    },
    is_active: true,
  },
  {
    id: '0e7e29df-7375-47ce-8c3f-bf03c14d6e67',
    name: 'Showroom Camera 201',
    location: 'Channel 201 (Showroom Floor & Display Area)',
    stream_url: 'rtsp://username:password@192.168.1.101:554/Streaming/Channels/201',
    entry_line_config: {
      line: [[200.0, 500.0], [1000.0, 500.0]],
      direction: 'top_to_bottom',
    },
    is_active: true,
  },
  {
    id: '8da1af51-b4fb-4745-bc47-a5a5c0e4571e',
    name: 'Showroom Camera 301',
    location: 'Channel 301 (Customer Lounge & Billing Desk)',
    stream_url: 'rtsp://username:password@192.168.1.101:554/Streaming/Channels/301',
    entry_line_config: {
      line: [[200.0, 500.0], [1000.0, 500.0]],
      direction: 'top_to_bottom',
    },
    is_active: true,
  },
  {
    id: 'f959f6fc-eb56-4515-85c6-510f5c7fb27d',
    name: 'Showroom Camera 401',
    location: 'Channel 401 (Delivery Bay & Service Gate)',
    stream_url: 'rtsp://username:password@192.168.1.101:554/Streaming/Channels/401',
    entry_line_config: {
      line: [[200.0, 500.0], [1000.0, 500.0]],
      direction: 'top_to_bottom',
    },
    is_active: true,
  },
];

class CameraService {
  private getBaseUrl(): string {
    // In local development, relative URLs like /api/v1 are proxied by Vite to http://127.0.0.1:8000
    // Direct FASTAPI_BASE_URL is also supported since CORS is enabled
    return FASTAPI_BASE_URL || '';
  }

  /**
   * Check whether FastAPI backend is responsive
   */
  async checkHealth(): Promise<{ online: boolean; message: string }> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/health`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        return { online: true, message: data.service || 'Backend Online' };
      }
      return { online: false, message: `Status ${res.status}` };
    } catch (e) {
      return { online: false, message: 'Backend unreachable' };
    }
  }

  /**
   * List all configured cameras
   */
  async listCameras(): Promise<CameraModel[]> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/cameras`, { signal: AbortSignal.timeout(4000) });
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          return data;
        }
      }
      console.info('[CameraService] Using configured fallback cameras');
      return FALLBACK_CAMERAS;
    } catch (err) {
      console.warn('[CameraService] Backend camera endpoint unreachable, using fallback cameras', err);
      return FALLBACK_CAMERAS;
    }
  }

  /**
   * Get specific camera details
   */
  async getCamera(cameraId: string): Promise<CameraModel | null> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/cameras/${cameraId}`, { signal: AbortSignal.timeout(4000) });
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.warn('[CameraService] Failed to get camera', e);
    }
    return FALLBACK_CAMERAS.find((c) => c.id === cameraId) || null;
  }

  /**
   * Get live processing status of a camera worker
   */
  async getCameraStatus(cameraId: string): Promise<CameraStatusModel> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/cameras/${cameraId}/status`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      // Ignore network errors in status poll
    }
    return {
      camera_id: cameraId,
      is_processing: false,
      active_run_id: null,
      frames_processed: 0,
      last_error: null,
    };
  }

  /**
   * Start AI tracking worker on the backend
   */
  async startWorker(cameraId: string): Promise<{ success: boolean; message: string }> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/cameras/${cameraId}/start`, {
        method: 'POST',
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        return { success: true, message: data.message || 'Worker started successfully' };
      }
      return { success: false, message: data.detail || 'Failed to start worker' };
    } catch (e: any) {
      return { success: false, message: e.message || 'Backend connection error' };
    }
  }

  /**
   * Stop AI tracking worker on the backend
   */
  async stopWorker(cameraId: string): Promise<{ success: boolean; message: string }> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/cameras/${cameraId}/stop`, {
        method: 'POST',
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        return { success: true, message: data.message || 'Worker stopped successfully' };
      }
      return { success: false, message: data.detail || 'Failed to stop worker' };
    } catch (e: any) {
      return { success: false, message: e.message || 'Backend connection error' };
    }
  }

  /**
   * Update camera stream URL, location, or virtual entry line
   */
  async updateCamera(cameraId: string, payload: Partial<CameraModel>): Promise<CameraModel | null> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/cameras/${cameraId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.error('[CameraService] Failed to update camera', e);
    }
    return null;
  }

  /**
   * Fetch footfall analytics from backend
   */
  async getFootfall(cameraId?: string): Promise<FootfallSummary | null> {
    try {
      const url = cameraId
        ? `${this.getBaseUrl()}/api/v1/analytics/footfall?camera_id=${cameraId}`
        : `${this.getBaseUrl()}/api/v1/analytics/footfall`;
      const res = await fetch(url, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      // Graceful fallback
    }
    return null;
  }

  /**
   * List all configured detection departments
   */
  async listDepartments(cameraId?: string): Promise<DepartmentModel[]> {
    try {
      const url = cameraId
        ? `${this.getBaseUrl()}/api/v1/departments?camera_id=${cameraId}`
        : `${this.getBaseUrl()}/api/v1/departments`;
      const res = await fetch(url, { signal: AbortSignal.timeout(4000) });
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.warn('[CameraService] Failed to fetch departments from backend', e);
    }
    return [
      { id: 'ai_department', name: 'AI Department', color: '#3B82F6', is_active: true },
      { id: 'bde', name: 'BDE Department', color: '#10B981', is_active: true },
      { id: 'sales', name: 'Sales Section', color: '#00D7FF', is_active: true },
      { id: 'accounts', name: 'Accounts Section', color: '#FF8C00', is_active: true },
      { id: 'service', name: 'Service Section', color: '#32E178', is_active: true },
    ];
  }

  /**
   * Create a new detection department
   */
  async createDepartment(payload: DepartmentCreatePayload): Promise<DepartmentModel | null> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/departments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.error('[CameraService] Failed to create department', e);
    }
    return null;
  }

  /**
   * Update department details
   */
  async updateDepartment(departmentId: string, payload: Partial<DepartmentCreatePayload>): Promise<DepartmentModel | null> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/departments/${departmentId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.error('[CameraService] Failed to update department', e);
    }
    return null;
  }

  /**
   * Delete a detection department
   */
  async deleteDepartment(departmentId: string): Promise<boolean> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/departments/${departmentId}`, {
        method: 'DELETE',
      });
      return res.status === 204 || res.ok;
    } catch (e) {
      console.error('[CameraService] Failed to delete department', e);
      return false;
    }
  }

  /**
   * Get dynamic camera & vision AI configuration from camera_config.json
   */
  async getCameraConfig(): Promise<DynamicCameraConfig> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/cameras/config`, { signal: AbortSignal.timeout(4000) });
      if (res.ok) {
        const data = await res.json();
        if (data && typeof data === 'object' && data.cameras) {
          try {
            localStorage.setItem('jms_dynamic_camera_config', JSON.stringify(data));
          } catch (_) {}
          return data as DynamicCameraConfig;
        }
      }
    } catch (err) {
      console.warn('[CameraService] Failed to fetch dynamic camera config from backend, using cached/fallback', err);
    }

    try {
      const cached = localStorage.getItem('jms_dynamic_camera_config');
      if (cached) {
        return JSON.parse(cached);
      }
    } catch (_) {}

    return DEFAULT_CAMERA_CONFIG;
  }

  /**
   * Save dynamic camera & vision AI configuration to camera_config.json and sync with DB
   */
  async saveCameraConfig(
    config: Partial<DynamicCameraConfig>
  ): Promise<{ success: boolean; message: string; config?: DynamicCameraConfig }> {
    try {
      localStorage.setItem('jms_dynamic_camera_config', JSON.stringify(config));
    } catch (_) {}

    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/cameras/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        return {
          success: true,
          message: data.message || 'Camera configuration updated and synced with Edge AI.',
          config: data.config,
        };
      }
      return {
        success: false,
        message: data.detail || 'Backend failed to save camera configuration.',
      };
    } catch (e: any) {
      console.warn('[CameraService] Backend unreachable, configuration cached locally for synchronization.', e);
      return {
        success: true,
        message: 'Settings saved locally and will auto-sync when Edge detector reconnects.',
      };
    }
  }

  /**
   * Test RTSP / Video Stream connectivity and retrieve resolution and FPS
   */
  async testStream(streamUrl: string): Promise<StreamTestResult> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/cameras/test-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stream_url: streamUrl }),
        signal: AbortSignal.timeout(6000),
      });
      if (res.ok) {
        return await res.json();
      }
      const data = await res.json().catch(() => ({}));
      return {
        online: false,
        error: data.detail || 'Stream test request rejected by server.',
      };
    } catch (e: any) {
      return {
        online: false,
        error: 'Unable to reach backend stream tester. Please ensure FastAPI backend is running.',
      };
    }
  }

  /**
   * Delete a camera channel from camera_config.json and DB
   */
  async deleteCameraChannel(channelOrId: string): Promise<boolean> {
    try {
      const cached = localStorage.getItem('jms_dynamic_camera_config');
      if (cached) {
        const parsed = JSON.parse(cached);
        if (parsed.cameras && parsed.cameras[channelOrId]) {
          delete parsed.cameras[channelOrId];
          localStorage.setItem('jms_dynamic_camera_config', JSON.stringify(parsed));
        }
      }
    } catch (_) {}

    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/cameras/channel/${channelOrId}`, {
        method: 'DELETE',
      });
      return res.ok;
    } catch (e) {
      console.error('[CameraService] Failed to delete camera channel', e);
      return false;
    }
  }

  /**
   * List detected visitors from backend with live photos and statuses
   */
  async listVisitors(taggingStatus?: string, visitPurpose?: string): Promise<{ visitors: any[]; total: number; tagged_count: number; untagged_count: number } | null> {
    try {
      let url = `${this.getBaseUrl()}/api/v1/visitors`;
      const params = new URLSearchParams();
      if (taggingStatus && taggingStatus !== 'all') params.append('tagging_status', taggingStatus);
      if (visitPurpose && visitPurpose !== 'all') params.append('visit_purpose', visitPurpose);
      if (params.toString()) url += `?${params.toString()}`;

      const res = await fetch(url);
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      return null;
    }
  }

  /**
   * Tag a visitor with visit purpose, salesperson, and notes
   */
  async tagVisitor(personId: string, visitPurpose: string, salesperson?: string, notes?: string): Promise<any | null> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/visitors/${encodeURIComponent(personId)}/tag`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          visit_purpose: visitPurpose,
          assigned_salesperson: salesperson || null,
          notes: notes || null,
        }),
      });
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      console.warn('[CameraService] Failed to tag visitor:', e);
      return null;
    }
  }

  /**
   * Log manual walk-in visitor from reception
   */
  async createWalkIn(payload: { visit_purpose?: string; assigned_salesperson?: string; notes?: string }): Promise<any | null> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/visitors/walk-in`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      console.warn('[CameraService] Failed to create walk-in:', e);
      return null;
    }
  }

  /**
   * Clear all visitors and snapshots from backend database
   */
  async clearAllVisitors(): Promise<boolean> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/visitors/clear`, {
        method: 'POST',
      });
      return res.ok;
    } catch (e) {
      console.warn('[CameraService] Failed to clear all visitors:', e);
      return false;
    }
  }

  /**
   * Update live visitor zone dwell breakdown and departure
   */
  async updateVisitorZone(payload: {
    person_id?: string;
    tracker_id?: string;
    current_zone?: string;
    zone_dwell?: Record<string, string>;
    exit_time?: string;
  }): Promise<any | null> {
    try {
      const res = await fetch(`${this.getBaseUrl()}/api/v1/visitors/update-zone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      console.warn('[CameraService] Failed to update visitor zone dwell:', e);
      return null;
    }
  }

  /**
   * Mark visitor departure with checkout time
   */
  async checkoutVisitor(personId: string, exitTime: string): Promise<any | null> {
    return this.updateVisitorZone({ person_id: personId, exit_time: exitTime });
  }
}

export const cameraService = new CameraService();
