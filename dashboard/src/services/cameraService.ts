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
}

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
}

export const cameraService = new CameraService();
