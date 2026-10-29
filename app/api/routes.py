HEALTH = "/health"

REGISTER_CAMERAS = "/api/v1/cameras"
LIST_CAMERAS = "/api/v1/cameras"
GET_CAMERA = "/api/v1/cameras/{camera_id}"
UPDATE_CAMERA = "/api/v1/cameras/{camera_id}"

START_CAMERA = "/api/v1/cameras/{camera_id}/start"
STOP_CAMERA = "/api/v1/cameras/{camera_id}/stop"
CAMERA_STATUS = "/api/v1/cameras/{camera_id}/status"
CAMERA_FRAME = "/api/v1/cameras/{camera_id}/frame"
CAMERA_STREAM = "/api/v1/cameras/{camera_id}/stream"
RECORD_CAMERA_FOOTFALL = "/api/v1/cameras/{camera_id}/footfall"
CAMERA_HEARTBEAT = "/api/v1/cameras/{camera_id}/heartbeat"

FOOTFALL_ANALYTICS = "/api/v1/analytics/footfall"
DAILY_FOOTFALL_ANALYTICS = "/api/v1/analytics/footfall/daily"

RETENTION_CLEANUP = "/api/v1/identity/cleanup"
RETENTION_STATUS = "/api/v1/identity/status"

EMPLOYEES = "/api/v1/employees"
EMPLOYEE_DETAIL = "/api/v1/employees/{employee_id}"
EMPLOYEE_FACE = "/api/v1/employees/{employee_id}/face"
EMPLOYEE_ATTENDANCE = "/api/v1/employees/{employee_id}/attendance"
ATTENDANCE_PUNCH = "/api/v1/attendance/punch"
ATTENDANCE_DAILY = "/api/v1/attendance/daily"