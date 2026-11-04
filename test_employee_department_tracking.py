import unittest
import time
from local_detector import EmployeeDepartmentTracker, ZoneManager


class TestEmployeeDepartmentTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = EmployeeDepartmentTracker(log_interval=5.0, exit_debounce=1.5)
        self.zone_manager = ZoneManager(
            config_path="camera_config.json",
            camera_channel="101",
            sections_cfg=[
                {
                    "id": "sales",
                    "name": "Sales Department",
                    "color": [0, 215, 255],
                    "box": [100, 100, 400, 400],
                },
                {
                    "id": "service",
                    "name": "Service Lounge",
                    "color": [246, 130, 59],
                    "box": [500, 100, 800, 400],
                },
            ],
            frame_width=1280,
            frame_height=720,
        )

    def test_find_section_for_box(self):
        # Box completely inside sales: feet at (250, 350)
        box_sales = (200.0, 150.0, 300.0, 350.0)
        dept = self.zone_manager.find_section_for_box(box_sales)
        self.assertIsNotNone(dept)
        self.assertEqual(dept[0], "sales")
        self.assertEqual(dept[1], "Sales Department")

        # Box outside all sections
        box_outside = (10.0, 10.0, 50.0, 80.0)
        dept_out = self.zone_manager.find_section_for_box(box_outside)
        self.assertIsNone(dept_out)

        # Box in service
        box_service = (550.0, 150.0, 650.0, 380.0)
        dept_serv = self.zone_manager.find_section_for_box(box_service)
        self.assertIsNotNone(dept_serv)
        self.assertEqual(dept_serv[0], "service")

    def test_employee_entry_dwell_exit_lifecycle(self):
        emp_id = "EMP-001"
        emp_name = "Rajesh Sharma"
        t0 = 1000.0

        # 1. Entry into Sales at t=1000.0
        self.tracker.update_employee_presence(emp_id, emp_name, ("sales", "Sales Department"), now_ts=t0, track_id="10")
        self.assertIn(emp_id, self.tracker.active_sessions)
        session = self.tracker.active_sessions[emp_id]
        self.assertEqual(session["dept_id"], "sales")
        self.assertEqual(session["entry_time"], t0)

        # Active dwell info check
        dwell_info = self.tracker.get_active_dwell_info(emp_id, now_ts=t0 + 25.0)
        self.assertIsNotNone(dwell_info)
        d_name, d_dig, d_human = dwell_info
        self.assertEqual(d_name, "Sales Department")
        self.assertEqual(d_dig, "00:25")
        self.assertEqual(d_human, "25s")

        # 2. Ongoing dwell updates (t=1010.0: 10 seconds spent)
        self.tracker.update_employee_presence(emp_id, emp_name, ("sales", "Sales Department"), now_ts=t0 + 10.0, track_id="10")
        self.assertEqual(self.tracker.active_sessions[emp_id]["last_seen_time"], t0 + 10.0)

        # 3. Employee steps outside at t=1020.0 (outside for 0.5s < exit_debounce 1.5s -> debounced, still active)
        self.tracker.update_employee_presence(emp_id, emp_name, None, now_ts=t0 + 20.0)
        self.assertIn(emp_id, self.tracker.active_sessions)
        self.assertIn(emp_id, self.tracker.pending_exits)

        # 4. Employee outside for 2.0s (> exit_debounce 1.5s) -> Exit finalized
        self.tracker.update_employee_presence(emp_id, emp_name, None, now_ts=t0 + 22.0)
        self.assertNotIn(emp_id, self.tracker.active_sessions)
        self.assertNotIn(emp_id, self.tracker.pending_exits)

        # Cumulative time should be 10.0s (t0+10.0 - t0)
        self.assertIn(emp_id, self.tracker.cumulative_dwell)
        self.assertAlmostEqual(self.tracker.cumulative_dwell[emp_id]["sales"], 10.0)
        self.assertEqual(self.tracker.visit_counts[emp_id]["sales"], 1)

    def test_candidate_track_pre_recognition_credit(self):
        track_id = "42"
        emp_id = "EMP-007"
        emp_name = "Anita Patel"
        t0 = 2000.0

        # Unconfirmed candidate track enters Sales at t=2000.0
        self.tracker.update_unconfirmed_track(track_id, ("sales", "Sales Department"), now_ts=t0)
        self.assertIn(track_id, self.tracker.unconfirmed_tracks)

        # 4 seconds later (t=2004.0), face recognition verifies track as Anita Patel
        self.tracker.promote_track_to_employee(track_id, emp_id, emp_name, now_ts=t0 + 4.0)

        # Session should be active with original entry time t0 = 2000.0!
        self.assertIn(emp_id, self.tracker.active_sessions)
        self.assertEqual(self.tracker.active_sessions[emp_id]["entry_time"], t0)

        # At t=2004.0, dwell time is 4.0s
        dwell = self.tracker.get_active_dwell_info(emp_id, now_ts=t0 + 4.0)
        self.assertEqual(dwell[1], "00:04")

    def test_department_transition(self):
        emp_id = "EMP-003"
        emp_name = "Vikram"
        t0 = 3000.0

        # Enters Sales
        self.tracker.update_employee_presence(emp_id, emp_name, ("sales", "Sales Department"), now_ts=t0)
        self.tracker.update_employee_presence(emp_id, emp_name, ("sales", "Sales Department"), now_ts=t0 + 30.0)

        # Direct transition into Service Lounge at t=3031.0
        self.tracker.update_employee_presence(emp_id, emp_name, ("service", "Service Lounge"), now_ts=t0 + 31.0)

        # Sales session finalized with 30s dwell
        self.assertAlmostEqual(self.tracker.cumulative_dwell[emp_id]["sales"], 30.0)
        # Service session active
        self.assertEqual(self.tracker.active_sessions[emp_id]["dept_id"], "service")

    def test_format_duration(self):
        self.assertEqual(EmployeeDepartmentTracker.format_duration(45), "45s")
        self.assertEqual(EmployeeDepartmentTracker.format_duration(125), "2m 05s")
        self.assertEqual(EmployeeDepartmentTracker.format_duration(3665), "1h 01m 05s")

    def test_format_digital(self):
        self.assertEqual(EmployeeDepartmentTracker.format_digital(45), "00:45")
        self.assertEqual(EmployeeDepartmentTracker.format_digital(125), "02:05")
        self.assertEqual(EmployeeDepartmentTracker.format_digital(3665), "01:01:05")


if __name__ == "__main__":
    unittest.main()

