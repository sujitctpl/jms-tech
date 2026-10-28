from app.vision.entry_counter import EntryLineCounter
from app.vision.tracker import TrackedPerson


def test_bottom_center_reference_point():
    # Box: x1=100, y1=50, x2=200, y2=250
    # Bottom-center should be ((100+200)/2, 250) = (150, 250)
    person = TrackedPerson(
        tracker_id="1",
        box=(100.0, 50.0, 200.0, 250.0),
        confidence=0.85,
        class_id=0,
    )
    assert person.bottom_center == (150.0, 250.0)


def test_entry_line_top_to_bottom_crossing():
    config = {
        "line": [[100.0, 300.0], [500.0, 300.0]],
        "direction": "top_to_bottom",
    }
    counter = EntryLineCounter(config)

    # Frame 1: Person 1 at y = 250 (above line)
    p1_f1 = TrackedPerson(tracker_id="1", box=(200.0, 150.0, 250.0, 250.0), confidence=0.9, class_id=0)
    entries_f1 = counter.process_frame([p1_f1])
    # Person appearing does not trigger an entry count
    assert len(entries_f1) == 0

    # Frame 2: Person 1 moves to y = 350 (below line)
    p1_f2 = TrackedPerson(tracker_id="1", box=(200.0, 250.0, 250.0, 350.0), confidence=0.9, class_id=0)
    entries_f2 = counter.process_frame([p1_f2])
    # Now crossed top_to_bottom!
    assert len(entries_f2) == 1
    assert entries_f2[0].tracker_id == "1"


def test_entry_line_wrong_direction_rejected():
    config = {
        "line": [[100.0, 300.0], [500.0, 300.0]],
        "direction": "top_to_bottom",
    }
    counter = EntryLineCounter(config)

    # Person starts at y = 350 (below line)
    p1_f1 = TrackedPerson(tracker_id="1", box=(200.0, 250.0, 250.0, 350.0), confidence=0.9, class_id=0)
    counter.process_frame([p1_f1])

    # Person moves upwards to y = 250 (bottom to top crossing)
    p1_f2 = TrackedPerson(tracker_id="1", box=(200.0, 150.0, 250.0, 250.0), confidence=0.9, class_id=0)
    entries = counter.process_frame([p1_f2])

    # Should NOT count because direction is top_to_bottom only
    assert len(entries) == 0


def test_duplicate_entry_prevention():
    config = {
        "line": [[100.0, 300.0], [500.0, 300.0]],
        "direction": "top_to_bottom",
    }
    counter = EntryLineCounter(config)

    # Frame 1: above line
    p1 = TrackedPerson(tracker_id="track_99", box=(200.0, 150.0, 250.0, 250.0), confidence=0.9, class_id=0)
    counter.process_frame([p1])

    # Frame 2: crosses line
    p2 = TrackedPerson(tracker_id="track_99", box=(200.0, 250.0, 250.0, 350.0), confidence=0.9, class_id=0)
    entries = counter.process_frame([p2])
    assert len(entries) == 1

    # Frame 3: same person moves back above line
    p3 = TrackedPerson(tracker_id="track_99", box=(200.0, 150.0, 250.0, 250.0), confidence=0.9, class_id=0)
    entries_back = counter.process_frame([p3])
    assert len(entries_back) == 0

    # Frame 4: same person crosses line AGAIN
    p4 = TrackedPerson(tracker_id="track_99", box=(200.0, 250.0, 250.0, 350.0), confidence=0.9, class_id=0)
    entries_again = counter.process_frame([p4])
    # MUST NOT be counted again!
    assert len(entries_again) == 0


def test_non_crossing_movements():
    config = {
        "line": [[100.0, 300.0], [500.0, 300.0]],
        "direction": "top_to_bottom",
    }
    counter = EntryLineCounter(config)

    # Movement entirely above the line (y: 200 -> 250)
    p1 = TrackedPerson(tracker_id="1", box=(100.0, 100.0, 150.0, 200.0), confidence=0.9, class_id=0)
    p2 = TrackedPerson(tracker_id="1", box=(100.0, 150.0, 150.0, 250.0), confidence=0.9, class_id=0)
    counter.process_frame([p1])
    assert len(counter.process_frame([p2])) == 0

    # Movement outside line x range (x: 600 -> 650, y: 250 -> 350)
    p3 = TrackedPerson(tracker_id="2", box=(600.0, 200.0, 650.0, 250.0), confidence=0.9, class_id=0)
    p4 = TrackedPerson(tracker_id="2", box=(600.0, 300.0, 650.0, 350.0), confidence=0.9, class_id=0)
    counter.process_frame([p3])
    assert len(counter.process_frame([p4])) == 0
