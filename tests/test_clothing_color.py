"""Clothing colour tagging and appearance-matching rules (app/vision/clothing_color.py).

Numeric inputs mirror values measured on real Main Gate snapshots; the pose model is stubbed so these
run without weights.
"""
import numpy as np
import pytest

from app.vision import clothing_color as cc


@pytest.fixture(autouse=True)
def no_pose(monkeypatch):
    monkeypatch.setattr(cc._POSE, "keypoints", lambda img: None)
    cc._CACHE.clear()
    yield
    cc._CACHE.clear()


def _part(tag, rel_l, a, b, patterned=False):
    hist = cc._soft_hist(np.full(50, rel_l), np.full(50, a), np.full(50, b))
    return cc.PartColor(tag=tag, rel_l=rel_l, a=a, b=b, patterned=patterned, hist=hist, pixels=50)


@pytest.mark.parametrize(
    "rel_l,a,b,expected",
    [
        (0.21, 3.3, -4.0, cc.TAG_BLACK),    # black tee: camera adds a faint blue cast to darks
        (0.45, 1.6, 1.1, cc.TAG_GRAY),
        (0.90, 1.0, -0.6, cc.TAG_WHITE),
        (0.85, -0.9, 5.9, cc.TAG_CREAM),
        (0.89, -0.4, 2.4, cc.TAG_CREAM),    # washed-out khaki
        (0.61, -1.8, 3.7, cc.TAG_GREEN),    # sage sweatshirt (VIS-2026-1041), used to read Cream/Gray
        (0.94, 4.1, 26.3, cc.TAG_YELLOW),
        (0.42, 35.1, 14.9, cc.TAG_RED),
        (0.89, 9.1, -3.0, cc.TAG_PINK),
        (0.37, 6.4, 2.5, cc.TAG_BROWN),
        (0.57, 0.6, -9.8, cc.TAG_BLUE),
        (0.40, 1.5, -6.9, cc.TAG_BLUE),     # navy, clearly bluer than the dark-garment cast
        (0.72, 40.0, -70.0, cc.TAG_BLUE),   # saturated blue sits near 300 deg hue in Lab
    ],
)
def test_name_color(rel_l, a, b, expected):
    assert cc._name_color(rel_l, a, b) == expected


def test_lighting_drift_is_compatible_but_real_contradictions_are_not():
    sage_lit = _part(cc.TAG_GREEN, 0.60, -1.8, 3.7)
    sage_shadow = _part(cc.TAG_GRAY, 0.50, -0.1, 0.6)
    assert cc.tags_compatible(sage_lit, sage_shadow)  # the missed VIS-2026-1041 exit
    assert not cc.tags_compatible(_part(cc.TAG_WHITE, 0.9, 0, 0), _part(cc.TAG_BLACK, 0.2, 0, 0))
    assert not cc.tags_compatible(_part(cc.TAG_RED, 0.4, 35, 15), _part(cc.TAG_BLUE, 0.5, 0, -12))
    assert cc.tags_compatible(_part(cc.TAG_BLUE, 0.3, 1.5, -7), _part(cc.TAG_BLACK, 0.2, 2, -4))


def test_patterned_front_is_compatible_with_plain_back():
    striped = _part(cc.TAG_PATTERN, 0.68, 2.3, -2.9, patterned=True)
    assert cc.tags_compatible(striped, _part(cc.TAG_BLACK, 0.24, 2.5, -5.2))
    assert not cc.tags_compatible(striped, _part(cc.TAG_YELLOW, 0.9, 4, 26))


def test_vivid_colours_are_not_dropped_from_histogram():
    red = _part(cc.TAG_RED, 0.4, 60.0, 40.0)  # far outside the +-24 bin range
    assert red.hist.sum() == pytest.approx(1.0, abs=1e-5)
    assert cc.hist_similarity(red, red) == pytest.approx(1.0, abs=1e-4)


def _person(shirt_bgr, trousers_bgr=(30, 30, 30), wall=(200, 200, 200)):
    img = np.full((320, 140, 3), wall, np.uint8)
    img[60:160, 35:105] = shirt_bgr
    img[170:300, 40:100] = trousers_bgr
    return img


def test_fallback_geometry_reads_shirt_relative_to_wall():
    assert cc.estimate_tag(_person((235, 235, 235))) == cc.TAG_WHITE
    assert cc.estimate_tag(_person((25, 25, 25))) == cc.TAG_BLACK
    desc = cc.describe(_person((235, 235, 235), trousers_bgr=(30, 30, 30)))
    assert desc.source == "fallback" and desc.legs is not None and desc.legs.rel_l < 0.35


@pytest.mark.parametrize("bad", [None, np.zeros((0, 0, 3), np.uint8), np.zeros((10, 5, 3), np.uint8)])
def test_describe_never_raises(bad):
    assert cc.describe(bad).tag == cc.TAG_UNKNOWN


def test_describe_is_cached_per_crop():
    img = _person((235, 235, 235))
    assert cc.describe(img) is cc.describe(img.copy())


def test_verify_appearance_vetoes_dark_vs_light_trousers(monkeypatch):
    reid, _, score, is_match = _verify(monkeypatch, reid=0.75, legs_reliable=(True, True),
                                       shirt2=_part(cc.TAG_GRAY, 0.55, 0.5, -0.5))
    assert reid > 0.7 and not is_match and score < 0.48


def _verify(monkeypatch, reid, legs_reliable, shirt2=None):
    from app.vision.body_embedder import BodyEmbedder

    shirt = _part(cc.TAG_WHITE, 0.9, 0.5, -0.5)
    descs = iter([
        cc.ClothingDescriptor(shirt, _part(cc.TAG_BLACK, 0.22, 0, 0), "pose", legs_reliable[0]),
        cc.ClothingDescriptor(shirt2 or shirt, _part(cc.TAG_WHITE, 0.80, 0, 0), "pose", legs_reliable[1]),
    ])
    monkeypatch.setattr(cc, "describe", lambda crop: next(descs))
    be = BodyEmbedder.__new__(BodyEmbedder)  # no ONNX session needed with explicit embeddings
    be._cached_face_embedder = type("NoFace", (), {"detect_and_crop_face": lambda self, c: (None, None, None)})()
    e1 = np.zeros(512, np.float32); e1[0] = 1.0
    e2 = np.zeros(512, np.float32); e2[0], e2[1] = reid, np.sqrt(1 - reid ** 2)
    crop = np.full((300, 120, 3), 120, np.uint8)
    return be.verify_appearance(crop, crop.copy(), e1, e2)


def test_trouser_veto_ignores_unreliable_legs(monkeypatch):
    # VIS-2026-1041 entry: walking side-on, pose put the knees between his legs and sampled the floor.
    _, _, score, is_match = _verify(monkeypatch, reid=0.75, legs_reliable=(False, True))
    assert is_match and score >= 0.48


def test_trouser_veto_yields_to_strong_body_and_shirt_match(monkeypatch):
    _, _, score, is_match = _verify(monkeypatch, reid=0.81, legs_reliable=(True, True))
    assert is_match and score >= 0.48


def _desc(shirt, sleeves=None, legs=None, legs_reliable=False):
    return cc.ClothingDescriptor(shirt, legs, "pose", legs_reliable, sleeves)


def test_weak_reid_band_accepts_matching_outfit():
    # VIS-2026-1044: black top, reid 0.650 front-in vs side-out; clothing near-identical, sleeves agree.
    top = _part(cc.TAG_GRAY, 0.45, 1.0, -2.0)
    ok, _ = cc.weak_reid_support(_desc(top, "long"), _desc(top, "long"))
    assert ok
    ok, _ = cc.weak_reid_support(_desc(top, None), _desc(top, "short"))  # one side undetermined
    assert ok


def test_weak_reid_band_rejects_sleeve_contradiction():
    # Wrong VIS-2026-1047 departure: black short-sleeve tee vs dark long-sleeve shirt, reid 0.685.
    top = _part(cc.TAG_BLACK, 0.25, 2.0, -4.0)
    ok, why = cc.weak_reid_support(_desc(top, "short"), _desc(top, "long"))
    assert not ok and "sleeves" in why


def test_weak_reid_band_rejects_weak_clothing_and_reliable_trouser_conflict():
    ok, _ = cc.weak_reid_support(_desc(_part(cc.TAG_GRAY, 0.45, 0, 0)), _desc(_part(cc.TAG_GRAY, 0.95, 0, 0)))
    assert not ok  # same tag but histograms far apart
    top = _part(cc.TAG_WHITE, 0.9, 0, 0)
    dark, light = _part(cc.TAG_BLACK, 0.2, 0, 0), _part(cc.TAG_WHITE, 0.8, 0, 0)
    ok, why = cc.weak_reid_support(_desc(top, legs=dark, legs_reliable=True), _desc(top, legs=light, legs_reliable=True))
    assert not ok and "trousers" in why


def test_sleeve_length_from_forearm_colour():
    shape = (40, 40)
    torso = _part(cc.TAG_BLACK, 0.25, 2.0, -4.0)
    skin_a, skin_b = np.full(shape, 8.0), np.full(shape, 9.0)
    cloth_a, cloth_b = np.full(shape, 1.0), np.full(shape, -4.0)
    arm = np.zeros(shape, bool); arm[10:20, 10:20] = True
    assert cc._sleeve_length(torso, [arm], skin_a, skin_b, (0.0, 0.0)) == "short"
    assert cc._sleeve_length(torso, [arm], cloth_a, cloth_b, (0.0, 0.0)) == "long"
    assert cc._sleeve_length(_part(cc.TAG_PINK, 0.8, 8, 6), [arm], skin_a, skin_b, (0.0, 0.0)) is None  # skin-like top
    assert cc._sleeve_length(torso, [], skin_a, skin_b, (0.0, 0.0)) is None
