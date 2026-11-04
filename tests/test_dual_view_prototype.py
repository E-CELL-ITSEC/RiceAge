"""
RiceAge Phase 4 Dual-View Integration Prototype Regression Tests.
Verifies all 10 core regression invariants, plus reproduction of the Phase 3.2 b_859 bug:
 1. A grain observed by both scanners produces exactly one physical grain record.
 2. Total physical grain count does not double after adding second view.
 3. A bottom detection cannot be assigned to two top detections.
 4. A top detection cannot be assigned to two bottom detections.
 5. Missing candidates do not trigger forced assignments.
 6. Ambiguous matches do not enter automatic fusion.
 7. Registration failure triggers safe fallback.
 8. Existing single-view behavior remains operational.
 9. Sample-level aggregation counts each physical grain once.
10. Unmatched observations remain traceable.
11. Regression test: Phase 3.2 dropped detection (b_859 / t_981) duplicate assignment prevented.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.dual_view.models import (
    ViewObservation,
    PhysicalGrain,
    CorrespondenceStatus,
    DualViewFusionResult
)
from src.dual_view.registration import DualViewRegistration
from src.dual_view.correspondence import DualViewCorrespondenceEngine
from src.dual_view.fusion import DualViewFusionEngine, dataframe_to_view_observations
from src.hardware.dual_platen import DualPlatenManager
from configs.default_config import PipelineConfig


@pytest.fixture
def calibrated_registration():
    """Provides a valid calibrated registration instance."""
    H = np.array([
        [-0.998463, -0.000463, 4917.872],
        [ 0.001606,  0.999657,   21.086],
        [ 0.000000,  0.000000,    1.000]
    ], dtype=np.float64)
    return DualViewRegistration(homography_matrix=H)


@pytest.fixture
def correspondence_engine(calibrated_registration):
    return DualViewCorrespondenceEngine(registration=calibrated_registration, max_matching_distance_px=35.0, ambiguity_margin_px=8.0)


@pytest.fixture
def fusion_engine(correspondence_engine):
    return DualViewFusionEngine(correspondence_engine=correspondence_engine, pixel_scale_mm=0.08467)


def make_obs(scanner_id: str, idx: int, cx: float, cy: float, area: float = 600.0, dclass: str = "Good / Head Rice", ar: float = 2.4):
    prefix = scanner_id[0].upper()
    return ViewObservation(
        observation_id=f"OBS_{prefix}_{idx:05d}",
        scanner_id=scanner_id,
        detection_id=f"{scanner_id[0]}_{idx}",
        centroid_x=cx,
        centroid_y=cy,
        bounding_box=(int(cx - 15), int(cy - 6), 30, 12),
        area_px=area,
        length_mm=round(30 * 0.08467, 2),
        width_mm=round(12 * 0.08467, 2),
        aspect_ratio=ar,
        defect_class=dclass,
        defect_confidence=0.95
    )


# Test 1: Single physical grain produced for dual-view observation
def test_grain_observed_by_both_scanners_produces_single_physical_grain(calibrated_registration, fusion_engine):
    bx, by = 500.0, 500.0
    _, tx_exp, ty_exp = calibrated_registration.project_bottom_to_top(bx, by)

    b_obs = [make_obs("bottom", 1, bx, by, area=620.0, dclass="Good / Head Rice")]
    t_obs = [make_obs("top", 1, tx_exp + 1.2, ty_exp - 0.8, area=610.0, dclass="Good / Head Rice")]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.total_physical_grains == 1
    assert res.matched_count == 1
    g = res.physical_grains[0]
    assert g.correspondence_status == CorrespondenceStatus.MATCHED
    assert g.bottom_observation_id == "OBS_B_00001"
    assert g.top_observation_id == "OBS_T_00001"
    assert g.quality_attributes["is_fused"] is True


# Test 2: Total physical grain count does NOT double after adding second view
def test_total_physical_grain_count_does_not_double(calibrated_registration, fusion_engine):
    N = 5
    b_obs = []
    t_obs = []

    for i in range(N):
        bx = 400.0 + i * 150.0
        by = 600.0 + i * 100.0
        _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)
        b_obs.append(make_obs("bottom", i, bx, by, area=650.0))
        t_obs.append(make_obs("top", i, tx, ty, area=640.0))

    res = fusion_engine.fuse_views(b_obs, t_obs)

    # Must be 5 physical grains, NOT 10!
    assert res.total_physical_grains == N
    assert res.matched_count == N
    assert res.unmatched_bottom_count == 0
    assert res.unmatched_top_count == 0
    # Mass must be approximately 5 grains worth, NOT 10!
    single_grain_mass = res.physical_grains[0].quality_attributes["estimated_mass_g"]
    assert abs(res.estimated_mass_g - (N * single_grain_mass)) < 1e-4


# Test 3: Bottom detection cannot map to two top detections
def test_bottom_cannot_map_to_two_top_detections(calibrated_registration, fusion_engine):
    bx, by = 600.0, 700.0
    _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)

    b_obs = [make_obs("bottom", 1, bx, by, area=600.0)]
    # Two top detections: one close (dist ~3px), one further (dist ~25px)
    t_obs = [
        make_obs("top", 1, tx + 2.0, ty + 2.0, area=605.0),
        make_obs("top", 2, tx + 25.0, ty, area=610.0)
    ]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.total_physical_grains == 2
    assert res.matched_count == 1
    assert res.unmatched_top_count == 1

    matched = [g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.MATCHED]
    assert len(matched) == 1
    assert matched[0].top_observation_id == "OBS_T_00001"


# Test 4: Top detection cannot map to two bottom detections
def test_top_cannot_map_to_two_bottom_detections(calibrated_registration, fusion_engine):
    bx1, by1 = 800.0, 800.0
    bx2, by2 = 820.0, 800.0  # 20px away
    _, tx1, ty1 = calibrated_registration.project_bottom_to_top(bx1, by1)

    b_obs = [
        make_obs("bottom", 1, bx1, by1, area=600.0),
        make_obs("bottom", 2, bx2, by2, area=600.0)
    ]
    # Single top detection right at projection of b_1
    t_obs = [make_obs("top", 1, tx1, ty1, area=600.0)]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.total_physical_grains == 2
    assert res.matched_count == 1
    assert res.unmatched_bottom_count == 1

    matched = [g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.MATCHED]
    assert matched[0].bottom_observation_id == "OBS_B_00001"


# Test 5: Missing candidates do NOT trigger forced assignments
def test_missing_candidates_do_not_trigger_forced_assignments(fusion_engine):
    b_obs = [make_obs("bottom", 1, 500.0, 500.0)]
    # Far-away top observation (over 1000px away)
    t_obs = [make_obs("top", 1, 2000.0, 2000.0)]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.matched_count == 0
    assert res.unmatched_bottom_count == 1
    assert res.unmatched_top_count == 1
    assert res.total_physical_grains == 2


# Test 6: Ambiguous matches do not enter automatic fusion
def test_ambiguous_matches_do_not_enter_automatic_fusion(calibrated_registration, fusion_engine):
    bx, by = 1000.0, 1000.0
    _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)

    b_obs = [make_obs("bottom", 1, bx, by, area=600.0)]
    # Two top detections equidistant (e.g. dist ~15px and dist ~16px, diff < 8px)
    t_obs = [
        make_obs("top", 1, tx + 15.0, ty, area=600.0),
        make_obs("top", 2, tx - 15.0, ty, area=600.0)
    ]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.matched_count == 0
    assert res.ambiguous_count > 0
    # Ambiguous records must NOT have automatic fusion applied
    for g in res.physical_grains:
        if g.correspondence_status == CorrespondenceStatus.AMBIGUOUS:
            assert g.quality_attributes["is_fused"] is False


# Test 7: Registration failure triggers safe fail-closed fallback
def test_registration_failure_triggers_safe_fallback():
    # Corrupted matrix with positive determinant (+1.0 -> no reflection)
    H_bad = np.array([
        [ 1.0, 0.0, 100.0],
        [ 0.0, 1.0, 100.0],
        [ 0.0, 0.0,   1.0]
    ])
    bad_reg = DualViewRegistration(homography_matrix=H_bad)
    assert bad_reg.is_healthy is False

    engine = DualViewCorrespondenceEngine(registration=bad_reg)
    fusion = DualViewFusionEngine(correspondence_engine=engine)

    b_obs = [make_obs("bottom", 1, 500.0, 500.0)]
    t_obs = [make_obs("top", 1, 500.0, 500.0)]

    res = fusion.fuse_views(b_obs, t_obs)

    assert res.registration_failed_count == 2
    assert res.matched_count == 0
    for g in res.physical_grains:
        assert g.correspondence_status == CorrespondenceStatus.REGISTRATION_FAILED
        assert g.quality_attributes["is_fused"] is False


# Test 8: Existing single-view / legacy behavior remains operational
def test_existing_single_view_behavior_remains_operational(tmp_path):
    p1 = str(tmp_path / "p1.jpg")
    p2 = str(tmp_path / "p2.jpg")

    import cv2
    img1 = np.full((400, 400, 3), 40, dtype=np.uint8)
    cv2.ellipse(img1, (200, 200), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p1, img1)

    img2 = np.full((400, 400, 3), 40, dtype=np.uint8)
    cv2.ellipse(img2, (200, 200), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p2, img2)

    from src.pipeline import RiceQualityPipeline
    cfg = PipelineConfig()
    cfg.segmentation.engine = "classic"
    pipeline = RiceQualityPipeline(config=cfg)

    # Legacy mode: use_dual_view_fusion=False
    mgr_legacy = DualPlatenManager(pipeline=pipeline, use_dual_view_fusion=False)
    res_legacy = mgr_legacy.run_dual_analysis(p1, p2, sample_id="TEST-LEGACY")

    assert res_legacy["status"] == "Complete"
    assert "is_dual_view_prototype" not in res_legacy or res_legacy.get("is_dual_view_prototype") is not True

    # Prototype mode: use_dual_view_fusion=True
    mgr_proto = DualPlatenManager(pipeline=pipeline, use_dual_view_fusion=True)
    res_proto = mgr_proto.run_dual_analysis(p1, p2, sample_id="TEST-PROTO")

    assert res_proto["status"] == "Complete"
    assert res_proto["is_dual_view_prototype"] is True
    assert "physical_grains" in res_proto
    assert "dual_view_diagnostics" in res_proto


# Test 9: Sample-level aggregation counts each physical grain once
def test_sample_level_aggregation_counts_each_physical_grain_once(calibrated_registration, fusion_engine):
    b_obs = [
        make_obs("bottom", 1, 500.0, 500.0, area=600.0, dclass="Good / Head Rice"),
        make_obs("bottom", 2, 700.0, 500.0, area=650.0, dclass="Chalky Grains"),
        make_obs("bottom", 3, 900.0, 500.0, area=580.0, dclass="Big Brokens (25-75%)")
    ]
    # Top has counterpart for grain 1 only
    _, tx1, ty1 = calibrated_registration.project_bottom_to_top(500.0, 500.0)
    t_obs = [
        make_obs("top", 1, tx1, ty1, area=610.0, dclass="Good / Head Rice"),
        make_obs("top", 99, 100.0, 100.0, area=500.0, dclass="Damaged Grains")  # Unmatched top
    ]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    # 1 matched + 2 unmatched bottom + 1 unmatched top = 4 physical grains
    assert res.total_physical_grains == 4
    cb = res.aggregation_summary["count_basis"]
    assert cb["total_physical_grains"] == 4
    assert cb["whole_count"] == 1
    assert cb["chalky_count"] == 1
    assert cb["broken_count"] == 1
    assert cb["damaged_count"] == 1
    total_cat_sum = (cb["whole_count"] + cb["chalky_count"] + cb["broken_count"] +
                     cb["damaged_count"] + cb["discoloured_count"] + cb["small_broken_count"] +
                     cb["foreign_matter_count"] + cb["unresolved_count"])
    assert total_cat_sum == 4


# Test 10: Unmatched observations remain traceable
def test_unmatched_observations_remain_traceable(fusion_engine):
    b_obs = [make_obs("bottom", 42, 500.0, 600.0, area=612.0)]
    t_obs = [make_obs("top", 99, 2000.0, 2000.0, area=580.0)]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.total_physical_grains == 2
    b_grain = next(g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.UNMATCHED_BOTTOM)
    assert b_grain.bottom_obs is not None
    assert b_grain.bottom_obs.detection_id == "b_42"
    assert b_grain.bottom_obs.centroid_x == 500.0

    t_grain = next(g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.UNMATCHED_TOP)
    assert t_grain.top_obs is not None
    assert t_grain.top_obs.detection_id == "t_99"
    assert t_grain.top_obs.centroid_x == 2000.0


# Test 11: Phase 3.2 Regression - Dropped sub-400px detection cannot cause distant b_859 double-assignment
def test_reproduce_phase3_2_dropped_detection_b859_bug(calibrated_registration, fusion_engine):
    """
    In Phase 3.2, grain CLEAN_GRAIN_140 was dropped from detection because area was 390px (<400px threshold).
    A naive argmin() found b_859 199.1px away, which was already the true detection for CLEAN_GRAIN_138.
    This caused b_859 to be double-assigned and generated a false positive mismatch.
    This regression test proves that our architecture prevents distant hijacking:
    Distance gating strictly rejects any candidate > 35px, and bipartite matching guarantees
    b_859 can NEVER be assigned to two records.
    """
    # Detection b_859 at (518.4, 31.4)
    b_859 = make_obs("bottom", 859, 518.44, 31.40, area=770.0)
    b_obs = [b_859]

    # Two queries:
    # 1. True top observation t_981 at (4400.2, 53.3) -> distance to b_859 proj is ~2.0px (VALID MATCH)
    _, tx_859, ty_859 = calibrated_registration.project_bottom_to_top(518.44, 31.40)
    t_981 = make_obs("top", 981, tx_859 + 1.5, ty_859 - 1.0, area=760.0)

    # 2. Second top observation t_distant at (4265.9, 200.4) -> distance to b_859 proj is ~199px away!
    t_distant = make_obs("top", 999, 4265.94, 200.39, area=390.0)

    t_obs = [t_981, t_distant]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    # b_859 must match t_981
    matched = [g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.MATCHED]
    assert len(matched) == 1
    assert matched[0].bottom_observation_id == "OBS_B_00859"
    assert matched[0].top_observation_id == "OBS_T_00981"

    # t_distant is 199px away: must NEVER be forced or hijack b_859!
    unmatched_t = [g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.UNMATCHED_TOP]
    assert len(unmatched_t) == 1
    assert unmatched_t[0].top_observation_id == "OBS_T_00999"

    # Total physical grains = 1 matched + 1 unmatched top = 2
    assert res.total_physical_grains == 2
