"""
Dual-View Fusion and Aggregation Engine.
Merges matched and unmatched ViewObservation instances into canonical PhysicalGrain records,
enforces agronomic defect precedence, and executes non-duplicating sample-level aggregation.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.dual_view.models import (
    ViewObservation,
    PhysicalGrain,
    CorrespondenceStatus,
    DualViewFusionResult
)
from src.dual_view.correspondence import DualViewCorrespondenceEngine


def dataframe_to_view_observations(
    df: Optional[pd.DataFrame],
    scanner_id: str,
    pixel_scale_mm: float = 0.08467
) -> List[ViewObservation]:
    """
    Converts a pipeline measurements DataFrame into a list of ViewObservation dataclass records.
    Safely parses missing or optional columns without raising KeyErrors.
    """
    if df is None or df.empty:
        return []

    observations: List[ViewObservation] = []
    prefix = scanner_id[0].upper()

    for idx, row in df.iterrows():
        det_id = str(row.get("grain_id", f"{scanner_id}_{idx}"))
        obs_id = f"OBS_{prefix}_{idx:05d}"

        cx = float(row.get("centroid_x", row.get("x", 0.0) + row.get("width", 0.0) / 2.0))
        cy = float(row.get("centroid_y", row.get("y", 0.0) + row.get("height", 0.0) / 2.0))
        bbox = (
            int(row.get("x", 0)),
            int(row.get("y", 0)),
            int(row.get("width", 0)),
            int(row.get("height", 0))
        )
        area_px = float(row.get("area_px", 0.0))
        len_mm = float(row.get("length_mm", row.get("length_px", 0.0) * pixel_scale_mm))
        wid_mm = float(row.get("width_mm", row.get("width_px", 0.0) * pixel_scale_mm))
        ar = float(row.get("aspect_ratio", len_mm / max(wid_mm, 0.01)))

        # Quality and defect labels
        dclass = str(row.get("classification", row.get("quality_class", "Good / Head Rice")))
        conf = float(row.get("confidence", 1.0))
        is_broken = bool(row.get("is_broken", False) or "Broken" in dclass or ar < 1.6)

        obs = ViewObservation(
            observation_id=obs_id,
            scanner_id=scanner_id,
            detection_id=det_id,
            centroid_x=cx,
            centroid_y=cy,
            bounding_box=bbox,
            area_px=area_px,
            length_mm=len_mm,
            width_mm=wid_mm,
            aspect_ratio=ar,
            defect_class=dclass,
            defect_confidence=conf,
            is_broken=is_broken,
            detection_confidence=conf,
            raw_features=row.to_dict()
        )
        observations.append(obs)

    return observations


class DualViewFusionEngine:
    """
    Fuses cross-view observations into canonical physical grains.
    Enforces:
      - Canonical grain identity: 1 physical grain = 1 record.
      - Agronomic defect precedence: surface defect visible on either side marks grain as defective.
      - Non-duplicating mass aggregation: mass estimated once per physical grain.
      - Safe handling of ambiguous, failed, and unresolved cluster records.
    """

    DEFECT_SEVERITY_ORDER = [
        "Foreign Matter",
        "Damaged Grains",
        "Discoloured Grains",
        "Small Brokens (<25%)",
        "Big Brokens (25-75%)",
        "Chalky Grains",
        "Good / Head Rice"
    ]

    def __init__(
        self,
        correspondence_engine: DualViewCorrespondenceEngine,
        pixel_scale_mm: float = 0.08467,  # Default for 300 DPI (25.4 / 300)
        mass_conversion_factor: Optional[float] = None
    ):
        self.correspondence_engine = correspondence_engine
        self.pixel_scale_mm = pixel_scale_mm
        self.mass_conv_factor = mass_conversion_factor

    def fuse_views(
        self,
        bottom_obs: List[ViewObservation],
        top_obs: List[ViewObservation]
    ) -> DualViewFusionResult:
        """
        Executes correspondence matching and synthesizes canonical PhysicalGrain records.
        """
        match_records = self.correspondence_engine.match_observations(bottom_obs, top_obs)

        physical_grains: List[PhysicalGrain] = []
        grain_counter = 1

        matched_cnt = 0
        unmatched_b_cnt = 0
        unmatched_t_cnt = 0
        ambiguous_cnt = 0
        reg_failed_cnt = 0
        unresolved_cluster_cnt = 0

        for m in match_records:
            status: CorrespondenceStatus = m["status"]
            b_obs: Optional[ViewObservation] = m.get("bottom_obs")
            t_obs: Optional[ViewObservation] = m.get("top_obs")
            conf: float = m.get("confidence", 0.0)
            cost_comps: dict = m.get("cost_components", {})

            phys_id = f"PHYS_{grain_counter:05d}"
            grain_counter += 1

            if status == CorrespondenceStatus.MATCHED:
                matched_cnt += 1
                q_attrs = self._fuse_matched_quality(b_obs, t_obs)
                prov = {
                    "matching_algorithm": "Hungarian_Bipartite",
                    "reprojection_distance_px": m.get("distance_px"),
                    "cost_components": cost_comps,
                    "fusion_policy": "agronomic_defect_precedence"
                }
            elif status == CorrespondenceStatus.UNMATCHED_BOTTOM:
                unmatched_b_cnt += 1
                q_attrs = self._extract_single_view_quality(b_obs, view="bottom")
                prov = {"matching_algorithm": "None", "reason": cost_comps.get("reason")}
            elif status == CorrespondenceStatus.UNMATCHED_TOP:
                unmatched_t_cnt += 1
                q_attrs = self._extract_single_view_quality(t_obs, view="top")
                prov = {"matching_algorithm": "None", "reason": cost_comps.get("reason")}
            elif status == CorrespondenceStatus.AMBIGUOUS:
                ambiguous_cnt += 1
                obs = b_obs if b_obs is not None else t_obs
                q_attrs = self._handle_unfused_record(obs, status=status, reason=cost_comps.get("reason"))
                prov = {"matching_algorithm": "Ambiguity_Detection", "reason": cost_comps.get("reason")}
            elif status == CorrespondenceStatus.REGISTRATION_FAILED:
                reg_failed_cnt += 1
                obs = b_obs if b_obs is not None else t_obs
                q_attrs = self._handle_unfused_record(obs, status=status, reason=cost_comps.get("reason"))
                prov = {"matching_algorithm": "Fail_Closed", "reason": cost_comps.get("reason")}
            elif status == CorrespondenceStatus.SEGMENTATION_UNRESOLVED:
                unresolved_cluster_cnt += 1
                obs = b_obs if b_obs is not None else t_obs
                q_attrs = self._handle_unfused_record(obs, status=status, reason=cost_comps.get("reason"))
                prov = {"matching_algorithm": "Cluster_Segregation", "reason": cost_comps.get("reason")}
            else:
                obs = b_obs if b_obs is not None else t_obs
                q_attrs = self._handle_unfused_record(obs, status=status, reason="Unknown")
                prov = {"matching_algorithm": "Unknown"}

            phys_grain = PhysicalGrain(
                physical_grain_id=phys_id,
                correspondence_status=status,
                correspondence_confidence=conf,
                bottom_observation_id=b_obs.observation_id if b_obs else None,
                top_observation_id=t_obs.observation_id if t_obs else None,
                bottom_obs=b_obs,
                top_obs=t_obs,
                quality_attributes=q_attrs,
                provenance=prov
            )
            physical_grains.append(phys_grain)

        # Aggregate sample-level metrics across canonical physical grains
        agg_summary, total_mass_g = self._aggregate_physical_sample(physical_grains)

        return DualViewFusionResult(
            physical_grains=physical_grains,
            total_physical_grains=len(physical_grains),
            matched_count=matched_cnt,
            unmatched_bottom_count=unmatched_b_cnt,
            unmatched_top_count=unmatched_t_cnt,
            ambiguous_count=ambiguous_cnt,
            registration_failed_count=reg_failed_cnt,
            unresolved_cluster_count=unresolved_cluster_cnt,
            estimated_mass_g=total_mass_g,
            is_experimental=True,
            registration_metadata=self.correspondence_engine.registration.get_health_report(),
            aggregation_summary=agg_summary
        )

    def _fuse_matched_quality(self, b_obs: ViewObservation, t_obs: ViewObservation) -> Dict[str, Any]:
        """
        Fuses quality attributes for a confirmed one-to-one matched grain pair.
        Implements agronomic defect precedence: defect visible on either side classifies grain.
        """
        # 1. Defect class precedence
        b_class = b_obs.defect_class
        t_class = t_obs.defect_class

        if b_class == t_class:
            fused_class = b_class
            fused_conf = float(np.mean([b_obs.defect_confidence, t_obs.defect_confidence]))
            defect_source = "both_views_agree"
        else:
            # Select higher severity defect according to agronomic inspection standard
            b_rank = self.DEFECT_SEVERITY_ORDER.index(b_class) if b_class in self.DEFECT_SEVERITY_ORDER else 99
            t_rank = self.DEFECT_SEVERITY_ORDER.index(t_class) if t_class in self.DEFECT_SEVERITY_ORDER else 99
            if b_rank <= t_rank:
                fused_class = b_class
                fused_conf = b_obs.defect_confidence
                defect_source = "bottom_view_defect_precedence"
            else:
                fused_class = t_class
                fused_conf = t_obs.defect_confidence
                defect_source = "top_view_defect_precedence"

        # 2. Geometric attributes (average of both views for stable metric sizing)
        fused_len_mm = float(np.mean([b_obs.length_mm, t_obs.length_mm]))
        fused_wid_mm = float(np.mean([b_obs.width_mm, t_obs.width_mm]))
        fused_area_px = float(np.mean([b_obs.area_px, t_obs.area_px]))
        fused_aspect_ratio = round(fused_len_mm / max(fused_wid_mm, 0.01), 2)
        is_broken = bool(b_obs.is_broken or t_obs.is_broken or "Broken" in fused_class)

        # 3. Single mass calculation (NO DOUBLING!)
        mass_g = self._calculate_single_grain_mass(fused_area_px, fused_len_mm, fused_wid_mm)

        return {
            "defect_class": fused_class,
            "defect_confidence": round(fused_conf, 4),
            "defect_precedence_source": defect_source,
            "is_broken": is_broken,
            "length_mm": round(fused_len_mm, 2),
            "width_mm": round(fused_wid_mm, 2),
            "area_px": round(fused_area_px, 1),
            "aspect_ratio": fused_aspect_ratio,
            "estimated_mass_g": round(mass_g, 5),
            "is_fused": True,
            "bottom_prediction": b_class,
            "top_prediction": t_class
        }

    def _extract_single_view_quality(self, obs: ViewObservation, view: str) -> Dict[str, Any]:
        """Extracts quality attributes for an unmatched grain observed on one platen only."""
        mass_g = self._calculate_single_grain_mass(obs.area_px, obs.length_mm, obs.width_mm)
        return {
            "defect_class": obs.defect_class,
            "defect_confidence": round(obs.defect_confidence, 4),
            "defect_precedence_source": f"single_view_{view}",
            "is_broken": obs.is_broken,
            "length_mm": round(obs.length_mm, 2),
            "width_mm": round(obs.width_mm, 2),
            "area_px": round(obs.area_px, 1),
            "aspect_ratio": round(obs.aspect_ratio, 2),
            "estimated_mass_g": round(mass_g, 5),
            "is_fused": False,
            "view_present": view
        }

    def _handle_unfused_record(self, obs: Optional[ViewObservation], status: CorrespondenceStatus, reason: Optional[str]) -> Dict[str, Any]:
        """Preserves record without automatic fusion for ambiguous/unresolved/failed cases."""
        area = obs.area_px if obs else 0.0
        len_mm = obs.length_mm if obs else 0.0
        wid_mm = obs.width_mm if obs else 0.0
        mass_g = self._calculate_single_grain_mass(area, len_mm, wid_mm) if obs else 0.0

        return {
            "defect_class": "Unresolved / Pending Review",
            "defect_confidence": 0.0,
            "defect_precedence_source": "none",
            "is_broken": False,
            "length_mm": round(len_mm, 2),
            "width_mm": round(wid_mm, 2),
            "area_px": round(area, 1),
            "aspect_ratio": round(len_mm / max(wid_mm, 0.01), 2) if wid_mm > 0 else 0.0,
            "estimated_mass_g": round(mass_g, 5),
            "is_fused": False,
            "unfused_reason": reason or status.value
        }

    def _calculate_single_grain_mass(self, area_px: float, length_mm: float, width_mm: float) -> float:
        """Estimates single grain mass. Guaranteed never to double-count."""
        if self.mass_conv_factor is not None:
            return float(area_px * self.mass_conv_factor)
        if length_mm > 0.0 and width_mm > 0.0:
            thickness_mm = 0.70 * width_mm
            vol_mm3 = (4.0 / 3.0) * np.pi * (length_mm / 2.0) * (width_mm / 2.0) * (thickness_mm / 2.0)
            density_g_mm3 = 0.00142  # ~1.42 g/cm3 standard rice endosperm density
            return float(vol_mm3 * density_g_mm3)
        else:
            return float(area_px * (self.pixel_scale_mm ** 2) * 0.00025)

    def _aggregate_physical_sample(self, grains: List[PhysicalGrain]) -> Tuple[Dict[str, Any], float]:
        """
        Executes sample-level aggregation strictly on canonical PhysicalGrain records.
        Each physical grain contributes exactly once to count and mass totals.
        """
        total_physical_count = len(grains)
        total_sample_mass_g = 0.0

        category_counts = {
            "Good / Head Rice": 0,
            "Big Brokens (25-75%)": 0,
            "Small Brokens (<25%)": 0,
            "Foreign Matter": 0,
            "Damaged Grains": 0,
            "Discoloured Grains": 0,
            "Chalky Grains": 0,
            "Unresolved / Pending Review": 0
        }
        category_masses = {k: 0.0 for k in category_counts}

        lengths, breadths = [], []

        for g in grains:
            qa = g.quality_attributes
            mass = qa.get("estimated_mass_g", 0.0)
            dclass = qa.get("defect_class", "Unresolved / Pending Review")

            total_sample_mass_g += mass

            if dclass in category_counts:
                category_counts[dclass] += 1
                category_masses[dclass] += mass
            else:
                category_counts["Unresolved / Pending Review"] += 1
                category_masses["Unresolved / Pending Review"] += mass

            if qa.get("length_mm", 0.0) > 0.0:
                lengths.append(qa["length_mm"])
            if qa.get("width_mm", 0.0) > 0.0:
                breadths.append(qa["width_mm"])

        avg_len = float(np.mean(lengths)) if lengths else 0.0
        avg_brd = float(np.mean(breadths)) if breadths else 0.0

        denom_cnt = max(total_physical_count, 1)
        denom_mass = max(total_sample_mass_g, 1e-6)

        count_basis = {
            "total_physical_grains": total_physical_count,
            "whole_count": category_counts["Good / Head Rice"],
            "whole_pct": round(category_counts["Good / Head Rice"] / denom_cnt * 100, 2),
            "broken_count": category_counts["Big Brokens (25-75%)"],
            "broken_pct": round(category_counts["Big Brokens (25-75%)"] / denom_cnt * 100, 2),
            "small_broken_count": category_counts["Small Brokens (<25%)"],
            "small_broken_pct": round(category_counts["Small Brokens (<25%)"] / denom_cnt * 100, 2),
            "foreign_matter_count": category_counts["Foreign Matter"],
            "foreign_matter_pct": round(category_counts["Foreign Matter"] / denom_cnt * 100, 2),
            "damaged_count": category_counts["Damaged Grains"],
            "damaged_pct": round(category_counts["Damaged Grains"] / denom_cnt * 100, 2),
            "discoloured_count": category_counts["Discoloured Grains"],
            "discoloured_pct": round(category_counts["Discoloured Grains"] / denom_cnt * 100, 2),
            "chalky_count": category_counts["Chalky Grains"],
            "chalky_pct": round(category_counts["Chalky Grains"] / denom_cnt * 100, 2),
            "unresolved_count": category_counts["Unresolved / Pending Review"],
            "unresolved_pct": round(category_counts["Unresolved / Pending Review"] / denom_cnt * 100, 2)
        }

        mass_basis = {
            "total_estimated_mass_g": round(total_sample_mass_g, 4),
            "is_measured_scale_mass": False,
            "mass_estimation_type": "Volumetric Optical Proxy (Unvalidated for Statutory Trade)",
            "whole_mass_g": round(category_masses["Good / Head Rice"], 4),
            "whole_pct": round(category_masses["Good / Head Rice"] / denom_mass * 100, 2),
            "broken_mass_g": round(category_masses["Big Brokens (25-75%)"], 4),
            "broken_pct": round(category_masses["Big Brokens (25-75%)"] / denom_mass * 100, 2),
            "small_broken_mass_g": round(category_masses["Small Brokens (<25%)"], 4),
            "small_broken_pct": round(category_masses["Small Brokens (<25%)"] / denom_mass * 100, 2),
            "foreign_matter_mass_g": round(category_masses["Foreign Matter"], 4),
            "foreign_matter_pct": round(category_masses["Foreign Matter"] / denom_mass * 100, 2),
            "damaged_mass_g": round(category_masses["Damaged Grains"], 4),
            "damaged_pct": round(category_masses["Damaged Grains"] / denom_mass * 100, 2),
            "discoloured_mass_g": round(category_masses["Discoloured Grains"], 4),
            "discoloured_pct": round(category_masses["Discoloured Grains"] / denom_mass * 100, 2),
            "chalky_mass_g": round(category_masses["Chalky Grains"], 4),
            "chalky_pct": round(category_masses["Chalky Grains"] / denom_mass * 100, 2),
            "unresolved_mass_g": round(category_masses["Unresolved / Pending Review"], 4),
            "unresolved_pct": round(category_masses["Unresolved / Pending Review"] / denom_mass * 100, 2)
        }

        stats = [
            {"param": "Good / Head Rice", "count": count_basis["whole_count"], "pct": count_basis["whole_pct"], "mass_g": mass_basis["whole_mass_g"], "mass_pct": mass_basis["whole_pct"], "limit": "Min 75.00%"},
            {"param": "Big Brokens (25-75%)", "count": count_basis["broken_count"], "pct": count_basis["broken_pct"], "mass_g": mass_basis["broken_mass_g"], "mass_pct": mass_basis["broken_pct"], "limit": "Max 15.00%"},
            {"param": "Small Brokens (<25%)", "count": count_basis["small_broken_count"], "pct": count_basis["small_broken_pct"], "mass_g": mass_basis["small_broken_mass_g"], "mass_pct": mass_basis["small_broken_pct"], "limit": "Max 1.00%"},
            {"param": "Foreign Matter", "count": count_basis["foreign_matter_count"], "pct": count_basis["foreign_matter_pct"], "mass_g": mass_basis["foreign_matter_mass_g"], "mass_pct": mass_basis["foreign_matter_pct"], "limit": "Max 0.50%"},
            {"param": "Damaged Grains", "count": count_basis["damaged_count"], "pct": count_basis["damaged_pct"], "mass_g": mass_basis["damaged_mass_g"], "mass_pct": mass_basis["damaged_pct"], "limit": "Max 2.00%"},
            {"param": "Discoloured Grains", "count": count_basis["discoloured_count"], "pct": count_basis["discoloured_pct"], "mass_g": mass_basis["discoloured_mass_g"], "mass_pct": mass_basis["discoloured_pct"], "limit": "Max 1.00%"},
            {"param": "Chalky Grains", "count": count_basis["chalky_count"], "pct": count_basis["chalky_pct"], "mass_g": mass_basis["chalky_mass_g"], "mass_pct": mass_basis["chalky_pct"], "limit": "Max 3.00%"},
            {"param": "Unresolved Grains", "count": count_basis["unresolved_count"], "pct": count_basis["unresolved_pct"], "mass_g": mass_basis["unresolved_mass_g"], "mass_pct": mass_basis["unresolved_pct"], "limit": "Pending Review"}
        ]

        summary = {
            "total_physical_grains": total_physical_count,
            "total_estimated_mass_g": round(total_sample_mass_g, 4),
            "avg_length_mm": round(avg_len, 2),
            "avg_breadth_mm": round(avg_brd, 2),
            "length_breadth_ratio": round(avg_len / max(avg_brd, 0.01), 2),
            "count_basis": count_basis,
            "mass_basis": mass_basis,
            "stats": stats,
            "is_single_grain_non_doubled": True
        }

        return summary, total_sample_mass_g
