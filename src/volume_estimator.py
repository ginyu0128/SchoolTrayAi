from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Set

from src.tray_specs import load_volume_estimation_config


def get_cell_floor_area_cm2(cell_spec: Dict[str, Any]) -> float:
    """Return the top-view floor area used for area-based volume estimates."""
    if cell_spec["shape"] == "rectangle":
        return float(cell_spec["width_cm"]) * float(cell_spec["height_cm"])
    if cell_spec["shape"] == "circular_bowl":
        radius_cm = float(cell_spec["diameter_cm"]) / 2.0
        return math.pi * radius_cm**2
    raise ValueError(f"Unsupported tray cell shape: {cell_spec['shape']}")


def get_food_area_cm2(cell_spec: Dict[str, Any], area_ratio: float) -> float:
    """Estimate detected food top-view area in square centimeters."""
    return get_cell_floor_area_cm2(cell_spec) * area_ratio


def estimate_standard_volume_ml(cell_spec: Dict[str, Any], area_ratio: float) -> float:
    """Estimate volume for foods that roughly fill cell depth as area grows."""
    return float(cell_spec["max_volume_ml"]) * area_ratio


def estimate_flat_piece_volume_ml(
    cell_spec: Dict[str, Any],
    area_ratio: float,
    thickness_cm: float,
) -> float:
    """Estimate volume for wide, flat foods using fixed thickness."""
    return get_food_area_cm2(cell_spec, area_ratio) * thickness_cm


def infer_flat_piece_cell_ids(area_ratios: Dict[str, float]) -> Set[str]:
    """Infer flat-piece cells from configured tray positions and area ratios.

    This is intentionally conservative and limited to the requested cases:
    edge side-dish cells and rare flat pieces spanning the two middle cells.
    """
    config = load_volume_estimation_config()["flat_piece"]
    flat_cell_ids: Set[str] = set()

    edge_threshold = float(config["edge_min_area_ratio"])
    for cell_id in config["edge_cell_ids"]:
        if area_ratios.get(cell_id, 0.0) >= edge_threshold:
            flat_cell_ids.add(cell_id)

    span_threshold = float(config["middle_span_min_area_ratio"])
    for group in config["middle_span_cell_groups"]:
        if _all_cells_above_threshold(group, area_ratios, span_threshold):
            flat_cell_ids.update(group)

    return flat_cell_ids


def _all_cells_above_threshold(
    cell_ids: Iterable[str],
    area_ratios: Dict[str, float],
    threshold: float,
) -> bool:
    return all(area_ratios.get(cell_id, 0.0) >= threshold for cell_id in cell_ids)


def estimate_volume_ml(
    cell_spec: Dict[str, Any],
    area_ratio: float,
    volume_profile: str,
) -> float:
    """Estimate volume using the selected volume profile."""
    if volume_profile == "rice_mound":
        config = load_volume_estimation_config()["rice_mound"]
        return estimate_flat_piece_volume_ml(
            cell_spec,
            area_ratio,
            thickness_cm=float(config["thickness_cm"]),
        )
    if volume_profile == "flat_piece":
        config = load_volume_estimation_config()["flat_piece"]
        return estimate_flat_piece_volume_ml(
            cell_spec,
            area_ratio,
            thickness_cm=float(config["thickness_cm"]),
        )
    return estimate_standard_volume_ml(cell_spec, area_ratio)
