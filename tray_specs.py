from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


TRAY_SPECS_PATH = Path(__file__).resolve().parents[1] / "data" / "tray_specs.json"


def _rectangular_volume_ml(cell: Dict[str, Any]) -> float:
    return float(cell["width_cm"]) * float(cell["height_cm"]) * float(cell["depth_cm"])


def _spherical_cap_volume_ml(cell: Dict[str, Any]) -> float:
    radius_cm = float(cell["diameter_cm"]) / 2.0
    depth_cm = float(cell["depth_cm"])
    cap_height = min(depth_cm, radius_cm)
    return math.pi * cap_height**2 * (3.0 * radius_cm - cap_height) / 3.0


def calculate_cell_volume_ml(cell: Dict[str, Any]) -> float:
    """Calculate maximum cell volume from measured dimensions."""
    if cell["shape"] == "rectangle":
        return _rectangular_volume_ml(cell)
    if cell["shape"] == "circular_bowl":
        return _spherical_cap_volume_ml(cell)
    raise ValueError(f"Unsupported tray cell shape: {cell['shape']}")


@lru_cache(maxsize=1)
def load_tray_config() -> Dict[str, Any]:
    with TRAY_SPECS_PATH.open("r", encoding="utf-8") as specs_file:
        return json.load(specs_file)


@lru_cache(maxsize=1)
def load_tray_specs() -> List[Dict[str, Any]]:
    """Load tray cell specs and attach derived max_volume_ml values."""
    payload = load_tray_config()

    cells = []
    for cell in payload["cells"]:
        enriched = dict(cell)
        enriched["max_volume_ml"] = calculate_cell_volume_ml(enriched)
        cells.append(enriched)
    return cells


def get_tray_spec_map() -> Dict[str, Dict[str, Any]]:
    return {cell["cell_id"]: cell for cell in load_tray_specs()}


def load_tray_roi_layout() -> Dict[str, tuple[float, float, float, float]]:
    """Load normalized ROI layout as x, y, width, height tuples."""
    payload = load_tray_config()
    roi_payload = payload["layout"]["roi"]
    return {
        cell_id: (
            float(values["x"]),
            float(values["y"]),
            float(values["width"]),
            float(values["height"]),
        )
        for cell_id, values in roi_payload.items()
    }


def load_volume_estimation_config() -> Dict[str, Any]:
    payload = load_tray_config()
    return payload["volume_estimation"]
