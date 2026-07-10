from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from src.tray_specs import load_tray_roi_layout, load_tray_specs


def _scale_bbox(
    tray_bbox: tuple[int, int, int, int],
    normalized_bbox: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x, y, width, height = tray_bbox
    nx, ny, nw, nh = normalized_bbox
    return (
        int(round(x + nx * width)),
        int(round(y + ny * height)),
        max(1, int(round(nw * width))),
        max(1, int(round(nh * height))),
    )


def split_tray_cells(tray_bbox: tuple[int, int, int, int]) -> List[Dict[str, Any]]:
    """Split a detected tray bbox into measured tray compartments.

    Until perspective correction is implemented, the split uses normalized
    locations inside the detected tray rectangle. This keeps the app usable and
    makes layout errors visible through the debug overlay.
    """
    cells = []
    cell_layout = load_tray_roi_layout()
    for spec in load_tray_specs():
        cell_id = spec["cell_id"]
        bbox = _scale_bbox(tray_bbox, cell_layout[cell_id])
        cells.append({
            "cell_id": cell_id,
            "display_name": spec["display_name"],
            "shape": spec["shape"],
            "bbox": bbox,
            "spec": spec,
        })
    return cells


def draw_cell_overlay(
    image: np.ndarray,
    cells: List[Dict[str, Any]],
    volume_profiles: Dict[str, str] | None = None,
) -> np.ndarray:
    """Return an RGB image annotated with the current tray cell split."""
    annotated = image.copy()
    volume_profiles = volume_profiles or {}
    for index, cell in enumerate(cells, start=1):
        x, y, w, h = cell["bbox"]
        volume_profile = volume_profiles.get(cell["cell_id"], "standard")
        is_flat_piece = volume_profile == "flat_piece"
        color = (255, 80, 80) if is_flat_piece else (0, 180, 255)
        if cell["shape"] == "circular_bowl":
            center = (x + w // 2, y + h // 2)
            radius = max(1, min(w, h) // 2)
            cv2.circle(annotated, center, radius, color, 3)
        else:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

        label = f"{index}:{cell['cell_id']}"
        if is_flat_piece:
            label = f"{label} FLAT"
        cv2.putText(
            annotated,
            label,
            (max(x, 5), max(y + 24, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated
