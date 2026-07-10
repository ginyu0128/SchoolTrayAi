from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class TrayDetection:
    """Detected tray region in image pixel coordinates."""

    bbox: tuple[int, int, int, int]
    found: bool
    area_ratio: float


def _full_image_detection(image: np.ndarray) -> TrayDetection:
    height, width = image.shape[:2]
    return TrayDetection(
        bbox=(0, 0, width, height),
        found=False,
        area_ratio=1.0 if width and height else 0.0,
    )


def detect_tray(
    image: np.ndarray,
    min_area_ratio: float = 0.08,
    min_bbox_width_ratio: float = 0.70,
    min_bbox_height_ratio: float = 0.60,
) -> TrayDetection:
    """Detect the largest tray-like region from an RGB image.

    This first OpenCV version intentionally stays conservative: it finds the
    largest external contour after edge extraction and falls back to the full
    image when the candidate is too small or invalid.
    """
    if image.size == 0:
        return _full_image_detection(image)

    image_area = image.shape[0] * image.shape[1]
    if image_area == 0:
        return _full_image_detection(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), dtype=np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox: Optional[tuple[int, int, int, int]] = None
    best_area = 0.0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area <= best_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        best_bbox = (x, y, w, h)
        best_area = float(contour_area)

    if best_bbox is None:
        return _full_image_detection(image)

    area_ratio = best_area / float(image_area)
    _, _, bbox_width, bbox_height = best_bbox
    bbox_width_ratio = bbox_width / float(image.shape[1])
    bbox_height_ratio = bbox_height / float(image.shape[0])
    if area_ratio < min_area_ratio:
        return _full_image_detection(image)
    if bbox_width_ratio < min_bbox_width_ratio or bbox_height_ratio < min_bbox_height_ratio:
        return _full_image_detection(image)

    return TrayDetection(bbox=best_bbox, found=True, area_ratio=area_ratio)


def draw_tray_detection(image: np.ndarray, detection: TrayDetection) -> np.ndarray:
    """Return an RGB image annotated with the detected tray bounding box."""
    annotated = image.copy()
    x, y, w, h = detection.bbox
    color = (0, 255, 0) if detection.found else (255, 180, 0)
    label = "tray detected" if detection.found else "fallback: full image"

    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
    cv2.putText(
        annotated,
        label,
        (max(x, 5), max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated
