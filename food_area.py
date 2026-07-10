from __future__ import annotations

import cv2
import numpy as np


def estimate_food_area_ratio(image: np.ndarray, cell_id: str | None = None) -> float:
    """Estimate food-covered top-view area ratio inside a cell crop.

    Stainless tray backgrounds are often bright and low-saturation, while many
    foods are saturated, dark, or textured. Rice is a special case because it is
    bright, so rice-cell estimates keep a white-pixel component.
    """
    if image.size == 0:
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    saturated_food = saturation > 28
    dark_food = value < 120
    food_mask = saturated_food | dark_food

    if cell_id == "rice":
        white_food = (saturation < 45) & (value > 145)
        food_mask = food_mask | white_food

    food_mask = food_mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    food_pixels = np.count_nonzero(food_mask)
    total_pixels = food_mask.size
    return food_pixels / total_pixels if total_pixels else 0.0
