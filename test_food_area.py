import numpy as np

from src.food_area import estimate_food_area_ratio


def test_estimate_food_area_ratio_detects_saturated_food():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = [210, 210, 210]
    image[:, :80] = [60, 130, 30]

    ratio = estimate_food_area_ratio(image)

    assert ratio >= 0.75


def test_estimate_food_area_ratio_detects_dark_food():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = [210, 210, 210]
    image[:, :80] = [65, 45, 25]

    ratio = estimate_food_area_ratio(image)

    assert ratio >= 0.75


def test_estimate_food_area_ratio_keeps_white_rice_special_case():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = [210, 210, 210]
    image[:, :80] = [245, 245, 245]

    generic_ratio = estimate_food_area_ratio(image)
    rice_ratio = estimate_food_area_ratio(image, cell_id="rice")

    assert generic_ratio < 0.2
    assert rice_ratio >= 0.75
