import pytest

from src.tray_specs import get_tray_spec_map, load_volume_estimation_config
from src.volume_estimator import (
    estimate_flat_piece_volume_ml,
    estimate_standard_volume_ml,
    estimate_volume_ml,
    get_food_area_cm2,
    get_cell_floor_area_cm2,
    infer_flat_piece_cell_ids,
)


def test_flat_piece_volume_uses_fixed_thickness():
    spec = get_tray_spec_map()["side_right"]

    volume_ml = estimate_flat_piece_volume_ml(spec, area_ratio=0.5, thickness_cm=1.5)

    assert volume_ml == pytest.approx(11.1 * 8.8 * 0.5 * 1.5)


def test_food_area_uses_cell_floor_area_and_area_ratio():
    spec = get_tray_spec_map()["side_right"]

    food_area = get_food_area_cm2(spec, area_ratio=0.5)

    assert food_area == pytest.approx(11.1 * 8.8 * 0.5)


def test_standard_volume_uses_cell_max_volume():
    spec = get_tray_spec_map()["side_right"]

    volume_ml = estimate_standard_volume_ml(spec, area_ratio=0.5)

    assert volume_ml == pytest.approx(spec["max_volume_ml"] * 0.5)


def test_estimate_volume_selects_flat_piece_profile():
    spec = get_tray_spec_map()["side_left"]

    flat_volume = estimate_volume_ml(spec, area_ratio=1.0, volume_profile="flat_piece")
    standard_volume = estimate_volume_ml(spec, area_ratio=1.0, volume_profile="standard")

    assert flat_volume == pytest.approx(get_cell_floor_area_cm2(spec) * 1.5)
    assert flat_volume < standard_volume


def test_estimate_volume_selects_rice_mound_profile():
    spec = get_tray_spec_map()["rice"]

    rice_volume = estimate_volume_ml(spec, area_ratio=1.0, volume_profile="rice_mound")
    standard_volume = estimate_volume_ml(spec, area_ratio=1.0, volume_profile="standard")

    assert rice_volume == pytest.approx(get_cell_floor_area_cm2(spec) * 1.5)
    assert rice_volume < standard_volume


def test_infer_flat_piece_for_edge_side_cells():
    flat_cells = infer_flat_piece_cell_ids({
        "side_left": 0.8,
        "side_middle_left": 0.1,
        "side_middle_right": 0.1,
        "side_right": 0.85,
    })

    assert flat_cells == {"side_left", "side_right"}


def test_edge_side_cells_are_not_flat_below_edge_threshold():
    flat_cells = infer_flat_piece_cell_ids({
        "side_left": 0.79,
        "side_middle_left": 0.1,
        "side_middle_right": 0.1,
        "side_right": 0.79,
    })

    assert flat_cells == set()


def test_infer_flat_piece_for_middle_span_group():
    flat_cells = infer_flat_piece_cell_ids({
        "side_left": 0.1,
        "side_middle_left": 0.7,
        "side_middle_right": 0.75,
        "side_right": 0.1,
    })

    assert flat_cells == {"side_middle_left", "side_middle_right"}


def test_middle_span_cells_are_not_flat_below_span_threshold():
    flat_cells = infer_flat_piece_cell_ids({
        "side_left": 0.1,
        "side_middle_left": 0.69,
        "side_middle_right": 0.75,
        "side_right": 0.1,
    })

    assert flat_cells == set()


def test_volume_estimation_config_uses_requested_thickness():
    config = load_volume_estimation_config()

    assert config["rice_mound"]["thickness_cm"] == 1.5
    assert config["flat_piece"]["thickness_cm"] == 1.5
    assert config["flat_piece"]["edge_min_area_ratio"] == 0.8
    assert config["flat_piece"]["middle_span_min_area_ratio"] == 0.7
