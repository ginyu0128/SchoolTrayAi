import pytest

from src.tray_specs import get_tray_spec_map, load_tray_roi_layout, load_tray_specs


def test_load_tray_specs_contains_measured_cells():
    specs = load_tray_specs()

    assert len(specs) == 6
    assert {spec["cell_id"] for spec in specs} == {
        "rice",
        "side_left",
        "side_middle_left",
        "side_middle_right",
        "side_right",
        "soup",
    }


def test_rectangular_cell_volumes_are_derived_from_dimensions():
    spec_map = get_tray_spec_map()

    assert spec_map["rice"]["max_volume_ml"] == pytest.approx(19.3 * 15.2 * 3.0)
    assert spec_map["side_left"]["max_volume_ml"] == pytest.approx(11.1 * 8.8 * 3.0)
    assert spec_map["side_middle_left"]["max_volume_ml"] == pytest.approx(11.4 * 8.0 * 2.0)


def test_soup_bowl_uses_curved_bowl_volume_model():
    spec_map = get_tray_spec_map()

    assert spec_map["soup"]["max_volume_ml"] == pytest.approx(508.94, abs=0.01)


def test_tray_roi_layout_is_loaded_from_config():
    layout = load_tray_roi_layout()

    assert layout["side_left"] == (0.0, 0.025, 0.265, 0.405)
    assert layout["side_middle_left"][2] < layout["side_right"][2]
    assert layout["side_middle_right"][2] < layout["side_right"][2]
