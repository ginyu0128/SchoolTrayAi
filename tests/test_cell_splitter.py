import numpy as np

from src.cell_splitter import draw_cell_overlay, split_tray_cells


def test_split_tray_cells_returns_measured_layout_cells():
    cells = split_tray_cells((10, 20, 300, 200))

    assert len(cells) == 6
    assert [cell["cell_id"] for cell in cells] == [
        "rice",
        "side_left",
        "side_middle_left",
        "side_middle_right",
        "side_right",
        "soup",
    ]
    assert all(cell["spec"]["max_volume_ml"] > 0 for cell in cells)


def test_split_tray_cells_scales_bboxes_inside_tray_bbox():
    tray_bbox = (10, 20, 300, 200)
    cells = split_tray_cells(tray_bbox)
    tray_x, tray_y, tray_w, tray_h = tray_bbox

    for cell in cells:
        x, y, w, h = cell["bbox"]
        assert tray_x <= x <= tray_x + tray_w
        assert tray_y <= y <= tray_y + tray_h
        assert x + w <= tray_x + tray_w
        assert y + h <= tray_y + tray_h


def test_split_tray_cells_uses_configured_roi_adjustments():
    cells = {cell["cell_id"]: cell for cell in split_tray_cells((0, 0, 1000, 1000))}

    assert cells["side_left"]["bbox"] == (0, 25, 265, 405)
    assert cells["side_middle_left"]["bbox"][2] == 180
    assert cells["side_middle_right"]["bbox"][2] == 180
    assert cells["side_right"]["bbox"] == (690, 50, 250, 380)
    assert cells["rice"]["bbox"] == (50, 500, 435, 450)
    assert cells["soup"]["bbox"] == (525, 475, 420, 525)


def test_draw_cell_overlay_preserves_image_shape():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cells = split_tray_cells((0, 0, 300, 200))

    annotated = draw_cell_overlay(image, cells)

    assert annotated.shape == image.shape


def test_draw_cell_overlay_marks_flat_piece_cells():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cells = split_tray_cells((0, 0, 300, 200))

    standard_overlay = draw_cell_overlay(image, cells)
    flat_overlay = draw_cell_overlay(image, cells, {"side_left": "flat_piece"})

    assert flat_overlay.shape == image.shape
    assert np.any(flat_overlay != standard_overlay)
