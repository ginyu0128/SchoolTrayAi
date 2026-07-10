import numpy as np

from src.tray_pipeline import estimate_tray_nutrition


def test_estimate_tray_nutrition_returns_summary():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    image[:, :, 1] = 255
    image[:, :, 2] = 255

    result = estimate_tray_nutrition(image)

    assert "cell_estimates" in result
    assert "nutrition_summary" in result
    assert len(result["cell_estimates"]) == 6
    assert "cell_split" in result["debug_images"]
    assert "volume_profile" in result["cell_estimates"][0]
    assert "flat_piece_cells" in result
    assert "classification" in result["cell_estimates"][0]
    assert "nutrition_search_name" in result["cell_estimates"][0]
    assert "classification_source" in result["nutrition_summary"].columns
    assert "nutrition_source" in result["nutrition_summary"].columns
