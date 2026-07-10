from io import BytesIO

import pytest
from PIL import Image

from src.image_io import ImageLoadError, load_uploaded_image


def test_load_uploaded_image_returns_rgb_numpy_array():
    image_file = BytesIO()
    Image.new("RGBA", (12, 8), color=(255, 0, 0, 128)).save(image_file, format="PNG")
    image_file.seek(0)

    image_np = load_uploaded_image(image_file)

    assert image_np.shape == (8, 12, 3)
    assert image_np[0, 0].tolist() == [255, 0, 0]


def test_load_uploaded_image_rejects_invalid_file():
    invalid_file = BytesIO(b"not an image")

    with pytest.raises(ImageLoadError):
        load_uploaded_image(invalid_file)
