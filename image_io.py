from __future__ import annotations

from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageLoadError(ValueError):
    """Raised when an uploaded file cannot be used as an image."""


def load_uploaded_image(uploaded_file: BinaryIO) -> np.ndarray:
    """Load a Streamlit uploaded image as an RGB NumPy array.

    The pipeline uses RGB arrays as its public image format. OpenCV functions
    should therefore use COLOR_RGB2... conversions internally.
    """
    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError("The uploaded file is not a valid image.") from exc

    image_np = np.array(image)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ImageLoadError("The uploaded image must be an RGB image.")

    return image_np
