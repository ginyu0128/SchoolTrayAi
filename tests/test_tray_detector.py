import cv2
import numpy as np

from src.tray_detector import detect_tray, draw_tray_detection


def test_detect_tray_finds_large_rectangular_candidate():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[30:220, 30:295] = 230

    detection = detect_tray(image)

    assert detection.found is True
    x, y, w, h = detection.bbox
    assert 20 <= x <= 40
    assert 20 <= y <= 40
    assert 250 <= w <= 280
    assert 180 <= h <= 205


def test_detect_tray_falls_back_when_no_candidate_exists():
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    detection = detect_tray(image)

    assert detection.found is False
    assert detection.bbox == (0, 0, 120, 100)


def test_detect_tray_falls_back_when_candidate_is_only_a_bowl_region():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2_center = (230, 150)

    cv2.circle(image, cv2_center, 55, (230, 230, 230), 4)

    detection = detect_tray(image)

    assert detection.found is False
    assert detection.bbox == (0, 0, 320, 240)


def test_draw_tray_detection_preserves_image_shape():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    detection = detect_tray(image)

    annotated = draw_tray_detection(image, detection)

    assert annotated.shape == image.shape
