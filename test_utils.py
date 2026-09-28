"""
Basic unit tests for src/utils.py. Run with:
    pytest tests/
"""

import numpy as np
import pytest

from src.utils import clean_plate_text, crop_with_margin, is_image_file, is_video_file


def test_clean_plate_text_basic():
    assert clean_plate_text("lea-1234") == "LEA-1234"


def test_clean_plate_text_strips_noise():
    assert clean_plate_text(" ab#12 34! ") == "AB1234"


def test_clean_plate_text_empty():
    assert clean_plate_text("") == ""
    assert clean_plate_text(None) == ""


def test_is_image_file():
    assert is_image_file("car.jpg")
    assert is_image_file("car.PNG")
    assert not is_image_file("clip.mp4")


def test_is_video_file():
    assert is_video_file("clip.mp4")
    assert not is_video_file("photo.jpg")


def test_crop_with_margin_stays_in_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_with_margin(frame, (90, 90, 99, 99), margin=0.5)
    assert crop.shape[0] <= 100
    assert crop.shape[1] <= 100


def test_crop_with_margin_empty_box():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_with_margin(frame, (10, 10, 10, 10))
    assert crop.size == 0
