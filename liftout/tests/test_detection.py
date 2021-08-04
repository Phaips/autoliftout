#!/usr/bin/env python3


# https://changhsinlee.com/pytest-mock/
# https://docs.pytest.org/en/6.2.x/monkeypatch.html

import pytest
from liftout.detection import detection, utils
import segmentation_models_pytorch as smp
import numpy as np

@pytest.fixture
def weights_file():
    yield r"C:\Users\Admin\Github\autoliftout\liftout\model\models\fresh_full_n10.pt"

@pytest.fixture
def detector(weights_file):

    yield detection.Detector(weights_file=weights_file)


def test_Detector_init(detector):

    assert type(detector) == detection.Detector
    assert type(detector.detection_model) == detection.DetectionModel.DetectionModel
    assert detector.supported_shift_types == [
            "needle_tip_to_lamella_centre",
            "lamella_centre_to_image_centre",
            "lamella_edge_to_landing_post",
            "needle_tip_to_image_centre",
            "thin_lamella_top_to_centre",
            "thin_lamella_bottom_to_centre"
        ]


def test_image():

    img = np.zeros(shape=(254, 364, 3))

    assert img.shape == (254, 364, 3)


def test_extract_class_pixels_is_null():

    # null mask
    mask = np.zeros(shape=(256, 384, 3))
    color = (255, 0, 0)

    class_mask, idx = detection.extract_class_pixels(mask, color)

    assert np.all(class_mask == (0, 0, 0))
    assert len(idx[0]) == 0 # empty tuple

def test_extract_class_pixels():

    # red mask
    mask = np.zeros(shape=(256, 384, 3))
    mask[:, :, :] = (255, 0, 0)
    color = (255, 0, 0)

    class_mask, idx = detection.extract_class_pixels(mask, color)

    assert np.all(class_mask == (255, 0, 0))
    assert len(idx[0]) == class_mask.shape # empty tuple


# def test_scale_invariant_coordinates():
#
#     mask = np.ones(256, 384, 3)
#     px = mask.shape[0]//2, mask.shape[1]//2
#
#     scaled_px =



