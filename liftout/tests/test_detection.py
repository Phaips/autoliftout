#!/usr/bin/env python3


# https://changhsinlee.com/pytest-mock/
# https://docs.pytest.org/en/6.2.x/monkeypatch.html

import numpy as np
import pytest
from liftout.detection import detection, utils


@pytest.fixture
def weights_file():
    # yield r"C:\Users\Admin\Github\autoliftout\liftout\model\models\fresh_full_n10.pt"
    yield "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/liftout/model/models/fresh_full_n10.pt"


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
        "thin_lamella_bottom_to_centre",
    ]


def test_extract_class_pixels_is_null():

    # null mask
    mask = np.zeros(shape=(256, 384, 3))
    color = (255, 0, 0)

    class_mask, idx = detection.extract_class_pixels(mask, color)

    assert np.all(class_mask == (0, 0, 0))
    assert len(idx[0]) == 0  # empty tuple


def test_extract_class_pixels():

    # red mask
    mask = np.zeros(shape=(256, 384, 3))
    mask[:, :, :] = (255, 0, 0)
    color = (255, 0, 0)

    class_mask, idx = detection.extract_class_pixels(mask, color)

    assert np.all(class_mask == (255, 0, 0))
    assert len(idx[0]) == class_mask.shape[0] * class_mask.shape[1]  # full tuple





def test_detect_centre_point():

    # red mask
    mask = np.zeros(shape=(256, 384, 3))
    mask[:, :, :] = (255, 0, 0)
    color = (255, 0, 0)

    centre_px = detection.detect_centre_point(mask, color)

    # check that centre detection is within 1px of image centre
    assert np.isclose(centre_px[0], mask.shape[1] // 2, 1.0)
    assert np.isclose(centre_px[1], mask.shape[0] // 2, 1.0)


def test_detect_centre_point_is_zero():

    # zero mask
    mask = np.zeros(shape=(256, 384, 3))
    color = (255, 0, 0)

    centre_px = detection.detect_centre_point(mask, color)

    # check that centre px is (0, 0)
    assert centre_px == (0, 0)


def test_detect_right_edge():

    # red mask
    mask = np.zeros(shape=(256, 384, 3))
    mask[:, :, :] = (255, 0, 0)
    color = (255, 0, 0)

    right_edge_px = detection.detect_right_edge(mask, color)
    left_edge_px = detection.detect_right_edge(mask, color, left=True)

    assert right_edge_px[1] == mask.shape[1] - 1
    assert left_edge_px[1] == 0


def test_detect_bounding_box():

    mask = np.zeros(shape=(256, 384, 3))
    mask[:, :, :] = (255, 0, 0)
    color = (255, 0, 0)

    bbox = detection.detect_bounding_box(mask=mask, color=color)

    assert bbox == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
