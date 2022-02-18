#!/usr/bin/env python3


# https://changhsinlee.com/pytest-mock/
# https://docs.pytest.org/en/6.2.x/monkeypatch.html

import os

import numpy as np
import pandas as pd
import pytest
from liftout.detection import utils


@pytest.fixture
def test_image_fname():
    """ Return the filename of a test image """

    # fname = filenames[0]
    from liftout.tests import data
    yield os.path.join(os.path.dirname(data.__file__), "test_image.tif")


def test_scale_invariant_coordinates():

    mask = np.ones((256, 384, 3))
    px = mask.shape[0] // 2, mask.shape[1] // 2

    scaled_px = utils.scale_invariant_coordinates(px, mask)

    assert scaled_px == (0.5, 0.5)


def test_scale_invariant_coordinates_is_zeros():

    mask = np.ones((256, 384, 3))
    px = (0, 0)

    scaled_px = utils.scale_invariant_coordinates(px, mask)

    assert scaled_px == (0.0, 0.0)


def test_load_image_from_file(test_image_fname):

    img = utils.load_image_from_file(fname=test_image_fname)
    
    assert type(img) == np.ndarray
    assert img.shape == (1024, 1536)



def test_parse_metadata_df(test_image_fname):

    df_metadata = utils.parse_metadata(test_image_fname)

    # assert len(df_metadata.columns) == 50
    assert type(df_metadata) == pd.DataFrame
