#!/usr/bin/env python3


# https://changhsinlee.com/pytest-mock/
# https://docs.pytest.org/en/6.2.x/monkeypatch.html

import numpy as np
import pytest
from liftout.model import models
from liftout.detection import detection, utils
import glob
import os
import pandas as pd

@pytest.fixture
def weights_file():
    # yield r"C:\Users\Admin\Github\autoliftout\liftout\model\models\fresh_full_n10.pt"
    # yield "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/liftout/model/models/fresh_full_n10.pt"
    yield os.path.dirname(models.__file__) + "/fresh_full_n10.pt"


@pytest.fixture
def detector(weights_file):

    yield detection.Detector(weights_file=weights_file)

@pytest.fixture
def test_image_fname():
    """ Return the filename of a test image """
    #TODO:  there has to be a better way to do this
    filenames = glob.glob("./**/*test_image.tif", recursive=True)
    # fname = os.getcwd() + "liftout/tests/test_image.tif"

    fname = filenames[0]
    yield fname


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
    # TODO: create the AdornedImage version of this and test



def test_parse_metadata_df(test_image_fname):

    df_metadata = utils.parse_metadata(test_image_fname)

    # assert len(df_metadata.columns) == 50 # TODO: actually implement this properly
    assert type(df_metadata) == pd.DataFrame


def test_match_filenames_from_path():
    pass

def test_extract_img_for_labelling():

    pass