#!/usr/bin/env python3


import os

import pytest
from liftout.fibsem.sample import *

try:
    from autoscript_sdb_microscope_client.structures import *
except:
    from liftout.tests.mock_autoscript_sdb_microscope_client import *


@pytest.fixture
def tmp_data_path():

    # TODO: create some fake data and directories
    yield os.path.dirname(os.getcwd())


def test_create_new_sample(tmp_data_path):

    sample_no = 1
    sample = Sample(data_path=tmp_data_path, sample_no=sample_no)

    assert sample.data_path == tmp_data_path + "/"
    assert sample.sample_no == sample_no


def test_create_new_yaml_file(tmp_data_path):

    sample = Sample(tmp_data_path, 1)  # TODO: make fixture

    yaml_file = sample.setup_yaml_file()

    assert set(yaml_file.keys()) == set(["timestamp", "data_path", "sample"])
    assert yaml_file["sample"] == {}


def test_save_sample_data(tmp_data_path):

    pass


def test_load_sample_data_from_file(tmp_data_path):

    pass


def test_get_initial_sample_data(tmp_data_path):

    pass


def test_load_sample_data_from_file_fails(tmp_data_path):

    pass


def test_save_current_position(tmp_data_path):

    pass
