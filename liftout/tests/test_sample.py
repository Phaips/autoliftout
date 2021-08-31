#!/usr/bin/env python3


import os

import pytest
from liftout.fibsem.sample import *

from liftout import tests
try:
    from autoscript_sdb_microscope_client.structures import *
except:
    from liftout.tests.mock_autoscript_sdb_microscope_client import *


@pytest.fixture
def tmp_data_path():

    path = os.path.dirname(tests.__file__)
    sample_file = os.path.join(os.path.dirname(tests.__file__), "sample.yaml")
    if os.path.exists(sample_file):
        os.remove(sample_file)


    # TODO: create some fake data and directories
    yield path


# TODO: fix this
def test_create_new_sample(tmp_data_path):

    sample_no = 1
    sample = Sample(data_path=tmp_data_path, sample_no=sample_no)

    assert sample.data_path == os.path.join(tmp_data_path) #TODO: make this os agnostic
    assert sample.sample_no == sample_no


def test_create_new_yaml_file(tmp_data_path):

    sample = Sample(tmp_data_path, 1)  # TODO: make fixture

    yaml_file = sample.setup_yaml_file()

    assert set(yaml_file.keys()) == set(["timestamp", "data_path", "sample"])
    assert yaml_file["sample"] == {}


def test_save_sample_data(tmp_data_path):

    pass

def test_save_sample_data_with_existing_data(tmp_data_path):

    pass

def test_load_sample_data_from_file(tmp_data_path):

    sample = Sample(tmp_data_path, 1)

    pass



def test_get_initial_sample_data(tmp_data_path):

    pass


def test_load_sample_data_from_file_fails(tmp_data_path):

    pass


def test_save_current_position(tmp_data_path):

    stage_position = StagePosition(1, 2, 3, 4, 5)
    needle_position = ManipulatorPosition(1, 2, 3, 4)

    sample = Sample(tmp_data_path, 1)
    sample.save_current_position(stage_position=stage_position, needle_position=needle_position)

    assert sample.last_stage_position == stage_position
    assert sample.last_needle_position == needle_position


# #https://mq-software-carpentry.github.io/python-testing/07-fixtures/

# # TODO: use tmpdir
# # def test_remove_all_tmp_sample_data_files(tmp_data_path):
# #     # needs to be run first, and last
# #     sample_file = os.path.join(tmp_data_path, "sample.yaml")
# #     if os.path.exists(sample_file):
# #         os.remove(sample_file)
    
# #     assert os.path.exists(sample_file) == False