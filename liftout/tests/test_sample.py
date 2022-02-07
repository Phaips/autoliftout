#!/usr/bin/env python3


import os

import pytest
from liftout.fibsem.sampleposition import *
from liftout.tests import data as test_data_path

try:
    from autoscript_sdb_microscope_client.structures import *
except:
    from liftout.tests.mock_autoscript_sdb_microscope_client import *


def remove_tmp_file(path):
    sample_file = os.path.join(path, "sample.yaml")
    if os.path.exists(sample_file):
        os.remove(sample_file)


@pytest.fixture
def tmp_data_path():

    path = os.path.dirname(test_data_path.__file__)

    # TODO: create some fake data and directories
    yield path


@pytest.fixture()
def saved_sample(tmp_data_path):

    sample_no = 1
    x, y, z, r, t = 1, 2, 3, 4, 5
    lamella_coordinates = StagePosition(x=x, y=y, z=z, r=r, t=t)

    sample = SamplePosition(tmp_data_path, sample_no)
    sample.lamella_coordinates = lamella_coordinates
    sample.save_data()

    yield sample


# TODO: fix this
def test_create_new_sample(tmp_data_path):

    sample_no = 1
    sample = SamplePosition(data_path=tmp_data_path, sample_no=sample_no)

    assert sample.data_path == os.path.join(
        tmp_data_path
    )  # TODO: make this os agnostic
    assert sample.sample_no == sample_no


def test_create_new_yaml_file(tmp_data_path):

    sample = SamplePosition(tmp_data_path, 1)  # TODO: make fixture

    yaml_file = sample.setup_yaml_file()

    assert set(yaml_file.keys()) == {"timestamp", "data_path", "sample"}
    assert yaml_file["sample"] == {}


def test_save_sample_data(tmp_data_path):

    x, y, z, r, t = 1, 2, 3, 4, 5
    lamella_coordinates = StagePosition(x=x, y=y, z=z, r=r, t=t)

    sample = SamplePosition(tmp_data_path, 1)
    sample.lamella_coordinates = lamella_coordinates
    sample.save_data()
    sample.load_data_from_file()

    assert sample.lamella_coordinates.x == lamella_coordinates.x
    assert sample.lamella_coordinates.y == lamella_coordinates.y
    assert sample.lamella_coordinates.z == lamella_coordinates.z
    assert sample.lamella_coordinates.r == lamella_coordinates.r
    assert sample.lamella_coordinates.t == lamella_coordinates.t


def test_save_sample_data_with_existing_data(tmp_data_path, saved_sample):

    sample_no = 1

    x, y, z, r, t = 6, 7, 8, 9, 10
    new_lamella_coordinates = StagePosition(x=x, y=y, z=z, r=r, t=t)
    saved_sample.lamella_coordinates = new_lamella_coordinates
    saved_sample.save_data()

    del saved_sample
    new_sample = SamplePosition(tmp_data_path, sample_no)
    new_sample.load_data_from_file()

    assert new_sample.lamella_coordinates.x == new_lamella_coordinates.x
    assert new_sample.lamella_coordinates.y == new_lamella_coordinates.y
    assert new_sample.lamella_coordinates.z == new_lamella_coordinates.z
    assert new_sample.lamella_coordinates.r == new_lamella_coordinates.r
    assert new_sample.lamella_coordinates.t == new_lamella_coordinates.t

def test_load_sample_data_from_file(tmp_data_path, saved_sample):

    new_sample = SamplePosition(tmp_data_path, 1)
    new_sample.load_data_from_file()

    assert saved_sample.lamella_coordinates.x == new_sample.lamella_coordinates.x
    assert saved_sample.lamella_coordinates.y == new_sample.lamella_coordinates.y
    assert saved_sample.lamella_coordinates.z == new_sample.lamella_coordinates.z
    assert saved_sample.lamella_coordinates.r == new_sample.lamella_coordinates.r
    assert saved_sample.lamella_coordinates.t == new_sample.lamella_coordinates.t

def test_load_sample_data_from_file_fails():

    sample = SamplePosition(".", 1)
    with pytest.raises(FileNotFoundError):
        sample.load_data_from_file()


def test_load_sample_data_from_file_with_no_sample_fails(saved_sample):

    saved_sample.sample_no = 2
    with pytest.raises(KeyError):
        saved_sample.load_data_from_file()


# https://stackoverflow.com/questions/23337471/how-to-properly-assert-that-an-exception-gets-raised-in-pytest

def test_get_sample_data(tmp_data_path, saved_sample):

    sample = SamplePosition(tmp_data_path, 1)
    sample.load_data_from_file()

    lamella_coord, landing_coord, lamella_imgs, landing_imgs = sample.get_sample_data()

    example_stage_position = StagePosition(1, 2, 3, 4, 5)
    empty_stage_position = StagePosition()

    assert lamella_coord.x == example_stage_position.x
    assert lamella_coord.y == example_stage_position.y
    assert lamella_coord.z == example_stage_position.z
    assert lamella_coord.r == example_stage_position.r
    assert lamella_coord.t == example_stage_position.t
    assert len(lamella_imgs) == 4
    assert landing_coord.x == empty_stage_position.x
    assert landing_coord.y == empty_stage_position.y
    assert landing_coord.z == empty_stage_position.z
    assert landing_coord.r == empty_stage_position.r
    assert landing_coord.t == empty_stage_position.t
    assert len(landing_imgs) == 4


def test_save_current_position(tmp_data_path):

    stage_position = StagePosition(1, 2, 3, 4, 5)
    needle_position = ManipulatorPosition(1, 2, 3, 4)

    sample = SamplePosition(tmp_data_path, 1)
    sample.save_current_position(
        stage_position=stage_position, needle_position=needle_position
    )

    assert sample.last_stage_position == stage_position
    assert sample.last_needle_position == needle_position


# #https://mq-software-carpentry.github.io/python-testing/07-fixtures/

# # TODO: use tmpdir
def test_remove_all_tmp_sample_data_files(tmp_data_path):
    # needs to be run first, and last
    sample_file = os.path.join(tmp_data_path, "sample.yaml")
    if os.path.exists(sample_file):
        os.remove(sample_file)

    assert os.path.exists(sample_file) == False
