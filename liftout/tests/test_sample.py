#!/usr/bin/env python3


import os

import pytest
from liftout.fibsem.sample import *
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

    sample = Sample(tmp_data_path, sample_no)
    sample.lamella_coordinates = lamella_coordinates
    sample.save_data()

    yield sample


# TODO: fix this
def test_create_new_sample(tmp_data_path):

    sample_no = 1
    sample = Sample(data_path=tmp_data_path, sample_no=sample_no)

    assert sample.data_path == os.path.join(
        tmp_data_path
    )  # TODO: make this os agnostic
    assert sample.sample_no == sample_no


def test_create_new_yaml_file(tmp_data_path):

    sample = Sample(tmp_data_path, 1)  # TODO: make fixture

    yaml_file = sample.setup_yaml_file()

    assert set(yaml_file.keys()) == set(["timestamp", "data_path", "sample"])
    assert yaml_file["sample"] == {}


def test_save_sample_data(tmp_data_path):

    x, y, z, r, t = 1, 2, 3, 4, 5
    lamella_coordinates = StagePosition(x=x, y=y, z=z, r=r, t=t)

    sample = Sample(tmp_data_path, 1)
    sample.lamella_coordinates = lamella_coordinates
    sample.save_data()
    sample.load_data_from_file()

    assert sample.lamella_coordinates == lamella_coordinates


def test_save_sample_data_with_existing_data(tmp_data_path, saved_sample):

    sample_no = 1

    x, y, z, r, t = 6, 7, 8, 9, 10
    new_lamella_coordinates = StagePosition(x=x, y=y, z=z, r=r, t=t)
    saved_sample.lamella_coordinates = new_lamella_coordinates
    saved_sample.save_data()

    del saved_sample
    new_sample = Sample(tmp_data_path, sample_no)
    new_sample.load_data_from_file()

    assert new_sample.lamella_coordinates == new_lamella_coordinates


def test_load_sample_data_from_file(tmp_data_path, saved_sample):

    new_sample = Sample(tmp_data_path, 1)
    new_sample.load_data_from_file()

    assert saved_sample.lamella_coordinates == new_sample.lamella_coordinates


def test_load_sample_data_from_file_fails():

    sample = Sample(".", 1)
    with pytest.raises(FileNotFoundError):
        sample.load_data_from_file()


def test_load_sample_data_from_file_with_no_sample_fails(saved_sample):

    saved_sample.sample_no = 2
    with pytest.raises(KeyError):
        saved_sample.load_data_from_file()


# https://stackoverflow.com/questions/23337471/how-to-properly-assert-that-an-exception-gets-raised-in-pytest


def test_initial_get_sample_data_returns_none(tmp_data_path):
    remove_tmp_file(tmp_data_path)
    sample = Sample(tmp_data_path, 1)

    lam_coord, land_coord, lam_imgs, landing_imgs = sample.get_sample_data()

    assert lam_coord.x == StagePosition().x
    assert land_coord.x == StagePosition().x
    assert not lam_imgs  # list is empty
    assert not landing_imgs  # list is empty


def test_get_sample_data(tmp_data_path, saved_sample):

    sample = Sample(tmp_data_path, 1)
    sample.load_data_from_file()

    lamella_coord, landing_coord, lamella_imgs, landing_imgs = sample.get_sample_data()

    assert lamella_coord == StagePosition(1, 2, 3, 4, 5)
    assert landing_coord == StagePosition()
    assert len(lamella_imgs) == 4
    assert len(landing_imgs) == 4


def test_save_current_position(tmp_data_path):

    stage_position = StagePosition(1, 2, 3, 4, 5)
    needle_position = ManipulatorPosition(1, 2, 3, 4)

    sample = Sample(tmp_data_path, 1)
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
