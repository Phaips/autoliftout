import glob
from enum import Enum
from pprint import pprint
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np

try:
    from autoscript_sdb_microscope_client.structures import *
except:
    from liftout.tests.mock_autoscript_sdb_microscope_client import *


class Sample:
    def __init__(self, data_path, sample_no):
        self.landing_coordinates = StagePosition()
        self.lamella_coordinates = StagePosition()
        self.milling_coordinates = StagePosition()
        self.jcut_coordinates = StagePosition()
        self.liftout_coordinates = StagePosition()
        self.park_position = ManipulatorPosition()

        self.last_stage_position = StagePosition()
        self.last_needle_position = ManipulatorPosition()

        self.landing_ref_images = list()
        self.lamella_ref_images = list()
        self.status = NotImplemented

        self.data_path = os.path.join(data_path)
        self.timestamp = os.path.basename(self.data_path)
        self.sample_no = sample_no

    def setup_yaml_file(self):
        # check if yaml file already exists for this timestamp..
        yaml_file = os.path.join(self.data_path, "sample.yaml")

        if os.path.exists(yaml_file):
            # read and open existing yaml file
            with open(yaml_file, "r") as f:
                sample_yaml = yaml.safe_load(f)

        else:
            # create new yaml file
            sample_yaml = {
                "timestamp": self.timestamp,
                "data_path": self.data_path,
                "sample": {},
            }
        return sample_yaml

    def save_data(self):
        """Save the lamella and landing coordinates, and reference to data path"""

        # check if yaml file already exists for this timestamp..
        sample_yaml = self.setup_yaml_file()

        # format coordinate data for saving
        lamella_coordinates_dict = {
            "x": self.lamella_coordinates.x,
            "y": self.lamella_coordinates.y,
            "z": self.lamella_coordinates.z,
            "r": self.lamella_coordinates.r,
            "t": self.lamella_coordinates.t,
            "coordinate_system": self.lamella_coordinates.coordinate_system
        }

        landing_coordinates_dict = {
            "x": self.landing_coordinates.x,
            "y": self.landing_coordinates.y,
            "z": self.landing_coordinates.z,
            "r": self.landing_coordinates.r,
            "t": self.landing_coordinates.t,
            "coordinate_system": self.landing_coordinates.coordinate_system

        }

        milling_coordinates_dict = {
            "x": self.milling_coordinates.x,
            "y": self.milling_coordinates.y,
            "z": self.milling_coordinates.z,
            "r": self.milling_coordinates.r,
            "t": self.milling_coordinates.t,
            "coordinate_system": self.milling_coordinates.coordinate_system

        }

        jcut_coordinates_dict = {
            "x": self.jcut_coordinates.x,
            "y": self.jcut_coordinates.y,
            "z": self.jcut_coordinates.z,
            "r": self.jcut_coordinates.r,
            "t": self.jcut_coordinates.t,
            "coordinate_system": self.jcut_coordinates.coordinate_system

        }

        liftout_coordinates_dict = {
            "x": self.liftout_coordinates.x,
            "y": self.liftout_coordinates.y,
            "z": self.liftout_coordinates.z,
            "r": self.liftout_coordinates.r,
            "t": self.liftout_coordinates.t,
            "coordinate_system": self.liftout_coordinates.coordinate_system

        }

        park_position_dict = {
            "x": self.park_position.x,
            "y": self.park_position.y,
            "z": self.park_position.z,
            "r": self.park_position.r,

        }

        last_stage_position_dict = {
            "x": self.last_stage_position.x,
            "y": self.last_stage_position.y,
            "z": self.last_stage_position.z,
            "r": self.last_stage_position.r,
            "t": self.last_stage_position.t,
            "coordinate_system": self.last_stage_position.coordinate_system

        }

        last_needle_position_dict = {
            "x": self.last_needle_position.x,
            "y": self.last_needle_position.y,
            "z": self.last_needle_position.z,
            "r": self.last_needle_position.r,
        }

        # save stage position to yml file
        save_dict = {
            "timestamp": self.timestamp,
            "sample_no": self.sample_no,
            "lamella_coordinates": lamella_coordinates_dict,
            "landing_coordinates": landing_coordinates_dict,
            "milling_coordinates": milling_coordinates_dict,
            "jcut_coordinates": jcut_coordinates_dict,
            "liftout_coordinates": liftout_coordinates_dict,
            "park_position": park_position_dict,
            "last_stage_position": last_stage_position_dict,
            "last_needle_position": last_needle_position_dict
            # "data_path": self.data_path
        }

        # format dictionary
        sample_yaml["sample"][self.sample_no] = save_dict

        # should we save the images separately? or just use previously saved?
        with open(os.path.join(self.data_path, "sample.yaml"), "w") as outfile:
            yaml.dump(sample_yaml, outfile)

    def load_data_from_file(self, fname=None):

        if fname is None:
            fname = os.path.join(self.data_path, "sample.yaml")

        # load yaml file
        with open(fname, "r") as f:
            sample_yaml = yaml.safe_load(f)

        sample_dict = sample_yaml["sample"][self.sample_no]

        # load stage positions from yaml
        self.lamella_coordinates = StagePosition(
            x=sample_dict["lamella_coordinates"]["x"],
            y=sample_dict["lamella_coordinates"]["y"],
            z=sample_dict["lamella_coordinates"]["z"],
            r=sample_dict["lamella_coordinates"]["r"],
            t=sample_dict["lamella_coordinates"]["t"],
            coordinate_system=sample_dict["lamella_coordinates"]["coordinate_system"]
        )

        self.landing_coordinates = StagePosition(
            x=sample_dict["landing_coordinates"]["x"],
            y=sample_dict["landing_coordinates"]["y"],
            z=sample_dict["landing_coordinates"]["z"],
            r=sample_dict["landing_coordinates"]["r"],
            t=sample_dict["landing_coordinates"]["t"],
            coordinate_system=sample_dict["landing_coordinates"]["coordinate_system"]
        )

        self.milling_coordinates = StagePosition(
            x=sample_dict["milling_coordinates"]["x"],
            y=sample_dict["milling_coordinates"]["y"],
            z=sample_dict["milling_coordinates"]["z"],
            r=sample_dict["milling_coordinates"]["r"],
            t=sample_dict["milling_coordinates"]["t"],
            coordinate_system=sample_dict["milling_coordinates"]["coordinate_system"]

        )

        self.jcut_coordinates = StagePosition(
            x=sample_dict["jcut_coordinates"]["x"],
            y=sample_dict["jcut_coordinates"]["y"],
            z=sample_dict["jcut_coordinates"]["z"],
            r=sample_dict["jcut_coordinates"]["r"],
            t=sample_dict["jcut_coordinates"]["t"],
            coordinate_system=sample_dict["jcut_coordinates"]["coordinate_system"]

        )

        self.liftout_coordinates = StagePosition(
            x=sample_dict["liftout_coordinates"]["x"],
            y=sample_dict["liftout_coordinates"]["y"],
            z=sample_dict["liftout_coordinates"]["z"],
            r=sample_dict["liftout_coordinates"]["r"],
            t=sample_dict["liftout_coordinates"]["t"],
            coordinate_system=sample_dict["liftout_coordinates"]["coordinate_system"]
        )

        self.park_position = ManipulatorPosition(
            x=sample_dict["park_position"]["x"],
            y=sample_dict["park_position"]["y"],
            z=sample_dict["park_position"]["z"],
            r=sample_dict["park_position"]["r"],
        )

        # load images from disk
        sample_no = sample_dict["sample_no"]
        ref_landing_lowres_eb = os.path.join(
            self.data_path, "img", f"{sample_no:02d}_ref_landing_low_res_eb.tif"
        )
        ref_landing_highres_eb = os.path.join(
            self.data_path, "img", f"{sample_no:02d}_ref_landing_high_res_eb.tif"
        )
        ref_landing_lowres_ib = os.path.join(
            self.data_path, "img", f"{sample_no:02d}_ref_landing_low_res_ib.tif"
        )
        ref_landing_highres_ib = os.path.join(
            self.data_path, "img", f"{sample_no:02d}_ref_landing_high_res_ib.tif"
        )
        ref_lamella_lowres_eb = os.path.join(
            self.data_path, "img", f"{sample_no:02d}_ref_lamella_low_res_eb.tif"
        )
        ref_lamella_highres_eb = os.path.join(
            self.data_path, "img", f"{sample_no:02d}_ref_lamella_high_res_eb.tif"
        )
        ref_lamella_lowres_ib = os.path.join(
            self.data_path, "img", f"{sample_no:02d}_ref_lamella_low_res_ib.tif"
        )
        ref_lamella_highres_ib = os.path.join(
            self.data_path, "img", f"{sample_no:02d}_ref_lamella_high_res_ib.tif"
        )
        # ref_landing_lowres_eb = os.path.join(self.data_path, "img", f"{sample_no:02d}_ref_landing_low_res_eb.tif")
        # ref_landing_highres_eb = os.path.join(self.data_path, "img", f"{sample_no:02d}_ref_landing_high_res_eb.tif")
        # ref_landing_lowres_ib = os.path.join(self.data_path, "img", f"{sample_no:02d}_ref_landing_low_res_eb_ib.tif")
        # ref_landing_highres_ib = os.path.join(self.data_path, "img", f"{sample_no:02d}_ref_landing_high_res_eb_ib.tif")
        # ref_lamella_lowres_eb = os.path.join(self.data_path, "img", f"{sample_no:02d}_ref_lamella_low_res_eb.tif")
        # ref_lamella_highres_eb = os.path.join(self.data_path, "img", f"{sample_no:02d}_ref_lamella_high_res_eb.tif")
        # ref_lamella_lowres_ib = os.path.join(self.data_path, "img", f"{sample_no:02d}_ref_lamella_low_res_eb_ib.tif")
        # ref_lamella_highres_ib = os.path.join(self.data_path, "img", f"{sample_no:02d}_ref_lamella_high_res_eb_ib.tif")

        # load the adorned images and format
        for fname in [
            ref_landing_lowres_eb,
            ref_landing_highres_eb,
            ref_landing_lowres_ib,
            ref_landing_highres_ib,
        ]:
            img = AdornedImage.load(fname)

            self.landing_ref_images.append(img)

        for fname in [
            ref_lamella_lowres_eb,
            ref_lamella_highres_eb,
            ref_lamella_lowres_ib,
            ref_lamella_highres_ib,
        ]:
            img = AdornedImage.load(fname)

            self.lamella_ref_images.append(img)

    def get_sample_data(self):
        """Return the sample data formatted for liftout from the specificed data_path."""

        if not self.lamella_ref_images or not self.landing_ref_images:
            self.load_data_from_file()  # TODO: probably only need to load the images.. separate functionality for load

        return (
            self.lamella_coordinates,
            self.landing_coordinates,
            self.lamella_ref_images,
            self.landing_ref_images,
        )

    def save_current_position(
        self,
        stage_position: StagePosition,
        needle_position: ManipulatorPosition,
    ):
        """Save the current sample stage and needle positions and state for recovery"""

        # save the current sample positions
        self.last_stage_position = stage_position
        self.last_needle_position = needle_position
        self.last_status = self.status

        # save to disk
        self.save_data()
