
from enum import Enum

import yaml
import os
import datetime
import time
# from liftout.gui.main import AutoLiftoutStatus
from dataclasses import dataclass
import uuid

try:
    from autoscript_sdb_microscope_client.structures import *
except:
    from liftout.tests.mock_autoscript_sdb_microscope_client import *


# TODO: move this to a separate file (and the one from main too)
class AutoLiftoutStatus(Enum):
    Initialisation = -1
    Setup = 0
    Milling = 1
    Liftout = 2
    Landing = 3
    Reset = 4
    Thinning = 5
    Finished = 6


@dataclass
class MicroscopeState:
    timestamp: float = None
    absolute_position: StagePosition = StagePosition()
    eb_working_distance: float = None
    ib_working_distance: float = None
    eb_beam_current: float = None
    ib_beam_current: float = None
    eucentric_calibration: bool = False  # whether eucentricity has been recently verified
    last_completed_stage: AutoLiftoutStatus = None

@dataclass
class ReferenceImages:
    low_res_eb: AdornedImage
    high_res_eb: AdornedImage
    low_res_ib: AdornedImage
    high_res_ib: AdornedImage

    def __iter__(self):

        yield self.low_res_eb, self.high_res_eb, self.low_res_ib, self.high_res_ib


class SamplePosition:
    def __init__(self, data_path, sample_no):
        self.landing_coordinates = StagePosition()
        self.lamella_coordinates = StagePosition()

        self.landing_ref_images = list()  # TODO: change to ReferenceImages
        self.lamella_ref_images = list()  # TODO: change to ReferenceImages

        self.data_path = os.path.join(data_path)
        self.created_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')
        self.sample_no = sample_no
        self.landing_selected: bool = False

        self.sample_id = uuid.uuid4()
        self.microscope_state: MicroscopeState = MicroscopeState()

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
                "created": datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S'),
                "data_path": self.data_path,
                "sample": {},
            }
        return sample_yaml

    def save_data(self):
        """Save the lamella and landing coordinates, and reference to data path"""

        # check if yaml file already exists for this sample..
        sample_yaml = self.setup_yaml_file()

        # create dir for reference images
        os.makedirs(os.path.join(self.data_path, str(self.sample_id)), exist_ok=True)

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

        microscope_state_dict = {
            "timestamp": self.microscope_state.timestamp,
            "absolute_position": {
                "x": self.microscope_state.absolute_position.x,
                "y": self.microscope_state.absolute_position.y,
                "z": self.microscope_state.absolute_position.z,
                "r": self.microscope_state.absolute_position.r,
                "t": self.microscope_state.absolute_position.t,
                "coordinate_system": self.microscope_state.absolute_position.coordinate_system
            },
            "eb_working_distance": self.microscope_state.eb_working_distance,
            "ib_working_distance": self.microscope_state.ib_working_distance,
            "eb_beam_current": self.microscope_state.eb_beam_current,
            "ib_beam_current": self.microscope_state.ib_beam_current,
            "eucentric_calibration": self.microscope_state.eucentric_calibration,
            "last_completed_stage": str(self.microscope_state.last_completed_stage),
        }

        # save stage position to yml file
        save_dict = {

            "created": self.created_timestamp,
            "updated": datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S'),
            "sample_no": self.sample_no,
            "sample_id": str(self.sample_id),
            "landing_selected": self.landing_selected,
            "lamella_coordinates": lamella_coordinates_dict,
            "landing_coordinates": landing_coordinates_dict,
            "microscope_state": microscope_state_dict
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

        self.created_timestamp = sample_dict["created"]
        self.updated_timestamp = sample_dict["updated"]
        self.sample_id = sample_dict["sample_id"]
        self.landing_selected = sample_dict["landing_selected"]

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

        # load micrscope state
        self.microscope_state = MicroscopeState(
            timestamp=sample_dict["microscope_state"]["timestamp"],
            absolute_position=StagePosition(
                x=sample_dict["microscope_state"]["absolute_position"]["x"],
                y=sample_dict["microscope_state"]["absolute_position"]["y"],
                z=sample_dict["microscope_state"]["absolute_position"]["z"],
                r=sample_dict["microscope_state"]["absolute_position"]["r"],
                t=sample_dict["microscope_state"]["absolute_position"]["t"],
                coordinate_system=sample_dict["microscope_state"]["absolute_position"]["coordinate_system"]
            ),
            eb_working_distance=sample_dict["microscope_state"]["eb_working_distance"],
            ib_working_distance=sample_dict["microscope_state"]["ib_working_distance"],
            eb_beam_current=sample_dict["microscope_state"]["eb_beam_current"],
            ib_beam_current=sample_dict["microscope_state"]["ib_beam_current"],
            eucentric_calibration=sample_dict["microscope_state"]["eucentric_calibration"],
            last_completed_stage=sample_dict["microscope_state"]["last_completed_stage"]
        )

        # load images from disk
        ref_landing_lowres_eb = os.path.join(self.data_path, str(self.sample_id), "ref_landing_low_res_eb.tif")
        ref_landing_highres_eb = os.path.join(self.data_path, str(self.sample_id), "ref_landing_high_res_eb.tif")
        ref_landing_lowres_ib = os.path.join(self.data_path, str(self.sample_id), "ref_landing_low_res_ib.tif")
        ref_landing_highres_ib = os.path.join(self.data_path, str(self.sample_id), "ref_landing_high_res_ib.tif")
        ref_lamella_lowres_eb = os.path.join(self.data_path, str(self.sample_id), "ref_lamella_low_res_eb.tif")
        ref_lamella_highres_eb = os.path.join(self.data_path, str(self.sample_id), "ref_lamella_high_res_eb.tif")
        ref_lamella_lowres_ib = os.path.join(self.data_path, str(self.sample_id), "ref_lamella_low_res_ib.tif")
        ref_lamella_highres_ib = os.path.join(self.data_path, str(self.sample_id), "ref_lamella_high_res_ib.tif")

        # load the adorned images and format
        if self.landing_selected:
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

    def save_current_state(self):
        """Save the current microscope state"""
        return NotImplemented

    def load_sample_state(self):
        """Load the current microscope state"""
        return NotImplemented
