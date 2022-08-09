import os
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import petname
import yaml
from autoscript_sdb_microscope_client.structures import (AdornedImage,
                                                         StagePosition)
from fibsem import utils as fibsem_utils
from fibsem.structures import MicroscopeState, ReferenceImages, stage_position_from_dict

from liftout import utils

# move this up into liftout..


class AutoLiftoutStage(Enum):
    Initialisation = -1
    Setup = 0
    MillTrench = 1
    MillJCut = 2
    Liftout = 3
    Landing = 4
    Reset = 5
    Thinning = 6
    Polishing = 7
    Finished = 8
    Failure = 99


class Sample:
    def __init__(self, path: Path = None, name: str = "default") -> None:

        self.name: str = name
        self.path: Path = utils.make_logging_directory(path=path, name=name)
        self.log_path: Path = utils.configure_logging(
            path=self.path, log_filename="logfile"
        )

        self.state = None
        self.positions: dict = {}

    def __to_dict__(self) -> dict:

        state_dict = {
            "name": self.name,
            "path": self.path,
            "log_path": self.log_path,
            "positions": [lamella.__to_dict__() for lamella in self.positions.values()],
        }

        return state_dict

    def save(self) -> None:
        """Save the sample data to yaml file"""

        with open(os.path.join(self.path, "sample.yaml"), "w") as f:
            yaml.dump(self.__to_dict__(), f, indent=4)

    def __repr__(self) -> str:

        return f"""Sample: 
        Path: {self.path}
        State: {self.state}
        Lamella: {len(self.positions)}
        """


def load_sample(fname: str) -> Sample:
    """Load a sample from disk."""

    # read and open existing yaml file
    if os.path.exists(fname):
        with open(fname, "r") as f:
            sample_dict = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"No file with name {fname} found.")

    # create sample
    path = os.path.dirname(sample_dict["path"])
    name = sample_dict["name"]
    sample = Sample(path=path, name=name)

    # load lamella from dict
    for lamella_dict in sample_dict["positions"]:
        lamella = lamella_from_dict(path=sample.path, lamella_dict=lamella_dict)
        sample.positions[lamella._number] = lamella

    return sample


class Lamella:
    def __init__(self, path: Path, number: int = 0, _petname: str = None) -> None:

        self._number: int = number
        self._id = str(uuid.uuid4())
        if _petname is None:
            self._petname = f"{self._number:02d}-{petname.generate(2)}"
        else:
            self._petname = _petname

        # filesystem
        self.base_path = path
        self.path = os.path.join(self.base_path, self._petname)
        os.makedirs(self.path, exist_ok=True)

        self.lamella_coordinates: StagePosition = StagePosition()
        self.landing_coordinates: StagePosition = StagePosition()

        self.lamella_ref_images: ReferenceImages = None
        self.landing_ref_images: ReferenceImages = None

        self.landing_selected: bool = False

        self.current_state: AutoLiftoutState = AutoLiftoutState()

        # state history
        self.history: list[AutoLiftoutState] = []

    def __repr__(self) -> str:

        return f"""
        Lamella {self._number} ({self._petname}). 
        Lamella Coordinates: {self.lamella_coordinates}, 
        Landing Coordinates: {self.landing_coordinates}, 
        Current Stage: {self.current_state.stage},
        History: {len(self.history)} stages completed ({[state.stage.name for state in self.history]}).
        """

    def __to_dict__(self):

        state_dict = {
            "id": str(self._id),
            "petname": self._petname,
            "number": self._number,
            "base_path": self.base_path,
            "path": self.path,
            "lamella_coordinates": {
                "x": self.lamella_coordinates.x,
                "y": self.lamella_coordinates.y,
                "z": self.lamella_coordinates.z,
                "r": self.lamella_coordinates.r,
                "t": self.lamella_coordinates.t,
                "coordinate_system": self.lamella_coordinates.coordinate_system,
            },
            "landing_coordinates": {
                "x": self.landing_coordinates.x,
                "y": self.landing_coordinates.y,
                "z": self.landing_coordinates.z,
                "r": self.landing_coordinates.r,
                "t": self.landing_coordinates.t,
                "coordinate_system": self.landing_coordinates.coordinate_system,
            },
            "current_state": self.current_state.__to_dict__(),
            "history": [state.__to_dict__() for state in self.history],
        }

        return state_dict

    def load_reference_image(self, fname) -> AdornedImage:
        """Load a specific reference image for this lamella from disk
        Args:
            fname: str
                the filename of the reference image to load
        Returns:
            adorned_img: AdornedImage
                the reference image loaded as an AdornedImage
        """

        adorned_img = AdornedImage.load(os.path.join(self.path, f"{fname}.tif"))

        return adorned_img


def lamella_from_dict(path: str, lamella_dict: dict) -> Lamella:

    lamella = Lamella(
        path=path, number=lamella_dict["number"], _petname=lamella_dict["petname"]
    )

    lamella._petname = lamella_dict["petname"]
    lamella._id = lamella_dict["id"]

    # load stage positions from yaml
    lamella.lamella_coordinates = stage_position_from_dict(
        lamella_dict["lamella_coordinates"]
    )
    lamella.landing_coordinates = stage_position_from_dict(
        lamella_dict["landing_coordinates"]
    )

    # load current state
    lamella.current_state = AutoLiftoutState.__from_dict__(lamella_dict["current_state"])

    # load history
    lamella.history = [
        AutoLiftoutState.__from_dict__(state_dict)
        for state_dict in lamella_dict["history"]
    ]

    return lamella


@dataclass
class AutoLiftoutState:
    stage: AutoLiftoutStage = AutoLiftoutStage.Setup
    microscope_state: MicroscopeState = MicroscopeState()
    start_timestamp: float = None  # TODO
    end_timestamp: float = None

    def __to_dict__(self) -> dict:

        state_dict = {
            "stage": self.stage.name,
            "microscope_state": self.microscope_state.__to_dict__(),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
        }

        return state_dict

    @classmethod
    def __from_dict__(state_dict: dict) -> 'AutoLiftoutState':

        autoliftout_state = AutoLiftoutState(
            stage=AutoLiftoutStage[state_dict["stage"]],
            microscope_state=MicroscopeState.__from_dict__(state_dict["microscope_state"]),
            start_timestamp=state_dict["start_timestamp"],
            end_timestamp=state_dict["end_timestamp"],
        )

        return autoliftout_state





# Sample:
#   data_path: Path
#
#   positions: [Lamella, Lamella, Lamella]

# Lamella
#   lamella_coordinates: StagePosition
#   landing_coordinates: StagePosition
#   lamella_ref_images: ReferenceImages
#   landing_ref_images: ReferenceImages
#   state: AutoLiftoutState
#       stage: AutoLiftoutStage
#       microscope_state: MicroscopeState
#           eb_settings: BeamSettings
#           ib_settings: BeamSettings
#           absolute_position: StagePosition


######################## UTIL ########################


def create_experiment(experiment_name: str, path: Path = None):

    # create unique experiment name
    exp_name = f"{experiment_name}_{fibsem_utils.current_timestamp()}"

    # create sample data struture
    sample = Sample(path=path, name=exp_name)

    # save sample to disk
    sample.save()

    return sample


def load_experiment(path: Path) -> Sample:

    sample_fname = os.path.join(path, "sample.yaml")

    if not os.path.exists(sample_fname):
        raise ValueError(f"No sample file found for this path: {path}")

    return load_sample(fname=sample_fname)


def sample_to_dataframe(sample: Sample) -> pd.DataFrame:

    lamella_list = []
    for lamella in sample.positions.values():

        # lamella
        lamella_dict = {
            "number": lamella._number,
            "petname": lamella._petname,
            # "path": lamella.path,
            "lamella.x": lamella.lamella_coordinates.x,
            "lamella.y": lamella.lamella_coordinates.y,
            "lamella.z": lamella.lamella_coordinates.z,
            "lamella.r": lamella.lamella_coordinates.r,
            "lamella.t": lamella.lamella_coordinates.t,
            "lamella.coordinate_system": lamella.lamella_coordinates.coordinate_system,
            "landing.x": lamella.landing_coordinates.x,
            "landing.y": lamella.landing_coordinates.y,
            "landing.z": lamella.landing_coordinates.z,
            "landing.r": lamella.landing_coordinates.r,
            "landing.t": lamella.landing_coordinates.t,
            "landing.coordinate_system": lamella.landing_coordinates.coordinate_system,
            "landing_selected": lamella.landing_selected,
            "current_stage": lamella.current_state.stage.name,
            "last_timestamp": lamella.current_state.microscope_state.timestamp,
            "history: ": len(lamella.history),
        }

        lamella_list.append(lamella_dict)

    df = pd.DataFrame.from_dict(lamella_list)

    return df