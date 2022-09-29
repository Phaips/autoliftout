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
from fibsem.structures import (FibsemStage, FibsemState, ImageSettings,
                               MicroscopeState, ReferenceImages, StageSettings,
                               SystemSettings)

from liftout import utils

# TODO: make this config adjustable, change to dataclass? 
class ReferenceHFW(Enum):
    Wide: float = 2750.0e-6
    Low: float = 900.0e-6
    Medium: float = 400.0e-6
    High: float = 150.0e-6
    Super: float = 80.0e-6
    Ultra: float = 50.0e-6

@dataclass
class AutoLiftoutOptions:
    high_throughput: bool = True
    piescope: bool = False
    autolamella: bool = False

    @staticmethod  # TODO: add test
    def __from_dict__(settings: dict) -> "AutoLiftoutOptions":

        options = AutoLiftoutOptions(
            high_throughput=settings["high_throughput"],
            piescope=settings["piescope_enabled"],
            autolamella=settings["autolamella"],
        )
        return options

class AutoLiftoutMode(Enum):
    Manual = 1
    Auto = 2 

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
    
    def __to_dataframe__(self) -> pd.DataFrame:

        lamella_list = []
        lamella: Lamella
        for lamella in self.positions.values():

            # lamella
            lamella_dict = {
                "number": lamella._number,
                "petname": lamella._petname,
                # "path": lamella.path,
                "lamella.x": lamella.lamella_state.absolute_position.x,
                "lamella.y": lamella.lamella_state.absolute_position.y,
                "lamella.z": lamella.lamella_state.absolute_position.z,
                "lamella.r": lamella.lamella_state.absolute_position.r,
                "lamella.t": lamella.lamella_state.absolute_position.t,
                "lamella.coordinate_system": lamella.lamella_state.absolute_position.coordinate_system,
                "landing.x": lamella.landing_state.absolute_position.x,
                "landing.y": lamella.landing_state.absolute_position.y,
                "landing.z": lamella.landing_state.absolute_position.z,
                "landing.r": lamella.landing_state.absolute_position.r,
                "landing.t": lamella.landing_state.absolute_position.t,
                "landing.coordinate_system": lamella.landing_state.absolute_position.coordinate_system,
                "landing_selected": lamella.landing_selected,
                "current_stage": lamella.current_state.stage.name,
                "last_timestamp": lamella.current_state.microscope_state.timestamp,
                "history: ": len(lamella.history),
            }

            lamella_list.append(lamella_dict)

        df = pd.DataFrame.from_dict(lamella_list)

        return df


    @staticmethod
    def load(fname: Path) -> 'Sample':
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
            lamella = Lamella.__from_dict__(path=sample.path, lamella_dict=lamella_dict)
            sample.positions[lamella._number] = lamella

        return sample

# TODO: move to fibsem?
# TODO: need to inherit the state class?
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

        self.lamella_state: MicroscopeState = MicroscopeState()
        self.landing_state: MicroscopeState = MicroscopeState()

        self.landing_selected: bool = False

        self.current_state: AutoLiftoutState = AutoLiftoutState()

        # state history
        self.history: list[AutoLiftoutState] = []

    def __repr__(self) -> str:

        return f"""
        Lamella {self._number} ({self._petname}). 
        Lamella Coordinates: {self.lamella_state.absolute_position}, 
        Landing Coordinates: {self.landing_state.absolute_position}, 
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
            "landing_selected": self.landing_selected,
            "lamella_state": self.lamella_state.__to_dict__(),
            "landing_state": self.landing_state.__to_dict__(),
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

    @staticmethod
    def __from_dict__(path: str, lamella_dict: dict) -> 'Lamella':

        lamella = Lamella(
            path=path, number=lamella_dict["number"], _petname=lamella_dict["petname"]
        )

        lamella._petname = lamella_dict["petname"]
        lamella._id = lamella_dict["id"]

        # load stage positions from yaml
        lamella.lamella_state = MicroscopeState.__from_dict__(lamella_dict["lamella_state"])
        lamella.landing_state = MicroscopeState.__from_dict__(lamella_dict["landing_state"])
        lamella.landing_selected = bool(lamella_dict["landing_selected"])

        # load current state
        lamella.current_state = AutoLiftoutState.__from_dict__(lamella_dict["current_state"])

        # load history
        lamella.history = [
            AutoLiftoutState.__from_dict__(state_dict)
            for state_dict in lamella_dict["history"]
        ]

        return lamella

    # convert to method
    def get_reference_images(self, label: str) -> ReferenceImages:
        reference_images = ReferenceImages(
            low_res_eb=self.load_reference_image(f"{label}_low_res_eb"),
            high_res_eb=self.load_reference_image(f"{label}_high_res_eb"),
            low_res_ib=self.load_reference_image(f"{label}_low_res_ib"),
            high_res_ib=self.load_reference_image(f"{label}_high_res_ib"),
        )

        return reference_images

@dataclass
class AutoLiftoutState(FibsemState):
    stage: AutoLiftoutStage = AutoLiftoutStage.Setup
    microscope_state: MicroscopeState = MicroscopeState()
    start_timestamp: float = None
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
    def __from_dict__(self, state_dict: dict) -> 'AutoLiftoutState':

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
#   lamella_state: MicroscopeState
#   landing_state: MicroscopeState
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

    return Sample.load(fname=sample_fname)



