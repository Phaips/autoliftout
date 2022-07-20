

from enum import Enum
from pathlib import Path

import yaml
import os
import datetime
import time
from dataclasses import dataclass
import uuid
import petname

from liftout.fibsem.acquire import BeamType

from autoscript_sdb_microscope_client.structures import StagePosition, AdornedImage, ManipulatorPosition
from liftout import utils
from liftout.fibsem.sampleposition import AutoLiftoutStage, ReferenceImages


class Sample:
    def __init__(self, path: Path) -> None:
        
        
        self.path: Path = path
        # self.sample_yaml: dict = self.setup_sample_yaml_file(path)

        self.positions: dict = {}

    def __to_dict__(self) -> dict:

        state_dict = {
            "path": self.path, 
            "positions": [lamella.__to_dict__() for lamella in self.positions.values()]
        }

        return state_dict

    def __from__dict__(self, state_dict) -> None:
        # restore class from dict / yaml
        self.path = state_dict["path"]
        self.positions = state_dict["positions"]

        return 

    def save(self) -> None:
        """Save the sample data to yaml file"""

        with open(os.path.join(self.path, "sample.yaml"), "w") as f:
            yaml.dump(self.__to_dict__(), f, indent=4)

class Lamella:

    def __init__(self, number: int = 0) -> None:
        
        self._number: int = number
        self._id = uuid.uuid4()
        self._petname = f"{self._number:02d}-{petname.generate(2)}"

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
        Last Completed: {self.current_state.microscope_state.last_completed_stage}
        History: {len(self.history)} stages completed.
        """ 

    def __to_dict__(self):

        state_dict = {
            "id": str(self._id),
            "petname": self._petname,
            "number": self._number,
            "lamella_coordinates": {
                "x": self.lamella_coordinates.x,
                "y": self.lamella_coordinates.y,
                "z": self.lamella_coordinates.z,
                "r": self.lamella_coordinates.r,
                "t": self.lamella_coordinates.t,
                "coordinate_system": self.lamella_coordinates.coordinate_system
            },
            "landing_coordinates": {
                "x": self.landing_coordinates.x,
                "y": self.landing_coordinates.y,
                "z": self.landing_coordinates.z,
                "r": self.landing_coordinates.r,
                "t": self.landing_coordinates.t,
                "coordinate_system": self.landing_coordinates.coordinate_system
            },
            "current_state":  self.current_state.__to_dict__(),
            "history": [state.__to_dict__() for state in self.history]
        }

        return state_dict

def lamella_from_dict(lamella_dict: dict) -> Lamella:

    lamella = Lamella(number = lamella_dict["number"])

    lamella._petname = lamella_dict["petname"]
    lamella._id = lamella_dict["id"]

    # load stage positions from yaml
    lamella.lamella_coordinates = stage_position_from_dict(lamella_dict["lamella_coordinates"])
    lamella.landing_coordinates = stage_position_from_dict(lamella_dict["landing_coordinates"])

    # load current state
    lamella.current_state = autoliftout_state_from_dict(lamella_dict["current_state"])

    # load history
    
    return lamella

@dataclass
class BeamSettings:
    beam_type: BeamType
    working_distance: float = None
    beam_current: float = None
    hfw: float = None
    resolution: float = None
    dwell_time: float = None
    stigmation: float = None

    def __to_dict__(self):

        state_dict = {
            "beam_type": self.beam_type.name,
            "working_distance": self.working_distance,
            "beam_current": self.beam_current,
            "hfw": self.hfw,
            "resolution": self.resolution,
            "dwell_time": self.dwell_time,
            "stigmation": self.stigmation
        }

        return state_dict

def beam_settings_from_dict(state_dict: dict) -> None:

    beam_settings = BeamSettings(
        beam_type = BeamType[state_dict["beam_type"]],
        working_distance = state_dict["working_distance"],
        beam_current=state_dict["beam_current"],
        hfw=state_dict["hfw"], 
        resolution=state_dict["resolution"],
        dwell_time=state_dict["dwell_time"],
        stigmation=state_dict["stigmation"]
    )

    return beam_settings

@dataclass
class MicroscopeState:
    timestamp: float = None
    absolute_position: StagePosition = StagePosition()
    last_completed_stage: AutoLiftoutStage = None
    eb_settings: BeamSettings = BeamSettings(beam_type=BeamType.ELECTRON)
    ib_settings: BeamSettings = BeamSettings(beam_type=BeamType.ION)

    def __to_dict__(self) -> dict:

        state_dict = {
            "timestamp": self.timestamp,
            "absolute_position": {
                "x": self.absolute_position.x,
                "y": self.absolute_position.y,
                "z": self.absolute_position.z,
                "r": self.absolute_position.r,
                "t": self.absolute_position.t,
                "coordinate_system": self.absolute_position.coordinate_system
            },
            "last_completed_stage": str(self.last_completed_stage),
            "eb_settings": self.eb_settings.__to_dict__(),
            "ib_settings": self.ib_settings.__to_dict__()
        }

        return state_dict

def microscope_state_from_dict(state_dict: dict) -> MicroscopeState:

    microscope_state = MicroscopeState(
        timestamp=state_dict["timestamp"],
        absolute_position=StagePosition(
            x=state_dict["absolute_position"]["x"],
            y=state_dict["absolute_position"]["y"],
            z=state_dict["absolute_position"]["z"],
            r=state_dict["absolute_position"]["r"],
            t=state_dict["absolute_position"]["t"],
            coordinate_system=state_dict["absolute_position"]["coordinate_system"]
        ),
        last_completed_stage=AutoLiftoutStage[state_dict["last_completed_stage"]],
        eb_settings=beam_settings_from_dict(state_dict["eb_settings"]),
        ib_settings=beam_settings_from_dict(state_dict["ib_settings"])
    )

    return microscope_state

@dataclass
class AutoLiftoutState:
    stage: AutoLiftoutStage = AutoLiftoutStage.Initialisation
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


def autoliftout_state_from_dict(state_dict: dict) -> AutoLiftoutState:

    autoliftout_state = AutoLiftoutState(
        stage = AutoLiftoutStage[state_dict["stage"]],
        microscope_state=microscope_state_from_dict(state_dict["microscope_state"]),
        start_timestamp=state_dict["start_timestamp"], 
        end_timestamp=state_dict["end_timestamp"]
    )

    return autoliftout_state


def stage_position_from_dict(state_dict: dict) -> StagePosition:

    stage_position = StagePosition(
        x=state_dict["x"],
        y=state_dict["y"],
        z=state_dict["z"],
        r=state_dict["r"],
        t=state_dict["t"],
        coordinate_system=state_dict["coordinate_system"]
    )

    return stage_position



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