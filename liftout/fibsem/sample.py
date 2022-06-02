

from enum import Enum
from pathlib import Path

import yaml
import os
import datetime
import time
from dataclasses import dataclass
import uuid
import petname

from autoscript_sdb_microscope_client.structures import StagePosition, AdornedImage, ManipulatorPosition
from liftout import utils
from liftout.fibsem.sampleposition import AutoLiftoutStage, MicroscopeState, ReferenceImages


class Sample:
    def __init__(self, data_path: Path) -> None:
        
        
        self.data_path: Path = data_path
        self.sample_yaml: dict = self.setup_sample_yaml_file(data_path)

        self.positions: list = []


    def setup_sample_yaml_file(self, data_path: Path):
        # check if yaml file already exists
        yaml_file = os.path.join(data_path, "sample.yaml")

        if os.path.exists(yaml_file):
            # read and open existing yaml file
            with open(yaml_file, "r") as f:
                sample_yaml = yaml.safe_load(f)

        else:
            # create new yaml file
            sample_yaml = { 
                "created": utils.current_timestamp(),
                "data_path": data_path,
                "sample": {},
            }
        return sample_yaml


class Lamella:

    def __init__(self) -> None:
        
        self.lamella_coordinates: StagePosition = StagePosition()
        self.landing_coordinates: StagePosition = StagePosition()

        self.lamella_ref_images: ReferenceImages = None
        self.landing_ref_images: ReferenceImages = None

        self.landing_selected: bool = False

        self._id = uuid.uuid4()
        self._petname = petname.generate(2)
        self._number: int 

        self.microscope_state: MicroscopeState = MicroscopeState()

        # state history
        self.history: list = [] 




@dataclass
class AutoLiftoutStageState:
    stage: AutoLiftoutStage
    microscope_state: MicroscopeState
    start_timestamp: float
    end_timestamp: float


# Sample:
#   data_path: Path
#   
#   positions: [Lamella, Lamella, Lamella]



# Lamella
#   lamella_coordinates: StagePosition
#   landing_coordinates: StagePosition
#   lamella_ref_images: ReferenceImages       
#   landing_ref_images: ReferenceImages
#   state: AutoLiftoutStageState
#       stage: AutoLiftoutStage
#       microscope_state: MicroscopeState
#           eb_settings: BeamSettings
#           ib_settings: BeamSettings
#           absolute_position: StagePosition