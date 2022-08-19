from dataclasses import dataclass
from fibsem.structures import (
    SystemSettings,
    ImageSettings,
    StageSettings,
)
from autoscript_sdb_microscope_client.structures import StagePosition
from enum import Enum

class ReferenceHFW(Enum):
    Wide: float = 2750.0e-6
    Low: float = 900.0e-6
    Medium: float = 400.0e-6
    High: float = 150.0e-6
    Super: float = 80.0e-6
    Ultra: float = 50.0e-6

@dataclass
class ReferenceHFW_DATACLASS:
    wide: float = 2750.0e-6
    low: float = 900.0e-6
    medium: float = 400.0e-6
    high: float = 150.0e-6
    super: float = 80.0e-6
    ultra: float = 50.0e-6

    @classmethod
    def __from_dict__(self, settings: dict) -> 'ReferenceHFW_DATACLASS':
        
        reference_hfw = ReferenceHFW_DATACLASS(
            wide = settings["wide"],
            low = settings["low"],
            medium = settings["medium"],
            high = settings["high"],
            super = settings["super"],
            ultra = settings["ultra"]
        )

        return reference_hfw

@dataclass
class AutoLiftoutOptions:
    high_throughput: bool = True
    piescope: bool = False
    autolamella: bool = False

    @classmethod  # TODO: add test
    def __from_dict__(self, settings: dict) -> "AutoLiftoutOptions":

        options = AutoLiftoutOptions(
            high_throughput=settings["high_throughput"],
            piescope=settings["piescope_enabled"],
            autolamella=settings["autolamella"],
        )
        return options

@dataclass
class AutoLiftoutSettings:
    system: SystemSettings
    options: AutoLiftoutOptions
    stage: StageSettings
    image_settings: ImageSettings
    grid_position: StagePosition  
    landing_position: StagePosition
    protocol: dict

    def __repr__(self) -> str:

        return f"""AutoLiftoutSettings:
System: {self.system},
Options: {self.options},
Stage: {self.stage},
Calibration: {self.calibration},
Imaging: {self.image_settings},
Grid: {self.grid_position}, 
Landing: {self.landing_position},
Protocol: {len(self.protocol)} stages ({[k for k in self.protocol.keys()]}         
"""
