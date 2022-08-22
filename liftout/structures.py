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

