from dataclasses import dataclass
from fibsem.structures import (
    SystemSettings,
    CalibrationSettings,
    ImageSettings,
    StageSettings,
)
from autoscript_sdb_microscope_client.structures import StagePosition


@dataclass
class ReferenceHFW:
    wide: 2750.0e-6
    low: 900.0e-6
    medium: 400.0e-6
    high: 150.0e-6
    super: 80.0e-6
    ultra: 50.0e-6

    @classmethod
    def __from_dict__(self, settings: dict) -> 'ReferenceHFW':
        
        reference_hfw = ReferenceHFW(
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
    calibration: CalibrationSettings
    reference_hfw: ReferenceHFW
    image_settings: ImageSettings
    grid_position: StagePosition  
    landing_position: StagePosition

    def __repr__(self) -> str:

        return f"""AutoLiftoutSettings:
System: {self.system},
Options: {self.options},
Stage: {self.stage},
Calibration: {self.calibration},
ReferenceHFW: {self.reference_hfw},
Imaging: {self.image_settings},
Grid: {self.grid_position}, 
Landing: {self.landing_position}        
"""
