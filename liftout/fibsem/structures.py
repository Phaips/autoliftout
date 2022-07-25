# fibsem structures


from autoscript_sdb_microscope_client.structures import AdornedImage, StagePosition, ManipulatorPosition

from dataclasses import dataclass


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0


