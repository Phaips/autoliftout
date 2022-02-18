
from liftout.tests import data
from attr import dataclass
import numpy as np
import PIL

@dataclass
class StagePosition:
    """ Mock StagePosition because dont have access to autoscript"""
    x: float = 0
    y: float = 0
    z: float = 0
    r: float = 0
    t: float = 0

    def __init__(self, x=0, y=0, z=0, r=0, t=0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.t = t

    def __repr__(self) -> str:
        return f"x={self.x}, y={self.y}, z={self.z}, r={self.r}, t={self.t}"

@dataclass
class ManipulatorPosition:
    """Mock StagePosition because dont have access to autoscript"""
    x: float = 0
    y: float = 0
    z: float = 0
    r: float = 0
    t: float = 0

    def __init__(self, x=0, y=0, z=0, r=0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.r = r

    def __repr__(self) -> str:
        return str(f"x={self.x}, y={self.y}, z={self.z}, r={self.r}")

class AdornedImage:
    
    @staticmethod
    def load(fname):
        adorned_img = AdornedImage()
        adorned_img.data = np.array(PIL.Image.open(fname))
        
        return adorned_img