from autoscript_sdb_microscope_client.structures import *
from enum import Enum


class SampleStatus(Enum):
    Setup = 0
    Milling = 1
    Liftout = 2
    Landing = 3
    Reset = 4
    Cleanup = 5
    Finished = 6


class Sample:
    def __init__(self):
        self.landing_coordinates = StagePosition()
        self.lamella_coordinates = StagePosition()
        self.landing_ref_images = list()
        self.lamella_ref_images = list()
        self.status = NotImplemented

    def save_data(self):

        return NotImplemented

    def load_data_from_file(self, fname):

        return NotImplemented


# Sample
# - lamella_coordinates
# - lamella_ref_images
# - landing_coordinates
# - landing_ref_images