# from autoscript_sdb_microscope_client.structures import *
import glob
from enum import Enum
from pprint import pprint

import matplotlib.pyplot as plt
import yaml
from liftout.detection import utils


class SampleStatus(Enum):
    Setup = 0
    Milling = 1
    Liftout = 2
    Landing = 3
    Reset = 4
    Cleanup = 5
    Finished = 6

class FakeStagePosition():
    """ Mock stage position because dont have access to autoscript"""
    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.z = 0
        self.r = 0
        self.t = 0

class Sample:
    def __init__(self):
        self.landing_coordinates = FakeStagePosition() # StagePosition()
        self.lamella_coordinates = FakeStagePosition() # StagePosition()
        self.landing_ref_images = list()
        self.lamella_ref_images = list()
        self.status = NotImplemented
        self.data_path = "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/liftout/gui/log/run/20210804.151912/"
        self.timestamp = self.data_path.split("/")[-2]

    def save_data(self):
        

        lamella_coordinates_dict = {
            "x": self.lamella_coordinates.x,
            "y": self.lamella_coordinates.y,
            "z": self.lamella_coordinates.z,
            "r": self.lamella_coordinates.r,
            "t": self.lamella_coordinates.t,
        }
        
        landing_coordinates_dict = {
            "x": self.landing_coordinates.x,
            "y": self.landing_coordinates.y,
            "z": self.landing_coordinates.z,
            "r": self.landing_coordinates.r,
            "t": self.landing_coordinates.t,
        }        

        # save stage position to yml file
        save_dict = {
            "timestamp": self.timestamp,
            "lamella_coordinates": lamella_coordinates_dict,
            "landing_coordinates": landing_coordinates_dict,
            "data_path": self.data_path}

        # should we save the images separately? or just use previously saved?
        with open(f"{self.data_path}sample.yaml", "w") as outfile:
            yaml.dump(save_dict, outfile)



        return NotImplemented

    def load_data_from_file(self, fname):

        # load yaml file

        with open(fname, 'r') as f:
            load_dict = yaml.safe_load(f)

        pprint(load_dict)

        # TODO: improve this
        ref_landing_lowres_eb = self.data_path + "img/01_ref_landing_low_res_eb.tif"
        ref_landing_highres_eb = self.data_path + "img/01_ref_landing_high_res_eb.tif"
        ref_landing_lowres_ib = self.data_path + "img/01_ref_landing_low_res_ib.tif"
        ref_landing_highres_ib = self.data_path + "img/01_ref_landing_high_res_ib.tif"
        ref_lamella_lowres_eb = self.data_path + "img/01_ref_lamella_low_res_eb.tif"
        ref_lamella_highres_eb = self.data_path + "img/01_ref_lamella_high_res_eb.tif"
        ref_lamella_lowres_ib = self.data_path + "img/01_ref_lamella_low_res_ib.tif"
        ref_lamella_highres_ib = self.data_path + "img/01_ref_lamella_high_res_ib.tif"

        ref_landing_fnames = [
            ref_landing_lowres_eb,
            ref_landing_highres_eb,
            ref_landing_lowres_ib,
            ref_landing_highres_ib
        ]

        ref_lamella_fnames = [
            ref_lamella_lowres_eb,
            ref_lamella_highres_eb,
            ref_lamella_lowres_ib,
            ref_lamella_highres_ib
        ]

        
        for fname in ref_landing_fnames:
            img = utils.load_image_from_file(fname)

            plt.title(fname)
            plt.imshow(img)
            plt.show()
            self.landing_ref_images.append(img) #TODO: change to AdornedImage

        for fname in ref_lamella_fnames:
            img = utils.load_image_from_file(fname)
            
            plt.title(fname)
            plt.imshow(img)
            plt.show()
            self.lamella_ref_images.append(img) #TODO: change to AdornedImage



        # print(self.landing_ref_images)
        # print(self.lamella_ref_images)
        # load stage positions from yaml

        # load images from disk

        # format for lifout

        return NotImplemented

if __name__ == "__main__":

    sample = Sample()
    
    sample.lamella_coordinates.x = 1
    sample.lamella_coordinates.y = 1
    sample.lamella_coordinates.z = 1
    sample.lamella_coordinates.r = 1
    sample.lamella_coordinates.t = 1
    
    sample.save_data()
    print(sample)

    fname = f"{sample.data_path}sample.yaml"
    sample.load_data_from_file(fname=fname)


# timestamp="20210804.151912"

# self.landing_coordinates = [StagePosition(x=-0.0033364968,y=-0.003270375,z=0.0039950765,t=0.75048248,r=4.0317333)]

# self.lamella_coordinates = [StagePosition(x=0.0028992212,y=-0.0033259583,z=0.0039267467,t=0.43632805,r=4.0318009)]
# # self.zipped_coordinates = None
# self.zipped_coordinates = list(zip(self.lamella_coordinates, self.landing_coordinates))

# original_landing_images = ((f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_landing_low_res_eb.tif',
#                             f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_landing_high_res_eb.tif',
#                             f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_landing_low_res_eb_ib.tif',
#                             f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_landing_high_res_eb_ib.tif'))
# self.original_landing_images = list()
# for image in original_landing_images:
#     self.original_landing_images.append(AdornedImage.load(image))


# original_trench_images= ((f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_lamella_low_res_eb.tif',
#                             f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_lamella_high_res_eb.tif',
#                             f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_lamella_low_res_eb_ib.tif',
#                             f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_lamella_high_res_eb_ib.tif'))
# self.original_trench_images = list()
# for image in original_trench_images:
#     self.original_trench_images.append(AdornedImage.load(image))

# self.original_landing_images = [self.original_landing_images]
# self.original_trench_images = [self.original_trench_images]



# Sample
# - lamella_coordinates
# - lamella_ref_images
# - landing_coordinates
# - landing_ref_images
