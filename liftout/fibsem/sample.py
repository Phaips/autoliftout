# from autoscript_sdb_microscope_client.structures import *
import glob
from enum import Enum
from pprint import pprint
import random
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
    """ Mock StagePosition because dont have access to autoscript"""
    def __init__(self, x=0, y=0, z=0, r=0, t=0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.t = t

    def __repr__(self) -> str:
        
        return f"x={self.x}, y={self.y}, z={self.z}, r={self.r}, t={self.t}"

class Sample:
    def __init__(self, data_path, sample_no):
        self.landing_coordinates = FakeStagePosition() # StagePosition()
        self.lamella_coordinates = FakeStagePosition() # StagePosition()
        self.landing_ref_images = list()
        self.lamella_ref_images = list()
        self.status = NotImplemented
        self.data_path = data_path
        self.timestamp = self.data_path.split("/")[-2]
        self.sample_no = sample_no

    def setup_yaml_file(self):
        # check if yaml file already exists for this timestamp..
        yaml_file =glob.glob(self.data_path + "*sample.yaml")

        # TODO: investigate if there will there ever be more than 1 yaml file?
        if yaml_file:
            # read and open existing yaml file
            with open(yaml_file[0], 'r') as f:
                sample_yaml = yaml.safe_load(f)

        else:
            # create new yaml file
            sample_yaml = {
                        "timestamp": self.timestamp,
                        "data_path": self.data_path, 
                        "sample": {}
                        }
        return sample_yaml
    def save_data(self):
        """ Save the lamella and landing coordinates, and reference to data path"""
        
        # check if yaml file already exists for this timestamp..
        sample_yaml = self.setup_yaml_file()

        # format coordinate data for saving        
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
            "sample_no": self.sample_no,
            "lamella_coordinates": lamella_coordinates_dict,
            "landing_coordinates": landing_coordinates_dict,
            # "data_path": self.data_path
            }

        # format dictionary
        sample_yaml["sample"][self.sample_no] = save_dict

        # should we save the images separately? or just use previously saved?
        with open(f"{self.data_path}sample.yaml", "w") as outfile:
            yaml.dump(sample_yaml, outfile)

    def load_data_from_file(self, fname=None):
        
        if fname is None:
            fname = f"{sample.data_path}sample.yaml"

        # load yaml file
        with open(fname, 'r') as f:
            sample_yaml = yaml.safe_load(f)
        
        sample_dict = sample_yaml["sample"][self.sample_no]

        # load stage positions from yaml
        self.lamella_coordinates = FakeStagePosition(x=sample_dict["lamella_coordinates"]["x"],
                                                    y=sample_dict["lamella_coordinates"]["y"],
                                                    z=sample_dict["lamella_coordinates"]["z"],
                                                    r=sample_dict["lamella_coordinates"]["r"],
                                                    t=sample_dict["lamella_coordinates"]["t"])


        self.landing_coordinates = FakeStagePosition(x=sample_dict["landing_coordinates"]["x"],
                                                    y=sample_dict["landing_coordinates"]["y"],
                                                    z=sample_dict["landing_coordinates"]["z"],
                                                    r=sample_dict["landing_coordinates"]["r"],
                                                    t=sample_dict["landing_coordinates"]["t"])
        
        # TODO: improve this
        # load images from disk
        sample_no = sample_dict["sample_no"]
        ref_landing_lowres_eb = self.data_path + f"img/{sample_no:02d}_ref_landing_low_res_eb.tif"
        ref_landing_highres_eb = self.data_path + f"img/{sample_no:02d}_ref_landing_high_res_eb.tif"
        ref_landing_lowres_ib = self.data_path + f"img/{sample_no:02d}_ref_landing_low_res_ib.tif"
        ref_landing_highres_ib = self.data_path + f"img/{sample_no:02d}_ref_landing_high_res_ib.tif"
        ref_lamella_lowres_eb = self.data_path + f"img/{sample_no:02d}_ref_lamella_low_res_eb.tif"
        ref_lamella_highres_eb = self.data_path + f"img/{sample_no:02d}_ref_lamella_high_res_eb.tif"
        ref_lamella_lowres_ib = self.data_path + f"img/{sample_no:02d}_ref_lamella_low_res_ib.tif"
        ref_lamella_highres_ib = self.data_path + f"img/{sample_no:02d}_ref_lamella_high_res_ib.tif"


        # load the adorned images and format        
        for fname in [ref_landing_lowres_eb, ref_landing_highres_eb, ref_landing_lowres_ib, ref_landing_highres_ib]:
            
            #img = AdornedImage.load(fname)
            img = utils.load_image_from_file(fname)

            self.landing_ref_images.append(img) #TODO: change to AdornedImage

        for fname in [ref_lamella_lowres_eb, ref_lamella_highres_eb, ref_lamella_lowres_ib, ref_lamella_highres_ib]:
            
            #img = AdornedImage.load(fname)
            img = utils.load_image_from_file(fname)

            self.lamella_ref_images.append(img) #TODO: change to AdornedImage

        
        # for img in self.landing_ref_images + self.lamella_ref_images:
        #     plt.title(fname.split("/")[-1])
        #     plt.imshow(img)
        #     plt.show()


    def get_sample_data(self):
        """Return the sample data formatted for liftout from the specificed data_path. """
        self.load_data_from_file()

        return (self.lamella_coordinates, self.landing_coordinates, self.lamella_ref_images, self.landing_ref_images)

        
if __name__ == "__main__":

    data_path = "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/liftout/gui/log/run/20210804.151912/"
    
    sample = Sample(data_path=data_path, sample_no=random.randint(1, 2))
    
    # create fake data
    sample.lamella_coordinates.x = sample.sample_no
    sample.lamella_coordinates.y = sample.sample_no
    sample.lamella_coordinates.z = sample.sample_no
    sample.lamella_coordinates.r = sample.sample_no
    sample.lamella_coordinates.t = sample.sample_no
    
    sample.save_data()

    data_path = "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/liftout/gui/log/run/20210804.151912/"
    
    sample_new = Sample(data_path=data_path, sample_no=sample.sample_no)

    lamella_coordinates, land_coordinates, ref_land_imgs, ref_lamella_imgs = sample_new.get_sample_data()
    
    print(f"Sample {sample_new.sample_no:02d}:",  sample_new.timestamp)
    print("Land: ", land_coordinates)
    print("Lamella: ", lamella_coordinates)
    print(len(ref_land_imgs), len(ref_lamella_imgs))



# folder structure
# log/run/timestamp/
#   sample.yaml
#   img/

# the names of the images are fixed, wiht a sample_no prefix.
# so we dont need to save the fnames, just the timestamp...

# Sample file structure
# sample.yaml:
#   timestamp:
#   data_path:
#   sample:
#       1: Sample()
#           timestamp: float
#           sample_no: int
#           lamella_coordinates: StagePosition()
#           landing_coordinates: StagePosition()
#       2: Sample()
#       3: Sample()
#       ...


# TODO:
# might be worth adding .current_sample to Liftout
# then we can change the status and display to user?