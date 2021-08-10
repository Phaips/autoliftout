
import pandas as pd
import numpy as np
from PIL import Image
import re

import glob
from random import shuffle
import shutil
 

def scale_invariant_coordinates(px, mask):
    """ Return the scale invariant coordinates of the features in the given mask

    args:
        px (tuple): pixel coordinates of the feature (y, x)
        mask (PIL.Image): PIL Image of detection mask

    returns:
        scaled_px (tuple): pixel coordinates of the feature as proportion of mask size

    """

    scaled_px = (px[0] / mask.shape[0], px[1] / mask.shape[1])

    return scaled_px

def draw_crosshairs(draw, mask, idx, color="white"):
    """ helper function for drawing crosshairs on an image"""
    draw.line([0, idx[0], mask.size[0], idx[0]], color)
    draw.line([idx[1], 0, idx[1], mask.size[1]], color)

def load_image_from_file(fname):

    """ Load a .tif image from disk as np.array """

    img = np.asarray(Image.open(fname))

    return img

def match_filenames_from_path(filepath, pattern=".tif", sort=True):

    # load image filenames, randomise
    filenames = sorted(glob.glob(filepath + pattern))
    
    if not sort:
        shuffle(filenames)

    return filenames



def parse_metadata(filename):

    # FIB meta data key is 34682, comes as a string
    img = Image.open(filename)
    img_metadata = img.tag[34682][0]

    # parse metadata
    parsed_metadata = img_metadata.split("\r\n")

    metadata_dict = {}
    for item in parsed_metadata:

        if item == "":
            # skip blank lines
            pass
        elif re.match(r"\[(.*?)\]", item):
            # find category, dont add to dict
            category = item
        else:
            # meta data point
            datum = item.split("=")

            # save to dictionary
            metadata_dict[category + "." + datum[0]] = datum[1]

    # add filename to metadata
    metadata_dict["filename"] = filename

    # convert to pandas df
    df = pd.DataFrame.from_dict(metadata_dict, orient="index").T

    return df

def extract_img_for_labelling(path, logfile="logfile"):
    """Extract all the images that have been identified for retraining"""

    log_dir = path+f"{logfile}/"
    label_dir = path+"label"
    dest_dir = path
    # identify images with _label postfix
    filenames = glob.glob(log_dir+ "*label*.tif")


    for fname in filenames:
        # print(fname)
        basename = fname.split("/")[-1]
        print(fname, basename)
        shutil.copyfile(fname, path+"label/"+basename)

    # zip the image folder
    shutil.make_archive(f"{path}/images", 'zip', label_dir)