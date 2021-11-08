
import pandas as pd
import numpy as np
from PIL import Image
import re

import glob
from random import shuffle
import shutil
import os

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


def extract_img_for_labelling(path, show=False):
    """Extract all the images that have been identified for retraining.

    path: path to directory containing logged images

    """
    import liftout
    import matplotlib.pyplot as plt
    from PIL import Image
    import datetime, time
    import random


    # mkdir for copying images to
    data_path = os.path.join(os.path.dirname(liftout.__file__), "data", "retrain")
    os.makedirs(
        data_path, exist_ok=True
    )
    print(f"Searching in {path} for retraining images...")

    # find all files for retraining (with _label postfix
    filenames = glob.glob(os.path.join(path, "/**/*label*.tif"), recursive=True)
    print(f"{len(filenames)} images found for relabelling")
    print(f"Copying images to {data_path}...")

    for i, fname in enumerate(filenames):
        # tqdm?
        print(f"Copying {i}/{len(filenames)}")
        # basename = os.path.basename(fname)
        datetime_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')
        basename = f"{datetime_str}.{random.random()}.tif" # use a random number to prevent duplicates at seconds time resolution
        # print(fname, basename)
        if show:
            img = Image.open(fname)
            plt.imshow(img, cmap="gray")
            plt.show()

        source_path = os.path.join(fname)
        destination_path = os.path.join(data_path, basename)
        # print(f"Source: {source_path}")
        # print(f"Destination: {destination_path}")
        print("-"*50)
        shutil.copyfile(source_path, destination_path)

    # zip the image folder
    # shutil.make_archive(f"{path}/images", 'zip', label_dir)