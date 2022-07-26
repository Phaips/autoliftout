import glob
import logging
import os
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autoscript_sdb_microscope_client.structures import AdornedImage
from liftout.fibsem.movement import pixel_to_realspace_coordinate
from liftout.fibsem.structures import Point
from PIL import Image


class DetectionType(Enum):
    LamellaCentre = 1
    NeedleTip = 2
    LamellaEdge = 3
    LandingPost = 4
    ImageCentre = 5


@dataclass
class DetectionFeature:
    detection_type: DetectionType
    feature_px: Point  # x, y


@dataclass
class DetectionResult:
    features: list[DetectionFeature]
    adorned_image: AdornedImage
    display_image: np.ndarray
    distance_metres: Point = Point(0, 0)  # x, y
    downscale_image: np.ndarray = None
    microscope_coordinate: list[Point] = None


def convert_pixel_distance_to_metres(
    p1: Point, p2: Point, adorned_image: AdornedImage, display_image: np.ndarray
):
    """Convert from pixel coordinates to distance in metres """
    # NB: need to use this func, not pixel_to_realspace because display_iamge and adorned image are no the same size...

    # upscale the pixel coordinates to adorned image size
    scaled_px_1 = scale_pixel_coordinates(p1, display_image, adorned_image)
    scaled_px_2 = scale_pixel_coordinates(p2, display_image, adorned_image)

    # convert pixel coordinate to realspace coordinate
    x1_real, y1_real = pixel_to_realspace_coordinate(
        (scaled_px_1.x, scaled_px_1.y), adorned_image
    )
    x2_real, y2_real = pixel_to_realspace_coordinate(
        (scaled_px_2.x, scaled_px_2.y), adorned_image
    )

    p1_real = Point(x1_real, y1_real)
    p2_real = Point(x2_real, y2_real)

    # calculate distance between points along each axis
    x_distance_m, y_distance_m = coordinate_distance(p1_real, p2_real)

    return x_distance_m, y_distance_m


def scale_pixel_coordinates(px: Point, from_image: np.ndarray, to_image=None) -> Point:
    """Scale the pixel coordinate from one image to another"""
    if isinstance(to_image, AdornedImage):
        to_image = to_image.data

    x_scale, y_scale = (
        px.x / from_image.shape[1],
        px.y / from_image.shape[0],
    )  # (x, y)

    scaled_px = Point(x_scale * to_image.shape[1], y_scale * to_image.shape[0])

    return scaled_px


def coordinate_distance(p1: Point, p2: Point):
    """Calculate the distance between two points in each coordinate"""

    return p2.x - p1.x, p2.y - p1.y


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


def get_scale_invariant_coordinates(point: Point, shape: tuple) -> Point:

    scaled_pt = Point(x=point.x / shape[1], y=point.y / shape[0])

    return scaled_pt


def scale_coordinate_to_image(point: Point, shape: tuple) -> Point:
    """Scale invariant coordinates to image shape"""
    scaled_pt = Point(x=point.x * shape[1], y=point.y * shape[0])

    return scaled_pt


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
    import datetime
    import random
    import time

    import liftout
    import matplotlib.pyplot as plt
    from PIL import Image

    # mkdir for copying images to
    data_path = os.path.join(os.path.dirname(liftout.__file__), "data", "retrain")
    os.makedirs(data_path, exist_ok=True)
    print(f"Searching in {path} for retraining images...")

    # find all files for retraining (with _label postfix
    filenames = glob.glob(os.path.join(path, "/**/*label*.tif"), recursive=True)
    print(f"{len(filenames)} images found for relabelling")
    print(f"Copying images to {data_path}...")

    for i, fname in enumerate(filenames):
        # tqdm?
        print(f"Copying {i}/{len(filenames)}")
        # basename = os.path.basename(fname)
        datetime_str = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%Y%m%d.%H%M%S"
        )
        basename = f"{datetime_str}.{random.random()}.tif"  # use a random number to prevent duplicates at seconds time resolution
        # print(fname, basename)
        if show:
            img = Image.open(fname)
            plt.imshow(img, cmap="gray")
            plt.show()

        source_path = os.path.join(fname)
        destination_path = os.path.join(data_path, basename)
        # print(f"Source: {source_path}")
        # print(f"Destination: {destination_path}")
        print("-" * 50)
        shutil.copyfile(source_path, destination_path)

    # zip the image folder
    # shutil.make_archive(f"{path}/images", 'zip', label_dir)

def write_data_to_csv(path: Path, info) -> None:

    dataframe_path = os.path.join(path, "data.csv")

    cols = ["label", "p1.type", "p1.x", "p1.y", "p2.type", "p2.x", "p2.y"]
    df_tmp = pd.DataFrame([info], columns=cols)
    if os.path.exists(dataframe_path):
        df = pd.read_csv(dataframe_path)
        df = pd.concat([df, df_tmp], axis=0, ignore_index=True)
    else:
        df = df_tmp
    df.to_csv(dataframe_path,index=False)

    logging.info(f"Logged data to {dataframe_path}.")



def load_detection_result(path: Path, data) -> DetectionResult:
    """Read detection result from dataframe row, and return"""

    label = data["label"]
    p1_type = DetectionType[data["p1.type"]]
    p1 = Point(x=data["p1.x"], y=data["p1.y"])
    p2_type = DetectionType[data["p2.type"]]
    p2 = Point(x=data["p2.x"], y=data["p2.y"])

    fname = glob.glob(os.path.join(path, f"*{label}*.tif"))[0]
    img = AdornedImage.load(fname)

    p1 = scale_coordinate_to_image(p1, img.data.shape)
    p2 = scale_coordinate_to_image(p2, img.data.shape)

    det = DetectionResult(
        features=[
            DetectionFeature(detection_type=p1_type, feature_px=p1),
            DetectionFeature(detection_type=p2_type, feature_px=p2),
        ],
        adorned_image=img,
        display_image=None,
        downscale_image=None,
    )

    return det


def plot_detection_result(det_result: DetectionResult):
    """Plot the Detection Result using matplotlib"""
    from liftout.config import config

    p1 = det_result.features[0].feature_px
    p2 = det_result.features[1].feature_px

    c1 = config.DETECTION_TYPE_COLOURS[det_result.features[0].detection_type]
    c2 = config.DETECTION_TYPE_COLOURS[det_result.features[1].detection_type]

    fig = plt.figure()
    plt.imshow(det_result.adorned_image.data, cmap="gray")
    plt.plot(p1.x, p1.y, color=c1, marker="+", ms=20)
    plt.plot(p2.x, p2.y, color=c2, marker="+", ms=20)

    return fig


def write_data_to_disk(path: Path, detection_result: DetectionResult) -> None:
    
    # TODO: move this
    from liftout import utils
    label = utils.current_timestamp() + "_label"

    utils.save_image(
        image=detection_result.adorned_image,
        save_path=path,
        label=label,
    )


    # get scale invariant coords
    shape = detection_result.downscale_image.shape
    scaled_p0 = get_scale_invariant_coordinates(detection_result.features[0].feature_px, shape=shape)
    scaled_p1 = get_scale_invariant_coordinates(detection_result.features[1].feature_px, shape=shape)

    # get info
    logging.info(f"Label: {label}")
    info = [label, 
        detection_result.features[0].detection_type.name, 
        scaled_p0.x, 
        scaled_p0.y, 
        detection_result.features[1].detection_type.name, 
        scaled_p1.x, 
        scaled_p1.y
        ]

    write_data_to_csv(path, info)
