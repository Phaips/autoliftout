#!/usr/bin/env python3


import numpy as np
import PIL
from autoscript_sdb_microscope_client.structures import AdornedImage
from liftout.detection import utils
from liftout.detection.DetectionModel import DetectionModel
from liftout.detection.utils import (DetectionFeature, DetectionResult,
                                     DetectionType, Point)
from PIL import Image
from scipy.spatial import distance
from skimage import feature

# TODO: START_HERE
# better edge detections for landing...
# dont reload the model all the time
# dont show unused masks (e..g dont show needle when detecting only lamella...)

# TODO: 
# import to get rid of downscalling the images, it causes very annoying bugs, tricky to debug, and
# leads to lots of extra book keeping also probably not required.
# however, will probably need to pad so images are square?


DETECTION_COLOURS_UINT8 = {
    DetectionType.ImageCentre: (255, 255, 255),
    DetectionType.LamellaCentre: (255, 0, 0),
    DetectionType.LamellaEdge: (255, 0, 0),
    DetectionType.NeedleTip: (0, 255, 0),
    DetectionType.LandingPost: (255, 255, 255),
}

def detect_features(img, mask, shift_type) -> list[DetectionFeature]:
    """

    args:
        img: the input img (np.array)
        mask: the output rgb prediction mask (np.array)
        shift_type: the type of feature detection to run (tuple)

    return:
        detection_features [DetectionFeature, DetectionFeature]: the detected feature coordinates and types
    """

    detection_features = []

    for det_type in shift_type:

        if not isinstance(det_type, DetectionType):
            raise TypeError(f"Detection Type {det_type} is not supported.")

        if det_type == DetectionType.ImageCentre:
            feature_px = (mask.shape[1] // 2, mask.shape[0] // 2)  # midpoint

        if det_type == DetectionType.NeedleTip:
            feature_px = detect_needle_tip(mask)  # needle_tip
            feature_px = feature_px[::-1] # yx -> xy

        if det_type == DetectionType.LamellaCentre:
            feature_px = detect_lamella_centre(mask)  # lamella_centre
            feature_px = feature_px[::-1] # yx -> xy

        if det_type == DetectionType.LamellaEdge:
            feature_px = detect_lamella_edge(mask)  # lamella_edge
            feature_px = feature_px[::-1] # yx -> xy

        if det_type == DetectionType.LandingPost:

            landing_px = (img.shape[0] // 2, img.shape[1] // 2)
            feature_px, landing_mask = detect_landing_edge(img, landing_px)  # landing post
            feature_px = feature_px[::-1] # yx -> xy

        detection_features.append(
            DetectionFeature(detection_type=det_type, feature_px=Point(*feature_px))
        )

    return detection_features

def locate_shift_between_features_v2(model: DetectionModel, adorned_img: AdornedImage, shift_type: tuple):
    """
    Calculate the distance between two features in the image coordinate system (as a proportion of the image).

    args:
        adorned_img: input image (AdornedImage, or np.array)
        shift_type: the type of feature detection shift to calculation

    return:
        detection_result (DetectionResult): The detection result containing the feature coordinates, and images

    """

    # check image type
    if isinstance(adorned_img, AdornedImage):
        img = adorned_img.data  # extract image data from AdornedImage

    # model inference
    mask = model.inference(img)

    # upscale the mask # TODO: remove once model takes full sized images
    mask = np.array(Image.fromarray(mask).resize((img.shape[1], img.shape[0])))

    # detect features for calculation
    feature_1, feature_2 = detect_features(img, mask, shift_type)

    # filter to selected masks
    mask = filter_selected_masks(mask, shift_type)

    # display features for validation
    display_image = draw_overlay(img, mask)

    # calculate movement distance
    x_distance_m, y_distance_m = utils.convert_pixel_distance_to_metres(
        feature_1.feature_px, feature_2.feature_px, adorned_img
    )

    detection_result = DetectionResult(
        features=[feature_1, feature_2],
        adorned_image=adorned_img,
        display_image=display_image,
        distance_metres=Point(x_distance_m, y_distance_m),
        microscope_coordinate=[Point(0, 0), Point(0, 0)],
    )

    return detection_result

def filter_selected_masks(mask: np.ndarray, shift_type: tuple[DetectionType]) -> np.ndarray:
    """Combine only the masks for the selected detection types"""
    c1 = DETECTION_COLOURS_UINT8[shift_type[0]]
    c2 = DETECTION_COLOURS_UINT8[shift_type[1]]
    
    # get mask for first detection type
    mask1, _ = extract_class_pixels(mask, color=c1)
    # get mask for second detection type
    mask2, _ = extract_class_pixels(mask, color=c2)

    # combine masks
    mask_combined = mask1 + mask2 

    return mask_combined

# Detection and Drawing Tools
def extract_class_pixels(mask, color):
    """ Extract only the pixels that are classified as the desired class (color)

    args:
        mask: detection mask containing all detection classes (np.array)
        color: the color of the specified class in the mask (rgb tuple)

    return:
        class_mask: the mask containing only the selected class
        idx: the indexes of the detected class in the mask

    """
    # extract only label pixels to find edges
    class_mask = np.zeros_like(mask)
    idx = np.where(np.all(mask == color, axis=-1))
    class_mask[idx] = color

    return class_mask, idx


def draw_overlay(img: np.ndarray, mask: np.ndarray, alpha:float=0.2) -> np.ndarray:
    """ Draw the detection overlay onto base image

    args:
        img: orignal image (np.array or PIL.Image)
        mask: detection mask (np.array or PIL.Image)

    returns:
        alpha_blend: mask overlaid with original image (PIL.Image)

    """

    # convert to PIL Image from np.array
    if isinstance(mask, np.ndarray):
        mask = PIL.Image.fromarray(mask)
    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img)

    # resize to same size as mask
    img = img.resize((mask.size[0], mask.size[1]), resample=PIL.Image.BILINEAR)

    # required for blending
    img = img.convert("RGB")
    mask = mask.convert("RGB")

    # blend images together
    alpha_blend = PIL.Image.blend(img, mask, alpha)

    return np.array(alpha_blend)


def detect_centre_point(mask, color, threshold=25):
    """ Detect the centre (mean) point of the mask for a given color (label)

    args:
        mask: the detection mask (PIL.Image)
        color: the color of the label for the feature to detect (rgb tuple)
        threshold: the minimum number of required pixels for a detection to count (int)

    return:

        centre_px: the pixel coordinates of the centre point of the feature (tuple)
    """
    centre_px = (0, 0)

    # extract class pixels
    class_mask, idx = extract_class_pixels(mask, color)

    # only return a centre point if detection is above a threshold
    if len(idx[0]) > threshold:
        # get the centre point of each coordinate
        x_mid = int(np.mean(idx[0]))
        y_mid = int(np.mean(idx[1]))

        # centre coordinate as tuple
        centre_px = (x_mid, y_mid)

    return centre_px


# - detect_right_edge(mask, color, threshold)


def detect_right_edge(mask, color, threshold=25, left=False) -> tuple[int]:
    """ Detect the right edge point of the mask for a given color (label)

    args:
        mask: the detection mask (PIL.Image)
        color: the color of the label for the feature to detect (rgb tuple)
        threshold: the minimum number of required pixels for a detection to count (int)

    return:

        edge_px: the pixel coordinates of the right edge point of the feature (tuple)
    """

    edge_px = (0, 0)

    # extract class pixels
    class_mask, idx = extract_class_pixels(mask, color)

    # only return an edge point if detection is above a threshold

    if len(idx[0]) > threshold:
        # convert mask to coordinates
        px = list(zip(idx[0], idx[1]))

        # get index of max value (right)
        max_idx = np.argmax(idx[1])
        if left:
            max_idx = np.argmin(idx[1])

        edge_px = px[max_idx]  # right edge px

    return edge_px


def detect_needle_tip(mask, threshold=200) -> tuple[int]:
    """Detect the needle tip"""
    color = (0, 255, 0)  # fixed color
    edge_px = detect_right_edge(mask, color, threshold=threshold)

    return edge_px


def detect_lamella_centre(mask, threshold=25) -> tuple[int]:
    """Detect the centre of the lamella"""
    color = (255, 0, 0)  # fixed color

    centre_px = detect_centre_point(mask, color, threshold=threshold)

    return centre_px


def detect_lamella_edge(mask, threshold=25) -> tuple[int]:
    """Detect the right edge of the lamella"""
    color = (255, 0, 0)  # fixed color
    edge_px = detect_right_edge(mask, color, threshold=threshold)

    return edge_px

def edge_detection(img: np.ndarray, sigma=3) -> np.ndarray:
    return feature.canny(img, sigma=3)  # sigma higher usually better


def detect_closest_edge(img: np.ndarray, landing_px: tuple[int]) -> tuple[tuple[int], np.ndarray]:
    """ Identify the closest edge point to the initially selected point

    args:
        img: base image (np.ndarray)
        landing_px: the initial landing point pixel (tuple)
    return:
        landing_edge_pt: the closest edge point to the intitially selected point (tuple)
        edges: the edge mask (np.array)
    """

    # identify edge pixels
    edges = edge_detection(img, sigma=3)
    edge_mask = np.where(edges)
    edge_px = list(zip(edge_mask[0], edge_mask[1]))

    # set min distance
    min_dst = np.inf

    # TODO: vectorise this like
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html

    landing_edge_px = (0, 0)
    for px in edge_px:

        # distance between edges and landing point
        dst = distance.euclidean(landing_px, px)

        # select point with min
        if dst < min_dst:
            min_dst = dst
            landing_edge_px = px

    return landing_edge_px, edges


def detect_landing_edge(img: np.ndarray, landing_px: tuple[int]) -> tuple[tuple[int], np.ndarray]:
    """ Detect the landing edge point closest to the initial point using canny edge detection

    args:
        img: base image (np.array)
        landing_px: the initial landing point pixel (tuple)

    returns:
        landing_edge_px: the closest edge point to the intitially selected point (tuple)
        landing_mask: the edge mask (PIL.Image)
    """
    landing_edge_px, edges_mask = detect_closest_edge(img, landing_px)
    landing_mask = draw_overlay(img, edges_mask, alpha=0.8)

    return landing_edge_px, landing_mask


def detect_bounding_box(mask, color, threshold=25):
    """ Detect the bounding edge points of the mask for a given color (label)

    args:
        mask: the detection mask (PIL.Image)
        color: the color of the label for the feature to detect (rgb tuple)
        threshold: the minimum number of required pixels for a detection to count (int)

    return:

        edge_px: the pixel coordinates of the edge points of the feature list((tuple))
    """

    top_px, bottom_px, left_px, right_px = (0, 0), (0, 0), (0, 0), (0, 0)

    # extract class pixels
    class_mask, idx = extract_class_pixels(mask, color)

    # only return an edge point if detection is above a threshold

    if len(idx[0]) > threshold:
        # convert mask to coordinates
        px = list(zip(idx[0], idx[1]))

        # get index of each value
        top_idx = np.argmin(idx[0])
        bottom_idx = np.argmax(idx[0])
        left_idx = np.argmin(idx[1])
        right_idx = np.argmax(idx[1])

        # pixel coordinates
        top_px = px[top_idx]
        bottom_px = px[bottom_idx]
        left_px = px[left_idx]
        right_px = px[right_idx]

    # bbox should be (x0, y0), (x1, y1)
    x0 = top_px[0]
    y0 = left_px[1]
    x1 = bottom_px[0]
    y1 = right_px[1]

    bbox = (x0, y0, x1, y1)

    return bbox

