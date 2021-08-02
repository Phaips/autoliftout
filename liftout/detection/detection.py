#!/usr/bin/env python3


import glob
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from skimage import feature
from scipy.spatial import distance

import liftout.detection.DetectionModel as DetectionModel

from liftout.detection.utils_old import *
# from utils import load_image, draw_crosshairs, scale_invariant_coordinates_NEW, parse_metadata, validate_detection, select_point_new
# import liftout.liftout.main as liftout_main

class Detector:

    def __init__(self, weights_file) -> None:

        self.detection_model = DetectionModel.DetectionModel(weights_file)

        self.supported_shift_types = [
            "needle_tip_to_lamella_centre",
            "lamella_centre_to_image_centre",
            "lamella_edge_to_landing_post",
            "needle_tip_to_image_centre",
            "thin_lamella_top_to_centre",
            "thin_lamella_bottom_to_centre"
        ]

    def detect_features(self, img, mask, shift_type):
        """

        args:
            img: the input img (np.array)
            mask: the output rgb prediction mask (np.array)
            shift_type: the type of feature detection to run (str)

        return:
            feature_1_px: the pixel coordinates of feature one (tuple)
            feature_1_type: the type of detection for feature one (str)
            feature_1_color: color to plot feature one detection (str)
            feature_2_px: the pixel coordinates of feature two (tuple)
            feature_2_type: the type of detection for feature two (str)
            feature_2_color: color to plot feature two detection (str)
        """
        # TODO: convert feature attributes to dict?

        if shift_type not in self.supported_shift_types:
            raise ValueError("ERROR: shift type calculation is not supported")

        # detect feature shift
        if shift_type=="needle_tip_to_lamella_centre":
            feature_1_px, lamella_mask = detect_lamella_centre(img, mask) # lamella_centre
            feature_2_px, needle_mask = detect_needle_tip(img, mask) # needle_tip

            feature_1_type = "lamella_centre"
            feature_2_type = "needle_tip"

            feature_1_color = "red"
            feature_2_color = "green"

        if shift_type=="lamella_centre_to_image_centre":
            feature_1_px, lamella_mask = detect_lamella_centre(img, mask) # lamella_centre
            feature_2_px = (mask.shape[0] // 2, mask.shape[1] // 2) # midpoint

            feature_1_type = "lamella_centre"
            feature_2_type = "image_centre"

            feature_1_color = "red"
            feature_2_color = "white"

        if shift_type=="lamella_edge_to_landing_post":

            # need to resize image
            img_landing = Image.fromarray(img).resize((mask.shape[1], mask.shape[0]))
            landing_px=(img_landing.size[1]//2, img_landing.size[0]//2)

            feature_1_px, lamella_mask = detect_lamella_edge(img, mask) # lamella_centre
            feature_2_px, landing_mask = detect_landing_edge(img_landing, landing_px) # landing post # TODO: initial landing point?

            feature_1_type = "lamella_edge"
            feature_2_type = "landing_post"

            feature_1_color = "red"
            feature_2_color = "white"

        if shift_type=="needle_tip_to_image_centre":
            feature_1_px, needle_mask = detect_needle_tip(img, mask) # needle_tip
            feature_2_px = (mask.shape[0] // 2, mask.shape[1] // 2) # midpoint

            feature_1_type = "needle_tip"
            feature_2_type = "image_centre"

            feature_1_color = "green"
            feature_2_color = "white"

        if shift_type == "thin_lamella_top_to_centre":
            # top_thin_px
            feature_1_px, left_mask = detect_thin_region(img, mask, top=True)
            # img centre
            feature_2_px = (mask.shape[0] // 2, mask.shape[1] // 2) # midpoint

            feature_1_type = "top_thin"
            feature_2_type = "image_centre"

            feature_1_color = "blue"
            feature_2_color = "white"

        if shift_type == "thin_lamella_bottom_to_centre":

            # bottom_thin_px
            feature_1_px, left_mask = detect_thin_region(img, mask, top=False)
            # img centre
            feature_2_px = (mask.shape[0] // 2, mask.shape[1] // 2) # midpoint

            feature_1_type = "bottom_thin"
            feature_2_type = "image_centre"

            feature_1_color = "blue"
            feature_2_color = "white"

        return feature_1_px, feature_1_type, feature_1_color, feature_2_px, feature_2_type, feature_2_color

    def locate_shift_between_features(self, adorned_img, shift_type="needle_to_lamella_centre", show=False, validate=True):
        """
        Calculate the distance between two features in the image coordinate system (as a proportion of the image).

        args:
            adorned_img: input image (AdornedImage, or np.array)
            shift_type: the type of feature detection shift to calculation
            show: display a plot of feature detections over the input image (bool)
            validate: enable manual validation of the feature detections (bool)

        return:
            (x_distance, y_distance): the distance between the two features in the image coordinate system (as a proportion of the image) (tuple)

        """
        # TODO: fix display colours for this?

        # check image type
        if hasattr(adorned_img, 'data'):
            img = adorned_img.data # extract image data from AdornedImage
        if isinstance(adorned_img, np.ndarray):
            img = adorned_img # adorned image is just numpy array

        # run image through model
        mask = self.detection_model.model_inference(img)

        # detect features for calculation
        feature_1_px, feature_1_type, feature_1_color, feature_2_px, feature_2_type, feature_2_color = self.detect_features(img, mask, shift_type)

        # display features for validation
        mask_combined = draw_two_features(mask, feature_1_px, feature_2_px, color_1=feature_1_color, color_2=feature_2_color)
        img_blend = draw_overlay(img, mask_combined, show=show, title=shift_type)

        # # need to use the same scale images for both detection selections
        img_downscale = Image.fromarray(img).resize((mask_combined.size[0], mask_combined.size[1]))

        return img_blend, img_downscale, feature_1_px, feature_1_type, feature_2_px, feature_2_type


    def calculate_shift_between_features(self, adorned_img, shift_type="needle_to_lamella_centre", show=False, validate=True):
        """
        Calculate the distance between two features in the image coordinate system (as a proportion of the image).

        args:
            adorned_img: input image (AdornedImage, or np.array)
            shift_type: the type of feature detection shift to calculation
            show: display a plot of feature detections over the input image (bool)
            validate: enable manual validation of the feature detections (bool)

        return:
            (x_distance, y_distance): the distance between the two features in the image coordinate system (as a proportion of the image) (tuple)

        """
        # TODO: fix display colours for this?

        # # check image type
        if hasattr(adorned_img, 'data'):
            img = adorned_img.data # extract image data from AdornedImage
        if isinstance(adorned_img, np.ndarray):
            img = adorned_img # adorned image is just numpy array

        # load image from file
        mask = self.detection_model.model_inference(img)

        # detect features for calculation
        feature_1_px, feature_1_type, feature_1_color, feature_2_px, feature_2_type, feature_2_color = self.detect_features(img, mask, shift_type)

        # display features for validation
        mask_combined = draw_two_features(mask, feature_1_px, feature_2_px, color_1=feature_1_color, color_2=feature_2_color)
        img_blend = draw_overlay(img, mask_combined, show=show, title=shift_type)

        # # need to use the same scale images for both detection selections
        img_downscale = Image.fromarray(img).resize((mask_combined.size[0], mask_combined.size[1]))

        # validate detection
        if validate:
            feature_1_px = validate_detection(img_downscale, img, feature_1_px, feature_1_type)
            feature_2_px = validate_detection(img_downscale, img, feature_2_px, feature_2_type)

        # scale invariant coordinatesss
        scaled_feature_1_px = scale_invariant_coordinates_NEW(feature_1_px, mask_combined)
        scaled_feature_2_px = scale_invariant_coordinates_NEW(feature_2_px, mask_combined)

        # if no detection is found, something has gone wrong
        if scaled_feature_1_px is None or scaled_feature_2_px is None:
            raise ValueError("No detections available")

        # x, y distance (proportional)
        return scaled_feature_2_px[1] - scaled_feature_1_px[1], scaled_feature_2_px[0] - scaled_feature_1_px[0]
        #TODO: this will probably be wrong now for most, need to re-validate

# REFACTOR DETECTION AND DRAWING TOOLS

def calculate_shift_distance_in_metres(img, distance_x, distance_y, metadata=None):
    """Convert the shift distance from proportion of img to metres using image metadata"""

    # check image type
    if isinstance(img, np.ndarray):
        # use extracted metadata
        pixelsize_x = float(metadata["[Scan].PixelWidth"])
        width = img.shape[1]
        height = img.shape[0]
    else:
        # use embedded metadata in Adorned Image
        pixelsize_x = img.metadata.binary_result.pixel_size.x #5.20833e-008
        width = img.width
        height = img.height

    # scale distances by pixelsize
    field_width   = pixelsize_x  * width
    field_height  = pixelsize_x  * height
    x_shift_metres = distance_x * field_width
    y_shift_metres = distance_y * field_height

    return x_shift_metres, y_shift_metres


def extract_class_pixels(mask, color):
    # TODO: get a better name for this
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


def draw_feature(mask, px, color, RECT_WIDTH=2, crosshairs=False):
    """ Helper function to draw the feature on the mask

    args:
        mask: base mask to draw the feature (np.array)
        px: pixel coordinates of the feature to draw (tuple)
        color: color to draw the feature (rgb tuple)
        crosshairs: flag to draw crosshairs (bool)

    return:
        mask: mask with feature drawn (PIL.Image)


    """
    # convert to PIL Image
    if isinstance(mask, np.ndarray):
        mask = PIL.Image.fromarray(mask)

    # rectangular coordinates
    rect_px = (
        px[1] + RECT_WIDTH // 2,
        px[0] + RECT_WIDTH // 2,
        px[1] - RECT_WIDTH // 2,
        px[0] - RECT_WIDTH // 2,
    )

    # draw mask and feature
    draw = PIL.ImageDraw.Draw(mask)
    draw.rectangle(rect_px, fill="white", width=5)

    if crosshairs:
        draw_crosshairs(draw, mask, px, color=color)

    return mask

def draw_two_features(
    mask, feature_1, feature_2, color_1="red", color_2="green", line=True):
    """ Draw two detected features on the same mask, and optionally a line between

    args:
        mask: the detection mask with all classes (PIL.Image or np.array)
        feature_1: the first feature to draw (tuple)
        feature_2: the second feature to draw (tuple)
        line: flag to draw a line between features (bool)


    return:
        mask_combined: the detection mask with both features drawn
    """

    mask = draw_feature(mask, feature_1, color_1, crosshairs=True)
    mask = draw_feature(mask, feature_2, color_2, crosshairs=True)

    # draw a line between features
    if line:
        draw = PIL.ImageDraw.Draw(mask)
        draw.line(
            [feature_1[1], feature_1[0], feature_2[1], feature_2[0]],
            fill="white",
            width=1,
        )

    return mask

def draw_overlay(img, mask, alpha=0.4, show=False, title="Overlay Image"):
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

    if show:
        plt.title(title)
        plt.imshow(alpha_blend)
        plt.show()

    return alpha_blend

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

def detect_right_edge(mask, color, threshold, left=False):
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

def detect_needle_tip(img, mask, threshold=200):
    """Detect the needle tip"""
    color = (0, 255, 0) # fixed color

    # TODO: extract filter from detection?
    mask_filt, px_filt = extract_class_pixels(mask, color)

    edge_px = detect_right_edge(mask, color, threshold=threshold)
    mask_draw = draw_feature(mask_filt, edge_px, color, crosshairs=True)
    needle_mask = draw_overlay(img, mask_draw)

    return edge_px, needle_mask

def detect_lamella_centre(img, mask, threshold=25):
    """Detect the centre of the lamella"""
    color = (255, 0, 0) # fixed color

    # TODO: extract filter from detection?
    mask_filt, px_filt = extract_class_pixels(mask, color)

    centre_px = detect_centre_point(mask, color, threshold=threshold)
    mask_draw = draw_feature(mask_filt, centre_px, color, crosshairs=True)
    lamella_mask = draw_overlay(img, mask_draw)

    return centre_px, lamella_mask

def detect_lamella_edge(img, mask, threshold=25):
    """Detect the right edge of the lamella"""
    color = (255, 0, 0) # fixed color

    # TODO: extract filter from detection?
    mask_filt, px_filt = extract_class_pixels(mask, color) # this is duplicate in detect_func
    edge_px = detect_right_edge(mask, color, threshold=threshold)
    mask_draw = draw_feature(mask_filt, edge_px, color, crosshairs=True)
    lamella_mask = draw_overlay(img, mask_draw)

    return edge_px, lamella_mask

def detect_closest_edge(img, landing_px):
    """ Identify the closest edge point to the initially selected point

    args:
        img: base image (PIL.Image)
        landing_px: the initial landing point pixel (tuple)
    return:
        landing_edge_pt: the closest edge point to the intitially selected point (tuple)
        edges: the edge mask (np.array)
    """
    if isinstance(img, PIL.Image.Image):
        img = np.asarray(img)

    # identify edge pixels
    edges = feature.canny(img, sigma=3) # sigma higher usually better
    edge_mask = np.where(edges)
    edge_px = list(zip(edge_mask[0], edge_mask[1]))

    # set min distance
    min_dst = np.inf

    # TODO: vectorise this like
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html

    for px in edge_px:

        # distance between edges and landing point
        dst = distance.euclidean(landing_px, px)

        # select point with min
        if dst < min_dst:

            min_dst = dst
            landing_edge_px = px

    return landing_edge_px, edges


def detect_landing_edge(img, landing_px):
    """ Detect the landing edge point closest to the initial point using canny edge detection

    args:
        img: base image (np.array)
        landing_px: the initial landing point pixel (tuple)

    returns:
        landing_edge_px: the closest edge point to the intitially selected point (tuple)
        landing_mask: the edge mask (PIL.Image)
    """
    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img)

    landing_edge_px, edges_mask = detect_closest_edge(img, landing_px)
    mask_draw = draw_feature(img, landing_edge_px, color="blue", crosshairs=True)
    landing_mask = draw_overlay(img, edges_mask, alpha=0.5)

    return landing_edge_px, landing_mask


def detect_bounding_box(mask, color, threshold):
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

    return [top_px, bottom_px, left_px, right_px], bbox


def detect_thin_region(img, mask, top=True):
    """Detect the region of the lamella to thin"""

    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img)

    color = (255, 0, 0)
    threshold=25
    min_height = 20 # minimum lamella thin height

    # detect centre
    centre_px, centre_mask = detect_lamella_centre(img, mask)

    # detect bounding box around lamella
    pts, bbox = detect_bounding_box(mask, color, threshold=threshold) #top, bot, left, right

    mask_draw = draw_feature(img, bbox[:2], color="blue", crosshairs=True)
    landing_mask = draw_overlay(img, mask_draw, alpha=0.5)

    # detect top and bottom edges
    w = bbox[3] - bbox[1]
    h = min(bbox[2] - bbox[0], min_height) #

    # thin top and bottom edges to leave only small lamella slice
    # bbox method
    # top_px = bbox[0] + h/4
    # bottom_px = bbox[2] - h/4
    # left_px = bbox[1] + w/2

    # centre px method
    top_px = centre_px[0] - h/6
    bottom_px = centre_px[0] + h /6
    left_px = centre_px[1]

    # top and bottom cut start (left and up/down)
    top_thin_px = top_px, left_px
    bottom_thin_px = bottom_px, left_px

    if top:
        return top_thin_px, landing_mask
    else:
        return bottom_thin_px, landing_mask
    # TODO: this is fooled by small red specks need a way to aggregate a whole detection clump and ignore outliers...
