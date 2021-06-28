#!/usr/bin/env python3


import glob
import json
import re
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import PIL
import segmentation_models_pytorch as smp
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms, utils

from utils import load_image, draw_crosshairs
from liftout.main import validate_detection

from DetectionModel import DetectionModel




    





# REFACTOR DRAWING TOOLS
# - refactor drawing to use a single function:

# TODO: make the mask consistently a np.array or PIL.Images

# extract_class_pixels(mask, color)
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


# - draw_feature(mask, px, color, crosshair)


def draw_feature(mask, px, color, RECT_WIDTH=4, crosshairs=False):
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


# - draw_two_features(mask, feature_1, feature_2, line=False)


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


# - detect_centre_point(mask, color, threshold)
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

def detect_right_edge(mask, color, threshold):
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
        edge_px = px[max_idx]  # right edge px

    return edge_px


# - draw_overlay(img, mask, alpha, show) - DONE
def draw_overlay(img, mask, alpha=0.4, show=False):
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
    img = img.convert("RGB")  # required for blending

    # blend images together
    alpha_blend = PIL.Image.blend(img, mask, alpha)

    if show:
        plt.imshow(alpha_blend)
        plt.show()

    return alpha_blend


def detect_needle_tip(img, mask, threshold=200):

    color = (0, 255, 0) # fixed color

    # TODO: extract filter from detection?
    mask_filt, px_filt = extract_class_pixels(mask, color)

    edge_px = detect_right_edge(mask, color, threshold=threshold)
    mask_draw = draw_feature(mask_filt, edge_px, color, crosshairs=True)
    needle_mask = draw_overlay(img, mask_draw, show=True)

    return edge_px, needle_mask

def detect_lamella_centre(img, mask, threshold=25):

    color = (255, 0, 0) # fixed color

    # TODO: extract filter from detection?
    mask_filt, px_filt = extract_class_pixels(mask, color)

    centre_px = detect_centre_point(mask, color, threshold=threshold)
    mask_draw = draw_feature(mask_filt, centre_px, color, crosshairs=True)
    lamella_mask = draw_overlay(img, mask_draw, show=True)

    return centre_px, lamella_mask

def detect_lamella_edge(img, mask, threshold=25):

    color = (255, 0, 0) # fixed color

    # TODO: extract filter from detection?
    mask_filt, px_filt = extract_class_pixels(mask, color) # this is duplicate in detect_func
    edge_px = detect_right_edge(mask, color, threshold=threshold)
    mask_draw = draw_feature(mask_filt, edge_px, color, crosshairs=True)
    lamella_mask = draw_overlay(img, mask_draw, show=True)

    return edge_px, lamella_mask



class Detector:

    def __init__(self, weights_file) -> None:
        self.detection_model = DetectionModel(weights_file)
        
