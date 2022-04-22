#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np
import PIL

from PIL import Image, ImageDraw
from scipy.fftpack import shift
from skimage import feature
from scipy.spatial import distance

import liftout.detection.DetectionModel as DetectionModel
from liftout.detection import utils
from liftout.detection.utils import Point, DetectionType, DetectionFeature, DetectionResult
from liftout.tests.mock_autoscript_sdb_microscope_client import AdornedImage 

class Detector:

    def __init__(self, weights_file) -> None:

        self.detection_model = DetectionModel.DetectionModel(weights_file)

        self.supported_feature_types = utils.DetectionType


    def detect_features(self, img, mask, shift_type):
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
                feature_px, needle_mask = detect_needle_tip(img, mask)  # needle_tip
                feature_px = feature_px[::-1]

            if det_type == DetectionType.LamellaCentre:
                feature_px, lamella_mask = detect_lamella_centre(img, mask)  # lamella_centre
                feature_px = feature_px[::-1]
            
            if det_type == DetectionType.LamellaEdge:
                feature_px, lamella_mask = detect_lamella_edge(img, mask)  # lamella_centre
                feature_px = feature_px[::-1]

            if det_type == DetectionType.LandingPost:
                img_landing = Image.fromarray(img).resize((mask.shape[1], mask.shape[0]))
                landing_px = (img_landing.size[0] // 2, img_landing.size[1] // 2)
                feature_px, landing_mask = detect_landing_edge(img_landing, landing_px)  # landing post 
                # feature_px = feature_px[::-1] # TODO: validate if this needs to be done.

            detection_features.append(DetectionFeature(
                detection_type=det_type,
                feature_px=Point(*feature_px)
            ))

        return detection_features


    def locate_shift_between_features(self, adorned_img, shift_type:tuple):
        """
        Calculate the distance between two features in the image coordinate system (as a proportion of the image).

        args:
            adorned_img: input image (AdornedImage, or np.array)
            shift_type: the type of feature detection shift to calculation

        return:
            detection_result (DetectionResult): The detection result containing the feature coordinates, and images

        """

        # check image type
        if hasattr(adorned_img, 'data'):
            img = adorned_img.data  # extract image data from AdornedImage
        if isinstance(adorned_img, np.ndarray):
            img = adorned_img  # adorned image is just numpy array

        # run image through model
        mask = self.detection_model.model_inference(img)

        # detect features for calculation
        feature_1, feature_2 = self.detect_features(img, mask, shift_type)

        # display features for validation
        mask_combined = PIL.Image.fromarray(mask)
        img_blend = np.array(draw_overlay(img, mask_combined, show=False, title=shift_type))

        # # need to use the same scale images for both detection selections
        img_downscale = np.array(Image.fromarray(img).resize((mask_combined.size[0], mask_combined.size[1])))

        # calculate movement distance
        x_distance_m, y_distance_m = utils.convert_pixel_distance_to_metres(
            feature_1.feature_px, feature_2.feature_px, adorned_img, img_downscale)

        detection_result = DetectionResult(
            feature_1=feature_1,
            feature_2=feature_2,
            adorned_image=adorned_img,
            display_image=img_blend,
            downscale_image=img_downscale,
            distance_metres=Point(x_distance_m, y_distance_m)
        )

        return detection_result

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
        utils.draw_crosshairs(draw, mask, px, color=color)

    return mask


def draw_two_features(
        mask, feature_1, feature_2, color_1="red", color_2="green", line=False):
    """ Draw two detected features on the same mask, and optionally a line between

    args:
        mask: the detection mask with all classes (PIL.Image or np.array)
        feature_1: the first feature to draw (tuple)
        feature_2: the second feature to draw (tuple)
        line: flag to draw a line between features (bool)


    return:
        mask_combined: the detection mask with both features drawn
    """

    mask = draw_feature(mask, feature_1, color_1, crosshairs=False)
    mask = draw_feature(mask, feature_2, color_2, crosshairs=False)

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

def detect_right_edge(mask, color, threshold=25, left=False):
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
    color = (0, 255, 0)  # fixed color

    # TODO: extract filter from detection?
    mask_filt, px_filt = extract_class_pixels(mask, color)

    edge_px = detect_right_edge(mask, color, threshold=threshold)
    mask_draw = draw_feature(mask_filt, edge_px, color, crosshairs=True)
    needle_mask = draw_overlay(img, mask_draw)

    return edge_px, needle_mask


def detect_lamella_centre(img, mask, threshold=25):
    """Detect the centre of the lamella"""
    color = (255, 0, 0)  # fixed color

    # TODO: extract filter from detection?
    mask_filt, px_filt = extract_class_pixels(mask, color)

    centre_px = detect_centre_point(mask, color, threshold=threshold)
    mask_draw = draw_feature(mask_filt, centre_px, color, crosshairs=True)
    lamella_mask = draw_overlay(img, mask_draw)

    return centre_px, lamella_mask


def detect_lamella_edge(img, mask, threshold=25):
    """Detect the right edge of the lamella"""
    color = (255, 0, 0)  # fixed color

    mask_filt, px_filt = extract_class_pixels(mask, color)  # this is duplicate in detect_func
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
    edges = feature.canny(img, sigma=3)  # sigma higher usually better
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


def detect_thin_region(img, mask, top=True):
    """Detect the region of the lamella to thin"""

    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img)

    color = (255, 0, 0)
    threshold = 25
    min_height = 20  # minimum lamella thin height

    # detect centre
    centre_px, centre_mask = detect_lamella_centre(img, mask)

    # detect bounding box around lamella
    bbox = detect_bounding_box(mask, color, threshold=threshold)  # top, bot, left, right

    mask_draw = draw_feature(img, bbox[:2], color="blue", crosshairs=True)
    landing_mask = draw_overlay(img, mask_draw, alpha=0.5)

    # detect top and bottom edges
    w = bbox[3] - bbox[1]
    h = min(bbox[2] - bbox[0], min_height)  #

    # thin top and bottom edges to leave only small lamella slice
    # bbox method
    # top_px = bbox[0] + h/4
    # bottom_px = bbox[2] - h/4
    # left_px = bbox[1] + w/2

    # centre px method
    top_px = centre_px[0] - h / 6
    bottom_px = centre_px[0] + h / 6
    left_px = centre_px[1]

    # top and bottom cut start (left and up/down)
    top_thin_px = top_px, left_px
    bottom_thin_px = bottom_px, left_px

    if top:
        return top_thin_px, landing_mask
    else:
        return bottom_thin_px, landing_mask
    # TODO: this is fooled by small red specks need a way to aggregate a whole detection clump and ignore outliers...


def draw_final_detection_image(img: np.ndarray, feature_1_px, feature_2_px) -> np.ndarray:
    """Draw the final features on the image"""
    final_detection_img = Image.fromarray(img).convert("RGB")
    final_detection_img = draw_two_features(final_detection_img, feature_1_px, feature_2_px)
    final_detection_img = np.array(final_detection_img.convert("RGB"))
    return final_detection_img
