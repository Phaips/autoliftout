#!/usr/bin/env python3


import glob
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms, utils
from skimage import feature
from scipy.spatial import distance


from utils import load_image, draw_crosshairs, scale_invariant_coordinates_NEW
from DetectionModel import DetectionModel

class Detector:

    def __init__(self, weights_file) -> None:

        self.detection_model = DetectionModel(weights_file)

        self.supported_shift_types = [
            "needle_tip_to_lamella_centre",
            "lamella_centre_to_image_centre",
            "lamella_edge_to_landing_post",
            "needle_tip_to_image_centre"
        ]
    
    def detect_features(self, img, mask, shift_type):

        if shift_type not in self.supported_shift_types:
            raise ValueError("ERROR: shift type calculation is not supported")

        # detect feature shift
        if shift_type=="needle_tip_to_lamella_centre":
            feature_1_px, lamella_mask = detect_lamella_centre(img, mask) # lamella_centre
            feature_2_px, needle_mask = detect_needle_tip(img, mask) # needle_tip

            feature_1_type = "lamella_centre"
            feature_2_type = "needle_tip"

        if shift_type=="lamella_centre_to_image_centre":
            feature_1_px, lamella_mask = detect_lamella_centre(img, mask) # lamella_centre
            feature_2_px = (mask.shape[0] // 2, mask.shape[1] // 2) # midpoint

            feature_1_type = "lamella_centre"
            feature_2_type = "image_centre"

        if shift_type=="lamella_edge_to_landing_post":
            # TODO: This doesnt work yet
            # TODO: The directions and shapes are wrong and messing things up needs to be fixed
            feature_1_px, lamella_mask = detect_lamella_edge(img, mask) # lamella_centre
            
            # need to resize image
            img_landing = Image.fromarray(img).resize((mask.shape[1], mask.shape[0]))

            landing_px=(img_landing.size[0]//2, img_landing.size[1]//2)

            print(img_landing.size)
            # print(mask.shape)
            print(landing_px)
            print("NOTE: landing not yet working")
            feature_2_px, landing_mask = detect_landing_edge(img_landing, landing_px) # landing post # TODO: initial landing point?

            feature_1_type = "lamella_edge"
            feature_2_type = "landing_post"

        if shift_type=="needle_tip_to_image_centre":
            feature_1_px = (mask.shape[0] // 2, mask.shape[1] // 2) # midpoint
            feature_2_px, needle_mask = detect_needle_tip(img, mask) # needle_tip

            feature_1_type = "image_centre"
            feature_2_type = "needle_tip"

        return feature_1_px, feature_1_type, feature_2_px, feature_2_type

    def calculate_shift_between_features(self, adorned_img, shift_type="needle_to_lamella_centre", show=False):
        """
        NOTE: img.data is np.array for Adorned Image

        """
        # TODO: fix display colours for this?

        if hasattr(adorned_img, 'data'):
            img = adorned_img.data # extract image data
        if isinstance(adorned_img, np.ndarray):
            img = adorned_img # adorned image is just numpy array

        # load image from file
        mask = self.detection_model.model_inference(img)

        # detect features for calculation
        feature_1_px, feature_1_type, feature_2_px, feature_2_type = self.detect_features(img, mask, shift_type)

        # display features for validation
        mask_combined = draw_two_features(mask, feature_1_px, feature_2_px)
        img_blend = draw_overlay(img, mask_combined, show=show)

        # # need to use the same scale images for both detection selections
        img_downscale = Image.fromarray(img).resize((mask_combined.size[0], mask_combined.size[1]))

        # validate detection
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




# Detection, Drawing helper functions

def select_point_new(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    coords = []

    def on_click(event):
        print(event.xdata, event.ydata)
        coords.append(event.ydata)
        coords.append(event.xdata)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    return tuple(coords[-2:])

def validate_detection(img, img_base, detection_coord, det_type):
    correct = input(f"Is the detection for {det_type} correct? (y/n)")
    #TODO: change this to user_input
    if correct == "n":

        detection_coord_initial = detection_coord # save initial coord

        print(f"Please click the {det_type} position")
        detection_coord = select_point_new(img)

        # TODO: need to resolve behaviour when user exits plot without selecting?
        # if detection_coord is None:
        #     detection_coord = detection_coord # use initial coord


        # save image for training here
        print("Saving image for labelling")
        #storage.step_counter +=1
        #storage.SaveImage(img_base, id="label_")


    print(f"{det_type}: {detection_coord}")
    return detection_coord


def load_image_from_live(img):

    """ Load a live image from the microscope as np.array """

    return np.asarray(img.data)


def load_image_from_file(fname):

    """ Load a .tif image from disk as np.array """

    img = np.asarray(Image.open(fname))

    return img


# REFACTOR DETECTION AND DRAWING TOOLS

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
    
    # required for blending
    img = img.convert("RGB") 
    mask = mask.convert("RGB") 

    # blend images together
    alpha_blend = PIL.Image.blend(img, mask, alpha)

    if show:
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

    print("Dist: ", px, landing_px, dst)
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



        

####################### # UNUSED #######################

# def calculate_needletip_shift_from_lamella_centre(adorned_img, show=False):
#     """
#     NOTE: img.data is np.array for Adorned Image

#     """

#     if hasattr(adorned_img, 'data'):
#         img = adorned_img.data # extract image data
#     if isinstance(adorned_img, np.ndarray):
#         img = adorned_img # adorned image is just numpy array

#     weights_file = r"\\ad.monash.edu\home\User007\prcle2\Documents\demarco\autoliftout\patrick\models\fresh_full_n10.pt"
#     detector = Detector(weights_file) # TODO: wrap in class

#     # load image from file
#     mask = detector.detection_model.model_inference(img)

#     # detect features
#     needle_tip_px, needle_mask = detect_needle_tip(img, mask)
#     lamella_centre_px, lamella_mask = detect_lamella_centre(img, mask)

#     # display features
#     mask_combined = draw_two_features(mask, lamella_centre_px, needle_tip_px)
#     alpha_blend = draw_overlay(img, mask_combined, show=show)

#     # # need to use the same scale images for both detection selections
#     img_downscale = Image.fromarray(img).resize((mask_combined.size[0], mask_combined.size[1]))

#     # validate detection
#     needle_tip_px = validate_detection(img_downscale, img, needle_tip_px, "needle tip")
#     lamella_centre_px = validate_detection(img_downscale, img, lamella_centre_px, "lamella_centre")

#     # scale invariant coordinatesss
#     scaled_lamella_centre_px = scale_invariant_coordinates_NEW(lamella_centre_px, mask_combined)
#     scaled_needle_tip_px = scale_invariant_coordinates_NEW(needle_tip_px, mask_combined)

#     # if no detection is found, something has gone wrong
#     if scaled_needle_tip_px is None or scaled_lamella_centre_px is None:
#         raise ValueError("No detections available")

#     # # x, y
#     return -(scaled_lamella_centre_px[1] - scaled_needle_tip_px[1]), scaled_lamella_centre_px[0] - scaled_needle_tip_px[0]


