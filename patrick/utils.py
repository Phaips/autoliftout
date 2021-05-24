# /usr/bin/env python3

import glob
import json
import re
from random import shuffle

# import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import segmentation_models_pytorch as smp
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms, utils

# user functions

# transformations
transformation = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((1024 // 4, 1536 // 4)),
        transforms.ToTensor(),
    ]
)


def load_image(fname):

    """ Load a .tif image from disk as np.array """

    img = np.asarray(Image.open(fname))

    return img


def preprocess_image(img, transformation):
    """ preprocess an image for model inference """

    return transformation(img).unsqueeze(0)


def load_model(weights_file):
    """ helper function for loading model"""

    # load model
    model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=3,)
    # load model weights
    model.load_state_dict(torch.load(weights_file, map_location="cpu"))
    model.eval()

    return model


# @st.cache
def model_inference(model, fname, img=None):

    """
        Helper function to run the image through model,
        and return image and predicted mask
    """
    if img is None:
        # load selected image
        img = load_image(fname)

    # pre-process image (+ batch dim)
    img_t = preprocess_image(img=img, transformation=transformation)

    # model inference
    output = model(img_t)

    # calculate mask
    rgb_mask = decode_output(output)

    return img, rgb_mask


def draw_mask(fname):
    """ Helper function for drawing the label mask directly from .json"""

    basename = fname.split(".tif")[0]
    label_fname = basename + ".json"

    with open(label_fname) as json_file:
        label = json.load(json_file)

    # label not showing???
    img = np.asarray(PIL.Image.open(fname), dtype=np.uint8)

    im = Image.fromarray(img)
    im = im.convert("RGBA")
    draw = ImageDraw.Draw(im)

    for shape in label["shapes"]:
        pts = shape["points"]
        class_label = shape["label"]

        if class_label == "needle":
            col = (0, 255, 0, 127)
        else:
            col = (255, 0, 0, 127)
        pts = [tuple(pt) for pt in pts]

        draw.polygon(pts, fill=col, outline=col)

    return im


def decode_output(output):
    """decodes the output of segmentation model to RGB mask"""
    output = F.softmax(output, dim=1)
    mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    mask = decode_segmap(mask)
    return mask


def decode_segmap(image, nc=3):

    """ Decode segmentation class mask into an RGB image mask"""

    # 0=background, 1=lamella, 2= needle
    label_colors = np.array([(0, 0, 0), (255, 0, 0), (0, 255, 0)])

    # pre-allocate r, g, b channels as zero
    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)

    # apply the class label colours to each pixel
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    # stack rgb channels to form an image
    rgb_mask = np.stack([r, g, b], axis=2)
    return rgb_mask


def show_overlay(img, rgb_mask):

    # convert to np if required
    if isinstance(rgb_mask, PIL.Image.Image):
        rgb_mask = np.asarray(rgb_mask)

    # resize img to same size as mask
    # img_shape = (rgb_mask.shape[1], rgb_mask.shape[0])
    # img_resize_np = cv2.resize(img, img_shape, interpolation=cv2.INTER_AREA)

    # # add transparent overlay
    # alpha = 0.4
    # img_rgb = cv2.cvtColor(img_resize_np, cv2.COLOR_GRAY2RGB)
    # img_overlay = cv2.addWeighted(img_rgb, 1 - alpha, rgb_mask, alpha, 1)

    return rgb_mask

def calculate_distance_between_points(lamella_centre_px, needle_tip_px):

    distance, vertical_distance, horizontal_distance = None, None, None

    if lamella_centre_px and needle_tip_px:

        # vertical and horizontal distance
        vertical_distance = lamella_centre_px[0] - needle_tip_px[0]
        horizontal_distance = lamella_centre_px[1] - needle_tip_px[1]

        # calculate distance between needle point and lamella centre
        distance = np.sqrt(
            np.power(vertical_distance, 2) +
            np.power(horizontal_distance, 2)
        )
    return distance, vertical_distance, horizontal_distance


def get_rectangle_coordinates(px, RECT_WIDTH=4):
    """ helper function for getting a rectangle centred on a pixel"""
    rect_px = (
        px[1] + RECT_WIDTH // 2,
        px[0] + RECT_WIDTH // 2,
        px[1] - RECT_WIDTH // 2,
        px[0] - RECT_WIDTH // 2,
    )

    return rect_px


def detect_lamella_centre(px):

    """detect lamella centre from list of lamella pixels, idx"""
    # # convert mask to coordinates
    # px = list(zip(idx[0], idx[1]))

    # get centre of lamella detections
    x_mid = int(np.mean(px[0]))
    y_mid = int(np.mean(px[1]))

    lamella_centre_px = x_mid, y_mid

    return lamella_centre_px


def draw_rectangle_feature(mask, px, RECT_WIDTH=4):

    # rectangle coordinates
    rect_px = get_rectangle_coordinates(px, RECT_WIDTH)

    # draw mask
    draw = PIL.ImageDraw.Draw(mask)
    draw.rectangle(rect_px, fill="white", width=5)

    return mask


def draw_lamella_centre(rgb_mask_l, lamella_centre_px=None):

    """draw the centre of the lamella on the rgb mask
    args:
        rgb_mask_l: rgb_mask/image to draw on
        lamella_centre_px: coordinates (in px) of lamella centre

    returns
        rgb_mask_l: rgb_mask with lamella centre drawn on

    """

    # draw rectangle on lamella centre
    rgb_mask_l = draw_rectangle_feature(rgb_mask_l, lamella_centre_px)

    return rgb_mask_l


def detect_needle_tip(idx):

    """detect needle tip from list of needle pixels"""
    # convert mask to coordinates
    px = list(zip(idx[0], idx[1]))

    # get index of max value
    max_idx = np.argmax(idx[1])
    needle_tip_px = px[max_idx]  # needle tip px

    return needle_tip_px


def draw_needle_tip(rgb_mask_n, needle_tip_px=None):

    """draw the needle tip on the rgb mask"""

    # rectangle coordinates
    rect_px = get_rectangle_coordinates(needle_tip_px)

    # draw mask
    draw = PIL.ImageDraw.Draw(rgb_mask_n)
    draw.rectangle(rect_px, fill="white", width=5)

    return rgb_mask_n


def draw_crosshairs(draw, mask, idx, color="white"):
    """ helper function for drawing crosshairs on an image"""
    draw.line([0, idx[0], mask.size[0], idx[0]], color)
    draw.line([idx[1], 0, idx[1], mask.size[1]], color)


def draw_needle_and_lamella(rgb_mask_c, needle_tip_px=None, lamella_centre_px=None):
    """ helper function for drawing needle and lamella features on an image/mask"""

    # set draw object
    draw = PIL.ImageDraw.Draw(rgb_mask_c)

    # if needle detected, draw
    if needle_tip_px:

        # draw cross-hairs
        draw_crosshairs(draw=draw, mask=rgb_mask_c, idx=needle_tip_px, color="green")

        # draw needle tip
        rgb_mask_c = draw_needle_tip(rgb_mask_c, needle_tip_px=needle_tip_px)

    # if lamella detected, draw
    if lamella_centre_px:

        # draw cross-hairs
        draw_crosshairs(draw, rgb_mask_c, idx=lamella_centre_px, color="red")

        # # draw lamella centre on mask
        rgb_mask_c = draw_lamella_centre(rgb_mask_c, lamella_centre_px)

    # if needle and lamella detected, draw line
    if needle_tip_px and lamella_centre_px:

        # draw line between needle and lamella
        draw.line(
            [
                needle_tip_px[1],
                needle_tip_px[0],
                lamella_centre_px[1],
                lamella_centre_px[0],
            ],
            fill="white",
            width=1,
        )

    return rgb_mask_c


def detect_and_draw_lamella_and_needle(rgb_mask):
    """ Detect the lamella centre and needle.

    args:
        rgb_mask: mask of needle, and lamella detections

    returns:
        lamella_centre_px: pixel coordinate of lamella centre
        needle_tip_px:  pixel coordinate of needle tip
        rgb_mask_l: rgb_mask showing lamella centre
        rgb_mask_n: rgb_mask showing needle tip
        rgb_mask_c: rgb_mask showing both lamella, needle and line
    """
    # Mask separation
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
    labels = ["background", "lamella", "needle"]

    # masks
    rgb_mask_l = PIL.Image.fromarray(rgb_mask)
    rgb_mask_n = PIL.Image.fromarray(rgb_mask)
    lamella_centre_px = None
    needle_tip_px = None

    for i, (col, label) in enumerate(zip(colors, labels)):

        # extract only label pixels to find edges
        mask_copy = np.zeros_like(rgb_mask)
        idx = np.where(np.all(rgb_mask == col, axis=-1))
        mask_copy[idx] = col

        # find lamella centre
        if label == "lamella" and len(idx[0]) > 25:

            # detect lamella centre and draw
            lamella_centre_px = detect_lamella_centre(idx)
            rgb_mask_l = draw_lamella_centre(
                rgb_mask_l=rgb_mask_l, lamella_centre_px=lamella_centre_px
            )

        # find needle tip:
        if label == "needle" and len(idx[0]) > 200:

            # show needle tip
            needle_tip_px = detect_needle_tip(idx)
            rgb_mask_n = draw_needle_tip(
                rgb_mask_n=rgb_mask_n, needle_tip_px=needle_tip_px
            )

    # draw both needle and lamella and line between
    rgb_mask_c = PIL.Image.fromarray(rgb_mask)
    rgb_mask_c = draw_needle_and_lamella(
        rgb_mask_c, needle_tip_px=needle_tip_px, lamella_centre_px=lamella_centre_px
    )

    return lamella_centre_px, rgb_mask_l, needle_tip_px, rgb_mask_n, rgb_mask_c


def scale_invariant_coordinates(
    needle_tip_px=None, lamella_centre_px=None, rgb_mask_combined=None
):
    """ Return the scale invariant coordinates of the features in the given mask """

    scaled_lamella_centre_px = None
    scaled_needle_tip_px = None

    if lamella_centre_px:

        scaled_lamella_centre_px = (
            lamella_centre_px[0] / rgb_mask_combined.size[1],
            lamella_centre_px[1] / rgb_mask_combined.size[0],
        )

    if needle_tip_px:

        scaled_needle_tip_px = (
            needle_tip_px[0] / rgb_mask_combined.size[1],
            needle_tip_px[1] / rgb_mask_combined.size[0],
        )

    return scaled_lamella_centre_px, scaled_needle_tip_px


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


def match_filenames_from_path(filepath, pattern=".tif"):

    # load image filenames, randomise
    filenames = sorted(glob.glob(filepath + ".tif"))
    shuffle(filenames)

    return filenames