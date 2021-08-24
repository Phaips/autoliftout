#!/usr/bin/env python3


import streamlit as st
import plotly.express as px
import PIL
import numpy as np
import glob
from PIL import Image, ImageDraw
import yaml


st.header("Lamella Parameter Selection")


filenames = glob.glob("*.tif")
fname = filenames[0]

img = PIL.Image.open(fname)


with open("../protocol_liftout.yml") as f:
    settings = yaml.safe_load(f)


st.sidebar.subheader("Lamella Parameters")
lamella_height = st.sidebar.slider(
    "lamella_height", 1.0, 10.0, float(settings["lamella"]["lamella_height"]) * 1e6, step=0.5
)
lamella_width = st.sidebar.slider(
    "lamella_width", 5.0, 20.0, float(settings["lamella"]["lamella_width"]) * 1e6, step=0.5
)
total_cut_height = st.sidebar.slider(
    "total_cut_height", 1.0, 10.0, float(settings["lamella"]["total_cut_height"]) * 1e6, step=0.5
)
milling_depth = st.sidebar.slider(
    "milling_depth", 0.5, 5.0, float(settings["lamella"]["milling_depth"]) * 1e6, step=0.5
)


st.subheader("Stage 1 Parameters")
percentage_roi_height_stage_1 = st.slider(
    "percentage_roi_height_stage_1",
    0.0,
    1.0,
    settings["lamella"]["protocol_stages"][0]["percentage_roi_height"],
)
percentage_from_lamella_surface_stage_1 = st.slider(
    "percentage_from_lamella_surface_stage_1",
    0.0,
    1.0,
    settings["lamella"]["protocol_stages"][0]["percentage_from_lamella_surface"],
)

st.subheader("Stage 2 Parameters")
percentage_roi_height_stage_2 = st.slider(
    "percentage_roi_height_stage_2",
    0.0,
    1.0,
    settings["lamella"]["protocol_stages"][1]["percentage_roi_height"],
)
percentage_from_lamella_surface_stage_2 = st.slider(
    "percentage_from_lamella_surface_stage_2",
    0.0,
    1.0,
    settings["lamella"]["protocol_stages"][1]["percentage_from_lamella_surface"],
)

# select milling protocol parameters


# TODO: the box is calculated in microns which dont translate well to pixels
# need to find a way to move between...


# from centre of img, these are the offsets
#


def calc_milling_pattern(
    lamella_height,
    total_cut_height,
    percentage_from_lamella_surface,
    percentage_roi_height,
):

    center_y = (
        0
        + (0.5 * lamella_height)
        + (total_cut_height * percentage_from_lamella_surface)
        + (0.5 * total_cut_height * percentage_roi_height)
    )
    height = float(total_cut_height * percentage_roi_height)

    return center_y, height


def calc_milling_draw_px(center_y, height):
    LAMELLA_WIDTH_PX = lamella_width * 10
    LAMELLA_HEIGHT_PX = lamella_height * 10
    TOTAL_CUT_HEIGHT_PX = total_cut_height * 10
    CENTRE_Y_OFFSET_PX = center_y * 10
    CUT_HEIGHT_PX = height * 10

    CUT_BOTTOM_PX = CENTRE_Y_OFFSET_PX - CUT_HEIGHT_PX / 2
    CUT_TOP_PX = CENTRE_Y_OFFSET_PX + CUT_HEIGHT_PX / 2
    # st.write("CUT: ", CUT_BOTTOM_PX, CUT_TOP_PX)

    return LAMELLA_WIDTH_PX, LAMELLA_HEIGHT_PX, CUT_BOTTOM_PX, CUT_TOP_PX

def draw_milling_stage(
    draw, img_centre_x, img_centre_y, CUT_TOP_PX, CUT_BOTTOM_PX, color
):
    draw.rectangle(
        [
            (img_centre_x - LAMELLA_WIDTH_PX / 2, img_centre_y - CUT_TOP_PX),
            (img_centre_x + LAMELLA_WIDTH_PX / 2, img_centre_y - CUT_BOTTOM_PX),
        ],
        fill=color,
        outline=color,
    )

    draw.rectangle(
        [
            (img_centre_x - LAMELLA_WIDTH_PX / 2, img_centre_y + CUT_BOTTOM_PX),
            (img_centre_x + LAMELLA_WIDTH_PX / 2, img_centre_y + CUT_TOP_PX),
        ],
        fill=color,
        outline=color,
    )

# draw both stages as different colours..

rgbimg = Image.new("RGBA", img.size)
rgbimg.paste(img)

img_centre_x = rgbimg.size[0] // 2
img_centre_y = rgbimg.size[1] // 2

# create rectangle image
mask = PIL.Image.fromarray(np.zeros_like(rgbimg))
draw = ImageDraw.Draw(mask)

# stage 1
center_y, height = calc_milling_pattern(
    lamella_height,
    total_cut_height,
    percentage_from_lamella_surface_stage_1,
    percentage_roi_height_stage_1,
)
LAMELLA_WIDTH_PX, LAMELLA_HEIGHT_PX, CUT_BOTTOM_PX, CUT_TOP_PX = calc_milling_draw_px(
    center_y, height
)

draw_milling_stage(draw, img_centre_x, img_centre_y, CUT_TOP_PX, CUT_BOTTOM_PX, "red")

# stage 2
center_y, height = calc_milling_pattern(
    lamella_height,
    total_cut_height,
    percentage_from_lamella_surface_stage_2,
    percentage_roi_height_stage_2,
)
LAMELLA_WIDTH_PX, LAMELLA_HEIGHT_PX, CUT_BOTTOM_PX, CUT_TOP_PX = calc_milling_draw_px(
    center_y, height
)

draw_milling_stage(draw, img_centre_x, img_centre_y, CUT_TOP_PX, CUT_BOTTOM_PX, "blue")


# SHOW IMAGE
# blend image
alpha = 0.3
alpha_img = PIL.Image.blend(rgbimg, mask, alpha)

# make interactive
st.image(alpha_img, caption="test image", use_column_width=True)
