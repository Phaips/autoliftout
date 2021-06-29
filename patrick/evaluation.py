#!/usr/bin/env python3

import glob
from patrick.detection import Detector
from random import shuffle

import PIL
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# user functions
from utils import load_image, parse_metadata, match_filenames_from_path
from detection import *
from DetectionModel import *

@st.cache()
def cached_streamlit_setup(images_path, weights_file, device="cpu"):

    # load image filenames, randomise
    filenames = match_filenames_from_path(images_path, pattern=".tif")

    # load detector
    weights_file = "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/patrick/test_model.pt"
    detector = Detector(weights_file=weights_file)

    return filenames, detector


def streamlit_setup():
    """ Helper function for setting up streamlit app"""

    # streamlit options
    st.title("Liftout Project")
    st.markdown(
        """App for evaluating the predicted segmenation mask for the needle and lamella.
            Shows a random selection of images, and corresponding masks, labels, and detections."""
    )

    # select images
    st.sidebar.subheader("Dataset Options")
    images_path = st.sidebar.text_input("Select a image dataset", "data/train/raw/*")
    
    # select model
    model_paths = sorted(glob.glob("models/*.pt"))
    weights_file = st.sidebar.selectbox("Select a model", model_paths, 1)

    # load with caching for performance
    filenames, model = cached_streamlit_setup(images_path, weights_file)


    # select number of images to show
    n_images = st.sidebar.number_input(
        "Select a number of images to display", 1, len(filenames) - 1, 1
    )

    return filenames, n_images, model

@st.cache
def extract_img_metadata(filenames):
    

    df_metadata = pd.DataFrame()

    for filename in filenames:

        df = parse_metadata(filename)

        df_metadata = df_metadata.append(df, ignore_index=True)

    # cleanup
    df_metadata["[Stage].StageT"] = round(df_metadata["[Stage].StageT"].apply(float), 4)

    # save metadata
    # df_metadata.to_csv("meta_data.csv")
    # st.success("Metadata file saved.")

    return df_metadata


@st.cache
def filter_data(df_metadata, filter_col, filter_val):
    
    # filter datas
    df_metadata = df_metadata[df_metadata[filter_col] == filter_val]

    # filter filenames
    filenames = list(df_metadata["filename"])

    return df_metadata, filenames

def extract_metadata_and_filter(filenames):
    
    st.sidebar.subheader("Filter Options")

    # extract all metadata
    df_metadata = extract_img_metadata(filenames)

    # select a filter column
    filter_col = st.sidebar.selectbox("Select a column to filter metadata.", df_metadata.columns)

    # select a filter value
    filter_val = st.sidebar.selectbox(
        "Select a filter value", df_metadata[filter_col].unique()
    )

    # filter data
    df_metadata, filenames = filter_data(df_metadata, filter_col, filter_val)

    st.sidebar.write(
        f"{len(df_metadata)} images selected." # for {filter_val} ({filter_col})."
    )

    return df_metadata, filenames


if __name__ == "__main__":

    # setup for streamlit app
    filenames, n_images, model = streamlit_setup()

    if st.sidebar.checkbox("Filter Data?"):
        # extract metadata and filter
        df_metadata, filenames = extract_metadata_and_filter(filenames)
    

    # saving images for evaluation
    st.sidebar.subheader("Saving Options")
    _save_prefix = st.sidebar.text_input("Image prefix")
    SAVE_IMAGES_FOR_EVAL = True if st.sidebar.checkbox("Save Images for Evaluation") else False
    

    # loop through images, and display masks, labels, and detections
    for i, fname in enumerate(filenames[:n_images+1]):

        st.subheader(f"{fname}")

        # display setup
        cols_pred = st.beta_columns(2)
        cols_points = st.beta_columns(2)

        # load image from file
        img = load_image_from_file(fname)

        # model inference + display
        img, rgb_mask = model_inference(model, img)

        # detect and draw lamella centre, and needle tip
        (
            lamella_centre_px,
            rgb_mask_lamella,
            needle_tip_px,
            rgb_mask_needle,
            rgb_mask_combined,
        ) = detect_and_draw_lamella_and_needle(rgb_mask)

        # scale invariant coordinatess
        scaled_lamella_centre_px, scaled_needle_tip_px = scale_invariant_coordinates(
            needle_tip_px, lamella_centre_px, rgb_mask_combined
        )

        # prediction overlay
        img_overlay = show_overlay_streamlit(img, rgb_mask)

        # find label
        basename = fname.split("/")[-1].split(".tif")[0] # TODO: fix path so it is universal
        if "train" in fname:
            label_path = f"data/train/{basename}/label_viz.png"  # or label.png
            label_fname = glob.glob(label_path)
        elif "aug" in fname:
            label_path = f"data/aug/{basename}/label_viz.png"  # or label.png
            label_fname = glob.glob(label_path)
        else:
            label_fname = None
            

        # if label exists, show both pred, and label
        if label_fname:
            label_img = PIL.Image.open(label_path).convert("RGB")

            # show prediction and label
            cols_pred[0].image(
                img_overlay, caption=f"Predicted Mask ({fname})", use_column_width=True
            )
            cols_pred[1].image(
                label_img, caption=f"Label ({label_path})", use_column_width=True
            )
        else:
            # just show prediction
            cols_pred[0].image(
                img_overlay, caption=f"Predicted Mask ({fname})", use_column_width=True
            )

        # show invidual detection masks and points
        lamella_caption, needle_caption = "Lamella Middle (None)", "Needle Tip (None)"
        if scaled_lamella_centre_px:
            lamella_caption = f"Lamella Middle ({scaled_lamella_centre_px[0]:.2f}, {scaled_lamella_centre_px[1]:.2f})"

        if scaled_needle_tip_px:
            needle_caption = f"Needle Tip ({scaled_needle_tip_px[0]:.2f}, {scaled_needle_tip_px[1]:.2f})"

        cols_points[0].image(
            rgb_mask_lamella, caption=lamella_caption, use_column_width=True
        )
        cols_points[1].image(
            rgb_mask_needle, caption=needle_caption, use_column_width=True
        )

        # show overlay of both detections
        img_overlay_c = show_overlay_streamlit(img, rgb_mask_combined)
        st.image(
            img_overlay_c,
            caption=f"Needle/Lamella Detection ({fname})",
            use_column_width=True,
        )

        # draw a line between the lamella centre and centre of the image.
        if lamella_centre_px:
            mask_centre = draw_line_between_lamella_and_img_centre(rgb_mask_lamella, lamella_centre_px)
            centre_overlay = show_overlay_streamlit(img, mask_centre)
            fname_base = fname.split("\\")[-1]
            st.image(centre_overlay, caption=f"Centre Distance ({fname_base})", use_column_width=True)

        # show image metadata
        df = parse_metadata(fname)
        st.write("Image Metadata: ", df)


        if SAVE_IMAGES_FOR_EVAL:
            
            # datetime object containing current date and time
            now = datetime.now()
            # format
            # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S") + f"
            
            img_fname =f"example/tmp/{_save_prefix}_0{i:0d}.png"
            PIL.Image.fromarray(img_overlay_c).save(img_fname)
            st.success(f"Image saved as {img_fname}")




# TODO: might be worth using opening/morphology to clean up masks
# TODO: fix mixing and matching between passing draw and/or mask
# TODO: improve detecting the centre of lamella/ tip of needle (only basic atm)
# TODO: add metadata to evaluation display (e.g. beam type, angle, etc)
# TODO: add tracking file, is this detection correct yes/no. for data engine

