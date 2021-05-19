#!/usr/bin/env python3

import glob
from random import shuffle

import PIL
import streamlit as st
import numpy as np
import pandas as pd

# user functions
from utils import (
    load_model, model_inference, show_overlay,
    detect_and_draw_lamella_and_needle,
    scale_invariant_coordinates,
    calculate_distance_between_points,
    parse_metadata, match_filenames_from_path
)

@st.cache(allow_output_mutation=True)
def cached_streamlit_setup(images_path, weights_file):

    # load image filenames, randomise
    filenames = match_filenames_from_path(images_path, pattern=".tif")

    # load model
    model = load_model(weights_file=weights_file)

    return filenames, model


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


    
    # loop through images, and display masks, labels, and detections
    for fname in filenames[:n_images]:

        st.subheader(f"{fname}")

        # display setup
        cols_pred = st.beta_columns(2)
        cols_masks = st.beta_columns(3)
        cols_points = st.beta_columns(2)

        # model inference + display
        img, rgb_mask = model_inference(model, fname)

        # detect and draw lamella centre, and needle tip
        (
            lamella_centre_px,
            rgb_mask_lamella,
            needle_tip_px,
            rgb_mask_needle,
            rgb_mask_combined,
        ) = detect_and_draw_lamella_and_needle(rgb_mask, cols_masks)
        # TODO: this col masks still needs to be extracted out

        # scale invariant coordinatess
        scaled_lamella_centre_px, scaled_needle_tip_px = scale_invariant_coordinates(
            needle_tip_px, lamella_centre_px, rgb_mask_combined
        )

        # calculate distance between features
        (
            distance,
            vertical_distance,
            horizontal_distance,
        ) = calculate_distance_between_points(scaled_lamella_centre_px, scaled_needle_tip_px)

        # prediction overlay
        img_overlay = show_overlay(img, rgb_mask)

        # find label
        basename = fname.split("\\")[-1].split(".tif")[0]
        label_path = f"data/train/{basename}/label_viz.png"  # or label.png
        label_fname = glob.glob(label_path)

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
            cols_pred[0].image(
                img_overlay, caption=f"Predicted Mask ({fname})", use_column_width=True
            )

        # show masks
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

        # show overlay of detections
        img_overlay_c = show_overlay(img, rgb_mask_combined)
        st.image(
            img_overlay_c,
            caption=f"Needle/Lamella Detection ({fname})",
            use_column_width=True,
        )

        # show image metadata
        df = parse_metadata(fname)
        st.write("Image Metadata: ", df)

        try:
            st.write(f"Horizontal Distance: {horizontal_distance:.3f} px")
            st.write(f"Vertical Distance: {vertical_distance:.3f} px")
            st.write(f"Total Distance: {distance:.3f} px")

            # TODO: Check these values are coorect.
            rescaled_horizontal_distance = float(df["[Image].ResolutionY"].values[0]) * horizontal_distance
            rescaled_vertical_distance = float(df["[Image].ResolutionX"].values[0]) * vertical_distance

            horizontal_distance_microns = rescaled_horizontal_distance * float(df["[Scan].PixelWidth"].values[0]) *10e6
            vertical_distance_microns = rescaled_vertical_distance * float(df["[Scan].PixelHeight"].values[0]) * 10e6

            st.write(f"horizontal distance:  {horizontal_distance_microns}um")
            st.write(f"vertical distance:  {vertical_distance_microns}um") 

        except TypeError:
            pass




# TODO: might be worth using opening/morphology to clean up masks
# TODO: fix mixing and matching between passing draw and/or mask
# TODO: improve detecting the centre of lamella/ tip of needle (only basic atm)
# TODO: add metadata to evaluation display (e.g. beam type, angle, etc)
# TODO: add tracking file, is this detection correct yes/no. for data engine

