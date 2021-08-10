#!/usr/bin/env python3

import glob
from random import shuffle

import PIL
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
#
# user functions
from liftout.detection.detection import *
from liftout.detection.DetectionModel import *
from liftout.detection.utils import *


@st.cache(allow_output_mutation=True)
def cached_streamlit_setup(images_path, weights_file):

    # load image filenames, randomise
    filenames = match_filenames_from_path(images_path, pattern=".tif")

    # load detector
    detector = Detector(weights_file=weights_file)

    return filenames, detector

def main():
    st.title("AutoLiftout Model Evaluation")

    st.sidebar.subheader("Data Selection")
    images_path = st.sidebar.text_input("Image Path", "test_images/**/*")
    weights_file = st.sidebar.text_input("Model", "models/fresh_full_n10.pt")
    filenames, detector = cached_streamlit_setup(images_path, weights_file)

    st.sidebar.subheader("Filter Options")
    st.sidebar.write(f"{len(filenames)} filenames matched")
    n_files = st.sidebar.number_input("Number of Images ", 1, len(filenames), 5)
    shuffle(filenames)

    for fname in filenames[:n_files]:

        # load image from file
        img = load_image_from_file(fname)

        # model inference
        mask = detector.detection_model.model_inference(img)

        # individual detection modes
        lamella_mask, lamella_idx = extract_class_pixels(mask, color=(255, 0, 0)) # red
        needle_mask, needle_idx = extract_class_pixels(mask, color=(0, 255, 0)) # green

        feature_1_px, lamella_centre_detection = detect_lamella_centre(img, mask) # lamella_centre
        feature_2_px, needle_tip_detection = detect_needle_tip(img, mask)
        feature_3_px, lamella_edge_detection = detect_lamella_edge(img, mask)

        mask_combined = draw_two_features(mask, feature_1_px, feature_2_px)
        img_blend = draw_overlay(img, mask_combined, show=False)

        # show images
        st.subheader(fname)
        cols_raw = st.columns(2)
        cols_raw[0].image(img, caption="base_img")
        cols_raw[1].image(mask, caption="label_img") # TODO: replace with label if available?

        cols_mask = st.columns(2)
        cols_mask[0].image(lamella_mask, caption="lamella_mask")
        cols_mask[1].image(needle_mask, caption="needle_mask")

        cols_detection = st.columns(3)
        cols_detection[0].image(lamella_centre_detection, caption="lamella_centre")
        cols_detection[1].image(needle_tip_detection, caption="needle_tip")
        cols_detection[2].image(lamella_edge_detection, caption="lamella_edge")

        cols_combined = st.columns(2)
        cols_combined[0].image(mask_combined, caption="combined_mask")
        cols_combined[1].image(img_blend, caption="combined_overlay")


if __name__ == "__main__":
    main()


# TODO: add save images