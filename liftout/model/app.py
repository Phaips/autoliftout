#!/usr/bin/env python3

import glob
from random import shuffle
import os
import PIL
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
#
# user functions

from liftout.detection.DetectionModel import DetectionModel
from fibsem.detection import utils, detection


@st.cache(allow_output_mutation=True)
def cached_streamlit_setup(images_path, weights_file, sort=True):

    # load image filenames, randomise
    filenames = sorted(glob.glob(os.path.join(images_path, "*.tif")))

    if sort is False:
        shuffle(filenames)

    # load detector
    detector = DetectionModel(weights_file=weights_file)

    return filenames, detector

def main():
    st.title("AutoLiftout Model Evaluation")

    st.sidebar.subheader("Data Selection")
    images_path = st.sidebar.text_input("Image Path", "test_images/**/*")
    weights_file = st.sidebar.text_input("Model", "models/boost_n05_model.pt")
    filenames, detector = cached_streamlit_setup(images_path, weights_file)

    st.sidebar.subheader("Filter Options")
    st.sidebar.write(f"{len(filenames)} filenames matched")
    n_files = st.sidebar.number_input("Number of Images ", 1, len(filenames), 5)
    shuffle(filenames)

    for fname in filenames[:n_files]:

        # load image from file
        img = utils.load_image_from_file(fname)

        # model inference
        mask = detector.detection_model.inference(img)

        # individual detection modes
        lamella_mask, lamella_idx = detection.extract_class_pixels(mask, color=(255, 0, 0)) # red
        needle_mask, needle_idx = detection.extract_class_pixels(mask, color=(0, 255, 0)) # green

        feature_1_px, lamella_centre_detection = detection.detect_centre_point(mask, (255, 0, 0)) # lamella_centre
        feature_2_px, needle_tip_detection = detection.detect_right_edge(mask, (0, 255, 0))
        feature_3_px, lamella_edge_detection = detection.detect_closest_edge(mask, (mask.shape[1]//2, mask.shape[0] // 2))

        img_blend = detection.draw_overlay(img, mask)

        # show images
        st.subheader(fname)
        cols_raw = st.columns(2)
        cols_raw[0].image(img, caption="base_img")
        cols_raw[1].image(mask, caption="label_img")

        cols_mask = st.columns(2)
        cols_mask[0].image(lamella_mask, caption="lamella_mask")
        cols_mask[1].image(needle_mask, caption="needle_mask")

        cols_detection = st.columns(3)
        cols_detection[0].image(lamella_centre_detection, caption="lamella_centre")
        cols_detection[1].image(needle_tip_detection, caption="needle_tip")
        cols_detection[2].image(lamella_edge_detection, caption="lamella_edge")

        cols_combined = st.columns(2)
        cols_combined[0].image(mask, caption="combined_mask")
        cols_combined[1].image(img_blend, caption="combined_overlay")


if __name__ == "__main__":
    main()