import glob
import os

import liftout
import matplotlib.pyplot as plt
import numpy as np
import PIL
import skimage
import streamlit as st
import yaml
from liftout import tools
from liftout.detection.detection import *
from liftout.model import models
from skimage import exposure



def gamma_correction(img, gam, mean, median):
    gamma_corrected = exposure.adjust_gamma(img, gam)

    bins = 30
    bin_counts, bin_edges = np.histogram(img, bins)
    fig = plt.figure()
    plt.hist(img.ravel(), bins, color="blue", label="raw", alpha=0.5)
    plt.hist(
        gamma_corrected.ravel(), bins, color="red", label="gamma_adjusted", alpha=0.5
    )
    plt.axvline(median, color='k', linestyle='solid', linewidth=1, label = "median")
    plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label="mean")
    plt.legend(loc="best")

    return gamma_corrected, fig


st.set_page_config(page_title="Gamma Correction", layout="wide")
st.title("Gamma Correction Calibration")

protocol_path = os.path.join(os.path.dirname(liftout.__file__), "protocol_liftout.yml")

with open(protocol_path) as f:
    settings = yaml.safe_load(f)

default_path = os.path.join(
    os.path.dirname(liftout.__file__),
    "log", "run", "20210830.171910", "img"
)
# TODO: make directory selectable.
path = st.sidebar.text_input("Image Directory", default_path)

filenames = sorted(glob.glob(path + "*.tif", recursive=True))

from random import shuffle
shuffle(filenames)

if len(filenames) > 0:

    st.sidebar.write(f"{len(filenames)} images found.")
    NUM_IMAGES = int(
        st.sidebar.number_input(
            "Number of Images to display", 1, len(filenames), min(5, len(filenames))
        )
    )

    weights_file = os.path.join(
        os.path.dirname(models.__file__), settings["machine_learning"]["weights"]
    )
    detector = Detector(weights_file=weights_file)

    min_gamma = st.sidebar.slider("min_gamma", 0.1, 0.5, 0.15)
    max_gamma = st.sidebar.slider("max_gamma", 1.5, 3.0, 2.0)
    gamma_threshold = st.sidebar.slider("gamma_threshold", 25, 100, 46)
    gamma_scale = st.sidebar.slider("gamma_scale", 0.01, 0.1, 0.01)

    for fname in filenames[:NUM_IMAGES]:

        st.subheader(fname.split("/")[-1])
        img = np.array(PIL.Image.open(os.path.join(path, fname)))

        std = np.std(img)
        mean = np.mean(img)
        median = np.median(img)
        diff = mean - 255/2.
        gam = np.clip(min_gamma, 1 + diff * gamma_scale, max_gamma)
        if abs(diff) < gamma_threshold:
            gam = 1.0
        st.write(f"mean: {mean:.3f}, median: {median}, std: {std:.3f}, diff: {diff:.3f}, gam: {gam:.3f}")

        gamma_corrected, fig = gamma_correction(img, gam, mean, median)

        raw_mask = detector.detection_model.model_inference(np.asarray(img))
        raw_blend = draw_overlay(img, raw_mask)

        gamma_mask = detector.detection_model.model_inference(np.asarray(gamma_corrected))
        gamma_blend = draw_overlay(gamma_corrected, gamma_mask)

        cols = st.columns(5)
        cols[0].image(img, caption="raw_image")
        cols[1].image(gamma_corrected, caption="gamma correction")
        cols[2].pyplot(fig, caption="pixel intensity")
        cols[3].image(raw_blend, caption="raw detection mask")
        cols[4].image(gamma_blend, caption="gamma detection mask")
