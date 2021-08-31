

from liftout.model import models
import numpy as np
import streamlit as st
import glob
import PIL
import matplotlib.pyplot as plt
import numpy as np

import skimage
from skimage import exposure
import os
from liftout import tools
import yaml

from liftout.detection.detection import *

import liftout

def gamma_correction(img, gam):
    gamma_corrected = exposure.adjust_gamma(img, gam)

    bins = 30
    bin_counts, bin_edges = np.histogram(img, bins)
    fig = plt.figure()
    plt.hist(img.ravel(), bins, color="blue", label="raw", alpha=0.5)
    plt.hist(gamma_corrected.ravel(), bins, color="red", label="gamma_adjusted", alpha=0.5)
    plt.legend(loc="best")

    return gamma_corrected, fig


st.set_page_config(page_title="Gamma Correction", layout="wide")
st.title("Gamma Correction Calibration")

protocol_path = os.path.join(os.path.dirname(liftout.__file__),"protocol_liftout.yml")

with open(protocol_path) as f:
    settings = yaml.safe_load(f)

default_path = os.path.join(os.path.dirname(tools.__file__), "20210823.153535_manual/20210823.153535_manual/img/")

path = st.sidebar.text_input("Image Directory", default_path)

filenames = sorted(glob.glob(path + "*.tif", recursive=True))

NUM_IMAGES = int(st.sidebar.number_input("Number of Images to display", 1, 
            len(filenames)-1, min(5, len(filenames))))

gam = st.sidebar.slider("gamma", 0.0, 5.0, 2.0)


weights_file = os.path.join(os.path.dirname(models.__file__), settings["machine_learning"]["weights"])
detector = Detector(weights_file=weights_file)


for fname in filenames[:NUM_IMAGES]:

    st.subheader(fname.split("/")[-1])
    img = np.array(PIL.Image.open(os.path.join(path, fname)))
    
    gamma_corrected, fig = gamma_correction(img, gam)

    mask = detector.detection_model.model_inference(np.asarray(gamma_corrected))


    cols = st.columns(4)
    cols[0].image(img, caption="raw")
    cols[1].image(gamma_corrected, caption="gamma correction")
    cols[2].pyplot(fig, caption="pixel intensity")
    cols[3].image(mask, caption="detection mask")


# save gamma value in protocol

st.sidebar.subheader("Save Protocol")
protocol_file_new = st.sidebar.text_input("Protocol File Name", "protocol_file_new.yml")
if st.sidebar.button("Save to protocol file"):
    settings["imaging"] = {}
    settings["imaging"]["gamma_correction"] = gam
    st.write(settings)
    st.sidebar.success(f"Saved to protocol file {protocol_file_new}")

    with open(protocol_file_new, "w") as file:
        yaml.dump(settings, file)


# TODO: fit the histogram for gamma adjustment to a set of 'good' images
# TODO: add gamma adjustment to images? where? in acquire_image?
# TODO: add detections to help calibrate?s