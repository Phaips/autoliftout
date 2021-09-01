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



def gamma_correction(img, gam):
    gamma_corrected = exposure.adjust_gamma(img, gam)

    bins = 30
    bin_counts, bin_edges = np.histogram(img, bins)
    fig = plt.figure()
    plt.hist(img.ravel(), bins, color="blue", label="raw", alpha=0.5)
    plt.hist(
        gamma_corrected.ravel(), bins, color="red", label="gamma_adjusted", alpha=0.5
    )
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
            "Number of Images to display", 1, len(filenames) - 1, min(5, len(filenames))
        )
    )

    gam = st.sidebar.slider("gamma", 0.0, 5.0, 2.0)


    weights_file = os.path.join(
        os.path.dirname(models.__file__), settings["machine_learning"]["weights"]
    )
    detector = Detector(weights_file=weights_file)


    for fname in filenames[:NUM_IMAGES]:

        st.subheader(fname.split("/")[-1])
        img = np.array(PIL.Image.open(os.path.join(path, fname)))
        raw_img = img
        gamma_corrected, fig = gamma_correction(img, gam)

        raw_mask = detector.detection_model.model_inference(np.asarray(img))
        raw_blend = draw_overlay(img, raw_mask, show=False)

        gamma_mask = detector.detection_model.model_inference(np.asarray(gamma_corrected))
        gamma_blend = draw_overlay(gamma_corrected, gamma_mask, show=False)

        # std = np.std(img)
        # mean = np.mean(img)
        #
        # st.write(f"mean: {mean}, std: {std}")
        # histogram, bin_edges = np.histogram(img, bins=256)
        #
        # if mean > (255 - std/2):
        #     st.write("Mean too high, clipping to zero")
        #     indices = np.where(img < mean)
        #     img[indices] = 0
        # elif (mean<(255/2. - std/2) and (mean < 2*std)):
        #     st.write("Mean too low, clipping to 255")
        #     indices = np.where(img > mean)
        #     img[indices] = 255
        #
        # bins =256
        # fig = plt.figure()
        # plt.hist(raw_img.ravel(), bins, color="blue", label="raw", alpha=0.5)
        # plt.hist(
        #     gamma_corrected.ravel(), bins, color="red", label="gamma_adjusted", alpha=0.5
        # )
        # plt.hist(
        #     img.ravel(), bins, color="black", label="clipped_image", alpha=0.5
        # )
        # plt.legend(loc="best")


        cols = st.columns(6)
        cols[0].image(raw_img, caption="raw_image")
        cols[1].image(img, caption="clipped image")
        cols[2].image(gamma_corrected, caption="gamma correction")
        cols[3].pyplot(fig, caption="pixel intensity")
        cols[4].image(raw_blend, caption="raw detection mask")
        cols[5].image(gamma_blend, caption="gamma detection mask")

    # save gamma value in protocol
    st.sidebar.subheader("Save Protocol")
    protocol_file_new = st.sidebar.text_input("Protocol File Name", "protocol_file_new.yml")
    if st.sidebar.button("Save to protocol file"):
        # settings["imaging"] = {}
        settings["imaging"]["gamma_correction"] = gam
        st.write(settings)
        st.sidebar.success(f"Saved to protocol file {protocol_file_new}")

        with open(protocol_file_new, "w") as file:
            yaml.dump(settings, file, sort_keys=False)


# TODO: fit the histogram for gamma adjustment to a set of 'good' images
# TODO: probably should have two separate gamma corrections for eb / ib?

"""
# TODO: initial setup:
# - show autocontrasted images:
- ask if it is ok
    - if not show gamma corrected autocontrast
        - allow changing gamma here?
- if still not ok, allow them to set manual brightness / contrast?
- and gamma correction?
"""

#
# STD = np.std(img_orig)
# MEAN  = np.mean(img_orig)
#
# print('STD = ', STD, '; MEAN = ', MEAN)
#
# histogram, bin_edges = np.histogram(img_orig, bins=256)
# plt.figure(11)
# plt.plot(bin_edges[0:-1], histogram, 'ob')  # <- or here
#
# if MEAN > (255 - STD/2):
#     print('1---------- doing renormalisation, recalibration and padding ----------------- ')
#     indices = np.where(img_orig < MEAN  )
#     img_orig[indices] = 0
# elif (MEAN<(255/2. - STD/2) and (MEAN < 2*STD)):
#     print('2---------- doing renormalisation, recalibration and padding ----------------- ')
#     indices = np.where(img_orig > MEAN  )
#     img_orig[indices] = 255
#
# elif (MEAN<(255/2. - STD/2) and MEAN>=STD/2) or (MEAN>=(255/2. + STD/2) and MEAN <= (255 - STD/2)):
#     print('3---------- doing renormalisation, recalibration and padding ----------------- ')
#     # normalisation
#     img_sigma = np.std(img_orig)
#     img_mean  = np.mean(img_orig)
#     img_orig =  (img_orig  - img_mean) / img_sigma
#
#     ### filtering begin
#     img_orig = img_orig - img_orig.min()
#     img_sigma = np.std(img_orig)
#     img_mean  = np.mean(img_orig)
#
#     if MEAN<(255/2. - STD/2):
#         indices = np.where( img_orig >= (img_mean + 2*img_sigma) )
#     if MEAN>=(255/2. + STD/2):
#         indices = np.where( img_orig <= (img_mean - 2*img_sigma) )
#     img_orig[indices] = img_mean
#     img_orig = img_orig/(img_mean + 2*img_sigma )  * 255
#     img_orig = img_orig.astype(np.uint8)
#     #### filtering end
#     histogram3, bin_edges3 = np.histogram(img_orig, bins=256)
#
#
#     ### renormalisation
#     img_sigma = np.std(img_orig)
#     img_mean  = np.mean(img_orig)
#     img_orig =  (img_orig  - img_mean) / img_sigma
#     #### BINNING, PADDING
#     scale_factor = 1
#     #image_resized = resize(img_orig, (img_orig.shape[0] // 2, img_orig.shape[1] // 2),  anti_aliasing=False)
#     #image_resized = rescale(img_orig, 0.50, anti_aliasing=True)
#     image_resized = skimage.transform.downscale_local_mean(img_orig, (scale_factor,scale_factor), cval=0, clip=True)
#     cmask = circ_mask(size=(image_resized.shape[1], image_resized.shape[0]), radius=image_resized.shape[0]//scale_factor-80, sigma=20)  # circular mask
#     img_orig = image_resized * cmask
#     if scale_factor > 1:
#         ddx = image_resized.shape[0]//scale_factor
#         ddy = image_resized.shape[1]//scale_factor
#         img_orig = np.pad(img_orig, ((ddx,ddx), (ddy,ddy)), 'constant'  )
#     img_orig = img_orig/img_orig.max()  * 255
#     img_orig = img_orig.astype(np.uint8)
#     #img_orig = np.asarray(img.data)
#     #img_orig1 = ndi.median_filter(img_orig1, size=4)
#
#     # model inference + display
#
#     #img_sigma = np.std(img_orig)
#     #img_mean  = np.mean(img_orig)
#     #img_orig =  (img_orig  - img_mean) / img_sigma
#
#
#     plt.figure(12)
#     plt.plot(bin_edges[0:-1], histogram, 'b')  # <- or here
#     plt.figure(12)
#     plt.plot(bin_edges3[0:-1], histogram3, 'r')  # <- or here