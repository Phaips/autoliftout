#!/usr/bin/env python3

# from liftout.fibsem import acquire, movement, calibration
# from microscope_verification import initial_test_setup
# from liftout.fibsem.acquire import BeamType

from numpy.core.defchararray import title
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Cross Correlation Calibration")

# # setup microscope
# # move to sample grid
# microscope, settings = initial_test_setup()

# # move to sample grid
# movement.move_to_sample_grid(microscope, settings)

# # TODO: get lowres, and highres hfw for cross correlation
# image_settings = {'resolution': "1536x1024", 'dwell_time': 1e-6,
#                         'hfw': 150e-6,
#                         'autocontrast': True,
#                         'beam_type': BeamType.ELECTRON,
#                         "gamma": settings["gamma"]}  # TODO: test if this dictionary works...

# # take reference images, record position
# eb_image_reference, ib_image_reference = acquire.take_reference_images(microscope=microscope, settings=image_settings)
# start_position = microscope.specimen.stage.current_position

# # randomly perturb position
# x_shift, y_shift = np.random.randint(1, 50, 2) * 10e-6
# x_move = movement.x_corrected_stage_movement(x_shift)
# y_move = movement.y_corrected_stage_movement(y_shift)
# microscope.specimen.stage.relative_move(x_move)
# microscope.specimen.stage.relative_move(y_move)


# # take reference images, record position
# eb_image, ib_image = acquire.take_reference_images(microscope=microscope, settings=image_settings)
# end_position = microscope.specimen.stage.current_position


# calculate shift

# run through sweep of cross correlation paramters,

df_results = pd.DataFrame()
x_shift, y_shift = np.random.randint(1, 50, 2) * 10e-6
# new_image = eb_image
# ref_image = eb_image_reference

# TODO: make params selectable?

image = np.array(Image.open("test_image.tif"))
ref_image = image
new_image = image

SIGMA = int(
    max(ref_image.shape) / 1536 * 10
)  # need to scan SIGMA too, from 1 to 10 or something like that
sigma_max = int(max(ref_image.shape) / 1536 * 10)
LOW_PASS_FILTER = np.arange(1, max(ref_image.shape) // 2 + 1, 10)
HIGH_PASS_FILTER = np.arange(1, max(ref_image.shape) // 32 + 1, 10)

SIGMA = np.arange(1, sigma_max + 1)

st.subheader("Parameter Sweep")
st.write(f"LOW_PASS: {LOW_PASS_FILTER}")
st.write(f"HIGH_PASS: {HIGH_PASS_FILTER}")
st.write(f"SIGMA: {SIGMA}")

sigma_factor = sigma_max
for hp_ratio in HIGH_PASS_FILTER:
    for lp_ratio in LOW_PASS_FILTER:
        # for sigma_factor in SIGMA:
        print(f"hp: {hp_ratio}, lp: {lp_ratio}, sig: {sigma_factor}")

        # record params and results

        # TODO: set mode for eb or ib imaging?
        # mostly eb for

        # something like this:

        # lp_ratio = 6
        # hp_ratio = 64
        # sigma_factor = 10
        sigma_ratio = 1536

        # These are the old cross-correlation values
        # elif mode is not "land":
        #     lp_ratio = 12
        #     hp_ratio = 256
        #     sigma_factor = 2
        #     sigma_ratio = 1536
        #     beam_type = BeamType.ELECTRON

        # TODO: possibly hard-code these numbers at fixed resolutions?
        lowpass_pixels = int(
            max(new_image.data.shape) / lp_ratio
        )  # =128 @ 1536x1024, good for e-beam images
        highpass_pixels = int(
            max(new_image.data.shape) / hp_ratio
        )  # =6 @ 1536x1024, good for e-beam images
        sigma = int(
            sigma_factor * max(new_image.data.shape) / sigma_ratio
        )  # =2 @ 1536x1024, good for e-beam images

        # do cross correlation
        # dx_metres, dy_metres = calibration.shift_from_crosscorrelation_AdornedImages(
        #     new_image, ref_image, lowpass=lowpass_pixels,
        #     highpass=highpass_pixels, sigma=sigma)
        dx_metres, dy_metres = np.random.randint(1, 200, 2) * 10e-6

        # TODO: do we want to calculate the pixels or ratio?
        results = {}
        results["dx"] = abs(dx_metres - x_shift)
        results["dy"] = abs(dy_metres - y_shift)
        results["x_shift"] = dx_metres
        results["y_shift"] = dy_metres
        results["lowpass_pixels"] = lowpass_pixels
        results["lowpass_ratio"] = lp_ratio
        results["highpass_pixels"] = highpass_pixels
        results["highpass_ratio"] = hp_ratio
        results["sigma"] = sigma
        results["sigma_factor"] = sigma_factor
        results["sigma_ratio"] = sigma_ratio
        results["timestamp"] = datetime.now().strftime("%Y%m%d.%H%M%S")

        df_results = df_results.append(pd.DataFrame.from_records([results]))



# check which ones get closest to correct shift.

# minimum results
st.subheader("Minimum Differences")
st.write(df_results[df_results["dx"] == min(df_results["dx"])])
st.write(df_results[df_results["dy"] == min(df_results["dy"])])

# Parameter Sweep Results
fig_pt = px.line(
    df_results,
    x="lowpass_ratio",
    y=["dx", "dy"],
    facet_row="highpass_ratio",
    facet_col="sigma_factor",
    title="Cross Correlation Calibration",
)
st.plotly_chart(fig_pt, use_container_width=True)

# Heatmap results
arr_dx = np.zeros((len(HIGH_PASS_FILTER), len(LOW_PASS_FILTER)))
arr_dy = np.zeros_like(arr_dx)

# TODO: do this in the main loop?

for ii, hp_ratio in enumerate(HIGH_PASS_FILTER):
    tmp_hp = df_results[df_results["highpass_ratio"] == hp_ratio]

    for jj, lp_ratio in enumerate(LOW_PASS_FILTER):

        val = tmp_hp[tmp_hp["lowpass_ratio"] == lp_ratio]
        arr_dx[ii, jj] = val["dx"]
        arr_dy[ii, jj] = val["dy"]

# TODO: get the actual range values for each tick

cols = st.columns(2)
fig_img = px.imshow(arr_dx, title="Difference in x", color_continuous_scale="jet")
cols[0].plotly_chart(fig_img)

fig_img = px.imshow(arr_dy, title="Difference in y", color_continuous_scale="jet")
cols[1].plotly_chart(fig_img)

# save those parameters in protocol
# TODO