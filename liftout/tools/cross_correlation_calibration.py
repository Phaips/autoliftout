#!/usr/bin/env python3

# from liftout.fibsem import acquire, movement, calibration
# from microscope_verification import initial_test_setup
# from liftout.fibsem.acquire import BeamType

import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

"""
# setup microscope
# move to sample grid
microscope, settings = initial_test_setup()

# move to sample grid
movement.move_to_sample_grid(microscope, settings)

# TODO: get lowres, and highres hfw for cross correlation
image_settings = {'resolution': "1536x1024", 'dwell_time': 1e-6,
                        'hfw': 150e-6,
                        'autocontrast': True,
                        'beam_type': BeamType.ELECTRON,
                        "gamma": settings["gamma"]}  # TODO: test if this dictionary works...

# take reference images, record position
eb_image_reference, ib_image_reference = acquire.take_reference_images(microscope=microscope, settings=image_settings)
start_position = microscope.specimen.stage.current_position

# randomly perturb position
x_shift, y_shift = np.random.randint(1, 50, 2) * 10e-6
x_move = movement.x_corrected_stage_movement(x_shift)
y_move = movement.y_corrected_stage_movement(y_shift)
microscope.specimen.stage.relative_move(x_move)
microscope.specimen.stage.relative_move(y_move)


# take reference images, record position
eb_image, ib_image = acquire.take_reference_images(microscope=microscope, settings=image_settings)
end_position = microscope.specimen.stage.current_position
"""

# calculate shift

# run through sweep of cross correlation paramters, 

df_results = pd.DataFrame()
x_shift, y_shift = np.random.randint(1, 50, 2) * 10e-6
# new_image = eb_image
# ref_image = eb_image_reference



image = np.array(Image.open("test_image.tif"))
ref_image = image
new_image = image

SIGMA = int( max(ref_image.shape)/1536  * 10 ) # need to scan SIGMA too, from 1 to 10 or something like that
sigma_max = int( max(ref_image.shape)/1536 * 10 )
LOW_PASS_FILTER  = np.arange(1, max(ref_image.shape)//2  + 1, 10)
HIGH_PASS_FILTER = np.arange(1, max(ref_image.shape)//32 + 1, 10)

SIGMA = np.arange(1, sigma_max+1)


for hp_ratio in HIGH_PASS_FILTER:
    for lp_ratio in LOW_PASS_FILTER:
        for sigma_factor in SIGMA:
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
            lowpass_pixels = int(max(
                new_image.data.shape) / lp_ratio)  # =128 @ 1536x1024, good for e-beam images
            highpass_pixels = int(max(
                new_image.data.shape) / hp_ratio)  # =6 @ 1536x1024, good for e-beam images
            sigma = int(sigma_factor * max(
                new_image.data.shape) / sigma_ratio)  # =2 @ 1536x1024, good for e-beam images

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

# print(df_results)

print(
    df_results[df_results["dx"] == min(df_results["dx"])]
)
print(
    df_results[df_results["dy"] == min(df_results["dy"])]
)

import matplotlib.pyplot as plt



for sigma_factor in SIGMA:

    tmp = df_results[df_results["sigma_factor"]==sigma_factor]

    for hp_ratio in HIGH_PASS_FILTER:
        tmp_hp = tmp[tmp["highpass_ratio"]==hp_ratio]
        plt.figure(22)
        plt.suptitle(f"HP: {hp_ratio}, sigma: {sigma_factor}")
        plt.subplot(2,1,1)
        plt.plot(LOW_PASS_FILTER, tmp_hp["dx"], label="x")
        plt.subplot(2,1,2)
        plt.plot(LOW_PASS_FILTER, tmp_hp["dy"], label="y")
        plt.show()

# TODO: colourmap
# TODO: show parameters that minimise the difference


# check which ones get closest to correct shift.

# save those parameters in protocol