#!/usr/bin/env python3


import PIL
import skimage
import matplotlib.pyplot as plt

import glob
import numpy as np

from skimage import feature

from PIL import ImageDraw

from scipy.spatial import distance

from utils import (draw_rectangle_feature, draw_crosshairs,
    detect_closest_landing_point, draw_landing_edges_and_point,
    scale_invariant_coordinates_NEW

)


import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu

if __name__ == "__main__":

    print("Hello Edge Detection")

    filenames = glob.glob("patrick/landing_edge_img/*.tif")

    for i, fname in enumerate(filenames):

        img = np.array(PIL.Image.open(fname))
        # use img.data to get this from Adorned Image

        w_min, w_max = int(img.shape[1] * 0.3), int(img.shape[1]*0.7)
        h_min, h_max = int(img.shape[0] * 0.3), int(img.shape[0]*0.7)
        img = img[h_min:h_max, w_min:w_max]

        # binarise
        thresh = threshold_otsu(img)
        binary = img > thresh

        plt.imshow(binary, cmap="gray")
        plt.show()

        test_landing_pxs = [(img.shape[0]//4, img.shape[1]//4), (img.shape[0]//2, img.shape[1]//2), (700, 750)]
        # Note: this will only work if the landing pt is close to selection, if calibration is off all bets are off

        for landing_px in test_landing_pxs:

            # use the initially selected landing point, and snap to the nearest edge
            edge_landing_px, edges = detect_closest_landing_point(binary, landing_px)
            landing_px_mask = draw_landing_edges_and_point(edges, edges, edge_landing_px)

            # TODO: validate detection here

            scaled_edge_landing_px = scale_invariant_coordinates_NEW(edge_landing_px, landing_px_mask)

            print("Landing Point:", edge_landing_px)
            print("Proportional: ", scaled_edge_landing_px)



            # TODO: ensure the image size is the same between detection and this
            # TODO: convert to proportional / metres
            # TODO: refactor utils.scale_invariant_coordinates to be general

            # https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html