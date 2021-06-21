#!/usr/bin/env python3


import PIL
import skimage
import matplotlib.pyplot as plt

import glob
import numpy as np

from skimage import feature

from PIL import ImageDraw

from scipy.spatial import distance

from utils import draw_rectangle_feature, draw_crosshairs


def detect_closest_landing_point(img, landing_px):
    """ Identify the closest edge landing point to the initially selected landing point"""
    
    # identify edge pixels
    edges = feature.canny(img, sigma=3) # sigma higher usually better
    edge_mask = np.where(edges)
    edge_px = list(zip(edge_mask[0], edge_mask[1]))

    # set min distance
    min_dst = np.inf

    # TODO: vectorise this like
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html

    for px in edge_px:

        # distance between edges and landing point
        dst = distance.euclidean(landing_px, px)

        # select point with min
        if dst < min_dst:

            min_dst = dst
            landing_edge_pt = px

    print("Dist: ", px, landing_px, dst)
    return landing_edge_pt, edges


def draw_landing_edges_and_point(img, edges, edge_landing_px, show=True):
    """Draw the detected landing edge ontop of the edge detection image"""
    # draw landing point
    landing_px_mask = PIL.Image.fromarray(np.zeros_like(img))
    landing_px_mask = draw_rectangle_feature(
        landing_px_mask, edge_landing_px, RECT_WIDTH=10
    )

    # draw crosshairs
    draw = PIL.ImageDraw.Draw(landing_px_mask)
    draw_crosshairs(draw=draw, mask=landing_px_mask, idx=edge_landing_px)

    # show landing spot
    if show:

        fig, ax = plt.subplots(1, 1)
        ax.set_title(f"Edge sigma={3}")
        ax.imshow(edges, cmap="gray", alpha=0.9)
        ax.imshow(landing_px_mask, cmap="gray", alpha=0.5)
        plt.show()

    return landing_px_mask


if __name__ == "__main__":

    print("Hello Edge Detection")

    filenames = glob.glob("landing_edge_img/*.tif")

    for i, fname in enumerate(filenames):

        img = np.array(PIL.Image.open(fname))
        # use img.data to get this from Adorned Image

        test_landing_pxs = [(400, 750), (700, 750), (200, 500), (600, 1000)]
        # Note: this will only work if the landing pt is close to selection, if calibration is off all bets are off
        
        for landing_px in test_landing_pxs:

            # use the initially selected landing point, and snap to the nearest edge
            edge_landing_px, edges = detect_closest_landing_point(img, landing_px)
            landing_px_mask = draw_landing_edges_and_point(img, edges, edge_landing_px)

            # TODO: validate detection here

            scaled_edge_landing_px = (
                    edge_landing_px[0] / landing_px_mask.size[1],  # mask is a PIL.Image (x, y) landing_px is px (y, x)
                    edge_landing_px[1] / landing_px_mask.size[0],
            )

            print("Landing Point:", edge_landing_px)
            print("Proportional: ", scaled_edge_landing_px)



            # TODO: ensure the image size is the same between detection and this
            # TODO: convert to proportional / metres
            # TODO: refactor utils.scale_invariant_coordinates to be general