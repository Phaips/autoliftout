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


def detect_closest_landing_point(img, landing_pt):

    # identify edge pixels
    edges = feature.canny(img, sigma=3)
    edge_mask = np.where(edges)
    edge_px = list(zip(edge_mask[0], edge_mask[1]))

    # set min distance
    min_dst = np.inf

    for px in edge_px:
                            
        # distance between edges and landing point
        dst = distance.euclidean(landing_pt, px)

        # select point with min
        if dst < min_dst:

            min_dst = dst
            landing_edge_pt = px

    print("Dist: ", px, landing_pt, dst)
    return landing_edge_pt, edges


def draw_landing_edges_and_point(img, edges, edge_landing_pt, show=True):

    # draw landing point
    landing_pt_mask = PIL.Image.fromarray(np.zeros_like(img))
    landing_pt_mask = draw_rectangle_feature(landing_pt_mask, edge_landing_pt, RECT_WIDTH=10)
    
    # draw crosshairs
    draw = PIL.ImageDraw.Draw(landing_pt_mask)
    draw_crosshairs(draw=draw, mask=landing_pt_mask, idx=edge_landing_pt, color="red")

    if show:
        
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f"Edge sigma={3}")
        ax.imshow(edges, cmap="gray", alpha=0.9)
        ax.imshow(landing_pt_mask, cmap="gray", alpha=0.5)
        plt.show()

    return landing_pt_mask




if __name__ == "__main__":

    print("Hello Edge Detection")

    filenames = glob.glob("landing_edge_img/*.tif")

    for i, fname in enumerate(filenames):

        img = np.array(PIL.Image.open(fname))
        # use img.data to get this from Adorned Image


        # crop image to centre 
        # h, w = img.shape[0], img.shape[1]
        # h_min, h_max = int(h*0.4), int(h*0.6)
        # w_min, w_max = int(w*0.4), int(w*0.6)
        # print(w, h)
        
        # img = img[h_min:h_max, w_min: w_max]



        landing_pt = (700, 750)
        edge_landing_pt, edges = detect_closest_landing_point(img, landing_pt)
        landing_pt_mask = draw_landing_edges_and_point(img, edges, edge_landing_pt)

        # only detect near vertical edges?
        # only detect near the centre of image?
        # detect edge nearest the selected point?