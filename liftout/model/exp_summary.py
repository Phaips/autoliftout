#!/usr/bin/env python3

import matplotlib.pyplot as plt
import PIL
import glob
import sys
import numpy as np
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        help="filepath to experiment images ",
        dest="filepath",
        action="store",
        type=str
    )
    parser.add_argument(
        "--exp",
        help="number of experiments to display",
        dest="exp",
        action="store",
        default=10,
        type=int
    )
    parser.add_argument(
        "--images",
        help="number of test images in an experiment",
        dest="images",
        action="store",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--title",
        help="title for the experiment figure",
        dest="title",
        action="store",
        type=str,
        default="Model Experiment",
    )

    args = parser.parse_args()

    # cmd line arguments 
    h, w = args.exp, args.images
    filepath = args.filepath

    # read files
    filenames = sorted(glob.glob(filepath +"*g"))
    filenames = np.array(filenames).reshape(h, w) # reshape for conveinence

    # plotting setup
    fig, ax = plt.subplots(w, h, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.01, wspace=0.05)
    
    # loop through experiment images, and show
    for i, exp in enumerate(filenames):
        for j, fname in enumerate(exp):
            
            img = PIL.Image.open(fname)
            if "baseline" in fname:
                exp_name = "baseline"
            else:
                exp_name = fname.split("\\")[-1].split(".")[0].split("_")[0]

            # show image
            ax[j][i].imshow(img)
            ax[j][i].get_xaxis().set_ticks([])
            ax[j][i].get_yaxis().set_ticks([])
            ax[j][0].set(ylabel=f"img {j:02d}")



            if j == len(exp)-1:
                ax[j][i].set(xlabel=exp_name)
   
    # show all experiment images
    fig.suptitle(args.title, fontsize=24)
    plt.show()

    # save summary figure
    summary_fname = filepath + "exp_summary.png"
    fig.savefig(summary_fname)
    print(f"Saving summary as: {summary_fname}")