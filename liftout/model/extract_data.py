#!/usr/bin/env python3

""" extract images from folders and save to a common dataset. Preserve .tif metadata """

import glob
import sys
from PIL import Image
from PIL.TiffTags import TAGS

# could probably do this as a bash script

if __name__ == "__main__":

    folder = "images/new_images"
    filenames = glob.glob(folder + "/**/*.tif", recursive=True)

    idx = int(sys.argv[1])
    print(f"\nReading files from: {folder}")
    print(f"Incrementing count by {idx} \n")

    for i, fname in enumerate(filenames):

        # rename img
        new_fname = f"data/aug/raw/{idx+i:09d}.tif"
        # save img and preserve metadata
        img = Image.open(fname)
        # img.save(new_fname, tiffinfo=img.tag, format="TIFF")

        print(f"Image {i} saved as {new_fname}")

    print(f"{len(filenames)} images saved.")