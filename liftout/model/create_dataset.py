#!/usr/bin/env python3

import numpy as np
import PIL
import glob
import shutil
import os

# move and rename files into dataset format.

DATA_PATH = "/Users/patrickcleeve/Documents/university/bio/demarco/liftout/data/retrain/raw"
SEARCH_PATH = os.path.join(DATA_PATH, "*.tif")
START_IDX = 955
filenames = sorted(glob.glob(
    SEARCH_PATH, recursive=True))

print(DATA_PATH)
print(SEARCH_PATH)
print(f"Starting index at {START_IDX}")
print(f"{len(filenames)} images found.. ")


for idx, fname in enumerate(filenames, START_IDX):

    new_fname = os.path.join(DATA_PATH, f"{idx:09d}.tif")
    
    assert os.path.dirname(fname) == DATA_PATH

    print(fname, new_fname)
    # rename the file
    os.rename(fname, new_fname)


print(f"Sucessfully renamed {len(filenames)} files.")