#!/usr/bin/env python3


from PIL.Image import new
import numpy as np
import PIL
import glob

import os

DATA_PATH = "/Users/patrickcleeve/Documents/university/bio/demarco/liftout/data"
train_filenames = sorted(glob.glob(DATA_PATH + "/train/raw/*.tif", recursive=True))
train_json = sorted(glob.glob(DATA_PATH + "/train/raw/*.json", recursive=True))
aug_tif = sorted(glob.glob(DATA_PATH + "/aug/raw/*.tif", recursive=True))
aug_json = sorted(glob.glob(DATA_PATH + "/aug/raw/*.json", recursive=True))


print(len(aug_tif))
print(len(aug_json))

# # rename all the aug/ files to match the latest train

import shutil
idx = 687
# # for i, fname in enumerate(aug_tif, 1):
    
# #     print(idx+i, f"{DATA_PATH}/train/raw/{idx+i:09d}.tif")

# #     shutil.copyfile(fname, f"{DATA_PATH}/train/raw/{idx+i:09d}.tif")

new_tif = []
new_json = []
print("AUG TIF")
for i, fname in enumerate(aug_tif):

    # increment by basename number?
    basename = fname.split("/")[-1].split(".")[0]

    new_idx = idx + int(basename)
    new_fname = f"{DATA_PATH}/train/raw/{new_idx:09d}.tif"
    print(f"{idx}+{int(basename)}: {new_fname}")
    new_tif.append(new_fname)

    shutil.copyfile(fname, new_fname)


# basename tif == basename json

print("AUG JSON")
for i, fname in enumerate(aug_json):

    # increment by basename number?
    basename = fname.split("/")[-1].split(".")[0]
    new_idx = idx + int(basename)
    new_fname = f"{DATA_PATH}/train/raw/{new_idx:09d}.json"
    print(f"{idx}+{int(basename)}: {new_fname}")
    new_json.append(new_fname)

    shutil.copyfile(fname, new_fname)


# check lengths are the same...
# basename tif == basename json
print(len(aug_tif), len(aug_json))
print(len(new_tif), len(new_json))