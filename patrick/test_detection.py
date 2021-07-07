#!/usr/bin/env python3

import glob
from random import shuffle
import detection
import argparse

import utils
import new_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="store_false", dest="validate", default=True)
    parser.add_argument("-m", action="store", type=str, dest="model", default="models/fresh_full_n10.pt")
    parser.add_argument("-d", action="store", type=str, dest="data", default="test_images/")
    args = parser.parse_args()

    # weights_file = "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/patrick/test_model.pt"
    # weights_file = r"\\ad.monash.edu\home\User007\prcle2\Documents\demarco\autoliftout\patrick\models\fresh_full_n10.pt"

    # cmd line arguments
    validate = args.validate
    weights_file = args.model
    data_path = args.data

    # data
    filenames = glob.glob(data_path + "*.tif")
    shuffle(filenames)

    # detector class (model)
    detector = detection.Detector(weights_file)

    for fname in filenames:

        # load image from file
        img = new_utils.load_image_from_file(fname)

        # img metadata
        df_metadata = utils.parse_metadata(fname)

        supported_shift_types = [
            # "needle_tip_to_lamella_centre",
            # "lamella_centre_to_image_centre",
            # "lamella_edge_to_landing_post",
            # "needle_tip_to_image_centre",
            "trim_lamella_top_to_centre",
            "trim_lamella_bottom_to_centre"
        ]

        print(f"image: {fname}")

        for shift_type in supported_shift_types:

            print("shift_type: ", shift_type)
            x_distance, y_distance = detector.calculate_shift_between_features(img, shift_type=shift_type, show=True, validate=validate)
            print(f"x_distance = {x_distance:.4f}, y_distance = {y_distance:.4f}")

            x_shift, y_shift = detection.calculate_shift_distance_in_metres(img, x_distance, y_distance, df_metadata)
            print(f"x_shift =  {x_shift/1e-6:.4f}, um; y_shift = {y_shift/1e-6:.4f} um; ")

