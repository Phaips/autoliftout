#!/usr/bin/env python3

import glob
from DetectionModel import *
from detection import *






if __name__ == "__main__":
    # load model
    weights_file = "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/patrick/test_model.pt"
    # r"\\ad.monash.edu\home\User007\prcle2\Documents\demarco\autoliftout\patrick\models\fresh_full_n10.pt"
    # model = DetectionModel(weights_file)

    filenames = glob.glob("test_images/*.tif")
    shuffle(filenames)

    # detector class
    detector = Detector(weights_file)

    for fname in filenames:

        # load image from file
        img = load_image_from_file(fname)

        supported_shift_types = [
            # "needle_tip_to_lamella_centre",
            # "lamella_centre_to_image_centre",
            "lamella_edge_to_landing_post",
            # "needle_tip_to_image_centre"
        ]

        for shift_type in supported_shift_types:
            
            print("SHIFT TYPE: ", shift_type)
            x_distance, y_distance = detector.calculate_shift_between_features(img, shift_type=shift_type, show=True) # more general version?

            print("Distance: ", x_distance, y_distance)



        


