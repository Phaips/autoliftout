#!/usr/bin/env python3

import glob
from random import shuffle
import detection


if __name__ == "__main__":
    
    # load model
    # weights_file = "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/patrick/test_model.pt"
    weights_file = r"\\ad.monash.edu\home\User007\prcle2\Documents\demarco\autoliftout\patrick\models\fresh_full_n10.pt"

    filenames = glob.glob("test_images/*.tif")
    shuffle(filenames)

    # detector class
    detector = detection.Detector(weights_file)

    for fname in filenames:

        # load image from file
        img = detection.load_image_from_file(fname)

        supported_shift_types = [
            "needle_tip_to_lamella_centre",
            "lamella_centre_to_image_centre",
            "lamella_edge_to_landing_post",
            "needle_tip_to_image_centre"
        ]

        print(f"IMAGE: {fname}")
        for shift_type in supported_shift_types:
            
            x_distance, y_distance = detector.calculate_shift_between_features(img, shift_type=shift_type, show=True) # more general version?

            print("Shift Type: ", shift_type)
            print("Distance: ", x_distance, y_distance)



        


