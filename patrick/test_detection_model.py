#!/usr/bin/env python3

import glob
from DetectionModel import *
from detection import *

# load model
weights_file = r"\\ad.monash.edu\home\User007\prcle2\Documents\demarco\autoliftout\patrick\models\fresh_full_n10.pt"
# model = DetectionModel(weights_file)

filenames = glob.glob("test_images/*.tif")
shuffle(filenames)

# detector class
detector = Detector(weights_file)

for fname in filenames:

    # load image from file
    img = load_image(fname)

    # calculate_needletip_shift_from_lamella_centre(img, show=True)
    # calculate_shift_between_features(img, show=True) # more general version?
    supported_shift_types = [
        # "needle_tip_to_lamella_centre",
        # "lamella_centre_to_image_centre",
        "lamella_edge_to_landing_post",
        # "needle_tip_to_image_centre"
    ]

    # landing_edge_px, landing_mask = detect_landing_edge(img, landing_px=(img.shape[0]//2, img.shape[1]//2))

    for shift_type in supported_shift_types:
        
        print("SHIFT TYPE: ", shift_type)
        x_distance, y_distance = calculate_shift_between_features(img, shift_type=shift_type, show=True) # more general version?

        print("Distance: ", x_distance, y_distance)


    

    # calculate_shift_between_features(img, shift_type="lamella_centre_to_image_centre", show=True) # more general version?
    # calculate_shift_between_features(img, shift_type="lamella_centre_to_image_centre", show=True) # more general version?

    # mask = detector.detection_model.model_inference(img)

    # needle_tip_px, needle_mask = detect_needle_tip(img, mask)
    # lamella_centre_px, lamella_mask = detect_lamella_centre(img, mask)
    # lamella_edge_px, lamella_mask = detect_lamella_edge(img, mask)

    # mask_combined = draw_two_features(mask, lamella_centre_px, needle_tip_px)
    # alpha_blend = draw_overlay(img, mask_combined, show=True)

    # # test class extraction
    # for color in [(255, 0, 0), (0, 255, 0)]:
        
    #     # mask filtering
    #     mask_filt, px_filt = extract_class_pixels(mask, color)
    #     print("N_DETECTIONS: ", len(px_filt[0]), len(px_filt[1]))

    #     assert len(px_filt[0]) == len(px_filt[1])

    #     # alpha_blend = draw_overlay(img, mask_filt, show=True)

    #     # centre detection
    #     centre_px = detect_centre_point(mask, color, threshold=25)
    #     print("CENTRE: ", centre_px)

    #     mask_draw = draw_feature(mask_filt, centre_px, color, crosshairs=True)
    #     alpha_blend = draw_overlay(img, mask_draw, show=True)

    #     # right edge detection
    #     edge_px = detect_right_edge(mask, color, threshold=50)
    #     print("EDGE: ", edge_px)
    #     mask_draw = draw_feature(mask_filt, edge_px, color, crosshairs=True)
    #     alpha_blend = draw_overlay(img, mask_draw, show=True)

    #     # draw two features
    #     mask_combined = draw_two_features(mask, centre_px, edge_px, color_1="red", color_2="green",  line=True)
    #     alpha_blend = draw_overlay(img, mask_combined, show=True)


        


