# TODO:

- remove unused functions
- refactor out old / new functions
- convert to object oriented design

- replace / refactor all instances of taking and saving images
- rename calculate distance functions to be more logical - DONE

- refactor drawing to use a single function:
- draw_feature(mask, px, color, crosshair) - DONE
- draw_two_features(mask, feature_1, feature_2, line=False) - DONE
- detect_centre_point(mask, color, threshold) - DONE
- detect_right_edge(mask, color, threshold) - DONE
- draw_overlay(img, mask, alpha, show) - DONE


- change feature distance calculation to be consistent (stationary minus moving)
-   this should make the direction consistent
-   in some cases the stage (lamella) moves, other times needle (consistent with naming, moving_to_stationary)
-           "needle_tip_to_lamella_centre",
            "lamella_centre_to_image_centre",
            "lamella_edge_to_landing_post",
            "needle_tip_to_image_centre"
- need to resolve the colours so that they are consistent with this and not determined by postition - Done

- packaging, setup.py
https://www.youtube.com/watch?v=GaWs-LenLYE
https://www.youtube.com/watch?v=wCGsLqHOT2I
https://iq-inc.com/importerror-attempted-relative-import/

# Conceptual Breakdown

- setup
- milling
- detection
- measurement (calibration)
- imaging (acquire)
- movement
- display
- utils



# TODO:
- change confirmation: "Is the feature centred now? yes/no to:
    - please centre the feature in the SEM/ion and enter yes
    - needs user to actually do the centring

- add a global flag for confirmation of milling
- add a global flag for validation of detections

- sharpen needle pattern very sensitive to initial position / params

- GUI
-   add move stage from gui
-   add user_msg to displat in gui
-   add yes / no buttons to gui that feedback to autoliftout
-   pass image to gui?

- wrap some functions to update the gui?