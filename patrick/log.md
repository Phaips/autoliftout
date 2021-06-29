# TODO:

- remove unused functions
- add function to take and save images in 1 line - DONE
- refactor out old / new functions
- convert to object oriented design
- make sure that each shift calculation is consistent in direction so that dont need to keep swapping negative signs
- consider refactoring the calculation so it is always to - from (or similiar)
- rethink whether taking images should be done inside functions or outside and passed in

- refactor sharpen_needle
- code the cleanup stage

- replace / refactor all instances of taking and saving images
- rename calculate distance functions to be more logical

- refactor drawing to use a single function: 
- draw_feature(mask, px, color, crosshair) - DONE
- draw_two_features(mask, feature_1, feature_2, line=False) - DONE
- detect_centre_point(mask, color, threshold) - DONE
- detect_right_edge(mask, color, threshold) - DONE
- draw_overlay(img, mask, alpha, show) - DONE


- packaging, setup.py
https://www.youtube.com/watch?v=GaWs-LenLYE
https://www.youtube.com/watch?v=wCGsLqHOT2I