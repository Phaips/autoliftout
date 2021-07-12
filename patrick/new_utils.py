import numpy as np
import PIL
from PIL import Image
import patrick.detection
import matplotlib.pyplot as plt
import glob

import shutil

def extract_img_for_labelling(path, logfile="logfile"):
    """Extract all the images that have been identified for retraining"""

    log_dir = path+f"{logfile}/"
    label_dir = path+"label"
    dest_dir = path
    # identify images with _label postfix
    filenames = glob.glob(log_dir+ "*label*.tif")


    for fname in filenames:
        # print(fname)
        basename = fname.split("/")[-1]
        print(fname, basename)
        shutil.copyfile(fname, path+"label/"+basename)

    # zip the image folder
    shutil.make_archive(f"{path}/images", 'zip', label_dir)



###################################################################################################################################################

# Detection, Drawing helper functions

# def select_point_new(image):
#     fig, ax = plt.subplots()
#     ax.imshow(image, cmap="gray")
#     coords = []

#     def on_click(event):
#         print(event.xdata, event.ydata)
#         coords.append(event.ydata)
#         coords.append(event.xdata)

#     fig.canvas.mpl_connect("button_press_event", on_click)
#     plt.show()

#     return tuple(coords[-2:])

# def validate_detection(img, img_base, detection_coord, det_type):
#     correct = input(f"Is the detection for {det_type} correct? (y/n)")
#     #TODO: change this to user_input
#     if correct == "n":

#         detection_coord_initial = detection_coord # save initial coord

#         print(f"Please click the {det_type} position")
#         detection_coord = select_point_new(img)

#         # TODO: need to resolve behaviour when user exits plot without selecting?
#         # if detection_coord is None:
#         #     detection_coord = detection_coord # use initial coord


#         # save image for training here
#         print("Saving image for labelling")
#         #storage.step_counter +=1
#         #storage.SaveImage(img_base, id="label_")


#     print(f"{det_type}: {detection_coord}")
#     return detection_coord




