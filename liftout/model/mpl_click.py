import glob
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib.widgets import Button, RectangleSelector

from data_engine import relabel_img

from utils import load_model, model_inference, detect_and_draw_lamella_and_needle, show_overlay_streamlit 


def show_img_prediction(model, img):
    img, rgb_mask = model_inference(model, img)

    # detect and draw lamella centre, and needle tip
    (
        lamella_centre_px,
        rgb_mask_lamella,
        needle_tip_px,
        rgb_mask_needle,
        rgb_mask_combined,
    ) = detect_and_draw_lamella_and_needle(rgb_mask)

        # prediction overlay
    img_overlay = show_overlay_streamlit(img, rgb_mask)

    return img_overlay



class Index:
    def __init__(self, imgs, fig, ax, img_plot):

        self.ind = 0

        self.fig = fig
        self.ax = ax
        self.img_plot = img_plot

        self.label_json = {}


        self.model = load_model("models/fresh_full_n10.pt")


        self.imgs = imgs
        self.img = show_img_prediction(self.model, np.asarray(self.imgs[0]))
        self.rs = RectangleSelector(
            ax,
            self.line_select_callback,
            drawtype="box",
            useblit=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            rectprops=dict(color="red", alpha=0.4, fill=True),
        )
        self.click = [None, None]
        self.release = [None, None]

    def line_select_callback(self, eclick, erelease):
        """Rectanglular selection callback """
        self.click[:] = eclick.xdata, eclick.ydata
        self.release[:] = erelease.xdata, erelease.ydata

        x1, y1 = self.click
        x2, y2 = self.release

        rect = plt.Rectangle(
            (min(x1, x2), min(y1, y2)),
            np.abs(x1 - x2),
            np.abs(y1 - y2),
            color="red",
            alpha=0.4,
        )
        self.ax.add_patch(rect)

    def correct(self, event):

        self.next_img()
        print("CORRECT", self.ind)

    def incorrect(self, event):

        self.label_json = {
            "shapes": [{"label": "lamella", 
                        "points": [self.click, self.release]}]
        }

        print("WRONG")
        print("Label: ", self.label_json)

        self.next_img()

    def next_img(self):

        # remove rectanglular selector and patches
        self.ax.patches = []
        self.rs.set_visible(False)
        self.rs.update()

        # update img
        self.ind += 1
        self.img = show_img_prediction(self.model, np.asarray(self.imgs[self.ind]))

        self.img_plot.set_data(self.img)
        plt.draw()


class DataEngine:
    def __init__(self, images):
        
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        img_plot = plt.imshow(np.zeros_like(images[0]), cmap="gray")

        fig.suptitle("Active Learning")
        plt.text(700, 1200, "Correct Prediction?")


        self.callback = Index(images, fig, ax, img_plot)
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, "Yes")
        bnext.on_clicked(self.callback.correct)
        bprev = Button(axprev, "No")
        bprev.on_clicked(self.callback.incorrect)

        plt.show()

        
if __name__ == "__main__":

    print("Starting Active Learning...")

    filenames = glob.glob("images/eval/model/*.tif")
    shuffle(filenames)

    images = [PIL.Image.open(fname) for fname in filenames]

    # run data engine
    data_engine = DataEngine(images)



# TODO:

# incorporate logging
# check if run out of images. and just return

# relabel_img(img)

# {
#   "version": "4.5.7",
#   "flags": {},
#   "shapes": [
#     {
#       "label": "lamella",
#       "points": [
#         [
#           622.6842105263157,
#           556.8421052631578
#         ],
#         [
#           609.0,
#           753.6842105263157
#         ],
#         [
#           858.4736842105262,
#           765.2631578947368
#         ],
#         [
#           860.578947368421,
#           562.1052631578947
#         ]
#       ],
#       "group_id": null,
#       "shape_type": "polygon",
#       "flags": {}
#     }
#   ],
#   "imagePath": "000000000.tif",


# Ref
# https://stackoverflow.com/questions/12052379/matplotlib-draw-a-selection-area-in-the-shape-of-a-rectangle-with-the-mouse
# https://matplotlib.org/3.1.1/gallery/widgets/rectangle_selector.html
# https://stackoverflow.com/questions/46510355/i-wants-to-get-coordinate-from-matplotlib-rectangle-selector

# buttons:
# https://matplotlib.org/stable/gallery/widgets/buttons.html

# https://stackoverflow.com/questions/63722736/matplotlib-remove-rectangleselector-widget-from-the-plot
