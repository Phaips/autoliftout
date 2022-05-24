
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from autoscript_sdb_microscope_client.structures import AdornedImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QGridLayout, QLabel, QSizePolicy, QVBoxLayout,
                             QWidget)


class _WidgetPlot(QWidget):
    def __init__(self, *args, display_image, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = _PlotCanvas(self, image=display_image)
        self.layout().addWidget(self.canvas)


class _PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, image=None):
        self.fig = Figure()
        FigureCanvasQTAgg.__init__(self, self.fig)

        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        self.image = image
        self.plot()
        self.createConn()

        self.figureActive = False
        self.axesActive = None
        self.cursorGUI = "arrow"
        self.cursorChanged = False

    def plot(self):
        gs0 = self.fig.add_gridspec(1, 1)

        self.ax11 = self.fig.add_subplot(
            gs0[0], xticks=[], yticks=[], title="")

        if self.image.ndim == 3:
            self.ax11.imshow(self.image,) 
        else:
            self.ax11.imshow(self.image, cmap="gray")

    def updateCanvas(self, event=None):
        ax11_xlim = self.ax11.get_xlim()
        ax11_xvis = ax11_xlim[1] - ax11_xlim[0]

        while len(self.ax11.patches) > 0:
            [p.remove() for p in self.ax11.patches]
        while len(self.ax11.texts) > 0:
            [t.remove() for t in self.ax11.texts]

        ax11_units = ax11_xvis * 0.003
        self.fig.canvas.draw()

    def createConn(self):
        self.fig.canvas.mpl_connect("figure_enter_event", self.activeFigure)
        self.fig.canvas.mpl_connect("figure_leave_event", self.leftFigure)
        self.fig.canvas.mpl_connect("button_press_event", self.mouseClicked)
        self.ax11.callbacks.connect("xlim_changed", self.updateCanvas)

    def activeFigure(self, event):
        self.figureActive = True

    def leftFigure(self, event):
        self.figureActive = False
        if self.cursorGUI != "arrow":
            self.cursorGUI = "arrow"
            self.cursorChanged = True

    def mouseClicked(self, event):
        x = event.xdata
        y = event.ydata


@dataclass
class Crosshair:
    rectangle_horizontal: plt.Rectangle
    rectangle_vertical: plt.Rectangle


def create_crosshair(image: np.ndarray or AdornedImage, x=None, y=None, colour="xkcd:yellow"):
    if type(image) == AdornedImage:
        image = image.data

    midx = int(image.shape[1] / 2) if x is None else x
    midy = int(image.shape[0] / 2) if y is None else y

    cross_width = int(
        0.05 / 100 * image.shape[1]
    )
    cross_length = int(5 / 100 * image.shape[1]
                       )

    rect_horizontal = plt.Rectangle(
        (midx - cross_length / 2, midy - cross_width / 2), cross_length, cross_width
    )
    rect_vertical = plt.Rectangle(
        (midx - cross_width, midy - cross_length / 2), cross_width * 2, cross_length
    )

    # set colours
    rect_horizontal.set_color(colour)
    rect_vertical.set_color(colour)

    return Crosshair(
        rectangle_horizontal=rect_horizontal, rectangle_vertical=rect_vertical
    )

def draw_crosshair(image, canvas):
    # draw crosshairs
    crosshair = create_crosshair(image)
    canvas.ax11.patches = []
    for patch in crosshair.__dataclass_fields__:
        canvas.ax11.add_patch(getattr(crosshair, patch))
        getattr(crosshair, patch).set_visible(True)

###################


def draw_grid_layout(samples: list):
    gridLayout = QGridLayout()

    # Only add data is sample positions are added
    if len(samples) == 0:
        label = QLabel()
        label.setText("No Sample Positions Selected. Please press initialise to begin.")
        gridLayout.addWidget(label)
        return gridLayout

    sample_images = [[] for _ in samples]

    # initial, mill, jcut, liftout, land, reset, thin, polish (TODO: add to protocol / external file)
    exemplar_filenames = ["ref_lamella_low_res_ib", "ref_trench_high_res_ib", "jcut_highres_ib",
                        "needle_liftout_landed_highres_ib", "landing_lamella_final_cut_highres_ib", "sharpen_needle_final_ib",
                        "thin_lamella_post_superres_ib", "polish_lamella_post_superres_ib"]

    # headers
    headers = ["Sample No", "Position", "Reference", "Milling", "J-Cut", "Liftout", "Landing", "Reset", "Thinning", "Polishing"]
    for j, title in enumerate(headers):
        label_header = QLabel()
        label_header.setText(title)
        label_header.setMaximumHeight(80)
        label_header.setStyleSheet("font-family: Arial; font-weight: bold; font-size: 18px;")
        label_header.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(label_header, 0, j)

    for i, sp in enumerate(samples):

        # load the exemplar images for each sample
        qimage_labels = []
        for img_basename in exemplar_filenames:
            fname = os.path.join(sp.data_path, str(sp.sample_id), f"{img_basename}.tif")
            imageLabel = QLabel()
            imageLabel.setMaximumHeight(150)

            if os.path.exists(fname):
                adorned_img = sp.load_reference_image(img_basename)
                image = QImage(adorned_img.data, adorned_img.data.shape[1], adorned_img.data.shape[0], QImage.Format_Grayscale8)
                imageLabel.setPixmap(QPixmap.fromImage(image).scaled(125, 125))

            qimage_labels.append(imageLabel)

        sample_images[i] = qimage_labels

        # display information on grid
        row_id = i + 1

        # display sample no
        label_sample = QLabel()
        label_sample.setText(f"""Sample {sp.sample_no:02d} \n{sp.petname} ({str(sp.sample_id)[-6:]}) \nStage: {sp.microscope_state.last_completed_stage.name}""")
        label_sample.setStyleSheet("font-family: Arial; font-size: 12px;")
        label_sample.setMaximumHeight(150)
        gridLayout.addWidget(label_sample, row_id, 0)

        # display sample position
        label_pos = QLabel()
        pos_text = f"Pos: x:{sp.lamella_coordinates.x:.2f}, y:{sp.lamella_coordinates.y:.2f}, z:{sp.lamella_coordinates.z:.2f}\n"
        if sp.landing_coordinates.x is not None:

            pos_text += f"Land: x:{sp.landing_coordinates.x:.2f}, y:{sp.landing_coordinates.y:.2f}, z:{sp.landing_coordinates.z:.2f}\n"
            
        label_pos.setText(pos_text)
        label_pos.setStyleSheet("font-family: Arial; font-size: 12px;")
        label_pos.setMaximumHeight(150)

        gridLayout.addWidget(label_pos, row_id, 1)

        # display exemplar images
        gridLayout.addWidget(sample_images[i][0], row_id, 2, Qt.AlignmentFlag.AlignCenter) #TODO: fix missing visual boxes
        gridLayout.addWidget(sample_images[i][1], row_id, 3, Qt.AlignmentFlag.AlignCenter)
        gridLayout.addWidget(sample_images[i][2], row_id, 4, Qt.AlignmentFlag.AlignCenter)
        gridLayout.addWidget(sample_images[i][3], row_id, 5, Qt.AlignmentFlag.AlignCenter)
        gridLayout.addWidget(sample_images[i][4], row_id, 6, Qt.AlignmentFlag.AlignCenter)
        gridLayout.addWidget(sample_images[i][5], row_id, 7, Qt.AlignmentFlag.AlignCenter)
        gridLayout.addWidget(sample_images[i][6], row_id, 8, Qt.AlignmentFlag.AlignCenter)
        gridLayout.addWidget(sample_images[i][7], row_id, 9, Qt.AlignmentFlag.AlignCenter)

    gridLayout.setRowStretch(9, 1) # grid spacing
    return gridLayout
