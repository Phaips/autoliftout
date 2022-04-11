
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from autoscript_sdb_microscope_client.structures import AdornedImage
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
from liftout.gui.qtdesigner_files import milling

from PyQt5 import QtWidgets
import numpy as np




@dataclass
class Crosshair:
    rectangle_horizontal: plt.Rectangle
    rectangle_vertical: plt.Rectangle


def create_crosshair(image: np.ndarray or AdornedImage, x=None, y=None):
    if type(image) == AdornedImage:
        image = image.data

    if x is None:
        midx = int(image.shape[1] / 2)
    else:
        midx = x
    if y is None:
        midy = int(image.shape[0] / 2)
    else:
        midy = y

    cross_width = int(
        0.1 / 100 * image.shape[1]
    )
    cross_length = int(10 / 100 * image.shape[1]
                       )

    rect_horizontal = plt.Rectangle(
        (midx - cross_length / 2, midy - cross_width / 2), cross_length, cross_width
    )
    rect_vertical = plt.Rectangle(
        (midx - cross_width, midy - cross_length / 2), cross_width * 2, cross_length
    )

    # set colours
    colour = "xkcd:yellow"
    rect_horizontal.set_color(colour)
    rect_vertical.set_color(colour)

    return Crosshair(
        rectangle_horizontal=rect_horizontal, rectangle_vertical=rect_vertical
    )


class GUIMillingWindow(milling.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(GUIMillingWindow, self).__init__()
        self.setupUi(self)

        # select pattern
        self.patterns = ["Pattern One", "Pattern Two", "Pattern Three", "Pattern Four"]
        self.comboBox_select_pattern.addItems(self.patterns)

        # test image
        self.image = np.random.randint(0, 255, size=(1024, 1536), dtype='uint16')
        global image
        image = self.image

        self.wp = _WidgetPlot(self)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)

        self.upper_rectangle = Rectangle((0, 0), 0.2, 0.2, color='yellow', fill=None, alpha=1)
        self.lower_rectangle = Rectangle((0, 0), 0.2, 0.2, color='yellow', fill=None, alpha=1)
        self.wp.canvas.ax11.add_patch(self.upper_rectangle)
        self.wp.canvas.ax11.add_patch(self.lower_rectangle)

        self.wp.canvas.mpl_connect('button_press_event', self.on_click)


        # create sliders for each pattern param?
        # width
        # height
        # rotation

        self.setup_connections()


    def setup_connections(self):

        self.pushButton_runMilling.clicked.connect(self.run_milling)
        self.comboBox_select_pattern.currentIndexChanged.connect(self.change_pattern)

        self.doubleSpinBox_height.valueChanged.connect(self.update_milling_pattern)
        self.doubleSpinBox_width.valueChanged.connect(self.update_milling_pattern)
        self.doubleSpinBox_rotation.valueChanged.connect(self.update_milling_pattern)
        self.doubleSpinBox_depth.valueChanged.connect(self.update_milling_pattern)
        self.doubleSpinBox_direction.valueChanged.connect(self.update_milling_pattern)

    def update_milling_pattern(self):

        idx = self.comboBox_select_pattern.currentIndex()
        current_pattern = self.patterns[idx]

        print("UPDATING: ", current_pattern)
        # TODO: get senderr?


    def change_pattern(self):

        opt = self.comboBox_select_pattern.currentIndex()
        text = self.comboBox_select_pattern.currentText()
        print(f"Changed to {self.patterns[opt]}")

        # update pattern values in double spin box too

    def run_milling(self):

        print("Run Milling Button Pressed")

        # TODO: run milling func

    def on_click(self, event):
        print("ON CLICK")

        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata
            print(self.xclick, self.yclick)
            # self.center_x, self.center_y = fibsem.pixel_to_realspace_coordinate((self.xclick, self.yclick), self.adorned_image)
            crosshair = create_crosshair(self.image, x=self.xclick, y=self.yclick)
            self.wp.canvas.ax11.patches = []
            for patch in crosshair.__dataclass_fields__:
                self.wp.canvas.ax11.add_patch(getattr(crosshair, patch))
                getattr(crosshair, patch).set_visible(True)
            self.wp.canvas.draw()

    def draw_milling_patterns(self):
        
        lower_pattern, upper_pattern = mill_trench_patterns(self.parent().microscope, self.center_x, self.center_y, self.settings)

        def draw_rectangle_pattern(adorned_image, rectangle, pattern):
            image_width = adorned_image.width
            image_height = adorned_image.height
            pixel_size =  adorned_image.metadata.binary_result.pixel_size.x

            width = pattern.width / pixel_size
            height = pattern.height / pixel_size
            rectangle_left = (image_width / 2) + (pattern.center_x / pixel_size) - (width/2)
            rectangle_bottom = (image_height / 2) - (pattern.center_y / pixel_size) - (height/2)
            rectangle.set_width(width)
            rectangle.set_height(height)
            rectangle.set_xy((rectangle_left, rectangle_bottom))
            rectangle.set_visible(True)

        try:
            draw_rectangle_pattern(adorned_image=self.adorned_image, rectangle=self.upper_rectangle, pattern=upper_pattern)
            draw_rectangle_pattern(adorned_image=self.adorned_image, rectangle=self.lower_rectangle, pattern=lower_pattern)
        except:
            # NOTE: these exceptions happen when the pattern is too far outside of the FOV
            print("Pattern outside FOV") #TODO
        self.wp.canvas.draw()




class _WidgetPlot(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = _PlotCanvas(self)
        self.layout().addWidget(self.canvas)


class _PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure()
        FigureCanvasQTAgg.__init__(self, self.fig)

        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
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
        self.ax11.imshow(image, cmap="gray")

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


def main():

    app = QtWidgets.QApplication([])
    qt_app = GUIMillingWindow()
    qt_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
