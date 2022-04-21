

import logging
import sys

import matplotlib.patches as mpatches
from liftout import utils
from liftout.detection.utils import (DetectionFeature, DetectionResult,
                                     DetectionType, Point,
                                     convert_pixel_distance_to_metres)
from liftout.fibsem import acquire, calibration, movement
from liftout.detection import detection
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem.milling import BeamType
from liftout.gui import utils as gui_utils
from liftout.gui.qtdesigner_files import detection_dialog as detection_gui
from PyQt5 import QtCore, QtWidgets


class GUIDetectionWindow(detection_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, microscope, settings: dict, image_settings: dict, detection_result: DetectionResult,  parent=None):
        super(GUIDetectionWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.microscope = microscope
        self.settings = settings
        self.image_settings = image_settings
        self.detection_result = detection_result

        # images
        self.adorned_image = self.detection_result.adorned_image
        self.image = self.detection_result.display_image

        # pattern drawing
        self.wp = gui_utils._WidgetPlot(self, display_image=self.image)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)

        self.wp.canvas.mpl_connect('button_press_event', self.on_click)

        self.detection_type_colours = {
            DetectionType.LamellaCentre: (1, 0, 0, 1),
            DetectionType.NeedleTip: (0, 1, 0, 1),
            DetectionType.LamellaEdge: (1, 0, 0, 1),
            DetectionType.LandingPost: (0, 1, 1, 1),
            DetectionType.ImageCentre: (1, 1, 1, 1)
        }

        self.detection_data = {}
        self.det_types = (self.detection_result.feature_1.detection_type,
                          self.detection_result.feature_2.detection_type)
        self.current_detection_selected = self.det_types[0]

        self.detection_data[self.detection_result.feature_1.detection_type] = {
            "colour": self.detection_type_colours[self.detection_result.feature_1.detection_type],
            "microscope_coordinate": (0, 0),  # tuple (x, y) #TODO: initialise?
            "image_coordinate": self.detection_result.feature_1.feature_px,  # Point (x, y)
            "crosshair": [],
        }

        self.detection_data[self.detection_result.feature_2.detection_type] = {
            "colour": self.detection_type_colours[self.detection_result.feature_2.detection_type],
            "microscope_coordinate": (0, 0),  # tuple (x, y) #TODO: initialise?
            "image_coordinate": self.detection_result.feature_2.feature_px,  # Point (x, y)
            "crosshair": [],
        }

        self.setup_connections()

        self.update_display()

    def setup_connections(self):

        self.comboBox_detection_type.clear()
        self.comboBox_detection_type.addItems([det.name for det in self.det_types])
        self.comboBox_detection_type.setCurrentText(self.det_types[0].name)
        self.comboBox_detection_type.currentTextChanged.connect(self.update_detection_type)

        self.pushButton_continue.clicked.connect(self.continue_button_pressed)

    def update_detection_type(self):

        current_detection_type = self.comboBox_detection_type.currentText()

        logging.info(f"Changed to {current_detection_type}")

        self.current_detection_selected = DetectionType[current_detection_type]

    def continue_button_pressed(self):

        logging.info(f"Continue button pressed: {self.sender()}")

        # TODO: do movement...
        self.close()

    def on_click(self, event):
        """Redraw the patterns and update the display on user click"""
        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata
            self.center_x, self.center_y = movement.pixel_to_realspace_coordinate(
                (self.xclick, self.yclick), self.adorned_image)

            logging.info(f"on_click: {event.button} | {self.center_x}, {self.center_y}")

            # update detection data
            self.detection_data[self.current_detection_selected]["image_coordinate"] = Point(
                self.xclick, self.yclick)
            self.detection_data[self.current_detection_selected]["microscope_coordinate"] = Point(
                self.center_x, self.center_y)

            self.update_display()

    def update_display(self):
        """Update the window display. Redraw the crosshair"""

        # redraw all crosshairs
        self.wp.canvas.ax11.patches = []
        for det_type, data in self.detection_data.items():
            crosshair = gui_utils.create_crosshair(self.image,
                                                   x=data["image_coordinate"].x,
                                                   y=data["image_coordinate"].y,
                                                   colour=data["colour"])
            data["crosshair"] = crosshair
            if isinstance(crosshair, gui_utils.Crosshair):
                for patch in crosshair.__dataclass_fields__:
                    self.wp.canvas.ax11.add_patch(getattr(crosshair, patch))
                    getattr(crosshair, patch).set_visible(True)

        # draw arrow
        point_1 = self.detection_data[self.det_types[0]]["image_coordinate"]
        point_2 = self.detection_data[self.det_types[1]]["image_coordinate"]

        x1, y1 = point_1.x, point_1.y
        x2, y2 = point_2.x, point_2.y

        line = mpatches.Arrow(x1, y1, x2-x1, y2-y1, color="white")

        self.wp.canvas.ax11.add_patch(line)

        # calculate movement distance
        x_distance_m, y_distance_m = convert_pixel_distance_to_metres(
            point_1, point_2, self.adorned_image, self.image)
        self.detection_result.distance_metres = Point(
            x_distance_m, y_distance_m)  # move from 1 to 2 (reverse direction)

        # update labels
        self.label_movement_header.setText(f"Movement")
        self.label_movement_header.setStyleSheet("font-weight:bold")
        self.label_movement.setText(f"""Moving {self.det_types[0].name} to {self.det_types[1].name}
         \nx distance: {self.detection_result.distance_metres.x*1e6:.2f}um 
         \ny distance: {self.detection_result.distance_metres.y*1e6:.2f}um""")

        self.wp.canvas.draw()


def main():

    settings = utils.load_config(r"C:\Users\Admin\Github\autoliftout\liftout\protocol_liftout.yml")
    microscope = fibsem_utils.initialise_fibsem(ip_address=settings["system"]["ip_address"])
    image_settings = {
        "resolution": settings["imaging"]["resolution"],
        "dwell_time": settings["imaging"]["dwell_time"],
        "hfw": settings["imaging"]["horizontal_field_width"],
        "autocontrast": True,
        "beam_type": BeamType.ION,
        "gamma": settings["gamma"],
        "save": False,
        "label": "test",
    }
    app = QtWidgets.QApplication([])

    detection_result = calibration.identify_shift_using_machine_learning(microscope,
                                                            image_settings,
                                                            settings,
                                                            shift_type=(DetectionType.NeedleTip, DetectionType.ImageCentre))

    qt_app = GUIDetectionWindow(microscope=microscope,
                                settings=settings,
                                image_settings=image_settings,
                                detection_result=detection_result,
                                )
    qt_app.show()
    qt_app.exec_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
