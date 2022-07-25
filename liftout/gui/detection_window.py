

from dataclasses import dataclass
import logging
import sys

import matplotlib.patches as mpatches
from liftout import utils
from liftout.detection import detection
from liftout.detection.utils import (DetectionFeature, DetectionResult,
                                     DetectionType, Point,
                                     convert_pixel_distance_to_metres)
from liftout.fibsem import acquire, calibration, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem.acquire import ImageSettings, BeamType
from liftout.fibsem.sample import Lamella
from liftout.gui import utils as ui_utils
from liftout.gui.qtdesigner_files import detection_dialog as detection_gui
from PyQt5 import QtCore, QtWidgets

from liftout.config import config


@dataclass
class DetectionData:
    detection_result: DetectionResult
    colour: list[tuple]
    image_coordinate: list[Point]
    microscope_coordinate: list[Point]

class GUIDetectionWindow(detection_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, microscope, settings: dict, image_settings: ImageSettings, 
        detection_result: DetectionResult,  lamella: Lamella, parent=None):
        super(GUIDetectionWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.microscope = microscope
        self.settings = settings
        self.image_settings = image_settings
        self.detection_result = detection_result
        self.lamella = lamella

        # images
        self.adorned_image = self.detection_result.adorned_image
        self.image = self.detection_result.display_image
        self._IMAGE_SAVED = False
        self._USER_CORRECTED = True

        # pattern drawing
        self.wp = ui_utils._WidgetPlot(self, display_image=self.image)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)

        self.wp.canvas.mpl_connect('button_press_event', self.on_click)



        self.det_types = (self.detection_result.feature_1.detection_type,
                          self.detection_result.feature_2.detection_type)
        self.current_selected_feature = 0
        self.current_detection_selected = self.det_types[self.current_selected_feature]


        self.det_data_v2 = DetectionData(
            detection_result=detection_result,
            colour=[
                 config.DETECTION_TYPE_COLOURS[self.detection_result.feature_1.detection_type],
                 config.DETECTION_TYPE_COLOURS[self.detection_result.feature_2.detection_type]
            ],
            image_coordinate=[
                self.detection_result.feature_1.feature_px,
                self.detection_result.feature_2.feature_px
            ],
            microscope_coordinate=[
                Point(0, 0),
                Point(0, 0)
            ]
        )    


        self.logged_detection_types = []

        self.setup_connections()

        self.update_display()

    def setup_connections(self):

        self.comboBox_detection_type.clear()
        self.comboBox_detection_type.addItems([det.name for det in self.det_types])
        self.comboBox_detection_type.setCurrentText(self.det_types[0].name)
        self.comboBox_detection_type.currentTextChanged.connect(self.update_detection_type)

        self.pushButton_continue.clicked.connect(self.continue_button_pressed)

    def update_detection_type(self):

        self.current_selected_feature = self.comboBox_detection_type.currentIndex()
        self.current_detection_selected = self.det_types[self.current_selected_feature]
        logging.info(f"Changed to {self.current_selected_feature}, {self.current_detection_selected.name}")

    def continue_button_pressed(self):

        logging.info(f"Continue button pressed: {self.sender()}")

        self.log_active_learning_data()

        # self.close()
    
    def closeEvent(self, event):
        logging.info("Closing Detection Window")

        # log correct detection types
        try:
            petname = self.lamella._petname
            current_stage = self.lamella.current_state.stage
            for det_type in self.det_types:
                if det_type not in self.logged_detection_types:
                    logging.info(f"{petname} | {current_stage} | ml_detection | {self.current_detection_selected} | {True}")
        except:
            pass

        event.accept()

    def on_click(self, event):
        """Redraw the patterns and update the display on user click"""
        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata
            self.center_x, self.center_y = movement.pixel_to_realspace_coordinate(
                (self.xclick, self.yclick), self.adorned_image)

            logging.info(f"on_click: {event.button} | IMAGE COORDINATE | {self.xclick:.2e}, {self.yclick:.2e}")
            logging.info(f"on_click: {event.button} | REAL COORDINATE | {self.center_x:.2e}, {self.center_y:.2e}")


            # update detection data
            self.det_data_v2.image_coordinate[self.current_selected_feature] = Point(self.xclick, self.yclick)
            self.det_data_v2.microscope_coordinate[self.current_selected_feature] = Point(self.center_x, self.center_y)

            # logging statistics
            petname = self.lamella._petname
            current_stage = self.lamella.current_state.stage
            logging.info(f"{petname} | {current_stage} | ml_detection | {self.current_detection_selected} | {False}")
            
            self.logged_detection_types.append(self.current_detection_selected)


            # flag that the user corrected a detection.
            self._USER_CORRECTED = True

            # save to disk

            # req info
            # image
            # image coordinate
            # detection type
            # flag for completion




            # DO this on close...









            self.update_display()

    def update_display(self):
        """Update the window display. Redraw the crosshair"""

        # TODO: daataclass the data?

        # redraw all crosshairs
        self.wp.canvas.ax11.patches.clear()

        # draw cross hairs
        ui_utils.draw_crosshair(self.image, self.wp.canvas, x=self.det_data_v2.image_coordinate[0].x, y=self.det_data_v2.image_coordinate[0].y, colour=self.det_data_v2.colour[0])
        ui_utils.draw_crosshair(self.image, self.wp.canvas, x=self.det_data_v2.image_coordinate[1].x, y=self.det_data_v2.image_coordinate[1].y, colour=self.det_data_v2.colour[1])

        # draw arrow
        point_1 = self.det_data_v2.image_coordinate[0]
        point_2 = self.det_data_v2.image_coordinate[1]

        x1, y1 = point_1.x, point_1.y
        x2, y2 = point_2.x, point_2.y
        line = mpatches.Arrow(x1, y1, x2-x1, y2-y1, color="white")

        self.wp.canvas.ax11.add_patch(line)

        # legend
        patch_one = mpatches.Patch(color=self.det_data_v2.colour[0], label=self.det_types[0].name)
        patch_two = mpatches.Patch(color=self.det_data_v2.colour[1], label=self.det_types[1].name)
        self.wp.canvas.ax11.legend(handles=[patch_one, patch_two])

        # calculate movement distance
        x_distance_m, y_distance_m = convert_pixel_distance_to_metres(
            point_1, point_2, self.adorned_image, self.image)
        self.det_data_v2.detection_result.distance_metres = Point(
            x_distance_m, y_distance_m)  # move from 1 to 2 (reverse direction)

        # update labels
        self.label_movement_header.setText(f"Movement")
        self.label_movement_header.setStyleSheet("font-weight:bold")
        self.label_movement.setText(f"""Moving {self.det_types[0].name} to {self.det_types[1].name}
         \nx distance: {self.det_data_v2.detection_result.distance_metres.x*1e6:.2f}um 
         \ny distance: {self.det_data_v2.detection_result.distance_metres.y*1e6:.2f}um""")

        self.wp.canvas.draw()

    def log_active_learning_data(self):

        logging.info(f"IM LOGGING DATA WOOW")


        if self._USER_CORRECTED:
            label = self.image_settings.label+"_label"
            utils.save_image(image=self.adorned_image, save_path=self.image_settings.save_path, label=label)
            self._IMAGE_SAVED = True

            # get info 
            logging.info(f"Label: {label}")

            info = list(zip(self.det_types, self.det_data_v2.image_coordinate))
            info.insert(0, label)
            logging.info(f"Data: ", info)
            logging.info(f"Finished logging data...")

            # save to file....
            # TODO: START_HERE log this data to a file, find a way to restore it. 
            # NB: it is not the resized images coordinate system, but the downscaled!!!!
            # THIS should also be resolved ASAP, no more downscale

        # TODO: flag if detection has been corrected...



def main():


    microscope, settings, image_settings = fibsem_utils.quick_setup()
    
    lamella = Lamella("tools/test", 999, _petname="999-test-mule")
    image_settings.save_path = lamella.path

    print(image_settings.save_path)

    import os
    os.makedirs(image_settings.save_path, exist_ok=True)

    app = QtWidgets.QApplication([])

    calibration.validate_detection_v2(
        microscope,
        settings,
        image_settings,
        lamella,
        shift_type=(DetectionType.ImageCentre, DetectionType.LamellaCentre),
        beam_type=BeamType.ELECTRON,
        parent=None
    )
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
