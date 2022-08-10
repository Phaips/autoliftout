import logging
import os
import sys
from pprint import pprint

import matplotlib.patches as mpatches
from fibsem import calibration, movement
from liftout import utils
from fibsem.structures import BeamType
from liftout.config import config
from liftout.detection import utils as det_utils
from liftout.detection.utils import (DetectionResult, DetectionType, Point,
                                     convert_pixel_distance_to_metres)
from liftout.gui import utils as ui_utils
from liftout.gui.qtdesigner_files import detection_dialog as detection_gui
from liftout.sample import Lamella
from PyQt5 import QtCore, QtWidgets


class GUIDetectionWindow(detection_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        microscope,
        settings: dict,
        detection_result: DetectionResult,
        lamella: Lamella,
    ):
        super(GUIDetectionWindow, self).__init__()
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        # microscope settings
        self.microscope = microscope
        self.settings = settings
        self.lamella = lamella

        # detection data
        self.detection_result = detection_result
        self.current_selected_feature = 0
        self.current_detection_selected = (
            self.detection_result.features[
                self.current_selected_feature
            ].detection_type,
        )
        self.logged_detection_types = []

        # images
        self.adorned_image = self.detection_result.adorned_image
        self.image = self.detection_result.display_image
        self._USER_CORRECTED = False

        # pattern drawing
        self.wp = ui_utils._WidgetPlot(self, display_image=self.image)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)
        self.wp.canvas.mpl_connect("button_press_event", self.on_click)

        self.setup_connections()
        self.update_display()

        AUTO_CONTINUE = False
        if AUTO_CONTINUE:
            self.continue_button_pressed() # automatically continue

    def setup_connections(self):

        self.comboBox_detection_type.clear()
        self.comboBox_detection_type.addItems(
            [feature.detection_type.name for feature in self.detection_result.features]
        )
        self.comboBox_detection_type.setCurrentText(
            self.detection_result.features[0].detection_type.name
        )
        self.comboBox_detection_type.currentTextChanged.connect(
            self.update_detection_type
        )

        self.pushButton_continue.clicked.connect(self.continue_button_pressed)

    def update_detection_type(self):

        self.current_selected_feature = self.comboBox_detection_type.currentIndex()
        self.current_detection_selected = self.detection_result.features[
            self.current_selected_feature
        ].detection_type
        logging.info(
            f"Changed to {self.current_selected_feature}, {self.current_detection_selected.name}"
        )

    def on_click(self, event):
        """Redraw the patterns and update the display on user click"""
        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata
            self.center_x, self.center_y = calibration.pixel_to_realspace_coordinate(
                (self.xclick, self.yclick), self.adorned_image
            )

            logging.info(
                f"""on_click: {event.button} | {self.current_detection_selected} | IMAGE COORDINATE | {int(self.xclick)}, {int(self.yclick)} | REAL COORDINATE | {self.center_x:.2e}, {self.center_y:.2e}"""
            )

            # update detection data
            self.detection_result.features[
                self.current_selected_feature
            ].feature_px = Point(self.xclick, self.yclick)
            self.detection_result.microscope_coordinate[
                self.current_selected_feature
            ] = Point(self.center_x, self.center_y)

            # logging statistics
            petname = self.lamella._petname
            current_stage = self.lamella.current_state.stage
            logging.info(
                f"{petname} | {current_stage} | ml_detection | {self.current_detection_selected} | {False}"
            )

            self.logged_detection_types.append(self.current_detection_selected)

            # flag that the user corrected a detection.
            self._USER_CORRECTED = True

            self.update_display()

    def update_display(self):
        """Update the window display. Redraw the crosshair"""

        # TODO: consolidate with plot_detection_result

        # point position, image coordinates
        point_1 = self.detection_result.features[0].feature_px
        point_2 = self.detection_result.features[1].feature_px

        # colours
        c1 = config.DETECTION_TYPE_COLOURS[
            self.detection_result.features[0].detection_type
        ]
        c2 = config.DETECTION_TYPE_COLOURS[
            self.detection_result.features[1].detection_type
        ]

        # redraw all crosshairs
        self.wp.canvas.ax11.patches.clear()

        # draw cross hairs
        ui_utils.draw_crosshair(
            self.image, self.wp.canvas, x=point_1.x, y=point_1.y, colour=c1,
        )
        ui_utils.draw_crosshair(
            self.image, self.wp.canvas, x=point_2.x, y=point_2.y, colour=c2,
        )

        # draw arrow
        ui_utils.draw_arrow(point_1, point_2, self.wp.canvas)

        # legend
        patch_one = mpatches.Patch(
            color=c1, label=self.detection_result.features[0].detection_type.name
        )
        patch_two = mpatches.Patch(
            color=c2, label=self.detection_result.features[1].detection_type.name
        )
        self.wp.canvas.ax11.legend(handles=[patch_one, patch_two])

        # calculate movement distance
        x_distance_m, y_distance_m = convert_pixel_distance_to_metres(
            point_1, point_2, self.adorned_image
        )
        self.detection_result.distance_metres = Point(
            x_distance_m, y_distance_m
        )  # move from 1 to 2 (reverse direction)

        # update labels
        self.label_movement_header.setText(f"Movement")
        self.label_movement_header.setStyleSheet("font-weight:bold")
        self.label_movement.setText(
            f"""Moving {self.detection_result.features[0].detection_type.name} to {self.detection_result.features[1].detection_type.name}
         \nx distance: {self.detection_result.distance_metres.x*1e6:.2f}um 
         \ny distance: {self.detection_result.distance_metres.y*1e6:.2f}um"""
        )

        self.wp.canvas.draw()
    
    def continue_button_pressed(self):

        logging.info(f"Continue button pressed: {self.sender()}")

        self.close() #exit

    def closeEvent(self, event):
        logging.info("Closing Detection Window")

        # log active learning data...
        logging.info(f"Writing machine learning data to disk...")
        if self._USER_CORRECTED:
            path = os.path.join(self.lamella.base_path, "label")
            det_utils.write_data_to_disk(path, self.detection_result)

        # log correct detection types
        petname = self.lamella._petname
        current_stage = self.lamella.current_state.stage
        for feature in self.detection_result.features:
            if feature.detection_type not in self.logged_detection_types:
                logging.info(
                    f"{petname} | {current_stage} | ml_detection | {self.current_detection_selected} | {True}"
                )

        event.accept()

def main():

    microscope, settings, image_settings, sample, lamella = utils.full_setup()
    
    app = QtWidgets.QApplication([])

    from liftout.gui import windows
    from liftout.detection.detection import DetectionFeature

    # select features
    features = [DetectionFeature(detection_type=DetectionType.ImageCentre, feature_px=None),
                DetectionFeature(detection_type=DetectionType.LamellaCentre, feature_px=None)]
    det = windows.detect_features(microscope=microscope, 
        settings=settings, image_settings=image_settings, lamella=lamella, ref_image=None, features=features, validate=True)


    pprint(det)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
