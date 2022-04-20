

import sys
from enum import Enum

import logging
from attr import dataclass

from liftout import detection, utils
from liftout.fibsem import acquire, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem.milling import BeamType
from liftout.gui import utils as gui_utils
from liftout.gui.qtdesigner_files import detection_dialog as detection_gui
from PyQt5 import QtCore, QtWidgets

import numpy as np
from liftout.detection import utils as detection_utils
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from autoscript_sdb_microscope_client.structures import AdornedImage
from liftout.fibsem import calibration

class DetectionType(Enum):
    LamellaCentre = 1
    NeedleTip = 2
    LamellaEdge = 3
    LandingPost = 4
    ImageCentre = 5 

@dataclass
class Point:
    x: float
    y: float


@dataclass
class DetectionFeature:
    detection_type: DetectionType
    feature_px: Point # x, y


@dataclass
class DetectionResult:
    feature_1: DetectionFeature
    feature_2: DetectionFeature
    adorned_image: AdornedImage
    display_image: np.ndarray
    distance_metres: Point # x, y
    



class GUIDetectionWindow(detection_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, microscope, settings: dict, image_settings: dict, detection_result: DetectionResult,  parent=None):
        super(GUIDetectionWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)


        # TODO: dont know if we need the microscope, settings, etc
        # detection_type: (moving, stationary) tuple(DetectionType, DetectionType)

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
        self.det_types = (self.detection_result.feature_1.detection_type, self.detection_result.feature_2.detection_type)
        self.current_detection_selected = self.det_types[0]

        self.detection_data[self.detection_result.feature_1.detection_type] = {
                "colour": self.detection_type_colours[self.detection_result.feature_1.detection_type],
                "microscope_coordinate": (0, 0), # tuple (x, y) #TODO: initialise?
                "image_coordinate": self.detection_result.feature_1.feature_px, # Point (x, y)
                "crosshair": [],
        }

        self.detection_data[self.detection_result.feature_2.detection_type] = {
            "colour": self.detection_type_colours[self.detection_result.feature_2.detection_type],
            "microscope_coordinate": (0, 0), # tuple (x, y) #TODO: initialise?
            "image_coordinate": self.detection_result.feature_2.feature_px, # Point (x, y)
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
                self.detection_data[self.current_detection_selected]["image_coordinate"] = Point(self.xclick, self.yclick)
                self.detection_data[self.current_detection_selected]["microscope_coordinate"] = Point(self.center_x, self.center_y)
                
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

        line =  mpatches.Arrow(x1, y1, x2-x1, y2-y1, color="white")

        self.wp.canvas.ax11.add_patch(line)
        
        def convert_pixel_distance_to_metres(p1:Point, p2: Point, adorned_image: AdornedImage, display_image: np.ndarray):
            """Convert from pixel coordinates to distance in metres """        
            # NB: need to use this func, not pixel_to_realspace because display_iamge and adorned image are no the same size...
            
            # upscale the pixel coordinates to adorned image size
            scaled_px_1 = scale_pixel_coordinates(p1, display_image, adorned_image)
            scaled_px_2 = scale_pixel_coordinates(p2, display_image, adorned_image)

            # convert pixel coordinate to realspace coordinate
            x1_real, y1_real = movement.pixel_to_realspace_coordinate((scaled_px_1.x, scaled_px_1.y), adorned_image)
            x2_real, y2_real = movement.pixel_to_realspace_coordinate((scaled_px_2.x, scaled_px_2.y), adorned_image)
            
            p1_real = Point(x1_real, y1_real)
            p2_real = Point(x2_real, y2_real)

            # calculate distance between points along each axis
            x_distance_m, y_distance_m = coordinate_distance(p1_real, p2_real)

            return x_distance_m, y_distance_m

        def scale_pixel_coordinates(px:Point, downscale_image, upscale_image=None):
            """Scale the pixel coordinate from one image to another"""
            if isinstance(upscale_image, AdornedImage):
                upscale_image = upscale_image.data

            x_scale, y_scale = (px.x / downscale_image.shape[1], px.y / downscale_image.shape[0])  # (x, y)
            
            scaled_px = Point(x_scale * upscale_image.shape[1], y_scale * upscale_image.shape[0])

            return scaled_px

        def coordinate_distance(p1:Point, p2:Point):
            """Calculate the distance between two points in each coordinate"""

            return p2.x - p1.x, p2.y - p1.y

        # calculate movement distance
        x_distance_m, y_distance_m = convert_pixel_distance_to_metres(point_1, point_2, self.adorned_image, self.image)
        self.detection_result.distance_metres = Point(x_distance_m, y_distance_m) # move from 1 to 2 (reverse direction)


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

    ret = calibration.identify_shift_using_machine_learning(microscope, 
                                                image_settings, 
                                                settings, 
                                                shift_type="needle_tip_to_image_centre") 

    adorned_image, overlay_image, downscale_image, feature_1_px, feature_1_type, feature_2_px, feature_2_type = ret 

    det_type_dict = {
        "needle_tip": DetectionType.NeedleTip,
        "lamella_centre": DetectionType.LamellaCentre,
        "image_centre": DetectionType.ImageCentre,
        "lamella_edge": DetectionType.LamellaEdge,
        "landing_post": DetectionType.LandingPost,
    }
    
    # TODO: double check the direction consistency
    feature_1 = DetectionFeature(
        detection_type=det_type_dict[feature_1_type],
        feature_px=Point(*feature_1_px),
    )

    feature_2 = DetectionFeature(
        detection_type=det_type_dict[feature_2_type],
        feature_px=Point(*feature_2_px),
    )

    detection_result = DetectionResult(
        feature_1=feature_1,
        feature_2=feature_2,
        adorned_image=adorned_image,
        display_image=downscale_image,
        distance_metres=None
    )

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
