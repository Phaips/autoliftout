
from PyQt5 import QtWidgets, QtCore
from liftout.fibsem.milling import BeamType
from liftout.gui.milling_window import MillingPattern
from liftout.gui.qtdesigner_files import detection_dialog as detection_gui
import sys
from liftout import utils, detection
from liftout.fibsem import acquire, movement
from liftout.fibsem import utils as fibsem_utils

from liftout.gui import utils as gui_utils
from enum import Enum

class DetectionType(Enum):
    LamellaCentre = 1
    NeedleTip = 2
    LamellaEdge = 3
    LandingPost = 4 


class GUIDetectionWindow(detection_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, microscope, settings: dict, image_settings: dict, detection_type: DetectionType = None,  parent=None):
        super(GUIDetectionWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.microscope = microscope
        self.settings = settings
        self.image_settings = image_settings
        self.detection_type = detection_type

        self.adorned_image = acquire.last_image(self.microscope, beam_type=BeamType.ION)
        global image
        self.image = self.adorned_image.data
        image = self.image

        # pattern drawing
        self.wp = gui_utils._WidgetPlot(self, display_image=image)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)

        self.wp.canvas.mpl_connect('button_press_event', self.on_click)


        self.detection_type_colours = {
            DetectionType.LamellaCentre: (1, 0, 0, 1),
            DetectionType.NeedleTip: (0, 1, 0, 1),
            DetectionType.LamellaEdge: (1, 0, 0, 1),
            DetectionType.LandingPost: (0, 1, 1, 1)
        }

        # TODO: use this instead of others...
        self.detection_data = {}
        self.det_types = [det for det in DetectionType][:2]

        for det in self.det_types:
            self.detection_data[det] = {
                "colour": self.detection_type_colours[det],
                "microscope_coordinate": (0, 0), # tuple (x, y)
                "image_coordinate": (0, 0), # tuple (x, y)
                "crosshair": [],
            }

        self.setup_connections()
    
    def setup_connections(self):
        
        self.comboBox_detection_type.clear()
        self.comboBox_detection_type.addItems([det.name for det in self.det_types])
        self.comboBox_detection_type.setCurrentText(self.detection_type.name)
        self.comboBox_detection_type.currentTextChanged.connect(self.update_detection_type)
        
    
    def update_detection_type(self):

        current_detection_type = self.comboBox_detection_type.currentText()

        print(f"Changed to {current_detection_type}")

        self.detection_type = DetectionType[current_detection_type]

    def continue_button_pressed(self):

        print(f"Continue button pressed: {self.sender()}")

    def on_click(self, event):
            """Redraw the patterns and update the display on user click"""
            if event.button == 1 and event.inaxes is not None:
                self.xclick = event.xdata
                self.yclick = event.ydata
                self.center_x, self.center_y = movement.pixel_to_realspace_coordinate(
                    (self.xclick, self.yclick), self.adorned_image)
                
                print("on_click: ", self.center_x, self.center_y)


                # update detection data
                self.detection_data[self.detection_type]["image_coordinate"] = (self.xclick, self.yclick)
                self.detection_data[self.detection_type]["microscope_coordinate"] = (self.center_x, self.center_y)
                
                self.update_display()

                # TODO: get the initial positions for the detections in crosshair space
                # TODO: only remove if it is the same detection type...
                # TODO: draw a line between the two...
                # TODO: calculate the actual distance...



    def update_display(self):
        """Update the window display. Redraw the crosshair"""
        colour = self.detection_data[self.detection_type]["colour"]
        crosshair = gui_utils.create_crosshair(self.image, x=self.xclick, y=self.yclick, colour=colour)
        self.detection_data[self.detection_type]["crosshair"] = crosshair       
        
        from pprint import pprint
        print("-"*50)
        pprint(self.detection_data)
        # redraw all crosshairs
        self.wp.canvas.ax11.patches = []
        for det_type, data in self.detection_data.items():
            crosshair = data["crosshair"]
            if isinstance(crosshair, gui_utils.Crosshair):
                print(f"Drawing crosshair for {det_type.name}")
                for patch in crosshair.__dataclass_fields__:
                    self.wp.canvas.ax11.add_patch(getattr(crosshair, patch))
                    getattr(crosshair, patch).set_visible(True)


        x1, y1, = self.detection_data[self.det_types[0]]["image_coordinate"]
        x2, y2, = self.detection_data[self.det_types[1]]["image_coordinate"]

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        line =  mpatches.Arrow(x1, y1, x2-x1, y2-y1, color="white")

        self.wp.canvas.ax11.add_patch(line)
        print(x1, y1)
        print(x2, y2)
    
        # self.draw_milling_patterns()
        # for rect in self.pattern_rectangles:
        #     self.wp.canvas.ax11.add_patch(rect)

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
    qt_app = GUIDetectionWindow(microscope=microscope, 
                                settings=settings, 
                                image_settings=image_settings, 
                                detection_type=DetectionType.NeedleTip
                                )
    qt_app.show()
    qt_app.exec_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
