
import logging
import sys
from typing import Iterable
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from autoscript_sdb_microscope_client.structures import AdornedImage
from liftout import fibsem, utils
from liftout.fibsem import acquire, milling, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem.acquire import BeamType
from liftout.gui.qtdesigner_files import milling_dialog as milling_gui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from liftout.gui.utils import _PlotCanvas, _WidgetPlot, create_crosshair

MICRON_TO_METRE = 1e-6
METRE_TO_MICRON = 1e6



class MillingPattern(Enum):
    Trench = 1
    JCut = 2
    Sever = 3
    Weld = 4
    Cut = 5
    Sharpen = 6
    Thin = 7
    Polish = 8
    Flatten = 9


# TODO:
# separate thin and polish stages in the config...
# add rotation for plt rectangles in display...

class GUIMillingWindow(milling_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, microscope, settings: dict, image_settings: dict, milling_pattern_type: MillingPattern, parent=None):
        super(GUIMillingWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.microscope = microscope
        self.settings = settings
        self.image_settings = image_settings
        self.milling_pattern_type = milling_pattern_type

        self.wp = None # plotting widget
        self.USER_UPDATE = True
                
        # milling parameters
        self.parameter_labels = [self.label_01, self.label_02, self.label_03,
                                 self.label_04, self.label_05, self.label_06,
                                 self.label_07, self.label_08, self.label_09,
                                 self.label_10, self.label_11, self.label_12]

        self.parameter_values = [self.doubleSpinBox_01,
                                 self.doubleSpinBox_02,
                                 self.doubleSpinBox_03,
                                 self.doubleSpinBox_04,
                                 self.doubleSpinBox_05,
                                 self.doubleSpinBox_06,
                                 self.doubleSpinBox_07,
                                 self.doubleSpinBox_08,
                                 self.doubleSpinBox_09,
                                 self.doubleSpinBox_10,
                                 self.doubleSpinBox_11,
                                 self.doubleSpinBox_12
                                 ]

        # setup ui
        for label, spinBox in zip(self.parameter_labels, self.parameter_values):
            label.setVisible(False)
            spinBox.setVisible(False)

        self.non_changeable_params = ["milling_current", "hfw", "jcut_angle", "rotation_angle", "tilt_angle"]
        self.non_scaled_params = ["size_ratio", "rotation", "tip_angle",
                                  "needle_angle", "percentage_roi_height", "percentage_from_lamella_surface"]


        self.INITIALISED = False


    def setup_milling_window(self, x=0.0, y=0.0):
        """Setup the milling window"""

        self.milling_settings = None
        self.milling_stages = {}
        self.image_settings["beam_type"] = BeamType.ION
        self.adorned_image = acquire.new_image(self.microscope, self.image_settings)
        self.image = self.adorned_image.data

        # pattern drawing
        if self.wp is not None:
            self.label_image.layout().removeWidget(self.wp)
            self.wp.deleteLater()

        # pattern drawing
        self.wp = _WidgetPlot(self, display_image=self.image)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)
        self.wp.canvas.mpl_connect('button_press_event', self.on_click)

        # setup
        milling.setup_ion_milling(self.microscope, ion_beam_field_of_view=self.image_settings["hfw"])
        self.setup_milling_patterns()
        self.setup_connections()

        # initial pattern
        self.center_x, self.center_y = x, y
        self.xclick, self.yclick = None, None
        self.update_display()
        self.INITIALISED = True

    def setup_connections(self):
        """Setup connections for milling window"""
        
        if self.INITIALISED:
            # disconnect buttons if initialised
            self.pushButton_runMilling.clicked.disconnect()
            for param_spinBox in self.parameter_values:
                param_spinBox.valueChanged.disconnect()
            self.comboBox_pattern_stage.currentTextChanged.disconnect()


        # reconnect buttons
        self.pushButton_runMilling.clicked.connect(self.run_milling_button_pressed)

        for param_spinBox in self.parameter_values:
            param_spinBox.valueChanged.connect(self.update_milling_settings)


        self.comboBox_pattern_stage.clear()
        milling_keys_list = list(self.milling_stages.keys())
        self.comboBox_pattern_stage.addItems(milling_keys_list)
        self.comboBox_pattern_stage.currentTextChanged.connect(self.update_stage_settings)

    def update_milling_pattern_type(self, milling_pattern: MillingPattern, x=0.0, y=0.0):
        """Update the milling pattern type and reset the milling window

        Args:
            milling_pattern (MillingPattern): desired milling pattern
        """

        # update milling pattern
        self.milling_pattern_type = milling_pattern

        # update to latest imaging settings
        if self.parent():
            self.image_settings = self.parent().image_settings
        
        self.setup_milling_window(x, y)
        self.show()
        self.exec_()

    def update_stage_settings(self):
        """Update the current milling stage and settings"""
        
        key = self.comboBox_pattern_stage.currentText()
        try:
            self.milling_settings = self.milling_stages[key]
            logging.info(f"Stage: {key} - Milling Settings: {self.milling_stages[key]}")

            self.update_parameter_elements()
            self.update_display()
        except:
            logging.warning(f"Pattern not found in Milling Stages")

    def update_milling_settings(self):
        """Update the milling settings when parameter ui elements are changed"""
        for i, param_spinBox in enumerate(self.parameter_values):

            if param_spinBox == self.sender():
                param = self.parameter_labels[i].text()
                param_value = param_spinBox.value()
                if param not in self.non_scaled_params:
                    param_value = param_value * MICRON_TO_METRE
                self.milling_settings[param] = param_value

                if self.USER_UPDATE:
                    self.update_display()



    def update_display(self):
        """Update the millig window display. Redraw the crosshair, and milling patterns"""
        crosshair = create_crosshair(self.image, x=self.xclick, y=self.yclick)
        self.wp.canvas.ax11.patches = []
        for patch in crosshair.__dataclass_fields__:
            self.wp.canvas.ax11.add_patch(getattr(crosshair, patch))
            getattr(crosshair, patch).set_visible(True)

        self.draw_milling_patterns()

        for rect in self.pattern_rectangles:
            self.wp.canvas.ax11.add_patch(rect)
        self.wp.canvas.draw()

    def on_click(self, event):
        """Redraw the patterns and update the display on user click"""
        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata
            self.center_x, self.center_y = movement.pixel_to_realspace_coordinate(
                (self.xclick, self.yclick), self.adorned_image)

            self.update_display()

    def setup_milling_patterns(self):
        """Load the milling stages and settings for the select milling pattern type"""
 
        if self.milling_pattern_type == MillingPattern.Trench:
            milling_protocol_stages = milling.get_milling_protocol_stages(settings=self.settings, stage_name="lamella")
            
        if self.milling_pattern_type == MillingPattern.JCut:
            milling_protocol_stages = self.settings["jcut"]

        if self.milling_pattern_type == MillingPattern.Sever:
            milling_protocol_stages = self.settings["jcut"]

        if self.milling_pattern_type == MillingPattern.Weld:
            milling_protocol_stages = self.settings["weld"]

        if self.milling_pattern_type == MillingPattern.Cut:
            milling_protocol_stages = self.settings["cut"]

        if self.milling_pattern_type == MillingPattern.Sharpen:
            milling_protocol_stages = self.settings["sharpen"]

        if self.milling_pattern_type == MillingPattern.Thin:
            milling_protocol_stages = milling.get_milling_protocol_stages(
                settings=self.settings, stage_name="thin_lamella")

        if self.milling_pattern_type == MillingPattern.Polish:
            milling_protocol_stages = milling.get_milling_protocol_stages(
                settings=self.settings, stage_name="thin_lamella")

        if self.milling_pattern_type == MillingPattern.Flatten:
            milling_protocol_stages = self.settings["flatten_landing"]

        if isinstance(milling_protocol_stages, list):
            # generic
            for i, stage_settings in enumerate(milling_protocol_stages, 1):
                try:
                    # remove list element from settings
                    del stage_settings["protocol_stages"]
                except:
                    pass
                self.milling_stages[f"{self.milling_pattern_type.name}_{i}"] = stage_settings
        else:
            self.milling_stages[f"{self.milling_pattern_type.name}_1"] = milling_protocol_stages
        
        # set the first milling stage settings
        first_stage = list(self.milling_stages.keys())[0]
        self.milling_settings = self.milling_stages[first_stage]
        self.update_parameter_elements()

    def update_parameter_elements(self):
        """Update the parameter labels and values with updated milling settings"""
        # update ui elements
        self.USER_UPDATE = False
        i = 0
        for key, value in self.milling_settings.items():
            if key not in self.non_changeable_params:
                if key not in self.non_scaled_params:
                    value = float(value) * METRE_TO_MICRON
                self.parameter_labels[i].setText(key)
                self.parameter_values[i].setValue(value)
                self.parameter_labels[i].setVisible(True)
                self.parameter_values[i].setVisible(True)
                i += 1
            else:
                self.parameter_labels[i].setVisible(False)
                self.parameter_values[i].setVisible(False)
        
        self.USER_UPDATE = True

    def update_milling_patterns(self):
        """Redraw the milling patterns with updated milling settings"""

        if self.milling_pattern_type == MillingPattern.Trench:

            self.patterns = milling.mill_trench_patterns(microscope=self.microscope,
                                                         settings=self.milling_settings,
                                                         centre_x=self.center_x, centre_y=self.center_y)

        if self.milling_pattern_type == MillingPattern.JCut:

            self.patterns = milling.jcut_milling_patterns(microscope=self.microscope,
                                                          settings=self.settings, centre_x=self.center_x, centre_y=self.center_y)

        if self.milling_pattern_type == MillingPattern.Sever:

            self.patterns = milling.jcut_severing_pattern(microscope=self.microscope,
                                                          settings=self.settings, centre_x=self.center_x, centre_y=self.center_y)

        if self.milling_pattern_type == MillingPattern.Weld:

            self.patterns = milling.weld_to_landing_post(microscope=self.microscope, settings=self.settings,
                                                         centre_x=self.center_x, centre_y=self.center_y)

        if self.milling_pattern_type == MillingPattern.Cut:

            self.patterns = milling.cut_off_needle(microscope=self.microscope, settings=self.settings,
                                                   centre_x=self.center_x, centre_y=self.center_y)

        if self.milling_pattern_type == MillingPattern.Sharpen:

            cut_coord_bottom, cut_coord_top = milling.calculate_sharpen_needle_pattern(microscope=self.microscope, settings=self.settings,
                                                                                       x_0=self.center_x, y_0=self.center_y)

            self.patterns = milling.create_sharpen_needle_patterns(self.microscope, cut_coord_bottom, cut_coord_top)

        if self.milling_pattern_type == MillingPattern.Thin:
            self.patterns = milling.mill_thin_patterns(self.microscope, self.milling_settings, 
                                                        centre_x=self.center_x, centre_y=self.center_y)

        if self.milling_pattern_type == MillingPattern.Polish:
            self.patterns = milling.mill_thin_patterns(self.microscope, self.milling_settings, 
                                                        centre_x=self.center_x, centre_y=self.center_y)

        if self.milling_pattern_type == MillingPattern.Flatten:
            self.patterns = milling.flatten_landing_pattern(microscope=self.microscope, settings=self.settings,
                                                            centre_x=self.center_x, centre_y=self.center_y)
        # convert patterns is list
        if not isinstance(self.patterns, list):
            self.patterns = [self.patterns]

    def draw_milling_patterns(self):

        self.microscope.imaging.set_active_view(2)  # the ion beam view
        self.microscope.patterning.clear_patterns()
        try:
            self.update_milling_patterns()
        except Exception as e:
            logging.error(f"Error during milling pattern update: {e}")

        # create a rectangle for each pattern
        self.pattern_rectangles = []
        for pattern in self.patterns:
            rectangle = Rectangle((0, 0), 0.2, 0.2, color='yellow', fill=None, alpha=1)
            rectangle.set_visible(False)
            rectangle.set_hatch('//////')
            self.pattern_rectangles.append(rectangle)

        def draw_rectangle_pattern(adorned_image, rectangle, pattern):
            image_width = adorned_image.width
            image_height = adorned_image.height
            pixel_size = adorned_image.metadata.binary_result.pixel_size.x

            width = pattern.width / pixel_size
            height = pattern.height / pixel_size
            rectangle_left = (image_width / 2) + (pattern.center_x / pixel_size) - (width/2)
            rectangle_bottom = (image_height / 2) - (pattern.center_y / pixel_size) - (height/2)
            rectangle.set_width(width)
            rectangle.set_height(height)
            rectangle.set_xy((rectangle_left, rectangle_bottom))
            rectangle.set_visible(True)

        try:
            for pattern, rectangle in zip(self.patterns, self.pattern_rectangles):

                draw_rectangle_pattern(adorned_image=self.adorned_image,
                                       rectangle=rectangle, pattern=pattern)
        except Exception as e:
            # NOTE: these exceptions happen when the pattern is too far outside of the FOV
            logging.error(f"Pattern outside FOV: {e}") 

    def run_milling_button_pressed(self):
        """Run ion beam milling for the selected milling pattern"""

        logging.info("Run Milling Button Pressed")
        logging.info(f"Running milling for {len(self.milling_stages)} Stages")

        for stage_name, milling_settings in self.milling_stages.items():
            
            logging.info(f"Stage {stage_name}: {milling_settings}")

            self.milling_settings = milling_settings 
            self.update_milling_patterns()
            self.update_display()
            
            # run_milling
            milling.run_milling(microscope=self.microscope, settings=self.settings)

        # reset to imaging mode
        milling.finish_milling(microscope=self.microscope, settings=self.settings)

        self.close() 

    def closeEvent(self, event):
        logging.info("Closing Milling Window")
        self.microscope.patterning.clear_patterns()
        event.accept()


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
    qt_app = GUIMillingWindow(microscope=microscope, 
                                settings=settings, 
                                image_settings=image_settings, 
                                milling_pattern_type=MillingPattern.Trench)
    qt_app.show()
    qt_app.update_milling_pattern_type(MillingPattern.Flatten)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
