import logging
import sys
import time
from enum import Enum

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client.structures import \
    Rectangle as RectangleArea
from liftout.fibsem import acquire, milling, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem.acquire import BeamType, ImageSettings
from liftout.fibsem.constants import METRE_TO_MICRON, MICRON_TO_METRE
from liftout.gui.qtdesigner_files import milling_dialog as milling_gui
from liftout.gui.utils import _WidgetPlot, create_crosshair
from matplotlib.patches import Rectangle
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox


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
    Fiducial = 10


class GUIMillingWindow(milling_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, microscope, settings: dict, image_settings: ImageSettings, milling_pattern_type: MillingPattern, x: float = 0.0, y:float = 0.0, parent=None):
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

        # TODO: make milling current changable separately? combobox
        self.non_changeable_params = ["milling_current", "hfw", "jcut_angle", "rotation_angle", "tilt_angle", "tilt_offset", 
            "resolution", "dwell_time", "reduced_area"]
        self.non_scaled_params = ["size_ratio", "rotation", "tip_angle",
                                  "needle_angle", "percentage_roi_height", "percentage_from_lamella_surface"]

        self.INITIALISED = False

        self.update_milling_pattern_type(milling_pattern_type, x=x, y=y)

    def setup_milling_image(self):


        # image with a reduced area for thin/polishing.
        if self.milling_pattern_type in [MillingPattern.Thin, MillingPattern.Polish]:
            reduced_area = RectangleArea(0.3, 0.3, 0.4, 0.4)
        else: 
            reduced_area = None

        self.image_settings.beam_type = BeamType.ION
        self.adorned_image = acquire.new_image(self.microscope, self.image_settings, reduced_area=reduced_area)    
        self.image = ndi.median_filter(self.adorned_image.data, size=3)

        # pattern drawing
        if self.wp is not None:
            self.label_image.layout().removeWidget(self.wp)
            self.wp.deleteLater()

        # pattern drawing
        self.wp = _WidgetPlot(self, display_image=self.image)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)
        self.wp.canvas.mpl_connect('button_press_event', self.on_click)

    def setup_milling_window(self, x=0.0, y=0.0):
        """Setup the milling window"""

        self.milling_settings = None
        self.milling_stages = {}

        self.setup_milling_image()

        # setup
        milling.setup_ion_milling(self.microscope, ion_beam_field_of_view=self.image_settings.hfw)
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
        self.pushButton_exitMilling.clicked.connect(self.exit_milling_button_pressed)

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

    def update_display(self, draw_patterns=True):
        """Update the millig window display. Redraw the crosshair, and milling patterns"""
        crosshair = create_crosshair(self.image, x=self.xclick, y=self.yclick)
        self.wp.canvas.ax11.patches = []
        for patch in crosshair.__dataclass_fields__:
            self.wp.canvas.ax11.add_patch(getattr(crosshair, patch))
            getattr(crosshair, patch).set_visible(True)

        if draw_patterns:
            self.draw_milling_patterns()

            for rect in self.pattern_rectangles:
                self.wp.canvas.ax11.add_patch(rect)
    
            self.update_estimated_time()
        
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
            milling_protocol_stages = milling.get_milling_protocol_stages(settings=self.settings, stage_name="thin_lamella")

        if self.milling_pattern_type == MillingPattern.Polish:
            milling_protocol_stages = self.settings["polish_lamella"]

        if self.milling_pattern_type == MillingPattern.Flatten:
            milling_protocol_stages = self.settings["flatten_landing"]

        if self.milling_pattern_type == MillingPattern.Fiducial:
            milling_protocol_stages = self.settings["fiducial"]

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

        if "milling_current" in self.milling_settings:
            self.milling_current = float(self.milling_settings["milling_current"])
        else: 
            self.milling_current = float(self.settings["calibration"]["imaging"]["milling_current"]) 
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

    def update_estimated_time(self):
        # volume (width * height * depth) / total_volume_sputter_rate

        # sputtering rates
        self.sputter_rate_dict = {
            20e-12: 6.85e-3,    # 30kv
            0.2e-9: 6.578e-2,   # 30kv
            0.74e-9: 3.349e-1,  # 30kv
            0.89e-9: 3.920e-1,  # 20kv
            2.0e-9: 9.549e-1,   # 30kv
            2.4e-9: 1.309,      # 20kv
            6.2e-9: 2.907,      # 20kv
            7.6e-9: 3.041       # 30kv
        }
        # 0.89nA : 3.920e-1 um3/s
        # 2.4nA : 1.309e0 um3/s
        # 6.2nA : 2.907e0 um3/s # from microscope application files

        # 30kV
        # 7.6nA: 3.041e0 um3/s

        if self.milling_current in self.sputter_rate_dict:
            total_volume_sputter_rate = self.sputter_rate_dict[self.milling_current]
        else:
            total_volume_sputter_rate = 3.920e-1

        volume = 0
        for pattern in self.patterns:
            width = pattern.width * METRE_TO_MICRON
            height = pattern.height * METRE_TO_MICRON
            depth = pattern.depth * METRE_TO_MICRON
            volume += width * height * depth
        
        self.estimated_milling_time_s = volume / total_volume_sputter_rate # um3 * 1/ (um3 / s) = seconds

        logging.info(f"WHDV: {width:.2f}um, {height:.2f}um, {depth:.2f}um, {volume:.2f}um3")
        logging.info(f"Milling Volume Sputter Rate: {total_volume_sputter_rate} um3/s")
        logging.info(f"Milling Estimated Time: {self.estimated_milling_time_s / 60:.2f}m")

        # update labels
        self.label__milling_current.setText(f"Milling Current: {self.milling_current:.2e}A")
        self.label_estimated_time.setText(f"Estimated Time: {self.estimated_milling_time_s / 60:.2f} minutes") # TODO: formulaa

        # set progress bar
        self.progressBar.setValue(0)
        # self.progressBar.setStyleSheet(
        #     """
        #     border: 2px solid #2196F3;
        #     border-radius: 5px;
        #     background-color: #E0E0E0;
        #     """
        # )

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
            self.patterns = milling.mill_trench_patterns(microscope=self.microscope,
                                    settings=self.milling_settings,
                                    centre_x=self.center_x, centre_y=self.center_y)


        if self.milling_pattern_type == MillingPattern.Polish:
            self.patterns = milling.mill_trench_patterns(microscope=self.microscope,
                                                settings=self.milling_settings,
                                                centre_x=self.center_x, centre_y=self.center_y)

        if self.milling_pattern_type == MillingPattern.Flatten:
            self.patterns = milling.flatten_landing_pattern(microscope=self.microscope, settings=self.settings,
                                                            centre_x=self.center_x, centre_y=self.center_y)

        if self.milling_pattern_type == MillingPattern.Fiducial:
            self.patterns = milling.fiducial_marker_patterns(microscope=self.microscope, settings=self.settings,
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

        def draw_rectangle_pattern(adorned_image, rectangle, pattern):
            image_width = adorned_image.width
            image_height = adorned_image.height
            pixel_size = adorned_image.metadata.binary_result.pixel_size.x

            width = pattern.width / pixel_size
            height = pattern.height / pixel_size
            rotation = -pattern.rotation
            rectangle_left = (image_width / 2) + (pattern.center_x / pixel_size) - (width / 2) * np.cos(rotation) + (height / 2) * np.sin(rotation)
            rectangle_bottom = (image_height / 2) - (pattern.center_y / pixel_size) - (height / 2) * np.cos(rotation) - (width / 2) * np.sin(rotation)
            rectangle.set_width(width)
            rectangle.set_height(height)
            rectangle.set_xy((rectangle_left, rectangle_bottom))
            rectangle.set_visible(True)

        # create a rectangle for each pattern
        self.pattern_rectangles = []
        try:
            for pattern in self.patterns:
                rectangle = Rectangle((0, 0), 0.2, 0.2, color='yellow', fill=None, alpha=1, angle=np.rad2deg(-pattern.rotation))
                rectangle.set_visible(False)
                rectangle.set_hatch('//////')
                self.pattern_rectangles.append(rectangle)
                draw_rectangle_pattern(adorned_image=self.adorned_image,
                                       rectangle=rectangle, pattern=pattern)
        except Exception as e:
            # NOTE: these exceptions happen when the pattern is too far outside of the FOV
            logging.error(f"Pattern outside FOV: {e}") 
    
    def exit_milling_button_pressed(self):
        """Exit the Milling Window"""
        self.close()

    def run_milling_button_pressed(self):
        """Run ion beam milling for the selected milling pattern"""

        logging.info(f"Running milling for {len(self.milling_stages)} Stages")

        if self.milling_pattern_type == MillingPattern.Polish:
            milling.mill_polish_lamella(
                        microscope=self.microscope, 
                        settings=self.settings, 
                        image_settings=self.image_settings, 
                        patterns=self.patterns, 
                        ) # TODO: move this into the standard flow if titling is not required
        else:
            for stage_name, milling_settings in self.milling_stages.items():
                
                logging.info(f"Stage {stage_name}: {milling_settings}")

                self.milling_settings = milling_settings 
                self.update_milling_patterns()
                self.update_display()
                
                # run_milling
                milling.run_milling(microscope=self.microscope, settings=self.settings, milling_current=self.milling_current, asynch=True)
                
                time.sleep(5) # wait for milling to start
                elapsed_time = 0
                while self.microscope.patterning.state == "Running":                  
                    
                    elapsed_time+=1
                    prog_val = elapsed_time / self.estimated_milling_time_s * 100
                    self.progressBar.setValue(prog_val)
                    time.sleep(1)
                
                logging.info(f"Milling finished: {self.microscope.patterning.state}")


        # reset to imaging mode
        milling.finish_milling(microscope=self.microscope, settings=self.settings)

        # refresh image
        self.image_settings.save = False
        self.setup_milling_image()
        self.update_display(draw_patterns=False)
        
        # ask user if the milling succeeded
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Milling Confirmation")
        dlg.setText("Do you need to redo milling?")
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dlg.setIcon(QMessageBox.Question)
        button = dlg.exec()

        if button == QMessageBox.Yes:
            logging.info("Redoing milling")
            self.update_display()
        else:
            self.close() 

    def closeEvent(self, event):
        logging.info("Closing Milling Window")
        self.microscope.patterning.clear_patterns()
        event.accept()


def main():

    microscope, settings, image_settings = fibsem_utils.quick_setup()

    import os
    os.makedirs(image_settings.save_path, exist_ok=True)

    # image_settings.hfw = settings["thin_lamella"]["hfw"]
    # image_settings.resolution = settings["thin_lamella"]["resolution"]
    # image_settings.dwell_time = settings["thin_lamella"]["dwell_time"]

    image_settings.hfw = 80.e-6
    from liftout.fibsem import calibration
    calibration.reset_beam_shifts(microscope)       

    app = QtWidgets.QApplication([])
    qt_app = GUIMillingWindow(microscope=microscope, 
                                settings=settings, 
                                image_settings=image_settings, 
                                milling_pattern_type=MillingPattern.JCut)
    qt_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
