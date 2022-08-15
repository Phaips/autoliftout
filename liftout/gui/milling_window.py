import datetime
import logging
import sys
import time

import liftout.gui.utils as ui_utils
import matplotlib.patches as mpatches
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import \
    Rectangle as RectangleArea
from fibsem import acquire, constants, calibration, milling
from fibsem.structures import BeamType, ImageSettings, Point
from liftout import patterning, utils
from liftout.config import config
from liftout.gui.qtdesigner_files import milling_dialog as milling_gui
from PyQt5 import QtCore, QtWidgets


class GUIMillingWindow(milling_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        microscope: SdbMicroscopeClient,
        settings: dict,
        image_settings: ImageSettings,
        milling_pattern_type: patterning.MillingPattern,
        x: float = 0.0,
        y: float = 0.0,
        parent=None,
    ):
        super(GUIMillingWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.microscope = microscope
        self.settings = settings
        self.image_settings = image_settings
        self.milling_pattern = milling_pattern_type

        self.wp = None  # plotting widget
        self.USER_UPDATE = True

        self.available_milling_currents = (
            self.microscope.beams.ion_beam.beam_current.available_values
        )
        self.comboBox_milling_current.addItems(
            [f"{current:.2e}" for current in self.available_milling_currents]
        )

        # milling parameters
        self.parameter_labels = [
            self.label_01,
            self.label_02,
            self.label_03,
            self.label_04,
            self.label_05,
            self.label_06,
            self.label_07,
            self.label_08,
            self.label_09,
            self.label_10,
            self.label_11,
            self.label_12,
        ]

        self.parameter_values = [
            self.doubleSpinBox_01,
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
            self.doubleSpinBox_12,
        ]

        # setup ui
        for label, spinBox in zip(self.parameter_labels, self.parameter_values):
            label.setVisible(False)
            spinBox.setVisible(False)

        self.INITIALISED = False

        # update milling pattern
        self.milling_pattern = milling_pattern_type

        self.milling_stages = {}
        self.setup_milling_image()

        # setup
        milling.setup_milling(
            microscope=self.microscope,
            application_file=self.settings["system"]["application_file"],
            hfw=self.image_settings.hfw,
        )
        self.setup_milling_patterns()
        self.setup_connections()

        # initial pattern
        self.center_x, self.center_y = x, y
        self.xclick, self.yclick = None, None
        self.update_display()

        self.INITIALISED = True

        AUTO_CONTINUE = False
        if AUTO_CONTINUE:
            self.run_milling_button_pressed() # automatically continue

    def setup_milling_image(self):

        # image with a reduced area for thin/polishing.
        if self.milling_pattern in [
            patterning.MillingPattern.Thin,
            patterning.MillingPattern.Polish,
        ]:
            reduced_area = RectangleArea(0.3, 0.3, 0.4, 0.4)
        else:
            reduced_area = None

        self.image_settings.beam_type = BeamType.ION
        self.adorned_image = acquire.new_image(
            self.microscope, self.image_settings, reduced_area=reduced_area
        )
        self.image = ndi.median_filter(self.adorned_image.data, size=3)

        # pattern drawing
        if self.wp is not None:
            self.label_image.layout().removeWidget(self.wp)
            self.wp.deleteLater()

        # pattern drawing
        self.wp = ui_utils._WidgetPlot(self, display_image=self.image)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)
        self.wp.canvas.mpl_connect("button_press_event", self.on_click)

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
        self.comboBox_pattern_stage.currentTextChanged.connect(
            self.update_stage_settings
        )

        # update milling currents
        self.comboBox_milling_current.currentTextChanged.connect(
            self.update_milling_current
        )

    def update_milling_current(self):

        # update the milling current
        milling_current = float(self.comboBox_milling_current.currentText())
        self.milling_stages[self.current_selected_stage][
            "milling_current"
        ] = milling_current

        print(f"updating milling current to: {milling_current:.2e}")
        self.update_estimated_time()

    def update_stage_settings(self):
        """Update the current milling stage and settings"""

        self.current_selected_stage = self.comboBox_pattern_stage.currentText()

        try:
            logging.info(
                f"Stage: {self.current_selected_stage} - Milling Settings: {self.milling_stages[self.current_selected_stage]}"
            )

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
                if param not in config.NON_SCALED_MILLING_PARAMETERS:
                    param_value = param_value * constants.MICRON_TO_METRE
                self.milling_stages[self.current_selected_stage][param] = param_value

        if self.USER_UPDATE:
            self.update_display()

    def update_display(self, draw_patterns=True):
        """Update the millig window display. Redraw the crosshair, and milling patterns"""

        self.wp.canvas.ax11.patches.clear()

        # draw crosshar
        ui_utils.draw_crosshair(
            self.image, self.wp.canvas, x=self.xclick, y=self.yclick
        )

        if draw_patterns:
            try:
                self.draw_milling_patterns()
                self.update_estimated_time()
            except Exception as e:
                logging.error(f"Error during display update: {e}")

        # reset progress bar
        self.progressBar.setValue(0)

        self.wp.canvas.draw()

    def on_click(self, event):
        """Redraw the patterns and update the display on user click"""
        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata
            self.center_x, self.center_y = calibration.pixel_to_realspace_coordinate(
                (self.xclick, self.yclick), self.adorned_image
            )

            self.update_display()

    def setup_milling_patterns(self):
        """Load the milling stages and settings for the selected milling pattern"""

        milling_protocol_stages = patterning.get_milling_protocol_stage_settings(
            self.settings, self.milling_pattern
        )

        for i, stage_settings in enumerate(milling_protocol_stages, 1):
            self.milling_stages[f"{self.milling_pattern.name}_{i}"] = stage_settings

        # set the first milling stage settings
        self.current_selected_stage = list(self.milling_stages.keys())[0]

        # set the milling current
        milling_current = float(
            self.milling_stages[self.current_selected_stage]["milling_current"]
        )
        closest_current = min(
            self.available_milling_currents, key=lambda x: abs(x - milling_current)
        )  # https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
        self.comboBox_milling_current.setCurrentText(f"{closest_current:.2e}")

        self.update_parameter_elements()

    def update_parameter_elements(self):
        """Update the parameter labels and values with updated milling settings"""

        # update ui elements
        self.USER_UPDATE = False
        i = 0
        for key, value in self.milling_stages[self.current_selected_stage].items():
            if key not in config.NON_CHANGEABLE_MILLING_PARAMETERS:
                if key not in config.NON_SCALED_MILLING_PARAMETERS:
                    value = float(value) * constants.METRE_TO_MICRON
                self.parameter_labels[i].setText(key)
                self.parameter_values[i].setValue(value)
                self.parameter_labels[i].setVisible(True)
                self.parameter_values[i].setVisible(True)
                i += 1
            else:
                self.parameter_labels[i].setVisible(False)
                self.parameter_values[i].setVisible(False)

    def update_estimated_time(self):

        # update estimted milling time

        # TODO: change over:
        total_time_seconds = milling.estimate_milling_time_in_seconds(self.patterns) # only works for current milling current.

        self.milling_time_seconds = milling.calculate_milling_time(
            self.patterns,
            self.milling_stages[self.current_selected_stage]["milling_current"],
        )
        time_str = str(datetime.timedelta(seconds=self.milling_time_seconds)).split(
            "."
        )[0]
        self.label_estimated_time.setText(f"Estimated Time: {time_str}")
        self.USER_UPDATE = True

    def draw_milling_patterns(self):

        self.microscope.imaging.set_active_view(2)  # the ion beam view
        self.microscope.patterning.clear_patterns()
        try:
            self.patterns = []
            for stage_name, stage_settings in self.milling_stages.items():

                patterns = patterning.create_milling_patterns(
                    self.microscope,
                    stage_settings,
                    self.milling_pattern,
                    Point(self.center_x, self.center_y),
                )
                self.patterns.append(patterns)  # 2D

        except Exception as e:
            print(e)
            logging.error(f"Error during milling pattern update: {e}")

        # create a rectangle for each pattern
        self.pattern_rectangles = []
        try:
            for i, stage in enumerate(self.patterns):
                for pattern in stage:
                    colour = "cyan" if i == 1 else "yellow"
                    rectangle = ui_utils.draw_rectangle_pattern(
                        adorned_image=self.adorned_image, pattern=pattern, colour=colour
                    )
                    self.pattern_rectangles.append(rectangle)
        except Exception as e:
            # NOTE: these exceptions happen when the pattern is too far outside of the FOV
            logging.error(f"Pattern outside FOV: {e}")

        for rect in self.pattern_rectangles:
            self.wp.canvas.ax11.add_patch(rect)

            # legend
            yellow_patch = mpatches.Patch(
                color="yellow", hatch="//////", label="Stage 1"
            )
            cyan_patch = mpatches.Patch(color="cyan", hatch="//////", label="Stage 2")
            self.wp.canvas.ax11.legend(handles=[yellow_patch, cyan_patch])

    def run_milling_button_pressed(self):
        """Run ion beam milling for the selected milling pattern"""

        logging.info(f"Running milling for {len(self.milling_stages)} Stages")

        # clear state
        self.microscope.imaging.set_active_view(2)  # the ion beam view
        for stage_name, stage_settings in self.milling_stages.items():

            logging.info(f"Stage {stage_name}: {stage_settings}")

            # redraw patterns, and run milling
            self.microscope.patterning.clear_patterns()
            self.patterns = patterning.create_milling_patterns(
                self.microscope,
                stage_settings,
                self.milling_pattern,
                Point(self.center_x, self.center_y),
            )
            milling.run_milling(
                microscope=self.microscope,
                settings=self.settings,
                milling_current=stage_settings["milling_current"],
                asynch=True,
            )

            # update progress bar
            time.sleep(3)  # wait for milling to start
            elapsed_time = 0
            while self.microscope.patterning.state == "Running":

                elapsed_time += 1
                prog_val = int(elapsed_time / self.milling_time_seconds * 100)
                self.progressBar.setValue(prog_val)
                time.sleep(1)
            logging.info(f"Milling finished: {self.microscope.patterning.state}")

        # reset to imaging mode
        milling.finish_milling(
            microscope=self.microscope,
            imaging_current=self.settings["calibration"]["imaging"]["imaging_current"],
        )

        # refresh image
        self.image_settings.save = False
        self.setup_milling_image()
        self.update_display(draw_patterns=False)

        # finish milling
        self.finalise_milling()

    def finalise_milling(self):
        # ask user if the milling succeeded
        response = ui_utils.message_box_ui(
            title="Milling Confirmation", text="Do you want to redo milling?"
        )

        if response:
            logging.info("Redoing milling")
            self.update_display(draw_patterns=False)
        else:
            response = ui_utils.message_box_ui(
                title="Save Milling Protocol?",
                text="Do you want to save this milling protocol?",
            )

            if response:
                try:
                    ui_utils.update_milling_protocol_ui(
                        self.milling_pattern, self.milling_stages, self
                    )
                except Exception as e:
                    logging.error(f"Unable to update protocol file: {e}")
            self.close()

    def exit_milling_button_pressed(self):
        """Exit the Milling Window"""
        self.close()

    def closeEvent(self, event):
        logging.info("Closing Milling Window")
        self.microscope.patterning.clear_patterns()
        event.accept()


def main():

    microscope, settings, image_settings = utils.quick_setup()

    import os

    image_settings.save_path = "tools/test"
    image_settings.hfw = 80.0e-6

    os.makedirs(image_settings.save_path, exist_ok=True)
    # acquire.reset_beam_shifts(microscope)

    from liftout.patterning import MillingPattern
    from liftout.gui import windows

    app = QtWidgets.QApplication([])
    windows.open_milling_window(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern=MillingPattern.Trench,
        x=0, y=0,
    )
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
