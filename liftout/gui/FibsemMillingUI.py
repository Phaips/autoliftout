import logging
import time
from pprint import pprint

import liftout.gui.utils as ui_utils
import napari
import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import acquire, constants, conversions, milling, movement
from fibsem import utils as fibsem_utils
from fibsem.structures import BeamType, MicroscopeSettings, MillingSettings, Point
from fibsem.ui import utils as fibsem_ui
from liftout import patterning
from liftout.config import config
from liftout.gui.qtdesigner_files import MillingUI
from liftout.patterning import MillingPattern
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel

class FibsemMillingUI(MillingUI.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        viewer: napari.Viewer,
        microscope: SdbMicroscopeClient,
        settings: MicroscopeSettings,
        milling_pattern: MillingPattern = MillingPattern.Trench,
        point: Point = None,
        change_pattern: bool = False,
        auto_continue: bool = False,
    ):
        super(FibsemMillingUI, self).__init__()

        # setup ui
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings

        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.milling_pattern = milling_pattern
        self.point = point
        self.auto_continue = auto_continue
        self.USER_UPDATED_PROTOCOL = False 
        self.CHANGE_PATTERN_ENABLED = change_pattern

        self.setup_ui()
        self.setup_connections()

        milling.setup_milling(
            microscope = self.microscope,
            application_file = self.settings.system.application_file,
            hfw=self.settings.image.hfw
        )

        self.update_milling_stages()

    def setup_connections(self):

        # combobox
        self.comboBox_select_pattern.addItems(
            pattern.name for pattern in MillingPattern
        )
        self.comboBox_select_pattern.setCurrentText(self.milling_pattern.name)
        self.comboBox_select_pattern.currentTextChanged.connect(
            self.update_milling_stages
        )

        self.comboBox_milling_current.addItems(
            [
                f"{current*constants.SI_TO_NANO:.2f}"
                for current in self.microscope.beams.ion_beam.beam_current.available_values
            ]
        )

        # buttons
        self.pushButton_run_milling.clicked.connect(self.run_milling)
        self.pushButton_exit_milling.clicked.connect(self.exit_milling)

        # instructions
        self.label_message.setText(
            f"Double-click to move the pattern, adjust parameters to change pattern dimensions. Press Run Milling to start milling."
        )
        self.comboBox_select_pattern.setEnabled(self.CHANGE_PATTERN_ENABLED)

    def setup_ui(self):

        # take image
        self.settings.image.beam_type = BeamType.ION
        self.image = acquire.new_image(self.microscope, self.settings.image)

        # draw image
        self.viewer.layers.clear()
        self.image_layer = self.viewer.add_image(
            ndi.median_filter(self.image.data, 3), name="Image"
        )
        self.image_layer.mouse_double_click_callbacks.append(
            self._double_click
        )  # append callback
        self.viewer.layers.selection.active = self.image_layer

    def _double_click(self, layer, event):

        coords = layer.world_to_data(event.position)

        image_shape = self.image.data.shape

        if (coords[0] > 0 and coords[0] < image_shape[0]) and (
            coords[1] > 0 and coords[1] < image_shape[1]
        ):
            logging.info(
                f"click inside image: {coords[0]:.2f}, {coords[1]:.2f}"
            )

        else:
            napari.utils.notifications.show_info(
                f"Clicked outside image dimensions. Please click inside the image."
            )
            return

        # for trenches, move the stage, not the pattern
        # (trench should always be in centre of image)
        if self.milling_pattern is MillingPattern.Trench:

            point = conversions.image_to_microscope_image_coordinates(Point(x=coords[1], y=coords[0]), 
                    self.image.data, self.image.metadata.binary_result.pixel_size.x)  

            movement.move_stage_relative_with_corrected_movement(
                microscope=self.microscope,
                settings=self.settings,
                dx=point.x,
                dy=point.y,
                beam_type=BeamType.ION,
            )

            # update image
            self.setup_ui()
            coords = None

        # get image coordinate
        if coords is None:
            coords = np.asarray(self.image.data.shape) // 2

        self.point = conversions.image_to_microscope_image_coordinates(Point(x=coords[1], y=coords[0]), 
            self.image.data, self.image.metadata.binary_result.pixel_size.x)  

        logging.info(
            f"Milling, {BeamType.ION}, {self.milling_pattern.name}, p=({coords[1]:.2f}, {coords[0]:.2f})  c=({self.point.x:.2e}, {self.point.y:.2e}) "
        )

        self.update_milling_pattern()

    def update_milling_pattern(self):
        
        try:
            # draw patterns in microscope
            self.microscope.patterning.clear_patterns()
            all_patterns = []
            for stage_name, stage_settings in self.milling_stages.items():

                patterns = patterning.create_milling_patterns(
                    self.microscope,
                    stage_settings,
                    self.milling_pattern,
                    self.point,
                )
                all_patterns.append(patterns)  # 2D

            # draw patterns in napari

            if "Stage 1" in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers["Stage 1"])
            if "Stage 2" in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers["Stage 2"])

            pixelsize = self.image.metadata.binary_result.pixel_size.x

            for i, stage in enumerate(all_patterns, 1):
                shape_patterns = []
                for pattern in stage:
                    shape = convert_pattern_to_napari_rect(
                        pattern, self.image.data, pixelsize
                    )
                    shape_patterns.append(shape)

                # draw shapes
                colour = "yellow" if i == 1 else "cyan"
                self.viewer.add_shapes(
                    shape_patterns,
                    name=f"Stage {i}",
                    shape_type="rectangle",
                    edge_width=0.5,
                    edge_color=colour,
                    face_color=colour,
                    opacity=0.5,
                )

            self.update_estimated_time(all_patterns)

            self.viewer.layers.selection.active = self.image_layer
        
        except Exception as e:
            napari.utils.notifications.show_info(
                f"Error: {e}"
            )

    def update_estimated_time(self, patterns: list):

        milling_time_seconds = milling.estimate_milling_time_in_seconds(patterns)
        time_str = fibsem_utils._format_time_seconds(milling_time_seconds)
        self.label_milling_time.setText(f"Estimated Time: {time_str}")

    def update_milling_stages(self):

        logging.info("update_milling_stages")

        self.milling_pattern = MillingPattern[
            self.comboBox_select_pattern.currentText()
        ]

        # get all milling stages for the pattern
        milling_protocol_stages = patterning.get_milling_protocol_stage_settings(
            self.settings, self.milling_pattern
        )

        self.milling_stages = {}
        for i, stage_settings in enumerate(milling_protocol_stages, 1):
            self.milling_stages[f"{self.milling_pattern.name}_{i}"] = stage_settings

        try:
            self.comboBox_milling_stage.disconnect()
        except:
            pass
        self.comboBox_milling_stage.clear()
        self.comboBox_milling_stage.addItems(list(self.milling_stages.keys()))
        self.comboBox_milling_stage.currentTextChanged.connect(
            self.setup_milling_parameters_ui
        )

        # setup parameter ui
        self.setup_milling_parameters_ui()

        # draw milling patterns
        self.update_milling_pattern()

    def setup_milling_parameters_ui(self):

        # remove existing elements
        for i in reversed(range(self.gridLayout_2.count())):
            self.gridLayout_2.itemAt(i).widget().setParent(None)

        i = 0
        current_stage = self.comboBox_milling_stage.currentText()

        self.milling_ui_dict = {}

        for (k, v) in self.milling_stages[current_stage].items():

            if k not in config.NON_CHANGEABLE_MILLING_PARAMETERS:
                if k not in config.NON_SCALED_MILLING_PARAMETERS:
                    v = float(v) * constants.METRE_TO_MICRON

                label = QLabel()
                label.setText(str(k))
                spinBox_value = QtWidgets.QDoubleSpinBox()
                spinBox_value.setValue(v)
                spinBox_value.valueChanged.connect(self.update_milling_settings_from_ui)
                
                if k == "rotation":
                    spinBox_value.setRange(-360, 360)

                self.gridLayout_2.addWidget(label, i, 0)
                self.gridLayout_2.addWidget(spinBox_value, i, 1)

                self.milling_ui_dict[k] = spinBox_value

                i += 1

        milling_current = (
            self.milling_stages[current_stage]["milling_current"] * constants.SI_TO_NANO
        )
        self.comboBox_milling_current.setCurrentText(f"{milling_current:.2f}")
        try:
            self.comboBox_milling_current.disconnect()
        except:
            pass
        self.comboBox_milling_current.currentTextChanged.connect(
            self.update_milling_settings_from_ui
        )

    def update_milling_settings_from_ui(self):

        # flag that user has changed protocol
        self.USER_UPDATED_PROTOCOL = True

        # map keys to ui widgets
        current_stage = self.comboBox_milling_stage.currentText()
        for k, v in self.milling_ui_dict.items():
            value = v.value()
            if k not in config.NON_SCALED_MILLING_PARAMETERS:
                value = float(value) * constants.MICRON_TO_METRE
            self.milling_stages[current_stage][k] = value

        self.milling_stages[current_stage]["milling_current"] = (
            float(self.comboBox_milling_current.currentText()) * constants.NANO_TO_SI
        )

        self.update_milling_pattern()

    def run_milling(self):
        """Run ion beam milling for the selected milling pattern"""

        logging.info(f"Running milling for {len(self.milling_stages)} Stages")

        # clear state
        self.microscope.imaging.set_active_view(BeamType.ION.value)
        self.microscope.imaging.set_active_device(
            BeamType.ION.value
        )  # set ion beam view
        for stage_name, stage_settings in self.milling_stages.items():

            # redraw patterns, and run milling
            self.microscope.patterning.clear_patterns()
            self.patterns = patterning.create_milling_patterns(
                self.microscope,
                stage_settings,
                self.milling_pattern,
                self.point,
            )
            milling.run_milling(
                microscope=self.microscope,
                milling_current=stage_settings["milling_current"],
                asynch=True,
            )

            # update progress bar
            time.sleep(3)  # wait for milling to start
            elapsed_time = 0

            milling_time_seconds = milling.estimate_milling_time_in_seconds(
                [self.patterns]
            )
            logging.info(f"milling time: {milling_time_seconds}")
            progressbar = napari.utils.progress(milling_time_seconds)

            # TODO: thread https://forum.image.sc/t/napari-progress-bar-modification-on-the-fly/62496/7
            while self.microscope.patterning.state == "Running":

                elapsed_time += 1
                prog_val = int(elapsed_time)
                progressbar.update(prog_val)
                time.sleep(1)

            logging.info(f"Milling finished: {self.microscope.patterning.state}")
            progressbar.clear()
            progressbar.close()

        # reset to imaging mode
        milling.finish_milling(
            microscope=self.microscope,
            imaging_current=self.settings.default.imaging_current,
        )
        napari.utils.notifications.show_info(f"Milling Complete.")

        # update image
        self.setup_ui()

        # confirm finish
        self.finalise_milling()

    def finalise_milling(self) -> bool:

        if self.auto_continue:
            self.close()
            return

        # ask user if the milling succeeded
        response = fibsem_ui.message_box_ui(
            title="Milling Confirmation", text="Do you want to redo milling?"
        )

        if response:
            logging.info("Redoing milling")
            
            # re-update patterns
            self.update_milling_pattern()
            return response

        # only update if the protocol has been changed...
        if self.USER_UPDATED_PROTOCOL:
            response = fibsem_ui.message_box_ui(
                title="Save Milling Protocol?",
                text="Do you want to update the protocol to use these milling parameters?",
            )

            if response:
                try:
                    ui_utils.update_milling_protocol_ui(
                        self.milling_pattern, self.milling_stages, self
                    )
                except Exception as e:
                    logging.error(f"Unable to update protocol file: {e}")
        self.close()

    def exit_milling(self):
        self.close()

    def closeEvent(self, event):
        self.microscope.patterning.clear_patterns()
        event.accept()
        self.viewer.window.close()


def convert_pattern_to_napari_rect(
    pattern, image: np.ndarray, pixelsize: float
) -> np.ndarray:

    # image centre
    icy, icx = image.shape[0] // 2, image.shape[1] // 2

    # pattern to pixel coords
    w = int(pattern.width / pixelsize)
    h = int(pattern.height / pixelsize)
    cx = int(icx + (pattern.center_x / pixelsize))
    cy = int(icy - (pattern.center_y / pixelsize))

    r = -pattern.rotation #

    xmin, xmax = - w / 2, w / 2
    ymin, ymax = - h / 2, h / 2

    px0 = cx + (xmin * np.cos(r) - ymin * np.sin(r))
    py0 = cy + (xmin * np.sin(r) + ymin * np.cos(r))

    px1 = cx + (xmax * np.cos(r) - ymin * np.sin(r))
    py1 = cy + (xmax * np.sin(r) + ymin * np.cos(r))

    px2 = cx + (xmax * np.cos(r) - ymax * np.sin(r))
    py2 = cy + (xmax * np.sin(r) + ymax * np.cos(r))

    px3 = cx + (xmin * np.cos(r) - ymax * np.sin(r))
    py3 = cy + (xmin * np.sin(r) + ymax * np.cos(r))

    # napari shape format
    shape = [[py0, px0], [py1, px1], [py2, px2], [py3, px3]]

    return shape


def convert_napari_rect_to_mill_settings(
    arr: np.array, image: np.array, pixelsize: float, depth: float = 10e-6
) -> MillingSettings:
    # convert napari rect to milling pattern

    # get centre of image
    cy_mid, cx_mid = image.data.shape[0] // 2, image.shape[1] // 2

    # TODO: account for rotation, different shape types

    # get rect dimensions in px
    ymin, xmin = arr[0]
    ymax, xmax = arr[2]

    width = int(xmax - xmin)
    height = int(ymax - ymin)

    cx = int(xmin + width / 2)
    cy = int(ymin + height / 2)

    # get rect dimensions in real space
    cy_real = (cy_mid - cy) * pixelsize
    cx_real = -(cx_mid - cx) * pixelsize
    width = width * pixelsize
    height = height * pixelsize

    # set milling settings
    mill_settings = MillingSettings(
        width=width, height=height, depth=depth, centre_x=cx_real, centre_y=cy_real
    )

    return mill_settings


def main():

    microscope, settings = fibsem_utils.setup_session()
    settings = fibsem_utils.load_settings_from_config(
        config_path=config.config_path, protocol_path=config.protocol_path,
    )
    milling_pattern = MillingPattern.Cut
    point = None
    change_pattern = True
    auto_continue = False

    settings.image.hfw = 80e-6


    from liftout.autoliftout import milling_ui

    milling_ui(microscope, settings, milling_pattern, point = point, change_pattern = change_pattern, auto_continue=auto_continue)



if __name__ == "__main__":
    main()
