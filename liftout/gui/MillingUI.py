import logging
import sys
from pprint import pprint

import napari
import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import AdornedImage
from torch import autocast
from fibsem import calibration, constants, acquire, conversions, milling
from fibsem import utils as fibsem_utils
from fibsem.structures import MicroscopeSettings, BeamType, Point, MillingSettings
from liftout import autoliftout, patterning, utils
from liftout.gui.qtdesigner_files import MillingUI
from liftout.patterning import MillingPattern
from napari.utils import notifications, progress
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel
from fibsem.ui import utils as fibsem_ui
import liftout.gui.utils as ui_utils
import scipy.ndimage as ndi


from liftout.config import config
import time

# TODO: connect parameter changes

class MillingUI(MillingUI.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        viewer: napari.Viewer,
        microscope: SdbMicroscopeClient,
        settings: MicroscopeSettings,
        milling_pattern: MillingPattern = MillingPattern.Trench,
        point: Point = None,
        auto_continue: bool = False
    ):
        super(MillingUI, self).__init__()

        # setup ui
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings

        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.milling_pattern = milling_pattern
        self.auto_continue = auto_continue
        self.USER_UPDATED_PROTOCOL = False # TODO: implement this

        self.setup_ui()
        self.setup_connections()

        self.update_milling_stages()


    def setup_connections(self):

        logging.info("setup connections")

        # combobox
        self.comboBox_select_pattern.addItems(
            pattern.name for pattern in MillingPattern
        )
        self.comboBox_select_pattern.setCurrentText(self.milling_pattern.name)
        self.comboBox_select_pattern.currentTextChanged.connect(
            self.update_milling_stages
        )

        self.comboBox_milling_current.addItems(
            [f"{current*constants.SI_TO_NANO:.2f}" 
            for current in self.microscope.beams.ion_beam.beam_current.available_values]
        )

        # buttons
        self.pushButton_run_milling.clicked.connect(self.run_milling)
        self.pushButton_exit_milling.clicked.connect(self.exit_milling)

    def setup_ui(self):
        
        # take image
        self.settings.image.beam_type = BeamType.ION
        self.image = acquire.new_image(self.microscope, self.settings.image)

        # draw image
        self.viewer.layers.clear()
        self.image_layer = self.viewer.add_image(ndi.median_filter(self.image.data, 3), name="Image")
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
            logging.info(f"click inside image: {coords[0]:.2f}, {coords[1]:.2f}")
        else:
            napari.utils.notifications.show_info(
                f"Clicked outside image dimensions. Please click inside the image."
            )
            return

        self.update_milling_pattern(coords)

    def update_milling_pattern(self, coords: tuple = None):

        logging.info(f"update milling pattern")

        # get image coordinate
        if coords is None:
            coords = np.asarray(self.image.data.shape) // 2
        pixelsize = self.image.metadata.binary_result.pixel_size.x
        self.center_x, self.center_y = conversions.pixel_to_realspace_coordinate(
                (coords[1], coords[0]), self.image
            )

        # draw patterns in microscope
        self.microscope.patterning.clear_patterns()
        all_patterns = []
        for stage_name, stage_settings in self.milling_stages.items():

            patterns = patterning.create_milling_patterns(
                self.microscope,
                stage_settings,
                self.milling_pattern,
                Point(self.center_x, self.center_y),
            )
            all_patterns.append(patterns)  # 2D

        # draw patterns in napari

        # TODO: clear all shape layers...add a legend to the view?
        if "Stage 1" in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers["Stage 1"])
        if "Stage 2" in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers["Stage 2"])

        for i, stage in enumerate(all_patterns, 1):
            shape_patterns = []
            for pattern in stage:
                shape = convert_pattern_to_napari_rect(pattern, self.image.data, pixelsize)
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
                opacity=0.5
            )

        self.viewer.layers.selection.active = self.image_layer

    def update_milling_stages(self):

        logging.info("update_milling_stages")

        self.milling_pattern = MillingPattern[self.comboBox_select_pattern.currentText()]

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

    # TODO: support for other element types
    # TODO: pattern rotation
    # TODO: milling time

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

                self.gridLayout_2.addWidget(label, i, 0)
                self.gridLayout_2.addWidget(spinBox_value, i, 1)

                self.milling_ui_dict[k] = spinBox_value

                i += 1

        milling_current = self.milling_stages[current_stage]["milling_current"] * constants.SI_TO_NANO
        self.comboBox_milling_current.setCurrentText(f'{milling_current:.2f}')
        try:
            self.comboBox_milling_current.disconnect()
        except:
            pass
        self.comboBox_milling_current.currentTextChanged.connect(self.update_milling_settings_from_ui)

    def update_milling_settings_from_ui(self):
        # map keys to ui widgets
        current_stage = self.comboBox_milling_stage.currentText()
        for k, v in self.milling_ui_dict.items():
            value = v.value()
            if k not in config.NON_SCALED_MILLING_PARAMETERS:
                value = float(value) * constants.MICRON_TO_METRE
            self.milling_stages[current_stage][k] = value

        self.milling_stages[current_stage]["milling_current"] = float(self.comboBox_milling_current.currentText()) * constants.NANO_TO_SI
        
        self.update_milling_pattern()

    def run_milling(self):
        """Run ion beam milling for the selected milling pattern"""

        logging.info(f"Running milling for {len(self.milling_stages)} Stages")

        # clear state
        self.microscope.imaging.set_active_view(BeamType.ION.value)
        self.microscope.imaging.set_active_device(BeamType.ION.value)  # set ion beam view
        # for stage_name, stage_settings in self.milling_stages.items():

        #     logging.info(f"Stage {stage_name}: {stage_settings}")

        #     # redraw patterns, and run milling
        #     self.microscope.patterning.clear_patterns()
        #     self.patterns = patterning.create_milling_patterns(
        #         self.microscope,
        #         stage_settings,
        #         self.milling_pattern,
        #         Point(self.center_x, self.center_y),
        #     )
        #     milling.run_milling(
        #         microscope=self.microscope,
        #         milling_current=stage_settings["milling_current"],
        #         asynch=True,
        #     )

        #     # update progress bar
        #     time.sleep(3)  # wait for milling to start
        #     elapsed_time = 0
        #     progressbar = napari.utils.progress()

        #     self.milling_time_seconds = 600 # TODO: update

        #     while self.microscope.patterning.state == "Running":

        #         elapsed_time += 1
        #         prog_val = int(elapsed_time / self.milling_time_seconds * 100)
        #         progressbar.update(prog_val)
        #         time.sleep(1)
        #     logging.info(f"Milling finished: {self.microscope.patterning.state}")
        #     progressbar.close()

        # # reset to imaging mode
        # milling.finish_milling(
        #     microscope=self.microscope,
        #     imaging_current=self.settings.default.imaging_current,
        # )

        # update image
        self.setup_ui()

        # confirm finish
        self.finalise_milling()    

        # re-update patterns
        self.update_milling_pattern()    
    
    def finalise_milling(self) -> bool:

        if self.auto_continue:
            self.close()

        # ask user if the milling succeeded
        response = fibsem_ui.message_box_ui(
            title="Milling Confirmation", text="Do you want to redo milling?"
        )

        if response:
            logging.info("Redoing milling")
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
        self.viewer.window.close()
        event.accept()


def convert_pattern_to_napari_rect(pattern, image: np.ndarray, pixelsize: float) -> np.ndarray:
    
    # image centre
    icy, icx = image.shape[0] // 2, image.shape[1]// 2

    # pattern to pixel coords
    w = int(pattern.width / pixelsize)
    h = int(pattern.height  / pixelsize)
    cx = int(icx + (pattern.center_x / pixelsize))
    cy = int(icy - (pattern.center_y / pixelsize))

    xmin, xmax = cx-w/2, cx+w/2
    ymin, ymax = cy-h/2, cy+h/2

    # napari shape format
    shape = [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]

    return shape

def convert_napari_rect_to_mill_settings(arr: np.array, image: np.array, pixelsize: float, depth: float = 10e-6) -> MillingSettings:
    # convert napari rect to milling pattern

    # get centre of image
    cy_mid, cx_mid = image.data.shape[0] // 2, image.shape[1] // 2

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
    mill_settings = MillingSettings(width=width, height=height, depth=depth, centre_x=cx_real, centre_y=cy_real)

    return mill_settings



def main():

    microscope, settings = fibsem_utils.setup_session()
    settings = fibsem_utils.load_settings_from_config(
        config_path=config.config_path,
        protocol_path=config.protocol_path,
    )
    milling_pattern = MillingPattern.JCut
    auto_continue = False

    viewer = napari.Viewer()
    milling_ui = MillingUI(viewer=viewer, 
            microscope=microscope, settings=settings, 
            milling_pattern=milling_pattern, 
            auto_continue=auto_continue)
    viewer.window.add_dock_widget(milling_ui, area="right", add_vertical_stretch=False)
    napari.run()


if __name__ == "__main__":
    main()
