import logging
import sys
from pprint import pprint

import napari
import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import AdornedImage
from fibsem import calibration, constants
from fibsem import utils as fibsem_utils
from fibsem.structures import MicroscopeSettings
from liftout import autoliftout, patterning, utils
from liftout.gui.qtdesigner_files import MillingUI
from liftout.patterning import MillingPattern
from napari.utils import notifications
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel

from liftout.config import config

# TODO: connect parameter changes


class MillingUI(MillingUI.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, viewer: napari.Viewer=None, microscope: SdbMicroscopeClient = None, settings: MicroscopeSettings = None):
        super(MillingUI, self).__init__()


        # setup ui
        self.setupUi(self)

        self.viewer = viewer
        # self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        # self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        if microscope is None:
            self.microscope, self.settings = fibsem_utils.setup_session()

        self.settings = fibsem_utils.load_settings_from_config(
            config_path = "/home/patrick/github/autoliftout/liftout/config",
            protocol_path = "/home/patrick/github/autoliftout/liftout/protocol/protocol.yaml"
        )

        self.setup_connections()
        self.update_ui()

    def setup_connections(self):

        logging.info("setup connections")

        # combobox
        self.comboBox_select_pattern.currentTextChanged.connect(self.update_milling_stages)
        self.comboBox_select_pattern.addItems(pattern.name for pattern in MillingPattern)

        # buttons
        self.pushButton_run_milling.clicked.connect(self.run_milling)
        self.pushButton_exit_milling.clicked.connect(self.exit_milling)
    

    def update_ui(self):

        # draw image
        arr = np.ones(shape=(1024, 1536))
        self.image  = arr#AdornedImage(data=arr)

        self.viewer.layers.clear()
        self.image_layer = self.viewer.add_image(self.image, name="Image")
        self.image_layer.mouse_double_click_callbacks.append(self._double_click) # append callback
        self.pattern_layer = self.viewer.add_shapes(None, name="Patterns")

        self.viewer.layers.selection.active = self.image_layer

        # draw milling patterns


    def _double_click(self, layer, event):

        coords = layer.world_to_data(event.position)

        image_shape = self.image.shape

        if (coords[0] > 0 and coords[0] < image_shape[0]) and (coords[1] > 0 and coords[1] < image_shape[1]):
            logging.info(f"click inside image: {coords[0]:.2f}, {coords[1]:.2f}")
        else:
            napari.utils.notifications.show_info(f"Clicked outside image dimensions. Please click inside the image to move.")
            return

        self.update_milling_pattern(coords)

    def update_milling_pattern(self, coords):

        logging.info(f"update milling pattern")
        
        # TODO: use convert pattern to napari rect...

        w, h = 300, 200
        icy, icx = self.image.shape[0]//2, self.image.shape[1]//2

        cy, cx = coords

        xmin, xmax = cx-w/2, cx+w/2
        ymin, ymax = cy-h/2, cy+h/2
        
        # napari shape format
        shape = [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]

        shape_patterns = []
        shape_patterns.append(shape)

        self.viewer.layers.remove(self.viewer.layers["Patterns"])
        self.viewer.add_shapes(shape_patterns, name="Patterns", shape_type='rectangle', edge_width=1,
                            edge_color='yellow', face_color='yellow')
        self.viewer.layers.selection.active = self.image_layer


    def update_milling_stages(self):

        logging.info("update_milling_stages")

        milling_pattern = MillingPattern[self.comboBox_select_pattern.currentText()]

        milling_protocol_stages = patterning.get_milling_protocol_stage_settings(
            self.settings, milling_pattern
        )

        n_stages = len(milling_protocol_stages)

        self.milling_stages = {}
        for i, stage_settings in enumerate(milling_protocol_stages, 1):
            self.milling_stages[f"{milling_pattern.name}_{i}"] = stage_settings

        try:
            self.comboBox_milling_stage.disconnect()
        except:
            pass
        self.comboBox_milling_stage.clear()
        self.comboBox_milling_stage.currentTextChanged.connect(
            self.update_milling_parameters_ui
        )
        self.comboBox_milling_stage.addItems(list(self.milling_stages.keys()))


    # TODO: draw pattern
    # TODO: milling current
    # TODO: update on param change
    # TODO: support for other element types

    def update_milling_parameters_ui(self):

        # remove existing elements
        for i in reversed(range(self.gridLayout_2.count())): 
            self.gridLayout_2.itemAt(i).widget().setParent(None)

        i = 0
        current_stage = self.comboBox_milling_stage.currentText()
        for (k, v) in self.milling_stages[current_stage].items():
            
            if k not in config.NON_CHANGEABLE_MILLING_PARAMETERS:
                if k not in config.NON_SCALED_MILLING_PARAMETERS:
                    v = float(v) * constants.METRE_TO_MICRON

                label = QLabel()
                label.setText(str(k))
                spinBox_value = QtWidgets.QDoubleSpinBox()
                spinBox_value.setValue(v)
                spinBox_value.valueChanged.connect(self.update_milling_settings)

                self.gridLayout_2.addWidget(label, i, 0)
                self.gridLayout_2.addWidget(spinBox_value, i, 1)

                i+=1

    def update_milling_settings(self):

        logging.info(f"sender: {self.sender()}")

        # TODO: START_HERE

    def run_milling(self):

        logging.info("run milling")

    
    def exit_milling(self):

        logging.info("exit milling")

        self.close()

    def closeEvent(self, event):
        self.microscope.patterning.clear_patterns()
        event.accept()


def main():
    """Launch the `autoliftout` main application window."""
    viewer = napari.Viewer()
    milling_ui = MillingUI(viewer=viewer)
    viewer.window.add_dock_widget(milling_ui, area="right", add_vertical_stretch=False)
    napari.run()

if __name__ == "__main__":
    main()
