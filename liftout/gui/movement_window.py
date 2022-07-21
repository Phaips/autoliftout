
import logging
import sys

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client.structures import StagePosition
from liftout import utils
from liftout.fibsem import acquire, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem.acquire import BeamType, GammaSettings, ImageSettings
from liftout.fibsem.constants import METRE_TO_MICRON, MICRON_TO_METRE, MILLIMETRE_TO_METRE, METRE_TO_MILLIMETRE
from liftout.gui.qtdesigner_files import movement_dialog as movement_gui
from liftout.gui.utils import _WidgetPlot, draw_crosshair
from PyQt5 import QtCore, QtWidgets


class GUIMMovementWindow(movement_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, microscope, settings: dict, image_settings: ImageSettings, msg_type: str = None, msg: str = None, parent=None):
        super(GUIMMovementWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.microscope = microscope
        self.settings = settings
        self.image_settings = image_settings

        self.wp_ib = None
        self.wp_eb = None

        # set message
        msg_dict = {
            "centre_ib": "Please centre the lamella in the Ion Beam (Double click to move).",
            "centre_eb": "Please centre the lamella in the Electron Beam (Double click to move). ",
            "eucentric": "Please centre a feature in both Beam views (Double click to move). ",
            "alignment": "Please centre the lamella in the Ion Beam, and tilt so the lamella face is perpendicular to the Ion Beam."
        }
        if msg is None:
            msg = msg_dict[msg_type]
        self.label_message.setText(msg)

        # enable / disable view movement
        self.eb_movement_enabled = False
        self.ib_movement_enabled = False
        self.tilt_movement_enabled = False
        self.vertical_movement_enabled = False

        if msg_type in ["eucentric", "centre_eb", "alignment"]:
            self.eb_movement_enabled = True

        if msg_type in ["eucentric", "centre_ib", "alignment"]:
            self.ib_movement_enabled = True

        if msg_type in ["alignment"]:
            self.tilt_movement_enabled = True

        if msg_type in ["eucentric"]:
            self.vertical_movement_enabled = True

        # movement permissions:
        # eucentric: eb, ib, vertical
        # centre_eb: eb
        # centre_ib: ib
        # alignment: eb, ib, tilt

        self.setup_connections()
        self.update_displays()

    def update_displays(self):
        """Update the displays for both Electron and Ion Beam views"""

        logging.info("updating displays for Electron and Ion beam views")
        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        # median filter image for better display
        eb_image_smooth = ndi.median_filter(self.eb_image.data, size=3)
        ib_image_smooth = ndi.median_filter(self.ib_image.data, size=3)

        # update eb view
        if self.wp_eb is not None:
            self.label_image_eb.layout().removeWidget(self.wp_eb)
            # TODO: remove layouts properly
            self.wp_eb.deleteLater()

        self.wp_eb = _WidgetPlot(self, display_image=eb_image_smooth)
        self.label_image_eb.setLayout(QtWidgets.QVBoxLayout())
        self.label_image_eb.layout().addWidget(self.wp_eb)

        # update ib view
        if self.wp_ib is not None:
            self.label_image_ib.layout().removeWidget(self.wp_ib)
            # TODO: remove layouts properly
            self.wp_ib.deleteLater()

        self.wp_ib = _WidgetPlot(self, display_image=ib_image_smooth)
        self.label_image_ib.setLayout(QtWidgets.QVBoxLayout())
        self.label_image_ib.layout().addWidget(self.wp_ib)

        # draw crosshairs on both images
        draw_crosshair(self.eb_image, self.wp_eb.canvas)
        draw_crosshair(self.ib_image, self.wp_ib.canvas)

        self.wp_eb.canvas.ax11.set_title("Electron Beam")
        self.wp_ib.canvas.ax11.set_title("Ion Beam")

        self.wp_eb.canvas.draw()
        self.wp_ib.canvas.draw()

        # reconnect buttons
        if self.eb_movement_enabled:
            self.wp_eb.canvas.mpl_connect('button_press_event', self.on_click)

        if self.ib_movement_enabled:
            self.wp_ib.canvas.mpl_connect('button_press_event', self.on_click)

    def setup_connections(self):

        logging.info("setup connections")

        self.pushButton_continue.clicked.connect(self.continue_button_pressed)
        self.pushButton_take_image.clicked.connect(self.take_image_button_pressed)

        self.doubleSpinBox_hfw.setMinimum(30e-6 * METRE_TO_MICRON)
        self.doubleSpinBox_hfw.setMaximum(self.settings["calibration"]["limits"]["max_ib_hfw"] * METRE_TO_MICRON)
        self.doubleSpinBox_hfw.setValue(self.image_settings.hfw * METRE_TO_MICRON)
        self.doubleSpinBox_hfw.valueChanged.connect(self.update_image_settings)

        # tilt functionality
        self.doubleSpinBox_tilt_degrees.setMinimum(0)
        self.doubleSpinBox_tilt_degrees.setMaximum(25.0)
        if self.tilt_movement_enabled:
            self.pushButton_tilt_stage.setVisible(True)
            self.doubleSpinBox_tilt_degrees.setVisible(True)
            self.pushButton_tilt_stage.setEnabled(True)
            self.doubleSpinBox_tilt_degrees.setEnabled(True)
            self.label_tilt.setVisible(True)
            self.label_header_tilt.setVisible(True)
            self.pushButton_tilt_stage.clicked.connect(self.stage_tilt)
        else:
            self.label_tilt.setVisible(False)
            self.label_header_tilt.setVisible(False)
            self.pushButton_tilt_stage.setVisible(False)
            self.doubleSpinBox_tilt_degrees.setVisible(False)
            self.pushButton_tilt_stage.setVisible(False)
            self.doubleSpinBox_tilt_degrees.setVisible(False)

        # vertical movement functionality
        self.doubleSpinBox_stage_height.setMinimum(3.8)
        self.doubleSpinBox_stage_height.setMaximum(4.1)
        if self.vertical_movement_enabled:
            self.pushButton_move_stage.setVisible(True)
            self.pushButton_move_to_eucentric.setVisible(True)
            self.doubleSpinBox_stage_height.setVisible(True)
            self.label_height.setVisible(True)
            self.label_header_height.setVisible(True)
            self.pushButton_move_stage.clicked.connect(self.move_stage_vertical)
            self.pushButton_move_to_eucentric.clicked.connect(self.move_to_eucentric_point)
        else:
            self.pushButton_move_stage.setVisible(False)
            self.pushButton_move_to_eucentric.setVisible(False)
            self.doubleSpinBox_stage_height.setVisible(False)
            self.label_height.setVisible(False)
            self.label_header_height.setVisible(False)

    def take_image_button_pressed(self):
        """Take a new image with the current image settings."""

        self.update_displays()

    def update_image_settings(self):
        """Update the image settings when ui elements change"""

        # TODO: validate these values....
        self.image_settings.hfw = self.doubleSpinBox_hfw.value() * MICRON_TO_METRE

    def continue_button_pressed(self):
        logging.info("continue button pressed")
        self.close()

    def closeEvent(self, event):
        logging.info("closing movement window")
        # TODO: update parent image settings on close?
        event.accept()

    def on_click(self, event):
        """Move to the selected position on user double click"""

        if event.inaxes is self.wp_ib.canvas.ax11:
            beam_type = BeamType.ION
            adorned_image = self.ib_image

        if event.inaxes is self.wp_eb.canvas.ax11:
            beam_type = BeamType.ELECTRON
            adorned_image = self.eb_image

        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata
            self.center_x, self.center_y = movement.pixel_to_realspace_coordinate(
                (self.xclick, self.yclick), adorned_image)

            logging.info(f"Clicked on {beam_type} Image at {self.center_x:.2e}, {self.center_y:.2e}")
            # draw crosshair?
            if event.dblclick:
                self.stage_movement(beam_type=beam_type)

                if self.parent():
                    logging.info(f"{self.parent().current_stage} | DOUBLE CLICK")

    def stage_movement(self, beam_type: BeamType):

        stage = self.microscope.specimen.stage

        # calculate stage movement
        x_move = movement.x_corrected_stage_movement(self.center_x,
                                                     stage_tilt=stage.current_position.t)
        yz_move = movement.y_corrected_stage_movement(microscope=self.microscope,
                                                      expected_y=self.center_y,
                                                      stage_tilt=stage.current_position.t,
                                                      settings=self.settings,
                                                      beam_type=beam_type)
        logging.info(f"x_move: {x_move}, yz_move: {yz_move}")

        # move stage
        stage.relative_move(x_move)
        stage.relative_move(yz_move)

        # update displays
        self.update_displays()

    def stage_tilt(self):
        """Tilt the stage to the desired angle

        Args:
            stage_tilt (float): desired stage tilt angle (degrees)
        """

        stage_tilt_rad: float = np.deg2rad(self.doubleSpinBox_tilt_degrees.value())
        stage = self.microscope.specimen.stage
        stage_position = StagePosition(t=stage_tilt_rad)
        stage.absolute_move(stage_position)

        # update displays
        self.update_displays()

    def move_to_eucentric_point(self):
        print("move to eucentric point")

    def move_stage_vertical(self):
        print("move stage vertical")
        z_height_m = self.doubleSpinBox_stage_height.value() * MILLIMETRE_TO_METRE

        print(f"stage_height: {z_height_m:.4f}m")

        stage_position = StagePosition(z=z_height_m)


# TODO: override enter


def main():

    settings = utils.load_config(r"C:\Users\Admin\Github\autoliftout\liftout\protocol_liftout.yml")
    microscope = fibsem_utils.initialise_fibsem(ip_address=settings["system"]["ip_address"])
    image_settings = ImageSettings(
        resolution=settings["imaging"]["resolution"],
        dwell_time=settings["imaging"]["dwell_time"],
        hfw=settings["imaging"]["horizontal_field_width"],
        autocontrast=True,
        beam_type=BeamType.ION,
        gamma=GammaSettings(
            enabled=settings["gamma"]["correction"],
            min_gamma=settings["gamma"]["min_gamma"],
            max_gamma=settings["gamma"]["max_gamma"],
            scale_factor=settings["gamma"]["scale_factor"],
            threshold=settings["gamma"]["threshold"]
        ),
        save=False,
        label="test",
    )
    app = QtWidgets.QApplication([])
    qt_app = GUIMMovementWindow(microscope=microscope,
                                settings=settings,
                                image_settings=image_settings,
                                msg_type="eucentric"
                                )
    qt_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
