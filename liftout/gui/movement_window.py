
import sys
import logging

from liftout import utils
from liftout.fibsem import acquire, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem.acquire import BeamType, ImageSettings
from liftout.gui.qtdesigner_files import movement_dialog as movement_gui
from liftout.gui.utils import _WidgetPlot, draw_crosshair
from PyQt5 import QtCore, QtWidgets
import scipy.ndimage as ndi

MICRON_TO_METRE = 1e-6 
METRE_TO_MICRON = 1e6

class GUIMMovementWindow(movement_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, microscope, settings: dict, image_settings: ImageSettings, msg_type: str=None, parent=None):
        super(GUIMMovementWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.microscope = microscope
        self.settings = settings
        self.image_settings = image_settings

        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        self.wp_ib = None
        self.wp_eb = None

        # set message
        msg_dict = {
            "centre_ib": "Please centre the lamella in the Ion Beam (Double click to move).",
            "centre_eb": "Please centre the lamella in the Electron Beam (Double click to move). ",
            "eucentric": "Please centre a feature in both Beam views (Double click to move). ",
        }
        self.label_message.setText(msg_dict[msg_type])

        # enable / disable view movement
        self.eb_movement_enabled = False
        self.ib_movement_enabled = False

        if msg_type in ["eucentric", "centre_eb"]:
            self.eb_movement_enabled = True

        if msg_type in ["eucentric", "centre_ib"]:
            self.ib_movement_enabled = True

        self.setup_connections()
        self.update_displays()

    def update_displays(self):
        """Update the displays for both Electron and Ion Beam views"""
        
        logging.info("updating displays for Electron and Ion beam views")
        
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

        if self.eb_movement_enabled:
            self.wp_eb.canvas.mpl_connect('button_press_event', self.on_click)

        if self.ib_movement_enabled:
            self.wp_ib.canvas.mpl_connect('button_press_event', self.on_click)

    def setup_connections(self):

        logging.info("setup connections")

        self.pushButton_continue.clicked.connect(self.continue_button_pressed)
        self.pushButton_take_image.clicked.connect(self.take_image_button_pressed)

        self.doubleSpinBox_hfw.setMinimum(30e-6 * METRE_TO_MICRON)
        self.doubleSpinBox_hfw.setMaximum(self.settings["imaging"]["max_ib_hfw"] * METRE_TO_MICRON)
        self.doubleSpinBox_hfw.setValue(self.image_settings.hfw * METRE_TO_MICRON)
        self.doubleSpinBox_hfw.valueChanged.connect(self.update_image_settings)

    def take_image_button_pressed(self):
        """Take a new image with the current image settings."""

        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

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

    def stage_movement(self, beam_type: BeamType):

        stage = self.microscope.specimen.stage

        # calculate stage movement
        x_move = movement.x_corrected_stage_movement(self.center_x,
                                                     stage_tilt=stage.current_position.t)
        yz_move = movement.y_corrected_stage_movement(self.center_y,
                                                      stage_tilt=stage.current_position.t,
                                                      settings=self.settings,
                                                      beam_type=beam_type)
        logging.info(f"x_move: {x_move}, yz_move: {yz_move}")

        
        # move stage
        stage.relative_move(x_move)
        stage.relative_move(yz_move)

        # take new images
        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        # update displays
        self.update_displays()

# TODO: override enter

def main():

    settings = utils.load_config(r"C:\Users\Admin\Github\autoliftout\liftout\protocol_liftout.yml")
    microscope = fibsem_utils.initialise_fibsem(ip_address=settings["system"]["ip_address"])
    image_settings = ImageSettings(
        resolution = settings["imaging"]["resolution"],
        dwell_time = settings["imaging"]["dwell_time"],
        hfw = settings["imaging"]["horizontal_field_width"],
        autocontrast = True,
        beam_type = BeamType.ION,
        gamma = settings["gamma"],
        save = False,
        label = "test",
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
