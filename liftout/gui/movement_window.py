
from liftout.fibsem import acquire, movement
from liftout.gui.qtdesigner_files import movement_dialog as movement_gui

from PyQt5 import QtWidgets, QtCore
import sys

from liftout import utils
from liftout.fibsem import utils as fibsem_utils

from liftout.fibsem.acquire import BeamType
from liftout.gui.utils import _WidgetPlot, create_crosshair


class GUIMMovementWindow(movement_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, microscope, settings: dict, image_settings: dict, msg_type: str, parent=None):
        super(GUIMMovementWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.microscope = microscope
        self.settings = settings
        self.image_settings = image_settings

        # self.eb_image = acquire.last_image(self.microscope, BeamType.ELECTRON)
        # self.ib_image = acquire.last_image(self.microscope, BeamType.ION)
        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        self.wp_ib = None
        self.wp_eb = None

        # set message
        msg_dict = {
            "centre_ib": "Please centre the lamella in the ion beam (Double click to move). \nPress continue when complete.",
            "centre_eb": "Please centre the lamella in the electron beam (Double click to move). \nPress continue when complete.",
            "eucentric": "Please centre a feature in both beam views (Double click to move). \nPress continue when complete.",
            "milling_success": "Was the milling successful?\nIf not, please manually fix, and then press continue.",
            "working_distance": "The working distance seems to be incorrect, please manually fix the focus. \nPress continue when complete.",
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
        
        # update eb view
        if self.wp_eb is not None:
            self.label_image_eb.layout().removeWidget(self.wp_eb)
            self.wp_eb.deleteLater()

        self.wp_eb = _WidgetPlot(self, display_image=self.eb_image.data)
        self.label_image_eb.setLayout(QtWidgets.QVBoxLayout())
        self.label_image_eb.layout().addWidget(self.wp_eb)

        # update ib view
        if self.wp_ib is not None:
            self.label_image_ib.layout().removeWidget(self.wp_ib)
            self.wp_ib.deleteLater()

        self.wp_ib = _WidgetPlot(self, display_image=self.ib_image.data)
        self.label_image_ib.setLayout(QtWidgets.QVBoxLayout())
        self.label_image_ib.layout().addWidget(self.wp_ib)

        def draw_crosshair(image, canvas):
            # draw crosshairs
            crosshair = create_crosshair(image)
            canvas.ax11.patches = []
            for patch in crosshair.__dataclass_fields__:
                canvas.ax11.add_patch(getattr(crosshair, patch))
                getattr(crosshair, patch).set_visible(True)

        # draw crosshairs on both images
        draw_crosshair(self.eb_image, self.wp_eb.canvas)
        draw_crosshair(self.ib_image, self.wp_ib.canvas)

        self.wp_eb.canvas.draw()
        self.wp_ib.canvas.draw()

        if self.eb_movement_enabled:
            self.wp_eb.canvas.mpl_connect('button_press_event', self.on_click)

        if self.ib_movement_enabled:
            self.wp_ib.canvas.mpl_connect('button_press_event', self.on_click)


    def setup_connections(self):
        
        print("Setup Connections")

        self.pushButton_continue.clicked.connect(self.continue_button_pressed)
        self.pushButton_take_image.clicked.connect(self.take_image_button_pressed)

        self.doubleSpinBox_hfw.setMinimum(30e-6)
        self.doubleSpinBox_hfw.setMaximum(self.settings["imaging"]["max_ib_hfw"] * 1e6)
        self.doubleSpinBox_hfw.valueChanged.connect(self.update_image_settings)

    def take_image_button_pressed(self):
        print("take image button pressed")

        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_displays()

    def update_image_settings(self):
        print("updating image settings")
        print(f"SENDER: ", self.sender())

        # TODO: validate these values....
        # TODO: fix MICRON_TO_METRE, and METRE_TO_MICRON
        self.image_settings["hfw"] = self.doubleSpinBox_hfw.value() * 1e-6

        from pprint import pprint
        pprint(self.image_settings)

    def continue_button_pressed(self):
        # TODO: do something
        print(f"Continue button pressed: {self.sender()}")

        self.close()

    def closeEvent(self, event):
        
        print("Closing Movement Window")
        event.accept()

    def on_click(self, event):
        """Move to the selected position on user double click"""

        if event.inaxes is self.wp_ib.canvas.ax11:
            print("Clicked in IB Axes")
            beam_type = BeamType.ION
            adorned_image = self.ib_image
        
        if event.inaxes is self.wp_eb.canvas.ax11:
            print("Clicked in EB Axes")
            beam_type = BeamType.ELECTRON
            adorned_image = self.eb_image        

        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata
            self.center_x, self.center_y = movement.pixel_to_realspace_coordinate(
                (self.xclick, self.yclick), adorned_image)

            print(f"Clicked on {beam_type} Image at {self.center_x:.2e}, {self.center_y:.2e}")
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

        print("X MOVE: ", x_move)
        print("Y_MOVE: ", yz_move)

        # move stage
        stage.relative_move(x_move)
        stage.relative_move(yz_move) # TODO: consolidate?

        # take new images
        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        # update displays
        self.update_displays()


    



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
    qt_app = GUIMMovementWindow(microscope=microscope, 
                                settings=settings, 
                                image_settings=image_settings,
                                msg_type="centre_ib" 
                                )
    qt_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()