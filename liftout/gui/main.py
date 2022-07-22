import logging
import sys
from pprint import pprint

import matplotlib
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from liftout import autoliftout, utils
from liftout.fibsem import acquire, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem import validation
from liftout.fibsem.acquire import ImageSettings
from liftout.fibsem.sample import AutoLiftoutStage, Lamella, Sample
from liftout.gui import utils as ui_utils
from liftout.gui.qtdesigner_files import main as gui_main
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGroupBox, QInputDialog

matplotlib.use("Agg")

_translate = QtCore.QCoreApplication.translate


class AutoLiftoutMainWindow(gui_main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(AutoLiftoutMainWindow, self).__init__()

        # load config
        self.settings: dict = utils.load_config_v2()
        self.image_settings: ImageSettings = acquire.update_image_settings_v3(
            self.settings
        )

        # setup ui
        self.setupUi(self)
        self.set_ui_style()
        self.showNormal()

        # load experiment
        self.sample: Sample = self.setup_experiment()

        # return
        logging.info(f"INIT | {AutoLiftoutStage.Initialisation.name} | STARTED")

        # initialise hardware
        self.microscope: SdbMicroscopeClient = self.initialize_hardware(
            settings=self.settings,
            log_path=self.sample.log_path,
            ip_address=self.settings["system"]["ip_address"]
        )
        # self.microscope: SdbMicroscopeClient = None
        self.MICROSCOPE_CONNECTED: bool = bool(self.microscope)  # offline mode

        # setup connections
        self.setup_connections()

        # update display
        self.update_ui_display()

        logging.info(f"INIT | {AutoLiftoutStage.Initialisation.name} | FINISHED")

    ########################## AUTOLIFTOUT ##########################

    def run_setup_autoliftout(self):
        """Run the autoliftout setup workflow."""
        autoliftout.run_setup_autoliftout(
            microscope=self.microscope,
            settings=self.settings,
            image_settings=self.image_settings,
            sample=self.sample,
            parent_ui=self,
        )

    def run_autoliftout(self):
        """Run the autoliftout main workflow."""
        autoliftout.run_autoliftout_workflow(
            microscope=self.microscope,
            settings=self.settings,
            image_settings=self.image_settings,
            sample=self.sample,
            parent_ui=self,
        )

    def run_autoliftout_thinning(self):
        "Run the autoliftout thinning workflow."
        autoliftout.run_thinning_workflow(
            microscope=self.microscope,
            settings=self.settings,
            image_settings=self.image_settings,
            sample=self.sample,
            parent_ui=self,
        )

    ########################## UTILS ##########################

    def setup_experiment(self) -> Sample:

        # TODO: add a select folder option for new experiment

        try:
            sample: Sample = ui_utils.setup_experiment_sample_ui(parent_ui=self)

            if sample.positions:
                # enable autoliftout buttons
                self.pushButton_autoliftout.setEnabled(True)
                self.pushButton_thinning.setEnabled(True)

        except Exception as e:
            ui_utils.display_error_message(message=f"Unable to setup sample: {e}")
            sample = None

        return sample

    def set_lamella_failed(self):
        """Mark the indicated sample position as failed."""

        # TODO: redo this better..
        # think it is currently broken
        if self.sample:

            # show the user the sample number and petname
            lamella_str = [
                f"Lamella {lamella._number}"
                for lamella in self.sample.positions.values()
            ]
            lamella_petname, okPressed = QInputDialog.getItem(
                self, "Select a Lamella", "Lamella No:", lamella_str, 0, False
            )
            lamella_idx = int(lamella_petname.split(" ")[-1])

            if okPressed:
                # mark sample as failure
                lamella = self.sample.positions[lamella_idx]
                lamella.current_state.microscope_state.last_completed_stage = (
                    AutoLiftoutStage.Failure
                )
                self.sample = autoliftout.update_sample_lamella_data(
                    self.sample, lamella
                )
                logging.info(
                    f"Marked {lamella_petname} as {AutoLiftoutStage.Failure.name}"
                )

                # update UI
                self.update_scroll_ui()

    def load_protocol_from_file(self):
        # TODO: add protocol file name to ui?

        # reload the protocol from file
        try:
            self.settings = ui_utils.load_configuration_from_ui(self)
        except Exception as e:
            ui_utils.display_error_message(f"Unable to load selected protocol: {e}")

    def run_load_experiment_utility(self):
        """Run the laod experiment utility"""
        self.sample = self.setup_experiment()

        self.update_ui_display()

    def run_sharpen_needle_utility(self):
        """Run the sharpen needle utility, e.g. reset stage"""
        # TODO: fix this so it doesnt rely on lamella...
        self.image_settings.save = False
        autoliftout.reset_needle(
            self.microscope,
            self.settings,
            self.image_settings,
            Lamella(self.sample.path, 999),
        )

    def run_sputter_platinum_utility(self):
        """Run the sputter platinum utility"""

        # move to the initial sample grid position
        movement.move_to_sample_grid(self.microscope, self.settings)
        # sputter
        fibsem_utils.sputter_platinum_on_whole_sample_grid_v2(
            self.microscope, self.settings, self.image_settings
        )

    def testing_function(self):
        print("yay testing function")

    ########################## HARDWARE ##########################

    def initialize_hardware(
        self, settings: dict, log_path: str, ip_address: str = "10.0.0.1"
    ) -> SdbMicroscopeClient:

        microscope = fibsem_utils.connect_to_microscope(
            ip_address=ip_address, parent_ui=self
        )

        if microscope:
            # run validation and show in ui
            validation.run_validation_ui(
                microscope=microscope, settings=settings, log_path=log_path,
            )

        return microscope

    def disconnect(self):
        logging.info("Running cleanup/teardown")
        logging.debug("Running cleanup/teardown")
        if self.microscope:
            self.microscope.disconnect()

    ########################## USER INTERFACE ##########################

    def setup_connections(self):
        logging.info("gui: setup connections started")
        # connect buttons
        self.pushButton_initialise.clicked.connect(self.run_setup_autoliftout)
        self.pushButton_autoliftout.clicked.connect(self.run_autoliftout)
        self.pushButton_thinning.clicked.connect(self.run_autoliftout_thinning)
        self.pushButton_autoliftout.setEnabled(False)  # disable unless lamella selected
        self.pushButton_thinning.setEnabled(False)  # disable unless lamella selected

        # load data
        self.pushButton_add_sample_position.setVisible(False)

        # configuration management
        self.actionLoad_Experiment.triggered.connect(self.run_load_experiment_utility)
        self.actionLoad_Protocol.triggered.connect(self.load_protocol_from_file)

        # actions
        self.actionMark_Lamella_Failed.triggered.connect(self.set_lamella_failed)

        # utilities
        self.actionSharpen_Needle.triggered.connect(self.run_sharpen_needle_utility)
        self.actionSputter_Platinum.triggered.connect(self.run_sputter_platinum_utility)

        # TESTING METHODS TODO: TO BE REMOVED
        self.pushButton_test_popup.clicked.connect(lambda: self.testing_function())

        logging.info("gui: setup connections finished")

    def set_ui_style(self):
        # UI style
        self.setWindowTitle("AutoLiftout")
        self.label_title.setStyleSheet(
            "font-family: Arial; font-weight: bold; font-size: 36px; border: 0px solid lightgray"
        )
        self.label_stage.setStyleSheet(
            "background-color: gray; padding: 10px; border-radius: 5px; color: white;"
        )
        self.label_stage.setFont(QtGui.QFont("Arial", 10, weight=QtGui.QFont.Bold))
        self.label_stage.setAlignment(QtCore.Qt.AlignCenter)
        self.label_stage.setText(f"{AutoLiftoutStage.Initialisation.name}")

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

    def update_scroll_ui(self):
        """Update the central ui grid with current sample data."""

        horizontalGroupBox = QGroupBox() 
        gridLayout = ui_utils.draw_grid_layout(self.sample)
        horizontalGroupBox.setLayout(gridLayout)
        horizontalGroupBox.update()
        self.scroll_area.setWidget(horizontalGroupBox)
        self.scroll_area.update()

    def update_status(self, lamella: Lamella = None):
        """Update status information."""

        if lamella is None:
            if self.sample.positions != {}:
                key = list(self.sample.positions.keys())[0]
                lamella = self.sample.positions[key]
            else: 
                lamella = Lamella(self.sample.path, 0)
        ui_utils.update_stage_label(self.label_stage, lamella)

        # log info
        self.statusBar.showMessage(utils.get_last_log_message(self.sample.log_path))
        self.statusBar.repaint()

    def update_ui_display(self):
        try:
            self.update_status()
            self.update_scroll_ui()
        except Exception as e:
            print(f"Exception updating ui: {e}")
        return


def main():
    """Launch the `autoliftout` main application window."""
    app = QtWidgets.QApplication([])
    autoliftout_ui = AutoLiftoutMainWindow()
    app.aboutToQuit.connect(autoliftout_ui.disconnect)  # cleanup & teardown
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
