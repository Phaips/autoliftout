import logging
import sys
from pprint import pprint

import fibsem.ui.utils as fibsem_ui_utils
import fibsem.ui.windows as fibsem_ui_windows
import matplotlib
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import utils as fibsem_utils
from fibsem import calibration
from liftout import autoliftout, utils
from liftout.config import config
from liftout.gui import utils as ui_utils
from liftout.gui.qtdesigner_files import AutoLiftoutUI 
from liftout.structures import AutoLiftoutStage, Lamella, Sample
from PyQt5 import QtCore, QtGui, QtWidgets

import napari 
from napari.utils import notifications

class AutoLiftoutUI(AutoLiftoutUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: napari.Viewer=None):
        super(AutoLiftoutUI, self).__init__()


        # setup ui
        self.setupUi(self)

        self.viewer = viewer

        # load experiment
        self.setup_experiment()

        # connect to microscope session
        # self.microscope, self.settings = fibsem_utils.setup_session(
        #     log_path = self.sample.path,
        #     config_path = config.config_path,
        #     protocol_path = config.protocol_path
        # )

        logging.info(f"INIT | {AutoLiftoutStage.Initialisation.name} | STARTED")

        # load settings / protocol
        self.settings = fibsem_utils.load_settings_from_config(
            config_path = config.config_path,
            protocol_path = config.protocol_path
        )
        # NB cant use setup session, because log dir cant be loaded until later...very silly


        # initialise hardware
        self.microscope = fibsem_utils.connect_to_microscope(ip_address=self.settings.system.ip_address)

        self.MICROSCOPE_CONNECTED: bool = bool(self.microscope)  # offline mode

        # run validation and show in ui
        if self.MICROSCOPE_CONNECTED:
            fibsem_ui_windows.run_validation_ui(
                microscope=self.microscope,
                settings=self.settings,
                log_path=self.sample.log_path,
            )
        else:
            notifications.show_info(
                f"AutoLiftout is unavailable. Unable to connect to microscope. Please see the console for more information."
            )

        # setup connections
        self.setup_connections()

        # update display
        self.update_ui()

        logging.info(f"INIT | {AutoLiftoutStage.Initialisation.name} | FINISHED")

    def setup_connections(self):

        logging.info("Setup Connections")

        # actions
        self.actionLoad_Protocol.triggered.connect(self.load_protocol_from_file)
        self.actionLoad_Experiment.triggered.connect(self.load_experiment_utility)

        # buttons
        self.pushButton_setup_autoliftout.clicked.connect(self.run_setup_autoliftout)
        self.pushButton_run_autoliftout.clicked.connect(self.run_autoliftout)
        self.pushButton_run_polishing.clicked.connect(self.run_autoliftout_thinning)
        self.pushButton_test_button.clicked.connect(self.testing_function)

        # widgets
        if self.sample.positions:
            self.comboBox_lamella_select.addItems([lamella._petname for lamella in self.sample.positions.values()])
        self.comboBox_lamella_select.currentTextChanged.connect(self.update_lamella_ui)
        self.checkBox_lamella_mark_failure.toggled.connect(self.mark_lamella_failure)

        
    def testing_function(self):
        logging.info(f"Test button pressed")

    def setup_experiment(self) -> None:

        # TODO: add a select folder option for new experiment
        try:
            sample: Sample = ui_utils.setup_experiment_sample_ui(parent_ui=self)

        except Exception as e:
            notifications.show_info(message=f"Unable to setup sample: {e}")
            sample = None

        self.sample = sample

    def load_experiment_utility(self):

        self.setup_experiment()
        self.update_ui()

    def update_lamella_ui(self):

        logging.info(f"Updating Lamella UI")

        # # update if the sample changes
        # self.comboBox_lamella_select.clear()
        # if self.sample.positions:
        #     self.comboBox_lamella_select.addItems([lamella._petname for lamella in self.sample.positions.values()])
        # TODO: need to update this combobox when sample changes, and disconnect signal to prevent inifite loop


        lamella = self.get_current_selected_lamella()
        self.checkBox_lamella_mark_failure.setChecked(lamella.is_failure)

        # info
        fail_string = "(Active)" if lamella.is_failure is False else "(Failure)" 
        self.label_lamella_status.setText(f"Stage: {lamella.current_state.stage.name} {fail_string}")

    def mark_lamella_failure(self):
        
        lamella = self.get_current_selected_lamella()
        lamella.is_failure = bool(self.checkBox_lamella_mark_failure.isChecked())
        self.sample = autoliftout.update_sample_lamella_data(
                self.sample, lamella
            )
        logging.info(f"Marked {lamella._petname} as Failure")

        self.update_lamella_ui()

    def get_current_selected_lamella(self) -> Lamella:

        lamella_petname = self.comboBox_lamella_select.currentText()
        lamella_idx = int(lamella_petname.split("-")[0])

        # mark sample as failure
        lamella: Lamella = self.sample.positions[lamella_idx]

        return lamella

# TODO: fix logging issue so loading exp doesnt have to be the first thing
# TODO: move validate to a button or on setup start rather than program start
# TODO: load protocol
# TODO: load experiment
# TODO: run info      
# TODO: add is_failure to workflow checks  

    def update_ui(self):

        self.update_lamella_ui()

        # update run info
        # n_stages / n_total
        # next stage..? how to calc?

        n_stages, active_lam, c_stages, t_stages, perc = ui_utils.get_completion_stats(self.sample)

        self.label_general_info.setText(f"{c_stages}/{t_stages} Stages Complete ({perc*100:.2f}%)")
        
    
        # enable autoliftout buttons
        LIFTOUT_ENABLED = bool(self.sample.positions)
        self.pushButton_run_autoliftout.setEnabled(LIFTOUT_ENABLED)
        self.pushButton_run_autoliftout.setVisible(LIFTOUT_ENABLED)
        self.pushButton_run_polishing.setEnabled(LIFTOUT_ENABLED)
        self.pushButton_run_polishing.setVisible(LIFTOUT_ENABLED)

        # TODO: stage labels, lamella labels
        # update main display
        overview_image = ui_utils.create_overview_image(self.sample)
        self.viewer.layers.clear()
        self.viewer.add_image(overview_image, name="AutoLiftout")

    def load_protocol_from_file(self):
        # TODO: add protocol file name to ui?

        # reload the protocol from file
        try:
            self.settings.protocol = ui_utils.load_configuration_from_ui(self)
        except Exception as e:
            notifications.show_info(f"Unable to load selected protocol: {e}")



    ########################## AUTOLIFTOUT ##########################

    def run_setup_autoliftout(self):
        """Run the autoliftout setup workflow."""
        logging.info(f"Run setup autoliftout")
        # self.sample = autoliftout.run_setup_autoliftout(
        #     microscope=self.microscope,
        #     settings=self.settings,
        #     sample=self.sample,
        #     parent_ui=self,
        # )


    def run_autoliftout(self):
        """Run the autoliftout main workflow."""
        logging.info(f"Run autoliftout")

        # self.sample = autoliftout.run_autoliftout_workflow(
        #     microscope=self.microscope,
        #     settings=self.settings,
        #     sample=self.sample,
        #     parent_ui=self,
        # )


    def run_autoliftout_thinning(self):
        "Run the autoliftout thinning workflow."
        logging.info(f"Run setup autoliftout")
        # self.sample = autoliftout.run_thinning_workflow(
        #     microscope=self.microscope,
        #     settings=self.settings,
        #     sample=self.sample,
        #     parent_ui=self,
        # )

    # ########################## UTILS ##########################


# TODO
# TODO



def main():
    """Launch the `autoliftout` main application window."""
    app = QtWidgets.QApplication([])
    viewer = napari.Viewer()
    autoliftout_ui = AutoLiftoutUI(viewer=viewer)
    viewer.window.add_dock_widget(autoliftout_ui, area="right", add_vertical_stretch=False)

    app.aboutToQuit.connect(autoliftout_ui.disconnect)  # cleanup & teardown
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
