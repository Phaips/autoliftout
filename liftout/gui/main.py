import datetime
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from pprint import pprint

import liftout
import matplotlib
import numpy as np
from liftout import utils
from liftout import autoliftout
from liftout.detection.utils import DetectionType
from liftout.fibsem import acquire, calibration, milling, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem import validation
from liftout.fibsem.acquire import ImageSettings
from liftout.fibsem.sample import Lamella, Sample, AutoLiftoutStage
from liftout.fibsem.sampleposition import SamplePosition
from liftout.gui import utils as ui_utils
from liftout.gui import windows
from liftout.gui.milling_window import MillingPattern
from liftout.gui.qtdesigner_files import main as gui_main
from liftout.gui.utils import draw_grid_layout
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGroupBox, QInputDialog, QLineEdit, QMessageBox

from liftout.config import config

matplotlib.use("Agg")

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
from autoscript_sdb_microscope_client.structures import (
    AdornedImage,
    MoveSettings,
    StagePosition,
)

# Required to not break imports
BeamType = acquire.BeamType

_translate = QtCore.QCoreApplication.translate


class GUIMainWindow(gui_main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(GUIMainWindow, self).__init__()

        # load config
        self.settings: dict = utils.load_config_v2()
        self.image_settings: ImageSettings =  acquire.update_image_settings_v3(self.settings)

        # setup ui
        self.setupUi(self)
        self.set_ui_style()
        self.showNormal()

        # load experiment
        self.sample: Sample = self.setup_experiment()

        # return
        logging.info(f"INIT | {AutoLiftoutStage.Initialisation.name} | STARTED")

        # initialise hardware
        # self.microscope: SdbMicroscopeClient = self.initialize_hardware(
        #     settings=self.settings, 
        #     log_path=self.sample.log_path,
        #     ip_address=self.settings["system"]["ip_address"]
        # )
        self.microscope: SdbMicroscopeClient = None
        self.MICROSCOPE_CONNECTED: bool = bool(self.microscope) # offline mode

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
            parent_ui = self
        )


    def run_autoliftout(self):
        """Run the autoliftout main workflow."""
        autoliftout.run_autoliftout_workflow(
            microscope=self.microscope,
            settings=self.settings,
            image_settings=self.image_settings,
            sample=self.sample,
            parent_ui = self
        )
    
    def run_autoliftout_thinning(self):
        "Run the autoliftout thinning workflow."
        autoliftout.run_thinning_workflow(
            microscope=self.microscope, 
            settings=self.settings,
            image_settings=self.image_settings,
            sample=self.sample,
            parent_ui = self
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

    def mark_sample_position_failed(self):
        """Mark the indicated sample position as failed."""
        
        # TODO: redo this better..
        #think it is currently broken
        if self.sample:

            # show the user the sample number and petname
            lamella_str = [f"Lamella {lamella._number}" for lamella in self.sample.positions.values()]
            lamella_petname, okPressed = QInputDialog.getItem(self, "Select a Lamella", "Lamella No:", lamella_str, 0, False
            )
            lamella_idx = int(lamella_petname.split(" ")[-1])

            pprint(self.sample.positions)
            if okPressed:
                # mark sample as failure
                lamella = self.sample.positions[lamella_idx]
                lamella.current_state.microscope_state.last_completed_stage = AutoLiftoutStage.Failure
                self.sample = autoliftout.update_sample_lamella_data(self.sample, lamella)
                logging.info(f"Marked {lamella_petname} as {AutoLiftoutStage.Failure.name}")

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


    def run_sharpen_needle_utility(self):
        """Run the sharpen needle utility, e.g. reset stage"""
        # TODO: fix this so it doesnt rely on lamella...
        self.image_settings.save = False
        autoliftout.reset_needle(self.microscope, self.settings, self.image_settings, Lamella(self.sample.path, 999))

    def run_sputter_platinum_utility(self):
        """Run the sputter platinum utility"""


        # move to the initial sample grid position
        movement.move_to_sample_grid(self.microscope, self.settings)        
        # sputter
        fibsem_utils.sputter_platinum_on_whole_sample_grid_v2(self.microscope, self.settings, self.image_settings)

    def testing_function(self):

        TEST_SAMPLE_POSITIONS = False
        TEST_MILLING_WINDOW = False
        TEST_DETECTION_WINDOW = False
        TEST_MOVEMENT_WINDOW = False
        TEST_USER_WINDOW = False

        TEST_RESET_NEEDLE = True
        if TEST_RESET_NEEDLE:
            self.reset_needle()

        if TEST_SAMPLE_POSITIONS:
            self.select_sample_positions()

        if TEST_MILLING_WINDOW:

            self.update_image_settings(hfw=150e-6, beam_type=BeamType.ION)
            # windows.open_milling_window_v2(MillingPattern.JCut)
            # print("hello jcut")

            windows.open_milling_window_v2(MillingPattern.Thin)
            print("hello thin")

            self.update_image_settings(hfw=50.0e-6)
            windows.open_milling_window_v2(MillingPattern.Polish)
            print("hello polish")

        if TEST_DETECTION_WINDOW:

            self.current_sample_position = SamplePosition(self.save_path, 9999)
            # self.current_sample_position.sample_id = "9999"

            from pprint import pprint

            det = calibration.validate_detection_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
            )
            pprint(det)

        if TEST_MOVEMENT_WINDOW:

            windows.ask_user_movement_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg_type="eucentric",
                flat_to_sem=True,
            )
            windows.ask_user_movement_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg_type="centre_eb",
                flat_to_sem=True,
            )
            windows.ask_user_movement_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg_type="centre_ib",
                flat_to_sem=True,
            )

        if TEST_USER_WINDOW:

            ret = windows.ask_user_interaction_v2(
                self.microscope,
                msg="Hello 1",
                beam_type=None,
            )
            print("RETURN: ", ret, self.response)


    ########################## HARDWARE ##########################

    def initialize_hardware(self, settings: dict, log_path: str, ip_address: str = "10.0.0.1" ) -> SdbMicroscopeClient:
        
        microscope = fibsem_utils.connect_to_microscope(ip_address=ip_address, parent_ui=self)

        if microscope:
            # run validation and show in ui
            validation.run_validation_ui(
                microscope=microscope,
                settings=settings,
                log_path=log_path,
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
        self.pushButton_autoliftout.setEnabled(False)  # disable unless sample positions are loaded.
        self.pushButton_thinning.setEnabled(False)  # disable unless sample positions are loaded

        # load data
        self.pushButton_add_sample_position.setVisible(False)

        # configuration management
        self.actionLoad_Experiment.triggered.connect(self.run_load_experiment_utility)
        self.actionLoad_Protocol.triggered.connect(self.load_protocol_from_file)

        # mode selection
        self.actionMark_Sample_Position_Failed.triggered.connect(self.mark_sample_position_failed)
        self.actionAutoLamella.triggered.connect(self.enable_autolamella)
        self.actionAutoLiftout.triggered.connect(self.enable_autoliftout)

        # utilities
        self.actionSharpen_Needle.triggered.connect(self.run_sharpen_needle_utility)
        self.actionSputter_Platinum.triggered.connect(self.run_sputter_platinum_utility)

        # TESTING METHODS TODO: TO BE REMOVED
        self.pushButton_test_popup.clicked.connect(lambda: self.testing_function())

        # if self.PIESCOPE_ENABLED:
        #     self.pushButton_add_sample_position.setVisible(False)
        #     self.pushButton_add_sample_position.clicked.connect(
        #         lambda: self.select_sample_positions_piescope(initialisation=False)
        #     )

        logging.info("gui: setup connections finished")

    def enable_autolamella(self):
        self.AUTOLAMELLA_ENABLED = True
        self.pushButton_autoliftout.setText("Run AutoLamella")
        self.label_title.setText("AutoLamella")
        self.pushButton_initialise.setText("Setup Autolamella")
        self.pushButton_thinning.setVisible(False)

        # connect autolamella workflow
        self.pushButton_autoliftout.disconnect()
        self.pushButton_autoliftout.clicked.connect(self.run_autolamella_workflow)

    def enable_autoliftout(self):
        self.AUTOLAMELLA_ENABLED = False
        self.pushButton_autoliftout.setText("Run AutoLiftout")
        self.label_title.setText("AutoLiftout")
        self.pushButton_initialise.setText("Setup AutoLiftout")
        self.pushButton_thinning.setVisible(True)

        # connect autoliftout workflow
        self.pushButton_autoliftout.disconnect()
        self.pushButton_autoliftout.clicked.connect(self.run_autoliftout)

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

        horizontalGroupBox = QGroupBox()  # TODO: move to title section
        gridLayout = draw_grid_layout(self.sample)

        #########
        # table options..
        # TODO: get current row
        # add images...
        # 
        # from PyQt5 import QtCore
        # ref https://learndataanalysis.org/display-pandas-dataframe-with-pyqt5-qtableview-widget/
        # class pandasModel(QtCore.QAbstractTableModel):

        #     def __init__(self, data):
        #         QtCore.QAbstractTableModel.__init__(self)
        #         self._data = data

        #     def rowCount(self, parent=None):
        #         return self._data.shape[0]

        #     def columnCount(self, parnet=None):
        #         return self._data.shape[1]

        #     def data(self, index, role=QtCore.Qt.DisplayRole):
        #         if index.isValid():
        #             if role == QtCore.Qt.DisplayRole:
        #                 return str(self._data.iloc[index.row(), index.column()])
        #         return None

        #     def headerData(self, col, orientation, role):
        #         if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
        #             return self._data.columns[col]
        #         return None

        # from liftout.fibsem.sample import sample_to_dataframe
        # df = sample_to_dataframe(self.sample)
        # model = pandasModel(df)
        # view = QtWidgets.QTableView()
        # view.setModel(model)

        # view.horizontalHeader().setStretchLastSection(True)
        # view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # gridLayout = QtWidgets.QGridLayout()
        # gridLayout.addWidget(view, 1, 0)
        ####################

        horizontalGroupBox.setLayout(gridLayout)
        horizontalGroupBox.update()
        self.scroll_area.setWidget(horizontalGroupBox)
        self.scroll_area.update()


    def update_status(self, lamella: Lamella = None):
        """Update status information
        """

        # TODO: get current lamella?
        if lamella is None:
            lamella = Lamella("test", 999)
            import random
            lamella.current_state.stage   = random.choice([stage for stage in AutoLiftoutStage])     
        ui_utils.update_stage_label(self.label_stage, lamella)

        # log info
        with open(self.sample.log_path) as f:
            lines = f.read().splitlines()
            log_line = "\n".join(lines[-3:])  # last log msg
            log_msg = log_line.split("—")[-1].strip()
            self.statusBar.showMessage(log_msg)
            self.statusBar.repaint()


    def update_ui_display(self):
        try:
            self.update_status()
            self.update_scroll_ui()
        except Exception as e:
            print(f"Exception updating ui: {e}")
        return
    


    ########################## PIESCOPE ##########################

    
    def select_sample_positions_piescope(self, initialisation=False):
        import piescope_gui.main

        if initialisation:
            # get the current sample positions..
            select_another_sample_position = self.get_current_sample_positions()

            if select_another_sample_position is False:
                return

            if self.piescope_gui_main_window is None:
                self.piescope_gui_main_window = piescope_gui.main.GUIMainWindow(
                    parent_gui=self
                )
                self.piescope_gui_main_window.window_close.connect(
                    lambda: self.finish_select_sample_positions_piescope()
                )

        if self.piescope_gui_main_window:
            # continue selecting points
            self.piescope_gui_main_window.milling_position = None
            self.piescope_gui_main_window.show()


    def get_initial_lamella_position_piescope(self):
        """Select the initial sample positions for liftout"""
        sample_position = SamplePosition(
            data_path=self.save_path, sample_no=self.sample_no
        )

        movement.safe_absolute_stage_movement(
            self.microscope, self.piescope_gui_main_window.milling_position
        )

        # save lamella coordinates
        sample_position.lamella_coordinates = StagePosition(
            x=float(self.piescope_gui_main_window.milling_position.x),
            y=float(self.piescope_gui_main_window.milling_position.y),
            z=float(self.piescope_gui_main_window.milling_position.z),
            r=float(self.piescope_gui_main_window.milling_position.r),
            t=float(self.piescope_gui_main_window.milling_position.t),
            coordinate_system=str(
                self.piescope_gui_main_window.milling_position.coordinate_system
            ),
        )
        # save microscope state
        sample_position.microscope_state = calibration.get_current_microscope_state(
            microscope=self.microscope, stage=self.current_stage, eucentric=True
        )
        sample_position.save_data()

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_med_res"],
            save=True,
            save_path=os.path.join(self.save_path, str(sample_position.sample_id)),
            label=f"ref_lamella_low_res",
        )
        acquire.take_reference_images(
            self.microscope, image_settings=self.image_settings
        )

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
            save=True,
            save_path=os.path.join(self.save_path, str(sample_position.sample_id)),
            label="ref_lamella_high_res",
        )
        acquire.take_reference_images(
            self.microscope, image_settings=self.image_settings
        )

        return sample_position

    def finish_select_sample_positions_piescope(self):

        try:
            self.piescope_gui_main_window.milling_window.hide()
            self.piescope_gui_main_window.hide()
            time.sleep(1)
        except:
            logging.error("Unable to close the PIEScope windows?")

        if self.piescope_gui_main_window.milling_position is not None:
            # get the lamella milling position from piescope...
            sample_position = self.get_initial_lamella_position_piescope()
            self.samples.append(sample_position)
            self.sample_no += 1

        finished_selecting = windows.ask_user_interaction_v2(
            self.microscope,
            msg=f"Do you want to select landing positions?\n"
            f"{len(self.samples)} positions selected so far.",
        )

        self.update_scroll_ui()

        # enable adding more samples with piescope
        if self.samples:
            self.pushButton_add_sample_position.setVisible(True)
            self.pushButton_add_sample_position.setEnabled(True)

        if finished_selecting:

            # only select landing positions for liftout
            if not self.AUTOLAMELLA_ENABLED:
                self.select_landing_positions()

            self.finish_setup()


    ########################## AUTOLAMELLA ##########################

    def run_autolamella_workflow(self):
        """Run the autolamella workflow"""

        logging.info(
            f"AutoLamella Workflow started for {len(self.samples)} sample positions."
        )

        for next_stage in [
            AutoLiftoutStage.MillTrench,
            AutoLiftoutStage.Thinning,
            AutoLiftoutStage.Polishing,
        ]:
            for sp in self.samples:
                self.current_sample_position = sp

                msg = (
                    f"The last completed stage for sample position {sp.sample_no} ({sp.petname}) \nis {sp.microscope_state.last_completed_stage.name}. "
                    f"\nWould you like to continue from {next_stage.name}?\n"
                )
                response = windows.ask_user_interaction_v2(
                    self.microscope,
                    msg=msg,
                    beam_type=BeamType.ION,
                )

                if response:

                    # reset to the previous state
                    self.start_of_stage_update(next_stage=next_stage)

                    # run the next workflow stage
                    self.autolamella_stages[next_stage]()

                    # advance workflow
                    self.end_of_stage_update(eucentric=True)
                else:
                    break  # go to the next sample
        # TODO: maybe move polishing outside?

    def mill_autolamella(self):
        """Milling stage for autolamella"""

        # tilt and rotate to the correct position... TODO: also do this when selecting?
        autolamella_position = StagePosition(r=np.deg2rad(50), t=np.deg2rad(7))
        movement.safe_absolute_stage_movement(self.microscope, autolamella_position)

        # TODO: reference images...
        windows.ask_user_movement_v2(
            self.microscope, self.settings, self.image_settings, msg_type="eucentric"
        )

        windows.open_milling_window_v2(MillingPattern.Trench)

    def thin_autolamella(self):
        """Thinning stage for autolamella"""
        # move to correct angle ?

        self.update_image_settings(
            resolution=self.settings["protocol"]["thin_lamella"]["resolution"],
            dwell_time=self.settings["protocol"]["thin_lamella"]["dwell_time"],
            hfw=self.settings["protocol"]["thin_lamella"]["hfw"],
        )
        windows.ask_user_movement_v2(
            self.microscope, self.settings, self.image_settings, msg_type="centre_ib"
        )

        windows.open_milling_window_v2(MillingPattern.Thin)

    def polish_autolamella(self):
        """Polishing Stage for autolamella"""

        windows.ask_user_movement_v2(
            self.microscope, self.settings, self.image_settings, msg_type="centre_ib"
        )

        self.update_image_settings(
            resolution=self.settings["protocol"]["polish_lamella"]["resolution"],
            dwell_time=self.settings["protocol"]["polish_lamella"]["dwell_time"],
            hfw=self.settings["protocol"]["polish_lamella"]["hfw"],
        )
        windows.open_milling_window_v2(MillingPattern.Polish)












def main():
    """Launch the `autoliftout` main application window."""
    app = QtWidgets.QApplication([])
    qt_app = GUIMainWindow()
    app.aboutToQuit.connect(qt_app.disconnect)  # cleanup & teardown
    qt_app.showNormal()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
