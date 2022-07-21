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
from liftout.detection.utils import DetectionType
from liftout.fibsem import acquire, calibration, milling, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem import validation
from liftout.fibsem.acquire import ImageSettings
from liftout.fibsem.sample import Sample, AutoLiftoutStage
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

        # start initialisation
        self.current_stage = AutoLiftoutStage.Initialisation

        # load config
        self.settings = utils.load_config_v2()

        self.setupUi(self)
        self.set_ui_style()
        self.showNormal()

        # load experiment
        self.setup_experiment()

        # return
        logging.info(f"INIT | {self.current_stage.name} | STARTED")

        # initialise hardware
        # self.microscope = self.initialize_hardware(
        #     ip_address=self.settings["system"]["ip_address"]
        # )

        # # offline
        # self.MICROSCOPE_CONNECTED = bool(self.microscope)

        # set different modes
        self.HIGH_THROUGHPUT = bool(self.settings["system"]["high_throughput"])
        self.AUTOLAMELLA_ENABLED = bool(self.settings["system"]["autolamella"])
        self.PIESCOPE_ENABLED = bool(self.settings["system"]["piescope_enabled"])

        # initialise piescope
        if self.PIESCOPE_ENABLED:
            self.piescope_gui_main_window = None

        # setup connections
        self.setup_connections()

        pprint(self.settings)

        return

        # initial image settings
        self.update_image_settings()

        if self.microscope:

            self.stage = self.microscope.specimen.stage
            self.needle = self.microscope.specimen.manipulator
            self.current_sample_position = None

            # run validation and show in ui
            validation.run_validation_ui(
                microscope=self.microscope,
                settings=self.settings,
                log_path=self.log_path,
            )

        # setup status information
        self.update_status()

        # save the metadata
        utils.save_metadata(self.settings, self.save_path)

        # enable liftout if samples are loaded
        if self.samples:
            self.pushButton_autoliftout.setEnabled(True)
            self.pushButton_thinning.setEnabled(True)

        # autoliftout_workflow
        self.autoliftout_stages = {
            AutoLiftoutStage.Setup: self.setup_autoliftout,
            AutoLiftoutStage.MillTrench: self.mill_lamella_trench,
            AutoLiftoutStage.MillJCut: self.mill_lamella_jcut,
            AutoLiftoutStage.Liftout: self.liftout_lamella,
            AutoLiftoutStage.Landing: self.land_lamella,
            AutoLiftoutStage.Reset: self.reset_needle,
            AutoLiftoutStage.Thinning: self.thin_lamella,
            AutoLiftoutStage.Polishing: self.polish_lamella,
        }

        # autolamella workflow
        self.autolamella_stages = {  # TODO: need to create separate mode / stages for these
            AutoLiftoutStage.Setup: self.setup_autoliftout,
            AutoLiftoutStage.MillTrench: self.mill_autolamella,
            AutoLiftoutStage.Thinning: self.thin_autolamella,
            AutoLiftoutStage.Polishing: self.polish_autolamella,
        }

        # Set up scroll area for display
        self.update_scroll_ui()

        logging.info(f"INIT | {self.current_stage.name} | FINISHED")

    



    ########################## SETUP ##########################

    def get_current_sample_positions(self):
        # check if samples already has been loaded, and then append from there
        self.current_sample_position = None  # reset the current sample
        if self.samples:
            select_another_sample_position = windows.ask_user_interaction_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg=f"Do you want to select another lamella position?\n"
                f"{len(self.samples)} positions selected so far.",
                beam_type=BeamType.ELECTRON,
            )
            self.sample_no = (
                max([sample_position.sample_no for sample_position in self.samples]) + 1
            )

        else:
            # select the initial positions to mill lamella
            select_another_sample_position = True
            self.samples = []
            self.sample_no = 1

        return select_another_sample_position


    def select_sample_positions(self):

        select_another_sample_position = self.get_current_sample_positions()

        # allow the user to select additional lamella positions
        eucentric_calibration = False
        while select_another_sample_position:
            sample_position = self.select_initial_lamella_positions(
                sample_no=self.sample_no, eucentric_calibration=eucentric_calibration
            )
            self.samples.append(sample_position)
            self.sample_no += 1

            eucentric_calibration = True

            select_another_sample_position = windows.ask_user_interaction_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg=f"Do you want to select another lamella position?\n"
                f"{len(self.samples)} positions selected so far.",
                beam_type=BeamType.ION,
            )
            self.update_scroll_ui()

        # select landing positions
        if not self.AUTOLAMELLA_ENABLED:
            self.select_landing_positions()

        # finish setup
        self.finish_setup()

    def select_landing_positions(self):
        """Select landing positions for autoliftout"""

        ####################################
        # # move to landing grid
        movement.move_to_landing_grid(
            self.microscope, settings=self.settings, flat_to_sem=False
        )
        # movement.auto_link_stage(self.microscope, hfw=900e-6)
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_low_res"]
        )
        windows.ask_user_movement_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            msg_type="eucentric",
            flat_to_sem=False,
        )
        ####################################

        # select corresponding sample landing positions
        for current_sample_position in self.samples:

            # check if landing position already selected? so it doesnt overwrite
            if current_sample_position.landing_selected is False:
                self.select_landing_sample_positions(current_sample_position)

    def finish_setup(self):
        """Finish the setup stage for autolifout/autolamella"""
        # load all the data from disk (to load images)
        for sample_position in self.samples:
            sample_position.load_data_from_file()

        self.update_scroll_ui()

        # reset microscope coordinate system
        self.microscope.specimen.stage.set_default_coordinate_system(
            CoordinateSystem.SPECIMEN
        )

        logging.info(
            f"Selected {len(self.samples)} initial sample and landing positions."
        )

        if self.samples and self.PIESCOPE_ENABLED:
            self.pushButton_autoliftout.setEnabled(True)
            self.pushButton_thinning.setEnabled(True)

            logging.info(
                f"{len(self.samples)} samples selected and saved to {self.save_path}."
            )
            logging.info(f"INIT | {self.current_stage.name} | FINISHED")



    def select_initial_lamella_positions(
        self, sample_no, eucentric_calibration: bool = False
    ):
        """Select the initial sample positions for liftout"""
        sample_position = SamplePosition(data_path=self.save_path, sample_no=sample_no)

        if eucentric_calibration is False:
            movement.move_to_sample_grid(self.microscope, settings=self.settings)
            movement.auto_link_stage(self.microscope)

            windows.ask_user_movement_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg_type="eucentric",
                flat_to_sem=True,
            )
            movement.move_to_trenching_angle(self.microscope, settings=self.settings)

        sample_position.lamella_coordinates = self.user_select_feature(
            feature_type="lamella"
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

    def select_landing_sample_positions(self, current_sample_position: SamplePosition):
        logging.info(f"Selecting Landing Position: {current_sample_position.sample_id}")

        # select landing coordinates
        current_sample_position.landing_coordinates = self.user_select_feature(
            feature_type="landing"
        )

        # mill the landing edge flat
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            beam_type=BeamType.ION,
            save=False,
        )
        windows.open_milling_window_v2(MillingPattern.Flatten)

        current_sample_position.landing_selected = True

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_med_res"],
            save=True,
            save_path=os.path.join(
                self.save_path, str(current_sample_position.sample_id)
            ),
            label="ref_landing_low_res",
        )
        acquire.take_reference_images(
            self.microscope, image_settings=self.image_settings
        )

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            save=True,
            save_path=os.path.join(
                self.save_path, str(current_sample_position.sample_id)
            ),
            label="ref_landing_high_res",
        )
        acquire.take_reference_images(
            self.microscope, image_settings=self.image_settings
        )

        # save coordinates
        current_sample_position.save_data()





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
            self.settings,
            self.image_settings,
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
                    self.settings,
                    self.image_settings,
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


    ########################## AUTOLIFTOUT ##########################

    def setup_autoliftout(self):

        self.current_stage = AutoLiftoutStage.Setup
        logging.info(f"INIT | {self.current_stage.name} | STARTED")

        # initial image settings
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_low_res"],
            beam_type=BeamType.ELECTRON,
            save=True,
            label="grid",
        )

        # move to the initial sample grid position
        movement.move_to_sample_grid(self.microscope, self.settings)

        # NOTE: can't take ion beam image with such a high hfw, will default down to max ion beam hfw
        acquire.new_image(self.microscope, self.image_settings)

        # sputter platinum to protect grid and prevent charging...
        fibsem_utils.sputter_platinum_on_whole_sample_grid_v2(
            self.microscope, self.settings, self.image_settings
        )

        # select initial lamella and landing points
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_low_res"],
            beam_type=BeamType.ELECTRON,
            save=True,
            label="centre_grid",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        # check if focus is good enough
        if (
            self.microscope.beams.electron_beam.working_distance.value
            >= self.settings["calibration"]["limits"]["max_working_distance_eb"]
        ):
            windows.ask_user_interaction_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg="The working distance seems to be incorrect, please manually fix the focus. \nPress Yes to continue.",
                beam_type=BeamType.ELECTRON,
            )

        # select initial lamella positions
        if self.PIESCOPE_ENABLED:
            self.select_sample_positions_piescope(initialisation=True)
        else:
            # select the positions
            self.select_sample_positions()

            # enable autoliftout workflow
            self.pushButton_autoliftout.setEnabled(True)
            self.pushButton_thinning.setEnabled(True)
            self.microscope.specimen.stage.set_default_coordinate_system(
                CoordinateSystem.SPECIMEN
            )

            logging.info(
                f"{len(self.samples)} samples selected and saved to {self.save_path}."
            )
            logging.info(f"INIT | {self.current_stage.name} | FINISHED")


    def run_autoliftout_workflow(self):

        logging.info(
            f"AutoLiftout Workflow started for {len(self.samples)} sample positions."
        )

        # high throughput workflow
        if self.HIGH_THROUGHPUT:
            for terminal_stage in [
                AutoLiftoutStage.MillTrench,
                AutoLiftoutStage.MillJCut,
            ]:
                for sp in self.samples:
                    self.current_sample_position = sp

                    while (
                        sp.microscope_state.last_completed_stage.value
                        < terminal_stage.value
                    ):

                        next_stage = AutoLiftoutStage(
                            sp.microscope_state.last_completed_stage.value + 1
                        )

                        # reset to the previous state
                        self.start_of_stage_update(next_stage=next_stage)

                        # run the next workflow stage
                        self.autoliftout_stages[next_stage]()

                        # advance workflow
                        self.end_of_stage_update(eucentric=True)

        # standard workflow
        for sp in self.samples:
            self.current_sample_position = sp

            while (
                sp.microscope_state.last_completed_stage.value
                < AutoLiftoutStage.Reset.value
            ):

                next_stage = AutoLiftoutStage(
                    sp.microscope_state.last_completed_stage.value + 1
                )
                msg = (
                    f"The last completed stage for sample position {sp.sample_no} ({sp.petname}) \nis {sp.microscope_state.last_completed_stage.name}. "
                    f"\nWould you like to continue from {next_stage.name}?\n"
                )
                response = windows.ask_user_interaction_v2(
                    self.microscope,
                    self.settings,
                    self.image_settings,
                    msg=msg,
                    beam_type=BeamType.ION,
                )

                if response:

                    # reset to the previous state
                    self.start_of_stage_update(next_stage=next_stage)

                    # run the next workflow stage
                    self.autoliftout_stages[next_stage]()

                    # advance workflow
                    self.end_of_stage_update(eucentric=True)
                else:
                    break  # go to the next sample

    def run_thinning_workflow(self):

        # thinning
        for sp in self.samples:
            self.current_sample_position = sp
            if (
                self.current_sample_position.microscope_state.last_completed_stage
                == AutoLiftoutStage.Reset
            ):
                self.start_of_stage_update(next_stage=AutoLiftoutStage.Thinning)
                self.thin_lamella()
                self.end_of_stage_update(eucentric=True)

        # polish
        for sp in self.samples:
            self.current_sample_position = sp
            if (
                self.current_sample_position.microscope_state.last_completed_stage
                == AutoLiftoutStage.Thinning
            ):
                self.start_of_stage_update(next_stage=AutoLiftoutStage.Polishing)
                self.polish_lamella()
                self.end_of_stage_update(eucentric=True)

        # finish the experiment
        self.current_stage = AutoLiftoutStage.Finished
        for sp in self.samples:
            self.current_sample_position = sp
            if (
                self.current_sample_position.microscope_state.last_completed_stage
                == AutoLiftoutStage.Polishing
            ):
                self.end_of_stage_update(eucentric=True)

    def end_of_stage_update(self, eucentric: bool) -> None:
        """Save the current microscope state configuration to disk, and log that the stage has been completed."""
        # save state information
        microscope_state = calibration.get_current_microscope_state(
            microscope=self.microscope, stage=self.current_stage, eucentric=eucentric
        )
        self.current_sample_position.microscope_state = microscope_state
        self.current_sample_position.save_data()

        # update ui
        self.update_scroll_ui()

        logging.info(
            f"{self.current_sample_position.sample_id} | {self.current_stage.name} | FINISHED"
        )

        return

    def start_of_stage_update(self, next_stage: AutoLiftoutStage) -> None:
        """Check the last completed stage and reload the microscope state if required. Log that the stage has started. """
        last_completed_stage = (
            self.current_sample_position.microscope_state.last_completed_stage
        )

        if last_completed_stage.value == next_stage.value - 1:
            logging.info(
                f"{self.current_sample_position.sample_id} restarting from end of stage: {last_completed_stage.name}"
            )
            calibration.set_microscope_state(
                self.microscope, self.current_sample_position.microscope_state
            )

        self.current_stage = next_stage
        logging.info(
            f"{self.current_sample_position.sample_id} | {self.current_stage.name}  | STARTED"
        )

        self.update_status()

        return

    def mill_lamella_trench(self):

        # move flat to the ion beam, stage tilt 25 (total image tilt 52)
        movement.move_to_trenching_angle(self.microscope, self.settings)

        # Take an ion beam image at the *milling current*
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"]
        )
        windows.ask_user_movement_v2(
            self.microscope, self.settings, self.image_settings, msg_type="centre_ib"
        )

        # update the lamella coordinates, and save
        self.microscope.specimen.stage.set_default_coordinate_system(
            CoordinateSystem.RAW
        )
        self.current_sample_position.lamella_coordinates = (
            self.microscope.specimen.stage.current_position
        )
        self.current_sample_position.save_data()
        self.microscope.specimen.stage.set_default_coordinate_system(
            CoordinateSystem.SPECIMEN
        )

        # MILL_TRENCHES
        windows.open_milling_window_v2(MillingPattern.Trench)

        # reference images of milled trenches
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_med_res"],
            save=True,
            label=f"ref_trench_low_res",
        )
        acquire.take_reference_images(
            self.microscope, image_settings=self.image_settings
        )

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
            save=True,
            label=f"ref_trench_high_res",
        )
        acquire.take_reference_images(
            self.microscope, image_settings=self.image_settings
        )

    def mill_lamella_jcut(self):

        ####################################### JCUT #######################################

        # load the reference images
        reference_images_low_and_high_res = []
        for fname in [
            "ref_trench_low_res_eb",
            "ref_trench_high_res_eb",
            "ref_trench_low_res_ib",
            "ref_trench_high_res_ib",
        ]:

            img = self.current_sample_position.load_reference_image(fname)
            reference_images_low_and_high_res.append(img)

        # move flat to electron beam
        movement.flat_to_beam(
            self.microscope, self.settings, beam_type=BeamType.ELECTRON
        )

        # make sure drift hasn't been too much since milling trenches
        # first using reference images
        ret = calibration.correct_stage_drift(
            self.microscope,
            self.image_settings,
            reference_images_low_and_high_res,
            mode="ib",
        )

        if ret is False:
            windows.ask_user_movement_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg_type="centre_eb",
            )

            # # cross-correlation has failed, manual correction required
            logging.info(
                f"{self.current_stage.name}: cross-correlation manually corrected"
            )

        self.update_image_settings(
            hfw=self.settings["calibration"]["drift_correction_hfw_highres"],
            save=True,
            label=f"drift_correction_ML",
        )

        # then using ML, tilting/correcting in steps so drift isn't too large
        stage_settings = MoveSettings(rotate_compucentric=True)
        movement.move_relative(
            self.microscope,
            t=np.deg2rad(self.settings["jcut"]["jcut_angle"]),
            settings=stage_settings,
        )
        self.update_image_settings(
            hfw=self.settings["calibration"]["drift_correction_hfw_highres"],
            save=True,
            label=f"drift_correction_ML",
        )
        calibration.correct_stage_drift_with_ML_v2(
            self.microscope, self.settings, self.image_settings
        )

        ## MILL_JCUT
        # now we are at the angle for jcut, perform jcut
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"]
        )
        windows.open_milling_window_v2(MillingPattern.JCut)

        # take reference images of the jcut
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_med_res"],
            save=True,
            label=f"jcut_lowres",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_ultra_res"],
            save=True,
            label=f"jcut_highres",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

    def liftout_lamella(self):

        # get ready to do liftout by moving to liftout angle
        movement.move_to_liftout_angle(self.microscope, self.settings)

        # check eucentric height
        windows.ask_user_movement_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            msg_type="eucentric",
            flat_to_sem=True,
        )  # liftout angle is flat to SEM

        # check focus distance is within tolerance
        movement.auto_link_stage(self.microscope)  # TODO: remove?

        if not calibration.check_working_distance_is_within_tolerance(
            self.microscope, settings=self.settings, beam_type=BeamType.ELECTRON
        ):
            logging.warning("Autofocus has failed")
            windows.ask_user_interaction_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg="The AutoFocus routine has failed, please correct the focus manually.",
                beam_type=BeamType.ELECTRON,
            )

        # correct stage drift from mill_lamella stage
        calibration.correct_stage_drift_with_ML_v2(
            self.microscope, self.settings, self.image_settings
        )

        # move needle to liftout start position
        if (
            self.stage.current_position.z
            < self.settings["calibration"]["limits"]["stage_height_limit"]
        ):  # 3.7e-3
            # [FIX] autofocus cannot be relied upon, if this condition is met, we need to stop.

            # movement.auto_link_stage(self.microscope) # This is too unreliable to fix the miscalibration
            logging.warning(f"Calibration error detected: stage position height")
            logging.warning(f"Stage Position: {self.stage.current_position}")

            windows.ask_user_interaction_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg="The system has identified the distance between the sample and the pole piece is less than 3.7mm. "
                "The needle will contact the sample, and it is unsafe to insert the needle. "
                "\nPlease manually refocus and link the stage, then press OK to continue. ",
                beam_type=BeamType.ELECTRON,
            )

        park_position = movement.move_needle_to_liftout_position(self.microscope)
        logging.info(
            f"{self.current_stage.name}: needle inserted to park positon: {park_position}"
        )

        # land needle on lamella
        self.land_needle_on_milled_lamella()

        # sputter platinum
        fibsem_utils.sputter_platinum(self.microscope, self.settings, whole_grid=False)
        logging.info(f"{self.current_stage.name}: lamella to needle welding complete.")

        self.update_image_settings(
            save=True,
            hfw=self.settings["platinum"]["weld"]["hfw"],
            label=f"needle_landed_Pt_sputter",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        # jcut sever pattern
        windows.open_milling_window_v2(MillingPattern.Sever)

        self.update_image_settings(
            save=True,
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
            label=f"jcut_sever",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        # Raise needle 30um from trench
        logging.info(f"{self.current_stage.name}: start removing needle from trench")
        for i in range(3):
            z_move_out_from_trench = movement.z_corrected_needle_movement(
                10e-6, self.stage.current_position.t
            )
            self.needle.relative_move(z_move_out_from_trench)
            self.image_settings.label = f"liftout_trench_{i}"
            acquire.take_reference_images(self.microscope, self.image_settings)
            logging.info(
                f"{self.current_stage.name}: removing needle from trench at {z_move_out_from_trench} ({i + 1}/3)"
            )
            time.sleep(1)

        # reference images after liftout complete
        self.image_settings.label = f"liftout_of_trench"
        acquire.take_reference_images(self.microscope, self.image_settings)

        # move needle to park position
        movement.retract_needle(self.microscope, park_position)

    def land_needle_on_milled_lamella(self):

        needle = self.microscope.specimen.manipulator

        ### REFERENCE IMAGES
        # low res
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            save=True,
            label=f"needle_liftout_start_position_lowres",
        )

        det = calibration.validate_detection_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
            beam_type=BeamType.ELECTRON,
        )

        x_move = movement.x_corrected_needle_movement(
            det.distance_metres.x, stage_tilt=self.stage.current_position.t
        )
        yz_move = movement.y_corrected_needle_movement(
            det.distance_metres.y, stage_tilt=self.stage.current_position.t
        )
        needle.relative_move(x_move)
        needle.relative_move(yz_move)
        logging.info(f"{self.current_stage.name}: needle x-move: {x_move}")
        logging.info(f"{self.current_stage.name}: needle yz-move: {yz_move}")

        ###

        ### Z-HALF MOVE (ION)
        # calculate shift between lamella centre and needle tip in the ion view
        self.image_settings.label = f"needle_liftout_post_xy_movement_lowres"
        det = calibration.validate_detection_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
            beam_type=BeamType.ION,
        )

        # calculate shift in xyz coordinates
        z_distance = det.distance_metres.y / np.cos(self.stage.current_position.t)

        # Calculate movement
        zy_move_half = movement.z_corrected_needle_movement(
            -z_distance / 2, self.stage.current_position.t
        )
        needle.relative_move(zy_move_half)
        logging.info(f"{self.current_stage.name}: needle z-half-move: {zy_move_half}")

        ###

        # repeat the final movement until user confirms.
        response = False
        while response is False:

            ### Z-MOVE FINAL (ION)
            self.image_settings.hfw = self.settings["calibration"]["reference_images"][
                "hfw_super_res"
            ]
            self.image_settings.label = f"needle_liftout_post_z_half_movement_highres"
            det = calibration.validate_detection_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
                beam_type=BeamType.ION,
            )

            # calculate shift in xyz coordinates
            z_distance = det.distance_metres.y / np.cos(self.stage.current_position.t)

            # move in x
            x_move = movement.x_corrected_needle_movement(det.distance_metres.x)
            self.needle.relative_move(x_move)

            # move in z
            # detection is based on centre of lamella, we want to land near the edge
            # gap = 0.2e-6 #lamella_height / 10
            zy_move_gap = movement.z_corrected_needle_movement(
                -z_distance, self.stage.current_position.t
            )
            self.needle.relative_move(zy_move_gap)

            logging.info(f"{self.current_stage.name}: needle x-move: {x_move}")
            logging.info(f"{self.current_stage.name}: needle zy-move: {zy_move_gap}")

            self.image_settings.save = False
            acquire.take_reference_images(self.microscope, self.image_settings)

            response = windows.ask_user_interaction_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg="Has the needle landed on the lamella? \nPress Yes to continue, or No to redo the final movement",
                beam_type=BeamType.ION,
            )

        # take final reference images
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            save=True,
            label=f"needle_liftout_landed_lowres",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
            save=True,
            label=f"needle_liftout_landed_highres",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

    def land_lamella(self):

        # move to landing coordinate
        movement.safe_absolute_stage_movement(
            microscope=self.microscope,
            stage_position=self.current_sample_position.landing_coordinates,
        )
        movement.auto_link_stage(
            self.microscope,
            hfw=self.settings["calibration"]["reference_images"]["hfw_med_res"],
        )

        # confirm eucentricity
        self.update_image_settings()
        windows.ask_user_movement_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            msg_type="eucentric",
            flat_to_sem=False,
        )

        # after eucentricity... we should be at 4mm,
        # so we should set wd to 4mm and link

        logging.info(
            f"{self.current_stage.name}: initial landing calibration complete."
        )

        ############################## LAND_LAMELLA ##############################
        park_position = movement.move_needle_to_landing_position(self.microscope)

        #### Y-MOVE (ELECTRON)
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            beam_type=BeamType.ELECTRON,
            save=True,
            label=f"landing_needle_land_sample_lowres",
        )

        det = calibration.validate_detection_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
            beam_type=BeamType.ELECTRON,
        )

        y_move = movement.y_corrected_needle_movement(
            det.distance_metres.y, self.stage.current_position.t
        )
        self.needle.relative_move(y_move)
        logging.info(f"{self.current_stage.name}: y-move complete: {y_move}")
        acquire.take_reference_images(self.microscope, self.image_settings)

        #### Z-MOVE (ION)
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            beam_type=BeamType.ION,
            save=True,
            label=f"landing_needle_land_sample_lowres_after_y_move",
        )
        det = calibration.validate_detection_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
            beam_type=BeamType.ION,
        )

        # up is down
        z_distance = -det.distance_metres.y / np.sin(
            np.deg2rad(self.settings["system"]["stage_tilt_flat_to_ion"])
        )
        z_move = movement.z_corrected_needle_movement(
            z_distance, self.stage.current_position.t
        )
        self.needle.relative_move(z_move)
        logging.info(f"{self.current_stage.name}: z-move complete: {z_move}")

        acquire.take_reference_images(self.microscope, self.image_settings)

        # TODO: change this to use ion view...
        #### X-HALF-MOVE (ELECTRON)
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            beam_type=BeamType.ELECTRON,
            save=True,
            label=f"landing_needle_land_sample_lowres_after_z_move",
        )

        det = calibration.validate_detection_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
            beam_type=BeamType.ELECTRON,
        )

        # half move
        x_move = movement.x_corrected_needle_movement(det.distance_metres.x / 2)
        self.needle.relative_move(x_move)
        logging.info(f"{self.current_stage.name}: x-half-move complete: {x_move}")
        acquire.take_reference_images(self.microscope, self.image_settings)

        # repeat final movement until user confirms landing
        response = False
        while response is False:
            #### X-MOVE
            self.update_image_settings(
                hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
                beam_type=BeamType.ELECTRON,
                save=True,
                label=f"landing_needle_land_sample_lowres_after_z_move",
            )

            det = calibration.validate_detection_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
                beam_type=BeamType.ELECTRON,
            )

            x_move = movement.x_corrected_needle_movement(det.distance_metres.x)
            self.needle.relative_move(x_move)
            logging.info(f"{self.current_stage.name}: x-move complete: {x_move}")

            # final reference images
            self.update_image_settings(
                hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
                beam_type=BeamType.ELECTRON,
                save=True,
                label=f"landing_lamella_final_weld_highres",
            )
            acquire.take_reference_images(
                microscope=self.microscope, image_settings=self.image_settings
            )

            response = windows.ask_user_interaction_v2(
                self.microscope,
                self.settings,
                self.image_settings,
                msg="Has the lamella landed on the post? \nPress Yes to continue, or No to redo the final movement",
                beam_type=BeamType.ION,
            )

        #################################################################################################

        ############################## WELD TO LANDING POST #############################################

        windows.open_milling_window_v2(MillingPattern.Weld)

        # final reference images
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
            save=True,
            label=f"landing_lamella_final_weld_highres",
        )
        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        #################################################################################################

        ###################################### CUT_OFF_NEEDLE ######################################

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            beam_type=BeamType.ION,
            save=True,
            label=f"landing_lamella_pre_cut_off",
        )

        det = calibration.validate_detection_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            shift_type=(DetectionType.NeedleTip, DetectionType.ImageCentre),
            beam_type=BeamType.ION,
        )

        # cut off needle
        windows.open_milling_window_v2(
            MillingPattern.Cut, x=det.distance_metres.x, y=det.distance_metres.y
        )

        #################################################################################################

        # reference images
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            beam_type=BeamType.ION,
            save=True,
            label=f"landing_lamella_final_cut_lowres",
        )
        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
            beam_type=BeamType.ION,
            save=True,
            label=f"landing_lamella_final_cut_highres",
        )
        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        logging.info(f"{self.current_stage.name}: removing needle from landing post")
        # move needle out of trench slowly at first
        for i in range(3):
            z_move_out_from_post = movement.z_corrected_needle_movement(
                10e-6, self.stage.current_position.t
            )
            self.needle.relative_move(z_move_out_from_post)
            logging.info(
                f"{self.current_stage.name}: moving needle out: {z_move_out_from_post} ({i + 1}/3)"
            )
            time.sleep(1)

        # move needle to park position
        movement.retract_needle(self.microscope, park_position)
        logging.info(f"{self.current_stage.name}: needle retracted.")

        # reference images
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            save=True,
            label=f"landing_lamella_final_lowres",
        )

        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            save=True,
            label=f"landing_lamella_final_highres",
        )
        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

    def reset_needle(self):

        # move sample stage out
        movement.move_sample_stage_out(self.microscope)
        logging.info(f"{self.current_stage.name}: moved sample stage out")

        ###################################### SHARPEN_NEEDLE ######################################

        # move needle in
        park_position = movement.insert_needle(self.microscope)
        z_move_in = movement.z_corrected_needle_movement(
            -180e-6, self.stage.current_position.t
        )
        self.needle.relative_move(z_move_in)
        logging.info(f"{self.current_stage.name}: insert needle for reset")

        # needle images
        self.update_image_settings(
            beam_type=BeamType.ION, save=True, label=f"sharpen_needle_initial",
        )
        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        det = calibration.validate_detection_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            shift_type=(DetectionType.NeedleTip, DetectionType.ImageCentre),
            beam_type=BeamType.ION,
        )

        x_move = movement.x_corrected_needle_movement(det.distance_metres.x)
        self.needle.relative_move(x_move)
        z_distance = -det.distance_metres.y / np.sin(
            np.deg2rad(self.settings["system"]["stage_tilt_flat_to_ion"])
        )
        z_move = movement.z_corrected_needle_movement(
            z_distance, self.stage.current_position.t
        )
        self.needle.relative_move(z_move)
        logging.info(
            f"{self.current_stage.name}: moving needle to centre: x_move: {x_move}, z_move: {z_move}"
        )

        self.image_settings.label = f"sharpen_needle_centre"
        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        det = calibration.validate_detection_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            shift_type=(DetectionType.NeedleTip, DetectionType.ImageCentre),
            beam_type=BeamType.ION,
        )

        # create sharpening patterns
        windows.open_milling_window_v2(
            MillingPattern.Sharpen, x=det.distance_metres.x, y=det.distance_metres.y
        )

        #################################################################################################

        # take reference images
        self.image_settings.label = f"sharpen_needle_final"
        self.image_settings.save = True
        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        # retract needle
        movement.retract_needle(self.microscope, park_position)

        # reset stage position
        stage_settings = MoveSettings(rotate_compucentric=True)
        self.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)
        # self.stage.absolute_move(StagePosition(r=lamella_coordinates.r))
        self.stage.absolute_move(StagePosition(x=0.0, y=0.0))

        # TODO: replace and test
        # stage_position = StagePosition(x=0, y=0, t=0)
        # movement.safe_absolute_stage_movement(self.microscope, stage_position)

    def thin_lamella(self):

        # move to the initial landing coordinates
        movement.safe_absolute_stage_movement(
            microscope=self.microscope,
            stage_position=self.current_sample_position.landing_coordinates,
        )

        # ensure_eucentricity # TODO: Maybe remove, not required?
        windows.ask_user_movement_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            msg_type="eucentric",
            flat_to_sem=False,
        )

        # rotate_and_tilt_to_thinning_angle
        self.image_settings.hfw = self.settings["calibration"]["imaging"][
            "horizontal_field_width"
        ]
        movement.move_to_thinning_angle(
            microscope=self.microscope, settings=self.settings
        )

        # ensure_eucentricity at thinning angle
        windows.ask_user_movement_v2(
            self.microscope,
            self.settings,
            self.image_settings,
            msg_type="eucentric",
            flat_to_sem=False,
        )

        # lamella images
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_med_res"],
            save=True,
            label=f"thin_lamella_0_deg_tilt",
        )

        acquire.take_reference_images(self.microscope, self.image_settings)

        # realign lamella to image centre
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            save=True,
            label=f"thin_drift_correction_medres",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["high_super_res"],
            save=False,
        )

        windows.ask_user_movement_v2(
            self.microscope, self.settings, self.image_settings, msg_type="alignment"
        )

        # take reference images
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
            save=True,
            label=f"thin_drift_correction_highres",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        # thin_lamella (align and mill)
        self.update_image_settings(
            resolution=self.settings["protocol"]["thin_lamella"]["resolution"],
            dwell_time=self.settings["protocol"]["thin_lamella"]["dwell_time"],
            hfw=self.settings["thin_lamella"]["hfw"],
        )
        # windows.open_milling_window_v2(MillingPattern.Fiducial)
        windows.open_milling_window_v2(MillingPattern.Thin)

        # take reference images
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"][
                "thinning_ref_img_hfw_superres"
            ],
            save=True,
            label=f"thin_lamella_post_superres",
        )

        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        return

    def polish_lamella(self):

        # restore state from thinning stage
        # ref_image = self.current_sample_position.load_reference_image("thin_lamella_crosscorrelation_ref_ib")

        # realign lamella to image centre
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"][
                "thinning_ref_img_hfw_medres"
            ],
            save=True,
            label=f"polish_drift_correction_medres",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"][
                "thinning_ref_img_hfw_highres"
            ],
            save=False,
        )

        windows.ask_user_movement_v2(
            self.microscope, self.settings, self.image_settings, msg_type="alignment"
        )

        # take reference images
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"][
                "thinning_ref_img_hfw_highres"
            ],
            save=True,
            label=f"polish_drift_correction_highres",
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        # polish (align and mill)
        self.update_image_settings(
            resolution=self.settings["protocol"]["polish_lamella"]["resolution"],
            dwell_time=self.settings["protocol"]["polish_lamella"]["dwell_time"],
            hfw=self.settings["protocol"]["polish_lamella"]["hfw"],
        )
        windows.open_milling_window_v2(MillingPattern.Polish)

        # take reference images
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_ultra_res"],
            save=True,
            label=f"polish_lamella_post_ultra_res",
        )

        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
            save=True,
            label=f"polish_lamella_post_super_res",
        )

        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_high_res"],
            save=True,
            label=f"polish_lamella_post_high_res",
        )

        acquire.take_reference_images(
            microscope=self.microscope, image_settings=self.image_settings
        )

        logging.info(
            f"{self.current_stage.name}: polish lamella {self.current_sample_position.sample_no} complete."
        )

        return


    ########################## UTILS ##########################

    
    def setup_experiment(self):

        # TODO: add a select folder option for new experiment

        try:
            sample = ui_utils.setup_experiment_sample_ui(parent=self)

        except Exception as e:
            ui_utils.display_error_message(message=f"Unable to setup sample: {e}")
            return

        # TODO: rename sample to experiment or vice versa?
        # class variables... TODO: remove...
        self.save_path = sample.path
        self.log_path = sample.log_path
        self.experiment_name = sample.name
        self.experiment_path = sample.path
        self.current_sample_position = None
        self.sample = sample

        logging.info(f"Experiment {self.sample.name} loaded.")
        logging.info(
            f"{len(self.sample.positions)} lamella loaded from {self.sample.path}"
        )

        # update the ui
        self.statusBar.showMessage(f"Experiment {self.sample.name} loaded.")
        self.statusBar.repaint()
        self.label_experiment_name.setText(f"Experiment: {self.sample.name}")
        # self.update_scroll_ui()
        self.update_status()

        return

    def user_select_feature(self, feature_type):
        """Get the user to centre the beam on the desired feature"""

        # ask user
        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_med_res"],
            save=False,
        )
        windows.ask_user_movement_v2(
            self.microscope, self.settings, self.image_settings, msg_type="centre_ib"
        )

        self.microscope.specimen.stage.set_default_coordinate_system(
            CoordinateSystem.RAW
        )

        return self.stage.current_position


    def mark_sample_position_failed(self):
        """Mark the indicated sample position as failed."""

        if self.samples:

            # show the user the sample number and petname
            sample_str = [
                f"Sample {sp.sample_no} ({sp.petname})" for sp in self.samples
            ]
            sp_name, okPressed = QInputDialog.getItem(
                self, "Select a Sample Position", "Sample No:", sample_str, 0, False
            )
            sp_idx = sample_str.index(sp_name)

            if okPressed:
                # mark sample as failure
                sp = self.samples[int(sp_idx)]
                sp.microscope_state.last_completed_stage = AutoLiftoutStage.Failure
                sp.save_data()
                logging.info(
                    f"Marked {sp_name} as {sp.microscope_state.last_completed_stage.name}"
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

    def testing_function(self):

        TEST_SAMPLE_POSITIONS = False
        TEST_PIESCOPE = False
        TEST_MILLING_WINDOW = False
        TEST_DETECTION_WINDOW = False
        TEST_MOVEMENT_WINDOW = False
        TEST_USER_WINDOW = False

        TEST_RESET_NEEDLE = True
        if TEST_RESET_NEEDLE:
            self.reset_needle()

        if TEST_SAMPLE_POSITIONS:
            self.select_sample_positions()

        if TEST_PIESCOPE:
            self.select_sample_positions_piescope(initialisation=False)

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
                self.settings,
                self.image_settings,
                msg="Hello 1",
                beam_type=None,
            )
            print("RETURN: ", ret, self.response)

    def update_image_settings(
        self,
        resolution=None,
        dwell_time=None,
        hfw=None,
        autocontrast=None,
        beam_type=None,
        gamma=None,
        save=None,
        label=None,
        save_path=None,
    ):
        """Update image settings. Uses default values if not supplied

        Args:
            resolution (str, optional): image resolution. Defaults to None.
            dwell_time (float, optional): image dwell time. Defaults to None.
            hfw (float, optional): image horizontal field width. Defaults to None.
            autocontrast (bool, optional): use autocontrast. Defaults to None.
            beam_type (BeamType, optional): beam type to image with (Electron, Ion). Defaults to None.
            gamma (GammaSettings, optional): gamma correction settings. Defaults to None.
            save (bool, optional): save the image. Defaults to None.
            label (str, optional): image filename . Defaults to None.
            save_path (Path, optional): directory to save image. Defaults to None.
        """
        gamma_settings = acquire.GammaSettings(
            enabled=self.settings["calibration"]["gamma"]["correction"],
            min_gamma=self.settings["calibration"]["gamma"]["min_gamma"],
            max_gamma=self.settings["calibration"]["gamma"]["max_gamma"],
            scale_factor=self.settings["calibration"]["gamma"]["scale_factor"],
            threshold=self.settings["calibration"]["gamma"]["threshold"],
        )

        self.image_settings = acquire.ImageSettings(
            resolution=self.settings["calibration"]["imaging"]["resolution"]
            if resolution is None
            else resolution,
            dwell_time=self.settings["calibration"]["imaging"]["dwell_time"]
            if dwell_time is None
            else dwell_time,
            hfw=self.settings["calibration"]["imaging"]["horizontal_field_width"]
            if hfw is None
            else hfw,
            autocontrast=self.settings["calibration"]["imaging"]["autocontrast"]
            if autocontrast is None
            else autocontrast,
            beam_type=BeamType.ELECTRON if beam_type is None else beam_type,
            gamma=gamma_settings if gamma is None else gamma,
            save=bool(self.settings["calibration"]["imaging"]["save"])
            if save is None
            else save,
            save_path=self.save_path if save_path is None else save_path,
            label=datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d.%H%M%S")
            if label is None
            else label,
        )

        # change the save path to the current sample if available
        if self.current_sample_position:
            self.image_settings.save_path = os.path.join(
                self.save_path, str(self.current_sample_position.sample_id)
            )

        logging.debug(f"Image Settings: {self.image_settings}")


    ########################## HARDWARE ##########################

    def connect_to_microscope(self, ip_address="10.0.0.1"):
        """Connect to the FIBSEM microscope."""
        try:
            microscope = fibsem_utils.initialise_fibsem(ip_address=ip_address)
        except Exception as e:
            ui_utils.display_error_message(
                f"AutoLiftout is unavailable. Unable to connect to microscope: {e}"
            )
            microscope = None

        return microscope

    def initialize_hardware(self, ip_address: str = "10.0.0.1"):

        return self.connect_to_microscope(ip_address=ip_address)

    def disconnect(self):
        logging.info("Running cleanup/teardown")
        logging.debug("Running cleanup/teardown")
        if self.microscope:
            self.microscope.disconnect()

    ########################## USER INTERFACE ##########################




    def setup_connections(self):
        logging.info("gui: setup connections started")
        # connect buttons
        self.pushButton_initialise.clicked.connect(lambda: self.setup_autoliftout())
        self.pushButton_autoliftout.clicked.connect(
            lambda: self.run_autoliftout_workflow()
        )
        self.pushButton_thinning.clicked.connect(lambda: self.run_thinning_workflow())
        self.pushButton_autoliftout.setEnabled(
            False
        )  # disable unless sample positions are loaded.
        self.pushButton_thinning.setEnabled(
            False
        )  # disable unless sample positions are loaded

        # load data
        self.pushButton_add_sample_position.setVisible(False)

        # configuration management
        self.actionLoad_Experiment.triggered.connect(self.setup_experiment)
        self.actionLoad_Protocol.triggered.connect(self.load_protocol_from_file)

        # mode selection
        self.actionMark_Sample_Position_Failed.triggered.connect(
            lambda: self.mark_sample_position_failed()
        )
        self.actionAutoLamella.triggered.connect(self.enable_autolamella)
        self.actionAutoLiftout.triggered.connect(self.enable_autoliftout)

        # utilities
        self.actionSharpen_Needle.triggered.connect(self.reset_needle)
        self.actionSputter_Platinum.triggered.connect(
            lambda: fibsem_utils.sputter_platinum_on_whole_sample_grid_v2(
                self.microscope, self.settings, self.image_settings
            )
        )

        # TESTING METHODS TODO: TO BE REMOVED
        self.pushButton_test_popup.clicked.connect(lambda: self.testing_function())

        if self.PIESCOPE_ENABLED:
            self.pushButton_add_sample_position.setVisible(False)
            self.pushButton_add_sample_position.clicked.connect(
                lambda: self.select_sample_positions_piescope(initialisation=False)
            )

        logging.info("gui: setup connections finished")

    def enable_autolamella(self):
        self.AUTOLAMELLA_ENABLED = True
        self.pushButton_autoliftout.setText("Run AutoLamella")
        self.label_title.setText("AutoLamella")
        self.pushButton_initialise.setText("Setup Autolamella")
        self.pushButton_thinning.setVisible(False)

        # connect autolamella workflow
        self.pushButton_autoliftout.disconnect()
        self.pushButton_autoliftout.clicked.connect(
            lambda: self.run_autolamella_workflow()
        )

    def enable_autoliftout(self):
        self.AUTOLAMELLA_ENABLED = False
        self.pushButton_autoliftout.setText("Run AutoLiftout")
        self.label_title.setText("AutoLiftout")
        self.pushButton_initialise.setText("Setup AutoLiftout")
        self.pushButton_thinning.setVisible(True)

        # connect autoliftout workflow
        self.pushButton_autoliftout.disconnect()
        self.pushButton_autoliftout.clicked.connect(
            lambda: self.run_autoliftout_workflow()
        )

    def set_ui_style(self):
        # UI style
        self.setWindowTitle("AutoLiftout")
        self.label_title.setStyleSheet(
            "font-family: Arial; font-weight: bold; font-size: 36px; border: 0px solid lightgray"
        )
        self.label_stage.setStyleSheet(
            "background-color: gray; padding: 10px; border-radius: 5px; color: white;"
        )
        self.label_stage.setFont(QtGui.QFont("Arial", 12, weight=QtGui.QFont.Bold))
        self.label_stage.setAlignment(QtCore.Qt.AlignCenter)
        self.label_stage.setText(f"{self.current_stage.name}")

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

    def update_scroll_ui(self):
        """Update the central ui grid with current sample data."""

        self.horizontalGroupBox = QGroupBox()  # TODO: move to title section
        gridLayout = draw_grid_layout(self.samples)
        self.horizontalGroupBox.setLayout(gridLayout)
        self.scroll_area.setWidget(self.horizontalGroupBox)
        self.horizontalGroupBox.update()
        self.scroll_area.update()




    def update_status(self):
        """Update status information
        """

        def update_stage_label(label: QtWidgets.QLabel, stage: AutoLiftoutStage):

            status_colors = {
                "Initialisation": "gray",
                "Setup": "gold",
                "MillTrench": "coral",
                "MillJCut": "coral",
                "Liftout": "seagreen",
                "Landing": "dodgerblue",
                "Reset": "salmon",
                "Thinning": "mediumpurple",
                "Polishing": "cyan",
                "Finished": "silver",
            }
            label.setText(f"{stage.name}")
            label.setStyleSheet(
                str(
                    f"background-color: {status_colors[stage.name]}; color: white; border-radius: 5px"
                )
            )

        update_stage_label(self.label_stage, self.current_stage)

        # log info
        with open(self.log_path) as f:
            lines = f.read().splitlines()
            log_line = "\n".join(lines[-3:])  # last log msg
            log_msg = log_line.split("—")[-1].strip()
            self.statusBar.showMessage(log_msg)
            self.statusBar.repaint()



def main():
    """Launch the `autoliftout` main application window."""
    app = QtWidgets.QApplication([])
    qt_app = GUIMainWindow()
    app.aboutToQuit.connect(qt_app.disconnect)  # cleanup & teardown
    qt_app.showNormal()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
