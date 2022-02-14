import datetime
import logging
import sys
import time
import traceback
import os
from enum import Enum


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import liftout
from liftout import utils
from liftout.detection import utils as detection_utils
from liftout.fibsem import acquire, calibration, milling, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.gui.qtdesigner_files import main as gui_main
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as _FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as _NavigationToolbar
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QLineEdit, QGroupBox, QGridLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from liftout.gui.DraggablePatch import DraggablePatch
matplotlib.use('Agg')

import scipy.ndimage as ndi
from autoscript_sdb_microscope_client.structures import *
from autoscript_sdb_microscope_client.enumerations import *
from liftout.fibsem.sampleposition import SamplePosition, AutoLiftoutStage
from PIL import Image

# Required to not break imports
BeamType = acquire.BeamType

# test_image = PIL.Image.open('C:/Users/David/images/mask_test.tif')
test_image = np.random.randint(0, 255, size=(1024, 1536), dtype='uint16')
test_image = np.array(test_image)
# test_image = np.zeros_like(test_image, dtype='uint16')
test_jcut = [(0.e-6, 200.e-6, 200.e-6, 30.e-6), (100.e-6, 175.e-6, 30.e-6, 100.e-6), (-100.e-6, 0.e-6, 30.e-6, 400.e-6)]


# conversions
MICRON_TO_METRE = 1e6
METRE_TO_MICRON = 1e-6

_translate = QtCore.QCoreApplication.translate

#
# class AutoLiftoutStage(Enum):
#     Initialisation = -1
#     Setup = 0
#     Milling = 1
#     Liftout = 2
#     Landing = 3
#     Reset = 4
#     Thinning = 5
#     Finished = 6


class GUIMainWindow(gui_main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, offline=False):
        super(GUIMainWindow, self).__init__()

        self.UI_LOADED = False

        # load experiment
        self.setup_experiment()

        # load config
        config_filename = os.path.join(os.path.dirname(liftout.__file__), "protocol_liftout.yml")
        self.settings = utils.load_config(config_filename)
        self.pretilt_degrees = self.settings["system"]["pretilt_angle"]
        assert self.pretilt_degrees == 27  # TODO: remove this once this has been cleaned up in other files

        self.current_stage = AutoLiftoutStage.Initialisation
        logging.info(f"{self.current_stage.name} STARTED")
        logging.info(f"gui: starting in {'offline' if offline else 'online'} mode")

        self.offline = offline
        self.setupUi(self)
        self.UI_LOADED = True # flag to check if ui has been loaded
        self.setWindowTitle('Autoliftout User Interface Main Window')
        self.popup_window = None
        self.popup_canvas = None
        self.raw_image = None
        self.overlay_image = None
        self.downscaled_image = None

        self.filter_strength = 3
        self.button_height = 50
        self.button_width = 100

        # initialise image frames and images
        self.image_SEM = None
        self.image_FIB = None
        # self.figure_SEM = None
        # self.figure_FIB = None
        # self.canvas_SEM = None
        # self.canvas_FIB = None
        # self.ax_SEM = None
        # self.ax_FIB = None
        # self.toolbar_SEM = None
        # self.toolbar_FIB = None

        # self.initialise_image_frames()

        # image frame interaction
        self.xclick = None
        self.yclick = None

        # initialise hardware
        self.ip_address = self.settings["system"]["ip_address"]
        self.microscope = None

        self.initialize_hardware(offline=offline)

        if self.microscope:
            self.microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
            self.stage = self.microscope.specimen.stage
            self.needle = self.microscope.specimen.manipulator

        self.current_sample_position = None

        # initial display
        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        self.update_display(beam_type=BeamType.ION, image_type='last')

        # popup initialisations
        self.popup_window = None
        self.new_image = None
        self.hfw_slider = None
        self.popup_settings = {'message': 'startup',
                               'allow_new_image': False,
                               'click': None,
                               'filter_strength': 0,
                               'crosshairs': True,
                               'milling_patterns': None}

        # initial image settings
        self.image_settings = {}
        self.USE_AUTOCONTRAST = self.settings["imaging"]["autocontrast"]
        self.update_image_settings()

        if self.microscope:
            self.microscope.beams.ion_beam.beam_current.value = self.settings["imaging"]["imaging_current"]

        self.current_stage = AutoLiftoutStage.Initialisation

        # setup status information
        self.status_timer = QtCore.QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)

        # setup status labels
        self.label_stage.setStyleSheet("background-color: coral; padding: 10px; border-radius: 5px")
        self.label_stage.setFont(QtGui.QFont("Arial", 12, weight=QtGui.QFont.Bold))
        self.label_stage.setAlignment(QtCore.Qt.AlignCenter)
        self.label_status.setStyleSheet("background-color: black;  color: white; padding:10px")
        self.update_status()

        self.setup_connections()

        if self.microscope:
            self.pre_run_validation()

        # enable liftout if samples are loaded
        if self.samples:
            self.pushButton_autoliftout.setEnabled(True)

        # DEVELOPER ONLY
        self.ADDITIONAL_CONFIRMATION = True
        self.MILLING_COMPLETED_THIS_RUN = False

        # Set up scroll area for display
        # ui setup part 2
        self.label_title.setStyleSheet("font-family: Arial; font-weight: bold; font-size: 24px")
        self.update_scroll_ui()

        logging.info(f"{self.current_stage.name} FINISHED")

    def update_scroll_ui(self):

        if self.UI_LOADED:
            self.draw_sample_grid_ui()
            self.scroll_area.setWidget(self.horizontalGroupBox)
            self.horizontalGroupBox.update()
            self.scroll_area.update()

    def pre_run_validation(self):
        logging.info(f"Running pre run validation")

        # TODO populate the validation checks
        validation_errors = []
        validation_checks = [
            "axes_homed",
            "beam_calibration",
            "needle_calibration",
            "link_and_focus",
            "stage_coordinate_system",
            "check_beam_shift_is_zero",
            "ion_beam_working_distance"   # 16.5mm
            "check_if_needle_is_inserted"

        ]
        import random
        for check in validation_checks:
            if random.random() > 0.5:
                validation_errors.append(check)

        # validate zero beamshift
        logging.info("BEAM SHIFT: SHOULD BE ZERO")
        logging.info(f"ELECTRON BEAM: {self.microscope.beams.electron_beam.beam_shift.value}")
        logging.info(f"ION BEAM: {self.microscope.beams.ion_beam.beam_shift.value}")

        # DOESNT WORK
        # self.microscope.beams.electron_beam.beam_shift.value = (0, 0)
        # self.microscope.beams.ion_beam.beam_shift.value = (0, 0)
        # logging.info(f"ELECTRON BEAM: {self.microscope.beams.electron_beam.beam_shift.value}")
        # logging.info(f"ION BEAM: {self.microscope.beams.ion_beam.beam_shift.value}")
        # logging.info(f"BEAM SHIFT RESET")


        # ion beam currents
        # 20e-12, 0.2e-9, 0.89e-9, 2.4e-9, 6.2e-9

        if validation_errors:
            logging.warning(f"validation_errors={validation_errors}")
        logging.info(f"Finished pre run validation: {len(validation_errors)} issues identified.")

    def initialise_autoliftout(self):

        self.current_stage = AutoLiftoutStage.Setup
        logging.info(f"{self.current_stage.name} STARTED")

        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["grid_ref_img_hfw_lowres"],
            beam_type=BeamType.ELECTRON,
            save=True,
            label="grid",
        )

        # check if focus is good enough
        eb_image = acquire.new_image(self.microscope, self.image_settings)
        if eb_image.metadata.optics.working_distance >= 6.0e-3:  # TODO: MAXIMUM_WORKING_DISTANCE
            self.update_popup_settings(message="The working distance seems to be incorrect, please manually fix the focus.", crosshairs=False)
            self.ask_user(image=None)

        # move to the initial sample grid position
        movement.move_to_sample_grid(self.microscope, self.settings)
        # TODO: do we need to link here?
        # movement.auto_link_stage(self.microscope)

        # NOTE: can't take ion beam image with such a high hfw
        acquire.new_image(self.microscope, self.image_settings)

        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        self.update_display(beam_type=BeamType.ION, image_type='last')

        # Whole-grid platinum deposition
        self.update_popup_settings(message='Do you want to sputter the whole sample grid with platinum?',
                                   crosshairs=False,
                                   filter_strength=self.filter_strength)
        self.ask_user(image=self.image_SEM)
        if self.response:
            fibsem_utils.sputter_platinum(self.microscope, self.settings, whole_grid=True)
            self.update_image_settings(hfw=self.settings["reference_images"]["grid_ref_img_hfw_lowres"], save=True, label="grid_Pt_deposition")
            self.update_display(beam_type=BeamType.ELECTRON, image_type='new')


        # select initial lamella and landing points
        # TODO: replace with PIESCOPE
        # TODO: MAGIC_NUMBER
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=900e-6,
            beam_type=BeamType.ELECTRON,
            save=True,
            label="centre_grid",
        )
        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.update_display(beam_type=BeamType.ION, image_type='new')




        #### TODO: ENABLE THIS BLOCK
        self.new_select_sample_positions()
        #####


        #### TODO: REMOVE THIS BLOCK
        # self.landing_coordinates, self.original_landing_images = self.select_initial_feature_coordinates(feature_type='landing')
        # self.lamella_coordinates, self.original_trench_images = self.select_initial_feature_coordinates(feature_type='lamella')
        # self.zipped_coordinates = list(zip(self.lamella_coordinates, self.landing_coordinates))
        #
        # # save coordinates
        # self.samples = []
        # for i, (lamella_coordinates, landing_coordinates) in enumerate(self.zipped_coordinates, 1):
        #     sample = SamplePosition(self.save_path, i)
        #     sample.lamella_coordinates = lamella_coordinates
        #     sample.landing_coordinates = landing_coordinates
        #     sample.save_data()
        #     self.samples.append(sample)

        ###############
        self.pushButton_autoliftout.setEnabled(True)
        self.microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)

        logging.info(f"{len(self.samples)} samples selected and saved to {self.save_path}.")
        logging.info(f"{self.current_stage.name} FINISHED")

    def new_select_sample_positions(self):


        # check if samples already has been loaded, and then append from there
        if self.samples:
            self.update_popup_settings(message=f'Do you want to select another lamella position?\n'
                                               f'{len(self.samples)} positions selected so far.', crosshairs=False)
            self.ask_user()
            sample_no = max([sample_position.sample_no for sample_position in self.samples]) + 1
            select_another_sample_position = self.response
        else:
            # select the initial positions to mill lamella
            select_another_sample_position = True
            self.samples = []
            sample_no = 0

        # allow the user to select additional lamella positions
        while select_another_sample_position:
            sample_position = self.select_initial_lamella_positions(sample_no=sample_no)
            self.samples.append(sample_position)
            sample_no += 1

            self.update_popup_settings(message=f'Do you want to select another lamella position?\n'
                                               f'{len(self.samples)} positions selected so far.', crosshairs=False)
            self.ask_user()
            select_another_sample_position = self.response
            self.update_scroll_ui()

        ####################################
        # # move to landing grid
        movement.move_to_landing_grid(self.microscope, settings=self.settings, flat_to_sem=False)
        movement.auto_link_stage(self.microscope, hfw=900e-6)
        self.ensure_eucentricity(flat_to_sem=False)
        ####################################

        # select corresponding sample landing positions
        for current_sample_position in self.samples:

            # check if landing position already selected? so it doesnt overwrite
            if current_sample_position.landing_selected is False:
                self.select_landing_sample_positions(current_sample_position)

        # load all the data from disk (to load images)
        for sample_position in self.samples:
            sample_position.load_data_from_file()

        self.update_scroll_ui()

        # reset microscope coordinate system
        self.microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
        logging.info(f"Selected {len(self.samples)} initial sample and landing positions.")

    def load_experiment_button_pressed(self, btn):

        if "Yes" in btn.text():
            self.load_experiment = True
        else:
            self.load_experiment = False

    def setup_experiment(self):

        CONTINUE_SETUP_EXPERIMENT = True
        experiment_path = os.path.join(os.path.dirname(liftout.__file__), "log")
        experiment_name = os.path.basename(experiment_path)

        # ui
        msg = QMessageBox()
        msg.setWindowTitle("AutoLiftout Startup")
        msg.setText("Do you want to load a previous experiment?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.buttonClicked.connect(self.load_experiment_button_pressed)
        msg.exec_()

        # cmdline
        # user_response = input("Do you want to load a previous experiment? (y/n)")
        # load_experiment = True if user_response.lower() == "y" else False

        if self.load_experiment:
            # load_experiment
            experiment_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Log Folder to Load",
                                                                         directory=experiment_path)
            # if the user doesnt select a folder, start a new experiment
            # TODO: include a check for invalid folders here too?
            if experiment_path is "":
                self.load_experiment = False
                experiment_path = os.path.join(os.path.dirname(liftout.__file__), "log")
            else:
                experiment_name = os.path.basename(experiment_path)
                self.save_path = experiment_path
                self.log_path = utils.configure_logging(save_path=self.save_path, log_filename="logfile")

        # start from scratch
        if self.load_experiment is False:

            # create_new_experiment
            experiment_name, okPressed = QInputDialog.getText(self, "New AutoLiftout Experiment",
                                                              "Enter a name for your experiment:", QLineEdit.Normal, "default_experiment")
            if not okPressed or experiment_name == "":
                experiment_name = "default_experiment"

            # experiment_name = input("Enter a name for the experiment: ")

            run_name = f"{experiment_name}_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d.%H%M%S')}"
            self.save_path = utils.make_logging_directory(prefix=run_name)
            self.log_path = utils.configure_logging(save_path=self.save_path, log_filename="logfile")
            self.samples = []
            experiment_name = run_name
            experiment_path = self.save_path
            CONTINUE_SETUP_EXPERIMENT = False

        if CONTINUE_SETUP_EXPERIMENT:
            # check if there are valid sample positions stored in this directory
            # check if the file exists
            yaml_file = os.path.join(experiment_path, "sample.yaml")
            if not os.path.exists(yaml_file):
                error_msg = "This experiment does not contain any saved sample positions."
                logging.warning(error_msg)
                self.samples = []
                CONTINUE_SETUP_EXPERIMENT = False

        if CONTINUE_SETUP_EXPERIMENT:
            # check if there are any sample positions stored in the file
            sample = SamplePosition(experiment_path, None)
            yaml_file = sample.setup_yaml_file()
            if len(yaml_file["sample"]) == 0:
                error_msg = "This experiment does not contain any saved sample positions."
                logging.error(error_msg)
                display_error_message(error_msg)
                CONTINUE_SETUP_EXPERIMENT = False

        if CONTINUE_SETUP_EXPERIMENT:
            # reset previously loaded samples
            self.samples = []
            self.current_sample_position = None

            # load the samples
            sample = SamplePosition(experiment_path, None)
            yaml_file = sample.setup_yaml_file()
            for sample_no in yaml_file["sample"].keys():
                sample = SamplePosition(experiment_path, sample_no)
                sample.load_data_from_file()
                self.samples.append(sample)

        self.experiment_name = experiment_name
        self.experiment_path = experiment_path

        # update the ui
        self.update_scroll_ui()

        logging.info(f"{len(self.samples)} samples loaded from {experiment_path}")
        logging.info(f"Reload Experiment Finished")

    def select_initial_lamella_positions(self, sample_no):
        """Select the initial sample positions for liftout"""
        sample_position = SamplePosition(data_path=self.save_path, sample_no=sample_no)

        movement.move_to_sample_grid(self.microscope, settings=self.settings)
        movement.auto_link_stage(self.microscope)

        self.ensure_eucentricity()
        movement.move_to_trenching_angle(self.microscope, settings=self.settings)

        sample_position.lamella_coordinates = self.user_select_feature(feature_type="lamella")

        # save microscope state
        sample_position.microscope_state = calibration.get_current_microscope_state(microscope=self.microscope,
                                                                                    stage=self.current_stage,
                                                                                    eucentric=True)
        sample_position.save_data()

        self.update_image_settings(
            resolution=self.settings['reference_images']['trench_area_ref_img_resolution'],
            dwell_time=self.settings['reference_images']['trench_area_ref_img_dwell_time'],
            hfw=self.settings['reference_images']['trench_area_ref_img_hfw_lowres'],
            save=True,
            save_path=os.path.join(self.save_path, str(sample_position.sample_id)),
            label=f"ref_lamella_low_res"
        )
        acquire.take_reference_images(self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings['reference_images']['trench_area_ref_img_resolution'],
            dwell_time=self.settings['reference_images']['trench_area_ref_img_dwell_time'],
            hfw=self.settings['reference_images']['trench_area_ref_img_hfw_highres'],
            save=True,
            save_path=os.path.join(self.save_path, str(sample_position.sample_id)),
            label="ref_lamella_high_res"
        )
        acquire.take_reference_images(self.microscope, settings=self.image_settings)

        return sample_position

    def select_landing_sample_positions(self, current_sample_position: SamplePosition):
        logging.info(f"Selecting Landing Position: {current_sample_position.sample_id}")

        # select landing coordinates
        current_sample_position.landing_coordinates = self.user_select_feature(feature_type="landing")

        # mill the landing edge flat
        self.mill_flat_landing_edge()

        current_sample_position.landing_selected = True

        self.update_image_settings(
            resolution=self.settings['reference_images']['landing_post_ref_img_resolution'],
            dwell_time=self.settings['reference_images']['landing_post_ref_img_dwell_time'],
            hfw=self.settings['reference_images']['landing_post_ref_img_hfw_lowres'],
            save=True,
            save_path=os.path.join(self.save_path, str(current_sample_position.sample_id)),
            label="ref_landing_low_res"
        )
        acquire.take_reference_images(self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings['reference_images']['landing_post_ref_img_resolution'],
            dwell_time=self.settings['reference_images']['landing_post_ref_img_dwell_time'],
            hfw=self.settings['reference_images']['landing_post_ref_img_hfw_highres'],
            save=True,
            save_path=os.path.join(self.save_path, str(current_sample_position.sample_id)),
            label="ref_landing_high_res"
        )
        acquire.take_reference_images(self.microscope, settings=self.image_settings)

        # save coordinates
        current_sample_position.save_data()

    def user_select_feature(self, feature_type):
        """Get the user to centre the beam on the desired feature"""
        # refresh
        self.update_image_settings(hfw=self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"], save=False)
        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.update_display(beam_type=BeamType.ION, image_type='new')

        self.update_popup_settings(message=f'Please double click to centre the {feature_type} coordinate in the ion beam.\n'
                                           f'Press Yes when the feature is centered', click='double',
                                   filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_FIB)

        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')

        self.microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.RAW)

        return self.stage.current_position

    def mill_flat_landing_edge(self):

        # mill the edge of the landing post flat
        logging.info(f"Preparing to flatten landing surface.")
        flatten_landing_pattern = milling.flatten_landing_pattern(microscope=self.microscope, settings=self.settings)

        self.update_display(beam_type=BeamType.ION, image_type='last')
        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength,
                                   crosshairs=False, milling_patterns=flatten_landing_pattern)
        self.ask_user(image=self.image_FIB)
        if self.response:
            # TODO: refactor this into draw_patterns_and_mill
            # additional args: pattern_type, scan_direction, milling_current
            self.microscope.imaging.set_active_view(2)  # the ion beam view
            self.microscope.patterning.clear_patterns()
            for pattern in self.patterns:
                tmp_pattern = self.microscope.patterning.create_cleaning_cross_section(
                    center_x=pattern.center_x,
                    center_y=pattern.center_y,
                    width=pattern.width,
                    height=pattern.height,
                    depth=self.settings["flatten_landing"]["depth"]
                )
                tmp_pattern.rotation = -np.deg2rad(pattern.rotation)
                tmp_pattern.scan_direction = "LeftToRight"
            milling.run_milling(microscope=self.microscope, settings=self.settings, milling_current=6.2e-9)
        else:
            self.microscope.patterning.clear_patterns()
        logging.info(f"{self.current_stage.name} | FLATTEN_LANDING | FINISHED")

    # def select_initial_feature_coordinates(self, feature_type=''):
    #     """
    #     Options are 'lamella' or 'landing'
    #     """
    #
    #     # TODO: REMOVE THIS FUNCTION
    #     select_another_position = True
    #     coordinates = []
    #     images = []
    #
    #     if feature_type == 'lamella':
    #         movement.move_to_sample_grid(self.microscope, settings=self.settings)
    #         movement.auto_link_stage(self.microscope)
    #     elif feature_type == 'landing':
    #         movement.move_to_landing_grid(self.microscope, settings=self.settings, flat_to_sem=False)
    #         movement.auto_link_stage(self.microscope, hfw=900e-6)
    #         self.ensure_eucentricity(flat_to_sem=False)
    #     else:
    #         raise ValueError(f'Expected "lamella" or "landing" as feature_type')
    #
    #     while select_another_position:
    #         if feature_type == 'lamella':
    #             self.ensure_eucentricity()
    #             movement.move_to_trenching_angle(self.microscope, settings=self.settings)
    #
    #         # refresh
    #         self.update_image_settings(hfw=self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"], save=False)
    #         self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
    #         self.update_display(beam_type=BeamType.ION, image_type='new')
    #
    #         self.update_popup_settings(message=f'Please double click to centre the {feature_type} coordinate in the ion beam.\n'
    #                                     f'Press Yes when the feature is centered', click='double',
    #                                     filter_strength=self.filter_strength, allow_new_image=True)
    #         self.ask_user(image=self.image_FIB)
    #
    #         self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
    #
    #         self.microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.RAW)
    #         coordinates.append(self.stage.current_position)
    #         if feature_type == 'landing':
    #             self.update_image_settings(
    #                 resolution=self.settings['reference_images']['landing_post_ref_img_resolution'],
    #                 dwell_time=self.settings['reference_images']['landing_post_ref_img_dwell_time'],
    #                 hfw=self.settings['reference_images']['landing_post_ref_img_hfw_lowres'],
    #                 save=True,
    #                 label=f"{len(coordinates):02d}_ref_landing_low_res"
    #             )
    #             eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.image_settings)
    #
    #             self.update_image_settings(
    #                 resolution=self.settings['reference_images']['landing_post_ref_img_resolution'],
    #                 dwell_time=self.settings['reference_images']['landing_post_ref_img_dwell_time'],
    #                 hfw=self.settings['reference_images']['landing_post_ref_img_hfw_highres'],
    #                 save=True,
    #                 label=f"{len(coordinates):02d}_ref_landing_high_res"
    #             )
    #             eb_highres, ib_highres = acquire.take_reference_images(self.microscope, settings=self.image_settings)
    #
    #             #######################################################################
    #             # mill the edge of the landing post flat
    #             logging.info(f"Preparing to flatten landing surface.")
    #             flatten_landing_pattern = milling.flatten_landing_pattern(microscope=self.microscope, settings=self.settings)
    #
    #             self.update_display(beam_type=BeamType.ION, image_type='last')
    #             self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength,
    #                                        crosshairs=False, milling_patterns=flatten_landing_pattern)
    #             self.ask_user(image=self.image_FIB)
    #             if self.response:
    #                 # TODO: refactor this into draw_patterns_and_mill
    #                 # additional args: pattern_type, scan_direction, milling_current
    #                 self.microscope.imaging.set_active_view(2)  # the ion beam view
    #                 self.microscope.patterning.clear_patterns()
    #                 for pattern in self.patterns:
    #                     tmp_pattern = self.microscope.patterning.create_cleaning_cross_section(
    #                         center_x=pattern.center_x,
    #                         center_y=pattern.center_y,
    #                         width=pattern.width,
    #                         height=pattern.height,
    #                         depth=self.settings["flatten_landing"]["depth"]
    #                     )
    #                     tmp_pattern.rotation = -np.deg2rad(pattern.rotation)
    #                     tmp_pattern.scan_direction = "LeftToRight"
    #                 milling.run_milling(microscope=self.microscope, settings=self.settings, milling_current=6.2e-9)
    #             else:
    #                 self.microscope.patterning.clear_patterns()
    #             logging.info(f"{self.current_status.name} | FLATTEN_LANDING | FINISHED")
    #             #######################################################################
    #
    #         elif feature_type == 'lamella':
    #             self.update_image_settings(
    #                 resolution=self.settings['reference_images']['trench_area_ref_img_resolution'],
    #                 dwell_time=self.settings['reference_images']['trench_area_ref_img_dwell_time'],
    #                 hfw=self.settings['reference_images']['trench_area_ref_img_hfw_lowres'],
    #                 save=True,
    #                 label=f"{len(coordinates):02d}_ref_lamella_low_res"
    #             )
    #             eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.image_settings)
    #
    #             self.update_image_settings(
    #                 resolution=self.settings['reference_images']['trench_area_ref_img_resolution'],
    #                 dwell_time=self.settings['reference_images']['trench_area_ref_img_dwell_time'],
    #                 hfw=self.settings['reference_images']['trench_area_ref_img_hfw_highres'],
    #                 save=True,
    #                 label=f"{len(coordinates):02d}_ref_lamella_high_res"
    #             )
    #             eb_highres, ib_highres = acquire.take_reference_images(self.microscope, settings=self.image_settings)
    #
    #         images.append((eb_lowres, eb_highres, ib_lowres, ib_highres))
    #
    #         self.update_popup_settings(message=f'Do you want to select another {feature_type} position?\n'
    #                                     f'{len(coordinates)} positions selected so far.', crosshairs=False)
    #         self.ask_user()
    #         select_another_position = self.response
    #
    #     self.microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
    #     logging.info(f"{self.current_status.name}: finished selecting {len(coordinates)} {feature_type} points.")
    #
    #     return coordinates, images

    def ensure_eucentricity(self, flat_to_sem=True):
        calibration.validate_scanning_rotation(self.microscope)
        if flat_to_sem:
            movement.flat_to_beam(self.microscope, settings=self.settings, beam_type=BeamType.ELECTRON)

        # TODO: maybe update images here?
        # lowres calibration
        self.user_based_eucentric_height_adjustment(hfw=self.settings["calibration"]["eucentric_hfw_lowres"])  # 900e-6

        # midres eucentric calibration
        self.user_based_eucentric_height_adjustment(hfw=self.settings["calibration"]["eucentric_hfw_midres"])  # 400e-6

        # highres calibration
        self.user_based_eucentric_height_adjustment(hfw=self.settings["calibration"]["eucentric_hfw_highres"])  # 200e-6

    def user_based_eucentric_height_adjustment(self, hfw=None):

        self.update_image_settings(
            resolution='1536x1024',
            dwell_time=1e-6,
            beam_type=BeamType.ELECTRON,
            hfw=hfw,
            save=False
        )

        self.image_SEM = acquire.new_image(self.microscope, settings=self.image_settings)
        self.update_popup_settings(message=f'Please double click to centre a feature in the SEM\n'
                                                           f'Press Yes when the feature is centered', click='double',
                                   filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_SEM)

        if self.response:
            self.image_settings['beam_type'] = BeamType.ION
            self.image_settings["hfw"] = float(min(self.image_settings["hfw"], self.settings["imaging"]["max_ib_hfw"]))  # clip to max hfw for ion, 900e-6 #TODO: implement this before taking images...?
            self.update_display(beam_type=BeamType.ION, image_type='new')
            self.update_popup_settings(message=f'Please click the same location in the ion beam\n'
                                                           f'Press Yes when happy with the location', click='single',
                                       filter_strength=self.filter_strength, crosshairs=False, allow_new_image=False)
            self.ask_user(image=self.image_FIB, second_image=self.image_SEM)

        else:
            logging.warning('calibration: electron image not centered')
            return

        self.image_FIB = acquire.last_image(self.microscope, beam_type=BeamType.ION)
        real_x, real_y = movement.pixel_to_realspace_coordinate([self.xclick, self.yclick], self.image_FIB)
        delta_z = -np.cos(self.stage.current_position.t) * real_y
        self.stage.relative_move(StagePosition(z=delta_z))
        logging.info(f"eucentric: moving height by {delta_z:.4f}m")
        if self.response:
            self.update_display(beam_type=BeamType.ION, image_type='new')
        # TODO: Could replace this with an autocorrelation (maybe with a fallback to asking for a user click if the correlation values are too low)
        self.image_settings['beam_type'] = BeamType.ELECTRON
        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.update_popup_settings(message=f'Please double click to centre a feature in the SEM\n'
                                                           f'Press Yes when the feature is centered', click='double',
                                   filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_SEM)

    def run_liftout(self):
        logging.info(f"gui: run liftout started")
        logging.info(f"gui: running liftout on {len(self.samples)} samples.")

        for sample in self.samples:

            self.current_sample_position = sample
            self.single_liftout()

        logging.info(f"{len(self.samples)} lamella landings complete. Ready for thinning.")

    def run_autoliftout_workflow(self):

        logging.info(f"AutoLiftout Workflow started for {len(self.samples)} sample positions.")

        # TODO: move this to the INIT
        self.autoliftout_stages = {
            AutoLiftoutStage.Setup: self.initialise_autoliftout,  # TODO: rename this to setup_autoliftout, and button too
            AutoLiftoutStage.MillTrench: self.mill_lamella_trench,
            AutoLiftoutStage.MillJCut: self.mill_lamella_jcut,
            AutoLiftoutStage.Liftout: self.liftout_lamella,
            AutoLiftoutStage.Landing: self.land_lamella,
            AutoLiftoutStage.Reset: self.reset_needle,
            AutoLiftoutStage.Thinning: self.new_thin_lamella,
            AutoLiftoutStage.Polishing: self.polish_lamella
        }

        for sp in self.samples:
            self.current_sample_position = sp

            while sp.microscope_state.last_completed_stage.value < AutoLiftoutStage.Reset.value:

                next_stage = AutoLiftoutStage(sp.microscope_state.last_completed_stage.value + 1)
                msg = f"The last completed stage for sample position {str(sp.sample_id)[-6:]} is {sp.microscope_state.last_completed_stage.name}. " \
                      f"\nWould you like to continue from {next_stage.name}?\n"
                self.update_popup_settings(message=msg, crosshairs=False)
                self.ask_user()

                # TODO: what to do if we say no?
                # if self.response:

                # TODO: check the interaction between start_of_stage_update, running the stage, and end_of_stage_update
                self.start_of_stage_update(next_stage=next_stage)

                # run the next workflow stage
                self.autoliftout_stages[next_stage]()

                # advance workflow
                self.end_of_stage_update(eucentric=True) # TODO: how to find out if we are eucentric?

    def run_thinning_workflow(self):
        # TODO: make button for this

        self.THINNING_EUCENTRICITY_COMPLETED = False
        # thinning
        for sp in self.samples:
            self.current_sample_position = sp
            self.new_thin_lamella()
            self.end_of_stage_update(eucentric=True)

        # polish
        for sp in self.samples:
            self.current_sample_position = sp
            self.polish_lamella()
            self.end_of_stage_update(eucentric=True)

        # finish the experiment
        self.current_stage = AutoLiftoutStage.Finished
        for sp in self.samples:
            self.current_sample_position = sp
            self.end_of_stage_update(eucentric=True)

        #
        # # NOTE: thinning needs to happen after all lamella landed due to contamination buildup...
        # self.update_popup_settings(message="Do you want to start lamella thinning?", crosshairs=False)
        # self.ask_user()
        # logging.info(f"Perform Thinning: {self.response}")
        # if self.response:
        #     for sample in self.samples:
        #         self.current_sample_position = sample
        #         landing_coord = self.current_sample_position.landing_coordinates
        #         self.current_status = AutoLiftoutStage.Thinning
        #         self.thin_lamella(landing_coord=landing_coord)


    # def load_coordinates(self):
    #     # TODO: REMOVE
    #     logging.info(f"LOAD COORDINATES STARTED")
    #     # input save path
    #     save_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Log Folder to Load",
    #                                                            directory=os.path.join(os.path.dirname(liftout.__file__), "log"))
    #     if not save_path:
    #         error_msg = "Load Coordinates: No Folder selected."
    #         logging.warning(error_msg)
    #         display_error_message(error_msg)
    #         return
    #
    #     ##########
    #     # read the sample.yaml: get how many samples in there, loop through
    #
    #     # test if the file exists
    #     yaml_file = os.path.join(save_path, "sample.yaml")
    #
    #     if not os.path.exists(yaml_file):
    #         error_msg = "sample.yaml file could not be found in this directory."
    #         logging.error(error_msg)
    #         display_error_message(error_msg)
    #         return
    #
    #     # test if there are any samples in the file?
    #     sample = SamplePosition(save_path, None)
    #     yaml_file = sample.setup_yaml_file()
    #
    #
    #     num_of_samples = len(yaml_file["sample"])
    #
    #     if num_of_samples == 0:
    #         # error out if no sample.yaml found...
    #         error_msg = "sample.yaml file has no stored sample coordinates."
    #         logging.error(error_msg)
    #         display_error_message(error_msg)
    #         return
    #
    #     else:
    #         # load the samples
    #         self.samples = []
    #         for sample_no in yaml_file["sample"].keys():
    #             sample = SamplePosition(save_path, sample_no)
    #             sample.load_data_from_file()
    #             self.samples.append(sample)
    #     self.pushButton_autoliftout.setEnabled(True)
    #
    #     logging.info(f"{len(self.samples)} samples loaded from {save_path}")
    #     logging.info(f"LOAD COORDINATES FINISHED")

    def single_liftout(self):

        # mill trench
        self.update_popup_settings(message="Do you want to start milling trenches?", crosshairs=False)
        self.ask_user()
        logging.info(f"Perform MillTrench: {self.response}")
        if self.response:
            self.mill_lamella_trench()

        # mill jcut
        self.update_popup_settings(message="Do you want to start milling j-cut?", crosshairs=False)
        self.ask_user()
        logging.info(f"Perform MillJcut: {self.response}")
        if self.response:
            self.mill_lamella_jcut()

        # liftout
        self.update_popup_settings(message="Do you want to start liftout?", crosshairs=False)
        self.ask_user()
        logging.info(f"Perform Liftout: {self.response}")
        if self.response:
            self.liftout_lamella()

        # landing
        self.update_popup_settings(message="Do you want to start landing?", crosshairs=False)
        self.ask_user()
        logging.info(f"Perform Landing: {self.response}")
        if self.response:
            self.land_lamella()

        # reset
        self.update_popup_settings(message="Do you want to start reset?", crosshairs=False)
        self.ask_user()
        logging.info(f"Perform Reset: {self.response}")
        if self.response:
            self.reset_needle()

    def end_of_stage_update(self, eucentric: bool) -> None:
        """Save the current microscope state configuration to disk, and log that the stage has been completed."""
        # TODO: this can only be called if current_sample_position exists, how to indicate
        # save state information
        microscope_state = calibration.get_current_microscope_state(microscope=self.microscope,
                                                                    stage=self.current_stage,
                                                                    eucentric=eucentric)
        self.current_sample_position.microscope_state = microscope_state
        self.current_sample_position.save_data()

        # update ui
        self.update_scroll_ui()

        logging.info(f"{self.current_sample_position.sample_id} | {self.current_stage.name} | FINISHED")

        return

    def start_of_stage_update(self, next_stage: AutoLiftoutStage) -> None:
        """Check the last completed stage and reload the microscope state if required. Log that the stage has started. """
        last_completed_stage = self.current_sample_position.microscope_state.last_completed_stage

        # TODO: if the last completed stage is liftout... need to confirm the lamella is still attached to needle?

        if last_completed_stage.value == next_stage.value - 1:
            logging.info(f"{self.current_sample_position.sample_id} restarting from end of stage: {last_completed_stage.name}")
            calibration.set_microscope_state(self.microscope, self.current_sample_position.microscope_state)

        self.current_stage = next_stage
        logging.info(f"{self.current_sample_position.sample_id} | {self.current_stage.name}  | STARTED")

        return

    def mill_lamella_trench(self):
        # self.current_stage = AutoLiftoutStage.MillTrench # TODO: REMOVE

        # TODO: change this to load ref images from disk
        (lamella_coordinates, landing_coordinates,
            original_lamella_area_images, original_landing_images) = self.current_sample_position.get_sample_data()
        # TODO: some bug here maybe?

        # initial state
        self.MILLING_COMPLETED_THIS_RUN = False

        # move to saved lamella position
        movement.safe_absolute_stage_movement(microscope=self.microscope, stage_position=lamella_coordinates)

        ret = calibration.correct_stage_drift(self.microscope, self.image_settings,
                                              original_lamella_area_images, mode='eb')
        self.image_SEM = acquire.last_image(self.microscope, beam_type=BeamType.ELECTRON)

        # TODO: remove the cross correlation failure mode, as it does nothing because we immediately ask the user about it?
        if ret is False:
            # cross-correlation has failed, manual correction required
            self.update_popup_settings(message=f'Please double click to centre the lamella in the image.',
                                       click='double', filter_strength=self.filter_strength, allow_new_image=True)
            self.ask_user(image=self.image_SEM)
            logging.info(f"{self.current_stage.name}: cross-correlation manually corrected")


        self.update_popup_settings(message=f'Is the lamella currently centered in the image?\n'
                                           f'If not, double click to center the lamella, press Yes when centered.',
                                   click='double', filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_SEM)

        self.update_image_settings(
            save=True,
            label="initial_position_post_drift_correction"
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        # move flat to the ion beam, stage tilt 25 (total image tilt 52)
        movement.move_to_trenching_angle(self.microscope, self.settings)

        # Take an ion beam image at the *milling current*
        self.update_image_settings(hfw=self.settings["reference_images"]["trench_area_ref_img_hfw_highres"])
        self.update_display(beam_type=BeamType.ION, image_type='new')
        self.update_popup_settings(message=f'Have you centered the lamella position in the ion beam?\n'
                                                      f'If not, double click to center the lamella position', click='double',
                                   filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_FIB)

        # TODO: we should update the lamella_coordinates here, and save the data?

        # MILL_TRENCHES
        protocol_stages = milling.get_milling_protocol_stages(settings=self.settings, stage_name="new_lamella")
        trench_patterns = milling.mill_trench_patterns(microscope=self.microscope, settings=protocol_stages[1])  # only show final trenches...

        # TODO: support moving the trench patterns?
        self.update_display(beam_type=BeamType.ION, image_type='last')
        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength,
                                   crosshairs=False, milling_patterns=trench_patterns)
        self.ask_user(image=self.image_FIB)
        if self.response:

            milling.mill_lamella_trenches(microscope=self.microscope, settings=self.settings)

        if self.ADDITIONAL_CONFIRMATION:
            self.update_popup_settings(message="Was the milling successful?\nIf not, please manually fix, and then press yes.", filter_strength=self.filter_strength, crosshairs=False)
            self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
            self.update_display(beam_type=BeamType.ION, image_type="new")
            self.ask_user(image=self.image_FIB)

        # reference images of milled trenches
        self.update_image_settings(
            resolution=self.settings['reference_images']['trench_area_ref_img_resolution'],
            dwell_time=self.settings['reference_images']['trench_area_ref_img_dwell_time'],
            hfw=self.settings['reference_images']['trench_area_ref_img_hfw_lowres'],
            save=True,
            label=f"ref_trench_low_res"
        )
        acquire.take_reference_images(self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings['reference_images']['trench_area_ref_img_resolution'],
            dwell_time=self.settings['reference_images']['trench_area_ref_img_dwell_time'],
            hfw=self.settings['reference_images']['trench_area_ref_img_hfw_highres'],
            save=True,
            label=f"ref_trench_high_res"
        )
        acquire.take_reference_images(self.microscope, settings=self.image_settings)

        # Mill Trenches is Finished Here...
        self.end_of_stage_update(eucentric=True)


    def mill_lamella_jcut(self):
        ####################################### JCUT #######################################

        # Movement to JCut is below

        # TODO: remove once using start_of stage update
        self.current_stage = AutoLiftoutStage.MillJCut
        logging.info(f"{self.current_sample_position.sample_id} | {self.current_stage.name}  | STARTED")
        movement.safe_absolute_stage_movement(microscope=self.microscope,
                                              stage_position=self.current_sample_position.lamella_coordinates)

        # load the reference images # TODO: TEST
        reference_images_low_and_high_res = []
        for fname in ["ref_trench_low_res_eb", "ref_trench_high_res_eb", "ref_trench_low_res_ib", "ref_trench_high_res_ib"]:

            img = self.current_sample_position.load_reference_image(fname)
            reference_images_low_and_high_res.append(img)

        # move flat to electron beam
        movement.flat_to_beam(self.microscope, self.settings, beam_type=BeamType.ELECTRON, )

        # make sure drift hasn't been too much since milling trenches
        # first using reference images
        ret = calibration.correct_stage_drift(self.microscope, self.image_settings,
                                              reference_images_low_and_high_res, mode='ib')

        if ret is False:
            # cross-correlation has failed, manual correction required
            self.update_popup_settings(message=f'Please double click to centre the lamella in the image.',
                         click='double', filter_strength=self.filter_strength, allow_new_image=True)
            self.image_SEM = acquire.last_image(self.microscope, beam_type=BeamType.ELECTRON)
            self.ask_user(image=self.image_SEM)
            logging.info(f"{self.current_stage.name}: cross-correlation manually corrected")

        self.update_image_settings(hfw=self.settings["calibration"]["drift_correction_hfw_highres"],
                                   save=True, label=f"drift_correction_ML")

        # then using ML, tilting/correcting in steps so drift isn't too large
        # self.correct_stage_drift_with_ML() # TODO: test if we can remove this
        stage_settings = MoveSettings(rotate_compucentric=True)
        movement.move_relative(self.microscope, t=np.deg2rad(self.settings["jcut"]["jcut_angle"]), settings=stage_settings)
        self.update_image_settings(hfw=self.settings["calibration"]["drift_correction_hfw_highres"],
                                   save=True, label=f"drift_correction_ML")
        self.correct_stage_drift_with_ML()

        ## MILL_JCUT
        # now we are at the angle for jcut, perform jcut

        jcut_patterns = milling.mill_jcut(self.microscope, self.settings)

        self.update_display(beam_type=BeamType.ION, image_type='last')
        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength,
                            crosshairs=False, milling_patterns=jcut_patterns)
        self.ask_user(image=self.image_FIB)
        if self.response:

            milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                           patterns=self.patterns, depth=self.settings["jcut"]['jcut_milling_depth'])

        logging.info(f"{self.current_sample_position.sample_id} | {self.current_stage.name} | MILL_JCUT | FINISHED")
        ##

        # take reference images of the jcut
        self.update_image_settings(hfw=self.settings["reference_images"]["milling_ref_img_hfw_lowres"],
                                   save=True, label=f"jcut_lowres")
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_image_settings(hfw=self.settings["reference_images"]["milling_ref_img_hfw_highres"],
                                   save=True, label=f"jcut_highres")
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.MILLING_COMPLETED_THIS_RUN = True
        self.end_of_stage_update(eucentric=True)
        logging.info(f" {self.current_stage.name} FINISHED")

    def correct_stage_drift_with_ML(self):
        # correct stage drift using machine learning
        label = self.image_settings['label']
        # if self.image_settings["hfw"] > 200e-6:
        #     self.image_settings["hfw"] = 150e-6
        for beamType in (BeamType.ION, BeamType.ELECTRON, BeamType.ION): # TODO this probably only needs to happen once, not three times...
            self.image_settings['label'] = label + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')
            distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_centre_to_image_centre', beamType=beamType)

            # TODO: make moves consistent from moving to stationary
            # yz-correction
            x_move = movement.x_corrected_stage_movement(-distance_x_m, stage_tilt=self.stage.current_position.t)
            yz_move = movement.y_corrected_stage_movement(distance_y_m, stage_tilt=self.stage.current_position.t, beam_type=beamType)
            self.stage.relative_move(x_move)
            self.stage.relative_move(yz_move)
            self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
            self.update_display(beam_type=BeamType.ION, image_type="new")

        # TODO: label will overwrite previous, needs a unique identifier
        self.update_image_settings(
            save=True,
            label=f'drift_correction_ML_final_' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')
        )
        self.image_SEM, self.image_FIB = acquire.take_reference_images(self.microscope, self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        self.update_display(beam_type=BeamType.ION, image_type='last')

    def liftout_lamella(self):
        self.current_stage = AutoLiftoutStage.Liftout
        logging.info(f" {self.current_stage.name} STARTED")

        # get ready to do liftout by moving to liftout angle
        movement.move_to_liftout_angle(self.microscope, self.settings)
        movement.auto_link_stage(self.microscope)

        # check focus distance is within tolerance
        eb_image, ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        if abs(eb_image.metadata.optics.working_distance - 4e-3) > 0.5e-3:
            logging.warning("Autofocus has failed")
            self.update_popup_settings(message='Autofocus has failed, please correct the focus manually', filter_strength=self.filter_strength, crosshairs=False)
            self.ask_user()

        # if not self.MILLING_COMPLETED_THIS_RUN:
        #     self.ensure_eucentricity(flat_to_sem=True) # liftout angle is flat to SEM
        #     self.image_settings["hfw"] = self.settings["imaging"]["horizontal_field_width"]
        #     movement.move_to_liftout_angle(self.microscope, self.settings)

        # correct stage drift from mill_lamella stage
        self.correct_stage_drift_with_ML()

        # move needle to liftout start position
        if self.stage.current_position.z < self.settings["calibration"]["stage_height_limit"]: # 3.7e-3
            # [FIX] autofocus cannot be relied upon, if this condition is met, we need to stop.

            # movement.auto_link_stage(self.microscope) # This is too unreliable to fix the miscalibration
            logging.warning(f"Calibration error detected: stage position height")
            logging.warning(f"Stage Position: {self.stage.current_position}")
            display_error_message(message="The system has identified the distance between the sample and the pole piece is less than 3.7mm. "
                "The needle will contact the sample, and it is unsafe to insert the needle. "
                "\nPlease manually recalibrate the focus and restart the program. "
                "\n\nThe AutoLiftout GUI will now exit.",
                title="Calibration Error"
            )

            # Aborting Liftout
            self.disconnect()
            exit(0)

        park_position = movement.move_needle_to_liftout_position(self.microscope)
        logging.info(f"{self.current_stage.name}: needle inserted to park positon: {park_position}")

        # land needle on lamella
        self.land_needle_on_milled_lamella()

        # sputter platinum
        fibsem_utils.sputter_platinum(self.microscope, self.settings, whole_grid=False)
        logging.info(f"{self.current_stage.name}: lamella to needle welding complete.")

        self.update_image_settings(save=True, hfw=self.settings["platinum"]["weld"]["hfw"], label=f"needle_landed_Pt_sputter")
        acquire.take_reference_images(self.microscope, self.image_settings)

        logging.info(f"{self.current_stage.name} | MILL_SEVERING | STARTED")
        jcut_severing_pattern = milling.jcut_severing_pattern(self.microscope, self.settings)
        self.update_display(beam_type=BeamType.ION, image_type='last')

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False,
                                   milling_patterns=jcut_severing_pattern)
        self.ask_user(image=self.image_FIB)
        if self.response:
            milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                           patterns=self.patterns, depth=self.settings["jcut"]['jcut_milling_depth'])

        self.update_image_settings(save=True, hfw=self.settings["reference_images"]["needle_ref_img_hfw_highres"], label=f"jcut_sever")
        acquire.take_reference_images(self.microscope, self.image_settings)
        logging.info(f"{self.current_stage.name} | MILL_SEVERING | FINISHED")

        if self.ADDITIONAL_CONFIRMATION:
            self.update_popup_settings(message="Was the milling successful?\nIf not, please manually fix, and then press yes.", filter_strength=self.filter_strength, crosshairs=False)
            self.update_display(beam_type=BeamType.ION, image_type="last")
            self.ask_user(image=self.image_FIB)

        # Raise needle 30um from trench
        logging.info(f"{self.current_stage.name}: start removing needle from trench")
        for i in range(3):
            z_move_out_from_trench = movement.z_corrected_needle_movement(10e-6, self.stage.current_position.t)
            self.needle.relative_move(z_move_out_from_trench)
            self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
            self.update_display(beam_type=BeamType.ION, image_type="new")
            logging.info(f"{self.current_stage.name}: removing needle from trench at {z_move_out_from_trench}")
            time.sleep(1)

        # reference images after liftout complete
        self.image_settings['label'] = f"liftout_of_trench"
        acquire.take_reference_images(self.microscope, self.image_settings)

        # move needle to park position
        movement.retract_needle(self.microscope, park_position)

        self.end_of_stage_update(eucentric=True)
        logging.info(f"{self.current_stage.name}: needle retracted. ")
        logging.info(f" {self.current_stage.name} FINISHED")

    def land_needle_on_milled_lamella(self):

        logging.info(f"{self.current_stage.name} | LAND_NEEDLE_ON_LAMELLA | STARTED")

        needle = self.microscope.specimen.manipulator

        ### REFERENCE IMAGES
        # low res
        self.update_image_settings(
            resolution=self.settings["reference_images"]["needle_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["needle_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["needle_ref_img_hfw_lowres"],
            save=True,
            label=f"needle_liftout_start_position_lowres"
        )
        acquire.take_reference_images(self.microscope, self.image_settings)
        # high res
        self.update_image_settings(
            resolution=self.settings["reference_images"]["needle_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["needle_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["needle_ref_img_hfw_highres"],
            save=True,
            label=f"needle_liftout_start_position_highres"
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")
        ###

        ### XY-MOVE (ELECTRON)
        self.image_settings['hfw'] = self.settings["reference_images"]["liftout_ref_img_hfw_lowres"]
        self.image_settings["label"] = f"needle_liftout_pre_movement_lowres"
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='needle_tip_to_lamella_centre', beamType=BeamType.ELECTRON)

        x_move = movement.x_corrected_needle_movement(-distance_x_m, stage_tilt=self.stage.current_position.t)
        yz_move = movement.y_corrected_needle_movement(distance_y_m, stage_tilt=self.stage.current_position.t)
        needle.relative_move(x_move)
        needle.relative_move(yz_move)
        logging.info(f"{self.current_stage.name}: needle x-move: {x_move}")
        logging.info(f"{self.current_stage.name}: needle yz-move: {yz_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")
        ###

        ### Z-HALF MOVE (ION)
        # calculate shift between lamella centre and needle tip in the ion view
        self.image_settings["label"] = f"needle_liftout_post_xy_movement_lowres"
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='needle_tip_to_lamella_centre', beamType=BeamType.ION)

        # calculate shift in xyz coordinates
        z_distance = distance_y_m / np.cos(self.stage.current_position.t)

        # Calculate movement
        zy_move_half = movement.z_corrected_needle_movement(-z_distance / 2, self.stage.current_position.t)
        needle.relative_move(zy_move_half)
        logging.info(f"{self.current_stage.name}: needle z-half-move: {zy_move_half}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.update_display(beam_type=BeamType.ION, image_type='new')
        ###

        self.update_popup_settings(message='Is the needle safe to move another half step?', click=None, filter_strength=self.filter_strength)
        self.ask_user(image=self.image_FIB)

        if self.response:
            ### Z-MOVE FINAL (ION)
            self.image_settings['hfw'] = self.settings['reference_images']['needle_with_lamella_shifted_img_highres']
            self.image_settings["label"] = f"needle_liftout_post_z_half_movement_highres"
            distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='needle_tip_to_lamella_centre', beamType=BeamType.ION)

            # calculate shift in xyz coordinates
            z_distance = distance_y_m / np.cos(self.stage.current_position.t)

            # move in x
            x_move = movement.x_corrected_needle_movement(-distance_x_m)
            self.needle.relative_move(x_move)

            # move in z
            # detection is based on centre of lamella, we want to land of the edge.
            # therefore subtract half the height from the movement.
            lamella_height = self.settings["lamella"]["lamella_height"]
            gap = lamella_height / 6
            zy_move_gap = movement.z_corrected_needle_movement(-(z_distance - gap), self.stage.current_position.t)
            self.needle.relative_move(zy_move_gap)

            logging.info(f"{self.current_stage.name}: needle x-move: {x_move}")
            logging.info(f"{self.current_stage.name}: needle zy-move: {zy_move_gap}")

            self.update_image_settings(
                hfw=self.settings["reference_images"]["needle_ref_img_hfw_lowres"],
                save=True,
                label=f"needle_liftout_landed_lowres"
            )
            acquire.take_reference_images(self.microscope, self.image_settings)

            self.update_image_settings(
                hfw=self.settings["reference_images"]["needle_ref_img_hfw_highres"],
                save=True,
                label=f"needle_liftout_landed_highres"
            )
            acquire.take_reference_images(self.microscope, self.image_settings)
            logging.info(f"{self.current_stage.name} | LAND_NEEDLE_ON_LAMELLA | FINISHED")
            ###

        else:
            logging.warning(f"{self.current_stage.name}: needle not safe to move onto lamella.")
            logging.warning(f"{self.current_stage.name}: needle landing cancelled by user.")
            return

    def calculate_shift_distance_metres(self, shift_type, beamType=BeamType.ELECTRON):
        self.image_settings['beam_type'] = beamType
        self.raw_image, self.overlay_image, self.downscaled_image, feature_1_px, feature_1_type, feature_2_px, feature_2_type = \
            calibration.identify_shift_using_machine_learning(microscope=self.microscope, image_settings=self.image_settings, settings=self.settings, shift_type=shift_type)
        feature_1_px, feature_2_px = self.validate_detection(feature_1_px=feature_1_px, feature_1_type=feature_1_type, feature_2_px=feature_2_px, feature_2_type=feature_2_type)
        # scaled features
        scaled_feature_1_px = detection_utils.scale_invariant_coordinates(feature_1_px, self.overlay_image) #(y, x)
        scaled_feature_2_px = detection_utils.scale_invariant_coordinates(feature_2_px, self.overlay_image) # (y, x)
        # distance
        distance_x = scaled_feature_2_px[1] - scaled_feature_1_px[1]
        distance_y = scaled_feature_2_px[0] - scaled_feature_1_px[0]
        x_pixel_size = self.raw_image.metadata.binary_result.pixel_size.x
        y_pixel_size = self.raw_image.metadata.binary_result.pixel_size.y
        # distance in metres
        distance_x_m = x_pixel_size * self.raw_image.width * distance_x
        distance_y_m = y_pixel_size * self.raw_image.height * distance_y
        logging.info(f"calculated detection distance: x = {distance_x_m:.4f}m , y = {distance_y_m:.4f}m")
        return distance_x_m, distance_y_m

    def validate_detection(self, feature_1_px=None, feature_1_type=None, feature_2_px=None, feature_2_type=None):
        self.update_popup_settings(message=f'Has the model correctly identified the {feature_1_type} and {feature_2_type} positions?', click=None, crosshairs=False)
        self.ask_user(image=self.overlay_image)

        DETECTIONS_ARE_CORRECT = self.response
        if DETECTIONS_ARE_CORRECT:
            logging.info(f"ml_detection: {feature_1_type}: {self.response}")
            logging.info(f"ml_detection: {feature_2_type}: {self.response}")

        # if something wasn't correctly identified
        # if not self.response:
        while not DETECTIONS_ARE_CORRECT:
            utils.save_image(image=self.raw_image, save_path=self.image_settings['save_path'], label=self.image_settings['label'] + '_label')

            self.update_popup_settings(message=f'Has the model correctly identified the {feature_1_type} position?', click=None, crosshairs=False)
            self.ask_user(image=self.overlay_image)

            logging.info(f"ml_detection: {feature_1_type}: {self.response}")

            # if feature 1 wasn't correctly identified
            if not self.response:

                self.update_popup_settings(message=f'Please click on the correct {feature_1_type} position.'
                                                                   f'Press Yes button when happy with the position', click='single', crosshairs=False)
                self.ask_user(image=self.downscaled_image)

                # if new feature position selected
                if self.response:
                    feature_1_px = (self.yclick, self.xclick)

            # skip image centre 'detections'
            if feature_2_type != "image_centre": 
                self.update_popup_settings(message=f'Has the model correctly identified the {feature_2_type} position?', click=None, crosshairs=False)
                self.ask_user(image=self.overlay_image)

                logging.info(f"ml_detection: {feature_2_type}: {self.response}")
                # if feature 2 wasn't correctly identified 
                if not self.response:

                    self.update_popup_settings(message=f'Please click on the correct {feature_2_type} position.'
                                                                    f'Press Yes button when happy with the position', click='single',
                                                                    filter_strength=self.filter_strength, crosshairs=False)
                    self.ask_user(image=self.downscaled_image)

                    # if new feature position selected
                    if self.response:
                        feature_2_px = (self.yclick, self.xclick)

            # TODO: wrap this in a function
            # show the user the manually corrected movement and confirm
            from liftout.detection.detection import draw_two_features
            final_detection_img = Image.fromarray(self.downscaled_image).convert("RGB")
            final_detection_img = draw_two_features(final_detection_img, feature_1_px, feature_2_px)
            final_detection_img = np.array(final_detection_img.convert("RGB"))
            self.update_popup_settings(
                message=f'Are the {feature_1_type} and {feature_2_type} positions now correctly identified?', 
                click=None, crosshairs=False
            )
            self.ask_user(image=final_detection_img)
            DETECTIONS_ARE_CORRECT = self.response
            #####

        return feature_1_px, feature_2_px

    def land_lamella(self):

        self.current_stage = AutoLiftoutStage.Landing
        logging.info(f"{self.current_stage.name} STARTED")

        # load positions and reference images,
        (lamella_coordinates, landing_coordinates,
         original_lamella_area_images, original_landing_images) = self.current_sample_position.get_sample_data()

        # stage_settings = MoveSettings(rotate_compucentric=True)
        # self.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)
        # self.stage.absolute_move(StagePosition(x=landing_coord.x, y=landing_coord.y, r=landing_coord.r, coordinate_system=landing_coord.coordinate_system))
        # self.stage.absolute_move(landing_coord)

        # TODO: TEST THIS
        # move to landing coordinate
        movement.safe_absolute_stage_movement(microscope=self.microscope, stage_position=landing_coordinates)
        movement.auto_link_stage(self.microscope, hfw=400e-6)

        # eucentricity correction
        self.ensure_eucentricity(flat_to_sem=False)  # liftout angle is flat to SEM
        self.image_settings["hfw"] = self.settings["imaging"]["horizontal_field_width"]

        ret = calibration.correct_stage_drift(self.microscope, self.image_settings,
                                              original_landing_images, mode="land")

        if ret is False:
            # cross-correlation has failed, manual correction required
            self.update_popup_settings(message=f'Please double click to centre the lamella in the image.',
                         click='double', filter_strength=self.filter_strength, allow_new_image=True)
            self.ask_user(image=self.image_FIB)
            logging.info(f"{self.current_stage.name}: cross-correlation manually corrected")

        logging.info(f"{self.current_stage.name}: initial landing calibration complete.")

        ############################## LAND_LAMELLA ##############################
        logging.info(f"{self.current_stage.name} | LAND_LAMELLA | STARTED")
        park_position = movement.move_needle_to_landing_position(self.microscope)

        #### Y-MOVE (ELECTRON)
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["liftout_ref_img_hfw_lowres"],
            beam_type=BeamType.ELECTRON,
            save=True,
            label=f"landing_needle_land_sample_lowres"
        )

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=BeamType.ELECTRON)

        y_move = movement.y_corrected_needle_movement(-distance_y_m, self.stage.current_position.t)
        self.needle.relative_move(y_move)
        logging.info(f"{self.current_stage.name}: y-move complete: {y_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")


        #### Z-MOVE (ION)
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["liftout_ref_img_hfw_lowres"],
            beam_type=BeamType.ION,
            save=True,
            label=f"landing_needle_land_sample_lowres_after_y_move"
        )
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=BeamType.ION)

        z_distance = distance_y_m / np.sin(np.deg2rad(52))  # TODO: MAGIC_NUMBER
        z_move = movement.z_corrected_needle_movement(z_distance, self.stage.current_position.t)
        self.needle.relative_move(z_move)
        logging.info(f"{self.current_stage.name}: z-move complete: {z_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")

        #### X-HALF-MOVE (ELECTRON)
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["landing_lamella_ref_img_hfw_lowres"],
            beam_type=BeamType.ELECTRON,
            save=True,
            label=f"landing_needle_land_sample_lowres_after_z_move"
        )

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=BeamType.ELECTRON)

        # half move
        x_move = movement.x_corrected_needle_movement(distance_x_m / 2)
        self.needle.relative_move(x_move)
        logging.info(f"{self.current_stage.name}: x-half-move complete: {x_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")

        #### X-MOVE
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["landing_lamella_ref_img_hfw_highres"],
            beam_type=BeamType.ELECTRON,
            save=True,
            label=f"landing_needle_land_sample_lowres_after_z_move"
        )
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=BeamType.ELECTRON)

        x_move = movement.x_corrected_needle_movement(distance_x_m)
        self.needle.relative_move(x_move)
        logging.info(f"{self.current_stage.name}: x-move complete: {x_move}")

        # final reference images
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["landing_lamella_ref_img_hfw_highres"],
            beam_type=BeamType.ELECTRON,
            save=True,
            label=f"landing_lamella_final_weld_highres"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        if self.ADDITIONAL_CONFIRMATION:
            self.update_popup_settings(message="Was the landing successful?\nIf not, please manually fix, and then press yes.", filter_strength=self.filter_strength, crosshairs=False)
            self.ask_user(image=self.image_SEM)

        logging.info(f"{self.current_stage.name} | LAND_LAMELLA | FINISHED")
        #################################################################################################

        ############################## WELD TO LANDING POST #############################################
        logging.info(f"{self.current_stage.name} | MILLING_WELD | STARTED")

        weld_pattern = milling.weld_to_landing_post(self.microscope, self.settings)
        self.update_display(beam_type=BeamType.ION, image_type='last')

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?',
                                   filter_strength=self.filter_strength, crosshairs=False, milling_patterns=weld_pattern)
        self.ask_user(image=self.image_FIB)

        if self.response:
            logging.info(f"{self.current_stage.name}: welding to post started.")
            milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                           patterns=self.patterns, depth=self.settings["weld"]["depth"])

        logging.info(f"{self.current_stage.name} | MILLING_WELD | FINISHED")

        #################################################################################################

        # final reference images
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["landing_lamella_ref_img_hfw_highres"],
            save=True,
            label=f"landing_lamella_final_weld_highres"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")


        ###################################### CUT_OFF_NEEDLE ######################################
        logging.info(f"{self.current_stage.name} | CUT_OFF_NEEDLE | STARTED")

        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["cut"]["hfw"],
            beam_type=BeamType.ION,
            save=True,
            label=f"landing_lamella_pre_cut_off"
        )

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=BeamType.ION)

        height = self.settings["cut"]["height"]
        width = self.settings["cut"]["width"]
        depth = self.settings["cut"]["depth"]
        rotation = self.settings["cut"]["rotation"]
        hfw = self.settings["cut"]["hfw"]
        vertical_gap = self.settings["cut"]["gap"]
        horizontal_gap = self.settings["cut"]["hgap"] # TODO:  TO_TEST

        cut_coord = {"center_x": -distance_x_m - horizontal_gap,
                     "center_y": distance_y_m - vertical_gap,
                     "width": width,
                     "height": height,
                     "depth": depth,
                     "rotation": rotation, "hfw": hfw}

        logging.info(f"{self.current_stage.name}: calculating needle cut-off pattern")

        # cut off needle tip
        cut_off_pattern = milling.cut_off_needle(self.microscope, cut_coord=cut_coord)
        self.update_display(beam_type=BeamType.ION, image_type='last')

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False,
                                   milling_patterns=cut_off_pattern)
        self.ask_user(image=self.image_FIB)

        if self.response:
            logging.info(f"{self.current_stage.name}: needle cut-off started")
            milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                           patterns=self.patterns, depth=cut_coord["depth"])

        logging.info(f"{self.current_stage.name} | CUT_OFF_NEEDLE | FINISHED")
        #################################################################################################

        # reference images
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["landing_lamella_ref_img_hfw_lowres"],
            beam_type=BeamType.ION,
            save=True,
            label=f"landing_lamella_final_cut_lowres"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["landing_lamella_ref_img_hfw_highres"],
            beam_type=BeamType.ION,
            save=True,
            label=f"landing_lamella_final_cut_highres"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        if self.ADDITIONAL_CONFIRMATION:
            self.update_popup_settings(message="Was the milling successful?\nIf not, please manually fix, and then press yes.", filter_strength=self.filter_strength, crosshairs=False)
            self.update_display(beam_type=BeamType.ION, image_type="last")
            self.ask_user(image=self.image_FIB)

        logging.info(f"{self.current_stage.name}: removing needle from landing post")
        # move needle out of trench slowly at first
        for i in range(3):
            z_move_out_from_post = movement.z_corrected_needle_movement(10e-6, self.stage.current_position.t)
            self.needle.relative_move(z_move_out_from_post)
            self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
            self.update_display(beam_type=BeamType.ION, image_type="new")
            logging.info(f"{self.current_stage.name}: moving needle out: {z_move_out_from_post} ({i + 1} / 3")
            time.sleep(1)

        # move needle to park position
        movement.retract_needle(self.microscope, park_position)
        logging.info(f"{self.current_stage.name}: needle retracted.")

        # reference images
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["landing_lamella_ref_img_hfw_lowres"],
            save=True,
            label=f"landing_lamella_final_lowres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["landing_lamella_ref_img_hfw_highres"],
            save=True,
            label=f"landing_lamella_final_highres"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.end_of_stage_update(eucentric=True)
        logging.info(f"{self.current_stage.name} FINISHED")

    def reset_needle(self):

        self.current_stage = AutoLiftoutStage.Reset
        logging.info(f" {self.current_stage.name} STARTED")

        # move sample stage out
        movement.move_sample_stage_out(self.microscope)
        logging.info(f"{self.current_stage.name}: moved sample stage out")

        ###################################### SHARPEN_NEEDLE ######################################
        logging.info(f"{self.current_stage.name} | SHARPEN_NEEDLE | STARTED")

        # move needle in
        park_position = movement.insert_needle(self.microscope)
        z_move_in = movement.z_corrected_needle_movement(-180e-6, self.stage.current_position.t)
        self.needle.relative_move(z_move_in)
        logging.info(f"{self.current_stage.name}: insert needle for reset")

        # needle images
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=self.settings["imaging"]["horizontal_field_width"],
            beam_type=BeamType.ION,
            save=True,
            label=f"sharpen_needle_initial"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=BeamType.ION)

        x_move = movement.x_corrected_needle_movement(distance_x_m)
        self.needle.relative_move(x_move)
        z_distance = distance_y_m / np.sin(np.deg2rad(52))  # TODO: MAGIC_NUMBER
        z_move = movement.z_corrected_needle_movement(z_distance, self.stage.current_position.t)
        self.needle.relative_move(z_move)
        logging.info(f"{self.current_stage.name}: moving needle to centre: x_move: {x_move}, z_move: {z_move}")

        self.image_settings["label"] = f"sharpen_needle_centre"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=BeamType.ION)

        # create sharpening patterns
        cut_coord_bottom, cut_coord_top = milling.calculate_sharpen_needle_pattern(microscope=self.microscope, settings=self.settings, x_0=distance_x_m, y_0=distance_y_m)
        logging.info(f"{self.current_stage.name}: calculate needle sharpen pattern")

        sharpen_patterns = milling.create_sharpen_needle_patterns(
            self.microscope, cut_coord_bottom, cut_coord_top
        )

        # confirm and run milling
        self.update_display(beam_type=BeamType.ION, image_type='last')

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength,
                                   crosshairs=False, milling_patterns=sharpen_patterns)
        self.ask_user(image=self.image_FIB)
        if self.response:
            logging.info(f"{self.current_stage.name}: needle sharpening milling started")
            milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                           patterns=self.patterns, depth=cut_coord_bottom["depth"], milling_current=6.2e-9)

        logging.info(f"{self.current_stage.name} | SHARPEN_NEEDLE | FINISHED")
        #################################################################################################

        # take reference images
        self.image_settings["label"] = f"sharpen_needle_final"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        # retract needle
        movement.retract_needle(self.microscope, park_position)

        # reset stage position
        stage_settings = MoveSettings(rotate_compucentric=True)
        self.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)
        # self.stage.absolute_move(StagePosition(r=lamella_coordinates.r))
        self.stage.absolute_move(StagePosition(x=0.0, y=0.0))

        self.end_of_stage_update(eucentric=False)
        logging.info(f"{self.current_stage.name} FINISHED")

    def new_thin_lamella(self):

        self.current_stage = AutoLiftoutStage.Thinning
        logging.info(f"{self.current_sample_position.sample_id} | {self.current_stage.name} | STARTED")

        thinning_coordinate = self.current_sample_position.landing_coordinates
        thinning_coordinate.r = np.deg2rad(self.settings["thin_lamella"]["rotation_angle"])  # TODO: check this rotation
        thinning_coordinate.t = np.deg2rad(self.settings["thin_lamella"]["tilt_angle"])

        # complete eucentricity at this angle if it hasn't been conducted
        if not self.THINNING_EUCENTRICITY_COMPLETED:

            # move to the initial landing coordinates
            movement.safe_absolute_stage_movement(microscope=self.microscope,
                                                  stage_position=self.current_sample_position.landing_coordinates)
            # ensure_eucentricity
            self.ensure_eucentricity(flat_to_sem=False)

            # rotate_and_tilt_to_thinning_angle
            self.image_settings["hfw"] = self.settings["imaging"]["horizontal_field_width"]
            movement.safe_absolute_stage_movement(microscope=self.microscope, stage_position=thinning_coordinate)

            # ensure_eucentricity at thinning angle
            self.ensure_eucentricity(flat_to_sem=False)
            self.THINNING_EUCENTRICITY_COMPLETED = True

        # move to thinning coord
        movement.safe_absolute_stage_movement(microscope=self.microscope, stage_position=thinning_coordinate)

        # lamella images
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_lowres"],
            save=True,
            label=f"thinning_lamella_0_deg_tilt"
        )

        acquire.take_reference_images(self.microscope, self.image_settings)

        # realign lamella to image centre
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_medres"],
            save=True,
            label=f"thinning_drift_correction_medres"
        )

        self.image_SEM, self.image_FIB = acquire.take_reference_images(self.microscope, self.image_settings)
        self.update_popup_settings(message=f'Please double click to centre the lamella coordinate in the ion beam.\n'
                                           f'Press Yes when the feature is centered', click='double',
                                   filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_FIB)

        # take reference images
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_highres"],
            save=True,
            label=f"thinning_drift_correction_highres"
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        # thin_lamella (align and mill)
        self.update_image_settings()
        milling.mill_thin_lamella(microscope=self.microscope, settings=self.settings,
                                  image_settings=self.image_settings, milling_type="thin")

        # TODO: check the beam currents etc are restored here

        # take reference images
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_superres"],
            save=True,
            label=f"thin_lamella_post_superres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        #
        # self.update_image_settings(
        #     resolution=self.settings["imaging"]["resolution"],
        #     dwell_time =self.settings["imaging"]["dwell_time"],
        #     hfw=self.settings["reference_images"]["thinning_ref_img_hfw_highres"],
        #     save=True,
        #     label=f"thinning_lamella_post_highres"
        # )
        #
        # acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        #
        # self.update_image_settings(
        #     resolution=self.settings["imaging"]["resolution"],
        #     dwell_time =self.settings["imaging"]["dwell_time"],
        #     hfw=self.settings["reference_images"]["thinning_ref_img_hfw_medres"],
        #     save=True,
        #     label=f"thinning_lamella_post_lowres"
        # )
        #
        # acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        return

    def thin_lamella(self, landing_coord):
        """Thinning: Thin the lamella thickness to size for imaging."""

        self.current_stage = AutoLiftoutStage.Thinning
        logging.info(f" {self.current_stage.name} STARTED")

        # move to landing coord
        # TODO: safe rotation first?
        movement.safe_absolute_stage_movement(microscope=self.microscope, stage_position=landing_coord)
        # self.microscope.specimen.stage.absolute_move(landing_coord)
        logging.info(f"{self.current_stage.name}: move to landing coordinates: {landing_coord}")

        self.ensure_eucentricity(flat_to_sem=False)  # liftout angle is flat to SEM
        self.image_settings["hfw"] = self.settings["imaging"]["horizontal_field_width"]

        # move to the thinning position
        thinning_coordinate = landing_coord
        thinning_coordinate.r = np.deg2rad(self.settings["thin_lamella"]["rotation_angle"])  # TODO: check this rotation
        thinning_coordinate.t = np.deg2rad(self.settings["thin_lamella"]["tilt_angle"])
        movement.safe_absolute_stage_movement(microscope=self.microscope, stage_position=thinning_coordinate)


        # tilt to 0 rotate 180 move to 21 deg
        # tilt to zero, to prevent hitting anything
        # stage_settings = MoveSettings(rotate_compucentric=True)
        # self.microscope.specimen.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)
        #
        # # thinning position
        # thinning_rotation_angle = self.settings["thin_lamella"]["rotation_angle"]  # 180 deg # TODO: convert to absolute movement for safety (50deg, aka start angle)
        # thinning_tilt_angle = self.settings["thin_lamella"]["tilt_angle"]  # 0 deg
        #
        # # rotate to thinning angle
        # self.microscope.specimen.stage.relative_move(StagePosition(r=np.deg2rad(thinning_rotation_angle)), stage_settings)
        #
        # # tilt to thinning angle
        # self.microscope.specimen.stage.absolute_move(StagePosition(t=np.deg2rad(thinning_tilt_angle)), stage_settings)
        # logging.info(f"{self.current_status.name}: rotate to thinning angle: {thinning_rotation_angle}")
        # logging.info(f"{self.current_status.name}: tilt to thinning angle: {thinning_tilt_angle}")

        # ensure eucentricity for thinning milling
        self.ensure_eucentricity(flat_to_sem=False)

        # TODO: check if working distance of the ion beam is 16.5mm once eucentric

        # lamella images
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_lowres"],
            save=True,
            label=f"thinning_lamella_0_deg_tilt"
        )

        acquire.take_reference_images(self.microscope, self.image_settings)

        # realign lamella to image centre
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_medres"],
            save=True,
            label=f"thinning_drift_correction_medres"
        )

        self.image_SEM, self.image_FIB = acquire.take_reference_images(self.microscope, self.image_settings)
        self.update_popup_settings(message=f'Please double click to centre the lamella coordinate in the ion beam.\n'
                                           f'Press Yes when the feature is centered', click='double',
                                   filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_FIB)

        # take reference images
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_highres"],
            save=True,
            label=f"thinning_drift_correction_highres"
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        ###################################### THIN_LAMELLA ######################################

        # NEW THINNING
        self.update_image_settings()
        calibration.test_thin_lamella(microscope=self.microscope, settings=self.settings, image_settings=self.image_settings, sample_no=self.current_sample_position.sample_no)

        ###################################################################################################

        # take reference images and finish

        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_superres"],
            save=True,
            label=f"thinning_lamella_post_superres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_highres"],
            save=True,
            label=f"thinning_lamella_post_highres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_medres"],
            save=True,
            label=f"thinning_lamella_post_lowres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        logging.info(f"{self.current_stage.name}: thin lamella {self.current_sample_position.sample_no} complete.")
        logging.info(f" {self.current_stage.name} FINISHED")

    def polish_lamella(self):

        # TODO: we will also need to save the reference images from the thinning stage if we want to crosscorrelate back to them here?
        # we can probably just load them from disk? See jcut for an example
        # TODO: add a helper method to SamplePosition?

        self.current_stage = AutoLiftoutStage.Polishing
        logging.info(f"{self.current_sample_position.sample_id} | {self.current_stage} | STARTED")

        # restore state from thinning stage
        if self.current_sample_position.microscope_state.last_completed_stage == AutoLiftoutStage.Thinning:
            calibration.set_microscope_state(microscope=self.microscope, microscope_state=self.current_sample_position.microscope_state)
            ref_image = AdornedImage.load(os.path.join(self.current_sample_position.data_path,
                                                       self.current_sample_position.sample_id, "thin_lamella_crosscorrelation_ref"))
        else:
            # cannot polish if the last stage was not thinning
            logging.warning(f"Unable to polish sample no {self.current_sample_position.sample_no}. Thinning was not completed previously.")
            return

        # realign lamella to image centre
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_medres"],
            save=True,
            label=f"polish_drift_correction_medres"
        )

        # TODO: check if we need to manually realign or if the cross correlation is good enough?
        self.image_SEM, self.image_FIB = acquire.take_reference_images(self.microscope, self.image_settings)
        self.update_popup_settings(message=f'Please double click to centre the lamella coordinate in the ion beam.\n'
                                           f'Press Yes when the feature is centered', click='double',
                                   filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_FIB)

        # take reference images
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_highres"],
            save=True,
            label=f"polish_drift_correction_highres"
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        # polish (align and mill)
        self.update_image_settings()
        milling.mill_thin_lamella(microscope=self.microscope, settings=self.settings,
                                  image_settings=self.image_settings, milling_type="polish", ref_image=ref_image)

        # TODO: check the beam currents etc are restored here

        # take reference images
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_superres"],
            save=True,
            label=f"polish_lamella_post_superres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_highres"],
            save=True,
            label=f"polish_lamella_post_highres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=self.settings["reference_images"]["thinning_ref_img_hfw_medres"],
            save=True,
            label=f"polish_lamella_post_lowres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        logging.info(f"{self.current_stage.name}: polish lamella {self.current_sample_position.sample_no} complete.")
        logging.info(f"{self.current_sample_position.sample_id} | {self.current_stage.name} | FINISHED")

        return

    # def initialise_image_frames(self):
        # self.figure_SEM = plt.figure()
        # plt.axis('off')
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.01)
        # self.canvas_SEM = _FigureCanvas(self.figure_SEM)
        # self.toolbar_SEM = _NavigationToolbar(self.canvas_SEM, self)
        # self.label_SEM.setLayout(QtWidgets.QVBoxLayout())
        # self.label_SEM.layout().addWidget(self.toolbar_SEM)
        # self.label_SEM.layout().addWidget(self.canvas_SEM)

        # self.canvas_SEM.mpl_connect('button_press_event', lambda event: self.on_gui_click(event, beam_type=BeamType.ELECTRON, click=None))
        # TODO: if grid not centred before initialise allow click = double temporarily

        # self.figure_FIB = plt.figure()
        # plt.axis('off')
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.01)
        # self.canvas_FIB = _FigureCanvas(self.figure_FIB)
        # self.toolbar_FIB = _NavigationToolbar(self.canvas_FIB, self)
        # self.label_FIB.setLayout(QtWidgets.QVBoxLayout())
        # self.label_FIB.layout().addWidget(self.toolbar_FIB)
        # self.label_FIB.layout().addWidget(self.canvas_FIB)
        # self.canvas_FIB.mpl_connect('button_press_event', lambda event: self.on_gui_click(event, beam_type=BeamType.ION, click=None))

    def initialize_hardware(self, offline=False):
        if offline is False:
            self.connect_to_microscope(ip_address=self.ip_address)
        elif offline is True:
            pass
            # self.connect_to_microscope(ip_address='localhost')

    def setup_connections(self):
        logging.info("gui: setup connections started")
        # Protocol and information table connections
        self.pushButton_initialise.clicked.connect(lambda: self.initialise_autoliftout())
        # self.pushButton_autoliftout.clicked.connect(lambda: self.run_liftout())
        self.pushButton_autoliftout.clicked.connect(lambda: self.run_autoliftout_workflow())
        self.pushButton_autoliftout.setEnabled(0)  # disable unless sample positions are loaded.

        # FIBSEM methods
        self.pushButton_load_sample_data.clicked.connect(lambda: self.setup_experiment())

        # TESTING METHODS TODO: TO BE REMOVED
        self.pushButton_test_popup.clicked.connect(lambda: self.testing_function())

        logging.info("gui: setup connections finished")

    def testing_function(self):

        TEST_VALIDATE_DETECTION = False
        TEST_DRAW_PATTERNS = False
        TEST_BEAM_SHIFT = False
        TEST_AUTO_LINK = False
        TEST_FLATTEN_LANDING = False
        TEST_ROTATED_PATTERNS = False
        TEST_SAMPLE_POSITIONS = True
        TEST_SAVE_PATH = False

        # if self.current_sample_position is None:
            # self.current_sample_position = SamplePosition(".", 99)
            # self.current_sample_position.sample_no = 99
        if TEST_VALIDATE_DETECTION:

            self.raw_image = AdornedImage(data=test_image)
            self.overlay_image = test_image
            self.downscaled_image = test_image
            import random
            supported_feature_types = ["image_centre", "lamella_centre", "needle_tip", "lamella_edge", "landing_post"]
            feature_1_px = (0, 0)
            feature_1_type = random.choice(supported_feature_types)
            feature_2_px = (test_image.shape[0] // 2 , test_image.shape[1] //2)
            feature_2_type = random.choice(supported_feature_types)

            feature_1_px, feature_2_px = self.validate_detection(feature_1_px=feature_1_px, feature_1_type=feature_1_type, feature_2_px=feature_2_px, feature_2_type=feature_2_type)

        if TEST_DRAW_PATTERNS:
            self.test_draw_patterns()

        if TEST_BEAM_SHIFT:
            self.update_image_settings()
            calibration.test_thin_lamella(microscope=self.microscope, settings=self.settings, image_settings=self.image_settings)

        if TEST_AUTO_LINK:
            logging.info("TESTING AUTO LINK STAGE")

            # 4e-3 is an arbitary amount, we can focus at any distance, but the eucentric height (hardware defined) is at 4e-3
            # if there is a large difference between the stage z and working distance we need to refocus /link

            eb_image, ib_image = acquire.take_reference_images(self.microscope, self.image_settings)
            # working distance = focus distance
            #  stage.working_distance
            movement.auto_link_stage(microscope=self.microscope, expected_z=3.9e-3)

        if TEST_FLATTEN_LANDING:
            # logging.info(f"Flatten Landing Pattern")
            # flatten_landing_pattern = milling.flatten_landing_pattern(microscope=self.microscope, settings=self.settings)


            logging.info(f"Preparing to flatten landing surface.")
            flatten_landing_pattern = milling.flatten_landing_pattern(microscope=self.microscope, settings=self.settings)

            self.update_display(beam_type=BeamType.ION, image_type='last')
            self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength,
                                       crosshairs=False, milling_patterns=flatten_landing_pattern)
            self.ask_user(image=self.image_FIB)
            if self.response:
                self.microscope.imaging.set_active_view(2)  # the ion beam view
                self.microscope.patterning.clear_patterns()
                for pattern in self.patterns:
                    tmp_pattern = self.microscope.patterning.create_cleaning_cross_section(
                        center_x=pattern.center_x,
                        center_y=pattern.center_y,
                        width=pattern.width,
                        height=pattern.height,
                        depth=self.settings["flatten_landing"]["depth"]
                    )
                    tmp_pattern.rotation = -np.deg2rad(pattern.rotation)
                    tmp_pattern.scan_direction = "LeftToRight"
            milling.run_milling(microscope=self.microscope, settings=self.settings, milling_current=6.4e-9)
            logging.info(f"{self.current_stage.name} | FLATTEN_LANDING | FINISHED")

        if TEST_ROTATED_PATTERNS:
            # create sharpening patterns
            cut_coord_bottom, cut_coord_top = milling.calculate_sharpen_needle_pattern(microscope=self.microscope, settings=self.settings, x_0=0, y_0=0)
            logging.info(f"{self.current_stage.name}: calculate needle sharpen pattern")

            sharpen_patterns = milling.create_sharpen_needle_patterns(
                self.microscope, cut_coord_bottom, cut_coord_top
            )

            # confirm and run milling
            self.update_display(beam_type=BeamType.ION, image_type='last')

            self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength,
                                       crosshairs=False, milling_patterns=sharpen_patterns)
            self.ask_user(image=self.image_FIB)

            if self.response:
                logging.info(f"{self.current_stage.name}: needle sharpening milling started")
                milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                               patterns=self.patterns, depth=cut_coord_bottom["depth"], milling_current=6.2e-9)

        if TEST_SAMPLE_POSITIONS:
            self.new_select_sample_positions()

        if TEST_SAVE_PATH:
            if self.samples:
                self.current_sample_position = self.samples[0]
                self.update_image_settings()

    def ask_user(self, image=None, second_image=None):
        self.select_all_button = None

        if image is not None:
            self.popup_settings['image'] = image
        else:
            self.popup_settings['image'] = None
        if second_image is not None:
            self.popup_settings['second_image'] = second_image
        else:
            self.popup_settings['second_image'] = None

        if image is self.image_FIB:
            self.popup_settings['beam_type'] = BeamType.ION
            self.image_settings['beam_type'] = BeamType.ION
        elif image is self.image_SEM:
            self.popup_settings['beam_type'] = BeamType.ELECTRON
            self.image_settings['beam_type'] = BeamType.ELECTRON
        else:
            self.popup_settings['beam_type'] = None

        beam_type = self.image_settings['beam_type']

        # turn off main window while popup window in use
        self.setEnabled(False)

        # used to avoid bug of closing and reopening test popup window (might not be an issue for regular running)
        self.popup_settings['click_crosshair'] = None

        # settings of the popup window
        self.popup_window = QtWidgets.QDialog()
        self.popup_window.setLayout(QtWidgets.QGridLayout())
        self.popup_window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.popup_window.destroyed.connect(lambda: self.setEnabled(True))
        self.popup_window.destroyed.connect(lambda: self.popup_settings.pop('image'))
        self.popup_window.destroyed.connect(lambda: self.popup_settings.pop('second_image'))
        self.popup_window.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.popup_canvas = None

        if self.popup_settings['message']:
            # set up message
            message_frame = QtWidgets.QWidget(self.popup_window)
            message_layout = QtWidgets.QHBoxLayout()
            message_frame.setLayout(message_layout)
            font = QtGui.QFont()
            font.setPointSize(16)

            message = QtWidgets.QLabel()
            message.setText(self.popup_settings['message'])
            message.setFont(font)
            message.setFixedHeight(50)
            message.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            message.setAlignment(QtCore.Qt.AlignCenter)
            message_frame.layout().addWidget(message)
        else:
            logging.warning('Popup called without a message')
            return

        # yes/no buttons
        button_box = QtWidgets.QWidget(self.popup_window)
        button_box.setFixedHeight(int(self.button_height*1.2)) #TODO: MAGIC NUMBER
        button_layout = QtWidgets.QGridLayout()
        yes = QtWidgets.QPushButton('Yes')
        yes.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        yes.setFixedHeight(self.button_height)
        yes.setFixedWidth(self.button_width)
        yes.clicked.connect(lambda: self.set_response(True))
        yes.clicked.connect(lambda: self.popup_window.close())

        no = QtWidgets.QPushButton('No')
        no.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        no.setFixedHeight(self.button_height)
        no.setFixedWidth(self.button_width)
        no.clicked.connect(lambda: self.set_response(False))
        no.clicked.connect(lambda: self.popup_window.close())

        # spacers
        h_spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        h_spacer2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        # button layout
        button_box.setLayout(button_layout)
        button_box.layout().addItem(h_spacer, 0, 0, 2, 1)
        button_box.layout().addWidget(yes, 0, 1, 2, 1)
        button_box.layout().addWidget(no, 0, 2, 2, 1)
        button_box.layout().addItem(h_spacer2, 0, 3, 2, 1)

        # set up the rest of the window and update display
        if second_image is not None:
            self.popup_settings['second_image'] = second_image

        if image is not None:
            # extra check in case new_image set on ML images
            if self.popup_settings['allow_new_image'] and beam_type:
                self.new_image = QtWidgets.QPushButton()
                self.new_image.setFixedHeight(self.button_height)
                self.new_image.setFixedWidth(self.button_width)
                self.new_image.setText('New Image')
                message_frame.layout().addWidget(self.new_image)
                # extra check in case hfw set on ML images
                hfw_widget = QtWidgets.QWidget()
                hfw_widget_layout = QtWidgets.QGridLayout()
                hfw_widget.setLayout(hfw_widget_layout)

                # slider (set as a property so it can be accessed to set hfw)
                self.hfw_slider = QtWidgets.QSlider()
                self.hfw_slider.setOrientation(QtCore.Qt.Horizontal)
                self.hfw_slider.setMinimum(1)
                if beam_type == BeamType.ELECTRON:
                    self.hfw_slider.setMaximum(self.settings["imaging"]["max_eb_hfw"] * MICRON_TO_METRE)
                else:
                    self.hfw_slider.setMaximum(self.settings["imaging"]["max_ib_hfw"] * MICRON_TO_METRE)
                self.hfw_slider.setValue(self.image_settings['hfw'] * MICRON_TO_METRE)

                # spinbox (not a property as only slider value needed)
                hfw_spinbox = QtWidgets.QSpinBox()
                hfw_spinbox.setMinimum(1)
                if beam_type == BeamType.ELECTRON:
                    hfw_spinbox.setMaximum(self.settings["imaging"]["max_eb_hfw"] * MICRON_TO_METRE)
                else:
                    hfw_spinbox.setMaximum(self.settings["imaging"]["max_ib_hfw"] * MICRON_TO_METRE)
                hfw_spinbox.setValue(self.image_settings['hfw'] * MICRON_TO_METRE)

                self.hfw_slider.valueChanged.connect(lambda: hfw_spinbox.setValue(self.hfw_slider.value()))
                self.hfw_slider.valueChanged.connect(lambda: hfw_spinbox.setValue(self.hfw_slider.value()))

                hfw_spinbox.valueChanged.connect(lambda: self.hfw_slider.setValue(hfw_spinbox.value()))

                hfw_widget.layout().addWidget(self.hfw_slider)
                hfw_widget.layout().addWidget(hfw_spinbox)

                self.popup_window.layout().addWidget(hfw_widget, 7, 1, 1, 1)

            self.update_popup_display()

        self.popup_window.layout().addWidget(message_frame, 6, 1, 1, 1)
        self.popup_window.layout().addWidget(button_box, 8, 1, 1, 1)

        # show window
        self.popup_window.show()
        self.popup_window.exec_()

    def update_popup_display(self):
        toolbar_active = True
        second_image_array = None

        figure = plt.figure(1)
        plt.axis('off')
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.01)

        # reset the canvas and toolbar if they exist
        if self.popup_canvas:
            self.popup_window.layout().removeWidget(self.popup_canvas)
            self.popup_window.layout().removeWidget(self.popup_toolbar)
            self.popup_canvas.deleteLater()
            self.popup_toolbar.deleteLater()
        self.popup_canvas = _FigureCanvas(figure)

        if self.popup_settings['allow_new_image'] and self.popup_settings['beam_type']:
            # reset function connection
            self.new_image.clicked.connect(lambda: print(''))
            self.new_image.clicked.disconnect()

            # connect button to functions
            self.new_image.clicked.connect(lambda: self.image_settings.update(
                {'hfw': self.hfw_slider.value()*METRE_TO_MICRON}))
            self.new_image.clicked.connect(lambda: self.popup_settings.update(
                {'image': acquire.new_image(self.microscope, self.image_settings)}))
            self.new_image.clicked.connect(lambda: self.update_popup_display())

        if self.popup_settings['click']:
            self.popup_canvas.mpl_connect('button_press_event', lambda event: self.on_gui_click(event))

        # do second image first as it has less settings
        if self.popup_settings['second_image'] is not None:
            if type(self.popup_settings['second_image']) == np.ndarray:
                second_image_array = self.popup_settings['second_image'].astype(np.uint8)
            else:
                second_image_array = self.popup_settings['second_image'].data.astype(np.uint8)

            if self.popup_settings['filter_strength']:
                second_image_array = ndi.median_filter(second_image_array,
                                                size=self.popup_settings['filter_strength'])

            if second_image_array.ndim != 3:
                second_image_array = np.stack((second_image_array,) * 3, axis=-1)

        if self.popup_settings['image'] is not None:
            if type(self.popup_settings['image']) == np.ndarray:
                image_array = self.popup_settings['image'].astype(np.uint8)
            else:
                image_array = self.popup_settings['image'].data.astype(np.uint8)

            if self.popup_settings['filter_strength']:
                image_array = ndi.median_filter(image_array,
                                                size=self.popup_settings['filter_strength'])

            if image_array.ndim != 3:
                image_array = np.stack((image_array,) * 3, axis=-1)

            # Cross hairs
            xshape = image_array.shape[1]
            yshape = image_array.shape[0]
            midx = int(xshape / 2)
            midy = int(yshape / 2)

            cross_size = 120
            half_cross = cross_size / 2
            cross_thickness = 2
            half_thickness = cross_thickness / 2

            h_rect = plt.Rectangle((midx, midy - half_thickness), half_cross, cross_thickness)
            h_rect2 = plt.Rectangle((midx - half_cross, midy - half_thickness), half_cross, cross_thickness)
            v_rect = plt.Rectangle((midx - half_thickness, midy), cross_thickness, half_cross)
            v_rect2 = plt.Rectangle((midx - half_thickness, midy - half_cross), cross_thickness, half_cross)

            h_rect.set_color('xkcd:yellow')
            h_rect2.set_color('xkcd:yellow')
            v_rect.set_color('xkcd:yellow')
            v_rect2.set_color('xkcd:yellow')

            figure.clear()
            if self.popup_settings['second_image'] is not None:
                self.ax = figure.add_subplot(122)
                self.ax.set_title(' ')
                self.ax.imshow(image_array)
                ax2 = figure.add_subplot(121)
                ax2.imshow(second_image_array)
                ax2.set_title('Previous Image')

            else:
                self.ax = figure.add_subplot(111)
                self.ax.imshow(image_array)
                self.ax.set_title(' ')

            self.ax.patches = []
            if self.popup_settings['crosshairs']:
                self.ax.add_patch(h_rect)
                self.ax.add_patch(v_rect)
                self.ax.add_patch(h_rect2)
                self.ax.add_patch(v_rect2)
            if self.popup_settings['second_image'] is not None:
                ax2.add_patch(h_rect)
                ax2.add_patch(v_rect)
                ax2.add_patch(h_rect2)
                ax2.add_patch(v_rect2)

            if self.popup_settings['milling_patterns'] is not None:
                toolbar_active = False
                if self.select_all_button is None:
                    self.select_all_button = QtWidgets.QCheckBox('Select all')
                    self.popup_window.layout().addWidget(self.select_all_button)
                self.patterns = []
                for pattern in self.popup_settings['milling_patterns']:
                    if type(self.popup_settings['image']) == np.ndarray:
                        # TODO: remove this block as it is only used for testing (NEXT)
                        image_width = self.popup_settings['image'].shape[1]
                        image_height = self.popup_settings['image'].shape[0]
                        pixel_size = 1e-6
                        width = pattern[2] / pixel_size
                        height = pattern[3] / pixel_size

                        # Rectangle is defined from bottom left due to mpl, y+ down in image (so bottom left is top left)
                        # Microscope (0, 0) is middle of image, y+ = up in image
                        # Image (0, 0) is top left corner, y+ = down in image
                        rectangle_left = (image_width / 2) + (pattern[0] / pixel_size) - (width / 2)
                        rectangle_bottom = (image_height / 2) - (pattern[1] / pixel_size) - (height / 2)
                        rotation = 0
                    else:
                        image_width = self.popup_settings['image'].width
                        image_height = self.popup_settings['image'].height
                        pixel_size = self.popup_settings['image'].metadata.binary_result.pixel_size.x

                        width = pattern.width / pixel_size
                        height = pattern.height / pixel_size
                        rotation = -pattern.rotation
                        # Rectangle is defined from bottom left due to mpl
                        # Microscope (0, 0) is middle of image, y+ = up
                        # Image (0, 0) is top left corner, y+ = down

                        rectangle_left = (image_width / 2) + (pattern.center_x / pixel_size) - (width / 2) * np.cos(rotation) + (height / 2) * np.sin(rotation)
                        rectangle_bottom = (image_height / 2) - (pattern.center_y / pixel_size) - (height / 2) * np.cos(rotation) - (width / 2) * np.sin(rotation)

                    pattern = plt.Rectangle((rectangle_left, rectangle_bottom), width, height)
                    pattern.set_hatch('/////')
                    pattern.angle = np.rad2deg(rotation)
                    pattern.set_edgecolor('xkcd:pink')
                    pattern.set_fill(False)
                    self.ax.add_patch(pattern)
                    pattern = DraggablePatch(pattern)
                    pattern.pixel_size = pixel_size
                    pattern.image_width = image_width
                    pattern.image_height = image_height
                    pattern.connect()
                    pattern.update_position()
                    self.patterns.append(pattern)
                self.select_all_button.clicked.connect(lambda: self.toggle_select_all())

            if self.popup_settings['click_crosshair']:
                for patch in self.popup_settings['click_crosshair']:
                    self.ax.add_patch(patch)
            self.popup_canvas.draw()

            if toolbar_active:
                self.popup_toolbar = _NavigationToolbar(self.popup_canvas, self)
                self.popup_window.layout().addWidget(self.popup_toolbar, 1, 1, 1, 1)
            self.popup_window.layout().addWidget(self.popup_canvas, 2, 1, 4, 1)

    def update_popup_settings(self, message='default message', allow_new_image=False, click=None, filter_strength=0, crosshairs=True, milling_patterns=None):
        self.popup_settings["message"] = message
        self.popup_settings['allow_new_image'] = allow_new_image
        self.popup_settings['click'] = click
        self.popup_settings['filter_strength'] = filter_strength
        self.popup_settings['crosshairs'] = crosshairs
        if milling_patterns is not None:
            if type(milling_patterns) is list:
                self.popup_settings['milling_patterns'] = milling_patterns  # needs to be an iterable for display
            else:
                self.popup_settings['milling_patterns'] = [milling_patterns]  # needs to be an iterable for display
        else:
            self.popup_settings["milling_patterns"] = milling_patterns

    def update_image_settings(self, resolution=None, dwell_time=None, hfw=None,
                              autocontrast=None, beam_type=None, gamma=None,
                              save=None, label=None, save_path=None):

        self.image_settings["resolution"] = self.settings["imaging"]["resolution"] if resolution is None else resolution
        self.image_settings["dwell_time"] = self.settings["imaging"]["dwell_time"] if dwell_time is None else dwell_time
        self.image_settings["hfw"] = self.settings["imaging"]["horizontal_field_width"] if hfw is None else hfw
        self.image_settings["autocontrast"] = self.USE_AUTOCONTRAST if autocontrast is None else autocontrast
        self.image_settings["beam_type"] = BeamType.ELECTRON if beam_type is None else beam_type
        self.image_settings["gamma"] = self.settings["gamma"] if gamma is None else gamma
        self.image_settings["save"] = bool(self.settings["imaging"]["save"]) if save is None else save
        self.image_settings["label"] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S') if label is None else label

        # change the save path to the current sample if available
        if self.current_sample_position:
            self.image_settings["save_path"] = os.path.join(self.save_path, str(self.current_sample_position.sample_id))
        else:
            self.image_settings["save_path"] = self.save_path if save_path is None else save_path

        logging.debug(f"Image Settings: {self.image_settings}")

    def on_gui_click(self, event):
        click = self.popup_settings['click']
        image = self.popup_settings['image']
        beam_type = self.image_settings['beam_type']

        if self.popup_toolbar._active == 'ZOOM' or self.popup_toolbar._active == 'PAN':
            return
        else:
            if event.inaxes.get_title() != 'Previous Image':
                if event.button == 1:
                    if event.dblclick and (click in ('double', 'all')):
                        if image:
                            logging.info(f"{self.current_stage} | DOUBLE CLICK")
                            self.xclick = event.xdata
                            self.yclick = event.ydata
                            x, y = movement.pixel_to_realspace_coordinate(
                                [self.xclick, self.yclick], image)

                            x_move = movement.x_corrected_stage_movement(x,
                                                                         stage_tilt=self.stage.current_position.t)
                            yz_move = movement.y_corrected_stage_movement(
                                y,
                                stage_tilt=self.stage.current_position.t,
                                beam_type=beam_type)
                            self.stage.relative_move(x_move)
                            self.stage.relative_move(yz_move)
                            self.popup_settings['image'] = acquire.new_image(
                                microscope=self.microscope,
                                settings=self.image_settings)
                            if beam_type: # TODO: probably remove this if statement (useless)
                                self.update_display(beam_type=beam_type,
                                                    image_type='last')
                            self.update_popup_display()

                    elif click in ('single', 'all'):
                        logging.info(f"{self.current_stage} | SINGLE CLICK")
                        self.xclick = event.xdata
                        self.yclick = event.ydata

                        cross_size = 120
                        half_cross = cross_size / 2
                        cross_thickness = 2
                        half_thickness = cross_thickness / 2

                        h_rect = plt.Rectangle(
                            (event.xdata, event.ydata - half_thickness),
                            half_cross, cross_thickness)
                        h_rect2 = plt.Rectangle((event.xdata - half_cross,
                                                 event.ydata - half_thickness),
                                                half_cross,
                                                cross_thickness)

                        v_rect = plt.Rectangle(
                            (event.xdata - half_thickness, event.ydata),
                            cross_thickness, half_cross)
                        v_rect2 = plt.Rectangle((
                                                event.xdata - half_thickness,
                                                event.ydata - half_cross),
                                                cross_thickness,
                                                half_cross)

                        h_rect.set_color('xkcd:yellow')
                        h_rect2.set_color('xkcd:yellow')
                        v_rect.set_color('xkcd:yellow')
                        v_rect2.set_color('xkcd:yellow')

                        self.popup_settings['click_crosshair'] = (h_rect, h_rect2, v_rect, v_rect2)
                        self.update_popup_display()

    def set_response(self, response):
        self.response = response
        # xy is top left in image coords, bottom left in

        self.setEnabled(True) # enable the main window

    def test_draw_patterns(self):
        # TODO: REMOVE
        self.update_display(beam_type=BeamType.ION, image_type='last')

        if not self.offline:
            cut_coord_bottom, cut_coord_top = milling.calculate_sharpen_needle_pattern(microscope=self.microscope,
                                                                                       settings=self.settings,
                                                                                       x_0=0, y_0=0)

            # testing rotation passing from FIB to GUI
            sharpen_patterns = milling.create_sharpen_needle_patterns(self.microscope, cut_coord_bottom, cut_coord_top)
            self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False, milling_patterns=sharpen_patterns)

            self.ask_user(image=self.image_FIB)
            if self.response:
                milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                               patterns=self.patterns, depth=self.settings["jcut"]['jcut_milling_depth'])

    def connect_to_microscope(self, ip_address='10.0.0.1'):
        """Connect to the FIBSEM microscope."""
        try:
            self.microscope = fibsem_utils.initialise_fibsem(ip_address=ip_address)
        except Exception:
            display_error_message(traceback.format_exc())

    def update_display(self, beam_type=BeamType.ELECTRON, image_type='last'):
        """Update the GUI display with the current image"""
        if self.microscope:
            try:
                if beam_type is BeamType.ELECTRON:
                    if image_type == 'new':
                        self.image_settings['beam_type'] = beam_type
                        self.image_SEM = acquire.new_image(self.microscope, self.image_settings)
                    else:
                        self.image_SEM = acquire.last_image(self.microscope, beam_type=beam_type)
                    image_array = self.image_SEM.data
                    # self.figure_SEM.clear()
                    # self.figure_SEM.patch.set_facecolor((240/255, 240/255, 240/255))
                    # self.ax_SEM = self.figure_SEM.add_subplot(111)
                    # self.ax_SEM.get_xaxis().set_visible(False)
                    # self.ax_SEM.get_yaxis().set_visible(False)
                    # self.ax_SEM.imshow(image_array, cmap='gray')
                    # self.canvas_SEM.draw()
                if beam_type is BeamType.ION:
                    if image_type == 'new':
                        self.image_settings['beam_type'] = beam_type
                        self.image_FIB = acquire.new_image(self.microscope, self.image_settings)
                    else:
                        self.image_FIB = acquire.last_image(self.microscope, beam_type=beam_type)
                    image_array = self.image_FIB.data
                    # self.figure_FIB.clear()
                    # self.figure_FIB.patch.set_facecolor((240/255, 240/255, 240/255))
                    # self.ax_FIB = self.figure_FIB.add_subplot(111)
                    # self.ax_FIB.get_xaxis().set_visible(False)
                    # self.ax_FIB.get_yaxis().set_visible(False)
                    # self.ax_FIB.imshow(image_array, cmap='gray')
                    # self.canvas_FIB.draw()

            except Exception:
                display_error_message(traceback.format_exc())

    def disconnect(self):
        logging.info('Running cleanup/teardown')
        logging.debug('Running cleanup/teardown')
        if self.microscope:
            self.microscope.disconnect()

    def toggle_select_all(self, onoff=None):
        for pattern in self.patterns:

            if onoff is not None:
                pattern.toggle_move_all(onoff=onoff)
            else:
                pattern.toggle_move_all(onoff=self.select_all_button.isChecked())

    def update_status(self):

        # need to enable the window to update status bars when popup is open)
        WINDOW_ENABLED = self.isEnabled()
        if not WINDOW_ENABLED:
            self.setEnabled(True)

        mode = "" if not self.offline else "\n(Offline Mode)"
        self.label_stage.setText(f"{self.current_stage.name}{mode}")
        status_colors = {"Initialisation": "gray", "Setup": "gold",
                         "MillTrench": "coral", "MillJCut": "coral", "Liftout": "seagreen", "Landing": "dodgerblue",
                         "Reset": "salmon", "Thinning": "mediumpurple", "Finished": "cyan"}
        self.label_stage.setStyleSheet(str(f"background-color: {status_colors[self.current_stage.name]}; color: white; border-radius: 5px"))

        # log info
        with open(self.log_path) as f:
            lines = f.read().splitlines()
            log_line = "\n".join(lines[-3:])  # last log msg
            log_msg = log_line.split("—")[-1].strip()
            self.label_status.setText(log_msg)

        if not WINDOW_ENABLED:
            self.setEnabled(False)

    def draw_sample_grid_ui(self):
        self.horizontalGroupBox = QGroupBox(f"Experiment: {self.experiment_name} ({self.experiment_path})")

        ###################

        gridLayout = QGridLayout()

        # Only add data is sample positions are added
        if len(self.samples) == 0:
            label = QLabel()
            label.setText("No Sample Positions Selected. Please press initialise to begin.")
            gridLayout.addWidget(label)
            self.horizontalGroupBox.setLayout(gridLayout)
            return

        sample_images = [[] for _ in self.samples]

        # initial, mill, jcut, liftout, land, reset, thin, polish
        exemplar_filenames = ["ref_lamella_low_res_eb", "ref_trench_high_res_ib", "jcut_highres_ib",
                              "needle_liftout_landed_highres_ib", "landing_lamella_final_cut_highres_ib", "sharpen_needle_final_ib",
                              "thinning_lamella_stage_2_ib", "thinning_lamella_final_polishing_ib"]

        # headers
        headers = ["Sample No", "Position", "Reference", "Milling", "J-Cut", "Liftout", "Landing", "Reset", "Thinning", "Polishing"]
        for j, title in enumerate(headers):
            label_header = QLabel()
            label_header.setText(title)
            label_header.setStyleSheet("font-family: Arial; font-weight: bold; font-size: 18px;")
            label_header.setAlignment(Qt.AlignCenter)
            gridLayout.addWidget(label_header, 0, j)

        # TODO: can add more fields as neccessary
        for i, sp in enumerate(self.samples):

            # load the exemplar images for each sample
            qimage_labels = []
            for img_basename in exemplar_filenames:
                fname = os.path.join(sp.data_path, str(sp.sample_id), f"{img_basename}.tif")
                imageLabel = QLabel()

                if os.path.exists(fname):
                    adorned_img = sp.load_reference_image(img_basename)
                    image = QImage(adorned_img.data, adorned_img.data.shape[1], adorned_img.data.shape[0], QImage.Format_Grayscale8)
                    imageLabel.setPixmap(QPixmap.fromImage(image).scaled(125, 125))

                qimage_labels.append(imageLabel)

            # TODO: sort out sizing / spacing when there are no images present
            sample_images[i] = qimage_labels

            # diplay information on grid
            row_id = i + 1

            # display sample no
            label_sample = QLabel()
            label_sample.setText(f"""Sample {i:02d} \n\nStage: {sp.microscope_state.last_completed_stage.name}""")
            label_sample.setStyleSheet("font-family: Arial; font-size: 12px;")

            gridLayout.addWidget(label_sample, row_id, 0)

            # display sample position
            # TOD0: replace this with a plot showing the position?
            def testColourMap(x, y):
                # ref: https://stackoverflow.com/questions/30646152/python-matplotlib-pyqt-plot-to-qpixmap
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_qt5agg import \
                    FigureCanvasQTAgg as _FigureCanvas

                fig = plt.Figure()
                canvas = _FigureCanvas(fig)
                ax = fig.add_subplot(111)
                # gradient = np.linspace(0, 1, 256)
                # gradient = np.vstack((gradient, gradient))
                # ax.imshow(gradient, aspect=10, cmap="magma")
                ax.scatter(x=x, y=y, c="blue")
                ax.set_axis_off()
                canvas.draw()
                size = canvas.size()
                width, height = size.width(), size.height()
                im = QImage(canvas.buffer_rgba(), width, height, QImage.Format_ARGB32RGB32).scaled(150, 150)
                return QPixmap(im)

            label_pos = QLabel()
            # label_pos.setPixmap(testColourMap(sp.lamella_coordinates.x, sp.landing_coordinates.y))
            pos_text = f"Pos: x:{sp.lamella_coordinates.x:.2f}, y:{sp.lamella_coordinates.y:.2f}, z:{sp.lamella_coordinates.z:.2f}\n"
            if sp.landing_coordinates.x is not None:

                pos_text += f"Land: x:{sp.landing_coordinates.x:.2f}, y:{sp.landing_coordinates.y:.2f}, z:{sp.landing_coordinates.z:.2f}\n"
                
            label_pos.setText(pos_text)
            label_pos.setStyleSheet("font-family: Arial; font-size: 12px;")

            gridLayout.addWidget(label_pos, row_id, 1)

            # display exemplar images
            gridLayout.addWidget(sample_images[i][0], row_id, 2)
            gridLayout.addWidget(sample_images[i][1], row_id, 3)
            gridLayout.addWidget(sample_images[i][2], row_id, 4)
            gridLayout.addWidget(sample_images[i][3], row_id, 5)
            gridLayout.addWidget(sample_images[i][4], row_id, 6)
            gridLayout.addWidget(sample_images[i][5], row_id, 7)
            gridLayout.addWidget(sample_images[i][6], row_id, 8)
            gridLayout.addWidget(sample_images[i][7], row_id, 9)

        self.horizontalGroupBox.setLayout(gridLayout)
        ###################


def display_error_message(message, title="Error"):
    """PyQt dialog box displaying an error message."""
    logging.info('display_error_message')
    logging.exception(message)

    error_dialog = QtWidgets.QMessageBox()
    error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
    error_dialog.setText(message)
    error_dialog.setWindowTitle(title)
    error_dialog.exec_()

    return error_dialog


def main(offline=True):
    if offline is False:
        launch_gui(offline=offline)
    else:
        try:
            launch_gui(offline=offline)
        except Exception:
            import pdb
            traceback.print_exc()
            pdb.set_trace()


def launch_gui(offline=False):
    """Launch the `autoliftout` main application window."""
    app = QtWidgets.QApplication([])
    qt_app = GUIMainWindow(offline=offline)
    app.aboutToQuit.connect(qt_app.disconnect)  # cleanup & teardown
    qt_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    offline_mode = False
    main(offline=offline_mode)
