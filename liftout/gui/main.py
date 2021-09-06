from re import M
from liftout.gui.qtdesigner_files import main as gui_main
from liftout.fibsem import acquire
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem import movement
from liftout.fibsem import calibration
from liftout.fibsem import milling
from liftout.detection import utils as detection_utils
from liftout import utils
import time
import datetime
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import traceback
import logging
import sys
import mock
from tkinter import Tk, filedialog
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from autoscript_sdb_microscope_client.structures import *

import scipy.ndimage as ndi
import skimage
import PIL
from PIL import Image
from enum import Enum
from liftout.fibsem.sample import Sample
import os

import liftout

# Required to not break imports
BeamType = acquire.BeamType

# test_image = PIL.Image.open('C:/Users/David/images/mask_test.tif')
test_image = np.random.randint(0, 255, size=(1000, 1000, 3))
test_image = np.array(test_image)

pretilt = 27  # TODO: put in protocol

_translate = QtCore.QCoreApplication.translate

protocol_template_path = '..\\protocol_liftout.yml'
starting_positions = 1
information_keys = ['x', 'y', 'z', 'rotation', 'tilt', 'comments']

class AutoLiftoutStatus(Enum):
    Initialisation = -1
    Setup = 0
    Milling = 1
    Liftout = 2
    Landing = 3
    Reset = 4
    Cleanup = 5
    Finished = 6



class GUIMainWindow(gui_main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, ip_address='10.0.0.1', offline=False):
        super(GUIMainWindow, self).__init__()
        # TODO: replace "SEM, FIB" with BeamType calls
        self.offline = offline
        self.setupUi(self)
        self.setWindowTitle('Autoliftout User Interface Main Window')
        self.liftout_counter = 0
        self.popup_window = None
        self.popup_canvas = None
        self.raw_image = None
        self.overlay_image = None
        self.downscaled_image = None

        self.filter_strength = 3
        self.button_height = 50
        self.button_width = 100

        # TODO: status bar update with prints from Autoliftout
        self.statusbar.setSizeGripEnabled(0)
        self.status = QtWidgets.QLabel(self.statusbar)
        self.status.setAlignment(QtCore.Qt.AlignRight)
        self.statusbar.addPermanentWidget(self.status, 1)

        self.setup_connections()

        # initialise image frames and images
        self.image_SEM = None
        self.image_FIB = None
        self.figure_SEM = None
        self.figure_FIB = None
        self.canvas_SEM = None
        self.canvas_FIB = None
        self.ax_SEM = None
        self.ax_FIB = None
        self.toolbar_SEM = None
        self.toolbar_FIB = None

        self.initialise_image_frames()

        # image frame interaction
        self.xclick = None
        self.yclick = None

        # initialise template and information
        self.edit = None
        self.position = 0
        self.save_destination = None
        self.key_list_protocol = list()
        self.params = {}
        for parameter in range(len(information_keys)):
            self.params[information_keys[parameter]] = 0

        self.new_protocol()

        # initialise hardware
        self.ip_address = ip_address
        self.microscope = None
        self.detector = None
        self.lasers = None
        self.objective_stage = None
        self.camera_settings = None
        self.comboBox_resolution.setCurrentIndex(2)  # resolution '3072x2048'

        self.initialize_hardware(offline=offline)

        if self.microscope:
            self.stage = self.microscope.specimen.stage
            self.needle = self.microscope.specimen.manipulator


        # TODO: remove these?
        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        self.update_display(beam_type=BeamType.ION, image_type='last')

        self.save_path = utils.make_logging_directory(prefix="run")
        utils.configure_logging(save_path=self.save_path, log_filename='logfile_')

        config_filename = os.path.join(os.path.dirname(liftout.__file__),"protocol_liftout.yml")

        self.settings = utils.load_config(config_filename)
        self.pretilt_degrees = 27  # TODO: add pretilt_degrees to protocol

        # TODO: need to consolidate this so there arent multiple different paths, its too confusing
        # currently needed to stop it crashing running immediately after init
        # self.sample_save_path = self.save_path  # NOTE: this gets overwritten when load_coords is called...

        # popup initialisations
        self.popup_window = None
        self.new_image = None
        self.hfw_slider = None
        self.popup_settings = {'message': 'startup',
                               'allow_new_image': False,
                               'click': None,
                               'filter_strength': 0,
                               'crosshairs': True}

        self.USE_AUTOCONTRAST = True

        # initial image settings # TODO: add to protocol
        self.image_settings = {'resolution': "1536x1024", 'dwell_time': 1e-6,
                               'hfw': 2750e-6,
                               'brightness': self.settings["machine_learning"]["ib_brightness"],
                               'contrast': self.settings["machine_learning"]["ib_contrast"],
                               'autocontrast': self.USE_AUTOCONTRAST,
                               'save': True, 'label': 'grid',
                               'beam_type': BeamType.ELECTRON,
                               'save_path': self.save_path,
                               "gamma_correction": self.settings["imaging"]["gamma_correction"]}
        if self.microscope:
            self.microscope.beams.ion_beam.beam_current.value = self.settings["imaging"]["imaging_current"]

        self.current_status = AutoLiftoutStatus.Initialisation
        logging.info(f"Status: {self.current_status}")

    def initialise_autoliftout(self):
        # TODO: check if needle i
        self.current_status = AutoLiftoutStatus.Setup
        logging.info(f"Status: {self.current_status}")

        # move to the initial sample grid position
        self.image_settings = {'resolution': "1536x1024", 'dwell_time': 1e-6,
                               'hfw': 2750e-6,
                               'brightness': self.settings["machine_learning"]["ib_brightness"],
                               'contrast': self.settings["machine_learning"]["ib_contrast"],
                               'autocontrast': self.USE_AUTOCONTRAST,
                               'save': True, 'label': 'grid',
                               'beam_type': BeamType.ELECTRON,
                               'save_path': self.save_path,
                               "gamma_correction": self.settings["imaging"]["gamma_correction"]}

        movement.move_to_sample_grid(self.microscope, self.settings)

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
            logging.info("setup: sputtering platinum over the whole grid")
            self.image_settings['label'] = 'grid_Pt_deposition'
            self.image_settings['save'] = True
            self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        logging.info(f"Sputter Platinum: {self.response}")

        # movement.auto_link_stage(self.microscope) # Removed as it causes problems, and should be done before starting

        # Select landing points and check eucentric height
        movement.move_to_landing_grid(self.microscope, self.settings, flat_to_sem=True)
        self.ensure_eucentricity()
        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.update_display(beam_type=BeamType.ION, image_type='new')
        self.landing_coordinates, self.original_landing_images = self.select_initial_feature_coordinates(feature_type='landing')
        self.lamella_coordinates, self.original_trench_images = self.select_initial_feature_coordinates(feature_type='lamella')
        self.zipped_coordinates = list(zip(self.lamella_coordinates, self.landing_coordinates))

        # # save
        # TODO: move sample structure into select initial feature coordinates?
        self.samples = []
        for i, (lamella_coordinates, landing_coordinates) in enumerate(self.zipped_coordinates, 1):
            sample = Sample(self.save_path, i)
            sample.lamella_coordinates = lamella_coordinates
            sample.landing_coordinates = landing_coordinates
            sample.save_data()
            self.samples.append(sample)
        
        logging.info(f"{len(self.samples)} samples selected and saved to {self.save_path}.")

    def select_initial_feature_coordinates(self, feature_type=''):
        """
        Options are 'lamella' or 'landing'
        """

        select_another_position = True
        coordinates = []
        images = []

        if feature_type == 'lamella':
            movement.move_to_sample_grid(self.microscope, settings=self.settings)
        elif feature_type == 'landing':
            movement.move_to_landing_grid(self.microscope, settings=self.settings, flat_to_sem=False)
            self.ensure_eucentricity(flat_to_sem=False)
        else:
            raise ValueError(f'Expected "lamella" or "landing" as feature_type')

        while select_another_position:
            if feature_type == 'lamella':
                self.ensure_eucentricity()
                movement.move_to_trenching_angle(self.microscope, settings=self.settings)

            self.image_settings['hfw'] = 400e-6  # TODO: set this value in protocol

            # refresh TODO: fix protocol structure
            self.image_settings['save'] = False
            self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
            self.update_display(beam_type=BeamType.ION, image_type='new')

            self.update_popup_settings(message=f'Please double click to centre the {feature_type} coordinate in the ion beam.\n'
                                                          f'Press Yes when the feature is centered', click='double', filter_strength=self.filter_strength, allow_new_image=True)
            self.ask_user(image=self.image_FIB)

            self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
            # TODO: does this need to be new image?  Can it be last?  Can it be view set?

            coordinates.append(self.stage.current_position)
            self.image_settings['save'] = True
            if feature_type == 'landing':
                self.image_settings['resolution'] = self.settings['reference_images']['landing_post_ref_img_resolution']
                self.image_settings['dwell_time'] = self.settings['reference_images']['landing_post_ref_img_dwell_time']
                self.image_settings['hfw'] = self.settings['reference_images']['landing_post_ref_img_hfw_lowres']  # TODO: watch image settings through run
                self.image_settings['label'] = f'{len(coordinates):02d}_ref_landing_low_res'  # TODO: add to protocol
                eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.image_settings)
                self.image_settings['hfw'] = self.settings['reference_images']['landing_post_ref_img_hfw_highres']
                self.image_settings['label'] = f'{len(coordinates):02d}_ref_landing_high_res'  # TODO: add to protocol
                eb_highres, ib_highres = acquire.take_reference_images(self.microscope, settings=self.image_settings)
            elif feature_type == 'lamella':
                self.image_settings['resolution'] = self.settings['reference_images']['trench_area_ref_img_resolution']
                self.image_settings['dwell_time'] = self.settings['reference_images']['trench_area_ref_img_dwell_time']
                self.image_settings['hfw'] = self.settings['reference_images']['trench_area_ref_img_hfw_lowres']  # TODO: watch image settings through run
                self.image_settings['label'] = f'{len(coordinates):02d}_ref_lamella_low_res'  # TODO: add to protocol
                eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.image_settings)
                self.image_settings['hfw'] = self.settings['reference_images']['trench_area_ref_img_hfw_highres']
                self.image_settings['label'] = f'{len(coordinates):02d}_ref_lamella_high_res'  # TODO: add to protocol
                eb_highres, ib_highres = acquire.take_reference_images(self.microscope, settings=self.image_settings)

            images.append((eb_lowres, eb_highres, ib_lowres, ib_highres))

            self.update_popup_settings(message=f'Do you want to select another {feature_type} position?\n'
                                  f'{len(coordinates)} positions selected so far.', crosshairs=False)
            self.ask_user()
            select_another_position = self.response

        logging.info(f"{self.current_status.name}: finished selecting {len(coordinates)} {feature_type} points.")

        return coordinates, images

    def ensure_eucentricity(self, flat_to_sem=True):
        calibration.validate_scanning_rotation(self.microscope)
        if flat_to_sem:
            movement.flat_to_beam(self.microscope, settings=self.settings, pretilt_angle=pretilt, beam_type=BeamType.ELECTRON)

        self.image_settings['hfw'] = 900e-6  # TODO: add to protocol
        self.microscope.beams.electron_beam.horizontal_field_width.value = self.image_settings['hfw']
        self.microscope.beams.ion_beam.horizontal_field_width.value = self.image_settings['hfw']
        acquire.autocontrast(self.microscope, beam_type=BeamType.ELECTRON)
        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        acquire.autocontrast(self.microscope, beam_type=BeamType.ION)
        self.update_display(beam_type=BeamType.ION, image_type='last')
        self.user_based_eucentric_height_adjustment()

        # TODO: status
        self.image_settings['hfw'] = 200e-6  # TODO: add to protocol
        self.microscope.beams.electron_beam.horizontal_field_width.value = self.image_settings['hfw']
        self.microscope.beams.ion_beam.horizontal_field_width.value = self.image_settings['hfw']
        self.user_based_eucentric_height_adjustment()

    def user_based_eucentric_height_adjustment(self):
        self.image_settings['resolution'] = '1536x1024'  # TODO: add to protocol
        self.image_settings['dwell_time'] = 1e-6  # TODO: add to protocol
        self.image_settings['beam_type'] = BeamType.ELECTRON
        self.image_settings['save'] = False
        self.image_SEM = acquire.new_image(self.microscope, settings=self.image_settings)
        self.update_popup_settings(message=f'Please double click to centre a feature in the SEM\n'
                                                           f'Press Yes when the feature is centered', click='double', filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_SEM)
    
        if self.response:
            self.image_settings['beam_type'] = BeamType.ION
            self.update_display(beam_type=BeamType.ION, image_type='new')
            self.update_popup_settings(message=f'Please click the same location in the ion beam\n'
                                                           f'Press Yes when happy with the location', click='single', filter_strength=self.filter_strength, crosshairs=False, allow_new_image=True)
            self.ask_user(image=self.image_FIB, second_image=self.image_SEM)

        else:
            logging.warning('calibration: electron image not centered')
            return

        self.image_FIB = acquire.last_image(self.microscope, beam_type=BeamType.ION)
        real_x, real_y = movement.pixel_to_realspace_coordinate([self.xclick, self.yclick], self.image_FIB)
        delta_z = -np.cos(self.stage.current_position.t) * real_y
        self.stage.relative_move(StagePosition(z=delta_z))
        if self.response:
            self.update_display(beam_type=BeamType.ION, image_type='new')
        # TODO: Could replace this with an autocorrelation (maybe with a fallback to asking for a user click if the correlation values are too low)
        self.image_settings['beam_type'] = BeamType.ELECTRON
        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.update_popup_settings(message=f'Please double click to centre a feature in the SEM\n'
                                                           f'Press Yes when the feature is centered', click='double', filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_SEM)

    def run_liftout(self):
        logging.info("gui: run liftout started")

        # recalibrate park position coordinates
        # reset_needle_park_position(microscope=self.microscope, new_park_position=)

        for sample in self.samples:

            self.current_sample = sample
            self.liftout_counter = self.current_sample.sample_no
            (lamella_coord, landing_coord,
                lamella_area_reference_images,
                landing_reference_images) = self.current_sample.get_sample_data()

          # TODO: this can probably just use self.current_sample rather than passing arguments?
            self.single_liftout(landing_coord, lamella_coord,
                    landing_reference_images,
                    lamella_area_reference_images)


        # # TODO: remove below once the above is tested
        # for i, (lamella_coord, landing_coord) in enumerate(self.zipped_coordinates):
        #     self.liftout_counter += 1
        #
        #     # TODO: I think this is what is crashing when immediately initalising...
        #     # Should be fixed, by initialising self.sample_save_path
        #     self.current_sample = Sample(self.sample_save_path, i+1)
        #     self.current_sample.load_data_from_file()
        #
        #
        #
        #     landing_reference_images = self.original_landing_images[i]
        #     lamella_area_reference_images = self.original_trench_images[i]
        #
        #
        #     self.single_liftout(landing_coord, lamella_coord,
        #                         landing_reference_images,
        #                         lamella_area_reference_images)




        # NOTE: cleanup needs to happen after all lamellas landed due to platinum depositions...
        # TODO: confirm this is still true
        self.update_popup_settings(message="Do you want to start lamella cleanup?", crosshairs=False)
        self.ask_user()
        logging.info(f"Perform Cleanup: {self.response}")
        if self.response:
            for sample in self.samples:
                self.current_sample = sample
                self.liftout_counter = self.current_sample.sample_no
                landing_coord = self.current_sample.landing_coordinates
                self.current_status = AutoLiftoutStatus.Cleanup
                self.cleanup_lamella(landing_coord=landing_coord)


    def load_coords(self):

        # input save path
        save_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Log Folder to Load",
                                                               directory=os.path.join(os.path.dirname(liftout.__file__), "log")) # TODO: make this path not hard coded
        if not save_path:
            logging.warning("Load Coordinates: No Folder selected.")
            display_error_message("Load Coordinates: No Folder selected.")
            return

        ##########
        # read the sample.yaml: get how many samples in there, loop through
        # TODO: maybe change sample to start at no. 0 for consistency?
        # TODO: assume all sample no are consecutive?
        # TODO: it doesnt really matter what number a sample is, just store them in a list...
        # and have a browser for them? as long as they are consistently in the same place so we can retrieve the images too?

        # test if the file exists
        yaml_file = os.path.join(save_path, "sample.yaml")

        if not os.path.exists(yaml_file):
            error_msg = "sample.yaml file could not be found in this directory."
            logging.error(error_msg)
            display_error_message(error_msg)
            return

        # test if there are any samples in the file?
        sample = Sample(save_path, None)
        yaml_file = sample.setup_yaml_file()

        num_of_samples = len(yaml_file["sample"])
        if num_of_samples == 0:
            # error out if no sample.yaml found...
            error_msg = "sample.yaml file has no stored sample coordinates."
            logging.error(error_msg)
            display_error_message(error_msg)
            return

        else:
            # load the samples
            self.samples = []
            for sample_no in range(num_of_samples):
                sample = Sample(save_path, sample_no+1) # TODO: watch out for this kind of thing with the numbering... improve
                sample.load_data_from_file()
                self.samples.append(sample)


        # TODO: test whether this is accurate, maybe move to start of run_liftout
        if sample.park_position.x is not None:
            movement.reset_needle_park_position(microscope=self.microscope, new_park_position=sample.park_position)

        logging.info(f"Load Coordinates complete from {save_path}")

    def single_liftout(self, landing_coordinates, lamella_coordinates,
                       original_landing_images, original_lamella_area_images):
        logging.info(f"Starting Liftout No {self.liftout_counter}")
        
        # initial state
        self.MILLING_COMPLETED_THIS_RUN = False # maybe make this a struct? inclcude status?

        self.stage.absolute_move(lamella_coordinates)
        calibration.correct_stage_drift(self.microscope, self.image_settings, original_lamella_area_images, self.liftout_counter, mode='eb')
        self.image_SEM = acquire.last_image(self.microscope, beam_type=BeamType.ELECTRON)
        # TODO: possibly new image
        self.update_popup_settings(message=f'Is the lamella currently centered in the image?\n'
                                                           f'If not, double click to center the lamella, press Yes when centered.', click='double', filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_SEM)
        if not self.response:
            logging.warning(f'calibration: drift correction for sample {self.liftout_counter} did not work')
            return

        self.image_settings['save'] = True
        self.image_settings['label'] = f'{self.liftout_counter:02d}_post_drift_correction'
        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')

        # mill
        self.update_popup_settings(message="Do you want to start milling?", crosshairs=False)
        self.ask_user()
        logging.info(f"Perform Milling: {self.response}")
        if self.response:
            self.mill_lamella()

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
            self.land_lamella(landing_coordinates, original_landing_images)

        # reset
        self.update_popup_settings(message="Do you want to start reset?", crosshairs=False)
        self.ask_user()
        logging.info(f"Perform Reset: {self.response}")
        if self.response:
            self.reset_needle()

    def mill_lamella(self):
        self.current_status = AutoLiftoutStatus.Milling
        logging.info(f"Status: {self.current_status}")
        stage_settings = MoveSettings(rotate_compucentric=True)

        # move flat to the ion beam, stage tilt 25 (total image tilt 52)
        movement.move_to_trenching_angle(self.microscope, self.settings)

        # Take an ion beam image at the *milling current*
        self.image_settings['hfw'] = 100e-6  # TODO: add to protocol
        self.microscope.beams.ion_beam.horizontal_field_width.value = self.image_settings['hfw']
        self.microscope.beams.electron_beam.horizontal_field_width.value = self.image_settings['hfw']
        self.update_display(beam_type=BeamType.ION, image_type='new')

        self.update_popup_settings(message=f'Have you centered the lamella position in the ion beam?\n'
                                                      f'If not, double click to center the lamella position', click='double', filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_FIB)

        # TODO: remove ask user wrapping once mill_trenches is refactored
        self.update_popup_settings(message="Do you want to start milling trenches?", crosshairs=False)
        self.ask_user()
        logging.info(f"{self.current_status.name}: perform milling trenches: {self.response}")
        if self.response:
            # mills trenches for lamella
            milling.mill_trenches(self.microscope, self.settings)

        self.current_sample.milling_coordinates = self.stage.current_position
        self.current_sample.save_data()
        logging.info(f"{self.current_status.name}: mill trenches complete.")

        # reference images of milled trenches
        self.image_settings['save'] = True
        self.image_settings['resolution'] = self.settings['reference_images']['trench_area_ref_img_resolution']
        self.image_settings['dwell_time'] = self.settings['reference_images']['trench_area_ref_img_dwell_time']

        self.image_settings['hfw'] = self.settings['reference_images']['trench_area_ref_img_hfw_lowres']  # TODO: watch image settings through run
        self.image_settings['label'] = f'{self.liftout_counter:02d}_ref_trench_low_res'  # TODO: add to protocol
        eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.image_settings)

        self.image_settings['hfw'] = self.settings['reference_images']['trench_area_ref_img_hfw_highres']
        self.image_settings['label'] = f'{self.liftout_counter:02d}_ref_trench_high_res'  # TODO: add to protocol
        eb_highres, ib_highres = acquire.take_reference_images(self.microscope, settings=self.image_settings)

        reference_images_low_and_high_res = (eb_lowres, eb_highres, ib_lowres, ib_highres)

        # move flat to electron beam
        movement.flat_to_beam(self.microscope, self.settings, pretilt_angle=pretilt, beam_type=BeamType.ELECTRON, )

        # make sure drift hasn't been too much since milling trenches
        # first using reference images
        calibration.correct_stage_drift(self.microscope, self.image_settings, reference_images_low_and_high_res, self.liftout_counter, mode='ib')
        logging.info(f"{self.current_status.name}: finished cross-correlation")

        # TODO: check dwell time value/add to protocol
        # TODO: check artifact of 0.5 into 1 dwell time
        # setup for ML drift correction
        self.image_settings['resolution'] = '1536x1024'
        self.image_settings['dwell_time'] = 1e-6
        self.image_settings['hfw'] = 80e-6
        self.image_settings['autocontrast'] = self.USE_AUTOCONTRAST
        self.image_settings['save'] = True
        # TODO: deal with resetting label requirement
        self.image_settings['label'] = f'{self.liftout_counter:02d}_drift_correction_ML'
        # then using ML, tilting/correcting in steps so drift isn't too large
        self.correct_stage_drift_with_ML()
        movement.move_relative(self.microscope, t=np.deg2rad(6), settings=stage_settings) #  TODO: test movement by 6 deg
        self.image_settings['label'] = f'{self.liftout_counter:02d}_drift_correction_ML'
        self.correct_stage_drift_with_ML()

        # save jcut position
        self.current_sample.jcut_coordinates = self.stage.current_position
        self.current_sample.save_data()

        # now we are at the angle for jcut, perform jcut
        milling.mill_jcut(self.microscope, self.settings)

        # TODO: adjust hfw? check why it changes to 100
        self.update_display(beam_type=BeamType.ION, image_type='last')
        # TODO: return image with patterning marks
        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False)
        self.ask_user(image=self.image_FIB)
        if self.response:
            milling.run_milling(self.microscope, self.settings)
        self.microscope.patterning.mode = 'Serial'

        logging.info(f"{self.current_status.name}: mill j-cut complete.")

        # take reference images of the jcut
        self.image_settings['save'] = True
        self.image_settings['label'] = 'jcut_lowres'
        self.image_settings['hfw'] = 150e-6  # TODO: add to protocol
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.image_settings['label'] = 'jcut_highres'
        self.image_settings['hfw'] = 50e-6  # TODO: add to protocol
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.MILLING_COMPLETED_THIS_RUN = True
        logging.info(f"{self.current_status.name}: milling complete.")

    def correct_stage_drift_with_ML(self):
        # correct stage drift using machine learning
        label = self.image_settings['label']
        if self.image_settings["hfw"] > 200e-6:
            self.image_settings["hfw"] = 150e-6
        for beamType in (BeamType.ION, BeamType.ELECTRON, BeamType.ION):
            # TODO: more elegant labelling convention
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

        # take reference images after drift correction
        self.image_settings['save'] = True
        self.image_settings['label'] = f'{self.liftout_counter:02d}_drift_correction_ML_final'
        self.image_settings['autocontrast'] = self.USE_AUTOCONTRAST
        self.image_SEM, self.image_FIB = acquire.take_reference_images(self.microscope, self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        self.update_display(beam_type=BeamType.ION, image_type='last')

    def liftout_lamella(self):
        self.current_status = AutoLiftoutStatus.Liftout
        logging.info(f"{self.current_status.name}: liftout started.")

        # get ready to do liftout by moving to liftout angle
        movement.move_to_liftout_angle(self.microscope, self.settings)
        logging.info(f"{self.current_status.name}: move to liftout angle.")

        if not self.MILLING_COMPLETED_THIS_RUN:
            self.ensure_eucentricity(flat_to_sem=True) # liftout angle is flat to SEM
            self.image_settings["hfw"] = 150e-6
            movement.move_to_liftout_angle(self.microscope, self.settings)
            logging.info(f"{self.current_status.name}: move to liftout angle.")

        # correct stage drift from mill_lamella stage
        self.correct_stage_drift_with_ML()

        # move needle to liftout start position
        park_position = movement.move_needle_to_liftout_position(self.microscope)
        logging.info(f"{self.current_status.name}: needle inserted to park positon: {park_position}")

        # save liftout position
        self.current_sample.park_position = park_position
        self.current_sample.liftout_coordinates = self.stage.current_position
        self.current_sample.save_data()

        # land needle on lamella
        self.land_needle_on_milled_lamella()

        # sputter platinum
        # TODO: protocol sputter
        fibsem_utils.sputter_platinum(self.microscope, self.settings, whole_grid=False, sputter_time=20) # TODO: check sputter time
        logging.info(f"{self.current_status.name}: lamella to needle welding complete.")

        self.image_settings['save'] = True
        self.image_settings['autocontrast'] = self.USE_AUTOCONTRAST
        self.image_settings['hfw'] = 100e-6
        self.image_settings['label'] = 'landed_Pt_sputter'
        acquire.take_reference_images(self.microscope, self.image_settings)

        milling.jcut_severing_pattern(self.microscope, self.settings) #TODO: tune jcut severing pattern
        self.update_display(beam_type=BeamType.ION, image_type='last')

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False)
        self.ask_user(image=self.image_FIB)
        if self.response:
            milling.run_milling(self.microscope, self.settings)
        else:
            logging.warning(f"{self.current_status.name}: user not happy with jcut sever milling pattern")
            return

        self.image_settings['label'] = 'jcut_sever'
        acquire.take_reference_images(self.microscope, self.image_settings)
        logging.info(f"{self.current_status.name}: jcut sever milling complete.")

        # Raise needle 30um from trench
        logging.info(f"{self.current_status.name}: start removing needle from trench")
        for i in range(3):
            z_move_out_from_trench = movement.z_corrected_needle_movement(10e-6, self.stage.current_position.t)
            self.needle.relative_move(z_move_out_from_trench)
            self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
            self.update_display(beam_type=BeamType.ION, image_type="new")
            logging.info(f"{self.current_status.name}: removing needle from trench at {z_move_out_from_trench}")
            time.sleep(1)

        # reference images after liftout complete
        self.image_settings['label'] = 'liftout_of_trench'
        acquire.take_reference_images(self.microscope, self.image_settings)

        # move needle to park position
        movement.retract_needle(self.microscope, park_position)

        logging.info(f"{self.current_status.name}: needle retracted. ")
        logging.info(f"{self.current_status.name}: liftout complete.")


    def land_needle_on_milled_lamella(self):

        logging.info(f"{self.current_status.name}: land needle on lamella started.")

        needle = self.microscope.specimen.manipulator
        # setup and take reference images of liftout starting position
        self.image_settings['resolution'] = self.settings["reference_images"]["needle_ref_img_resolution"]
        self.image_settings['dwell_time'] = self.settings["reference_images"]["needle_ref_img_dwell_time"]
        self.image_settings['autocontrast'] = self.USE_AUTOCONTRAST
        self.image_settings['save'] = True

        # low res
        self.image_settings['hfw'] = self.settings["reference_images"]["needle_ref_img_hfw_lowres"]
        self.image_settings['label'] = 'needle_liftout_start_position_lowres'
        acquire.take_reference_images(self.microscope, self.image_settings)
        # high res
        self.image_settings['hfw'] = self.settings["reference_images"]["needle_ref_img_hfw_highres"]
        self.image_settings['label'] = 'needle_liftout_start_position_highres'
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        # calculate shift between lamella centre and needle tip in the electron view
        self.image_settings['hfw'] = self.settings["reference_images"]["needle_ref_img_hfw_lowres"]
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='needle_tip_to_lamella_centre', beamType=BeamType.ELECTRON)

        x_move = movement.x_corrected_needle_movement(-distance_x_m, stage_tilt=self.stage.current_position.t)
        yz_move = movement.y_corrected_needle_movement(distance_y_m, stage_tilt=self.stage.current_position.t)
        needle.relative_move(x_move)
        needle.relative_move(yz_move)
        logging.info(f"{self.current_status.name}: needle x-move: {x_move}")
        logging.info(f"{self.current_status.name}: needle yz-move: {yz_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")

        # calculate shift between lamella centre and needle tip in the ion view
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='needle_tip_to_lamella_centre', beamType=BeamType.ION)
        # calculate shift in xyz coordinates
        # TODO: status

        z_distance = distance_y_m / np.cos(self.stage.current_position.t)

        # Calculate movement
        zy_move_half = movement.z_corrected_needle_movement(-z_distance / 2, self.stage.current_position.t)
        needle.relative_move(zy_move_half)
        logging.info(f"{self.current_status.name}: needle z-half-move: {zy_move_half}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.update_display(beam_type=BeamType.ION, image_type='new')

        self.update_popup_settings(message='Is the needle safe to move another half step?', click=None, filter_strength=self.filter_strength)
        self.ask_user(image=self.image_FIB)
        # TODO: crosshairs here?
        if self.response:
            self.image_settings['hfw'] = self.settings['reference_images']['needle_with_lamella_shifted_img_highres']
            distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='needle_tip_to_lamella_centre', beamType=BeamType.ION)

            # calculate shift in xyz coordinates

            z_distance = distance_y_m / np.cos(self.stage.current_position.t)

            # Calculate movement
            # move in x
            x_move = movement.x_corrected_needle_movement(-distance_x_m)
            self.needle.relative_move(x_move)
            
            # move in z
             # TODO: might want to make this /3 or /4 or add a constant factor to make sure it lands
            # detection is based on centre of lamella, we want to land of the edge.
            # therefore subtract half the height from the movement.
            lamella_height = self.settings["lamella"]["lamella_height"]
            gap = lamella_height / 2 # 0.5e-6
            zy_move_gap = movement.z_corrected_needle_movement(-(z_distance - gap), self.stage.current_position.t)
            self.needle.relative_move(zy_move_gap)

            logging.info(f"{self.current_status.name}: needle x-move: {x_move}")
            logging.info(f"{self.current_status.name}: needle zy-move: {zy_move_gap}")


            self.image_settings['save'] = True
            self.image_settings['autocontrast'] = self.USE_AUTOCONTRAST
            self.image_settings['hfw'] = self.settings["reference_images"]["needle_ref_img_hfw_lowres"]
            self.image_settings['label'] = 'needle_ref_img_lowres'
            acquire.take_reference_images(self.microscope, self.image_settings)
            self.image_settings['hfw'] = self.settings["reference_images"]["needle_ref_img_hfw_highres"]
            self.image_settings['label'] = 'needle_ref_img_highres'
            acquire.take_reference_images(self.microscope, self.image_settings)
            logging.info(f"{self.current_status.name}: land needle on lamella complete.")
        else:
            logging.warning(f"{self.current_status.name}: needle not safe to move onto lamella.")
            logging.warning(f"{self.current_status.name}: needle landing cancelled by user.")
            return

    def calculate_shift_distance_metres(self, shift_type, beamType=BeamType.ELECTRON):
        self.image_settings['beam_type'] = beamType
        self.raw_image, self.overlay_image, self.downscaled_image, feature_1_px, feature_1_type, feature_2_px, feature_2_type = \
            calibration.identify_shift_using_machine_learning(self.microscope, self.image_settings, self.settings, self.liftout_counter,
                                                              shift_type=shift_type)
        feature_1_px, feature_2_px = self.validate_detection(feature_1_px=feature_1_px, feature_1_type=feature_1_type, feature_2_px=feature_2_px, feature_2_type=feature_2_type)
        # scaled features
        scaled_feature_1_px = detection_utils.scale_invariant_coordinates(feature_1_px, self.overlay_image) #(y, x)
        scaled_feature_2_px = detection_utils.scale_invariant_coordinates(feature_2_px, self.overlay_image) # (y, x)
        # TODO: check x/y here
        # distance
        distance_x = scaled_feature_2_px[1] - scaled_feature_1_px[1]
        distance_y = scaled_feature_2_px[0] - scaled_feature_1_px[0]
        x_pixel_size = self.raw_image.metadata.binary_result.pixel_size.x
        y_pixel_size = self.raw_image.metadata.binary_result.pixel_size.y
        # distance in metres
        distance_x_m = x_pixel_size * self.raw_image.width * distance_x
        distance_y_m = y_pixel_size * self.raw_image.height * distance_y
        return distance_x_m, distance_y_m

    def validate_detection(self, feature_1_px=None, feature_1_type=None, feature_2_px=None, feature_2_type=None):
        self.update_popup_settings(message=f'Has the model correctly identified the {feature_1_type} and {feature_2_type} positions?', click=None, crosshairs=False)
        self.ask_user(image=self.overlay_image)

        # if something wasn't correctly identified
        if not self.response:
            utils.save_image(image=self.raw_image, save_path=self.image_settings['save_path'], label=self.image_settings['label'] + '_label')

            self.update_popup_settings(message=f'Has the model correctly identified the {feature_1_type} position?', click=None, crosshairs=False)
            self.ask_user(image=self.overlay_image)

            # if feature 1 wasn't correctly identified
            if not self.response:

                self.update_popup_settings(message=f'Please click on the correct {feature_1_type} position.'
                                                                   f'Press Yes button when happy with the position', click='single', crosshairs=False)
                self.ask_user(image=self.downscaled_image)

                # if new feature position selected
                if self.response:
                    # TODO: check x/y here
                    feature_1_px = (self.yclick, self.xclick)

            self.update_popup_settings(message=f'Has the model correctly identified the {feature_2_type} position?', click=None, crosshairs=False)
            self.ask_user(image=self.overlay_image)

            # if feature 2 wasn't correctly identified
            if not self.response:

                self.update_popup_settings(message=f'Please click on the correct {feature_2_type} position.'
                                                                   f'Press Yes button when happy with the position', click='single', filter_strength=self.filter_strength, crosshairs=False)
                self.ask_user(image=self.downscaled_image)
                # TODO: do we want filtering on this image?

                # if new feature position selected
                if self.response:
                    feature_2_px = (self.yclick, self.xclick)

        return feature_1_px, feature_2_px

    def land_lamella(self, landing_coord, original_landing_images):

        logging.info(f"{self.current_status.name}: land lamella stage started")
        self.current_status = AutoLiftoutStatus.Landing

        stage_settings = MoveSettings(rotate_compucentric=True)
        self.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)

        # move to landing coordinate
        self.stage.absolute_move(landing_coord)
        # TODO: image settings?
        calibration.correct_stage_drift(self.microscope, self.image_settings, original_landing_images, self.liftout_counter, mode="land")
        logging.info(f"{self.current_status.name}: initial landing calibration complete.")
        park_position = movement.move_needle_to_landing_position(self.microscope)
        logging.info(f"{self.current_status.name}: needle inserted to park_position: {park_position}")

        # # Y-MOVE
        self.image_settings["resolution"] = self.settings["reference_images"]["landing_post_ref_img_resolution"]
        self.image_settings["dwell_time"] = self.settings["reference_images"]["landing_post_ref_img_dwell_time"]
        self.image_settings["hfw"]  = 150e-6 # self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
        self.image_settings["beam_type"] = BeamType.ELECTRON
        self.image_settings["save"] = True
        self.image_settings["label"] = "landing_needle_land_sample_lowres"

        # needle_eb_lowres, needle_ib_lowres = acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=self.image_settings["beam_type"])

        y_move = movement.y_corrected_needle_movement(-distance_y_m, self.stage.current_position.t)
        self.needle.relative_move(y_move)
        logging.info(f"{self.current_status.name}: y-move complete: {y_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")


        # Z-MOVE
        self.image_settings["label"] = "landing_needle_land_sample_lowres_after_y_move"
        self.image_settings["beam_type"] = BeamType.ION
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=self.image_settings["beam_type"])

        z_distance = distance_y_m / np.sin(np.deg2rad(52))
        z_move = movement.z_corrected_needle_movement(z_distance, self.stage.current_position.t)
        self.needle.relative_move(z_move)
        logging.info(f"{self.current_status.name}: z-move complete: {z_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")


        ## X-HALF-MOVE
        self.image_settings["label"] = "landing_needle_land_sample_lowres_after_z_move"
        self.image_settings["beam_type"] = BeamType.ELECTRON
        self.image_settings["hfw"] = 150e-6
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=self.image_settings["beam_type"])

        # half move
        x_move = movement.x_corrected_needle_movement(distance_x_m / 2)
        self.needle.relative_move(x_move)
        logging.info(f"{self.current_status.name}: x-half-move complete: {x_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")

        ## X-MOVE
        self.image_settings["label"] = "landing_needle_land_sample_lowres_after_z_move"
        self.image_settings["beam_type"] = BeamType.ELECTRON
        self.image_settings["hfw"] = 80e-6
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=self.image_settings["beam_type"])

        # TODO: gap?
        x_move = movement.x_corrected_needle_movement(distance_x_m)
        self.needle.relative_move(x_move)
        logging.info(f"{self.current_status.name}: x-move complete: {x_move}")

        # final reference images
        self.image_settings["save"] = True
        self.image_settings["hfw"]  = 80e-6 # self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
        self.image_settings["label"] = "E_landing_lamella_final_weld_highres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        # WELD TO LANDING POST
        # TODO: this is not joining the lamella to the post
        milling.weld_to_landing_post(self.microscope)
        self.update_display(beam_type=BeamType.ION, image_type='last')
        # TODO: return image with patterning marks

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False)
        self.ask_user(image=self.image_FIB)
        if self.response:
            logging.info(f"{self.current_status.name}: welding to post started.")
            milling.run_milling(self.microscope, self.settings)
        self.microscope.patterning.mode = 'Serial'
        logging.info(f"{self.current_status.name}: weld to post complete")

        # final reference images

        self.image_settings["hfw"]  = 100e-6 # self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
        self.image_settings["label"] = "landing_lamella_final_weld_highres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")


        # CUT OFF NEEDLE
        logging.info("landing: start cut off needle. detecting needle distance from centre.")
        self.image_settings["hfw"] = self.settings["cut"]["hfw"]
        self.image_settings["label"] = "landing_lamella_pre_cut_off"
        self.image_settings["beam_type"] = BeamType.ION
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=self.image_settings["beam_type"])

        height = self.settings["cut"]["height"]
        width = self.settings["cut"]["width"]
        depth = self.settings["cut"]["depth"]
        rotation = self.settings["cut"]["rotation"]
        hfw = self.settings["cut"]["hfw"]

        cut_coord = {"center_x": -distance_x_m,
                     "center_y": distance_y_m,
                     "width": width,
                     "height": height,
                     "depth": depth,  # TODO: might need more to get through needle
                     "rotation": rotation, "hfw": hfw}  # TODO: check rotation

        logging.info(f"{self.current_status.name}: calculating needle cut-off pattern")

        # cut off needle tip
        milling.cut_off_needle(self.microscope, cut_coord=cut_coord)
        self.update_display(beam_type=BeamType.ION, image_type='last')
        # TODO: return image with patterning marks

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False)
        self.ask_user(image=self.image_FIB)
        if self.response:
            logging.info(f"{self.current_status.name}: needle cut-off started")
            milling.run_milling(self.microscope, self.settings)
        self.microscope.patterning.mode = 'Serial'

        logging.info(f"{self.current_status.name}: needle cut-off complete")

        # reference images
        self.image_settings["hfw"]  = 150e-6 # self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
        self.image_settings["label"] = "landing_lamella_final_cut_lowres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)


        self.image_settings["hfw"]  = 80e-6 # self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
        self.image_settings["label"] = "landing_lamella_final_cut_highres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        logging.info(f"{self.current_status.name}: removing needle from landing post")
        # move needle out of trench slowly at first
        for i in range(3):
            z_move_out_from_post = movement.z_corrected_needle_movement(10e-6, self.stage.current_position.t)
            self.needle.relative_move(z_move_out_from_post)
            self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
            self.update_display(beam_type=BeamType.ION, image_type="new")
            logging.info(f"{self.current_status.name}: moving needle out: {z_move_out_from_post} ({i+1} / 3")
            time.sleep(1)

        # move needle to park position
        movement.retract_needle(self.microscope, park_position)
        logging.info(f"{self.current_status.name}: needle retracted.")

        # reference images
        self.image_settings["hfw"]  = 150e-6 #TODO: fix protocol
        self.image_settings["label"] = "landing_lamella_final_lowres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.image_settings["hfw"]  = 80e-6  #TODO: fix protocol
        self.image_settings["label"] = "landing_lamella_final_highres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        logging.info(f"{self.current_status.name}: landing stage complete")


    def reset_needle(self):

        self.current_status = AutoLiftoutStatus.Reset
        logging.info(f"{self.current_status.name}: reset stage started")

        # move sample stage out
        movement.move_sample_stage_out(self.microscope)
        logging.info(f"{self.current_status.name}: moved sample stage out")

        # move needle in
        park_position = movement.insert_needle(self.microscope)
        z_move_in = movement.z_corrected_needle_movement(-180e-6, self.stage.current_position.t)
        self.needle.relative_move(z_move_in)
        logging.info(f"{self.current_status.name}: insert needle for reset")

        # needle images
        self.image_settings["resolution"] = self.settings["imaging"]["resolution"]
        self.image_settings["dwell_time"] = self.settings["imaging"]["dwell_time"]
        self.image_settings["hfw"] = self.settings["imaging"]["horizontal_field_width"]
        self.image_settings["save"] = True
        self.image_settings["label"] = "sharpen_needle_initial"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")
        self.image_settings["beam_type"] = BeamType.ION

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=self.image_settings["beam_type"])

        x_move = movement.x_corrected_needle_movement(distance_x_m)
        self.needle.relative_move(x_move)
        z_distance = distance_y_m / np.sin(np.deg2rad(52))
        z_move = movement.z_corrected_needle_movement(z_distance, self.stage.current_position.t)
        self.needle.relative_move(z_move)
        logging.info(f"{self.current_status.name}: moving needle to centre: x_move: {x_move}, z_move: {z_move}")

        self.image_settings["label"] = "sharpen_needle_centre"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=self.image_settings["beam_type"])

        # create sharpening patterns
        cut_coord_bottom, cut_coord_top = milling.calculate_sharpen_needle_pattern(microscope=self.microscope, settings=self.settings, x_0=distance_x_m, y_0=distance_y_m)
        logging.info(f"{self.current_status.name}: calculate needle sharpen pattern")

        milling.create_sharpen_needle_patterns(
            self.microscope, cut_coord_bottom, cut_coord_top
        )

        # confirm and run milling
        self.update_display(beam_type=BeamType.ION, image_type='last')
        # TODO: return image with patterning marks

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False)
        self.ask_user(image=self.image_FIB)
        if self.response:
            logging.info(f"{self.current_status.name}: needle sharpening milling started")
            milling.run_milling(self.microscope, self.settings)
        self.microscope.patterning.mode = 'Serial'
        logging.info(f"{self.current_status.name}: needle sharpening milling complete")


        # take reference images
        self.image_settings["label"] = "sharpen_needle_final"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        # retract needle
        movement.retract_needle(self.microscope, park_position)
        logging.info(f"{self.current_status.name}: needle retracted")
        logging.info(f"{self.current_status.name}: reset stage complete")

    def cleanup_lamella(self, landing_coord):
        """Cleanup: Thin the lamella thickness to size for imaging."""
        
        self.current_status = AutoLiftoutStatus.Cleanup
        logging.info(f"{self.current_status.name}: cleanup stage started")

        # move to landing coord
        self.microscope.specimen.stage.absolute_move(landing_coord)
        logging.info(f"{self.current_status.name}: move to landing coordinates: {landing_coord}")

        # tilt to 0 rotate 180 move to 21 deg
        # tilt to zero, to prevent hitting anything
        stage_settings = MoveSettings(rotate_compucentric=True)
        self.microscope.specimen.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)

        # thinning position # TODO: add to protocol
        thinning_rotation_angle = 180
        thinning_tilt_angle = 21

        # rotate to thinning angle
        self.microscope.specimen.stage.relative_move(StagePosition(r=np.deg2rad(thinning_rotation_angle)), stage_settings)

        # tilt to thinning angle
        self.microscope.specimen.stage.absolute_move(StagePosition(t=np.deg2rad(thinning_tilt_angle)), stage_settings)
        logging.info(f"{self.current_status.name}: rotate to thinning angle: {thinning_rotation_angle}")
        logging.info(f"{self.current_status.name}: tilt to thinning angle: {thinning_tilt_angle}")

        # lamella images # TODO: check and add to protocol
        self.image_settings["resolution"] = self.settings["imaging"]["resolution"]
        self.image_settings["dwell_time"] = self.settings["imaging"]["dwell_time"]
        self.image_settings["label"] = f"thinning_lamella_21deg_tilt_{self.current_sample.sample_no}"
        self.image_settings["save"] = True
        self.image_settings["hfw"] = 100e-6
        acquire.take_reference_images(self.microscope, self.image_settings)

        # TODO: OLD REMOVE
        # resolution = self.settings["imaging"]["resolution"]
        # dwell_time = self.settings["imaging"]["dwell_time"]
        # image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
        # hfw_lowres = 100e-6  # storage.settings["imaging"]["horizontal_field_width"]
        #
        # lamella_eb, lamella_ib = take_electron_and_ion_reference_images(
        #     microscope, hor_field_width=hfw_lowres, image_settings=image_settings,
        #     save=True, save_label="A_thinning_lamella_21deg_tilt")


        # realign lamella to image centre
        self.image_settings['label'] = f'{self.liftout_counter:02d}_drift_correction_ML_cleanup'
        self.image_settings["resolution"] ="1536x1024" #TODO: add to protocol
        self.image_settings["dwell_time"] = 0.5e-6
        self.correct_stage_drift_with_ML()

        # image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=0.5e-6)  # TODO: user input resolution
        # realign_eucentric_with_machine_learning(microscope, image_settings=image_settings_ML, hor_field_width=100e-6)

        # # LAMELLA EDGE TO IMAGE CENTRE?
        # # x-movement
        # storage.step_counter += 1
        # lamella_eb, lamella_ib = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6,
        #                                                                 image_settings=image_settings,
        #                                                                 save=True, save_label="A_lamella_pre_thinning")

        self.image_settings["label"] = f"cleanup_lamella_pre_cleanup_{self.current_sample}"
        self.image_settings["save"] = True
        self.image_settings["hfw"] = 80e-6
        acquire.take_reference_images(self.microscope, self.image_settings)

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=BeamType.ION)

        # x_shift, y_shift = calculate_shift_between_features_in_metres(lamella_ib, "lamella_edge_to_landing_post")

        # z-movement (shouldnt really be needed if eucentric calibration is correct)
        z_distance = distance_y_m / np.sin(np.deg2rad(52))
        z_move = movement.z_corrected_stage_movement(z_distance, self.stage.current_position.t)
        self.stage.relative_move(z_move)

        # x-move the rest of the way
        x_move = movement.x_corrected_stage_movement(-distance_x_m)
        self.stage.relative_move(x_move)

        # move half the width of lamella to centre the edge..
        width = self.settings["thin_lamella"]["lamella_width"]
        x_move_half_width = movement.x_corrected_stage_movement(-width / 2)
        self.stage.relative_move(x_move_half_width)

        # mill thin lamella pattern
        self.update_popup_settings(message="Run lamella thinning?", crosshairs=False)
        self.ask_user()
        if self.response:
            milling.mill_thin_lamella(self.microscope, self.settings)

        # take reference images and finish
        self.image_settings["hfw"] = 800e-6
        self.image_settings["save"] = True
        self.image_settings["label"] = f"cleanup_lamella_post_cleanup_{self.current_sample.sample_no}"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        logging.info(f"{self.current_status.name}: thin lamella {self.current_sample.sample_no} complete.")

    def initialise_image_frames(self):
        self.figure_SEM = plt.figure()
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.01)
        self.canvas_SEM = _FigureCanvas(self.figure_SEM)
        self.toolbar_SEM = _NavigationToolbar(self.canvas_SEM, self)
        self.label_SEM.setLayout(QtWidgets.QVBoxLayout())
        self.label_SEM.layout().addWidget(self.toolbar_SEM)
        self.label_SEM.layout().addWidget(self.canvas_SEM)

        # self.canvas_SEM.mpl_connect('button_press_event', lambda event: self.on_gui_click(event, beam_type=BeamType.ELECTRON, click=None))
        # TODO: if grid not centred before initialise allow click = double temporarily

        self.figure_FIB = plt.figure()
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.01)
        self.canvas_FIB = _FigureCanvas(self.figure_FIB)
        self.toolbar_FIB = _NavigationToolbar(self.canvas_FIB, self)
        self.label_FIB.setLayout(QtWidgets.QVBoxLayout())
        self.label_FIB.layout().addWidget(self.toolbar_FIB)
        self.label_FIB.layout().addWidget(self.canvas_FIB)

        # self.canvas_FIB.mpl_connect('button_press_event', lambda event: self.on_gui_click(event, beam_type=BeamType.ION, click=None))

    def initialize_hardware(self, offline=False):
        if offline is False:
            self.connect_to_microscope(ip_address=self.ip_address)
        elif offline is True:
            pass
            # self.connect_to_microscope(ip_address='localhost')

    def setup_connections(self):
        # Protocol and information table connections
        self.pushButton_initialise.clicked.connect(lambda: self.initialise_autoliftout())
        self.pushButton_autoliftout.clicked.connect(lambda: self.run_liftout())

        self.pushButton_Protocol_Load.clicked.connect(lambda: self.load_yaml())
        self.pushButton_Protocol_New.clicked.connect(lambda: self.new_protocol())
        self.pushButton_Protocol_Delete.clicked.connect(lambda: self.delete_protocol())
        self.pushButton_Protocol_Save.clicked.connect(lambda: self.save_protocol(self.save_destination))
        self.pushButton_Protocol_Save_As.clicked.connect(lambda: self.save_protocol())
        self.pushButton_Protocol_Rename.clicked.connect(lambda: self.rename_protocol())
        self.tabWidget_Protocol.tabBarDoubleClicked.connect(lambda: self.rename_protocol())
        self.tabWidget_Protocol.tabBar().tabMoved.connect(lambda: self.tab_moved('protocol'))
        self.tabWidget_Information.tabBar().tabMoved.connect(lambda: self.tab_moved('information'))

        self.button_last_image_FIB.clicked.connect(lambda: self.update_display(beam_type=BeamType.ION, image_type='last'))

        # FIBSEM methods
        self.connect_microscope.clicked.connect(lambda: self.connect_to_microscope(ip_address=self.ip_address))

        self.pushButton_load_sample_data.clicked.connect(lambda: self.load_coords())

        self.pushButton_test_popup.clicked.connect(lambda: self.update_popup_settings(click='single'))
        # self.pushButton_test_popup.clicked.connect(lambda: self.ask_user(image=test_image, second_image=test_image))
        self.pushButton_test_popup.clicked.connect(lambda: self.ask_user(image=test_image))

    def ask_user(self, image=None, second_image=None):

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
        button_box.setFixedHeight(int(self.button_height*1.2))
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
                    self.hfw_slider.setMaximum(2700)
                else:
                    self.hfw_slider.setMaximum(900)
                self.hfw_slider.setValue(self.image_settings['hfw']*1e6)

                # spinbox (not a property as only slider value needed)
                hfw_spinbox = QtWidgets.QSpinBox()
                hfw_spinbox.setMinimum(1)
                if beam_type == BeamType.ELECTRON:
                    hfw_spinbox.setMaximum(2700)
                else:
                    hfw_spinbox.setMaximum(900)
                hfw_spinbox.setValue(self.image_settings['hfw'] * 1e6)

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

            # TODO: account for HFW and New image being together
            # connect button to functions
            self.new_image.clicked.connect(lambda: self.image_settings.update(
                {'hfw': self.hfw_slider.value()*1e-6}))
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

            if self.popup_settings['click_crosshair']:
                for patch in self.popup_settings['click_crosshair']:
                    self.ax.add_patch(patch)
            self.popup_canvas.draw()

            self.popup_toolbar = _NavigationToolbar(self.popup_canvas, self)
            self.popup_window.layout().addWidget(self.popup_toolbar, 1, 1, 1, 1)
            self.popup_window.layout().addWidget(self.popup_canvas, 2, 1, 4, 1)

    def update_popup_settings(self, message='default message', allow_new_image=False, click=None, filter_strength=0, crosshairs=True):
        self.popup_settings["message"] = message
        self.popup_settings['allow_new_image'] = allow_new_image
        self.popup_settings['click'] = click
        self.popup_settings['filter_strength'] = filter_strength
        self.popup_settings['crosshairs'] = crosshairs

    def on_gui_click(self, event):
        click = self.popup_settings['click']
        image = self.popup_settings['image']
        beam_type = self.image_settings['beam_type']

        if event.inaxes.get_title() != 'Previous Image':
            if event.button == 1:
                if event.dblclick and (click in ('double', 'all')):
                    if image:
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
                        # TODO: refactor beam type here
                        self.popup_settings['image'] = acquire.new_image(
                            microscope=self.microscope,
                            settings=self.image_settings)
                        if beam_type:
                            self.update_display(beam_type=beam_type,
                                                image_type='last')
                        self.update_popup_display()

                elif click in ('single', 'all'):
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
        self.setEnabled(True)

    def new_protocol(self):
        num_index = self.tabWidget_Protocol.__len__() + 1

        # new tab for protocol
        new_protocol_tab = QtWidgets.QWidget()
        new_protocol_tab.setObjectName(f'Protocol {num_index}')
        layout_protocol_tab = QtWidgets.QGridLayout(new_protocol_tab)
        layout_protocol_tab.setObjectName(f'gridLayout_{num_index}')

        # new text edit to hold protocol
        protocol_text_edit = QtWidgets.QTextEdit()
        font = QtGui.QFont()
        font.setPointSize(10)
        protocol_text_edit.setFont(font)
        protocol_text_edit.setObjectName(f'protocol_text_edit_{num_index}')
        layout_protocol_tab.addWidget(protocol_text_edit, 0, 0, 1, 1)

        self.tabWidget_Protocol.addTab(new_protocol_tab, f'Protocol {num_index}')
        self.tabWidget_Protocol.setCurrentWidget(new_protocol_tab)
        self.load_template_protocol()

        #

        # new tab for information from FIBSEM
        new_information_tab = QtWidgets.QWidget()
        new_information_tab.setObjectName(f'Protocol {num_index}')
        layout_new_information_tab = QtWidgets.QGridLayout(new_information_tab)
        layout_new_information_tab.setObjectName(f'layout_new_information_tab_{num_index}')

        # new empty table for information to fill
        new_table_widget = QtWidgets.QTableWidget()
        new_table_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        new_table_widget.setAlternatingRowColors(True)
        new_table_widget.setObjectName(f'tableWidget_{num_index}')
        new_table_widget.setColumnCount(len(information_keys))
        new_table_widget.setRowCount(starting_positions)

        # set up rows
        for row in range(starting_positions):
            table_item = QtWidgets.QTableWidgetItem()
            new_table_widget.setVerticalHeaderItem(row, table_item)
            item = new_table_widget.verticalHeaderItem(row)
            item.setText(_translate('MainWindow', f'Position {row+1}'))

        # set up columns
        for column in range(len(information_keys)):
            table_item = QtWidgets.QTableWidgetItem()
            new_table_widget.setHorizontalHeaderItem(column, table_item)
            item = new_table_widget.horizontalHeaderItem(column)
            item.setText(_translate('MainWindow', information_keys[column]))

        new_table_widget.horizontalHeader().setDefaultSectionSize(174)
        new_table_widget.horizontalHeader().setHighlightSections(True)

        layout_new_information_tab.addWidget(new_table_widget, 0, 0, 1, 1)
        self.tabWidget_Information.addTab(new_information_tab, f'Protocol {num_index}')
        self.tabWidget_Information.setCurrentWidget(new_information_tab)

    def rename_protocol(self):
        index = self.tabWidget_Protocol.currentIndex()

        top_margin = 4
        left_margin = 10

        rect = self.tabWidget_Protocol.tabBar().tabRect(index)
        self.edit = QtWidgets.QLineEdit(self.tabWidget_Protocol)
        self.edit.move(rect.left() + left_margin, rect.top() + top_margin)
        self.edit.resize(rect.width() - 2 * left_margin, rect.height() - 2 * top_margin)
        self.edit.show()
        self.edit.setFocus()
        self.edit.selectAll()
        self.edit.editingFinished.connect(lambda: self.finish_rename())

    def finish_rename(self):
        self.tabWidget_Protocol.setTabText(self.tabWidget_Protocol.currentIndex(), self.edit.text())
        tab = self.tabWidget_Protocol.currentWidget()
        tab.setObjectName(self.edit.text())

        self.tabWidget_Information.setTabText(self.tabWidget_Information.currentIndex(), self.edit.text())
        tab2 = self.tabWidget_Information.currentWidget()
        tab2.setObjectName(self.edit.text())
        self.edit.deleteLater()

    def delete_protocol(self):
        index = self.tabWidget_Protocol.currentIndex()
        self.tabWidget_Information.removeTab(index)
        self.tabWidget_Protocol.removeTab(index)
        pass

    def load_template_protocol(self):
        with open(protocol_template_path, 'r') as file:
            _dict = yaml.safe_load(file)

        for key, value in _dict.items():
            self.key_list_protocol.append(key)
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if key2 not in self.key_list_protocol:
                        self.key_list_protocol.append(key)
                if isinstance(value, dict):
                    for key3, value3 in value.items():
                        if key3 not in self.key_list_protocol:
                            self.key_list_protocol.append(key)
                elif isinstance(value, list):
                    for dictionary in value:
                        for key4, value4 in dictionary.items():
                            if key4 not in self.key_list_protocol:
                                self.key_list_protocol.append(key)

        self.load_protocol_text(_dict)

    def load_protocol_text(self, dictionary):
        protocol_text = str()
        count = 0
        jcut = 0
        _dict = dictionary

        for key in _dict:
            if key not in self.key_list_protocol:
                logging.info(f'Unexpected parameter in template file')
                return

        for key, value in _dict.items():
            if type(value) is dict:
                if jcut == 1:
                    protocol_text += f'{key}:'  # jcut
                    jcut = 0
                else:
                    protocol_text += f'\n{key}:'  # first level excluding jcut
                for key2, value2 in value.items():
                    if type(value2) is list:
                        jcut = 1
                        protocol_text += f'\n  {key2}:'  # protocol stages
                        for item in value2:
                            if count == 0:
                                protocol_text += f'\n    # rough_cut'
                                count = 1
                                count2 = 0
                            else:
                                protocol_text += f'    # regular_cut'
                                count = 0
                                count2 = 0
                            protocol_text += f'\n    -'
                            for key6, value6 in item.items():
                                if count2 == 0:
                                    protocol_text += f' {key6}: {value6}\n'  # first after list
                                    count2 = 1
                                else:
                                    protocol_text += f'      {key6}: {value6}\n'  # rest of list
                    else:
                        protocol_text += f'\n  {key2}: {value2}'  # values not in list
            else:
                protocol_text += f'{key}: {value}'  # demo mode
        self.tabWidget_Protocol.currentWidget().findChild(QtWidgets.QTextEdit).setText(protocol_text)

    def save_protocol(self, destination=None):
        dest = destination
        if dest is None:
            dest = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder to save protocol in')
            index = self.tabWidget_Protocol.currentIndex()
            tab = self.tabWidget_Protocol.tabBar().tabText(index)
            dest = f'{dest}/{tab}.yml'
        protocol_text = self.tabWidget_Protocol.currentWidget().findChild(QtWidgets.QTextEdit).toPlainText()
        # protocol_text = self.tabWidget_Protocol.currentWidget().findChild(QtWidgets.QTextEdit).toMarkdown()
        logging.info(protocol_text)
        p = yaml.safe_load(protocol_text)
        logging.info(p)
        with open(dest, 'w') as file:
            yaml.dump(p, file, sort_keys=False)
        self.save_destination = dest
        # save

    def load_yaml(self):
        """Ask the user to choose a protocol file to load

        Returns
        -------
        str
            Path to file for parameter loading
        """
        checked = 0
        logging.info(f'Please select protocol file (yml)')
        root = Tk()
        root.withdraw()
        _dict = None
        while not checked:
            checked = 1
            try:
                load_directory = filedialog.askopenfile(mode='r', filetypes=[('yml files', '*.yml')])
                if load_directory is None:
                    return

                while not load_directory.name.endswith('.yml'):
                    logging.info('Not a yml configuration file')
                    load_directory = filedialog.askopenfile(mode='r', filetypes=[('yml files', '*.yml')])

                with open(load_directory.name, 'r') as file:
                    _dict = yaml.safe_load(file)
                    file.close()
            except Exception:
                display_error_message(traceback.format_exc())
            for key in _dict:
                if key not in self.key_list_protocol:
                    if checked:
                        logging.info(f'Unexpected parameter in protocol file')
                    checked = 0
        root.destroy()
        logging.info(_dict)

        self.load_protocol_text(_dict)

    def tab_moved(self, moved):
        if moved == 'protocol':
            self.tabWidget_Information.tabBar().moveTab(self.tabWidget_Information.currentIndex(), self.tabWidget_Protocol.currentIndex())
        elif moved == 'information':
            self.tabWidget_Protocol.tabBar().moveTab(self.tabWidget_Protocol.currentIndex(), self.tabWidget_Information.currentIndex())

    def random_data(self):
        for parameter in range(len(self.params)):
            self.params[information_keys[parameter]] = np.random.randint(0, 1)
        self.fill_information()
        self.position = (self.position + 1) % self.tabWidget_Information.currentWidget().findChild(QtWidgets.QTableWidget).rowCount()

    def fill_information(self):
        information = self.tabWidget_Information.currentWidget().findChild(QtWidgets.QTableWidget)
        row = self.position

        for column in range(len(information_keys)):
            item = QtWidgets.QTableWidgetItem()
            item.setText(str(self.params[information_keys[column]]))
            information.setItem(row, column, item)

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
                    self.figure_SEM.patch.set_facecolor((240/255, 240/255, 240/255))
                    self.ax_SEM = self.figure_SEM.add_subplot(111)
                    self.ax_SEM.get_xaxis().set_visible(False)
                    self.ax_SEM.get_yaxis().set_visible(False)
                    self.ax_SEM.imshow(image_array, cmap='gray')
                    self.canvas_SEM.draw()
                if beam_type is BeamType.ION:
                    if image_type == 'new':
                        self.image_settings['beam_type'] = beam_type
                        self.image_FIB = acquire.new_image(self.microscope, self.image_settings)
                    else:
                        self.image_FIB = acquire.last_image(self.microscope, beam_type=beam_type)
                    image_array = self.image_FIB.data
                    self.figure_FIB.clear()
                    self.figure_FIB.patch.set_facecolor((240/255, 240/255, 240/255))
                    self.ax_FIB = self.figure_FIB.add_subplot(111)
                    self.ax_FIB.get_xaxis().set_visible(False)
                    self.ax_FIB.get_yaxis().set_visible(False)
                    self.ax_FIB.imshow(image_array, cmap='gray')
                    self.canvas_FIB.draw()

            except Exception:
                display_error_message(traceback.format_exc())

    def disconnect(self):
        logging.info('Running cleanup/teardown')
        logging.debug('Running cleanup/teardown')
        if self.objective_stage and self.offline is False:
            # Return objective lens stage to the 'out' position and disconnect.
            self.move_absolute_objective_stage(self.objective_stage, position=0)
            self.objective_stage.disconnect()
        if self.microscope:
            self.microscope.disconnect()


def display_error_message(message):
    """PyQt dialog box displaying an error message."""
    logging.info('display_error_message')
    logging.exception(message)
    error_dialog = QtWidgets.QErrorMessage()
    error_dialog.showMessage(message)
    error_dialog.exec_()
    return error_dialog


def main(offline='True'):
    if offline.lower() == 'false':
        # logging.basicConfig(level=logging.DEBUG)
        launch_gui(ip_address='10.0.0.1', offline=False)
    elif offline.lower() == 'true':
        # logging.basicConfig(level=logging.DEBUG)
        with mock.patch.dict('os.environ', {'PYLON_CAMEMU': '1'}):
            try:
                launch_gui(ip_address='localhost', offline=True)
            except Exception:
                import pdb
                traceback.print_exc()
                pdb.set_trace()


def launch_gui(ip_address='10.0.0.1', offline=False):
    """Launch the `autoliftout` main application window."""
    app = QtWidgets.QApplication([])
    qt_app = GUIMainWindow(ip_address=ip_address, offline=offline)
    app.aboutToQuit.connect(qt_app.disconnect)  # cleanup & teardown
    qt_app.show()
    sys.exit(app.exec_())


# main(offline='False')
# main(offline='True')


# TODO: use this instead of above,
if __name__ == "__main__":
    offline_mode = "True"  # TODO: change offline to bool not str
    main(offline=offline_mode)
