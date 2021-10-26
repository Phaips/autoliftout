import datetime
import logging
import sys
import time
import traceback
from re import M
from tkinter import Tk, filedialog

import matplotlib
import matplotlib.pyplot as plt
import mock
import numpy as np
import yaml
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

from liftout.gui.DraggablePatch import  DraggablePatch
matplotlib.use('Agg')
import os
from enum import Enum

import liftout
import PIL
import scipy.ndimage as ndi
import skimage
from autoscript_sdb_microscope_client.structures import *
from liftout.fibsem.sample import Sample
from PIL import Image

# Required to not break imports
BeamType = acquire.BeamType

# test_image = PIL.Image.open('C:/Users/David/images/mask_test.tif')
test_image = np.random.randint(0, 255, size=(1024, 1536), dtype='uint16')
test_image = np.array(test_image)
# test_image = np.zeros_like(test_image, dtype='uint16')
test_jcut = [(0.e-6, 200.e-6, 200.e-6, 30.e-6), (100.e-6, 175.e-6, 30.e-6, 100.e-6), (-100.e-6, 0.e-6, 30.e-6, 400.e-6)]


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
    Thinning = 5
    Finished = 6

class GUIMainWindow(gui_main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, ip_address='10.0.0.1', offline=False):
        super(GUIMainWindow, self).__init__()

        # setup logging
        self.save_path = utils.make_logging_directory(prefix="run")
        self.log_path = utils.configure_logging(save_path=self.save_path, log_filename='logfile_')
        config_filename = os.path.join(os.path.dirname(liftout.__file__),"protocol_liftout.yml")

        # load config
        self.settings = utils.load_config(config_filename)
        self.pretilt_degrees = self.settings["system"]["pretilt_angle"]
        assert self.pretilt_degrees == 27  # TODO: remove this once this has been cleaned up in other files

        self.current_status = AutoLiftoutStatus.Initialisation
        logging.info(f"{self.current_status.name} STARTED")
        logging.info(f"gui: starting in {'offline' if offline else 'online'} mode")

        # TODO: replace "SEM, FIB" with BeamType calls
        self.offline = offline
        self.setupUi(self)
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

        # initialise hardware
        self.ip_address = ip_address
        self.microscope = None

        self.initialize_hardware(offline=offline)

        if self.microscope:
            self.stage = self.microscope.specimen.stage
            self.needle = self.microscope.specimen.manipulator

        self.samples = []
        self.current_sample = None

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

        self.current_status = AutoLiftoutStatus.Initialisation

        # setup status information
        self.status_timer = QtCore.QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)

        # setup status labels
        self.label_status_1.setStyleSheet("background-color: coral; padding: 10px")
        self.label_status_1.setFont(QtGui.QFont("Arial", 14, weight=QtGui.QFont.Bold))
        self.label_status_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_status_2.setStyleSheet("background-color: coral; padding: 10px")
        self.label_status_3.setStyleSheet("background-color: black;  color: white; padding:10px")
        self.update_status()

        self.setup_connections()

        logging.info(f"{self.current_status.name} FINISHED")

    def initialise_autoliftout(self):
        # TODO: check if needle i
        self.current_status = AutoLiftoutStatus.Setup
        logging.info(f"{self.current_status.name} STARTED")

        # TODO: add to protocol
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=2750e-6,
            beam_type=BeamType.ELECTRON,
            save=True,
            label='grid',
        )

        # move to the initial sample grid position
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
            self.update_image_settings(hfw=2750e-6, save=True, label='grid_Pt_deposition')
            self.update_display(beam_type=BeamType.ELECTRON, image_type='new')

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

        self.pushButton_autoliftout.setEnabled(True)

        logging.info(f"{len(self.samples)} samples selected and saved to {self.save_path}.")
        logging.info(f"{self.current_status.name} FINISHED")

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


            # refresh TODO: fix protocol structure
            self.update_image_settings(hfw=400e-6, save=False)
            self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
            self.update_display(beam_type=BeamType.ION, image_type='new')

            self.update_popup_settings(message=f'Please double click to centre the {feature_type} coordinate in the ion beam.\n'
                                        f'Press Yes when the feature is centered', click='double',
                                        filter_strength=self.filter_strength, allow_new_image=True)
            self.ask_user(image=self.image_FIB)

            # TODO: does this need to be new image?  Can it be last?  Can it be view set?
            self.update_display(beam_type=BeamType.ELECTRON, image_type='new')

            coordinates.append(self.stage.current_position)
            if feature_type == 'landing':
                self.update_image_settings(
                    resolution=self.settings['reference_images']['landing_post_ref_img_resolution'],
                    dwell_time=self.settings['reference_images']['landing_post_ref_img_dwell_time'],
                    hfw=self.settings['reference_images']['landing_post_ref_img_hfw_lowres'],
                    save=True,
                    label=f'{len(coordinates):02d}_ref_landing_low_res'
                )
                eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.image_settings)

                self.update_image_settings(
                    resolution=self.settings['reference_images']['landing_post_ref_img_resolution'],
                    dwell_time=self.settings['reference_images']['landing_post_ref_img_dwell_time'],
                    hfw=self.settings['reference_images']['landing_post_ref_img_hfw_highres'],
                    save=True,
                    label=f'{len(coordinates):02d}_ref_landing_high_res'
                )
                eb_highres, ib_highres = acquire.take_reference_images(self.microscope, settings=self.image_settings)
            elif feature_type == 'lamella':
                self.update_image_settings(
                    resolution=self.settings['reference_images']['trench_area_ref_img_resolution'],
                    dwell_time=self.settings['reference_images']['trench_area_ref_img_dwell_time'],
                    hfw=self.settings['reference_images']['trench_area_ref_img_hfw_lowres'],
                    save=True,
                    label=f'{len(coordinates):02d}_ref_lamella_low_res'
                )
                eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.image_settings)

                self.update_image_settings(
                    resolution=self.settings['reference_images']['trench_area_ref_img_resolution'],
                    dwell_time=self.settings['reference_images']['trench_area_ref_img_dwell_time'],
                    hfw=self.settings['reference_images']['trench_area_ref_img_hfw_highres'],
                    save=True,
                    label=f'{len(coordinates):02d}_ref_lamella_high_res'
                )
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
            movement.flat_to_beam(self.microscope, settings=self.settings, pretilt_angle=self.pretilt_degrees, beam_type=BeamType.ELECTRON)

        # lowres calibration
        self.image_settings['hfw'] = 900e-6  # TODO: add to protocol
        self.microscope.beams.electron_beam.horizontal_field_width.value = self.image_settings['hfw'] # TODO: why do we do this hfw setting?
        self.microscope.beams.ion_beam.horizontal_field_width.value = self.image_settings['hfw']
        acquire.autocontrast(self.microscope, beam_type=BeamType.ELECTRON)  # TODO: why
        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        acquire.autocontrast(self.microscope, beam_type=BeamType.ION)  # TODO: why
        self.update_display(beam_type=BeamType.ION, image_type='last')
        self.user_based_eucentric_height_adjustment()

        # highres calibration
        self.image_settings['hfw'] = 200e-6  # TODO: add to protocol
        self.microscope.beams.electron_beam.horizontal_field_width.value = self.image_settings['hfw']
        self.microscope.beams.ion_beam.horizontal_field_width.value = self.image_settings['hfw']
        self.user_based_eucentric_height_adjustment()

    def user_based_eucentric_height_adjustment(self):
        self.image_settings['resolution'] = '1536x1024'  # TODO: add to protocol
        self.image_settings['dwell_time'] = 1e-6  # TODO: add to protocol
        self.image_settings['beam_type'] = BeamType.ELECTRON
        self.image_settings['save'] = False
        # self.update_image_settings(
        #     resolution='1536x1024',  # TODO: add to protocol
        #     dwell_time=1e-6,  # TODO: add to protocol
        #     beam
        # ) TODO: need to pass the hfw as parameter as it is currently defined outside func
        self.image_SEM = acquire.new_image(self.microscope, settings=self.image_settings)
        self.update_popup_settings(message=f'Please double click to centre a feature in the SEM\n'
                                                           f'Press Yes when the feature is centered', click='double',
                                   filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_SEM)

        if self.response:
            self.image_settings['beam_type'] = BeamType.ION
            self.image_settings["hfw"] = float(min(self.image_settings["hfw"], 900e-6)) # clip to max hfw for ion #TODO: implement this before taking images...?
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
        logging.info("gui: run liftout started")
        logging.info(f"gui: running liftout on {len(self.samples)} samples.")

        # recalibrate park position coordinates
        # reset_needle_park_position(microscope=self.microscope, new_park_position=)

        for sample in self.samples:

            self.current_sample = sample
            (lamella_coord, landing_coord,
                lamella_area_reference_images,
                landing_reference_images) = self.current_sample.get_sample_data()

          # TODO: this can probably just use self.current_sample rather than passing arguments?
            self.single_liftout(landing_coord, lamella_coord,
                            landing_reference_images, lamella_area_reference_images)

        # NOTE: thinning needs to happen after all lamella landed due to platinum depositions...
        self.update_popup_settings(message="Do you want to start lamella thinning?", crosshairs=False)
        self.ask_user()
        logging.info(f"Perform Thinning: {self.response}")
        if self.response:
            for sample in self.samples:
                self.current_sample = sample
                landing_coord = self.current_sample.landing_coordinates
                self.current_status = AutoLiftoutStatus.Thinning
                self.thin_lamella(landing_coord=landing_coord)
        logging.info(f"autoliftout complete")

    def load_coordinates(self):

        logging.info(f"LOAD COORDINATES STARTED")
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

        # tes if the file exists
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
                sample = Sample(save_path, sample_no+1)  # TODO: watch out for this kind of thing with the numbering... improve
                sample.load_data_from_file()
                self.samples.append(sample)


        # TODO: test whether this is accurate, maybe move to start of run_liftout
        # if sample.park_position.x is not None:
            # movement.reset_needle_park_position(microscope=self.microscope, new_park_position=sample.park_position)

        self.pushButton_autoliftout.setEnabled(True)

        logging.info(f"{len(self.samples)} samples loaded from {save_path}.")
        logging.info(f"LOAD COORDINATES FINISHED")

    def single_liftout(self, landing_coordinates, lamella_coordinates,
                       original_landing_images, original_lamella_area_images):

        logging.info(f"gui: starting liftout no. {self.current_sample.sample_no}")

        # initial state
        self.MILLING_COMPLETED_THIS_RUN = False

        # TODO: use or code a safe_move function
        stage_settings = MoveSettings(rotate_compucentric=True)
        self.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)
        self.stage.absolute_move(StagePosition(r=lamella_coordinates.r))
        self.stage.absolute_move(lamella_coordinates)
        ret = calibration.correct_stage_drift(self.microscope, self.image_settings, original_lamella_area_images, self.current_sample.sample_no, mode='eb')
        self.image_SEM = acquire.last_image(self.microscope, beam_type=BeamType.ELECTRON)

        if ret is False:
            # cross-correlation has failed, manual correction required
            self.update_popup_settings(message=f'Please double click to centre the lamella in the image.',
                         click='double', filter_strength=self.filter_strength, allow_new_image=True)
            self.ask_user(image=self.image_SEM) # TODO: might need to update image?
            logging.info(f"{self.current_status.name}: cross-correlation manually corrected")

        # TODO: possibly new image
        self.update_popup_settings(message=f'Is the lamella currently centered in the image?\n'
                                                           f'If not, double click to center the lamella, press Yes when centered.',
                                   click='double', filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_SEM)

        # self.image_settings['save'] = True
        # self.image_settings['label'] = f'{self.current_sample.sample_no:02d}_post_drift_correction'
        self.update_image_settings(
            save=True,
            label=f'{self.current_sample.sample_no:02d}_post_drift_correction'
        )
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
        logging.info(f"{self.current_status.name} STARTED")

        # move flat to the ion beam, stage tilt 25 (total image tilt 52)
        stage_settings = MoveSettings(rotate_compucentric=True)
        movement.move_to_trenching_angle(self.microscope, self.settings)

        # Take an ion beam image at the *milling current*
        self.update_image_settings(hfw=100e-6)
        self.microscope.beams.ion_beam.horizontal_field_width.value = self.image_settings['hfw']  # TODO: why are these two lines here?
        self.microscope.beams.electron_beam.horizontal_field_width.value = self.image_settings['hfw']
        self.update_display(beam_type=BeamType.ION, image_type='new')

        self.update_popup_settings(message=f'Have you centered the lamella position in the ion beam?\n'
                                                      f'If not, double click to center the lamella position', click='double',
                                   filter_strength=self.filter_strength, allow_new_image=True)
        self.ask_user(image=self.image_FIB)

        # # TODO: remove ask user wrapping once mill_trenches is refactored
        self.update_popup_settings(message="Do you want to start milling trenches?", crosshairs=False)
        self.ask_user()
        if self.response:
            # mills trenches for lamella
            milling.mill_trenches(self.microscope, self.settings)

        self.current_sample.milling_coordinates = self.stage.current_position
        self.current_sample.save_data()

        # reference images of milled trenches
        self.update_image_settings(
            resolution=self.settings['reference_images']['trench_area_ref_img_resolution'],
            dwell_time=self.settings['reference_images']['trench_area_ref_img_dwell_time'],
            hfw=self.settings['reference_images']['trench_area_ref_img_hfw_lowres'],
            save=True,
            label=f'{self.current_sample.sample_no:02d}_ref_trench_low_res'
        )
        eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings['reference_images']['trench_area_ref_img_resolution'],
            dwell_time=self.settings['reference_images']['trench_area_ref_img_dwell_time'],
            hfw=self.settings['reference_images']['trench_area_ref_img_hfw_highres'],
            save=True,
            label=f'{self.current_sample.sample_no:02d}_ref_trench_high_res'
        )
        eb_highres, ib_highres = acquire.take_reference_images(self.microscope, settings=self.image_settings)

        reference_images_low_and_high_res = (eb_lowres, eb_highres, ib_lowres, ib_highres)

        # move flat to electron beam
        movement.flat_to_beam(self.microscope, self.settings, pretilt_angle=self.pretilt_degrees, beam_type=BeamType.ELECTRON, )

        # make sure drift hasn't been too much since milling trenches
        # first using reference images
        ret = calibration.correct_stage_drift(self.microscope, self.image_settings, reference_images_low_and_high_res, self.current_sample.sample_no, mode='ib')

        if ret is False:
            # cross-correlation has failed, manual correction required
            # TODO: we need to take a new image here? / use last image
            self.update_popup_settings(message=f'Please double click to centre the lamella in the image.',
                         click='double', filter_strength=self.filter_strength, allow_new_image=True)
            self.image_SEM = acquire.last_image(self.microscope, beam_type=BeamType.ELECTRON)
            self.ask_user(image=self.image_SEM)
            logging.info(f"{self.current_status.name}: cross-correlation manually corrected")


        # TODO: add to protocol
        # TODO: deal with resetting label requirement

        self.update_image_settings(resolution='1536x1024', dwell_time=1e-6, hfw=80e-6,
                                   save=True, label=f'{self.current_sample.sample_no:02d}_drift_correction_ML')
        # then using ML, tilting/correcting in steps so drift isn't too large
        self.correct_stage_drift_with_ML()
        movement.move_relative(self.microscope, t=np.deg2rad(6), settings=stage_settings)
        self.update_image_settings(resolution='1536x1024', dwell_time=1e-6, hfw=80e-6,
                                   save=True, label=f'{self.current_sample.sample_no:02d}_drift_correction_ML')
        self.correct_stage_drift_with_ML()

        # save jcut position
        self.current_sample.jcut_coordinates = self.stage.current_position
        self.current_sample.save_data()

        # now we are at the angle for jcut, perform jcut
        jcut_patterns = milling.mill_jcut(self.microscope, self.settings)

        # TODO: adjust hfw? check why it changes to 100
        self.update_display(beam_type=BeamType.ION, image_type='last')
        # TODO: return image with patterning marks
        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False, milling_patterns=jcut_patterns)
        self.ask_user(image=self.image_FIB)
        if self.response:

            milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                           patterns=self.patterns, depth=self.settings["jcut"]['jcut_milling_depth'])

        # take reference images of the jcut
        # TODO: add to protocol
        self.update_image_settings(hfw=150e-6, save=True, label='jcut_lowres')
        acquire.take_reference_images(self.microscope, self.image_settings)

        # TODO: add to protocol
        self.update_image_settings(hfw=50e-6, save=True, label='jcut_lowres')
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.MILLING_COMPLETED_THIS_RUN = True
        logging.info(f" {self.current_status.name} FINISHED")

    def correct_stage_drift_with_ML(self):
        # correct stage drift using machine learning
        label = self.image_settings['label']
        # if self.image_settings["hfw"] > 200e-6:
        #     self.image_settings["hfw"] = 150e-6
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

        # TODO: update to update_image_settings
        # take reference images after drift correction
        self.image_settings['save'] = True
        self.image_settings['label'] = f'{self.current_sample.sample_no:02d}_drift_correction_ML_final'
        self.image_settings['autocontrast'] = self.USE_AUTOCONTRAST
        self.image_SEM, self.image_FIB = acquire.take_reference_images(self.microscope, self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        self.update_display(beam_type=BeamType.ION, image_type='last')

    def liftout_lamella(self):
        self.current_status = AutoLiftoutStatus.Liftout
        logging.info(f" {self.current_status.name} STARTED")

        # get ready to do liftout by moving to liftout angle
        movement.move_to_liftout_angle(self.microscope, self.settings)

        if not self.MILLING_COMPLETED_THIS_RUN:
            self.ensure_eucentricity(flat_to_sem=True) # liftout angle is flat to SEM
            self.image_settings["hfw"] = 150e-6
            movement.move_to_liftout_angle(self.microscope, self.settings)

        # correct stage drift from mill_lamella stage
        self.correct_stage_drift_with_ML()

        # move needle to liftout start position
        if self.stage.current_position.z < 3.7e-3:
            # TODO: [FIX] autofocus cannot be relied upon, if this condition is met, we need to stop.

            # movement.auto_link_stage(self.microscope) # This is too unreliable to fix the miscalibration
            logging.warning(f"Calibration error detected: stage position height")
            logging.warning(f"Stage Position: {self.stage.current_position}")
            display_error_message(message="The system has identified the distance between the sample and the pole piece is less than 3.7mm. "
                "The needle will contact the sample, and it is unsafe to insert the needle. "
                "\nPlease manually recalibrate the focus and restart the program. "
                "\n\nThe AutoLiftout GUI will now exit.",
                title="Calibration Error"
            )

            # Aborting Liftout # TODO: safe shutdown.
            self.disconnect()
            exit(0)

        park_position = movement.move_needle_to_liftout_position(self.microscope)
        logging.info(f"{self.current_status.name}: needle inserted to park positon: {park_position}")

        # save liftout position
        self.current_sample.park_position = park_position
        self.current_sample.liftout_coordinates = self.stage.current_position
        self.current_sample.save_data()

        # land needle on lamella
        self.land_needle_on_milled_lamella()

        # sputter platinum
        fibsem_utils.sputter_platinum(self.microscope, self.settings, whole_grid=False)
        logging.info(f"{self.current_status.name}: lamella to needle welding complete.")

        self.update_image_settings(save=True, hfw=100e-6, label='landed_Pt_sputter')
        acquire.take_reference_images(self.microscope, self.image_settings)

        jcut_severing_pattern = milling.jcut_severing_pattern(self.microscope, self.settings) # TODO: tune jcut severing pattern
        self.update_display(beam_type=BeamType.ION, image_type='last')

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False,
                                   milling_patterns=jcut_severing_pattern)
        self.ask_user(image=self.image_FIB)
        if self.response:
            milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                           patterns=self.patterns, depth=self.settings["jcut"]['jcut_milling_depth'])

        self.update_image_settings(save=True, hfw=100e-6, label='jcut_sever')
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
        logging.info(f" {self.current_status.name} FINISHED")

    def land_needle_on_milled_lamella(self):

        logging.info(f"{self.current_status.name}: land needle on lamella started.")

        needle = self.microscope.specimen.manipulator

        # low res
        # TODO: remove lowres/highres ref images?
        self.update_image_settings(
            resolution=self.settings["reference_images"]["needle_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["needle_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["needle_ref_img_hfw_lowres"],
            save=True,
            label='needle_liftout_start_position_lowres'
        )
        acquire.take_reference_images(self.microscope, self.image_settings)
        # high res
        self.update_image_settings(
            resolution=self.settings["reference_images"]["needle_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["needle_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["needle_ref_img_hfw_highres"],
            save=True,
            label='needle_liftout_start_position_highres'
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        # calculate shift between lamella centre and needle tip in the electron view
        self.image_settings['hfw'] = 180e-6 #self.settings["reference_images"]["needle_ref_img_hfw_lowres"]
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
        z_distance = distance_y_m / np.cos(self.stage.current_position.t)

        # Calculate movement
        zy_move_half = movement.z_corrected_needle_movement(-z_distance / 2, self.stage.current_position.t)
        needle.relative_move(zy_move_half)
        logging.info(f"{self.current_status.name}: needle z-half-move: {zy_move_half}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.update_display(beam_type=BeamType.ION, image_type='new')

        self.update_popup_settings(message='Is the needle safe to move another half step?', click=None, filter_strength=self.filter_strength)
        self.ask_user(image=self.image_FIB)

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


            self.update_image_settings(
                hfw=self.settings["reference_images"]["needle_ref_img_hfw_lowres"],
                save=True,
                label='needle_ref_img_lowres'
            )
            acquire.take_reference_images(self.microscope, self.image_settings)

            self.update_image_settings(
                hfw=self.settings["reference_images"]["needle_ref_img_hfw_highres"],
                save=True,
                label='needle_ref_img_highres'
            )
            acquire.take_reference_images(self.microscope, self.image_settings)
            logging.info(f"{self.current_status.name}: land needle on lamella complete.")
        else:
            logging.warning(f"{self.current_status.name}: needle not safe to move onto lamella.")
            logging.warning(f"{self.current_status.name}: needle landing cancelled by user.")
            return

    def calculate_shift_distance_metres(self, shift_type, beamType=BeamType.ELECTRON):
        self.image_settings['beam_type'] = beamType
        self.raw_image, self.overlay_image, self.downscaled_image, feature_1_px, feature_1_type, feature_2_px, feature_2_type = \
            calibration.identify_shift_using_machine_learning(self.microscope, self.image_settings, self.settings, self.current_sample.sample_no,
                                                              shift_type=shift_type)
        feature_1_px, feature_2_px = self.validate_detection(feature_1_px=feature_1_px, feature_1_type=feature_1_type, feature_2_px=feature_2_px, feature_2_type=feature_2_type)
        # TODO: assert that self.overlay_image and self.downscale_image are the same size?
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

            # TODO: change this to a loop instead of two individual detections?            
            logging.info(f"ml_detection: {feature_1_type}: {self.response}")

            # if feature 1 wasn't correctly identified
            if not self.response:

                self.update_popup_settings(message=f'Please click on the correct {feature_1_type} position.'
                                                                   f'Press Yes button when happy with the position', click='single', crosshairs=False)
                self.ask_user(image=self.downscaled_image)

                # if new feature position selected
                if self.response:
                    # TODO: check x/y here
                    feature_1_px = (self.yclick, self.xclick)

            # skip image centre 'detections'
            if feature_2_type != "image_centre": 
                self.update_popup_settings(message=f'Has the model correctly identified the {feature_2_type} position?', click=None, crosshairs=False)
                self.ask_user(image=self.overlay_image)

                logging.info(f"ml_detection: {feature_2_type}: {self.response}")
                # if feature 2 wasn't correctly identified 
                if not self.response:

                    self.update_popup_settings(message=f'Please click on the correct {feature_2_type} position.'
                                                                    f'Press Yes button when happy with the position', click='single', filter_strength=self.filter_strength, crosshairs=False)
                    self.ask_user(image=self.downscaled_image)
                    # TODO: do we want filtering on this image?

                    # if new feature position selected
                    if self.response:
                        feature_2_px = (self.yclick, self.xclick)


            # TODO: wrap this in a function
            #### show the user the manually corrected movement and confirm
            from liftout.detection.detection import draw_two_features
            final_detection_img = draw_two_features(self.downscaled_image, feature_1_px, feature_2_px)
            final_detection_img = np.array(final_detection_img.convert("RGB"))
            self.update_popup_settings(
                message=f'Are the {feature_1_type} and {feature_2_type} positions now correctly identified?', 
                click=None, crosshairs=False
            )
            self.ask_user(image=final_detection_img)
            DETECTIONS_ARE_CORRECT = self.response
            #####

        return feature_1_px, feature_2_px

    def land_lamella(self, landing_coord, original_landing_images):

        logging.info(f"{self.current_status.name}: land lamella stage started")
        self.current_status = AutoLiftoutStatus.Landing

        # move to landing coordinate # TODO: wrap in func
        stage_settings = MoveSettings(rotate_compucentric=True)
        self.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)
        self.stage.absolute_move(landing_coord)

        # eucentricity correction
        # TODO: check if we want to recalibrate... should this be done after drift correction?
        self.ensure_eucentricity(flat_to_sem=False)  # liftout angle is flat to SEM
        self.image_settings["hfw"] = 150e-6

        # TODO: image settings?
        ret = calibration.correct_stage_drift(self.microscope, self.image_settings, original_landing_images, self.current_sample.sample_no, mode="land")

        if ret is False:
            # cross-correlation has failed, manual correction required
            self.update_popup_settings(message=f'Please double click to centre the lamella in the image.',
                         click='double', filter_strength=self.filter_strength, allow_new_image=True)
            self.ask_user(image=self.image_FIB) # TODO: might need to update image?
            logging.info(f"{self.current_status.name}: cross-correlation manually corrected")

        logging.info(f"{self.current_status.name}: initial landing calibration complete.")
        park_position = movement.move_needle_to_landing_position(self.microscope)


        # # Y-MOVE

        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=180e-6, # self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol,
            beam_type=BeamType.ELECTRON,
            save=True,
            label="landing_needle_land_sample_lowres"
        )

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=BeamType.ELECTRON)

        y_move = movement.y_corrected_needle_movement(-distance_y_m, self.stage.current_position.t)
        self.needle.relative_move(y_move)
        logging.info(f"{self.current_status.name}: y-move complete: {y_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")


        # Z-MOVE
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=180e-6, # self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol,
            beam_type=BeamType.ION,
            save=True,
            label="landing_needle_land_sample_lowres_after_y_move"
        )
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=BeamType.ION)

        z_distance = distance_y_m / np.sin(np.deg2rad(52))
        z_move = movement.z_corrected_needle_movement(z_distance, self.stage.current_position.t)
        self.needle.relative_move(z_move)
        logging.info(f"{self.current_status.name}: z-move complete: {z_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")

        # X-HALF-MOVE
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=150e-6,  # TODO: fix protocol,
            beam_type=BeamType.ELECTRON,
            save=True,
            label="landing_needle_land_sample_lowres_after_z_move"
        )

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=BeamType.ELECTRON)

        # half move
        x_move = movement.x_corrected_needle_movement(distance_x_m / 2)
        self.needle.relative_move(x_move)
        logging.info(f"{self.current_status.name}: x-half-move complete: {x_move}")

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")

        ## X-MOVE

        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=80e-6,  # TODO: fix protocol,
            beam_type=BeamType.ELECTRON,
            save=True,
            label="landing_needle_land_sample_lowres_after_z_move"
        )
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=BeamType.ELECTRON)

        # TODO: gap?
        x_move = movement.x_corrected_needle_movement(distance_x_m)
        self.needle.relative_move(x_move)
        logging.info(f"{self.current_status.name}: x-move complete: {x_move}")

        # final reference images
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=80e-6,  # TODO: fix protocol,
            beam_type=BeamType.ELECTRON,
            save=True,
            label="landing_lamella_final_weld_highres"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        ############################## WELD TO LANDING POST #############################################
        weld_pattern = milling.weld_to_landing_post(self.microscope)
        self.update_display(beam_type=BeamType.ION, image_type='last')

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?',
                                   filter_strength=self.filter_strength, crosshairs=False, milling_patterns=weld_pattern)
        self.ask_user(image=self.image_FIB)

        if self.response:
            logging.info(f"{self.current_status.name}: welding to post started.")
            milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                           patterns=self.patterns, depth=15e-9) # TODO: add to protocol

        logging.info(f"{self.current_status.name}: weld to post complete")

        # final reference images

        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=100e-6,  # TODO: fix protocol,
            save=True,
            label="landing_lamella_final_weld_highres"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")


        ###################################### CUT OFF NEEDLE ######################################
        logging.info(f"{self.current_status.name}: start cut off needle. detecting needle distance from centre.")

        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["cut"]["hfw"],
            beam_type=BeamType.ION,
            save=True,
            label="landing_lamella_pre_cut_off"
        )

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=BeamType.ION)

        height = self.settings["cut"]["height"]
        width = self.settings["cut"]["width"]
        depth = self.settings["cut"]["depth"]
        rotation = self.settings["cut"]["rotation"]
        hfw = self.settings["cut"]["hfw"]
        vertical_gap = 2e-6

        cut_coord = {"center_x": -distance_x_m,
                     "center_y": distance_y_m - vertical_gap, # TODO: check direction?
                     "width": width,
                     "height": height,
                     "depth": depth,  # TODO: might need more to get through needle
                     "rotation": rotation, "hfw": hfw}  # TODO: check rotation

        logging.info(f"{self.current_status.name}: calculating needle cut-off pattern")

        # cut off needle tip
        cut_off_pattern = milling.cut_off_needle(self.microscope, cut_coord=cut_coord)
        self.update_display(beam_type=BeamType.ION, image_type='last')

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False,
                                   milling_patterns=cut_off_pattern)
        self.ask_user(image=self.image_FIB)
        # TODO: add rotation

        if self.response:
            logging.info(f"{self.current_status.name}: needle cut-off started")
            milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                           patterns=self.patterns, depth=cut_coord["depth"])

        logging.info(f"{self.current_status.name}: needle cut-off complete")

        # reference images
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=150e-6, # self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
            beam_type=BeamType.ION,
            save=True,
            label="landing_lamella_final_cut_lowres"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=80e-6, # self.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
            beam_type=BeamType.ION,
            save=True,
            label="landing_lamella_final_cut_highres"
        )
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
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["landing_lamella_ref_img_hfw_lowres"],
            save=True,
            label="landing_lamella_final_lowres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.image_settings["hfw"] = self.settings["reference_images"]["landing_lamella_ref_img_hfw_lowres"] # 80e-6  #TODO: fix protocol
        self.image_settings["label"] = "landing_lamella_final_highres"
        self.update_image_settings(
            resolution=self.settings["reference_images"]["landing_post_ref_img_resolution"],
            dwell_time=self.settings["reference_images"]["landing_post_ref_img_dwell_time"],
            hfw=self.settings["reference_images"]["landing_lamella_ref_img_hfw_lowres"],
            save=True,
            label="landing_lamella_final_highres"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        logging.info(f"{self.current_status.name}: landing stage complete")


    def reset_needle(self):

        self.current_status = AutoLiftoutStatus.Reset
        logging.info(f" {self.current_status.name} STARTED")

        # move sample stage out
        movement.move_sample_stage_out(self.microscope)
        logging.info(f"{self.current_status.name}: moved sample stage out")

        # move needle in
        park_position = movement.insert_needle(self.microscope)
        z_move_in = movement.z_corrected_needle_movement(-180e-6, self.stage.current_position.t)
        self.needle.relative_move(z_move_in)
        logging.info(f"{self.current_status.name}: insert needle for reset")

        # needle images
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=self.settings["imaging"]["horizontal_field_width"],
            beam_type=BeamType.ION,
            save=True,
            label="sharpen_needle_initial"
        )
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")
        # self.image_settings["beam_type"] = BeamType.ION

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=BeamType.ION)

        x_move = movement.x_corrected_needle_movement(distance_x_m)
        self.needle.relative_move(x_move)
        z_distance = distance_y_m / np.sin(np.deg2rad(52))  # TODO: magic number
        z_move = movement.z_corrected_needle_movement(z_distance, self.stage.current_position.t)
        self.needle.relative_move(z_move)
        logging.info(f"{self.current_status.name}: moving needle to centre: x_move: {x_move}, z_move: {z_move}")

        self.image_settings["label"] = "sharpen_needle_centre"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=BeamType.ION)

        # create sharpening patterns
        cut_coord_bottom, cut_coord_top = milling.calculate_sharpen_needle_pattern(microscope=self.microscope, settings=self.settings, x_0=distance_x_m, y_0=distance_y_m)
        logging.info(f"{self.current_status.name}: calculate needle sharpen pattern")

        sharpen_patterns = milling.create_sharpen_needle_patterns(
            self.microscope, cut_coord_bottom, cut_coord_top
        )

        # confirm and run milling
        self.update_display(beam_type=BeamType.ION, image_type='last')

        self.update_popup_settings(message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength,
                                   crosshairs=False, milling_patterns=sharpen_patterns)
        self.ask_user(image=self.image_FIB)
        if self.response:
            logging.info(f"{self.current_status.name}: needle sharpening milling started")
            # TODO: TEST ROTATION
            milling.draw_patterns_and_mill(microscope=self.microscope, settings=self.settings,
                                           patterns=self.patterns, depth=cut_coord_bottom["depth"])
            # milling.run_milling(self.microscope, self.settings)

        logging.info(f"{self.current_status.name}: needle sharpening milling complete")

        # take reference images
        self.image_settings["label"] = "sharpen_needle_final"
        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        # retract needle
        movement.retract_needle(self.microscope, park_position)

        logging.info(f"{self.current_status.name} FINISHED")

    def thin_lamella(self, landing_coord):
        """Thinning: Thin the lamella thickness to size for imaging."""

        self.current_status = AutoLiftoutStatus.Thinning
        logging.info(f" {self.current_status.name} STARTED")

        # move to landing coord
        self.microscope.specimen.stage.absolute_move(landing_coord)
        logging.info(f"{self.current_status.name}: move to landing coordinates: {landing_coord}")

        self.ensure_eucentricity(flat_to_sem=False)  # liftout angle is flat to SEM
        self.image_settings["hfw"] = 150e-6

        # tilt to 0 rotate 180 move to 21 deg
        # tilt to zero, to prevent hitting anything
        stage_settings = MoveSettings(rotate_compucentric=True)
        self.microscope.specimen.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)

        # thinning position
        thinning_rotation_angle = self.settings["thin_lamella"]["rotation_angle"]  # 180 deg
        thinning_tilt_angle = self.settings["thin_lamella"]["tilt_angle"]  # 21 deg

        # rotate to thinning angle
        self.microscope.specimen.stage.relative_move(StagePosition(r=np.deg2rad(thinning_rotation_angle)), stage_settings)

        # tilt to thinning angle
        self.microscope.specimen.stage.absolute_move(StagePosition(t=np.deg2rad(thinning_tilt_angle)), stage_settings)
        logging.info(f"{self.current_status.name}: rotate to thinning angle: {thinning_rotation_angle}")
        logging.info(f"{self.current_status.name}: tilt to thinning angle: {thinning_tilt_angle}")

        # lamella images # TODO: check and add to protocol
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=400e-6,
            save=True,
            label=f"{self.current_sample.sample_no:02d}_thinning_lamella_21deg_tilt"
        )

        acquire.take_reference_images(self.microscope, self.image_settings)

        # realign lamella to image centre
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=300e-6,
            save=True,
            label=f'{self.current_sample.sample_no:02d}_drift_correction_ML_thinning'
        )
        self.correct_stage_drift_with_ML()

        # take reference images
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time=self.settings["imaging"]["dwell_time"],
            hfw=80e-6,
            save=True,
            label=f"{self.current_sample.sample_no:02d}_cleanup_lamella_pre_movement"
        )
        acquire.take_reference_images(self.microscope, self.image_settings)

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_centre_to_image_centre', beamType=BeamType.ION)
        # TODO: this should be image_centre not landing post?
        # x_shift, y_shift = calculate_shift_between_features_in_metres(lamella_ib, "lamella_edge_to_landing_post")

        # z-movement (shouldnt really be needed if eucentric calibration is correct)
        z_distance = distance_y_m / np.sin(np.deg2rad(52))
        z_move = movement.z_corrected_stage_movement(z_distance, self.stage.current_position.t)
        self.stage.relative_move(z_move)

        # x-move the rest of the way
        x_move = movement.x_corrected_stage_movement(-distance_x_m)
        self.stage.relative_move(x_move)

        # TODO: check the direction of this movement?
        # move half the width of lamella to centre the edge..
        width = self.settings["lamella"]["lamella_width"]
        x_move_half_width = movement.x_corrected_stage_movement(width / 4)
        self.stage.relative_move(x_move_half_width)
        # TODO: take reference images

        # take reference images and finish
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=100e-6,
            save=True,
            label=f"{self.current_sample.sample_no:02d}_lamella_pre_thinning"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        # mill thin lamella pattern
        self.update_popup_settings(message="Run lamella thinning?", crosshairs=False)
        # TODO: refactor this to use the movable pattern structure like other milling
        self.ask_user()
        if self.response:
            milling.mill_thin_lamella(self.microscope, self.settings)

        # take reference images and finish
        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=80e-6,
            save=True,
            label=f"{self.current_sample.sample_no:02d}_lamella_post_thinning_highres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)

        self.update_image_settings(
            resolution=self.settings["imaging"]["resolution"],
            dwell_time =self.settings["imaging"]["dwell_time"],
            hfw=150e-6,
            save=True,
            label=f"{self.current_sample.sample_no:02d}_lamella_post_thinning_lowres"
        )

        acquire.take_reference_images(microscope=self.microscope, settings=self.image_settings)
        logging.info(f"{self.current_status.name}: thin lamella {self.current_sample.sample_no} complete.")
        logging.info(f" {self.current_status.name} FINISHED")


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
        logging.info("gui: setup connections started")
        # Protocol and information table connections
        self.pushButton_initialise.clicked.connect(lambda: self.initialise_autoliftout())
        self.pushButton_autoliftout.clicked.connect(lambda: self.run_liftout())
        self.pushButton_autoliftout.setEnabled(0)  # disable unless sample positions are loaded.

        # FIBSEM methods
        self.pushButton_load_sample_data.clicked.connect(lambda: self.load_coordinates())

        # self.update_popup_settings(click=None, crosshairs=True, milling_patterns=test_jcut)

        # self.pushButton_test_popup.clicked.connect(lambda: self.update_popup_settings(click=None, crosshairs=True))
        self.pushButton_test_popup.clicked.connect(lambda: self.update_popup_settings(click=None, crosshairs=True, milling_patterns=test_jcut))

        # self.pushButton_test_popup.clicked.connect(lambda: self.ask_user(image=test_image, second_image=test_image))
        self.pushButton_test_popup.clicked.connect(lambda: self.ask_user(image=test_image)) # only one image works with jcut

        # self.pushButton_test_popup.clicked.connect(lambda: self.calculate_shift_distance_metres(shift_type='lamella_centre_to_image_centre', beamType=BeamType.ELECTRON))

        # self.pushButton_test_popup.clicked.connect(lambda: self.testing_function())
        # self.pushButton_test_popup.clicked.connect(lambda: self.update_image_settings())

        # self.pushButton_test_popup.clicked.connect(lambda: self.test_draw_patterns())

        logging.info("gui: setup connections finished")

    def testing_function(self):

        TEST_VALIDATE_DETECTION = True

        if TEST_VALIDATE_DETECTION:

            self.raw_image = AdornedImage(data=test_image)
            self.overlay_image = test_image
            self.downscaled_image = test_image
            print("Hello World")
            import random
            supported_feature_types = ["image_centre", "lamella_centre", "needle_tip", "lamella_edge", "landing_post"]
            feature_1_px = (0, 0)
            feature_1_type = random.choice(supported_feature_types)
            feature_2_px = (test_image.shape[0] // 2 , test_image.shape[1] //2)
            feature_2_type = random.choice(supported_feature_types)

            feature_1_px, feature_2_px = self.validate_detection(feature_1_px=feature_1_px, feature_1_type=feature_1_type, feature_2_px=feature_2_px, feature_2_type=feature_2_type)




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
                    self.hfw_slider.setMaximum(2700) # TODO: update to CONST
                else:
                    self.hfw_slider.setMaximum(900) # TODO: update to CONST
                self.hfw_slider.setValue(self.image_settings['hfw']*1e6)

                # spinbox (not a property as only slider value needed)
                hfw_spinbox = QtWidgets.QSpinBox()
                hfw_spinbox.setMinimum(1)
                if beam_type == BeamType.ELECTRON:
                    hfw_spinbox.setMaximum(2700) # TODO: update to CONST
                else:
                    hfw_spinbox.setMaximum(900) # TODO: update to CONST
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

            if self.popup_settings['milling_patterns'] is not None:
                toolbar_active = False
                if self.select_all_button is None:
                    self.select_all_button = QtWidgets.QCheckBox('Select all')
                    self.popup_window.layout().addWidget(self.select_all_button)
                self.patterns = []
                for pattern in self.popup_settings['milling_patterns']:
                    if type(self.popup_settings['image']) == np.ndarray:
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
                        rotation = pattern.rotation
                        # Rectangle is defined from bottom left due to mpl
                        # Microscope (0, 0) is middle of image, y+ = up
                        # Image (0, 0) is top left corner, y+ = down
                        rectangle_left = (image_width / 2) + (pattern.center_x / pixel_size) - (width / 2)
                        rectangle_bottom = (image_height / 2) - (pattern.center_y / pixel_size) - (height / 2)

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
        # xy is top left in image coords, bottom left in

        self.setEnabled(True) # enable the main window

    def test_draw_patterns(self):
        # TODO: adjust hfw? check why it changes to 100
        self.update_display(beam_type=BeamType.ION, image_type='last')
        # TODO: return image with patterning marks
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

    # def new_protocol(self):
    #     num_index = self.tabWidget_Protocol.__len__() + 1
    #
    #     # new tab for protocol
    #     new_protocol_tab = QtWidgets.QWidget()
    #     new_protocol_tab.setObjectName(f'Protocol {num_index}')
    #     layout_protocol_tab = QtWidgets.QGridLayout(new_protocol_tab)
    #     layout_protocol_tab.setObjectName(f'gridLayout_{num_index}')
    #
    #     # new text edit to hold protocol
    #     protocol_text_edit = QtWidgets.QTextEdit()
    #     font = QtGui.QFont()
    #     font.setPointSize(10)
    #     protocol_text_edit.setFont(font)
    #     protocol_text_edit.setObjectName(f'protocol_text_edit_{num_index}')
    #     layout_protocol_tab.addWidget(protocol_text_edit, 0, 0, 1, 1)
    #
    #     self.tabWidget_Protocol.addTab(new_protocol_tab, f'Protocol {num_index}')
    #     self.tabWidget_Protocol.setCurrentWidget(new_protocol_tab)
    #     self.load_template_protocol()
    #
    #     #
    #
    #     # new tab for information from FIBSEM
    #     new_information_tab = QtWidgets.QWidget()
    #     new_information_tab.setObjectName(f'Protocol {num_index}')
    #     layout_new_information_tab = QtWidgets.QGridLayout(new_information_tab)
    #     layout_new_information_tab.setObjectName(f'layout_new_information_tab_{num_index}')
    #
    #     # new empty table for information to fill
    #     new_table_widget = QtWidgets.QTableWidget()
    #     new_table_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
    #     new_table_widget.setAlternatingRowColors(True)
    #     new_table_widget.setObjectName(f'tableWidget_{num_index}')
    #     new_table_widget.setColumnCount(len(information_keys))
    #     new_table_widget.setRowCount(starting_positions)
    #
    #     # set up rows
    #     for row in range(starting_positions):
    #         table_item = QtWidgets.QTableWidgetItem()
    #         new_table_widget.setVerticalHeaderItem(row, table_item)
    #         item = new_table_widget.verticalHeaderItem(row)
    #         item.setText(_translate('MainWindow', f'Position {row+1}'))
    #
    #     # set up columns
    #     for column in range(len(information_keys)):
    #         table_item = QtWidgets.QTableWidgetItem()
    #         new_table_widget.setHorizontalHeaderItem(column, table_item)
    #         item = new_table_widget.horizontalHeaderItem(column)
    #         item.setText(_translate('MainWindow', information_keys[column]))
    #
    #     new_table_widget.horizontalHeader().setDefaultSectionSize(174)
    #     new_table_widget.horizontalHeader().setHighlightSections(True)
    #
    #     layout_new_information_tab.addWidget(new_table_widget, 0, 0, 1, 1)
    #     self.tabWidget_Information.addTab(new_information_tab, f'Protocol {num_index}')
    #     self.tabWidget_Information.setCurrentWidget(new_information_tab)
    #
    # def rename_protocol(self):
    #     index = self.tabWidget_Protocol.currentIndex()
    #
    #     top_margin = 4
    #     left_margin = 10
    #
    #     rect = self.tabWidget_Protocol.tabBar().tabRect(index)
    #     self.edit = QtWidgets.QLineEdit(self.tabWidget_Protocol)
    #     self.edit.move(rect.left() + left_margin, rect.top() + top_margin)
    #     self.edit.resize(rect.width() - 2 * left_margin, rect.height() - 2 * top_margin)
    #     self.edit.show()
    #     self.edit.setFocus()
    #     self.edit.selectAll()
    #     self.edit.editingFinished.connect(lambda: self.finish_rename())
    #
    # def finish_rename(self):
    #     self.tabWidget_Protocol.setTabText(self.tabWidget_Protocol.currentIndex(), self.edit.text())
    #     tab = self.tabWidget_Protocol.currentWidget()
    #     tab.setObjectName(self.edit.text())
    #
    #     self.tabWidget_Information.setTabText(self.tabWidget_Information.currentIndex(), self.edit.text())
    #     tab2 = self.tabWidget_Information.currentWidget()
    #     tab2.setObjectName(self.edit.text())
    #     self.edit.deleteLater()
    #
    # def delete_protocol(self):
    #     index = self.tabWidget_Protocol.currentIndex()
    #     self.tabWidget_Information.removeTab(index)
    #     self.tabWidget_Protocol.removeTab(index)
    #     pass
    #
    # def load_template_protocol(self):
    #     with open(protocol_template_path, 'r') as file:
    #         _dict = yaml.safe_load(file)
    #
    #     for key, value in _dict.items():
    #         self.key_list_protocol.append(key)
    #         if isinstance(value, dict):
    #             for key2, value2 in value.items():
    #                 if key2 not in self.key_list_protocol:
    #                     self.key_list_protocol.append(key)
    #             if isinstance(value, dict):
    #                 for key3, value3 in value.items():
    #                     if key3 not in self.key_list_protocol:
    #                         self.key_list_protocol.append(key)
    #             elif isinstance(value, list):
    #                 for dictionary in value:
    #                     for key4, value4 in dictionary.items():
    #                         if key4 not in self.key_list_protocol:
    #                             self.key_list_protocol.append(key)
    #
    #     self.load_protocol_text(_dict)
    #
    # def load_protocol_text(self, dictionary):
    #     protocol_text = str()
    #     count = 0
    #     jcut = 0
    #     _dict = dictionary
    #
    #     for key in _dict:
    #         if key not in self.key_list_protocol:
    #             logging.info(f'Unexpected parameter in template file')
    #             return
    #
    #     for key, value in _dict.items():
    #         if type(value) is dict:
    #             if jcut == 1:
    #                 protocol_text += f'{key}:'  # jcut
    #                 jcut = 0
    #             else:
    #                 protocol_text += f'\n{key}:'  # first level excluding jcut
    #             for key2, value2 in value.items():
    #                 if type(value2) is list:
    #                     jcut = 1
    #                     protocol_text += f'\n  {key2}:'  # protocol stages
    #                     for item in value2:
    #                         if count == 0:
    #                             protocol_text += f'\n    # rough_cut'
    #                             count = 1
    #                             count2 = 0
    #                         else:
    #                             protocol_text += f'    # regular_cut'
    #                             count = 0
    #                             count2 = 0
    #                         protocol_text += f'\n    -'
    #                         for key6, value6 in item.items():
    #                             if count2 == 0:
    #                                 protocol_text += f' {key6}: {value6}\n'  # first after list
    #                                 count2 = 1
    #                             else:
    #                                 protocol_text += f'      {key6}: {value6}\n'  # rest of list
    #                 else:
    #                     protocol_text += f'\n  {key2}: {value2}'  # values not in list
    #         else:
    #             protocol_text += f'{key}: {value}'  # demo mode
    #     self.tabWidget_Protocol.currentWidget().findChild(QtWidgets.QTextEdit).setText(protocol_text)
    #
    # def save_protocol(self, destination=None):
    #     dest = destination
    #     if dest is None:
    #         dest = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder to save protocol in')
    #         index = self.tabWidget_Protocol.currentIndex()
    #         tab = self.tabWidget_Protocol.tabBar().tabText(index)
    #         dest = f'{dest}/{tab}.yml'
    #     protocol_text = self.tabWidget_Protocol.currentWidget().findChild(QtWidgets.QTextEdit).toPlainText()
    #     # protocol_text = self.tabWidget_Protocol.currentWidget().findChild(QtWidgets.QTextEdit).toMarkdown()
    #     logging.info(protocol_text)
    #     p = yaml.safe_load(protocol_text)
    #     logging.info(p)
    #     with open(dest, 'w') as file:
    #         yaml.dump(p, file, sort_keys=False)
    #     self.save_destination = dest
        # save
    #
    # def load_yaml(self):
    #     """Ask the user to choose a protocol file to load
    #
    #     Returns
    #     -------
    #     str
    #         Path to file for parameter loading
    #     """
    #     checked = 0
    #     logging.info(f'Please select protocol file (yml)')
    #     root = Tk()
    #     root.withdraw()
    #     _dict = None
    #     while not checked:
    #         checked = 1
    #         try:
    #             load_directory = filedialog.askopenfile(mode='r', filetypes=[('yml files', '*.yml')])
    #             if load_directory is None:
    #                 return
    #
    #             while not load_directory.name.endswith('.yml'):
    #                 logging.info('Not a yml configuration file')
    #                 load_directory = filedialog.askopenfile(mode='r', filetypes=[('yml files', '*.yml')])
    #
    #             with open(load_directory.name, 'r') as file:
    #                 _dict = yaml.safe_load(file)
    #                 file.close()
    #         except Exception:
    #             display_error_message(traceback.format_exc())
    #         for key in _dict:
    #             if key not in self.key_list_protocol:
    #                 if checked:
    #                     logging.info(f'Unexpected parameter in protocol file')
    #                 checked = 0
    #     root.destroy()
    #     logging.info(_dict)
    #
    #     self.load_protocol_text(_dict)

    # def tab_moved(self, moved):
    #     if moved == 'protocol':
    #         self.tabWidget_Information.tabBar().moveTab(self.tabWidget_Information.currentIndex(), self.tabWidget_Protocol.currentIndex())
    #     elif moved == 'information':
    #         self.tabWidget_Protocol.tabBar().moveTab(self.tabWidget_Protocol.currentIndex(), self.tabWidget_Information.currentIndex())
    #
    # def random_data(self):
    #     for parameter in range(len(self.params)):
    #         self.params[information_keys[parameter]] = np.random.randint(0, 1)
    #     self.fill_information()
    #     self.position = (self.position + 1) % self.tabWidget_Information.currentWidget().findChild(QtWidgets.QTableWidget).rowCount()
    #
    # def fill_information(self):
    #     information = self.tabWidget_Information.currentWidget().findChild(QtWidgets.QTableWidget)
    #     row = self.position
    #
    #     for column in range(len(information_keys)):
    #         item = QtWidgets.QTableWidgetItem()
    #         item.setText(str(self.params[information_keys[column]]))
    #         information.setItem(row, column, item)

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

    def toggle_select_all(self, onoff=None):
        for pattern in self.patterns:
            # if self.popup_toolbar._active == 'ZOOM' or self.popup_toolbar._active == 'PAN':
            #     pattern.movable = False
            # else:
            #     pattern.movable = True

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
        self.label_status_1.setText(f"{self.current_status.name}{mode}")
        status_colors = {"Initialisation": "gray", "Setup": "gold",
                         "Milling": "coral", "Liftout": "seagreen", "Landing": "dodgerblue",
                         "Reset": "salmon", "Thinning": "mediumpurple", "Finished": "cyan"}
        self.label_status_1.setStyleSheet(str(f"background-color: {status_colors[self.current_status.name]}"))

        if self.samples:
            if self.current_sample:
                self.label_status_2.setText(f"{len(self.samples)} Sample Positions Loaded"
                                            f"\n\tCurrent Sample: {self.current_sample.sample_no} "
                                            f"\n\tLamella Coordinate: {self.current_sample.lamella_coordinates}"
                                            f"\n\tLanding Coordinate: {self.current_sample.landing_coordinates}"
                                            f"\n\tPark Position: {self.current_sample.park_position}")
            else:
                self.label_status_2.setText(f"{len(self.samples)} Sample Positions Loaded"
                                            f"\n\tSample No: {self.samples[0].sample_no} "
                                            f"\n\tLamella Coordinate: {self.samples[0].lamella_coordinates}"
                                            f"\n\tLanding Coordinate: {self.samples[0].landing_coordinates}"
                                            f"\n\tPark Position: {self.samples[0].park_position}")
            self.label_status_2.setStyleSheet("background-color: lightgreen; padding: 10px")
        else:
            self.label_status_2.setText("No Sample Positions Loaded")
            self.label_status_2.setStyleSheet("background-color: gray; padding: 10px")

        # log info
        with open(self.log_path) as f:
            lines = f.read().splitlines()
            log_line = "\n".join(lines[-5:])  # last five log msgs
            self.label_status_3.setText(log_line)

        # TODO: test (this might cause lag...)
        # TODO: remove all other calls to updating these displays once this works
        # self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        # self.update_display(beam_type=BeamType.ION, image_type="last")

        # logging.info(f"Random No: {np.random.random():.5f}")

        if not WINDOW_ENABLED:
            self.setEnabled(False)

# TODO: remove
# class DraggablePatch:
#     def __init__(self, patch):
#         self.patch = patch
#         self.press = None
#         self.cidpress = None
#         self.cidrelease = None
#         self.cidmotion = None
#         self.move_all = False
#         self.movable = True
#         self.center_x = None
#         self.center_y = None
#         self.pixel_size = None
#         self.image_width = None
#         self.image_height = None
#         self.rotation = 0
#         self.rotating = False
#
#     def connect(self):
#         self.cidpress = self.patch.figure.canvas.mpl_connect(
#             'button_press_event', self.on_press)
#         self.cidrelease = self.patch.figure.canvas.mpl_connect(
#             'button_release_event', self.on_release)
#         self.cidmotion = self.patch.figure.canvas.mpl_connect(
#             'motion_notify_event', self.on_motion)
#
#     def update_position(self):
#         relative_center_x, relative_center_y = self.calculate_center()
#         center_x_px = relative_center_x - self.image_width / 2
#         center_y_px = relative_center_y - self.image_height / 2
#
#         self.center_x = center_x_px * self.pixel_size
#         self.center_y = - center_y_px * self.pixel_size  # centre coordinate systems
#
#         self.width = self.patch._width * self.pixel_size
#         self.height = self.patch._height * self.pixel_size
#         self.rotation = self.patch.angle
#
#     def on_press(self, event):
#         # movement enabled check
#         if not self.movable:
#             self.press = None
#             return
#
#         # left click check
#         if event.button != 1: return
#
#         # if only moving what's under the cursor:
#         if not self.move_all:
#             # discard all changes if this isn't a hovered patch
#             if event.inaxes != self.patch.axes: return
#             contains, attrd = self.patch.contains(event)
#             if not contains: return
#
#         # get the top left corner of the patch
#         x0, y0 = self.patch.xy
#         self.press = x0, y0, event.xdata, event.ydata
#
#         if self.rotation_check(event):
#             self.rotating = True
#         else:
#             self.rotating = False
#             QtWidgets.QApplication.restoreOverrideCursor()
#
#     def on_motion(self, event):
#         """on motion we will move the rect if the mouse is over us"""
#         if self.rotation_check(event):
#             QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.OpenHandCursor)
#         else:
#             QtWidgets.QApplication.restoreOverrideCursor()
#
#         if not self.press: return
#
#         if self.rotating and not self.move_all:
#             QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.OpenHandCursor)
#             center_x, center_y = self.calculate_center()
#
#             angle_dx = event.xdata - center_x
#             angle_dy = event.ydata - center_y
#             angle = np.rad2deg(np.arctan2(angle_dy, angle_dx))
#             self.rotate_about_center(angle+90)
#         else:
#             QtWidgets.QApplication.restoreOverrideCursor()
#             x0, y0, x_intial_press, y_initial_press = self.press
#             dx = event.xdata - x_intial_press
#             dy = event.ydata - y_initial_press
#             self.patch.set_x(x0+dx)
#             self.patch.set_y(y0+dy)
#
#         self.patch.figure.canvas.draw()
#
#     def on_release(self, event):
#         """on release we reset the press data"""
#         QtWidgets.QApplication.restoreOverrideCursor()
#         self.update_position()
#         self.press = None
#         self.patch.figure.canvas.draw()
#
#     def disconnect(self):
#         """disconnect all the stored connection ids"""
#         self.patch.figure.canvas.mpl_disconnect(self.cidpress)
#         self.patch.figure.canvas.mpl_disconnect(self.cidrelease)
#         self.patch.figure.canvas.mpl_disconnect(self.cidmotion)
#
#     def toggle_move_all(self, onoff=False):
#         self.move_all = onoff
#
#     def calculate_center(self):
#         x0, y0 = self.patch.xy
#         w = self.patch._width/2
#         h = self.patch._height/2
#         theta = np.deg2rad(self.patch.angle)
#         x_center = x0 + w * np.cos(theta) - h * np.sin(theta)
#         y_center = y0 + w * np.sin(theta) + h * np.cos(theta)
#
#         return x_center, y_center
#
#     def calculate_corners(self):
#         x0, y0 = self.patch.xy
#         w = self.patch._width/2
#         h = self.patch._height/2
#         theta = np.deg2rad(self.patch.angle)
#
#         x_shift = 2 * (w * np.cos(theta) - h * np.sin(theta))
#         y_shift = 2 * (w * np.sin(theta) + h * np.cos(theta))
#
#         top_left = x0, y0
#         top_right = x0 + x_shift, y0
#         bottom_left = x0, y0 + y_shift
#         bottom_right = x0 + x_shift, y0 + y_shift
#
#         return top_left, top_right, bottom_left, bottom_right
#
#     def rotation_check(self, event):
#         xpress = event.xdata
#         ypress = event.ydata
#         if xpress and ypress:
#             ratio = 5
#             abs_min = 30
#             distance_check = max(min(self.patch._height / ratio, self.patch._width / ratio), abs_min)
#             corners = self.calculate_corners()
#             for corner in corners:
#                 dist = np.sqrt((xpress-corner[0]) ** 2 + (ypress-corner[1]) ** 2)
#                 if dist < distance_check:
#                     return True
#         return False
#
#     def rotate_about_center(self, angle):
#         print(angle)
#         # calculate the center position in the unrotated, original position
#         old_x_center, old_y_center = self.calculate_center()
#
#         # move the pattern to have x0, y0 at 0, 0
#         self.patch.set_x(0)
#         self.patch.set_y(0)
#
#         # rotate by angle
#         self.patch.angle = angle
#         new_theta = np.deg2rad(self.patch.angle)
#
#         # calculate new center position at the rotated, 0, 0 position
#         w = self.patch._width/2
#         h = self.patch._height/2
#         new_x_center = w * np.cos(new_theta) - h * np.sin(new_theta)
#         new_y_center = w * np.sin(new_theta) + h * np.cos(new_theta)
#
#         # move the center to the 0, 0, position (can be removed as a nondebugging step)
#         # self.patch.set_x(-new_x_center)
#         # self.patch.set_y(-new_y_center)
#
#         # move pattern back to centered on original center position
#         self.patch.set_x(-new_x_center + old_x_center)
#         self.patch.set_y(-new_y_center + old_y_center)
#
#         self.update_position()


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
    # logging.basicConfig(level=logging.INFO)
    if offline is False:
        launch_gui(ip_address='10.0.0.1', offline=offline)
    else:
        try:
            launch_gui(ip_address='localhost', offline=offline)
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


if __name__ == "__main__":
    offline_mode = True
    main(offline=offline_mode)
