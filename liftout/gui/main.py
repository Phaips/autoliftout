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
# from liftout.fibsem.utils import *
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
from autoscript_sdb_microscope_client.structures import *
from liftout.main2 import AutoLiftout
from liftout.main2 import AutoLiftoutStatus
import scipy.ndimage as ndi
import skimage
import PIL
from PIL import Image

# Required to not break imports
BeamType = acquire.BeamType

# test_image = PIL.Image.open('C:/Users/David/images/mask_test.tif')
test_image = np.random.randint(0, 255, size=(1000, 1000, 3))
test_image = np.array(test_image)

pretilt = 27  # TODO: put in protocol

_translate = QtCore.QCoreApplication.translate
logger = logging.getLogger(__name__)

protocol_template_path = '..\\protocol_liftout.yml'
starting_positions = 1
information_keys = ['x', 'y', 'z', 'rotation', 'tilt', 'comments']


class GUIMainWindow(gui_main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, ip_address='10.0.0.1', offline=False):
        super(GUIMainWindow, self).__init__()
        # TODO: replace "SEM, FIB" with BeamType calls
        self.offline = offline
        self.setupUi(self)
        self.setWindowTitle('Autoliftout User Interface Main Window')
        self.liftout_counter = 0
        self.popup = None
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
        # TODO: check needle creation

        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        self.update_display(beam_type=BeamType.ION, image_type='last')
        self.auto = AutoLiftout(microscope=self.microscope)

    def initialise_autoliftout(self):

        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        self.update_display(beam_type=BeamType.ION, image_type='last')

        # Whole-grid platinum deposition
        self.ask_user(beam_type=BeamType.ELECTRON, message='Do you want to sputter the whole sample grid with platinum?', click='double', filter_strength=self.filter_strength)
        if self.auto.response:
            fibsem_utils.sputter_platinum(self.auto.microscope, self.auto.settings, whole_grid=True)
            print('Sputtering over whole grid')
            self.auto.image_settings['label'] = 'grid_Pt_deposition'
            self.auto.image_settings['save'] = True
            self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        movement.auto_link_stage(self.microscope)
        # # TODO: Add zooming of microscope to GUI control
        # Select landing points and check eucentric height
        movement.move_to_landing_grid(self.microscope, self.auto.settings, flat_to_sem=True)
        self.ensure_eucentricity()
        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.update_display(beam_type=BeamType.ION, image_type='new')
        self.landing_coordinates, self.original_landing_images = self.select_initial_feature_coordinates(feature_type='landing')
        self.lamella_coordinates, self.original_trench_images = self.select_initial_feature_coordinates(feature_type='lamella')
        self.zipped_coordinates = list(zip(self.lamella_coordinates, self.landing_coordinates))

    def select_initial_feature_coordinates(self, feature_type=''):
        """
        Options are 'lamella' or 'landing'
        """

        select_another_position = True
        coordinates = []
        images = []

        if feature_type == 'lamella':
            movement.move_to_sample_grid(self.microscope, settings=self.auto.settings)
        elif feature_type == 'landing':
            movement.move_to_landing_grid(self.microscope, settings=self.auto.settings, flat_to_sem=False)
            self.ensure_eucentricity(flat_to_sem=False)
        else:
            raise ValueError(f'Expected "lamella" or "landing" as feature_type')

        while select_another_position:
            if feature_type == 'lamella':
                self.ensure_eucentricity()
                movement.move_to_trenching_angle(self.microscope, settings=self.auto.settings)
            # elif feature_type == 'landing':
            # self.ensure_eucentricity()
            self.auto.image_settings['hfw'] = 400e-6  # TODO: set this value in protocol

            # refresh TODO: fix protocol structure
            self.auto.image_settings['save'] = False
            self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
            self.update_display(beam_type=BeamType.ION, image_type='new')

            self.ask_user(beam_type=BeamType.ION, message=f'Please double click to centre the {feature_type} coordinate in the ion beam.\n'
                                                          f'Press Yes when the feature is centered', click='double', filter_strength=self.filter_strength)

            self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
            # TODO: does this need to be new image?  Can it be last?  Can it be view set?

            coordinates.append(self.stage.current_position)
            self.auto.image_settings['save'] = True
            if feature_type == 'landing':
                self.auto.image_settings['resolution'] = self.auto.settings['reference_images']['landing_post_ref_img_resolution']
                self.auto.image_settings['dwell_time'] = self.auto.settings['reference_images']['landing_post_ref_img_dwell_time']
                self.auto.image_settings['hfw'] = self.auto.settings['reference_images']['landing_post_ref_img_hfw_lowres']  # TODO: watch image settings through run
                self.auto.image_settings['label'] = f'{len(coordinates):02d}_ref_landing_low_res'  # TODO: add to protocol
                eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.auto.image_settings)
                self.auto.image_settings['hfw'] = self.auto.settings['reference_images']['landing_post_ref_img_hfw_highres']
                self.auto.image_settings['label'] = f'{len(coordinates):02d}_ref_landing_high_res'  # TODO: add to protocol
                eb_highres, ib_highres = acquire.take_reference_images(self.microscope, settings=self.auto.image_settings)
            elif feature_type == 'lamella':
                self.auto.image_settings['resolution'] = self.auto.settings['reference_images']['trench_area_ref_img_resolution']
                self.auto.image_settings['dwell_time'] = self.auto.settings['reference_images']['trench_area_ref_img_dwell_time']
                self.auto.image_settings['hfw'] = self.auto.settings['reference_images']['trench_area_ref_img_hfw_lowres']  # TODO: watch image settings through run
                self.auto.image_settings['label'] = f'{len(coordinates):02d}_ref_lamella_low_res'  # TODO: add to protocol
                eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.auto.image_settings)
                self.auto.image_settings['hfw'] = self.auto.settings['reference_images']['trench_area_ref_img_hfw_highres']
                self.auto.image_settings['label'] = f'{len(coordinates):02d}_ref_lamella_high_res'  # TODO: add to protocol
                eb_highres, ib_highres = acquire.take_reference_images(self.microscope, settings=self.auto.image_settings)

            images.append((eb_lowres, eb_highres, ib_lowres, ib_highres))

            self.ask_user(message=f'Do you want to select another {feature_type} position?\n'
                                  f'{len(coordinates)} positions selected so far.', crosshairs=False)
            select_another_position = self.auto.response
        return coordinates, images

    def ensure_eucentricity(self, flat_to_sem=True):
        calibration.validate_scanning_rotation(self.microscope)
        # print("Rough eucentric alignment")  TODO: status
        if flat_to_sem:
            movement.flat_to_beam(self.microscope, settings=self.auto.settings, pretilt_angle=pretilt, beam_type=BeamType.ELECTRON)

        self.auto.image_settings['hfw'] = 900e-6  # TODO: add to protocol
        self.microscope.beams.electron_beam.horizontal_field_width.value = self.auto.image_settings['hfw']
        self.microscope.beams.ion_beam.horizontal_field_width.value = self.auto.image_settings['hfw']
        acquire.autocontrast(self.microscope, beam_type=BeamType.ELECTRON)
        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        acquire.autocontrast(self.microscope, beam_type=BeamType.ION)
        self.update_display(beam_type=BeamType.ION, image_type='last')
        self.user_based_eucentric_height_adjustment()

        # print("Final eucentric alignment")  TODO: status
        self.auto.image_settings['hfw'] = 200e-6  # TODO: add to protocol
        self.microscope.beams.electron_beam.horizontal_field_width.value = self.auto.image_settings['hfw']
        self.microscope.beams.ion_beam.horizontal_field_width.value = self.auto.image_settings['hfw']
        self.user_based_eucentric_height_adjustment()

    def user_based_eucentric_height_adjustment(self):
        self.auto.image_settings['resolution'] = '1536x1024'  # TODO: add to protocol
        self.auto.image_settings['dwell_time'] = 1e-6  # TODO: add to protocol
        self.auto.image_settings['beam_type'] = BeamType.ELECTRON
        self.auto.image_settings['save'] = False
        # self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.image_SEM = acquire.new_image(self.microscope, settings=self.auto.image_settings)
        self.ask_user(beam_type=BeamType.ELECTRON, message=f'Please double click to centre a feature in the SEM\n'
                                                           f'Press Yes when the feature is centered', click='double', filter_strength=self.filter_strength)
        if self.auto.response:
            self.auto.image_settings['beam_type'] = BeamType.ION
            self.update_display(beam_type=BeamType.ION, image_type='new')
            self.ask_user(beam_type=BeamType.ION,  message=f'Please click the same location in the ion beam\n'
                                                           f'Press Yes when happy with the location', click='single', filter_strength=self.filter_strength)

            # TODO: show users their click, they click Yes on single click
        else:
            print('SEM image not centered')
            return

        real_x, real_y = movement.pixel_to_realspace_coordinate([self.xclick, self.yclick], self.image_FIB)
        delta_z = -np.cos(self.stage.current_position.t) * real_y
        self.stage.relative_move(StagePosition(z=delta_z))
        if self.auto.response:
            self.update_display(beam_type=BeamType.ION, image_type='new')
        # Could replace this with an autocorrelation (maybe with a fallback to asking for a user click if the correlation values are too low)
        self.auto.image_settings['beam_type'] = BeamType.ELECTRON
        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.ask_user(beam_type=BeamType.ELECTRON, message=f'Please double click to centre a feature in the SEM\n'
                                                           f'Press Yes when the feature is centered', click='double', filter_strength=self.filter_strength)
        # TODO: can we remove this? not used
        # resolution = storage.settings["reference_images"][
        #     "needle_ref_img_resolution"]
        # dwell_time = storage.settings["reference_images"][
        #     "needle_ref_img_dwell_time"]
        # image_settings = GrabFrameSettings(resolution=resolution,
        #                                    dwell_time=dwell_time)

    def run_liftout(self):

        for i, (lamella_coord, landing_coord) in enumerate(self.zipped_coordinates):
            self.liftout_counter += 1
            landing_reference_images = self.original_landing_images[i]
            lamella_area_reference_images = self.original_trench_images[i]
            self.single_liftout(landing_coord, lamella_coord,
                                landing_reference_images,
                                lamella_area_reference_images)

        self.auto.current_status = AutoLiftoutStatus.Cleanup
        self.cleanup_lamella()

    def load_coords(self):
        timestamp="20210804.151912"

        self.landing_coordinates = [StagePosition(x=-0.0033364968,y=-0.003270375,z=0.0039950765,t=0.75048248,r=4.0317333)]

        self.lamella_coordinates = [StagePosition(x=0.0028992212,y=-0.0033259583,z=0.0039267467,t=0.43632805,r=4.0318009)]
        # self.zipped_coordinates = None
        self.zipped_coordinates = list(zip(self.lamella_coordinates, self.landing_coordinates))

        original_landing_images = ((f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_landing_low_res_eb.tif',
                                    f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_landing_high_res_eb.tif',
                                    f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_landing_low_res_eb_ib.tif',
                                    f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_landing_high_res_eb_ib.tif'))
        self.original_landing_images = list()
        for image in original_landing_images:
            self.original_landing_images.append(AdornedImage.load(image))


        original_trench_images= ((f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_lamella_low_res_eb.tif',
                                  f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_lamella_high_res_eb.tif',
                                  f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_lamella_low_res_eb_ib.tif',
                                  f'C:/Users/Admin/Github/autoliftout/liftout/gui/log/run/{timestamp}/img/01_ref_lamella_high_res_eb_ib.tif'))
        self.original_trench_images = list()
        for image in original_trench_images:
            self.original_trench_images.append(AdornedImage.load(image))

        self.original_landing_images = [self.original_landing_images]
        self.original_trench_images = [self.original_trench_images]
        # TODO: not this

    def single_liftout(self, landing_coordinates, lamella_coordinates,
                       original_landing_images, original_lamella_area_images):
        # self.stage.absolute_move(lamella_coordinates)
        # calibration.correct_stage_drift(self.microscope, self.auto.image_settings, original_lamella_area_images, self.liftout_counter, mode='eb')
        # self.image_SEM = acquire.last_image(self.microscope, beam_type=BeamType.ELECTRON)
        # # TODO: possibly new image
        # self.ask_user(beam_type=BeamType.ELECTRON, message=f'Is the lamella currently centered in the image?\n'
        #                                                    f'If not, double click to center the lamella, press Yes when centered.', click='double', filter_strength=self.filter_strength)
        # if not self.auto.response:
        #     print(f'Drift correction for sample {self.liftout_counter} did not work')
        #     return
        #
        # self.auto.image_settings['save'] = True
        # self.auto.image_settings['label'] = f'{self.liftout_counter:02d}_post_drift_correction'
        # self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        #
        # # mill
        # self.mill_lamella()
        #
        # # liftout
        # self.liftout_lamella()
        #
        # # landing
        # self.land_lamella(landing_coordinates, original_landing_images)

        # reset
        self.reset_needle()

    def mill_lamella(self):
        self.auto.current_status = AutoLiftoutStatus.Milling
        stage_settings = MoveSettings(rotate_compucentric=True)

        # move flat to the ion beam, stage tilt 25 (total image tilt 52)
        movement.move_to_trenching_angle(self.microscope, self.auto.settings)

        # Take an ion beam image at the *milling current*
        self.auto.image_settings['hfw'] = 100e-6  # TODO: add to protocol
        self.microscope.beams.ion_beam.horizontal_field_width.value = self.auto.image_settings['hfw']
        self.microscope.beams.electron_beam.horizontal_field_width.value = self.auto.image_settings['hfw']
        self.update_display(beam_type=BeamType.ION, image_type='new')

        self.ask_user(beam_type=BeamType.ION, message=f'Have you centered the lamella position in the ion beam?'
                                                      f'If not, double click to center the lamella position', click='double', filter_strength=self.filter_strength)

        # mills trenches for lamella
        # milling.mill_trenches(self.microscope, self.auto.settings)

        # reference images of milled trenches
        self.auto.image_settings['save'] = True
        self.auto.image_settings['resolution'] = self.auto.settings['reference_images']['trench_area_ref_img_resolution']
        self.auto.image_settings['dwell_time'] = self.auto.settings['reference_images']['trench_area_ref_img_dwell_time']

        self.auto.image_settings['hfw'] = self.auto.settings['reference_images']['trench_area_ref_img_hfw_lowres']  # TODO: watch image settings through run
        self.auto.image_settings['label'] = f'{self.liftout_counter:02d}_ref_trench_low_res'  # TODO: add to protocol
        eb_lowres, ib_lowres = acquire.take_reference_images(self.microscope, settings=self.auto.image_settings)

        self.auto.image_settings['hfw'] = self.auto.settings['reference_images']['trench_area_ref_img_hfw_highres']
        self.auto.image_settings['label'] = f'{self.liftout_counter:02d}_ref_trench_high_res'  # TODO: add to protocol
        eb_highres, ib_highres = acquire.take_reference_images(self.microscope, settings=self.auto.image_settings)

        reference_images_low_and_high_res = (eb_lowres, eb_highres, ib_lowres, ib_highres)

        # move flat to electron beam
        movement.flat_to_beam(self.microscope, self.auto.settings, pretilt_angle=pretilt, beam_type=BeamType.ELECTRON, )
        # self.auto.image_settings['hfw'] = 400e-6   # TODO: can this be ignored?

        # make sure drift hasn't been too much since milling trenches
        # first using reference images
        calibration.correct_stage_drift(self.microscope, self.auto.image_settings, reference_images_low_and_high_res, self.liftout_counter, mode='ib')

        # TODO: check dwell time value/add to protocol
        # TODO: check artifact of 0.5 into 1 dwell time
        # setup for ML drift correction
        self.auto.image_settings['resolution'] = '1536x1024'
        self.auto.image_settings['dwell_time'] = 1e-6
        self.auto.image_settings['hfw'] = 80e-6
        self.auto.image_settings['autocontrast'] = True
        self.auto.image_settings['save'] = True
        # TODO: deal with resetting label requirement
        self.auto.image_settings['label'] = f'{self.liftout_counter:02d}_drift_correction_ML'
        # then using ML, tilting/correcting in steps so drift isn't too large
        self.correct_stage_drift_with_ML()
        movement.move_relative(self.microscope, t=np.deg2rad(3), settings=stage_settings)
        self.auto.image_settings['label'] = f'{self.liftout_counter:02d}_drift_correction_ML'
        self.correct_stage_drift_with_ML()
        movement.move_relative(self.microscope, t=np.deg2rad(3), settings=stage_settings)
        self.auto.image_settings['label'] = f'{self.liftout_counter:02d}_drift_correction_ML'
        self.correct_stage_drift_with_ML()

        # now we are at the angle for jcut, perform jcut
        # milling.mill_jcut(self.microscope, self.auto.settings)
        # TODO: adjust hfw? check why it changes to 100?
        self.update_display(beam_type=BeamType.ION, image_type='last')
        # TODO: return image with patterning marks
        self.ask_user(beam_type=BeamType.ION, message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False)
        if self.auto.response:
            milling.run_milling(self.microscope, self.auto.settings)
        self.microscope.patterning.mode = 'Parallel'

        # take reference images of the jcut
        self.auto.image_settings['save'] = True
        self.auto.image_settings['label'] = 'jcut_lowres'
        self.auto.image_settings['hfw'] = 150e-6  # TODO: add to protocol
        acquire.take_reference_images(self.microscope, self.auto.image_settings)

        self.auto.image_settings['label'] = 'jcut_highres'
        self.auto.image_settings['hfw'] = 50e-6  # TODO: add to protocol
        acquire.take_reference_images(self.microscope, self.auto.image_settings)
        print('done, ready for liftout')  # TODO: status  bar

    def correct_stage_drift_with_ML(self):
        # TODO: add this autocontrast to a protocol? (because it changes depending on sample)
        # correct stage drift using machine learning
        label = self.auto.image_settings['label']
        for beamType in (BeamType.ION, BeamType.ELECTRON, BeamType.ION):
            # TODO: more elegant labelling convention
            self.auto.image_settings['label'] = label + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')
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
        self.auto.image_settings['save'] = True
        self.auto.image_settings['label'] = f'{self.liftout_counter:02d}_drift_correction_ML_final'
        self.auto.image_settings['autocontrast'] = True
        self.image_SEM, self.image_FIB = acquire.take_reference_images(self.microscope, self.auto.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type='last')
        self.update_display(beam_type=BeamType.ION, image_type='last')

    def liftout_lamella(self):
        self.auto.current_status = AutoLiftoutStatus.Liftout
        # get ready to do liftout by moving to liftout angle
        movement.move_to_liftout_angle(self.microscope, self.auto.settings)

        needle = self.microscope.specimen.manipulator

        # correct stage drift from mill_lamella stage
        self.correct_stage_drift_with_ML()

        # move needle to liftout start position
        park_position = movement.move_needle_to_liftout_position(self.microscope)

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")

        # land needle on lamella
        self.land_needle_on_milled_lamella()

        # sputter platinum
        # TODO: protocol sputter
        fibsem_utils.sputter_platinum(self.microscope, self.auto.settings, whole_grid=False, sputter_time=30) # TODO: check sputter time

        self.auto.image_settings['save'] = True
        self.auto.image_settings['autocontrast'] = True
        self.auto.image_settings['hfw'] = 100e-6
        self.auto.image_settings['label'] = 'landed_Pt_sputter'
        acquire.take_reference_images(self.microscope, self.auto.image_settings)

        milling.jcut_severing_pattern(self.microscope, self.auto.settings) #TODO: tune jcut severing pattern
        self.update_display(beam_type=BeamType.ION, image_type='last')
        self.ask_user(beam_type=BeamType.ION, message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False)
        if self.auto.response:
            milling.run_milling(self.microscope, self.auto.settings)
        else:
            print('Not happy with the pattern')
            return

        self.auto.image_settings['label'] = 'jcut_sever'
        acquire.take_reference_images(self.microscope, self.auto.image_settings)

        # Raise needle 30um
        # TODO: status
        # print('Stage tilt is ', np.rad2deg(self.stage.current_position.t), ' deg...')

        for i in range(3):
            # print("Moving out of trench by 10um")
            z_move_out_from_trench = movement.z_corrected_needle_movement(10e-6, self.stage.current_position.t)
            self.needle.relative_move(z_move_out_from_trench)
            self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
            self.update_display(beam_type=BeamType.ION, image_type="new")
            time.sleep(1)

        # reference images after liftout complete
        self.auto.image_settings['label'] = 'liftout_of_trench'
        acquire.take_reference_images(self.microscope, self.auto.image_settings)

        # move needle to park position
        movement.retract_needle(self.microscope, park_position)

    def land_needle_on_milled_lamella(self):
        needle = self.microscope.specimen.manipulator
        # setup and take reference images of liftout starting position
        self.auto.image_settings['resolution'] = self.auto.settings["reference_images"]["needle_ref_img_resolution"]
        self.auto.image_settings['dwell_time'] = self.auto.settings["reference_images"]["needle_ref_img_dwell_time"]
        self.auto.image_settings['autocontrast'] = True
        self.auto.image_settings['save'] = True

        # low res
        self.auto.image_settings['hfw'] = self.auto.settings["reference_images"]["needle_ref_img_hfw_lowres"]
        self.auto.image_settings['label'] = 'needle_liftout_start_position_lowres'
        acquire.take_reference_images(self.microscope, self.auto.image_settings)
        # high res
        self.auto.image_settings['hfw'] = self.auto.settings["reference_images"]["needle_ref_img_hfw_highres"]
        self.auto.image_settings['label'] = 'needle_liftout_start_position_highres'
        acquire.take_reference_images(self.microscope, self.auto.image_settings)

        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        # calculate shift between lamella centre and needle tip in the electron view
        self.auto.image_settings['hfw'] = self.auto.settings["reference_images"]["needle_ref_img_hfw_lowres"]
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='needle_tip_to_lamella_centre', beamType=BeamType.ELECTRON)

        x_move = movement.x_corrected_needle_movement(-distance_x_m, stage_tilt=self.stage.current_position.t)
        yz_move = movement.y_corrected_needle_movement(distance_y_m, stage_tilt=self.stage.current_position.t)
        needle.relative_move(x_move)
        needle.relative_move(yz_move)

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")
        # calculate shift between lamella centre and needle tip in the ion view
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='needle_tip_to_lamella_centre', beamType=BeamType.ION)
        # calculate shift in xyz coordinates
        # TODO: status
        # print('Stage tilt is ', np.rad2deg(self.stage.current_position.t), ' deg...')
        # print('cos(t) = ', np.cos(self.stage.current_position.t))
        z_distance = distance_y_m / np.cos(self.stage.current_position.t)

        # Calculate movement
        # print('Needle approach from i-beam low res - Z: landing')
        # print('Needle move in Z by half the distance...', z_distance)
        zy_move_half = movement.z_corrected_needle_movement(-z_distance / 2, self.stage.current_position.t)
        needle.relative_move(zy_move_half)

        self.update_display(beam_type=BeamType.ELECTRON, image_type='new')
        self.update_display(beam_type=BeamType.ION, image_type='new')

        self.ask_user(beam_type=BeamType.ION, message='Is the needle safe to move another half step?', click=None, filter_strength=self.filter_strength)
        # TODO: crosshairs here?
        if self.auto.response:
            self.auto.image_settings['hfw'] = self.auto.settings['reference_images']['needle_with_lamella_shifted_img_highres']
            distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='needle_tip_to_lamella_centre', beamType=BeamType.ION)

            # calculate shift in xyz coordinates

            z_distance = distance_y_m / np.cos(self.stage.current_position.t)

            # Calculate movement
            # move in x
            x_move = movement.x_corrected_needle_movement(-distance_x_m)
            # print('x_move = ', x_move)
            self.needle.relative_move(x_move)
            # move in z
            gap = 0.5e-6
            zy_move_gap = movement.z_corrected_needle_movement(-z_distance - gap, self.stage.current_position.t)
            self.needle.relative_move(zy_move_gap)
            # print('Needle move in Z minus gap ... LANDED')

            self.auto.image_settings['save'] = True
            self.auto.image_settings['autocontrast'] = True
            self.auto.image_settings['hfw'] = self.auto.settings["reference_images"]["needle_ref_img_hfw_lowres"]
            self.auto.image_settings['label'] = 'needle_ref_img_lowres'
            acquire.take_reference_images(self.microscope, self.auto.image_settings)
            self.auto.image_settings['hfw'] = self.auto.settings["reference_images"]["needle_ref_img_hfw_highres"]
            self.auto.image_settings['label'] = 'needle_ref_img_highres'
            acquire.take_reference_images(self.microscope, self.auto.image_settings)

    def calculate_shift_distance_metres(self, shift_type, beamType=BeamType.ELECTRON):
        self.auto.image_settings['beam_type'] = beamType
        self.raw_image, self.overlay_image, self.downscaled_image, feature_1_px, feature_1_type, feature_2_px, feature_2_type = \
            calibration.identify_shift_using_machine_learning(self.microscope, self.auto.image_settings, self.auto.settings, self.liftout_counter,
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
        self.ask_user(image=self.overlay_image, message=f'Has the model correctly identified the {feature_1_type} and {feature_2_type} positions?', click=None, crosshairs=False)

        # if something wasn't correctly identified
        if not self.auto.response:
            utils.save_image(image=self.raw_image, save_path=self.auto.image_settings['save_path'], label=self.auto.image_settings['label'] + '_label')

            self.ask_user(image=self.overlay_image, message=f'Has the model correctly identified the {feature_1_type} position?', click=None, crosshairs=False)

            # if feature 1 wasn't correctly identified
            if not self.auto.response:
                self.ask_user(image=self.downscaled_image, message=f'Please click on the correct {feature_1_type} position.'
                                                                   f'Press Yes button when happy with the position', click='single', filter_strength=self.filter_strength, crosshairs=False)
                # TODO: do we want filtering on this image?

                # if new feature position selected
                if self.auto.response:
                    # TODO: check x/y here
                    feature_1_px = (self.yclick, self.xclick)

            self.ask_user(image=self.overlay_image, message=f'Has the model correctly identified the {feature_2_type} position?', click=None, crosshairs=False)

            # if feature 2 wasn't correctly identified
            if not self.auto.response:
                self.ask_user(image=self.downscaled_image, message=f'Please click on the correct {feature_2_type} position.'
                                                                   f'Press Yes button when happy with the position', click='single', filter_strength=self.filter_strength, crosshairs=False)
                # TODO: do we want filtering on this image?

                # if new feature position selected
                if self.auto.response:
                    feature_2_px = (self.yclick, self.xclick)

        return feature_1_px, feature_2_px

    def land_lamella(self, landing_coord, original_landing_images):

        print("Hello Landing")

        stage_settings = MoveSettings(rotate_compucentric=True)
        self.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)

        # move to landing coordinate
        self.stage.absolute_move(landing_coord)
        # TODO: image settings?
        calibration.correct_stage_drift(self.microscope, self.auto.image_settings, original_landing_images, self.liftout_counter, mode="land")
        park_position = movement.move_needle_to_landing_position(self.microscope)


        # # Y-MOVE
        self.auto.image_settings["resolution"] = self.auto.settings["reference_images"]["landing_post_ref_img_resolution"]
        self.auto.image_settings["dwell_time"] = self.auto.settings["reference_images"]["landing_post_ref_img_dwell_time"]
        self.auto.image_settings["hfw"]  = 150e-6 # self.auto.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
        self.auto.image_settings["beam_type"] = BeamType.ELECTRON
        self.auto.image_settings["save"] = True
        self.auto.image_settings["label"] = "landing_needle_land_sample_lowres"

        # needle_eb_lowres, needle_ib_lowres = acquire.take_reference_images(microscope=self.microscope, settings=self.auto.image_settings)

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=self.auto.image_settings["beam_type"])

        y_move = movement.y_corrected_needle_movement(-distance_y_m, self.stage.current_position.t)
        self.needle.relative_move(y_move)

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")


        # Z-MOVE
        self.auto.image_settings["label"] = "landing_needle_land_sample_lowres_after_y_move"
        self.auto.image_settings["beam_type"] = BeamType.ION
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=self.auto.image_settings["beam_type"])

        z_distance = distance_y_m / np.sin(np.deg2rad(52))
        z_move = movement.z_corrected_needle_movement(z_distance, self.stage.current_position.t)
        self.needle.relative_move(z_move)

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")


        ## X-HALF-MOVE
        self.auto.image_settings["label"] = "landing_needle_land_sample_lowres_after_z_move"
        self.auto.image_settings["beam_type"] = BeamType.ELECTRON
        self.auto.image_settings["hfw"] = 150e-6
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=self.auto.image_settings["beam_type"])

        # half move
        x_move = movement.x_corrected_needle_movement(distance_x_m / 2)
        self.needle.relative_move(x_move)

        self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
        self.update_display(beam_type=BeamType.ION, image_type="new")

        ## X-MOVE
        self.auto.image_settings["label"] = "landing_needle_land_sample_lowres_after_z_move"
        self.auto.image_settings["beam_type"] = BeamType.ELECTRON
        self.auto.image_settings["hfw"] = 80e-6
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type='lamella_edge_to_landing_post', beamType=self.auto.image_settings["beam_type"])

        # TODO: gap?
        x_move = movement.x_corrected_needle_movement(distance_x_m)
        self.needle.relative_move(x_move)

        # final reference images
        self.auto.image_settings["save"] = True
        self.auto.image_settings["hfw"]  = 80e-6 # self.auto.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
        self.auto.image_settings["label"] = "E_landing_lamella_final_weld_highres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.auto.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        # WELD TO LANDING POST

        milling.weld_to_landing_post(self.microscope)
        self.update_display(beam_type=BeamType.ION, image_type='last')
        # TODO: return image with patterning marks
        self.ask_user(beam_type=BeamType.ION, message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False)
        if self.auto.response:
            milling.run_milling(self.microscope, self.auto.settings)
        self.microscope.patterning.mode = 'Parallel'

        # final reference images

        self.auto.image_settings["hfw"]  = 100e-6 # self.auto.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
        self.auto.image_settings["label"] = "landing_lamella_final_weld_highres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.auto.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")



        # CUT OFF NEEDLE

        self.auto.image_settings["hfw"]  = self.auto.settings["cut"]["hfw"]
        self.auto.image_settings["label"] = "landing_lamella_pre_cut_off"
        self.auto.image_settings["beam_type"] = BeamType.ION
        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=self.auto.image_settings["beam_type"])

        height = self.auto.settings["cut"]["height"]
        width = self.auto.settings["cut"]["width"]
        depth = self.auto.settings["cut"]["depth"]
        rotation = self.auto.settings["cut"]["rotation"]
        hfw = self.auto.settings["cut"]["hfw"]

        cut_coord = {"center_x": -distance_x_m,
                     "center_y": distance_y_m,
                     "width": width,
                     "height": height,
                     "depth": depth,  # TODO: might need more to get through needle
                     "rotation": rotation, "hfw": hfw}  # TODO: check rotation

        # cut off needle tip
        milling.cut_off_needle(self.microscope, cut_coord=cut_coord)
        self.update_display(beam_type=BeamType.ION, image_type='last')
        # TODO: return image with patterning marks
        self.ask_user(beam_type=BeamType.ION, message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False)
        if self.auto.response:
            milling.run_milling(self.microscope, self.auto.settings)
        self.microscope.patterning.mode = 'Parallel'


        # reference images
        self.auto.image_settings["hfw"]  = 150e-6 # self.auto.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
        self.auto.image_settings["label"] = "landing_lamella_final_cut_lowres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.auto.image_settings)


        self.auto.image_settings["hfw"]  = 80e-6 # self.auto.settings["reference_images"]["landing_post_ref_img_hfw_lowres"] #TODO: fix protocol
        self.auto.image_settings["label"] = "landing_lamella_final_cut_highres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.auto.image_settings)

        # move needle out of trench slowly at first
        for i in range(3):
            z_move_out_from_trench = movement.z_corrected_needle_movement(10e-6, self.stage.current_position.t)
            self.needle.relative_move(z_move_out_from_trench)
            self.update_display(beam_type=BeamType.ELECTRON, image_type="new")
            self.update_display(beam_type=BeamType.ION, image_type="new")
            time.sleep(1)

        # move needle to park position
        movement.retract_needle(self.microscope, park_position)

        # reference images
        self.auto.image_settings["hfw"]  = 150e-6 #TODO: fix protocol
        self.auto.image_settings["label"] = "landing_lamella_final_lowres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.auto.image_settings)

        self.auto.image_settings["hfw"]  = 80e-6  #TODO: fix protocol
        self.auto.image_settings["label"] = "landing_lamella_final_highres"
        acquire.take_reference_images(microscope=self.microscope, settings=self.auto.image_settings)

        # TODO: test this
    def reset_needle(self):

        print("hello sharpen needle")

        # move sample stage out
        movement.move_sample_stage_out(self.microscope)

        # move needle in
        park_position = movement.insert_needle(self.microscope)
        z_move_in = movement.z_corrected_needle_movement(-180e-6, self.stage.current_position.t)
        self.needle.relative_move(z_move_in)

        # needle images
        self.auto.image_settings["resolution"] = self.auto.settings["imaging"]["resolution"]
        self.auto.image_settings["dwell_time"] = self.auto.settings["imaging"]["dwell_time"]
        self.auto.image_settings["hfw"] = self.auto.settings["imaging"]["horizontal_field_width"]
        self.auto.image_settings["save"] = True
        self.auto.image_settings["label"] = "sharpen_needle_initial"
        acquire.take_reference_images(microscope=self.microscope, settings=self.auto.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")
        self.auto.image_settings["beam_type"] = BeamType.ION

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=self.auto.image_settings["beam_type"])

        x_move = movement.x_corrected_needle_movement(distance_x_m)
        self.needle.relative_move(x_move)
        z_distance = distance_y_m / np.sin(np.deg2rad(52))
        z_move = movement.z_corrected_needle_movement(z_distance, self.stage.current_position.t)
        self.needle.relative_move(z_move)

        self.auto.image_settings["label"] = "sharpen_needle_centre"
        acquire.take_reference_images(microscope=self.microscope, settings=self.auto.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        distance_x_m, distance_y_m = self.calculate_shift_distance_metres(shift_type="needle_tip_to_image_centre", beamType=self.auto.image_settings["beam_type"])

        # sharpening parameters
        # height = self.auto.settings["sharpen"]["height"]
        # width = self.auto.settings["sharpen"]["width"]
        # depth = self.auto.settings["sharpen"]["depth"]
        # bias = self.auto.settings["sharpen"]["bias"]
        # hfw = self.auto.settings["sharpen"]["hfw"]
        # tip_angle = self.auto.settings["sharpen"]["tip_angle"]  # 2NA of the needle   2*alpha
        # needle_angle = self.auto.settings["sharpen"][
        #     "needle_angle"
        # ]  # needle tilt on the screen 45 deg +/-
        # milling_current = self.auto.settings["sharpen"]["sharpen_milling_current"]

        # create sharpening patterns
        cut_coord_bottom, cut_coord_top = milling.calculate_sharpen_needle_pattern(microscope=self.microscope, settings=self.auto.settings, x_0=distance_x_m, y_0=distance_y_m)

        milling.create_sharpen_needle_patterns(
            self.microscope, cut_coord_bottom, cut_coord_top
        )

        # confirm and run milling
        self.update_display(beam_type=BeamType.ION, image_type='last')
        # TODO: return image with patterning marks
        self.ask_user(beam_type=BeamType.ION, message='Do you want to run the ion beam milling with this pattern?', filter_strength=self.filter_strength, crosshairs=False)
        if self.auto.response:
            milling.run_milling(self.microscope, self.auto.settings)
        self.microscope.patterning.mode = 'Parallel'

        # take reference images
        self.auto.image_settings["label"] = "sharpen_needle_final"
        acquire.take_reference_images(microscope=self.microscope, settings=self.auto.image_settings)
        self.update_display(beam_type=BeamType.ELECTRON, image_type="last")
        self.update_display(beam_type=BeamType.ION, image_type="last")

        # retract needle
        movement.retract_needle(self.microscope, park_position)

    def cleanup_lamella(self):

        return NotImplemented

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
        self.pushButton_test_popup.clicked.connect(lambda: self.ask_user(beam_type=BeamType.ION, image=test_image, message='test message\n single click', click='single', crosshairs=False, filter_strength=0))

    def ask_user(self, image=None, beam_type=None, message=None, click=None, crosshairs=True, filter_strength=0):
        self.crosshairs = crosshairs
        self.setEnabled(False)
        self.popup = QtWidgets.QDialog()
        self.popup.setLayout(QtWidgets.QGridLayout())
        self.popup.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.popup_canvas = None

        if message is None:
            message = "ok?"

        # Question space
        question_frame = QtWidgets.QWidget(self.popup)
        question = QtWidgets.QLabel()
        question_layout = QtWidgets.QHBoxLayout()
        question_frame.setLayout(question_layout)

        question.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        question.setFixedHeight(50)
        question.setText(message)
        font = QtGui.QFont()
        font.setPointSize(16)
        question.setFont(font)
        question.setAlignment(QtCore.Qt.AlignCenter)
        if filter_strength:
            # filter combo box

            filter_label = QtWidgets.QLabel('Median Filter')
            self.filter_check = QtWidgets.QCheckBox()
            self.filter_check.setChecked(1)
            self.filter_check.clicked.connect(lambda: (self.update_popup_display(click=click, beam_type=beam_type, image=image, crosshairs=self.crosshairs, filter_strength=filter_strength*int(self.filter_check.checkState()/2))))

            filter_frame = QtWidgets.QWidget()
            filter_frame_layout = QtWidgets.QGridLayout()
            filter_frame.setLayout(filter_frame_layout)
            filter_frame.layout().addWidget(filter_label, 0, 0, 1, 1)
            filter_frame.layout().addWidget(self.filter_check, 0, 1, 1, 1)

        new_image = QtWidgets.QPushButton()
        new_image.setFixedHeight(self.button_height)
        new_image.setFixedWidth(self.button_width)
        new_image.setText('New Image')

        if self.crosshairs:
            crosshair_label = QtWidgets.QLabel()
            crosshair_label.setText('Central Crosshair')

            crosshair_check = QtWidgets.QCheckBox()
            crosshair_check.setChecked(1)
            crosshair_check.clicked.connect(lambda: update_crosshair_value())
            crosshair_check.clicked.connect(lambda: self.update_popup_display(click=click, image=image, beam_type=beam_type, crosshairs=self.crosshairs, filter_strength=filter_strength))

        def update_crosshair_value():
            self.crosshairs = not self.crosshairs


        if self.crosshairs:
            question_frame.layout().addWidget(crosshair_label)
            question_frame.layout().addWidget(crosshair_check)
        question_frame.layout().addWidget(question)
        if filter_strength:
            question_frame.layout().addWidget(filter_frame)
        question_frame.layout().addWidget(new_image)

        # HFW changing
        hfw_widget = QtWidgets.QWidget()
        hfw_widget_layout = QtWidgets.QGridLayout()
        hfw_widget.setLayout(hfw_widget_layout)

        hfw_slider = QtWidgets.QSlider()
        hfw_slider.setOrientation(QtCore.Qt.Horizontal)
        hfw_slider.setMinimum(1)
        if beam_type == BeamType.ELECTRON:
            hfw_slider.setMaximum(2700)
        else:
            hfw_slider.setMaximum(900)
        hfw_slider.setValue(self.auto.image_settings['hfw']*1e6)

        hfw_spinbox = QtWidgets.QSpinBox()
        hfw_spinbox.setMinimum(1)
        if beam_type == BeamType.ELECTRON:
            hfw_spinbox.setMaximum(2700)
        else:
            hfw_spinbox.setMaximum(900)
        hfw_spinbox.setValue(self.auto.image_settings['hfw']*1e6)


        hfw_slider.valueChanged.connect(lambda: hfw_spinbox.setValue(hfw_slider.value()))
        hfw_slider.valueChanged.connect(lambda: hfw_spinbox.setValue(hfw_slider.value()))

        hfw_spinbox.valueChanged.connect(lambda: hfw_slider.setValue(hfw_spinbox.value()))

        hfw_widget.layout().addWidget(hfw_spinbox)
        hfw_widget.layout().addWidget(hfw_slider)

        new_image.clicked.connect(lambda: self.image_from_popup(hfw=hfw_slider.value()*1e-6, beam_type=beam_type,
                                  click=click, crosshairs=self.crosshairs, filter_strength=filter_strength))

        # Button space
        button_box = QtWidgets.QWidget(self.popup)
        button_box.setFixedHeight(int(self.button_height*1.2))
        button_layout = QtWidgets.QGridLayout()
        yes = QtWidgets.QPushButton('Yes')
        yes.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        yes.setFixedHeight(self.button_height)
        yes.setFixedWidth(self.button_width)
        yes.clicked.connect(lambda: self.set_response(True))
        yes.clicked.connect(lambda: self.popup.close())

        no = QtWidgets.QPushButton('No')
        no.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        no.setFixedHeight(self.button_height)
        no.setFixedWidth(self.button_width)
        no.clicked.connect(lambda: self.set_response(False))
        no.clicked.connect(lambda: self.popup.close())

        # spacers
        h_spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        h_spacer2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        # button layout
        button_box.setLayout(button_layout)
        button_box.layout().addItem(h_spacer, 0, 0, 2, 1)
        button_box.layout().addWidget(yes, 0, 1, 2, 1)
        button_box.layout().addWidget(no, 0, 2, 2, 1)
        button_box.layout().addItem(h_spacer2, 0, 3, 2, 1)

        # image space
        self.update_popup_display(click=click, image=image, beam_type=beam_type, crosshairs=self.crosshairs, filter_strength=filter_strength)


        self.popup.destroyed.connect(lambda: self.setEnabled(True))
        self.popup.layout().addWidget(question_frame, 6, 1, 1, 1)
        self.popup.layout().addWidget(hfw_widget, 7, 1, 1, 1)
        self.popup.layout().addWidget(button_box, 8, 1, 1, 1)
        self.popup.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.popup.show()
        self.popup.exec_()

    def image_from_popup(self, hfw, beam_type=None, click=None, crosshairs=False, filter_strength=0):
        self.auto.image_settings['hfw'] = hfw
        # print(self.auto.image_settings['hfw']*1e6)
        image = acquire.new_image(self.microscope, self.auto.image_settings)
        if beam_type == BeamType.ELECTRON:
            self.image_SEM = image
        elif beam_type == BeamType.ION:
            self.image_FIB = image

        self.update_popup_display(click=click, beam_type=beam_type, image=image, crosshairs=self.crosshairs, filter_strength=filter_strength)

    def on_gui_click(self, event, click,  image, beam_type=None, crosshairs=True, filter_strength=0):
        if event.inaxes:
            if event.button == 1:
                if event.dblclick and (click in ('double', 'all')):
                    if image:
                        self.xclick = event.xdata
                        self.yclick = event.ydata
                        x, y = movement.pixel_to_realspace_coordinate([self.xclick, self.yclick], image)

                        x_move = movement.x_corrected_stage_movement(x, stage_tilt=self.stage.current_position.t)
                        yz_move = movement.y_corrected_stage_movement(y, stage_tilt=self.stage.current_position.t, beam_type=beam_type)
                        self.stage.relative_move(x_move)
                        self.stage.relative_move(yz_move)
                        # TODO: refactor beam type here
                        acquire.new_image(microscope=self.microscope, settings=self.auto.image_settings)
                        if beam_type:
                            self.update_display(beam_type=beam_type, image_type='last')
                        self.update_popup_display(click=click, beam_type=beam_type, image=image, crosshairs=self.crosshairs, filter_strength=filter_strength)

                elif click in ('single', 'all'):
                    self.xclick = event.xdata
                    self.yclick = event.ydata

                    cross_size = 120
                    half_cross = cross_size/2
                    cross_thickness = 2
                    half_thickness = cross_thickness/2

                    h_rect = plt.Rectangle((event.xdata, event.ydata-half_thickness), half_cross, cross_thickness)
                    h_rect2 = plt.Rectangle((event.xdata-half_cross, event.ydata-half_thickness), half_cross, cross_thickness)

                    v_rect = plt.Rectangle((event.xdata-half_thickness, event.ydata), cross_thickness, half_cross)
                    v_rect2 = plt.Rectangle((event.xdata-half_thickness, event.ydata-half_cross), cross_thickness, half_cross)

                    h_rect.set_color('xkcd:yellow')
                    h_rect2.set_color('xkcd:yellow')
                    v_rect.set_color('xkcd:yellow')
                    v_rect2.set_color('xkcd:yellow')

                    click_crosshair = (h_rect, h_rect2, v_rect, v_rect2)
                    print(click_crosshair)
                    self.update_popup_display(click=click, image=image, beam_type=beam_type, crosshairs=self.crosshairs, filter_strength=filter_strength, click_crosshair=click_crosshair)

    def update_popup_display(self, click, image, beam_type=None, crosshairs=True, filter_strength=0, click_crosshair=None):
        print(f'Filter strength: {filter_strength} x {filter_strength}')
        # if self.popup.canvas:
        #     for i in reversed(range(self.popup.layout.count())):
        #         self.popup.layout.itemAt(i).widget().setParent(None)

        fig = plt.figure(99)
        if self.popup_canvas:
            self.popup.layout().removeWidget(self.popup_canvas)
            self.popup.layout().removeWidget(self.popup_toolbar)
            self.popup_canvas.deleteLater()
            self.popup_toolbar.deleteLater()
        self.popup_canvas = _FigureCanvas(fig)

        if click:
            self.popup_canvas.mpl_connect('button_press_event', lambda event: self.on_gui_click(event, image=image, beam_type=beam_type, click=click, crosshairs=self.crosshairs, filter_strength=filter_strength))
        image_array = image.data

        if filter_strength:
            image_array = ndi.median_filter(image_array, filter_strength)

        if image_array.ndim != 3:
            image_array = np.stack((image_array,) * 3, axis=-1)

        # Cross hairs
        xshape = image_array.shape[0]
        yshape = image_array.shape[1]
        midx = int(xshape / 2)
        midy = int(yshape / 2)

        cross_size = 120
        half_cross = cross_size / 2
        cross_thickness = 2
        half_thickness = cross_thickness / 2

        self.h_rect = plt.Rectangle((midx, midy - half_thickness),
                               half_cross, cross_thickness)

        self.h_rect2 = plt.Rectangle(
            (midx - half_cross, midy - half_thickness),
            half_cross, cross_thickness)

        self.v_rect = plt.Rectangle((midx - half_thickness, midy),
                               cross_thickness, half_cross)
        self.v_rect2 = plt.Rectangle(
            (midx - half_thickness, midy - half_cross),
            cross_thickness, half_cross)

        self.h_rect.set_color('xkcd:yellow')
        self.h_rect2.set_color('xkcd:yellow')
        self.v_rect.set_color('xkcd:yellow')
        self.v_rect2.set_color('xkcd:yellow')

        fig.clear()
        self.ax = fig.add_subplot(111)
        self.ax.imshow(image_array)

        self.ax.patches = []
        if self.crosshairs:
            self.ax.add_patch(self.h_rect)
            self.ax.add_patch(self.v_rect)
            self.ax.add_patch(self.h_rect2)
            self.ax.add_patch(self.v_rect2)
        if click_crosshair:
            for patch in click_crosshair:
                self.ax.add_patch(patch)
        self.popup_canvas.draw()

        self.popup_toolbar = _NavigationToolbar(self.popup_canvas, self)
        self.popup.layout().addWidget(self.popup_toolbar, 1, 1, 1, 1)
        self.popup.layout().addWidget(self.popup_canvas, 2, 1, 4, 1)

    def set_response(self, response):
        self.auto.response = response
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
                print(f'Unexpected parameter in template file')
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
        print(protocol_text)
        p = yaml.safe_load(protocol_text)
        print(p)
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
        print(f'Please select protocol file (yml)')
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
                    print('Not a yml configuration file')
                    load_directory = filedialog.askopenfile(mode='r', filetypes=[('yml files', '*.yml')])

                with open(load_directory.name, 'r') as file:
                    _dict = yaml.safe_load(file)
                    file.close()
            except Exception:
                display_error_message(traceback.format_exc())
            for key in _dict:
                if key not in self.key_list_protocol:
                    if checked:
                        print(f'Unexpected parameter in protocol file')
                    checked = 0
        root.destroy()
        print(_dict)

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
                        self.auto.image_settings['beam_type'] = beam_type
                        self.image_SEM = acquire.new_image(self.microscope, self.auto.image_settings)
                    else:
                        self.image_SEM = acquire.last_image(self.microscope, beam_type=beam_type)
                    image_array = self.image_SEM.data
                    self.figure_SEM.clear()
                    self.figure_SEM.patch.set_facecolor((240/255, 240/255, 240/255))
                    self.ax_SEM = self.figure_SEM.add_subplot(111)
                    self.ax_SEM.get_xaxis().set_visible(False)
                    self.ax_SEM.get_yaxis().set_visible(False)
                    self.ax_SEM.imshow(image_array, cmap='gray')
                    self.canvas_SEM.draw()
                if beam_type is BeamType.ION:
                    if image_type == 'new':
                        self.auto.image_settings['beam_type'] = beam_type
                        self.image_FIB = acquire.new_image(self.microscope, self.auto.image_settings)
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
        print('Running cleanup/teardown')
        logging.debug('Running cleanup/teardown')
        if self.objective_stage and self.offline is False:
            # Return objective lens stage to the 'out' position and disconnect.
            self.move_absolute_objective_stage(self.objective_stage, position=0)
            self.objective_stage.disconnect()
        if self.microscope:
            self.microscope.disconnect()


def display_error_message(message):
    """PyQt dialog box displaying an error message."""
    print('display_error_message')
    logging.exception(message)
    error_dialog = QtWidgets.QErrorMessage()
    error_dialog.showMessage(message)
    error_dialog.exec_()
    return error_dialog


def main(offline='True'):
    if offline.lower() == 'false':
        logging.basicConfig(level=logging.WARNING)
        launch_gui(ip_address='10.0.0.1', offline=False)
    elif offline.lower() == 'true':
        logging.basicConfig(level=logging.DEBUG)
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
main(offline='True')
