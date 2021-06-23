"""Main entry script."""
import click
from datetime import datetime
import time
import os, sys, glob, getopt
import logging
from enum import Enum
import numpy as np
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy import fftpack, misc
from PIL import Image, ImageDraw, ImageFilter
from matplotlib.patches import Circle
from scipy.ndimage.morphology import binary_dilation
from skimage.filters import gaussian, threshold_otsu, median
from skimage.measure import label
from skimage.morphology import disk
import datetime

from scipy import *
from scipy import signal
from scipy import fftpack, misc
import scipy
import os, sys, glob

import skimage.draw
import skimage.io

from operator import itemgetter

from liftout.calibration import setup
from liftout.user_input import load_config, protocol_stage_settings
# from liftout.milling import mill_lamella
# from liftout.needle import liftout_lamella, land_lamella

#from liftout.align import *
#from liftout.milling import *
#from liftout.needle import *
#from liftout.acquire import *
#from liftout.calibration import *
#from liftout.display import *
#from liftout.stage_movement import *
from liftout.user_input import *

PRETILT_DEGREES = 27

class BeamType(Enum):
    ION = 'ION'
    ELECTRON = 'ELECTRON'


# GLOBAL VARIABLE
class Storage():
    def __init__(self, DIR=''):
        self.DIR = DIR
        self.NEEDLE_REF_IMGS         = [] # dict()
        self.NEEDLE_WITH_SAMPLE_IMGS = [] # dict()
        self.LANDING_POSTS_REF       = []
        self.TRECHNING_POSITIONS_REF = []
        self.MILLED_TRENCHES_REF     = []
        self.liftout_counter = 0
        self.step_counter   = 0
        self.settings = ''
    def AddDirectory(self,DIR):
        self.DIR = DIR
    def NewRun(self, prefix='RUN'):
        self.__init__(self.DIR)
        if self.DIR == '':
            self.DIR = os.getcwd() # # dirs = glob.glob(saveDir + "/ALIGNED_*")        # nn = len(dirs) + 1
        stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')
        self.saveDir = self.DIR + '/' + prefix + '_' + stamp
        self.saveDir = self.saveDir.replace('\\', '/')
        os.mkdir(self.saveDir)
    def SaveImage(self, image, dir_prefix='', id=''):
        if len(dir_prefix) > 0:
            self.path_for_image = self.saveDir + '/'  + dir_prefix + '/'
        else:
            self.path_for_image = self.saveDir + '/'  + 'liftout%03d'%(self.liftout_counter) + '/'
        print(self.path_for_image)
        if not os.path.isdir( self.path_for_image ):
            print('creating directory')
            os.mkdir(self.path_for_image)
        self.fileName = self.path_for_image + 'step%02d'%(self.step_counter) + '_'  + id + '.tif'
        print(self.fileName)
        image.save(self.fileName)
storage = Storage() # global variable

def configure_logging(log_filename='logfile', log_level=logging.INFO):
    """Log to the terminal and to file simultaneously."""
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')#datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[
            logging.FileHandler(log_filename+timestamp+'.log'),
            logging.StreamHandler(),
        ])


def initialize(ip_address='10.0.0.1'):
    """Initialize connection to FIBSEM microscope with Autoscript."""
    from autoscript_sdb_microscope_client import SdbMicroscopeClient

    microscope = SdbMicroscopeClient()
    microscope.connect(ip_address)
    return microscope


def get_all_metadata(image):
    from autoscript_sdb_microscope_client.utilities import IniMetadataReader
    metadata_reader = IniMetadataReader()
    metadata_dictionary = metadata_reader.read_from_string(image.metadata.metadata_as_ini)
    for key in [k for k in metadata_dictionary]:
        value = metadata_dictionary[key]
        print(key, "=", value)


def beamtype_from_image(image):
    """Find the beam type used to acquire an AdornedImage.
    Parameters
    ----------
    image : AdornedImage
    """
    metadata_string = image.metadata.metadata_as_ini
    if "Beam=EBeam" in metadata_string:
        return BeamType.ELECTRON
    elif "Beam=IBeam" in metadata_string:
        return BeamType.ION
    else:
        raise RuntimeError("Beam type not recorded in image metadata!")



def quick_plot(image, median_smoothing=3, show=True):
    """Display image with matplotlib.pyplot
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    display_image = image.data
    if median_smoothing is not None:
        display_image = ndi.median_filter(display_image, size=median_smoothing)
    height, width = display_image.shape
    try:
        pixelsize_x = image.metadata.binary_result.pixel_size.x
        pixelsize_y = image.metadata.binary_result.pixel_size.y
    except AttributeError:
        extent_kwargs = [-(width / 2), +(width / 2), -(height / 2), +(height / 2)]
        ax.set_xlabel("Distance from origin (pixels)")
    else:
        extent_kwargs = [
            -(width  / 2) * pixelsize_x,
            +(width  / 2) * pixelsize_x,
            -(height / 2) * pixelsize_y,
            +(height / 2) * pixelsize_y,
        ]
        ax.set_xlabel(
            "Distance from origin (meters) \n" "1 pixel = {} meters".format(pixelsize_x)
        )
    ax.set_xlim(extent_kwargs[0], extent_kwargs[1])
    ax.set_ylim(extent_kwargs[2], extent_kwargs[3])
    ax.imshow(display_image, cmap="gray", extent=extent_kwargs)
    if show is True:
        fig.show()
    return fig, ax


def select_point(image):
    """Return location of interactive user click on image.
    Parameters
    ----------
    image : AdornedImage or 2D numpy array.
    Returns
    -------
    coords
          Coordinates of last point clicked in the image.
          Coordinates are in x, y format.
          Units are the same as the matplotlib figure axes.
    """
    fig, ax = quick_plot(image)
    coords = []

    def on_click(event):
        print(event.xdata, event.ydata)
        coords.append(event.ydata)
        coords.append(event.xdata)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return np.flip(coords[-2:], axis=0)  # coordintes in x, y format



def plot_overlaid_images(image_1, image_2, show=True):
    """Plot two images overlaid with partial transparency.
    Parameters
    ----------
    image_1 : AdornedImage or numpy.ndarray
        The first image to overlay (will appear blue)
    image_2 : AdornedImage or numpy.ndarray
        The second image to overlay (will appear orange)
    show : boolean, optional.
        Whether to display the matplotlib figure on screen immedicately.
    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """
    # If AdornedImage are passed in, convert to bare numpy arrays
    if hasattr(image_1, 'data'):
        image_1 = image_1.data
    if hasattr(image_2, 'data'):
        image_2 = image_2.data
    # Axes shown in pixels, not in real space.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(image_1, cmap='Blues_r', alpha=1)
    ax.imshow(image_2, cmap='Oranges_r', alpha=0.5)
    if show is True:
        fig.show()
    return fig, ax







def autocontrast(microscope, beam_type=BeamType.ELECTRON):
    """Atuomatically adjust the microscope image contrast.
    """
    from autoscript_sdb_microscope_client.structures import RunAutoCbSettings

    if beam_type == BeamType.ELECTRON:
        microscope.imaging.set_active_view(1)  # the electron beam view
    elif beam_type == BeamType.ION:
        microscope.imaging.set_active_view(2)  # the ion beam view

    autocontrast_settings = RunAutoCbSettings(
        method="MaxContrast",
        resolution="768x512",  # low resolution, so as not to damage the sample
        number_of_frames=5,
    )
    logging.info("Automatically adjusting contrast...")
    microscope.auto_functions.run_auto_cb()
    return autocontrast_settings


# def _reduced_area_rectangle(reduced_area_coords):
#     assert len(reduced_area_coords) == 4
#     top_corner_x, top_corner_y, width, height = reduced_area_coords
#     return Rectangle(top_corner_x, top_corner_y, width, height)


# def create_camera_settings(imaging_settings, reduced_area_coords=[0, 0, 1, 1]):
#     """Camera settings for acquiring images on the microscope.
#     Parameters
#     ----------
#     imaging_settings : dictionary
#         User input as dictionary containing keys "resolution" and "dwell_time".
#     reduced_area_coords : Rectangle, optional
#         Reduced area view for image acquisition.
#         By default None, which will create a Rectangle(0, 0, 1, 1),
#         which means the whole field of view will be imaged.
#     Returns
#     -------
#     GrabFrameSettings
#         Camera acquisition settings
#     """
#     from autoscript_sdb_microscope_client.structures import (GrabFrameSettings,
#                                                              Rectangle)
#     reduced_area = _reduced_area_rectangle(reduced_area_coords)
#     camera_settings = GrabFrameSettings(
#         resolution=imaging_settings["resolution"],
#         dwell_time=imaging_settings["dwell_time"],
#         reduced_area=reduced_area,
#     )
#     return camera_settings


def new_electron_image(microscope, settings=None, brightness=None, contrast=None):
    """Take new electron beam image.
    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    microscope.imaging.set_active_view(1)  # the electron beam view
    if brightness:
        microscope.detector.brightness.value = brightness
    if contrast:
        microscope.detector.contrast.value = contrast
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image


def new_ion_image(microscope, settings=None, brightness=None, contrast=None):
    """Take new ion beam image.
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    microscope.imaging.set_active_view(2)  # the ion beam view
    if brightness:
        microscope.detector.brightness.value = brightness
    if contrast:
        microscope.detector.contrast.value = contrast
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image

def take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6,  image_settings=None, __autocontrast=True,
                eb_brightness=None, ib_brightness=None, eb_contrast=None, ib_contrast=None):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    # image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    #############
    # Take reference images with lower resolution, wider field of view
    microscope.beams.electron_beam.horizontal_field_width.value = hor_field_width
    microscope.beams.ion_beam.horizontal_field_width.value      = hor_field_width
    microscope.imaging.set_active_view(1)
    if __autocontrast:
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
    eb_reference = new_electron_image(microscope, image_settings, eb_brightness, eb_contrast)
    microscope.imaging.set_active_view(2)
    if __autocontrast:
        autocontrast(microscope, beam_type=BeamType.ION)
    ib_reference = new_ion_image(microscope, image_settings, ib_brightness, ib_contrast)
    return eb_reference, ib_reference



def flat_to_electron_beam(microscope, *, pretilt_angle=PRETILT_DEGREES):
    """Make the sample surface flat to the electron beam.
    """
    from autoscript_sdb_microscope_client.structures import (StagePosition,
                                                             MoveSettings)
    stage = microscope.specimen.stage
    rotation = storage.settings["system"]["stage_rotation_flat_to_electron"]
    rotation = np.deg2rad(rotation)
    tilt = np.deg2rad(pretilt_angle)
    stage_settings = MoveSettings(rotate_compucentric=True)
    # If we rotating by a lot, tilt to zero so stage doesn't hit anything
    if abs(np.rad2deg(rotation - stage.current_position.r)) > 90:
        stage.absolute_move(StagePosition(t=0), stage_settings)  # just in case
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position


def flat_to_ion_beam(microscope, *, pretilt_angle=PRETILT_DEGREES):
    """Make the sample surface flat to the ion beam.
    """
    from autoscript_sdb_microscope_client.structures import (StagePosition,
                                                             MoveSettings)
    stage = microscope.specimen.stage
    rotation = storage.settings["system"]["stage_rotation_flat_to_ion"]
    rotation = np.deg2rad(rotation)
    tilt = np.deg2rad(52 - pretilt_angle)
    stage_settings = MoveSettings(rotate_compucentric=True)
    # If we rotating by a lot, tilt to zero so stage doesn't hit anything
    if abs(np.rad2deg(rotation - stage.current_position.r)) > 90:
        stage.absolute_move(StagePosition(t=0), stage_settings)  # just in case
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position



def move_to_trenching_angle(microscope, *, pretilt_angle=PRETILT_DEGREES):
    """Tilt the sample stage to the correct angle for milling trenches.
    Assumes trenches should be milled with the sample surface flat to ion beam.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.
    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    flat_to_ion_beam(microscope, pretilt_angle=pretilt_angle)
    return microscope.specimen.stage.current_position


def move_to_liftout_angle(microscope, *, liftout_angle=10, pretilt_angle=PRETILT_DEGREES):
    """Tilt the sample stage to the correct angle for liftout.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    flat_to_electron_beam(microscope, pretilt_angle=pretilt_angle)
    microscope.specimen.stage.relative_move(StagePosition(t=np.deg2rad(liftout_angle)))
    return microscope.specimen.stage.current_position


def move_to_landing_angle(microscope, *, landing_angle=18, pretilt_angle=PRETILT_DEGREES):
    """Tilt the sample stage to the correct angle for the landing posts.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    flat_to_ion_beam(microscope, pretilt_angle=pretilt_angle) # stage tilt 25
    microscope.specimen.stage.relative_move(StagePosition(t=np.deg2rad(landing_angle))) # more tilt by 18
    new_ion_image(microscope)
    return microscope.specimen.stage.current_position


def move_to_sample_grid(microscope, *, pretilt_angle=PRETILT_DEGREES):
    """Move stage and zoom out to see the whole sample grid.
    Assumes sample grid is mounted on the left hand side of the holder.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    flat_to_electron_beam(microscope, pretilt_angle=pretilt_angle)
    sample_grid_center = StagePosition(x=-0.0025868173, y=0.0031794167, z=0.0039457213)
    microscope.specimen.stage.absolute_move(sample_grid_center)
    # Zoom out so you can see the whole sample grid
    microscope.beams.ion_beam.horizontal_field_width.value = 100e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 100e-6
    #new_electron_image(microscope)
    return microscope.specimen.stage.current_position

def move_to_landing_grid(microscope, *, pretilt_angle=PRETILT_DEGREES,
                         flat_to_sem=True):
    """Move stage and zoom out to see the whole landing post grid.
    Assumes the landing grid is mounted on the right hand side of the holder.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.
    flat_to_sem : bool, optional
        Whether to keep the landing post grid surface flat to the SEM.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    if flat_to_sem:
        flat_to_electron_beam(microscope)
        landing_grid_position = StagePosition(x=+0.0034580609,
                                              y=+0.0032461667,
                                              z=0.0039338733)
    else:
        move_to_landing_angle(microscope, pretilt_angle=pretilt_angle)
        landing_grid_position = StagePosition(x=-0.0034580609,
                                              y=-0.0032461667,
                                              z=0.0039338733)
    microscope.specimen.stage.absolute_move(landing_grid_position)
    # Zoom out so you can see the whole landing grid
    microscope.beams.ion_beam.horizontal_field_width.value = 100e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 100e-6
    #new_electron_image(microscope)
    return microscope.specimen.stage.current_position

def move_to_jcut_angle(microscope, *, jcut_angle=6., pretilt_angle=PRETILT_DEGREES):
    """Tilt the sample to the Jcut angle.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    jcut_angle : float, optional
        Tilt angle for the stage when milling the J-cut, in degrees
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.
    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    flat_to_electron_beam(microscope, pretilt_angle=pretilt_angle)
    microscope.specimen.stage.relative_move(StagePosition(t=np.deg2rad(jcut_angle)))
    return microscope.specimen.stage.current_position


def move_sample_stage_out(microscope):
    """Move stage completely out of the way, so it is not visible at all.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    # Must set tilt to zero, so we don't see reflections from metal stage base
    microscope.specimen.stage.absolute_move(StagePosition(t=0))  # important!
    sample_stage_out = StagePosition(x=-0.002507,
                                     y=0.025962792,
                                     z=0.0039559049)
    microscope.specimen.stage.absolute_move(sample_stage_out)
    return microscope.specimen.stage.current_position


def linked_within_z_tolerance(microscope, expected_z=3.9e-3, tolerance=1e-6):
    """Check if the sample stage is linked and at the expected z-height.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    expected_z : float, optional
        Correct height for linked stage in z, in meters, by default 4e-3
    tolerance : float, optional
        Must be within this absolute tolerance of expected stage z height,
        in meters, by default 1e-4
    """
    # Check the microscope stage is at the correct height
    z_stage_height = microscope.specimen.stage.current_position.z
    if np.isclose(z_stage_height, expected_z, atol=tolerance):
        return True
    else:
        return False


def z_corrected_stage_movement(expected_z, stage_tilt):
    """Stage movement in Z, corrected for tilt of sample surface plane.
    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in radians
    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    y_move = -np.sin(stage_tilt) * expected_z
    z_move = +np.cos(stage_tilt) * expected_z
    return StagePosition(x=0, y=y_move, z=z_move)


def auto_link_stage(microscope, expected_z=3.9e-3, tolerance=1e-6):
    """Automatically focus and link sample stage z-height.
    expected_z : float, optional
        Correct height for linked stage in z, in meters, by default 4e-3
    tolerance : float, optional
        Must be within this absolute tolerance of expected stage z height,
        in meters, by default 1e-4
    """
    # SAMPLE GRID expected_z = 3.9e-3
    # LANDING GRID expected_z = 4.05e-3
    # How to auto-link z for the landing posts
    #    1. Make landing grid flat to SEM
    #    2. Zoom really far in on a flat part that isn't part of the posts
    #    3. Auto-link z, using a DIFFERENT expected_z height (4.05 mm)
    from autoscript_sdb_microscope_client.structures import StagePosition
    print("auto_link_stage")
    print(microscope.specimen.stage.current_position.z)
    microscope.imaging.set_active_view(1)  # the electron beam view
    original_hfw = microscope.beams.electron_beam.horizontal_field_width.value
    microscope.beams.electron_beam.horizontal_field_width.value = 0.000400
    print('Automatically focusing and linking stage z-height.')
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    microscope.auto_functions.run_auto_focus()
    microscope.specimen.stage.link()
    z_difference = expected_z - microscope.specimen.stage.current_position.z
    if abs(z_difference) > 3e-3:
        raise RuntimeError("ERROR: the reported stage position is likely incorrect!")
    z_move = z_corrected_stage_movement(
        z_difference, microscope.specimen.stage.current_position.t)
    microscope.specimen.stage.relative_move(z_move)
    print(microscope.specimen.stage.current_position.z)
    # iteration if need be
    counter = 0
    while not linked_within_z_tolerance(microscope,
                                        expected_z=expected_z,
                                        tolerance=tolerance):
        if counter > 3:
            raise(UserWarning("Could not auto-link z stage height."))
            break
        # Focus and re-link z stage height
        print('Automatically focusing and linking stage z-height.')
        microscope.auto_functions.run_auto_focus()
        microscope.specimen.stage.link()
        z_difference = expected_z - microscope.specimen.stage.current_position.z
        z_move = z_corrected_stage_movement(
            z_difference, microscope.specimen.stage.current_position.t)
        microscope.specimen.stage.relative_move(z_move)
        print(microscope.specimen.stage.current_position.z)
    # Restore original settings
    microscope.beams.electron_beam.horizontal_field_width.value = original_hfw
    new_electron_image(microscope)


def refocus_and_relink(microscope, expected_z=3.95e-3):
    #from autoscript_sdb_microscope_client.structures import RunAutoFocusSettings
    #autofocus_settings = RunAutoFocusSettings(resolution="6144x4096")
    """Automatically focus and link sample stage z-height.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    print("refocus_and_relink_stage")
    print(microscope.specimen.stage.current_position.z)
    microscope.imaging.set_active_view(1)  # the electron beam view
    original_hfw = microscope.beams.electron_beam.horizontal_field_width.value
    #microscope.beams.electron_beam.horizontal_field_width.value = 0.000400
    print('Automatically refocusing and relinking stage z-height.')
    microscope.auto_functions.run_auto_focus()
    microscope.specimen.stage.link()
    z_difference = expected_z - microscope.specimen.stage.current_position.z
    if abs(z_difference) > 3e-3:
        raise RuntimeError("ERROR: the reported stage position is likely incorrect!")
    #microscope.beams.electron_beam.horizontal_field_width.value = original_hfw
    new_electron_image(microscope)


def sputter_platinum(microscope, sputter_time=60, *,
                     sputter_application_file="cryo_Pt_dep",
                     default_application_file="autolamella",
                     horizontal_field_width=100e-6,
                     line_pattern_length=15e-6):
    """Sputter platinum over the sample.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    sputter_time : int, optionalye
        Time in seconds for platinum sputtering. Default is 60 seconds.
    sputter_application_file : str
        Application file for platinum sputtering/deposition.
    default_application_file : str
        Default application file, to return to after the platinum sputtering.
    """
    # Setup
    original_active_view = microscope.imaging.get_active_view()
    microscope.imaging.set_active_view(1)  # the electron beam view
    microscope.patterning.clear_patterns()
    microscope.patterning.set_default_application_file(sputter_application_file)
    microscope.patterning.set_default_beam_type(1)  # set electron beam for patterning
    multichem = microscope.gas.get_multichem()
    multichem.insert()
    # Create sputtering pattern
    microscope.beams.electron_beam.horizontal_field_width.value = horizontal_field_width
    pattern = microscope.patterning.create_line(-line_pattern_length/2,  # x_start
                                                +line_pattern_length,    # y_start
                                                +line_pattern_length/2,  # x_end
                                                +line_pattern_length,    # y_end
                                                2e-6)                    # milling depth
    pattern.time = sputter_time + 0.1
    # Run sputtering with progress bar
    microscope.beams.electron_beam.blank()
    if microscope.patterning.state == "Idle":
        print('Sputtering with platinum for {} seconds...'.format(sputter_time))
        microscope.patterning.start()  # asynchronous patterning
    else:
        raise RuntimeError(
            "Can't sputter platinum, patterning state is not ready."
        )
    for i in tqdm.tqdm(range(int(sputter_time))):
        time.sleep(1)  # update progress bar every second
    if microscope.patterning.state == "Running":
        microscope.patterning.stop()
    else:
        logging.warning("Patterning state is {}".format(microscope.patterning.state))
        logging.warning("Consider adjusting the patterning line depth.")
    # Cleanup

    microscope.beams.electron_beam.unblank()
    microscope.patterning.set_default_application_file(default_application_file)
    microscope.imaging.set_active_view(original_active_view)
    microscope.patterning.set_default_beam_type(2)  # set ion beam
    multichem.retract()
    logging.info("Sputtering finished.")


def sputter_platinum_over_whole_grid(microscope):
    """Sputter platnium over whole grid."""
    stage = microscope.specimen.stage
    move_to_sample_grid(microscope)
    auto_link_stage(microscope, expected_z=5e-3)
    # TODO: yaml user input for sputtering application file choice
    sputter_platinum(microscope, sputter_time=5, horizontal_field_width=30e-6, line_pattern_length=7e-6)
    auto_link_stage(microscope)  # return stage to default linked z height


def insert_needle(microscope):
    """Insert the needle and return the needle parking position.
    Returns
    -------
    park_position : autoscript_sdb_microscope_client.structures.ManipulatorPosition
        The parking position for the needle manipulator when inserted.
    """
    needle = microscope.specimen.manipulator
    needle.insert()
    park_position = needle.current_position
    return park_position


def retract_needle(microscope, park_position):
    """Retract the needle and multichem, preserving the correct park position.
    park_position : autoscript_sdb_microscope_client.structures.ManipulatorPosition
        The parking position for the needle manipulator when inserted.
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    # Retract the multichem
    multichem = microscope.gas.get_multichem()
    multichem.retract()
    # Retract the needle, preserving the correct parking postiion
    needle = microscope.specimen.manipulator
    current_position = needle.current_position
    # To prevent collisions with the sample; first retract in z, then y, then x
    needle.relative_move(ManipulatorPosition(z=park_position.z - current_position.z))  # noqa: E501
    needle.relative_move(ManipulatorPosition(y=park_position.y - current_position.y))  # noqa: E501
    needle.relative_move(ManipulatorPosition(x=park_position.x - current_position.x))  # noqa: E501
    time.sleep(1)  # AutoScript sometimes throws errors if you retract too quick?
    needle.retract()
    retracted_position = needle.current_position
    return retracted_position


def move_needle_closer(microscope, *, x_shift=-20e-6, z_shift=-160e-6):
    """Move the needle closer to the sample surface, after inserting.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.sdb_microscope.SdbMicroscopeClient
        The Autoscript microscope object.
    x_shift : float
        Distance to move the needle from the parking position in x, in meters.
    z_shift : float
        Distance to move the needle towards the sample in z, in meters.
        Negative values move the needle TOWARDS the sample surface.
    """
    needle = microscope.specimen.manipulator
    stage = microscope.specimen.stage
    # Needle starts from the parking position (after inserting it)
    # Move the needle back a bit in x, so the needle is not overlapping target
    x_move = x_corrected_needle_movement(x_shift)
    needle.relative_move(x_move)
    # Then move the needle towards the sample surface.
    z_move = z_corrected_needle_movement(z_shift, stage.current_position.t)
    needle.relative_move(z_move)
    # The park position is always the same,
    # so the needletip will end up about 20 microns from the surface.
    return needle.current_position

def x_corrected_needle_movement(expected_x, stage_tilt=None):
    """Needle movement in X, XTGui coordinates (Electron coordinate).
    Parameters
    ----------
    expected_x : float
        in meters
    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    return ManipulatorPosition(x=expected_x, y=0, z=0)  # no adjustment needed

def y_corrected_needle_movement(expected_y, stage_tilt):
    """Needle movement in Y, XTGui coordinates (Electron coordinate).
    Parameters
    ----------
    expected_y : in meters
    stage_tilt : in radians
    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    y_move = +np.cos(stage_tilt) * expected_y
    z_move = +np.sin(stage_tilt) * expected_y
    return ManipulatorPosition(x=0, y=y_move, z=z_move)

def z_corrected_needle_movement(expected_z, stage_tilt):
    """Needle movement in Z, XTGui coordinates (Electron coordinate).
    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in radians
    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    y_move = -np.sin(stage_tilt) * expected_z
    z_move = +np.cos(stage_tilt) * expected_z
    return ManipulatorPosition(x=0, y=y_move, z=z_move)

def move_needle_to_liftout_position(microscope):
    """Move the needle into position, ready for liftout.
    """
    park_position = insert_needle(microscope)
    move_needle_closer(microscope)
    multichem = microscope.gas.get_multichem()
    multichem.insert()
    return park_position

def move_needle_to_landing_position(microscope):
    """Move the needle into position, ready for landing.
    """
    park_position = insert_needle(microscope)
    move_needle_closer(microscope, x_shift=-25e-6)
    return park_position


# def needle_reference_images(microscope, move_needle_to="liftout", dwell_time=10e-6):
#     from autoscript_sdb_microscope_client.structures import GrabFrameSettings
#     move_sample_stage_out(microscope)
#     if move_needle_to == "liftout":
#         park_position = move_needle_to_liftout_position(microscope)
#     elif move_needle_to == "landing":
#         park_position = move_needle_to_landing_position(microscope)
#     # TODO: set field of view in electron & ion beam to match
#     # microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # can't be smaller than 150e-6
#     # microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # can't be smaller than 150e-6
#     # autocontrast(microscope, beam_type=BeamType.ELECTRON)
#     # needle_reference_eb_lowres = new_electron_image(microscope, image_settings)
#     # autocontrast(microscope, beam_type=BeamType.ION)
#     # needle_reference_ib_lowres = new_ion_image(microscope, image_settings)
#     resolution = storage.settings["reference_images"]["needle_ref_img_resolution"]
#     dwell_time = storage.settings["reference_images"]["needle_ref_img_dwell_time"]
#     image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
#     hfw_lowres  = storage.settings["reference_images"]["needle_ref_img_hfw_lowres"]
#     hfw_highres = storage.settings["reference_images"]["needle_ref_img_hfw_highres"]
#     needle_reference_eb_lowres,  needle_reference_ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres,  image_settings=image_settings)
#     needle_reference_eb_highres, needle_reference_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres, image_settings=image_settings)
#     ####
#     retract_needle(microscope, park_position)
#     return (needle_reference_eb_lowres, needle_reference_eb_highres, needle_reference_ib_lowres, needle_reference_ib_highres)



def validate_scanning_rotation(microscope):
    """Ensure the scanning rotation is set to zero."""
    rotation = microscope.beams.ion_beam.scanning.rotation.value
    if rotation is None:
        microscope.beams.ion_beam.scanning.rotation.value = 0
        rotation = microscope.beams.ion_beam.scanning.rotation.value
    if not np.isclose(rotation, 0.0):
        raise ValueError(
            "Ion beam scanning rotation must be 0 degrees."
            "\nPlease change your system settings and try again."
            "\nCurrent rotation value is {}".format(rotation)
        )

def ensure_eucentricity(microscope, *, pretilt_angle=PRETILT_DEGREES):
    """Check the sample stage is at the eucentric height.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object.
    pretilt_angle : float
        Extra tilt added by the cryo-grid sample holder, in degrees.
    """
    # TODO autofocus
    validate_scanning_rotation(microscope)  # ensure scan rotation is zero
    flat_to_electron_beam(microscope, pretilt_angle=pretilt_angle)
    print("Rough eucentric alignment")
    microscope.beams.electron_beam.horizontal_field_width.value = 900e-6
    microscope.beams.ion_beam.horizontal_field_width.value      = 900e-6
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    autocontrast(microscope, beam_type=BeamType.ION)
    _eucentric_height_adjustment(microscope)
    print("Final eucentric alignment")
    microscope.beams.electron_beam.horizontal_field_width.value = 200e-6
    microscope.beams.ion_beam.horizontal_field_width.value      = 200e-6
    _eucentric_height_adjustment(microscope)
    final_electron_image = new_electron_image(microscope)
    final_ion_image = new_ion_image(microscope)
    return final_electron_image, final_ion_image

def _eucentric_height_adjustment(microscope):
    from autoscript_sdb_microscope_client.structures import StagePosition
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings
    #resolution = storage.settings["reference_images"]["needle_ref_img_resolution"]
    #dwell_time = storage.settings["reference_images"]["needle_ref_img_dwell_time"]
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    new_electron_image(microscope, settings=image_settings)
    ask_user("Please double click to center a feature in the SEM?\n"
             "Is the feature centered now? yes/no: ")
    print("Please click the same location in the ion beam image")
    ion_image = new_ion_image(microscope, settings=image_settings)
    click_location = select_point(ion_image)
    _x, fib_delta_y = click_location
    stage = microscope.specimen.stage
    tilt_radians = microscope.specimen.stage.current_position.t
    delta_z = -np.cos(tilt_radians) * fib_delta_y
    microscope.specimen.stage.relative_move(StagePosition(z=delta_z))
    # Could replace this with an autocorrelation (maybe with a fallback to asking for a user click if the correlation values are too low)
    electron_image = new_electron_image(microscope)
    ask_user("Please double click to center a feature in the SEM?\n"
             "Is the feature centered now? yes/no: ")

    resolution = storage.settings["reference_images"]["needle_ref_img_resolution"]
    dwell_time = storage.settings["reference_images"]["needle_ref_img_dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)





def find_coordinates(microscope, name="", move_stage_angle=None):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    """Manually select stage coordinate positions."""
    if move_stage_angle == "trench":
        move_to_sample_grid(microscope)
    elif move_stage_angle == "landing":
        move_to_landing_grid(microscope)
        ensure_eucentricity(microscope)
    coordinates  = []
    landing_post_reference_images = []
    trench_area_reference_images  = []
    ###
    select_another_position = True
    while select_another_position:
        if move_stage_angle == "trench":
            ensure_eucentricity(microscope)
            move_to_trenching_angle(microscope) # flat to ion_beam
        elif move_stage_angle == "landing":
            move_to_landing_angle(microscope)
        microscope.beams.electron_beam.horizontal_field_width.value = 400e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 400e-6  # TODO: yaml use input
        #refocus_and_relink(microscope)
        eb = new_electron_image(microscope)
        ib = new_ion_image(microscope)
        if ask_user(f"Please center the {name} coordinate in the ion beam.\n"
                    f"Is the {name} feature centered in the ion beam? yes/no: "):
            eb = new_electron_image(microscope)
            coordinates.append(microscope.specimen.stage.current_position)
            if move_stage_angle == "landing":
                #microscope.beams.electron_beam.horizontal_field_width.value = 400e-6  # TODO: yaml use input
                #microscope.beams.ion_beam.horizontal_field_width.value      = 400e-6  # TODO: yaml use input
                #ib_low_res = new_ion_image(microscope)
                #eb_low_res = new_electron_image(microscope)
                #microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # TODO: yaml use input
                #microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # TODO: yaml use input
                #ib_high_res = new_ion_image(microscope)
                #eb_high_res = new_electron_image(microscope)
                resolution = storage.settings["reference_images"]["landing_post_ref_img_resolution"]
                dwell_time = storage.settings["reference_images"]["landing_post_ref_img_dwell_time"]
                image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
                hfw_lowres  = storage.settings["reference_images"]["landing_post_ref_img_hfw_lowres"]
                hfw_highres = storage.settings["reference_images"]["landing_post_ref_img_hfw_highres"]
                eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres,  image_settings=image_settings)
                eb_highres, ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres, image_settings=image_settings)
                landing_post_reference_images.append( (eb_lowres, eb_highres, ib_lowres, ib_highres) )
            if move_stage_angle == "trench":
                #microscope.beams.electron_beam.horizontal_field_width.value = 400e-6  # TODO: yaml use input
                #microscope.beams.ion_beam.horizontal_field_width.value      = 400e-6  # TODO: yaml use input
                #ib_low_res = new_ion_image(microscope)
                #eb_low_res = new_electron_image(microscope)
                #microscope.beams.electron_beam.horizontal_field_width.value =  50e-6  # TODO: yaml use input
                #microscope.beams.ion_beam.horizontal_field_width.value      =  50e-6  # TODO: yaml use input
                #ib_high_res = new_ion_image(microscope)
                #eb_high_res = new_electron_image(microscope)
                resolution = storage.settings["reference_images"]["trench_area_ref_img_resolution"]
                dwell_time = storage.settings["reference_images"]["trench_area_ref_img_dwell_time"]
                image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
                hfw_lowres  = storage.settings["reference_images"]["trench_area_ref_img_hfw_lowres"]
                hfw_highres = storage.settings["reference_images"]["trench_area_ref_img_hfw_highres"]
                eb_lowres, ib_lowres = take_electron_and_ion_reference_images(microscope,   hor_field_width=hfw_lowres,  image_settings=image_settings)
                eb_highres, ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres, image_settings=image_settings)
                trench_area_reference_images.append( (eb_lowres, eb_highres, ib_lowres, ib_highres) )

            print(microscope.specimen.stage.current_position)
            select_another_position = ask_user(
                f"Do you want to select another {name} position? "
                f"{len(coordinates)} selected so far. yes/no: ")
    if move_stage_angle == "landing":
        return coordinates, landing_post_reference_images
    else:
        return coordinates, trench_area_reference_images






##################################   CORNER FINDING : FOR NEEDLETIP IDENTIFICATION ##############################
# def gauss_derivative_kernels(size, sizey=None):
#     """ returns x and y derivatives of a 2D
#         gauss kernel array for convolutions """
#     size = int(size)
#     if not sizey:
#         sizey = size
#     else:
#         sizey = int(sizey)
#     y, x = np.mgrid[-size:size+1, -sizey:sizey+1]
#     #x and y derivatives of a 2D gaussian with standard dev half of size
#     # (ignore scale factor)
#     gx = - x * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2)))
#     gy = - y * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2)))
#     return gx,gy
# 
# def gauss_derivatives(im, n, ny=None):
#     """ returns x and y derivatives of an image using gaussian
#         derivative filters of size n. The optional argument
#         ny allows for a different size in the y direction."""
#     gx,gy = gauss_derivative_kernels(n, sizey=ny)
#     imx = signal.convolve(im,gx, mode='same')
#     imy = signal.convolve(im,gy, mode='same')
#     return imx,imy
# 
# def gauss_kernel(size, sizey = None):
#     """ Returns a normalized 2D gauss kernel array for convolutions """
#     size = int(size)
#     if not sizey:
#         sizey = size
#     else:
#         sizey = int(sizey)
#     x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
#     g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
#     return g / g.sum()
# 
# 
# def compute_harris_response(image):
#     """ compute the Harris corner detector response function
#         for each pixel in the image"""
#     #derivatives
#     imx,imy = gauss_derivatives(image, 3)
#     #kernel for blurring
#     gauss = gauss_kernel(3)
#     #compute components of the structure tensor
#     Wxx = signal.convolve(imx*imx,gauss, mode='same')
#     Wxy = signal.convolve(imx*imy,gauss, mode='same')
#     Wyy = signal.convolve(imy*imy,gauss, mode='same')
#     #determinant and trace
#     Wdet = Wxx*Wyy - Wxy**2
#     Wtr  = Wxx + Wyy
#     return Wdet / Wtr
# 
# def get_harris_points(harrisim, min_distance=10, threshold=0.1):
#     """ return corners from a Harris response image
#         min_distance is the minimum nbr of pixels separating
#         corners and image boundary"""
#     #find top corner candidates above a threshold
#     corner_threshold = max(harrisim.ravel()) * threshold
#     harrisim_t = (harrisim > corner_threshold) * 1
#     #get coordinates of candidates
#     candidates = harrisim_t.nonzero()
#     coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
#     #...and their values
#     candidate_values = [harrisim[c[0]][c[1]] for c in coords]
#     #sort candidates
#     index = argsort(candidate_values)
#     #store allowed point locations in array
#     allowed_locations = zeros(harrisim.shape)
#     allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1
#     #select the best points taking min_distance into account
#     filtered_coords = []
#     for i in index:
#         if allowed_locations[coords[i][0]][coords[i][1]] == 1:
#             filtered_coords.append(coords[i])
#             allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),(coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0
#     return filtered_coords
# 
# 
# def OLD___find_needletip_and_target_locations(image):
#     print("Please click the needle tip position")
#     needletip_location = select_point(image)
#     print("Please click the lamella target position")
#     target_location = select_point(image)
#     return needletip_location, target_location
# 
# 
# 
# def find_needletip_shift_in_image_ELECTRON(needle_with_sample_Adorned, needle_reference_Adorned, show=False, median_smoothing=2):
#     try:
#         pixelsize_x = needle_with_sample_Adorned.metadata.binary_result.pixel_size.x
#         pixelsize_y = needle_with_sample_Adorned.metadata.binary_result.pixel_size.y
#     except AttributeError:
#         pixelsize_x = 1
#         pixelsize_y = 1
#     ### TEST
#     ### Find the tip using corner-finding algorithm
#     needle_reference   = needle_reference_Adorned.data
#     needle_with_sample = needle_with_sample_Adorned.data
#     field_width  = pixelsize_x  * needle_with_sample_Adorned.width
#     height, width = needle_reference.shape # search for the tip in only the left half of the image
#     harrisim = compute_harris_response( ndi.median_filter(needle_reference[:, 0:width//2], size=1) )
#     filtered_coords_ref = get_harris_points(harrisim,4)
#     right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))
#     #############
#     if show:
#         plt.figure(1)
#         plt.imshow(needle_reference, cmap='gray')
#         plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
#         plt.plot( [right_outermost_point_ref[1]], [right_outermost_point_ref[0]],   'ro')

#     ### Find the tip from binarized image (mask) and corner-finding algorithm
#     filt   = ndi.median_filter(needle_reference, size=5)
#     thresh = threshold_otsu(filt)
#     binary = filt > thresh
#     mask   = gaussian(binary_dilation(binary, iterations=15), 5)
#     mask_binary = (mask >= 0.51).astype(int)
#     needle_ref_masked = needle_reference * mask_binary
#     ysize, xsize = mask.shape
#     harrisim = compute_harris_response(mask_binary[:, 0:width//2])
#     filtered_coords_mask_ref = get_harris_points(harrisim,4)
#     right_outermost_point_mask_ref = max(filtered_coords_mask_ref, key=itemgetter(1))
#     ####
#     if show:
#         plt.figure(2)
#         plt.imshow(mask_binary,   cmap='gray',     alpha=1)
#         plt.imshow(needle_reference, cmap='Oranges_r',   alpha=0.5)
#         plt.plot([p[1] for p in filtered_coords_mask_ref], [p[0] for p in filtered_coords_mask_ref], 'bo')
#         plt.plot([right_outermost_point_mask_ref[1]], [right_outermost_point_mask_ref[0]], 'ro')
#         plt.plot([right_outermost_point_ref[1]], [right_outermost_point_ref[0]], 'rd')
# 
#     def R(p1, p2):
#         return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) # find distance between two points
#     # two tips found, if the tip coordinate from the noisy real needle image is way off, rely on the tip found from the binarized image
#     #if R(right_outermost_point_ref, right_outermost_point_mask_ref  ) <= 20:
#     #    right_outermost_point = right_outermost_point_ref
#     #else:
#     #    right_outermost_point = right_outermost_point_mask_ref
#     # if ion beam - use harris points, if electron - check R bwt two points and select corner from
#     right_outermost_point = right_outermost_point_ref # for now only use the corner found from the real image, not binarised one

#     # tip position in the reference image :
#     old_tip_x = right_outermost_point[0]
#     old_tip_y = right_outermost_point[1]

#     xmin = min(old_tip_y, mask.shape[1] - old_tip_y)
#     ymin = min(old_tip_x, mask.shape[0] - old_tip_x)
#     rmin = min(xmin, ymin)
#     Dmin = 2 * rmin
#     ####################
#     cmask = circ_mask(size=(Dmin, Dmin), radius=Dmin // 2 - 15, sigma=10)  # circular mask
#     CMASK = np.zeros(needle_reference.shape)
#     CMASK[old_tip_x - Dmin // 2: old_tip_x + Dmin // 2, old_tip_y - Dmin // 2: old_tip_y + Dmin // 2] = cmask
#     # first normalise the reference image, then mask
#     needle_reference_norm = ( needle_reference - np.mean(needle_reference) ) / np.std(needle_reference)
#     reference_circ_norm = needle_reference_norm * CMASK * mask
#     ####################
#     xmin = min(old_tip_x, mask.shape[1] - old_tip_x)
#     ymin = min(old_tip_y, mask.shape[0] - old_tip_y)
#     ELLPS_MASK = np.zeros(needle_reference.shape)
#     elps_mask = ellipse_mask(size=(xmin*2, ymin*2), radius1=xmin-15, radius2=ymin-15, sigma=10)
#     ELLPS_MASK[old_tip_y - ymin : old_tip_y + ymin, old_tip_x - xmin : old_tip_x + xmin] = elps_mask
#     reference_elps_norm = needle_reference_norm * ELLPS_MASK * mask
#     ####################
#     lowpass_pixels  = int( max(needle_reference.shape) / 12 ) # =128 for 1536x1024 image
#     highpass_pixels = int( max(needle_reference.shape)/ 256 ) # =6   for 1536x1024 image
#     sigma = int( 2 * max(needle_reference.data.shape)/1536)   # =2 @ 1536x1024, good for e-beam images
#     print(': High pass filter = ', lowpass_pixels, '; low pass filter = ', highpass_pixels)
#     # nornalise the image before the cross-correlation
#     needle_with_sample_norm = ( needle_with_sample - np.mean(needle_with_sample) ) / np.std(needle_with_sample)
#     xcorr = crosscorrelation(needle_with_sample_norm, reference_elps_norm, bp='yes', lp=lowpass_pixels, hp=highpass_pixels, sigma=2)
#     maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
#     print('\n', maxX, maxY)
#     cen = np.asarray(xcorr.shape) / 2
#     print('centre = ', cen)
#     err = np.array(cen - [maxX, maxY], int)
#     print("Shift between 1 and 2 is = " + str(err))
#     print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
#     new_tip_x = old_tip_x - err[0]
#     new_tip_y = old_tip_y - err[1]
#     x_shift = +1 * ( cen[1] - new_tip_y ) * pixelsize_x
#     y_shift = -1 * ( cen[0] - new_tip_x ) * pixelsize_y
#     print("X-shift to the image centre =  {} meters".format(x_shift))
#     print("Y-shift to the image centre =  {} meters".format(y_shift))
#     if show:
#         plt.figure()
#         plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
#         plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
#         plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='original position')
#         plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='current position')
#         plt.legend()
#     return x_shift, y_shift # shift of location of the tip


# def find_needletip_shift_in_image_ION(needle_with_sample_Adorned, needle_reference_Adorned, show=False, median_smoothing=2):
#     try:
#         pixelsize_x = needle_with_sample_Adorned.metadata.binary_result.pixel_size.x
#         pixelsize_y = needle_with_sample_Adorned.metadata.binary_result.pixel_size.y
#     except AttributeError:
#         pixelsize_x = 1
#         pixelsize_y = 1
#     ### Find the tip using corner-finding algorithm
#     needle_reference   = needle_reference_Adorned.data
#     needle_with_sample = needle_with_sample_Adorned.data
#     field_width  = pixelsize_x  * needle_with_sample_Adorned.width
#     height, width = needle_reference.shape # search for the tip in only the left half of the image
#     # ION images are more "noisy", need more median smooting to find the tip/top position
#     harrisim = compute_harris_response( ndi.median_filter(needle_reference[:, 0:width//2], size=5) )
#     filtered_coords_ref = get_harris_points(harrisim,4)
#     right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))
#     topmost_point_ref         = min(filtered_coords_ref, key=itemgetter(0)) # from ion image we need the topmost point,not the rightmost
#     #############
#     if show:
#         plt.figure(1)
#         plt.imshow(needle_reference, cmap='gray')
#         plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
#         plt.plot( [topmost_point_ref[1]], [topmost_point_ref[0]],   'ro')
#     ### Find the tip from binarized image (mask) and corner-finding algorithm
#     filt   = ndi.median_filter(needle_reference, size=2)
#     thresh = threshold_otsu(filt)
#     binary = filt > thresh
#     mask   = gaussian(binary_dilation(binary, iterations=15), 5)
#     mask_binary = (mask >= 0.51).astype(int)
#     needle_ref_masked = needle_reference * mask_binary
#     ysize, xsize = mask.shape
#     harrisim = compute_harris_response(mask_binary[:, 0:width//2])
#     filtered_coords_mask_ref = get_harris_points(harrisim,4)
#     right_outermost_point_mask_ref = max(filtered_coords_mask_ref, key=itemgetter(1))
#     topmost_point_mask_ref         = min(filtered_coords_mask_ref, key=itemgetter(0))
#     ####
#     if show:
#         plt.figure(2)
#         plt.imshow(mask_binary,   cmap='gray',     alpha=1)
#         plt.imshow(needle_reference, cmap='Oranges_r',   alpha=0.5)
#         plt.plot([p[1] for p in filtered_coords_mask_ref], [p[0] for p in filtered_coords_mask_ref], 'bo')
#         plt.plot([topmost_point_mask_ref[1]], [topmost_point_mask_ref[0]], 'ro')
#         plt.plot([topmost_point_ref[1]], [topmost_point_ref[0]], 'rd')
#     def R(p1, p2):
#         return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) # find distance between two points
#     # two tips found, if the tip coordinate from the noisy real needle image is way off, rely on the tip found from the binarized image
#     #if R(right_outermost_point_ref, right_outermost_point_mask_ref  ) <= 20:
#     #    right_outermost_point = right_outermost_point_ref
#     #else:
#     #    right_outermost_point = right_outermost_point_mask_ref
#     # if ion beam - use harris points, if electron - check R bwt two points and select corner from
#     right_outermost_point = right_outermost_point_ref
#     topmost_point         = topmost_point_ref

#     # tip position in the reference image :
#     old_tip_x = topmost_point[0]
#     old_tip_y = topmost_point[1]

#     xmin = min(old_tip_y, mask.shape[1] - old_tip_y)
#     ymin = min(old_tip_x, mask.shape[0] - old_tip_x)
#     rmin = min(xmin, ymin)
#     Dmin = 2 * rmin
#     ####################
#     cmask = circ_mask(size=(Dmin, Dmin), radius=Dmin // 2 - 15, sigma=10)  # circular mask
#     CMASK = np.zeros(needle_reference.shape)
#     CMASK[old_tip_x - Dmin // 2: old_tip_x + Dmin // 2, old_tip_y - Dmin // 2: old_tip_y + Dmin // 2] = cmask
#     needle_reference_norm = ( needle_reference - np.mean(needle_reference) ) / np.std(needle_reference)
#     reference_circ_norm = needle_reference_norm * CMASK * mask
#     ####################
#     xmin = min(old_tip_x, mask.shape[1] - old_tip_x)
#     ymin = min(old_tip_y, mask.shape[0] - old_tip_y)
#     ELLPS_MASK = np.zeros(needle_reference.shape)
#     elps_mask = ellipse_mask(size=(xmin*2, ymin*2), radius1=xmin-15, radius2=ymin-15, sigma=10)
#     ELLPS_MASK[old_tip_y - ymin : old_tip_y + ymin, old_tip_x - xmin : old_tip_x + xmin] = elps_mask
#     reference_elps_norm = needle_reference_norm * ELLPS_MASK * mask
#     ####################
#     lowpass_pixels  = int( max(needle_reference.shape) / 6 ) # =256 @ 1536x1024, good for i-beam images
#     highpass_pixels = int( max(needle_reference.shape)/ 64 ) # =24  @ 1536x1024, good for i-beam images => need a large highpass to remove noise and ringing
#     sigma = int( 10 * max(needle_reference.data.shape)/1536) # =10 @ 1536x1024, good for i-beam images
#     print(': High pass filter = ', lowpass_pixels, '; low pass filter = ', highpass_pixels)
#     needle_with_sample_norm = ( needle_with_sample - np.mean(needle_with_sample) ) / np.std(needle_with_sample)
#     xcorr = crosscorrelation(needle_with_sample_norm, reference_elps_norm, bp='yes', lp=lowpass_pixels, hp=highpass_pixels, sigma=sigma)
#     maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
#     print('\n', maxX, maxY)
#     cen = np.asarray(xcorr.shape) / 2
#     print('centre = ', cen)
#     err = np.array(cen - [maxX, maxY], int)
#     print("Shift between 1 and 2 is = " + str(err))
#     print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
#     new_tip_x = old_tip_x - err[0]
#     new_tip_y = old_tip_y - err[1]
#     x_shift = ( cen[1] - new_tip_y ) * pixelsize_x
#     y_shift = ( cen[0] - new_tip_x ) * pixelsize_y
#     print("X-shift to the image centre =  {} meters".format(x_shift))
#     print("Y-shift to the image centre =  {} meters".format(y_shift))
#     if show:
#         plt.figure()
#         plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
#         plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
#         plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='original position')
#         plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='current position')
#         plt.legend()
#     return x_shift, y_shift # shift of location of the tip


def calculate_proportional_needletip_distance_from_img_centre(img, show=False):
    """Calculate the shift from the needletip to the image centre as a proportion of the image """
    weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\fresh_full_n10.pt"
    model = load_model(weights_file=weights_file)

    # convert img to numpy array
    img_orig = np.asarray(img.data)

    # model inference + display
    img_np, rgb_mask = model_inference(model, img=img_orig)

    # detect and draw lamella centre, and needle tip
    (
        lamella_centre_px,
        rgb_mask_lamella,
        needle_tip_px,
        rgb_mask_needle,
        rgb_mask_combined,
    ) = detect_and_draw_lamella_and_needle(rgb_mask)

    # prediction overlay
    img_overlay = show_overlay(img_np, rgb_mask_combined)

    # resize overlay back to full sized image for display
    img_overlay_resized = Image.fromarray(img_overlay).resize((img_np.shape[1], img_np.shape[0]))

    if show:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img_orig, cmap='Blues_r', alpha=1)
        ax.imshow(img_overlay_resized, cmap='Oranges_r', alpha=0.5)

        plt.show()

    # # need to use the same scale images for both detection selections
    img_downscale = Image.fromarray(img_orig).resize((rgb_mask_combined.size[0], rgb_mask_combined.size[1]))

    print("Confirm Needle Tip Position")
    needle_tip_px = validate_detection(img_downscale, img, needle_tip_px, "needle tip")

    # scale invariant coordinatesss
    scaled_lamella_centre_px, scaled_needle_tip_px = scale_invariant_coordinates(
        needle_tip_px, lamella_centre_px, rgb_mask_combined
    )

    # if no needle tip is found, something has gone wrong
    if scaled_needle_tip_px is None:
        raise ValueError("No needle tip detected")

   # x, y
    return -(0.5 - scaled_needle_tip_px[1]), 0.5 - scaled_needle_tip_px[0]


def needletip_shift_from_img_centre(img, show=True):
    """Calculate the shift in metres from needle tip to lamella centre """
    needle_distance_x, needle_distance_y = calculate_proportional_needletip_distance_from_img_centre(img, show=True)
    # pixelsize_x = img.metadata.binary_result.pixel_size.x
    # field_width   = pixelsize_x  * img.width
    # field_height  = pixelsize_x  * img.height
    # x_shift = needle_distance_x * field_width
    # y_shift = needle_distance_y * field_height
    # print('x_shift = ', x_shift/1e-6, 'um; ', 'y_shift = ', y_shift/1e-6, 'um; ')

    x_shift, y_shift = calculate_shift_distance_in_metres(img, needle_distance_x, needle_distance_y)

    return x_shift, y_shift


def calculate_needletip_shift_from_lamella_centre(img, show=False):
    """ Calculate the shift from the needle tip to the lamella centre as a proportion of the
    image.
    """

    weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\fresh_full_n10.pt"
    model = load_model(weights_file=weights_file)

    # convert img to numpy array
    img_orig = np.asarray(img.data)

    # model inference + display
    img_np, rgb_mask = model_inference(model, img=img_orig)

    # detect and draw lamella centre, and needle tip
    (
        lamella_centre_px,
        rgb_mask_lamella,
        needle_tip_px,
        rgb_mask_needle,
        rgb_mask_combined,
    ) = detect_and_draw_lamella_and_needle(rgb_mask)

    # prediction overlay
    img_overlay = show_overlay(img_np, rgb_mask_combined)

    # resize overlay back to full sized image for display
    img_overlay_resized = Image.fromarray(img_overlay).resize((img_np.shape[1], img_np.shape[0]))

    if show:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img_orig, cmap='Blues_r', alpha=1)
        ax.imshow(img_overlay_resized, cmap='Oranges_r', alpha=0.5)

        plt.show()

    # # need to use the same scale images for both detection selections
    img_downscale = Image.fromarray(img_orig).resize((rgb_mask_combined.size[0], rgb_mask_combined.size[1]))

    print("Confirm Needle Tip Position")
    needle_tip_px = validate_detection(img_downscale, img, needle_tip_px, "needle tip")

    print("Confirm Lamella Centre Position")
    lamella_centre_px = validate_detection(img_downscale, img, lamella_centre_px, "lamella_centre")


    # scale invariant coordinatesss
    scaled_lamella_centre_px, scaled_needle_tip_px = scale_invariant_coordinates(
        needle_tip_px, lamella_centre_px, rgb_mask_combined
    )

    # if no needle tip is found, something has gone wrong
    if scaled_needle_tip_px is None:
        raise ValueError("No needle tip detected")

    # x, y
    return -(scaled_lamella_centre_px[1] - scaled_needle_tip_px[1]), scaled_lamella_centre_px[0] - scaled_needle_tip_px[0]

def needletip_shift_from_lamella_centre(img, show=True):
    """Calculate the shift in metres from needle tip to lamella centre """
    needle_distance_x, needle_distance_y = calculate_needletip_shift_from_lamella_centre(img, show=True)

    # pixelsize_x = img.metadata.binary_result.pixel_size.x
    # field_width   = pixelsize_x  * img.width
    # field_height  = pixelsize_x  * img.height
    # x_shift = needle_distance_x * field_width
    # y_shift = needle_distance_y * field_height
    # print('x_shift = ', x_shift/1e-6, 'um; ', 'y_shift = ', y_shift/1e-6, 'um; ')

    x_shift, y_shift = calculate_shift_distance_in_metres(img, needle_distance_x, needle_distance_y)

    return x_shift, y_shift

def calculate_lamella_edge_shift_from_landing_post(img, landing_px, show=False):
    """Calculate the shift from the right edge of the lamella to the edge of the landing post

    The distance is calculated as a proportion of the image

    """
    weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\fresh_full_n10.pt"
    model = load_model(weights_file=weights_file)

    # convert img to numpy array
    img_orig = np.asarray(img.data)

    # model inference + display
    img_np, rgb_mask = model_inference(model, img=img_orig)

    # detect and draw lamella right edge
    lamella_right_edge_px, rgb_mask_combined = detect_and_draw_lamella_right_edge(rgb_mask)
    print("Lamella Right Edge Pixel: ", lamella_right_edge_px)

    # prediction overlay
    img_overlay = show_overlay(img_np, rgb_mask_combined)

    # resize overlay back to full sized image for display
    img_overlay_resized = Image.fromarray(img_overlay).resize((img_np.shape[1], img_np.shape[0]))

    if show:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img_orig, cmap='Blues_r', alpha=1)
        ax.imshow(img_overlay_resized, cmap='Oranges_r', alpha=0.5)

        plt.show()

    # # need to use the same scale images for both detection selections
    img_downscale = Image.fromarray(img_orig).resize((rgb_mask_combined.size[0], rgb_mask_combined.size[1]))

    print("Confirm Lamella Edge Position")
    lamella_right_edge_px = validate_detection(img_downscale, img, lamella_right_edge_px, "lamella_edge")

    scaled_lamella_right_edge_px = scale_invariant_coordinates_NEW(lamella_right_edge_px, rgb_mask_combined)

    ## LANDING EDGE

    # use the initially selected landing point, and snap to the nearest edge
    edge_landing_px, edges = detect_closest_landing_point(img_orig, landing_px)
    landing_px_mask = draw_landing_edges_and_point(img_orig, edges, edge_landing_px)

    print("Confirm Landing Post Position")
    edge_landing_px = validate_detection(img_orig, img, edge_landing_px, "landing_post")

    # scale landing coordinates
    scaled_edge_landing_px = scale_invariant_coordinates_NEW(edge_landing_px, landing_px_mask)

    print("Landing Point:", edge_landing_px)
    print("Proportional: ", scaled_edge_landing_px)

    # if no needle tip is found, something has gone wrong
    if scaled_lamella_right_edge_px is None:
        raise ValueError("No Lamella Edge detected")

    # if no landing post is found, something has gone wrong
    if scaled_edge_landing_px is None:
        raise ValueError("No Landing Post detected")

    # x, y
    return -(scaled_lamella_right_edge_px[1] - scaled_edge_landing_px[1]), scaled_lamella_right_edge_px[0] - scaled_edge_landing_px[0],

# TODO: replace all functions with this helper
def calculate_shift_distance_in_metres(img, distance_x, distance_y):
    """Convert the shift distance from proportion of img to metres"""

    pixelsize_x = img.metadata.binary_result.pixel_size.x #5.20833e-008
    field_width   = pixelsize_x  * img.width
    field_height  = pixelsize_x  * img.height
    x_shift = distance_x * field_width
    y_shift = distance_y * field_height
    print('x_shift = ', x_shift/1e-6, 'um; ', 'y_shift = ', y_shift/1e-6, 'um; ')

    return x_shift, y_shift

def lamella_edge_to_landing_post(img, landing_px, show=True):
    """Calculate the shift in metres from needle tip to lamella centre """
    landing_post_distance_x, landing_post_distance_y = calculate_lamella_edge_shift_from_landing_post(img, landing_px=landing_px, show=True)

    x_shift, y_shift = calculate_shift_distance_in_metres(img, landing_post_distance_x, landing_post_distance_y)
    return x_shift, y_shift

def land_needle_on_milled_lamella(microscope, move_in_x=True, move_in_y=True, xcorrection=1e-6, ycorrection=1.5e-6):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    # image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=2e-6)
    stage  = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    resolution = storage.settings["reference_images"]["needle_ref_img_resolution"]
    dwell_time = storage.settings["reference_images"]["needle_ref_img_dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
    hfw_lowres  = storage.settings["reference_images"]["needle_ref_img_hfw_lowres"]
    hfw_highres = storage.settings["reference_images"]["needle_ref_img_hfw_highres"]

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]

    needle_eb_lowres_with_lamella, needle_ib_lowres_with_lamella = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres, image_settings=image_settings,
                                __autocontrast=False,
                                eb_brightness=eb_brightness, ib_brightness=ib_brightness, eb_contrast=eb_contrast, ib_contrast=ib_contrast)
    needle_eb_highres_with_lamella, needle_ib_highres_with_lamella = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres, image_settings=image_settings,
                                __autocontrast=False,
                                eb_brightness=eb_brightness, ib_brightness=ib_brightness, eb_contrast=eb_contrast, ib_contrast=ib_contrast)

    storage.SaveImage(needle_eb_lowres_with_lamella,  id='B_needle_land_sample_eb_lowres' )
    storage.SaveImage(needle_eb_highres_with_lamella, id='B_needle_land_sample_eb_highres')
    storage.SaveImage(needle_ib_lowres_with_lamella,  id='B_needle_land_sample_ib_lowres' )
    storage.SaveImage(needle_ib_highres_with_lamella, id='B_needle_land_sample_ib_highres')

    ######
    #################### LOW & HIGH RES IMAGES ########################
    ############### take images, eb & ib, with the sample+needle ###############
    # pixelsize_x_lowres = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.x
    # field_width_lowres = pixelsize_x_lowres * needle_ref_eb_lowres_nobg.width
    # needle_eb_lowres_with_lamella, needle_ib_lowres_with_lamella = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres, image_settings=image_settings)
    ###
    # pixelsize_x_highres = needle_ref_eb_highres_nobg.metadata.binary_result.pixel_size.x
    # field_width_highres = pixelsize_x_highres * needle_ref_eb_highres_nobg.width
    # needle_eb_highres_with_lamella, needle_ib_highres_with_lamella = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres, image_settings=image_settings)
    ###
    # storage.SaveImage(needle_eb_lowres_with_lamella,  id='B_needle_land_sample_eb_lowres' )
    # storage.SaveImage(needle_eb_highres_with_lamella, id='B_needle_land_sample_eb_highres')
    # storage.SaveImage(needle_ib_lowres_with_lamella,  id='B_needle_land_sample_ib_lowres' )
    # storage.SaveImage(needle_ib_highres_with_lamella, id='B_needle_land_sample_ib_highres')

    #refocus_and_relink(microscope)
    ############ FIND dx, dy from LOW_RES ELECTRON images ############

    x_shift, y_shift = needletip_shift_from_lamella_centre(needle_eb_lowres_with_lamella)

    # (x,y)-correction for e-beam due to the shift in the SEM, (ion beam is aligned, but SEM is shifted due to the eucentricity blah blah)
    xcorrection = 1e-6*0 # empirically found
    ycorrection = 2e-6*0 # empirically found
    x_move = x_corrected_needle_movement(-x_shift + xcorrection)
    y_move = y_corrected_needle_movement(-y_shift + ycorrection, stage.current_position.t)
    print('Needle approach from e-beam high row res:')
    print('x_move = ', x_move, ';\ny_move = ', y_move)
    needle.relative_move(x_move)
    needle.relative_move(y_move)

    ############################### take IMAGES ###############################
    print('Needle approach from e-beam high res. Taking images after shifting the needle in XY')
    ### take low and high images again

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
    microscope.beams.ion_beam.horizontal_field_width.value      = hfw_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = hfw_lowres
    needle_eb_lowres_with_lamella_shifted = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    needle_ib_lowres_with_lamella_shifted = new_ion_image(microscope,      settings=image_settings, brightness=ib_brightness, contrast=ib_contrast)

    microscope.beams.ion_beam.horizontal_field_width.value      = hfw_highres
    microscope.beams.electron_beam.horizontal_field_width.value = hfw_highres
    needle_eb_highres_with_lamella_shifted = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    needle_ib_highres_with_lamella_shifted = new_ion_image(microscope,      settings=image_settings, brightness=ib_brightness, contrast=ib_contrast)

    storage.SaveImage(needle_eb_lowres_with_lamella_shifted,  id='C_needle_land_sample_eb_lowres_shifted' )
    storage.SaveImage(needle_eb_highres_with_lamella_shifted, id='C_needle_land_sample_eb_highres_shifted')
    storage.SaveImage(needle_ib_lowres_with_lamella_shifted,  id='C_needle_land_sample_ib_lowres_shifted' )
    storage.SaveImage(needle_ib_highres_with_lamella_shifted, id='C_needle_land_sample_ib_highres_shifted')

    '''
    ############ FIND dx, dy from HIGH_RES ELECTRON images ############
    x_shift, y_shift = find_needletip_shift_in_image_ELECTRON(needle_eb_highres_with_lamella_shifted,
                                                               needle_ref_eb_highres_nobg, show=False, median_smoothing=2)
    x_move = x_corrected_needle_movement(x_shift + xcorrection)
    y_move = y_corrected_needle_movement(y_shift + ycorrection, stage.current_position.t)
    print('Needle approach from e-beam high res:')
    print('x_move = ', x_move, ';\ny_move = ', y_move)
    needle.relative_move(x_move)
    needle.relative_move(y_move)
    #######
    #######
    '''

    #############################  find dz from LOW RES ION  ############################
    # take images, eb & ib, with the sample+needle - correct X of the needle from lowres eb
    ### RETAKE THE IMAGES AFTER dx,dy SHIFT
    hfw_lowres = storage.settings["reference_images"]["needle_with_lamella_shifted_img_lowres"]
    needle_eb_highres_with_lamella_shifted, needle_ib_highres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres,
                                            image_settings=image_settings, __autocontrast=False)
    # save
    storage.SaveImage(needle_eb_lowres_with_lamella_shifted,  id='D_needle_land_sample_eb_lowres_shifted' )
    storage.SaveImage(needle_ib_lowres_with_lamella_shifted,  id='D_needle_land_sample_ib_lowres_shifted' )

    # Z-MOVEMENT 1, HALF THE DISTANCE
    x_shift, y_shift = needletip_shift_from_lamella_centre(needle_ib_lowres_with_lamella_shifted)

    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    print('cos(t) = ', np.cos(stage_tilt))
    z_distance = y_shift / np.cos(stage_tilt)
    # Calculate movement
    print('Needle approach from i-beam low res - Z: landing')
    zy_move_half = z_corrected_needle_movement(z_distance / 2 , stage_tilt)
    print('Needle move in Z by half the distance...')
    needle.relative_move(zy_move_half)



    # Z-MOVEMENT 2 - TO LAND, higher res images
    hfw_highres = storage.settings["reference_images"]["needle_with_lamella_shifted_img_highres"]
    needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6,
        image_settings=image_settings, __autocontrast=False)
    storage.SaveImage(needle_eb_lowres_with_lamella_shifted,  id='E_needle_land_sample_eb_lowres_shifted' )
    storage.SaveImage(needle_ib_lowres_with_lamella_shifted,  id='E_needle_land_sample_ib_lowres_shifted' )

    x_shift, y_shift = needletip_shift_from_lamella_centre(needle_ib_lowres_with_lamella_shifted)

    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    print('cos(t) = ', np.cos(stage_tilt))
    z_distance = y_shift / np.cos(stage_tilt)
    # Calculate movement
    print('Needle approach from i-beam low res - Z: landing')

    gap = 1e-6
    zy_move_gap = z_corrected_needle_movement(z_distance - gap , stage_tilt)
    print('Needle move in Z minus gap ... LANDED')

    # move in x
    x_move = x_corrected_needle_movement(-x_shift)
    print('x_move = ', x_move)
    needle.relative_move(x_move)

    needle.relative_move(zy_move_gap)



    #############################  LANDED, take pictures  ############################
    ### RETAKE THE IMAGES AFTER dx,dy,dz SHIFT and landing

    hfw_lowres  = storage.settings["reference_images"]["needle_ref_img_hfw_lowres"]
    hfw_highres = storage.settings["reference_images"]["needle_ref_img_hfw_highres"]
    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
    microscope.beams.ion_beam.horizontal_field_width.value      = hfw_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = hfw_lowres
    needle_eb_lowres_with_lamella_landed = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    needle_ib_lowres_with_lamella_landed = new_ion_image(microscope,      settings=image_settings, brightness=ib_brightness, contrast=ib_contrast)

    microscope.beams.ion_beam.horizontal_field_width.value      = hfw_highres
    microscope.beams.electron_beam.horizontal_field_width.value = hfw_highres
    needle_eb_highres_with_lamella_landed = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    needle_ib_highres_with_lamella_landed = new_ion_image(microscope,      settings=image_settings, brightness=ib_brightness, contrast=ib_contrast)

    storage.SaveImage(needle_eb_lowres_with_lamella_landed,  id='E_needle_land_sample_eb_lowres_landed' )
    storage.SaveImage(needle_eb_highres_with_lamella_landed, id='E_needle_land_sample_eb_highres_landed')
    storage.SaveImage(needle_ib_lowres_with_lamella_landed,  id='E_needle_land_sample_ib_lowres_landed' )
    storage.SaveImage(needle_ib_highres_with_lamella_landed, id='E_needle_land_sample_ib_highres_landed')
    storage.step_counter += 1


# def OLD__manual_needle_movement_in_xy(microscope, move_in_x=True, move_in_y=True):
#     from autoscript_sdb_microscope_client.structures import GrabFrameSettings
#     stage = microscope.specimen.stage
#     needle = microscope.specimen.manipulator
#     electron_image = new_electron_image(microscope, settings=GrabFrameSettings(dwell_time=500e-9, resolution="1536x1024"))  # TODO: User input imaging settings
#     needletip_location, target_location = find_needletip_and_target_locations(electron_image)
#     # Calculate needle movements
#     x_needletip_location = needletip_location[0]  # coordinates in x-y format
#     y_needletip_location = needletip_location[1]  # coordinates in x-y format
#     x_target_location = target_location[0]  # pixels, coordinates in x-y format
#     y_target_location = target_location[1]  # pixels, coordinates in x-y format
#     if move_in_y is True:
#         y_distance = y_target_location - y_needletip_location
#         y_move = y_corrected_needle_movement(y_distance, stage.current_position.t)
#         needle.relative_move(y_move)
#     if move_in_x is True:  # MUST MOVE X LAST! Avoids hitting the sample
#         x_distance = x_target_location - x_needletip_location
#         x_move = x_corrected_needle_movement(x_distance)
#         needle.relative_move(x_move)


# def OLD__manual_needle_movement_in_z(microscope):
#     from autoscript_sdb_microscope_client.structures import GrabFrameSettings
#     stage = microscope.specimen.stage
#     needle = microscope.specimen.manipulator
#     ion_image = new_ion_image(microscope, settings=GrabFrameSettings(dwell_time=500e-9, resolution="1536x1024"))  # TODO: user input imaging settings
#     print("Please click the needle tip position")
#     needletip_location = select_point(ion_image)
#     print("Please click the lamella target position")
#     target_location = select_point(ion_image)
#     # Calculate movement
#     z_safety_buffer = 400e-9  # in meters TODO: yaml user input
#     z_distance = -(target_location[1] - needletip_location[1] / np.sin(np.deg2rad(52))) - z_safety_buffer
#     z_move = z_corrected_needle_movement(z_distance, stage.current_position.t)
#     needle.relative_move(z_move)


########################################LIFT-OUT#########################################################
def liftout_lamella(microscope, settings):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    needle = microscope.specimen.manipulator
    stage = microscope.specimen.stage
    # fix field of view to match the reference images
    microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # can't be smaller than 150e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # can't be smaller than 150e-6
    # needletip_ref_location_eb = ??? TODO: automated needletip identification
    # needletip_ref_location_ib = ??? TODO: automated needletip identification
    park_position = move_needle_to_liftout_position(microscope)
    land_needle_on_milled_lamella(microscope, move_in_x=True, move_in_y=True)
    sputter_platinum(microscope, sputter_time=30)  # TODO: yaml user input for sputtering application file choice
    eb, ib = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6, image_settings=image_settings)
    storage.step_counter += 1
    storage.SaveImage(eb, id='eb_landed_Pt_sputter')
    storage.SaveImage(ib, id='ib_landed_Pt_sputter')
    storage.step_counter += 1


    mill_to_sever_jcut(microscope, settings['jcut'], confirm=False)  # TODO: yaml user input for jcut milling current
    eb, ib = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6, image_settings=image_settings)
    storage.SaveImage(eb, id='eb_jcut_sever' )
    storage.SaveImage(ib, id='ib_jcut_sever')
    storage.step_counter += 1

    # TAKE NEEDLE z_UP (>30 MICRONS), TAKE GIS OUT, RESTRACT TO PARKING
    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    z_move_out_from_trench = z_corrected_needle_movement(10e-6, stage_tilt)
    needle.relative_move(z_move_out_from_trench)

    eb, ib = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
    storage.step_counter += 1
    storage.SaveImage(eb, id='eb_jcut_sever_liftout' )
    storage.SaveImage(ib, id='ib_jcut_sever_liftout')
    storage.step_counter += 1



    retract_needle(microscope, park_position)

    # liftout is finished, no need for reference images


    ####################### reference lamella-on-needle images without background #########################
    #needle_reference_images_with_lamella = needle_reference_images(microscope, move_needle_to="landing")
    # need smaller dwell time when there is a sample on the needle
    # move_sample_stage_out(microscope)
    # park_position = move_needle_to_landing_position(microscope)

    # resolution = storage.settings["reference_images"]["needle_with_lamella_ref_img_resolution"]
    # dwell_time_electron = storage.settings["reference_images"]["needle_with_lamella_ref_img_dwell_time_electron"]
    # dwell_time_ion      = storage.settings["reference_images"]["needle_with_lamella_ref_img_dwell_time_ion"]
    # hfw_highres = storage.settings["reference_images"]["needle_with_lamella_ref_img_hfw_highres"]
    # image_settings_electron = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time_electron)
    # image_settings_ion      = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time_ion)
    # microscope.beams.ion_beam.horizontal_field_width.value      = hfw_highres
    # microscope.beams.electron_beam.horizontal_field_width.value = hfw_highres
    # microscope.imaging.set_active_view(1)
    # autocontrast(microscope, beam_type=BeamType.ELECTRON)
    # ref_landingPos_eb_highres = microscope.imaging.grab_frame(image_settings_electron)
    # microscope.imaging.set_active_view(2)
    # autocontrast(microscope, beam_type=BeamType.ION)
    # ref_landingPos_ib_highres = microscope.imaging.grab_frame(image_settings_ion)
    # needle_reference_images_highres_with_lamella = (ref_landingPos_eb_highres, ref_landingPos_ib_highres)
    # retract_needle(microscope, park_position)
    # return None #needle_reference_images_highres_with_lamella





############################################# MILLING #######################################################

def mill_trenches(microscope, settings, confirm=True):
    """Mill the trenches for the lamella.
    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings :  Dictionary of user input argument settings.
    confirm : bool, optional
        Whether to ask the user to confirm before milling.
    """
    if confirm is True:
        if not ask_user("Have you centered the lamella position in the ion beam? yes/no \n"):
            print("Ok, cancelling trench milling.")
            return
    print('Milling trenches')
    protocol_stages = protocol_stage_settings(settings)
    for stage_number, stage_settings in enumerate(protocol_stages):
        print("Protocol stage {} of {}".format(
            stage_number + 1, len(protocol_stages)))
        mill_single_stage(
            microscope,
            settings,
            stage_settings,
            stage_number)
    # Restore ion beam imaging current (20 pico-Amps)
    microscope.beams.ion_beam.beam_current.value = 20e-12


def mill_single_stage(microscope, settings, stage_settings, stage_number):
    """Run ion beam milling for a single milling stage in the protocol.
    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings :  Dictionary of user input argument settings.
    stage_settings : Dictionary of settings for a single protocol milling stage
    stage_number : int. Current milling protocol stage number.
    """
    logging.info(f'Milling trenches, protocol stage {stage_number}')
    demo_mode = settings["demo_mode"]
    upper_milling(microscope, settings, stage_settings, demo_mode=demo_mode, )
    lower_milling(microscope, settings, stage_settings, demo_mode=demo_mode, )

def setup_milling(microscope, settings, stage_settings):
    """Setup the ion beam system ready for milling.
    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings :  Dictionary of user input argument settings.
    stage_settings : Dictionary of settings for a single protocol milling stage
    Returns
    -------
    Autoscript microscope object.
    """
    ccs_file = settings['system']["application_file_cleaning_cross_section"]
    microscope = reset_state(microscope, settings, application_file=ccs_file)
    microscope.beams.ion_beam.beam_current.value = stage_settings["milling_current"]
    return microscope

def reset_state(microscope, settings, application_file=None):
    """Reset the microscope state.
    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings :  Dictionary of user input argument settings.
    application_file : str, optional
        Name of the application file for milling, by default None
    """
    microscope.patterning.clear_patterns()
    if application_file:  # optionally specified
        microscope.patterning.set_default_application_file(application_file)
    resolution = settings["imaging"]["resolution"]
    dwell_time = settings["imaging"]["dwell_time"]
    hfw = settings["imaging"]["horizontal_field_width"]
    microscope.beams.ion_beam.scanning.resolution.value = resolution
    microscope.beams.ion_beam.scanning.dwell_time.value = dwell_time
    microscope.beams.ion_beam.horizontal_field_width.value = hfw
    microscope.imaging.set_active_view(2)  # the ion beam view
    return microscope

def upper_milling(microscope, settings, stage_settings, demo_mode=False,):
    from autoscript_core.common import ApplicationServerException
    # Setup and realign to fiducial marker
    setup_milling(microscope, settings, stage_settings)
    # Create and mill patterns
    _upper_milling_coords(microscope, stage_settings)
    if not demo_mode:
        print("Milling pattern...")
        microscope.imaging.set_active_view(2)  # the ion beam view
        try:
            microscope.patterning.run()
        except ApplicationServerException:
            logging.error("ApplicationServerException: could not mill!")
    microscope.patterning.clear_patterns()
    return microscope

def lower_milling(microscope, settings, stage_settings, demo_mode=False,):
    from autoscript_core.common import ApplicationServerException
    # Setup and realign to fiducial marker
    setup_milling(microscope, settings, stage_settings)
    # Create and mill patterns
    _lower_milling_coords(microscope, stage_settings)
    if not demo_mode:
        print("Milling pattern...")
        microscope.imaging.set_active_view(2)  # the ion beam view
        try:
            microscope.patterning.run()
        except ApplicationServerException:
            logging.error("ApplicationServerException: could not mill!")
    microscope.patterning.clear_patterns()
    return microscope

def _upper_milling_coords(microscope, stage_settings):
    """Create cleaning cross section milling pattern above lamella position."""
    microscope.imaging.set_active_view(2)  # the ion beam view
    lamella_center_x = 0
    lamella_center_y = 0
    milling_depth = stage_settings["milling_depth"]
    center_y = (
        lamella_center_y
        + (0.5 * stage_settings["lamella_height"])
        + (
            stage_settings["total_cut_height"]
            * stage_settings["percentage_from_lamella_surface"]
        )
        + (
            0.5
            * stage_settings["total_cut_height"]
            * stage_settings["percentage_roi_height"]
        )
    )
    height = float(
        stage_settings["total_cut_height"] *
        stage_settings["percentage_roi_height"]
    )
    milling_roi = microscope.patterning.create_cleaning_cross_section(
        lamella_center_x,
        center_y,
        stage_settings["lamella_width"],
        height,
        milling_depth,
    )
    milling_roi.scan_direction = "TopToBottom"
    return milling_roi


def _lower_milling_coords(microscope, stage_settings):
    """Create cleaning cross section milling pattern below lamella position."""
    microscope.imaging.set_active_view(2)  # the ion beam view
    lamella_center_x = 0
    lamella_center_y = 0
    milling_depth = stage_settings["milling_depth"]
    center_y = (
        lamella_center_y
        - (0.5 * stage_settings["lamella_height"])
        - (
            stage_settings["total_cut_height"]
            * stage_settings["percentage_from_lamella_surface"]
        )
        - (
            0.5
            * stage_settings["total_cut_height"]
            * stage_settings["percentage_roi_height"]
        )
    )
    height = float(
        stage_settings["total_cut_height"] *
        stage_settings["percentage_roi_height"]
    )
    milling_roi = microscope.patterning.create_cleaning_cross_section(
        lamella_center_x,
        center_y,
        stage_settings["lamella_width"],
        height,
        milling_depth,
    )
    milling_roi.scan_direction = "BottomToTop"
    return milling_roi


def setup_ion_milling(microscope, *,
                      application_file="Si_Alex",
                      patterning_mode="Parallel",
                      ion_beam_field_of_view=100e-6):
    """Setup for rectangle ion beam milling patterns.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    application_file : str, optional
        Application file for ion beam milling, by default "Si_Alex"
    patterning_mode : str, optional
        Ion beam milling pattern mode, by default "Parallel".
        The available options are "Parallel" or "Serial".
    ion_beam_field_of_view : float, optional
        Width of ion beam field of view in meters, by default 59.2e-6
    """
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_beam_type(2)  # ion beam default
    microscope.patterning.set_default_application_file(application_file)
    microscope.patterning.mode = patterning_mode
    microscope.patterning.clear_patterns()  # clear any existing patterns
    microscope.beams.ion_beam.horizontal_field_width.value = ion_beam_field_of_view


def _run_milling(microscope, milling_current, *, imaging_current=20e-12):
        print("Ok, running ion beam milling now...")
        microscope.imaging.set_active_view(2)  # the ion beam view
        microscope.beams.ion_beam.beam_current.value = milling_current
        microscope.patterning.run()
        print("Returning to the ion beam imaging current now.")
        microscope.patterning.clear_patterns()
        microscope.beams.ion_beam.beam_current.value = imaging_current
        print("Ion beam milling complete.")


def confirm_and_run_milling(microscope, milling_current, *,
                            imaging_current=20e-12, confirm=True):
    """Run all the ion beam milling pattterns, after user confirmation.
    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    milling_current : float
        The ion beam milling current to use, in Amps.
    imaging_current : float, optional
        The ion beam imaging current to return to, by default 20 pico-Amps.
    confirm : bool, optional
        Whether to wait for user confirmation before milling.
    """
    # TODO: maybe display to the user how long milling will take
    if confirm is True:
        if ask_user("Do you want to run the ion beam milling? yes/no: "):
            _run_milling(microscope, milling_current, imaging_current=imaging_current)
        else:
            microscope.patterning.clear_patterns()
    else:
        _run_milling(microscope, milling_current, imaging_current=imaging_current)



def jcut_milling_patterns(microscope, jcut_settings,  pretilt_degrees=PRETILT_DEGREES):
    """Create J-cut milling pattern in the center of the ion beam field of view.
    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    jcut_settings : dict
        Dictionary of J-cut parameter settings.
    pretilt_degrees : int, optional
        Pre-tilt of sample holder in degrees, by default 27
    Returns
    -------
    (autoscript_sdb_microscope_client.structures.RectanglePattern,
     autoscript_sdb_microscope_client.structures.RectanglePattern,
     autoscript_sdb_microscope_client.structures.RectanglePattern)
        Tuple containing the three milling patterns comprising the J-cut.
    """
    jcut_top = None
    jcut_lhs = None
    jcut_rhs = None

    # Unpack settings
    jcut_angle_degrees = jcut_settings['jcut_angle']
    jcut_lamella_depth = jcut_settings['jcut_lamella_depth']
    jcut_length = jcut_settings['jcut_length']
    jcut_trench_thickness = jcut_settings['jcut_trench_thickness']
    jcut_milling_depth = jcut_settings['jcut_milling_depth']
    extra_bit = jcut_settings['extra_bit']

    # Setup
    setup_ion_milling(microscope)
    # Create milling patterns
    angle_correction = np.sin(np.deg2rad(52 - jcut_angle_degrees))
    # Top bar of J-cut
    if bool(jcut_settings['mill_top_jcut_pattern']) is True:
        print('Creating top J-cut pattern')
        jcut_top = microscope.patterning.create_rectangle(
            0.0,                                    # center_x
            jcut_lamella_depth * angle_correction,  # center_y
            jcut_length,                            # width
            jcut_trench_thickness,                  # height
            jcut_milling_depth)                     # depth
    # Left hand side of J-cut (long side)
    if bool(jcut_settings['mill_lhs_jcut_pattern']) is True:
        print('Creating LHS J-cut pattern')
        jcut_lhs = microscope.patterning.create_rectangle(
            -((jcut_length - jcut_trench_thickness) / 2),           # center_x
            ((jcut_lamella_depth - (extra_bit / 2)) / 2) * angle_correction,  # center_y
            jcut_trench_thickness,                                  # width
            (jcut_lamella_depth + extra_bit) * angle_correction,    # height
            jcut_milling_depth)                                     # depth
    # Right hand side of J-cut (short side)
    if bool(jcut_settings['mill_rhs_jcut_pattern']) is True:
        print('Creating RHS J-cut pattern')
        jcut_rightside_remaining = 1.5e-6  # in microns, how much to leave attached
        height = (jcut_lamella_depth - jcut_rightside_remaining) * angle_correction
        center_y = jcut_rightside_remaining + (height / 2)
        jcut_rhs = microscope.patterning.create_rectangle(
            +((jcut_length - jcut_trench_thickness) / 2),  # center_x
            center_y,                                      # center_y
            jcut_trench_thickness,                         # width
            height,                                        # height
            jcut_milling_depth)                            # depth
    if jcut_top is None and jcut_lhs is None and jcut_rhs is None:
        raise RuntimeError('No J-cut patterns created, check your protocol file')
    return jcut_top, jcut_lhs, jcut_rhs


def mill_jcut(microscope, jcut_settings, confirm=True):
    """Create and mill the rectangle patter to sever the jcut completely.
    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    jcut_settings : dict
        Dictionary of J-cut parameter settings.
    confrim : bool, optional
        Whether to ask the user to confirm before milling.
    """
    jcut_milling_patterns(microscope, jcut_settings)
    microscope.patterning.mode = 'Serial'
    confirm_and_run_milling(microscope, jcut_settings['jcut_milling_current'], confirm=confirm)
    microscope.patterning.mode = 'Parallel'



def jcut_severing_pattern(microscope, jcut_settings, pretilt_degrees=PRETILT_DEGREES):
    """Create J-cut milling pattern in the center of the ion beam field of view.
    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    jcut_settings : dict
        Sample surface angle for J-cut in degrees, by default 6
    pretilt_degrees : int, optional
        Pre-tilt of sample holder in degrees, by default 27
    Returns
    -------
    autoscript_sdb_microscope_client.structures.RectanglePattern
        Rectangle milling pattern used to sever the remaining bit of the J-cut.
    """
    # Unpack settings
    jcut_angle_degrees = jcut_settings['jcut_angle']
    jcut_lamella_depth = jcut_settings['jcut_lamella_depth']
    jcut_length = jcut_settings['jcut_length']
    jcut_trench_thickness = jcut_settings['jcut_trench_thickness']
    jcut_milling_depth = jcut_settings['jcut_milling_depth']
    extra_bit = jcut_settings['extra_bit']
    # Setup
    setup_ion_milling(microscope)
    # Create milling pattern - right hand side of J-cut
    angle_correction_factor = np.sin(np.deg2rad(52 - jcut_angle_degrees))
    center_x = +((jcut_length - jcut_trench_thickness) / 2)
    center_y = ((jcut_lamella_depth - (extra_bit / 2)) / 2) * angle_correction_factor  # noqa: E501
    width = jcut_trench_thickness
    height =  (jcut_lamella_depth + extra_bit) * angle_correction_factor
    jcut_severing_pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, jcut_milling_depth)
    return jcut_severing_pattern


def mill_to_sever_jcut(microscope, jcut_settings, *, milling_current=0.74e-9,
                       confirm=True):
    """Create and mill the rectangle pattern to sever the jcut completely.
    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    confirm : bool, optional
        Whether to wait for user confirmation before milling.
    """
    jcut_severing_pattern(microscope, jcut_settings)
    confirm_and_run_milling(microscope, milling_current, confirm=confirm)



def weld_to_landing_post(microscope, *, milling_current=20e-12, confirm=True):
    """Create and mill the sample to the landing post.
    Stick the lamella to the landing post by melting the ice with ion milling.
    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    confirm : bool, optional
        Whether to wait for user confirmation before milling.
    """
    pattern = _create_welding_pattern(microscope)
    confirm_and_run_milling(microscope, milling_current, confirm=confirm)


def _create_welding_pattern(microscope, *,
                            center_x=0,
                            center_y=0,
                            width=3.5e-6,
                            height=10e-6,
                            depth=5e-9):
    """Create milling pattern for welding liftout sample to the landing post.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    center_x : float
        Center position of the milling pattern along x-axis, in meters.
        Zero coordinate is at the centerpoint of the image field of view.
    center_y : float
        Center position of the milling pattern along x-axis, in meters.
        Zero coordinate is at the centerpoint of the image field of view.
    width : float
        Width of the milling pattern, in meters.
    height: float
        Height of the milling pattern, in meters.
    depth : float
        Depth of the milling pattern, in meters.
    """
    # TODO: user input yaml for welding pattern parameters
    setup_ion_milling(microscope)
    pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, depth)
    return pattern


def cut_off_needle(microscope, cut_coord, milling_current=0.74e-9, confirm=True):
    pattern = _create_cutoff_pattern(microscope,
        center_x=cut_coord["center_x"], center_y=cut_coord["center_y"],
        width=cut_coord["width"], height=cut_coord["height"],
        depth=cut_coord["depth"], rotation_degrees=cut_coord["rotation"], ion_beam_field_of_view=cut_coord["hfw"] )
    confirm_and_run_milling(microscope, milling_current, confirm=confirm)


def _create_cutoff_pattern(microscope, *,
                           center_x=-10.5e-6,
                           center_y=-5e-6,
                           width=8e-6,
                           height=2e-6,
                           depth=1e-6,
                           rotation_degrees=40,
                           ion_beam_field_of_view=100e-6):
    setup_ion_milling(microscope, ion_beam_field_of_view=ion_beam_field_of_view)
    pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, depth)
    pattern.rotation = np.deg2rad(rotation_degrees)
    return pattern

def _create_sharpen_pattern(microscope, *,
                           center_x=-10.5e-6,
                           center_y=-5e-6,
                           width=8e-6,
                           height=2e-6,
                           depth=1e-6,
                           rotation_degrees=40,
                           ion_beam_field_of_view=100e-6):

    pattern = microscope.patterning.create_rectangle(
    center_x, center_y, width, height, depth)
    pattern.rotation = np.deg2rad(rotation_degrees)
    return pattern



import matplotlib.pyplot as plt
import numpy as np
from patrick.utils import (load_model, model_inference, detect_and_draw_lamella_and_needle,
                        scale_invariant_coordinates, show_overlay, detect_and_draw_lamella_right_edge,
                        detect_closest_landing_point, draw_landing_edges_and_point,
                        scale_invariant_coordinates_NEW)
from PIL import Image

# def renormalise_and_mask_image(image, plot=False):
#     from autoscript_sdb_microscope_client.structures import AdornedImage
#     data = np.copy(image.data)

#     data_norm = ( data - np.mean(data) ) / np.std(data)
#     data_norm =  data_norm - data_norm.min()
#     data_norm = data_norm / data_norm.max()  *  255
#     data_norm = data_norm.astype(np.uint8)

#     Nx, Ny = data.shape
#     #################### elliptical mask ####################
#     dx = int( Nx//2 * 0.1 )
#     dy = int( Ny//2 * 0.1 )
#     elps_mask = ellipse_mask(size=data.shape, radius1=Nx//2-dx, radius2=Ny//2-dy, sigma=min(dx//2,dy//2))
#     elps_mask = np.transpose(elps_mask)
#     #################### rectangular mask ####################
#     #start  = np.round(np.array( [ int(Nx*0.1) , int(Ny*0.1) ] ) )
#     #extent = np.round(np.array( [ int(Nx*0.8) , int(Ny*0.8),  ] ) )
#     #rr, cc = skimage.draw.rectangle(start, extent=extent, shape=data.shape)
#     #rect_mask = np.zeros(data.shape)
#     #rect_mask[rr.astype(int), cc.astype(int)] = 1.0
#     #rect_mask = ndi.gaussian_filter(rect_mask, sigma= min(dx//2,dy//2)   )

#     image_norm = AdornedImage(data=(data_norm * elps_mask).astype(uint8)  )
#     image_norm.metadata = image.metadata

#     return image_norm

def select_point_new(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    coords = []

    def on_click(event):
        print(event.xdata, event.ydata)
        coords.append(event.ydata)
        coords.append(event.xdata)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    return tuple(coords[-2:])

def validate_detection(img, img_base, detection_coord, det_type):
    correct = input("Is this correct (y/n)")

    if correct == "n":

        print(f"Please click the {det_type} position")
        detection_coord = select_point_new(img)

        # save image for training here
        print("Saving image for labelling")
        storage.step_counter +=1
        storage.SaveImage(img_base, id="label_")


    print(detection_coord)
    return detection_coord


def lamella_shift_from_img_centre(img, show=False):

    #weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\12_04_2021_10_32_23_model.pt"
    weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\fresh_full_n10.pt"
    model = load_model(weights_file=weights_file)

    # convert img to numpy array
    img_orig = np.asarray(img.data)

    # model inference + display
    img_np, rgb_mask = model_inference(model, img=img_orig)

    # detect and draw lamella centre, and needle tip
    (
        lamella_centre_px,
        rgb_mask_lamella,
        needle_tip_px,
        rgb_mask_needle,
        rgb_mask_combined,
    ) = detect_and_draw_lamella_and_needle(rgb_mask)

    # prediction overlay
    img_overlay = show_overlay(img_np, rgb_mask_lamella)

    img_overlay_resized = Image.fromarray(img_overlay).resize((img_np.shape[1], img_np.shape[0]))

    if show:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img_orig, cmap='Blues_r', alpha=1)
        ax.imshow(img_overlay_resized, cmap='Oranges_r', alpha=0.5)

        plt.show()


    # # need to use the same scale images for both detection selections
    img_downscale = Image.fromarray(img_orig).resize((rgb_mask_combined.size[0], rgb_mask_combined.size[1]))
    lamella_centre_px = validate_detection(img_downscale, img, lamella_centre_px, "lamella centre")

    # scale invariant coordinatess
    scaled_lamella_centre_px, scaled_needle_tip_px = scale_invariant_coordinates(
        needle_tip_px, lamella_centre_px, rgb_mask_combined
    )

    if scaled_lamella_centre_px is None:
        raise ValueError("No lamella centre detected")

    return 0.5 - scaled_lamella_centre_px[0], -(0.5 - scaled_lamella_centre_px[1])


def quick_eucentric_test():
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import StagePosition
    from autoscript_sdb_microscope_client.structures import MoveSettings
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage = microscope.specimen.stage


    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=0.5e-6)  # TODO: user input resolution

    realign_eucentric_with_machine_learning(microscope, image_settings = image_settings_ML, hor_field_width=80e-6)


def realign_eucentric_with_machine_learning(microscope, image_settings, hor_field_width=150e-6, _autocontrast=False):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    from autoscript_sdb_microscope_client.structures import StagePosition
    """ Realign image to lamella centre using ML detection"""
    stage  = microscope.specimen.stage

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]

    #eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=hor_field_width, image_settings=image_settings, __autocontrast=False)
    microscope.beams.ion_beam.horizontal_field_width.value      = hor_field_width
    microscope.beams.electron_beam.horizontal_field_width.value = hor_field_width

    if _autocontrast:
        eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=hor_field_width, image_settings=image_settings, __autocontrast=True)
    else:
        autocontrast(microscope, beam_type=BeamType.ELECTRON) # try to charge the area evenly
        eb_lowres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
        autocontrast(microscope, beam_type=BeamType.ION) # # try to charge the area evenly
        ib_lowres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)


    storage.step_counter +=1
    storage.SaveImage(eb_lowres, id='A_eb_lowres')
    storage.SaveImage(ib_lowres, id='A_ib_lowres')

    # in microscope coordinates (not img coordinate)
    # renormalise_and_mask_image (for elipse soft mask)
    # eb_distance_y, eb_distance_x = lamella_shift_from_img_centre( eb_lowres, show=True)
    ib_distance_y, ib_distance_x = lamella_shift_from_img_centre( ib_lowres, show=True)

    # tested, can align within 20um

    # z correction
    pixelsize_x   = ib_lowres.metadata.binary_result.pixel_size.x
    field_width   = pixelsize_x  * ib_lowres.width
    dy_meters_ion = ib_distance_y * field_width
    tilt_radians = stage.current_position.t

    if 0: # simple way, from eucentric correction for landing poles
        delta_z = -np.cos(tilt_radians) * dy_meters_ion
        # delta_z = -dy_meters_ion / np.cos(np.deg2rad(-38))
        stage.relative_move(StagePosition(z=delta_z))
    yz_move_ion = y_corrected_stage_movement(dy_meters_ion, stage_tilt=tilt_radians, beam_type=BeamType.ION)
    stage.relative_move(yz_move_ion)

    # x correction
    field_height   = pixelsize_x  * ib_lowres.height
    dx_meters_ion = ib_distance_x * field_height
    stage.relative_move(StagePosition(x=dx_meters_ion))

    print("Realigning x and y in ion beam")
    #print(f"Electron Beam: {eb_distance_y}, {eb_distance_x}")
    print(f"Ion Beam: {ib_distance_y}, {ib_distance_x}")
    print(f"Delta X movement: {dx_meters_ion}")
    print(f"Delta Y movement: {yz_move_ion.y}")
    print(f"Delta Z movement: {yz_move_ion.z}")


    # electron dy shift
    #eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=hor_field_width, image_settings=image_settings,  __autocontrast=False)
    microscope.beams.ion_beam.horizontal_field_width.value      = hor_field_width
    microscope.beams.electron_beam.horizontal_field_width.value = hor_field_width


    if _autocontrast:
        eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=hor_field_width, image_settings=image_settings, __autocontrast=True)
    else:
        autocontrast(microscope, beam_type=BeamType.ELECTRON) # try to charge the area evenly
        eb_lowres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
        autocontrast(microscope, beam_type=BeamType.ION) # # try to charge the area evenly
        ib_lowres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)


    storage.SaveImage(eb_lowres, id='A_eb_lowres_moved1')
    storage.SaveImage(ib_lowres, id='A_ib_lowres_moved1')

    # in microscope coordinates (not img coordinate)
    # renormalise_and_mask_image
    eb_distance_y, eb_distance_x = lamella_shift_from_img_centre( eb_lowres, show=True)
    # z correction
    pixelsize_x   = eb_lowres.metadata.binary_result.pixel_size.x
    field_width   = pixelsize_x  * eb_lowres.width
    dy_meters_elec = eb_distance_y * field_width
    tilt_radians = stage.current_position.t
    #delta_z = -np.cos(tilt_radians) * dy_meters_elec
    #stage.relative_move(StagePosition(z=delta_z))
    ##### stage.relative_move(StagePosition(y=dy_meters_elec))
    yz_move_elec = y_corrected_stage_movement(dy_meters_elec, stage_tilt=tilt_radians, beam_type=BeamType.ELECTRON)
    stage.relative_move(yz_move_elec)



    # x correction
    field_height   = pixelsize_x  * eb_lowres.height
    dx_meters_elec = eb_distance_x * field_height
    stage.relative_move(StagePosition(x=dx_meters_elec))

    print("Realigning x and y in electron beam")
    print(f"Electron Beam: {eb_distance_y}, {eb_distance_x}")
    print(f"Delta Y movement: {yz_move_elec.y}")
    print(f"Delta Z movement: {yz_move_elec.z}")
    print(f"Delta X movement: {dx_meters_elec}")


    # ion again
    #eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=hor_field_width, image_settings=image_settings)
    microscope.beams.ion_beam.horizontal_field_width.value      = hor_field_width
    microscope.beams.electron_beam.horizontal_field_width.value = hor_field_width

    if _autocontrast:
        eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=hor_field_width, image_settings=image_settings, __autocontrast=True)
    else:
        autocontrast(microscope, beam_type=BeamType.ELECTRON) # try to charge the area evenly
        eb_lowres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
        autocontrast(microscope, beam_type=BeamType.ION) # # try to charge the area evenly
        ib_lowres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)


    storage.SaveImage(eb_lowres, id='A_eb_lowres_moved2')
    storage.SaveImage(ib_lowres, id='A_ib_lowres_moved2')

    # in microscope coordinates (not img coordinate)
    # renormalise_and_mask_image
    ib_distance_y, ib_distance_x = lamella_shift_from_img_centre( ib_lowres, show=True)

    # z correction
    pixelsize_x   = ib_lowres.metadata.binary_result.pixel_size.x
    field_width   = pixelsize_x  * ib_lowres.width
    dy_meters_ion = ib_distance_y * field_width
    tilt_radians = stage.current_position.t
    delta_z = -np.cos(tilt_radians) * dy_meters_ion
    # delta_z = -dy_meters_ion / np.cos(np.deg2rad(-38))
    #stage.relative_move(StagePosition(z=delta_z))
    yz_move_ion = y_corrected_stage_movement(dy_meters_ion, stage_tilt=tilt_radians, beam_type=BeamType.ION)
    stage.relative_move(yz_move_ion)

    # x correction
    field_height   = pixelsize_x  * ib_lowres.height
    dx_meters_ion = ib_distance_x * field_height
    stage.relative_move(StagePosition(x=dx_meters_ion))

    print("Realigning x and y in ion beam")
    print(f"Ion Beam: {ib_distance_y}, {ib_distance_x}")
    print(f"Delta Y movement: {yz_move_ion.y}")
    print(f"Delta Z movement: {yz_move_ion.z}")
    print(f"Delta X movement: {dy_meters_ion}")


    # take final position images
    if _autocontrast:
        eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=hor_field_width, image_settings=image_settings, __autocontrast=True)
    else:
        autocontrast(microscope, beam_type=BeamType.ELECTRON) # try to charge the area evenly
        eb_lowres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
        autocontrast(microscope, beam_type=BeamType.ION) # # try to charge the area evenly
        ib_lowres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
    #microscope.beams.ion_beam.horizontal_field_width.value      = hor_field_width
    #microscope.beams.electron_beam.horizontal_field_width.value = hor_field_width
    #eb_lowres = new_electron_image(microscope, settings = image_settings, brightness = 0.41, contrast = 0.8)
    #ib_lowres = new_ion_image(microscope, settings = image_settings, brightness = 0.41, contrast = 0.8)
    storage.SaveImage(eb_lowres, id='A_eb_lowres_final')
    storage.SaveImage(ib_lowres, id='A_ib_lowres_final')

    storage.step_counter += 1




def mill_lamella(microscope, settings, confirm=True):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import StagePosition
    from autoscript_sdb_microscope_client.structures import MoveSettings
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage = microscope.specimen.stage
    ############# Set the correct magnification / field of view
    field_of_view = 100e-6  # in meters  TODO: user input from yaml settings
    microscope.beams.ion_beam.horizontal_field_width.value      = field_of_view
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
    ############# Move to trench position
    move_to_trenching_angle(microscope) ####<----flat to the ion, stage tilt 25 (total image tilt 52)
    ############# Take an ion beam image at the *milling current*
    ib = new_ion_image(microscope)
    mill_trenches(microscope, settings, confirm=confirm)
    #############
    # Take reference images after trech milling, use them to realign after stage rotation to flat_to_electron position
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution
    #refocus_and_relink(microscope)

    resolution = storage.settings["reference_images"]["trench_area_ref_img_resolution"]
    dwell_time = storage.settings["reference_images"]["trench_area_ref_img_dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
    hfw_lowres  = storage.settings["reference_images"]["trench_area_ref_img_hfw_lowres"]
    hfw_highres = storage.settings["reference_images"]["trench_area_ref_img_hfw_highres"]
    eb_lowres_reference,  ib_lowres_reference =  take_electron_and_ion_reference_images(microscope,  hor_field_width=hfw_lowres,  image_settings=image_settings)
    eb_highres_reference, ib_highres_reference = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres, image_settings=image_settings)
    reference_images_low_and_high_res = (eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference) #use these images for future alignment

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
    microscope.beams.ion_beam.horizontal_field_width.value      = hfw_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = hfw_lowres
    eb_lowres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_lowres = new_ion_image(microscope,      settings=image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.step_counter +=1
    storage.SaveImage(eb_lowres, id='ref_A_eb_lowres__brightnessContrast')
    storage.SaveImage(ib_lowres, id='ref_A_ib_lowres__brightnessContrast')
    microscope.beams.ion_beam.horizontal_field_width.value      = hfw_highres
    microscope.beams.electron_beam.horizontal_field_width.value = hfw_highres
    eb_highres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_highres = new_ion_image(microscope,      settings=image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.SaveImage(eb_highres, id='ref_A_eb_highres__brightnessContrast')
    storage.SaveImage(ib_highres, id='ref_A_ib_highres__brightnessContrast')
    reference_images_low_and_high_res_BC = (eb_lowres, eb_highres, ib_lowres, ib_highres) #use these images for future alignment

    ############# Move to flat_to_electron, take electron beam images, align using ion-beam image from tenching angle, Move to Jcut angle(+6 deg)
    flat_to_electron_beam(microscope, pretilt_angle=PRETILT_DEGREES) # rotate to flat_to_electron
    microscope.beams.ion_beam.horizontal_field_width.value      = 400e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 400e-6
    #refocus_and_relink(microscope)
    realign_using_reference_eb_and_ib_images(microscope, reference_images_low_and_high_res, plot=True) # correct the stage drift after 180 deg rotation using treched lamella images as reference
    #realign_using_reference_eb_and_ib_images(microscope, reference_images_low_and_high_res_BC, plot=False) # correct the stage drift after 180 deg rotation using treched lamella images as reference
    ##
    #MACHINE LEARNING eucentic height adjustment two steps
    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=0.5e-6)  # TODO: user input resolution
    realign_eucentric_with_machine_learning(microscope, image_settings = image_settings_ML, hor_field_width=300e-6)
    realign_eucentric_with_machine_learning(microscope, image_settings = image_settings_ML, hor_field_width=150e-6)
    realign_eucentric_with_machine_learning(microscope, image_settings = image_settings_ML, hor_field_width=80e-6, _autocontrast=True)

    ##
    #   the lamella is now aligned
    #############
    # Take new set of images after alignment
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution
    eb_lowres_reference,  ib_lowres_reference  = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings) # TODO: yaml use input
    eb_highres_reference, ib_highres_reference = take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6,  image_settings=image_settings) # TODO: yaml use input
    reference_images_low_and_high_res = (eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference)

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
    microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 150e-6
    eb_lowres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_lowres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.step_counter +=1
    storage.SaveImage(eb_lowres, id='aligned_A_eb_lowres__brightnessContrast')
    storage.SaveImage(ib_lowres, id='aligned_A_ib_lowres__brightnessContrast')
    microscope.beams.ion_beam.horizontal_field_width.value      = 50e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 50e-6
    eb_highres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_highres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.SaveImage(eb_highres, id='aligned_A_eb_highres__brightnessContrast')
    storage.SaveImage(ib_highres, id='aligned_A_ib_highres__brightnessContrast')
    storage.step_counter +=1

    #############
    ############# Mill J-cut
    #############
    ############################################################################################
    ###############################  tilted images alignment # 1 ###############################
    ############################################################################################
    ### Need to tilt +6 deg to j-cut, BUT tilt first +3 deg only:
    previous_stage_tilt = stage.current_position.t
    tilting = StagePosition(x=0, y=0, z=0, t=np.deg2rad(3)) #1/2 tilt: move_to_jcut_angle(microscope)<--flat to electron beam + jcut_angle=6, stage tilt total 33
    print(tilting)
    stage.relative_move(tilting, stage_settings)
    #realign_at_different_stage_tilts(microscope, reference_images_low_and_high_res, previous_stage_tilt=previous_stage_tilt, beam_type=BeamType.ION)
    #refocus_and_relink(microscope)
    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution
    realign_eucentric_with_machine_learning(microscope, image_settings_ML, hor_field_width=150e-6, _autocontrast=False) # TODO: investigate autocontrast problem (spiked pixel intensities)
    ############################################################################################
    ###############################  tilted images alignment # 2 ###############################
    ############################################################################################
    # Take new images, use them ans reference to align for the next 3 deg tilt
    eb_lowres_ref, ib_lowres_ref   = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
    eb_highres_ref, ib_highres_ref = take_electron_and_ion_reference_images(microscope, hor_field_width= 50e-6, image_settings=image_settings)
    reference_images_low_and_high_res = (eb_lowres_ref, eb_highres_ref, ib_lowres_ref, ib_highres_ref) #use these images for future alignment

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
    microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 150e-6
    eb_lowres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_lowres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.step_counter +=1
    storage.SaveImage(eb_lowres, id='tilt3deg_A_eb_lowres__brightnessContrast')
    storage.SaveImage(ib_lowres, id='tilt3deg_A_ib_lowres__brightnessContrast')
    microscope.beams.ion_beam.horizontal_field_width.value      = 50e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 50e-6
    eb_highres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_highres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.SaveImage(eb_highres, id='tilt3deg_A_eb_highres__brightnessContrast')
    storage.SaveImage(ib_highres, id='tilt3deg_A_ib_highres__brightnessContrast')
    reference_images_low_and_high_res_BC = (eb_lowres, eb_highres, eb_highres, ib_highres) #use these images for future alignment
    storage.step_counter +=1


    ### Need to tilt +6 deg, tilt first +3 deg only, again +3deg, Now +6 deg
    previous_stage_tilt = stage.current_position.t
    tilting = StagePosition(x=0, y=0, z=0, t=np.deg2rad(3))
    print(tilting)
    stage.relative_move(tilting, stage_settings)
    # realign_at_different_stage_tilts(microscope, reference_images_low_and_high_res, previous_stage_tilt=previous_stage_tilt, beam_type=BeamType.ION)
    #refocus_and_relink(microscope)y
    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=1.0e-6)  # TODO: user input resolution
    realign_eucentric_with_machine_learning(microscope, image_settings_ML, hor_field_width=150e-6)
    realign_eucentric_with_machine_learning(microscope, image_settings_ML, hor_field_width=80e-6, _autocontrast=False)

    ############################################################################################
    #
    mill_jcut(microscope, settings['jcut'], confirm=False)
    #
    # images after j-cut is done
    eb_lowres_ref_jcut,  ib_lowres_ref_jcut  = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings) # TODO: yaml use input
    eb_highres_ref_jcut, ib_highres_ref_jcut = take_electron_and_ion_reference_images(microscope, hor_field_width= 50e-6, image_settings=image_settings) # TODO: yaml use input
    storage.SaveImage(eb_lowres_ref_jcut,  id='eb_lowres_Jcut' )
    storage.SaveImage(ib_lowres_ref_jcut,  id='ib_lowres_Jcut' )
    storage.SaveImage(eb_highres_ref_jcut, id='eb_highres_Jcut')
    storage.SaveImage(ib_highres_ref_jcut, id='ib_highres_Jcut')

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
    microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 150e-6
    eb_lowres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_lowres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.step_counter +=1
    storage.SaveImage(eb_lowres, id='jcut_A_eb_lowres__brightnessContrast')
    storage.SaveImage(ib_lowres, id='jcut_A_ib_lowres__brightnessContrast')
    microscope.beams.ion_beam.horizontal_field_width.value      = 50e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 50e-6
    eb_highres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_highres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.SaveImage(eb_highres, id='jcut_A_eb_highres__brightnessContrast')
    storage.SaveImage(ib_highres, id='jcut_A_ib_highres__brightnessContrast')


    storage.step_counter += 1

    ########################## Ready for liftout ##########################
    # go from j-cut (33 deg) to liftout angle (37)
    previous_stage_tilt = stage.current_position.t
    move_to_liftout_angle(microscope) ####<----flat to electron beam + 10 deg - MOVE 33->37 AND CORRECT the drift in ibeam
    # realign j-cut images to lift-out images
    # reference_images_low_and_high_res = (eb_lowres_ref_jcut, eb_highres_ref_jcut, ib_lowres_ref_jcut, ib_highres_ref_jcut)
    # realign_at_different_stage_tilts(microscope, reference_images_low_and_high_res, previous_stage_tilt, beam_type=BeamType.ION) ### REALIGN to center in the ion beam after +4 deg tilt
    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=0.5e-6)  # TODO: user input resolution
    realign_eucentric_with_machine_learning(microscope, image_settings_ML, hor_field_width=150e-6)
    realign_eucentric_with_machine_learning(microscope, image_settings_ML, hor_field_width=80e-6)
    realign_eucentric_with_machine_learning(microscope, image_settings_ML, hor_field_width=50e-6, _autocontrast=False)

    print("Done, ready for liftout!")


# def OLD__mill_lamella(microscope, settings, confirm=True):
#     from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
#     stage = microscope.specimen.stage
#     # Set the correct magnification / field of view
#     field_of_view = 100e-6  # in meters  TODO: user input from yaml settings
#     microscope.beams.ion_beam.horizontal_field_width.value = field_of_view
#     microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
#     # Move to trench position
#     move_to_trenching_angle(microscope)  # flat to the ion
#     ib = new_ion_image(microscope)       # Take an ion beam image at the *milling current*
#     mill_trenches(microscope, settings, confirm=confirm)
#     autocontrast(microscope, beam_type=BeamType.ION)
#     image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution
#     ib_original = new_ion_image(microscope, settings=image_settings)
#     template = create_reference_image(ib_original)
#     # Low res template image
#     scaling_factor = 4
#     microscope.beams.ion_beam.horizontal_field_width.value = field_of_view * scaling_factor
#     microscope.beams.electron_beam.horizontal_field_width.value = field_of_view * scaling_factor
#     ib_lowres_original = new_ion_image(microscope, settings=image_settings)
#     lowres_template = AdornedImage(data=np.rot90(np.rot90(ib_lowres_original.data)))
#     lowres_template.metadata = ib_lowres_original.metadata
#     # Move to Jcut angle and take electron beam image
#     move_to_jcut_angle(microscope)  # flat to electron beam
#     # Low res resolution
#     autocontrast(microscope, beam_type=BeamType.ELECTRON)
#     microscope.beams.ion_beam.horizontal_field_width.value = field_of_view * scaling_factor
#     microscope.beams.electron_beam.horizontal_field_width.value = field_of_view * scaling_factor
#     image = new_electron_image(microscope, settings=image_settings)
#     location = match_locations(microscope, image, lowres_template)
#     realign_hog_matcher(microscope, location)
#     eb = new_electron_image(microscope, settings=image_settings)
#     # Realign first to the electron beam image
#     autocontrast(microscope, beam_type=BeamType.ELECTRON)
#     microscope.beams.ion_beam.horizontal_field_width.value = field_of_view
#     microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
#     image = new_electron_image(microscope, settings=image_settings)
#     location = match_locations(microscope, image, template)
#     realign_hog_matcher(microscope, location)
#     eb = new_electron_image(microscope, settings=image_settings)
#     # Fine tune alignment of ion beam image
#     autocontrast(microscope, beam_type=BeamType.ION)
#     image = new_ion_image(microscope, settings=image_settings)
#     location = match_locations(microscope, image, template)
#     realign_hog_matcher(microscope, location)
#     ib = new_ion_image(microscope, settings=image_settings)
#     eb = new_electron_image(microscope, settings=image_settings)
#     # Mill J-cut
#     mill_jcut(microscope, settings['jcut'], confirm=False)
#     final_ib = new_ion_image(microscope, settings=image_settings)
#     final_eb = new_electron_image(microscope, settings=image_settings)
#     # Ready for liftout
#     move_to_liftout_angle(microscope)
#     print("Done, ready for liftout!")


def take_and_save_electron_and_ion_reference_images(microscope, hor_field_width, image_settings, label):
    """ Helper function for taking and saving reference images for both beams"""
    
    # take images
    eb_image, ib_image = take_electron_and_ion_reference_images(microscope, hor_field_width=hor_field_width,
            image_settings=image_settings)

    # file labels
    eb_label = label + "_eb"
    ib_label = label + "_ib"

    # save images
    storage.SaveImage(eb_image,  id = eb_label)
    storage.SaveImage(ib_image,  id = ib_label)

    return eb_image, ib_image



def land_lamella(microscope, landing_coord, original_landing_images):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator
    microscope.specimen.stage.absolute_move(landing_coord)
    realign_landing_post(microscope, original_landing_images)
    park_position = move_needle_to_landing_position(microscope)

    # Lamella to Landing Post
    # TODO: check if the landing post is cslose enough to the centre

    # y-movement
    needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6,
            image_settings=image_settings)

    storage.SaveImage(needle_eb_lowres_with_lamella_shifted,  id='A_landing_needle_land_sample_eb_lowres' )
    storage.SaveImage(needle_ib_lowres_with_lamella_shifted,  id='A_landing_needle_land_sample_ib_lowres' )

    # NEW 
    # take_and_save_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings, label="A_landing_needle_land_sample")


    # TODO: make distance measurement calculation direction consistent
    landing_px = needle_eb_lowres_with_lamella_shifted.height // 2, needle_eb_lowres_with_lamella_shifted.width // 2
    x_shift, y_shift = lamella_edge_to_landing_post(needle_eb_lowres_with_lamella_shifted, landing_px)
    x_move = x_corrected_needle_movement(x_shift)
    y_move = y_corrected_needle_movement(y_shift, stage.current_position.t)
    print('x_move = ', x_move, ';\ny_move = ', y_move)
    needle.relative_move(y_move)

    # z-movement
    needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6,
                image_settings=image_settings)

    storage.SaveImage(needle_eb_lowres_with_lamella_shifted,  id='B_landing_needle_land_sample_eb_lowres_after_y_move' )
    storage.SaveImage(needle_ib_lowres_with_lamella_shifted,  id='B_landing_needle_land_sample_ib_lowres_after_y_move' )

    # NEW
    # take_and_save_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings, label="B_landing_needle_land_sample_after_y_move")

    x_shift, y_shift = lamella_edge_to_landing_post(needle_ib_lowres_with_lamella_shifted, landing_px)
    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    z_distance = y_shift / np.sin(np.deg2rad(52))
    z_move = z_corrected_needle_movement(-z_distance, stage_tilt)
    print('z_move = ', z_move)
    needle.relative_move(z_move)


    # x-movement
    needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6,
                    image_settings=image_settings)
    storage.SaveImage(needle_eb_lowres_with_lamella_shifted,  id='C_landing_needle_land_sample_eb_lowres_after_z_move' )
    storage.SaveImage(needle_ib_lowres_with_lamella_shifted,  id='C_landing_needle_land_sample_ib_lowres_after_z_move' )

    # NEW
    # take_and_save_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings, label="C_landing_needle_land_sample_after_z_move")


    x_shift, y_shift = lamella_edge_to_landing_post(needle_eb_lowres_with_lamella_shifted, landing_px)
    x_move = x_corrected_needle_movement(x_shift)
    print('x_move = ', x_move)


    # half move
    x_move = x_corrected_needle_movement(x_shift / 2)
    print('x_move = ', x_move)

    needle.relative_move(x_move)

    # x-movement
    needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6,
                    image_settings=image_settings)
    storage.SaveImage(needle_eb_lowres_with_lamella_shifted,  id='C_landing_needle_land_sample_eb_lowres_after_x_half_move' )
    storage.SaveImage(needle_ib_lowres_with_lamella_shifted,  id='C_landing_needle_land_sample_ib_lowres_after_x_half_move' )

    # NEW
    # take_and_save_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings, label="C_landing_needle_land_sample_after_x_half_move")

    print("ion:")
    x_shift, y_shift = lamella_edge_to_landing_post(needle_eb_lowres_with_lamella_shifted, landing_px)

    # x-move the rest of the way
    x_move = x_corrected_needle_movement(x_shift)
    print('x_move = ', x_move)

    needle.relative_move(x_move)



    # take final landing images
    landing_eb_highres, landing_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6, image_settings=image_settings)

    storage.SaveImage(landing_eb_highres,  id='D_landing_eb_lamella_final' )
    storage.SaveImage(landing_ib_highres,  id='D_landing_ib_lamella_final' )
    
    # NEW
    # take_and_save_electron_and_ion_reference_images(microscope, hor_field_width=50e-6, image_settings=image_settings, label="D_landing_lamella_final")

    storage.step_counter += 1

    # finish landing
    weld_to_landing_post(microscope, confirm=True)



    # calculate cut off position
    #landing_eb_highres, landing_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6, image_settings=image_settings)
    #landing_eb_highres2, landing_ib_highres2 = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6, image_settings=image_settings)
    landing_eb_highres3, landing_ib_highres3 = take_electron_and_ion_reference_images(microscope, hor_field_width=100e-6, image_settings=image_settings)


    # storage.SaveImage(landing_eb_highres,  id='E_landing_eb_lamella_final_1' )
    # storage.SaveImage(landing_ib_highres,  id='E_landing_ib_lamella_final_1' )

    # storage.SaveImage(landing_eb_highres2,  id='E_landing_eb_lamella_final_2' )
    # storage.SaveImage(landing_ib_highres2,  id='E_landing_ib_lamella_final_2' )

    storage.SaveImage(landing_eb_highres3,  id='E_landing_eb_lamella_final_3' )
    storage.SaveImage(landing_ib_highres3,  id='E_landing_ib_lamella_final_3' )


    # x_shift, y_shift = needletip_shift_from_lamella_centre(landing_ib_highres)
    # x_shift, y_shift = needletip_shift_from_lamella_centre(landing_ib_highres2)
    # x_shift, y_shift = needletip_shift_from_lamella_centre(landing_ib_highres3)


    # calculate shift from needle top to centre of image
    x_shift, y_shift = needletip_shift_from_img_centre(landing_ib_highres3)

    cut_coord = {"center_x": x_shift,
                "center_y": y_shift,
                "width":8e-6,
                "height": 0.5e-6,
                "depth": 4e-6,
                "rotation": 0, "hfw":100e-6}  # TODO: check rotation


    # cut off needle tip
    cut_off_needle(microscope, cut_coord=cut_coord, confirm=True)
    landing_eb_highres3, landing_ib_highres3 = take_electron_and_ion_reference_images(microscope, hor_field_width=100e-6, image_settings=image_settings)
    storage.SaveImage(landing_eb_highres3,  id='F_landing_eb_lamella_final_cut' )
    storage.SaveImage(landing_ib_highres3,  id='F_landing_ib_lamella_final_cut' )

    # retract needle from landing position
    retract_needle(microscope, park_position)

    landing_eb_highres2, landing_ib_highres2 = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6, image_settings=image_settings)
    landing_eb_highres3, landing_ib_highres3 = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)

    storage.SaveImage(landing_eb_highres2,  id='F_cut_needle_eb_lamella_1' )
    storage.SaveImage(landing_ib_highres2,  id='F_cut_needle_ib_lamella_1' )
    storage.SaveImage(landing_eb_highres3,  id='F_cut_needle_eb_lamella_2' )
    storage.SaveImage(landing_ib_highres3,  id='F_cut_needle_ib_lamella_2' )

    storage.step_counter += 1






    # eb_highres_reference, ib_highres_reference = needle_reference_images_highres_with_lamella
    # storage.SaveImage(eb_highres_reference, id='A_landing_ref_eb_highres')
    # storage.SaveImage(ib_highres_reference, id='A_landing_ref_ib_highres')
    # pixelsize_x = eb_highres_reference.metadata.binary_result.pixel_size.x
    # field_width = pixelsize_x * eb_highres_reference.width
    # microscope.beams.ion_beam.horizontal_field_width.value      = field_width
    # microscope.beams.electron_beam.horizontal_field_width.value = field_width
    # image_settings_electron = GrabFrameSettings(resolution="3072x2048", dwell_time=1.0e-6)
    # image_settings_ion      = GrabFrameSettings(resolution="3072x2048", dwell_time=0.5e-6)

    # microscope.imaging.set_active_view(1)
    # autocontrast(microscope, beam_type=BeamType.ELECTRON)
    # landing_eb_highres = microscope.imaging.grab_frame(image_settings_electron)
    # microscope.imaging.set_active_view(2)
    # autocontrast(microscope, beam_type=BeamType.ION)
    # landing_ib_highres = microscope.imaging.grab_frame(image_settings_ion)
    # storage.SaveImage(landing_eb_highres, id='B_landingLamella_eb_highres')
    # storage.SaveImage(landing_ib_highres, id='B_landingLamella_ib_highres')

    # ############ FIND dx, dy from HIGH_RES ELECTRON images ############
    # x_shift, y_shift = find_needletip_shift_in_image_ELECTRON(landing_eb_highres, eb_highres_reference, show=False, median_smoothing=2)
    # x_move = x_corrected_needle_movement(x_shift)
    # y_move = y_corrected_needle_movement(y_shift, stage.current_position.t)
    # print('x_move = ', x_move, ';\ny_move = ', y_move)
    # #yy = input('press ENTER to move the needle in Y only...')
    # needle.relative_move(y_move)

    # ############ FIND dz from HIGH_RES ION images ############
    # # landing_eb_lowres02, landing_ib_lowres02   = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
    # microscope.imaging.set_active_view(1)
    # autocontrast(microscope, beam_type=BeamType.ELECTRON)
    # landing_eb_highres02 = microscope.imaging.grab_frame(image_settings_electron)
    # microscope.imaging.set_active_view(2)
    # autocontrast(microscope, beam_type=BeamType.ION)
    # landing_ib_highres02 = microscope.imaging.grab_frame(image_settings_ion)
    # storage.SaveImage(landing_eb_highres02, id='C_landingLamella_eb_highres_yShifted')
    # storage.SaveImage(landing_ib_highres02, id='C_landingLamella_ib_highres_yShifted')
    # x_shift, y_shift = find_needletip_shift_in_image_ION(landing_ib_highres02, ib_highres_reference, show=False, median_smoothing=2)
    # stage_tilt = stage.current_position.t
    # print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    # z_distance = y_shift / np.sin(np.deg2rad(52))
    # z_move = z_corrected_needle_movement(z_distance, stage_tilt)
    # needle.relative_move(z_move)

    # ############ FIND dx from HIGH_RES ElEC images ############
    # # landing_eb_highres, landing_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6, image_settings=image_settings)
    # microscope.imaging.set_active_view(1)
    # autocontrast(microscope, beam_type=BeamType.ELECTRON)
    # landing_eb_highres03 = microscope.imaging.grab_frame(image_settings_electron)
    # microscope.imaging.set_active_view(2)
    # autocontrast(microscope, beam_type=BeamType.ION)
    # landing_ib_highres03 = microscope.imaging.grab_frame(image_settings_ion)
    # storage.SaveImage(landing_eb_highres02, id='D_landingLamella_eb_highres_yShifted')
    # storage.SaveImage(landing_ib_highres02, id='D_landingLamella_ib_highres_yShifted')
    # x_shift, y_shift = find_needletip_shift_in_image_ELECTRON(landing_eb_highres03, eb_highres_reference, show=False,  median_smoothing=2)
    # x_move = x_corrected_needle_movement(x_shift)
    # print('x_move = ', x_move)
    # needle.relative_move(x_move)

    # landed_eb_highres, landed_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6, image_settings=image_settings_ion)
    # storage.SaveImage(landed_eb_highres, id='E_landed_eb_highres')
    # storage.SaveImage(landed_ib_highres, id='E_landed_ib_highres')
    # storage.step_counter += 1



def sharpen_needle(microscope):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    def Rotate(x, y, angle):
        angle = np.deg2rad(angle)
        x_rot = x * math.cos(angle) - y * math.sin(angle)
        y_rot = x * math.sin(angle) + y * math.cos(angle)
        return x_rot, y_rot

    move_sample_stage_out(microscope)

    park_position = insert_needle(microscope)
    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    z_move_in = z_corrected_needle_movement(-180e-6, stage_tilt)
    needle.relative_move(z_move_in)

    if 0: # focus ion beam image : does not work
        microscope.imaging.set_active_view(2)  # the ion beam view
        original_hfw = microscope.beams.ion_beam.horizontal_field_width.value
        microscope.beams.ion_beam.horizontal_field_width.value = 0.000400
        print('Automatically refocusing ion  beam.')
        microscope.auto_functions.run_auto_focus()
        microscope.beams.ion_beam.horizontal_field_width.value = original_hfw

    # needle images
    resolution = storage.settings["imaging"]["resolution"]
    dwell_time = storage.settings["imaging"]["dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
    hfw_lowres  = storage.settings["imaging"]["horizontal_field_width"]
    needle_eb, needle_ib = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres, image_settings=image_settings)
    storage.SaveImage(needle_eb,  id='A_sharpen_needle_eb' )
    storage.SaveImage(needle_ib,  id='A_sharpen_needle_ib' )

    x_0, y_0 = needletip_shift_from_img_centre(needle_ib, show=True)

    # move needle to the centre
    x_move = x_corrected_needle_movement(-x_0)
    needle.relative_move(x_move)
    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    z_distance = y_0 / np.sin(np.deg2rad(52))
    z_move = z_corrected_needle_movement(z_distance, stage_tilt)
    print('z_move = ', z_move)
    needle.relative_move(z_move)



    # needle images after centering
    resolution = storage.settings["imaging"]["resolution"]
    dwell_time = storage.settings["imaging"]["dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
    hfw_lowres  = storage.settings["imaging"]["horizontal_field_width"]
    needle_eb, needle_ib = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres, image_settings=image_settings)
    storage.SaveImage(needle_eb,  id='A_sharpen_needle_eb_centre' )
    storage.SaveImage(needle_ib,  id='A_sharpen_needle_ib_centre' )

    x_0, y_0 = needletip_shift_from_img_centre(needle_ib, show=True)

    height = storage.settings["sharpen"]["height"]
    width  = storage.settings["sharpen"]["width"]
    depth  = storage.settings["sharpen"]["depth"]
    bias   = storage.settings["sharpen"]["bias"]
    hfw    = storage.settings["sharpen"]["hfw"]
    tip_angle    = storage.settings["sharpen"]["tip_angle"] # 2NA of the needle   2*alpha
    needle_angle = storage.settings["sharpen"]["needle_angle"] # needle tilt on the screen 45 deg +/-
    milling_current = storage.settings["sharpen"]["sharpen_milling_current"]

    alpha = tip_angle/2 # half of NA of the needletip
    beta =  np.rad2deg( np.arctan( width / height ) ) # box's width and length, beta is the diagonal angle
    D = np.sqrt( width**2 + height**2 )  / 2 # half of box diagonal
    rotation_1 = -(needle_angle + alpha)
    rotation_2 = -(needle_angle - alpha) - 180

    ############################################################################
    # dx_1 = D * math.cos( np.deg2rad(needle_angle + alpha) )
    # dy_1 = D * math.sin( np.deg2rad(needle_angle + alpha) )
    # x_1 = x_0 - dx_1 # centre of the bottom box
    # y_1 = y_0 - dy_1 # centre of the bottom box

    # dx_2 = D * math.cos( np.deg2rad(needle_angle - alpha - beta) )
    # dy_2 = D * math.sin( np.deg2rad(needle_angle - alpha - beta) )
    # x_2 = x_0 - dx_2 # centre of the top box
    # y_2 = y_0 - dy_2 # centre of the top box

    # x_1_origin = x_1 - x_0
    # y_1_origin = y_1 - y_0 # shift the x1,y1 to the origin
    # x_2_origin_rot, y_2_origin_rot = Rotate( x_1_origin, y_1_origin, 360-(2*alpha+2*beta) ) # rotate to get the x2,y2 point
    # x_2_rot = x_2_origin_rot + x_0 # shift to the old centre at x0,y0
    # y_2_rot = y_2_origin_rot + y_0

    ############################################################################
    dx_1 = (width/2) * math.cos( np.deg2rad(needle_angle + alpha) )
    dy_1 = (width/2) * math.sin( np.deg2rad(needle_angle + alpha) )
    ddx_1 = (height/2) * math.sin( np.deg2rad(needle_angle + alpha) )
    ddy_1 = (height/2) * math.cos( np.deg2rad(needle_angle + alpha) )
    x_1 = x_0 - dx_1 + ddx_1 # centre of the bottom box
    y_1 = y_0 - dy_1 - ddy_1 # centre of the bottom box

    dx_2 = D * math.cos( np.deg2rad(needle_angle - alpha) )
    dy_2 = D * math.sin( np.deg2rad(needle_angle - alpha ) )
    ddx_2 = (height/2) * math.sin( np.deg2rad(needle_angle - alpha) )
    ddy_2 = (height/2) * math.cos( np.deg2rad(needle_angle - alpha) )
    x_2 = x_0 - dx_2 - ddx_2  # centre of the top box
    y_2 = y_0 - dy_2 + ddy_2 # centre of the top box


    print("needletip xshift offcentre: ", x_0, "; needletip yshift offcentre: ", y_0)
    print("width: ", width)
    print("height: ", height)
    print("depth: ", depth)
    print("needle_angle: ", needle_angle)
    print("tip_angle: ", tip_angle)
    print("rotation1 :",  rotation_1)
    print("rotation2 :",  rotation_2)
    print("=================================================")
    print("centre of bottom box: x1 = ",  x_1, '; y1 = ', y_1)
    print("centre of top box:    x2 = ",  x_2, '; y2 = ', y_2)
    print("=================================================")


    #pattern = microscope.patterning.create_rectangle(x_3, y_3, width+2*bias, height+2*bias, depth)
    #pattern.rotation = np.deg2rad(rotation_1)
    #pattern = microscope.patterning.create_rectangle(x_4, y_4, width+2*bias, height+2*bias, depth)
    #pattern.rotation = np.deg2rad(rotation_2)

    # bottom cut pattern
    cut_coord_bottom = {"center_x": x_1,
                    "center_y": y_1,
                    "width": width,
                    "height": height-bias,
                    "depth": depth,
                    "rotation": rotation_1,
                    "hfw": hfw}

    # top cut pattern
    cut_coord_top = {"center_x": x_2 ,
                    "center_y": y_2,
                    "width": width,
                    "height": height-bias,
                    "depth": depth,
                    "rotation": rotation_2,
                    "hfw": hfw}


    # create sharpening patterns
    setup_ion_milling(microscope, ion_beam_field_of_view=hfw)

    sharpen_patterns = []
    for cut_coord in [cut_coord_bottom, cut_coord_top]:
        pattern = _create_sharpen_pattern(microscope,
            center_x=cut_coord["center_x"], center_y=cut_coord["center_y"],
            width=cut_coord["width"], height=cut_coord["height"],
            depth=cut_coord["depth"], rotation_degrees=cut_coord["rotation"], ion_beam_field_of_view=cut_coord["hfw"] )
        sharpen_patterns.append(pattern)


    confirm_and_run_milling(microscope, milling_current, confirm=True)

    needle_eb, needle_ib = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres, image_settings=image_settings)
    storage.step_counter +=1
    storage.SaveImage(needle_eb,  id='A_sharpen_needle_eb_sharp' )
    storage.SaveImage(needle_ib,  id='A_sharpen_needle_ib_sharp' )
    storage.step_counter +=1

    retract_needle(microscope, park_position)


def x_corrected_stage_movement(expected_x, stage_tilt=None, beam_type=None):
    """Stage movement in X.
    ----------
    expected_x : in meters
    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    return StagePosition(x=expected_x, y=0, z=0)


def y_corrected_stage_movement(expected_y, stage_tilt, beam_type=BeamType.ELECTRON):
    """Stage movement in Y, corrected for tilt of sample surface plane.
    ----------
    expected_y : in meters
    stage_tilt : in radians        Can pass this directly microscope.specimen.stage.current_position.t
    beam_type : BeamType, optional
        BeamType.ELECTRON or BeamType.ION
    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    if beam_type == BeamType.ELECTRON:
        tilt_adjustment =  np.deg2rad(-PRETILT_DEGREES)
    elif beam_type == BeamType.ION:
        tilt_adjustment =  np.deg2rad(52 - PRETILT_DEGREES)
    tilt_radians = stage_tilt + tilt_adjustment
    y_move = +np.cos(tilt_radians) * expected_y
    z_move = -np.sin(tilt_radians) * expected_y
    print(' ------------  drift correction ---------------  '  )
    print('the corrected Y shift is ', y_move, 'meters')
    print('the corrected Z shift is ', z_move, 'meters')
    return StagePosition(x=0, y=y_move, z=z_move)


def z_corrected_stage_movement(expected_z, stage_tilt):
    """Stage movement in Z, corrected for tilt of sample surface plane.
    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in radians
    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    y_move = -np.sin(stage_tilt) * expected_z
    z_move = +np.cos(stage_tilt) * expected_z
    return StagePosition(x=0, y=y_move, z=z_move)




#############################   cross-correlation, alignment   ##################################################################################################################
def circ_mask(size=(128, 128), radius=32, sigma=3):
	x=size[0]
	y=size[1]
	img = Image.new('I', size)
	draw = ImageDraw.Draw(img)
	draw.ellipse((x / 2 - radius, y / 2 - radius, x / 2 + radius, y / 2 + radius), fill='white', outline='white')
	tmp = np.array(img, float) / 255
	if sigma > 0:
		mask = ndi.filters.gaussian_filter(tmp, sigma=sigma)
	else:
		mask = tmp
	return mask

def ellipse_mask(size=(128, 128), radius1=32,radius2=32, sigma=3):
    x=size[0]
    y=size[1]
    img = Image.new('I', size)
    draw = ImageDraw.Draw(img)
    draw.ellipse((x / 2 - radius1, y / 2 - radius2, x / 2 + radius1, y / 2 + radius2), fill='white', outline='white')
    tmp = np.array(img, float) / 255
    if sigma > 0:
        mask = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        mask = tmp
    return mask

def bandpass_mask(size=(128, 128), lp=32, hp=2, sigma=3):
	x = size[0]
	y = size[1]
	lowpass = circ_mask(size=(x, y), radius=lp, sigma=0)
	hpass_tmp = circ_mask(size=(x, y), radius=hp, sigma=0)
	highpass = -1*(hpass_tmp - 1)
	tmp = lowpass * highpass
	if sigma > 0:
		bandpass = ndi.filters.gaussian_filter(tmp, sigma=sigma)
	else:
		bandpass = tmp
	return bandpass

def crosscorrelation(img1, img2, bp='no', *args, **kwargs):
	if img1.shape != img2.shape:
		print('### ERROR in xcorr2: img1 and img2 do not have the same size ###')
		return -1
	if img1.dtype != 'float64':
		img1 = np.array(img1, float)
	if img2.dtype != 'float64':
		img2 = np.array(img2, float)

	if bp == 'yes':
		lpv = kwargs.get('lp', None)
		hpv = kwargs.get('hp', None)
		sigmav = kwargs.get('sigma', None)
		if lpv == 'None' or hpv == 'None' or sigmav == 'None':
			print('ERROR in xcorr2: check bandpass parameters')
			return -1
		bandpass = bandpass_mask(size=(img1.shape[1], img1.shape[0]), lp=lpv, hp=hpv, sigma=sigmav)
		img1ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img1)))
		s = img1.shape[0] * img1.shape[1]
		tmp = img1ft * np.conj(img1ft)
		img1ft = s * img1ft / np.sqrt(tmp.sum())
		img2ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img2)))
		img2ft[0, 0] = 0
		tmp = img2ft * np.conj(img2ft)
		img2ft = s * img2ft / np.sqrt(tmp.sum())
		xcorr = np.real(fftpack.fftshift(fftpack.ifft2(img1ft * np.conj(img2ft))))
	elif bp == 'no':
		img1ft = fftpack.fft2(img1)
		img2ft = np.conj(fftpack.fft2(img2))
		img1ft[0, 0] = 0
		xcorr = np.abs(fftpack.fftshift(fftpack.ifft2(img1ft * img2ft)))
	else:
		print('ERROR in xcorr2: bandpass value ( bp= ' + str(bp) + ' ) not recognized')
		return -1
	return xcorr


def shift_from_crosscorrelation_simple_images(img1, img2, lowpass=256, highpass=22, sigma=2  ):
    img1_norm = (img1 - np.mean(img1)) / np.std(img1)
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)
    xcorr = crosscorrelation(img1_norm, img2_norm, bp='yes', lp=lowpass, hp=highpass, sigma=sigma)
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    print('\n', maxX, maxY)
    cen = np.asarray(xcorr.shape) / 2
    print('centre = ', cen)
    err = np.array(cen - [maxX, maxY], int)
    print("Shift between 1 and 2 is = " + str(err))
    print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
    return err[1], err[0]

def shift_from_crosscorrelation_AdornedImages(img1, img2, lowpass=128, highpass=6, sigma=6):
    pixelsize_x_1 = img1.metadata.binary_result.pixel_size.x
    pixelsize_y_1 = img1.metadata.binary_result.pixel_size.y
    pixelsize_x_2 = img2.metadata.binary_result.pixel_size.x
    pixelsize_y_2 = img2.metadata.binary_result.pixel_size.y
    # normalise both images
    img1_data_norm = (img1.data - np.mean(img1.data)) / np.std(img1.data)
    img2_data_norm = (img2.data - np.mean(img2.data)) / np.std(img2.data)
    # cross-correlate normalised images
    xcorr = crosscorrelation(img1_data_norm, img2_data_norm, bp='yes', lp=lowpass, hp=highpass, sigma=sigma)
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    print('\n', maxX, maxY)
    cen = np.asarray(xcorr.shape) / 2
    print('centre = ', cen)
    err = np.array(cen - [maxX, maxY], int)
    print("Shift between 1 and 2 is = " + str(err))
    print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
    x_shift = err[1] * pixelsize_x_2
    y_shift = err[0] * pixelsize_y_2
    print("X-shift =  {} meters".format(x_shift))
    print("Y-shift =  {} meters".format(y_shift))
    return x_shift, y_shift


def rotate_AdornedImage(image):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    data = np.rot90(np.rot90(np.copy(image.data)))
    #data = ndi.rotate(image.data, 180, reshape=False)
    reference = AdornedImage(data=data)
    reference.metadata = image.metadata
    return reference

def shift_from_correlation_electronBeam_and_ionBeam(eb_image, ib_image, lowpass=128, highpass=6, sigma=2):
    ib_image_rotated = rotate_AdornedImage(ib_image)
    x_shift, y_shift = shift_from_crosscorrelation_AdornedImages(eb_image, ib_image_rotated, lowpass=lowpass, highpass=highpass, sigma=sigma)
    return x_shift, y_shift



def correct_stage_drift_using_reference_eb_images(microscope, reference_images_low_and_high_res, plot=False):
    print('stage shift correction by image cross-correlation')
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    ### Unpack reference images
    eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference = reference_images_low_and_high_res
    storage.SaveImage(eb_lowres_reference,  id='A_ref_eb_lowres')
    storage.SaveImage(eb_highres_reference, id='A_ref_eb_highres')
    storage.SaveImage(ib_lowres_reference,  id='A_ref_ib_lowres')
    storage.SaveImage(ib_highres_reference, id='A_ref_ib_highres')
    ##############
    stage = microscope.specimen.stage
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
    ####
    pixelsize_x_lowres  = ib_lowres_reference.metadata.binary_result.pixel_size.x
    field_width_lowres  = pixelsize_x_lowres  * ib_lowres_reference.width
    pixelsize_x_highres = ib_highres_reference.metadata.binary_result.pixel_size.x
    field_width_highres = pixelsize_x_highres * ib_highres_reference.width
    ########################################  LOW resolution alignment  #############################################
    print('- - - - - - - - - - - - - - Coarse alignment- - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    new_eb_lowres = microscope.imaging.grab_frame(image_settings)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    new_ib_lowres = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(new_eb_lowres,  id='B_sample_eb_lowres')
    storage.SaveImage(new_ib_lowres,  id='B_sample_ib_lowres')
    #
    lowpass_pixels  = int( max(new_eb_lowres.data.shape) / 12 ) # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int( max(new_eb_lowres.data.shape)/ 256 ) # =6 @ 1536x1024, good for e-beam images
    sigma = int( 2 * max(new_eb_lowres.data.shape)/1536)       # =2 @ 1536x1024, good for e-beam images
    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(new_eb_lowres, eb_lowres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    x_move  = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ELECTRON) ##check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    #yy = input('press Enter to move...')
    stage.relative_move(x_move)
    stage.relative_move(yz_move)
    ###
    ########################################  HIGH resolution alignment  #############################################
    print('- - - - - - - - - - - - - - Finer alignment- - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value      = field_width_highres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    new_eb_highres = microscope.imaging.grab_frame(image_settings)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    new_ib_highres = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(new_eb_highres,  id='C_sample_eb_highres_shifted')
    storage.SaveImage(new_ib_highres,  id='C_sample_ib_highres_shifted')
    ########   ------------- correlate------------
    lowpass_pixels  = int( max(new_eb_highres.data.shape) / 12 ) # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int( max(new_eb_highres.data.shape)/ 256 ) # =6 @ 1536x1024, good for e-beam images
    sigma = int( 2 * max(new_eb_highres.data.shape)/1536)        # =2 @ 1536x1024, good for e-beam images
    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(new_eb_highres, eb_highres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    x_move  = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ELECTRON) ##check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    #yy = input('press Enter to move...')
    stage.relative_move(x_move)
    stage.relative_move(yz_move)
    storage.step_counter += 1


def realign_using_reference_eb_and_ib_images(microscope, reference_images_low_and_high_res, plot=False):
    print('stage shift correction by image cross-correlation : using only eBeam images')
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    ### Unpack reference images
    eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference = reference_images_low_and_high_res
    storage.SaveImage(eb_lowres_reference,  id='A_ebTOib_ref_eb_lowres')
    storage.SaveImage(eb_highres_reference, id='A_ebTOib_ref_eb_highres')
    storage.SaveImage(ib_lowres_reference,  id='A_ebTOib_ref_ib_lowres')
    storage.SaveImage(ib_highres_reference, id='A_ebTOib_ref_ib_highres')
    #
    stage = microscope.specimen.stage
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
    ####
    pixelsize_x_lowres  = eb_lowres_reference.metadata.binary_result.pixel_size.x
    field_width_lowres  = pixelsize_x_lowres  * eb_lowres_reference.width
    pixelsize_x_highres = eb_highres_reference.metadata.binary_result.pixel_size.x
    field_width_highres = pixelsize_x_highres * eb_highres_reference.width
    ########################################  LOW resolution alignment #1  #############################################
    print(' - - - - - - - - - - - - - - Coarse alignment #1 - - - - - - - - - - - - - - ...')
    #refocus_and_relink(microscope)
    microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    new_eb_lowres = microscope.imaging.grab_frame(image_settings)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    new_ib_lowres = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(new_eb_lowres,  id='B_01_ebTOib_sample_eb_lowres')
    storage.SaveImage(new_ib_lowres,  id='B_01_ebTOib_sample_ib_lowres')

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
    microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
    eb_lowres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_lowres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.SaveImage(eb_lowres, id='B_01_ebTOib_sample_eb_lowres_BC')
    storage.SaveImage(ib_lowres, id='B_01_ebTOib_sample_ib_lowres_BC')

    #
    lowpass_pixels  = int( max(new_eb_lowres.data.shape) / 12 ) # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int( max(new_eb_lowres.data.shape)/ 256 ) # =6   @ 1536x1024, good for e-beam images
    sigma = int( 2 * max(new_eb_lowres.data.shape)/1536)        # =2   @ 1536x1024, good for e-beam images
    dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(new_eb_lowres, ib_lowres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    #
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ELECTRON) ##check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    #yy = input('press Enter to move...')
    stage.relative_move(x_move)
    stage.relative_move(yz_move)

    ########################################  LOW resolution alignment #2  #############################################
    print(' - - - - - - - - - - - - - - Coarse alignment # 2 - - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    new_eb_lowres = microscope.imaging.grab_frame(image_settings)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    new_ib_lowres = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(new_eb_lowres,  id='B_02_ebTOib_sample_eb_lowres')
    storage.SaveImage(new_ib_lowres,  id='B_02_ebTOib_sample_ib_lowres')

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
    microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
    eb_lowres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_lowres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.SaveImage(eb_lowres, id='B_02_ebTOib_sample_eb_lowres_BC')
    storage.SaveImage(ib_lowres, id='B_02_ebTOib_sample_ib_lowres_BC')


    #
    lowpass_pixels  = int( max(new_eb_lowres.data.shape) / 12 ) # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int( max(new_eb_lowres.data.shape)/ 256 ) # =6   @ 1536x1024, good for e-beam images
    sigma = int( 2 * max(new_eb_lowres.data.shape)/1536)        # =2   @ 1536x1024, good for e-beam images
    dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(new_eb_lowres, ib_lowres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    #
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ELECTRON) ##check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    #yy = input('Press Enter to move...')
    stage.relative_move(x_move)
    stage.relative_move(yz_move)

   ########################################  HIGH resolution alignment  #############################################
    print(' - - - - - - - - - - - - - - Finer alignment - - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value      = field_width_highres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    new_eb_highres = microscope.imaging.grab_frame(image_settings)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    new_ib_highres = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(new_eb_lowres,  id='C_ebTOib_sample_eb_highres_shifted')
    storage.SaveImage(new_ib_lowres,  id='C_ebTOib_sample_ib_highres_shifted')

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
    microscope.beams.ion_beam.horizontal_field_width.value      = field_width_highres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
    eb_highres = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
    ib_highres = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.SaveImage(eb_highres, id='C_ebTOib_sample_eb_highres_shifted_BC')
    storage.SaveImage(ib_highres, id='C_ebTOib_sample_ib_highres_shifted_BC')

    lowpass_pixels  = int( max(new_eb_highres.data.shape) / 12 ) # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int( max(new_eb_highres.data.shape)/ 256 ) # =6   @ 1536x1024, good for e-beam images
    sigma = int(2 * max(new_eb_highres.data.shape) / 1536)       # =2   @ 1536x1024, good for e-beam images
    dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(new_eb_highres, ib_highres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ELECTRON) ##check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    #yy = input('Press Enter to move...')
    stage.relative_move(x_move)
    stage.relative_move(yz_move)
    storage.step_counter += 1



def realign_at_different_stage_tilts(microscope, reference_images_low_and_high_res, previous_stage_tilt, beam_type=BeamType.ION):
    print('stage shift correction by image cross-correlation : different stage/image tilts')
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import StagePosition
    stage = microscope.specimen.stage
    ### Unpack reference images
    eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference = reference_images_low_and_high_res
    storage.SaveImage(eb_lowres_reference,  id='A_tiltAlign_ref_eb_lowres')
    storage.SaveImage(eb_highres_reference, id='A_tiltAlign_ref_eb_highres')
    storage.SaveImage(ib_lowres_reference,  id='A_tiltAlign_ref_ib_lowres')
    storage.SaveImage(ib_highres_reference, id='A_tiltAlign_ref_ib_highres')
    pixelsize_x_lowres  = eb_lowres_reference.metadata.binary_result.pixel_size.x
    pixelsize_y_lowres  = eb_lowres_reference.metadata.binary_result.pixel_size.y
    field_width_lowres  = pixelsize_x_lowres  * eb_lowres_reference.width
    pixelsize_x_highres = eb_highres_reference.metadata.binary_result.pixel_size.x
    pixelsize_y_highres = eb_highres_reference.metadata.binary_result.pixel_size.y
    field_width_highres = pixelsize_x_highres * eb_highres_reference.width
    height, width       = eb_lowres_reference.data.shape
    eb_lowres_reference_norm  = (eb_lowres_reference.data - np.mean(eb_lowres_reference.data))   / np.std(eb_lowres_reference.data)
    eb_highres_reference_norm = (eb_highres_reference.data - np.mean(eb_highres_reference.data)) / np.std(eb_highres_reference.data)
    ib_lowres_reference_norm  = (ib_lowres_reference.data - np.mean(ib_lowres_reference.data))   / np.std(ib_lowres_reference.data)
    ib_highres_reference_norm = (ib_highres_reference.data - np.mean(ib_highres_reference.data)) / np.std(ib_highres_reference.data)
    # current_stage_tilt  = stage.current_position.t
    # current_image_tilt  = PRETILT_DEGREES + current_stage_tilt
    # previous_image_tilt = PRETILT_DEGREES + previous_stage_tilt
    # if beam_type==BeamType.ION:
    #     previous_image_tilt_from_ion_flat = 52 - np.red2deg(previous_image_tilt)
    #     current_image_tilt_from_ion_flat  = 52 - np.red2deg(current_image_tilt)
    #     if abs(previous_image_tilt_from_ion_flat) > abs(current_image_tilt_from_ion_flat):
    #         print('Previous image was tilted more, stretching it for alignment..')
    #         delta_angle = abs(previous_image_tilt_from_ion_flat) - abs(current_image_tilt_from_ion_flat)
    #         stretch_image= 1
    #         stretch =  1. / math.cos( np.deg2rad(delta_angle) )
    #     else:
    #         print('Current image is tilted more, stretching it for aligment..')
    #         delta_angle = abs(current_image_tilt_from_ion_flat) - abs(previous_image_tilt_from_ion_flat)
    #         stretch_image = 2
    #         stretch = 1. / math.cos(np.deg2rad(delta_angle))
    ####
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
    ####
    ########################################  LOW resolution alignment  #############################################
    print(' - - - - - - - - - - - - - - Coarse alignment - - - - - - - - - - - - - - ...')
    new_eb_lowres,  new_ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=field_width_lowres, image_settings=image_settings)
    storage.SaveImage(new_eb_lowres,  id='B_tiltAlign_sample_eb_lowres')
    storage.SaveImage(new_ib_lowres,  id='B_tiltAlign_sample_ib_lowres')
    new_eb_lowres_norm  = (new_eb_lowres.data - np.mean(new_eb_lowres.data)) / np.std(new_eb_lowres.data)
    new_ib_lowres_norm  = (new_ib_lowres.data - np.mean(new_ib_lowres.data)) / np.std(new_ib_lowres.data)
    ###
    cmask = circ_mask(size=(width,height), radius=height // 3 - 15, sigma=10)  # circular mask, align only the central areas
    ###
    if beam_type==BeamType.ELECTRON:
        lowpass_pixels  = int( max(new_eb_lowres.data.shape) / 12 ) # =128 @ 1536x1024, good for e-beam images
        highpass_pixels = int( max(new_eb_lowres.data.shape)/ 256 ) # =6   @ 1536x1024, good for e-beam images
        sigma = int( 2 * max(new_eb_lowres.data.shape)/1536)        # =2   @ 1536x1024, good for e-beam images
        dx_pixels, dy_pixels = shift_from_crosscorrelation_simple_images(new_eb_lowres_norm * cmask, eb_lowres_reference_norm * cmask, lowpass=lowpass_pixels,
                                                                 highpass=highpass_pixels, sigma=sigma)
        dx_meters = dx_pixels * pixelsize_x_lowres
        dy_meters = dy_pixels * pixelsize_y_lowres
        x_move = x_corrected_stage_movement(-dx_meters)
        yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t, beam_type=BeamType.ELECTRON) ##check electron/ion movement
    if beam_type==BeamType.ION:
        lowpass_pixels  = int(max(new_ib_lowres.data.shape) / 6)  # =256 @ 1536x1024,  good for i-beam images
        highpass_pixels = int(max(new_ib_lowres.data.shape) / 64) # =24  @ 1536x1024, good for i-beam images => need a large highpass to remove noise and ringing
        sigma = int(10 * max(new_ib_lowres.data.shape)    / 1536) # =10 @ 1536x1024,  good for i-beam images
        dx_pixels, dy_pixels = shift_from_crosscorrelation_simple_images(new_ib_lowres_norm * cmask, ib_lowres_reference_norm * cmask, lowpass=lowpass_pixels,
                                                                 highpass=highpass_pixels, sigma=sigma)
        dx_meters = dx_pixels * pixelsize_x_lowres
        dy_meters = dy_pixels * pixelsize_y_lowres
        x_move = x_corrected_stage_movement(-dx_meters)
        yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t, beam_type=BeamType.ION) ##check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    stage.relative_move(x_move)
    stage.relative_move(yz_move)

   ########################################  HIGH resolution alignment  #############################################
    print(' - - - - - - - - - - - - - - Finer alignment - - - - - - - - - - - - - - ...')
    new_eb_highres,  new_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=field_width_highres, image_settings=image_settings)
    storage.SaveImage(new_eb_highres,  id='C_tiltAlign_sample_eb_highres')
    storage.SaveImage(new_ib_highres,  id='C_tiltAlign_sample_ib_highres')
    new_eb_highres_norm  = (new_eb_highres.data - np.mean(new_eb_highres.data)) / np.std(new_eb_highres.data)
    new_ib_highres_norm  = (new_ib_highres.data - np.mean(new_ib_highres.data)) / np.std(new_ib_highres.data)
    ###
    if beam_type==BeamType.ELECTRON:
        lowpass_pixels  = int( max(new_eb_highres.data.shape) / 12 ) # =128 @ 1536x1024, good for e-beam images
        highpass_pixels = int( max(new_eb_highres.data.shape)/ 256 ) # =6   @ 1536x1024, good for e-beam images
        sigma = int( 2 * max(new_eb_highres.data.shape)/1536)        # =2   @ 1536x1024, good for e-beam images
        #dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(new_eb_lowres, ib_lowres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
        dx_pixels, dy_pixels = shift_from_crosscorrelation_simple_images(new_eb_highres_norm * cmask, eb_highres_reference_norm * cmask, lowpass=lowpass_pixels,
                                                                 highpass=highpass_pixels, sigma=sigma_eb)
        dx_meters = dx_pixels * pixelsize_x_highres
        dy_meters = dy_pixels * pixelsize_y_highres
        x_move = x_corrected_stage_movement(-dx_meters)
        yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t, beam_type=BeamType.ELECTRON) ##check electron/ion movement
    if beam_type==BeamType.ION:
        lowpass_pixels  = int( max(new_ib_highres.data.shape) / 6)    # =256 @ 1536x1024,  good for i-beam images
        highpass_pixels = int( max(new_ib_highres.data.shape)/ 64 )   # =24  @ 1536x1024,  good for i-beam images => need a large highpass to remove noise and ringing
        sigma = int( 10 * max(new_ib_highres.data.shape)/1536)        # =10   @ 1536x1024, good for i-beam images
        dx_pixels, dy_pixels = shift_from_crosscorrelation_simple_images(new_ib_highres_norm * cmask, ib_highres_reference_norm * cmask, lowpass=lowpass_pixels,
                                                                 highpass=highpass_pixels, sigma=sigma)
        dx_meters = dx_pixels * pixelsize_x_highres
        dy_meters = dy_pixels * pixelsize_y_highres
        x_move = x_corrected_stage_movement(-dx_meters)
        yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t, beam_type=BeamType.ION) ##check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    stage.relative_move(x_move)
    stage.relative_move(yz_move)

    new_eb_highres, new_ib_highres = take_electron_and_ion_reference_images(microscope,  hor_field_width=field_width_highres, image_settings=image_settings)
    storage.SaveImage(new_eb_highres, id='D_tiltAlign_sample_eb_highres_aligned')
    storage.SaveImage(new_ib_highres, id='D_tiltAlign_sample_ib_highres_aligned')
    storage.step_counter += 1


# def realign_landing_post(microscope, original_landing_images):
#     from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
#     # Unpack reference images
#     ib_low_res_reference, ib_high_res_reference, eb_low_res_reference, eb_high_res_reference = original_landing_images
#     # Low resolution alignment (TODO: magnifications must match, yaml user input)
#     template = ib_low_res_reference
#     microscope.beams.ion_beam.horizontal_field_width.value      = 400e-6  # TODO: user input, can't be smaller than 150e-6
#     microscope.beams.electron_beam.horizontal_field_width.value = 400e-6  # TODO: user input, can't be smaller than 150e-6
#     image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
#     image = new_ion_image(microscope, settings=image_settings)
#     realign_sample_stage(microscope, image, template, beam_type=BeamType.ION, correct_z_height=False)
#     ib = new_ion_image(microscope, settings=image_settings)
#     eb = new_electron_image(microscope, settings=image_settings)
#     # High resolution alignment (TODO: magnifications must match, yaml user input)
#     template = ib_high_res_reference
#     microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # TODO: user input, can't be smaller than 150e-6
#     microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # TODO: user input, can't be smaller than 150e-6
#     image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
#     image = new_ion_image(microscope, settings=image_settings)
#     realign_sample_stage(microscope, image, template, beam_type=BeamType.ION, correct_z_height=False)
#     ib = new_ion_image(microscope, settings=image_settings)
#     eb = new_electron_image(microscope, settings=image_settings)

def realign_landing_post(microscope, reference_images_low_and_high_res, plot=False):
    print('stage shift correction by image cross-correlation')
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    ### Unpack reference images
    eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference = reference_images_low_and_high_res
    storage.SaveImage(eb_lowres_reference,  id='A_ref_eb_lowres')
    storage.SaveImage(eb_highres_reference, id='A_ref_eb_highres')
    storage.SaveImage(ib_lowres_reference,  id='A_ref_ib_lowres')
    storage.SaveImage(ib_highres_reference, id='A_ref_ib_highres')
    ##############
    stage = microscope.specimen.stage
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
    ####
    pixelsize_x_lowres  = ib_lowres_reference.metadata.binary_result.pixel_size.x
    field_width_lowres  = pixelsize_x_lowres  * ib_lowres_reference.width
    pixelsize_x_highres = ib_highres_reference.metadata.binary_result.pixel_size.x
    field_width_highres = pixelsize_x_highres * ib_highres_reference.width
    ########################################  LOW resolution alignment  #############################################
    print('- - - - - - - - - - - - - - Coarse alignment- - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    new_eb_lowres = microscope.imaging.grab_frame(image_settings)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    new_ib_lowres = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(new_eb_lowres,  id='B_sample_eb_lowres')
    storage.SaveImage(new_ib_lowres,  id='B_sample_ib_lowres')
    #
    lowpass_pixels = int(max(new_ib_lowres.data.shape) / 6)   # =256 @ 1536x1024,  good for i-beam images
    highpass_pixels = int(max(new_ib_lowres.data.shape) / 64) # =24  @ 1536x1024,  good for i-beam images => need a large highpass to remove noise and ringing
    sigma = int(10 * max(new_ib_lowres.data.shape) / 1536)    # =10  @ 1536x1024, good for i-beam images
    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(new_ib_lowres, ib_lowres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    x_move  = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ION) ##check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    stage.relative_move(x_move)
    stage.relative_move(yz_move)
    ###
    ########################################  HIGH resolution alignment  #############################################
    print('- - - - - - - - - - - - - - Finer alignment- - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value      = field_width_highres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    new_eb_highres = microscope.imaging.grab_frame(image_settings)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    new_ib_highres = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(new_eb_highres,  id='C_sample_eb_highres_shifted')
    storage.SaveImage(new_ib_highres,  id='C_sample_ib_highres_shifted')
    ########   ------------- correlate------------
    lowpass_pixels = int(max(new_ib_highres.data.shape) / 6)   # =256 @ 1536x1024,  good for i-beam images
    highpass_pixels = int(max(new_ib_highres.data.shape) / 64) # =24  @ 1536x1024,  good for i-beam images => need a large highpass to remove noise and ringing
    sigma = int(10 * max(new_ib_highres.data.shape) / 1536)    # =10  @ 1536x1024, good for i-beam images
    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(new_ib_highres, ib_highres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    x_move  = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ION) ##check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    stage.relative_move(x_move)
    stage.relative_move(yz_move)
    storage.step_counter += 1


###########################################################################################################################################################################




def single_liftout(microscope, settings, landing_coord, lamella_coord, original_landing_images, original_lamella_area_images):

    ### move to the previously stored position and correct the position using the reference images:
    microscope.specimen.stage.absolute_move(lamella_coord)
    #realign_sample_stage(microscope, image, template, beam_type=BeamType.ION, correct_z_height=False)
    correct_stage_drift_using_reference_eb_images(microscope, original_lamella_area_images, plot=False)

    # mill
    mill_lamella(microscope, settings, confirm=False)

    # lift-out
    liftout_lamella(microscope, settings)

    # land
    land_lamella(microscope, landing_coord, original_landing_images)

    # resharpen needle
    sharpen_needle(microscope)


def reload_config(config_filename):
    settings = load_config(config_filename)
    storage.settings = settings




@click.command()
@click.argument("config_filename")
def main_cli(config_filename):
    """Run the main command line interface.
    Parameters
    ----------
    config_filename : str
        Path to protocol file with input parameters given in YAML (.yml) format
    """
    settings = load_config(config_filename)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S') #datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    output_log_filename = os.path.join('logfile' + timestamp + '.log')
    configure_logging(log_filename=output_log_filename)
    main(settings)


def main(settings):
    storage.NewRun()
    microscope = initialize(settings["system"]["ip_address"])
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    autocontrast(microscope, beam_type=BeamType.ION)
    eb = new_electron_image()
    if ask_user("Do you want to sputter the whole sample grid with platinum? yes/no: "):
        sputter_platinum_over_whole_grid(microscope)
    print("Please select the landing positions and check eucentric height manually.")
    landing_coordinates, original_landing_images = find_coordinates(microscope, name="landing position", move_stage_angle="landing")
    lamella_coordinates, original_trench_images  = find_coordinates(microscope, name="lamella",          move_stage_angle="trench")
    zipped_coordinates = list(zip(lamella_coordinates, landing_coordinates))
    storage.LANDING_POSTS_POS_REF = original_landing_images
    storage.LAMELLA_POS_REF       = original_trench_images
    # Start liftout for each lamella
    for i, (lamella_coord, landing_coord) in enumerate(zipped_coordinates):
        landing_reference_images      = original_landing_images[i]
        lamella_area_reference_images = original_trench_images[i]
        single_liftout(microscope, settings, landing_coord, lamella_coord, landing_reference_images, lamella_area_reference_images)
        storage.liftout_counter += 1
    print("Finished.")



if __name__ == '__main__':
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    # try:
    #     main_cli()
    # except KeyboardInterrupt:
    #     logging.error('Keyboard Interrupt: Cancelling program.')
    opts, args = getopt.getopt(sys.argv[1:], 'q:', ['run='])
    print(opts)
    print(args)
    config_filename =args[0]
    settings = load_config(config_filename)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S') #datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    output_log_filename = os.path.join('logfile' + timestamp + '.log')
    configure_logging(log_filename=output_log_filename)



    storage.NewRun(prefix='edge_landing_detection_')
    storage.settings = settings

    microscope = initialize(settings["system"]["ip_address"])
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    autocontrast(microscope, beam_type=BeamType.ION)



    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    microscope.imaging.set_active_view(1)
    microscope.beams.electron_beam.horizontal_field_width.value = 2750e-6
    eb = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(eb, id='grid')
    storage.step_counter += 1

    if ask_user("Do you want to sputter the whole sample grid with platinum? yes/no: "):
        sputter_platinum_over_whole_grid(microscope)

    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    microscope.beams.electron_beam.horizontal_field_width.value = 2750e-6
    eb = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(eb, id='grid_Pt_deposition')
    storage.step_counter += 1

    print("Please select the landing positions and check eucentric height manually.")
    landing_coordinates, original_landing_images = find_coordinates(microscope, name="landing position", move_stage_angle="landing")
    lamella_coordinates, original_trench_images  = find_coordinates(microscope, name="lamella",          move_stage_angle="trench")
    zipped_coordinates = list(zip(lamella_coordinates, landing_coordinates))
    storage.LANDING_POSTS_POS_REF = original_landing_images
    storage.LAMELLA_POS_REF       = original_trench_images
    # Start liftout for each lamella
    for i, (lamella_coord, landing_coord) in enumerate(zipped_coordinates):
        landing_reference_images      = original_landing_images[i]
        lamella_area_reference_images = original_trench_images[i]
        single_liftout(microscope, settings, landing_coord, lamella_coord, landing_reference_images, lamella_area_reference_images)
        storage.liftout_counter += 1
    print("Finished.")






    '''###quick test
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    microscope = initialize(settings["system"]["ip_address"])
    park_position = move_needle_to_landing_position(microscope)
    image_settings = GrabFrameSettings(resolution="3072x2048", dwell_time=0.1e-6)
    image_settings_electron = GrabFrameSettings(resolution="3072x2048", dwell_time=1e-6)
    image_settings_ion      = GrabFrameSettings(resolution="3072x2048", dwell_time=1e-6)
    microscope.beams.ion_beam.horizontal_field_width.value      = 80e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 80e-6
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    eb = microscope.imaging.grab_frame(image_settings_electron)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    ib  = microscope.imaging.grab_frame(image_settings_ion)
    x_shift, y_shift = find_needletip_shift_in_image_ELECTRON(eb, eb, show=True, median_smoothing=2)
    retract_needle(microscope, park_position)'''