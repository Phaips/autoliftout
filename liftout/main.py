"""Main entry script."""


import click
from datetime import datetime
import time
import os
import sys
import glob
import getopt
import logging
from enum import Enum
import numpy as np
import tqdm
# import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
# from scipy import fftpack, misc
from PIL import Image, ImageDraw#, ImageFilter
# from matplotlib.patches import Circle
# from scipy.ndimage.morphology import binary_dilation
# from skimage.filters import gaussian, threshold_otsu, median
# from skimage.measure import label
# from skimage.morphology import disk
import datetime

from scipy import *
from scipy import signal
from scipy import fftpack, misc
import scipy
# import os
# import sys
# import glob

import skimage.draw
import skimage.io

from operator import itemgetter

# from liftout.calibration import setup
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
import liftout.detection as detection
# import liftout.AutoLiftout as AutoLiftout

PRETILT_DEGREES = 27


class BeamType(Enum):
    ION = 'ION'
    ELECTRON = 'ELECTRON'


# GLOBAL VARIABLE
class Storage():
    def __init__(self, DIR=''):
        self.DIR = DIR
        self.NEEDLE_REF_IMGS = []  # dict()
        self.NEEDLE_WITH_SAMPLE_IMGS = []  # dict()
        self.LANDING_POSTS_REF = []
        self.TRECHNING_POSITIONS_REF = []
        self.MILLED_TRENCHES_REF = []
        self.liftout_counter = 0
        self.step_counter = 0
        self.settings = ''

    def AddDirectory(self, DIR):
        self.DIR = DIR

    def NewRun(self, prefix='RUN'):
        self.__init__(self.DIR)
        if self.DIR == '':
            self.DIR = os.getcwd()  # dirs = glob.glob(saveDir + "/ALIGNED_*")        # nn = len(dirs) + 1
        stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')
        self.saveDir = self.DIR + '/' + prefix + '_' + stamp
        self.saveDir = self.saveDir.replace('\\', '/')
        os.mkdir(self.saveDir)

    def SaveImage(self, image, dir_prefix='', id=''):
        if len(dir_prefix) > 0:
            self.path_for_image = self.saveDir + '/' + dir_prefix + '/'
        else:
            self.path_for_image = self.saveDir + '/' + 'liftout%03d' % (self.liftout_counter) + '/'
        if not os.path.isdir(self.path_for_image):
            print('creating directory')
            os.mkdir(self.path_for_image)
            print(self.path_for_image)
        self.fileName = self.path_for_image + 'step%02d' % (self.step_counter) + '_' + id + '.tif'
        print(self.fileName)
        image.save(self.fileName)


# storage = Storage()  # global variable


def configure_logging(log_filename='logfile', log_level=logging.INFO):
    """Log to the terminal and to file simultaneously."""
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        '%Y%m%d.%H%M%S')  # datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
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
            -(width / 2) * pixelsize_x,
            +(width / 2) * pixelsize_x,
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


def take_electron_and_ion_reference_images(
    microscope,
    hor_field_width=50e-6,
    image_settings=None,
    __autocontrast=True,
    eb_brightness=None,
    ib_brightness=None,
    eb_contrast=None,
    ib_contrast=None,
    save=False,
    save_label="default",
):
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        GrabFrameSettings,
    )

    # image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    #############

    # Take reference images with lower resolution, wider field of view
    microscope.beams.electron_beam.horizontal_field_width.value = hor_field_width
    microscope.beams.ion_beam.horizontal_field_width.value = hor_field_width
    microscope.imaging.set_active_view(1)
    if __autocontrast:
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        eb_brightness, eb_contrast = None, None # use autoconstrast values
    eb_reference = new_electron_image(
        microscope, image_settings, eb_brightness, eb_contrast
    )
    microscope.imaging.set_active_view(2)
    if __autocontrast:
        autocontrast(microscope, beam_type=BeamType.ION)
        ib_brightness, ib_contrast = None, None # use autoconstrast values
    ib_reference = new_ion_image(microscope, image_settings, ib_brightness, ib_contrast)

    # save images
    if save:
        storage.SaveImage(eb_reference, id=save_label + "_eb")
        storage.SaveImage(ib_reference, id=save_label + "_ib")

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
    flat_to_ion_beam(microscope, pretilt_angle=pretilt_angle)  # stage tilt 25
    microscope.specimen.stage.relative_move(StagePosition(t=np.deg2rad(landing_angle)))  # more tilt by 18
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
    # new_electron_image(microscope)
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
    # new_electron_image(microscope)
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
    sputter_platinum(microscope, sputter_time=20, horizontal_field_width=30e-6, line_pattern_length=7e-6)
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
    microscope.beams.ion_beam.horizontal_field_width.value = 900e-6
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    autocontrast(microscope, beam_type=BeamType.ION)
    _eucentric_height_adjustment(microscope)
    print("Final eucentric alignment")
    microscope.beams.electron_beam.horizontal_field_width.value = 200e-6
    microscope.beams.ion_beam.horizontal_field_width.value = 200e-6
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
    coordinates = []
    landing_post_reference_images = []
    trench_area_reference_images = []
    ###
    select_another_position = True
    while select_another_position:
        if move_stage_angle == "trench":
            ensure_eucentricity(microscope)
            move_to_trenching_angle(microscope)  # flat to ion_beam
        elif move_stage_angle == "landing":
            move_to_landing_angle(microscope)
        microscope.beams.electron_beam.horizontal_field_width.value = 400e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value = 400e-6  # TODO: yaml use input
        # refocus_and_relink(microscope)
        eb = new_electron_image(microscope)
        ib = new_ion_image(microscope)
        if ask_user(f"Please center the {name} coordinate in the ion beam.\n"
                    f"Is the {name} feature centered in the ion beam? yes/no: "):
            eb = new_electron_image(microscope)
            coordinates.append(microscope.specimen.stage.current_position)
            if move_stage_angle == "landing":

                resolution = storage.settings["reference_images"]["landing_post_ref_img_resolution"]
                dwell_time = storage.settings["reference_images"]["landing_post_ref_img_dwell_time"]
                image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
                hfw_lowres = storage.settings["reference_images"]["landing_post_ref_img_hfw_lowres"]
                hfw_highres = storage.settings["reference_images"]["landing_post_ref_img_hfw_highres"]
                eb_lowres,  ib_lowres = take_electron_and_ion_reference_images(
                    microscope, hor_field_width=hfw_lowres,  image_settings=image_settings)
                eb_highres, ib_highres = take_electron_and_ion_reference_images(
                    microscope, hor_field_width=hfw_highres, image_settings=image_settings)
                landing_post_reference_images.append((eb_lowres, eb_highres, ib_lowres, ib_highres))
            if move_stage_angle == "trench":

                resolution = storage.settings["reference_images"]["trench_area_ref_img_resolution"]
                dwell_time = storage.settings["reference_images"]["trench_area_ref_img_dwell_time"]
                image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
                hfw_lowres = storage.settings["reference_images"]["trench_area_ref_img_hfw_lowres"]
                hfw_highres = storage.settings["reference_images"]["trench_area_ref_img_hfw_highres"]
                eb_lowres, ib_lowres = take_electron_and_ion_reference_images(
                    microscope,   hor_field_width=hfw_lowres,  image_settings=image_settings)
                eb_highres, ib_highres = take_electron_and_ion_reference_images(
                    microscope, hor_field_width=hfw_highres, image_settings=image_settings)
                trench_area_reference_images.append((eb_lowres, eb_highres, ib_lowres, ib_highres))

            print(microscope.specimen.stage.current_position)
            select_another_position = ask_user(
                f"Do you want to select another {name} position? "
                f"{len(coordinates)} selected so far. yes/no: ")
    if move_stage_angle == "landing":
        return coordinates, landing_post_reference_images
    else:
        return coordinates, trench_area_reference_images

def calculate_shift_between_features_in_metres(img, shift_type, show=True, validate=True):

    # detector class (model)
    weights_file = storage.settings["machine_learning"]["weights"]
    detector = detection.Detector(weights_file)

    print("shift_type: ", shift_type)
    x_distance, y_distance = detector.calculate_shift_between_features(img, shift_type=shift_type, show=show, validate=validate)
    print(f"x_distance = {x_distance:.4f}, y_distance = {y_distance:.4f}")

    x_shift, y_shift = detection.calculate_shift_distance_in_metres(img, x_distance, y_distance)
    print(f"x_shift =  {x_shift/1e-6:.4f}, um; y_shift = {y_shift/1e-6:.4f} um; ")

    return x_shift, y_shift

########################################LIFT-OUT#########################################################

def land_needle_on_milled_lamella(microscope):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # image settings from config
    resolution = storage.settings["reference_images"]["needle_ref_img_resolution"]
    dwell_time = storage.settings["reference_images"]["needle_ref_img_dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
    hfw_lowres = storage.settings["reference_images"]["needle_ref_img_hfw_lowres"]
    hfw_highres = storage.settings["reference_images"]["needle_ref_img_hfw_highres"]

    # ml settings
    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]

    # take initial reference images
    needle_eb_lowres_with_lamella, needle_ib_lowres_with_lamella = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres, image_settings=image_settings,
                                                                                                          __autocontrast=True,
                                                                                                          eb_brightness=eb_brightness, ib_brightness=ib_brightness,
                                                                                                          eb_contrast=eb_contrast, ib_contrast=ib_contrast,
                                                                                                          save=True, save_label="B_needle_land_sample_lowres")
    needle_eb_highres_with_lamella, needle_ib_highres_with_lamella = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres, image_settings=image_settings,
                                                                                                            __autocontrast=True,
                                                                                                            eb_brightness=eb_brightness, ib_brightness=ib_brightness,
                                                                                                            eb_contrast=eb_contrast, ib_contrast=ib_contrast,
                                                                                                            save=True, save_label="B_needle_land_sample_highres")
    ############################### Initial X-Y Movement ###############################

    # calculate shift in image coordinates
    x_shift, y_shift = calculate_shift_between_features_in_metres(needle_eb_lowres_with_lamella, "needle_tip_to_lamella_centre")

    # move needle in x and y (e-beam)
    x_move = x_corrected_needle_movement(-x_shift)
    y_move = y_corrected_needle_movement(y_shift, stage.current_position.t)
    print('Needle approach from e-beam high row res:')
    print('x_move = ', x_move, ';\ny_move = ', y_move)
    needle.relative_move(x_move)
    needle.relative_move(y_move)

    ############################### Take Reference Images of XY Movement ###############################

    needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres, image_settings=image_settings,
                                                                                                          __autocontrast=True,
                                                                                                          eb_brightness=eb_brightness, ib_brightness=ib_brightness,
                                                                                                          eb_contrast=eb_contrast, ib_contrast=ib_contrast,
                                                                                                          save=True, save_label="C_needle_land_sample_lowres_after_xy_shift")
    needle_eb_highres_with_lamella_shifted, needle_ib_highres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres, image_settings=image_settings,
                                                                                                            __autocontrast=True,
                                                                                                            eb_brightness=eb_brightness, ib_brightness=ib_brightness,
                                                                                                            eb_contrast=eb_contrast, ib_contrast=ib_contrast,
                                                                                                            save=True, save_label="C_needle_land_sample_highres_after_xy_shift")




    #############################  Initial Z Half Movement  ############################

    hfw_lowres = storage.settings["reference_images"]["needle_with_lamella_shifted_img_lowres"]
    needle_eb_highres_with_lamella_shifted, needle_ib_highres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres,
                                                                                                                            image_settings=image_settings, __autocontrast=False,
                                                                                                                            save=True, save_label="D_needle_land_sample_highres_before_z_shift")
    # calculate shift in image coordinates (i-beam)
    x_shift, y_shift = calculate_shift_between_features_in_metres(needle_ib_lowres_with_lamella_shifted, "needle_tip_to_lamella_centre")

    # calculate shift in xyz coordinates
    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    print('cos(t) = ', np.cos(stage_tilt))
    z_distance = y_shift / np.cos(stage_tilt)

    # Calculate movement
    print('Needle approach from i-beam low res - Z: landing')
    print('Needle move in Z by half the distance...', z_distance)
    zy_move_half = z_corrected_needle_movement(-z_distance / 2, stage_tilt)
    needle.relative_move(zy_move_half)


    #############################  Final Z-Movement to Land  ############################
    # final z-movement to land
    hfw_highres = storage.settings["reference_images"]["needle_with_lamella_shifted_img_highres"]
    needle_eb_highres_with_lamella_shifted, needle_ib_highres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6,
                                                                                                                          image_settings=image_settings, __autocontrast=True,
                                                                                                                          save=True, save_label="E_needle_land_sample_lowres_shifted_half_z" )
    # calculate shift in image coordinates (-beam)
    x_shift, y_shift = calculate_shift_between_features_in_metres(needle_ib_highres_with_lamella_shifted, "needle_tip_to_lamella_centre")

    # calculate shift in xyz coordinates
    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    print('cos(t) = ', np.cos(stage_tilt))
    z_distance = y_shift / np.cos(stage_tilt)
    print("z_distance: ", z_distance)

    # Calculate movement
    print('Needle approach from i-beam low res - Z: landing')

    # move in x
    x_move = x_corrected_needle_movement(-x_shift)
    print('x_move = ', x_move)
    needle.relative_move(x_move)

    # move in z
    gap = 0.5e-6
    zy_move_gap = z_corrected_needle_movement(-z_distance - gap, stage_tilt)
    needle.relative_move(zy_move_gap)
    print('Needle move in Z minus gap ... LANDED')

    #############################  Needle Landed  ############################

    # Take reference images after dx,dy,dz shift and landing
    hfw_lowres = storage.settings["reference_images"]["needle_ref_img_hfw_lowres"]
    hfw_highres = storage.settings["reference_images"]["needle_ref_img_hfw_highres"]

    eb_lowres_needle_landed, ib_lowres_needle_landed = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres, image_settings=image_settings,
                                                                                                          __autocontrast=True,
                                                                                                          eb_brightness=eb_brightness, ib_brightness=ib_brightness,
                                                                                                          eb_contrast=eb_contrast, ib_contrast=ib_contrast,
                                                                                                          save=True, save_label="E_needle_land_sample_ib_lowres_landed")
    eb_highres_needle_landed, ib_highres_needle_landed = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres, image_settings=image_settings,
                                                                                                            __autocontrast=True,
                                                                                                            eb_brightness=eb_brightness, ib_brightness=ib_brightness,
                                                                                                            eb_contrast=eb_contrast, ib_contrast=ib_contrast,
                                                                                                            save=True, save_label="E_needle_land_sample_highres_landed")
    storage.step_counter += 1

def liftout_lamella(microscope, settings):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    needle = microscope.specimen.manipulator
    stage = microscope.specimen.stage

    # image settings from config
    resolution = storage.settings["reference_images"]["needle_ref_img_resolution"]
    dwell_time = storage.settings["reference_images"]["needle_ref_img_dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)

    # move needle to liftout start position
    park_position = move_needle_to_liftout_position(microscope)

    # land needle on lamella
    land_needle_on_milled_lamella(microscope)

    # weld needle to lamella
    sputter_platinum(microscope, sputter_time=30)  # TODO: yaml user input for sputtering application file choice

    storage.step_counter += 1

    eb, ib = take_electron_and_ion_reference_images(
        microscope, hor_field_width=80e-6, image_settings=image_settings,
        save=True, save_label="landed_Pt_sputter")
    storage.step_counter += 1

    # TODO: yaml user input for jcut milling current
    mill_to_sever_jcut(microscope, settings['jcut'], confirm=True)
    eb, ib = take_electron_and_ion_reference_images(
        microscope, hor_field_width=80e-6, image_settings=image_settings,
        save=True, save_label="jcut_sever")
    storage.step_counter += 1

    # TAKE NEEDLE z_UP (>30 MICRONS), TAKE GIS OUT, RESTRACT TO PARKING
    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')

    for i in range(3):
        print("Moving out of trench by 10um")
        z_move_out_from_trench = z_corrected_needle_movement(10e-6, stage_tilt)
        needle.relative_move(z_move_out_from_trench)
        time.sleep(1)

    eb, ib = take_electron_and_ion_reference_images(
        microscope, hor_field_width=150e-6, image_settings=image_settings,
        save=True, save_label="liftout_of_trench")
    storage.step_counter += 1

    retract_needle(microscope, park_position)

    # liftout is finished, no need for reference images

############################################# MILLING #######################################################



def mill_thin_lamella(microscope, settings, confirm=True):
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
            # return
    print('Milling trenches')
    protocol_stages = []

    for stage_settings in settings["thin_lamella"]["protocol_stages"]:
        tmp_settings = settings["thin_lamella"].copy()
        tmp_settings.update(stage_settings)

        protocol_stages.append(tmp_settings)

    # protocol_stages = protocol_stage_settings(settings)
    for stage_number, stage_settings in enumerate(protocol_stages):
        print("Protocol stage {} of {}".format(
            stage_number + 1, len(protocol_stages)))
        mill_single_stage(
            microscope,
            settings,
            stage_settings,
            stage_number)
    # Restore ion beam imaging current (20 pico-Amps)
    microscope.beams.ion_beam.beam_current.value = 30e-12


def mill_trenches(microscope, settings, confirm=True):
    """Mill the trenches for thinning the lamella.
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
    microscope.beams.ion_beam.beam_current.value = 30e-12


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
    if storage.settings["imaging"]["imaging_current"]:
        imaging_current = storage.settings["imaging"]["imaging_current"]
    print("Ok, running ion beam milling now...")
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.beams.ion_beam.beam_current.value = milling_current
    microscope.patterning.run()
    print("Returning to the ion beam imaging current now.")
    microscope.patterning.clear_patterns()
    microscope.beams.ion_beam.beam_current.value = imaging_current
    print("Ion beam milling complete.")


def confirm_and_run_milling(microscope, milling_current, *,
                            imaging_current=30e-12, confirm=True):
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
    if storage.settings["imaging"]["imaging_current"]:
        imaging_current = storage.settings["imaging"]["imaging_current"]
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
    height = (jcut_lamella_depth + extra_bit) * angle_correction_factor
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
    pattern = _create_mill_pattern(microscope,
                                     center_x=cut_coord["center_x"], center_y=cut_coord["center_y"],
                                     width=cut_coord["width"], height=cut_coord["height"],
                                     depth=cut_coord["depth"], rotation_degrees=cut_coord["rotation"], ion_beam_field_of_view=cut_coord["hfw"])
    confirm_and_run_milling(microscope, milling_current, confirm=confirm)


def _create_mill_pattern(microscope, *,
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


# def _create_sharpen_pattern(microscope, *,
#                             center_x=-10.5e-6,
#                             center_y=-5e-6,
#                             width=8e-6,
#                             height=2e-6,
#                             depth=1e-6,
#                             rotation_degrees=40,
#                             ion_beam_field_of_view=100e-6):

#     pattern = microscope.patterning.create_rectangle(
#         center_x, center_y, width, height, depth)
#     pattern.rotation = np.deg2rad(rotation_degrees)
#     return pattern

# def _create_mill_pattern(microscope, *,
#                             center_x=-10.5e-6,
#                             center_y=-5e-6,
#                             width=8e-6,
#                             height=2e-6,
#                             depth=1e-6,
#                             rotation_degrees=40,
#                             ion_beam_field_of_view=100e-6):

#     pattern = microscope.patterning.create_rectangle(
#         center_x, center_y, width, height, depth)
#     pattern.rotation = np.deg2rad(rotation_degrees)
#     return pattern

# def select_point_new(image):
#     fig, ax = plt.subplots()
#     ax.imshow(image, cmap="gray")
#     coords = []

#     def on_click(event):
#         print(event.xdata, event.ydata)
#         coords.append(event.ydata)
#         coords.append(event.xdata)

#     fig.canvas.mpl_connect("button_press_event", on_click)
#     plt.show()

#     return tuple(coords[-2:])


# def validate_detection(img, img_base, detection_coord, det_type):
#     correct = input(f"Is {det_type} correct? (y/n)")

#     if correct == "n":

#         print(f"Please click the {det_type} position")
#         detection_coord = select_point_new(img)

#         # save image for training here
#         print("Saving image for labelling")
#         storage.step_counter += 1
#         storage.SaveImage(img_base, id="label_")

#     print(detection_coord)
#     return detection_coord

def quick_eucentric_test():
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import StagePosition
    from autoscript_sdb_microscope_client.structures import MoveSettings
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage = microscope.specimen.stage

    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=0.5e-6)  # TODO: user input resolution

    realign_eucentric_with_machine_learning(microscope, image_settings=image_settings_ML, hor_field_width=80e-6)


def realign_eucentric_with_machine_learning(microscope, image_settings, hor_field_width=150e-6, _autocontrast=False):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    from autoscript_sdb_microscope_client.structures import StagePosition
    """ Realign image to lamella centre using ML detection"""
    stage = microscope.specimen.stage

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]

    eb_lowres,  ib_lowres = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hor_field_width, image_settings=image_settings,
        __autocontrast=_autocontrast,
        eb_brightness=eb_brightness, eb_contrast=eb_contrast,
        ib_brightness=ib_brightness, ib_contrast=ib_contrast,
        save=True, save_label="A_eucentric_calibration_lowres")

    storage.step_counter += 1

    # new
    x_shift, y_shift = calculate_shift_between_features_in_metres(ib_lowres, "lamella_centre_to_image_centre")

    # yz-correction
    tilt_radians = stage.current_position.t
    yz_move = y_corrected_stage_movement(y_shift, stage_tilt=tilt_radians, beam_type=BeamType.ION)
    stage.relative_move(yz_move)

    # x correction
    stage.relative_move(StagePosition(x=-x_shift))

    # electron dy shift
    eb_lowres,  ib_lowres = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hor_field_width, image_settings=image_settings,
        __autocontrast=_autocontrast,
        eb_brightness=eb_brightness, eb_contrast=eb_contrast,
        ib_brightness=ib_brightness, ib_contrast=ib_contrast,
        save=True, save_label="A_eucentric_calibration_lowres_moved_1")

    # new
    x_shift, y_shift = calculate_shift_between_features_in_metres(eb_lowres, "lamella_centre_to_image_centre")

    # yz-correction
    tilt_radians = stage.current_position.t
    yz_move = y_corrected_stage_movement(y_shift, stage_tilt=tilt_radians, beam_type=BeamType.ELECTRON)
    stage.relative_move(yz_move)

    # x correction
    stage.relative_move(StagePosition(x=-x_shift))

    # ion again
    eb_lowres,  ib_lowres = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hor_field_width, image_settings=image_settings,
        __autocontrast=_autocontrast,
        eb_brightness=eb_brightness, eb_contrast=eb_contrast,
        ib_brightness=ib_brightness, ib_contrast=ib_contrast,
        save=True, save_label="A_eucentric_calibration_lowres_moved_2",)

    # new
    x_shift, y_shift = calculate_shift_between_features_in_metres(ib_lowres, "lamella_centre_to_image_centre")

    # yz-correction
    tilt_radians = stage.current_position.t
    yz_move = y_corrected_stage_movement(y_shift, stage_tilt=tilt_radians, beam_type=BeamType.ION)
    stage.relative_move(yz_move)

    # x correction
    stage.relative_move(StagePosition(x=-x_shift))

    # # take final position images
    eb_lowres,  ib_lowres = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hor_field_width, image_settings=image_settings,
        __autocontrast=_autocontrast,
        eb_brightness=eb_brightness, eb_contrast=eb_contrast,
        ib_brightness=ib_brightness, ib_contrast=ib_contrast,
        save=True, save_label="A_eucentric_calibration_lowres_final",)

    storage.step_counter += 1


def mill_lamella(microscope, settings, confirm=True):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import StagePosition
    from autoscript_sdb_microscope_client.structures import MoveSettings
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage = microscope.specimen.stage

    # Set the correct magnification / field of view
    field_of_view = 100e-6  # in meters  TODO: user input from yaml settings
    microscope.beams.ion_beam.horizontal_field_width.value = field_of_view
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view

    # Move to trench position
    move_to_trenching_angle(microscope)  # <----flat to the ion, stage tilt 25 (total image tilt 52)

    # Take an ion beam image at the *milling current*
    ib = new_ion_image(microscope)
    mill_trenches(microscope, settings, confirm=confirm)

    #############
    # Take reference images after trech milling, use them to realign after stage rotation to flat_to_electron position
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution

    resolution = storage.settings["reference_images"]["trench_area_ref_img_resolution"]
    dwell_time = storage.settings["reference_images"]["trench_area_ref_img_dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
    hfw_lowres = storage.settings["reference_images"]["trench_area_ref_img_hfw_lowres"]
    hfw_highres = storage.settings["reference_images"]["trench_area_ref_img_hfw_highres"]

    eb_lowres_reference,  ib_lowres_reference = take_electron_and_ion_reference_images(
        microscope,  hor_field_width=hfw_lowres,  image_settings=image_settings)
    eb_highres_reference, ib_highres_reference = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hfw_highres, image_settings=image_settings)
    reference_images_low_and_high_res = (eb_lowres_reference, eb_highres_reference,
                                         ib_lowres_reference, ib_highres_reference)  # use these images for future alignment

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]


    storage.step_counter += 1
    eb_lowres, ib_lowres = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_lowres, image_settings=image_settings,
                                    __autocontrast=False,
                                    eb_brightness=eb_brightness, eb_contrast=eb_contrast,
                                    ib_brightness=ib_brightness, ib_contrast=ib_contrast,
                                    save=True, save_label="ref_A_lowres_brightness_contrast")


    eb_highres, ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=hfw_highres, image_settings=image_settings,
                                __autocontrast=False,
                                eb_brightness=eb_brightness, eb_contrast=eb_contrast,
                                ib_brightness=ib_brightness, ib_contrast=ib_contrast,
                                save=True, save_label="ref_A_highres_brightness_contrast")

    # use these images for future alignment

    # Move to flat_to_electron, take electron beam images, align using ion-beam image from tenching angle, Move to Jcut angle(+6 deg)
    flat_to_electron_beam(microscope, pretilt_angle=PRETILT_DEGREES)  # rotate to flat_to_electron
    microscope.beams.ion_beam.horizontal_field_width.value = 400e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 400e-6

    ############################### Realign with Ref Images ###############################

    # correct the stage drift after 180 deg rotation using treched lamella images as reference
    realign_using_reference_eb_and_ib_images(microscope, reference_images_low_and_high_res, plot=True)


    ############################### Realign with ML ###############################

    # realign eucentric height (centred) with model
    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=0.5e-6)  # TODO: user input resolution
    hfw_ml = 80e-6 # 150e-6, 80e-6, 50e-6
    realign_eucentric_with_machine_learning(microscope, image_settings=image_settings_ML, hor_field_width=hfw_ml, _autocontrast=True)


    # take reference images after alignment
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]

    storage.step_counter += 1
    eb_lowres, ib_lowres = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings,
                            __autocontrast=False,
                            eb_brightness=eb_brightness, eb_contrast=eb_contrast,
                            ib_brightness=ib_brightness, ib_contrast=ib_contrast,
                            save=True, save_label="aligned_A_lowres_brightness_contrast")
    eb_highres, ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6, image_settings=image_settings,
                            __autocontrast=False,
                            eb_brightness=eb_brightness, eb_contrast=eb_contrast,
                            ib_brightness=ib_brightness, ib_contrast=ib_contrast,
                            save=True, save_label="aligned_A_highres_brightness_contrast")

    storage.step_counter += 1

    # move to j-cut angle (tilt)

    ############################### Align After Tilt 01 ###############################

    # Need to tilt +6 deg to j-cut, BUT tilt first +3 deg only:
    previous_stage_tilt = stage.current_position.t
    # 1/2 tilt: move_to_jcut_angle(microscope)<--flat to electron beam + jcut_angle=6, stage tilt total 33
    tilting = StagePosition(x=0, y=0, z=0, t=np.deg2rad(3))
    print(tilting)
    stage.relative_move(tilting, stage_settings)

    # realign with model
    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution
    realign_eucentric_with_machine_learning(microscope, image_settings_ML,
                                            hor_field_width=150e-6, _autocontrast=True)


    ############################### Align After Tilt 02 ###############################

    # take reference images after alignment
    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
    storage.step_counter += 1
    eb_lowres, ib_lowres = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings,
                            __autocontrast=True,
                            eb_brightness=eb_brightness, eb_contrast=eb_contrast,
                            ib_brightness=ib_brightness, ib_contrast=ib_contrast,
                            save=True, save_label="tilt_3_deg_A_lowres_brightness_contrast")
    eb_highres, ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6, image_settings=image_settings,
                            __autocontrast=True,
                            eb_brightness=eb_brightness, eb_contrast=eb_contrast,
                            ib_brightness=ib_brightness, ib_contrast=ib_contrast,
                            save=True, save_label="tilt_3_deg_A_highres_brightness_contrast")

    # # use these images for future alignment
    storage.step_counter += 1

    # Need to tilt +6 deg, tilt first +3 deg only, again +3deg, Now +6 deg
    previous_stage_tilt = stage.current_position.t
    tilting = StagePosition(x=0, y=0, z=0, t=np.deg2rad(3))
    print(tilting)
    stage.relative_move(tilting, stage_settings)

    # realign with model
    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=1.0e-6)  # TODO: user input resolution
    realign_eucentric_with_machine_learning(microscope, image_settings_ML,
                                            hor_field_width=hfw_ml, _autocontrast=True)


    ############################### Mill J-Cut ###############################
    mill_jcut(microscope, settings['jcut'], confirm=True)

    # take reference images after jcut
    eb_lowres_ref_jcut,  ib_lowres_ref_jcut = take_electron_and_ion_reference_images(
        microscope, hor_field_width=150e-6, image_settings=image_settings,
        save=True, save_label="jcut_lowres")  # TODO: yaml use input
    eb_highres_ref_jcut, ib_highres_ref_jcut = take_electron_and_ion_reference_images(
        microscope, hor_field_width=50e-6, image_settings=image_settings,
        save=True, save_label="jcut_highres")  # TODO: yaml use input

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]

    storage.step_counter += 1
    eb_lowres, ib_lowres = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings,
                            __autocontrast=False,
                            eb_brightness=eb_brightness, eb_contrast=eb_contrast,
                            ib_brightness=ib_brightness, ib_contrast=ib_contrast,
                            save=True, save_label="jcut_lowres_brightness_contrast")
    eb_highres, ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6, image_settings=image_settings,
                            __autocontrast=False,
                            eb_brightness=eb_brightness, eb_contrast=eb_contrast,
                            ib_brightness=ib_brightness, ib_contrast=ib_contrast,
                            save=True, save_label="jcut_highres_brightness_contrast")
    storage.step_counter += 1

    ########################## Prepare for Liftout ##########################
    # go from j-cut (33 deg) to liftout angle (37)
    previous_stage_tilt = stage.current_position.t
    # <----flat to electron beam + 10 deg - MOVE 33->37 AND CORRECT the drift in ibeam
    move_to_liftout_angle(microscope)


    # realign with model after moving to liftout angle
    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=0.5e-6)  # TODO: user input resolution
    hfw_ml = 80e-6 # 150e-6, 80e-6, 50e-6
    realign_eucentric_with_machine_learning(microscope, image_settings_ML, hor_field_width=hfw_ml, _autocontrast=True)

    print("Done, ready for liftout!")

def land_lamella(microscope, landing_coord, original_landing_images):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition, StagePosition
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # tilt to zero to avoid contact
    stage_settings = MoveSettings(rotate_compucentric=True)
    microscope.specimen.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)

    # move to landing coordinate
    microscope.specimen.stage.absolute_move(landing_coord)
    realign_landing_post(microscope, original_landing_images)
    park_position = move_needle_to_landing_position(microscope)

    # Lamella to Landing Post
    # TODO: check if the landing post is cslose enough to the centre


    # image settings from config
    resolution = storage.settings["reference_images"]["landing_post_ref_img_resolution"]
    dwell_time = storage.settings["reference_images"]["landing_post_ref_img_dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)

    # y-movement
    needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6,
                                                                                                                          image_settings=image_settings,
                                                                                                                          save=True, save_label="A_landing_needle_land_sample_lowres")

    x_shift, y_shift = calculate_shift_between_features_in_metres(needle_eb_lowres_with_lamella_shifted, "lamella_edge_to_landing_post")

    # x_move = x_corrected_needle_movement(x_shift)
    y_move = y_corrected_needle_movement(-y_shift, stage.current_position.t)
    print('x_move = ', x_move, ';\ny_move = ', y_move)
    needle.relative_move(y_move)

    # z-movement
    needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6,
                                                                                                                          image_settings=image_settings,
                                                                                                                          save=True, save_label="B_landing_needle_land_sample_lowres_after_y_move")


    x_shift, y_shift = calculate_shift_between_features_in_metres(needle_ib_lowres_with_lamella_shifted, "lamella_edge_to_landing_post")


    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    z_distance = y_shift / np.sin(np.deg2rad(52))
    z_move = z_corrected_needle_movement(z_distance, stage_tilt)
    print('z_move = ', z_move)
    needle.relative_move(z_move)

    # x-movement
    needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6,
                                                                                                                          image_settings=image_settings,
                                                                                                                          save=True, save_label="C_landing_needle_land_sample_lowres_after_z_move")

    x_shift, y_shift = calculate_shift_between_features_in_metres(needle_eb_lowres_with_lamella_shifted, "lamella_edge_to_landing_post")

    # half move
    x_move = x_corrected_needle_movement(x_shift / 2)
    print('x_move (half) = ', x_move)

    needle.relative_move(x_move)

    # x-movement
    needle_eb_highres_with_lamella_shifted, needle_ib_highres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6,
                                                                                                                          image_settings=image_settings,
                                                                                                                          save=True, save_label="C_landing_needle_land_sample_lowres_after_x_half_move")


    x_shift, y_shift = calculate_shift_between_features_in_metres(needle_eb_highres_with_lamella_shifted, "lamella_edge_to_landing_post")

    # x-move the rest of the way
    x_move = x_corrected_needle_movement(x_shift)
    print('x_move = ', x_move)

    needle.relative_move(x_move)

    # take final landing images
    landing_eb_highres, landing_ib_highres = take_electron_and_ion_reference_images(
        microscope, hor_field_width=80e-6, image_settings=image_settings,
        save=True, save_label="D_landing_lamella_final_highres")

    storage.step_counter += 1

    # weld to lamella to landing post
    weld_to_landing_post(microscope, confirm=True)

    # calculate cut off position
    landing_eb_highres3, landing_ib_highres3 = take_electron_and_ion_reference_images(
        microscope, hor_field_width=100e-6, image_settings=image_settings,
        save=True, save_label="E_landing_lamella_final_weld")

    # calculate shift from needle top to centre of image
    x_shift, y_shift = calculate_shift_between_features_in_metres(needle_eb_highres_with_lamella_shifted, "needle_tip_to_image_centre")


    height = storage.settings["cut"]["height"]
    width = storage.settings["cut"]["width"]
    depth = storage.settings["cut"]["depth"]
    rotation = storage.settings["cut"]["rotation"]
    hfw = storage.settings["cut"]["hfw"]

    cut_coord = {"center_x": x_shift,
                 "center_y": y_shift,
                 "width": 8e-6,
                 "height": 0.5e-6,
                 "depth": 4e-6, # TODO: might need more to get through needle
                 "rotation": 0, "hfw": 100e-6}  # TODO: check rotation

    # cut off needle tip
    cut_off_needle(microscope, cut_coord=cut_coord, confirm=True)
    landing_eb_highres3, landing_ib_highres3 = take_electron_and_ion_reference_images(
        microscope, hor_field_width=80e-6, image_settings=image_settings,
        save=True, save_label="F_landing_lamella_final_cut")
    landing_eb_lowres, landing_ib_lowres = take_electron_and_ion_reference_images(
        microscope, hor_field_width=150e-6, image_settings=image_settings,
        save=True, save_label="F_landing_lamella_final_cut_lowres")

    # move needle out of trench slowly at first
    for i in range(3):
        print("Moving needle out by 10um")
        z_move_out_from_trench = z_corrected_needle_movement(10e-6, stage_tilt)
        needle.relative_move(z_move_out_from_trench)
        time.sleep(1)

    # retract needle from landing position
    retract_needle(microscope, park_position)

    # take final reference images
    landing_eb_final_highres, landing_ib_final_highres = take_electron_and_ion_reference_images(
        microscope, hor_field_width=80e-6, image_settings=image_settings,
        save=True, save_label="G_landing_lamella_final_needle_cut_highres")
    landing_eb_final_lowres, landing_ib__final_lowres = take_electron_and_ion_reference_images(
        microscope, hor_field_width=150e-6, image_settings=image_settings,
        save=True, save_label="G_landing_lamella_final_needle_cut_lowres")

    storage.step_counter += 1


def thinning_lamella(microscope):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition, MoveSettings, StagePosition
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # move to landing coord
    microscope.specimen.stage.absolute_move(landing_coord)

    # tilt to 0 rotate 180 move to 21 deg

    # tilt to zero, to prevent hitting anything
    stage_settings = MoveSettings(rotate_compucentric=True)
    microscope.specimen.stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)

    # thinning position
    thinning_rotation_angle = 180
    thinning_tilt_angle = 21

    # rotate to thinning angle
    microscope.specimen.stage.relative_move(StagePosition(r=np.deg2rad(thinning_rotation_angle)), stage_settings)

    # tilt to thinning angle
    microscope.specimen.stage.absolute_move(StagePosition(t=np.deg2rad(thinning_tilt_angle)), stage_settings)


    # lamella images
    resolution = storage.settings["imaging"]["resolution"]
    dwell_time = storage.settings["imaging"]["dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
    hfw_lowres = 100e-6#storage.settings["imaging"]["horizontal_field_width"]

    lamella_eb, lamella_ib = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hfw_lowres, image_settings=image_settings,
        save=True, save_label="A_thinning_lamella_21deg_tilt")

    # realign lamella to image centre
    image_settings_ML = GrabFrameSettings(resolution="1536x1024", dwell_time=0.5e-6)  # TODO: user input resolution
    realign_eucentric_with_machine_learning(microscope, image_settings=image_settings_ML, hor_field_width=100e-6)

    # LAMELLA EDGE TO IMAGE CENTRE?
    # x-movement
    storage.step_counter += 1
    lamella_eb, lamella_ib = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6,
                                                                    image_settings=image_settings,
                                                                    save=True, save_label="A_lamella_pre_thinning")


    x_shift, y_shift = calculate_shift_between_features_in_metres(lamella_ib, "lamella_edge_to_landing_post")

    # z-movement (shouldnt really be needed if eucentric calibration is correct)
    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    z_distance = y_shift / np.sin(np.deg2rad(52))
    z_move = z_corrected_stage_movement(z_distance, stage_tilt)
    print('z_move = ', z_move)
    stage.relative_move(z_move)

    # x-move the rest of the way
    x_move = x_corrected_stage_movement(-x_shift)
    print('x_move = ', x_move)

    stage.relative_move(x_move)

    width = settings["thin_lamella"]["lamella_width"]
    x_move_half_width = x_corrected_stage_movement(-width / 2)
    stage.relative_move(x_move_half_width)

    # lamella edge needs to be centred in image...
    # mill thin lamella pattern
    mill_thin_lamella(microscope, settings, confirm=True)

    # take reference images after cleaning
    storage.step_counter += 1
    lamella_eb, lamella_ib = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6,
                                                                    image_settings=image_settings,
                                                                    save=True, save_label="A_lamella_post_thinning")

    # cleaning finished
    print("Trim Lamella Finished.")

    # # calculate thinning patterns
    # height = storage.settings["thin"]["height"]
    # width = storage.settings["thin"]["width"]
    # depth = storage.settings["thin"]["depth"]
    # rotation = storage.settings["thin"]["rotation"]
    # hfw = storage.settings["thin"]["hfw"]

    # # # TODO: draw mill pattern
    # # TODO: use width / weight to adjust centre not magic numbers
    # thinning_coord_top = {
    #     "center_x": -thinning_shifts[0][0] + width / 2,
    #     "center_y": thinning_shifts[0][1] + height / 2,
    #     "width": width,
    #     "height": height,
    #     "depth": depth,
    #     "rotation": rotation,
    #     "hfw": hfw
    # }

    # thinning_coord_bottom = {
    #     "center_x": -thinning_shifts[1][0] + width / 2,
    #     "center_y": thinning_shifts[1][1] - height / 2,
    #     "width": width,
    #     "height": height,
    #     "depth": depth,
    #     "rotation": rotation,
    #     "hfw": hfw
    # }
    # microscope.patterning.clear_patterns()
    # create_thinning_lamella_patterns(microscope, thinning_coord_top, thinning_coord_bottom)

    # # TODO: this actually needs to be the shift from the img centre (needs to happen for top and bottom...)


# def create_thinning_lamella_patterns(microscope, thinning_coord_top, thinning_coord_bottom):
#     thinning_lamella_patterns = []
#     for thinning_coord in [thinning_coord_top, thinning_coord_bottom]:
#         pattern = _create_mill_pattern(
#             microscope,
#             center_x=thinning_coord["center_x"],
#             center_y=thinning_coord["center_y"],
#             width=thinning_coord["width"],
#             height=thinning_coord["height"],
#             depth=thinning_coord["depth"],
#             rotation_degrees=thinning_coord["rotation"],
#             ion_beam_field_of_view=thinning_coord["hfw"],
#         )
#         thinning_lamella_patterns.append(pattern)

#     return thinning_lamella_patterns



def sharpen_needle(microscope):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # def Rotate(x, y, angle):
    #     angle = np.deg2rad(angle)
    #     x_rot = x * math.cos(angle) - y * math.sin(angle)
    #     y_rot = x * math.sin(angle) + y * math.cos(angle)
    #     return x_rot, y_rot

    move_sample_stage_out(microscope)

    park_position = insert_needle(microscope)
    stage_tilt = stage.current_position.t
    print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...')
    z_move_in = z_corrected_needle_movement(-180e-6, stage_tilt)
    needle.relative_move(z_move_in)

    # needle images
    resolution = storage.settings["imaging"]["resolution"]
    dwell_time = storage.settings["imaging"]["dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
    hfw_lowres = storage.settings["imaging"]["horizontal_field_width"]

    needle_eb, needle_ib = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hfw_lowres, image_settings=image_settings,
        save=True, save_label="A_sharpen_needle_initial")

    # move needle to the centre
    x_0, y_0 = calculate_shift_between_features_in_metres(needle_ib, "needle_tip_to_image_centre")

    x_move = x_corrected_needle_movement(x_0)
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
    hfw_lowres = storage.settings["imaging"]["horizontal_field_width"]

    needle_eb, needle_ib = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hfw_lowres, image_settings=image_settings,
        save=True, save_label="A_sharpen_needle_centre")

    x_0, y_0 = calculate_shift_between_features_in_metres(needle_ib, "needle_tip_to_image_centre")

    # sharpening parameters
    height = storage.settings["sharpen"]["height"]
    width = storage.settings["sharpen"]["width"]
    depth = storage.settings["sharpen"]["depth"]
    bias = storage.settings["sharpen"]["bias"]
    hfw = storage.settings["sharpen"]["hfw"]
    tip_angle = storage.settings["sharpen"]["tip_angle"]  # 2NA of the needle   2*alpha
    needle_angle = storage.settings["sharpen"][
        "needle_angle"
    ]  # needle tilt on the screen 45 deg +/-
    milling_current = storage.settings["sharpen"]["sharpen_milling_current"]

    # create sharpening patterns
    cut_coord_bottom, cut_coord_top = calculate_sharpen_needle_pattern(x_0=x_0, y_0=y_0)

    sharpen_patterns = create_sharpen_needle_patterns(
        microscope, cut_coord_bottom, cut_coord_top
    )

    # run sharpening milling
    confirm_and_run_milling(microscope, milling_current, confirm=True)

    needle_eb, needle_ib = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hfw_lowres, image_settings=image_settings,
        save=True, save_label="A_sharpen_needle_sharp")

    storage.step_counter += 1

    retract_needle(microscope, park_position)


def calculate_sharpen_needle_pattern(x_0, y_0):

    height = storage.settings["sharpen"]["height"]
    width = storage.settings["sharpen"]["width"]
    depth = storage.settings["sharpen"]["depth"]
    bias = storage.settings["sharpen"]["bias"]
    hfw = storage.settings["sharpen"]["hfw"]
    tip_angle = storage.settings["sharpen"]["tip_angle"]  # 2NA of the needle   2*alpha
    needle_angle = storage.settings["sharpen"][
        "needle_angle"
    ]  # needle tilt on the screen 45 deg +/-
    milling_current = storage.settings["sharpen"]["sharpen_milling_current"]

    alpha = tip_angle / 2  # half of NA of the needletip
    beta = np.rad2deg(
        np.arctan(width / height)
    )  # box's width and length, beta is the diagonal angle
    D = np.sqrt(width ** 2 + height ** 2) / 2  # half of box diagonal
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
    dx_1 = (width / 2) * math.cos(np.deg2rad(needle_angle + alpha))
    dy_1 = (width / 2) * math.sin(np.deg2rad(needle_angle + alpha))
    ddx_1 = (height / 2) * math.sin(np.deg2rad(needle_angle + alpha))
    ddy_1 = (height / 2) * math.cos(np.deg2rad(needle_angle + alpha))
    x_1 = x_0 - dx_1 + ddx_1  # centre of the bottom box
    y_1 = y_0 - dy_1 - ddy_1  # centre of the bottom box

    dx_2 = D * math.cos(np.deg2rad(needle_angle - alpha))
    dy_2 = D * math.sin(np.deg2rad(needle_angle - alpha))
    ddx_2 = (height / 2) * math.sin(np.deg2rad(needle_angle - alpha))
    ddy_2 = (height / 2) * math.cos(np.deg2rad(needle_angle - alpha))
    x_2 = x_0 - dx_2 - ddx_2  # centre of the top box
    y_2 = y_0 - dy_2 + ddy_2  # centre of the top box

    print("needletip xshift offcentre: ", x_0, "; needletip yshift offcentre: ", y_0)
    print("width: ", width)
    print("height: ", height)
    print("depth: ", depth)
    print("needle_angle: ", needle_angle)
    print("tip_angle: ", tip_angle)
    print("rotation1 :", rotation_1)
    print("rotation2 :", rotation_2)
    print("=================================================")
    print("centre of bottom box: x1 = ", x_1, "; y1 = ", y_1)
    print("centre of top box:    x2 = ", x_2, "; y2 = ", y_2)
    print("=================================================")

    # pattern = microscope.patterning.create_rectangle(x_3, y_3, width+2*bias, height+2*bias, depth)
    # pattern.rotation = np.deg2rad(rotation_1)
    # pattern = microscope.patterning.create_rectangle(x_4, y_4, width+2*bias, height+2*bias, depth)
    # pattern.rotation = np.deg2rad(rotation_2)

    # bottom cut pattern
    cut_coord_bottom = {
        "center_x": x_1,
        "center_y": y_1,
        "width": width,
        "height": height - bias,
        "depth": depth,
        "rotation": rotation_1,
        "hfw": hfw,
    }

    # top cut pattern
    cut_coord_top = {
        "center_x": x_2,
        "center_y": y_2,
        "width": width,
        "height": height - bias,
        "depth": depth,
        "rotation": rotation_2,
        "hfw": hfw,
    }

    # setup ion milling
    setup_ion_milling(microscope, ion_beam_field_of_view=hfw, patterning_mode="Serial")

    return cut_coord_bottom, cut_coord_top


def create_sharpen_needle_patterns(microscope, cut_coord_bottom, cut_coord_top):
    sharpen_patterns = []
    for cut_coord in [cut_coord_bottom, cut_coord_top]:
        pattern = _create_mill_pattern(
            microscope,
            center_x=cut_coord["center_x"],
            center_y=cut_coord["center_y"],
            width=cut_coord["width"],
            height=cut_coord["height"],
            depth=cut_coord["depth"],
            rotation_degrees=cut_coord["rotation"],
            ion_beam_field_of_view=cut_coord["hfw"],
        )
        sharpen_patterns.append(pattern)

    return sharpen_patterns




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
        tilt_adjustment = np.deg2rad(-PRETILT_DEGREES)
    elif beam_type == BeamType.ION:
        tilt_adjustment = np.deg2rad(52 - PRETILT_DEGREES)
    tilt_radians = stage_tilt + tilt_adjustment
    y_move = +np.cos(tilt_radians) * expected_y
    z_move = -np.sin(tilt_radians) * expected_y
    print(' ------------  drift correction ---------------  ')
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
    x = size[0]
    y = size[1]
    img = Image.new('I', size)
    draw = ImageDraw.Draw(img)
    draw.ellipse((x / 2 - radius, y / 2 - radius, x / 2 + radius, y / 2 + radius), fill='white', outline='white')
    tmp = np.array(img, float) / 255
    if sigma > 0:
        mask = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        mask = tmp
    return mask


def ellipse_mask(size=(128, 128), radius1=32, radius2=32, sigma=3):
    x = size[0]
    y = size[1]
    img = Image.new('I', size)
    draw = ImageDraw.Draw(img)
    draw.ellipse((x / 2 - radius1, y / 2 - radius2, x / 2 + radius1,
                  y / 2 + radius2), fill='white', outline='white')
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


def shift_from_crosscorrelation_simple_images(img1, img2, lowpass=256, highpass=22, sigma=2):
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
    x_shift, y_shift = shift_from_crosscorrelation_AdornedImages(
        eb_image, ib_image_rotated, lowpass=lowpass, highpass=highpass, sigma=sigma)
    return x_shift, y_shift


def correct_stage_drift_using_reference_eb_images(microscope, reference_images_low_and_high_res, plot=False):
    print('stage shift correction by image cross-correlation')
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    # Unpack reference images
    eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference = reference_images_low_and_high_res
    storage.SaveImage(eb_lowres_reference,  id='A_ref_eb_lowres')
    storage.SaveImage(eb_highres_reference, id='A_ref_eb_highres')
    storage.SaveImage(ib_lowres_reference,  id='A_ref_ib_lowres')
    storage.SaveImage(ib_highres_reference, id='A_ref_ib_highres')
    ##############
    stage = microscope.specimen.stage
    # TODO: user input resolution, must match
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    ####
    pixelsize_x_lowres = ib_lowres_reference.metadata.binary_result.pixel_size.x
    field_width_lowres = pixelsize_x_lowres * ib_lowres_reference.width
    pixelsize_x_highres = ib_highres_reference.metadata.binary_result.pixel_size.x
    field_width_highres = pixelsize_x_highres * ib_highres_reference.width
    ########################################  LOW resolution alignment  #############################################
    print('- - - - - - - - - - - - - - Coarse alignment- - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_lowres
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
    lowpass_pixels = int(max(new_eb_lowres.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int(max(new_eb_lowres.data.shape) / 256)  # =6 @ 1536x1024, good for e-beam images
    sigma = int(2 * max(new_eb_lowres.data.shape)/1536)       # =2 @ 1536x1024, good for e-beam images
    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(
        new_eb_lowres, eb_lowres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t,
                                         beam_type=BeamType.ELECTRON)  # check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    #yy = input('press Enter to move...')
    stage.relative_move(x_move)
    stage.relative_move(yz_move)
    ###
    ########################################  HIGH resolution alignment  #############################################
    print('- - - - - - - - - - - - - - Finer alignment- - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_highres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    new_eb_highres = microscope.imaging.grab_frame(image_settings)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    new_ib_highres = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(new_eb_highres,  id='C_sample_eb_highres_shifted')
    storage.SaveImage(new_ib_highres,  id='C_sample_ib_highres_shifted')
    # ------------- correlate------------
    lowpass_pixels = int(max(new_eb_highres.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int(max(new_eb_highres.data.shape) / 256)  # =6 @ 1536x1024, good for e-beam images
    sigma = int(2 * max(new_eb_highres.data.shape)/1536)        # =2 @ 1536x1024, good for e-beam images
    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(
        new_eb_highres, eb_highres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t,
                                         beam_type=BeamType.ELECTRON)  # check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    #yy = input('press Enter to move...')
    stage.relative_move(x_move)
    stage.relative_move(yz_move)
    storage.step_counter += 1


def realign_using_reference_eb_and_ib_images(microscope, reference_images_low_and_high_res, plot=False):
    print('stage shift correction by image cross-correlation : using only eBeam images')
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    # Unpack reference images
    eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference = reference_images_low_and_high_res
    storage.SaveImage(eb_lowres_reference,  id='A_ebTOib_ref_eb_lowres')
    storage.SaveImage(eb_highres_reference, id='A_ebTOib_ref_eb_highres')
    storage.SaveImage(ib_lowres_reference,  id='A_ebTOib_ref_ib_lowres')
    storage.SaveImage(ib_highres_reference, id='A_ebTOib_ref_ib_highres')
    #
    stage = microscope.specimen.stage
    # TODO: user input resolution, must match
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    ####
    pixelsize_x_lowres = eb_lowres_reference.metadata.binary_result.pixel_size.x
    field_width_lowres = pixelsize_x_lowres * eb_lowres_reference.width
    pixelsize_x_highres = eb_highres_reference.metadata.binary_result.pixel_size.x
    field_width_highres = pixelsize_x_highres * eb_highres_reference.width
    ########################################  LOW resolution alignment #1  #############################################
    print(' - - - - - - - - - - - - - - Coarse alignment #1 - - - - - - - - - - - - - - ...')
    # refocus_and_relink(microscope)
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_lowres
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
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
    eb_lowres = new_electron_image(microscope, settings=image_settings,
                                   brightness=eb_brightness, contrast=eb_contrast)
    ib_lowres = new_ion_image(microscope, settings=image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.SaveImage(eb_lowres, id='B_01_ebTOib_sample_eb_lowres_BC')
    storage.SaveImage(ib_lowres, id='B_01_ebTOib_sample_ib_lowres_BC')

    #
    lowpass_pixels = int(max(new_eb_lowres.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int(max(new_eb_lowres.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
    sigma = int(2 * max(new_eb_lowres.data.shape)/1536)        # =2   @ 1536x1024, good for e-beam images
    dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(
        new_eb_lowres, ib_lowres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    #
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t,
                                         beam_type=BeamType.ELECTRON)  # check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    #yy = input('press Enter to move...')
    stage.relative_move(x_move)
    stage.relative_move(yz_move)

    ########################################  LOW resolution alignment #2  #############################################
    print(' - - - - - - - - - - - - - - Coarse alignment # 2 - - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_lowres
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
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
    eb_lowres = new_electron_image(microscope, settings=image_settings,
                                   brightness=eb_brightness, contrast=eb_contrast)
    ib_lowres = new_ion_image(microscope, settings=image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.SaveImage(eb_lowres, id='B_02_ebTOib_sample_eb_lowres_BC')
    storage.SaveImage(ib_lowres, id='B_02_ebTOib_sample_ib_lowres_BC')

    #
    lowpass_pixels = int(max(new_eb_lowres.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int(max(new_eb_lowres.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
    sigma = int(2 * max(new_eb_lowres.data.shape)/1536)        # =2   @ 1536x1024, good for e-beam images
    dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(
        new_eb_lowres, ib_lowres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    #
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t,
                                         beam_type=BeamType.ELECTRON)  # check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    #yy = input('Press Enter to move...')
    stage.relative_move(x_move)
    stage.relative_move(yz_move)

   ########################################  HIGH resolution alignment  #############################################
    print(' - - - - - - - - - - - - - - Finer alignment - - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_highres
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
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_highres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
    eb_highres = new_electron_image(microscope, settings=image_settings,
                                    brightness=eb_brightness, contrast=eb_contrast)
    ib_highres = new_ion_image(microscope, settings=image_settings, brightness=ib_brightness, contrast=ib_contrast)
    storage.SaveImage(eb_highres, id='C_ebTOib_sample_eb_highres_shifted_BC')
    storage.SaveImage(ib_highres, id='C_ebTOib_sample_ib_highres_shifted_BC')

    lowpass_pixels = int(max(new_eb_highres.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int(max(new_eb_highres.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
    sigma = int(2 * max(new_eb_highres.data.shape) / 1536)       # =2   @ 1536x1024, good for e-beam images
    dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(
        new_eb_highres, ib_highres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t,
                                         beam_type=BeamType.ELECTRON)  # check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    #yy = input('Press Enter to move...')
    stage.relative_move(x_move)
    stage.relative_move(yz_move)
    storage.step_counter += 1


# def realign_at_different_stage_tilts(microscope, reference_images_low_and_high_res, previous_stage_tilt, beam_type=BeamType.ION):
#     print('stage shift correction by image cross-correlation : different stage/image tilts')
#     from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
#     from autoscript_sdb_microscope_client.structures import StagePosition
#     stage = microscope.specimen.stage
#     # Unpack reference images
#     eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference = reference_images_low_and_high_res
#     storage.SaveImage(eb_lowres_reference,  id='A_tiltAlign_ref_eb_lowres')
#     storage.SaveImage(eb_highres_reference, id='A_tiltAlign_ref_eb_highres')
#     storage.SaveImage(ib_lowres_reference,  id='A_tiltAlign_ref_ib_lowres')
#     storage.SaveImage(ib_highres_reference, id='A_tiltAlign_ref_ib_highres')
#     pixelsize_x_lowres = eb_lowres_reference.metadata.binary_result.pixel_size.x
#     pixelsize_y_lowres = eb_lowres_reference.metadata.binary_result.pixel_size.y
#     field_width_lowres = pixelsize_x_lowres * eb_lowres_reference.width
#     pixelsize_x_highres = eb_highres_reference.metadata.binary_result.pixel_size.x
#     pixelsize_y_highres = eb_highres_reference.metadata.binary_result.pixel_size.y
#     field_width_highres = pixelsize_x_highres * eb_highres_reference.width
#     height, width = eb_lowres_reference.data.shape
#     eb_lowres_reference_norm = (eb_lowres_reference.data -
#                                 np.mean(eb_lowres_reference.data)) / np.std(eb_lowres_reference.data)
#     eb_highres_reference_norm = (eb_highres_reference.data -
#                                  np.mean(eb_highres_reference.data)) / np.std(eb_highres_reference.data)
#     ib_lowres_reference_norm = (ib_lowres_reference.data -
#                                 np.mean(ib_lowres_reference.data)) / np.std(ib_lowres_reference.data)
#     ib_highres_reference_norm = (ib_highres_reference.data -
#                                  np.mean(ib_highres_reference.data)) / np.std(ib_highres_reference.data)
#     # current_stage_tilt  = stage.current_position.t
#     # current_image_tilt  = PRETILT_DEGREES + current_stage_tilt
#     # previous_image_tilt = PRETILT_DEGREES + previous_stage_tilt
#     # if beam_type==BeamType.ION:
#     #     previous_image_tilt_from_ion_flat = 52 - np.red2deg(previous_image_tilt)
#     #     current_image_tilt_from_ion_flat  = 52 - np.red2deg(current_image_tilt)
#     #     if abs(previous_image_tilt_from_ion_flat) > abs(current_image_tilt_from_ion_flat):
#     #         print('Previous image was tilted more, stretching it for alignment..')
#     #         delta_angle = abs(previous_image_tilt_from_ion_flat) - abs(current_image_tilt_from_ion_flat)
#     #         stretch_image= 1
#     #         stretch =  1. / math.cos( np.deg2rad(delta_angle) )
#     #     else:
#     #         print('Current image is tilted more, stretching it for aligment..')
#     #         delta_angle = abs(current_image_tilt_from_ion_flat) - abs(previous_image_tilt_from_ion_flat)
#     #         stretch_image = 2
#     #         stretch = 1. / math.cos(np.deg2rad(delta_angle))
#     ####
#     # TODO: user input resolution, must match
#     image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
#     ####
#     ########################################  LOW resolution alignment  #############################################
#     print(' - - - - - - - - - - - - - - Coarse alignment - - - - - - - - - - - - - - ...')
#     new_eb_lowres,  new_ib_lowres = take_electron_and_ion_reference_images(
#         microscope, hor_field_width=field_width_lowres, image_settings=image_settings)
#     storage.SaveImage(new_eb_lowres,  id='B_tiltAlign_sample_eb_lowres')
#     storage.SaveImage(new_ib_lowres,  id='B_tiltAlign_sample_ib_lowres')
#     new_eb_lowres_norm = (new_eb_lowres.data - np.mean(new_eb_lowres.data)) / np.std(new_eb_lowres.data)
#     new_ib_lowres_norm = (new_ib_lowres.data - np.mean(new_ib_lowres.data)) / np.std(new_ib_lowres.data)
#     ###
#     # circular mask, align only the central areas
#     cmask = circ_mask(size=(width, height), radius=height // 3 - 15, sigma=10)
#     ###
#     if beam_type == BeamType.ELECTRON:
#         lowpass_pixels = int(max(new_eb_lowres.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
#         highpass_pixels = int(max(new_eb_lowres.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
#         sigma = int(2 * max(new_eb_lowres.data.shape)/1536)        # =2   @ 1536x1024, good for e-beam images
#         dx_pixels, dy_pixels = shift_from_crosscorrelation_simple_images(new_eb_lowres_norm * cmask, eb_lowres_reference_norm * cmask, lowpass=lowpass_pixels,
#                                                                          highpass=highpass_pixels, sigma=sigma)
#         dx_meters = dx_pixels * pixelsize_x_lowres
#         dy_meters = dy_pixels * pixelsize_y_lowres
#         x_move = x_corrected_stage_movement(-dx_meters)
#         yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t,
#                                              beam_type=BeamType.ELECTRON)  # check electron/ion movement
#     if beam_type == BeamType.ION:
#         lowpass_pixels = int(max(new_ib_lowres.data.shape) / 6)  # =256 @ 1536x1024,  good for i-beam images
#         # =24  @ 1536x1024, good for i-beam images => need a large highpass to remove noise and ringing
#         highpass_pixels = int(max(new_ib_lowres.data.shape) / 64)
#         sigma = int(10 * max(new_ib_lowres.data.shape) / 1536)  # =10 @ 1536x1024,  good for i-beam images
#         dx_pixels, dy_pixels = shift_from_crosscorrelation_simple_images(new_ib_lowres_norm * cmask, ib_lowres_reference_norm * cmask, lowpass=lowpass_pixels,
#                                                                          highpass=highpass_pixels, sigma=sigma)
#         dx_meters = dx_pixels * pixelsize_x_lowres
#         dy_meters = dy_pixels * pixelsize_y_lowres
#         x_move = x_corrected_stage_movement(-dx_meters)
#         yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t,
#                                              beam_type=BeamType.ION)  # check electron/ion movement
#     print('relative movement of the the stage by X  :',  x_move)
#     print('relative movement of the the stage by Y-Z:', yz_move)
#     stage.relative_move(x_move)
#     stage.relative_move(yz_move)

#    ########################################  HIGH resolution alignment  #############################################
#     print(' - - - - - - - - - - - - - - Finer alignment - - - - - - - - - - - - - - ...')
#     new_eb_highres,  new_ib_highres = take_electron_and_ion_reference_images(
#         microscope, hor_field_width=field_width_highres, image_settings=image_settings)
#     storage.SaveImage(new_eb_highres,  id='C_tiltAlign_sample_eb_highres')
#     storage.SaveImage(new_ib_highres,  id='C_tiltAlign_sample_ib_highres')
#     new_eb_highres_norm = (new_eb_highres.data - np.mean(new_eb_highres.data)) / np.std(new_eb_highres.data)
#     new_ib_highres_norm = (new_ib_highres.data - np.mean(new_ib_highres.data)) / np.std(new_ib_highres.data)
#     ###
#     if beam_type == BeamType.ELECTRON:
#         lowpass_pixels = int(max(new_eb_highres.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
#         highpass_pixels = int(max(new_eb_highres.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
#         sigma = int(2 * max(new_eb_highres.data.shape)/1536)        # =2   @ 1536x1024, good for e-beam images
#         #dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(new_eb_lowres, ib_lowres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
#         dx_pixels, dy_pixels = shift_from_crosscorrelation_simple_images(new_eb_highres_norm * cmask, eb_highres_reference_norm * cmask, lowpass=lowpass_pixels,
#                                                                          highpass=highpass_pixels, sigma=sigma_eb)
#         dx_meters = dx_pixels * pixelsize_x_highres
#         dy_meters = dy_pixels * pixelsize_y_highres
#         x_move = x_corrected_stage_movement(-dx_meters)
#         yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t,
#                                              beam_type=BeamType.ELECTRON)  # check electron/ion movement
#     if beam_type == BeamType.ION:
#         lowpass_pixels = int(max(new_ib_highres.data.shape) / 6)    # =256 @ 1536x1024,  good for i-beam images
#         # =24  @ 1536x1024,  good for i-beam images => need a large highpass to remove noise and ringing
#         highpass_pixels = int(max(new_ib_highres.data.shape) / 64)
#         sigma = int(10 * max(new_ib_highres.data.shape)/1536)        # =10   @ 1536x1024, good for i-beam images
#         dx_pixels, dy_pixels = shift_from_crosscorrelation_simple_images(new_ib_highres_norm * cmask, ib_highres_reference_norm * cmask, lowpass=lowpass_pixels,
#                                                                          highpass=highpass_pixels, sigma=sigma)
#         dx_meters = dx_pixels * pixelsize_x_highres
#         dy_meters = dy_pixels * pixelsize_y_highres
#         x_move = x_corrected_stage_movement(-dx_meters)
#         yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t,
#                                              beam_type=BeamType.ION)  # check electron/ion movement
#     print('relative movement of the the stage by X  :',  x_move)
#     print('relative movement of the the stage by Y-Z:', yz_move)
#     stage.relative_move(x_move)
#     stage.relative_move(yz_move)

#     new_eb_highres, new_ib_highres = take_electron_and_ion_reference_images(
#         microscope,  hor_field_width=field_width_highres, image_settings=image_settings)
#     storage.SaveImage(new_eb_highres, id='D_tiltAlign_sample_eb_highres_aligned')
#     storage.SaveImage(new_ib_highres, id='D_tiltAlign_sample_ib_highres_aligned')
#     storage.step_counter += 1

def realign_landing_post(microscope, reference_images_low_and_high_res, plot=False):
    print('stage shift correction by image cross-correlation')
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    # Unpack reference images
    eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference = reference_images_low_and_high_res
    storage.SaveImage(eb_lowres_reference,  id='A_ref_eb_lowres')
    storage.SaveImage(eb_highres_reference, id='A_ref_eb_highres')
    storage.SaveImage(ib_lowres_reference,  id='A_ref_ib_lowres')
    storage.SaveImage(ib_highres_reference, id='A_ref_ib_highres')
    ##############
    stage = microscope.specimen.stage
    # TODO: user input resolution, must match
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    ####
    pixelsize_x_lowres = ib_lowres_reference.metadata.binary_result.pixel_size.x
    field_width_lowres = pixelsize_x_lowres * ib_lowres_reference.width
    pixelsize_x_highres = ib_highres_reference.metadata.binary_result.pixel_size.x
    field_width_highres = pixelsize_x_highres * ib_highres_reference.width
    ########################################  LOW resolution alignment  #############################################
    print('- - - - - - - - - - - - - - Coarse alignment- - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_lowres
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
    # =24  @ 1536x1024,  good for i-beam images => need a large highpass to remove noise and ringing
    highpass_pixels = int(max(new_ib_lowres.data.shape) / 64)
    sigma = int(10 * max(new_ib_lowres.data.shape) / 1536)    # =10  @ 1536x1024, good for i-beam images
    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(
        new_ib_lowres, ib_lowres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t,
                                         beam_type=BeamType.ION)  # check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    stage.relative_move(x_move)
    stage.relative_move(yz_move)
    ###
    ########################################  HIGH resolution alignment  #############################################
    print('- - - - - - - - - - - - - - Finer alignment- - - - - - - - - - - - - - ...')
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_highres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    new_eb_highres = microscope.imaging.grab_frame(image_settings)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    new_ib_highres = microscope.imaging.grab_frame(image_settings)
    storage.SaveImage(new_eb_highres,  id='C_sample_eb_highres_shifted')
    storage.SaveImage(new_ib_highres,  id='C_sample_ib_highres_shifted')
    # ------------- correlate------------
    lowpass_pixels = int(max(new_ib_highres.data.shape) / 6)   # =256 @ 1536x1024,  good for i-beam images
    # =24  @ 1536x1024,  good for i-beam images => need a large highpass to remove noise and ringing
    highpass_pixels = int(max(new_ib_highres.data.shape) / 64)
    sigma = int(10 * max(new_ib_highres.data.shape) / 1536)    # =10  @ 1536x1024, good for i-beam images
    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(
        new_ib_highres, ib_highres_reference, lowpass=lowpass_pixels, highpass=highpass_pixels, sigma=sigma)
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t,
                                         beam_type=BeamType.ION)  # check electron/ion movement
    print('relative movement of the the stage by X  :',  x_move)
    print('relative movement of the the stage by Y-Z:', yz_move)
    stage.relative_move(x_move)
    stage.relative_move(yz_move)
    storage.step_counter += 1


###########################################################################################################################################################################


def single_liftout(microscope, settings, landing_coord, lamella_coord, original_landing_images, original_lamella_area_images):

    # move to the previously stored position and correct the position using the reference images:
    microscope.specimen.stage.absolute_move(lamella_coord)
    correct_stage_drift_using_reference_eb_images(microscope, original_lamella_area_images, plot=False)

    # mill
    mill_lamella(microscope, settings, confirm=False)

    # lift-out
    liftout_lamella(microscope, settings)

    # land
    land_lamella(microscope, landing_coord, original_landing_images)

    # resharpen needle
    sharpen_needle(microscope)

    # thinning lamella
    thinning_lamella(microscope)


def reload_config(config_filename):
    settings = load_config(config_filename)
    storage.settings = settings


# @click.command()
# @click.argument("config_filename")
# def main_cli(config_filename):
#     """Run the main command line interface.
#     Parameters
#     ----------
#     config_filename : str
#         Path to protocol file with input parameters given in YAML (.yml) format
#     """
#     settings = load_config(config_filename)
#     timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
#         '%Y%m%d.%H%M%S')  # datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
#     output_log_filename = os.path.join('logfile' + timestamp + '.log')
#     configure_logging(log_filename=output_log_filename)
#     main(settings)


# def main(settings):
#     storage.NewRun()
#     microscope = initialize(settings["system"]["ip_address"])
#     autocontrast(microscope, beam_type=BeamType.ELECTRON)
#     autocontrast(microscope, beam_type=BeamType.ION)
#     eb = new_electron_image()
#     if ask_user("Do you want to sputter the whole sample grid with platinum? yes/no: "):
#         sputter_platinum_over_whole_grid(microscope)
#     print("Please select the landing positions and check eucentric height manually.")
#     landing_coordinates, original_landing_images = find_coordinates(
#         microscope, name="landing position", move_stage_angle="landing")
#     lamella_coordinates, original_trench_images = find_coordinates(
#         microscope, name="lamella",          move_stage_angle="trench")
#     zipped_coordinates = list(zip(lamella_coordinates, landing_coordinates))
#     storage.LANDING_POSTS_POS_REF = original_landing_images
#     storage.LAMELLA_POS_REF = original_trench_images
#     # Start liftout for each lamella
#     for i, (lamella_coord, landing_coord) in enumerate(zipped_coordinates):
#         landing_reference_images = original_landing_images[i]
#         lamella_area_reference_images = original_trench_images[i]
#         single_liftout(microscope, settings, landing_coord, lamella_coord,
#                        landing_reference_images, lamella_area_reference_images)
#         storage.liftout_counter += 1

#     print("Finished.")



class AutoLiftoutStatus(Enum):
    Initialize = 0
    Setup = 1
    Milling = 2
    Liftout = 3
    Landing = 4
    Reset = 5
    Cleanup = 6
    Finished = 7

# TODO: logging and storage should be consistent and consolidated
class AutoLiftout:

    def __init__(self, config_filename, run_name="run") -> None:

        # initialise autoliftout
        configure_logging("logfile_")

        self.settings = load_config(config_filename)

        self.storage = Storage()
        self.storage.NewRun(prefix=run_name)
        self.storage.settings = self.settings

        self.microscope = initialize(self.settings["system"]["ip_address"])

        self.current_status = AutoLiftoutStatus.Initialize


    def _report_status(self):
        """Helper function for reporting liftout status"""
        print(f"\nCurrent Status: {self.current_status.name}")
        print(f"Liftout Counter: {self.storage.liftout_counter}")

    def setup(self):
        """ Initial setup of grid and selection for lamella and landing positions"""
        self.current_status = AutoLiftoutStatus.Setup
        self._report_status()
        autocontrast(self.microscope, beam_type=BeamType.ELECTRON)
        autocontrast(self.microscope, beam_type=BeamType.ION)

        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
        self.microscope.imaging.set_active_view(1)
        self.microscope.beams.electron_beam.horizontal_field_width.value = 2750e-6
        eb = self.microscope.imaging.grab_frame(image_settings)
        self.storage.SaveImage(eb, id='grid')
        self.storage.step_counter += 1

        if ask_user("Do you want to sputter the whole sample grid with platinum? yes/no: "):
            sputter_platinum_over_whole_grid(microscope)

        autocontrast(self.microscope, beam_type=BeamType.ELECTRON)
        self.microscope.beams.electron_beam.horizontal_field_width.value = 2750e-6
        eb = self.microscope.imaging.grab_frame(image_settings)
        self.storage.SaveImage(eb, id='grid_Pt_deposition')
        self.storage.step_counter += 1

        print("Please select the landing positions and check eucentric height manually.")
        self.landing_coordinates, self.original_landing_images = find_coordinates(self.microscope, name="landing position", move_stage_angle="landing")
        self.lamella_coordinates, self.original_trench_images  = find_coordinates(self.microscope, name="lamella",          move_stage_angle="trench")
        self.zipped_coordinates = list(zip(self.lamella_coordinates, self.landing_coordinates))
        self.storage.LANDING_POSTS_POS_REF = self.original_landing_images
        self.storage.LAMELLA_POS_REF       = self.original_trench_images

    def _get_fake_setup_data(self):
        self.zipped_coordinates = [[1, 1], [1, 1], [1, 1], [1, 1]]
        self.original_landing_images = [[1], [1], [1], [1]]
        self.original_trench_images = [[1], [1], [1], [1]]

    def run_liftout(self):

        self._report_status()
        # self._get_fake_setup_data()

        # Start liftout for each lamella
        for i, (lamella_coord, landing_coord) in enumerate(self.zipped_coordinates):
            landing_reference_images      = self.original_landing_images[i]
            lamella_area_reference_images = self.original_trench_images[i]
            self.single_liftout(self.microscope, self.settings, landing_coord, lamella_coord, landing_reference_images, lamella_area_reference_images)
            self.storage.liftout_counter += 1

        self.current_status = AutoLiftoutStatus.Finished
        self._report_status()

    def single_liftout(self, microscope, settings, landing_coord, lamella_coord, original_landing_images, original_lamella_area_images):

        ### move to the previously stored position and correct the position using the reference images:
        microscope.specimen.stage.absolute_move(lamella_coord)
        correct_stage_drift_using_reference_eb_images(microscope, original_lamella_area_images, plot=False)

        # mill
        self.current_status = AutoLiftoutStatus.Milling
        self._report_status()
        mill_lamella(microscope, settings, confirm=False)

        # lift-out
        self.current_status = AutoLiftoutStatus.Liftout
        self._report_status()
        liftout_lamella(microscope, settings)

        # land
        self.current_status = AutoLiftoutStatus.Landing
        self._report_status()
        land_lamella(microscope, landing_coord, original_landing_images)

        # resharpen needle
        self.current_status = AutoLiftoutStatus.Reset
        self._report_status()
        sharpen_needle(microscope)


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
    config_filename = args[0]

    # oop version
    auto_liftout = AutoLiftout(config_filename)
    global storage
    storage = auto_liftout.storage
    auto_liftout.setup()
    auto_liftout.run_liftout()

    # settings = load_config(config_filename)
    # timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
    #     '%Y%m%d.%H%M%S')  # datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    # output_log_filename = os.path.join('logfile' + timestamp + '.log')
    # configure_logging(log_filename=output_log_filename)

    # storage.NewRun(prefix='thin_lamella_exp')
    # storage.settings = settings

    # microscope = initialize(settings["system"]["ip_address"])
    # autocontrast(microscope, beam_type=BeamType.ELECTRON)
    # autocontrast(microscope, beam_type=BeamType.ION)

    # image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    # microscope.imaging.set_active_view(1)
    # microscope.beams.electron_beam.horizontal_field_width.value = 2750e-6
    # eb = microscope.imaging.grab_frame(image_settings)
    # storage.SaveImage(eb, id='grid')
    # storage.step_counter += 1

    # if ask_user("Do you want to sputter the whole sample grid with platinum? yes/no: "):
    #     sputter_platinum_over_whole_grid(microscope)

    # autocontrast(microscope, beam_type=BeamType.ELECTRON)
    # microscope.beams.electron_beam.horizontal_field_width.value = 2750e-6
    # eb = microscope.imaging.grab_frame(image_settings)
    # storage.SaveImage(eb, id='grid_Pt_deposition')
    # storage.step_counter += 1

    # print("Please select the landing positions and check eucentric height manually.")
    # landing_coordinates, original_landing_images = find_coordinates(
    #     microscope, name="landing position", move_stage_angle="landing")
    # lamella_coordinates, original_trench_images = find_coordinates(
    #     microscope, name="lamella",          move_stage_angle="trench")
    # zipped_coordinates = list(zip(lamella_coordinates, landing_coordinates))
    # storage.LANDING_POSTS_POS_REF = original_landing_images
    # storage.LAMELLA_POS_REF = original_trench_images
    # # Start liftout for each lamella
    # for i, (lamella_coord, landing_coord) in enumerate(zipped_coordinates):
    #     landing_reference_images = original_landing_images[i]
    #     lamella_area_reference_images = original_trench_images[i]
    #     single_liftout(microscope, settings, landing_coord, lamella_coord,
    #                    landing_reference_images, lamella_area_reference_images)
    #     storage.liftout_counter += 1
    # print("Finished.")

