import sys, getopt, glob, os
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import math
import time
import datetime
import scipy.ndimage as ndi
from scipy import fftpack, misc
from PIL import Image, ImageDraw, ImageFilter
from matplotlib.patches import Circle
from skimage.transform import resize



class BeamType(Enum):
    ION = 'ION'
    ELECTRON = 'ELECTRON'

PRETILT_DEGREES = 27

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
    microscope.auto_functions.run_auto_cb()
    return autocontrast_settings


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

class BeamType(Enum):
    ION = 'ION'
    ELECTRON = 'ELECTRON'





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
    return stage.current_position(t=tilt), stage_settings)
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
    sample_grid_center = StagePosition(x=-0.0025868173,
                                       y=0.0031794167,
                                       z=0.0039457213)
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

############################################################################################

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

def plot_overlaid_images(image_1, image_2, show=True, rotate_second_image=True):
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
        if rotate_second_image:
            image_2 = np.rot90(np.rot90(image_2))
    # Axes shown in pixels, not in real space.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(image_1, cmap='Blues_r', alpha=1)
    ax.imshow(image_2, cmap='Oranges_r', alpha=0.5)
    if show is True:
        fig.show()
    return fig, ax





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


############################################################################################


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
    yy = input('press Enter to move...')
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
    yy = input('press Enter to move...')
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
    refocus_and_relink(microscope)
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

###################################################################################


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


    ##### HEIDI-SERGEY CALCULATION
    #if beam_type == BeamType.ELECTRON:
    #    tilt_radians =  np.deg2rad(PRETILT_DEGREES)
    #    y_move = +np.cos(tilt_radians) * expected_y
    #    z_move = +np.sin(tilt_radians) * expected_y
    #elif beam_type == BeamType.ION:
    #    tilt_radians =  np.deg2rad(PRETILT_DEGREES)
    #    y_move = +np.cos(tilt_radians) * expected_y
    #    z_move = -np.sin(tilt_radians) * expected_y

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



def new_electron_image(microscope, settings=None):
    microscope.imaging.set_active_view(1)  # the electron beam view
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image

def new_ion_image(microscope, settings=None):
    microscope.imaging.set_active_view(2)  # the ion beam view
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image

def take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6,  image_settings=None):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    # image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    #############
    # Take reference images with lower resolution, wider field of view
    microscope.beams.electron_beam.horizontal_field_width.value = hor_field_width
    microscope.beams.ion_beam.horizontal_field_width.value      = hor_field_width
    microscope.imaging.set_active_view(1)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    eb_reference = new_electron_image(microscope, image_settings)
    microscope.imaging.set_active_view(2)
    autocontrast(microscope, beam_type=BeamType.ION)
    ib_reference = new_ion_image(microscope, image_settings)
    return eb_reference, ib_reference

def stretch_image(image, y_scale):
    #y_scale = 1. / math.cos(np.deg2rad(5))
    height, width = image.shape
    height_scaled = int(round(height * y_scale))
    if height_scaled % 2 != 0:
        height_scaled += 1
    dy = (height_scaled - height) // 2
    image_scaled = resize(image, (height_scaled, width))
    image_scaled = image_scaled[dy:-dy, :]
    return image_scaled

def read_image(DIR, fileName, gaus_smooth=1):
    fileName = DIR + '/' + fileName
    image = Image.open(fileName)
    image = np.array(image)
    if image.shape[1] == 1536:
        image = image[0:1024, :]
    if image.shape[1] == 3072:
        image = image[0:2048, :]
    #image = ndi.filters.gaussian_filter(image, sigma=gaus_smooth) #median(imageTif, disk(1))
    return image

if __name__ == '__main__':
    # 1 - load images and correlate
    # 2   - take sem/fib images and correlate
    # 2.5 - correlate rotated ion beam image
    # 3 - hog, template matching
    TEST = 7
    print('cross-correlation test')

    if TEST==8:
        #storage.NewRun(prefix='test_alignment')
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        from autoscript_sdb_microscope_client.structures import StagePosition
        ip_address = '10.0.0.1'
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        stage = microscope.specimen.stage
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match

        yy = input('MOVE TO POSITION TO CREATE REFERENCE IMAGES flat-to-ion after trenching: HIGH AND LOW RES, press Enter when ready...')
        eb_lowres_ref, ib_lowres_ref   = take_electron_and_ion_reference_images(microscope, hor_field_width=600e-6, image_settings=image_settings)
        eb_highres_ref, ib_highres_ref = take_electron_and_ion_reference_images(microscope, hor_field_width= 80e-6, image_settings=image_settings)
        reference_images_low_and_high_res = (eb_lowres_ref, eb_highres_ref, ib_lowres_ref, ib_highres_ref) #use these images for future alignment

        realign_using_reference_eb_and_ib_images(microscope, reference_images_low_and_high_res, plot=False) # correct the stage drift after 180 deg rotation using treched lamella images as reference


        flat_to_electron_beam(microscope, pretilt_angle=PRETILT_DEGREES) # rotate to flat_to_electron
        # the lamella is now aligned

        # take reference images:
        eb_lowres_ref, ib_lowres_ref   = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
        eb_highres_ref, ib_highres_ref = take_electron_and_ion_reference_images(microscope, hor_field_width= 50e-6, image_settings=image_settings)
        reference_images_low_and_high_res = (eb_lowres_ref, eb_highres_ref, ib_lowres_ref, ib_highres_ref) #use these images for future alignment

        ### Need to tilt +6 deg, tilt first +3 deg only:
        tilting = StagePosition(x=0, y=0, z=0, t=np.deg2rad(3))
        print(tilting)
        yy = input('TILTING the stage by +3 deg to 30, press Enter when ready...')
        stage.relative_move(tilting)
        realign_at_different_stage_tilts(microscope, reference_images_low_and_high_res, previous_stage_tilt=27, beam_type=BeamType.ION)


        eb_lowres_ref, ib_lowres_ref   = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
        eb_highres_ref, ib_highres_ref = take_electron_and_ion_reference_images(microscope, hor_field_width= 50e-6, image_settings=image_settings)
        reference_images_low_and_high_res = (eb_lowres_ref, eb_highres_ref, ib_lowres_ref, ib_highres_ref) #use these images for future alignment

        ### Need to tilt +6 deg, tilt first +3 deg only:
        tilting = StagePosition(x=0, y=0, z=0, t=np.deg2rad(3))
        print(tilting)
        yy = input('TILTING the stage by +3 deg to 30, press Enter when ready...')
        stage.relative_move(tilting)
        realign_at_different_stage_tilts(microscope, reference_images_low_and_high_res, previous_stage_tilt=27, beam_type=BeamType.ION)



    if TEST == 7:
        #storage.NewRun(prefix='test_alignment')
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        from autoscript_sdb_microscope_client.structures import StagePosition
        ip_address = '10.0.0.1'
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        stage = microscope.specimen.stage
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match

        yy = input('MOVE TO POSITION TO CREATE REFERENCE IMAGES flat-to-ion after trenching: HIGH AND LOW RES, press Enter when ready...')
        eb_lowres_ref, ib_lowres_ref   = take_electron_and_ion_reference_images(microscope, hor_field_width=600e-6, image_settings=image_settings)
        eb_highres_ref, ib_highres_ref = take_electron_and_ion_reference_images(microscope, hor_field_width= 80e-6, image_settings=image_settings)
        reference_images_low_and_high_res = (eb_lowres_ref, eb_highres_ref, ib_lowres_ref, ib_highres_ref) #use these images for future alignment

        flat_to_electron_beam(microscope, pretilt_angle=PRETILT_DEGREES) # rotate to flat_to_electron
        realign_using_reference_eb_and_ib_images(microscope, reference_images_low_and_high_res, plot=False) # correct the stage drift after 180 deg rotation using treched lamella images are reference
        # the lamella is now aligned
        #
        #############
        #############
        yy = input('Press Enter...')
        move_to_jcut_angle(microscope) ####<----flat to electron beam + jcut_angle=6, stage tilt total 33
        ################# LOW RES ALIGNMENT # 1, tilted to flat! ########################
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution
        eb_lowres,  ib_lowres   = take_electron_and_ion_reference_images(microscope, hor_field_width=400e-6, image_settings=image_settings) # TODO: yaml use input
        eb_highres, ib_highres  = take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6,  image_settings=image_settings) # TODO: yaml use input
        reference_images_low_and_high_res = (eb_lowres, eb_highres, ib_lowres, ib_highres)
        pixelsize_x_lowres  = ib_lowres.metadata.binary_result.pixel_size.x
        pixelsize_y_lowres  = ib_lowres.metadata.binary_result.pixel_size.y

        storage.SaveImage(ib_lowres,  id='A_tiltAlign_lowres')
        storage.SaveImage(ib_highres, id='A_tiltAlign_highres')
        storage.step_counter += 1

        height, width       = ib_lowres_ref.data.shape
        cmask = circ_mask(size=(width,height), radius=height // 5 - 15, sigma=10)  # circular mask, align only the central areas
        ib_lowres_ref_norm = ( ib_lowres_ref.data - np.mean(ib_lowres_ref.data) ) / np.std(ib_lowres_ref.data)
        ib_lowres_ref_norm = np.rot90(np.rot90(np.copy(ib_lowres_ref_norm)))
        ib_lowres_norm     = ( ib_lowres.data - np.mean(ib_lowres.data) ) / np.std(ib_lowres.data)

        y_scale = 1. / math.cos(np.deg2rad(46))
        ib_lowres_norm_stretched = stretch_image(ib_lowres_norm, y_scale)

        lowpass_pixels  = int( max(ib_lowres_norm.shape) / 6 ) # =256 @ 1536x1024, good for i-beam images
        highpass_pixels = int( max(ib_lowres_norm.shape)/ 64 ) # =24  @ 1536x1024, good for i-beam images => need a large highpass to remove noise and ringing
        sigma = int( 10 * max(ib_lowres_norm.data.shape)/1536) # =10 @ 1536x1024, good for i-beam images
        xcorr = crosscorrelation(ib_lowres_norm_stretched , ib_lowres_ref_norm , bp='yes', lp=lowpass_pixels, hp=highpass_pixels, sigma=sigma)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        dx_meters = err[1] * pixelsize_x_lowres
        dy_meters = err[0] * pixelsize_y_lowres
        print("X-shift =  {} meters".format(dx_meters))
        print("Y-shift =  {} meters".format(dy_meters))
        x_move = x_corrected_stage_movement(-dx_meters)
        yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t, beam_type=BeamType.ION) ##check electron/ion movement
        print('relative movement of the the stage by X  :',  x_move)
        print('relative movement of the the stage by Y-Z:', yz_move)
        yy = input('Press Enter to move...')
        stage.relative_move(x_move)
        stage.relative_move(yz_move)

        ################# LOW RES ALIGNMENT # 2, tilted to flat! ########################
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution
        eb_lowres,  ib_lowres   = take_electron_and_ion_reference_images(microscope, hor_field_width=400e-6, image_settings=image_settings) # TODO: yaml use input
        eb_highres, ib_highres  = take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6,  image_settings=image_settings) # TODO: yaml use input
        pixelsize_x_lowres  = ib_lowres.metadata.binary_result.pixel_size.x
        pixelsize_y_lowres  = ib_lowres.metadata.binary_result.pixel_size.y

        storage.SaveImage(ib_lowres,  id='B_tiltAlign_lowres')
        storage.SaveImage(ib_highres, id='B_tiltAlign_highres')
        storage.step_counter += 1

        height, width       = ib_lowres_ref.data.shape
        cmask = circ_mask(size=(width,height), radius=height // 4 - 15, sigma=10)  # circular mask, align only the central areas
        ib_lowres_norm     = ( ib_lowres.data - np.mean(ib_lowres.data) ) / np.std(ib_lowres.data)

        lowpass_pixels  = int( max(ib_lowres_norm.shape) / 6 ) # =256 @ 1536x1024, good for i-beam images
        highpass_pixels = int( max(ib_lowres_norm.shape)/ 64 ) # =24  @ 1536x1024, good for i-beam images => need a large highpass to remove noise and ringing
        sigma = int( 10 * max(ib_lowres_norm.data.shape)/1536) # =10 @ 1536x1024, good for i-beam images
        xcorr = crosscorrelation(ib_lowres_norm * cmask, ib_lowres_ref_norm * cmask, bp='yes', lp=lowpass_pixels, hp=highpass_pixels, sigma=sigma)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        dx_meters = err[1] * pixelsize_x_lowres
        dy_meters = err[0] * pixelsize_y_lowres
        print("X-shift =  {} meters".format(dx_meters))
        print("Y-shift =  {} meters".format(dy_meters))
        x_move = x_corrected_stage_movement(-dx_meters)
        yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t, beam_type=BeamType.ION) ##check electron/ion movement
        print('relative movement of the the stage by X  :',  x_move)
        print('relative movement of the the stage by Y-Z:', yz_move)
        yy = input('Press Enter to move...')
        stage.relative_move(x_move)
        stage.relative_move(yz_move)


        ################# HIGH RES ALIGNMENT, tilted to flat! ########################
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution
        eb_lowres,  ib_lowres   = take_electron_and_ion_reference_images(microscope, hor_field_width=400e-6, image_settings=image_settings) # TODO: yaml use input
        eb_highres, ib_highres  = take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6,  image_settings=image_settings) # TODO: yaml use input
        pixelsize_x_highres  = ib_highres.metadata.binary_result.pixel_size.x
        pixelsize_y_highres  = ib_highres.metadata.binary_result.pixel_size.y

        storage.SaveImage(ib_lowres,  id='C_tiltAlign_lowres')
        storage.SaveImage(ib_highres, id='C_tiltAlign_highres')
        storage.step_counter += 1

        height, width       = ib_highres_ref.data.shape
        cmask = circ_mask(size=(width,height), radius=height // 4 - 15, sigma=10)  # circular mask, align only the central areas
        ib_highres_ref_norm = ( ib_highres_ref.data - np.mean(ib_highres_ref.data) ) / np.std(ib_highres_ref.data)
        ib_highres_ref_norm = np.rot90(np.rot90(np.copy(ib_highres_ref_norm)))
        ib_highres_norm     = ( ib_highres.data - np.mean(ib_highres.data) ) / np.std(ib_highres.data)

        lowpass_pixels  = int( max(ib_highres_norm.shape) / 6 ) # =256 @ 1536x1024, good for i-beam images
        highpass_pixels = int( max(ib_highres_norm.shape)/ 64 ) # =24  @ 1536x1024, good for i-beam images => need a large highpass to remove noise and ringing
        sigma = int( 10 * max(ib_highres_norm.data.shape)/1536) # =10 @ 1536x1024, good for i-beam images
        xcorr = crosscorrelation(ib_highres_norm * cmask, ib_highres_ref_norm * cmask, bp='yes', lp=lowpass_pixels, hp=highpass_pixels, sigma=sigma)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        dx_meters = err[1] * pixelsize_x_highres
        dy_meters = err[0] * pixelsize_y_highres
        print("X-shift =  {} meters".format(dx_meters))
        print("Y-shift =  {} meters".format(dy_meters))
        x_move = x_corrected_stage_movement(-dx_meters)
        yz_move = y_corrected_stage_movement(dy_meters, stage.current_position.t, beam_type=BeamType.ION) ##check electron/ion movement
        print('relative movement of the the stage by X  :',  x_move)
        print('relative movement of the the stage by Y-Z:', yz_move)
        yy = input('Press Enter to move...')
        stage.relative_move(x_move)
        stage.relative_move(yz_move)



        #############
        ############# Mill J-cut
        #############
        move_to_jcut_angle(microscope) ####<----flat to electron beam + jcut_angle=6, stage tilt total 33




    if TEST == 5.9:
        DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.15_trench_alignment'
        fileName_ib_tilt27 = r'001_I_T27_HFW_50um_1us_1536_1024_001.tif'
        fileName_ib_tilt37 = r'002_I_T37_HFW_50um_1us_1536_1024_002.tif'
        sample_tilt27 = read_image(DIR, fileName_ib_tilt27, gaus_smooth=0)
        sample_tilt37 = read_image(DIR, fileName_ib_tilt37, gaus_smooth=0)
        sample_tilt27_norm = ( sample_tilt27 - np.mean(sample_tilt27) ) / np.std(sample_tilt27)
        sample_tilt37_norm = ( sample_tilt37 - np.mean(sample_tilt37) ) / np.std(sample_tilt37)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(sample_tilt27_norm, cmap='Blues_r', alpha=1)
        ax.imshow(sample_tilt37_norm, cmap='Oranges_r', alpha=0.5)
        dx_ib, dy_ib = shift_from_crosscorrelation_simple_images(sample_tilt27_norm, sample_tilt37_norm, lowpass=256, highpass=22, sigma=10  )
        plt.show()
    ################################################################################
    if TEST==6:
        print('TEST - correlate 27 and 37 and 32 degrees tilts! ion beam, find the shift in e-beam')
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        from autoscript_sdb_microscope_client.structures import StagePosition
        ip_address = '10.0.0.1'
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        stage = microscope.specimen.stage
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match

        yy = input('MOVE TO 27 tilt: HIGH AND LOW RES, press Enter when ready...')
        microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb_lowres_27 = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_lowres_27 = microscope.imaging.grab_frame(image_settings)
        microscope.beams.electron_beam.horizontal_field_width.value = 50e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 50e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb_highres_27 = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_highres_27 = microscope.imaging.grab_frame(image_settings)
        ############
        eb_lowres_27.save('01_eb_lowres_27.tif')
        ib_lowres_27.save('01_ib_lowres_27.tif')
        eb_highres_27.save('01_eb_highres_27.tif')
        ib_highres_27.save('01_ib_highres_27.tif')



        tilting = StagePosition(x=0, y=0, z=0, t=np.deg2rad(5))
        print(tilting)
        yy = input('TILTING the stage by +5 deg to 32, press Enter when ready...')
        stage.relative_move(tilting)
        # alignment (TODO: magnifications must match, yaml user input)
        microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # TODO: user input, can't be smaller than 150e-6
        microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # TODO: user input, can't be smaller than 150e-6
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
        microscope.imaging.set_active_view(1)
        eb_lowres_32 = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_lowres_32 = microscope.imaging.grab_frame(image_settings)
        microscope.beams.electron_beam.horizontal_field_width.value = 50e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 50e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb_highres_32 = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_highres_32 = microscope.imaging.grab_frame(image_settings)
        ###
        eb_lowres_32.save('02_eb_lowres_32.tif')
        ib_lowres_32.save('02_ib_lowres_32.tif')
        eb_highres_32.save('02_eb_highres_32.tif')
        ib_highres_32.save('02_ib_highres_32.tif')
        # correlate
        ### normalise!!!
        dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(ib_highres_32, ib_highres_27, lowpass=256, highpass=22, sigma=10)
        x_move  = x_corrected_stage_movement(-dx_ei_meters)
        yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ION)
        print('prepare to move -relative- the stage by X:', x_move)
        print('prepare to move -relative- the stage by Y-Z:', yz_move)
        yy = input('Do you want to move the stage for real? (yes/no)')
        if yy == 'yes':
            stage.relative_move(x_move)
            stage.relative_move(yz_move)
        else:
            print('Movement will not proceed!')

        microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # TODO: user input, can't be smaller than 150e-6
        microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # TODO: user input, can't be smaller than 150e-6
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
        microscope.imaging.set_active_view(1)
        eb_lowres_32_aligned = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_lowres_32_aligned = microscope.imaging.grab_frame(image_settings)
        microscope.beams.electron_beam.horizontal_field_width.value = 50e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 50e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb_highres_32_aligned = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_highres_32_aligned = microscope.imaging.grab_frame(image_settings)
        ###
        eb_lowres_32_aligned.save('03_eb_lowres_32_aligned.tif')
        ib_lowres_32_aligned.save('03_ib_lowres_32_aligned.tif')
        eb_highres_32_aligned.save('03_eb_highres_32_aligned.tif')
        ib_highres_32_aligned.save('03_ib_highres_32_aligned.tif')



        tilting = StagePosition(x=0, y=0, z=0, t=np.deg2rad(5))
        print(tilting)
        yy = input('TILTING the stage by +5 deg to 37, press Enter when ready...')
        stage.relative_move(tilting)
        # alignment (TODO: magnifications must match, yaml user input)
        microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # TODO: user input, can't be smaller than 150e-6
        microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # TODO: user input, can't be smaller than 150e-6
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
        microscope.imaging.set_active_view(1)
        eb_lowres_37 = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_lowres_37 = microscope.imaging.grab_frame(image_settings)
        microscope.beams.electron_beam.horizontal_field_width.value = 50e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 50e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb_highres_37 = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_highres_37 = microscope.imaging.grab_frame(image_settings)
        ###
        eb_lowres_37.save('04_eb_lowres_37.tif')
        ib_lowres_37.save('04_ib_lowres_37.tif')
        eb_highres_37.save('04_eb_highres_37.tif')
        ib_highres_37.save('04_ib_highres_37.tif')
        # correlate
        ### normalise!!!
        dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(ib_highres_37, ib_highres_32_aligned, lowpass=256, highpass=22, sigma=10)
        x_move  = x_corrected_stage_movement(-dx_ei_meters)
        yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ION)
        print('prepare to move -relative- the stage by X:', x_move)
        print('prepare to move -relative- the stage by Y-Z:', yz_move)
        yy = input('Do you want to move the stage for real? (yes/no)')
        if yy == 'yes':
            stage.relative_move(x_move)
            stage.relative_move(yz_move)
        else:
            print('Movement will not proceed!')

        microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # TODO: user input, can't be smaller than 150e-6
        microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # TODO: user input, can't be smaller than 150e-6
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
        microscope.imaging.set_active_view(1)
        eb_lowres_37_aligned = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_lowres_37_aligned = microscope.imaging.grab_frame(image_settings)
        microscope.beams.electron_beam.horizontal_field_width.value = 50e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 50e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb_highres_37_aligned = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_highres_37_aligned = microscope.imaging.grab_frame(image_settings)
        ###
        eb_lowres_37_aligned.save('05_eb_lowres_37_aligned.tif')
        ib_lowres_37_aligned.save('05_ib_lowres_37_aligned.tif')
        eb_highres_37_aligned.save('05_eb_highres_37_aligned.tif')
        ib_highres_37_aligned.save('05_ib_highres_37_aligned.tif')

        # SHIFT in the ebeam for the needle
        dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(eb_highres_27, eb_highres_37_aligned, lowpass=128, highpass=6, sigma=2)

        microscope.imaging.set_active_view(2)  #
        new_ib_highres = microscope.imaging.grab_frame(image_settings)
        plot_overlaid_images(ib_highres_37_aligned, ib_highres_27, rotate_second_image=False)




    if TEST == 5:
        print('TEST - correlate lamella/trench position, correct shift using reference flat_ion_beam image_rotated')
        original_lamella_area_images = []
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        ip_address = '10.0.0.1'
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        stage = microscope.specimen.stage
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match

        yy = input('MOVE TO POSITION TO CREATE REFERENCE IMAGES: HIGH AND LOW RES, press Enter when ready...')

        microscope.beams.electron_beam.horizontal_field_width.value = 400e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 400e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb_low_res = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_low_res = microscope.imaging.grab_frame(image_settings)
        microscope.beams.electron_beam.horizontal_field_width.value = 50e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 50e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb_high_res = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_high_res = microscope.imaging.grab_frame(image_settings)
        original_lamella_area_images.append((ib_low_res, ib_high_res, eb_low_res, eb_high_res))
        # Unpack reference images
        ib_low_res_reference, ib_high_res_reference, eb_low_res_reference, eb_high_res_reference = original_lamella_area_images[0]


        yy = input('SHIFT/rotate the stage for alignment testing and stange movement, press Enter when ready...')
        # use ib images (flat) to correlate with the electron beam image at the new position
        # LOW resolution alignment (TODO: magnifications must match, yaml user input)
        microscope.beams.ion_beam.horizontal_field_width.value      = 400e-6  # TODO: user input, can't be smaller than 150e-6
        microscope.beams.electron_beam.horizontal_field_width.value = 400e-6  # TODO: user input, can't be smaller than 150e-6
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
        microscope.imaging.set_active_view(1)
        new_eb_lowres = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        new_ib_lowres = microscope.imaging.grab_frame(image_settings)
        # correlate
        ### normalise!!!
        dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(new_eb_lowres, ib_low_res_reference)
        x_move  = x_corrected_stage_movement(-dx_ei_meters)
        yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ELECTRON)
        print('prepare to move -relative- the stage by X:', x_move)
        print('prepare to move -relative- the stage by Y-Z:', yz_move)
        yy = input('Do you want to move the stage for real? (yes/no)')
        if yy == 'yes':
            stage.relative_move(x_move)
            stage.relative_move(yz_move)
        else:
            print('Movement will not proceed!')

        microscope.imaging.set_active_view(1)  #
        new_eb_lowres = microscope.imaging.grab_frame(image_settings)
        plot_overlaid_images(new_eb_lowres, ib_low_res_reference, rotate_second_image=True)


        # HIGH resolution alignment NUMBER 1 (TODO: magnifications must match, yaml user input)
        microscope.beams.ion_beam.horizontal_field_width.value      = 50e-6  # TODO: user input, can't be smaller than 150e-6
        microscope.beams.electron_beam.horizontal_field_width.value = 50e-6  # TODO: user input, can't be smaller than 150e-6
        image_settings = GrabFrameSettings(resolution="1536x1024",
                                           dwell_time=1e-6)  # TODO: user input resolution, must match
        microscope.imaging.set_active_view(1)  #
        new_eb_highres = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)  #
        new_ib_highres = microscope.imaging.grab_frame(image_settings)
        # correlate
        dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(new_eb_highres, ib_high_res_reference)
        x_move  = x_corrected_stage_movement(-dx_ei_meters)
        yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ELECTRON)
        print('prepare to move relative the stage by X:', x_move)
        print('prepare to move relative the stage by Y-Z:', yz_move)
        yy = input('Do you want to move the stage for real? (yes/no)')
        if yy == 'yes':
            stage.relative_move(x_move)
            stage.relative_move(yz_move)
        else:
            print('Movement will not proceed!')

        microscope.imaging.set_active_view(1)  #
        new_eb_highres = microscope.imaging.grab_frame(image_settings)
        plot_overlaid_images(new_eb_highres, ib_high_res_reference, rotate_second_image=True)









    if TEST == 5.1:
        print('TEST correlate POLES - use ebeam for better results')
        print('for trenching positions toom REMOVE ION BEAM IMAGING THOUGH!')
        original_lamella_area_images = []
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        ip_address = '10.0.0.1'
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        stage = microscope.specimen.stage
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match

        yy = input('MOVE TO POSITION TO CREATE REFERENCE IMAGES: HIGH AND LOW RES, press Enter when ready...')

        microscope.beams.electron_beam.horizontal_field_width.value = 400e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 400e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb_low_res = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_low_res = microscope.imaging.grab_frame(image_settings)
        microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb_high_res = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib_high_res = microscope.imaging.grab_frame(image_settings)

        original_lamella_area_images.append((ib_low_res, ib_high_res, eb_low_res, eb_high_res))
        # Unpack reference images
        ib_low_res_reference, ib_high_res_reference, eb_low_res_reference, eb_high_res_reference = original_lamella_area_images[0]

        yy = input('SHIFT/rotate the stage for alignment testing and stange movement, press Enter when ready...')

        # use ib images (flat) to correlate with the electron beam image at the new position
        # LOW resolution alignment (TODO: magnifications must match, yaml user input)
        microscope.beams.ion_beam.horizontal_field_width.value      = 400e-6  # TODO: user input, can't be smaller than 150e-6
        microscope.beams.electron_beam.horizontal_field_width.value = 400e-6  # TODO: user input, can't be smaller than 150e-6
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match
        microscope.imaging.set_active_view(1)
        new_eb_lowres = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        new_ib_lowres = microscope.imaging.grab_frame(image_settings)
        # correlate
        dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(new_eb_lowres, eb_low_res_reference)
        x_move  = x_corrected_stage_movement(-dx_ei_meters)
        yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ELECTRON)
        print('prepare to move -relative- the stage by X:', x_move)
        print('prepare to move -relative- the stage by Y-Z:', yz_move)
        yy = input('Do you want to move the stage for real? (yes/no)')
        if yy == 'yes':
            stage.relative_move(x_move)
            stage.relative_move(yz_move)
        else:
            print('Movement will not proceed!')

        microscope.imaging.set_active_view(1)  #
        new_eb_lowres = microscope.imaging.grab_frame(image_settings)
        plot_overlaid_images(new_eb_lowres, eb_low_res_reference, rotate_second_image=False)



        # HIGH resolution alignment (TODO: magnifications must match, yaml user input)
        microscope.beams.ion_beam.horizontal_field_width.value = 150e-6  # TODO: user input, can't be smaller than 150e-6
        microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # TODO: user input, can't be smaller than 150e-6
        image_settings = GrabFrameSettings(resolution="1536x1024",
                                           dwell_time=1e-6)  # TODO: user input resolution, must match
        microscope.imaging.set_active_view(1)  #
        new_eb_highres = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)  #
        new_ib_highres = microscope.imaging.grab_frame(image_settings)
        # correlate
        dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(new_eb_highres, eb_high_res_reference)
        x_move  = x_corrected_stage_movement(-dx_ei_meters)
        yz_move = y_corrected_stage_movement(dy_ei_meters, stage.current_position.t, beam_type=BeamType.ELECTRON)
        print('prepare to move relative the stage by X:', x_move)
        print('prepare to move relative the stage by Y-Z:', yz_move)
        yy = input('Do you want to move the stage for real? (yes/no)')
        if yy == 'yes':
            stage.relative_move(x_move)
            stage.relative_move(yz_move)
        else:
            print('Movement will not proceed!')

        microscope.imaging.set_active_view(1)  #
        new_eb_highres = microscope.imaging.grab_frame(image_settings)
        plot_overlaid_images(new_eb_highres, eb_high_res_reference, rotate_second_image=False)














    if TEST == 2.5:
        ip_address = '10.0.0.1'
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)

        field_of_view = 4 * 100e-6 # in meters
        microscope.beams.ion_beam.horizontal_field_width.value      = field_of_view
        microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
        microscope.imaging.set_active_view(1)
        eb_1 = microscope.imaging.grab_frame()
        microscope.imaging.set_active_view(2)
        ib_1 = microscope.imaging.grab_frame()

        yy  = input('Prepare the next image set, e.g. rotate by 180 degrees')

        microscope.imaging.set_active_view(1)
        eb_2 = microscope.imaging.grab_frame()
        microscope.imaging.set_active_view(2)
        ib_2 = microscope.imaging.grab_frame()

        ### correlate from np.array data
        dx_ei_pixels, dy_ei_pixels = shift_from_crosscorrelation_simple_images(eb_2.data, ndi.rotate(ib_1.data, 180, reshape=False))

        ### correlate from AdournedImage
        dx_ei_meters, dy_ei_meters = shift_from_correlation_electronBeam_and_ionBeam(eb_2, ib_1)


    if TEST == 2:
        ip_address = '10.0.0.1'
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)

        microscope.imaging.set_active_view(1)
        eb_1 = microscope.imaging.grab_frame()
        microscope.imaging.set_active_view(2)
        ib_1 = microscope.imaging.grab_frame()

        yy  = input('Prepare the next image set, e.g. shift by some amount')

        microscope.imaging.set_active_view(1)
        eb_2 = microscope.imaging.grab_frame()
        microscope.imaging.set_active_view(2)
        ib_2 = microscope.imaging.grab_frame()

        ### correlate from np.array data
        dx_e_pixels, dy_e_pixels = shift_from_crosscorrelation_simple_images(eb_1.data, eb_2.data)
        dx_i_pixels, dy_i_pixels = shift_from_crosscorrelation_simple_images(ib_1.data, ib_2.data)

        ### correlate from AdournedImage
        dx_e_meters, dy_e_meters = shift_from_crosscorrelation_AdornedImages(eb_1, eb_2)
        dx_i_meters, dy_i_meters = shift_from_crosscorrelation_AdornedImages(ib_1, ib_2)













    ####################################################################################################################
    if TEST == 1:
        DIR = r'Y:\Sergey\codes\HS auto lamella1\01.15.2021_cross_corrolation_for_stage_rotation'
        DIR.replace('\\', '/')

        #fileName1 = DIR + '/' + r'E_flat_to_sem_1381x_HFW_150um_001.tif'
        fileName1 = DIR + '/' + r'E_flat_to_sem_414x_HFW_500um_001.tif'
        imageTif1 = Image.open(fileName1)
        imageTif1 = np.array(imageTif1)
        imageTif1 = imageTif1[0:1024, :]
        ind1 = np.unravel_index(imageTif1.argmax(), imageTif1.shape)
        Imax1 = imageTif1.max()
        ##############  correlate with the image #####################
        #fileName2 = DIR + '/' + r'I_flat_to_ion_1381x_HFW_150um_001.tif'
        fileName2 = DIR + '/' + r'I_flat_to_ion_414x_HFW_500um_001.tif'
        imageTif2 = Image.open(fileName2)
        imageTif2 = np.array(imageTif2)
        imageTif2 = imageTif2[0:1024, :]
        imageTif2 = ndi.rotate(imageTif2, 180, reshape=False)
        ind2 = np.unravel_index(imageTif2.argmax(), imageTif2.shape)
        print(ind1)
        print(ind2)
        dx, dy = shift_from_crosscorrelation_simple_images(imageTif1, imageTif2)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(imageTif1, cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(imageTif2, cmap='gray')

        x0 = int(imageTif1.shape[1] / 2)
        y0 = int(imageTif1.shape[0] / 2)
        width = 300
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(imageTif1[y0 - width: y0 + width, x0 - width: x0 + width], cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(imageTif2[y0 - width: y0 + width, x0 - width: x0 + width], cmap='gray')

        original_image = imageTif1[y0 - width      : y0 + width,       x0 - width      : x0 + width]
        aligned_image  = imageTif2[y0 - width + dy : y0 + width + dy , x0 - width + dx : x0 + width + dx ]
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(original_image, cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(aligned_image, cmap='gray')

        plot_overlaid_images(imageTif1[y0 - width: y0 + width, x0 - width: x0 + width], imageTif2[y0 - width: y0 + width, x0 - width: x0 + width], show=True)
        plot_overlaid_images(original_image, aligned_image, show=True)

    ####################################################################################################################

    if TEST == 1.1:
        DIR = r'Y:\Sergey\codes\HS auto lamella1\01.15.2021_images_landing_posts'
        DIR.replace('\\', '/')

        fileName1 = DIR + '/' + r'80x_HFW_2.59mm_001.tif'
        imageTif1 = Image.open(fileName1)
        imageTif1 = np.array(imageTif1)
        imageTif1 = imageTif1[0:1024, :]
        ind1 = np.unravel_index(imageTif1.argmax(), imageTif1.shape)
        Imax1 = imageTif1.max()
        ##############  correlate with the image #####################
        fileName2 = DIR + '/' + r'80x_HFW_2.59mm_002.tif'
        imageTif2 = Image.open(fileName2)
        imageTif2 = np.array(imageTif2)
        imageTif2 = imageTif2[0:1024, :]
        imageTif2 = ndi.rotate(imageTif2, 0, reshape=False)
        ind2 = np.unravel_index(imageTif2.argmax(), imageTif2.shape)
        print(ind1)
        print(ind2)
        dx, dy = shift_from_crosscorrelation_simple_images(imageTif1, imageTif2)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(imageTif1, cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(imageTif2, cmap='gray')

        x0 = int(imageTif1.shape[1] / 2)
        y0 = int(imageTif1.shape[0] / 2)
        width = 300
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(imageTif1[y0 - width: y0 + width, x0 - width: x0 + width], cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(imageTif2[y0 - width: y0 + width, x0 - width: x0 + width], cmap='gray')

        original_image = imageTif1[y0 - width: y0 + width, x0 - width: x0 + width]
        aligned_image = imageTif2[y0 - width + dy: y0 + width + dy, x0 - width + dx: x0 + width + dx]
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(original_image, cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(aligned_image, cmap='gray')

        plot_overlaid_images(imageTif1[y0 - width: y0 + width, x0 - width: x0 + width],
                             imageTif2[y0 - width: y0 + width, x0 - width: x0 + width], show=True)
        plot_overlaid_images(original_image, aligned_image, show=True)






    if TEST == 3: #hog template matching
        ip_address = '10.0.0.1'
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)

        # Set the correct magnification / field of view
        print('starting...')
        yy  = input('adjust contrast, press enter when movement complete...')
        ########################################################################
        field_of_view = 100e-6  # in meters
        microscope.beams.ion_beam.horizontal_field_width.value      = field_of_view
        microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
        microscope.imaging.set_active_view(1)
        reference_eb_highres = microscope.imaging.grab_frame()
        microscope.imaging.set_active_view(2)
        reference_ib_highres = microscope.imaging.grab_frame()
        #reference_ib_highres.data = ndi.rotate(reference_ib_highres.data, 180, reshape=False)
        # ndi.median_filter(display_image, size=median_smoothing)
        ########################################################################
        field_of_view = 4 * 100e-6 # in meters
        microscope.beams.ion_beam.horizontal_field_width.value      = field_of_view
        microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
        microscope.imaging.set_active_view(1)
        reference_eb_lowres = microscope.imaging.grab_frame()
        microscope.imaging.set_active_view(2)
        reference_ib_lowres = microscope.imaging.grab_frame()
        #reference_ib_lowres.data = ndi.rotate(reference_ib_lowres.data, 180, reshape=False)
        ########################################################################

        print('shift the sample...')
        yy = input('adjust autocontrast, press enter when movement complete...')
        ########################################################################
        microscope.beams.ion_beam.horizontal_field_width.value      = field_of_view
        microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
        microscope.imaging.set_active_view(1)
        eb_highres = microscope.imaging.grab_frame()
        microscope.imaging.set_active_view(2)
        ib_highres = microscope.imaging.grab_frame()
        #ib_highres.data = ndi.rotate(ib_highres.data, 180, reshape=False)
        ########################################################################
        field_of_view = 4 * 100e-6 # in meters
        microscope.beams.ion_beam.horizontal_field_width.value      = field_of_view
        microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
        microscope.imaging.set_active_view(1)
        eb_lowres = microscope.imaging.grab_frame()
        microscope.imaging.set_active_view(2)
        ib_lowres = microscope.imaging.grab_frame()
        #ib_lowres.data = ndi.rotate(ib_lowres.data, 180, reshape=False)
        ########################################################################

        #location = match_locations(microscope, image, lowres_template)
        import autoscript_toolkit.vision as vision_toolkit
        from autoscript_toolkit.template_matchers import HogMatcher
        hog_matcher = HogMatcher(microscope)

        # low res alignment
        original_feature_center = list(np.flip(np.array(reference_eb_lowres.data.shape) // 2, axis=0))
        location = vision_toolkit.locate_feature(eb_lowres, reference_eb_lowres, hog_matcher, original_feature_center=original_feature_center)
        location.print_all_information()  # displays in x-y coordinate order

        # high res alignment
        original_feature_center = list(np.flip(np.array(reference_eb_highres.data.shape) // 2, axis=0))
        location = vision_toolkit.locate_feature(eb_highres, reference_eb_highres, hog_matcher, original_feature_center=original_feature_center)
        location.print_all_information()  # displays in x-y coordinate order