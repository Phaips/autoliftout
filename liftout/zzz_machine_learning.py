#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('matplotlib', 'qt')
from scipy import *
from scipy import signal
import scipy.ndimage as ndi
from scipy import fftpack, misc
import scipy
import scipy.ndimage as ndi
import os, sys, glob

import skimage.draw
import skimage.io

from pylab import *
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from operator import itemgetter
from enum import Enum

from scipy.ndimage.morphology import binary_dilation
from skimage.filters import gaussian, threshold_otsu, median
from skimage.measure import label
from skimage.morphology import disk
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import probabilistic_hough_line
from skimage.draw import line
#from autoscript_sdb_microscope_client.structures import ManipulatorPosition

class BeamType(Enum):
    ION = 'ION'
    ELECTRON = 'ELECTRON'

PRETILT_DEGREES = 27



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

		img1ft = scipy.fftpack.ifftshift(bandpass * scipy.fftpack.fftshift(scipy.fftpack.fft2(img1)))
		s = img1.shape[0] * img1.shape[1]
		tmp = img1ft * np.conj(img1ft)
		img1ft = s * img1ft / np.sqrt(tmp.sum())

		img2ft = scipy.fftpack.ifftshift(bandpass * scipy.fftpack.fftshift(scipy.fftpack.fft2(img2)))
		img2ft[0, 0] = 0
		tmp = img2ft * np.conj(img2ft)
		img2ft = s * img2ft / np.sqrt(tmp.sum())
		xcorr = np.real(scipy.fftpack.fftshift(scipy.fftpack.ifft2(img1ft * np.conj(img2ft))))
	elif bp == 'no':
		img1ft = scipy.fftpack.fft2(img1)
		img2ft = np.conj(scipy.fftpack.fft2(img2))
		img1ft[0, 0] = 0
		xcorr = np.abs(scipy.fftpack.fftshift(scipy.fftpack.ifft2(img1ft * img2ft)))
	else:
		print('ERROR in xcorr2: bandpass value ( bp= ' + str(bp) + ' ) not recognized')
		return -1
	return xcorr


def shift_from_crosscorrelation_simple_images(img1, img2):
    xcorr = crosscorrelation(img1, img2, bp='yes', lp=int( max(img1.shape)/12 ), hp=2, sigma=2)
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    print('\n', maxX, maxY)
    cen = np.asarray(xcorr.shape) / 2
    print('centre = ', cen)
    err = np.array(cen - [maxX, maxY], int)
    print("Shift between 1 and 2 is = " + str(err))
    print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
    return err[1], err[0]



def shift_from_crosscorrelation_AdornedImages(img1, img2):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    pixelsize_x_1 = img1.metadata.binary_result.pixel_size.x
    pixelsize_y_1 = img1.metadata.binary_result.pixel_size.y
    pixelsize_x_2 = img2.metadata.binary_result.pixel_size.x
    pixelsize_y_2 = img2.metadata.binary_result.pixel_size.y
    xcorr = crosscorrelation(img1.data, img2.data, bp='yes', lp=int( max(img1.data.shape)/12 ), hp=2, sigma=2)
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


def gauss_derivative_kernels(size, sizey=None):
    """ returns x and y derivatives of a 2D
        gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = mgrid[-size:size+1, -sizey:sizey+1]

    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2)))
    gy = - y * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2)))

    return gx,gy

def gauss_derivatives(im, n, ny=None):
    """ returns x and y derivatives of an image using gaussian
        derivative filters of size n. The optional argument
        ny allows for a different size in the y direction."""
    gx,gy = gauss_derivative_kernels(n, sizey=ny)
    imx = signal.convolve(im,gx, mode='same')
    imy = signal.convolve(im,gy, mode='same')
    return imx,imy

def gauss_kernel(size, sizey = None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()


def compute_harris_response(image):
    """ compute the Harris corner detector response function
        for each pixel in the image"""

    #derivatives
    imx,imy = gauss_derivatives(image, 3)

    #kernel for blurring
    gauss = gauss_kernel(3)

    #compute components of the structure tensor
    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')

    #determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr  = Wxx + Wyy

    return Wdet / Wtr

def get_harris_points(harrisim, min_distance=10, threshold=0.1):
    """ return corners from a Harris response image
        min_distance is the minimum nbr of pixels separating
        corners and image boundary"""

    #find top corner candidates above a threshold
    corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    #get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
    #...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]

    #sort candidates
    index = argsort(candidate_values)

    #store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1

    #select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),(coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0

    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""
    print(filtered_coords)
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'o')
    axis('off')
    show()











def autocontrast(microscope, beam_type=BeamType.ELECTRON):
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


def _reduced_area_rectangle(reduced_area_coords):
    assert len(reduced_area_coords) == 4
    top_corner_x, top_corner_y, width, height = reduced_area_coords
    return Rectangle(top_corner_x, top_corner_y, width, height)

def new_electron_image(microscope, settings=None):
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
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image

def new_ion_image(microscope, settings=None):
    """Take new ion beam image.
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
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



def x_corrected_needle_movement(expected_x, stage_tilt=None):
    """Needle movement in X, XTGui coordinates (Electron coordinate).
    Parameters
    ----------
    expected_y : float
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

def move_needle_closer(microscope, *, x_shift=-20e-6, z_shift=-180e-6):
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


def needle_reference_images(microscope, move_needle_to="liftout", dwell_time=10e-6):
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings
    move_sample_stage_out(microscope)
    if move_needle_to == "liftout":
        park_position = move_needle_to_liftout_position(microscope)
    elif move_needle_to == "landing":
        park_position = move_needle_to_landing_position(microscope)
    # TODO: set field of view in electron & ion beam to match
    camera_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=dwell_time )
    ####
    microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # can't be smaller than 150e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # can't be smaller than 150e-6
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    needle_reference_eb_lowres = new_electron_image(microscope, camera_settings)
    autocontrast(microscope, beam_type=BeamType.ION)
    needle_reference_ib_lowres = new_ion_image(microscope, camera_settings)
    #####
    microscope.beams.ion_beam.horizontal_field_width.value      = 80e-6  # can't be smaller than 150e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 80e-6  # can't be smaller than 150e-6
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    needle_reference_eb_highres = new_electron_image(microscope, camera_settings)
    autocontrast(microscope, beam_type=BeamType.ION)
    needle_reference_ib_highres = new_ion_image(microscope, camera_settings)
    ####
    retract_needle(microscope, park_position)
    return [needle_reference_eb_lowres, needle_reference_eb_highres, needle_reference_ib_lowres, needle_reference_ib_highres]





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
###################################################################################






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

def find_needletip_and_target_locations_MANUAL(image):
    print("Please click the needle tip position")
    needletip_location = select_point(image)
    print("Please click the lamella target position")
    target_location = select_point(image)
    return needletip_location, target_location




#######################################################################################################################################
def find_tip_in_needle_image(imageAdourned, median_smoothing=3, show=True):
    print('needle tip search')
    image_data = imageAdourned.data
    if median_smoothing is not None:
        image_data = ndi.median_filter(image_data, size=median_smoothing)
    height, width = image_data.shape
    try:
        pixelsize_x = image.metadata.binary_result.pixel_size.x
        pixelsize_y = image.metadata.binary_result.pixel_size.y
        print('pixel size = ', pixelsize_x)
    except AttributeError:
        extent_kwargs = [-(width / 2), +(width / 2), -(height / 2), +(height / 2)]
        ax.set_xlabel("Distance from origin (pixels)")

    harrisim = compute_harris_response(image_data)
    filtered_coords = get_harris_points(harrisim,6)
    right_outermost_point = max(filtered_coords, key=itemgetter(1))

    print('outermost point pixel', right_outermost_point)
    cen = np.asarray(image_data.shape) / 2
    tip_shift_from_center = np.array(cen - np.array(right_outermost_point), int)
    print("Tip shift from the image center is = " + str(tip_shift_from_center))
    print("Tip shift from the image center is by ", tip_shift_from_center[1], '; Y-shifted by ', tip_shift_from_center[0])
    x_shift = -1 * tip_shift_from_center[1] * pixelsize_x
    y_shift = +1 * tip_shift_from_center[0] * pixelsize_y
    print("X-shift =  {} meters".format(x_shift))
    print("Y-shift =  {} meters".format(y_shift))
    if show:
        plt.figure()
        plt.gray()
        plt.imshow(image_data)
        plt.plot( [p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'bo'  )
        plt.plot( [right_outermost_point[1]], [right_outermost_point[0]],   'ro')
        plt.show()
    return x_shift, y_shift



def find_needle_tip_shift_in_image_ELECTRON(needle_with_sample_Adorned, needle_reference_Adorned, show=False, median_smoothing=2):
    try:
        pixelsize_x = needle_with_sample_Adorned.metadata.binary_result.pixel_size.x
        pixelsize_y = needle_with_sample_Adorned.metadata.binary_result.pixel_size.y
    except AttributeError:
        pixelsize_x = 1
        pixelsize_y = 1
    ### Find the tip using corner-finding algorithm
    needle_reference   = needle_reference_Adorned.data
    needle_with_sample = needle_with_sample_Adorned.data
    field_width  = pixelsize_x  * needle_with_sample_Adorned.width
    height, width = needle_reference.shape # search for the tip in only the left half of the image
    harrisim = compute_harris_response( ndi.median_filter(needle_reference[:, 0:width//2], size=1) )
    #harrisim = compute_harris_response(needle_reference)
    filtered_coords_ref = get_harris_points(harrisim,4)
    right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))
    #############
    if show:
        plt.figure(1)
        plt.imshow(needle_reference, cmap='gray')
        plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
        plt.plot( [right_outermost_point_ref[1]], [right_outermost_point_ref[0]],   'ro')

    ### Find the tip from binarized image (mask) and corner-finding algorithm
    filt   = ndi.median_filter(needle_reference, size=5)
    thresh = threshold_otsu(filt)
    binary = filt > thresh
    mask   = gaussian(binary_dilation(binary, iterations=15), 5)
    mask_binary = (mask >= 0.51).astype(int)
    needle_ref_masked = needle_reference * mask_binary
    ysize, xsize = mask.shape
    harrisim = compute_harris_response(mask_binary[:, 0:width//2])
    filtered_coords_mask_ref = get_harris_points(harrisim,4)
    right_outermost_point_mask_ref = max(filtered_coords_mask_ref, key=itemgetter(1))
    ####
    if show:
        plt.figure(2)
        plt.imshow(mask_binary,   cmap='gray',     alpha=1)
        plt.imshow(needle_reference, cmap='Oranges_r',   alpha=0.5)
        plt.plot([p[1] for p in filtered_coords_mask_ref], [p[0] for p in filtered_coords_mask_ref], 'bo')
        plt.plot([right_outermost_point_mask_ref[1]], [right_outermost_point_mask_ref[0]], 'ro')
        plt.plot([right_outermost_point_ref[1]], [right_outermost_point_ref[0]], 'rd')

    def R(p1, p2):
        return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) # find distance between two points
    # two tips found, if the tip coordinate from the noisy real needle image is way off, rely on the tip found from the binarized image
    #if R(right_outermost_point_ref, right_outermost_point_mask_ref  ) <= 20:
    #    right_outermost_point = right_outermost_point_ref
    #else:
    #    right_outermost_point = right_outermost_point_mask_ref
    # if ion beam - use harris points, if electron - check R bwt two points and select corner from
    right_outermost_point = right_outermost_point_ref

    # tip position in the reference image :
    old_tip_x = right_outermost_point[0]
    old_tip_y = right_outermost_point[1]

    xmin = min(old_tip_y, mask.shape[1] - old_tip_y)
    ymin = min(old_tip_x, mask.shape[0] - old_tip_x)
    rmin = min(xmin, ymin)
    Dmin = 2 * rmin
    ####################
    cmask = circ_mask(size=(Dmin, Dmin), radius=Dmin // 2 - 15, sigma=10)  # circular mask
    CMASK = np.zeros(needle_reference.shape)
    CMASK[old_tip_x - Dmin // 2: old_tip_x + Dmin // 2, old_tip_y - Dmin // 2: old_tip_y + Dmin // 2] = cmask
    needle_reference_norm = ( needle_reference - np.mean(needle_reference) ) / np.std(needle_reference)
    reference_circ_norm = needle_reference_norm * CMASK * mask
    ####################
    xmin = min(old_tip_x, mask.shape[1] - old_tip_x)
    ymin = min(old_tip_y, mask.shape[0] - old_tip_y)
    ELLPS_MASK = np.zeros(needle_reference.shape)
    elps_mask = ellipse_mask(size=(xmin*2, ymin*2), radius1=xmin-15, radius2=ymin-15, sigma=10)
    ELLPS_MASK[old_tip_y - ymin : old_tip_y + ymin, old_tip_x - xmin : old_tip_x + xmin] = elps_mask
    reference_elps_norm = needle_reference_norm * ELLPS_MASK * mask
    ####################
    lowpass_pixels  = int( max(needle_reference.shape) / 12 )
    highpass_pixels = int( max(needle_reference.shape)/ 256 )
    print(': High pass filter = ', lowpass_pixels, '; low pass filter = ', highpass_pixels)
    needle_with_sample_norm = ( needle_with_sample - np.mean(needle_with_sample) ) / np.std(needle_with_sample)
    xcorr = crosscorrelation(needle_with_sample_norm, needle_reference_norm, bp='yes', lp=lowpass_pixels, hp=highpass_pixels, sigma=2)
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    print('\n', maxX, maxY)
    cen = np.asarray(xcorr.shape) / 2
    print('centre = ', cen)
    err = np.array(cen - [maxX, maxY], int)
    print("Shift between 1 and 2 is = " + str(err))
    print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
    new_tip_x = old_tip_x - err[0]
    new_tip_y = old_tip_y - err[1]
    x_shift = +1 * ( cen[1] - new_tip_y ) * pixelsize_x
    y_shift = -1 * ( cen[0] - new_tip_x ) * pixelsize_y
    print("X-shift to the image centre =  {} meters".format(x_shift))
    print("Y-shift to the image centre =  {} meters".format(y_shift))
    if show:
        plt.figure()
        plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
        plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
        plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='original position')
        plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='current position')
        plt.legend()

    return x_shift, y_shift # shift of location of the tip


def find_needle_tip_shift_in_image_ION(needle_with_sample_Adorned, needle_reference_Adorned, show=False, median_smoothing=2):
    try:
        pixelsize_x = needle_with_sample_Adorned.metadata.binary_result.pixel_size.x
        pixelsize_y = needle_with_sample_Adorned.metadata.binary_result.pixel_size.y
    except AttributeError:
        pixelsize_x = 1
        pixelsize_y = 1
    ### Find the tip using corner-finding algorithm
    needle_reference   = needle_reference_Adorned.data
    needle_with_sample = needle_with_sample_Adorned.data
    field_width  = pixelsize_x  * needle_with_sample_Adorned.width
    height, width = needle_reference.shape # search for the tip in only the left half of the image
    harrisim = compute_harris_response( ndi.median_filter(needle_reference[:, 0:width//2], size=5) )
    #harrisim = compute_harris_response(needle_reference)
    filtered_coords_ref = get_harris_points(harrisim,4)
    right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))
    topmost_point_ref         = min(filtered_coords_ref, key=itemgetter(0))
    #############
    if show:
        plt.figure(1)
        plt.imshow(needle_reference, cmap='gray')
        plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
        plt.plot( [topmost_point_ref[1]], [topmost_point_ref[0]],   'ro')

    ### Find the tip from binarized image (mask) and corner-finding algorithm
    filt   = ndi.median_filter(needle_reference, size=2)
    thresh = threshold_otsu(filt)
    binary = filt > thresh
    mask   = gaussian(binary_dilation(binary, iterations=15), 5)
    mask_binary = (mask >= 0.51).astype(int)
    needle_ref_masked = needle_reference * mask_binary
    ysize, xsize = mask.shape
    harrisim = compute_harris_response(mask_binary[:, 0:width//2])
    filtered_coords_mask_ref = get_harris_points(harrisim,4)
    right_outermost_point_mask_ref = max(filtered_coords_mask_ref, key=itemgetter(1))
    topmost_point_mask_ref         = min(filtered_coords_mask_ref, key=itemgetter(0))
    ####
    if show:
        plt.figure(2)
        plt.imshow(mask_binary,   cmap='gray',     alpha=1)
        plt.imshow(needle_reference, cmap='Oranges_r',   alpha=0.5)
        plt.plot([p[1] for p in filtered_coords_mask_ref], [p[0] for p in filtered_coords_mask_ref], 'bo')
        plt.plot([topmost_point_mask_ref[1]], [topmost_point_mask_ref[0]], 'ro')
        plt.plot([topmost_point_ref[1]], [topmost_point_ref[0]], 'rd')

    def R(p1, p2):
        return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) # find distance between two points
    # two tips found, if the tip coordinate from the noisy real needle image is way off, rely on the tip found from the binarized image
    #if R(right_outermost_point_ref, right_outermost_point_mask_ref  ) <= 20:
    #    right_outermost_point = right_outermost_point_ref
    #else:
    #    right_outermost_point = right_outermost_point_mask_ref
    # if ion beam - use harris points, if electron - check R bwt two points and select corner from
    right_outermost_point = right_outermost_point_ref
    topmost_point         = topmost_point_ref

    # tip position in the reference image :
    old_tip_x = topmost_point[0]
    old_tip_y = topmost_point[1]

    xmin = min(old_tip_y, mask.shape[1] - old_tip_y)
    ymin = min(old_tip_x, mask.shape[0] - old_tip_x)
    rmin = min(xmin, ymin)
    Dmin = 2 * rmin
    ####################
    cmask = circ_mask(size=(Dmin, Dmin), radius=Dmin // 2 - 15, sigma=10)  # circular mask
    CMASK = np.zeros(needle_reference.shape)
    CMASK[old_tip_x - Dmin // 2: old_tip_x + Dmin // 2, old_tip_y - Dmin // 2: old_tip_y + Dmin // 2] = cmask
    needle_reference_norm = ( needle_reference - np.mean(needle_reference) ) / np.std(needle_reference)
    reference_circ_norm = needle_reference_norm * CMASK * mask
    ####################
    xmin = min(old_tip_x, mask.shape[1] - old_tip_x)
    ymin = min(old_tip_y, mask.shape[0] - old_tip_y)
    ELLPS_MASK = np.zeros(needle_reference.shape)
    elps_mask = ellipse_mask(size=(xmin*2, ymin*2), radius1=xmin-15, radius2=ymin-15, sigma=10)
    ELLPS_MASK[old_tip_y - ymin : old_tip_y + ymin, old_tip_x - xmin : old_tip_x + xmin] = elps_mask
    reference_elps_norm = needle_reference_norm * ELLPS_MASK * mask
    ####################
    lowpass_pixels  = int( max(needle_reference.shape) / 6 )
    highpass_pixels = int( max(needle_reference.shape)/ 64 )
    print(': High pass filter = ', lowpass_pixels, '; low pass filter = ', highpass_pixels)
    needle_with_sample_norm = ( needle_with_sample - np.mean(needle_with_sample) ) / np.std(needle_with_sample)
    xcorr = crosscorrelation(needle_with_sample_norm, needle_reference_norm, bp='yes', lp=lowpass_pixels, hp=highpass_pixels, sigma=10)
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    print('\n', maxX, maxY)
    cen = np.asarray(xcorr.shape) / 2
    print('centre = ', cen)
    err = np.array(cen - [maxX, maxY], int)
    print("Shift between 1 and 2 is = " + str(err))
    print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
    new_tip_x = old_tip_x - err[0]
    new_tip_y = old_tip_y - err[1]
    x_shift = ( cen[1] - new_tip_y ) * pixelsize_x
    y_shift = ( cen[0] - new_tip_x ) * pixelsize_y
    print("X-shift to the image centre =  {} meters".format(x_shift))
    print("Y-shift to the image centre =  {} meters".format(y_shift))
    if show:
        plt.figure()
        plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
        plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
        plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='original position')
        plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='current position')
        plt.legend()

    return x_shift, y_shift # shift of location of the tip
#######################################################################################################################################





def manual_needle_movement_in_z(microscope):
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator
    ion_image = new_ion_image(microscope, settings=GrabFrameSettings(dwell_time=500e-9, resolution="1536x1024"))  # TODO: user input imaging settings
    print("Please click the needle tip position")
    needletip_location = select_point(ion_image)
    print("Please click the lamella target position")
    target_location = select_point(ion_image)
    # Calculate movment
    z_safety_buffer = 400e-9  # in meters TODO: yaml user input
    z_distance = -( target_location[1] - needletip_location[1] / np.sin(np.deg2rad(52)) ) - z_safety_buffer
    z_move = z_corrected_needle_movement(z_distance, stage.current_position.t)
    needle.relative_move(z_move)



def liftout_lamella(microscope, settings, needle_reference_imgs):
    microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # can't be smaller than 150e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # can't be smaller than 150e-6
    needle_reference_eb, needle_reference_ib = needle_reference_imgs
    # needletip_ref_location_eb = ??? TODO: automated needletip identification
    # needletip_ref_location_ib = ??? TODO: automated needletip identification
    park_position = move_needle_to_liftout_position(microscope)
    manual_needle_movement_in_xy(microscope, move_in_x=False)
    manual_needle_movement_in_z(microscope)
    manual_needle_movement_in_xy(microscope)
    sputter_platinum(microscope, sputter_time=10)  # TODO: yaml user input for sputtering application file choice
    mill_to_sever_jcut(microscope, settings['jcut'], confirm=False)  # TODO: yaml user input for jcut milling current
    retract_needle(microscope, park_position)
    needle_reference_images_with_lamella = needle_reference_images(microscope, move_needle_to="landing", dwell_time=2e-6)
    retract_needle(microscope, park_position)
    return needle_reference_images_with_lamella


import matplotlib.pyplot as plt
import numpy as np
from patrick.utils import load_model, model_inference, detect_and_draw_lamella_and_needle, scale_invariant_coordinates, calculate_distance_between_points, parse_metadata, show_overlay
from PIL import Image


def lamella_shift_from_img(img, show=False):

    weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\12_04_2021_10_32_23_model.pt"
    model = load_model(weights_file=weights_file)

    img_orig = np.asarray(img.data)

    # model inference + display
    img, rgb_mask = model_inference(model, None, img=img_orig)

    # detect and draw lamella centre, and needle tip
    (
        lamella_centre_px,
        rgb_mask_lamella,
        needle_tip_px,
        rgb_mask_needle,
        rgb_mask_combined,
    ) = detect_and_draw_lamella_and_needle(rgb_mask, cols_masks=None)
    # TODO: this col masks still needs to be extracted out

    # scale invariant coordinatess
    scaled_lamella_centre_px, scaled_needle_tip_px = scale_invariant_coordinates(
        needle_tip_px, lamella_centre_px, rgb_mask_combined
    )

    # calculate distance between features
    (
        distance,
        vertical_distance,
        horizontal_distance,
    ) = calculate_distance_between_points(scaled_lamella_centre_px, scaled_needle_tip_px)

    # prediction overlay
    img_overlay = show_overlay(img, rgb_mask_combined)

    # df = parse_metadata(fname)

    img_overlay_resized = Image.fromarray(img_overlay).resize((img.shape[1], img.shape[0]))

    if show:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img_orig, cmap='Blues_r', alpha=1)
        ax.imshow(img_overlay_resized, cmap='Oranges_r', alpha=0.5)

        plt.show()

    # TODO: fix this
    if scaled_lamella_centre_px is None:
        scaled_lamella_centre_px = (0, 0)

    return 0.5 - scaled_lamella_centre_px[0]


if __name__ == "__main__":

    #weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\12_04_2021_10_32_23_model.pt"
    weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\fresh_full_n10.pt"
    model = load_model(weights_file=weights_file)

    if 1:
        eb_,  ib_  = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
        storage.SaveImage(eb_, id='A_eb_')
        storage.SaveImage(ib_, id='A_ib_')

        img_orig = np.asarray(ib_.data)

        # model inference + display
        img, rgb_mask = model_inference(model, None, img=img_orig)

        # detect and draw lamella centre, and needle tip
        (
            lamella_centre_px,
            rgb_mask_lamella,
            needle_tip_px,
            rgb_mask_needle,
            rgb_mask_combined,
        ) = detect_and_draw_lamella_and_needle(rgb_mask, cols_masks=None)
        # TODO: this col masks still needs to be extracted out

        # scale invariant coordinatess
        scaled_lamella_centre_px, scaled_needle_tip_px = scale_invariant_coordinates(
            needle_tip_px, lamella_centre_px, rgb_mask_combined
        )

        # calculate distance between features
        (
            distance,
            vertical_distance,
            horizontal_distance,
        ) = calculate_distance_between_points(scaled_lamella_centre_px, scaled_needle_tip_px)

        # prediction overlay
        img_overlay = show_overlay(img, rgb_mask_combined)

        # df = parse_metadata(fname)

        img_overlay_resized = Image.fromarray(img_overlay).resize((img.shape[1], img.shape[0]))

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img_orig, cmap='Blues_r', alpha=1)
        ax.imshow(img_overlay_resized, cmap='Oranges_r', alpha=0.5)

        plt.show()





    if 0:
        #storage.NewRun(prefix='test_machine_learning')
        ip_address = '10.0.0.1'
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        from autoscript_sdb_microscope_client.structures import ManipulatorPosition
        from autoscript_sdb_microscope_client.structures import StagePosition
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        stage  = microscope.specimen.stage
        needle = microscope.specimen.manipulator
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=0.5e-6)

        microscope.auto_functions.run_auto_focus()
        microscope.specimen.stage.link()

        eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=80e-6, image_settings=image_settings)
        storage.SaveImage(eb_lowres, id='A_eb_lowres')
        storage.SaveImage(ib_lowres, id='A_ib_lowres')

        eb_distance = lamella_shift_from_img(eb_lowres, show=True)
        ib_distance = lamella_shift_from_img(ib_lowres, show=True)

        print(f"Electron Beam: {eb_distance}")
        print(f"Ion Beam: {ib_distance}")

        pixelsize_x   = ib_lowres.metadata.binary_result.pixel_size.x
        field_width   = pixelsize_x  * ib_lowres.width
        dy_meters_ion = ib_distance * field_width

        if 1: # z correction
            tilt_radians = stage.current_position.t
            delta_z = -np.cos(tilt_radians) * dy_meters_ion
            stage.relative_move(StagePosition(z=delta_z))



            # elctron dy shift
            eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
            storage.SaveImage(eb_lowres, id='A_eb_lowres_moved1')
            storage.SaveImage(ib_lowres, id='A_ib_lowres_moved1')

            eb_distance = lamella_shift_from_img(eb_lowres, show=True)
            ib_distance = lamella_shift_from_img(ib_lowres, show=True)

            print(f"Electron Beam: {eb_distance}")
            print(f"Ion Beam: {ib_distance}")

            pixelsize_x   = eb_lowres.metadata.binary_result.pixel_size.x
            field_width   = pixelsize_x  * eb_lowres.width
            dy_meters_elec = eb_distance * field_width
            stage.relative_move(StagePosition(y=dy_meters_elec))



            # ion again
            eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
            storage.SaveImage(eb_lowres, id='A_eb_lowres_moved2')
            storage.SaveImage(ib_lowres, id='A_ib_lowres_moved2')

            eb_distance = lamella_shift_from_img(eb_lowres, show=True)
            ib_distance = lamella_shift_from_img(ib_lowres, show=True)

            print(f"Electron Beam: {eb_distance}")
            print(f"Ion Beam: {ib_distance}")

            pixelsize_x   = ib_lowres.metadata.binary_result.pixel_size.x
            field_width   = pixelsize_x  * ib_lowres.width
            dy_meters_ion = ib_distance * field_width

            tilt_radians = stage.current_position.t
            delta_z = -np.cos(tilt_radians) * dy_meters_ion
            stage.relative_move(StagePosition(z=delta_z))



        if 0: # y,z correction
            yz_move = y_corrected_stage_movement(dy_meters_ion, stage.current_position.t, beam_type=BeamType.ION)
            print('relative movement of the the stage by Y-Z:', yz_move)
            stage.relative_move(yz_move)

        eb_lowres,  ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
        storage.SaveImage(eb_lowres, id='B_eb_lowres')
        storage.SaveImage(ib_lowres, id='B_ib_lowres')

        storage.step_counter += 1






        plt.show()


        #storage.SaveImage(needle_ref_eb_lowres_nobg, id='A_ref_eb_lowres')
        #park_position = move_needle_to_liftout_position(microscope)
        #################### LOW & HIGH RES IMAGES ########################
        #take images, eb & ib, with the sample+needle
        #pixelsize_x_lowres  = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.x
        #field_width_lowres  = pixelsize_x_lowres  * needle_ref_eb_lowres_nobg.width
        #microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
        #microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
        #microscope.imaging.set_active_view(1)
        #autocontrast(microscope, beam_type=BeamType.ELECTRON)
        #needle_eb_lowres_with_lamella = microscope.imaging.grab_frame(image_settings)

        ############ FIND dx, dy from HIGH_RES ELECTRON images ############
        #retract_needle(microscope, park_position)
