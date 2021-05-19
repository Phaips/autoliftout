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







if __name__ == "__main__":
    TEST = 3.3

    if 0:
        storage.NewRun(prefix='test_needle_landing_lowDose')
        ip_address = '10.0.0.1'
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        from autoscript_sdb_microscope_client.structures import ManipulatorPosition
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        stage  = microscope.specimen.stage
        needle = microscope.specimen.manipulator
        sample_stage_out = StagePosition(x=-0.002507, y=0.025962792, z=0.0039559049)
        microscope.specimen.stage.absolute_move(sample_stage_out)

        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
        needle_eb_highres, needle_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width= 80e-6, image_settings=image_settings)
        needle_eb_lowres,  needle_ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
        ###
        storage.SaveImage(needle_eb_lowres,  id='_needle_eb_lowres' )
        storage.SaveImage(needle_eb_highres, id='_needle_eb_highres')
        storage.SaveImage(needle_ib_lowres,  id='_needle_ib_lowres' )
        storage.SaveImage(needle_ib_highres, id='_needle_ib_highres')
        storage.step_counter += 1

        #park_position = move_needle_to_liftout_position(microscope)
        #park_position = move_needle_to_landing_position(microscope)
        #park_position = needle.current_position
        #retract_needle(microscope, park_position)

        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=0.2e-6)  # TODO: user input resolution
        eb_lowres_reference,  ib_lowres_reference  = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings) # TODO: yaml use input
        eb_highres_reference, ib_highres_reference = take_electron_and_ion_reference_images(microscope, hor_field_width=50e-6,  image_settings=image_settings) # TODO: yaml use input
        reference_images_low_and_high_res = (eb_lowres_reference, eb_highres_reference, ib_lowres_reference, ib_highres_reference)
        storage.SaveImage(eb_lowres_reference,  id='_ref_eb_lowres' )
        storage.SaveImage(eb_highres_reference, id='_ref_eb_highres')
        storage.SaveImage(ib_lowres_reference,  id='_ref_ib_lowres' )
        storage.SaveImage(ib_highres_reference, id='_ref_ib_highres')
        storage.step_counter += 1


    # try this
    #sx = ndimage.sobel(im, axis=0, mode='constant')
    #sy = ndimage.sobel(im, axis=1, mode='constant')
    #sob = np.hypot(sx, sy)

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


    if TEST == 4:
        storage.NewRun(prefix='test_needle_landing_lowDose')
        ip_address = '10.0.0.1'
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        from autoscript_sdb_microscope_client.structures import ManipulatorPosition
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        stage  = microscope.specimen.stage
        needle = microscope.specimen.manipulator
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=2e-6)
        NEEDLE_REF_NOBG_IMAGES = needle_reference_images(microscope)
        needle_ref_eb_lowres_nobg  = NEEDLE_REF_NOBG_IMAGES[0]
        needle_ref_eb_highres_nobg = NEEDLE_REF_NOBG_IMAGES[1]
        needle_ref_ib_lowres_nobg  = NEEDLE_REF_NOBG_IMAGES[2]
        needle_ref_ib_highres_nobg = NEEDLE_REF_NOBG_IMAGES[3]
        quick_plot(needle_ref_eb_lowres_nobg)
        quick_plot(needle_ref_ib_lowres_nobg)
        #needle_ref_eb_lowres_nobg.save('ref_eb_lowres.tif')
        #needle_ref_eb_highres_nobg.save('ref_eb_highres.tif')
        #needle_ref_ib_lowres_nobg.save('ref_ib_lowres.tif')
        #needle_ref_ib_highres_nobg.save('ref_ib_highres.tif')
        storage.SaveImage(needle_ref_eb_lowres_nobg, id='A_ref_eb_lowres')
        storage.SaveImage(needle_ref_eb_highres_nobg, id='A_ref_eb_highres')
        storage.SaveImage(needle_ref_ib_lowres_nobg, id='A_ref_ib_lowres')
        storage.SaveImage(needle_ref_ib_highres_nobg, id='A_ref_ib_highres')


        if 0:
            needle_reference_images_nobg = [needle_ref_eb_lowres_nobg, needle_ref_eb_highres_nobg, needle_ref_ib_lowres_nobg, needle_ref_ib_highres_nobg]
            needle_reference_imgs = NEEDLE_REF_NOBG_IMAGES
            needle_ref_eb_lowres_nobg  = needle_reference_imgs[0]
            needle_ref_eb_highres_nobg = needle_reference_imgs[1]
            needle_ref_ib_lowres_nobg  = needle_reference_imgs[2]
            needle_ref_ib_highres_nobg = needle_reference_imgs[3]
            storage.SaveImage(needle_ref_eb_lowres_nobg, id='A_ref_eb_needle_nobg_lowres')
            storage.SaveImage(needle_ref_eb_highres_nobg, id='A_ref_eb_needle_nobg_highres')
            storage.SaveImage(needle_ref_ib_lowres_nobg, id='A_ref_ib_needle_nobg_lowres')
            storage.SaveImage(needle_ref_ib_highres_nobg, id='A_ref_ib_needle_nobg_highres')
            storage.step_counter += 1

        ######
        ######
        yy = input('Move the sample in and press ENTER...')
        ######
        ######
        park_position = move_needle_to_liftout_position(microscope)

        #################### LOW & HIGH RES IMAGES ########################
        #take images, eb & ib, with the sample+needle
        pixelsize_x_lowres  = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.x
        field_width_lowres  = pixelsize_x_lowres  * needle_ref_eb_lowres_nobg.width
        microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
        microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
        microscope.imaging.set_active_view(1)
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        needle_eb_lowres_with_lamella = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        autocontrast(microscope, beam_type=BeamType.ION)
        needle_ib_lowres_with_lamella  = microscope.imaging.grab_frame(image_settings)
        pixelsize_x_highres  = needle_ref_eb_highres_nobg.metadata.binary_result.pixel_size.x
        field_width_highres  = pixelsize_x_highres  * needle_ref_eb_highres_nobg.width
        microscope.beams.ion_beam.horizontal_field_width.value      = field_width_highres
        microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
        microscope.imaging.set_active_view(1)
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        needle_eb_highres_with_lamella = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        autocontrast(microscope, beam_type=BeamType.ION)
        needle_ib_highres_with_lamella  = microscope.imaging.grab_frame(image_settings)
        #needle_eb_lowres_with_lamella.save( 'sample_eb_lowres.tif')
        #needle_eb_highres_with_lamella.save('sample_eb_highres.tif')
        #needle_ib_lowres_with_lamella.save( 'sample_ib_lowres.tif')
        #needle_ib_highres_with_lamella.save('sample_ib_highres.tif')
        storage.SaveImage(needle_eb_lowres_with_lamella, id='B_sample_eb_lowres')
        storage.SaveImage(needle_eb_highres_with_lamella, id='B_sample_eb_highres')
        storage.SaveImage(needle_ib_lowres_with_lamella, id='B_sample_ib_lowres')
        storage.SaveImage(needle_ib_highres_with_lamella, id='B_sample_ib_highres')
        storage.step_counter +=1
        ############ FIND dx, dy from HIGH_RES ELECTRON images ############
        x_shift, y_shift = find_needletip_shift_in_image_ELECTRON(needle_eb_lowres_with_lamella, needle_ref_eb_lowres_nobg, show=True, median_smoothing=2)
        xcorrection = 1e-6
        ycorrection = 2e-6
        x_move = x_corrected_needle_movement(x_shift + xcorrection)
        y_move = y_corrected_needle_movement(y_shift + ycorrection, stage.current_position.t)
        print('x_move = ', x_move, ';\ny_move = ', y_move)
        yy = input('press ENTER to move the needle...')
        needle.relative_move(x_move)
        needle.relative_move(y_move)

        #################### HIGH RES IMAGES ########################
        pixelsize_x_lowres  = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.x
        field_width_lowres  = pixelsize_x_lowres  * needle_ref_eb_lowres_nobg.width
        microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
        microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
        microscope.imaging.set_active_view(1)
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        needle_eb_lowres_with_lamella_shifted = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        autocontrast(microscope, beam_type=BeamType.ION)
        needle_ib_lowres_with_lamella_shifted  = microscope.imaging.grab_frame(image_settings)
        pixelsize_x_highres  = needle_ref_eb_highres_nobg.metadata.binary_result.pixel_size.x
        field_width_highres  = pixelsize_x_highres  * needle_ref_eb_highres_nobg.width
        microscope.beams.ion_beam.horizontal_field_width.value      = field_width_highres
        microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
        microscope.imaging.set_active_view(1)
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        needle_eb_highres_with_lamella_shifted = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        autocontrast(microscope, beam_type=BeamType.ION)
        needle_ib_highres_with_lamella_shifted  = microscope.imaging.grab_frame(image_settings)
        storage.SaveImage(needle_eb_lowres_with_lamella_shifted, id='C_sample_eb_lowres_shifted')
        storage.SaveImage(needle_eb_highres_with_lamella_shifted, id='C_sample_eb_highres_shifted')
        storage.SaveImage(needle_ib_lowres_with_lamella_shifted, id='C_sample_ib_lowres_shifted')
        storage.SaveImage(needle_ib_highres_with_lamella_shifted, id='C_ref_ib_highres_shifted')
        ############ FIND dx, dy from HIGH_RES ELECTRON images ############
        x_shift, y_shift = find_needletip_shift_in_image_ELECTRON(needle_eb_highres_with_lamella_shifted, needle_ref_eb_highres_nobg, show=True, median_smoothing=2)
        x_move = x_corrected_needle_movement(x_shift + xcorrection)
        y_move = y_corrected_needle_movement(y_shift + ycorrection, stage.current_position.t)
        print('x_move = ', x_move, ';\ny_move = ', y_move)
        yy = input('press ENTER to move the needle...')
        needle.relative_move(x_move)
        needle.relative_move(y_move)

        #############################  find dz from LOW RES ION  ############################
        #take images, eb & ib, with the sample+needle - correct X of the needle from lowres eb
        ### RETAKE THE IMAGES AFTER dx,dy SHIFT
        pixelsize_x_lowres  = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.x
        field_width_lowres  = pixelsize_x_lowres  * needle_ref_eb_lowres_nobg.width
        microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
        microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
        microscope.imaging.set_active_view(1)
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        needle_eb_lowres_with_lamella_shifted = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        autocontrast(microscope, beam_type=BeamType.ION)
        needle_ib_lowres_with_lamella_shifted  = microscope.imaging.grab_frame(image_settings)
        pixelsize_x_highres  = needle_ref_eb_highres_nobg.metadata.binary_result.pixel_size.x
        field_width_highres  = pixelsize_x_highres  * needle_ref_eb_highres_nobg.width
        microscope.beams.ion_beam.horizontal_field_width.value      = field_width_highres
        microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
        microscope.imaging.set_active_view(1)
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        needle_eb_highres_with_lamella_shifted = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        autocontrast(microscope, beam_type=BeamType.ION)
        needle_ib_highres_with_lamella_shifted  = microscope.imaging.grab_frame(image_settings)
        storage.SaveImage(needle_eb_lowres_with_lamella_shifted, id='D_sample_eb_lowres_shifted2')
        storage.SaveImage(needle_eb_highres_with_lamella_shifted, id='D_sample_eb_highres_shifted2')
        storage.SaveImage(needle_ib_lowres_with_lamella_shifted, id='D_sample_ib_lowres_shifted2')
        storage.SaveImage(needle_ib_highres_with_lamella_shifted, id='D_ref_ib_highres_shifted2')
        ##########
        x_shift, y_shift = find_needletip_shift_in_image_ION(needle_ib_lowres_with_lamella_shifted, needle_ref_ib_lowres_nobg, show=True, median_smoothing=2)
        stage_tilt = stage.current_position.t
        print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...' )
        print('cos(t) = ', np.cos(stage_tilt) )
        z_distance = y_shift / np.cos(stage_tilt)
        zy_move_half = z_corrected_needle_movement(z_distance/2, stage_tilt)
        yy = input('press ENTER to move the needle in Z by HALF distance...')
        needle.relative_move(zy_move_half)
        yy = input('press ENTER to move the needle in Z by HALF distance...Landing now.')
        needle.relative_move(zy_move_half)
        print('The needle has landed.')


        #############################  LANDED, take pictures  ############################
        ### RETAKE THE IMAGES AFTER dx,dy,dz SHIFT
        pixelsize_x_lowres  = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.x
        field_width_lowres  = pixelsize_x_lowres * needle_ref_eb_lowres_nobg.width
        microscope.beams.ion_beam.horizontal_field_width.value      = field_width_lowres
        microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres
        microscope.imaging.set_active_view(1)
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        needle_eb_lowres_with_lamella_shifted = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        autocontrast(microscope, beam_type=BeamType.ION)
        needle_ib_lowres_with_lamella_shifted  = microscope.imaging.grab_frame(image_settings)
        pixelsize_x_highres  = needle_ref_eb_highres_nobg.metadata.binary_result.pixel_size.x
        field_width_highres  = pixelsize_x_highres  * needle_ref_eb_highres_nobg.width
        microscope.beams.ion_beam.horizontal_field_width.value      = field_width_highres
        microscope.beams.electron_beam.horizontal_field_width.value = field_width_highres
        microscope.imaging.set_active_view(1)
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        needle_eb_highres_with_lamella_shifted = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        autocontrast(microscope, beam_type=BeamType.ION)
        needle_ib_highres_with_lamella_shifted  = microscope.imaging.grab_frame(image_settings)
        storage.SaveImage(needle_eb_lowres_with_lamella_shifted,  id='E_sample_eb_lowres_landed')
        storage.SaveImage(needle_eb_highres_with_lamella_shifted, id='E_sample_eb_highres_landed')
        storage.SaveImage(needle_ib_lowres_with_lamella_shifted,  id='E_sample_ib_lowres_landed')
        storage.SaveImage(needle_ib_highres_with_lamella_shifted, id='E_ref_ib_highres_landed')

        if 0:
            new_eb,  new_ib  = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
            storage.SaveImage(new_eb,  id='eb_')
            storage.SaveImage(new_ib,  id='ib_')
            storage.step_counter +=1

        if 0:
            eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
            eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
            ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
            ib_contrast = storage.settings["machine_learning"]["ib_contrast"]
            microscope.beams.ion_beam.horizontal_field_width.value      = 80e-6 # hor_field_width
            microscope.beams.electron_beam.horizontal_field_width.value = 80e-6 # hor_field_width
            new_eb = new_electron_image(microscope, settings=image_settings, brightness=eb_brightness, contrast=eb_contrast)
            new_ib = new_ion_image(microscope, settings = image_settings, brightness=ib_brightness, contrast=ib_contrast)
            storage.SaveImage(new_eb,  id='eb__BC')
            storage.SaveImage(new_ib,  id='ib__BC')
            storage.step_counter +=1


        # TAKE NEEDLE z_UP (>30 MICRONS), TAKE GIS OUT, RESTRACT TO PARKING
        z_move_out_from_trench = z_corrected_needle_movement(30e-6, stage_tilt)
        needle.relative_move(z_move_out_from_trench)
        retract_needle(microscope, park_position)


        ######################### ATTACHING #######################################
        '''# TAKE IMAGES OF SAMPLE GLUED TO THE NEEDLE
        ATTCHED_REF_NOBG_IMAGES = needle_reference_images(microscope, dwell_time=1e-6)
        attached_ref_eb_lowres_nobg  = ATTCHED_REF_NOBG_IMAGES[0]
        attached_ref_eb_highres_nobg = ATTCHED_REF_NOBG_IMAGES[1]
        attached_ref_ib_lowres_nobg  = ATTCHED_REF_NOBG_IMAGES[2]
        attached_ref_ib_highres_nobg = ATTCHED_REF_NOBG_IMAGES[3]
        storage.SaveImage(attached_ref_eb_lowres_nobg, id='F_ref_eb_lowres_attached')
        storage.SaveImage(attached_ref_eb_highres_nobg, id='F_ref_eb_highres_attached')
        storage.SaveImage(attached_ref_ib_lowres_nobg, id='F_ref_ib_lowres_attached')
        storage.SaveImage(attached_ref_ib_highres_nobg, id='F_ref_ib_highres_attached')
        '''



        ########################### LANDING ON POST ###########################
        # move the stage out, take reference images with no background
        move_sample_stage_out(microscope)
        park_position = move_needle_to_landing_position(microscope)
        image_settings = GrabFrameSettings(resolution="3072x2048", dwell_time=1e-6)
        #ref_landingPos_eb_lowres,  ref_landingPos_ib_lowres  = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
        #ref_landingPos_eb_highres, ref_landingPos_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width= 80e-6, image_settings=image_settings)
        image_settings_electron = GrabFrameSettings(resolution="3072x2048", dwell_time=0.5e-6)
        image_settings_ion      = GrabFrameSettings(resolution="3072x2048", dwell_time=0.2e-6)
        microscope.beams.ion_beam.horizontal_field_width.value      = 80e-6
        microscope.beams.electron_beam.horizontal_field_width.value = 80e-6
        microscope.imaging.set_active_view(1)
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        ref_landingPos_eb_highres = microscope.imaging.grab_frame(image_settings_electron)
        microscope.imaging.set_active_view(2)
        autocontrast(microscope, beam_type=BeamType.ION)
        ref_landingPos_ib_highres  = microscope.imaging.grab_frame(image_settings_ion)
        #storage.SaveImage(ref_landingPos_eb_lowres,  id='G_ref_eb_lowres_LandingPosition')
        storage.SaveImage(ref_landingPos_eb_highres, id='G_ref_eb_highres_LandingPosition')
        #storage.SaveImage(ref_landingPos_ib_lowres,  id='G_ref_ib_lowres_LandingPosition')
        storage.SaveImage(ref_landingPos_ib_highres, id='G_ref_ib_highres_LandingPosition')

        retract_needle(microscope, park_position)


        #################### LANDING ########################
        # PUT THE SAMPLE BACK
        # move_sample_back().landing_posts_coor()
        park_position = move_needle_to_landing_position(microscope)
        #landing_eb_lowres, landing_ib_lowres   = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
        #landing_eb_highres, landing_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width= 80e-6, image_settings=image_settings)
        microscope.beams.ion_beam.horizontal_field_width.value      = 80e-6
        microscope.beams.electron_beam.horizontal_field_width.value = 80e-6
        microscope.imaging.set_active_view(1)
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
        landing_eb_highres = microscope.imaging.grab_frame(image_settings_electron)
        microscope.imaging.set_active_view(2)
        autocontrast(microscope, beam_type=BeamType.ION)
        landing_ib_highres  = microscope.imaging.grab_frame(image_settings_ion)
        #storage.SaveImage(landing_eb_lowres,  id='H_landingLamella_eb_lowres')
        storage.SaveImage(landing_eb_highres, id='H_landingLamella_eb_highres')
        #storage.SaveImage(landing_ib_lowres,  id='H_landingLamella_ib_lowres')
        storage.SaveImage(landing_ib_highres, id='H_landingLamella_ib_highres')
        ############ FIND dx, dy from HIGH_RES ELECTRON images ############
        #ndi.median_filter(display_image, size=median_smoothing)
        x_shift, y_shift = find_needle_tip_shift_in_image_ELECTRON(landing_eb_highres, ref_landingPos_eb_highres, show=True, median_smoothing=2)
        x_move = x_corrected_needle_movement(x_shift)
        y_move = y_corrected_needle_movement(y_shift, stage.current_position.t)
        print('x_move = ', x_move, ';\ny_move = ', y_move)
        yy = input('press ENTER to move the needle in Y only...')
        needle.relative_move(y_move)

        ############ FIND dz from HIGH_RES ION images ############
        #landing_eb_lowres02, landing_ib_lowres02   = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
        landing_eb_highres02, landing_ib_highres02 = take_electron_and_ion_reference_images(microscope, hor_field_width= 80e-6, image_settings=image_settings)
        #storage.SaveImage(landing_eb_lowres02,  id='I_landingLamella_eb_lowres_yShifted')
        storage.SaveImage(landing_eb_highres02, id='I_landingLamella_eb_highres_yShifted')
        #storage.SaveImage(landing_ib_lowres02,  id='I_landingLamella_ib_lowres_yShifted')
        storage.SaveImage(landing_ib_highres02, id='I_landingLamella_ib_highres_yShifted')
        x_shift, y_shift = find_needle_tip_shift_in_image_ION(landing_ib_highres02, ref_landingPos_ib_highres, show=True, median_smoothing=2)
        stage_tilt = stage.current_position.t
        print('Stage tilt is ', np.rad2deg(stage.current_position.t), ' deg...' )
        z_distance =  y_shift / np.sin(np.deg2rad(52))
        z_move = z_corrected_needle_movement(z_distance, stage_tilt)
        yy = input('press ENTER to move the needle in Z only...')
        needle.relative_move(z_move)
        #if 0: # manual movement:
        #    manual_move = ManipulatorPosition(x=0, y=0, z=y_shift)
        #    needle.relative_move(manual_move)

        ############ FIND dx from HIGH_RES ElEC images ############
        #landing_eb_lowres, landing_ib_lowres   = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
        landing_eb_highres, landing_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width= 80e-6, image_settings=image_settings)
        #storage.SaveImage(landing_eb_lowres,  id='J_landingLamella_eb_lowres')
        storage.SaveImage(landing_eb_highres, id='J_landingLamella_eb_highres')
        #storage.SaveImage(landing_ib_lowres,  id='J_landingLamella_ib_lowres')
        storage.SaveImage(landing_ib_highres, id='J_landingLamella_ib_highres')
        ############ FIND dx, dy from HIGH_RES ELECTRON images ############
        x_shift, y_shift = find_needle_tip_shift_in_image_ELECTRON(landing_eb_highres, ref_landingPos_eb_highres, show=True, median_smoothing=2)
        x_move = x_corrected_needle_movement(x_shift)
        print('x_move = ', x_move)
        yy = input('press ENTER to move the needle in X only...')
        needle.relative_move(x_move)

        #landed_eb_lowres, landed_ib_lowres   = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6, image_settings=image_settings)
        landed_eb_highres, landed_ib_highres = take_electron_and_ion_reference_images(microscope, hor_field_width= 80e-6, image_settings=image_settings)
        #storage.SaveImage(landed_eb_lowres,  id='K_landed_eb_lowres')
        storage.SaveImage(landed_eb_highres, id='K_landed_eb_highres')
        #storage.SaveImage(landed_ib_lowres,  id='K_landed_ib_lowres')
        storage.SaveImage(landed_ib_highres, id='K_landed_ib_highres')
        ############ FIND dx, dy from HIGH_RES ELECTRON images ############

        retract_needle(microscope, park_position)














        if 0: ### EB MANUAL STEP BY STEP CHECK
            needle_reference   = needle_ref_eb_lowres_nobg.data
            needle_with_sample = needle_eb_lowres_with_lamella.data
            field_width  = pixelsize_x  * needle_ref_eb_lowres_nobg.width
            height, width = needle_reference.shape # search for the tip in only the left half of the image
            harrisim = compute_harris_response( ndi.median_filter(needle_reference[:, 0:width//2], size=1) )
            #harrisim = compute_harris_response(needle_reference)
            filtered_coords_ref = get_harris_points(harrisim,4)
            right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))
            #############
            if 1:
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
            if 1:
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
            print(': High pass filter = ', 128, '; low pass filter = ', 5)
            needle_with_sample_norm = ( needle_with_sample - np.mean(needle_with_sample) ) / np.std(needle_with_sample)
            xcorr = crosscorrelation(needle_with_sample_norm, needle_reference_norm, bp='yes', lp=128, hp=5, sigma=2)
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
            if 1:
                plt.figure()
                plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
                plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
                plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='original position')
                plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='current position')
                plt.legend()

        if 0: # IB MANUAL CHECK STEP by STEP
            needle_reference   = needle_ref_ib_lowres_nobg.data
            needle_with_sample = needle_ib_lowres_with_lamella.data
            field_width  = pixelsize_x  * needle_ib_lowres_with_lamella.width
            height, width = needle_reference.shape # search for the tip in only the left half of the image
            harrisim = compute_harris_response( ndi.median_filter(needle_reference[:, 0:width//2], size=5) )
            #harrisim = compute_harris_response(needle_reference)
            filtered_coords_ref = get_harris_points(harrisim,4)
            right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))
            topmost_point_ref         = min(filtered_coords_ref, key=itemgetter(0))
            #############
            if 1:
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
            if 1:
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
            print(': High pass filter = ', 128, '; low pass filter = ', 5)
            needle_with_sample_norm = ( needle_with_sample - np.mean(needle_with_sample) ) / np.std(needle_with_sample)
            xcorr = crosscorrelation(needle_with_sample_norm, reference_elps_norm, bp='yes', lp=128, hp=5, sigma=5)
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
            if 1:
                plt.figure()
                plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
                plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
                plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='original position')
                plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='current position')
                plt.legend()



    if TEST == 'ION':
        needle_ref_images_nobgrnd = (needle_ref_eb_lowres_nobg, needle_ref_eb_highres_nobg, needle_ref_ib_lowres_nobg, needle_ref_ib_highres_nobg )
        #pixelsize_x = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.x
        #pixelsize_y = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.y
        pixelsize_x = needle_ref_ib_highres_nobg.metadata.binary_result.pixel_size.x
        pixelsize_y = needle_ref_ib_highres_nobg.metadata.binary_result.pixel_size.y

        #needle_ref_eb_lowres_nobg
        #needle_eb_lowres_with_lamella
        #manual_needle_movement_in_xy(microscope, needle_ref_images_nobgrnd, [], move_in_x=True, move_in_y=True)
        #tip_x, tip_y, mask, mask_binary = find_needle_tip(image, show=show, median_smoothing=median_smoothing)
        #old_tip_x, old_tip_y, reference = generate_reference_needle_image(needle_reference.data, show=show, median_smoothing=
        #old_tip_x, old_tip_y, reference = generate_reference_needle_image(needle_reference.data, show=show, median_smoothing=median_smoothing)
        #x_shift_eb, y_shift_eb, old_tip_x_eb, old_tip_y_eb = find_needle_tip_shift_in_image(needle_eb_lowres_with_lamella, needle_reference_eb_lowres_nobgrnd, show=False, median_smoothing=2)
        print('ION needle tip search')
        harrisim = compute_harris_response(  ndi.median_filter(needle_ref_ib_highres_nobg.data, size=2)  )
        filtered_coords = get_harris_points(harrisim, 4)
        right_outermost_point_image = max(filtered_coords, key=itemgetter(1)) # find tip needle using corner finding algorithm
        print('outermost point pixel from real image', right_outermost_point_image)

        ############
        filt = ndi.median_filter(needle_ref_ib_highres_nobg.data, size=10)  #median(needle_ref_eb_lowres_nobg.data, disk(5))
        thresh = threshold_otsu(filt)
        binary = filt > thresh
        mask = gaussian(binary_dilation(binary, iterations=15), 5)
        mask_binary = (mask >= 0.51).astype(int)
        ysize, xsize = mask.shape
        harrisim = compute_harris_response(binary)
        filtered_coords_mask = get_harris_points(harrisim, 4)
        right_outermost_point_mask = max(filtered_coords_mask, key=itemgetter(1))
        print('outermost point pixel from binarized image', right_outermost_point_mask)
        ##########
        def R(p1, p2):
            return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) # fimnd distance between two points
        if len(right_outermost_point_image)==0:
            right_outermost_point = right_outermost_point_mask
        if len(right_outermost_point_image)==0 and len(right_outermost_point_mask)==0:
            raise RuntimeError("Could not find the needle tip!")
        if R(right_outermost_point_image, right_outermost_point_mask  ) <= 20:
            right_outermost_point = right_outermost_point_image
        else:
            right_outermost_point = right_outermost_point_mask
        print('outermost point pixel is', right_outermost_point)
        # tip position in the reference image :
        tip_x = right_outermost_point_mask[0]
        tip_y = right_outermost_point_mask[1]

        xmin = min( tip_y, mask.shape[1] - tip_y )
        ymin = min( tip_x, mask.shape[0] - tip_x )
        rmin = min(xmin, ymin )
        Dmin = 2*rmin
        #xmin = min( tip_y, mask.shape[1] - tip_y )
        #ymin = min( tip_x, mask.shape[0] - tip_x )
        #rmin = min(xmin, ymin )
        #Dmin = min( tip_x,   mask.shape[0], tip_y,   mask.shape[1])
        #if Dmin%2 != 0 : Dmin += 1
        ########
        cmask = circ_mask( size=(Dmin, Dmin), radius=Dmin//2-15, sigma=10 ) #circular mask
        CMASK = np.zeros(mask.shape)
        CMASK[ tip_x-Dmin//2 : tip_x+Dmin//2 , tip_y-Dmin//2 : tip_y+Dmin//2] = cmask
        reference = needle_ref_ib_highres_nobg.data * CMASK * mask

        plt.figure(1)
        plt.imshow(needle_ref_ib_highres_nobg.data, cmap='gray')
        plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], 'bo')
        plt.plot([tip_y], [tip_x], 'ro')

        plt.figure(2)
        plt.imshow(mask_binary, cmap='gray')
        plt.plot([p[1] for p in filtered_coords_mask], [p[0] for p in filtered_coords_mask], 'bo')
        plt.plot([tip_y], [tip_x], 'ro')

        xcorr = crosscorrelation(needle_ib_highres_with_lamella.data, reference, bp='yes', lp=int(max(reference.shape) // 24), hp=5, sigma=2)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        ### new coordinates of the tip
        new_tip_y = tip_y - err[1]
        new_tip_x = tip_x - err[0]
        x_shift = ( cen[1] - new_tip_y ) * pixelsize_x
        y_shift = ( cen[0] - new_tip_x ) * pixelsize_y
        print("X-shift =  {} meters".format(x_shift))
        print("Y-shift =  {} meters".format(y_shift))

        plt.figure(3)
        plt.imshow(needle_ib_highres_with_lamella.data, cmap='gray')
        plt.plot([tip_y], [tip_x], 'ro')
        plt.plot([new_tip_y], [new_tip_x], 'rd')
        plt.show()

        print("ION: img2 is X-shifted by ", y_shift, '; Y-shifted by ', x_shift)
        print('Tip position in the reference image: dX = ', old_tip_y, '; dY = ', old_tip_x)



    if TEST == 'ELECTRON':
        needle_ref_images_nobgrnd = (needle_ref_eb_lowres_nobg, needle_ref_eb_highres_nobg, needle_ref_ib_lowres_nobg, needle_ref_ib_highres_nobg )
        #pixelsize_x = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.x
        #pixelsize_y = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.y
        pixelsize_x = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.x
        pixelsize_y = needle_ref_eb_lowres_nobg.metadata.binary_result.pixel_size.y

        #needle_ref_eb_lowres_nobg
        #needle_eb_lowres_with_lamella
        #manual_needle_movement_in_xy(microscope, needle_ref_images_nobgrnd, [], move_in_x=True, move_in_y=True)
        #tip_x, tip_y, mask, mask_binary = find_needle_tip(image, show=show, median_smoothing=median_smoothing)
        #old_tip_x, old_tip_y, reference = generate_reference_needle_image(needle_reference.data, show=show, median_smoothing=
        #old_tip_x, old_tip_y, reference = generate_reference_needle_image(needle_reference.data, show=show, median_smoothing=median_smoothing)
        #x_shift_eb, y_shift_eb, old_tip_x_eb, old_tip_y_eb = find_needle_tip_shift_in_image(needle_eb_lowres_with_lamella, needle_reference_eb_lowres_nobgrnd, show=False, median_smoothing=2)
        print('needle tip search')
        harrisim = compute_harris_response(  ndi.median_filter(needle_ref_eb_lowres_nobg.data, size=2)  )
        filtered_coords = get_harris_points(harrisim, 4)
        right_outermost_point_image = max(filtered_coords, key=itemgetter(1)) # find tip needle using corner finding algorithm
        print('outermost point pixel from real image', right_outermost_point_image)

        ############
        filt = ndi.median_filter(needle_ref_eb_lowres_nobg.data, size=10)  #median(needle_ref_eb_lowres_nobg.data, disk(5))
        thresh = threshold_otsu(filt)
        binary = filt > thresh
        mask = gaussian(binary_dilation(binary, iterations=15), 5)
        mask_binary = (mask >= 0.51).astype(int)
        ysize, xsize = mask.shape
        harrisim = compute_harris_response(mask_binary)
        filtered_coords_mask = get_harris_points(harrisim, 4)
        right_outermost_point_mask = max(filtered_coords_mask, key=itemgetter(1))
        print('outermost point pixel from binarized image', right_outermost_point_mask)
        ##########
        def R(p1, p2):
            return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) # fimnd distance between two points
        if len(right_outermost_point_image)==0:
            right_outermost_point = right_outermost_point_mask
        if len(right_outermost_point_image)==0 and len(right_outermost_point_mask)==0:
            raise RuntimeError("Could not find the needle tip!")
        if R(right_outermost_point_image, right_outermost_point_mask  ) <= 20:
            right_outermost_point = right_outermost_point_image
        else:
            right_outermost_point = right_outermost_point_mask
        print('outermost point pixel is', right_outermost_point)
        # tip position in the reference image :
        tip_x = right_outermost_point_mask[0]
        tip_y = right_outermost_point_mask[1]

        #xmin = min( tip_y, mask.shape[1] - tip_y )
        #ymin = min( tip_x, mask.shape[0] - tip_x )
        #rmin = min(xmin, ymin )
        #Dmin = min( tip_x + 20,   mask.shape[0]  )
        #if Dmin%2 != 0 : Dmin += 1
        xmin = min( tip_y, mask.shape[1] - tip_y )
        ymin = min( tip_x, mask.shape[0] - tip_x )
        rmin = min(xmin, ymin )
        Dmin = 2*rmin    ###min( tip_x,   mask.shape[0], tip_y,   mask.shape[1])
        ########
        cmask = circ_mask( size=(Dmin, Dmin), radius=Dmin//2-15, sigma=10 ) #circular mask
        CMASK = np.zeros(mask.shape)
        CMASK[ tip_x-Dmin//2 : tip_x+Dmin//2 , tip_y-Dmin//2 : tip_y+Dmin//2] = cmask
        reference = needle_ref_eb_lowres_nobg.data * CMASK * mask

        plt.figure(1)
        plt.imshow(needle_ref_eb_lowres_nobg.data, cmap='gray')
        plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], 'bo')
        plt.plot([tip_y], [tip_x], 'ro')

        plt.figure(2)
        plt.imshow(mask_binary, cmap='gray')
        plt.plot([p[1] for p in filtered_coords_mask], [p[0] for p in filtered_coords_mask], 'bo')
        plt.plot([tip_y], [tip_x], 'ro')

        xcorr = crosscorrelation(needle_eb_highres_with_lamella.data, reference, bp='yes', lp=int(max(reference.shape) // 12), hp=5, sigma=2)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        ### new coordinates of the tip
        new_tip_y = tip_y - err[1]
        new_tip_x = tip_x - err[0]
        x_shift = ( cen[1] - new_tip_y ) * pixelsize_x
        y_shift = ( cen[0] - new_tip_x ) * pixelsize_y
        print("X-shift =  {} meters".format(x_shift))
        print("Y-shift =  {} meters".format(y_shift))

        plt.figure(3)
        plt.imshow(needle_eb_highres_with_lamella.data, cmap='gray')
        plt.plot([tip_y], [tip_x], 'ro')
        plt.plot([new_tip_y], [new_tip_x], 'rd')
        plt.show()

        print("ELECTRON: img2 is X-shifted by ", y_shift, '; Y-shifted by ', x_shift)
        print('Tip position in the reference image: dX = ', old_tip_y, '; dY = ', old_tip_x)
        #print('ELECTRON: now tip off-centre X, meters: ', (y_shift - y_shift)*pixelsize_x_lowres)
        #print('ELECTRON: now tip off-centre Y, meters: ', (x_shift - x_shift_eb)*pixelsize_y_lowres)





    if TEST == 3.3:
        print('different tip detection mnethods')
        pixelsize_x = 1
        pixelsize_y = 1


        '''TYPE = 'ION'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
        fileName_sample = r'ion_beam.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
        fileName_ref = r'reference_ib_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        '''TYPE = 'ELECTRON'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.01.28_needle_images\B_inserted_position_moved_closerX20um_Z180'
        fileName_eb = r'001_e_needle_movedCloser_A_hfw150.tif'
        needle_with_sample = read_image(DIR1, fileName_eb, gaus_smooth=2)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.01.28_needle_images\C_reference_needle_images_no_background_insertedPosition'
        fileName_eb_ref = r'001_e_needle_movedCloser_A_hfw150.tif'
        needle_reference = read_image(DIR2, fileName_eb_ref, gaus_smooth=2)'''

        ########################################################################################################
        '''TYPE = 'ION'        # weird HALO at the edges...........
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.12_needle_image_A'
        fileName_sample = r'sample_ib_lowres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.12_needle_image_A'
        fileName_ref = r'ref_ib_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        '''TYPE = 'ELECTRON'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.12_needle_image_A'
        fileName_sample = r'sample_eb_lowres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.12_needle_image_A'
        fileName_ref = r'ref_eb_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        ########################################################################################################
        '''TYPE = 'ION'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.15_needle_image_E'
        fileName_sample = r'sample_ib_lowres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.15_needle_image_E'
        fileName_ref = r'ref_ib_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        '''TYPE = 'ELECTRON'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.15_needle_image_E'
        fileName_sample = r'sample_eb_lowres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.15_needle_image_E'
        fileName_ref = r'ref_eb_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        ########################################################################################################
        '''TYPE = 'ION'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.102713\liftout000'
        fileName_sample = r'step00_B_sample_ib_lowres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.102713\liftout000'
        fileName_ref = r'step00_A_ref_ib_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        '''TYPE = 'ELECTRON'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.102713\liftout000'
        fileName_sample = r'step00_B_sample_eb_lowres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.102713\liftout000'
        fileName_ref = r'step00_A_ref_eb_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        ########################################################################################################
        '''TYPE = 'ION'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.102713\liftout000'
        fileName_sample = r'step00_B_sample_ib_lowres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.102713\liftout000'
        fileName_ref = r'step00_A_ref_ib_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        '''TYPE = 'ELECTRON'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.102713\liftout000'
        fileName_sample = r'step00_B_sample_eb_lowres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.102713\liftout000'
        fileName_ref = r'step00_A_ref_eb_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''


        ########################################################################################################
        '''TYPE = 'ION'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\RUN_20210308.105112\liftout000'
        fileName_sample = r'step00_B_sample_ib_lowres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\RUN_20210308.105112\liftout000'
        fileName_ref = r'step00_A_ref_ib_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        TYPE = 'ELECTRON'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\RUN_20210308.105112\liftout000'
        fileName_sample = r'step00_B_sample_eb_lowres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\RUN_20210308.105112\liftout000'
        fileName_ref = r'step06_A_needle_land_needle_ref_eb_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)

        ###### soft rectangular mask
        start = np.round(np.array(needle_reference.shape) * 0.05)
        extent = np.round(np.array(needle_reference.shape) * 0.90)
        rr, cc = skimage.draw.rectangle(start, extent=extent, shape=needle_reference.shape)
        rect_mask = np.zeros(needle_reference.shape)
        rect_mask[rr.astype(int), cc.astype(int)] = 1.0
        rect_mask = ndi.gaussian_filter(rect_mask, sigma=10)

        if TYPE=='ELECTRON':
            lowpass_pixels  = int( max(needle_reference.shape) / 12 ) # =128 for 1536x1024 image
            highpass_pixels = int( max(needle_reference.shape)/ 256 ) # =6   for 1536x1024 image
            sigma = int( 2 * max(needle_reference.data.shape)/1536)   # =2 @ 1536x1024, good for e-beam images
        if TYPE=='ION':
            lowpass_pixels  = int( max(needle_reference.shape) / 6 ) # =256 @ 1536x1024, good for i-beam images
            highpass_pixels = int( max(needle_reference.shape)/ 64 ) # =24  @ 1536x1024, good for i-beam images => need a large highpass to remove noise and ringing
            sigma = int( 10 * max(needle_reference.data.shape)/1536) # =10 @ 1536x1024, good for i-beam images


        ### binary mask to remove background
        filter_type = 'fourier'
        if filter_type == 'median':
            filt   = ndi.median_filter(needle_reference * rect_mask, size=5)
        elif filter_type == 'fourier':
            print(': High pass filter = ', lowpass_pixels, '; low pass filter = ', highpass_pixels, '; sigma = ', sigma)
            bandpass = bandpass_mask(size=(needle_reference.shape[1], needle_reference.shape[0]), lp=lowpass_pixels, hp=highpass_pixels, sigma=sigma)
            img1ft = bandpass * scipy.fftpack.fftshift(scipy.fftpack.fft2(needle_reference * rect_mask))
            filt = np.real(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(((img1ft)))))
            #plt.imshow(filt, cmap='gray')
        thresh = threshold_otsu(filt)
        binary = filt > thresh
        mask   = gaussian(binary_dilation(binary, iterations=15), 5)
        mask_binary = (mask >= 0.50).astype(int)
        sigma_blur = 5
        mask_blurred = ndi.gaussian_filter(mask,  sigma=sigma_blur)


        ########### Harris corner detection
        height, width = needle_reference.shape # search for the tip in only the left half of the image
        if TYPE=='ELECTRON':
            median_filter_size = int( max(needle_reference.shape) / 512 )  #5@3072x2048 #milder
            harrisim = compute_harris_response( ndi.median_filter( (filt * rect_mask)[:, 0:width//2], size=median_filter_size) )
        elif TYPE=='ION':
            median_filter_size = int( max(needle_reference.shape) / 307 )  #10@3072x2048 #stronger noise filtering
            harrisim = compute_harris_response( ndi.median_filter( (filt * rect_mask)[:, 0:width//2], size=median_filter_size) )
        print('=====median filter size to remove noise = ',  median_filter_size)
        corner_threshold = max(harrisim.ravel()) * 0.1
        harrisim_t = (harrisim > corner_threshold) * 1
        filtered_coords_ref = get_harris_points(harrisim,4)
        right_outermost_point_harris = max(filtered_coords_ref, key=itemgetter(1))
        topmost_point_harris         = min(filtered_coords_ref, key=itemgetter(0))
        if 1:
            plt.figure(1)
            plt.imshow(needle_reference, cmap='gray', alpha=1)
            plt.imshow(mask_blurred, cmap='gray', alpha=0.5)
            plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
            plt.plot( [right_outermost_point_harris[1]], [right_outermost_point_harris[0]],   'ro', label='rightmost')
            plt.plot( [topmost_point_harris[1]], [topmost_point_harris[0]],   'rd', label='topmost')
            plt.title('Harris corner detection')
            plt.legend()




        ############## sobel edge detection
        sx = ndi.sobel(filt * rect_mask * mask_blurred, axis=0, mode='constant')
        sy = ndi.sobel(filt * rect_mask * mask_blurred, axis=1, mode='constant')
        needle_edges = np.hypot(sx, sy)
        edge_max = needle_edges.max()
        nonzero = np.where(needle_edges[:, 0:width//2] > needle_edges[:, 0:width//2].max()/10 )
        mean_edge_value = np.mean( needle_edges[nonzero] )
        #needle_edges_binary = (needle_edges >= needle_edges.max()*0.5).astype(int)
        needle_edges_binary = (needle_edges >=mean_edge_value).astype(int)
        coords = np.where(needle_edges_binary[:, 0:width//2]==1)
        coords1_max_location = np.argmax(coords[1]) # rightmost position
        coords0_min_location = np.argmin(coords[0]) #topmost position, min, because Y goes from top to bottom
        right_outermost_point_sobel = ( coords[0][coords1_max_location] , coords[1][coords1_max_location] )
        topmost_point_sobel         = ( coords[0][coords0_min_location] , coords[1][coords0_min_location] )
        if 1:
            plt.figure(3)
            plt.imshow(needle_edges_binary,   cmap='gray',     alpha=1)
            plt.imshow(needle_reference, cmap='Oranges_r',   alpha=0.5)
            plt.plot( [right_outermost_point_sobel[1]], [right_outermost_point_sobel[0]],   'ro', label='rightmost')
            plt.plot( [topmost_point_sobel[1]], [topmost_point_sobel[0]],   'rd', label='topmost')
            plt.title('Sobel edge detection')
            plt.legend()



        ########## canny edge detection
        edges = feature.canny( filt * rect_mask * mask_blurred, sigma=sigma_blur )
        edges = edges.astype(int)
        coords = np.where(edges==1)
        coords1_max_location = np.argmax(coords[1]) # rightmost position
        coords0_min_location = np.argmin(coords[0]) #topmost position, min, because Y goes from top to bottom
        right_outermost_point_canny = ( coords[0][coords1_max_location] , coords[1][coords1_max_location] )
        topmost_point_canny         = ( coords[0][coords0_min_location] , coords[1][coords0_min_location] )
        if 1:
            plt.figure(4)
            plt.imshow( needle_reference * rect_mask * mask_blurred, cmap='Blues_r', alpha=1)
            plt.imshow( edges, cmap='Oranges_r', alpha=0.5)
            plt.imshow( mask_blurred, cmap='gray', alpha=0.5)
            plt.plot( [right_outermost_point_canny[1]], [right_outermost_point_canny[0]],   'ro', label='rightmost')
            plt.plot( [topmost_point_canny[1]], [topmost_point_canny[0]],   'rd', label='topmost')
            plt.title('Canny edge detection')
            plt.legend()


        right_x_median = np.median( [right_outermost_point_canny[0], right_outermost_point_sobel[0], right_outermost_point_harris[0]] )
        right_y_median = np.median( [right_outermost_point_canny[1], right_outermost_point_sobel[1], right_outermost_point_harris[1]] )
        top_x_median = np.median( [topmost_point_canny[0], topmost_point_sobel[0], topmost_point_harris[0]] )
        top_y_median = np.median( [topmost_point_canny[1], topmost_point_sobel[1], topmost_point_harris[1]] )

        right_x_mean = np.mean( [right_outermost_point_canny[0], right_outermost_point_sobel[0], right_outermost_point_harris[0]] )
        right_y_mean = np.mean( [right_outermost_point_canny[1], right_outermost_point_sobel[1], right_outermost_point_harris[1]] )
        top_x_mean = np.mean( [topmost_point_canny[0], topmost_point_sobel[0], topmost_point_harris[0]] )
        top_y_mean = np.mean( [topmost_point_canny[1], topmost_point_sobel[1], topmost_point_harris[1]] )



        # tip position in the reference image :
        if TYPE=='ELECTRON':
            old_tip_x = right_outermost_point_harris[0]
            old_tip_y = right_outermost_point_harris[1]
        elif TYPE=='ION':
            old_tip_x = topmost_point_harris[0]
            old_tip_y = topmost_point_harris[1]

        xmin = min(old_tip_y, mask.shape[1] - old_tip_y)
        ymin = min(old_tip_x, mask.shape[0] - old_tip_x)
        rmin = min(xmin, ymin)
        Dmin = 2 * rmin
        #################### circular mask
        cmask = circ_mask(size=(Dmin, Dmin), radius=Dmin // 2 - 15, sigma=10)  # circular mask
        CMASK = np.zeros(needle_reference.shape)
        CMASK[old_tip_x - Dmin // 2: old_tip_x + Dmin // 2, old_tip_y - Dmin // 2: old_tip_y + Dmin // 2] = cmask
        # fist normalise the reference image, then mask
        needle_reference_norm = ( needle_reference - np.mean(needle_reference) ) / np.std(needle_reference)
        reference_circ_norm = needle_reference_norm * CMASK * mask
        #################### elliptical mask
        xmin = min(old_tip_x, mask.shape[1] - old_tip_x)
        ymin = min(old_tip_y, mask.shape[0] - old_tip_y)
        ELLPS_MASK = np.zeros(needle_reference.shape)
        elps_mask = ellipse_mask(size=(xmin*2, ymin*2), radius1=xmin-15, radius2=ymin-15, sigma=10)
        ELLPS_MASK[old_tip_y - ymin : old_tip_y + ymin, old_tip_x - xmin : old_tip_x + xmin] = elps_mask
        reference_elps_norm = needle_reference_norm * ELLPS_MASK * mask
        #################### rectangular mask
        start  = np.round(np.array( [old_tip_x - Dmin // 2, old_tip_y - Dmin // 2] ) )
        extent = np.round(np.array( [Dmin, Dmin] ) )
        rr, cc = skimage.draw.rectangle(start, extent=extent, shape=needle_reference.shape)
        RECT_MASK = np.zeros(needle_reference.shape)
        RECT_MASK[rr.astype(int), cc.astype(int)] = 1.0
        RECT_MASK = ndi.gaussian_filter(RECT_MASK, sigma=10)



        print(': High pass filter = ', lowpass_pixels, '; low pass filter = ', highpass_pixels)
        # nornalise the image before the cross-correlation
        needle_with_sample_norm = ( needle_with_sample - np.mean(needle_with_sample) ) / np.std(needle_with_sample)
        xcorr = crosscorrelation(needle_with_sample_norm, reference_elps_norm, bp='yes', lp=lowpass_pixels, hp=highpass_pixels, sigma=sigma)
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

        plt.show()











    if TEST == 3.2: # loop over low- high pass filters
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
        fileName_sample = r'electron_beam.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)

        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
        fileName_ref = r'reference_eb_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)

        '''DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.01.28_needle_images\B_inserted_position_moved_closerX20um_Z180'
        fileName_eb = r'001_e_needle_movedCloser_A_hfw150.tif'
        needle_with_sample = read_image(DIR1, fileName_eb, gaus_smooth=2)

        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.01.28_needle_images\C_reference_needle_images_no_background_insertedPosition'
        fileName_eb_ref = r'001_e_needle_movedCloser_A_hfw150.tif'
        needle_reference = read_image(DIR2, fileName_eb_ref, gaus_smooth=2)'''

        ### Find the tip using corner-finding algorithm
        harrisim = compute_harris_response( median(needle_reference, disk(2)) )
        filtered_coords_ref = get_harris_points(harrisim,4)
        right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))

        plt.figure(1)
        plt.imshow(needle_reference, cmap='gray')
        plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
        plt.plot( [right_outermost_point_ref[1]], [right_outermost_point_ref[0]],   'ro')

        ### Find the tip from binarized image (mask) and corner-finding algorithm
        filt   = median(needle_reference, disk(3))
        thresh = threshold_otsu(filt)
        binary = filt > thresh
        mask   = gaussian(binary_dilation(binary, iterations=15), 5)
        mask_binary = (mask >= 0.5).astype(int)
        needle_ref_masked = needle_reference * mask_binary

        ysize, xsize = mask.shape
        harrisim = compute_harris_response(mask_binary)
        filtered_coords_mask_ref = get_harris_points(harrisim,4)
        right_outermost_point_mask_ref = max(filtered_coords_mask_ref, key=itemgetter(1))

        plt.figure(2)
        plt.imshow(mask_binary, cmap='gray')
        plt.plot([p[1] for p in filtered_coords_mask_ref], [p[0] for p in filtered_coords_mask_ref], 'bo')
        plt.plot([right_outermost_point_mask_ref[1]], [right_outermost_point_mask_ref[0]], 'ro')

        def R(p1, p2):
            return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) # find distance between two points
        # two tips found, if the tip coordinate from the noisy real needle image is way off, rely on the tip found from the binarized image
        if R(right_outermost_point_ref, right_outermost_point_mask_ref  ) <= 20:
            right_outermost_point = right_outermost_point_ref
        else:
            right_outermost_point = right_outermost_point_mask_ref

        # tip position in the reference image :
        tip_x = right_outermost_point[0]
        tip_y = right_outermost_point[1]

        xmin = min( tip_y, mask.shape[1] - tip_y )
        ymin = min( tip_x, mask.shape[0] - tip_x )
        rmin = min(xmin, ymin )
        Dmin = min( tip_x + 100,   mask.shape[0]  )
        if Dmin%2 != 0 : Dmin +=1

        cmask = circ_mask( size=(Dmin, Dmin), radius=Dmin//2-15, sigma=10 )
        CMASK = np.zeros(needle_reference.shape)
        CMASK[ tip_y-Dmin//2 : tip_y+Dmin//2, 0:Dmin ] = cmask
        #####image_eb_ref_cut = image_eb_ref_masked[ right_outermost_point_mask_eb_ref[1]-Dmin//2 : right_outermost_point_mask_eb_ref[1]+Dmin//2, 0:Dmin ]
        reference     = needle_reference * CMASK * mask
        ######reference_eb     = np.pad( reference_eb, ((0,image_eb_ref.shape[0]-reference_eb.shape[0]), (0,image_eb_ref.shape[1]-reference_eb.shape[1])), 'constant' )


        CORR_MAX_X = []
        CORR_MAX_Y = []
        LOW_PASS_FILTER = range( max(needle_with_sample.shape) )

        for low_pass_filter in LOW_PASS_FILTER:
            print(low_pass_filter)
            xcorr = crosscorrelation(needle_with_sample, reference, bp='yes', lp=low_pass_filter,  hp=5, sigma=2)
            maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
            print('\n', maxX, maxY)
            cen = np.asarray(xcorr.shape) / 2
            err = np.array(cen - [maxX, maxY], int)
            print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
            CORR_MAX_X.append( maxX )
            CORR_MAX_Y.append( maxY )

        plt.figure()
        plt.plot( LOW_PASS_FILTER, cen[0] - np.array(CORR_MAX_X), '-ro', label='maxX')
        plt.plot(LOW_PASS_FILTER, cen[1] - np.array(CORR_MAX_Y), '-bo', label='maxY')
        plt.ylim(ymin = 0, ymax = 50)
        plt.xlabel('low pass filter, pixels')
        plt.ylabel('shift of the second image, pixels')
        plt.legend()
        plt.show()



    if TEST == 3.1:
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
        fileName_sample = r'ion_beam.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
        fileName_ref = r'reference_ib_lowres.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)

        '''DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.01.28_needle_images\B_inserted_position_moved_closerX20um_Z180'
        fileName_eb = r'001_e_needle_movedCloser_A_hfw150.tif'
        needle_with_sample = read_image(DIR1, fileName_eb, gaus_smooth=2)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.01.28_needle_images\C_reference_needle_images_no_background_insertedPosition'
        fileName_eb_ref = r'001_e_needle_movedCloser_A_hfw150.tif'
        needle_reference = read_image(DIR2, fileName_eb_ref, gaus_smooth=2)'''

        ### Find the tip using corner-finding algorithm
        harrisim = compute_harris_response( median(needle_reference, disk(2)) )
        filtered_coords_ref = get_harris_points(harrisim,4)
        right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))

        plt.figure(1)
        plt.imshow(needle_reference, cmap='gray')
        plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
        plt.plot( [right_outermost_point_ref[1]], [right_outermost_point_ref[0]],   'ro')


        ### Find the tip from binarized image (mask) and corner-finding algorithm
        filt   = median(needle_reference, disk(3))
        thresh = threshold_otsu(filt)
        binary = filt > thresh
        mask   = gaussian(binary_dilation(binary, iterations=15), 5)
        mask_binary = (mask >= 0.5).astype(int)
        needle_ref_masked = needle_reference * mask_binary

        ysize, xsize = mask.shape
        harrisim = compute_harris_response(mask_binary)
        filtered_coords_mask_ref = get_harris_points(harrisim,4)
        right_outermost_point_mask_ref = max(filtered_coords_mask_ref, key=itemgetter(1))

        plt.figure(2)
        plt.imshow(mask_binary, cmap='gray')
        plt.plot([p[1] for p in filtered_coords_mask_ref], [p[0] for p in filtered_coords_mask_ref], 'bo')
        plt.plot([right_outermost_point_mask_ref[1]], [right_outermost_point_mask_ref[0]], 'ro')

        def R(p1, p2):
            return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) # find distance between two points

        # two tips found, if the tip coordinate from the noisy real needle image is way off, rely on the tip found from the binarized image
        if R(right_outermost_point_ref, right_outermost_point_mask_ref  ) <= 20:
            right_outermost_point = right_outermost_point_ref
        else:
            right_outermost_point = right_outermost_point_mask_ref

        # tip position in the reference image :
        tip_x = right_outermost_point[0]
        tip_y = right_outermost_point[1]

        xmin = min( tip_y, mask.shape[1] - tip_y )
        ymin = min( tip_x, mask.shape[0] - tip_x )
        rmin = min(xmin, ymin )
        Dmin = min( tip_x + 100,   mask.shape[0]  )
        if Dmin%2 != 0 : Dmin +=1


        cmask = circ_mask(size=(Dmin, Dmin), radius=Dmin // 2 - 15, sigma=10)  # circular mask
        CMASK = np.zeros(needle_reference.shape)
        CMASK[old_tip_x - Dmin // 2: old_tip_x + Dmin // 2, old_tip_y - Dmin // 2: old_tip_y + Dmin // 2] = cmask

        #####image_eb_ref_cut = image_eb_ref_masked[ right_outermost_point_mask_eb_ref[1]-Dmin//2 : right_outermost_point_mask_eb_ref[1]+Dmin//2, 0:Dmin ]
        reference     = needle_reference * CMASK * mask
        ######reference_eb     = np.pad( reference_eb, ((0,image_eb_ref.shape[0]-reference_eb.shape[0]), (0,image_eb_ref.shape[1]-reference_eb.shape[1])), 'constant' )

        xcorr = crosscorrelation(needle_with_sample, reference, bp='yes', lp=int(max(needle_with_sample.shape) / 24), hp=5, sigma=2)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])

        plt.figure()
        plt.subplot(3,1,1)
        plt.imshow(reference, cmap='gray')
        plt.subplot(3,1,2)
        plt.imshow(needle_with_sample, cmap='gray')
        plt.subplot(3,1,3)
        plt.imshow(xcorr, cmap='gray')
        plt.plot( [maxY], [maxX], 'ro' )

        plt.figure()
        plt.subplot(1,2,1)
        #plt.imshow(image_eb_ref, cmap='gray')
        plt.imshow(needle_reference,   cmap='Blues_r',     alpha=1)
        plt.imshow(needle_with_sample, cmap='Oranges_r',   alpha=0.5)
        plt.plot([p[1] for p in filtered_coords_mask_ref], [p[0] for p in filtered_coords_mask_ref], 'bo')
        #plt.plot([right_outermost_point_mask_ref[1]], [right_outermost_point_mask_ref[0]], 'ro')
        plt.plot([tip_y], [tip_x], 'rd',  markersize=5)
        plt.subplot(1,2,2)
        plt.imshow(needle_with_sample, cmap='gray')
        plt.plot([tip_y], [tip_x], 'bd', markersize=5)
        plt.plot([tip_y - err[1]], [tip_x - err[0]], 'rd', markersize=3)
        plt.show()


    if TEST == 3:
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.01.28_needle_images\B_inserted_position_moved_closerX20um_Z180'
        fileName_eb = r'001_e_needle_movedCloser_A_hfw150.tif'
        fileName_ib = r'001_i_needle_movedCloser_A_hfw150.tif'
        image_eb = read_image(DIR1, fileName_eb, gaus_smooth=2)
        image_ib = read_image(DIR1, fileName_ib, gaus_smooth=2)

        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.01.28_needle_images\C_reference_needle_images_no_background_insertedPosition'
        fileName_eb_ref = r'001_e_needle_movedCloser_A_hfw150.tif'
        fileName_ib_ref = r'001_i_needle_movedCloser_A_hfw150.tif'
        image_eb_ref = read_image(DIR2, fileName_eb_ref, gaus_smooth=2)
        image_ib_ref = read_image(DIR2, fileName_ib_ref, gaus_smooth=2)


        image_eb_ref_smooth = median(image_eb_ref, disk(2))
        harrisim = compute_harris_response(image_eb_ref_smooth)
        filtered_coords_eb_ref = get_harris_points(harrisim,4)
        right_outermost_point_eb_ref = max(filtered_coords_eb_ref, key=itemgetter(1))
        plt.figure(1)
        plt.imshow(image_eb_ref, cmap='gray')
        plt.plot( [p[1] for p in filtered_coords_eb_ref],[p[0] for p in filtered_coords_eb_ref],'bo'  )
        plt.plot( [right_outermost_point_eb_ref[1]], [right_outermost_point_eb_ref[0]],   'ro')

        filt = median(image_eb_ref, disk(5))
        thresh = threshold_otsu(filt)
        binary = filt > thresh
        mask_eb        = gaussian(binary_dilation(binary, iterations=15), 5)
        mask_eb_binary = (mask_eb >= 0.5).astype(int)
        image_eb_ref_masked = image_eb_ref * mask_eb_binary

        ysize, xsize = mask_eb.shape
        harrisim = compute_harris_response(mask_eb_binary)
        filtered_coords_mask_eb_ref = get_harris_points(harrisim,4)
        right_outermost_point_mask_eb_ref = max(filtered_coords_mask_eb_ref, key=itemgetter(1))

        plt.figure()
        plt.imshow(mask_eb, cmap='gray')
        plt.plot([p[1] for p in filtered_coords_mask_eb_ref], [p[0] for p in filtered_coords_mask_eb_ref], 'bo')
        plt.plot([right_outermost_point_mask_eb_ref[1]], [right_outermost_point_mask_eb_ref[0]], 'ro')
        plt.plot([right_outermost_point_eb_ref[1]], [right_outermost_point_eb_ref[0]], 'rd')

        xmin = min( right_outermost_point_mask_eb_ref[1], mask_eb.shape[1] - right_outermost_point_mask_eb_ref[1] )
        ymin = min( right_outermost_point_mask_eb_ref[0], mask_eb.shape[0] - right_outermost_point_mask_eb_ref[0] )
        rmin = min(xmin, ymin )
        Dmin = min( right_outermost_point_eb_ref[1] + 100,   mask_eb.shape[0]  )
        if Dmin%2 != 0 : Dmin +=1

        cmask_eb         = circ_mask( size=(Dmin, Dmin), radius=Dmin//2-15, sigma=10 )
        CMASK = np.zeros(image_eb_ref.shape)
        CMASK[ right_outermost_point_mask_eb_ref[1]-Dmin//2 : right_outermost_point_mask_eb_ref[1]+Dmin//2, 0:Dmin ] = cmask_eb
        #####image_eb_ref_cut = image_eb_ref_masked[ right_outermost_point_mask_eb_ref[1]-Dmin//2 : right_outermost_point_mask_eb_ref[1]+Dmin//2, 0:Dmin ]
        reference_eb     = image_eb_ref * CMASK * mask_eb
        ######reference_eb     = np.pad( reference_eb, ((0,image_eb_ref.shape[0]-reference_eb.shape[0]), (0,image_eb_ref.shape[1]-reference_eb.shape[1])), 'constant' )

        xcorr = crosscorrelation(image_eb, reference_eb, bp='yes', lp=int(max(image_eb.shape) / 12), hp=5, sigma=2)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])

        # tip position in the reference image :
        tip_x = right_outermost_point_eb_ref[0]
        tip_y = right_outermost_point_eb_ref[1]

        plt.figure()
        plt.subplot(3,1,1)
        plt.imshow(reference_eb, cmap='gray')
        plt.subplot(3,1,2)
        plt.imshow(image_eb, cmap='gray')
        plt.subplot(3,1,3)
        plt.imshow(xcorr, cmap='gray')
        plt.plot( [maxY], [maxX], 'ro' )


        plt.figure()
        plt.subplot(1,2,1)
        #plt.imshow(image_eb_ref, cmap='gray')
        plt.imshow(image_eb_ref, cmap='Blues_r', alpha=1)
        plt.imshow(image_eb, cmap='Oranges_r',   alpha=0.5)
        plt.plot([p[1] for p in filtered_coords_mask_eb_ref], [p[0] for p in filtered_coords_mask_eb_ref], 'bo')
        plt.plot([right_outermost_point_mask_eb_ref[1]], [right_outermost_point_mask_eb_ref[0]], 'ro')
        plt.plot([tip_y], [tip_x], 'rd')
        plt.subplot(1,2,2)
        plt.imshow(image_eb, cmap='gray')
        plt.plot( [maxY], [maxX], 'bx' )
        plt.plot([tip_y], [tip_x], 'rd')
        plt.plot([tip_y - err[1]], [tip_x - err[0]], 'ro')


        plt.show()









    if TEST == 2:
        ip_address = '10.0.0.1'
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        stage  = microscope.specimen.stage
        needle = microscope.specimen.manipulator
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)

        needle_images_no_bgnd = needle_reference_images(microscope)
        eb_needle_image_no_bgnd = needle_images_no_bgnd[0]
        ib_needle_image_no_bgnd = needle_images_no_bgnd[1]
        quick_plot(eb_needle_image_no_bgnd)
        quick_plot(ib_needle_image_no_bgnd)

        needle_x_shift, needle_y_shift = find_tip_in_needle_image(eb_needle_image_no_bgnd, median_smoothing=3, show=True)
        x_move = x_corrected_needle_movement(needle_x_shift)
        y_move = y_corrected_needle_movement(needle_y_shift, stage.current_position.t)
        #needle.relative_move(x_move)











    if TEST == 1:
        DIR = r'Y:\Sergey\codes\HS auto lamella1\01.25.2021_Needle_Images'
        fileName = DIR + '/' + 'E_needle_reference_2us_1536_HFW150um_001.tif'
        #fileName = DIR + '/' + 'I_needle_reference_3us_1536_HFW150um_002.tif'
        #fileName = DIR + '/' + '1_E_pp_50pA_250x_1000ns_1536_10.tif'
        #fileName = DIR + '/' + '1_I_pp_20pA_250x_1000ns_1536_1.tif' # NO SUCCESS
        #fileName = DIR + '/' + '1_I_pp_20pA_250x_1000ns_3072_1.tif'
        #fileName = DIR + '/' + '3_E_Z180_50pA_1200x_1000ns_3072_2.tif'



        imageTif = Image.open(fileName)
        imageTif = np.array(imageTif)
        if imageTif.shape[1] == 1536:
            imageTif = imageTif[0:1024, :]
        if imageTif.shape[1] == 3072:
            imageTif = imageTif[0:2048, :]
        imageTif = median(imageTif, disk(1))

        harrisim = compute_harris_response(imageTif)
        filtered_coords = get_harris_points(harrisim,2)
        right_outermost_point = max(filtered_coords, key=itemgetter(1))
        plt.figure()
        plt.gray()
        plt.imshow(imageTif)
        plt.plot( [p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'bo'  )
        plt.plot( [right_outermost_point[1]], [right_outermost_point[0]],   'ro')

        # dx = 200
        # patch = imageTif[ right_outermost_point[0]-dx : right_outermost_point[0]+dx, right_outermost_point[1]-dx : right_outermost_point[1]+dx ]
        # image_cut = np.ones( imageTif.shape)
        # image_cut[ right_outermost_point[1]-dx:right_outermost_point[1]+dx, right_outermost_point[0]-dx:right_outermost_point[0]+dx ] = patch
        # harrisim_cut = compute_harris_response(image_cut)
        # filtered_coords_cut = get_harris_points(harrisim_cut, 6)
        # right_outermost_point_cut = max(filtered_coords_cut, key=itemgetter(1))
        # plt.figure()
        # plt.gray()
        # plt.imshow(image_cut)
        # plt.plot( [p[1] for p in filtered_coords_cut],[p[0] for p in filtered_coords_cut],'o'  )
        # plt.plot( [right_outermost_point_cut[1]], [right_outermost_point_cut[0]],   'ro')
        # #plt.axis('off')
        # plt.show()

        ############ G's proceedings
        filt = median(imageTif, disk(5))
        thresh = threshold_otsu(filt)
        binary = filt > thresh
        mask = gaussian(binary_dilation(binary, iterations=15), 5)
        cmask = circ_mask( size=(imageTif.shape[1], imageTif.shape[0]), radius=int(min(imageTif.shape)/2*0.75), sigma=10   )
        ysize, xsize = imageTif.shape
        x_start = 100
        y_start = ysize//3
        expected_needle_cropped = binary[y_start:ysize, x_start:xsize//2]
        labels = label(expected_needle_cropped)
        needle_mask = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        needletip_x = np.argwhere(np.max(needle_mask, axis=0))[-1][0] + x_start
        needletip_y = np.argwhere(np.max(needle_mask, axis=1))[0][0] + y_start
        plt.figure()
        plt.imshow(imageTif, cmap='Blues_r', alpha=1)
        plt.imshow(mask, cmap='Oranges_r', alpha=0.5)
        plt.plot(  [needletip_x], [needletip_y], 'ro' )

        plt.figure()
        plt.imshow(imageTif * cmask, cmap='gray')
        plt.plot( [needletip_x], [needletip_y], 'ro' )
        plt.plot( [right_outermost_point[1]], [right_outermost_point[0]],   'ro')

        plt.show()
