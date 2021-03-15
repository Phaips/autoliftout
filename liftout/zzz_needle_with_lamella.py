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
        storage.SaveImage(needle_ib_highres_with_lamella, id='B_ref_ib_highres')
        ############ FIND dx, dy from HIGH_RES ELECTRON images ############
        x_shift, y_shift = find_needle_tip_shift_in_image_ELECTRON(needle_eb_lowres_with_lamella, needle_ref_eb_lowres_nobg, show=False, median_smoothing=2)
        x_move = x_corrected_needle_movement(x_shift)
        y_move = y_corrected_needle_movement(y_shift, stage.current_position.t)
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
        x_shift, y_shift = find_needle_tip_shift_in_image_ELECTRON(needle_eb_highres_with_lamella_shifted, needle_ref_eb_highres_nobg, show=False, median_smoothing=2)
        x_move = x_corrected_needle_movement(x_shift)
        y_move = y_corrected_needle_movement(y_shift, stage.current_position.t)
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
        x_shift, y_shift = find_needle_tip_shift_in_image_ION(needle_ib_lowres_with_lamella_shifted, needle_ref_ib_lowres_nobg, show=False, median_smoothing=2)
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















    if TEST == 3.3:
        print('different lamella corners detection methods')
        pixelsize_x = 1
        pixelsize_y = 1

        #######################################################################################
        '''TYPE = 'ION'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.115423\liftout000'
        fileName_sample = r'step00_H_landingLamella_ib_highres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.115423\liftout000'
        fileName_ref = r'step00_G_ref_ib_highres_LandingPosition.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        '''TYPE = 'ELECTRON'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.115423\liftout000'
        fileName_sample = r'step00_H_landingLamella_eb_highres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.115423\liftout000'
        fileName_ref = r'step00_G_ref_eb_highres_LandingPosition.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        #######################################################################################
        '''TYPE = 'ION'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.140126\liftout000'
        fileName_sample = r'step00_H_landingLamella_ib_highres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.140126\liftout000'
        fileName_ref = r'step00_G_ref_ib_highres_LandingPosition.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        '''TYPE = 'ELECTRON'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.140126\liftout000'
        fileName_sample = r'step00_H_landingLamella_eb_highres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.140126\liftout000'
        fileName_ref = r'step00_G_ref_eb_highres_LandingPosition.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        #######################################################################################
        TYPE = 'ION'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.145719\liftout000'
        fileName_sample = r'step00_H_landingLamella_ib_highres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.145719\liftout000'
        fileName_ref = r'step00_G_ref_ib_highres_LandingPosition.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)

        '''TYPE = 'ELECTRON'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.145719\liftout000'
        fileName_sample = r'step00_H_landingLamella_eb_highres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_20210223.145719\liftout000'
        fileName_ref = r'step00_G_ref_eb_highres_LandingPosition.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        #######################################################################################
        '''TYPE = 'ION'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_lowDose_20210223.154612\liftout000'
        fileName_sample = r'step00_H_landingLamella_ib_highres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_lowDose_20210223.154612\liftout000'
        fileName_ref = r'step00_G_ref_ib_highres_LandingPosition.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''

        '''TYPE = 'ELECTRON'
        DIR1 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_lowDose_20210223.154612\liftout000'
        fileName_sample = r'step00_H_landingLamella_eb_highres.tif'
        needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=4)
        DIR2 = r'Y:\Sergey\codes\HS auto lamella1\test_needle_landing_lowDose_20210223.154612\liftout000'
        fileName_ref = r'step00_G_ref_eb_highres_LandingPosition.tif'
        needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=4)'''


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
        filter_type = 'median'
        #filter_type = 'fourier'
        if filter_type == 'median':
            #filt   = ndi.median_filter(needle_reference * rect_mask, size=5)
            filt   = ndi.gaussian_filter(needle_reference * rect_mask, sigma=5)
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
        height, width = needle_reference.shape # search for the tip in only the left half of the image
        if TYPE=='ELECTRON':
            median_filter_size = int( max(needle_reference.shape) / 512 )  #5@3072x2048 #milder
            harrisim = compute_harris_response( ndi.median_filter( (needle_reference * mask_blurred * rect_mask)[:, 0:width//2], size=median_filter_size) )
        elif TYPE=='ION':
            median_filter_size = int( max(needle_reference.shape) / 307 )  #10@3072x2048 #stronger noise filtering
            harrisim = compute_harris_response( ndi.median_filter( (needle_reference * mask_blurred * rect_mask)[:, 0:width//2], size=median_filter_size) )
        print('=====median filter size to remove noise = ',  median_filter_size)
        corner_threshold = max(harrisim.ravel()) * 0.1
        harrisim_t = (harrisim > corner_threshold) * 1
        filtered_coords_ref = get_harris_points(harrisim,4)
        right_outermost_point_harris = max(filtered_coords_ref, key=itemgetter(1))
        topmost_point_harris         = min(filtered_coords_ref, key=itemgetter(0))

        lamella_tip_length = 1.2e-6 # search for all the corners to left from the rightmost point by the distance of the tip
        #lamella_tip_pixels = lamella_tip_length / pixelsize_x
        lamella_tip_pixels = lamella_tip_length / (80e-6/3072)
        lamella_tip_coords = [ii for ii in filtered_coords_ref if ii[1]>=(right_outermost_point_harris[1]-lamella_tip_pixels)  ]
        tip_topmost_point_harris    = min(lamella_tip_coords, key=itemgetter(0))
        tip_bottommost_point_harris = max(lamella_tip_coords, key=itemgetter(0))
        tip_y_top    = tip_topmost_point_harris[0]
        tip_y_bottom = tip_bottommost_point_harris[0]

        tip_x_center = right_outermost_point_harris[1]
        tip_y_center = (tip_y_top + tip_y_bottom) // 2



        if 1:
            plt.figure(1)
            plt.imshow(needle_reference, cmap='gray', alpha=1)
            plt.imshow(mask_blurred, cmap='gray', alpha=0.5)
            plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
            plt.plot( [p[1] for p in lamella_tip_coords],[p[0] for p in lamella_tip_coords],'mx'  )
            plt.plot( [right_outermost_point_harris[1]], [right_outermost_point_harris[0]],   'ro', label='rightmost')
            plt.plot( [topmost_point_harris[1]], [topmost_point_harris[0]],   'rd', label='topmost')
            plt.plot( [tip_x_center], [tip_y_center],   'md', label='center')
            plt.title('Harris corner detection')
            plt.legend()


        ############## sobel edge detection
        sx = ndi.sobel(needle_reference * mask_blurred * rect_mask, axis=0, mode='constant')
        sy = ndi.sobel(needle_reference * mask_blurred * rect_mask, axis=1, mode='constant')
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
        edges = feature.canny( needle_reference * mask_blurred * rect_mask, sigma=sigma_blur )
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
        old_tip_x = right_outermost_point_harris[0]
        old_tip_y = right_outermost_point_harris[1]

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
        if 1:
            plt.figure()
            plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
            plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
            plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='original position')
            plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='current position')
            plt.legend()






        ##### detection of the edge of the landing post
        filter_type = 'fourier'
        if filter_type == 'median':
            filt   = ndi.median_filter(needle_with_sample * rect_mask, size=5)
        elif filter_type == 'fourier':
            print(': High pass filter = ', lowpass_pixels, '; low pass filter = ', highpass_pixels, '; sigma = ', sigma)
            bandpass = bandpass_mask(size=(needle_with_sample.shape[1], needle_with_sample.shape[0]), lp=lowpass_pixels, hp=highpass_pixels, sigma=sigma)
            img1ft = bandpass * scipy.fftpack.fftshift(scipy.fftpack.fft2(needle_with_sample * rect_mask))
            filt = np.real(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(((img1ft)))))
            #plt.imshow(filt, cmap='gray')

        maxRight = (needle_with_sample * rect_mask)[:, 1200:].max()
        needle_with_sample_binary = (needle_with_sample >= maxRight*0.5).astype(int)
        # plt.imshow(needle_with_sample_binary, cmap='gray')
        maxRight = filt[:, 1200:].max()
        filt_binary = (filt >= maxRight*0.5).astype(int)
        # plt.imshow(filt_binary, cmap='gray')

        ############## canny edge detection
        canny_edges_sample = feature.canny( needle_with_sample * rect_mask , sigma=sigma )
        canny_edges_sample = canny_edges_sample.astype(int)
        # plt.imshow(canny_edges_sample, cmap='gray')

        ############## sobel edge detection
        sx = ndi.sobel(filt * rect_mask, axis=0, mode='constant')
        sy = ndi.sobel(filt * rect_mask, axis=1, mode='constant')
        sobel_edges_sample = np.hypot(sx, sy)
        sobelMaxRight = sobel_edges_sample[:,1200:].max()
        sobel_edges_sample_binary = ( sobel_edges_sample >= sobelMaxRight*0.25 ).astype(int)
        #sobel_edges_sample_median = ndi.median_filter(sobel_edges_sample , size=20)
        # plt.imshow(sobel_edges_sample_binary, cmap='gray')

        ############## laplacian
        sample_laplacian = ndi.laplace(filt * rect_mask)
        sample_laplacian_binary = (sample_laplacian >= sample_laplacian.max()*0.25).astype(int)
        #sample_laplacian_median = ndi.median_filter(sample_laplacian , size=20)
        # plt.imshow(sample_laplacian_median, cmap='gray')

        ###################################################################################
        # Classic straight-line Hough transform
        # Set a precision of 0.5 degree.
        tested_angles = np.linspace(-np.deg2rad(25), np.deg2rad(25), 720, endpoint=False)
        h, theta, d = hough_line( canny_edges_sample, theta=tested_angles)
        DISTANCES = []
        ANGLES    = []
        plt.figure(40)
        plt.imshow(needle_with_sample , cmap=cm.gray,  alpha=1 )
        plt.imshow(filt_binary, cmap=cm.gray,  alpha=0.5 )
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            print(dist, np.rad2deg(angle))
            DISTANCES.append(dist )
            ANGLES.append(np.rad2deg(angle))
            x0 = dist * np.cos(angle)
            y0 = dist * np.sin(angle)
            plt.scatter(x0,y0)
            if angle!=0:
                k = -np.cos(angle) / np.sin(angle)
                b = dist/np.sin(angle)
                x1 = 0
                y1 = k*x1 + b
                x2 = 3071
                y2 = k*x2 + b
                plt.plot( [x1, x2], [y1, y2] )
            elif angle==0:
                x1 = 0
                y1 = x1
                x2 = 2048
                y2 = x1
            plt.plot([x1, x2], [y1, y2], '--w', linewidth=2)

        plt.xlim(xmin=0, xmax=3072)
        plt.ylim(ymin=2048, ymax=0)




        plt.show()








