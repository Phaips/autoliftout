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
from skimage.transform import rescale, resize, downscale_local_mean


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








import matplotlib.pyplot as plt
import numpy as np
from patrick.utils import load_model, model_inference, detect_and_draw_lamella_and_needle, scale_invariant_coordinates, calculate_distance_between_points, parse_metadata, show_overlay
from PIL import Image



#weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\12_04_2021_10_32_23_model.pt"
weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\fresh_full_n10.pt"
model = load_model(weights_file=weights_file)


#DIR = r'C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\liftout\cryo_machine_learning_20210510.110832\liftout000'
#DIR.replace('\\', '/')
#fileName = DIR + '/' + r'step11_A_ib_lowres.tif' # no success
#fileName = DIR + '/' + r'step12_A_ib_lowres.tif'
#fileName = DIR + '/' + r'step26_C_tiltAlign_sample_eb_highres.tif'
#fileName = DIR + '/' + r'step28_A_eb_lowres.tif'
#fileName = DIR + '/' + r'step41_A_eb_lowres_final.tif'
#fileName = DIR + '/' + r'step62_C_landingLamella_ib_highres_yShifted.tif'
#fileName = DIR + '/' + r'step43_B_needle_land_sample_ib_lowres.tif'
#fileName = DIR + '/' + r'step28_A_eb_lowres.tif'


DIR = r'C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\liftout\cryo_machine_learning_20210517.124020\liftout000'
DIR.replace('\\', '/')
fileName = DIR + '/' + r'step10_A_ib_lowres.tif' # no success

imageTif = Image.open(fileName)
img_orig = np.array(imageTif)
STD = np.std(img_orig)
MEAN  = np.mean(img_orig)

print('STD = ', STD, '; MEAN = ', MEAN)

histogram, bin_edges = np.histogram(img_orig, bins=256)
plt.figure(11)
plt.plot(bin_edges[0:-1], histogram, 'ob')  # <- or here

if MEAN > (255 - STD/2):
    print('1---------- doing renormalisation, recalibration and padding ----------------- ')
    indices = np.where(img_orig < MEAN  )
    img_orig[indices] = 0
elif (MEAN<(255/2. - STD/2) and (MEAN < 2*STD)):
    print('2---------- doing renormalisation, recalibration and padding ----------------- ')
    indices = np.where(img_orig > MEAN  )
    img_orig[indices] = 255

elif (MEAN<(255/2. - STD/2) and MEAN>=STD/2) or (MEAN>=(255/2. + STD/2) and MEAN <= (255 - STD/2)):
    print('3---------- doing renormalisation, recalibration and padding ----------------- ')
    # normalisation
    img_sigma = np.std(img_orig)
    img_mean  = np.mean(img_orig)
    img_orig =  (img_orig  - img_mean) / img_sigma

    ### filtering begin
    img_orig = img_orig - img_orig.min()
    img_sigma = np.std(img_orig)
    img_mean  = np.mean(img_orig)

    if MEAN<(255/2. - STD/2):
        indices = np.where( img_orig >= (img_mean + 2*img_sigma) )
    if MEAN>=(255/2. + STD/2):
        indices = np.where( img_orig <= (img_mean - 2*img_sigma) )
    img_orig[indices] = img_mean
    img_orig = img_orig/(img_mean + 2*img_sigma )  * 255
    img_orig = img_orig.astype(np.uint8)
    #### filtering end
    histogram3, bin_edges3 = np.histogram(img_orig, bins=256)


    ### renormalisation
    img_sigma = np.std(img_orig)
    img_mean  = np.mean(img_orig)
    img_orig =  (img_orig  - img_mean) / img_sigma
    #### BINNING, PADDING
    scale_factor = 1
    #image_resized = resize(img_orig, (img_orig.shape[0] // 2, img_orig.shape[1] // 2),  anti_aliasing=False)
    #image_resized = rescale(img_orig, 0.50, anti_aliasing=True)
    image_resized = skimage.transform.downscale_local_mean(img_orig, (scale_factor,scale_factor), cval=0, clip=True)
    cmask = circ_mask(size=(image_resized.shape[1], image_resized.shape[0]), radius=image_resized.shape[0]//scale_factor-80, sigma=20)  # circular mask
    img_orig = image_resized * cmask
    if scale_factor > 1:
        ddx = image_resized.shape[0]//scale_factor
        ddy = image_resized.shape[1]//scale_factor
        img_orig = np.pad(img_orig, ((ddx,ddx), (ddy,ddy)), 'constant'  )
    img_orig = img_orig/img_orig.max()  * 255
    img_orig = img_orig.astype(np.uint8)
    #img_orig = np.asarray(img.data)
    #img_orig1 = ndi.median_filter(img_orig1, size=4)

    # model inference + display

    #img_sigma = np.std(img_orig)
    #img_mean  = np.mean(img_orig)
    #img_orig =  (img_orig  - img_mean) / img_sigma


    plt.figure(12)
    plt.plot(bin_edges[0:-1], histogram, 'b')  # <- or here
    plt.figure(12)
    plt.plot(bin_edges3[0:-1], histogram3, 'r')  # <- or here











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








imageTif = Image.open(fileName)
img_orig = np.array(imageTif)
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





