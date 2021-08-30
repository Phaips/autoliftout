#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('matplotlib', 'qt')
from scipy import signal
import scipy.ndimage as ndi
from scipy import fftpack, misc
import scipy
import scipy.ndimage as ndi
import skimage.draw
import skimage.io
import datetime
import time
import os, glob

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from enum import Enum

from scipy.ndimage.morphology import binary_dilation
from skimage.filters import gaussian, threshold_otsu, median
from skimage.measure import label
from skimage.morphology import disk




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

def rect_mask(size=(128,128), sigma=None):
    # leave at least a 5% gap on each edge
    start = np.round(np.array(size) * 0.025)
    extent = np.round(np.array(size) * 0.95)
    rr, cc = skimage.draw.rectangle(start, extent=extent, shape=size)
    mask = np.zeros(size)
    mask[rr.astype(int), cc.astype(int)] = 1.0
    if sigma==None:
        sigma = min( start//2 )
    mask = ndi.gaussian_filter(mask, sigma=sigma)
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
    lowpass   = circ_mask(size=(x, y), radius=lp, sigma=0)
    hpass_tmp = circ_mask(size=(x, y), radius=hp, sigma=0)
    highpass = -1*(hpass_tmp - 1)
    tmp = lowpass * highpass
    if sigma > 0:
        bandpass = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        bandpass = tmp
    return bandpass

def crosscorrelation(img1, img2, bp='no', *args, **kwargs):
    # both images normalised :
    #img_norm = ( img - np.mean(img) ) / np.std(img)
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
        img1ft =         scipy.fftpack.fft2(img1)
        img2ft = np.conj(scipy.fftpack.fft2(img2))
        img1ft[0, 0] = 0
        xcorr = np.abs(scipy.fftpack.fftshift(scipy.fftpack.ifft2(img1ft * img2ft)))
    else:
        print('ERROR in xcorr2: bandpass value ( bp= ' + str(bp) + ' ) not recognized')
        return -1
    return xcorr



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

    return gx, gy

def gauss_derivatives(im, n, ny=None):
    """ returns x and y derivatives of an image using gaussian
        derivative filters of size n. The optional argument
        ny allows for a different size in the y direction."""
    gx, gy = gauss_derivative_kernels(n, sizey=ny)
    imx = signal.convolve(im, gx, mode='same')
    imy = signal.convolve(im, gy, mode='same')
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
    imx, imy = gauss_derivatives(image, 3)

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


def load_image(file_path):
    image = Image.open(file_path)
    image = np.array(image)
    return image

def normalise(img):
    img_normalised = (img - np.mean(img)) / np.std(img)
    return img_normalised

def rotate_180deg(img):
    rotated = np.rot90(np.rot90(np.copy(img)))
    return rotated

if __name__ == "__main__":
    # check the parameters of cross correlation
    # electron, low res, dwell time for refrence images

    # some older files for testing
    #DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.03_needle_image'
    #fileName_sample    = r'sample_eb_lowres.tif'
    #needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
    #fileName_ref       = r'ref_eb_lowres.tif'
    #needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)

    # ion, low res, dwell time for refrence images 10us, ion beam image strongly shifted
    # DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.09_needle_image_C'
    # fileName_sample = r'sample_ib_lowres_02_ebXYalign.tif'
    # image_1 = read_image(DIR, fileName_sample)
    # fileName_ref = r'ref_ib_lowres.tif'
    # image_2 = read_image(DIR, fileName_ref)

    # autocontrast images, yeast cells on grid, 20 keV Ar @ 20 pA, electron 2 keV @ 50 pA
    images = { 'eb_lowres' : r'Y:\Sergey\codes\images\20210823.143636_autocontrast\img\01_ib_drift_correction_lamella_low_res_0_eb.tif',
               'ib_lowres' : r'Y:\Sergey\codes\images\20210823.143636_autocontrast\img\01_ref_trench_low_res_eb_ib.tif',
               'eb_highres' : r'Y:\Sergey\codes\images\20210823.143636_autocontrast\img\01_drift_correction_ML20210823.145237_eb_ib_eb.tif',
               'ib_highres' : r'Y:\Sergey\codes\images\20210823.143636_autocontrast\img\01_ref_trench_high_res_eb_ib.tif',
               'eb_highres_shifted' : r'Y:\Sergey\codes\images\20210823.143636_autocontrast\img\01_drift_correction_ML20210823.145322_eb_ib_label.tif'}

    ############## low res images ##############
    image_1 = load_image( images['eb_lowres'] )
    image_2 = load_image( images['ib_lowres'] )
    ############## high res images #############
    #image_1 = load_image( images['eb_highres_shifted'] )
    #image_2 = load_image( images['ib_highres'] )
    ############# rotate the second image 180 deg for cross correlation #############
    image_2 = rotate_180deg(image_2)

    # rectangular mask to fluff the sharp edges
    mask = rect_mask(image_1.shape)

    # normalise the images for reliable crosscorrelation
    image_1 = normalise(image_1)
    image_2 = normalise(image_2)
    # smooth the noise out for plotting
    image_1_smooth = ndi.filters.gaussian_filter( image_1 * mask, sigma=3)
    image_2_smooth = ndi.filters.gaussian_filter( image_2 * mask, sigma=3)


    ############# FIRST QUICK TEST of PARAMETERS lp hp sigma, MANUAL ADJUSTMENT TO SEE THE SHIFTS GIVE ALIGNED IMAGES
    xcorr = crosscorrelation(image_1 * mask, image_2 * mask, bp='yes', lp=256, hp=24, sigma=10)
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    xcorr_01_max     = xcorr.max()
    xcorr_01_max_3x3 = np.sum(xcorr[maxX - 1:maxX + 2, maxY - 1:maxY + 2])
    print('\n', maxX, maxY)
    cen = np.asarray(xcorr.shape) / 2
    print('centre = ', cen)
    err = np.array(cen - [maxX, maxY], int)
    print("Shift between 1 and 2 is = " + str(err))
    print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
    shift_01_x = cen[0] - maxX
    shift_01_y = cen[1] - maxY



    aligned = np.copy(image_1)
    aligned = np.roll(aligned, +int(shift_01_x), axis=0)
    aligned = np.roll(aligned, +int(shift_01_y), axis=1)
    aligned_smooth = ndi.filters.gaussian_filter( aligned, sigma=3)


    #############
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow( image_1_smooth, cmap='gray')
    plt.title('image_1: to align')
    plt.subplot(2,2,2)
    plt.title('image_2: reference')
    plt.imshow( image_2_smooth, cmap='gray')
    plt.subplot(2,2,3)
    plt.title('image 1,2: overlay')
    plt.imshow( image_1_smooth, cmap='Blues_r',   alpha=0.5)
    plt.imshow( image_2_smooth, cmap='Oranges_r', alpha=0.5)
    plt.subplot(2,2,4)
    plt.title('image 1 shifted to align with 2')
    plt.imshow( image_2, cmap='Blues_r',   alpha=0.5)
    plt.imshow( aligned_smooth, cmap='Oranges_r', alpha=0.5)

    # scan the parameters hp lp, also sigma needs to be ideally scanned
    # bp lp sigma as a function of image resolution image.shape
    if 0:
        SIGMA = int( max(image_1.shape)/1536  * 10 ) # need to scan SIGMA too, from 1 to 10 or something like that
        sigma_max = int( max(image_1.shape)/1536 * 10 )
        LOW_PASS_FILTER  = np.arange(1, max(image_1.shape)//2  + 1, 10)
        HIGH_PASS_FILTER = np.arange(1, max(image_1.shape)//32 + 1, 10)

        NN = len(LOW_PASS_FILTER)
        Nx = len(LOW_PASS_FILTER)
        Ny = len(HIGH_PASS_FILTER)
        x_shift_matrix = np.zeros([Nx, Ny])
        y_shift_matrix = np.zeros([Nx, Ny])
        X = []
        Y = []
        LP = []
        HP = []

        stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')
        DIR = os.getcwd()
        DIR = DIR + '/' + stamp
        os.mkdir(DIR)

        for ii in range(len(HIGH_PASS_FILTER)):

            high_pass_filter = HIGH_PASS_FILTER[ii]
            dataOut = np.zeros([NN, 5])
            saveFile =  DIR + '/' + 'ionToElectronAlign_LP_x_y_xcorrMax_xCorrIntergMax_' + 'sigma_%d'%(SIGMA) +  '_hp_%02d'%(high_pass_filter) + '.txt'

            #eee = 0
            for jj in range(len(LOW_PASS_FILTER)):
                low_pass_filter = LOW_PASS_FILTER[jj]
                HP.append(high_pass_filter)
                LP.append(low_pass_filter)
                print(': High pass filter = ', high_pass_filter, '; low pass filter = ', low_pass_filter)
                xcorr = crosscorrelation(image_1 * mask, image_2 * mask, bp='yes', lp=low_pass_filter, hp=high_pass_filter, sigma=SIGMA)
                maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
                xcorr_max = xcorr.max() # maximum value of the crosscorrelation, may be used to check where the crosscorrelation is the strongest
                xcorr_max_3x3 = np.sum(xcorr[maxX - 1:maxX + 2, maxY - 1:maxY + 2]) #same as maximum correlation value, but integrating around the peak
                #print(maxX, maxY)
                cen = np.asarray(xcorr.shape) / 2
                #print('centre = ', cen)
                err = np.array(cen - [maxX, maxY], int)
                #print("Shift between 1 and 2 is = " + str(err))
                print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0], '\n')
                shift_x = cen[0] - maxX
                shift_y = cen[1] - maxY

                x_shift_matrix[jj,ii] = shift_x
                y_shift_matrix[jj,ii] = shift_y
                X.append(shift_x)
                Y.append(shift_y)

                dataOut[jj, 0] = low_pass_filter
                dataOut[jj, 1] = shift_x
                dataOut[jj, 2] = shift_y
                dataOut[jj, 3] = xcorr_max     # maximum value of the correlation
                dataOut[jj, 4] = xcorr_max_3x3 # maximum value of the correlation, integral about the peak

                #eee += 1

            plt.figure(22)
            plt.subplot(2,1,1)
            plt.plot(LOW_PASS_FILTER, dataOut[:, 1], label=str(high_pass_filter))
            plt.subplot(2,1,2)
            plt.plot(LOW_PASS_FILTER, dataOut[:, 2], label=str(high_pass_filter))

            np.savetxt(saveFile, dataOut, fmt='%f')

    plt.figure(33)
    plt.imshow( np.transpose((x_shift_matrix)), aspect='auto', cmap='jet', extent=[LOW_PASS_FILTER.min(), LOW_PASS_FILTER.max(), HIGH_PASS_FILTER.max(), HIGH_PASS_FILTER.min()])
    plt.title('From manual test dy is ~ +70')

    plt.figure(34)
    plt.imshow( np.transpose((y_shift_matrix)), aspect='auto', cmap='jet', extent=[LOW_PASS_FILTER.min(), LOW_PASS_FILTER.max(), HIGH_PASS_FILTER.max(), HIGH_PASS_FILTER.min()])
    plt.title('From manual test dy is ~ -250')


    plt.legend()
    plt.show()
