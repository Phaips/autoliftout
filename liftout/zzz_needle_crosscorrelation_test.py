#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('matplotlib', 'qt')
from scipy import *
from scipy import signal
import scipy.ndimage as ndi
from scipy import fftpack, misc
import scipy
import scipy.ndimage as ndi


from pylab import *
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from operator import itemgetter
from enum import Enum

from scipy.ndimage.morphology import binary_dilation
from skimage.filters import gaussian, threshold_otsu, median
from skimage.measure import label
from skimage.morphology import disk


class BeamType(Enum):
    ION = 'ION'
    ELECTRON = 'ELECTRON'


PRETILT_DEGREES = 27


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


















if __name__ == "__main__":
    TEST = 4.2

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


################################################################################################################################################
################################################################################################################################################

    if TEST == 5: # check the parameters of cross correlation
        # electron, low res, dwell time for refrence images 20us
        #DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.03_needle_image'
        #fileName_sample    = r'sample_eb_lowres.tif'
        #needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
        #fileName_ref       = r'ref_eb_lowres.tif'
        #needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)

        # ion, low res, dwell time for refrence images 10us, ion beam image strongly shifted
        DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.09_needle_image_C'
        fileName_sample = r'sample_ib_lowres_02_ebXYalign.tif'
        needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
        fileName_ref = r'ref_ib_lowres.tif'
        needle_reference = read_image(DIR, fileName_ref, gaus_smooth=0)

        needle_reference_norm   = ( needle_reference - np.mean(needle_reference) ) / np.std(needle_reference)
        needle_with_sample_norm = ( needle_with_sample - np.mean(needle_with_sample) ) / np.std(needle_with_sample)

        SIGMA = 2
        #SAVEFILE = DIR + '/' + 'electronBeam_lowRes_LP_x_y_simpleCorr_corrMaskedTip_corrCircMaskTip_'

        ### Find the tip using corner-finding algorithm
        harrisim = compute_harris_response( ndi.median_filter(needle_reference, size=2) )
        filtered_coords_ref = get_harris_points(harrisim,4)
        right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))
        #############
        plt.figure(1)
        plt.imshow(needle_reference, cmap='gray')
        plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
        plt.plot( [right_outermost_point_ref[1]], [right_outermost_point_ref[0]],   'ro')

        ### Find the tip from binarized image (mask) and corner-finding algorithm
        filt   = median(needle_reference, disk(2))
        thresh = threshold_otsu(filt)
        binary = filt > thresh
        mask   = gaussian(binary_dilation(binary, iterations=15), 5)
        mask_binary = (mask >= 0.51).astype(int)
        needle_ref_masked = needle_reference * mask_binary
        ysize, xsize = mask.shape
        harrisim = compute_harris_response(mask_binary)
        filtered_coords_mask_ref = get_harris_points(harrisim,4)
        right_outermost_point_mask_ref = max(filtered_coords_mask_ref, key=itemgetter(1))
        ####
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
        cmask = circ_mask(size=(Dmin, Dmin), radius=Dmin // 2 - 15, sigma=10)  # circular mask
        CMASK = np.zeros(needle_reference.shape)
        CMASK[old_tip_x - Dmin // 2: old_tip_x + Dmin // 2, old_tip_y - Dmin // 2: old_tip_y + Dmin // 2] = cmask
        reference           = needle_reference      * CMASK * mask
        reference_circ_norm = needle_reference_norm * CMASK * mask

        xmin = min(old_tip_x, mask.shape[1] - old_tip_x)
        ymin = min(old_tip_y, mask.shape[0] - old_tip_y)
        ELLPS_MASK = np.zeros(needle_reference.shape)
        elps_mask = ellipse_mask(size=(xmin*2, ymin*2), radius1=xmin-15, radius2=ymin-15, sigma=10)
        ELLPS_MASK[old_tip_y - ymin : old_tip_y + ymin, old_tip_x - xmin : old_tip_x + xmin] = elps_mask
        reference_elps_norm = needle_reference_norm * ELLPS_MASK * mask


        LOW_PASS_FILTER  = np.arange(32, 256, 2)#range( max(needle_with_sample.shape)//2  )
        HIGH_PASS_FILTER = np.arange(1, 25+1, 1)#range( max(needle_with_sample.shape)//100)
        NN = len(LOW_PASS_FILTER)
        dataOut = np.zeros([NN, 11])


        for high_pass_filter in HIGH_PASS_FILTER:
            dataOut = np.zeros([NN, 9])
            saveFile = DIR + '/' + 'ionBeam_lowRes_LP_x_y_simpleCorr_corrMaskedTip_corrCircMaskTip_corrEllpMaskTip_' + 'sigma_%d'%(SIGMA) +  '_hp_%02d'%(high_pass_filter) + '.txt'
            eee = 0
            for low_pass_filter in LOW_PASS_FILTER:
                print(eee, ': High pass filter = ', high_pass_filter, '; low pass filter = ', low_pass_filter)

                xcorr = crosscorrelation(needle_with_sample_norm, needle_reference_norm, bp='yes', lp=low_pass_filter, hp=high_pass_filter, sigma=SIGMA)
                maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
                print('\n', maxX, maxY)
                cen = np.asarray(xcorr.shape) / 2
                print('centre = ', cen)
                err = np.array(cen - [maxX, maxY], int)
                print("Shift between 1 and 2 is = " + str(err))
                print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
                new_tip_x = old_tip_x - err[0]
                new_tip_y = old_tip_y - err[1]
                shift_01_x = cen[0] - maxX
                shift_01_y = cen[1] - maxY

                xcorr = crosscorrelation(needle_with_sample_norm, needle_reference_norm * mask_binary, bp='yes', lp=low_pass_filter, hp=high_pass_filter, sigma=SIGMA)
                maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
                print('\n', maxX, maxY)
                cen = np.asarray(xcorr.shape) / 2
                print('centre = ', cen)
                err = np.array(cen - [maxX, maxY], int)
                print("Shift between 1 and 2 is = " + str(err))
                print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
                new_tip_x = old_tip_x - err[0]
                new_tip_y = old_tip_y - err[1]
                shift_02_x = cen[0] - maxX
                shift_02_y = cen[1] - maxY

                xcorr = crosscorrelation(needle_with_sample_norm, reference_circ_norm, bp='yes', lp=low_pass_filter, hp=high_pass_filter, sigma=SIGMA)
                maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
                print('\n', maxX, maxY)
                cen = np.asarray(xcorr.shape) / 2
                print('centre = ', cen)
                err = np.array(cen - [maxX, maxY], int)
                print("Shift between 1 and 2 is = " + str(err))
                print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
                new_tip_x = old_tip_x - err[0]
                new_tip_y = old_tip_y - err[1]
                shift_03_x = cen[0] - maxX
                shift_03_y = cen[1] - maxY

                xcorr = crosscorrelation(needle_with_sample_norm, reference_elps_norm, bp='yes', lp=low_pass_filter, hp=high_pass_filter, sigma=SIGMA)
                maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
                print('\n', maxX, maxY)
                cen = np.asarray(xcorr.shape) / 2
                print('centre = ', cen)
                err = np.array(cen - [maxX, maxY], int)
                print("Shift between 1 and 2 is = " + str(err))
                print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
                new_tip_x = old_tip_x - err[0]
                new_tip_y = old_tip_y - err[1]
                shift_04_x = cen[0] - maxX
                shift_04_y = cen[1] - maxY

                dataOut[eee, 0] = low_pass_filter
                dataOut[eee, 1] = shift_01_x
                dataOut[eee, 2] = shift_01_y
                dataOut[eee, 3] = shift_02_x
                dataOut[eee, 4] = shift_02_y
                dataOut[eee, 5] = shift_03_x
                dataOut[eee, 6] = shift_03_y
                dataOut[eee, 7] = shift_04_x
                dataOut[eee, 8] = shift_04_y

                eee += 1
            np.savetxt(saveFile, dataOut, fmt='%f')





    if TEST == 4.2:
        if 0: # electron, low res, dwell time for refrence images 20us
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.09_needle_image_C'
            #fileName_sample = r'sample_eb_lowres.tif'
            fileName_sample    = r'sample_eb_lowres_02_ebXYalign.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_eb_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: # ion, low res
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.09_needle_image_C'
            fileName_sample    = r'sample_ib_lowres.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_ib_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 1: # ion, low res
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.09_needle_image_C'
            fileName_sample    = r'sample_ib_lowres_02_ebXYalign.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_ib_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)


        if 0: # electron, low res, dwell time for refrence images 20us
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.03_needle_image'
            fileName_sample    = r'sample_eb_lowres.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_eb_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: # electron, high res
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.03_needle_image'
            fileName_sample    = r'sample_eb_highres.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_eb_highres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: # ion, low res
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.03_needle_image'
            fileName_sample    = r'sample_ib_lowres.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_ib_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: # ion, high res
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.03_needle_image'
            fileName_sample    = r'sample_ib_highres.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_ib_highres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)

        # Different set of data, dwell time for reference images 2 us
        if 0: # eb, DOES NOT WORK, too noisy
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
            fileName_sample    = r'electron_beam.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'reference_eb_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: #
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
            fileName_sample    = r'ion_beam.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'reference_ib_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)


        # Different set of data, dwell time for reference images 2 us
        if 0: # eb, DOES NOT WORK, too noisy
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
            fileName_sample    = r'electron_beam.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'reference_eb_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: #
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
            fileName_sample    = r'ion_beam.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'reference_ib_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)

        # Different set of data, dwell time for reference images 2 us
        if 0:
            DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
            fileName_sample = r'electron_beam.tif'
            needle_with_sample = read_image(DIR1, fileName_sample, gaus_smooth=0)
            DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
            fileName_ref = r'reference_eb_lowres.tif'
            needle_reference = read_image(DIR2, fileName_ref, gaus_smooth=0)
        if 0:
            DIR1 = r'Y:\Sergey\codes\HS auto lamella1\2021.01.28_needle_images\B_inserted_position_moved_closerX20um_Z180'
            fileName_eb = r'001_e_needle_movedCloser_A_hfw150.tif'
            needle_with_sample = read_image(DIR1, fileName_eb, gaus_smooth=0)
            DIR2 = r'Y:\Sergey\codes\HS auto lamella1\2021.01.28_needle_images\C_reference_needle_images_no_background_insertedPosition'
            fileName_eb_ref = r'001_e_needle_movedCloser_A_hfw150.tif'
            needle_reference = read_image(DIR2, fileName_eb_ref, gaus_smooth=0)


        ### Find the tip using corner-finding algorithm
        height, width = needle_reference.shape # search for the tip in only the left half of the image
        harrisim = compute_harris_response( ndi.median_filter(needle_reference[:, 0:width//2], size=1) )
        #harrisim = compute_harris_response(needle_reference)
        filtered_coords_ref = get_harris_points(harrisim,4)
        right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))
        #############
        plt.figure(1)
        plt.imshow(needle_reference, cmap='gray')
        plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
        plt.plot( [right_outermost_point_ref[1]], [right_outermost_point_ref[0]],   'ro')

        ### Find the tip from binarized image (mask) and corner-finding algorithm
        filt   = median(needle_reference, disk(2))
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
        '''
        needle_reference_sqr = np.pad(needle_reference*mask, ( (256,256), (0,0) ), 'constant'  )
        bandpass = bandpass_mask(size=(needle_reference.shape[1], needle_reference.shape[0]), lp=128, hp=22, sigma=10)
        img1ft = bandpass * scipy.fftpack.fftshift(scipy.fftpack.fft2(   ndi.filters.gaussian_filter(needle_reference, sigma=2)   ))
        qqq = np.real(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(((img1ft)) )))
        plt.imshow(qqq, cmap='gray')
        plt.imshow( np.log( np.abs(img1ft) ) )
        plt.imshow(ndi.filters.gaussian_filter(needle_reference_sqr, sigma=0))
        plt.imshow( np.log( np.abs(img1ft) + 1  ) )
        plt.imshow(bandpass)'''

        LOWPASS = 256
        HIGHPASS = 22
        print(': High pass filter = ', 128, '; low pass filter = ', 5)
        needle_with_sample_norm = ( needle_with_sample - np.mean(needle_with_sample) ) / np.std(needle_with_sample)
        xcorr = crosscorrelation(needle_with_sample_norm, reference_circ_norm, bp='yes', lp=LOWPASS, hp=HIGHPASS, sigma=10)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        new_tip_x = old_tip_x - err[0]
        new_tip_y = old_tip_y - err[1]

        x_shift = ( cen[1] - new_tip_y )
        y_shift = ( cen[0] - new_tip_x )
        print("X-shift to the image centre =  {} meters".format(x_shift))
        print("Y-shift to the image centre =  {} meters".format(y_shift))

        # LOW_PASS_FILTER  = [64, 128, 256]#range( max(needle_with_sample.shape)//2  )
        # HIGH_PASS_FILTER = [1,2, 3, 4, 5 ]#range( max(needle_with_sample.shape)//100)
        # SIGMA = 2
        # NN = len(LOW_PASS_FILTER)
        # XSHIFT = []
        # YSHIFT = []
        #        for high_pass_filter in HIGH_PASS_FILTER:
        #            for low_pass_filter in LOW_PASS_FILTER:
        #                print(': High pass filter = ', high_pass_filter, '; low pass filter = ', low_pass_filter)
        #                xcorr = crosscorrelation(needle_with_sample, needle_reference, bp='yes', lp=low_pass_filter, hp=high_pass_filter, sigma=SIGMA)
        #                maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        #                print('\n', maxX, maxY)
        #                cen = np.asarray(xcorr.shape) / 2
        #                print('centre = ', cen)
        #                err = np.array(cen - [maxX, maxY], int)
        #                print("Shift between 1 and 2 is = " + str(err))
        #                print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        #                x_shift = cen[0] - maxX
        #                y_shift = cen[1] - maxY
        #                XSHIFT.append(x_shift)
        #                YSHIFT.append(y_shift)
        #
        #        XSHIFT = np.array(XSHIFT)
        #        YSHIFT = np.array(YSHIFT)
        #        x_shift_mean = np.mean(XSHIFT, axis=0)
        #        y_shift_mean = np.mean(YSHIFT, axis=0)
        #        new_tip_x_mean = old_tip_x - x_shift_mean
        #        new_tip_y_mean = old_tip_y - y_shift_mean

        plt.figure()
        plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
        plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
        plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='original position')
        plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='current position')
        plt.legend()
        #plt.title('reference needle image WITH needle over the sample')


        plt.figure()
        plt.subplot(2,3,1)
        plt.imshow(  ndi.median_filter(needle_reference_norm, size=2), cmap='gray')
        plt.subplot(2,3,2)
        plt.imshow(  ndi.median_filter(needle_with_sample_norm, size=2), cmap='gray')
        plt.subplot(2,3,3)
        plt.imshow(reference_elps_norm, cmap='gray')
        plt.subplot(2,3,4)
        bandpass = bandpass_mask(size=(needle_reference.shape[1], needle_reference.shape[0]), lp=128, hp=5, sigma=5)
        img1ft = bandpass * scipy.fftpack.fftshift(scipy.fftpack.fft2(   ndi.filters.gaussian_filter(needle_reference, sigma=2)   ))
        qqq = np.real(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(((img1ft)) )))
        plt.imshow(qqq, cmap='gray')
        plt.title("lowpass = " + str(LOWPASS) + '; highpass = ' + str(HIGHPASS) )
        plt.subplot(2,3,5)
        plt.imshow(xcorr, cmap='gray')
        plt.plot( [maxY], [maxX], 'ro' )
        plt.title("img2 is X-shifted by " +  str(err[1]) + '; Y-shifted by ' +  str(err[0]))
        plt.subplot(2,3,6)
        plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
        plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
        plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='original position')
        plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='current position')
        plt.legend()
        plt.show()




    if TEST == 5.2:
        if 0: # electron, low res, dwell time for refrence images 20us
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.09_needle_image_C'
            #fileName_sample = r'sample_eb_lowres.tif'
            fileName_sample    = r'sample_eb_lowres_02_ebXYalign.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_eb_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: # ion, low res
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.09_needle_image_C'
            fileName_sample    = r'sample_ib_lowres.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_ib_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 1: # ion, low res
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.09_needle_image_C'
            fileName_sample    = r'sample_ib_lowres_02_ebXYalign.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_ib_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)

        ### Find the tip using corner-finding algorithm
        height, width = needle_reference.shape # search for the tip in only the left half of the image
        harrisim = compute_harris_response( ndi.median_filter(needle_reference[:, 0:width//2], size=1) )
        #harrisim = compute_harris_response(needle_reference)
        filtered_coords_ref = get_harris_points(harrisim,4)
        right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))
        #############
        plt.figure(1)
        plt.imshow(needle_reference, cmap='gray')
        plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
        plt.plot( [right_outermost_point_ref[1]], [right_outermost_point_ref[0]],   'ro')

        ### Find the tip from binarized image (mask) and corner-finding algorithm
        filt   = median(needle_reference, disk(2))
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
        '''
        needle_reference_sqr = np.pad(needle_reference*mask, ( (256,256), (0,0) ), 'constant'  )
        bandpass = bandpass_mask(size=(needle_reference.shape[1], needle_reference.shape[0]), lp=128, hp=22, sigma=10)
        img1ft = bandpass * scipy.fftpack.fftshift(scipy.fftpack.fft2(   ndi.filters.gaussian_filter(needle_reference, sigma=2)   ))
        qqq = np.real(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(((img1ft)) )))
        plt.imshow(qqq, cmap='gray')
        plt.imshow( np.log( np.abs(img1ft) ) )
        plt.imshow(ndi.filters.gaussian_filter(needle_reference_sqr, sigma=0))
        plt.imshow( np.log( np.abs(img1ft) + 1  ) )
        plt.imshow(bandpass)'''

        LOWPASS = 256
        HIGHPASS = 22
        print(': High pass filter = ', 128, '; low pass filter = ', 5)
        needle_with_sample_norm = ( needle_with_sample - np.mean(needle_with_sample) ) / np.std(needle_with_sample)
        xcorr = crosscorrelation(needle_with_sample_norm, reference_circ_norm, bp='yes', lp=LOWPASS, hp=HIGHPASS, sigma=10)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        new_tip_x = old_tip_x - err[0]
        new_tip_y = old_tip_y - err[1]

        x_shift = ( cen[1] - new_tip_y )
        y_shift = ( cen[0] - new_tip_x )
        print("X-shift to the image centre =  {} meters".format(x_shift))
        print("Y-shift to the image centre =  {} meters".format(y_shift))

        plt.figure(33)
        plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
        plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
        plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='original position')
        plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='current position')
        plt.legend()
        #plt.title('reference needle image WITH needle over the sample')

        HP = range(1,22)
        for HIGHPASS in HP:
            print(': High pass filter = ', HIGHPASS)
            xcorr = crosscorrelation(needle_with_sample_norm, reference_circ_norm, bp='yes', lp=LOWPASS, hp=HIGHPASS, sigma=10)
            maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
            print('\n', maxX, maxY)
            cen = np.asarray(xcorr.shape) / 2
            print('centre = ', cen)
            err = np.array(cen - [maxX, maxY], int)
            print("Shift between 1 and 2 is = " + str(err))
            print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
            new_tip_x = old_tip_x - err[0]
            new_tip_y = old_tip_y - err[1]

            x_shift = ( cen[1] - new_tip_y )
            y_shift = ( cen[0] - new_tip_x )
            print("X-shift to the image centre =  {} meters".format(x_shift))
            print("Y-shift to the image centre =  {} meters".format(y_shift))

            plt.figure(33)
            plt.plot([new_tip_y], [new_tip_x], 'o', markersize=8, label='hpass%d'%HIGHPASS)
            plt.legend()
            #plt.title('reference needle image WITH needle over the sample')







    if TEST == 4:
        if 1: # electron, low res, dwell time for refrence images 20us
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.03_needle_image'
            fileName_sample    = r'sample_eb_lowres.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_eb_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: # electron, high res
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.03_needle_image'
            fileName_sample    = r'sample_eb_highres.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_eb_highres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: # ion, low res
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.03_needle_image'
            fileName_sample    = r'sample_ib_lowres.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_ib_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: # ion, high res
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.03_needle_image'
            fileName_sample    = r'sample_ib_highres.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'ref_ib_highres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)

        # Different set of data, dwell time for reference images 2 us
        if 0: # eb, DOES NOT WORK, too noisy
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
            fileName_sample    = r'electron_beam.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'reference_eb_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)
        if 0: #
            DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.01_needle_image'
            fileName_sample    = r'ion_beam.tif'
            needle_with_sample = read_image(DIR, fileName_sample, gaus_smooth=0)
            fileName_ref       = r'reference_ib_lowres.tif'
            needle_reference   = read_image(DIR, fileName_ref, gaus_smooth=0)


        ### Find the tip using corner-finding algorithm
        harrisim = compute_harris_response( ndi.median_filter(needle_reference, size=2) )
        filtered_coords_ref = get_harris_points(harrisim,4)
        right_outermost_point_ref = max(filtered_coords_ref, key=itemgetter(1))
        #############
        plt.figure(1)
        plt.imshow(needle_reference, cmap='gray')
        plt.plot( [p[1] for p in filtered_coords_ref],[p[0] for p in filtered_coords_ref],'bo'  )
        plt.plot( [right_outermost_point_ref[1]], [right_outermost_point_ref[0]],   'ro')


        ### Find the tip from binarized image (mask) and corner-finding algorithm
        filt   = median(needle_reference, disk(2))
        thresh = threshold_otsu(filt)
        binary = filt > thresh
        mask   = gaussian(binary_dilation(binary, iterations=15), 5)
        mask_binary = (mask >= 0.51).astype(int)
        needle_ref_masked = needle_reference * mask_binary
        ysize, xsize = mask.shape
        harrisim = compute_harris_response(mask_binary)
        filtered_coords_mask_ref = get_harris_points(harrisim,4)
        right_outermost_point_mask_ref = max(filtered_coords_mask_ref, key=itemgetter(1))
        ####
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
        cmask = circ_mask(size=(Dmin, Dmin), radius=Dmin // 2 - 15, sigma=10)  # circular mask
        CMASK = np.zeros(needle_reference.shape)
        CMASK[old_tip_x - Dmin // 2: old_tip_x + Dmin // 2, old_tip_y - Dmin // 2: old_tip_y + Dmin // 2] = cmask
        reference = needle_reference * CMASK * mask


        xcorr = crosscorrelation(needle_with_sample, needle_reference, bp='yes', lp=int(max(needle_with_sample.shape) / 24), hp=5, sigma=2)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        new_tip_x = old_tip_x - err[0]
        new_tip_y = old_tip_y - err[1]
        plt.figure()
        plt.imshow(needle_reference,   cmap='Oranges_r',     alpha=1)
        plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
        plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='old tip')
        plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='new tip')
        plt.legend()
        plt.title('reference needle image WITH needle over the sample')



        xcorr = crosscorrelation(needle_with_sample, needle_reference * mask_binary, bp='yes', lp=int(max(needle_with_sample.shape) / 24), hp=5, sigma=2)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        new_tip_x = old_tip_x - err[0]
        new_tip_y = old_tip_y - err[1]
        plt.figure()
        plt.imshow(needle_reference * mask_binary,   cmap='Oranges_r',     alpha=1)
        plt.imshow(needle_with_sample,               cmap='Blues_r',   alpha=0.5)
        plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='old tip')
        plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='new tip')
        plt.legend()
        plt.title('reference needle image masked  bg=0 WITH needle over the sample')



        xcorr = crosscorrelation(needle_with_sample, reference, bp='yes', lp=int(max(needle_with_sample.shape) / 24), hp=5, sigma=2)
        maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        print('\n', maxX, maxY)
        cen = np.asarray(xcorr.shape) / 2
        print('centre = ', cen)
        err = np.array(cen - [maxX, maxY], int)
        print("Shift between 1 and 2 is = " + str(err))
        print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
        new_tip_x = old_tip_x - err[0]
        new_tip_y = old_tip_y - err[1]
        plt.figure()
        plt.imshow(reference,          cmap='Oranges_r',     alpha=1)
        plt.imshow(needle_with_sample, cmap='Blues_r',   alpha=0.5)
        plt.plot([old_tip_y], [old_tip_x], 'bd', markersize=5, label='old tip')
        plt.plot([new_tip_y], [new_tip_x], 'rd', markersize=5, label='new tip')
        plt.legend()
        plt.title('reference needle image circ_mask bg0 WITH needle over the sample')



        plt.show()


################################################################################################################################################
################################################################################################################################################
























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
