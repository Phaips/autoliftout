import sys, getopt, glob, os
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import math
import time
import scipy.ndimage as ndi
from scipy import fftpack, misc
from PIL import Image, ImageDraw, ImageFilter
from matplotlib.patches import Circle
from skimage.transform import resize



import os, sys, glob
import datetime

PRETILT_DEGREES = 27

class BeamType(Enum):
    ION = 'ION'
    ELECTRON = 'ELECTRON'


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
    xcorr = crosscorrelation(img1, img2, bp='yes', lp=lowpass, hp=highpass, sigma=sigma)
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    print('\n', maxX, maxY)
    cen = np.asarray(xcorr.shape) / 2
    print('centre = ', cen)
    err = np.array(cen - [maxX, maxY], int)
    print("Shift between 1 and 2 is = " + str(err))
    print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
    return err[1], err[0]


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
    TEST = 2

    if TEST == 2:
        print('Image stretching test.. MASKED CORRELATION.')
        DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.15_trench_alignment_A'
        fileName_ib_tilt27 = r'01_ib_highres_27.tif'
        fileName_ib_tilt32 = r'02_ib_highres_32.tif'
        fileName_ib_tilt32_aligned = r'03_ib_highres_32_aligned.tif'
        fileName_ib_tilt37 = r'04_ib_highres_37.tif'
        fileName_ib_tilt37_aligned = r'05_ib_highres_37_aligned.tif'

        ib27 = read_image(DIR, fileName_ib_tilt27, gaus_smooth=0)
        ib32 = read_image(DIR, fileName_ib_tilt32, gaus_smooth=0)
        ib37 = read_image(DIR, fileName_ib_tilt37_aligned, gaus_smooth=0)
        ib27_norm = (ib27 - np.mean(ib27)) / np.std(ib27)
        ib32_norm = (ib32 - np.mean(ib32)) / np.std(ib32)
        ib37_norm = (ib37 - np.mean(ib37)) / np.std(ib37)

        y_scale = 1. / math.cos(np.deg2rad(5))
        height, width = ib27_norm.shape
        height_scaled = int(round(height * y_scale))
        if height_scaled % 2 != 0:
            height_scaled += 1
        dy = (height_scaled - height) // 2
        ib27_norm_scaled = resize(ib27_norm, (height_scaled, width))
        ib27_norm_scaled = ib27_norm_scaled[dy:-dy, :]

        cmask = circ_mask(size=(width,height), radius=height // 3 - 15, sigma=10)  # circular mask


        dx_ib, dy_ib = shift_from_crosscorrelation_simple_images(ib27_norm_scaled * cmask, ib37_norm * cmask, lowpass=256,
                                                                 highpass=22, sigma=10)
        dx_ib, dy_ib = shift_from_crosscorrelation_simple_images(ib27_norm * cmask, ib37_norm * cmask, lowpass=256,
                                                                 highpass=22, sigma=10)

        x0 = int(ib27_norm_scaled.shape[1] / 2)
        y0 = int(ib27_norm_scaled.shape[0] / 2)
        w = 400
        original_image = ib27_norm[y0 - w: y0 + w, x0 - w: x0 + w]
        aligned_image = ib37_norm[y0 - w + dy_ib: y0 + w + dy_ib, x0 - w + dx_ib: x0 + w + dx_ib]

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(ib27_norm * cmask, cmap='Blues_r', alpha=1)
        plt.subplot(1, 3, 2)
        plt.imshow(ib27_norm_scaled * cmask, cmap='Blues_r', alpha=1)
        plt.subplot(1, 3, 3)
        plt.imshow(ib37_norm * cmask, cmap='Oranges_r', alpha=0.5)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(ib27_norm_scaled, cmap='Blues_r', alpha=1)
        ax.imshow(ib37_norm, cmap='Oranges_r', alpha=0.5)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(original_image, cmap='Blues_r', alpha=1)
        ax.imshow(aligned_image, cmap='Oranges_r', alpha=0.5)


    if TEST == 1:
        print('Image stretching test...')
        DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.15_trench_alignment_A'
        fileName_ib_tilt27 = r'01_ib_highres_27.tif'
        fileName_ib_tilt32 = r'02_ib_highres_32.tif'
        fileName_ib_tilt32_aligned = r'03_ib_highres_32_aligned.tif'
        fileName_ib_tilt37 = r'04_ib_highres_37.tif'
        fileName_ib_tilt37_aligned = r'05_ib_highres_37_aligned.tif'

        ib27 = read_image(DIR, fileName_ib_tilt27, gaus_smooth=0)
        ib32 = read_image(DIR, fileName_ib_tilt32, gaus_smooth=0)
        ib37 = read_image(DIR, fileName_ib_tilt37_aligned, gaus_smooth=0)
        ib27_norm = (ib27 - np.mean(ib27)) / np.std(ib27)
        ib32_norm = (ib32 - np.mean(ib32)) / np.std(ib32)
        ib37_norm = (ib37 - np.mean(ib37)) / np.std(ib37)

        y_scale = 1. / math.cos(np.deg2rad(5))
        height, width = ib27_norm.shape
        height_scaled = int(round(height * y_scale))
        if height_scaled % 2 != 0 :
            height_scaled += 1
        dy = (height_scaled - height)//2
        ib27_norm_scaled = resize(ib27_norm, (height_scaled, width) )
        ib27_norm_scaled = ib27_norm_scaled[dy:-dy,:]

        ib27_norm_scaled_cut = ib27_norm_scaled[ height//4:-height//4, width//4 : -width//4 ]
        ib37_norm_cut        = ib37_norm[ height//4:-height//4, width//4 : -width//4 ]

        dx_ib, dy_ib = shift_from_crosscorrelation_simple_images(ib27_norm_scaled_cut, ib37_norm_cut, lowpass=256, highpass=22, sigma=10)

        x0 = int(ib27_norm_scaled.shape[1] / 2)
        y0 = int(ib27_norm_scaled.shape[0] / 2)
        width = 400
        original_image = ib27_norm[y0 - width        : y0 + width        , x0 - width         : x0 + width]
        aligned_image =  ib37_norm[y0 - width + dy_ib: y0 + width + dy_ib, x0 - width + dx_ib  : x0 + width + dx_ib ]

        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(ib27_norm, cmap='Blues_r', alpha=1)
        plt.subplot(1,3,2)
        plt.imshow(ib27_norm_scaled, cmap='Blues_r', alpha=1)
        plt.subplot(1,3,3)
        plt.imshow(ib37_norm, cmap='Oranges_r', alpha=0.5)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(ib27_norm_scaled, cmap='Blues_r', alpha=1)
        ax.imshow(ib37_norm, cmap='Oranges_r', alpha=0.5)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(original_image, cmap='Blues_r', alpha=1)
        ax.imshow(aligned_image, cmap='Oranges_r', alpha=0.5)




    plt.show()



