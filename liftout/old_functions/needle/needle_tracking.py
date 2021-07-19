"""Functions for needle identification and liftout."""
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from skimage.filters import gaussian, threshold_otsu, median
from skimage.measure import label
from skimage.morphology import disk

from liftout.old_functions.acquire import new_electron_image, new_ion_image
from liftout.old_functions.needle import insert_needle, retract_needle
from liftout.old_functions.stage_movement import move_sample_stage_out


__all__ = [
    "needle_with_blank_background",
    "segment_needle",
    "locate_needletip",
]


def needle_with_blank_background(microscope, *,
                                 acquire_ion_image=True,
                                 acquire_electron_image=True):
    """Move the sample stage out of the way and take a picture of the needle.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    acquire_ion_image : bool, optional
        Whether to take an ion beam image of the needle, by default True
    acquire_electron_image : bool, optional
        Whether to take an electron beam image of the needle, by default True

    Returns
    -------
    tuple
        (AdornedImage, AdornerdImage)
        Returns a tuple containing the electron and ion beam images.
        Returns None instead of an AdornedImage if that image modality skipped
    """
    original_stage_position = microscope.specimen.stage.current_position
    move_sample_stage_out(microscope)
    park_position = insert_needle(microscope)
    move_needle_closer(microscope.specimen.manipulator)
    if acquire_ion_image is True:
        electron_image = new_electron_image(microscope)
    else:
        electron_image = None
    if acquire_electron_image is True:
        ion_image = new_ion_image(microscope)
    else:
        ion_image = None
    retract_needle(microscope, park_position)
    microscope.specimen.stage.absolute_move(original_stage_position)
    return electron_image, ion_image


def segment_needle(image):
    """Alex's classical needle segmentation method."""
    if hasattr(image, 'data'):
        image = np.copy(image.data)
    filt = median(image, disk(5))
    thresh = threshold_otsu(filt)
    binary = filt > thresh
    mask = gaussian(binary_dilation(binary, iterations=15), 5)
    return mask


def locate_needletip(image):
    """Locate needletip position in x-y pixel coordinates.

    Parameters
    ----------
    image : AdornedImage or numpy.ndarray
        Image of needle with no background

    Returns
    -------
    tuple
        Needletip coordinates in pixels (x, y) format.
    """
    if hasattr(image, 'data'):
        image_data = np.copy(image.data)
    else:
        image_data = np.copy(image)
    filt = median(image_data, disk(5))
    thresh = threshold_otsu(filt)
    binary = filt > thresh
    # expected needle location
    ysize, xsize = image_data.shape
    x_start = 100
    y_start = ysize//3
    expected_needle_cropped = binary[y_start:ysize, x_start:xsize//2]
    labels = label(expected_needle_cropped)
    needle_mask = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    needletip_x = np.argwhere(np.max(needle_mask, axis=0))[-1][0] + x_start
    needletip_y = np.argwhere(np.max(needle_mask, axis=1))[0][0] + y_start
    return (needletip_x, needletip_y)
