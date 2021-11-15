import os
from liftout.fibsem.movement import *
from autoscript_sdb_microscope_client.structures import GrabFrameSettings
import scipy.ndimage as ndi
from scipy import fftpack, misc
from PIL import Image, ImageDraw
from liftout.fibsem import acquire
from liftout.detection import detection
from liftout.model import models
BeamType = acquire.BeamType

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


def correct_stage_drift(microscope, image_settings, reference_images, liftout_counter, mode='eb'):

    ref_eb_lowres, ref_eb_highres, ref_ib_lowres, ref_ib_highres = reference_images

    lowres_count = 1
    if mode == 'eb':
        ref_lowres = ref_eb_lowres
        ref_highres = ref_eb_highres
    elif mode == 'ib':
        lowres_count = 2
        ref_lowres = rotate_AdornedImage(ref_ib_lowres)
        ref_highres = rotate_AdornedImage(ref_ib_highres)
    elif mode == "land":
        lowres_count = 1
        ref_lowres = ref_ib_lowres
        ref_highres = ref_ib_highres

    stage = microscope.specimen.stage
    # TODO: user input resolution, must match (protocol)
    image_settings['resolution'] = '1536x1024'
    image_settings['dwell_time'] = 1e-6

    pixelsize_x_lowres = ref_lowres.metadata.binary_result.pixel_size.x
    field_width_lowres = pixelsize_x_lowres * ref_lowres.width
    pixelsize_x_highres = ref_highres.metadata.binary_result.pixel_size.x
    field_width_highres = pixelsize_x_highres * ref_highres.width
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres

    # TODO: refactor this code
    # TODO: 'coarse alignment' status
    image_settings['hfw'] = field_width_lowres
    image_settings['save'] = True
    # image_settings['label'] = f'{liftout_counter:02d}_drift_correction_lamella_low_res'  # TODO: add to protocol

    if mode == "land":
        image_settings['label'] = f'{liftout_counter:02d}_{mode}_drift_correction_landing_low_res'  # TODO: add to protocol
        new_eb_lowres, new_ib_lowres = take_reference_images(microscope, settings=image_settings)
        ret = align_using_reference_images(ref_lowres, new_ib_lowres, stage, mode=mode)
        if ret is False:
            return ret
        # TODO: 'fine alignment' status
        image_settings['hfw'] = field_width_highres
        image_settings['save'] = True

        image_settings['label'] = f'{liftout_counter:02d}_drift_correction_landing_high_res'  # TODO: add to protocol
        new_eb_highres, new_ib_highres = take_reference_images(microscope, settings=image_settings)
        ret = align_using_reference_images(ref_highres, new_ib_highres, stage, mode=mode)
        # TODO: deduplicate this bit ^
    else:
        for i in range(lowres_count):

            image_settings['label'] = f'{liftout_counter:02d}_{mode}_drift_correction_lamella_low_res_{i}'  # TODO: add to protocol
            new_eb_lowres, new_ib_lowres = take_reference_images(microscope, settings=image_settings)
            ret = align_using_reference_images(ref_lowres, new_eb_lowres, stage)
            if ret is False:
                return ret

        # TODO: 'fine alignment' status
        image_settings['hfw'] = field_width_highres
        image_settings['save'] = True

        image_settings['label'] = f'{liftout_counter:02d}_drift_correction_lamella_high_res'  # TODO: add to protocol
        new_eb_highres, new_ib_highres = take_reference_images(microscope, settings=image_settings)
        ret = align_using_reference_images(ref_highres, new_eb_highres, stage)

    logging.info(f"calibration: image cross correlation finished {ret}")
    return ret

def align_using_reference_images(ref_image, new_image, stage, mode=None):

    #TODO: Read in from protocol
    #TODO: there are three different types of cross-corellation, E-E, E-I, I-I
    if mode == "land":
        beam_type = BeamType.ION
    if mode is None:
        beam_type = BeamType.ELECTRON
    lp_ratio = 6
    hp_ratio = 64
    sigma_factor = 10
    sigma_ratio = 1536

    # These are the old cross-correlation values
    # elif mode is not "land":
    #     lp_ratio = 12
    #     hp_ratio = 256
    #     sigma_factor = 2
    #     sigma_ratio = 1536
    #     beam_type = BeamType.ELECTRON
    logging.info(f"calibration: align using {beam_type.name} reference image in mode {mode}.")

    # TODO: possibly hard-code these numbers at fixed resolutions?
    lowpass_pixels = int(max(
        new_image.data.shape) / lp_ratio)  # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int(max(
        new_image.data.shape) / hp_ratio)  # =6 @ 1536x1024, good for e-beam images
    sigma = int(sigma_factor * max(
        new_image.data.shape) / sigma_ratio)  # =2 @ 1536x1024, good for e-beam images
    # TODO: ask Sergey about maths/check if we can use pixel_to_real_value on dx
    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(
        new_image, ref_image, lowpass=lowpass_pixels,
        highpass=highpass_pixels, sigma=sigma)
    
    # check if the cross correlation is trying to move more than a safety threshold.
    pixelsize_x = ref_image.metadata.binary_result.pixel_size.x
    hfw = pixelsize_x * ref_image.width
    vfw  = pixelsize_x * ref_image.height
    X_THRESHOLD = 0.25 * hfw
    Y_THRESHOLD = 0.25 * vfw
    if abs(dx_ei_meters) > X_THRESHOLD or abs(dy_ei_meters) > Y_THRESHOLD:
        logging.warning("calibration: calculated cross correlation movement too large.")
        logging.warning("calibration: cancelling automatic cross correlation.")
        return False
    else:
        
        x_move = x_corrected_stage_movement(-dx_ei_meters)
        yz_move = y_corrected_stage_movement(dy_ei_meters,
                                            stage.current_position.t,
                                            beam_type=beam_type)  # check electron/ion movement
        stage.relative_move(x_move)
        stage.relative_move(yz_move)
        return True



# TODO: figure out a better name
def identify_shift_using_machine_learning(microscope, image_settings, settings, shift_type):

    eb_image,  ib_image = take_reference_images(microscope, image_settings)
    weights_file = settings["machine_learning"]["weights"]
    weights_path = os.path.join(os.path.dirname(models.__file__), weights_file)
    detector = detection.Detector(weights_path)

    if image_settings['beam_type'] == BeamType.ION:
        image = ib_image
    else:
        image = eb_image
    image_w_overlay, downscaled_image, feature_1_px, feature_1_type, feature_2_px, feature_2_type = detector.locate_shift_between_features(image, shift_type=shift_type, show=False)
    return image, np.array(image_w_overlay), np.array(downscaled_image), feature_1_px, feature_1_type, feature_2_px, feature_2_type


def shift_from_correlation_electronBeam_and_ionBeam(eb_image, ib_image, lowpass=128, highpass=6, sigma=2):
    ib_image_rotated = rotate_AdornedImage(ib_image)
    x_shift, y_shift = shift_from_crosscorrelation_AdornedImages(
        eb_image, ib_image_rotated, lowpass=lowpass, highpass=highpass, sigma=sigma)
    return x_shift, y_shift


def rotate_AdornedImage(image):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    data = np.rot90(np.rot90(np.copy(image.data)))
    reference = AdornedImage(data=data)
    reference.metadata = image.metadata
    return reference


def shift_from_crosscorrelation_AdornedImages(img1, img2, lowpass=128, highpass=6, sigma=6, beamshift=False):
    pixelsize_x_1 = img1.metadata.binary_result.pixel_size.x
    pixelsize_y_1 = img1.metadata.binary_result.pixel_size.y
    pixelsize_x_2 = img2.metadata.binary_result.pixel_size.x
    pixelsize_y_2 = img2.metadata.binary_result.pixel_size.y
    # normalise both images
    img1_data_norm = (img1.data - np.mean(img1.data)) / np.std(img1.data)
    img2_data_norm = (img2.data - np.mean(img2.data)) / np.std(img2.data)
    # cross-correlate normalised images
    if beamshift:
        rect_mask = _mask_rectangular(img2_data_norm.shape)
        img1_data_norm = rect_mask * img1_data_norm
        img2_data_norm = rect_mask * img2_data_norm
    xcorr = crosscorrelation(img1_data_norm, img2_data_norm, bp='yes', lp=lowpass, hp=highpass, sigma=sigma)
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    logging.info(f"maxX: {maxX}, {maxY}")
    cen = np.asarray(xcorr.shape) / 2
    logging.info(f'centre = {cen}')
    err = np.array(cen - [maxX, maxY], int)
    logging.info(f"Shift between 1 and 2 is = {err}")
    logging.info(f"img2 is X-shifted by  {err[1]}; Y-shifted by {err[0]}")
    x_shift = err[1] * pixelsize_x_2
    y_shift = err[0] * pixelsize_y_2
    logging.info(f"X-shift =  {x_shift} meters")
    logging.info(f"Y-shift =  {y_shift} meters")
    return x_shift, y_shift # metres


def crosscorrelation(img1, img2, bp='no', *args, **kwargs):
    if img1.shape != img2.shape:
        logging.error('### ERROR in xcorr2: img1 and img2 do not have the same size ###')
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
            logging.error('ERROR in xcorr2: check bandpass parameters')
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
        logging.error('ERROR in xcorr2: bandpass value ( bp= ' + str(bp) + ' ) not recognized')
        return -1
    return xcorr


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



# FROM AUTOLAMELLA


def realign(microscope, new_image, reference_image):
    """Realign to reference image using beam shift.

    Parameters
    ----------
    microscope : Autoscript microscope object
    new_image : The most recent image acquired.
        Must have the same dimensions and relative position as the reference.
    reference_image : The reference image to align with.
        Muast have the same dimensions and relative position as the new image
    Returns
    -------
    microscope.beams.ion_beam.beam_shift.value
        The current beam shift position (after any realignment)
    """
    from autoscript_core.common import ApplicationServerException

    shift_in_meters = _calculate_beam_shift(new_image, reference_image)
    try:
        microscope.beams.ion_beam.beam_shift.value += shift_in_meters
    except ApplicationServerException:
        logging.warning(
            "Cannot move beam shift beyond limits, "
            "will continue with no beam shift applied."
        )
    return microscope.beams.ion_beam.beam_shift.value


def _calculate_beam_shift(image_1, image_2):
    """Cross correlation to find shift between two images.

    Parameters
    ----------
    image_1 : AdornedImage
        Original image to use as reference point.
    image_2 : AdornedImage
        Possibly shifted image to align with original.

    Returns
    -------
    realspace_beam_shift
        Beam shift in x, y format (meters), list of floats.

    Raises
    ------
    ValueError
        If images are not the same dimensions, raise a ValueError.
    """
    if image_1.data.shape != image_2.data.shape:
        raise ValueError("Images must be the same shape for cross correlation.")
    mask_image_1 = _mask_circular(image_1.data.shape)
    mask_image_2 = _mask_rectangular(image_2.data.shape)
    norm_image_1 = _normalize_image(image_1.data) * mask_image_1
    norm_image_2 = _normalize_image(image_2.data) * mask_image_2
    pixel_shift = _simple_register_translation(norm_image_2, norm_image_1)
    # Autoscript y-axis has an inverted positive direction
    pixel_shift[1] = -pixel_shift[1]
    pixelsize_x = image_1.metadata.binary_result.pixel_size.x
    realspace_beam_shift = pixel_shift * pixelsize_x
    logging.info("pixel_shift calculated = {}".format(pixel_shift))
    logging.info("realspace_beam_shift calculated = {}".format(realspace_beam_shift))
    return realspace_beam_shift


def _simple_register_translation(src_image, target_image, max_shift_mask=None):
    """Calculate pixel shift between two input images.

    This function runs with numpy or cupy for GPU acceleration.

    Parameters
    ----------
    src_image : array
        Reference image.
    target_image : array
        Image to register.  Must be same dimensionality as ``src_image``.
    max_shift_mask : array
        The fourier mask restricting the maximum allowable pixel shift.

    Returns
    -------
    shifts : ndarray
        Pixel shift in x, y order between target and source image.

    References
    ----------
    scikit-image register_translation function in the skimage.feature module.
    """
    src_freq = np.fft.fftn(src_image)
    target_freq = np.fft.fftn(target_image)
    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product)
    # Locate maximum
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )
    midpoints = np.array([float(np.fix(axis_size / 2)) for axis_size in shape])
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    shifts = np.flip(shifts, axis=0).astype(np.int)  # x, y order
    return shifts



def _normalize_image(image, mask=None):
    """Ensure the image mean is zero and the standard deviation is one.

    Parameters
    ----------
    image : ndarray
        The input image array.
    mask : ndarray, optional
        A mask image containing values between zero and one.
        Dimensions must match the input image.

    Returns
    -------
    ndarray
        The normalized image.
        The mean intensity is equal to zero and standard deviation equals one.
    """
    image = image - np.mean(image)
    image = image / np.std(image)
    if mask:
        image = image * mask
    return image


def _mask_circular(image_shape, sigma=5.0, *, radius=None):
    """Make a circular mask with soft edges for image normalization.

    Parameters
    ----------
    image_shape : tuple
        Shape of the original image array
    sigma : float, optional
        Sigma value (in pixels) for gaussian blur function, by default 5.
    radius : int, optional
        Radius of circle, by default None which will create a circle that fills
        90% of the smallest image dimension.

    Returns
    -------
    ndarray
        Circular mask with soft edges in array matching the input image_shape
    """
    if radius is None:
        # leave at least a 5% gap on each edge
        radius = 0.45 * min(image_shape)
    r, c = np.array(np.array(image_shape) / 2).astype(int)  # center point
    rr, cc = skimage.draw.circle(r, c, radius=radius, shape=image_shape)
    mask = np.zeros(image_shape)
    mask[rr, cc] = 1.0
    mask = ndi.gaussian_filter(mask, sigma=sigma)
    return mask


def _mask_rectangular(image_shape, sigma=5.0, *, start=None, extent=None):
    """Make a rectangular mask with soft edges for image normalization.

    Parameters
    ----------
    image_shape : tuple
        Shape of the original image array
    sigma : float, optional
        Sigma value (in pixels) for gaussian blur function, by default 5.
    start : tuple, optional
        Origin point of the rectangle, e.g., ([plane,] row, column).
        Default start is 5% of the total image width and height.
    extent : int, optional
        The extent (size) of the drawn rectangle.
        E.g., ([num_planes,] num_rows, num_cols).
        Default is for the rectangle to cover 95% of the image width & height.

    Returns
    -------
    ndarray
        Rectangular mask with soft edges in array matching input image_shape.
    """
    import skimage
    if extent is None:
        # leave at least a 5% gap on each edge
        start = np.round(np.array(image_shape) * 0.05)
        extent = np.round(np.array(image_shape) * 0.90)
    rr, cc = skimage.draw.rectangle(start, extent=extent, shape=image_shape)
    mask = np.zeros(image_shape)
    mask[rr.astype(int), cc.astype(int)] = 1.0
    mask = ndi.gaussian_filter(mask, sigma=sigma)
    return mask


def _bandpass_mask(image_shape, outer_radius, inner_radius=0, sigma=5):
    """Create a fourier bandpass mask.

    Parameters
    ----------
    image_shape : tuple
        Shape of the original image array
    outer_radius : int
        Outer radius for bandpass filter array.
    inner_radius : int, optional
        Inner radius for bandpass filter array, by default 0
    sigma : int, optional
        Sigma value for edge blending, by default 5 pixels.

    Returns
    -------
    _bandpass_mask : ndarray
        The bandpass image mask.
    """
    _bandpass_mask = np.zeros(image_shape)
    r, c = np.array(image_shape) / 2
    inner_circle_rr, inner_circle_cc = skimage.draw.circle(
        r, c, inner_radius, shape=image_shape
    )
    outer_circle_rr, outer_circle_cc = skimage.draw.circle(
        r, c, outer_radius, shape=image_shape
    )
    _bandpass_mask[outer_circle_rr, outer_circle_cc] = 1.0
    _bandpass_mask[inner_circle_rr, inner_circle_cc] = 0.0
    _bandpass_mask = ndi.gaussian_filter(_bandpass_mask, sigma)
    _bandpass_mask = np.array(_bandpass_mask)
    # fourier space origin should be in the corner
    _bandpass_mask = np.roll(
        _bandpass_mask, (np.array(image_shape) / 2).astype(int), axis=(0, 1)
    )
    return _bandpass_mask


def test_thin_lamella(microscope, settings, image_settings, ref_image=None):

    # rotate and tilt thinning angle

    # user clicks on lamella position

    image_settings["save"] = False
    image_settings["hfw"] = 30e-6
    image_settings["beam_type"] = BeamType.ION
    image_settings["gamma"]["correction"] = False
    image_settings["save"] = True
    image_settings["label"] = "crosscorrelation_1"

    # initial reference image
    # _, ref_image = acquire.take_reference_images(microscope, image_settings)
    ref_image = acquire.new_image(microscope, image_settings)

    # adjust beamshift by known amount
    microscope.beams.ion_beam.beam_shift.value += (1e-6, 2e-6)

    # align using cross correlation
    img1 = ref_image
    image_settings["label"] = "crosscorrelation_2"
    img2 = acquire.new_image(microscope, settings=image_settings)
    dx, dy = shift_from_crosscorrelation_AdornedImages(img1, img2, lowpass=256, highpass=24, sigma=10, beamshift=True)

    # adjust beamshift
    microscope.beams.ion_beam.beam_shift.value += (-dx, dy)

    # retake image
    _ = acquire.new_image(microscope, image_settings)


    # TODO: need to align for each imaging current...

    # mill
    from liftout.fibsem import milling
    milling.mill_thin_lamella(microscope, settings)

    return




