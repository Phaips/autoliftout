from liftout.fibsem.movement import *
from autoscript_sdb_microscope_client.structures import GrabFrameSettings
import scipy.ndimage as ndi
from scipy import fftpack, misc
from PIL import Image, ImageDraw
from liftout.fibsem import acquire
from liftout.detection import detection
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


    # TODO: 'coarse alignment' status
    image_settings['hfw'] = field_width_lowres
    image_settings['save'] = True
    # image_settings['label'] = f'{liftout_counter:02d}_drift_correction_lamella_low_res'  # TODO: add to protocol

    if mode == "land":
        image_settings['label'] = f'{liftout_counter:02d}_{mode}_drift_correction_landing_low_res'  # TODO: add to protocol
        new_eb_lowres, new_ib_lowres = take_reference_images(microscope, settings=image_settings)
        align_using_reference_images(ref_lowres, new_ib_lowres, stage, mode=mode)

        # TODO: 'fine alignment' status
        image_settings['hfw'] = field_width_highres
        image_settings['save'] = True

        image_settings['label'] = f'{liftout_counter:02d}_drift_correction_landing_high_res'  # TODO: add to protocol
        new_eb_highres, new_ib_highres = take_reference_images(microscope, settings=image_settings)
        align_using_reference_images(ref_highres, new_ib_highres, stage, mode=mode)
        # TODO: deduplicate this bit ^
    else:
        for i in range(lowres_count):
            # if i == 1:
            image_settings['label'] = f'{liftout_counter:02d}_{mode}_drift_correction_lamella_low_res_{i}'  # TODO: add to protocol
            new_eb_lowres, new_ib_lowres = take_reference_images(microscope, settings=image_settings)
            align_using_reference_images(ref_lowres, new_eb_lowres, stage)

        # TODO: 'fine alignment' status
        image_settings['hfw'] = field_width_highres
        image_settings['save'] = True

        image_settings['label'] = f'{liftout_counter:02d}_drift_correction_lamella_high_res'  # TODO: add to protocol
        new_eb_highres, new_ib_highres = take_reference_images(microscope, settings=image_settings)
        align_using_reference_images(ref_highres, new_eb_highres, stage)


def align_using_reference_images(ref_image, new_image, stage, mode=None):

    #TODO: add to protocol
    if mode == "land":
        lp_ratio = 6
        hp_ratio = 64
        sigma_factor = 10
        sigma_ratio = 1536
        beam_type = BeamType.ION
    else:
        lp_ratio = 12
        hp_ratio = 256
        sigma_factor = 2
        sigma_ratio = 1536
        beam_type = BeamType.ELECTRON


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
    x_move = x_corrected_stage_movement(-dx_ei_meters)
    yz_move = y_corrected_stage_movement(dy_ei_meters,
                                         stage.current_position.t,
                                         beam_type=beam_type)  # check electron/ion movement
    stage.relative_move(x_move)
    stage.relative_move(yz_move)


# TODO: figure out a better name
def identify_shift_using_machine_learning(microscope, image_settings, settings, liftout_counter, shift_type):
    if image_settings['beam_type'] == BeamType.ION:
        image_settings['brightness'] = settings["machine_learning"]["ib_brightness"]
        image_settings['contrast'] = settings["machine_learning"]["ib_contrast"]
    elif image_settings['beam_type'] == BeamType.ELECTRON:
        image_settings['brightness'] = settings["machine_learning"]["eb_brightness"]
        image_settings['contrast'] = settings["machine_learning"]["eb_contrast"]

    eb_image,  ib_image = take_reference_images(microscope, image_settings)
    weights_file = settings["machine_learning"]["weights"]
    detector = detection.Detector(weights_file)

    if image_settings['beam_type'] == BeamType.ION:
        image = ib_image
    else:
        image = eb_image
    image_w_overlay, downscaled_image, feature_1_px, feature_1_type, feature_2_px, feature_2_type = detector.locate_shift_between_features(image, shift_type=shift_type, show=False)
    return image, np.array(image_w_overlay), np.array(downscaled_image), feature_1_px, feature_1_type, feature_2_px, feature_2_type


def calculate_shift_between_features(settings):
    """
    in m
    :return:
    """
    # detector class (model)
    weights_file = settings["machine_learning"]["weights"]
    detector = detection.Detector(weights_file)

    print("shift_type: ", shift_type)
    x_distance, y_distance = detector.calculate_shift_between_features(img, shift_type=shift_type, show=show, validate=validate)
    print(f"x_distance = {x_distance:.4f}, y_distance = {y_distance:.4f}")

    x_shift, y_shift = detection.calculate_shift_distance_in_metres(img, x_distance, y_distance)
    print(f"x_shift =  {x_shift/1e-6:.4f}, um; y_shift = {y_shift/1e-6:.4f} um; ")

    return x_shift, y_shift




def shift_from_correlation_electronBeam_and_ionBeam(eb_image, ib_image, lowpass=128, highpass=6, sigma=2):
    ib_image_rotated = rotate_AdornedImage(ib_image)
    x_shift, y_shift = shift_from_crosscorrelation_AdornedImages(
        eb_image, ib_image_rotated, lowpass=lowpass, highpass=highpass, sigma=sigma)
    return x_shift, y_shift


def rotate_AdornedImage(image):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
    data = np.rot90(np.rot90(np.copy(image.data)))
    # data = ndi.rotate(image.data, 180, reshape=False)
    reference = AdornedImage(data=data)
    reference.metadata = image.metadata
    return reference


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
