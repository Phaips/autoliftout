import os
from liftout.fibsem.movement import *
import scipy.ndimage as ndi
from scipy import fftpack, misc
from PIL import Image, ImageDraw
from liftout.fibsem import acquire
from liftout.fibsem.sampleposition import MicroscopeState, AutoLiftoutStage
from liftout.model import models
from autoscript_sdb_microscope_client.enumerations import *
from autoscript_sdb_microscope_client import SdbMicroscopeClient
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


def correct_stage_drift(
    microscope, image_settings, reference_images, mode="eb"
):
    # TODO: refactor the whole cross-correlation workflow (de-duplicate it)
    ref_eb_lowres, ref_eb_highres, ref_ib_lowres, ref_ib_highres = reference_images

    lowres_count = 1
    if mode == "eb":
        ref_lowres = ref_eb_lowres
        ref_highres = ref_eb_highres
    elif mode == "ib":
        lowres_count = 2
        ref_lowres = rotate_AdornedImage(ref_ib_lowres)
        ref_highres = rotate_AdornedImage(ref_ib_highres)
    elif mode == "land":
        lowres_count = 1
        ref_lowres = ref_ib_lowres
        ref_highres = ref_ib_highres

    stage = microscope.specimen.stage
    image_settings.resolution = "1536x1024"
    image_settings.dwell_time = 1e-6

    pixelsize_x_lowres = ref_lowres.metadata.binary_result.pixel_size.x
    field_width_lowres = pixelsize_x_lowres * ref_lowres.width
    pixelsize_x_highres = ref_highres.metadata.binary_result.pixel_size.x
    field_width_highres = pixelsize_x_highres * ref_highres.width
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres

    image_settings.hfw = field_width_lowres
    image_settings.save = True

    if mode == "land":
        image_settings.label = f"{mode}_drift_correction_landing_low_res"
        new_eb_lowres, new_ib_lowres = take_reference_images(
            microscope, image_settings=image_settings
        )
        ret = align_using_reference_images(ref_lowres, new_ib_lowres, stage, mode=mode)
        if ret is False:
            return ret

        # fine alignment
        image_settings.hfw = field_width_highres
        image_settings.save = True
        image_settings.label = f"drift_correction_landing_high_res"
        new_eb_highres, new_ib_highres = take_reference_images(
            microscope, image_settings=image_settings
        )
        ret = align_using_reference_images(
            ref_highres, new_ib_highres, stage, mode=mode
        )

    else:
        for i in range(lowres_count):

            image_settings.label = f"{mode}_drift_correction_lamella_low_res_{i}"
            new_eb_lowres, new_ib_lowres = take_reference_images(
                microscope, image_settings=image_settings
            )
            ret = align_using_reference_images(ref_lowres, new_eb_lowres, stage)
            if ret is False:
                return ret

        # fine alignment
        image_settings.hfw = field_width_highres
        image_settings.save = True
        image_settings.label = f"drift_correction_lamella_high_res"
        new_eb_highres, new_ib_highres = take_reference_images(
            microscope, image_settings=image_settings
        )
        ret = align_using_reference_images(ref_highres, new_eb_highres, stage)

    logging.info(f"CROSSCORRELATION | {mode} | {ret}")
    return ret


def align_using_reference_images(ref_image, new_image, stage, mode=None):

    # three different types of cross correlation, E-E, E-I, I-I
    if mode == "land":
        beam_type = BeamType.ION
    if mode is None:
        beam_type = BeamType.ELECTRON
    lp_ratio = 3
    hp_ratio = 64

    logging.info(f"aligning using {beam_type.name} reference image in mode {mode}.")
    lowpass_pixels = int(max(new_image.data.shape) * 0.66)   # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int(max(new_image.data.shape) / hp_ratio)
    sigma = 6

    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(
        new_image,
        ref_image,
        lowpass=lowpass_pixels,
        highpass=highpass_pixels,
        sigma=sigma,
    )

    # check if the cross correlation is trying to move more than a safety threshold.
    pixelsize_x = ref_image.metadata.binary_result.pixel_size.x
    hfw = pixelsize_x * ref_image.width
    vfw = pixelsize_x * ref_image.height
    X_THRESHOLD = 0.25 * hfw
    Y_THRESHOLD = 0.25 * vfw
    if abs(dx_ei_meters) > X_THRESHOLD or abs(dy_ei_meters) > Y_THRESHOLD:
        logging.warning("calibration: calculated cross correlation movement too large.")
        logging.warning("calibration: cancelling automatic cross correlation.")
        return False
    else:

        tmp_settings = {  # TODO: MAGIC NUMBERS REMOVE when doing refactor
            "system": {
                "pretilt_angle": 27,  # degrees
                "stage_tilt_flat_to_ion": 52  # degrees
            }

        }
        x_move = x_corrected_stage_movement(-dx_ei_meters)
        yz_move = y_corrected_stage_movement(
            dy_ei_meters, stage.current_position.t,
            settings=tmp_settings,
            beam_type=beam_type
        )  # check electron/ion movement
        stage.relative_move(x_move)
        stage.relative_move(yz_move)
        return True


def identify_shift_using_machine_learning(
    microscope, image_settings, settings, shift_type):


    eb_image, ib_image = take_reference_images(microscope, image_settings)
    weights_file = settings["machine_learning"]["weights"]
    weights_path = os.path.join(os.path.dirname(models.__file__), weights_file)
    detector = detection.Detector(weights_path)

    if image_settings.beam_type == BeamType.ION:
        image = ib_image
    else:
        image = eb_image

    det_result = detector.locate_shift_between_features(image, shift_type=shift_type)

    return det_result

def shift_from_correlation_electronBeam_and_ionBeam(
    eb_image, ib_image, lowpass=128, highpass=6, sigma=2
):
    ib_image_rotated = rotate_AdornedImage(ib_image)
    x_shift, y_shift = shift_from_crosscorrelation_AdornedImages(
        eb_image, ib_image_rotated, lowpass=lowpass, highpass=highpass, sigma=sigma
    )
    return x_shift, y_shift


def rotate_AdornedImage(image):
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        GrabFrameSettings,
    )

    data = np.rot90(np.rot90(np.copy(image.data)))
    reference = AdornedImage(data=data)
    reference.metadata = image.metadata
    return reference


def shift_from_crosscorrelation_AdornedImages(
    img1: AdornedImage, img2: AdornedImage, lowpass: int = 128, highpass: int = 6, sigma: int = 6, use_rect_mask: bool = False
):
    pixelsize_x_1 = img1.metadata.binary_result.pixel_size.x
    pixelsize_y_1 = img1.metadata.binary_result.pixel_size.y
    pixelsize_x_2 = img2.metadata.binary_result.pixel_size.x
    pixelsize_y_2 = img2.metadata.binary_result.pixel_size.y
    # normalise both images
    img1_data_norm = (img1.data - np.mean(img1.data)) / np.std(img1.data)
    img2_data_norm = (img2.data - np.mean(img2.data)) / np.std(img2.data)
    # cross-correlate normalised images
    if use_rect_mask:
        rect_mask = _mask_rectangular(img2_data_norm.shape)
        img1_data_norm = rect_mask * img1_data_norm
        img2_data_norm = rect_mask * img2_data_norm
    xcorr = crosscorrelation(
        img1_data_norm, img2_data_norm, bp="yes", lp=lowpass, hp=highpass, sigma=sigma
    )
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    logging.info(f"maxX: {maxX}, {maxY}")
    cen = np.asarray(xcorr.shape) / 2
    logging.info(f"centre = {cen}")
    err = np.array(cen - [maxX, maxY], int)
    logging.info(f"Shift between 1 and 2 is = {err}")
    logging.info(f"img2 is X-shifted by  {err[1]}; Y-shifted by {err[0]}")
    x_shift = err[1] * pixelsize_x_2
    y_shift = err[0] * pixelsize_y_2
    logging.info(f"X-shift =  {x_shift:.2e} meters")
    logging.info(f"Y-shift =  {y_shift:.2e} meters")
    return x_shift, y_shift  # metres


def crosscorrelation(img1, img2, bp="no", *args, **kwargs):
    if img1.shape != img2.shape:
        logging.error(
            "### ERROR in xcorr2: img1 and img2 do not have the same size ###"
        )
        return -1
    if img1.dtype != "float64":
        img1 = np.array(img1, float)
    if img2.dtype != "float64":
        img2 = np.array(img2, float)

    if bp == "yes":
        lpv = kwargs.get("lp", None)
        hpv = kwargs.get("hp", None)
        sigmav = kwargs.get("sigma", None)
        if lpv == "None" or hpv == "None" or sigmav == "None":
            logging.error("ERROR in xcorr2: check bandpass parameters")
            return -1
        bandpass = bandpass_mask(
            size=(img1.shape[1], img1.shape[0]), lp=lpv, hp=hpv, sigma=sigmav
        )
        img1ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img1)))
        s = img1.shape[0] * img1.shape[1]
        tmp = img1ft * np.conj(img1ft)
        img1ft = s * img1ft / np.sqrt(tmp.sum())
        img2ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img2)))
        img2ft[0, 0] = 0
        tmp = img2ft * np.conj(img2ft)
        img2ft = s * img2ft / np.sqrt(tmp.sum())
        xcorr = np.real(fftpack.fftshift(fftpack.ifft2(img1ft * np.conj(img2ft))))
    elif bp == "no":
        img1ft = fftpack.fft2(img1)
        img2ft = np.conj(fftpack.fft2(img2))
        img1ft[0, 0] = 0
        xcorr = np.abs(fftpack.fftshift(fftpack.ifft2(img1ft * img2ft)))
    else:
        logging.error(
            "ERROR in xcorr2: bandpass value ( bp= " + str(bp) + " ) not recognized"
        )
        return -1
    return xcorr


def bandpass_mask(size=(128, 128), lp=32, hp=2, sigma=3):
    x = size[0]
    y = size[1]
    lowpass = circ_mask(size=(x, y), radius=lp, sigma=0)
    hpass_tmp = circ_mask(size=(x, y), radius=hp, sigma=0)
    highpass = -1 * (hpass_tmp - 1)
    tmp = lowpass * highpass
    if sigma > 0:
        bandpass = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        bandpass = tmp
    return bandpass


def circ_mask(size=(128, 128), radius=32, sigma=3):
    x = size[0]
    y = size[1]
    img = Image.new("I", size)
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        (x / 2 - radius, y / 2 - radius, x / 2 + radius, y / 2 + radius),
        fill="white",
        outline="white",
    )
    tmp = np.array(img, float) / 255
    if sigma > 0:
        mask = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        mask = tmp
    return mask


# FROM AUTOLAMELLA
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


# def auto_focus_and_link(microscope):
#     import skimage

#     from skimage.morphology import disk
#     from skimage.filters.rank import gradient

#     PLOT = False

#     image_settings = {}
#     image_settings["resolution"] = "768x512"  # "1536x1024"
#     image_settings["dwell_time"] = 0.3e-6
#     image_settings["hfw"] = 50e-6
#     image_settings["autocontrast"] = True
#     image_settings["beam_type"] = BeamType.ELECTRON
#     image_settings["gamma"] = {
#         "correction": True,
#         "min_gamma": 0.15,
#         "max_gamma": 1.8,
#         "scale_factor": 0.01,
#         "threshold": 46,
#     }
#     image_settings["save"] = False
#     image_settings["label"] = ""
#     image_settings["save_path"] = None
#     print(image_settings)

#     microscope.beams.electron_beam.working_distance.value = 4e-3
#     current_working_distance = microscope.beams.electron_beam.working_distance.value

#     print("Initial: ", current_working_distance)

#     working_distances = [
#         current_working_distance - 0.5e-3,
#         current_working_distance - 0.25e-3,
#         current_working_distance,
#         current_working_distance + 0.25e-3,
#         current_working_distance + 0.5e-3
#     ]

#     # loop through working distances and calculate the sharpness (acutance)
#     # highest acutance is best focus
#     sharpeness_metric = []
#     for i, wd in enumerate(working_distances):
#         microscope.beams.electron_beam.working_distance.value = wd
#         img = acquire.new_image(microscope, image_settings)

#         print(f"Img {i}: {img.metadata.optics.working_distance:.5f}")

#         # sharpness (Acutance: https://en.wikipedia.org/wiki/Acutance
#         out = gradient(skimage.filters.median(np.copy(img.data)), disk(5))

#         if PLOT:
#             import matplotlib.pyplot as plt
#             plt.subplot(1, 2, 1)
#             plt.imshow(img.data, cmap="gray")
#             plt.subplot(1, 2, 2)
#             plt.imshow(out, cmap="gray")
#             plt.show()

#         sharpness = np.mean(out)
#         sharpeness_metric.append(sharpness)

#     # select working distance with max acutance
#     idx = np.argmax(sharpeness_metric)

#     print(*zip(working_distances, sharpeness_metric))
#     print(idx, working_distances[idx], sharpeness_metric[idx])

#     # reset working distance
#     microscope.beams.electron_beam.working_distance.value = working_distances[idx]

#     # run fine auto focus and link
#     microscope.imaging.set_active_view(1)  # set to Ebeam
#     microscope.auto_functions.run_auto_focus()
#     microscope.specimen.stage.link()


def get_current_microscope_state(microscope, stage: AutoLiftoutStage, eucentric: bool = False):
    """Get the current microscope state """

    current_microscope_state = MicroscopeState()
    current_microscope_state.timestamp = utils.current_timestamp()
    current_microscope_state.eucentric_calibration = eucentric
    current_microscope_state.last_completed_stage = stage

    # get absolute stage coordinates (RAW)
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.RAW)
    current_microscope_state.absolute_position = microscope.specimen.stage.current_position
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)

    # working_distance
    eb_image = acquire.last_image(microscope, BeamType.ELECTRON)
    ib_image = acquire.last_image(microscope, BeamType.ION)
    current_microscope_state.eb_working_distance = eb_image.metadata.optics.working_distance
    current_microscope_state.ib_working_distance = ib_image.metadata.optics.working_distance

    # beam_currents
    current_microscope_state.eb_beam_current = microscope.beams.electron_beam.beam_current.value
    current_microscope_state.ib_beam_current = microscope.beams.ion_beam.beam_current.value

    return current_microscope_state


def set_microscope_state(microscope, microscope_state: MicroscopeState):
    """Reset the microscope state to the provided state"""

    logging.info(f"restoring microscope state.")
    # move to position
    safe_absolute_stage_movement(microscope=microscope, stage_position=microscope_state.absolute_position)

    # check eucentricity?

    # set working distance
    logging.info(f"restoring ebeam working distance:  {microscope_state.eb_working_distance*1e3:.4f}mm")
    microscope.beams.electron_beam.working_distance.value = microscope_state.eb_working_distance
    logging.info(f"restoring ibeam working distance:  {microscope_state.ib_working_distance*1e3:.4f}mm")
    microscope.beams.ion_beam.working_distance.value = microscope_state.ib_working_distance

    # set beam currents
    logging.info(f"restoring ebeam current: {microscope_state.eb_beam_current:.2e}A")
    microscope.beams.electron_beam.beam_current.value = microscope_state.eb_beam_current
    logging.info(f"restoring ibeam current: {microscope_state.ib_beam_current:.2e}A")
    microscope.beams.ion_beam.beam_current.value = microscope_state.ib_beam_current

    logging.info(f"microscope state restored")
    return


def reset_beam_shifts(microscope: SdbMicroscopeClient):
    """Set the beam shift to zero for the electron and ion beams

    Args:
        microscope (SdbMicroscopeClient): Autoscript microscope object
    """
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings, Point

    # reset zero beamshift
    logging.info(f"reseting ebeam shift to (0, 0) from: {microscope.beams.electron_beam.beam_shift.value} ")
    microscope.beams.electron_beam.beam_shift.value = Point(0, 0)
    logging.info(f"reseting ibeam shift to (0, 0) from: {microscope.beams.electron_beam.beam_shift.value} ")
    microscope.beams.ion_beam.beam_shift.value = Point(0, 0)
    logging.info(f"reset beam shifts to zero complete")


def check_working_distance_is_within_tolerance(eb_image, ib_image, settings):
    eb_eucentric_height = settings["calibration"]["eucentric_height_eb"]
    eb_eucentric_tolerance = settings["calibration"]["eucentric_height_tolerance"]
    # TODO: use np.isclose instead
    return abs(eb_image.metadata.optics.working_distance - eb_eucentric_height) <= eb_eucentric_tolerance


def validate_stage_calibration(microscope):

    if not microscope.specimen.stage.is_homed:
        logging.warning("Stage is not homed.")

    if not microscope.specimen.stage.is_linked:
        logging.warning("Stage is not linked.")

    logging.info("Stage calibration validation complete.")

    return


def validate_needle_calibration(microscope):

    if str(microscope.specimen.manipulator.state) == "Retracted":
        logging.info("Needle is retracted")
    else:
        logging.warning("Needle is inserted. Please retract before starting.")
        # TODO: retract automatically?

    # TODO: calibrate needle?

    return


def validate_beams_calibration(microscope, settings: dict):
    """Validate Beam Settings"""

    high_voltage = float(settings["system"]["high_voltage"])
    plasma_gas = str(settings["system"]["plasma_gas"]).capitalize()

    # TODO: check beam blanks?
    # TODO: check electron voltage?

    logging.info("Validating Electron Beam")
    if not microscope.beams.electron_beam.is_on:
        logging.warning("Electron Beam is not on, switching on now...")
        microscope.beams.electron_beam.turn_on()
        assert microscope.beams.electron_beam.is_on, "Unable to turn on Electron Beam."
        logging.warning("Electron Beam turned on.")

    microscope.imaging.set_active_view(1)
    if str(microscope.detector.type.value) != "ETD":
        logging.warning(
            f"Electron detector type is  should be ETD (Currently is {str(microscope.detector.type.value)})")
        if "ETD" in microscope.detector.type.available_values:
            microscope.detector.type.value = "ETD"
            logging.warning(f"Changed Electron detector type to {str(microscope.detector.type.value)}")

    if str(microscope.detector.mode.value) != "SecondaryElectrons":
        logging.warning(
            f"Electron detector mode is should be SecondaryElectrons (Currently is {str(microscope.detector.mode.value)}")
        if "SecondaryElectrons" in microscope.detector.mode.available_values:
            microscope.detector.mode.value = "SecondaryElectrons"
            logging.warning(f"Changed Electron detector mode to {str(microscope.detector.mode.value)}")

    # working distances
    logging.info(f"EB Working Distance: {microscope.beams.electron_beam.working_distance.value:.4f}m")
    if not np.isclose(microscope.beams.electron_beam.working_distance.value,
                      settings["calibration"]["eucentric_height_eb"],
                      atol=settings["calibration"]["eucentric_height_tolerance"]):
        logging.warning(
            f"""Electron Beam is not close to eucentric height. It should be {settings['calibration']['eucentric_height_eb']}m \
                (Currently is {microscope.beams.electron_beam.working_distance.value:.4f}m)""")

    logging.info(f"E OPTICAL MODE: {str(microscope.beams.electron_beam.optical_mode.value)}")
    logging.info(f"E OPTICAL MODES:  {str(microscope.beams.electron_beam.optical_mode.available_values)}")

    # Validate Ion Beam
    logging.info("Validating Ion Beam")

    if not microscope.beams.ion_beam.is_on:
        logging.warning("Ion Beam is not on, switching on now...")
        microscope.beams.ion_beam.turn_on()
        assert microscope.beams.ion_beam.is_on, "Unable to turn on Ion Beam."
        logging.warning("Ion Beam turned on.")

    microscope.imaging.set_active_view(2)
    if str(microscope.detector.type.value) != "ETD":
        logging.warning(
            f"Ion detector type is  should be ETD (Currently is {str(microscope.detector.type.value)})")
        if "ETD" in microscope.detector.type.available_values:
            microscope.detector.type.value = "ETD"
            logging.warning(f"Changed Ion detector type to {str(microscope.detector.type.value)}")

    if str(microscope.detector.mode.value) != "SecondaryElectrons":
        logging.warning(
            f"Ion detector mode is should be SecondaryElectrons (Currently is {str(microscope.detector.mode.value)}")
        if "SecondaryElectrons" in microscope.detector.mode.available_values:
            microscope.detector.mode.value = "SecondaryElectrons"
            logging.warning(f"Changed Ion detector mode to {str(microscope.detector.mode.value)}")

    # working distance
    logging.info(f"IB Working Distance: {microscope.beams.ion_beam.working_distance.value:.4f}m")
    if not np.isclose(microscope.beams.ion_beam.working_distance.value,
                      settings["calibration"]["eucentric_height_ib"],
                      atol=settings["calibration"]["eucentric_height_tolerance"]):
        logging.warning(f"Ion Beam is not close to eucentric height. It should be {settings['calibration']['eucentric_height_ib']}m \
        (Currently is {microscope.beams.ion_beam.working_distance.value:.4f}m)")

    # validate high voltage
    high_voltage_limits = str(microscope.beams.ion_beam.high_voltage.limits)
    logging.info(f"Ion Beam High Voltage Limits are: {high_voltage_limits}")

    if microscope.beams.ion_beam.high_voltage.value != high_voltage:
        logging.warning(
            f"Ion Beam High Voltage should be {high_voltage}V (Currently {microscope.beams.ion_beam.high_voltage.value}V)")

        if bool(microscope.beams.ion_beam.high_voltage.is_controllable):
            logging.warning(f"Changing Ion Beam High Voltage to {high_voltage}V.")
            microscope.beams.ion_beam.high_voltage.value = high_voltage
            assert microscope.beams.ion_beam.high_voltage.value == high_voltage, "Unable to change Ion Beam High Voltage"
            logging.warning(f"Ion Beam High Voltage Changed")

    # validate plasma gas
    if plasma_gas not in microscope.beams.ion_beam.source.plasma_gas.available_values:
        logging.warning("{plasma_gas} is not available as a plasma gas.")

    if microscope.beams.ion_beam.source.plasma_gas.value != plasma_gas:
        logging.warning(
            f"Plasma Gas is should be {plasma_gas} (Currently {microscope.beams.ion_beam.source.plasma_gas.value})")

    # reset beam shifts
    reset_beam_shifts(microscope=microscope)


def validate_chamber(microscope):
    """Validate the state of the chamber"""

    logging.info(f"Validating Vacuum Chamber State: {str(microscope.vacuum.chamber_state)}")
    if not str(microscope.vacuum.chamber_state) == "Pumped":
        logging.warning(
            f"Chamber vacuum state should be Pumped (Currently is {str(microscope.vacuum.chamber_state)})")

    logging.info(f"Validating Vacuum Chamber Pressure: {microscope.state.chamber_pressure.value:.6f} mbar")
    if microscope.state.chamber_pressure.value >= 1e-3:
        logging.warning(
            f"Chamber pressure is too high, please pump the system (Currently {microscope.state.chamber_pressure.value:.6f} mbar)")
