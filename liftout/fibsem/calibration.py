import logging
import os

from datetime import datetime
import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
from autoscript_sdb_microscope_client.structures import (
    AdornedImage,
    StagePosition,
)
from liftout import utils
from liftout.detection import detection
from liftout.detection.utils import DetectionResult
from liftout.fibsem import acquire, movement
from liftout.fibsem.acquire import ImageSettings, BeamType
from liftout.fibsem.sample import (
    AutoLiftoutStage,
    BeamSettings,
    Lamella,
    MicroscopeState,
    ReferenceImages,
)

from liftout.model import models
from PIL import Image, ImageDraw
from scipy import fftpack


def correct_stage_drift_v2(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    reference_images: ReferenceImages,
    alignment: tuple(BeamType) = (BeamType.ELECTRON, BeamType.ELECTRON),
    rotate: bool = False,
    parent_ui=None,
) -> bool:

    # set reference images
    if alignment[0] is BeamType.ELECTRON:
        ref_lowres, ref_highres = (
            reference_images.low_res_eb,
            reference_images.high_res_eb,
        )
    if alignment[0] is BeamType.ION:
        ref_lowres, ref_highres = (
            reference_images.low_res_ib,
            reference_images.high_res_ib,
        )

    # rotate reference
    if rotate:
        ref_lowres = rotate_AdornedImage(ref_lowres)
        ref_highres = rotate_AdornedImage(ref_highres)

    # align lowres, then highres
    for ref_image in [ref_lowres, ref_highres]:

        # take new images
        # set new image settings (same as reference)
        image_settings = match_image_settings(
            ref_image, image_settings, beam_type=alignment[1]
        )
        new_image = acquire.new_image(microscope, image_settings)

        # crosscorrelation alignment
        ret = align_using_reference_images_v2(
            microscope, settings, ref_lowres, new_image,
        )

        if ret is False and parent_ui:
            from liftout.gui import windows

            # # cross-correlation has failed, manual correction required
            windows.ask_user_movement_v2(
                microscope,
                settings,
                image_settings,
                msg_type="centre_eb",
                parent=parent_ui,
            )
            break

    return ret


def match_image_settings(
    ref_image: AdornedImage,
    image_settings: ImageSettings,
    beam_type: BeamType = BeamType.ELECTRON,
) -> ImageSettings:
    image_settings.hfw = f"{ref_image.height}x{ref_image.width}"
    image_settings.dwell_time = ref_image.metadata.scan_settings.dwell_time
    image_settings.beam_type = beam_type
    image_settings.save = True
    image_settings.label = f"drift_correction_{utils.current_timestamp()}"

    return image_settings


def align_using_reference_images_v2(
    microscope: SdbMicroscopeClient,
    settings: dict,
    ref_image: AdornedImage,
    new_image: AdornedImage,
) -> bool:

    # get beam type
    ref_beam_type = BeamType(ref_image.metadata.acquisition.beam_type.upper())
    new_beam_type = BeamType(new_image.metadata.acquisition.beam_type.upper())

    logging.info(
        f"aligning {ref_beam_type.name} reference image to {new_beam_type.name}."
    )
    lp_px = int(max(new_image.data.shape) * 0.66)
    hp_px = int(max(new_image.data.shape) / 64)
    sigma = 6

    dx, dy, _ = shift_from_crosscorrelation(
        new_image, ref_image, lowpass=lp_px, highpass=hp_px, sigma=sigma
    )

    shift_within_tolerance = check_shift_within_tolerance(
        dx=dx, dy=dy, ref_image=ref_image
    )

    if shift_within_tolerance:

        # move the stage
        movement.move_stage_relative_with_corrected_movement(microscope, 
            settings, 
            dx=-dx, 
            dy=dy, 
            beam_type=new_beam_type)

    return shift_within_tolerance


def check_shift_within_tolerance(
    dx: float, dy: float, ref_image: AdornedImage, limit: float = 0.25
) -> bool:
    """Limit automatic movements to 25% of the field of view"""
    # check if the cross correlation movement is within the safety limit

    pixelsize_x = ref_image.metadata.binary_result.pixel_size.x
    X_THRESHOLD = limit * pixelsize_x * ref_image.width
    Y_THRESHOLD = limit * pixelsize_x * ref_image.height

    return abs(dx) < X_THRESHOLD and abs(dy) < Y_THRESHOLD


def identify_shift_using_machine_learning(
    microscope: SdbMicroscopeClient,
    image_settings: ImageSettings,
    settings: dict,
    shift_type: tuple,
) -> DetectionResult:

    # load model
    weights_file = settings["calibration"]["machine_learning"]["weights"]
    weights_path = os.path.join(os.path.dirname(models.__file__), weights_file)
    detector = detection.Detector(weights_path)

    # take new image
    image = acquire.new_image(microscope, image_settings)
    det_result = detector.locate_shift_between_features(image, shift_type=shift_type)

    return det_result


def rotate_AdornedImage(image: AdornedImage):

    data = np.rot90(np.rot90(np.copy(image.data)))
    reference = AdornedImage(data=data)
    reference.metadata = image.metadata
    return reference


def normalise_image(img: AdornedImage) -> np.ndarray:
    """Normalise the image"""
    return (img.data - np.mean(img.data)) / np.std(img.data)


def shift_from_crosscorrelation(
    img1: AdornedImage,
    img2: AdornedImage,
    lowpass: int = 128,
    highpass: int = 6,
    sigma: int = 6,
    use_rect_mask: bool = False,
):

    # get pixel_size
    pixelsize_x = img2.metadata.binary_result.pixel_size.x
    pixelsize_y = img2.metadata.binary_result.pixel_size.y

    # normalise both images
    img1_data_norm = normalise_image(img1)
    img2_data_norm = normalise_image(img2)

    # cross-correlate normalised images
    if use_rect_mask:
        rect_mask = _mask_rectangular(img2_data_norm.shape)
        img1_data_norm = rect_mask * img1_data_norm
        img2_data_norm = rect_mask * img2_data_norm

    # run crosscorrelation
    xcorr = crosscorrelation_v2(
        img1_data_norm, img2_data_norm, bp=True, lp=lowpass, hp=highpass, sigma=sigma
    )

    # calculate maximum crosscorrelation
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    cen = np.asarray(xcorr.shape) / 2
    err = np.array(cen - [maxX, maxY], int)

    # calculate shift in metres
    x_shift = err[1] * pixelsize_x
    y_shift = err[0] * pixelsize_y
    
    logging.info(f"cross-correlation:")
    logging.info(f"maxX: {maxX}, {maxY}, centre: {cen}")
    logging.info(f"x: {err[1]}px, y: {err[0]}px")
    logging.info(f"x: {x_shift:.2e}m, y: {y_shift:.2e} meters")

    # metres
    return x_shift, y_shift, xcorr

def crosscorrelation_v2(img1: np.ndarray, img2: np.ndarray,  
    lp: int = 128, hp: int = 6, sigma: int = 6, bp: bool = False) -> np.ndarray:
    
    if img1.shape != img2.shape:
        err = f"Image 1 {img1.shape} and Image 2 {img2.shape} need to have the same shape"
        logging.error(err)
        raise ValueError(err)

    if bp: 
        bandpass = bandpass_mask(
            size=(img1.shape[1], img1.shape[0]), 
            lp=lp, hp=hp, sigma=sigma
        )
        n_pixels = img1.shape[0] * img1.shape[1]
        
        img1ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img1)))
        tmp = img1ft * np.conj(img1ft)
        img1ft = n_pixels * img1ft / np.sqrt(tmp.sum())
        
        img2ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img2)))
        img2ft[0, 0] = 0
        tmp = img2ft * np.conj(img2ft)
        
        img2ft = n_pixels * img2ft / np.sqrt(tmp.sum())

        xcorr = np.real(fftpack.fftshift(fftpack.ifft2(img1ft * np.conj(img2ft))))
    else:
        img1ft = fftpack.fft2(img1)
        img2ft = np.conj(fftpack.fft2(img2))
        img1ft[0, 0] = 0
        xcorr = np.abs(fftpack.fftshift(fftpack.ifft2(img1ft * img2ft)))
    
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


def get_raw_stage_position(microscope: SdbMicroscopeClient) -> StagePosition:
    """Get the current stage position in raw coordinate system, and switch back to specimen"""
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.RAW)
    stage_position = microscope.specimen.stage.current_position
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)

    return stage_position


def get_current_microscope_state_v2(
    microscope: SdbMicroscopeClient, stage: AutoLiftoutStage
) -> MicroscopeState:
    """Get the current microscope state v2 """

    current_microscope_state = MicroscopeState(
        timestamp=datetime.timestamp(datetime.now()),
        last_completed_stage=stage,
        # get absolute stage coordinates (RAW)
        absolute_position=get_raw_stage_position(microscope),
        # electron beam settings
        eb_settings=BeamSettings(
            beam_type=BeamType.ELECTRON,
            working_distance=microscope.beams.electron_beam.working_distance.value,
            beam_current=microscope.beams.electron_beam.beam_current.value,
            hfw=microscope.beams.electron_beam.horizontal_field_width.value,
            resolution=microscope.beams.electron_beam.scanning.resolution.value,
            dwell_time=microscope.beams.electron_beam.scanning.dwell_time.value,
            stigmation=microscope.beams.electron_beam.stigmator.value,
        ),
        # ion beam settings
        ib_settings=BeamSettings(
            beam_type=BeamType.ION,
            working_distance=microscope.beams.ion_beam.working_distance.value,
            beam_current=microscope.beams.ion_beam.beam_current.value,
            hfw=microscope.beams.ion_beam.horizontal_field_width.value,
            resolution=microscope.beams.ion_beam.scanning.resolution.value,
            dwell_time=microscope.beams.ion_beam.scanning.dwell_time.value,
            stigmation=microscope.beams.ion_beam.stigmator.value,
        ),
    )

    return current_microscope_state


def set_microscope_state_v2(microscope, microscope_state: MicroscopeState):
    """Reset the microscope state to the provided state"""

    logging.info(
        f"restoring microscope state to {microscope_state.last_completed_stage.name} stage."
    )

    # move to position
    movement.safe_absolute_stage_movement(
        microscope=microscope, stage_position=microscope_state.absolute_position
    )

    # restore electron beam
    logging.info(f"restoring electron beam settings...")
    microscope.beams.electron_beam.working_distance.value = (
        microscope_state.eb_settings.working_distance
    )
    microscope.beams.electron_beam.beam_current.value = (
        microscope_state.eb_settings.beam_current
    )
    microscope.beams.electron_beam.horizontal_field_width.value = (
        microscope_state.eb_settings.hfw
    )
    microscope.beams.electron_beam.scanning.resolution.value = (
        microscope_state.eb_settings.resolution
    )
    microscope.beams.electron_beam.scanning.dwell_time.value = (
        microscope_state.eb_settings.dwell_time
    )
    microscope.beams.electron_beam.stigmator.value = (
        microscope_state.eb_settings.stigmation
    )

    # restore ion beam
    logging.info(f"restoring ion beam settings...")
    microscope.beams.ion_beam.working_distance.value = (
        microscope_state.ib_settings.working_distance
    )
    microscope.beams.ion_beam.beam_current.value = (
        microscope_state.ib_settings.beam_current
    )
    microscope.beams.ion_beam.horizontal_field_width.value = (
        microscope_state.ib_settings.hfw
    )
    microscope.beams.ion_beam.scanning.resolution.value = (
        microscope_state.ib_settings.resolution
    )
    microscope.beams.ion_beam.scanning.dwell_time.value = (
        microscope_state.ib_settings.dwell_time
    )
    microscope.beams.ion_beam.stigmator.value = microscope_state.ib_settings.stigmation

    logging.info(f"microscope state restored")
    return


def reset_beam_shifts(microscope: SdbMicroscopeClient):
    """Set the beam shift to zero for the electron and ion beams

    Args:
        microscope (SdbMicroscopeClient): Autoscript microscope object
    """
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings, Point

    # reset zero beamshift
    logging.info(
        f"reseting ebeam shift to (0, 0) from: {microscope.beams.electron_beam.beam_shift.value} "
    )
    microscope.beams.electron_beam.beam_shift.value = Point(0, 0)
    logging.info(
        f"reseting ibeam shift to (0, 0) from: {microscope.beams.electron_beam.beam_shift.value} "
    )
    microscope.beams.ion_beam.beam_shift.value = Point(0, 0)
    logging.info(f"reset beam shifts to zero complete")


def check_working_distance_is_within_tolerance(
    microscope, settings, beam_type: BeamType = BeamType.ELECTRON
):

    if beam_type is BeamType.ELECTRON:
        working_distance = microscope.beams.electron_beam.working_distance
        eucentric_height = settings["calibration"]["limits"]["eucentric_height_eb"]
        eucentric_tolerance = settings["calibration"]["limits"][
            "eucentric_height_tolerance"
        ]

    if beam_type is BeamType.ION:
        working_distance = microscope.beams.electron_beam.working_distance
        eucentric_height = settings["calibration"]["limits"]["eucentric_height_ib"]
        eucentric_tolerance = settings["calibration"]["limits"][
            "eucentric_height_tolerance"
        ]

    return np.isclose(working_distance, eucentric_height, atol=eucentric_tolerance)


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

    high_voltage = float(settings["system"]["high_voltage"])  # ion
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
            f"Electron detector type is  should be ETD (Currently is {str(microscope.detector.type.value)})"
        )
        # if "ETD" in microscope.detector.type.available_values:
        #     microscope.detector.type.value = "ETD"
        #     logging.warning(
        #         f"Changed Electron detector type to {str(microscope.detector.type.value)}"
        #     )

    if str(microscope.detector.mode.value) != "SecondaryElectrons":
        logging.warning(
            f"Electron detector mode is should be SecondaryElectrons (Currently is {str(microscope.detector.mode.value)}"
        )
        if "SecondaryElectrons" in microscope.detector.mode.available_values:
            microscope.detector.mode.value = "SecondaryElectrons"
            logging.warning(
                f"Changed Electron detector mode to {str(microscope.detector.mode.value)}"
            )

    # working distances
    logging.info(
        f"EB Working Distance: {microscope.beams.electron_beam.working_distance.value:.4f}m"
    )
    if not np.isclose(
        microscope.beams.electron_beam.working_distance.value,
        settings["calibration"]["limits"]["eucentric_height_eb"],
        atol=settings["calibration"]["limits"]["eucentric_height_tolerance"],
    ):
        logging.warning(
            f"""Electron Beam is not close to eucentric height. It should be {settings['calibration']['limits']['eucentric_height_eb']}m
            (Currently is {microscope.beams.electron_beam.working_distance.value:.4f}m)"""
        )

    logging.info(
        f"E OPTICAL MODE: {str(microscope.beams.electron_beam.optical_mode.value)}"
    )
    logging.info(
        f"E OPTICAL MODES:  {str(microscope.beams.electron_beam.optical_mode.available_values)}"
    )

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
            f"Ion detector type is  should be ETD (Currently is {str(microscope.detector.type.value)})"
        )
        if "ETD" in microscope.detector.type.available_values:
            microscope.detector.type.value = "ETD"
            logging.warning(
                f"Changed Ion detector type to {str(microscope.detector.type.value)}"
            )

    if str(microscope.detector.mode.value) != "SecondaryElectrons":
        logging.warning(
            f"Ion detector mode is should be SecondaryElectrons (Currently is {str(microscope.detector.mode.value)}"
        )
        if "SecondaryElectrons" in microscope.detector.mode.available_values:
            microscope.detector.mode.value = "SecondaryElectrons"
            logging.warning(
                f"Changed Ion detector mode to {str(microscope.detector.mode.value)}"
            )

    # working distance
    logging.info(
        f"IB Working Distance: {microscope.beams.ion_beam.working_distance.value:.4f}m"
    )
    if not np.isclose(
        microscope.beams.ion_beam.working_distance.value,
        settings["calibration"]["limits"]["eucentric_height_ib"],
        atol=settings["calibration"]["limits"]["eucentric_height_tolerance"],
    ):
        logging.warning(
            f"Ion Beam is not close to eucentric height. It should be {settings['calibration']['limits']['eucentric_height_ib']}m \
        (Currently is {microscope.beams.ion_beam.working_distance.value:.4f}m)"
        )

    # validate high voltage
    high_voltage_limits = str(microscope.beams.ion_beam.high_voltage.limits)
    logging.info(f"Ion Beam High Voltage Limits are: {high_voltage_limits}")

    if microscope.beams.ion_beam.high_voltage.value != high_voltage:
        logging.warning(
            f"Ion Beam High Voltage should be {high_voltage}V (Currently {microscope.beams.ion_beam.high_voltage.value}V)"
        )

        if bool(microscope.beams.ion_beam.high_voltage.is_controllable):
            logging.warning(f"Changing Ion Beam High Voltage to {high_voltage}V.")
            microscope.beams.ion_beam.high_voltage.value = high_voltage
            assert (
                microscope.beams.ion_beam.high_voltage.value == high_voltage
            ), "Unable to change Ion Beam High Voltage"
            logging.warning(f"Ion Beam High Voltage Changed")

    # validate plasma gas
    if plasma_gas not in microscope.beams.ion_beam.source.plasma_gas.available_values:
        logging.warning("{plasma_gas} is not available as a plasma gas.")

    if microscope.beams.ion_beam.source.plasma_gas.value != plasma_gas:
        logging.warning(
            f"Plasma Gas is should be {plasma_gas} (Currently {microscope.beams.ion_beam.source.plasma_gas.value})"
        )

    # reset beam shifts
    reset_beam_shifts(microscope=microscope)


def validate_chamber(microscope):
    """Validate the state of the chamber"""

    logging.info(
        f"Validating Vacuum Chamber State: {str(microscope.vacuum.chamber_state)}"
    )
    if not str(microscope.vacuum.chamber_state) == "Pumped":
        logging.warning(
            f"Chamber vacuum state should be Pumped (Currently is {str(microscope.vacuum.chamber_state)})"
        )

    logging.info(
        f"Validating Vacuum Chamber Pressure: {microscope.state.chamber_pressure.value:.6f} mbar"
    )
    if microscope.state.chamber_pressure.value >= 1e-3:
        logging.warning(
            f"Chamber pressure is too high, please pump the system (Currently {microscope.state.chamber_pressure.value:.6f} mbar)"
        )


def beam_shift_alignment(
    microscope: SdbMicroscopeClient,
    image_settings: ImageSettings,
    ref_image: AdornedImage,
    reduced_area,
):
    """Align the images by adjusting the beam shift, instead of moving the stage
            (increased precision, lower range)

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope client
        image_settings (acquire.ImageSettings): settings for taking image
        ref_image (AdornedImage): reference image to align to
        reduced_area (Rectangle): The reduced area to image with.
    """

    # # align using cross correlation
    img1 = ref_image
    img2 = acquire.new_image(
        microscope, settings=image_settings, reduced_area=reduced_area
    )
    dx, dy, _ = shift_from_crosscorrelation(
        img1, img2, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )

    # adjust beamshift
    microscope.beams.ion_beam.beam_shift.value += (-dx, dy)


def automatic_eucentric_correction(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    eucentric_height: float = 3.9e-3,
):
    """Automatic procedure to reset to the eucentric position

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope client connection
        settings (dict): configuration dictionary
        image_settings (ImageSettings): imaging settings
        eucentric_height (float, optional): manually calibrated eucentric height. Defaults to 3.9e-3.
    """

    # autofocus in eb
    movement.auto_link_stage(microscope)

    # move stage to z=3.9
    microscope.specimen.stage.set_default_coordinate_system()

    # turn on z-y linked movement # NB: cant do this through API
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)

    eucentric_position = StagePosition(z=eucentric_height)
    movement.safe_absolute_stage_movement(microscope, eucentric_position)

    # retake images to check
    acquire.take_reference_images(microscope, image_settings)


def automatic_eucentric_correction_v2(
    microscope: SdbMicroscopeClient, settings: dict, image_settings: ImageSettings
) -> None:

    # assume the feature of interest is on the image centre.

    # iterative eucentric alignment

    hfw = 900e-6
    tolerance = 5.0e-6
    iteration = 0
    
    while True:

        # take low res reference image
        image_settings.hfw = hfw
        ref_eb, ref_ib = acquire.take_reference_images(microscope, image_settings)

        # calculate cross correlation...
        # x = horizontal, y = vertical

        # align using cross correlation
        dx, dy, _ = shift_from_crosscorrelation(
            ref_eb, ref_ib, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
        )

        # stop if both are within tolernace
        if dy <= tolerance:
            break

        # move z??

        # align eb (cross correlate) back to original ref (move eb back to centre)
        image_settings.beam_type = BeamType.ELECTRON
        new_eb = acquire.new_image(microscope, image_settings, reduced_area=None)
        dx, dy, _ = shift_from_crosscorrelation(
            ref_eb, new_eb, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
        )

        # move feature back to centre of eb
        movement.move_stage_relative_with_corrected_movement(
            microscope =microscope,
            settings=settings,
            dx=dx, dy=dy, beam_type=BeamType.ELECTRON)

        # repeat


        # increase count
        iteration += 1

        if iteration == 5:
            # unable to align within given iterations
            break
        

    # TODO: do we want to align in x too?

    return


def correct_stage_drift_with_ML_v2(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
    iterations: int = 3
):
    # correct stage drift using machine learning
    from liftout.detection.utils import DetectionType


    iteration_count = 0
    beam_type = BeamType.ION
    label = image_settings.label

    while True:
        image_settings.label = label + utils.current_timestamp()
        det = validate_detection_v2(
            microscope,
            settings,
            image_settings,
            lamella=lamella,
            shift_type=(DetectionType.ImageCentre, DetectionType.LamellaCentre),
            beam_type=beam_type,
        )

        movement.move_stage_relative_with_corrected_movement(
            microscope=microscope,
            settings=settings,
            dx=det.distance_metres.x,
            dy=det.distance_metres.y,
            beam_type=beam_type,
        )

        beam_type = BeamType.ELECTRON if beam_type is BeamType.ION else BeamType.ION

        iteration_count += 1
        if iteration_count >= iterations:
            break

    image_settings.save = True
    image_settings.label = f"drift_correction_ML_final_" + utils.current_timestamp()
    acquire.take_reference_images(microscope, image_settings)


def validate_detection_v2(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
    shift_type: tuple,
    beam_type: BeamType = BeamType.ELECTRON,
    parent=None,
):
    # TODO: validate the detection shift type...
    from liftout.gui.detection_window import GUIDetectionWindow

    image_settings.beam_type = beam_type  # change to correct beamtype

    # run model detection
    detection_result = identify_shift_using_machine_learning(
        microscope, image_settings, settings, shift_type=shift_type
    )

    # user validates detection result
    detection_window = GUIDetectionWindow(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        detection_result=detection_result,
        lamella=lamella,
        parent=parent,
    )
    detection_window.show()
    detection_window.exec_()

    return detection_window.detection_result


def validate_stage_height_for_needle_insertion(
    microscope: SdbMicroscopeClient, settings: dict, image_settings: ImageSettings
) -> None:
    from liftout.gui import windows

    stage = microscope.specimen.stage
    stage_height_limit = settings["calibration"]["limits"]["stage_height_limit"]

    if stage.current_position.z < stage_height_limit:

        # Unable to insert the needle if the stage height is below this limit (3.7e-3)
        logging.warning(f"Calibration error detected: stage position height")
        logging.warning(f"Stage Position: {stage.current_position}")

        windows.ask_user_interaction_v2(
            microscope,
            msg="""The system has identified the distance between the sample and the pole piece is less than 3.7mm. "
            "The needle will contact the sample, and it is unsafe to insert the needle. "
            "\nPlease manually refocus and link the stage, then press OK to continue. """,
            beam_type=BeamType.ELECTRON,
        )

    return


def validate_focus(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    link: bool = True,
) -> None:

    from liftout.gui import windows

    # check focus distance is within tolerance
    if link:
        movement.auto_link_stage(microscope)  # TODO: remove?

    if not check_working_distance_is_within_tolerance(
        microscope, settings=settings, beam_type=BeamType.ELECTRON
    ):
        logging.warning("Autofocus has failed")
        windows.ask_user_interaction_v2(
            microscope,
            msg="The AutoFocus routine has failed, please correct the focus manually.",
            beam_type=BeamType.ELECTRON,
        )

    return

