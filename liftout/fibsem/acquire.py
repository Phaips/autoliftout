from torch.nn.functional import threshold
from autoscript_sdb_microscope_client.structures import *
from enum import Enum
import logging
from liftout import utils
from skimage import exposure
import numpy as np

# from liftout.fibsem import calibration


class BeamType(Enum):
    ELECTRON = 1
    ION = 2


def autocontrast(microscope, beam_type=BeamType.ELECTRON):
    """Automatically adjust the microscope image contrast."""
    microscope.imaging.set_active_view(beam_type.value)

    RunAutoCbSettings(
        method="MaxContrast",
        resolution="768x512",  # low resolution, so as not to damage the sample
        number_of_frames=5,
    )
    logging.info("acquire: automatically adjusting contrast...")
    microscope.auto_functions.run_auto_cb()


# TODO: these settings should be image_settings for consistency
def take_reference_images(microscope, settings):
    tmp_beam_type = settings["beam_type"]
    settings["beam_type"] = BeamType.ELECTRON
    eb_image = new_image(microscope, settings)
    settings["beam_type"] = BeamType.ION
    ib_image = new_image(microscope, settings)
    settings["beam_type"] = tmp_beam_type  # reset to original beam type
    return eb_image, ib_image


def gamma_correction(
    image, min_gamma, max_gamma, scale_factor, threshold
) -> AdornedImage:
    """Automatic gamma correction"""
    std = np.std(image.data)
    mean = np.mean(image.data)
    diff = mean - 255 / 2.0
    gam = np.clip(min_gamma, 1 + diff * scale_factor, max_gamma)
    if abs(diff) < threshold:
        gam = 1.0
    logging.info(f"gamma_correction: diff: {diff:.3f}, gam: {gam:.3f} ")
    image_data = exposure.adjust_gamma(image.data, gam)
    reference = AdornedImage(data=image_data)
    reference.metadata = image.metadata
    image = reference
    return image


def new_image(microscope, settings):
    frame_settings = GrabFrameSettings(
        resolution=settings["resolution"], dwell_time=settings["dwell_time"]
    )
    tmp_settings = settings
    if settings["beam_type"] == BeamType.ELECTRON:
        microscope.beams.electron_beam.horizontal_field_width.value = settings["hfw"]
        # settings['label'] += '_eb'
        label = (
            settings["label"] + "_eb"
        )  # TODO: need to update sample images filenames too
    else:
        microscope.beams.ion_beam.horizontal_field_width.value = settings["hfw"]
        # settings['label'] += '_ib'
        label = (
            settings["label"] + "_ib"
        )  # TODO: need to update sample images filenames too

    if settings["autocontrast"]:
        autocontrast(microscope, beam_type=settings["beam_type"])
        settings["contrast"] = None
        settings["brightness"] = None

    image = acquire_image(
        microscope=microscope,
        settings=frame_settings,
        brightness=settings["brightness"],
        contrast=settings["contrast"],
        beam_type=settings["beam_type"],
    )

    # apply gamma correction
    if settings["gamma"]["correction"]:

        # gamma parameters
        min_gamma = settings["gamma"]["min_gamma"]
        max_gamma = settings["gamma"]["max_gamma"]
        scale_factor = settings["gamma"]["scale_factor"]
        threshold = settings["gamma"]["threshold"]

        image = gamma_correction(image, min_gamma, max_gamma, scale_factor, threshold)

    if settings["save"]:
        utils.save_image(
            image=image, save_path=settings["save_path"], label=label
        )  # TO_TEST
        #  label=settings['label'])

    settings = tmp_settings  # reset the settings to original # TODO: this doesnt work, need to reset
    return image


def last_image(microscope, beam_type=BeamType.ELECTRON):
    """Get the last previously acquired ion or electron beam image.

    Parameters
    ----------
    microscope : Autoscript microscope object.
    beam_type :

    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    microscope.imaging.set_active_view(beam_type.value)
    image = microscope.imaging.get_image()
    return image


def acquire_image(
    microscope,
    settings=None,
    brightness=None,
    contrast=None,
    beam_type=BeamType.ELECTRON,
):
    """Take new electron or ion beam image.
    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    logging.info(f"acquire: acquiring new {beam_type.name} image.")
    microscope.imaging.set_active_view(beam_type.value)
    if brightness:
        microscope.detector.brightness.value = brightness
    if contrast:
        microscope.detector.contrast.value = contrast
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image
