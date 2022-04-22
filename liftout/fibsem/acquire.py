
from tkinter import Image
from autoscript_sdb_microscope_client.structures import RunAutoCbSettings, GrabFrameSettings, AdornedImage
from enum import Enum
import logging
from liftout import utils
from skimage import exposure
import numpy as np

# from liftout.fibsem import calibration

from dataclasses import dataclass
from pathlib import Path


class BeamType(Enum):
    ELECTRON = 1
    ION = 2


@dataclass
class GammaSettings:
    enabled: bool
    min_gamma: float
    max_gamma: float
    scale_factor: float
    threshold: int  # px


@dataclass
class ImageSettings:
    resolution: str
    dwell_time: float
    hfw: float
    autocontrast: bool
    beam_type: BeamType
    save: bool
    save_path: Path
    label: str
    gamma: GammaSettings


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


def take_reference_images(microscope, image_settings: ImageSettings):
    tmp_beam_type = image_settings.beam_type
    image_settings.beam_type = BeamType.ELECTRON
    eb_image = new_image(microscope, image_settings)
    image_settings.beam_type = BeamType.ION
    ib_image = new_image(microscope, image_settings)
    image_settings.beam_type = tmp_beam_type  # reset to original beam type
    return eb_image, ib_image


def gamma_correction(image, settings: GammaSettings) -> AdornedImage:
    """Automatic gamma correction"""
    std = np.std(image.data)
    mean = np.mean(image.data)
    diff = mean - 255 / 2.0
    gam = np.clip(settings.min_gamma, 1 + diff * settings.scale_factor, settings.max_gamma)
    if abs(diff) < settings.threshold:
        gam = 1.0
    logging.info(f"GAMMA_CORRECTION | {image.metadata.acquisition.beam_type} | {diff:.3f} | {gam:.3f}")
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
        settings["hfw"] = np.clip(settings["hfw"],
                                  microscope.beams.electron_beam.horizontal_field_width.limits.min,
                                  microscope.beams.electron_beam.horizontal_field_width.limits.max,
                                  )
        microscope.beams.electron_beam.horizontal_field_width.value = settings["hfw"]
        label = (
            settings["label"] + "_eb"
        )
    else:
        settings["hfw"] = np.clip(settings["hfw"],
                                  microscope.beams.ion_beam.horizontal_field_width.limits.min,
                                  microscope.beams.ion_beam.horizontal_field_width.limits.max,
                                  )
        microscope.beams.ion_beam.horizontal_field_width.value = settings["hfw"]
        label = (
            settings["label"] + "_ib"
        )

    if settings["autocontrast"]:
        autocontrast(microscope, beam_type=settings["beam_type"])

    image = acquire_image(
        microscope=microscope,
        settings=frame_settings,
        beam_type=settings["beam_type"],
    )

    # apply gamma correction
    if settings["gamma"]["correction"]:

        # gamma parameters
        gamma_settings = GammaSettings(
            enabled=settings["gamma"]["corrections"],
            min_gamma=settings["gamma"]["min_gamma"],
            max_gamma=settings["gamma"]["max_gamma"],
            scale_factor=settings["gamma"]["scale_factor"],
            threshold=settings["gamma"]["threshold"],
        )

        image = gamma_correction(image, gamma_settings)

    if settings["save"]:
        utils.save_image(
            image=image, save_path=settings["save_path"], label=label
        )
    settings = tmp_settings  # reset the settings to original # TODO: this doesnt work, need to reset
    return image

def new_image2(microscope, settings: ImageSettings):

    frame_settings = GrabFrameSettings(resolution=settings.resolution, dwell_time=settings.dwell_time)
    tmp_settings = settings
    
    if settings.beam_type == BeamType.ELECTRON:
        hfw_limits = microscope.beams.electron_beam.horizontal_field_width.limits
        settings.hfw = np.clip(settings.hfw, hfw_limits.min, hfw_limits.max)
        microscope.beams.electron_beam.horizontal_field_width.value = settings.hfw
        label = settings.label + "_eb"
    if settings.beam_type == BeamType.ION:
        hfw_limits = microscope.beams.ion_beam.horizontal_field_width.limits
        settings.hfw = np.clip(settings.hfw, hfw_limits.min, hfw_limits.max)
        microscope.beams.ion_beam.horizontal_field_width.value = settings.hfw
        label = settings.label + "_ib"

    if settings.autocontrast:
        autocontrast(microscope, beam_type=settings.beam_type)

    image = acquire_image(
        microscope=microscope,
        settings=frame_settings,
        beam_type=settings.beam_type,
    )

    # apply gamma correction
    if settings.gamma.enabled:

        # gamma parameters
        image = gamma_correction(image, settings.gamma)

    if settings.save:
        utils.save_image(image=image, save_path=settings.save_path, label=label)
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
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image
