from autoscript_sdb_microscope_client.structures import *
from enum import Enum
import logging
from liftout import utils


class BeamType(Enum):
    ELECTRON = 1
    ION = 2


def autocontrast(microscope, beam_type=BeamType.ELECTRON):
    """Automatically adjust the microscope image contrast.
    """
    microscope.imaging.set_active_view(beam_type.value)

    RunAutoCbSettings(
        method="MaxContrast",
        resolution="768x512",  # low resolution, so as not to damage the sample
        number_of_frames=5,
    )
    logging.info("Automatically adjusting contrast...")
    microscope.auto_functions.run_auto_cb()


def new_image(microscope, settings):
    frame_settings  = GrabFrameSettings(resolution=settings['resolution'],
                                        dwell_time=settings['dwell_time'])

    if settings['beam_type'] == BeamType.ELECTRON:
        microscope.beams.electron_beam.horizontal_field_width.value = settings['hfw']
    else:
        microscope.beams.ion_beam.horizontal_field_width.value = settings['hfw']

    if settings['autocontrast']:
        autocontrast(microscope, beam_type=settings['beam_type'])
        settings['contrast'] = None
        settings['brightness'] = None

    image = acquire_image(microscope=microscope,
                          settings=frame_settings,
                          brightness=settings['brightness'],
                          contrast=settings['contrast'])

    if settings['save']:
        utils.save_image(image=image, save_path=settings['save_path'],
                         label=settings['label'])

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


def acquire_image(microscope, settings=None, brightness=None, contrast=None, beam_type=BeamType.ELECTRON, save_path='', id=''):
    """Take new electron or ion beam image.
    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
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


# def update_image_settings(settings):
