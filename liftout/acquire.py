import logging
import os

__all__ = [
    "autocontrast",
    "autofocus",
    "autocontrast_autofocus",
    "create_camera_settings",
    "new_electron_image",
    "new_ion_image",
]


def autocontrast(microscope):
    """Atuomatically adjust the microscope image contrast.

    Parameters
    ----------
    microscope : Autoscript microscope object.

    Returns
    -------
    RunAutoCbSettings
        Automatic contrast brightness settings.
    """
    from autoscript_sdb_microscope_client.structures import RunAutoCbSettings

    autocontrast_settings = RunAutoCbSettings(
        method="MaxContrast",
        resolution="768x512",  # low resolution, so as not to damage the sample
        number_of_frames=5,
    )
    logging.info("Automatically adjusting contrast...")
    microscope.auto_functions.run_auto_cb()
    return autocontrast_settings


def autofocus(microscope, reduced_area_coords=(0, 0, 1, 1)):
    """Run autofocus function."""
    from autoscript_sdb_microscope_client.structures import (
        Rectangle,
        RunAutoFocusSettings,
    )

    left, top, width, height = reduced_area_coords
    reduced_area = Rectangle(left, top, width, height)
    focus_settings = RunAutoFocusSettings(
        method="Volumescope",
        resolution="1536x1024",
        reduced_area=reduced_area,
        number_of_frames=5,
        working_distance_step=1e-6,
    )
    logging.info("Automatically focusing...")
    microscope.auto_functions.run_auto_focus(focus_settings)
    return focus_settings



def autocontrast_autofocus(
    microscope,
    run_autocontrast=False,
    run_autofocus=False,
    reduced_area_focus_coords=(0, 0, 1, 1),
):
    """Optionally run autocontrast and autofocus functions."""
    if run_autocontrast:
        autocontrast(microscope)
    if run_autofocus:
        autofocus(microscope, reduced_area_focus_coords)


def _reduced_area_rectangle(reduced_area_coords):
    assert len(reduced_area_coords) == 4
    top_corner_x, top_corner_y, width, height = reduced_area_coords
    return Rectangle(top_corner_x, top_corner_y, width, height)


def create_camera_settings(imaging_settings, reduced_area_coorda=[0, 0, 1, 1]):
    """Camera settings for acquiring images on the microscope.

    Parameters
    ----------
    imaging_settings : dictionary
        User input as dictionary containing keys "resolution" and "dwell_time".
    reduced_area_coords : Rectangle, optional
        Reduced area view for image acquisition.
        By default None, which will create a Rectangle(0, 0, 1, 1),
        which means the whole field of view will be imaged.

    Returns
    -------
    GrabFrameSettings
        Camera acquisition settings
    """
    from autoscript_sdb_microscope_client.structures import (GrabFrameSettings,
                                                             Rectangle)

    reduced_area = _reduced_area_rectangle(reduced_area_coords)
    camera_settings = GrabFrameSettings(
        resolution=imaging_settings["resolution"],
        dwell_time=imaging_settings["dwell_time"],
        reduced_area=reduced_area,
    )
    return camera_settings


def new_electron_image(microscope, settings=None):
    """Take new electron beam image.

    Uses whichever camera settings (resolution, dwell time, etc) are current.

    Parameters
    ----------
    microscope : Autoscript microscope object.

    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    microscope.imaging.set_active_view(1)  # the electron beam view
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image



def new_ion_image(microscope, settings=None):
    """Take new ion beam image.

    Uses whichever camera settings (resolution, dwell time, etc) are current.

    Parameters
    ----------
    microscope : Autoscript microscope object.

    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    microscope.imaging.set_active_view(2)  # the ion beam view
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image
