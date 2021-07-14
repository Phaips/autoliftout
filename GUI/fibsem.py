import logging

import numpy as np
"""Module for interacting with the FIBSEM using Autoscript."""


def initialize(ip_address='10.0.0.1'):
    """Initialize connection to FIBSEM microscope with Autoscript."""
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    microscope = SdbMicroscopeClient()
    microscope.connect(ip_address)
    return microscope


def new_image(microscope, settings=None, modality=None):
    """Take new ion or electron beam image.

    Uses whichever camera settings (resolution, dwell time, etc) are current.

    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings : Settings for image acquisition
    modality : Which microscope to acquire an image from.  Possible options:
        - 'SEM' for electron beam
        - 'FIB' for ion beam
    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    view = None
    if modality == 'SEM':
        view = 1
    elif modality == 'FIB':
        view = 2
    if view is not None:
        microscope.imaging.set_active_view(view)
        if settings is not None:
            image = microscope.imaging.grab_frame(settings)
        else:
            image = microscope.imaging.grab_frame()
    else:
        raise ValueError
    return image


def last_image(microscope, modality=None):
    """Get the last previously acquired ion or electron beam image.

    Parameters
    ----------
    microscope : Autoscript microscope object.
    modality : Which microscope to acquire an image from.  Possible options:
        - 'SEM' for electron beam
        - 'FIB' for ion beam

    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """

    view = None
    if modality == 'SEM':
        view = 1
    elif modality == 'FIB':
        view = 2
    if view is not None:
        microscope.imaging.set_active_view(view)  # the ion beam view
        image = microscope.imaging.get_image()
    else:
        raise ValueError
    return image


def move_relative(microscope, x=0.0, y=0.0):
    """Move the sample stage in ion or electron beam view and take new image

    Parameters
    ----------
    microscope : Autoscript microscope object.
    x : float, optional
        Relative movement in x in realspace co-ordinates.
    y : float, optional
        Relative movement in y in realspace co-ordinates.

    Returns
    -------
    StagePosition
        FIBSEM microscope sample stage position after moving.
        If the returned stage position is called 'stage_pos' then:
        stage_pos.x = the x position of the FIBSEM sample stage (in meters)
        stage_pos.y = the y position of the FIBSEM sample stage (in meters)
        stage_pos.z = the z position of the FIBSEM sample stage (in meters)
        stage_pos.r = the rotation of the FIBSEM sample stage (in radians)
        stage_pos.t = the tilt of the FIBSEM sample stage (in radians)
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    current_position_x = microscope.specimen.stage.current_position.x
    current_position_y = microscope.specimen.stage.current_position.y
    if current_position_x > 10e-3 or current_position_x < -10e-3:
        print('Not under electron microscope, please reposition')
        return
    new_position = StagePosition(x=x, y=y, z=0, r=0, t=0)
    microscope.specimen.stage.relative_move(new_position)
    print(f'Old pos ition: {current_position_x*1e6}, {current_position_y*1e6}')
    print(f'Moving by: {x*1e6}, {y*1e6}')
    print(f'New position: {(current_position_x + x)*1e6}, {(current_position_y + y)*1e6}\n')

    return microscope.specimen.stage.current_position


def pixel_to_realspace_coordinate(coord, image):
    """Convert pixel image coordinate to real space coordinate.

    This conversion deliberately ignores the nominal pixel size in y,
    as this can lead to inaccuracies if the sample is not flat in y.

    Parameters
    ----------
    coord : listlike, float
        In x, y format & pixel units. Origin is at the top left.

    image : AdornedImage
        Image the coordinate came from.

    Returns
    -------
    realspace_coord
        xy coordinate in real space. Origin is at the image center.
        Output is in (x, y) format.
    """
    coord = np.array(coord).astype(np.float64)
    if len(image.data.shape) > 2:
        y_shape, x_shape = image.data.shape[0:2]
    else:
        y_shape, x_shape = image.data.shape

    pixelsize_x = image.metadata.binary_result.pixel_size.x
    # deliberately don't use the y pixel size, any tilt will throw this off
    coord[1] = y_shape - coord[1]  # flip y-axis for relative coordinate system
    # reset origin to center
    coord -= np.array([x_shape / 2, y_shape / 2]).astype(np.int32)
    realspace_coord = list(np.array(coord) * pixelsize_x)  # to real space
    return realspace_coord


def autocontrast(microscope):
    """Automatically adjust the microscope image contrast.

    Parameters
    ----------
    microscope : Autoscript microscope object.

    Returns
    -------
    RunAutoCbSettings
        Automatic contrast brightness settings.
    """
    from autoscript_sdb_microscope_client.structures import RunAutoCbSettings
    microscope.imaging.set_active_view(2)
    RunAutoCbSettings(
        method='MaxContrast',
        resolution='768x512',  # low resolution, so as not to damage the sample
        number_of_frames=5,
    )
    logging.info('Automatically adjusting contrast...')
    microscope.auto_functions.run_auto_cb()
    image = last_image(microscope, modality='FIB')
    return image


def update_camera_settings(camera_dwell_time, image_resolution):
    """Create new FIBSEM camera settings using Austoscript GrabFrameSettings.

    Parameters
    ----------
    camera_dwell_time : float
        Image acquisition dwell time in seconds.
    image_resolution : str
        String describing image resolution. Format is pixel width by height.
        Common values include:
            '1536x1024'
            '3072x2048'
            '6144x4096'
            '768x512'
        The full list of available values may differ between instruments.
        See microscope.beams.ion_beam.scanning.resolution.available_values

    Returns
    -------
    camera_settings
        AutoScript GrabFrameSettings object instance.
    """
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings
    camera_settings = GrabFrameSettings(
        resolution=image_resolution,
        dwell_time=camera_dwell_time
    )
    return camera_settings
