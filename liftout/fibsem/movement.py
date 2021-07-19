from liftout.old_functions.acquire import *
import numpy as np
from autoscript_sdb_microscope_client.structures import (StagePosition,
                                                         MoveSettings)

pretilt = 27


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


def move_to_sample_grid(microscope, settings, *, pretilt_angle=pretilt):
    """Move stage and zoom out to see the whole sample grid.
    Assumes sample grid is mounted on the left hand side of the holder.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    flat_to_beam(microscope, settings, pretilt_angle=pretilt_angle, beam_type=BeamType.ELECTRON)
    sample_grid_center = StagePosition(x=-0.0025868173, y=0.0031794167, z=0.0039457213)
    microscope.specimen.stage.absolute_move(sample_grid_center)
    # Zoom out so you can see the whole sample grid
    microscope.beams.ion_beam.horizontal_field_width.value = 100e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 100e-6
    return microscope.specimen.stage.current_position


def flat_to_beam(microscope, settings, *, pretilt_angle=pretilt, beam_type=BeamType.ELECTRON):
    """Make the sample surface flat to the electron or ion beam.
    """
    stage = microscope.specimen.stage
    if beam_type is BeamType.ELECTRON:
        rotation = settings["system"]["stage_rotation_flat_to_electron"]
        tilt = np.deg2rad(pretilt_angle)
    if beam_type is BeamType.ION:
        rotation = settings["system"]["stage_rotation_flat_to_ion"]
        tilt = np.deg2rad(52 - pretilt_angle)
    rotation = np.deg2rad(rotation)
    stage_settings = MoveSettings(rotate_compucentric=True)
    # If we rotating by a lot, tilt to zero so stage doesn't hit anything
    if abs(np.rad2deg(rotation - stage.current_position.r)) > 90:
        stage.absolute_move(StagePosition(t=0), stage_settings)  # just in case
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position


def auto_link_stage(microscope, expected_z=3.9e-3, tolerance=1e-6):
    """Automatically focus and link sample stage z-height.
    expected_z : float, optional
        Correct height for linked stage in z, in meters, by default 4e-3
    tolerance : float, optional
        Must be within this absolute tolerance of expected stage z height,
        in meters, by default 1e-4
    """
    # SAMPLE GRID expected_z = 3.9e-3
    # LANDING GRID expected_z = 4.05e-3
    # How to auto-link z for the landing posts
    #    1. Make landing grid flat to SEM
    #    2. Zoom really far in on a flat part that isn't part of the posts
    #    3. Auto-link z, using a DIFFERENT expected_z height (4.05 mm)

    microscope.imaging.set_active_view(1)
    original_hfw = microscope.beams.electron_beam.horizontal_field_width.value
    microscope.beams.electron_beam.horizontal_field_width.value = 0.000400
    # TODO: check with Sergey on double autocontrast
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    microscope.auto_functions.run_auto_focus()
    microscope.specimen.stage.link()
    z_difference = expected_z - microscope.specimen.stage.current_position.z
    if abs(z_difference) > 3e-3:
        raise RuntimeError("ERROR: the reported stage position is likely incorrect!")
    z_move = z_corrected_stage_movement(
        z_difference, microscope.specimen.stage.current_position.t)
    microscope.specimen.stage.relative_move(z_move)
    counter = 0
    while not linked_within_z_tolerance(microscope,
                                        expected_z=expected_z,
                                        tolerance=tolerance):
        if counter > 3:
            raise (UserWarning("Could not auto-link z stage height."))
            break
        # Focus and re-link z stage height
        print('Automatically focusing and linking stage z-height.')
        microscope.auto_functions.run_auto_focus()
        microscope.specimen.stage.link()
        z_difference = expected_z - microscope.specimen.stage.current_position.z
        z_move = z_corrected_stage_movement(
            z_difference, microscope.specimen.stage.current_position.t)
        microscope.specimen.stage.relative_move(z_move)
        print(microscope.specimen.stage.current_position.z)
    # Restore original settings
    microscope.beams.electron_beam.horizontal_field_width.value = original_hfw
    new_electron_image(microscope)


def z_corrected_stage_movement(expected_z, stage_tilt):
    """Stage movement in Z, corrected for tilt of sample surface plane.
    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in radians
    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    y_move = -np.sin(stage_tilt) * expected_z
    z_move = +np.cos(stage_tilt) * expected_z
    return StagePosition(x=0, y=y_move, z=z_move)


def linked_within_z_tolerance(microscope, expected_z=3.9e-3, tolerance=1e-6):
    """Check if the sample stage is linked and at the expected z-height.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    expected_z : float, optional
        Correct height for linked stage in z, in meters, by default 4e-3
    tolerance : float, optional
        Must be within this absolute tolerance of expected stage z height,
        in meters, by default 1e-4
    """
    # Check the microscope stage is at the correct height
    z_stage_height = microscope.specimen.stage.current_position.z
    if np.isclose(z_stage_height, expected_z, atol=tolerance):
        return True
    else:
        return False