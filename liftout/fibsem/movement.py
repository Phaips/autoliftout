from liftout.fibsem.acquire import *
import numpy as np
from autoscript_sdb_microscope_client.structures import StagePosition, MoveSettings
from autoscript_sdb_microscope_client.enumerations import ManipulatorSavedPosition, ManipulatorCoordinateSystem, CoordinateSystem
import time

pretilt = 27  # TODO: add to protocol


def move_relative(microscope, x=0.0, y=0.0, z=0.0, r=0.0, t=0.0, settings=None):
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
        logging.error("Not under electron microscope, please reposition")
        return
    new_position = StagePosition(x=x, y=y, z=z, r=r, t=t)
    microscope.specimen.stage.relative_move(new_position, settings=settings)
    logging.info(f"Old position: {current_position_x*1e6}, {current_position_y*1e6}")
    logging.info(f"Moving by: {x*1e6}, {y*1e6}")
    logging.info(
        f"New position: {(current_position_x + x)*1e6}, {(current_position_y + y)*1e6}\n"
    )

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

        # do we have a sample image somewhere?
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


def move_to_trenching_angle(microscope, settings):
    """Tilt the sample stage to the correct angle for milling trenches.
    Assumes trenches should be milled with the sample surface flat to ion beam.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    settings: dict
        settings dictionary
    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    flat_to_beam(
        microscope,
        settings=settings,
        beam_type=BeamType.ION,
    )
    return microscope.specimen.stage.current_position


def move_to_liftout_angle(
    microscope, settings, liftout_angle=10
):
    """Tilt the sample stage to the correct angle for liftout."""
    flat_to_beam(
        microscope,
        settings=settings,
        beam_type=BeamType.ELECTRON,
    )
    # microscope.specimen.stage.relative_move(StagePosition(t=np.deg2rad(liftout_angle))) # TODO: REMOVE?
    logging.info(f"movement: move to liftout angle ({liftout_angle} deg) complete.")
    return microscope.specimen.stage.current_position


def move_to_landing_angle(microscope, settings, landing_angle=13):
    """Tilt the sample stage to the correct angle for the landing posts."""
    flat_to_beam(
        microscope,
        settings=settings,
        beam_type=BeamType.ION,
    )  # stage tilt 25
    microscope.specimen.stage.relative_move(
        StagePosition(t=np.deg2rad(landing_angle)) # TODO: MAGIC_NUMBER
    )  # more tilt by 13
    logging.info(f"movement: move to landing angle ({landing_angle} deg) complete.")
    return microscope.specimen.stage.current_position


def move_to_jcut_angle(microscope, settings, jcut_angle=6.0):
    """Tilt the sample to the Jcut angle.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    jcut_angle : float, optional
        Tilt angle for the stage when milling the J-cut, in degrees
    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    flat_to_beam(
        microscope,
        settings=settings,
        beam_type=BeamType.ELECTRON,
    )
    microscope.specimen.stage.relative_move(StagePosition(t=np.deg2rad(jcut_angle)))
    logging.info(f"movement: move to j-cut angle ({jcut_angle} deg) complete.")
    return microscope.specimen.stage.current_position


def move_to_sample_grid(microscope, settings):
    """Move stage and zoom out to see the whole sample grid.
    Assumes sample grid is mounted on the left hand side of the holder.
    """
    # TODO: reorder this function so that the movement is safe, and the tilt happens last

    sample_grid_center = StagePosition(
        x=float(settings["initial_position"]["sample_grid"]["x"]),
        y=float(settings["initial_position"]["sample_grid"]["y"]),
        z=float(settings["initial_position"]["sample_grid"]["z"]),
        r=np.deg2rad(float(settings["system"]["stage_rotation_flat_to_electron"])),
        coordinate_system=settings["initial_position"]["sample_grid"]["coordinate_system"]
    )
    logging.info(f"movement: moving to sample grid {sample_grid_center}")
    # microscope.specimen.stage.absolute_move(sample_grid_center)
    safe_absolute_stage_movement(microscope=microscope, stage_position=sample_grid_center)

    # move flat to the electron beam
    flat_to_beam(
        microscope,
        settings=settings,
        beam_type=BeamType.ELECTRON,
    )

    # Zoom out so you can see the whole sample grid
    microscope.beams.ion_beam.horizontal_field_width.value = 100e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 100e-6
    logging.info(f"movement: move to sample grid complete.")
    return microscope.specimen.stage.current_position


def move_to_landing_grid(
    microscope, settings, flat_to_sem=True
):
    """Move stage and zoom out to see the whole landing post grid.
    Assumes the landing grid is mounted on the right hand side of the holder.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    flat_to_sem : bool, optional
        Whether to keep the landing post grid surface flat to the SEM.
    """

    # initially tilt flat for safety

    # move to landing grid initial position
    landing_grid_position = StagePosition(
        x=float(settings["initial_position"]["landing_grid"]["x"]),
        y=float(settings["initial_position"]["landing_grid"]["y"]),
        z=float(settings["initial_position"]["landing_grid"]["z"]),
        r=np.deg2rad(float(settings["system"]["stage_rotation_flat_to_electron"])),
        coordinate_system=settings["initial_position"]["landing_grid"]["coordinate_system"]  # TODO: raw coordinates
    )
    logging.info(f"movement: moving to landing grid {landing_grid_position}")
    # microscope.specimen.stage.absolute_move(landing_grid_position)
    safe_absolute_stage_movement(microscope=microscope, stage_position=landing_grid_position)

    if flat_to_sem:
        flat_to_beam(microscope, settings=settings, beam_type=BeamType.ELECTRON)

    else:
        move_to_landing_angle(
            microscope, settings=settings)

    # Zoom out so you can see the whole landing grid
    microscope.beams.ion_beam.horizontal_field_width.value = 100e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 100e-6
    logging.info(f"movement: move to landing grid complete.")
    return microscope.specimen.stage.current_position


def move_sample_stage_out(microscope):
    """Move stage completely out of the way, so it is not visible at all."""
    # Must set tilt to zero, so we don't see reflections from metal stage base
    microscope.specimen.stage.absolute_move(StagePosition(t=0))  # important!
    sample_stage_out = StagePosition(
        x=-0.002507, y=0.025962792, z=0.0039559049
    )  # TODO: make these dynamically set based on initial_position
    logging.info(f"movement: move sample grid out to {sample_stage_out}")
    microscope.specimen.stage.absolute_move(sample_stage_out)
    logging.info(f"movement: move sample stage out complete.")
    return microscope.specimen.stage.current_position


def move_needle_to_liftout_position(microscope):
    """Move the needle into position, ready for liftout."""
    park_position = insert_needle(microscope)

    move_needle_closer(microscope)
    multichem = microscope.gas.get_multichem()
    multichem.insert()
    return park_position


def move_needle_to_landing_position(microscope):
    """Move the needle into position, ready for landing."""
    park_position = insert_needle(microscope)
    move_needle_closer(microscope, x_shift=-25e-6)
    return park_position


def insert_needle(microscope):
    """Insert the needle and return the needle parking position.
    Returns
    -------
    park_position : autoscript_sdb_microscope_client.structures.ManipulatorPosition
        The parking position for the needle manipulator when inserted.
    """
    needle = microscope.specimen.manipulator
    logging.info(f"movement: inserting needle to park position.")
    park_position = needle.get_saved_position(ManipulatorSavedPosition.PARK, ManipulatorCoordinateSystem.RAW)
    needle.insert(park_position)
    park_position = needle.current_position
    logging.info(f"movement: inserted needle to {park_position}.")
    return park_position


def move_needle_closer(microscope, *, x_shift=-20e-6, z_shift=-160e-6):
    """Move the needle closer to the sample surface, after inserting.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.sdb_microscope.SdbMicroscopeClient
        The Autoscript microscope object.
    x_shift : float
        Distance to move the needle from the parking position in x, in meters.
    z_shift : float
        Distance to move the needle towards the sample in z, in meters.
        Negative values move the needle TOWARDS the sample surface.
    """
    needle = microscope.specimen.manipulator
    stage = microscope.specimen.stage
    # Needle starts from the parking position (after inserting it)
    # Move the needle back a bit in x, so the needle is not overlapping target
    x_move = x_corrected_needle_movement(x_shift)
    logging.info(f"movement: moving needle by {x_move}")
    needle.relative_move(x_move)
    # Then move the needle towards the sample surface.
    z_move = z_corrected_needle_movement(z_shift, stage.current_position.t)
    logging.info(f"movement: moving needle by {z_move}")
    needle.relative_move(z_move)
    # The park position is always the same,
    # so the needletip will end up about 20 microns from the surface.
    logging.info(f"movement: move needle closer complete.")
    return needle.current_position


def x_corrected_needle_movement(expected_x, stage_tilt=None):
    """Needle movement in X, XTGui coordinates (Electron coordinate).
    Parameters
    ----------
    expected_x : float
        in meters
    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    return ManipulatorPosition(x=expected_x, y=0, z=0)  # no adjustment needed


def y_corrected_needle_movement(expected_y, stage_tilt):
    """Needle movement in Y, XTGui coordinates (Electron coordinate).
    Parameters
    ----------
    expected_y : in meters
    stage_tilt : in radians
    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    y_move = +np.cos(stage_tilt) * expected_y
    z_move = +np.sin(stage_tilt) * expected_y
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def z_corrected_needle_movement(expected_z, stage_tilt):
    """Needle movement in Z, XTGui coordinates (Electron coordinate).
    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in radians
    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    y_move = -np.sin(stage_tilt) * expected_z
    z_move = +np.cos(stage_tilt) * expected_z
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def retract_needle(microscope, park_position):
    """Retract the needle and multichem, preserving the correct park position.
    park_position : autoscript_sdb_microscope_client.structures.ManipulatorPosition
        The parking position for the needle manipulator when inserted.
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    # Retract the multichem
    logging.info(f"movement: retracting multichem")
    multichem = microscope.gas.get_multichem()
    multichem.retract()
    # Retract the needle, preserving the correct parking postiion
    needle = microscope.specimen.manipulator
    current_position = needle.current_position
    # To prevent collisions with the sample; first retract in z, then y, then x
    logging.info(f"movement: retracting needle to {park_position}")
    needle.relative_move(
        ManipulatorPosition(z=park_position.z - current_position.z)
    )  # noqa: E501
    needle.relative_move(
        ManipulatorPosition(y=park_position.y - current_position.y)
    )  # noqa: E501
    needle.relative_move(
        ManipulatorPosition(x=park_position.x - current_position.x)
    )  # noqa: E501
    time.sleep(1)  # AutoScript sometimes throws errors if you retract too quick?
    logging.info(f"movement: retracting needle")
    needle.retract()
    retracted_position = needle.current_position
    logging.info(f"movement: retract needle complete")
    return retracted_position


def flat_to_beam(
    microscope, settings, beam_type=BeamType.ELECTRON
):
    """Make the sample surface flat to the electron or ion beam."""

    stage = microscope.specimen.stage
    pretilt_angle = settings["system"]["pretilt_angle"]  # 27

    if beam_type is BeamType.ELECTRON:
        rotation = settings["system"]["stage_rotation_flat_to_electron"]
        tilt = np.deg2rad(pretilt_angle)
    if beam_type is BeamType.ION:
        rotation = settings["system"]["stage_rotation_flat_to_ion"]
        tilt = np.deg2rad(settings["system"]["stage_tilt_flat_to_ion"] - pretilt_angle)  # MAGIC_NUMBER
    rotation = np.deg2rad(rotation)
    stage_settings = MoveSettings(rotate_compucentric=True)
    logging.info(f"movement: moving flat to {beam_type.name}")

    # If we rotating by a lot, tilt to zero so stage doesn't hit anything
    if abs(np.rad2deg(rotation - stage.current_position.r)) > 90:
        stage.absolute_move(StagePosition(t=0), stage_settings)  # just in case
        logging.info(f"movement: tilting to flat for large rotation.")
    logging.info(f"movement: rotating stage to {rotation:.4f}")
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    logging.info(f"movement: tilting stage to {tilt:.4f}")
    stage.absolute_move(StagePosition(t=tilt), stage_settings)

    return stage.current_position


def safe_absolute_stage_movement(microscope, stage_position: StagePosition):
    """Move the stage to the desired position in a safe manner, using compucentric rotation.
        Supports movements in the stage_position coordinate system
    """

    stage = microscope.specimen.stage
    stage_settings = MoveSettings(rotate_compucentric=True)
    # stage_position.coordinate_system = CoordinateSystem.Raw

    # tilt flat for large rotations to prevent collisions
    if abs(np.rad2deg(stage_position.r - stage.current_position.r)) > 90:
        stage.absolute_move(StagePosition(t=np.deg2rad(0), coordinate_system=stage_position.coordinate_system), stage_settings)
        logging.info(f"tilting to flat for large rotation.")
    stage.absolute_move(StagePosition(r=stage_position.r, coordinate_system=stage_position.coordinate_system), stage_settings)
    logging.info(f"safe moving to {stage_position}")
    stage.absolute_move(stage_position, stage_settings)
    logging.info(f"safe movement complete.")
    return stage.current_position


def auto_link_stage(microscope, expected_z=3.9e-3, tolerance=1e-6, hfw=150e-6):
    """Automatically focus and link sample stage z-height.
    expected_z : float, optional
        Correct height for linked stage in z, in meters, by default 4e-3
    tolerance : float, optional
        Must be within this absolute tolerance of expected stage z height,
        in meters, by default 1e-4

    Notes:
        - Focusing determines the working distance (focal distance) of the
        - Relinking is required whenever there is a significant change in vertical distance, i.e. moving
          from the landing grid to the sample grid.
        - Linking determines the specimen coordinate system, as it is defined as the relative dimensions of the top of stage
          to the instruments.
    """
    microscope.imaging.set_active_view(1)
    original_hfw = microscope.beams.electron_beam.horizontal_field_width.value
    microscope.beams.electron_beam.horizontal_field_width.value = hfw
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    microscope.auto_functions.run_auto_focus()
    microscope.specimen.stage.link()
    # TODO: replace with auto_focus_and_link if performance of focus is poor
    # # Restore original settings
    microscope.beams.electron_beam.horizontal_field_width.value = original_hfw

def x_corrected_stage_movement(expected_x, stage_tilt=None, beam_type=None):
    """Stage movement in X.
    ----------
    expected_x : in meters
    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    return StagePosition(x=expected_x, y=0, z=0)


def y_corrected_stage_movement(expected_y, stage_tilt, beam_type=BeamType.ELECTRON):
    """Stage movement in Y, corrected for tilt of sample surface plane.
    ----------
    expected_y : in meters
    stage_tilt : in radians        Can pass this directly microscope.specimen.stage.current_position.t
    beam_type : BeamType, optional
        BeamType.ELECTRON or BeamType.ION
    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    # TODO: add settings, need to read pretilt
    from autoscript_sdb_microscope_client.structures import StagePosition

    assert pretilt == 27  # 27
    if beam_type == BeamType.ELECTRON:
        tilt_adjustment = np.deg2rad(-pretilt)
    elif beam_type == BeamType.ION:
        tilt_adjustment = np.deg2rad(52 - pretilt)  # MAGIC_NUMBER
    tilt_radians = stage_tilt + tilt_adjustment
    y_move = +np.cos(tilt_radians) * expected_y
    z_move = -np.sin(tilt_radians) * expected_y
    logging.info(f"drift correction: the corrected Y shift is {y_move:.3e} meters")
    logging.info(f"drift correction: the corrected Z shift is  {z_move:.3e} meters")
    return StagePosition(x=0, y=y_move, z=z_move)


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


def reset_needle_park_position(microscope, new_park_position):
    """Reset the needle to a safe park position to prevent crashes when inserted.

    This function is required as the insert() api call does not allow us to specify
    an insert position. The needle will return to the previously retracted positions.

    If the programs stops while the need is inserted and near the stage there is a chance
    it will hit the stage when reinserted if we do not do this.

    # TODO: Remove as this is now replaced by AutoScript 4.6

    """
    # # recalibrating needle park position
    # # move sample stage out
    move_sample_stage_out(microscope)

    # move needle in
    insert_needle(microscope)

    # retract needle to park position, then out
    retract_needle(microscope, new_park_position)
    logging.info(f"movement: reset park position to {new_park_position}")
