"""Sample stage movement functions."""
import numpy as np

from .acquire import BeamType, beamtype_from_image, new_electron_image

__all__ = [
    "flat_to_electron_beam",
    "flat_to_ion_beam",
    "move_to_jcut_angle",
    "move_to_trenching_angle",
    "move_to_liftout_angle",
    "move_to_landing_angle",
    "move_to_landing_grid",
    "move_to_sample_grid",
    "move_sample_stage_out",
    "x_corrected_stage_movement",
    "y_corrected_stage_movement",
    "z_corrected_stage_movement",
    ]

PRETILT_DEGREES = 27


def flat_to_electron_beam(microscope, *, pretilt_angle=PRETILT_DEGREES):
    """Make the sample surface flat to the electron beam.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.

    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    from autoscript_sdb_microscope_client.structures import (StagePosition,
                                                             MoveSettings)

    stage = microscope.specimen.stage
    rotation = np.deg2rad(290)
    tilt = np.deg2rad(pretilt_angle)
    stage_settings = MoveSettings(rotate_compucentric=True)
    # If we rotating by a lot, tilt to zero so stage doesn't hit anything
    if abs(np.rad2deg(rotation - stage.current_position.r)) > 90:
        stage.absolute_move(StagePosition(t=0), stage_settings)  # just in case
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position


def flat_to_ion_beam(microscope, *, pretilt_angle=PRETILT_DEGREES):
    """Make the sample surface flat to the ion beam.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.

    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    from autoscript_sdb_microscope_client.structures import (StagePosition,
                                                             MoveSettings)

    stage = microscope.specimen.stage
    rotation = np.deg2rad(290 - 180)
    tilt = np.deg2rad(52 - pretilt_angle)
    stage_settings = MoveSettings(rotate_compucentric=True)
    # If we rotating by a lot, tilt to zero so stage doesn't hit anything
    if abs(np.rad2deg(rotation - stage.current_position.r)) > 90:
        stage.absolute_move(StagePosition(t=0), stage_settings)  # just in case
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position


def move_to_jcut_angle(microscope, *, jcut_angle=6., pretilt_angle=PRETILT_DEGREES):
    """Tilt the sample to the Jcut angle.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    jcut_angle : float, optional
        Tilt angle for the stage when milling the J-cut, in degrees
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.

    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    flat_to_electron_beam(microscope, pretilt_angle=pretilt_angle)
    microscope.specimen.stage.relative_move(StagePosition(t=np.deg2rad(jcut_angle)))
    return microscope.specimen.stage.current_position


def move_to_trenching_angle(microscope, *, pretilt_angle=PRETILT_DEGREES):
    """Tilt the sample stage to the correct angle for milling trenches.

    Assumes trenches should be milled with the sample surface flat to ion beam.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.

    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    flat_to_ion_beam(microscope, pretilt_angle=pretilt_angle)
    return microscope.specimen.stage.current_position


def move_to_liftout_angle(microscope, *, liftout_angle=10, pretilt_angle=PRETILT_DEGREES):
    """Tilt the sample stage to the correct angle for liftout.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    liftout_angle : float, optional
        Tilt angle for the stage for lamella needle liftout, in degrees
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.

    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    flat_to_electron_beam(microscope, pretilt_angle=pretilt_angle)
    microscope.specimen.stage.relative_move(StagePosition(t=np.deg2rad(liftout_angle)))
    return microscope.specimen.stage.current_position


def move_to_landing_angle(microscope, *, landing_angle=18, pretilt_angle=PRETILT_DEGREES):
    """Tilt the sample stage to the correct angle for the landing posts.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    landing_angle : float, optional
        Tilt angle for the stage to orient landing stage posts, in degrees
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.

    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    flat_to_ion_beam(microscope, pretilt_angle=pretilt_angle)
    microscope.specimen.stage.relative_move(StagePosition(t=np.deg2rad(landing_angle)))
    return microscope.specimen.stage.current_position


def move_to_landing_grid(microscope, *, pretilt_angle=PRETILT_DEGREES,
                         flat_to_sem=True):
    """Move stage and zoom out to see the whole landing post grid.

    Assumes the landing grid is mounted on the right hand side of the holder.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.
    flat_to_sem : bool, optional
        Whether to keep the landing post grid surface flat to the SEM.

    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    if flat_to_sem:
        flat_to_electron_beam(microscope)
        landing_grid_position = StagePosition(x=+0.0034580609,
                                              y=+0.0032461667,
                                              z=0.0039338733)
    else:
        move_to_landing_angle(microscope, pretilt_angle=pretilt_angle)
        landing_grid_position = StagePosition(x=-0.0034580609,
                                              y=-0.0032461667,
                                              z=0.0039338733)
    microscope.specimen.stage.absolute_move(landing_grid_position)
    # Zoom out so you can see the whole landing grid
    microscope.beams.ion_beam.horizontal_field_width.value = 100e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 100e-6
    new_electron_image(microscope)
    return microscope.specimen.stage.current_position


def move_to_sample_grid(microscope, *, pretilt_angle=PRETILT_DEGREES):
    """Move stage and zoom out to see the whole sample grid.

    Assumes sample grid is mounted on the left hand side of the holder.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    pretilt_angle : float, optional
        The pre-tilt angle of the sample holder, in degrees.

    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    flat_to_electron_beam(microscope, pretilt_angle=pretilt_angle)
    sample_grid_center = StagePosition(x=-0.0025868173,
                                       y=0.0031794167,
                                       z=0.0039457213)
    microscope.specimen.stage.absolute_move(sample_grid_center)
    # Zoom out so you can see the whole sample grid
    microscope.beams.ion_beam.horizontal_field_width.value = 100e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 100e-6
    new_electron_image(microscope)
    return microscope.specimen.stage.current_position


def move_sample_stage_out(microscope):
    """Move stage completely out of the way, so it is not visible at all.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.

    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    # Must set tilt to zero, so we don't see reflections from metal stage base
    sample_stage_out = StagePosition(x=-0.00060541666,
                                     y=0.014089917,
                                     z=0.0039562348,
                                     t=0)  # zero tilt angle is important
    microscope.specimen.stage.absolute_move(sample_stage_out)
    return microscope.specimen.stage.current_position


def _correct_y_stage_shift(microscope, image, y_shift):
    """Correct y stage movement, relative to the SEM detector.

    Parameters
    ----------
    microscope : Autoscript microscope object
    image : AdornedImage
        The most recent image acquired, that we are aligning with the stage.
    y_shift : float
        The uncorrected y-shift, in meters.

    Returns
    -------
    y_shift_corrected, z_shift_corrected
        Tuple of float values.
    """
    beam_type = beamtype_from_image(image)
    if beam_type == BeamType.ELECTRON:
        angle = PRETILT_DEGREES + microscope.specimen.stage.current_position.t
        y_shift_corrected = np.cos(np.deg2rad(angle)) * y_shift
        z_shift_corrected = np.sin(np.deg2rad(angle)) * y_shift
    elif beam_type == BeamType.ION:
        angle = (52 - PRETILT_DEGREES) + microscope.specimen.stage.current_position.t
        y_shift_corrected = np.cos(np.deg2rad(angle)) * y_shift
        z_shift_corrected = np.sin(np.deg2rad(angle)) * y_shift
    # Fix up sign of z correction stage movement
    if y_shift < 0:
        z_shift_corrected = -z_shift_corrected
    return y_shift_corrected, z_shift_corrected


def x_corrected_stage_movement(expected_x, stage_tilt=None, beam_type=None):
    """Stage movement in X.

    Parameters
    ----------
    expected_x : in meters
    stage_tilt : in radians
    beam_type : BeamType, optional
        BeamType.ELECTRON or BeamType.ION

    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    return StagePosition(x=expected_x, y=0, z=0)


def y_corrected_stage_movement(expected_y, stage_tilt,
                               beam_type=BeamType.ELECTRON):
    """Stage movement in Y, corrected for tilt of sample surface plane.

    Parameters
    ----------
    expected_y : in meters
    stage_tilt : in radians
        Can pass this directly microscope.specimen.stage.current_position.t
    beam_type : BeamType, optional
        BeamType.ELECTRON or BeamType.ION

    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    if beam_type == BeamType.ELECTRON:
        tilt_adjustment =  np.deg2rad(-PRETILT_DEGREES)
    elif beam_type == BeamType.ION:
        tilt_adjustment =  np.deg2rad(52 - PRETILT_DEGREES)
    tilt_radians = stage_tilt + tilt_adjustment
    y_move = +np.cos(tilt_radians) * expected_y
    z_move = -np.sin(tilt_radians) * expected_y
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
    from autoscript_sdb_microscope_client.structures import StagePosition

    y_move = -np.sin(stage_tilt) * expected_z
    z_move = +np.cos(stage_tilt) * expected_z
    return StagePosition(x=0, y=y_move, z=z_move)
