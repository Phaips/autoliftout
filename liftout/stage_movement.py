"""Sample stage movement functions."""
import numpy as np

from .acquire import BeamType, beamtype_from_image

__all__ = [
    "flat_to_electron_beam",
    "flat_to_ion_beam",
    "move_to_jcut_position",
    "move_to_trenching_position",
    "move_to_liftout_position",
    "move_to_landing_position",
    "move_to_grid",
    "x_corrected_stage_movement",
    "y_corrected_stage_movement",
    "z_corrected_stage_movement",
    ]

PRETILT_DEGREES = 27


def flat_to_electron_beam(stage, *, pretilt_angle=PRETILT_DEGREES):
    """Make the sample surface flat to the electron beam."""
    from autoscript_sdb_microscope_client.structures import (StagePosition,
                                                             MoveSettings)

    rotation = np.deg2rad(290)
    tilt = np.deg2rad(pretilt_angle)
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position


def flat_to_ion_beam(stage, *, pretilt_angle=PRETILT_DEGREES):
    """Make the sample surface flat to the ion beam."""
    from autoscript_sdb_microscope_client.structures import (StagePosition,
                                                             MoveSettings)

    rotation = np.deg2rad(290 - 180)
    tilt = np.deg2rad(52 - pretilt_angle)
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position


def move_to_jcut_position(stage, *, jcut_angle=6, pretilt_angle=PRETILT_DEGREES):
    """Move the sample to the Jcut angle."""
    from autoscript_sdb_microscope_client.structures import StagePosition

    flat_to_electron_beam(stage, pretilt_angle=pretilt_angle)
    stage.relative_move(StagePosition(t=np.deg2rad(jcut_angle)))
    return stage.current_position


def move_to_trenching_position(stage, *, pretilt_angle=PRETILT_DEGREES):
    """Move the sample stage to the correct angle for milling trenches."""
    flat_to_ion_beam(stage, pretilt_angle=pretilt_angle)
    return stage.current_position


def move_to_liftout_position(stage, *, liftout_angle=10, pretilt_angle=PRETILT_DEGREES):
    """Move the sample stage to the correct angle for liftout."""
    from autoscript_sdb_microscope_client.structures import StagePosition

    flat_to_electron_beam(stage, pretilt_angle=pretilt_angle)
    stage.relative_move(StagePosition(t=np.deg2rad(liftout_angle)))
    return stage.current_position


def move_to_landing_position(stage, *, landing_angle=18, pretilt_angle=PRETILT_DEGREES):
    from autoscript_sdb_microscope_client.structures import StagePosition

    flat_to_ion_beam(stage, pretilt_angle=pretilt_angle)
    stage.relative_move(StagePosition(t=np.deg2rad(landing_angle)))
    return stage.current_position


def move_to_grid(stage):
    raise NotImplementedError


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
    stage_tilt : in degrees
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
    stage_tilt : in degrees
    beam_type : BeamType, optional
        BeamType.ELECTRON or BeamType.ION

    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    if beam_type == BeamType.ELECTRON:
        tilt_adjustment = PRETILT_DEGREES
    elif beam_type == BeamType.ION:
        tilt_adjustment = 52 - PRETILT_DEGREES
    tilt_radians = np.deg2rad(stage_tilt + tilt_adjustment)
    y_move = +np.cos(tilt_radians) * expected_y
    z_move = -np.sin(tilt_radians) * expected_y
    return StagePosition(x=0, y=y_move, z=z_move)


def z_corrected_stage_movement(expected_z, stage_tilt,
                               beam_type=BeamType.ELECTRON):
    """Stage movement in Z, corrected for tilt of sample surface plane.

    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in degrees
    beam_type : BeamType, optional
        BeamType.ELECTRON or BeamType.ION

    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    if beam_type == BeamType.ELECTRON:
        tilt_adjustment = PRETILT_DEGREES
    elif beam_type == BeamType.ION:
        tilt_adjustment = 52 - PRETILT_DEGREES
    tilt_radians = np.deg2rad(stage_tilt + tilt_adjustment)
    y_move = -np.sin(tilt_radians) * expected_z
    z_move = +np.cos(tilt_radians) * expected_z
    return StagePosition(x=0, y=y_move, z=z_move)
