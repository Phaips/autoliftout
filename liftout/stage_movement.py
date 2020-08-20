"""Sample stage movement functions."""
import numpy as np

__all__ = [
    "flat_to_electron_beam",
    "flat_to_ion_beam",
    "move_to_jcut_position",
    "move_to_trenching_position",
    "move_to_liftout_position",
    "move_to_landing_position",
    "move_to_grid",
    ]

PRETILT_DEGREES = 27


def flat_to_electron_beam(stage, *, pretilt_angle=PRETILT_DEGREES):
    """Make the sample surface flat to the electron beam."""
    from autoscript_sdb_microscope_client.structures import StagePosition

    rotation = np.deg2rad(290)
    tilt = np.deg2rad(pretilt_angle)
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position


def flat_to_ion_beam(stage, *, pretilt_angle=PRETILT_DEGREES):
    """Make the sample surface flat to the ion beam."""
    from autoscript_sdb_microscope_client.structures import StagePosition

    rotation = np.deg2rad(290 - 180)
    tilt = np.deg2rad(52 - pretilt_angle)
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position


def move_to_jcut_position(stage, *, jcut_angle=6, pretilt_angle=PRETILT_DEGREES):
    from autoscript_sdb_microscope_client.structures import StagePosition

    flat_to_electron_beam(stage, pretilt_angle=pretilt_angle)
    stage.relative_move(StagePosition(t=np.deg2rad(jcut_angle)))
    return stage.current_position


def move_to_trenching_position(stage, *, pretilt_angle=PRETILT_DEGREES):
    flat_to_ion_beam(stage, pretilt_angle=pretilt_angle)
    return stage.current_position


def move_to_liftout_position(stage, *, pretilt_angle=PRETILT_DEGREES):
    flat_to_ion_beam(stage, pretilt_angle=pretilt_angle)
    return stage.current_position


def move_to_landing_position(stage, *, landing_angle=18, pretilt_angle=PRETILT_DEGREES):
    from autoscript_sdb_microscope_client.structures import StagePosition

    flat_to_ion_beam(stage, pretilt_angle=pretilt_angle)
    stage.relative_move(StagePosition(t=np.deg2rad(landing_angle)))
    return stage.current_position


def move_to_grid(stage):
    raise NotImplementedError
