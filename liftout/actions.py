import logging
import time

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (ManipulatorPosition,
                                                         MoveSettings,
                                                         StagePosition)
from fibsem import movement
from fibsem.acquire import BeamType


def move_to_trenching_angle(
    microscope: SdbMicroscopeClient, settings: dict
) -> StagePosition:
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
    movement.flat_to_beam(
        microscope, settings=settings, beam_type=BeamType.ION,
    )
    return microscope.specimen.stage.current_position

def move_to_liftout_angle(
    microscope: SdbMicroscopeClient, settings: dict
) -> StagePosition:
    """Tilt the sample stage to the correct angle for liftout."""
    movement.flat_to_beam(
        microscope, settings=settings, beam_type=BeamType.ELECTRON,
    )
    logging.info(f"move to liftout angle complete.")
    return microscope.specimen.stage.current_position

def move_to_landing_angle(
    microscope: SdbMicroscopeClient, settings: dict
) -> StagePosition:
    """Tilt the sample stage to the correct angle for the landing posts."""

    landing_angle = np.deg2rad(settings["system"]["stage_tilt_landing"])
    movement.flat_to_beam(
        microscope, settings=settings, beam_type=BeamType.ION,
    )  # stage tilt 25
    microscope.specimen.stage.relative_move(
        StagePosition(t=landing_angle)
    )  # more tilt by 13
    logging.info(
        f"movement: move to landing angle ({np.rad2deg(landing_angle)} deg) complete."
    )
    return microscope.specimen.stage.current_position

# TODO: change this to use set_microscope_state? 
def move_to_sample_grid(
    microscope: SdbMicroscopeClient, settings: dict
) -> StagePosition:
    """Move stage and zoom out to see the whole sample grid.
    Assumes sample grid is mounted on the left hand side of the holder.
    """

    sample_grid_center = StagePosition(
        x=float(settings["system"]["initial_position"]["sample_grid"]["x"]),
        y=float(settings["system"]["initial_position"]["sample_grid"]["y"]),
        z=float(settings["system"]["initial_position"]["sample_grid"]["z"]),
        r=np.deg2rad(float(settings["system"]["stage_rotation_flat_to_electron"])),
        coordinate_system=settings["system"]["initial_position"]["sample_grid"][
            "coordinate_system"
        ],
    )
    logging.info(f"movement: moving to sample grid {sample_grid_center}")
    movement.safe_absolute_stage_movement(
        microscope=microscope, stage_position=sample_grid_center
    )

    # move flat to the electron beam
    movement.flat_to_beam(
        microscope, settings=settings, beam_type=BeamType.ELECTRON,
    )
    logging.info(f"move to sample grid complete.")
    return microscope.specimen.stage.current_position

def move_to_landing_grid(
    microscope: SdbMicroscopeClient, settings: dict) -> StagePosition:
    """Move stage to landing post grid.
    Assumes the landing grid is mounted on the right hand side of the holder.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    flat_to_sem : bool, optional
        Whether to keep the landing post grid surface flat to the SEM.
    """

    # move to landing grid initial position
    landing_grid_position = StagePosition(
        x=float(settings["system"]["initial_position"]["landing_grid"]["x"]),
        y=float(settings["system"]["initial_position"]["landing_grid"]["y"]),
        z=float(settings["system"]["initial_position"]["landing_grid"]["z"]),
        r=np.deg2rad(float(settings["system"]["stage_rotation_flat_to_electron"])),
        coordinate_system=settings["system"]["initial_position"]["landing_grid"][
            "coordinate_system"
        ],
    )
    logging.info(f"movement: moving to landing grid {landing_grid_position}")
    movement.safe_absolute_stage_movement(
        microscope=microscope, stage_position=landing_grid_position
    )

    # move to landing angle
    move_to_landing_angle(microscope, settings=settings)

    logging.info(f"movement: move to landing grid complete.")
    return microscope.specimen.stage.current_position

def move_sample_stage_out(microscope: SdbMicroscopeClient) -> StagePosition:
    """Move stage completely out of the way, so it is not visible at all."""
    # Must set tilt to zero, so we don't see reflections from metal stage base
    microscope.specimen.stage.absolute_move(StagePosition(t=0))  # important!
    sample_stage_out = StagePosition(
        x=-0.002507, y=0.025962792, z=0.0039559049
    )  # TODO: make these dynamically set based on initial_position
    logging.info(f"movement: move sample grid out to {sample_stage_out}")
    movement.safe_absolute_stage_movement(microscope, sample_stage_out)
    logging.info(f"movement: move sample stage out complete.")
    return microscope.specimen.stage.current_position


def move_needle_to_liftout_position(
    microscope: SdbMicroscopeClient,
) -> ManipulatorPosition:
    """Move the needle into position, ready for liftout."""
    park_position = movement.insert_needle(microscope)
    movement.move_needle_closer(microscope)
    return park_position


def move_needle_to_landing_position(
    microscope: SdbMicroscopeClient,
) -> ManipulatorPosition:
    """Move the needle into position, ready for landing."""
    park_position = movement.insert_needle(microscope)
    movement.move_needle_closer(microscope, x_shift=-25e-6)
    return park_position

def move_needle_to_reset_position(microscope: SdbMicroscopeClient):

    movement.insert_needle(microscope)
    z_move_in = movement.z_corrected_needle_movement(-180e-6, microscope.specimen.stage.current_position.t)
    microscope.specimen.manipulator.relative_move(z_move_in)
    logging.info(f"insert needle for reset")

# TODO: use safe_absolute_stage_movement instead
def move_to_thinning_angle(
    microscope: SdbMicroscopeClient, settings: dict
) -> StagePosition:
    """ Rotate and tilt the stage to the thinning angle, assumes from the landing position"""
    stage = microscope.specimen.stage

    # tilt to zero for safety
    stage_settings = MoveSettings(rotate_compucentric=True, tilt_compucentric=True)
    stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)

    # thinning position
    thinning_rotation_angle = np.deg2rad(settings["thin_lamella"]["rotation_angle"])
    thinning_tilt_angle = np.deg2rad(settings["thin_lamella"]["tilt_angle"])

    # rotate to thinning angle
    logging.info(f"rotate to thinning angle: {thinning_rotation_angle}")
    stage.absolute_move(StagePosition(r=thinning_rotation_angle), stage_settings)

    # tilt to thinning angle
    logging.info(f"tilt to thinning angle: {thinning_tilt_angle}")
    stage.absolute_move(StagePosition(t=thinning_tilt_angle), stage_settings)

    return stage.current_position
