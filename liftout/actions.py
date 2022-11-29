import logging
import time

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import (
    ManipulatorCoordinateSystem, ManipulatorSavedPosition)
from autoscript_sdb_microscope_client.structures import (ManipulatorPosition,
                                                         MoveSettings,
                                                         StagePosition)
from fibsem import calibration, movement
from fibsem.structures import BeamType, MicroscopeSettings, MicroscopeState


def move_to_trenching_angle(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings
) -> None:
    """Tilt the sample stage to the correct angle for milling trenches.
    Assumes trenches should be milled with the sample surface flat to ion beam.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        settings (MicroscopeSettings): microscope settings
    """
    movement.move_flat_to_beam(
        microscope,
        settings=settings,
        beam_type=BeamType.ION,
    )


def move_to_liftout_angle(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings
) -> None:
    """Tilt the sample stage to the correct angle for liftout.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        settings (MicroscopeSettings): microscope settings
    """

    movement.move_flat_to_beam(
        microscope,
        settings=settings,
        beam_type=BeamType.ELECTRON,
    )


def move_to_landing_angle(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings
) -> None:
    """Tilt the sample stage to the correct angle for the landing posts.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        settings (MicroscopeSettings): microscope settings
    """

    movement.move_flat_to_beam(
        microscope,
        settings=settings,
        beam_type=BeamType.ION,
    )

    # landing angle
    landing_tilt_angle = np.deg2rad(settings.protocol["initial_position"]["landing_tilt_angle"])
    landing_rotation_angle = np.deg2rad(settings.system.stage.rotation_flat_to_ion)
    landing_position = StagePosition(r=landing_rotation_angle, t=landing_tilt_angle, coordinate_system="Raw")
    movement.safe_absolute_stage_movement(microscope, landing_position)



# TODO: change this to use set_microscope_state?
def move_to_sample_grid(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, protocol: dict
) -> None:
    """Move stage and zoom out to see the whole sample grid.
    Assumes sample grid is mounted on the left hand side of the holder.
    """

    sample_grid_center = StagePosition(
        x=float(protocol["initial_position"]["sample_grid"]["x"]),
        y=float(protocol["initial_position"]["sample_grid"]["y"]),
        z=float(protocol["initial_position"]["sample_grid"]["z"]),
        r=np.deg2rad(float(settings.system.stage.rotation_flat_to_electron)),
        coordinate_system=protocol["initial_position"]["sample_grid"][
            "coordinate_system"
        ],
    )
    movement.safe_absolute_stage_movement(
        microscope=microscope, stage_position=sample_grid_center
    )

    # move flat to the electron beam
    movement.move_flat_to_beam(
        microscope,
        settings=settings,
        beam_type=BeamType.ELECTRON,
    )
    logging.info(f"move to sample grid complete.")


# TODO: get the initial sample/landing grid state some how....
def move_to_sample_grid_v2(microscope: SdbMicroscopeClient, microscope_state: MicroscopeState) -> None:
    """Restore to the initial sample grid state 

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope state
        microscope_state (MicroscopeState): sample grid microscope state
    """
    calibration.set_microscope_state(microscope, microscope_state)

def move_to_landing_grid_v2(microscope: SdbMicroscopeClient, microscope_state: MicroscopeState) -> None:
    """Restore to the initial landing grid state 

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        microscope_state (MicroscopeState): landing grid microscope state
    """
    calibration.set_microscope_state(microscope, microscope_state)

def move_to_landing_grid(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, protocol: dict
) -> StagePosition:
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
        x=float(protocol["initial_position"]["landing_grid"]["x"]),
        y=float(protocol["initial_position"]["landing_grid"]["y"]),
        z=float(protocol["initial_position"]["landing_grid"]["z"]),
        r=np.deg2rad(settings.system.stage.rotation_flat_to_electron), # TODO: fix to ib position
        coordinate_system=protocol["initial_position"]["landing_grid"][
            "coordinate_system"
        ],
    )
    logging.info(f"moving to landing grid {landing_grid_position}")
    movement.safe_absolute_stage_movement(
        microscope=microscope, stage_position=landing_grid_position
    )

    # move to landing angle
    move_to_landing_angle(microscope, settings=settings)

    logging.info(f"move to landing grid complete.")


def move_sample_stage_out(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings
) -> None:
    """Move stage completely out of the way, so it is not visible at all.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        settings (MicroscopeSettings): microscope settings
    """

    # Must set tilt to zero, so we don't see reflections from metal stage base
    microscope.specimen.stage.absolute_move(StagePosition(t=0))  # important!
    sample_stage_out = StagePosition(
        x=-0.002507,
        y=0.025962792,
        z=0.0039559049,
        r=np.deg2rad(settings.system.stage.rotation_flat_to_electron),
    )

    # TODO: probably good enought to just move down a fair bit.
    # TODO: make these dynamically set based on initial_position
    # TODO: MAGIC_NUMBER
    logging.info(f"move sample grid out to {sample_stage_out}")
    movement.safe_absolute_stage_movement(microscope, sample_stage_out)
    logging.info(f"move sample stage out complete.")


def move_needle_to_liftout_position(
    microscope: SdbMicroscopeClient,
    position: ManipulatorPosition = None,
    dx: float = -25.0e-6,
    dy: float = 0.0e-6,
    dz: float = 10.0e-6,
) -> None:
    """Insert the needle to just above the eucentric point, ready for liftout.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        dz (float): distance to move above the eucentric point (ManipulatorCoordinateSystem.RAW -> up = negative)
    """

    # insert to park position
    movement.insert_needle(microscope, ManipulatorSavedPosition.PARK)

    # move to  offset position
    movement.move_needle_to_position_offset(microscope, position, dx, dy, dz)


def move_needle_to_landing_position(
    microscope: SdbMicroscopeClient,
    position: ManipulatorPosition = None,
    dx: float = -30.0e-6,
    dy: float = 0.0e-6,
    dz: float = 15.0e-6,
) -> None:
    """Insert the needle to just above, and left of the eucentric point, ready for land
    .+ing.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        dz (float): distance to move above the eucentric point (ManipulatorCoordinateSystem.RAW -> up = negative)

    Returns:
        ManipulatorPosition: current needle position
    """

    # insert to park position
    movement.insert_needle(microscope, ManipulatorSavedPosition.PARK)

    # move to  offset position #TODO: add this back, but offset from PARK
    # movement.move_needle_to_position_offset(microscope, position, dx, dy, dz) 

    return microscope.specimen.manipulator.current_position


def move_needle_to_reset_position(microscope: SdbMicroscopeClient, position: ManipulatorPosition = None) -> None:
    """Move the needle into position, ready for reset"""

    # insert to park
    movement.insert_needle(microscope, ManipulatorSavedPosition.PARK)

    # move to eucentric
    movement.move_needle_to_position_offset(microscope, position)

def move_to_thinning_angle(
    microscope: SdbMicroscopeClient, protocol: dict
) -> StagePosition:
    """Rotate and tilt the stage to the thinning angle, assumes from the landing position"""

    # thinning position
    thinning_rotation_angle = np.deg2rad(protocol["thin_lamella"]["rotation_angle"])
    thinning_tilt_angle = np.deg2rad(protocol["thin_lamella"]["tilt_angle"])

    stage_position = StagePosition(r=thinning_rotation_angle, t=thinning_tilt_angle)
    movement.safe_absolute_stage_movement(microscope, stage_position)

