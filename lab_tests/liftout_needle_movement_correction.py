import numpy as np

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import ManipulatorPosition


def z_corrected(expected_z, stage_tilt):
    """Needle movement in Z, XTGui coordinates (Electron coordinate).

    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in degrees

    Returns
    -------
    ManipulatorPosition
    """
    tilt_radians = np.deg2rad(stage_tilt)
    y_move = -np.sin(tilt_radians) * expected_z
    z_move = +np.cos(tilt_radians) * expected_z
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def y_corrected(expected_y, stage_tilt):
    """Needle movement in Y, XTGui coordinates (Electron coordinate).

    Parameters
    ----------
    expected_y : in meters
    stage_tilt : in degrees

    Returns
    -------
    ManipulatorPosition
    """
    tilt_radians = np.deg2rad(stage_tilt)
    y_move = +np.cos(tilt_radians) * expected_y
    z_move = +np.sin(tilt_radians) * expected_y
    return ManipulatorPosition(x=0, y=y_move, z=z_move)
