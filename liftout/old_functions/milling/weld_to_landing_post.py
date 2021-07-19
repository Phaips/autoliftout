"""Functions to stick the lamella onto the landing stage post."""
import numpy as np

from liftout.old_functions.milling import setup_ion_milling, confirm_and_run_milling

__all__ = [
    "weld_to_landing_post",
    "cut_off_needle",
]


def weld_to_landing_post(microscope, *, milling_current=20e-12, confirm=True):
    """Create and mill the sample to the landing post.

    Stick the lamella to the landing post by melting the ice with ion milling.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    confirm : bool, optional
        Whether to wait for user confirmation before milling.
    """
    pattern = _create_welding_pattern(microscope)
    confirm_and_run_milling(microscope, milling_current, confirm=confirm)


def _create_welding_pattern(microscope, *,
                            center_x=0,
                            center_y=0,
                            width=3.5e-6,
                            height=10e-6,
                            depth=5e-9):
    """Create milling pattern for welding liftout sample to the landing post.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    center_x : float
        Center position of the milling pattern along x-axis, in meters.
        Zero coordinate is at the centerpoint of the image field of view.
    center_y : float
        Center position of the milling pattern along x-axis, in meters.
        Zero coordinate is at the centerpoint of the image field of view.
    width : float
        Width of the milling pattern, in meters.
    height: float
        Height of the milling pattern, in meters.
    depth : float
        Depth of the milling pattern, in meters.
    """
    # TODO: user input yaml for welding pattern parameters
    setup_ion_milling(microscope)
    pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, depth)
    return pattern


def cut_off_needle(microscope, *, milling_current=0.74e-9, confirm=True):
    pattern = _create_cutoff_pattern(microscope)
    confirm_and_run_milling(microscope, milling_current, confirm=confirm)


def _create_cutoff_pattern(microscope, *,
                           center_x=-10.5e-6,
                           center_y=-5e-6,
                           width=8e-6,
                           height=2e-6,
                           depth=1e-6,
                           rotation_degrees=40):
    setup_ion_milling(microscope)
    pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, depth)
    pattern.rotation = np.deg2rad(rotation_degrees)
    return pattern
