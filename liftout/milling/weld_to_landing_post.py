"""Functions to stick the lamella onto the landing stage post."""
from liftout.milling.util import setup_ion_milling

__all__ = [
    "weld_to_landing_post",
]


def weld_to_landing_post(microscope, *, milling_current=20e-12):
    """Create and mill the sample to the landing post.

    Stick the lamella to the landing post by melting the ice with ion milling.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    """
    pattern = _create_welding_pattern(microscope)
    confirm_and_run_milling(microscope, milling_current)


def _create_welding_pattern(microscope, *,
                            center_x=0,
                            center_y=0,
                            width=3.5e-6,
                            height=5e-6,
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
    setup_ion_milling(microscope)
    pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, depth)
    return pattern
