"""J-cut milling patterns for liftout sample preparation."""
import numpy as np

from liftout.milling.util import confirm_and_run_milling, setup_ion_milling
from liftout.stage_movement import PRETILT_DEGREES
from liftout.user_input import ask_user


__all__ = [
    "mill_jcut",
    "mill_to_sever_jcut",
    "jcut_milling_patterns",
    "jcut_severing_pattern",
]


def jcut_milling_patterns(microscope,
                          jcut_settings,
                          pretilt_degrees=PRETILT_DEGREES):
    """Create J-cut milling pattern in the center of the ion beam field of view.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    jcut_settings : dict
        Dictionary of J-cut parameter settings.
    pretilt_degrees : int, optional
        Pre-tilt of sample holder in degrees, by default 27

    Returns
    -------
    (autoscript_sdb_microscope_client.structures.RectanglePattern,
     autoscript_sdb_microscope_client.structures.RectanglePattern,
     autoscript_sdb_microscope_client.structures.RectanglePattern)
        Tuple containing the three milling patterns comprising the J-cut.
    """
    jcut_top = None
    jcut_lhs = None
    jcut_rhs = None

    # Unpack settings
    jcut_angle_degrees = jcut_settings['jcut_angle']
    jcut_lamella_depth = jcut_settings['jcut_lamella_depth']
    jcut_length = jcut_settings['jcut_length']
    jcut_trench_thickness = jcut_settings['jcut_trench_thickness']
    jcut_milling_depth = jcut_settings['jcut_milling_depth']
    extra_bit = jcut_settings['extra_bit']

    # Setup
    setup_ion_milling(microscope)
    # Create milling patterns
    angle_correction = np.sin(np.deg2rad(52 - jcut_angle_degrees))
    # Top bar of J-cut
    if bool(jcut_settings['mill_top_jcut_pattern']) is True:
        print('Creating top J-cut pattern')
        jcut_top = microscope.patterning.create_rectangle(
            0.0,                                    # center_x
            jcut_lamella_depth * angle_correction,  # center_y
            jcut_length,                            # width
            jcut_trench_thickness,                  # height
            jcut_milling_depth)                     # depth
    # Left hand side of J-cut (long side)
    if bool(jcut_settings['mill_lhs_jcut_pattern']) is True:
        print('Creating LHS J-cut pattern')
        jcut_lhs = microscope.patterning.create_rectangle(
            -((jcut_length - jcut_trench_thickness) / 2),           # center_x
            ((jcut_lamella_depth - (extra_bit / 2)) / 2) * angle_correction,  # center_y
            jcut_trench_thickness,                                  # width
            (jcut_lamella_depth + extra_bit) * angle_correction,    # height
            jcut_milling_depth)                                     # depth
    # Right hand side of J-cut (short side)
    if bool(jcut_settings['mill_rhs_jcut_pattern']) is True:
        print('Creating RHS J-cut pattern')
        jcut_rightside_remaining = 1.5e-6  # in microns, how much to leave attached
        height = (jcut_lamella_depth - jcut_rightside_remaining) * angle_correction
        center_y = jcut_rightside_remaining + (height / 2)
        jcut_rhs = microscope.patterning.create_rectangle(
            +((jcut_length - jcut_trench_thickness) / 2),  # center_x
            center_y,                                      # center_y
            jcut_trench_thickness,                         # width
            height,                                        # height
            jcut_milling_depth)                            # depth
    if jcut_top is None and jcut_lhs is None and jcut_rhs is None:
        raise RuntimeError('No J-cut patterns created, check your protocol file')
    return jcut_top, jcut_lhs, jcut_rhs


def mill_jcut(microscope, jcut_settings, confirm=True):
    """Create and mill the rectangle patter to sever the jcut completely.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    jcut_settings : dict
        Dictionary of J-cut parameter settings.
    confrim : bool, optional
        Whether to ask the user to confirm before milling.
    """
    jcut_milling_patterns(microscope, jcut_settings)
    confirm_and_run_milling(microscope, jcut_settings['jcut_milling_current'], confirm=confirm)


def jcut_severing_pattern(microscope,
                          jcut_settings,
                          pretilt_degrees=PRETILT_DEGREES):
    """Create J-cut milling pattern in the center of the ion beam field of view.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    jcut_settings : dict
        Sample surface angle for J-cut in degrees, by default 6
    pretilt_degrees : int, optional
        Pre-tilt of sample holder in degrees, by default 27

    Returns
    -------
    autoscript_sdb_microscope_client.structures.RectanglePattern
        Rectangle milling pattern used to sever the remaining bit of the J-cut.
    """
    # Unpack settings
    jcut_angle_degrees = jcut_settings['jcut_angle']
    jcut_lamella_depth = jcut_settings['jcut_lamella_depth']
    jcut_length = jcut_settings['jcut_length']
    jcut_trench_thickness = jcut_settings['jcut_trench_thickness']
    jcut_milling_depth = jcut_settings['jcut_milling_depth']
    extra_bit = jcut_settings['extra_bit']
    # Setup
    setup_ion_milling(microscope)
    # Create milling pattern - right hand side of J-cut
    angle_correction_factor = np.sin(np.deg2rad(52 - jcut_angle_degrees))
    center_x = +((jcut_length - jcut_trench_thickness) / 2)
    center_y = ((jcut_lamella_depth - (extra_bit / 2)) / 2) * angle_correction_factor  # noqa: E501
    width = jcut_trench_thickness
    height =  (jcut_lamella_depth + extra_bit) * angle_correction_factor
    jcut_severing_pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, jcut_milling_depth)
    return jcut_severing_pattern


def mill_to_sever_jcut(microscope, jcut_settings, *, milling_current=0.74e-9,
                       confirm=True):
    """Create and mill the rectangle pattern to sever the jcut completely.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    confirm : bool, optional
        Whether to wait for user confirmation before milling.
    """
    jcut_severing_pattern(microscope, jcut_settings)
    confirm_and_run_milling(microscope, milling_current, confirm=confirm)
