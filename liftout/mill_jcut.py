"""J-cut milling for liftout sample preparation."""
import numpy as np

from liftout.stage_movement import PRETILT_DEGREES
from liftout.user_input import ask_user


__all__ = [
    "setup_ion_milling",
    "confirm_and_run_milling",
    "mill_fiducial_marker",
    "mill_trenches",
    "mill_jcut",
    "mill_to_sever_jcut",
    "jcut_milling_patterns",
    "jcut_severing_pattern",
]


def setup_ion_milling(microscope, *,
                      application_file="Si_Alex",
                      patterning_mode="Parallel",
                      ion_beam_field_of_view=82.9e-6):
    """Setup for rectangle ion beam milling patterns.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    application_file : str, optional
        Application file for ion beam milling, by default "Si_Alex"
    patterning_mode : str, optional
        Ion beam milling pattern mode, by default "Parallel".
        The available options are "Parallel" or "Serial".
    ion_beam_field_of_view : float, optional
        Width of ion beam field of view in meters, by default 59.2e-6
    """
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_beam_type(2)  # ion beam default
    microscope.patterning.set_default_application_file(application_file)
    microscope.patterning.mode = patterning_mode
    microscope.patterning.clear_patterns()  # clear any existing patterns
    microscope.beams.ion_beam.horizontal_field_width.value = ion_beam_field_of_view


def confirm_and_run_milling(microscope, milling_current, *,
                            imaging_current=20e-12):
    # TODO: maybe display to the user how long milling will take
    if ask_user("Do you want to run the ion beam milling?"):
        print("Ok, running ion beam milling now...")
        microscope.beams.ion_beam.beam_current.value = milling_current
        microscope.patterning.run()
        microscope.patterning.clear_patterns()
        microscope.beams.ion_beam.beam_current.value = imaging_current
        print("Ion beam milling complete.")


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


def mill_jcut(microscope, jcut_settings):
    """Create and mill the rectangle patter to sever the jcut completely.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    jcut_settings : dict
        Dictionary of J-cut parameter settings.
    """
    jcut_milling_patterns(microscope, jcut_settings)
    confirm_and_run_milling(microscope, jcut_settings['jcut_milling_current'])


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
    angle_correction_factor = np.sin(np.deg2rad(52 - jcut_angle))
    center_x = +((jcut_length - jcut_trench_thickness) / 2)
    center_y = ((jcut_lamella_depth - (extra_bit / 2)) / 2) * angle_correction_factor  # noqa: E501
    width = jcut_trench_thickness
    height =  (jcut_lamella_depth + extra_bit) * angle_correction_factor
    jcut_severing_pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, jcut_milling_depth)
    return jcut_severing_pattern


def mill_to_sever_jcut(microscope, *, milling_current=0.74e-9):
    """Create and mill the rectangle pattern to sever the jcut completely.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    """
    jcut_severing_pattern(microscope)
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
    setup_ion_milling(microscope)
    pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, depth)
    return pattern


def weld_to_landing_post(microscope, *, milling_current=20e-12):
    """Create and mill the sample to the landing post.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    """
    pattern = _create_welding_pattern(microscope)
    confirm_and_run_milling(microscope, milling_current)
