"""J-cut milling for liftout sample preparation."""
import numpy as np

__all__ [
    "setup_ion_milling",
    "confirm_and_run_milling",
    "mill_fiducial_marker",
    "mill_trenches",
    "mill_jcut",
    "mill_to_sever_jcut",
]


def setup_ion_milling(microscope, *,
                      application_file="Si_Alex",
                      patterning_mode="Parallel",
                      ion_beam_field_of_view=59.2e-6):
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
    if ask_user("Do you want to run the ion beam milling?"):
        print("Ok, running ion beam milling now...")
        microscope.beams.ion_beam.beam_current.value = milling_current
        microscope.patterning.run()
        microscope.beams.ion_beam.beam_current.value = imaging_current
        microscope.patterning.clear_patterns()
        print("Ion beam milling complete.")


def _liftout_fiducial_pattern(microscope, *,
                              fiducial_length=5e-6,
                              fiducial_thickness=0.5e-6):
    """Create milling cross pattern for liftout fiducial marker.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    fiducial_length : float
        The length of both lines in the fiducial cross marker, in meters.
    fiducial_thickness : float
        Thickness of the fiducial marker cross lines, in meters.

    Returns
    -------
    (autoscript_sdb_microscope_client.structures.RectanglePattern,
     autoscript_sdb_microscope_client.structures.RectanglePattern)
        Tuple of the two milling patterns comprising the fiducial marker.
    """
    # Place the fiducial center at the midpoint hight in y,
    # and three quarters of the way across the image along the x-axis.
    fiducial_center_y = 0
    fiducial_center_x = 0.75 * microscope.beams.ion_beam.horizontal_field_width.value
    setup_ion_milling(microscope)
    pattern_1 = microscope.patterning.create_rectangle(
        fiducial_center_x, fiducial_center_y, )
    pattern_2 = microscope.patterning.create_rectangle(
        fiducial_center_x, fiducial_center_y, )
    return pattern_2, pattern_2


def mill_fiducial_marker(microscope, *, milling_current=0.74e-9):
    """Create and mill the fiducial cross shaped marker.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    """
    _liftout_fiducial_pattern(microscope)
    confirm_and_run_milling(microscope, milling_current)


def _trench_milling_patterns(microscope, *,
                             lamella_thickness=2e-6,
                             trench_width=15e-6,
                             trench_height=10e-6,
                             milling_depth=3e-6,
                             lamella_buffer=0.5e-6):
    """Create two cleaning cross sections to cut lamella from the bulk sample.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    lamella_thickness : float, optional
        The intended lamella thickness after cutting trenches, in meters.
    trench_width : float, optional
        The width of the milling trench patterns, in meters.
    trench_height : float, optional
        The height of the milling trench patterns, in meters.
    milling_depth : float, optional
        The milling pattern depth for the trenches, in meters.
    lamella_buffer : float, optional
        The extra spacing to allow around the lamella, in meters.

    Returns
    -------
    (autoscript_sdb_microscope_client.structures.RectanglePattern,
     autoscript_sdb_microscope_client.structures.RectanglePattern)
        Tuple containing the two lamella trench milling patterns.
    """
    setup_ion_milling(application_file="Si_Heidi", patterning_mode="Serial")
    center_y = +(lamella_thickness / 2) + lamella_buffer + (trench_height / 2)
    upper_trench = microscope.patterning.create_cleaning_cross_section(
        0, +center_y, trench_width, trench_height, milling_depth)
    lower_trench = microscope.patterning.create_cleaning_cross_section(
        0, -center_y, trench_width, trench_height, milling_depth)
    return upper_trench, lower_trench


def mill_trenches(microscope, *, milling_current=7.4e-9):
    """Create and mill the lamella trenches.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    """
    _trench_milling_patterns(microscope)
    confirm_and_run_milling(microscope, milling_current)


def _jcut_milling_patterns(microscope, *,
                           jcut_angle_degrees=6,
                           pretilt_degrees=27,
                           lamella_depth=5e-6,
                           jcut_length=12e-6,
                           jcut_trench_thickness=1e-6,
                           jcut_milling_depth=3e-6):
    """Create J-cut milling pattern in the center of the ion beam field of view.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    jcut_angle_degrees : int, optional
        Sample surface angle for J-cut in degrees, by default 6
    pretilt_degrees : int, optional
        Pre-tilt of sample holder in degrees, by default 27
    lamella_depth : float, optional
        Desired depth into sample bulk for lamella in meters.
    jcut_length : float, optional
        Length of J-cut from left to right in meters.
    jcut_trench_thickness : float, optional
        Thickness of the J-cut milling lines in meters.
    jcut_milling_depth : float, optional
        Ion beam milling depth for J-cut in meters.

    Returns
    -------
    (autoscript_sdb_microscope_client.structures.RectanglePattern,
     autoscript_sdb_microscope_client.structures.RectanglePattern,
     autoscript_sdb_microscope_client.structures.RectanglePattern)
        Tuple containing the three milling patterns comprising the J-cut.
    """
    setup_ion_milling(microscope)
    # Create milling patterns
    angle_correction = np.sin(np.deg2rad(52 - jcut_angle_degrees))
    # Top bar of J-cut
    jcut_top = microscope.patterning.create_rectangle(
        0.0,                               # center_x
        lamella_depth * angle_correction,  # center_y
        jcut_length,                       # width
        jcut_trench_thickness,             # height
        jcut_milling_depth)                # depth
    # Left hand side of J-cut (long side)
    extra_bit = 3e-6  # this cut should extend out a little past the lamella
    jcut_lhs = microscope.patterning.create_rectangle(
        -((jcut_length - jcut_trench_thickness) / 2),                # center_x
        ((lamella_depth - (extra_bit / 2)) / 2) * angle_correction,  # center_y
        jcut_trench_thickness,                                       # width
        (lamella_depth + extra_bit) * angle_correction,              # height
        jcut_milling_depth)                                          # depth
    # Right hand side of J-cut (short side)
    jcut_rightside_remaining = 1.5e-6  # in microns, how much to leave attached
    height = (lamella_depth - jcut_rightside_remaining) * angle_correction
    center_y = jcut_rightside_remaining + (height / 2)
    jcut_rhs = microscope.patterning.create_rectangle(
        +((jcut_top_length - jcut_trench_width) / 2),  # center_x
        center_y,                                      # center_y
        jcut_trench_thickness,                         # width
        height,                                        # height
        jcut_milling_depth)                            # depth
    return jcut_top, jcut_lhs, jcut_rhs


def mill_jcut(microscope, *, milling_current=0.74e-9):
    """Create and mill the rectangle patter to sever the jcut completely.

    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    """
    _jcut_milling_patterns(microscope)
    confirm_and_run_milling(microscope, milling_current)


def _jcut_severing_pattern(microscope, *,
                           jcut_angle=6,
                           pretilt_degrees=27,
                           lamella_depth=5e-6,
                           jcut_length=12e-6,
                           jcut_trench_thickness=1e-6,
                           jcut_milling_depth=3e-6):
    """Create J-cut milling pattern in the center of the ion beam field of view.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    jcut_angle_degrees : int, optional
        Sample surface angle for J-cut in degrees, by default 6
    pretilt_degrees : int, optional
        Pre-tilt of sample holder in degrees, by default 27
    lamella_depth : float, optional
        Desired depth into sample bulk for lamella in meters.
    jcut_length : float, optional
        Length of J-cut from left to right in meters.
    jcut_trench_thickness : float, optional
        Thickness of the J-cut milling lines in meters.
    jcut_milling_depth : float, optional
        Ion beam milling depth for J-cut in meters.

    Returns
    -------
    autoscript_sdb_microscope_client.structures.RectanglePattern
        Rectangle milling pattern used to sever the remaining bit of the J-cut.
    """
    setup_ion_milling(microscope)
    # Create milling pattern - right hand side of J-cut
    extra_bit = 3e-6  # this cut should extend out a little past the lamella
    angle_correction_factor = np.sin(np.deg2rad(52 - jcut_angle))
    center_x = +((jcut_length - jcut_trench_thickness) / 2)
    center_y = ((lamella_depth - (extra_bit / 2)) / 2) * angle_correction_factor  # noqa: E501
    width = jcut_trench_thickness
    height =  (lamella_depth + extra_bit) * angle_correction_factor
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
