import logging

import numpy as np

from .stage_movement import PRETILT_DEGREES

__all__ = [
    'setup',
    'validate_scanning_rotation',
    'zero_beam_shift',
    "linked_within_z_tolerance",
    "auto_link_stage",
    'ensure_eucentricity',
]


def setup(microscope):
    """Setup actions for liftout procedure."""
    assert microscope.specimen.stage.is_linked
    validate_scanning_rotation(microscope)
    zero_beam_shift(microscope)
    set_magnification(microscope)
    ensure_eucentricity(microscope, PRETILT_DEGREES)
    return microscope


def validate_scanning_rotation(microscope):
    """Ensure the scanning rotation is set to zero."""
    rotation = microscope.beams.ion_beam.scanning.rotation.value
    if rotation is None:
        microscope.beams.ion_beam.scanning.rotation.value = 0
        rotation = microscope.beams.ion_beam.scanning.rotation.value
    if not np.isclose(rotation, 0.0):
        raise ValueError(
            "Ion beam scanning rotation must be 0 degrees."
            "\nPlease change your system settings and try again."
            "\nCurrent rotation value is {}".format(rotation)
        )


def set_magnification(microscope, ion_beam_field_of_view=104e-6):
    """Set the correct magnification (i.e. field of view) for the ion beam.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object.
    ion_beam_field_of_view : float
        Width of the desired ion beam field of view, in meters.
    """
    microscope.beams.ion_beam.horizontal_field_width.value = ion_beam_field_of_view


def zero_beam_shift(microscope, *,
                    zero_electron_beam=True,
                    zero_ion_beam=True):
    """Zero the beam shift for electron and ion beams.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object.
    zero_electron_beam : bool
        Whether to zero the ELECTRON beam shift.
    zero_ion_beam : bool
        Whether to zero the ION beam shift.
    """
    from autoscript_sdb_microscope_client.structures import Point

    if zero_electron_beam:
        microscope.beams.electron_beam.beam_shift.value = Point(0, 0)
        assert microscope.beams.electron_beam.beam_shift.value == Point(0, 0)
    if zero_ion_beam:
        microscope.beams.ion_beam.beam_shift.value = Point(0, 0)
        assert microscope.beams.ion_beam.beam_shift.value == Point(0, 0)


def linked_within_z_tolerance(microscope, expected_z=3.9e-3, tolerance=1e-6):
    """Check if the sample stage is linked and at the expected z-height.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    expected_z : float, optional
        Correct height for linked stage in z, in meters, by default 4e-3
    tolerance : float, optional
        Must be within this absolute tolerance of expected stage z height,
        in meters, by default 1e-4

    Returns
    -------
    bool
        Returns True if stage is linked and at the correct z-height.
    """
    # Check the microscope stage is at the correct height
    z_stage_height = microscope.specimen.stage.current_position.z
    if np.isclose(z_stage_height, expected_z, atol=tolerance):
        return True
    else:
        return False


def auto_link_stage(microscope, expected_z=3.9e-3, tolerance=1e-6):
    """Automatically focus and link sample stage z-height.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    expected_z : float, optional
        Correct height for linked stage in z, in meters, by default 4e-3
    tolerance : float, optional
        Must be within this absolute tolerance of expected stage z height,
        in meters, by default 1e-4
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    counter = 0
    while not linked_within_z_tolerance(microscope,
                                        expected_z=expected_z,
                                        tolerance=tolerance):
        print(counter)
        if counter > 3:
            raise(UserWarning("Could not auto-link z stage height."))
            break
        # Focus and re-link z stage height
        microscope.auto_functions.run_auto_focus()
        microscope.specimen.stage.link()
        microscope.specimen.stage.absolute_move(StagePosition(z=expected_z))
        print(microscope.specimen.stage.current_position)


def ensure_eucentricity(microscope, *, pretilt_angle=PRETILT_DEGREES):
    """Check the sample stage is at the eucentric height.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object.
    pretilt_angle : float
        Extra tilt added by the cryo-grid sample holder, in degrees.
    """
    from autoscript_sdb_microscope_client.structures import StagePosition

    validate_scanning_rotation(microscope)  # ensure scan rotation is zero
    flat_to_electron_beam(microscope, pretilt)
    microscope.beams.electron_beam.horizontal_field_width.value = 0.000104
    logging.info("Please click a feature to center in the electron beam image")
    center_sem_location(microscope)
    logging.info("Please click the same location in the ion beam image")
    ion_image = new_ion_image(microscope)
    click_location = select_point(ion_image)
    _x, fib_delta_y = click_location
    tilt = microscope.specimen.stage.current_position.t
    delta_z = calculate_delta_z(fib_delta_y, tilt)
    microscope.specimen.stage.relative_move(StagePosition(z=delta_z))
    # Could replace this with an autocorrelation (maybe with a fallback to asking for a user click if the correlation values are too low)
    center_sem_location(microscope)
    final_electron_image = new_electron_image(microscope)
    final_ion_image = new_ion_image(microscope)
    return final_electron_image, final_ion_image
