"""Module controlling needle manipulator movements."""
import logging
import time

import numpy as np
import tqdm

from liftout.old_functions.calibration import auto_link_stage
from liftout.old_functions.stage_movement import move_to_sample_grid

__all__ = [
    "move_needle_to_liftout_position",
    "move_needle_to_landing_position",
    "sputter_platinum",
    "sputter_platinum_over_whole_grid",
    "insert_needle",
    "retract_needle",
    "move_needle_closer",
    "x_corrected_needle_movement",
    "y_corrected_needle_movement",
    "z_corrected_needle_movement",
]


def move_needle_to_liftout_position(microscope):
    """Move the needle into position, ready for liftout.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.sdb_microscope.SdbMicroscopeClient
        The Autoscript microscope object.
    """
    park_position = insert_needle(microscope)
    move_needle_closer(microscope)
    multichem = microscope.gas.get_multichem()
    multichem.insert()
    return park_position


def move_needle_to_landing_position(microscope):
    """Move the needle into position, ready for landing.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.sdb_microscope.SdbMicroscopeClient
        The Autoscript microscope object.
    """
    park_position = insert_needle(microscope)
    move_needle_closer(microscope, x_shift=-40e-6)
    return park_position


def sputter_platinum(microscope, sputter_time=60, *,
                     sputter_application_file="cryo_Pt_dep",
                     default_application_file="autolamella",
                     horizontal_field_width=100e-6,
                     line_pattern_length=15e-6):
    """Sputter platinum over the sample.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    sputter_time : int, optional
        Time in seconds for platinum sputtering. Default is 60 seconds.
    sputter_application_file : str
        Application file for platinum sputtering/deposition.
    default_application_file : str
        Default application file, to return to after the platinum sputtering.
    """
    # Setup
    original_active_view = microscope.imaging.get_active_view()
    microscope.imaging.set_active_view(1)  # the electron beam view
    microscope.patterning.clear_patterns()
    microscope.patterning.set_default_application_file(sputter_application_file)
    microscope.patterning.set_default_beam_type(1)  # set electron beam for patterning
    multichem = microscope.gas.get_multichem()
    multichem.insert()
    # Create sputtering pattern
    microscope.beams.electron_beam.horizontal_field_width.value = horizontal_field_width
    pattern = microscope.patterning.create_line(-line_pattern_length/2,  # x_start
                                                +line_pattern_length,    # y_start
                                                +line_pattern_length/2,  # x_end
                                                +line_pattern_length,    # y_end
                                                2e-6)                    # milling depth
    pattern.time = sputter_time + 0.1
    # Run sputtering with progress bar
    microscope.beams.electron_beam.blank()
    if microscope.patterning.state == "Idle":
        print('Sputtering with platinum for {} seconds...'.format(sputter_time))
        microscope.patterning.start()  # asynchronous patterning
    else:
        raise RuntimeError(
            "Can't sputter platinum, patterning state is not ready."
        )
    for i in tqdm.tqdm(range(int(sputter_time))):
        time.sleep(1)  # update progress bar every second
    if microscope.patterning.state == "Running":
        microscope.patterning.stop()
    else:
        logging.warning("Patterning state is {}".format(microscope.patterning.state))
        logging.warning("Consider adjusting the patterning line depth.")
    # Cleanup
    microscope.patterning.clear_patterns()
    microscope.beams.electron_beam.unblank()
    microscope.patterning.set_default_application_file(default_application_file)
    microscope.imaging.set_active_view(original_active_view)
    microscope.patterning.set_default_beam_type(2)  # set ion beam
    multichem.retract()
    logging.info("Sputtering finished.")


def sputter_platinum_over_whole_grid(microscope):
    """Sputter platnium over whole grid."""
    stage = microscope.specimen.stage
    move_to_sample_grid(microscope)
    auto_link_stage(microscope, expected_z=5e-3)
    # TODO: yaml user input for sputtering application file choice
    sputter_platinum(microscope, sputter_time=60, horizontal_field_width=30e-6, line_pattern_length=7e-6)
    auto_link_stage(microscope)  # return stage to default linked z height


def insert_needle(microscope):
    """Insert the needle and return the needle parking position.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.

    Returns
    -------
    park_position : autoscript_sdb_microscope_client.structures.ManipulatorPosition
        The parking position for the needle manipulator when inserted.
    """
    needle = microscope.specimen.manipulator
    needle.insert()
    park_position = needle.current_position
    return park_position


def retract_needle(microscope, park_position):
    """Retract the needle and multichem, preserving the correct park position.

    Parameters
    ----------
    microscope :  autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    park_position : autoscript_sdb_microscope_client.structures.ManipulatorPosition
        The parking position for the needle manipulator when inserted.
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    # Retract the multichem
    multichem = microscope.gas.get_multichem()
    multichem.retract()
    # Retract the needle, preserving the correct parking postiion
    needle = microscope.specimen.manipulator
    current_position = needle.current_position
    # To prevent collisions with the sample; first retract in z, then y, then x
    needle.relative_move(ManipulatorPosition(z=park_position.z - current_position.z))  # noqa: E501
    needle.relative_move(ManipulatorPosition(y=park_position.y - current_position.y))  # noqa: E501
    needle.relative_move(ManipulatorPosition(x=park_position.x - current_position.x))  # noqa: E501
    time.sleep(1)  # AutoScript sometimes throws errors if you retract too quick?
    needle.retract()
    retracted_position = needle.current_position
    return retracted_position


def move_needle_closer(microscope, *, x_shift=-20e-6, z_shift=-180e-6):
    """Move the needle closer to the sample surface, after inserting.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.sdb_microscope.SdbMicroscopeClient
        The Autoscript microscope object.
    x_shift : float
        Distance to move the needle from the parking position in x, in meters.
    z_shift : float
        Distance to move the needle towards the sample in z, in meters.
        Negative values move the needle TOWARDS the sample surface.
    """
    needle = microscope.specimen.manipulator
    stage = microscope.specimen.stage
    # Needle starts from the parking position (after inserting it)
    # Move the needle back a bit in x, so the needle is not overlapping target
    x_move = x_corrected_needle_movement(x_shift)
    needle.relative_move(x_move)
    # Then move the needle towards the sample surface.
    z_move = z_corrected_needle_movement(z_shift, stage.current_position.t)
    needle.relative_move(z_move)
    # The park position is always the same,
    # so the needletip will end up about 20 microns from the surface.
    return needle.current_position


def x_corrected_needle_movement(expected_x, stage_tilt=None):
    """Needle movement in X, XTGui coordinates (Electron coordinate).

    Parameters
    ----------
    expected_y : float
        in meters

    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    return ManipulatorPosition(x=expected_x, y=0, z=0)  # no adjustment needed


def y_corrected_needle_movement(expected_y, stage_tilt):
    """Needle movement in Y, XTGui coordinates (Electron coordinate).

    Parameters
    ----------
    expected_y : in meters
    stage_tilt : in radians

    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    y_move = +np.cos(stage_tilt) * expected_y
    z_move = +np.sin(stage_tilt) * expected_y
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def z_corrected_needle_movement(expected_z, stage_tilt):
    """Needle movement in Z, XTGui coordinates (Electron coordinate).

    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in radians

    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    y_move = -np.sin(stage_tilt) * expected_z
    z_move = +np.cos(stage_tilt) * expected_z
    return ManipulatorPosition(x=0, y=y_move, z=z_move)