""""""
import logging
import time

import tqdm

__all__ = [
    "sputter_platinum",
    "insert_needle",
    "retract_needle",
    "x_corrected_needle_movement",
    "y_corrected_needle_movement",
    "z_corrected_needle_movement",
]



def sputter_platinum(microscope, sputter_time=60, *,
                     sputter_application_file="cryo_Pt_dep",
                     default_application_file="autolamella"):
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
    # Create sputtering pattern
    pattern = microscope.patterning.create_line(-15e-6, +15e-6, +15e-6, +15e-6, 2e-6)  # 1um, at zero in the FOV
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
        logging.warning("Patterning state is {}".format(self.patterning.state))
        loggging.warning("Consider adjusting the patterning line depth.")
    # Cleanup
    microscope.patterning.clear_patterns()
    microscope.beams.electron_beam.unblank()
    microscope.patterning.set_default_application_file(default_application_file)
    microscope.imaging.set_active_view(original_active_view)
    microscope.patterning.set_default_beam_type(2)  # set ion beam
    logging.info("Sputtering finished.")


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


def x_corrected_needle_movement(expected_x):
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
    stage_tilt : in degrees

    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    tilt_radians = np.deg2rad(stage_tilt)
    y_move = +np.cos(tilt_radians) * expected_y
    z_move = +np.sin(tilt_radians) * expected_y
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def z_corrected_needle_movement(expected_z, stage_tilt):
    """Needle movement in Z, XTGui coordinates (Electron coordinate).

    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in degrees

    Returns
    -------
    ManipulatorPosition
    """
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    tilt_radians = np.deg2rad(stage_tilt)
    y_move = -np.sin(tilt_radians) * expected_z
    z_move = +np.cos(tilt_radians) * expected_z
    return ManipulatorPosition(x=0, y=y_move, z=z_move)
