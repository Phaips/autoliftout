from liftout.fibsem.movement import *
import time
import logging


def initialise_fibsem(ip_address='10.0.0.1'):
    """Initialize connection to FIBSEM microscope with Autoscript."""
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    microscope = SdbMicroscopeClient()
    microscope.connect(ip_address)
    return microscope


def sputter_platinum(microscope, settings, whole_grid=False, sputter_time=60,
                     horizontal_field_width=100e-6, line_pattern_length=15e-6,
                     sputter_application_file="cryo_Pt_dep",
                     default_application_file="autolamella",
                     ):
    """Sputter platinum over the sample.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    sputter_time : int, optionalye
        Time in seconds for platinum sputtering. Default is 60 seconds.
    sputter_application_file : str
        Application file for platinum sputtering/deposition.
    default_application_file : str
        Default application file, to return to after the platinum sputtering.
    """

    # TODO: Check if auto_link used outside sputter_whole_grid with Sergey

    # TODO: add whole_grid sputter parameters to protocol
    if whole_grid:
        stage = microscope.specimen.stage
        move_to_sample_grid(microscope, settings)
        auto_link_stage(microscope, expected_z=5e-3)
        sputter_time = 20
        horizontal_field_width = 30e-6
        line_pattern_length = 7e-6

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
    # Run sputtering
    microscope.beams.electron_beam.blank()
    if microscope.patterning.state == "Idle":
        print('Sputtering with platinum for {} seconds...'.format(sputter_time))
        microscope.patterning.start()  # asynchronous patterning
    else:
        raise RuntimeError(
            "Can't sputter platinum, patterning state is not ready."
        )
    if microscope.patterning.state == "Running":
        microscope.patterning.stop()
    else:
        logging.warning("Patterning state is {}".format(microscope.patterning.state))
        logging.warning("Consider adjusting the patterning line depth.")

    # Cleanup
    microscope.beams.electron_beam.unblank()
    microscope.patterning.set_default_application_file(default_application_file)
    microscope.imaging.set_active_view(original_active_view)
    microscope.patterning.set_default_beam_type(2)  # set ion beam
    multichem.retract()
    logging.info("Sputtering finished.")

    if whole_grid:
        auto_link_stage(microscope)