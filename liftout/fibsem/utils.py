import logging
import time

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from liftout.fibsem import movement
from liftout.fibsem.acquire import BeamType, ImageSettings
from liftout.utils import configure_logging


def connect_to_microscope(ip_address="10.0.0.1", parent_ui=None):
    """Connect to the FIBSEM microscope."""
    try:
        # TODO: get the port
        logging.info(f"Microscope client connecting to [{ip_address}]")
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        logging.info(f"Microscope client connected to [{ip_address}]")
    except Exception as e:

        if parent_ui:
            import liftout.gui.utils as ui_utils

            ui_utils.display_error_message(
                f"AutoLiftout is unavailable. Unable to connect to microscope: {e}"
            )
        else:
            raise e

        microscope = None

    return microscope


def sputter_platinum(microscope, settings, whole_grid=False):
    """Sputter platinum over the sample.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    settings: dict
        The protocol settings
    """

    if whole_grid:
        movement.move_to_sample_grid(microscope, settings)
        sputter_time = settings["platinum"]["whole_grid"]["time"]  # 20
        hfw = settings["platinum"]["whole_grid"]["hfw"]  # 30e-6
        line_pattern_length = settings["platinum"]["whole_grid"]["length"]  # 7e-6
        logging.info("sputtering platinum over the whole grid.")
    else:
        sputter_time = settings["platinum"]["weld"]["time"]  # 20
        hfw = settings["platinum"]["weld"]["hfw"]  # 100e-6
        line_pattern_length = settings["platinum"]["weld"]["length"]  # 15e-6
        logging.info("sputtering platinum to weld.")

    # Setup
    original_active_view = microscope.imaging.get_active_view()
    microscope.imaging.set_active_view(1)  # the electron beam view
    microscope.patterning.clear_patterns()
    microscope.patterning.set_default_application_file(
        settings["platinum"]["application_file"]
    )  # sputter_application_file)
    microscope.patterning.set_default_beam_type(1)  # set electron beam for patterning
    multichem = microscope.gas.get_multichem()
    multichem.insert(settings["platinum"]["position"])
    multichem.turn_heater_on(settings["platinum"]["gas"])  # "Pt cryo")
    time.sleep(3)

    # Create sputtering pattern
    microscope.beams.electron_beam.horizontal_field_width.value = hfw
    horizontal_offset = 30e-6
    pattern = microscope.patterning.create_line(
        -line_pattern_length / 2,  # x_start
        +line_pattern_length,  # y_start
        +line_pattern_length / 2,  # x_end
        +line_pattern_length,  # y_end
        2e-6,
    )  # milling depth
    pattern.time = sputter_time + 0.1
    # Run sputtering
    microscope.beams.electron_beam.blank()
    if microscope.patterning.state == "Idle":
        logging.info("Sputtering with platinum for {} seconds...".format(sputter_time))
        microscope.patterning.start()  # asynchronous patterning
        time.sleep(sputter_time + 5)
    else:
        raise RuntimeError("Can't sputter platinum, patterning state is not ready.")
    if microscope.patterning.state == "Running":
        microscope.patterning.stop()
    else:
        logging.warning("Patterning state is {}".format(microscope.patterning.state))
        logging.warning("Consider adjusting the patterning line depth.")

    # Cleanup
    microscope.patterning.clear_patterns()
    microscope.beams.electron_beam.unblank()
    microscope.patterning.set_default_application_file(
        settings["system"]["application_file"]
    )  # default_application_file)
    microscope.imaging.set_active_view(original_active_view)
    microscope.patterning.set_default_beam_type(2)  # set ion beam
    multichem.retract()
    logging.info("sputtering platinum finished.")


def sputter_platinum_on_whole_sample_grid(
    microscope: SdbMicroscopeClient = None,
    settings: dict = None,
    image_settings: ImageSettings = None,
) -> None:
    """Move to the sample grid and sputter platinum over the whole grid"""
    from liftout.gui.windows import ask_user_interaction

    # Whole-grid platinum deposition
    response = ask_user_interaction(
        microscope=microscope,
        msg="Do you want to sputter the whole \nsample grid with platinum?",
        beam_type=BeamType.ELECTRON,
    )

    if response:
        sputter_platinum(microscope, settings, whole_grid=True)

    return


def quick_setup():
    """Quick setup for microscope, settings, and iamge_settings"""
    from liftout import utils
    from liftout.fibsem import acquire
    from liftout.fibsem import utils as fibsem_utils

    settings = utils.load_full_config()

    import os

    path = os.path.join(os.getcwd(), "tools/test")
    os.makedirs(path, exist_ok=True)
    configure_logging(path)

    microscope = fibsem_utils.connect_to_microscope(
        ip_address=settings["system"]["ip_address"]
    )
    image_settings = acquire.update_image_settings_v3(settings, path=path)

    return microscope, settings, image_settings
