import click
from datetime import datetime
import os
import logging

from liftout.calibration import setup
from liftout.user_input import load_config, protocol_stage_settings
from liftout.milling import mill_lamella
from liftout.needle import liftout_lamella, land_lamella


def configure_logging(log_filename='logfile', log_level=logging.DEBUG):
    """Log to the terminal and to file simultaneously."""
    timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[
            logging.FileHandler(log_filename+timestamp+'.log'),
            logging.StreamHandler(),
        ])


def initialize(ip_address='10.0.0.1'):
    """Initialize connection to FIBSEM microscope with Autoscript."""
    from autoscript_sdb_microscope_client import SdbMicroscopeClient

    microscope = SdbMicroscopeClient()
    microscope.connect(ip_address)
    return microscope


@click.command()
@click.argument("config_filename")
def main_cli(config_filename):
    """Run the main command line interface.

    Parameters
    ----------
    config_filename : str
        Path to protocol file with input parameters given in YAML (.yml) format
    """
    settings = load_config(config_filename)
    output_log_filename = os.path.join(data_directory, 'logfile.log')
    configure_logging(log_filename=output_log_filename)
    main(settings)


def main(settings):
    microscope = initialize(settings["system"]["ip_address"])
    # single liftout
    setup(microscope)  # setup microscope
    # setup landing position, setup needle image
    mill_lamella(microscope, settings) # select position, trench, jcut
    liftout_lamella() # insert needle, touch needle, sputter, retract, take picture with no background
    land_lamella()  # move to landing grid, find/align landing post, touch needle, glue, cut off needle


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.error('Keyboard Interrupt: Cancelling program.')
