import os
import logging

from .calibration import setup


def configure_logging(log_filename='logfile.log', log_level=logging.DEBUG):
    """Log to the terminal and to file simultaneously."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ])


def initialize(ip_address='10.0.0.1'):
    """Initialize connection to FIBSEM microscope with Autoscript."""
    from autoscript_sdb_microscope_client import SdbMicroscopeClient

    microscope = SdbMicroscopeClient()
    microscope.connect(ip_address)
    return microscope


def main():
    data_directory = "D:\SharedData\MyData\genevieve.buckley@monash.edu\\200316_liftout\data\\"
    output_log_filename = os.path.join(data_directory, 'logfile.log')
    configure_logging(log_filename=output_log_filename)
    setup(microscope)
    # code here


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.error('Keyboard Interrupt: Cancelling program.')
