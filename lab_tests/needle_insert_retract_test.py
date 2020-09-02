import logging
import time


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


def retract_properly(microscope, park_position):
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition
    needle = microscope.specimen.manipulator
    multichem = microscope.gas.get_multichem()
    multichem.retract()
    current_position = needle.current_position
    # First retract in z, then y, then x
    needle.relative_move(ManipulatorPosition(z=park_position.z - current_position.z))
    needle.relative_move(ManipulatorPosition(y=park_position.y - current_position.y))
    needle.relative_move(ManipulatorPosition(x=park_position.x - current_position.x))
    time.sleep(1)  # AutoScript sometimes throws errors if you retract too quick?
    needle.retract()
    retracted_position = needle.current_position
    return retracted_position


def insert_properly(microscope):
    needle = microscope.specimen.manipulator
    needle.insert()
    park_position = needle.current_position
    return park_position


def main():
    microscope = initialize()
    needle = microscope.specimen.manipulator
    park_position = insert_properly(needle)
    logging.info('Parked position is at: {}'.format(park_position))
    logging.info('Retracting needle...')
    retract_position_original = retract_properly(needle, park_position)
    logging.info('Needle retracted successfully')
    time.sleep(0.5)
    logging.info('Inserting needle...')
    insert_properly(needle)
    logging.info('Needle inserted successfully')
    logging.info('Original parked position: {}'.format(park_position))
    current_position = needle.current_position
    logging.info('Current parked position: {}'.format(current_position))
    # User interaction
    logging.info('Please move the needle manually')
    while True:    # infinite loop
        user_input = input("Have you finished moving the needle manually? yes/no\n")
        if user_input == "yes":
            retract_position = retract_properly(needle, park_position)
            logging.info('Original retracted position: {}'.format(retract_position_original))
            logging.info('Current retracted position: {}'.format(retract_position))
            logging.info('Finished.')
            break  # stops the loop


if __name__ == "__main__":
    user_input = input("Is the needle calibrated and in the park position? yes/no\n")
    if user_input == 'yes':
        configure_logging(log_filename='needle_insert_remove_logfiles.log')
        main()
    else:
        print('Ok, cancelling program.')
