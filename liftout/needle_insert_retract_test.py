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
    current_position = microscope.specimen.manipulator.current_position
    relative_movement = ManipulatorPosition(
        x=park_position.x - current_position.x,
        y=park_position.y - current_position.y,
        z=park_position.z - current_position.z)
    microscope.specimen.manipulator.relative_move(relative_movement)
    microscope.specimen.manipulator.retract()


def main():
    microscope = initialize()
    park_position = microscope.specimen.manipulator.current_position
    logging.info('Parked position is at: {}'.format(park_position))
    logging.info('Retracting needle...')
    microscope.specimen.manipulator.retract()
    retract_position_original = microscope.specimen.manipulator.current_position
    logging.info('Needle retracted successfully')
    time.sleep(0.5)
    logging.info('Inserting needle...')
    microscope.specimen.manipulator.insert()
    logging.info('Needle inserted successfully')
    logging.info('Original parked position: {}'.format(park_position))
    current_position = microscope.specimen.manipulator.current_position
    logging.info('Current parked position: {}'.format(current_position))
    # User interaction
    logging.info('Please move the needle manually')
    while True:    # infinite loop
        user_input = input("Have you finished moving the needle manually? yes/no\n")
        if user_input == "yes":
            retract_properly(microscope, park_position)
            retract_position = microscope.specimen.manipulator.current_position
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
