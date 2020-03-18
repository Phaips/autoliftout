import time


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
    print('Parked position is at: {}'.format(park_position))
    print('Retracting needle...')
    microscope.specimen.manipulator.retract()
    retract_position_original = microscope.specimen.manipulator.current_position
    print('Needle retracted successfully')
    time.sleep(0.5)
    print('Inserting needle...')
    microscope.specimen.manipulator.insert()
    print('Needle inserted successfully')
    print('Original parked position: {}'.format(park_position))
    current_position = microscope.specimen.manipulator.current_position
    print('Current parked position: {}'.format(current_position))
    # User interaction
    print('Please move the needle manually')
    while True:    # infinite loop
        user_input = input("Have you finished moving the needle manually? yes/no")
        if user_input == "yes":
            retract_properly(microscope, park_position)
            retract_position = microscope.specimen.manipulator.current_position
            print('Original retracted position: {}'.format(retract_position_original))
            print('Current retracted position: {}'.format(retract_position))
            break  # stops the loop


if __name__ == "__main__":
    user_input = input("Is the needle in the park position? yes/no")
    if user_input == 'yes':
        main()
    else:
        print('Ok, cancelling program.')
