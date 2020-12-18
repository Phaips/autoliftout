"""Main entry script."""
import click
from datetime import datetime
import os
import logging

from liftout.calibration import setup
from liftout.user_input import load_config, protocol_stage_settings
# from liftout.milling import mill_lamella
# from liftout.needle import liftout_lamella, land_lamella

from liftout.align import *
from liftout.milling import *
from liftout.needle import *
from liftout.acquire import *
from liftout.calibration import *
from liftout.display import *
from liftout.stage_movement import *
from liftout.user_input import *


def configure_logging(log_filename='logfile', log_level=logging.INFO):
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


def needle_reference_images(microscope, move_needle_to="liftout"):
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings

    move_sample_stage_out(microscope)
    if move_needle_to == "liftout":
        park_position = move_needle_to_liftout_position(microscope)
    elif move_needle_to == "landing":
        park_position = move_needle_to_landing_position(microscope)
    # TODO: set field of view in electron & ion beam to match
    camera_settings = GrabFrameSettings(
        resolution="1536x1024",  # TODO: from yaml user input
        dwell_time=2e-6,
    )
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    needle_reference_eb = new_electron_image(microscope, camera_settings)
    autocontrast(microscope, beam_type=BeamType.ION)
    needle_reference_ib = new_ion_image(microscope, camera_settings)
    retract_needle(microscope, park_position)
    return needle_reference_eb, needle_reference_ib


def find_coordinates(microscope, name="", move_stage_angle=None):
    """Manually select stage coordinate positions."""
    if move_stage_angle == "trench":
        move_to_sample_grid(microscope)
    elif move_stage_angle == "landing":
        move_to_landing_grid(microscope)

    coordinates = []
    select_another_position = True
    while select_another_position:
        ensure_eucentricity(microscope)
        if move_stage_angle == "trench":
            move_to_trenching_angle(microscope)
        elif move_stage_angle == "landing":
            move_to_landing_angle(microscope)
        microscope.beams.electron_beam.horizontal_field_width.value = 400e-6
        microscope.beams.ion_beam.horizontal_field_width.value = 400e-6
        eb = new_electron_image(microscope)
        ib = new_ion_image(microscope)
        if ask_user(f"Please center the {name} coordinate in the ion beam.\n"
                    f"Is the {name} feature centered in the ion beam? yes/no: "):
            eb = new_electron_image(microscope)
            coordinates.append(microscope.specimen.stage.current_position)
            print(microscope.specimen.stage.current_position)
            select_another_position = ask_user(f"Do you want to select another {name} position? yes/no: ")
    return coordinates


def find_needletip_and_target_locations(image):
    print("Please click the needle tip position")
    needletip_location = select_point(image)
    print("Please click the lamella target position")
    target_location = select_point(image)
    return needletip_location, target_location


def manual_needle_movement_in_xy(microscope, move_in_x=True, move_in_y=True):
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings

    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator
    electron_image = new_electron_image(microscope, settings=GrabFrameSettings(dwell_time=500e-9, resolution="1536x1024"))  # TODO: User input imaging settings
    needletip_location, target_location = find_needletip_and_target_locations(electron_image)
    # Calculate needle movements
    x_needletip_location = needletip_location[0]  # coordinates in x-y format
    y_needletip_location = needletip_location[1]  # coordinates in x-y format
    x_target_location = target_location[0]  # pixels, coordinates in x-y format
    y_target_location = target_location[1]  # pixels, coordinates in x-y format
    if move_in_y is True:
        y_distance = y_target_location - y_needletip_location
        y_move = y_corrected_needle_movement(y_distance, stage.current_position.t)
        needle.relative_move(y_move)
    if move_in_x is True:  # MUST MOVE X LAST! Avoids hitting the sample
        x_distance = x_target_location - x_needletip_location
        x_move = x_corrected_needle_movement(x_distance)
        needle.relative_move(x_move)


def manual_needle_movement_in_z(microscope):
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings

    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator
    ion_image = new_ion_image(microscope, settings=GrabFrameSettings(dwell_time=500e-9, resolution="1536x1024"))  # TODO: user input imaging settings
    print("Please click the needle tip position")
    needletip_location = select_point(ion_image)
    print("Please click the lamella target position")
    target_location = select_point(ion_image)
    # Calculate movment
    z_safety_buffer = 400e-9  # in meters TODO: yaml user input
    z_distance = -(target_location[1] - needletip_location[1] / np.sin(np.deg2rad(52))) - z_safety_buffer
    z_move = z_corrected_needle_movement(z_distance, stage.current_position.t)
    needle.relative_move(z_move)


def liftout_lamella(microscope, settings, needle_reference_imgs):
    microscope.beams.ion_beam.horizontal_field_width.value = 150e-6  # can't be smaller than 150e-6
    microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # can't be smaller than 150e-6
    needle_reference_eb, needle_reference_ib = needle_reference_imgs
    # needletip_ref_location_eb = ??? TODO: automated needletip identification
    # needletip_ref_location_ib = ??? TODO: automated needletip identification
    park_position = move_needle_to_liftout_position(microscope)
    manual_needle_movement_in_xy(microscope, move_in_x=False)
    manual_needle_movement_in_z(microscope)
    manual_needle_movement_in_xy(microscope)
    sputter_platinum(microscope, sputter_time=60)  # TODO: yaml user input for sputtering application file choice
    mill_to_sever_jcut(microscope, settings['jcut'], confirm=False)  # TODO: yaml user input for jcut milling current
    retract_needle(microscope, park_position)
    needle_reference_images_with_lamella = needle_reference_images(
        microscope, move_needle_to="landing")
    return needle_reference_images_with_lamella


def land_lamella(microscope, landing_coord):
    move_to_landing_grid(microscope)
    microscope.specimen.stage.absolute_move(landing_coord)
    import pdb; pdb.set_trace()
    # realign landing post
    # move needle + lamella in
    park_position = move_needle_to_landing_position(microscope)
    manual_needle_movement_in_xy(microscope, move_in_x=False)
    manual_needle_movement_in_z(microscope)
    manual_needle_movement_in_xy(microscope)
    # weld lamella to post
    # cut off needle + retract it


def single_liftout(microscope, settings, lamella_coord, landing_coord):
    needle_reference_imgs = needle_reference_images(microscope)
    move_to_sample_grid(microscope)
    move_to_trenching_angle(microscope)
    microscope.specimen.stage.absolute_move(lamella_coord)
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    autocontrast(microscope, beam_type=BeamType.ION)
    mill_lamella(microscope, settings)
    needle_reference_images_with_lamella = liftout_lamella(microscope, settings, needle_reference_imgs)
    land_lamella(microscope)


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
    timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    output_log_filename = os.path.join('logfile' + timestamp + '.log')
    configure_logging(log_filename=output_log_filename)
    main(settings)


def main(settings):
    microscope = initialize(settings["system"]["ip_address"])
    sputter_platinum_over_whole_grid(microscope)
    print("Please select the lamella positions and check eucentric height manually.")
    lamella_coordinates = find_coordinates(microscope, name="lamella", move_stage_angle="trench")
    landing_coordinates = find_coordinates(microscope, name="landing position", move_stage_angle="landing")
    zipped_coordinates = list(zip(lamella_coordinates, landing_coordinates))
    # Start liftout for each lamella
    for lamella_coord, landing_coord in zipped_coordinates:
        single_liftout(microscope, settings, lamella_coord, landing_coord)
    print("Finished.")


if __name__ == '__main__':
    try:
        main_cli()
    except KeyboardInterrupt:
        logging.error('Keyboard Interrupt: Cancelling program.')
