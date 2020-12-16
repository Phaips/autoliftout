from liftout import *


def needle_reference_images(microscope, move_needle_to="liftout"):
    move_sample_stage_out(microscope)
    if move_needle_to == "liftout":
        move_needle_to_liftout_position(microscope)
    elif move_needle_to == "landing":
        move_needle_to_landing_position(microscope)
    # TODO: image acquisition settings
    needle_reference_eb = new_electron_image(microscope)
    needle_reference_ib = new_ion_image(microscope)
    retract_needle(microscope, park_position)
    return needle_reference_eb, needle_reference_ib


def find_coordinates(microscope, name="", move_stage_angle=None):
    """Manually select stage coordinate positions."""
    if move_stage_angle == "trench":
        move_to_samploe_grid(microscope)
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
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator
    microscope.beams.electron_beam.horizontal_field_width.value = 82.9e-6  # TODO: user input from yaml file
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
    microscope.beams.ion_beam.horizontal_field_width.value = 82.9e-6  # TODO: user input from yaml file
    ion_image = new_ion_image(microscope, settings=GrabFrameSettings(dwell_time=500e-9, resolution="1536x1024"))  # TODO: user input imaging settings
    print("Please click the needle tip position")
    needletip_location = select_point(ion_image)
    print("Please click the lamella target position")
    target_location = select_point(ion_image)
    # Calculate movment
    z_safety_buffer = 400e-9  # in meters TODO: yaml user input
    z_distance = -(ion_target[1] - ion_needletip[1] / np.sin(np.deg2rad(52))) - z_safety_buffer
    z_move = z_corrected_needle_movement(z_distance, stage.current_position.t)
    needle.relative_move(z_move)


def liftout_lamella(microscope, needle_reference_imgs):
    needle_reference_eb, needle_reference_ib = needle_reference_imgs
    # needletip_ref_location_eb = ??? TODO: automated needletip identification
    # needletip_ref_location_ib = ??? TODO: automated needletip identification
    park_position = move_needle_to_liftout_position(microscope)
    manual_needle_movement_in_xy(microscope, move_in_x=False)
    manual_needle_movement_in_z(microscope)
    manual_needle_movement_in_xy(microscope)
    sputter_platinum(microscope, sputter_time=60)  # TODO: yaml user input for sputtering application file choice
    mill_to_sever_jcut(microscope)  # TODO: yaml user input for jcut milling current
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
    microscope.specimen.stage.absolute_position(lamella_coord)
    mill_lamella(microscope, settings)
    liftout_lamella(microscope, needle_reference_imgs)
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
    output_log_filename = os.path.join(data_directory, 'logfile.log')
    configure_logging(log_filename=output_log_filename)
    main(settings)


def main(settings):
    # ASSUMES THE NEEDLE IS ALREADY CALIBRATION
    # AND ALL THE ALIGNMENTS ARE PERFECT
    microscope = initialize()
    # needle_reference_images = needle_reference_images(microscope)
    sputter_platinum_over_whole_grid(microscope)
    print("Please select the lamella positions and check eucentric height manually.")
    lamella_coordinates = find_coordinates(microscope, name="lamella", move_stage_angle="trench")
    landing_coordinates = find_coordinates(microscope, name="landing position", move_stage_angle="landing")
    zipped_coordinates = zip(lamella_coordinates, landing_coordinates)
    # Start liftout for each lamella
    for lamella_coord, landing_coord in zipped_coordinates:
        single_liftout(microscope, settings, lamella_coord, landing_coord)
    print("Finished.")


if __name__=="__main__":
    main_cli()
