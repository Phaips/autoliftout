from liftout import *

# ASSUMES THE NEEDLE IS ALREADY CALIBRATION
# AND ALL THE ALIGNMENTS ARE PERFECT

microscope = initialize()
yaml_filename = "protocol_liftout.yml"
settings = load_config(yaml_filename)


# Take needle picture with blank background
def needle_reference_images(microscope):
    move_sample_stage_out(microscope)
    move_needle_to_liftout_position(microscope)
    # TODO: image acquisition settings
    needle_reference_eb = new_electron_image(microscope)
    needle_reference_ib = new_ion_image(microscope)
    retract_needle(microscope, park_position)
    return needle_reference_eb, needle_reference_ib


# Sputter platnium over whole grid
stage = microscope.specimen.stage
move_to_sample_grid(microscope)
auto_link_stage(microscope)
# TODO: yaml user input for sputtering application file choice
sputter_platinum(microscope, sputter_time=60)


# Find sample positions
def find_coordinates(microscope, name="", move_stage_angle=None):
    coordinates = []
    select_another_position = True
    while select_another_position:
        flat_to_electron_beam(microscope)
        if ask_user("Please move to the eucentric height.\n
                    "Is the eucentric height now correct? yes/no:"):
            if move_stage_angle is not None:
                move_stage_angle(microscope)
            if ask_user("Please center the {name} coordinate in the ion beam?.\n"
                        "Is the {name} feature centered in the ion beam? yes/no:"):
                coordinates.append(stage.current_position)
            select_another_position = ask_user("Do you want to select another {name} position?")
    return coordinates


print("Please select the lamella positions and check eucentric height manually.")
lamella_coordinates = find_coordinates(microscope, name="lamella", move_stage_angle=move_to_trenching_angle)
landing_coordinates = find_coordinates(microscope, name="landing position", move_stage_angle=move_to_landing_angle)
zipped_coordinates = zip(lamella_coordinates, landing_coordinates)


def liftout_lamella(microscope, needle_reference_imgs):
    needle_reference_eb, needle_reference_ib = needle_reference_imgs
    # needletip_ref_location_eb = ??? TODO: automated needletip identification
    # needletip_ref_location_ib = ??? TODO: automated needletip identification
    park_position = move_needle_to_liftout_position(microscope)
    # Z NEEDLE MOVEMENT
    # TODO: user parameters for imaging settings & field of viewe from yaml file
    microscope.beams.ion_beam.horizontal_field_width.value = 82.9e-6
    ion_image = new_ion_image(microscope, settings=GrabFrameSettings(dwell_time=500e-9, resolution="1536x1024"))
    print("Please click the needle tip position")
    needletip_location = select_point(ion_image)
    print("Please click the lamella target position")
    target_location = select_point(ion_image)
    # Calculate movment
    z_safety_buffer = 400e-9  # in meters TODO: yaml user input
    z_distance = -(ion_target[1] - ion_needletip[1] / np.sin(np.deg2rad(52))) - z_safety_buffer
    z_move = z_corrected_needle_movement(z_distance, stage.current_position.t)
    needle.relative_move(z_move)
    # Y NEEDLE MOVMEMNT, AND FINALLY THE X NEEDLE MOVEMENT
    # TODO: user parameters for imaging settings & field of viewe from yaml file
    microscope.beams.electron_beam.horizontal_field_width.value = 82.9e-6
    electron_image = new_electron_image(microscope, settings=GrabFrameSettings(dwell_time=500e-9, resolution="1536x1024"))
    # USER INPUT - Click to mark needle tip and target position in the electron beam image.
    print("Please click the needle tip position")
    needletip_location = select_point(electron_image)
    print("Please click the lamella target position")
    target_location = select_point(electron_image)
    # Calculate needle movements
    x_needletip_location = needletip_location[0]  # coordinates in x-y format
    y_needletip_location = needletip_location[1]  # coordinates in x-y format
    x_target_location = target_location[0]  # pixels, coordinates in x-y format
    y_target_location = target_location[1]  # pixels, coordinates in x-y format
    # Calculate the distance between the needle tip and the target.
    x_distance = x_target_location - x_needletip_location
    y_distance = y_target_location - y_needletip_location
    x_move = x_corrected_needle_movement(x_distance)
    y_move = y_corrected_needle_movement(y_distance, stage.current_position.t)
    needle.relative_move(y_move)
    needle.relative_move(x_move)  # x needle movmement must be last!
    sputter_platinum(microscope, sputter_time=60)  # TODO: yaml user input for sputtering application file choice
    # Sever jcut edge
    mill_to_sever_jcut(microscope)  # TODO: yaml user input for jcut milling current
    retract_needle(microscope, park_position)
    # Take a picture of the needle + lamella with no background
    move_needle_to_landing_position(microscope)
    electron_image = new_electron_image(microscope)  # TODO: include imaging settings
    ion_image = new_ion_image(microscope)  # TODO: include imaging settings
    retract_needle(microscope, park_position)
    return electron_image, ion_image


def land_lamella(microscope, landing_coord):
    move_to_landing_grid(microscope)
    microscope.specimen.stage.absolute_move(landing_coord)
    # realign landing post
    # move needle + lamella in
    # weld lamella to post
    # cut off needle + retract it


def single_liftout(microscope, lamella_coord, landing_coord):
    needle_reference_imgs = needle_reference_images(microscope)
    move_to_sample_grid(microscope)
    move_to_trenching_angle(microscope)
    microscope.specimen.stage.absolute_position(lamella_coord)
    mill_lamella(microscope)
    liftout_lamella(microscope, needle_reference_imgs)
    land_lamella(microscope)


# Start liftout for each lamella
for lamella_coord, landing_coord in zipped_coordinates:
    single_liftout(microscope, lamella_coord, landing_coord)
print("Finished.")
