from liftout.acquire import (new_electron_image,
                             new_ion_image,
                             autocontrast,
                             BeamType)
from liftout.calibration import auto_link_stage


def create_lamella(microscope):
    stage = microscope.specimen.stage
    # Set the correct magnification / field of view
    field_of_view = 100e-6  # in meters
    microscope.beams.ion_beam.horizontal_field_width.value = field_of_view
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
    # Move to trench position
    move_to_trenching_angle(stage)
    auto_link_stage(microscope)
    # Take an ion beam image at the *milling current*
    ib = new_ion_image(microscope)
    if ask_user("Have you centered the lamella location? yes/no"):
        continue
    else:
        print("Ok, cancelling trench milling.")
        return
    mill_trenches()
    ib_original = new_ion_image(microscope, settings=)
    template = create_reference_image(ib_original)
    # Move to Jcut angle and take electron beam image
    move_to_jcut_angle(stage)
    autocontrast(microscope)
    image = new_electron_image(microscope, settings=)
    location = match_locations(microscope, image, template)
    # Visualize
    import autoscript_toolkit.vision as vision_toolkit
    vision_toolkit.plot_match(image, template, location.center_in_pixels)
    plot_expected_alignment(location, image, template)
    # Realign
    realign_hog_matcher(microscope, location)
    eb = new_electron_image(microscope)


def realign_hog_matcher(microscope):
    stage = microscope.specimen.stage
    x_move = x_corrected_stage_movement(location.center_in_meters.x)
    y_move = y_corrected_stage_movement(location.center_in_meters.y,
                                        stage.current_position.t,
                                        beam_type=BeamType.ELECTRON)
    logging.info(x_move)
    logging.info(y_move)
    stage.relative_move(x_move)
    stage.relative_move(y_move)


def create_reference_image(image):
    from autoscript_sdb_microscope_client.structures import AdornedImage

    data = np.rot90(np.rot90(np.copy(image.data)))
    reference = AdornedImage(data=data)
    reference.metadata = image.metadata
    return reference


def match_locations(microscope, image, template):
    import autoscript_toolkit.vision as vision_toolkit
    from autoscript_toolkit.template_matchers import HogMatcher

    hog_matcher = HogMatcher(microscope)
    original_feature_center = list(np.flip(np.array(template.data.shape)//2))
    location = vision_toolkit.locate_feature(image, template, hog_matcher, original_feature_center=original_feature_center)
    location.print_all_information()  # displays in x-y coordinate order
    return location


def plot_overlaid_images(image_1, image_2)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(image_1, cmap='Blues_r', alpha=1)
    ax.imshow(image_2, cmap='Oranges_r', alpha=0.5)
    plt.show()


def plot_expected_alignment(location, image, template):
    aligned = np.copy(image.data)
    aligned = np.roll(aligned, -int(location.shift_in_pixels.y), axis=0)
    aligned = np.roll(aligned, -int(location.shift_in_pixels.x), axis=1)
    view_overlaid_images(image.data, aligned)