"""Mill lamella trenches and J-cut, using fiducial marker for realignment."""
import logging
import time

import numpy as np

from autoscript_sdb_microscope_client.structures import *
from autoscript_sdb_microscope_client.enumerations import *

from liftout.acquire import new_ion_image, new_electron_image, create_camera_settings
from liftout.align import realign_sample_stage
from lifout.calibration import autocontrast, auto_link_stage
from liftout.milling.jcut import mill_jcut, mill_to_sever_jcut
from liftout.milling.trenches import mill_trenches
from liftout.stage_movement import (
    move_to_jcut_angle,
    move_to_trenching_angle,
    move_to_liftout_angle)


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


def plot_overlaid_images(image_1, image_2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(image_1, cmap='Blues_r', alpha=1)
    ax.imshow(image_2, cmap='Oranges_r', alpha=0.5)
    plt.show()


def plot_expected_alignment(location, image, template):
    aligned = np.copy(image.data)
    aligned = np.roll(aligned, -int(location.shift_in_pixels.y), axis=0)
    aligned = np.roll(aligned, -int(location.shift_in_pixels.x), axis=1)
    plot_overlaid_images(image.data, aligned)


def setup(microscope, settings):
    assert microscope.specimen.stage.is_linked
    assert np.isclose(0, microscope.beams.electron_beam.scanning.rotation.value)
    assert np.isclose(0, microscope.beams.ion_beam.scanning.rotation.value)
    # Set the correct magnification / field of view
    field_of_view = settings['imaging']['horizontal_field_width']
    microscope.beams.ion_beam.horizontal_field_width.value = field_of_view
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
    # Imaging setup
    image_settings = GrabFrameSettings(
        resolution=settings['imaging']['resolution'],
        dwell_time=settings['imaging']['dwell_time'])
    return image_settings


def realign_hog_matcher(microscope, location):
    stage = microscope.specimen.stage
    x_move = x_corrected_stage_movement(location.center_in_meters.x)
    y_move = y_corrected_stage_movement(location.center_in_meters.y,
                                        stage.current_position.t,
                                        beam_type=BeamType.ELECTRON)
    logging.info(x_move)
    logging.info(y_move)
    stage.relative_move(x_move)
    stage.relative_move(y_move)


def mill_lamella(microscope, settings):
    move_to_trenching_angle(microscope)
    mill_trenches(microscope, settings)
    # Always put the field of view back to where it should be before aligning
    field_of_view = settings['imaging']['horizontal_field_width']
    microscope.beams.ion_beam.horizontal_field_width.value = field_of_view
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
    # Take ion image and create a reference template
    ib_original = new_ion_image(microscope, settings=image_settings)
    template = create_reference_image(ib_original)
    move_to_jcut_angle(microscope)
    # Realign first to the electron beam image
    image = new_electron_image(microscope, settings=image_settings)
    location = match_locations(microscope, image, template)
    realign_hog_matcher(microscope, location)
    eb = new_electron_image(microscope, settings=image_settings)
    # Fine tune alignment of ion beam image?
    image = new_ion_image(microscope, settings=image_settings)
    realign_sample_stage(microscope, image, template)
    ib = new_ion_image(microscope, settings=image_settings)
    eb = new_electron_image(microscope, settings=image_settings)
    # Make the Jcut
    mill_jcut(microscope, settings['jcut'])
    move_to_liftout_angle(stage)
    print("Done, ready for liftout!")


def main():
    # ASSUMES SAMPLE STAGE IS ALREADY EUCENTRIC
    microscope = initialize()
    config_filename = '..\\protocol_liftout.yml'
    settings = load_config(config_filename)
    image_settings = setup(microscope, settings)
    mill_lamella(microscope, settings)



if __name__ == "__main__":
    message = """
    Have you double-checked that:
    * The computcentric rotation calibration has been done recently
    * The scan rotation of the microscope is zero
    * The needle park position has been calibrated
    * The sample is at eucentric height
    * The sample is in focus
    * The z-height has been linked accurately
    \nPlease answer yes/no\n
    """
    user_input = input(message)
    if user_input == 'yes':
        try:
            main()
        except KeyboardInterrupt:
            print("KeyboardInterrupt encountered, quitting program.")
    else:
        print("Ok, cancelling program.")
