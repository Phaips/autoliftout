"""Mill lamella trenches and J-cut, using fiducial marker for realignment."""
import logging
import time

import numpy as np

from autoscript_sdb_microscope_client.structures import *
from autoscript_sdb_microscope_client.enumerations import *

from liftout.acquire import new_ion_image, new_electron_image, create_camera_settings
from lifout.calibration import autocontrast, auto_link_stage
from lifout.jcut import mill_trenches, mill_jcut, mill_to_sever_jcut
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


def view_overlaid_images(image_1, image_2)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(image_1, cmap='Blues_r', alpha=1)
    ax.imshow(image_2, cmap='Oranges_r', alpha=0.5)
    plt.show()


def view_expected_alignment(location, image, template):
    aligned = np.copy(image.data)
    aligned = np.roll(aligned, -int(location.shift_in_pixels.y), axis=0)
    aligned = np.roll(aligned, -int(location.shift_in_pixels.x), axis=1)
    view_overlaid_images(image.data, aligned)


def main():
    microscope = initialize()
    assert microscope.specimen.stage.is_linked
    assert np.isclose(0, microscope.beams.electron_beam.scanning.rotation.value)
    assert np.isclose(0, microscope.beams.ion_beam.scanning.rotation.value)

    # Set the correct magnification / field of view
    field_of_view = 100e-6  # in meters
    microscope.beams.ion_beam.horizontal_field_width.value = field_of_view
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
    # Move to trench position
    move_to_trenching_angle(stage)
    ask_user("Have you centered the lamella position? yes/no")
    auto_link_stage(microscope)
    # Mill trenches
    trench_milling_current = 7.4e-9  # in Amps, for cryo
    # trench_milling_current = 2e-9  # in Amps, at room temperature
    # TODO: mill trenches

    # Take ion image
    settings = # microscope settings
    ib_original = new_ion_image(microscope)

    # Create reference image
    template = create_reference_image(ib_original)
    # Move to Jcut angle and take electron beam image
    move_to_jcut_angle(stage)
    autocontrast(microscope)
    settings = # microscope settings
    image = new_electron_image(microscope)

    # Calculate alignment, show image
    location = match_locations(microscope, image, template)
    import autoscript_toolkit.vision as vision_toolkit
    vision_toolkit.plot_match(image, template, location.center_in_pixels)
    view_expected_alignment(location, image, template)
    # Realign with stage motors
    stage = microscope.specimen.stage
    x_move = x_corrected_stage_movement(location.center_in_meters.x)
    y_move = y_corrected_stage_movement(location.center_in_meters.y,
                                        stage.current_position.t,
                                        beam_type=BeamType.ELECTRON)
    stage.relative_move(x_move)
    eb = new_electron_image(microscope)
    import pdb; pdb.set_trace()
    stage.relative_move(y_move)
    eb = new_electron_image(microscope)
    view_overlaid_images(template.data, eb.data)
    # Mill Jcut pattern
    jcut_milling_current = 0.74e-9  # in Amps, for cryo
    # jcut_milling_current = 2e-9  # in Amps, at room temperature (same as trench current)
    microscope.beams.ion_beam.beam_current.value = jcut_milling_current
    ib = new_ion_image(microscope)
    mill_jcut(microscope, milling_current=jcut_milling_current)
    # Tilt stage flat to electron beam, so we are ready for liftout
    move_to_liftout_angle(stage)
    print("Done, ready for liftout!")


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
