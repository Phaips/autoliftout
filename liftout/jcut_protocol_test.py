"""Mill lamella trenches and J-cut, using fiducial marker for realignment."""
import logging
import time

import numpy as np

from autoscript_sdb_microscope_client.structures import *
from autoscript_sdb_microscope_client.enumerations import *

from acquire import new_ion_image, new_electron_image, create_camera_settings
from jcut import (
    mill_fiducial_marker,
    mill_trenches,
    mill_jcut,
    mill_to_sever_jcut)
from needle_movement import (
    insert_needle,
    retract_needle,
    sputter_platinum)
from stage_movement import (
    PRETILT_DEGREES,
    flat_to_electron_beam,
    flat_to_ion_beam,
    move_to_jcut_position,
    move_to_trenching_position,
    move_to_liftout_position)


def zero_beam_shift(microscope, *,
                    zero_electron_beam=True,
                    zero_ion_beam=True):
    from autoscript_sdb_microscope_client.structures import Point

    if zero_electron_beam:
        microscope.beams.electron_beam.beam_shift.value = Point(0, 0)
    if zero_ion_beam:
        microscope.beams.ion_beam.beam_shift.value = Point(0, 0)


def setup_imaging_parameters(microscope):
    zero_beam_shift(microscope)
    # set resolution  - "1536x1024"
    # set magnification -
    # set field of view


def realign_fiducial_for_trenches(microscope):
    # change ion beam current for trenches
    from autoscript_sdb_microscope_client.structures import (GrabFrameSettings,
                                                             Rectangle)

    if reduced_area is None:
        reduced_area = Rectangle(0, 0, 1, 1)
    settings = GrabFrameSettings(
        resolution=imaging_settings["resolution"],
        dwell_time=imaging_settings["dwell_time"],
        reduced_area=reduced_area,
    )
    ion_image = grab_ion_image(settings)

    # Take sub-section of image where we expect fiducial marker to be
    match = microscope.imaging.match_template(image, template_image)
    # Realign with fiducial marker


def realign_fiducial_for_jcut(microscope):
    # Change back to jcut ion beam current
    # Take sub-section of image where we expect fiducial marker to be
    # Realign with fiducial marker
    pass


def main():
    microscope = initialize()
    assert microscope.specimen.stage.is_linked
    assert np.isclose(0, microscope.beams.electron_beam.scanning.rotation.value)
    assert np.isclose(0, microscope.beams.ion_beam.scanning.rotation.value)
    zero_beam_shift(microscope)
    assert microscope.beams.electron_beam.beam_shift.value == Point(0, 0)
    assert microscope.beams.ion_beam.beam_shift.value == Point(0, 0)

    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator
    multichem = microscope.gas.get_multichem()

    # Set the correct magnification / field of view for the ion beam
    ion_beam_field_of_view = 104e-6  # 82.9e-6  # in meters
    microscope.beams.ion_beam.horizontal_field_width.value = ion_beam_field_of_view

    # Move to trench position
    trench_milling_current = 7.4e-9  # in Amps, for cryo
    # trench_milling_current = 2e-9  # in Amps, at room temperature
    move_to_trenching_position(stage)
    ask_user("Have you centered the lamella position? yes/no")
    synthetic_image, original_location_xy = mill_fiducial_marker(microscope, milling_current=trench_milling_current)

    jcut_milling_current = 0.74e-9  # in Amps, for cryo
    # jcut_milling_current = 2e-9  # in Amps, at room temperature (same as trench current)
    microscope.beams.ion_beam.beam_current.value = jcut_milling_current
    move_to_jcut_position(stage)

    ib = new_ion_image(microscope)
    match = microscope.imaging.match_template(ib, synthetic)
    expected_location_x = ib.width - original_location_x
    expected_location_y = ib.height - original_location_y
    pixelsize = original_image.metadata.binary_result.pixel_size.x
    x_pixel_difference = expected_location_x - match.center.x
    shift_in_x = pixelsize * x_pixel_difference  # real space, in meters
    x_move = StagePosition(x=-shift_in_x)
    stage.relative_move(x_move)

    sample_surface_angle_to_sem = 6  # in degrees
    sample_surface_angle = 52 - sample_surface_angle_to_sem

    y_pixel_difference = expected_location_y - match.center.y
    shift_in_y = (pixelsize * y_pixel_difference *
                np.cos(np.deg2rad(sample_surface_angle)))
    shift_in_z = (pixelsize * y_pixel_difference *
                np.sin(np.deg2rad(sample_surface_angle)))
    y_move = StagePosition(x=0, y=+shift_in_y, z=-shift_in_z)
    # stage.relative_move(y_move)

    # Move to trench position
    move_to_trenching_position(stage)
    realign_fiducial_for_trenches(microscope, milling_current=trench_milling_current)
    mill_trenches(microscope, milling_current=trench_milling_current)

    # Return to jcut angle
    move_to_jcut_position(stage)
    realign_fiducial_for_jcut(microscope, milling_current=jcut_milling_current)
    mill_jcut(microscope, milling_current=jcut_milling_current)

    # Tilt stage flat to electron beam, so we are ready for liftout
    move_to_liftout_position(stage)
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
