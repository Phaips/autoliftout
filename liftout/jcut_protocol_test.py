"""Mill lamella trenches and J-cut, using fiducial marker for realignment."""
import logging
import time

import numpy as np

from autoscript_sdb_microscope_client.structures import *
from autoscript_sdb_microscope_client.enumerations import *

from acquire import grab_ion_image, create_camera_settings
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
    flat_to_electron_beam,
    flat_to_ion_beam,
    move_to_jcut_position,
    move_to_trenching_position,
    move_to_liftout_position)


def setup_imaging_parameters(microscope):
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
    microscope.imaging.match_template(image, template_image)
    # Realign with fiducial marker


def realign_fiducial_for_jcut(microscope):
    # Change back to jcut ion beam current
    # Take sub-section of image where we expect fiducial marker to be
    # Realign with fiducial marker


def main():
    microscope = initialize()
    assert microscope.specimen.stage.is_linked
    assert np.isclose(0, microscope.beams.electron_beam.scanning.rotation.value)
    assert np.isclose(0, microscope.beams.ion_beam.scanning.rotation.value)
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator
    multichem = microscope.gas.get_multichem()

    # Fiducial marker
    jcut_milling_current = 0.74e-9  # in Amps
    microscope.beams.ion_beam.beam_current.value = jcut_milling_current
    move_to_jcut_position(stage)
    ask_user("Have you centered the lamella position? yes/no")
    mill_fiducial_marker(microsope, milling_current=jcut_milling_current)

    # Move to trench position
    trench_milling_current = 7.4e-9  # in Amps
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
