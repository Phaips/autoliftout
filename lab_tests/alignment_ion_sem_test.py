from copy import deepcopy
import logging

import numpy as np

from liftout.main import initialize, configure_logging
from liftout.stage_movement import flat_to_electron_beam, flat_to_ion_beam
from liftout.acquire import new_electron_image, new_ion_image
from liftout.align import (realign_beam_shift,
                           realign_sample_stage,
                           _calculate_beam_shift,
                           _correct_y_stage_shift)


def main():
    import pdb;.pdb.set_trace()
    configure_logging('alignment_ion_sem_test')
    logging.info("Start alignment-ion-sem-test")
    microscope = initialize()
    flat_to_ion_beam(microscope)
    original_image = new_ion_image(microscope)
    original_image.save('0-IB-alignment-test-original.tif')
    original_stage_position = microscope.specimen.stage.current_position
    logging.info("original_stage_position:", original_stage_position)
    # Move flat to electron beam
    flat_to_electron_beam(microscope)
    new_image = new_electron_image(microscope)
    new_image.save('1-EB-alignment-test-newimage.tif')
    shifted_stage_position = microscope.specimen.stage.current_position
    logging.info('shifted_stage_position:', shifted_stage_position)
    # Try to align using the rotated original ion beam image as reference
    reference_image = deepcopy(original_image)
    rotated_image_data = np.rot90(np.rot90(reference_image.data))
    reference_image.data = rotated_image_data
    reference_image.save('2-IB-alignment-test-rotated-reference-image.tif')
    # Calculate the difference between images
    x_shift, y_shift = _calculate_beam_shift(new_image, reference_image)
    logging.info("x_shift:", x_shift)
    logging.info("Uncorrected y_shift:", y_shift)
    y_shift, z_shift = _correct_y_stage_shift(microscope, )
    logging.info("Corrected y_shift:", y_shift)
    logging.info("Corrected z_shift:", z_shift)
    # Attempt to realign sample with these values
    realign_sample_stage(microscope, new_image, reference_image)
    realigned_image = new_electron_image(microscope)
    realigned_image.save('3-EB-alignment-test-realigned.tif')
    logging.info("Finished alignment-ion-sem-test")
    logging.info("-----------------------------")


if __name__ == "__main__":
    main()
