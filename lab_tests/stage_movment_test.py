import logging

from liftout.main import initialize, configure_logging
from liftout.acquire import new_electron_image, new_ion_image
from liftout.align import realign_sample_stage, _calculate_beam_shift
from liftout.stage_movement import (flat_to_electron_beam,
                                    flat_to_ion_beam,
                                    x_corrected_stage_movement,
                                    y_corrected_stage_movement,
                                    z_corrected_stage_movement,
                                    _correct_y_stage_shift,
                                    _calculate_beam_shift)

from autoscript_sdb_microscope_client.structures import StagePosition


def check_stage_movement_functions(microscope):
    logging.info("Start check_stage_movement_functions")
    stage = microscope.specimen.stage
    flat_to_ion_beam(stage)
    eb = new_electron_image(microscope)
    eb.save('EB-stage-movement-test-original-image.tif')
    ib = new_ion_image(microscope)
    ib.save('IB-stage-movement-test-original-image.tif')
    stage_position = microscope.specimen.stage.current_position
    logging.info("Stage position:", stage_position)
    stage_tilt = microscope.specimen.stage.current_position.t

    # Move +500 pixels in y
    pixelsize = eb.metadata.binary_result.pixel_size.x
    move_amount = pixelsize * 500

    move = y_corrected_stage_movement(move_amount, stage_tilt)
    logging.info("Calculated stage movement:", move)
    import pdb; pdb.set_trace()
    microscope.specimen.stage.relative_move(move)

    eb2 = new_electron_image(microscope)
    eb2.save('EB-stage-movement-test-final-image.tif')
    ib2 = new_ion_image(microscope)
    ib2.save('IB-stage-movement-test-final-image.tif')
    logging.info("Finished check_stage_movement_functions")
    logging.info("-----------------------------")


def check_stage_alignment_functions(microscope):
    logging.info("Start check_stage_alignment_functions")
    stage = microscope.specimen.stage
    flat_to_ion_beam(stage)
    reference_image = new_electron_image(microscope)
    new_image.save('EB-stage-alignment-test-original.tif')
    # Move the stage a known amount
    known_movement = StagePosition(x=, y=, z=)
    logging.info(known_movement)
    microscope.specimen.stage.relative_move(known_movement)
    new_image = new_electron_image(microscope)
    new_image.save('EB-stage-alignment-test-shifted.tif')
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
    realigned_image.save('EB-stage-alignment-test-realigned.tif')
    logging.info("Finished check_stage_alignment_functions")
    logging.info("-----------------------------")


def main():
    configure_logging("log-stage-movement-and-alignemnt-test")
    microscope = initialize()
    check_stage_movement_functions(microscope)
    check_stage_alignment_functions(miicoscope)
    print("Finished.")


if __name__=="__main__":
    main()
