from .acquire import new_electron_image, new_ion_image
from .needle_movement import insert_needle, retract_needle
from .stage_movement import move_sample_stage_out


__all__ = [
    "needle_with_blank_background",
    "segment_needle",
    "locate_needle",
]


def needle_with_blank_background(microscope, *,
                                 acquire_ion_image=True,
                                 acquire_electron_image=True):
    original_stage_position = microscope.specimen.stage.current_position
    move_sample_stage_out(microscope)
    park_position = insert_needle(microscope)
    move_needle_closer(microscope.specimen.manipulator)
    if acquire_ion_image is True:
        electron_image = new_electron_image(microscope)
    else:
        electron_image = None
    if acquire_electron_image is True:
        ion_image = new_ion_image(microscope)
    else:
        ion_image = None
    retract_needle(microscope, park_position)
    microscope.specimen.stage.absolute_move(original_stage_position)
    return electron_image, ion_image


def segment_needle(image):
    raise NotImplementedError


def locate_needle(image):
    raise NotImplementedError
