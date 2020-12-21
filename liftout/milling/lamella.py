"""Liftout sample preparation, combined trench and J-cut milling of lamellae"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

from liftout.acquire import (new_electron_image,
                             new_ion_image,
                             autocontrast,
                             BeamType)
from liftout.align.hog_template_matching import (create_reference_image,
                                                 match_locations,
                                                 realign_hog_matcher)
from liftout.milling.trenches import mill_trenches
from liftout.milling.jcut import mill_jcut
from liftout.stage_movement import (move_to_trenching_angle,
                                    move_to_jcut_angle,
                                    move_to_liftout_angle)


def mill_lamella(microscope, settings, confirm=True):
    from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings

    stage = microscope.specimen.stage
    # Set the correct magnification / field of view
    field_of_view = 100e-6  # in meters  TODO: user input from yaml settings
    microscope.beams.ion_beam.horizontal_field_width.value = field_of_view
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
    # Move to trench position
    move_to_trenching_angle(microscope)
    # Take an ion beam image at the *milling current*
    ib = new_ion_image(microscope)
    mill_trenches(microscope, settings, confirm=confirm)
    autocontrast(microscope, beam_type=BeamType.ION)
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution
    ib_original = new_ion_image(microscope, settings=image_settings)
    template = create_reference_image(ib_original)
    # Low res template image
    scaling_factor = 4
    microscope.beams.ion_beam.horizontal_field_width.value = field_of_view * scaling_factor
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view * scaling_factor
    ib_lowres_original = new_ion_image(microscope, settings=image_settings)
    lowres_template = AdornedImage(data=np.rot90(np.rot90(ib_lowres_original.data)))
    lowres_template.metadata = ib_lowres_original.metadata
    # Move to Jcut angle and take electron beam image
    move_to_jcut_angle(microscope)
    # Low res resolution
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    microscope.beams.ion_beam.horizontal_field_width.value = field_of_view * scaling_factor
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view * scaling_factor
    image = new_electron_image(microscope, settings=image_settings)
    location = match_locations(microscope, image, lowres_template)
    realign_hog_matcher(microscope, location)
    eb = new_electron_image(microscope, settings=image_settings)
    # Realign first to the electron beam image
    autocontrast(microscope, beam_type=BeamType.ELECTRON)
    microscope.beams.ion_beam.horizontal_field_width.value = field_of_view
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
    image = new_electron_image(microscope, settings=image_settings)
    location = match_locations(microscope, image, template)
    realign_hog_matcher(microscope, location)
    eb = new_electron_image(microscope, settings=image_settings)
    # Fine tune alignment of ion beam image
    autocontrast(microscope, beam_type=BeamType.ION)
    image = new_ion_image(microscope, settings=image_settings)
    location = match_locations(microscope, image, template)
    realign_hog_matcher(microscope, location)
    ib = new_ion_image(microscope, settings=image_settings)
    eb = new_electron_image(microscope, settings=image_settings)
    # Mill J-cut
    mill_jcut(microscope, settings['jcut'], confirm=False)
    final_ib = new_ion_image(microscope, settings=image_settings)
    final_eb = new_electron_image(microscope, settings=image_settings)
    # Ready for liftout
    move_to_liftout_angle(microscope)
    print("Done, ready for liftout!")
