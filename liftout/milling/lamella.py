"""Liftout sample preparation, combined trench and J-cut milling of lamellae"""
import logging

import matplotlib.pyplot as plt
import numpy as np

from liftout.acquire import (new_electron_image,
                             new_ion_image,
                             autocontrast,
                             BeamType)
from liftout.align.hog_template_matching import (create_reference_image,
                                                 match_locations,
                                                 realign_hog_matcher)
from liftout.milling.trenches import mill_trenches
from liftout.milling.jcut import mill_jcut
from liftout.stage_movement import move_to_trenching_angle


def mill_lamella(microscope, settings):
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings

    stage = microscope.specimen.stage
    # Set the correct magnification / field of view
    field_of_view = 100e-6  # in meters  TODO: user input from yaml settings
    microscope.beams.ion_beam.horizontal_field_width.value = field_of_view
    microscope.beams.electron_beam.horizontal_field_width.value = field_of_view
    # Move to trench position
    move_to_trenching_angle(microscope)
    # Take an ion beam image at the *milling current*
    ib = new_ion_image(microscope)
    mill_trenches(microscope, settings, condifrm=True)
    image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    ib_original = new_ion_image(microscope, settings=image_settings)
    template = create_reference_image(ib_original)
    # Move to Jcut angle and take electron beam image
    move_to_jcut_angle(microscope)
    autocontrast(microscope)
    # Realign first to the electron beam image
    image = new_electron_image(microscope, settings=image_settings)
    location = match_locations(microscope, image, template)
    realign_hog_matcher(microscope, location)
    eb = new_electron_image(microscope, settings=image_settings)
    # Fine tune alignment of ion beam image
    image = new_ion_image(microscope, settings=image_settings)
    location = match_locations(microscope, image, template)
    realign_hog_matcher(microscope, location)
    ib = new_ion_image(microscope, settings=image_settings)
    eb = new_electron_image(microscope, settings=image_settings)
    # Mill J-cut
    mill_jcut(microscope, settings['jcut'])
    final_ib = new_ion_image(microscope, settings=image_settings)
    final_eb = new_electron_image(microscope, settings=image_settings)
    return (final_eb, final_ib, eb, ib)
