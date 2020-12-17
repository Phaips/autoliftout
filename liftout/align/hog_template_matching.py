"""Alignment functions using histogram of gaussians (HOG) temlate matching."""
import logging

import numpy as np

from liftout.display import plot_overlaid_images
from liftout.stage_movement import (x_corrected_stage_movement,
                                    y_corrected_stage_movement,
                                    BeamType)

__all__ = [
    "create_reference_image",
    "match_locations",
    "realign_hog_matcher",
    "plot_expected_alignment",
]


def create_reference_image(image):
    """Create a reference template image for feature matching and alignment.

    Because we take the first ion beam image then rotate the sample stage by
    180 degrres, we need to *also* rotate the ion beam image data by 180 deg
    to create the refernce template to match new images with.

    Parameters
    ----------
    image : autoscript_sdb_microscope_client.structures.AdornedImage
        The ion beam image to rotate and use as a template for HOG matching.

    Returns
    -------
    AdornedImage
        Image with data rotated 180 degrees, and identical metadata as input.
    """
    from autoscript_sdb_microscope_client.structures import AdornedImage

    data = np.rot90(np.rot90(np.copy(image.data)))
    reference = AdornedImage(data=data)
    reference.metadata = image.metadata
    return reference


def match_locations(microscope, image, template):
    """Find the matched location between an image and a reference template.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    image : autoscript_sdb_microscope_client.structures.AdornedImage
        The image to compare with the reference template.
    template : autoscript_sdb_microscope_client.structures.AdornedImage
        The reference template image.

    Returns
    -------
    FeatureLocation
        AutoScript feature location object.
    """
    import autoscript_toolkit.vision as vision_toolkit
    from autoscript_toolkit.template_matchers import HogMatcher

    hog_matcher = HogMatcher(microscope)
    original_feature_center = list(np.flip(np.array(template.data.shape)//2, axis=0))
    location = vision_toolkit.locate_feature(image, template, hog_matcher, original_feature_center=original_feature_center)
    location.print_all_information()  # displays in x-y coordinate order
    return location


def realign_hog_matcher(microscope, location):
    """Realisn the image to a given location by shifting the sample stage.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    location : FeatureLocation
        AutoScript feature location object.
    """
    stage = microscope.specimen.stage
    x_move = x_corrected_stage_movement(location.center_in_meters.x)
    y_move = y_corrected_stage_movement(location.center_in_meters.y,
                                        stage.current_position.t,
                                        beam_type=BeamType.ELECTRON)
    logging.info(x_move)
    logging.info(y_move)
    stage.relative_move(x_move)
    stage.relative_move(y_move)


def plot_expected_alignment(location, image, template):
    """Plot the expected alignment, given the matched feature location.

    Parameters
    ----------
    location : FeatureLocation
        AutoScript feature location object.
    image : autoscript_sdb_microscope_client.structures.AdornedImage
        The image to compare with the reference template.
    template : autoscript_sdb_microscope_client.structures.AdornedImage
        The reference template image.
    """
    aligned = np.copy(image.data)
    aligned = np.roll(aligned, -int(location.shift_in_pixels.y), axis=0)
    aligned = np.roll(aligned, -int(location.shift_in_pixels.x), axis=1)
    plot_overlaid_images(image.data, aligned, show=True)
