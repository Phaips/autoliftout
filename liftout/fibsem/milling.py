import logging
from autoscript_core.common import ApplicationServerException


def mill_trenches(microscope, settings, confirm=True):
    """Mill the trenches for thinning the lamella.
    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings :  Dictionary of user input argument settings.
    confirm : bool, optional
        Whether to ask the user to confirm before milling.
    """
    protocol_stages = protocol_stage_settings(settings)
    for stage_number, stage_settings in enumerate(protocol_stages):
        print("Protocol stage {} of {}".format(
            stage_number + 1, len(protocol_stages)))
        mill_single_stage(microscope, settings, stage_settings, stage_number)
    # Restore ion beam imaging current (20 pico-Amps)
    microscope.beams.ion_beam.beam_current.value = 30e-12  # TODO: add to protocol?


def mill_single_stage(microscope, settings, stage_settings, stage_number):
    """Run ion beam milling for a single milling stage in the protocol.
    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings :  Dictionary of user input argument settings.
    stage_settings : Dictionary of settings for a single protocol milling stage
    stage_number : int. Current milling protocol stage number.
    """
    logging.info(f'Milling trenches, protocol stage {stage_number}')
    demo_mode = settings["demo_mode"]
    lamella_region_milling(microscope, settings, stage_settings, region='upper', demo_mode=demo_mode)
    lamella_region_milling(microscope, settings, stage_settings, region='lower', demo_mode=demo_mode)


def _upper_milling_coords(microscope, stage_settings):
    """Create cleaning cross section milling pattern above lamella position."""
    microscope.imaging.set_active_view(2)  # the ion beam view
    lamella_center_x = 0
    lamella_center_y = 0
    milling_depth = stage_settings["milling_depth"]
    center_y = (
        lamella_center_y
        + (0.5 * stage_settings["lamella_height"])
        + (
            stage_settings["total_cut_height"]
            * stage_settings["percentage_from_lamella_surface"]
        )
        + (
            0.5
            * stage_settings["total_cut_height"]
            * stage_settings["percentage_roi_height"]
        )
    )
    height = float(
        stage_settings["total_cut_height"] *
        stage_settings["percentage_roi_height"]
    )
    milling_roi = microscope.patterning.create_cleaning_cross_section(
        lamella_center_x,
        center_y,
        stage_settings["lamella_width"],
        height,
        milling_depth,
    )
    milling_roi.scan_direction = "TopToBottom"
    return milling_roi


def _lower_milling_coords(microscope, stage_settings):
    """Create cleaning cross section milling pattern below lamella position."""
    microscope.imaging.set_active_view(2)  # the ion beam view
    lamella_center_x = 0
    lamella_center_y = 0
    milling_depth = stage_settings["milling_depth"]
    center_y = (
        lamella_center_y
        - (0.5 * stage_settings["lamella_height"])
        - (
            stage_settings["total_cut_height"]
            * stage_settings["percentage_from_lamella_surface"]
        )
        - (
            0.5
            * stage_settings["total_cut_height"]
            * stage_settings["percentage_roi_height"]
        )
    )
    height = float(
        stage_settings["total_cut_height"] *
        stage_settings["percentage_roi_height"]
    )
    milling_roi = microscope.patterning.create_cleaning_cross_section(
        lamella_center_x,
        center_y,
        stage_settings["lamella_width"],
        height,
        milling_depth,
    )
    milling_roi.scan_direction = "BottomToTop"
    return milling_roi


def lamella_region_milling(microscope, settings, stage_settings, region, demo_mode=False):
    # Setup and realign to fiducial marker
    setup_milling(microscope, settings, stage_settings)
    # Create and mill patterns
    if region == 'lower':
        _lower_milling_coords(microscope, stage_settings)
    elif region == 'upper':
        _upper_milling_coords(microscope, stage_settings)
    if not demo_mode:
        print("Milling pattern...")
        microscope.imaging.set_active_view(2)  # the ion beam view
        try:
            microscope.patterning.run()
        except ApplicationServerException:
            logging.error("ApplicationServerException: could not mill!")
    microscope.patterning.clear_patterns()
    return microscope


def setup_milling(microscope, settings, stage_settings):
    """Setup the ion beam system ready for milling.
    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings :  Dictionary of user input argument settings.
    stage_settings : Dictionary of settings for a single protocol milling stage
    Returns
    -------
    Autoscript microscope object.
    """
    ccs_file = settings['system']["application_file_cleaning_cross_section"]
    microscope = reset_state(microscope, settings, application_file=ccs_file)
    microscope.beams.ion_beam.beam_current.value = stage_settings["milling_current"]
    return microscope


def reset_state(microscope, settings, application_file=None):
    """Reset the microscope state.
    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings :  Dictionary of user input argument settings.
    application_file : str, optional
        Name of the application file for milling, by default None
    """
    microscope.patterning.clear_patterns()
    if application_file:  # optionally specified
        microscope.patterning.set_default_application_file(application_file)
    resolution = settings["imaging"]["resolution"]
    dwell_time = settings["imaging"]["dwell_time"]
    hfw = settings["imaging"]["horizontal_field_width"]
    microscope.beams.ion_beam.scanning.resolution.value = resolution
    microscope.beams.ion_beam.scanning.dwell_time.value = dwell_time
    microscope.beams.ion_beam.horizontal_field_width.value = hfw
    microscope.imaging.set_active_view(2)  # the ion beam view
    return microscope


def protocol_stage_settings(settings):
    """"Load settings for each milling stage, overwriting default values.

    Parameters
    ----------
    settings :  Dictionary of user input argument settings.

    Returns
    -------
    protocol_stages
        List containing a dictionary of settings for each protocol stage.
    """
    protocol_stages = []
    for stage_settings in settings["lamella"]["protocol_stages"]:
        tmp_settings = settings["lamella"].copy()
        tmp_settings.update(stage_settings)
        protocol_stages.append(tmp_settings)
    return protocol_stages
