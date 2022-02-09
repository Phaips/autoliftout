import logging
from autoscript_core.common import ApplicationServerException
import numpy as np
import math

from liftout.fibsem import acquire, calibration
BeamType = acquire.BeamType


def mill_jcut(microscope, settings):
    """Create and mill the rectangle patter to sever the jcut completely.
    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    jcut_settings : dict
        Dictionary of J-cut parameter settings.
    confirm : bool, optional
        Whether to ask the user to confirm before milling.
    """
    jcut_patterns = jcut_milling_patterns(microscope, settings)
    return jcut_patterns


def jcut_severing_pattern(microscope, settings):
    """Create J-cut milling pattern in the center of the ion beam field of view.
    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    jcut_settings : dict
        Sample surface angle for J-cut in degrees, by default 6
    Returns
    -------
    autoscript_sdb_microscope_client.structures.RectanglePattern
        Rectangle milling pattern used to sever the remaining bit of the J-cut.
    """
    # Unpack settings
    jcut_angle_degrees = settings["jcut"]["jcut_angle"]
    jcut_lamella_depth = settings["jcut"]["jcut_lamella_depth"]
    jcut_length = settings["jcut"]["jcut_length"]
    jcut_trench_thickness = settings["jcut"]["jcut_trench_thickness"]
    jcut_milling_depth = settings["jcut"]["jcut_milling_depth"]
    extra_bit = settings["jcut"]["extra_bit"]
    # Setup
    setup_ion_milling(microscope)
    # Create milling pattern - right hand side of J-cut
    angle_correction_factor = np.sin(np.deg2rad(52 - jcut_angle_degrees))
    center_x = +((jcut_length - jcut_trench_thickness) / 2)
    center_y = (
        (jcut_lamella_depth - (extra_bit / 2)) / 2
    ) * angle_correction_factor  # noqa: E501
    width = jcut_trench_thickness
    height = (jcut_lamella_depth + extra_bit) * angle_correction_factor
    jcut_severing_pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, jcut_milling_depth
    )
    return jcut_severing_pattern


def run_milling(microscope, settings, *, imaging_current=20e-12, milling_current=None):
    if settings["imaging"]["imaging_current"]:
        imaging_current = settings["imaging"]["imaging_current"]
    logging.info("milling: running ion beam milling now...")
    microscope.imaging.set_active_view(2)  # the ion beam view
    if milling_current is None:
        microscope.beams.ion_beam.beam_current.value = settings["jcut"][
            "jcut_milling_current"
        ]
    else:
        microscope.beams.ion_beam.beam_current.value = milling_current
    microscope.patterning.run()
    logging.info("milling: returning to the ion beam imaging current now.")
    microscope.patterning.clear_patterns()
    microscope.beams.ion_beam.beam_current.value = imaging_current
    microscope.patterning.mode = "Serial"
    logging.info("milling: ion beam milling complete.")


def draw_patterns_and_mill(microscope, settings, patterns: list, depth: float, milling_current: float = None):
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.clear_patterns()
    for pattern in patterns:
        tmp_pattern = microscope.patterning.create_rectangle(
            pattern.center_x,
            pattern.center_y,
            pattern.width,
            pattern.height,
            depth=depth,
        )
        tmp_pattern.rotation = -np.deg2rad(pattern.rotation)
    run_milling(microscope=microscope, settings=settings, milling_current=milling_current)


def mill_thin_lamella(microscope, settings, image_settings, milling_type="thin", ref_image=None):
    """Align and mill thin lamella"""

    image_settings["save"] = False
    image_settings["hfw"] = 30e-6
    image_settings["beam_type"] = BeamType.ION
    image_settings["gamma"]["correction"] = False
    image_settings["save"] = True
    image_settings["label"] = f"thin_lamella_crosscorrelation_ref"

    # initial reference image
    if ref_image is None:
        ref_image = acquire.new_image(microscope, image_settings)
    #
    # # align using cross correlation
    # img1 = ref_image
    # image_settings["label"] = f"thinning_lamella_crosscorrelation_shift"
    # img2 = acquire.new_image(microscope, settings=image_settings)
    # dx, dy = calibration.shift_from_crosscorrelation_AdornedImages(
    #     img1, img2, lowpass=256, highpass=24, sigma=10, use_rect_mask=True
    # )
    #
    # # adjust beamshift
    # microscope.beams.ion_beam.beam_shift.value += (-dx, dy)
    #
    # # retake image
    # _ = acquire.new_image(microscope, image_settings)

    ##########

    # load protocol settings
    protocol_stages = []

    for stage_settings in settings["thin_lamella"]["protocol_stages"]:
        tmp_settings = settings["thin_lamella"].copy()
        tmp_settings.update(stage_settings)

        protocol_stages.append(tmp_settings)

    # FOR thinning, we do the first two stages, for polishing the last.
    if milling_type == "thin":
        protocol_stages = protocol_stages[:2]
    if milling_type == "polishing":
        protocol_stages = protocol_stages[-1]

    # mill lamella
    for stage_number, stage_settings in enumerate(protocol_stages):

        # setup milling (change current etc)
        setup_milling(microscope, settings, stage_settings)

        # align using cross correlation
        img1 = ref_image
        image_settings["label"] = f"{milling_type}_lamella_stage_{stage_number + 1}"
        img2 = acquire.new_image(microscope, settings=image_settings)
        dx, dy = calibration.shift_from_crosscorrelation_AdornedImages(
            img1, img2, lowpass=256, highpass=24, sigma=10, use_rect_mask=True
        )

        # adjust beamshift
        microscope.beams.ion_beam.beam_shift.value += (-dx, dy)
        _ = acquire.new_image(microscope, image_settings)

        logging.info(
            "milling: protocol stage {} of {}".format(
                stage_number + 1, len(protocol_stages)
            )
        )

        # Create and mill patterns

        # upper region
        """Create cleaning cross section milling pattern above lamella position."""
        microscope.imaging.set_active_view(2)  # the ion beam view
        lamella_center_x = - stage_settings["lamella_width"] * 0.5 + 0.25e-6
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
            stage_settings["total_cut_height"] * stage_settings["percentage_roi_height"]
        )
        upper_milling_roi = microscope.patterning.create_cleaning_cross_section(
            lamella_center_x,
            center_y,
            stage_settings["lamella_width"],
            height,
            milling_depth,
        )
        upper_milling_roi.scan_direction = "TopToBottom"

        # lower region
        """Create cleaning cross section milling pattern below lamella position."""
        microscope.imaging.set_active_view(2)  # the ion beam view
        lamella_center_x = - stage_settings["lamella_width"] * 0.5 + 0.25e-6
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
            stage_settings["total_cut_height"] * stage_settings["percentage_roi_height"]
        )
        lower_milling_roi = microscope.patterning.create_cleaning_cross_section(
            lamella_center_x,
            center_y,
            stage_settings["lamella_width"],
            height,
            milling_depth,
        )
        lower_milling_roi.scan_direction = "BottomToTop"

        # TODO: visualise the milling patterns?

        logging.info(f"milling: milling thin lamella pattern...")
        microscope.beams.ion_beam.horizontal_field_width.value = stage_settings[
            "hfw"
        ]
        microscope.imaging.set_active_view(2)  # the ion beam view
        _ = acquire.new_image(microscope, settings=image_settings)

        try:
            microscope.patterning.run()
        except ApplicationServerException:
            logging.error("ApplicationServerException: could not mill!")
        microscope.patterning.clear_patterns()

    # reset milling state and return to imaging current
    logging.info("returning to the ion beam imaging current now.")
    microscope.patterning.clear_patterns()
    microscope.beams.ion_beam.beam_current.value = settings["imaging"]["imaging_current"]
    microscope.patterning.mode = "Serial"
    logging.info("ion beam milling complete.")

    # take final reference image
    image_settings["label"] = f"{milling_type}_lamella_final"
    _ = acquire.new_image(microscope, settings=image_settings)
    logging.info("Thin Lamella Finished.")

    return


def mill_trenches(microscope, settings):
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
        logging.info(
            "milling: protocol stage {} of {}".format(
                stage_number + 1, len(protocol_stages)
            )
        )
        mill_single_stage(microscope, settings, stage_settings, stage_number)

    # Restore ion beam imaging current (20 pico-Amps)
    logging.info(f"mill trenches complete, returning to imaging current")
    microscope.beams.ion_beam.beam_current.value = settings["imaging"][
        "imaging_current"
    ]


def mill_single_stage(microscope, settings, stage_settings, stage_number):
    """Run ion beam milling for a single milling stage in the protocol.
    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings :  Dictionary of user input argument settings.
    stage_settings : Dictionary of settings for a single protocol milling stage
    stage_number : int. Current milling protocol stage number.
    """
    # logging.info(f'Milling trenches, protocol stage {stage_number+1}')
    lamella_region_milling(microscope, settings, stage_settings, region="upper")
    lamella_region_milling(microscope, settings, stage_settings, region="lower")


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
        stage_settings["total_cut_height"] * stage_settings["percentage_roi_height"]
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
        stage_settings["total_cut_height"] * stage_settings["percentage_roi_height"]
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


def lamella_region_milling(microscope, settings, stage_settings, region):
    # Setup and realign to fiducial marker
    setup_milling(microscope, settings, stage_settings)
    # Create and mill patterns
    if region == "lower":
        _lower_milling_coords(microscope, stage_settings)
    elif region == "upper":
        _upper_milling_coords(microscope, stage_settings)
    logging.info(f"milling: milling {region} lamella pattern...")
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.beams.ion_beam.horizontal_field_width.value = stage_settings[
        "hfw"
    ]  # TODO: move to better place
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
    ccs_file = settings["system"]["application_file_cleaning_cross_section"]
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
    microscope.patterning.set_default_beam_type(2)  # ion beam default
    return microscope


def setup_ion_milling(
    microscope,
    *,
    application_file="Si_Alex",
    patterning_mode="Serial",
    ion_beam_field_of_view=100e-6,
):
    """Setup for rectangle ion beam milling patterns.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    application_file : str, optional
        Application file for ion beam milling, by default "Si_Alex"
    patterning_mode : str, optional
        Ion beam milling pattern mode, by default "Serial".
        The available options are "Parallel" or "Serial".
    ion_beam_field_of_view : float, optional
        Width of ion beam field of view in meters, by default 59.2e-6
    """
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_beam_type(2)  # ion beam default
    microscope.patterning.set_default_application_file(application_file)
    microscope.patterning.mode = patterning_mode
    microscope.patterning.clear_patterns()  # clear any existing patterns
    microscope.beams.ion_beam.horizontal_field_width.value = ion_beam_field_of_view
    logging.info(f"milling: setup ion beam milling")
    logging.info(f"milling: application file:  {application_file}")
    logging.info(f"milling: patterning mode: {patterning_mode}")
    logging.info(f"milling: ion horizontal field width: {ion_beam_field_of_view}")


def protocol_stage_settings(settings):
    """ "Load settings for each milling stage, overwriting default values.

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


def jcut_milling_patterns(microscope, settings):
    """Create J-cut milling pattern in the center of the ion beam field of view.
    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    settings["jcut"] : dict
        Dictionary of J-cut parameter settings.
    Returns
    -------
    (autoscript_sdb_microscope_client.structures.RectanglePattern,
     autoscript_sdb_microscope_client.structures.RectanglePattern,
     autoscript_sdb_microscope_client.structures.RectanglePattern)
        Tuple containing the three milling patterns comprising the J-cut.
    """
    jcut_top = None
    jcut_lhs = None
    jcut_rhs = None

    # Unpack settings
    jcut_angle_degrees = settings["jcut"]["jcut_angle"]
    jcut_lamella_depth = settings["jcut"]["jcut_lamella_depth"]
    jcut_length = settings["jcut"]["jcut_length"]
    jcut_trench_thickness = settings["jcut"]["jcut_trench_thickness"]
    jcut_milling_depth = settings["jcut"]["jcut_milling_depth"]
    extra_bit = settings["jcut"]["extra_bit"]
    jcut_hfw = (
        microscope.beams.ion_beam.horizontal_field_width.value
    )  # dont change the hfw from previous step

    # Setup
    setup_ion_milling(microscope, ion_beam_field_of_view=jcut_hfw)
    # Create milling patterns
    angle_correction = np.sin(np.deg2rad(52 - jcut_angle_degrees))
    # Top bar of J-cut
    if bool(settings["jcut"]["mill_top_jcut_pattern"]) is True:
        logging.info("milling: creating top J-cut pattern")
        jcut_top = microscope.patterning.create_rectangle(
            0.0,  # center_x
            jcut_lamella_depth * angle_correction,  # center_y
            jcut_length,  # width
            jcut_trench_thickness,  # height
            jcut_milling_depth,
        )  # depth
    # Left hand side of J-cut (long side)
    if bool(settings["jcut"]["mill_lhs_jcut_pattern"]) is True:
        logging.info("milling: creating LHS J-cut pattern")
        jcut_lhs = microscope.patterning.create_rectangle(
            -((jcut_length - jcut_trench_thickness) / 2),  # center_x
            ((jcut_lamella_depth - (extra_bit / 2)) / 2) * angle_correction,  # center_y
            jcut_trench_thickness,  # width
            (jcut_lamella_depth + extra_bit) * angle_correction,  # height
            jcut_milling_depth,
        )  # depth
    # Right hand side of J-cut (short side)
    if bool(settings["jcut"]["mill_rhs_jcut_pattern"]) is True:
        logging.info("milling: creating RHS J-cut pattern")
        jcut_rightside_remaining = 1.5e-6  # in microns, how much to leave attached
        height = (jcut_lamella_depth - jcut_rightside_remaining) * angle_correction
        center_y = jcut_rightside_remaining + (height / 2)
        jcut_rhs = microscope.patterning.create_rectangle(
            +((jcut_length - jcut_trench_thickness) / 2),  # center_x
            center_y,  # center_y
            jcut_trench_thickness,  # width
            height,  # height
            jcut_milling_depth,
        )  # depth
    if jcut_top is None and jcut_lhs is None and jcut_rhs is None:
        raise RuntimeError("No J-cut patterns created, check your protocol file")
    return [jcut_top, jcut_lhs, jcut_rhs]


def weld_to_landing_post(microscope, settings, milling_current=20e-12):
    """Create and mill the sample to the landing post.
    Stick the lamella to the landing post by melting the ice with ion milling.
    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    milling_current : float, optional
        The ion beam milling current, in Amps.
    confirm : bool, optional
        Whether to wait for user confirmation before milling.
    """
    logging.info("milling: weld to landing post")
    pattern = _create_mill_pattern(
        microscope,
        center_x=0,
        center_y=0,
        width=settings["weld"]["width"],
        height=settings["weld"]["height"],
        depth=settings["weld"]["depth"],
        rotation_degrees=settings["weld"]["rotation"],
    )

    return pattern


def cut_off_needle(microscope, cut_coord, milling_current=0.74e-9):
    logging.info(f"milling: cut off needle")
    pattern = _create_mill_pattern(
        microscope,
        center_x=cut_coord["center_x"],
        center_y=cut_coord["center_y"],
        width=cut_coord["width"],
        height=cut_coord["height"],
        depth=cut_coord["depth"],
        rotation_degrees=cut_coord["rotation"],
        ion_beam_field_of_view=cut_coord["hfw"],
    )
    return pattern


def _create_mill_pattern(
    microscope,
    *,
    center_x=-10.5e-6,
    center_y=-5e-6,
    width=8e-6,
    height=2e-6,
    depth=1e-6,
    rotation_degrees=40,
    ion_beam_field_of_view=100e-6,
):
    setup_ion_milling(microscope, ion_beam_field_of_view=ion_beam_field_of_view)
    pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, depth
    )
    pattern.rotation = np.deg2rad(rotation_degrees)
    logging.info(f"milling: create milling pattern,  x:{center_x:.2e}, y: {center_y:.2e}")
    logging.info(f"w: {width:.2e}, h: {height:.2e}, d: {depth:.2e}, r:{rotation_degrees:.3f}")
    return pattern


# ideal structure:
# def milling_operation(patterns):
# setup_ion_milling()
# for pattern in patterns:
#   _draw_milling_patterns()
# run_milling()


def calculate_sharpen_needle_pattern(microscope, settings, x_0, y_0):

    height = settings["sharpen"]["height"]
    width = settings["sharpen"]["width"]
    depth = settings["sharpen"]["depth"]
    bias = settings["sharpen"]["bias"]
    hfw = settings["sharpen"]["hfw"]
    tip_angle = settings["sharpen"]["tip_angle"]  # 2NA of the needle   2*alpha
    needle_angle = settings["sharpen"][
        "needle_angle"
    ]  # needle tilt on the screen 45 deg +/-
    milling_current = settings["sharpen"]["sharpen_milling_current"]

    alpha = tip_angle / 2  # half of NA of the needletip
    beta = np.rad2deg(
        np.arctan(width / height)
    )  # box's width and length, beta is the diagonal angle
    D = np.sqrt(width ** 2 + height ** 2) / 2  # half of box diagonal
    rotation_1 = -(needle_angle + alpha)
    rotation_2 = -(needle_angle - alpha) - 180

    dx_1 = (width / 2) * math.cos(np.deg2rad(needle_angle + alpha))
    dy_1 = (width / 2) * math.sin(np.deg2rad(needle_angle + alpha))
    ddx_1 = (height / 2) * math.sin(np.deg2rad(needle_angle + alpha))
    ddy_1 = (height / 2) * math.cos(np.deg2rad(needle_angle + alpha))
    x_1 = x_0 - dx_1 + ddx_1  # centre of the bottom box
    y_1 = y_0 - dy_1 - ddy_1  # centre of the bottom box

    dx_2 = D * math.cos(np.deg2rad(needle_angle - alpha))
    dy_2 = D * math.sin(np.deg2rad(needle_angle - alpha))
    ddx_2 = (height / 2) * math.sin(np.deg2rad(needle_angle - alpha))
    ddy_2 = (height / 2) * math.cos(np.deg2rad(needle_angle - alpha))
    x_2 = x_0 - dx_2 - ddx_2  # centre of the top box
    y_2 = y_0 - dy_2 + ddy_2  # centre of the top box

    logging.info(
        f"needletip xshift offcentre: {x_0}; needletip yshift offcentre: {y_0}"
    )
    logging.info(f"width: {width}")
    logging.info(f"height: {height}")
    logging.info(f"depth: {depth}")
    logging.info(f"needle_angle: {needle_angle}")
    logging.info(f"tip_angle: {tip_angle}")
    logging.info(f"rotation1 : {rotation_1}")
    logging.info(f"rotation2 : {rotation_2}")
    logging.info(f"centre of bottom box: x1 = {x_1}; y1 = {y_1}")
    logging.info(f"centre of top box:    x2 = {x_2}; y2 = {y_2}")

    # bottom cut pattern
    cut_coord_bottom = {
        "center_x": x_1,
        "center_y": y_1,
        "width": width,
        "height": height - bias,
        "depth": depth,
        "rotation": rotation_1,
        "hfw": hfw,
    }

    # top cut pattern
    cut_coord_top = {
        "center_x": x_2,
        "center_y": y_2,
        "width": width,
        "height": height - bias,
        "depth": depth,
        "rotation": rotation_2,
        "hfw": hfw,
    }

    return cut_coord_bottom, cut_coord_top


def create_sharpen_needle_patterns(microscope, cut_coord_bottom, cut_coord_top):
    sharpen_patterns = []

    setup_ion_milling(microscope, ion_beam_field_of_view=cut_coord_top["hfw"])

    for cut_coord in [cut_coord_bottom, cut_coord_top]:
        center_x = cut_coord["center_x"]
        center_y = cut_coord["center_y"]
        width = cut_coord["width"]
        height = cut_coord["height"]
        depth = cut_coord["depth"]
        rotation_degrees = cut_coord["rotation"]

        # create patterns
        pattern = microscope.patterning.create_rectangle(
            center_x, center_y, width, height, depth
        )
        pattern.rotation = -np.deg2rad(rotation_degrees)
        sharpen_patterns.append(pattern)
        logging.info(f"milling: create sharpen needle pattern")
        logging.info(f"x: {center_x}, y: {center_y}, w: {width}, h: {height}")
        logging.info(f"d: {depth}, r: {rotation_degrees}")

    return sharpen_patterns


def flatten_landing_pattern(microscope, settings):
    """Create flatten_landing milling pattern in the center of the ion beam field of view.
    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    settings : dict

    Returns
    -------
    autoscript_sdb_microscope_client.structures.RectanglePattern
        Rectangle milling pattern used to flatten the landing area.
    """

    # setup
    setup_ion_milling(microscope, application_file="autolamella")

    # draw flatten landing pattern
    pattern = microscope.patterning.create_cleaning_cross_section(
        center_x=settings["flatten_landing"]["center_x"],
        center_y=settings["flatten_landing"]["center_y"],
        width=settings["flatten_landing"]["width"],
        height=settings["flatten_landing"]["height"],
        depth=settings["flatten_landing"]["depth"]
    )
    pattern.scan_direction = "LeftToRight"

    return pattern
