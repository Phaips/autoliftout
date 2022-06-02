import logging
import math
from cv2 import reduce

import numpy as np
from autoscript_core.common import ApplicationServerException
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (AdornedImage,
                                                         StagePosition)
from liftout.fibsem import acquire, calibration
from liftout.fibsem.acquire import ImageSettings

BeamType = acquire.BeamType

def jcut_severing_pattern(microscope, settings: dict, centre_x: float = 0.0, centre_y: float = 0.0):
    """Create J-cut milling pattern in the center of the ion beam field of view.
    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    settings : dict
        Sample surface angle for J-cut in degrees, by default 6
    Returns
    -------
    autoscript_sdb_microscope_client.structures.RectanglePattern
        Rectangle milling pattern used to sever the remaining bit of the J-cut.
    """
    # Unpack settings
    jcut_angle_degrees = settings["jcut"]["jcut_angle"]
    jcut_lamella_depth = settings["jcut"]["height"]
    jcut_length = settings["jcut"]["length"]
    jcut_trench_thickness = settings["jcut"]["trench_thickness"]
    jcut_milling_depth = settings["jcut"]["milling_depth"]
    extra_bit = settings["jcut"]["extra_bit"]

    # Create milling pattern - right hand side of J-cut
    flat_to_ion_angle = settings["system"]["stage_tilt_flat_to_ion"]
    assert flat_to_ion_angle == 52
    angle_correction_factor = np.sin(np.deg2rad(flat_to_ion_angle - jcut_angle_degrees)) # MAGIC NUMBER
    center_x = centre_x + ((jcut_length - jcut_trench_thickness) / 2)
    center_y = centre_y + (
        (jcut_lamella_depth - (extra_bit / 2)) / 2
    ) * angle_correction_factor  # noqa: E501
    width = jcut_trench_thickness
    height = (jcut_lamella_depth + extra_bit) * angle_correction_factor
    jcut_severing_pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, jcut_milling_depth
    )
    return [jcut_severing_pattern]


def run_milling(microscope: SdbMicroscopeClient, settings: dict, milling_current: float = None, asynch: bool = False):
    """Run ion beam milling at specified current.
    
    - Change to milling current
    - Run milling (synchronous) or Start Milling (asynchronous)

    """
    logging.info("milling: running ion beam milling now...")

    # change to milling current
    microscope.imaging.set_active_view(2)  # the ion beam view
    if milling_current is None:
        milling_current = settings["imaging"]["milling_current"]
    if microscope.beams.ion_beam.beam_current.value != milling_current:     
        # if milling_current not in microscope.beams.ion_beam.beam_current.available_values:
        #   switch to closest

        microscope.beams.ion_beam.beam_current.value = milling_current

    # run milling (asynchronously)
    if asynch:
        microscope.patterning.start()
    else:
        microscope.patterning.run()
        microscope.patterning.clear_patterns()


def finish_milling(microscope: SdbMicroscopeClient, settings: dict) -> None:
    """Finish milling by clearing the patterns and restoring the default imaging current.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope client connection
        settings (dict): configuration settings
    """
    # restore imaging current
    logging.info("returning to the ion beam imaging current now.")
    if settings["imaging"]["imaging_current"]:
        imaging_current = settings["imaging"]["imaging_current"]
    microscope.patterning.clear_patterns()
    microscope.beams.ion_beam.beam_current.value = imaging_current
    microscope.patterning.mode = "Serial"
    logging.info("ion beam milling complete.")

def mill_polish_lamella(microscope: SdbMicroscopeClient, settings: dict, image_settings: ImageSettings, patterns: list):
    """Polish the lamella edges to the desired thickness. Tilt by one degree between polishing steps to ensure beam mills along the correct axis.
    
    Align using crosscorrelation between after tilting.
    """    

    from autoscript_sdb_microscope_client.structures import Rectangle

    # align (move settings outside?)
    image_settings.resolution = settings["polish_lamella"]["resolution"]
    image_settings.dwell_time = settings["polish_lamella"]["dwell_time"]
    image_settings.hfw = settings["polish_lamella"]["hfw"]
    image_settings.beam_type = BeamType.ION
    image_settings.gamma.enabled = False
    image_settings.save = True
    image_settings.label = f"polish_lamella_crosscorrelation_ref"

    # user defined reduced area
    reduced_area = Rectangle(
        settings["reduced_area"]["x"],
        settings["reduced_area"]["y"],
        settings["reduced_area"]["dx"],
        settings["reduced_area"]["dy"]
        )

    # TILT_OFFSET = settings["polish_lamella"]["tilt_offset"]
    # CROSSCORRELATION_STEPS = 3

    # reset beam shift
    calibration.reset_beam_shifts(microscope)

    # initial reference image
    ref_image = acquire.new_image(microscope, image_settings, reduced_area=reduced_area)

    # generate patterns (user change?)    
    # lower_pattern, upper_pattern = patterns

    # # retrieve pattern values, (the objects are deleted by clear_patterns)
    # l_cx = lower_pattern.center_x
    # l_cy = lower_pattern.center_y
    # l_w = lower_pattern.width
    # l_h = lower_pattern.height
    # l_d = lower_pattern.depth

    # u_cx = upper_pattern.center_x
    # u_cy = upper_pattern.center_y
    # u_w = upper_pattern.width
    # u_h = upper_pattern.height
    # u_d = upper_pattern.depth

    # # clear patterns...
    # microscope.patterning.clear_patterns()
    
    # # # tilt up for bottom pattern
    # # tilt_up = StagePosition(t=np.deg2rad(-TILT_OFFSET))
    # # microscope.specimen.stage.relative_move(tilt_up)

    # # absolute moves...    
    # tilt_negative = StagePosition(t=np.deg2rad(TILT_OFFSET))
    # microscope.specimen.stage.absolute_move(tilt_negative)

    # # multi-step alignment
    # for i in range(CROSSCORRELATION_STEPS):
    #     image_settings.label = f"polish_lamella_tilt_{tilt_negative.t:.2f}_stage_{i+1}"
    #     calibration.beam_shift_alignment(microscope, image_settings, ref_image, reduced_area=reduced_area)
    
    # image_settings.label = f"polish_lamella_tilt_{tilt_negative.t:.2f}_aligned"
    # _ = acquire.new_image(microscope, image_settings, reduced_area)


    # # mill bottom pattern
    # # draw bottom pattern
    # lower_pattern = microscope.patterning.create_cleaning_cross_section(
    #     center_x=l_cx,
    #     center_y=l_cy,
    #     width=l_w,
    #     height=l_h,
    #     depth=l_d
    # )
    # lower_pattern.scan_direction = "BottomToTop"

    # # run milling
    # run_milling(microscope, settings, milling_current=settings["polish_lamella"]["milling_current"], asynch=False)

    # # reset back to starting tilt
    # # tilt_back = StagePosition(t=np.deg2rad(TILT_OFFSET))
    # # microscope.specimen.stage.relative_move(tilt_back)

    # # reset beam shift
    # calibration.reset_beam_shifts(microscope)

    # # tilt down for top pattern
    # tilt_positive = StagePosition(t=np.deg2rad(-TILT_OFFSET))
    # microscope.specimen.stage.absolute_move(tilt_positive)
    
    # # multi-step alignment
    # for i in range(CROSSCORRELATION_STEPS):
    #     image_settings.label = f"polish_lamella_tilt_{tilt_positive.t:.2f}_stage_{i+1}"
    #     calibration.beam_shift_alignment(microscope, image_settings, ref_image, reduced_area=reduced_area)
    
    # image_settings.label = f"polish_lamella_tilt_{tilt_positive.t:.2f}_aligned"
    # _ = acquire.new_image(microscope, image_settings, reduced_area)
    
    # # mill top pattern
    # # draw top pattern
    # upper_pattern = microscope.patterning.create_cleaning_cross_section(
    #     center_x = u_cx,
    #     center_y = u_cy,
    #     width = u_w,
    #     height = u_h,
    #     depth = u_d
    # )
    # upper_pattern.scan_direction = "TopToBottom"
    
    # run milling
    run_milling(microscope, settings, milling_current=settings["polish_lamella"]["milling_current"], asynch=False)

    # reset back to starting tilt
    # tilt_zero = StagePosition(t=np.deg2rad(0))
    # microscope.specimen.stage.absolute_move(tilt_zero)

    # reset beam shift
    calibration.reset_beam_shifts(microscope)

    # finish milling
    finish_milling(microscope, settings)


def mill_trench_patterns(microscope: SdbMicroscopeClient, settings: dict, centre_x=0, centre_y=0):
    """Calculate the trench milling patterns"""

    lamella_width = settings["lamella_width"]
    lamella_height = settings["lamella_height"]
    trench_height = settings["trench_height"]
    upper_trench_height = trench_height / max(settings["size_ratio"], 1.0)
    offset = settings["offset"]
    milling_depth = settings["milling_depth"]

    centre_upper_y = centre_y + (lamella_height / 2 + upper_trench_height / 2 + offset)
    centre_lower_y = centre_y - (lamella_height / 2 + trench_height / 2 + offset)

    lower_pattern = microscope.patterning.create_cleaning_cross_section(
        centre_x,
        centre_lower_y,
        lamella_width,
        trench_height,
        milling_depth,
    )
    lower_pattern.scan_direction = "BottomToTop"

    upper_pattern = microscope.patterning.create_cleaning_cross_section(
        centre_x,
        centre_upper_y,
        lamella_width,
        upper_trench_height,
        milling_depth,
    )
    upper_pattern.scan_direction = "TopToBottom"

    return [lower_pattern, upper_pattern]


def get_milling_protocol_stages(settings, stage_name):
    protocol_stages = []
    for stage_settings in settings[stage_name]["protocol_stages"]:
        tmp_settings = settings[stage_name].copy()
        tmp_settings.update(stage_settings)
        protocol_stages.append(tmp_settings)

    return protocol_stages

# def setup_milling(microscope, settings, stage_settings):
#     """Setup the ion beam system ready for milling.
#     Parameters
#     ----------
#     microscope : Autoscript microscope object.
#     settings :  Dictionary of user input argument settings.
#     stage_settings : Dictionary of settings for a single protocol milling stage
#     Returns
#     -------
#     Autoscript microscope object.
#     """
#     ccs_file = settings["system"]["application_file_cleaning_cross_section"]
#     microscope = reset_state(microscope, settings, application_file=ccs_file)
#     microscope.beams.ion_beam.beam_current.value = stage_settings["milling_current"]
#     return microscope


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
    application_file="autolamella",
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

def jcut_milling_patterns(microscope: SdbMicroscopeClient, settings: dict, centre_x: float = 0, centre_y: float = 0) -> list:
    """Create J-cut milling pattern in the center of the ion beam field of view.
    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    settings : dict
        Dictionary of parameter settings.
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
    jcut_lamella_depth = settings["jcut"]["height"]
    jcut_length = settings["jcut"]["length"]
    jcut_trench_thickness = settings["jcut"]["trench_thickness"]
    jcut_milling_depth = settings["jcut"]["milling_depth"]
    extra_bit = settings["jcut"]["extra_bit"]

    flat_to_ion_angle = settings["system"]["stage_tilt_flat_to_ion"]

    # Create milling patterns
    angle_correction = np.sin(np.deg2rad(flat_to_ion_angle - jcut_angle_degrees))
    
    # Top bar of J-cut
    jcut_top = microscope.patterning.create_rectangle(
        centre_x,  # center_x
        centre_y + jcut_lamella_depth * angle_correction,  # center_y
        jcut_length,  # width
        jcut_trench_thickness,  # height
        jcut_milling_depth,
    )  # depth
    
    # Left hand side of J-cut (long side)
    jcut_lhs = microscope.patterning.create_rectangle(
        centre_x + -((jcut_length - jcut_trench_thickness) / 2),  # center_x
        centre_y + ((jcut_lamella_depth - (extra_bit / 2)) / 2) * angle_correction,  # center_y
        jcut_trench_thickness,  # width
        (jcut_lamella_depth + extra_bit) * angle_correction,  # height
        jcut_milling_depth,
    )  # depth
    
    # Right hand side of J-cut (short side)
    jcut_rightside_remaining = 1.5e-6  # in microns, how much to leave attached
    height = (jcut_lamella_depth - jcut_rightside_remaining) * angle_correction
    center_y = jcut_rightside_remaining + (height / 2)
    jcut_rhs = microscope.patterning.create_rectangle(
        centre_x +((jcut_length - jcut_trench_thickness) / 2),  # center_x
        centre_y + center_y,  # center_y
        jcut_trench_thickness,  # width
        height,  # height
        jcut_milling_depth,
    )  # depth

    # use parallel mode for jcut
    # microscope.patterning.mode = "Parallel"

    return [jcut_top, jcut_lhs, jcut_rhs]


def weld_to_landing_post(microscope: SdbMicroscopeClient, settings: dict, centre_x: float = 0.0, centre_y: float = 0.0):
    """Create and mill the sample to the landing post.
    Stick the lamella to the landing post by melting the ice with ion milling.
    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    settings: dict
        The protocol settings dictionary
    """
    pattern = _create_mill_pattern(
        microscope,
        center_x=centre_x,
        center_y=centre_y,
        width=settings["weld"]["width"],
        height=settings["weld"]["height"],
        depth=settings["weld"]["depth"],
        rotation_degrees=settings["weld"]["rotation"],
    )

    return [pattern]


def cut_off_needle(microscope, settings, centre_x: float = 0.0, centre_y: float = 0.0):
    logging.info(f"milling: cut off needle")

    height = settings["cut"]["height"]
    width = settings["cut"]["width"]
    depth = settings["cut"]["depth"]
    rotation = settings["cut"]["rotation"]
    hfw = settings["cut"]["hfw"]
    vertical_gap = settings["cut"]["gap"] 
    horizontal_gap = settings["cut"]["hgap"] 

    cut_coord = {"center_x": centre_x - horizontal_gap,
                    "center_y": centre_y - vertical_gap,
                    "width": width,
                    "height": height,
                    "depth": depth,
                    "rotation": rotation, "hfw": hfw}

    pattern = _create_mill_pattern(
        microscope,
        center_x=cut_coord["center_x"],
        center_y=cut_coord["center_y"],
        width=cut_coord["width"],
        height=cut_coord["height"],
        depth=cut_coord["depth"],
        rotation_degrees=cut_coord["rotation"],
    )
    return [pattern]


def _create_mill_pattern(
    microscope,
    center_x=-10.5e-6,
    center_y=-5e-6,
    width=8e-6,
    height=2e-6,
    depth=1e-6,
    rotation_degrees=40,
):
    pattern = microscope.patterning.create_rectangle(
        center_x, center_y, width, height, depth
    )
    pattern.rotation = np.deg2rad(rotation_degrees)
    logging.info(f"milling: create milling pattern,  x:{center_x:.2e}, y: {center_y:.2e}")
    logging.info(f"w: {width:.2e}, h: {height:.2e}, d: {depth:.2e}, r:{rotation_degrees:.3f}")
    return pattern

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
        "height": height,# - bias,
        "depth": depth,
        "rotation": rotation_1,
        "hfw": hfw,
    }

    # top cut pattern
    cut_coord_top = {
        "center_x": x_2,
        "center_y": y_2,
        "width": width,
        "height": height,# - bias,
        "depth": depth,
        "rotation": rotation_2,
        "hfw": hfw,
    }

    return cut_coord_bottom, cut_coord_top


def create_sharpen_needle_patterns(microscope, cut_coord_bottom, cut_coord_top):
    sharpen_patterns = []

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
        logging.info(f"create sharpen needle pattern")
        logging.info(f"x: {center_x:.2e}, y: {center_y:.2e}, w: {width:.2e}, h: {height:.2e}")
        logging.info(f"d: {depth:.2e}, r: {rotation_degrees} deg")

    return sharpen_patterns


def flatten_landing_pattern(microscope: SdbMicroscopeClient, settings: dict, centre_x: float = 0.0, centre_y: float = 0.0):
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

    # draw flatten landing pattern
    pattern = microscope.patterning.create_cleaning_cross_section(
        center_x=centre_x,
        center_y=centre_y,
        width=settings["flatten_landing"]["width"],
        height=settings["flatten_landing"]["height"],
        depth=settings["flatten_landing"]["depth"]
    )
    pattern.scan_direction = "LeftToRight"

    return pattern


def fiducial_marker_patterns(microscope: SdbMicroscopeClient, settings: dict, centre_x: float = 0.0, centre_y: float = 0.0):
    """_summary_

    Args:
        microscope (SdbMicroscopeClient): AutoScript microscope connection
        settings (dict): fiducial milling settings
        centre_x (float, optional): centre x coordinate. Defaults to 0.0.
        centre_y (float, optional): centre y coordinate. Defaults to 0.0.
    Returns
    -------
        patterns : list
            List of rectangular patterns used to create the fiducial marker.
    """

    pattern_1 = microscope.patterning.create_rectangle(
        center_x=centre_x,
        center_y=centre_y,
        width=settings["fiducial"]["width"],
        height=settings["fiducial"]["length"],
        depth=settings["fiducial"]["depth"],
    )
    pattern_1.rotation = np.deg2rad(settings["fiducial"]["rotation"])

    pattern_2 = microscope.patterning.create_rectangle(
        center_x=centre_x,
        center_y=centre_y,
        width=settings["fiducial"]["width"],
        height=settings["fiducial"]["length"],
        depth=settings["fiducial"]["depth"],
    )
    pattern_2.rotation = np.deg2rad(settings["fiducial"]["rotation"] + 90)
    
    return [pattern_1, pattern_2]