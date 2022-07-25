import logging
import math
from turtle import width

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from liftout.fibsem import acquire, calibration
from liftout.fibsem.acquire import ImageSettings, BeamType
from enum import Enum
# from autoscript_sdb_microscope_client.structures import RectanglePattern
class MillingPattern(Enum):
    Trench = 1
    JCut = 2
    Sever = 3
    Weld = 4
    Cut = 5
    Sharpen = 6
    Thin = 7
    Polish = 8
    Flatten = 9
    Fiducial = 10



def get_milling_protocol_stages(settings, stage_name):
    protocol_stages = []
    for stage_settings in settings[stage_name]["protocol_stages"]:
        tmp_settings = settings[stage_name].copy()
        tmp_settings.update(stage_settings)
        protocol_stages.append(tmp_settings)

    return protocol_stages

def get_milling_protocol_stages_v2(settings, stage_name) -> list:
    # TODO: generalise this...
    protocol_stages = []

    return protocol_stages


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

def run_milling(
    microscope: SdbMicroscopeClient,
    settings: dict,
    milling_current: float = None,
    asynch: bool = False,
):
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


def finish_milling(microscope: SdbMicroscopeClient, imaging_current: float = 20e-12) -> None:
    """Finish milling by clearing the patterns and restoring the default imaging current.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope client connection
        settings (dict): configuration settings
    """
    # restore imaging current
    logging.info("returning to the ion beam imaging current now.")
    microscope.patterning.clear_patterns()
    microscope.beams.ion_beam.beam_current.value = imaging_current
    microscope.patterning.mode = "Serial"
    logging.info("ion beam milling complete.")


# def mill_polish_lamella(
#     microscope: SdbMicroscopeClient,
#     settings: dict,
#     image_settings: ImageSettings,
#     patterns: list,
# ):
#     """Polish the lamella edges to the desired thickness. Tilt by one degree between polishing steps to ensure beam mills along the correct axis.
    
#     Align using crosscorrelation between after tilting.
#     """

#     from autoscript_sdb_microscope_client.structures import Rectangle

#     # align (move settings outside?)
#     image_settings.resolution = settings["polish_lamella"]["resolution"]
#     image_settings.dwell_time = settings["polish_lamella"]["dwell_time"]
#     image_settings.hfw = settings["polish_lamella"]["hfw"]
#     image_settings.beam_type = BeamType.ION
#     image_settings.gamma.enabled = False
#     image_settings.save = True
#     image_settings.label = f"polish_lamella_crosscorrelation_ref"

#     # user defined reduced area
#     reduced_area = Rectangle(
#         settings["reduced_area"]["x"],
#         settings["reduced_area"]["y"],
#         settings["reduced_area"]["dx"],
#         settings["reduced_area"]["dy"],
#     )

#     # TILT_OFFSET = settings["polish_lamella"]["tilt_offset"]
#     # CROSSCORRELATION_STEPS = 3

#     # reset beam shift
#     calibration.reset_beam_shifts(microscope)

#     # initial reference image
#     ref_image = acquire.new_image(microscope, image_settings, reduced_area=reduced_area)

#     # generate patterns (user change?)
#     # lower_pattern, upper_pattern = patterns

#     # # retrieve pattern values, (the objects are deleted by clear_patterns)
#     # l_cx = lower_pattern.center_x
#     # l_cy = lower_pattern.center_y
#     # l_w = lower_pattern.width
#     # l_h = lower_pattern.height
#     # l_d = lower_pattern.depth

#     # u_cx = upper_pattern.center_x
#     # u_cy = upper_pattern.center_y
#     # u_w = upper_pattern.width
#     # u_h = upper_pattern.height
#     # u_d = upper_pattern.depth

#     # # clear patterns...
#     # microscope.patterning.clear_patterns()

#     # # # tilt up for bottom pattern
#     # # tilt_up = StagePosition(t=np.deg2rad(-TILT_OFFSET))
#     # # microscope.specimen.stage.relative_move(tilt_up)

#     # # absolute moves...
#     # tilt_negative = StagePosition(t=np.deg2rad(TILT_OFFSET))
#     # microscope.specimen.stage.absolute_move(tilt_negative)

#     # # multi-step alignment
#     # for i in range(CROSSCORRELATION_STEPS):
#     #     image_settings.label = f"polish_lamella_tilt_{tilt_negative.t:.2f}_stage_{i+1}"
#     #     calibration.beam_shift_alignment(microscope, image_settings, ref_image, reduced_area=reduced_area)

#     # image_settings.label = f"polish_lamella_tilt_{tilt_negative.t:.2f}_aligned"
#     # _ = acquire.new_image(microscope, image_settings, reduced_area)

#     # # mill bottom pattern
#     # # draw bottom pattern
#     # lower_pattern = microscope.patterning.create_cleaning_cross_section(
#     #     center_x=l_cx,
#     #     center_y=l_cy,
#     #     width=l_w,
#     #     height=l_h,
#     #     depth=l_d
#     # )
#     # lower_pattern.scan_direction = "BottomToTop"

#     # # run milling
#     # run_milling(microscope, settings, milling_current=settings["polish_lamella"]["milling_current"], asynch=False)

#     # # reset back to starting tilt
#     # # tilt_back = StagePosition(t=np.deg2rad(TILT_OFFSET))
#     # # microscope.specimen.stage.relative_move(tilt_back)

#     # # reset beam shift
#     # calibration.reset_beam_shifts(microscope)

#     # # tilt down for top pattern
#     # tilt_positive = StagePosition(t=np.deg2rad(-TILT_OFFSET))
#     # microscope.specimen.stage.absolute_move(tilt_positive)

#     # # multi-step alignment
#     # for i in range(CROSSCORRELATION_STEPS):
#     #     image_settings.label = f"polish_lamella_tilt_{tilt_positive.t:.2f}_stage_{i+1}"
#     #     calibration.beam_shift_alignment(microscope, image_settings, ref_image, reduced_area=reduced_area)

#     # image_settings.label = f"polish_lamella_tilt_{tilt_positive.t:.2f}_aligned"
#     # _ = acquire.new_image(microscope, image_settings, reduced_area)

#     # # mill top pattern
#     # # draw top pattern
#     # upper_pattern = microscope.patterning.create_cleaning_cross_section(
#     #     center_x = u_cx,
#     #     center_y = u_cy,
#     #     width = u_w,
#     #     height = u_h,
#     #     depth = u_d
#     # )
#     # upper_pattern.scan_direction = "TopToBottom"

#     # run milling
#     run_milling(
#         microscope,
#         settings,
#         milling_current=settings["polish_lamella"]["milling_current"],
#         asynch=False,
#     )

#     # reset back to starting tilt
#     # tilt_zero = StagePosition(t=np.deg2rad(0))
#     # microscope.specimen.stage.absolute_move(tilt_zero)

#     # reset beam shift
#     calibration.reset_beam_shifts(microscope)

#     # finish milling
#     finish_milling(microscope, settings)


def mill_trench_patterns(
    microscope: SdbMicroscopeClient, settings: dict, centre_x=0, centre_y=0
):
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
        centre_x, centre_lower_y, lamella_width, trench_height, milling_depth,
    )
    lower_pattern.scan_direction = "BottomToTop"

    upper_pattern = microscope.patterning.create_cleaning_cross_section(
        centre_x, centre_upper_y, lamella_width, upper_trench_height, milling_depth,
    )
    upper_pattern.scan_direction = "TopToBottom"

    return [lower_pattern, upper_pattern]


def jcut_milling_patterns(
    microscope: SdbMicroscopeClient,
    settings: dict,
    centre_x: float = 0,
    centre_y: float = 0,
) -> list:
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

    jcut_lhs_height = settings["protocol"]["jcut"]["lhs_height"]
    jcut_rhs_height = settings["protocol"]["jcut"]["rhs_height"]
    jcut_lamella_height = settings["protocol"]["jcut"]["lamella_height"]
    jcut_width = settings["protocol"]["jcut"]["width"]
    jcut_trench_thickness = settings["protocol"]["jcut"]["trench_thickness"]
    jcut_milling_depth = settings["protocol"]["jcut"]["depth"]

    # top_jcut
    jcut_top = microscope.patterning.create_rectangle(
        center_x = centre_x,
        center_y = centre_y + jcut_lamella_height,
        width = jcut_width,
        height=jcut_trench_thickness,
        depth=jcut_milling_depth
    )

    jcut_half_width = ((jcut_width - jcut_trench_thickness) / 2)
    jcut_half_height = jcut_lamella_height / 2

    # lhs_jcut
    jcut_lhs = microscope.patterning.create_rectangle(
        center_x=centre_x - jcut_half_width,
        center_y=centre_y + jcut_half_height - (jcut_lhs_height / 2 - jcut_half_height),  
        width=jcut_trench_thickness,
        height=jcut_lhs_height,  
        depth=jcut_milling_depth,
    )  # depth

    # rhs jcut
    jcut_rhs = microscope.patterning.create_rectangle(
        center_x=centre_x + jcut_half_width,
        center_y=centre_y + jcut_half_height - (jcut_rhs_height / 2 - jcut_half_height),  
        width=jcut_trench_thickness,
        height=jcut_rhs_height,  
        depth=jcut_milling_depth,
    ) 

    # use parallel mode for jcut
    # microscope.patterning.mode = "Parallel"

    return [jcut_top, jcut_lhs, jcut_rhs]

def jcut_severing_pattern(
    microscope, settings: dict, centre_x: float = 0.0, centre_y: float = 0.0
):

    jcut_severing_pattern = _draw_rectangle_pattern(
        microscope=microscope,
        settings=settings["protocol"]["sever"],
        x=centre_x, y=centre_y

    )

    return [jcut_severing_pattern]



def weld_to_landing_post(
    microscope: SdbMicroscopeClient,
    settings: dict,
    centre_x: float = 0.0,
    centre_y: float = 0.0,
):
    """Create and mill the sample to the landing post.
    Stick the lamella to the landing post by melting the ice with ion milling.
    Parmaters
    ---------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    settings: dict
        The protocol settings dictionary
    """

    pattern = _draw_rectangle_pattern(
        microscope, settings["protocol"]["weld"], centre_x, centre_y
    )

    return [pattern]


def cut_off_needle(microscope, settings, centre_x: float = 0.0, centre_y: float = 0.0):
    logging.info(f"milling: cut off needle")

    pattern = _draw_rectangle_pattern(microscope, settings["protocol"]["cut"], centre_x, centre_y)

    return [pattern]

def calculate_sharpen_needle_pattern(settings, x_0,  y_0):

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

    # bottom cut pattern
    cut_coord_bottom = {
        "center_x": x_1,
        "center_y": y_1,
        "width": width,
        "height": height,  # - bias,
        "depth": depth,
        "rotation": rotation_1,
        "hfw": hfw,
    }

    # top cut pattern
    cut_coord_top = {
        "center_x": x_2,
        "center_y": y_2,
        "width": width,
        "height": height,  # - bias,
        "depth": depth,
        "rotation": rotation_2,
        "hfw": hfw,
    }

    return cut_coord_bottom, cut_coord_top


def create_sharpen_needle_patterns(microscope: SdbMicroscopeClient, settings:dict, x0: float, y0: float) -> list:

    # calculate the sharpening patterns
    cut_coord_bottom, cut_coord_top = calculate_sharpen_needle_pattern(settings, x0,  y0)

    # draw the patterns
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
        logging.info(
            f"x: {center_x:.2e}, y: {center_y:.2e}, w: {width:.2e}, h: {height:.2e}"
        )
        logging.info(f"d: {depth:.2e}, r: {rotation_degrees} deg")

    return sharpen_patterns

def _draw_rectangle_pattern(microscope:SdbMicroscopeClient, settings:dict , x: float = 0.0, y: float = 0.0):

    if settings["cleaning_cross_section"]:
        pattern = microscope.patterning.create_cleaning_cross_section(
        center_x=x,
        center_y=y,
        width=settings["width"],
        height=settings["height"],
        depth=settings["depth"],
    )
    else:
        pattern = microscope.patterning.create_rectangle(
            center_x=x,
            center_y=y,
            width=settings["width"],
            height=settings["height"],
            depth=settings["depth"],
        )
    
    # need to make each protocol setting have these....which means validation
    pattern.rotation=np.deg2rad(settings["rotation"])
    pattern.scan_direction = settings["scan_direction"]

    return pattern


def flatten_landing_pattern(
    microscope: SdbMicroscopeClient,
    settings: dict,
    centre_x: float = 0.0,
    centre_y: float = 0.0,
):
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

    # # draw flatten landing pattern
    pattern = _draw_rectangle_pattern(microscope, settings["protocol"]["flatten_landing"], centre_x, centre_y)

    return pattern


def fiducial_marker_patterns(
    microscope: SdbMicroscopeClient,
    settings: dict,
    centre_x: float = 0.0,
    centre_y: float = 0.0,
):
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

    pattern_1 = _draw_rectangle_pattern(microscope, settings["protocol"]["fiducial"], centre_x, centre_y)
    pattern_2 = _draw_rectangle_pattern(microscope, settings["protocol"]["fiducial"], centre_x, centre_y)
    pattern_2.rotation = np.deg2rad(settings["protocol"]["fiducial"]["rotation"] + 90)

    return [pattern_1, pattern_2]


def update_milling_patterns(microscope:SdbMicroscopeClient, settings:dict , milling_settings: dict, milling_pattern_type: MillingPattern, x:float = 0.0, y: float = 0.0) -> list:
    """Redraw the milling patterns with updated milling settings"""

    if milling_pattern_type == MillingPattern.Trench:

        patterns = mill_trench_patterns(microscope=microscope,
                                                        settings=milling_settings,
                                                        centre_x=x, centre_y=y)

    if milling_pattern_type == MillingPattern.JCut:

        patterns = jcut_milling_patterns(microscope=microscope,
                                                        settings=settings, centre_x=x, centre_y=y)

    if milling_pattern_type == MillingPattern.Sever:

        patterns = jcut_severing_pattern(microscope=microscope, settings=settings, centre_x=x, centre_y=y)

    if milling_pattern_type == MillingPattern.Weld:

        patterns = weld_to_landing_post(microscope=microscope, settings=settings,
                                                        centre_x=x, centre_y=y)

    if milling_pattern_type == MillingPattern.Cut:

        patterns = cut_off_needle(microscope=microscope, settings=settings,
                                                centre_x=x, centre_y=y)

    if milling_pattern_type == MillingPattern.Sharpen:
        patterns = create_sharpen_needle_patterns(microscope, settings, x, y)

    if milling_pattern_type == MillingPattern.Thin:
        patterns = mill_trench_patterns(microscope=microscope,
                                settings=milling_settings,
                                centre_x=x, centre_y=y)


    if milling_pattern_type == MillingPattern.Polish:
        patterns = mill_trench_patterns(microscope=microscope, settings=milling_settings, centre_x=x, centre_y=y)

    if milling_pattern_type == MillingPattern.Flatten:
        patterns = flatten_landing_pattern(microscope=microscope, settings=settings,
                                                        centre_x=x, centre_y=y)

    if milling_pattern_type == MillingPattern.Fiducial:
        patterns = fiducial_marker_patterns(microscope=microscope, settings=settings, centre_x=x, centre_y=y)
    
    # convert patterns is list
    if not isinstance(patterns, list):
        patterns = [patterns]

    return patterns


def get_milling_protocol_stage_settings(settings:dict, milling_pattern: MillingPattern):
    if milling_pattern == MillingPattern.Trench:
        milling_protocol_stages = get_milling_protocol_stages(settings=settings["protocol"], stage_name="lamella")
        
    if milling_pattern == MillingPattern.JCut:
        milling_protocol_stages = settings["protocol"]["jcut"]

    if milling_pattern == MillingPattern.Sever:
        milling_protocol_stages = settings["protocol"]["sever"]

    if milling_pattern == MillingPattern.Weld:
        milling_protocol_stages = settings["protocol"]["weld"]

    if milling_pattern == MillingPattern.Cut:
        milling_protocol_stages = settings["protocol"]["cut"]

    if milling_pattern == MillingPattern.Sharpen:
        milling_protocol_stages = settings["protocol"]["sharpen"]

    if milling_pattern == MillingPattern.Thin:
        milling_protocol_stages = get_milling_protocol_stages(settings=settings["protocol"], stage_name="thin_lamella")

    if milling_pattern == MillingPattern.Polish:
        milling_protocol_stages = settings["protocol"]["polish_lamella"]

    if milling_pattern == MillingPattern.Flatten:
        milling_protocol_stages = settings["protocol"]["flatten_landing"]

    if milling_pattern == MillingPattern.Fiducial:
        milling_protocol_stages = settings["protocol"]["fiducial"]
    
    # validate settings
    if not isinstance(milling_protocol_stages, list):
        milling_protocol_stages =  [milling_protocol_stages]

    return milling_protocol_stages

def calculate_milling_time(patterns: list, milling_current: float) -> float:
    from liftout.config import config
    from liftout.fibsem import constants

    # volume (width * height * depth) / total_volume_sputter_rate

    # calculate sputter rate
    if milling_current in config.MILLING_SPUTTER_RATE:
        total_volume_sputter_rate = config.MILLING_SPUTTER_RATE[milling_current]
    else:
        total_volume_sputter_rate = 3.920e-1

    # calculate total milling volume
    volume = 0
    for stage in patterns:
        for pattern in stage:
            width = pattern.width * constants.METRE_TO_MICRON
            height = pattern.height * constants.METRE_TO_MICRON
            depth = pattern.depth * constants.METRE_TO_MICRON
            volume += width * height * depth
    
    # estimate time
    milling_time_seconds = volume / total_volume_sputter_rate # um3 * 1/ (um3 / s) = seconds

    logging.info(f"WHDV: {width:.2f}um, {height:.2f}um, {depth:.2f}um, {volume:.2f}um3")
    logging.info(f"Volume: {volume:.2e}, Rate: {total_volume_sputter_rate:.2e} um3/s")
    logging.info(f"Milling Estimated Time: {milling_time_seconds / 60:.2f}m")

    return milling_time_seconds