import logging
import math
from enum import Enum

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import milling, validation
from fibsem.structures import Point

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

############################## PATTERNS ##############################

def mill_trench_patterns(
    microscope: SdbMicroscopeClient, settings: dict, point:Point = Point()
):
    """Calculate the trench milling patterns"""

    
    lamella_width = settings["lamella_width"]
    lamella_height = settings["lamella_height"]
    trench_height = settings["trench_height"]
    upper_trench_height = trench_height / max(settings["size_ratio"], 1.0)
    offset = settings["offset"]
    milling_depth = settings["milling_depth"]

    centre_upper_y = point.y + (lamella_height / 2 + upper_trench_height / 2 + offset)
    centre_lower_y = point.y - (lamella_height / 2 + trench_height / 2 + offset)

    lower_pattern = microscope.patterning.create_cleaning_cross_section(
        point.x, centre_lower_y, lamella_width, trench_height, milling_depth,
    )
    lower_pattern.scan_direction = "BottomToTop"

    upper_pattern = microscope.patterning.create_cleaning_cross_section(
        point.x, centre_upper_y, lamella_width, upper_trench_height, milling_depth,
    )
    upper_pattern.scan_direction = "TopToBottom"

    return [lower_pattern, upper_pattern]


def jcut_milling_patterns(
    microscope: SdbMicroscopeClient,
    settings: dict,
    point: Point = Point()
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

    jcut_lhs_height = settings["lhs_height"]
    jcut_rhs_height = settings["rhs_height"]
    jcut_lamella_height = settings["lamella_height"]
    jcut_width = settings["width"]
    jcut_trench_thickness = settings["trench_thickness"]
    jcut_milling_depth = settings["depth"]

    # top_jcut
    jcut_top = microscope.patterning.create_rectangle(
        center_x = point.x,
        center_y = point.y + jcut_lamella_height,
        width = jcut_width,
        height=jcut_trench_thickness,
        depth=jcut_milling_depth
    )

    jcut_half_width = ((jcut_width - jcut_trench_thickness) / 2)
    jcut_half_height = jcut_lamella_height / 2

    # lhs_jcut
    jcut_lhs = microscope.patterning.create_rectangle(
        center_x=point.x - jcut_half_width,
        center_y=point.y + jcut_half_height - (jcut_lhs_height / 2 - jcut_half_height),  
        width=jcut_trench_thickness,
        height=jcut_lhs_height,  
        depth=jcut_milling_depth,
    )  # depth

    # rhs jcut
    jcut_rhs = microscope.patterning.create_rectangle(
        center_x=point.x + jcut_half_width,
        center_y=point.y + jcut_half_height - (jcut_rhs_height / 2 - jcut_half_height),  
        width=jcut_trench_thickness,
        height=jcut_rhs_height,  
        depth=jcut_milling_depth,
    ) 

    # use parallel mode for jcut
    # microscope.patterning.mode = "Parallel"

    return [jcut_top, jcut_lhs, jcut_rhs]

def jcut_severing_pattern(
    microscope, settings: dict, point: Point
):

    jcut_severing_pattern = milling._draw_rectangle_pattern(
        microscope=microscope,
        settings=settings,
        x=point.x, y=point.y
    )

    return [jcut_severing_pattern]

def weld_to_landing_post(
    microscope: SdbMicroscopeClient,
    settings: dict,
    point: Point = Point()
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

    pattern = milling._draw_rectangle_pattern(
        microscope, settings, point.x, point.y
    )

    return [pattern]


def cut_off_needle(microscope, settings, point: Point = Point()):
    logging.info(f"milling: cut off needle")

    pattern = milling._draw_rectangle_pattern(microscope, settings, point.x, point.y)

    return [pattern]

def calculate_sharpen_needle_pattern(settings: dict, point: Point = Point()):

    x_0, y_0 = point.x, point.y
    height = settings["height"]
    width = settings["width"]
    depth = settings["depth"]
    bias = settings["bias"]
    hfw = settings["hfw"]
    tip_angle = settings["tip_angle"]  # 2NA of the needle   2*alpha
    needle_angle = settings["needle_angle"]  # needle tilt on the screen 45 deg +/-

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

def create_sharpen_needle_patterns(microscope: SdbMicroscopeClient, settings:dict,point:Point = Point()) -> list:

    # calculate the sharpening patterns
    cut_coord_bottom, cut_coord_top = calculate_sharpen_needle_pattern(settings, point)

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

def flatten_landing_pattern(
    microscope: SdbMicroscopeClient,
    settings: dict,
    point:Point = Point()
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
    pattern = milling._draw_rectangle_pattern(microscope, settings, point.x, point.y)

    return pattern


def fiducial_marker_patterns(
    microscope: SdbMicroscopeClient,
    settings: dict,
    point: Point = Point()
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

    pattern_1 = milling._draw_rectangle_pattern(microscope, settings, point.x, point.y)
    pattern_2 = milling._draw_rectangle_pattern(microscope, settings, point.x, point.y)
    pattern_2.rotation = np.deg2rad(settings["rotation"] + 90)

    return [pattern_1, pattern_2]

# TODO: can probably be consolidated more ... in particular the rectangular patterns...
def create_milling_patterns(microscope:SdbMicroscopeClient, milling_settings: dict, milling_pattern_type: MillingPattern, point: Point = Point(0.0, 0.0)) -> list:
    """Redraw the milling patterns with updated milling settings"""

    if milling_pattern_type == MillingPattern.Trench:
        patterns = mill_trench_patterns(microscope=microscope,settings=milling_settings, point=point)
                                                        

    if milling_pattern_type == MillingPattern.JCut:

        patterns = jcut_milling_patterns(microscope=microscope, settings=milling_settings, point=point)

    if milling_pattern_type == MillingPattern.Sever:

        patterns = jcut_severing_pattern(microscope=microscope, settings=milling_settings, point=point)

    if milling_pattern_type == MillingPattern.Weld:

        patterns = weld_to_landing_post(microscope=microscope, settings=milling_settings, point=point)
                                                        
    if milling_pattern_type == MillingPattern.Cut:

        patterns = cut_off_needle(microscope=microscope, settings=milling_settings, point=point)

    if milling_pattern_type == MillingPattern.Sharpen:
        patterns = create_sharpen_needle_patterns(microscope, milling_settings, point)

    if milling_pattern_type == MillingPattern.Thin:
        patterns = mill_trench_patterns(microscope=microscope, settings=milling_settings, point=point)

    if milling_pattern_type == MillingPattern.Polish:
        patterns = mill_trench_patterns(microscope=microscope, settings=milling_settings, point=point)

    if milling_pattern_type == MillingPattern.Flatten:
        patterns = flatten_landing_pattern(microscope=microscope, settings=milling_settings, point=point)

    if milling_pattern_type == MillingPattern.Fiducial:
        patterns = fiducial_marker_patterns(microscope=microscope, settings=milling_settings, point=point)
    
    # convert patterns is list
    if not isinstance(patterns, list):
        patterns = [patterns]

    return patterns

# UTILS

def get_milling_protocol_stage_settings(settings:dict, milling_pattern: MillingPattern):
    from liftout.config import config

    stage_name = config.PATTERN_PROTOCOL_MAP[milling_pattern]
    milling_protocol_stages = milling.read_protocol_dictionary(settings["protocol"], stage_name)

    # validate settings
    if not isinstance(milling_protocol_stages, list):
        milling_protocol_stages =  [milling_protocol_stages]

    for i, stage_settings in enumerate(milling_protocol_stages):
        
        milling_protocol_stages[i] = validation.validate_milling_settings(stage_settings, settings
            )

    return milling_protocol_stages
