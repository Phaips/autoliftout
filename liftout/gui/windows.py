import logging

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from liftout.detection.detection import DetectionResult
from liftout.fibsem import milling, movement
from liftout.fibsem.acquire import BeamType, ImageSettings
from liftout.fibsem.sample import Lamella
from liftout.gui.detection_window import GUIDetectionWindow
from liftout.gui.milling_window import GUIMillingWindow
from liftout.gui.movement_window import GUIMMovementWindow
from liftout.gui.user_window import GUIUserWindow


def ask_user_interaction(
    microscope: SdbMicroscopeClient, msg="Default Ask User Message", beam_type=None
):
    """Create user interaction window and get return response"""
    ask_user_window = GUIUserWindow(microscope=microscope, msg=msg, beam_type=beam_type)
    ask_user_window.show()

    response = bool(ask_user_window.exec_())
    return response


def ask_user_movement(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    msg_type="eucentric",
    msg: str = None,
    flat_to_sem: bool = False,
    parent=None,
):

    logging.info(f"Asking user for confirmation for {msg_type} movement")
    if flat_to_sem:
        movement.flat_to_beam(
            microscope, settings=settings, beam_type=BeamType.ELECTRON
        )

    movement_window = GUIMMovementWindow(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        msg_type=msg_type,
        msg=msg, 
        parent=parent,
    )
    movement_window.show()
    movement_window.exec_()

def open_milling_window(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    milling_pattern: milling.MillingPattern,
    x: float = 0.0,
    y: float = 0.0,
    parent=None,
):
    """Open the Milling Window ready for milling

    Args:
        milling_pattern (MillingPattern): The type of milling pattern
        x (float, optional): the initial pattern offset (x-direction). Defaults to 0.0.
        y (float, optional): the initial pattenr offset (y-direction). Defaults to 0.0.
    """
    milling_window = GUIMillingWindow(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern_type=milling_pattern,
        x=x,
        y=y,
        parent=parent,
    )

    milling_window.show()
    milling_window.exec_()

def validate_detection_v2(
    microscope: SdbMicroscopeClient,
    settings: dict,
    detection_result: DetectionResult,
    lamella: Lamella,
):
    # TODO: validate the detection shift type...

    # user validates detection result
    detection_window = GUIDetectionWindow(
        microscope=microscope,
        settings=settings,
        detection_result=detection_result,
        lamella=lamella,
    )
    detection_window.show()
    detection_window.exec_()

    return detection_window.detection_result
