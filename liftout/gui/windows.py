import logging

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from liftout.fibsem import movement, milling
from liftout.fibsem.acquire import BeamType, ImageSettings
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
