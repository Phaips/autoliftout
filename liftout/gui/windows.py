import logging
from pathlib import Path

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import AdornedImage
from fibsem import acquire, movement, validation
from fibsem import utils as fibsem_utils
from fibsem.acquire import BeamType
from fibsem.structures import MicroscopeSettings, Point
from liftout import actions, patterning
from liftout.gui.milling_window import GUIMillingWindow
from liftout.gui.movement_window import GUIMMovementWindow
from liftout.gui.user_window import GUIUserWindow
from liftout.sample import Lamella
from PyQt5.QtWidgets import QMessageBox


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
    settings: MicroscopeSettings,
    msg_type="eucentric",
    msg: str = None,
    flat_to_sem: bool = False,
    parent=None,
):

    logging.info(f"Asking user for confirmation for {msg_type} movement")
    if flat_to_sem:
        movement.move_flat_to_beam(
            microscope, settings=settings, beam_type=BeamType.ELECTRON
        )

    movement_window = GUIMMovementWindow(
        microscope=microscope,
        settings=settings,
        msg_type=msg_type,
        msg=msg,
        parent=parent,
    )
    movement_window.show()
    movement_window.exec_()


def open_milling_window(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    milling_pattern: patterning.MillingPattern,
    point: Point = Point(),
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
        milling_pattern_type=milling_pattern,
        point = point,
        parent=parent,
    )

    milling_window.show()
    milling_window.exec_()



def sputter_platinum_on_whole_sample_grid(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    protocol: dict,
) -> None:
    """Move to the sample grid and sputter platinum over the whole grid"""

    # Whole-grid platinum deposition
    response = ask_user_interaction(
        microscope=microscope,
        msg="Do you want to sputter the whole \nsample grid with platinum?",
        beam_type=BeamType.ELECTRON,
    )

    if response:
        actions.move_to_sample_grid(microscope, settings, protocol)
        fibsem_utils.sputter_platinum(
            microscope=microscope,
            protocol=protocol["platinum"],
            whole_grid=True,
            default_application_file=settings.system.application_file,
        )

    return
