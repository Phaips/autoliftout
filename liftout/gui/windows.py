import logging
from pathlib import Path

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem.acquire import BeamType
from fibsem.structures import MicroscopeSettings, Point
from liftout import actions, patterning
from liftout.gui.milling_window import GUIMillingWindow

from fibsem.ui import windows as fibsem_ui_windows
from fibsem import utils as fibsem_utils



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
    response = fibsem_ui_windows.ask_user_interaction(
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
