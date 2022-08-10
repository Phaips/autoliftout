import logging
from pathlib import Path

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import AdornedImage
from fibsem import acquire, movement, validation
from fibsem import utils as fibsem_utils
from fibsem.acquire import BeamType, ImageSettings
from fibsem.structures import ImageSettings
from liftout import patterning, utils
from liftout.detection import detection
from liftout.detection.detection import DetectionFeature, DetectionResult
from liftout.gui.detection_window import GUIDetectionWindow
from liftout.gui.milling_window import GUIMillingWindow
from liftout.gui.movement_window import GUIMMovementWindow
from liftout.gui.user_window import GUIUserWindow
from liftout.sample import Lamella
from PyQt5.QtWidgets import QMessageBox
from liftout.gui import utils as ui_utils

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
    milling_pattern: patterning.MillingPattern,
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


def detect_features(microscope: SdbMicroscopeClient, settings: dict, image_settings: ImageSettings, lamella: Lamella,  ref_image: AdornedImage, features: tuple[DetectionFeature], validate: bool = True) -> DetectionResult:
    """_summary_

    Args:
        microscope (SdbMicroscopeClient): _description_
        settings (dict): _description_
        image_settings (ImageSettings): _description_
        lamella (Lamella): _description_
        ref_image (AdornedImage): _description_
        features (tuple[DetectionFeature]): _description_
        validate (bool, optional): _description_. Defaults to True.

    Returns:
        DetectionResult: _description_
    """
    # detect features
    image = acquire.new_image(microscope, image_settings)

    if ref_image is None:
        ref_image = image

    # detect features
    det = detection.locate_shift_between_features_v2(
        image, ref_image, features=features
    )

    # user validate features...
    if validate:
        det = validate_detection_v2(
            microscope, settings, det, lamella,
        )

    return det


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


def validate_detection(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
    shift_type: tuple,
    beam_type: BeamType = BeamType.ELECTRON,
    parent=None,
):
    # TODO: validate the detection shift type...
    from liftout.detection import detection
    from liftout.gui.detection_window import GUIDetectionWindow

    image_settings.beam_type = beam_type  # change to correct beamtype

    # run model detection
    detection_result = detection.identify_shift_using_machine_learning(
        microscope, image_settings, settings, shift_type=shift_type
    )

    # user validates detection result
    detection_window = GUIDetectionWindow(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        detection_result=detection_result,
        lamella=lamella,
        parent=parent,
    )
    detection_window.show()
    detection_window.exec_()

    return detection_window.detection_result


def run_validation_ui(microscope: SdbMicroscopeClient, settings: dict, log_path: Path):
    """Run validation checks to confirm microscope state before run."""

    response = ui_utils.message_box_ui(
        title="Microscope State Validation", 
        text="Do you want to validate the microscope state?")

    if response is False:
        logging.info(f"PRE_RUN_VALIDATION cancelled by user.")
        return

    logging.info(f"INIT | PRE_RUN_VALIDATION | STARTED")

    # run validation
    validation.validate_initial_microscope_state(microscope, settings)

    # validate user configuration
    utils._validate_configuration_values(microscope, settings) # TODO: fix

    # reminders
    reminder_str = """Please check that the following steps have been completed:
    \n - Sample is inserted
    \n - Confirm Operating Temperature
    \n - Needle Calibration
    \n - Ion Column Calibration
    \n - Crossover Calibration
    \n - Plasma Gas Valve Open
    \n - Initial Grid and Landing Positions
    """

    response = ui_utils.message_box_ui(
        title="AutoLiftout Initialisation Checklist",
        text=reminder_str,
        buttons=QMessageBox.Ok)

    # Loop backwards through the log, until we find the start of validation
    with open(log_path) as f:
        lines = f.read().splitlines()
        validation_warnings = []
        for line in lines[::-1]:
            if "PRE_RUN_VALIDATION" in line:
                break
            if "WARNING" in line:
                logging.info(line)
                validation_warnings.append(line)
        logging.info(
            f"{len(validation_warnings)} warnings were identified during intial setup."
        )

    if validation_warnings:
        warning_str = f"The following {len(validation_warnings)} warnings were identified during initialisation."

        for warning in validation_warnings[::-1]:
            warning_str += f"\n{warning.split('—')[-1]}"

        ui_utils.message_box_ui(
            title="AutoLiftout Initialisation Warning",
            text=warning_str,
            buttons=QMessageBox.Ok
            )

    logging.info(f"INIT | PRE_RUN_VALIDATION | FINISHED")


def sputter_platinum_on_whole_sample_grid(
    microscope: SdbMicroscopeClient = None,
    settings: dict = None,
) -> None:
    """Move to the sample grid and sputter platinum over the whole grid"""

    # Whole-grid platinum deposition
    response = ask_user_interaction(
        microscope=microscope,
        msg="Do you want to sputter the whole \nsample grid with platinum?",
        beam_type=BeamType.ELECTRON,
    )

    if response:
        fibsem_utils.sputter_platinum(microscope, settings, whole_grid=True)

    return
