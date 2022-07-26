import logging
from pathlib import Path

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
from liftout import utils
from liftout.fibsem import calibration
from PyQt5.QtWidgets import QMessageBox


def validate_initial_microscope_state(
    microscope: SdbMicroscopeClient, settings: dict
) -> None:
    """Set the initial microscope state to default, and validate other settings."""

    # TODO: add validation checks for dwell time and resolution
    logging.info(
        f"Electron voltage: {microscope.beams.electron_beam.high_voltage.value:.2f}"
    )
    logging.info(
        f"Electron current: {microscope.beams.electron_beam.beam_current.value:.2f}"
    )

    # set default microscope state
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
    microscope.beams.ion_beam.beam_current.value = settings["calibration"]["imaging"][
        "imaging_current"
    ]
    microscope.beams.ion_beam.horizontal_field_width.value = settings["calibration"][
        "imaging"
    ]["horizontal_field_width"]
    microscope.beams.ion_beam.scanning.resolution.value = settings["calibration"][
        "imaging"
    ]["resolution"]
    microscope.beams.ion_beam.scanning.dwell_time.value = settings["calibration"][
        "imaging"
    ]["dwell_time"]

    microscope.beams.electron_beam.horizontal_field_width.value = settings[
        "calibration"
    ]["imaging"]["horizontal_field_width"]
    microscope.beams.electron_beam.scanning.resolution.value = settings["calibration"][
        "imaging"
    ]["resolution"]
    microscope.beams.electron_beam.scanning.dwell_time.value = settings["calibration"][
        "imaging"
    ]["dwell_time"]

    # validate chamber state
    calibration.validate_chamber(microscope=microscope)

    # validate stage calibration (homed, linked)
    calibration.validate_stage_calibration(microscope=microscope)

    # validate needle calibration (needle calibration, retracted)
    calibration.validate_needle_calibration(microscope=microscope)

    # validate beam settings and calibration
    calibration.validate_beams_calibration(microscope=microscope, settings=settings)

    # validate user configuration
    utils.validate_settings(microscope=microscope, config=settings)


def run_validation_ui(microscope: SdbMicroscopeClient, settings: dict, log_path: Path):
    """Run validation checks to confirm microscope state before run."""

    msg = QMessageBox()
    msg.setWindowTitle("Microscope State Validation")
    msg.setText("Do you want to validate the microscope state?")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg.setIcon(QMessageBox.Question)
    button = msg.exec()

    if button == QMessageBox.No:
        logging.info(f"PRE_RUN_VALIDATION cancelled by user.")
        return

    logging.info(f"INIT | PRE_RUN_VALIDATION | STARTED")

    # run validation
    validate_initial_microscope_state(microscope, settings)

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
    msg = QMessageBox()
    msg.setWindowTitle("AutoLiftout Initialisation Checklist")
    msg.setText(reminder_str)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()

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

        msg = QMessageBox()
        msg.setWindowTitle("AutoLiftout Initialisation Warnings")
        msg.setText(warning_str)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    logging.info(f"INIT | PRE_RUN_VALIDATION | FINISHED")


def validate_milling_settings(stage_settings: dict, settings: dict) -> dict:
    # validation?
    if "milling_current" not in stage_settings:
        stage_settings["milling_current"] = settings["calibration"]["imaging"][
            "milling_current"
        ]
    if "cleaning_cross_section" not in stage_settings:
        stage_settings["cleaning_cross_section"] = False
    if "rotation" not in stage_settings:
        stage_settings["rotation"] = 0.0
    if "scan_direction" not in stage_settings:
        stage_settings["scan_direction"] = "TopToBottom"

    # remove list element from settings
    if "protocol_stages" in stage_settings:
        del stage_settings["protocol_stages"]

    return stage_settings
