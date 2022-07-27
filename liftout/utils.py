import datetime
import json
import logging
import os
import time
from pathlib import Path

import yaml

import liftout


# TODO: better logs: https://www.toptal.com/python/in-depth-python-logging
def configure_logging(path: Path = "", log_filename="logfile", log_level=logging.INFO):
    """Log to the terminal and to file simultaneously."""
    logfile = os.path.join(path, f"{log_filename}.log")

    logging.basicConfig(
        format="%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(),],
    )

    return logfile


def load_config(yaml_filename):
    """Load user input from yaml settings file.

    Parameters
    ----------
    yaml_filename : str
        Filename path of user configuration file.

    Returns
    -------
    dict
        Dictionary containing user input settings.
    """
    with open(yaml_filename, "r") as f:
        settings_dict = yaml.safe_load(f)
    settings_dict = _format_dictionary(settings_dict)
    return settings_dict


def load_yaml(fname: Path) -> dict:

    with open(fname, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_full_config(
    system_config: Path = None,
    calibration_config: Path = None,
    protocol_config: Path = None,
) -> dict:
    """Load multiple config files into single settings dictionary."""

    from liftout.config import config

    # default paths
    if system_config is None:
        system_config = config.system_config
    if calibration_config is None:
        calibration_config = config.calibration_config
    if protocol_config is None:
        protocol_config = config.protocol_config

    # load individual configs
    config_system = load_yaml(system_config)
    config_calibration = load_yaml(calibration_config)
    config_protocol = load_yaml(protocol_config)

    # consolidate
    settings = dict()
    settings["system"] = config_system
    settings["calibration"] = config_calibration
    settings["protocol"] = config_protocol

    # validation
    settings = _format_dictionary(settings)

    return settings


def _format_dictionary(dictionary):
    """Recursively traverse dictionary and covert all numeric values to flaot.

    Parameters
    ----------
    dictionary : dict
        Any arbitrarily structured python dictionary.

    Returns
    -------
    dictionary
        The input dictionary, with all numeric values converted to float type.
    """
    for key, item in dictionary.items():
        if isinstance(item, dict):
            _format_dictionary(item)
        elif isinstance(item, list):
            dictionary[key] = [_format_dictionary(i) for i in item]
        else:
            if item is not None:
                try:
                    dictionary[key] = float(dictionary[key])
                except ValueError:
                    pass
    return dictionary


def validate_settings(microscope, config):
    from liftout import user_input

    user_input._validate_configuration_values(microscope=microscope, dictionary=config)
    user_input._validate_scanning_rotation(microscope=microscope)


def make_logging_directory(path: Path = None, name="run"):
    if path is None:
        path = os.path.join(os.path.dirname(liftout.__file__), "log")
    directory = os.path.join(path, name)
    os.makedirs(directory, exist_ok=True)
    return directory


def save_image(image, save_path, label=""):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"{label}.tif")
    image.save(path)


def current_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d.%I-%M-%S%p")


def save_metadata(settings, path):
    fname = os.path.join(path, "metadata.json")
    with open(fname, "w") as fp:
        json.dump(settings, fp, sort_keys=True, indent=4)


def get_last_log_message(path: Path) -> str:
    with open(path) as f:
        lines = f.read().splitlines()
        log_line = lines[-1:][-1]  # last log msg
        log_msg = log_line.split("—")[-1].strip()

    return log_msg