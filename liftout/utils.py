import os
from pathlib import Path

import numpy as np
import yaml
import liftout
from fibsem.utils import load_yaml, configure_logging
from fibsem.structures import ImageSettings, SystemSettings, StageSettings, CalibrationSettings, stage_position_from_dict
from liftout.structures import AutoLiftoutOptions, AutoLiftoutSettings, ReferenceHFW

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


def load_settings_from_config(fname: Path):

    config = load_yaml(fname)
    image_settings = ImageSettings.__from_dict__(config["calibration"]["imaging"])
    system_settings = SystemSettings.__from_dict__(config["system"])
    stage_settings = StageSettings.__from_dict__(config["system"])
    calibration_settings = CalibrationSettings.__from_dict__(config["calibration"]["limits"])
    reference_hfw = ReferenceHFW.__from_dict__(config["calibration"]["reference_hfw"])
    options = AutoLiftoutOptions.__from_dict__(config["system"])
    grid_position = stage_position_from_dict(config["system"]["initial_position"]["sample_grid"])
    landing_position = stage_position_from_dict(config["system"]["initial_position"]["landing_grid"])

    settings = AutoLiftoutSettings(
        system = system_settings,
        stage = stage_settings,
        calibration = calibration_settings,
        reference_hfw=reference_hfw,
        options=options,
        image_settings=image_settings,
        grid_position=grid_position,
        landing_position=landing_position,
    )

    return settings


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
    # settings = _format_dictionary(settings)

    return settings


def _format_dictionary(dictionary: dict):
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


def make_logging_directory(path: Path = None, name="run"):
    if path is None:
        path = os.path.join(os.path.dirname(liftout.__file__), "log")
    directory = os.path.join(path, name)
    os.makedirs(directory, exist_ok=True)
    return directory


def get_last_log_message(path: Path) -> str:
    with open(path) as f:
        lines = f.read().splitlines()
        log_line = lines[-1:][-1]  # last log msg
        log_msg = log_line.split("—")[-1].strip()

    return log_msg


def plot_two_images(img1, img2) -> None:
    import matplotlib.pyplot as plt
    from fibsem.structures import Point

    c = Point(img1.data.shape[1]//2, img1.data.shape[0]//2)

    fig, ax = plt.subplots(1, 2, figsize=(30, 30))
    ax[0].imshow(img1.data, cmap="gray")
    ax[0].plot(c.x, c.y, "y+", ms=50, markeredgewidth=2)
    ax[1].imshow(img2.data, cmap="gray")
    ax[1].plot(c.x, c.y, "y+", ms=50, markeredgewidth=2)
    plt.show()


# cross correlate
def crosscorrelate_and_plot(ref_image, new_image, rotate: bool = False, lp: int = 128, hp:int = 6, sigma: int = 6, ref_mask: np.ndarray = None):
    import matplotlib.pyplot as plt
    import numpy as np

    from fibsem import calibration
    from fibsem.structures import Point

    # rotate ref
    if rotate:
        ref_image = calibration.rotate_AdornedImage(ref_image)

    dx, dy, xcorr = calibration.shift_from_crosscorrelation(
        ref_image, new_image, lowpass=lp, highpass=hp, sigma=sigma, use_rect_mask=True, ref_mask=ref_mask
    )

    pixelsize = ref_image.metadata.binary_result.pixel_size.x
    dx_p, dy_p = int(dx / pixelsize), int(dy / pixelsize)

    print(f"shift_m: {dx}, {dy}")
    print(f"shift_px: {dx_p}, {dy_p}")

    shift = np.roll(new_image.data, (-dy_p, -dx_p), axis=(0, 1))

    mid = Point(shift.shape[1]//2, shift.shape[0]//2)

    fig, ax = plt.subplots(1, 4, figsize=(30, 30))
    ax[0].imshow(ref_image.data, cmap="gray")
    ax[0].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[0].set_title(f"Reference (rotate={rotate})")
    ax[1].imshow(new_image.data, cmap="gray")
    ax[1].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[1].set_title(f"New Image")
    ax[2].imshow(xcorr, cmap="turbo")
    ax[2].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[2].plot(mid.x-dx_p, mid.y-dy_p, "m+", ms=50, markeredgewidth=2)
    ax[2].set_title("XCORR")
    ax[3].imshow(shift, cmap="gray")
    ax[3].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[3].plot(mid.x-dx_p, mid.y-dy_p, "m+", ms=50, markeredgewidth=2)
    ax[3].set_title("New Image Shifted")
    plt.show()


### VALIDATION

def _validate_configuration_values(microscope, dictionary):
    """Recursively traverse dictionary and validate all parameters.

    Parameters
    ----------
    dictionary : dict
        Any arbitrarily structured python dictionary.

    Returns
    ---------
    dictionary: dict
        Validated Configuration Dictionary
    Raises
    -------
    ValueError
        The parameter is not within the available range for the microscope.
    """
    
    from fibsem import validation

    for key, item in dictionary.items():
        if isinstance(item, dict):
            _validate_configuration_values(microscope, item)
        elif isinstance(item, list):
            dictionary[key] = [_validate_configuration_values(microscope, i) for i in item]
        else:
            if isinstance(item, float):
                if "hfw" in key:
                    if "max" in key or "grid" in key:
                        continue  # skip checks on these keys
                    validation._validate_horizontal_field_width(microscope=microscope, 
                        horizontal_field_widths=[item])

                if "milling_current" in key:
                    validation._validate_ion_beam_currents(microscope, [item])

                if "imaging_current" in key:
                    validation._validate_electron_beam_currents(microscope, [item])

                if "resolution" in key:
                    validation._validate_scanning_resolutions(microscope, [item])

                if "dwell_time" in key:
                    validation._validate_dwell_time(microscope, [item])

            if isinstance(item, str):
                if "application_file" in key:
                    validation._validate_application_files(microscope, [item])
                if "weights" in key:
                    _validate_model_weights_file(item)
                    
    return dictionary

def _validate_model_weights_file(filename):
    import os

    from liftout.model import models
    weights_path = os.path.join(os.path.dirname(models.__file__), filename)
    if not os.path.exists(weights_path):
        raise ValueError(
            f"Unable to find model weights file {weights_path} specified."
        )

### SETUP
def quick_setup():
    """Quick setup for microscope, settings, and image_settings"""
    from fibsem import acquire
    from fibsem import utils as fibsem_utils

    settings = load_full_config()

    import os

    path = os.path.join(os.getcwd(), "tools/test")
    os.makedirs(path, exist_ok=True)
    configure_logging(path)

    microscope = fibsem_utils.connect_to_microscope(
        ip_address=settings["system"]["ip_address"]
    )
    image_settings = acquire.update_image_settings_v3(settings, path=path)

    return microscope, settings, image_settings


def full_setup():
    """Quick setup for microscope, settings,  image_settings, sample and lamella"""
    import os

    from liftout.sample import Lamella, Sample

    microscope, settings, image_settings = quick_setup()

    # sample
    sample = Sample(path = os.path.dirname(image_settings.save_path), name="test")
    
    # lamella
    lamella = Lamella(sample.path, 999, _petname="999-test-mule")
    sample.positions[lamella._number] = lamella
    sample.save()

    return microscope, settings, image_settings, sample, lamella
