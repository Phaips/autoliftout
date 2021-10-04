import os
import yaml
import time
import logging
import datetime
import liftout


def configure_logging(save_path='', log_filename='logfile', log_level=logging.INFO):
    """Log to the terminal and to file simultaneously."""
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        '%Y%m%d.%H%M%S')  # datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[
            logging.FileHandler(save_path+'/'+log_filename+timestamp+'.log'),
            logging.StreamHandler(),
        ])


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


def make_logging_directory(prefix='run'):
    directory = os.path.join(os.path.dirname(liftout.__file__), "log")
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')

    save_directory = os.path.join(directory, prefix, timestamp)
    os.makedirs(os.path.join(save_directory, "img"), exist_ok=True)
    return save_directory


def save_image(image, save_path, label=''):
    path = f'{save_path}/img/{label}.tif'
    image.save(path)
