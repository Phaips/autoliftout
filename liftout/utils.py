import os
import yaml
import time
import logging
import datetime
import liftout

# TODO: better logs: https://www.toptal.com/python/in-depth-python-logging
def configure_logging(save_path='', log_filename='logfile', log_level=logging.INFO):
    """Log to the terminal and to file simultaneously."""
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        '%Y%m%d.%H%M%S')

    logfile = os.path.join(save_path, f"{log_filename}{timestamp}.log")

    logging.basicConfig(
        format="%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[
            logging.FileHandler(logfile), #save_path+'/'+log_filename+timestamp+'.log'), 
            logging.StreamHandler(),
        ])

    # FEATURE_FLAG
    # assert logfile == save_path+'/'+log_filename+timestamp+'.log'
    return logfile
    # return save_path+'/'+log_filename+timestamp+'.log'

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
    path_old = f'{save_path}/img/{label}.tif'
    path = os.path.join(save_path, "img", f"{label}.tif")
    # assert path == path_new
    image.save(path)
    # FEATURE_FLAG
