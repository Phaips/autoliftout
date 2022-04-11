"""Functions for getting input from the user."""
import numpy as np


def validate_user_input(microscope, settings):
    application_files = [
        settings["system"]["application_file_rectangle"],
        settings["system"]["application_file_cleaning_cross_section"],
    ]
    _validate_application_files(microscope, application_files)
    scanning_resolutions = [
        settings["imaging"]["resolution"],
        settings["fiducial"]["reduced_area_resolution"],
    ]
    _validate_scanning_rotation(microscope)
    _validate_stage_coordinate_system(microscope)
    dwell_times = [settings["imaging"]["dwell_time"]]
    _validate_dwell_time(microscope, dwell_times)
    _validate_scanning_resolutions(microscope, scanning_resolutions)


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


def _validate_application_files(microscope, application_files):
    """Check that the user supplied application files exist on this system.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    application_files : list
        List of application files, eg: ['Si', 'Si_small']

    Raises
    ------
    ValueError
        Application file name not found in list of available application files.
    """
    available_files = microscope.patterning.list_all_application_files()
    for app_file in application_files:
        if app_file not in available_files:
            raise ValueError(
                "{} not found ".format(app_file)
                + "in list of available application files!\n"
                "Please choose one from the list: \n"
                "{}".format(available_files)
            )


def _validate_dwell_time(microscope, dwell_times):
    """Check that the user specified dwell times are within the limits.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    dwell_times : list
        List of dwell times, eg: [1e-7, 1e-6]

    Raises
    ------
    ValueError
        Dwell time is smaller than the minimum limit.
    ValueError
        Dwell time is larger than the maximum limit.
    """
    dwell_limits = microscope.beams.ion_beam.scanning.dwell_time.limits
    for dwell in dwell_times:
        if not isinstance(dwell, (int, float)):
            raise ValueError(
                "Dwell time {} must be a number!\n".format(dwell)
                + "Please choose a value between the limits: \n"
                "{}".format(dwell_limits)
            )
        if dwell < dwell_limits.min:
            raise ValueError(
                "{} dwell time is too small!\n".format(dwell)
                + "Please choose a value between the limits: \n"
                "{}".format(dwell_limits)
            )
        elif dwell > dwell_limits.max:
            raise ValueError(
                "{} dwell time is too large!\n".format(dwell)
                + "Please choose a value between the limits: \n"
                "{}".format(dwell_limits)
            )
        else:
            if dwell is np.nan:
                raise ValueError(
                    "{} dwell time ".format(dwell) + "is not a number!\n"
                    "Please choose a value between the limits:\n"
                    "{}".format(dwell_limits)
                )

def _validate_electron_beam_currents(microscope, electron_beam_currents):
    """Check that the user supplied electron beam current values are valid.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    electron_beam_currents : list
        List of electron beam currents, eg: [ 3e-10, 1e-09]

    Raises
    ------
    ValueError
        Beam current not within limits of available electron beam currents.
    """
    available_electron_beam_currents = (
        microscope.beams.electron_beam.beam_current.limits
        )
    for beam_current in electron_beam_currents:
        
        if not available_electron_beam_currents.is_in(beam_current):
            raise ValueError(
                "{} not found ".format(beam_current)
                + "in range of available electron beam currents!\n"
                "Please choose one from within the range: \n"
                "{}".format(available_electron_beam_currents)
            )


    # print(available_electron_beam_currents.)

def _validate_ion_beam_currents(microscope, ion_beam_currents):
    """Check that the user supplied ion beam current values are valid.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    ion_beam_currents : list
        List of ion beam currents, eg: [ 3e-10, 1e-09]

    Raises
    ------
    ValueError
        Beam current not found in list of available ion beam currents.
    """
    available_ion_beam_currents = (
        microscope.beams.ion_beam.beam_current.available_values
    )
    # TODO: decide how strict we want to be on the available currents (e.g. exact or within range)
    for beam_current in ion_beam_currents:
        if beam_current <= min(available_ion_beam_currents) or beam_current >= max(available_ion_beam_currents):
            raise ValueError(
                "{} not found ".format(beam_current)
                + "in list of available ion beam currents!\n"
                "Please choose one from the list: \n"
                "{}".format(available_ion_beam_currents)
            )


def _validate_horizontal_field_width(microscope, horizontal_field_widths):
    """Check that the ion beam horizontal field width is within the limits.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    horizontal_field_widths : list
        List of ion beam horizontal field widths, eg: [50e-6, 100e-6]

    Raises
    ------
    ValueError
        Ion beam horizontal field width is smaller than the minimum limit.
    ValueError
        Ion beam horizontal field width is larger than the maximum limit.
    """
    hfw_limits = microscope.beams.ion_beam.horizontal_field_width.limits
    for hfw in horizontal_field_widths:
        if not isinstance(hfw, (int, float)):
            raise ValueError(
                "Horizontal field width must be a number!\n"
                "Please choose a value between the limits: \n"
                "{}".format(hfw_limits)
            )
        if hfw < hfw_limits.min:
            raise ValueError(
                "{} ".format(hfw) + "horizontal field width is too small!\n"
                "Please choose a value between the limits: \n"
                "{}".format(hfw_limits)
            )
        elif hfw > hfw_limits.max:
            raise ValueError(
                "{} ".format(hfw) + "horizontal field width is too large!\n"
                "Please choose a value between the limits: \n"
                "{}".format(hfw_limits)
            )
        else:
            if hfw is np.nan:
                raise ValueError(
                    "{} horizontal field width ".format(hfw) + "is not a number!\n"
                    "Please choose a value between the limits: \n"
                    "{}".format(hfw_limits)
                )


def _validate_scanning_resolutions(microscope, scanning_resolutions):
    """Check that the user supplied scanning resolution values are valid.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    scanning_resolutions : list
        List of scanning resolutions, eg: ['1536x1024', '3072x2048']

    Raises
    ------
    ValueError
        Resolution not found in list of available scanning resolutions.
    """
    available_resolutions = (
        microscope.beams.ion_beam.scanning.resolution.available_values
    )
    microscope.beams.ion_beam.beam_current.available_values
    for resolution in scanning_resolutions:
        if resolution not in available_resolutions:
            raise ValueError(
                "{} not found ".format(resolution)
                + "in list of available scanning resolutions!\n"
                "Please choose one from the list: \n"
                "{}".format(available_resolutions)
            )


def _validate_scanning_rotation(microscope):
    """Check the microscope scan rotation is zero.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.

    Raises
    ------
    ValueError
        Raise an error to warn the user if the scan rotation is not zero.
    """
    rotation = microscope.beams.ion_beam.scanning.rotation.value
    if rotation is None:
        microscope.beams.ion_beam.scanning.rotation.value = 0
        rotation = microscope.beams.ion_beam.scanning.rotation.value
    if not np.isclose(rotation, 0.0):
        raise ValueError(
            "Ion beam scanning rotation must be 0 degrees."
            "\nPlease change your system settings and try again."
            "\nCurrent rotation value is {}".format(rotation)
        )


def _validate_stage_coordinate_system(microscope):
    """Ensure the stage coordinate system is RAW.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.

    Notes
    -----
    The two available stage coordinate systems are:
    1. CoordinateSystem.RAW
        Coordinate system based solely on location of stage.
        This coordinate system is not affected by any adjustments and should
        bring stage to the exactly same position on a particular microscope.
    2. CoordinateSystem.SPECIMEN
        Coordinate system based on location on specimen.
        This coordinate system is affected by various additional adjustments
        that make it easier to navigate on a particular specimen. The most
        important one is link between Z coordinate and working distance.
        Specimen coordinate system is also used in XTUI stage control panel.

    Users have reported unexpected/unwanted behaviour with the operation of
    autolamella in cases where the SPECIMEN coordinate system is used
    (i.e. if the Z-Y link checkbox is ticked in the XT GUI). Avoiding this
    problem is why this validation check is run.
    """
    from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.RAW)


def _validate_model_weights_file(filename):
    import os
    from liftout.model import models
    weights_path = os.path.join(os.path.dirname(models.__file__), filename)
    if not os.path.exists(weights_path):
        raise ValueError(
            f"Unable to find model weights file {weights_path} specified."
        )


def _validate_configuration_values(microscope, dictionary):
    """Recursively traverse dictionary and validate all parameters.

    Parameters
    ----------
    dictionary : dict
        Any arbitrarily structured python dictionary.

    Raises
    -------
    ValueError
        The parameter is not within the available range for the microscope.
    """

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
                    _validate_horizontal_field_width(microscope=microscope, 
                        horizontal_field_widths=[item])

                if "milling_current" in key:
                    _validate_ion_beam_currents(microscope, [item])

                if "imaging_current" in key:
                    _validate_electron_beam_currents(microscope, [item])

                if "resolution" in key:
                    _validate_scanning_resolutions(microscope, [item])

                if "dwell_time" in key:
                    _validate_dwell_time(microscope, [item])

            if isinstance(item, str):
                if "application_file" in key:
                    _validate_application_files(microscope, [item])
                if "weights" in key:
                    _validate_model_weights_file(item)
                    
    return dictionary