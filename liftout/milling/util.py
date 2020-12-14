from liftout.user_input import ask_user

__all__ = [
    "setup_ion_milling",
    "confirm_and_run_milling",
]


def setup_ion_milling(microscope, *,
                      application_file="Si_Alex",
                      patterning_mode="Parallel",
                      ion_beam_field_of_view=82.9e-6):
    """Setup for rectangle ion beam milling patterns.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    application_file : str, optional
        Application file for ion beam milling, by default "Si_Alex"
    patterning_mode : str, optional
        Ion beam milling pattern mode, by default "Parallel".
        The available options are "Parallel" or "Serial".
    ion_beam_field_of_view : float, optional
        Width of ion beam field of view in meters, by default 59.2e-6
    """
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_beam_type(2)  # ion beam default
    microscope.patterning.set_default_application_file(application_file)
    microscope.patterning.mode = patterning_mode
    microscope.patterning.clear_patterns()  # clear any existing patterns
    microscope.beams.ion_beam.horizontal_field_width.value = ion_beam_field_of_view


def confirm_and_run_milling(microscope, milling_current, *,
                            imaging_current=20e-12):
    """Run all the ion beam milling pattterns, after user confirmation.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    milling_current : float
        The ion beam milling current to use, in Amps.
    imaging_current : float, optional
        The ion beam imaging current to return to, by default 20 pico-Amps.
    """
    # TODO: maybe display to the user how long milling will take
    if ask_user("Do you want to run the ion beam milling?"):
        print("Ok, running ion beam milling now...")
        microscope.beams.ion_beam.beam_current.value = milling_current
        microscope.patterning.run()
        print("Returning to the ion beam imaging current now.")
        microscope.patterning.clear_patterns()
        microscope.beams.ion_beam.beam_current.value = imaging_current
        print("Ion beam milling complete.")
