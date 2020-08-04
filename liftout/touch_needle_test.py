import logging
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tqdm

from autoscript_sdb_microscope_client.structures import *
from autoscript_sdb_microscope_client.enumerations import *


def configure_logging(log_filename='logfile.log', log_level=logging.DEBUG):
    """Log to the terminal and to file simultaneously."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ])


def initialize(ip_address='10.0.0.1'):
    """Initialize connection to FIBSEM microscope with Autoscript."""
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    microscope = SdbMicroscopeClient()
    microscope.connect(ip_address)
    return microscope


def quick_plot(image, median_smoothig=3):
    """Display image with matplotlib.pyplot

    Parameters
    ----------
    image : Adorned image or numpy array
        Input image.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    display_image = image.data
    if median_smoothig is not None:
        display_image = ndi.median_filter(display_image, size=median_smoothig)
    height, width = display_image.shape
    try:
        pixelsize_x = image.metadata.binary_result.pixel_size.x
        pixelsize_y = image.metadata.binary_result.pixel_size.y
    except AttributeError:
        extent_kwargs = [-(width / 2), +(width / 2), -(height / 2), +(height / 2)]
        ax.set_xlabel("Distance from origin (pixels)")
    else:
        extent_kwargs = [
            -(width / 2) * pixelsize_x,
            +(width / 2) * pixelsize_x,
            -(height / 2) * pixelsize_y,
            +(height / 2) * pixelsize_y,
        ]
        ax.set_xlabel(
            "Distance from origin (meters) \n" "1 pixel = {} meters".format(pixelsize_x)
        )
    ax.set_xlim(extent_kwargs[0], extent_kwargs[1])
    ax.set_ylim(extent_kwargs[2], extent_kwargs[3])
    ax.imshow(display_image, cmap="gray", extent=extent_kwargs)
    return fig, ax


def select_point(image):
    """Return location of interactive user click on image.

    Parameters
    ----------
    image : AdornedImage or 2D numpy array.

    Returns
    -------
    coords
          Coordinates of last point clicked in the image.
          Coordinates are in x, y format.
          Units are the same as the matplotlib figure axes.
    """
    fig, ax = quick_plot(image)
    coords = []

    def on_click(event):
        print(event.xdata, event.ydata)
        coords.append(event.ydata)
        coords.append(event.xdata)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return np.flip(coords[-2:], axis=0)  # coordintes in x, y format


def new_ion_image(microscope, settings=None):
    """Take new ion beam image.

    Uses whichever camera settings (resolution, dwell time, etc) are current.

    Parameters
    ----------
    microscope : Autoscript microscope object.

    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    microscope.imaging.set_active_view(2)  # the ion beam view
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image


def new_electron_image(microscope, settings=None):
    """Take new electron beam image.

    Uses whichever camera settings (resolution, dwell time, etc) are current.

    Parameters
    ----------
    microscope : Autoscript microscope object.

    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    microscope.imaging.set_active_view(1)  # the electron beam view
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image

def sputter_platinum(microscope, sputter_time=60, *,
                     default_application_file="autolamella",
                     sputter_application_file="cryo_Pt_dep",
    ):
    """Sputter platinum over the sample.

    Parameters
    ----------
    sputter_time : int, optional
        Time in seconds for platinum sputtering. Default is 60 seconds.
    """
    # Setup
    original_active_view = microscope.imaging.get_active_view()
    microscope.imaging.set_active_view(1)  # the electron beam view
    microscope.patterning.clear_patterns()
    microscope.patterning.set_default_application_file(sputter_application_file)
    microscope.patterning.set_default_beam_type(1)  # set electron beam for patterning
    # Create sputtering pattern
    start_x = -15e-6
    start_y = +15e-6
    end_x = +15e-6
    end_y = +15e-6
    depth = 2e-6
    pattern = microscope.patterning.create_line(start_x, start_y, end_x, end_y, depth)  # 1um, at zero in the FOV
    pattern.time = sputter_time + 0.1
    # Run sputtering with progress bar
    microscope.beams.electron_beam.blank()
    if microscope.patterning.state == "Idle":
        print('Sputtering with platinum for {} seconds...'.format(sputter_time))
        microscope.patterning.start()  # asynchronous patterning
    else:
        raise RuntimeError(
            "Can't sputter platinum, patterning state is not ready."
        )
    for i in tqdm.tqdm(range(int(sputter_time))):
        time.sleep(1)  # update progress bar every second
    if microscope.patterning.state == "Running":
        microscope.patterning.stop()
    else:
        logging.warning("Patterning state is {}".format(self.patterning.state))
        loggging.warning("Consider adjusting the patterning line depth.")
    # Cleanup
    microscope.patterning.clear_patterns()
    microscope.beams.electron_beam.unblank()
    microscope.patterning.set_default_application_file(default_application_file)
    microscope.imaging.set_active_view(original_active_view)
    microscope.patterning.set_default_beam_type(2)  # set ion beam
    logging.info("Sputtering finished.")


def actual_angle(stage, pretilt_degrees=27):
    """The actual angle of the sample surface, relative to the electron beam.

    Parameters
    ----------
    stage : microscope.specimen.stage
        AutoScript sample stage.
    pretilt_degrees : float
        Pre-tilt of sample holder, in degrees. Default is 27 degrees.

    Returns
    -------
    float
        The actual angle of the sample surface, in degrees.
    """
    reported_angle = np.rad2deg(stage.current_position.t)
    actual_angle = reported_angle + pretilt_degrees
    return actual_angle


# def tilt(stage, target_angle, pretilt_degrees=27):
#     """Tilt sample stage, taking into account any stage pre-tilt.

#     Parameters
#     ----------
#     stage : microscope.specimen.stage
#         Autoscript sample stage.
#     target_angle : float
#         Target angle for sample surface after stage tilt, in degrees.
#     pretilt_degrees : float
#         Pre-tilt of sample holder, in degrees. Default is 27 degrees.

#     Returns
#     -------
#     angle_to_move : float
#         In degrees.
#         angle_to_move = target_angle - pretilt_degrees
#     """
# ONLY WORKS FOR ION BEAM TILTING!!
# FOR ELECTON BEAM TILTING WE NEED TO ADD pretilt_degrees
# THIS IS BECAUSE WE ROTATE BY 180 degrees between electron & ion beam views
#     angle_to_move = (target_angle - pretilt_degrees)
#     stage.absolute_move(StagePosition(t=np.deg2rad(angle_to_move)))
#     return angle_to_move


def tilt_to_jcut_angle(stage, *, jcut_angle_degrees=6, pretilt_degrees=27):
    flat_to_electron_beam(stage, pretilt_degrees=pretilt_degrees)
    stage.relative_move(StagePosition(t=np.deg2rad(jcut_angle_degrees)))
    return stage.current_position


def x_corrected_needle(expected_x):
    """Needle movement in X, XTGui coordinates (Electron coordinate).

    Parameters
    ----------
    expected_y : in meters
    stage_tilt : in degrees

    Returns
    -------
    ManipulatorPosition
    """
    return ManipulatorPosition(x=expected_x, y=0, z=0)  # no adjustment needed


def y_corrected_needle(expected_y, stage_tilt):
    """Needle movement in Y, XTGui coordinates (Electron coordinate).

    Parameters
    ----------
    expected_y : in meters
    stage_tilt : in degrees

    Returns
    -------
    ManipulatorPosition
    """
    tilt_radians = np.deg2rad(stage_tilt)
    y_move = +np.cos(tilt_radians) * expected_y
    z_move = +np.sin(tilt_radians) * expected_y
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def z_corrected_needle(expected_z, stage_tilt):
    """Needle movement in Z, XTGui coordinates (Electron coordinate).

    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in degrees

    Returns
    -------
    ManipulatorPosition
    """
    tilt_radians = np.deg2rad(stage_tilt)
    y_move = -np.sin(tilt_radians) * expected_z
    z_move = +np.cos(tilt_radians) * expected_z
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def y_height_adjusted_electron_beam(y_pixels, pixelsize_y, stage):

    sample_surface_angle = actual_angle(stage)
    shrink_factor = np.cos(np.deg2rad(sample_surface_angle))
    y_height_adjusted = (y_pixels * pixelsize_y) * shrink_factor
    return y_height_adjusted


def y_height_adjusted_ion_beam():
    pass

def flat_to_electron_beam(stage, pretilt_degrees=27):
    """Make the sample surface flat to the electron beam."""
    rotation = np.deg2rad(290)
    tilt = np.deg2rad(pretilt_degrees)
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position


def flat_to_ion_beam(stage, pretilt_degrees=27):
    """Make the sample surface flat to the ion beam."""
    rotation = np.deg2rad(290 - 180)
    tilt = np.deg2rad(52 - pretilt_degrees)
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    stage.absolute_move(StagePosition(t=tilt), stage_settings)
    return stage.current_position


def mill_jcut_edge(microscope, pretilt_degrees=27, application_file="Si_Alex"):
    # USER INPUT PARAMETERS
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_beam_type(2)  # ion beam default
    # milling_current = 2e-9  # ROOM TEMP copper sample, smaller milling current for J-cut
    milling_current = 0.74e-9 # CRYO YEAST sample
    jcut_angle = 6  # in degrees
    angle_correction_factor = np.sin(np.deg2rad(52 - jcut_angle))
    expected_lamella_depth = 5e-6  # in microns
    jcut_trench_width = 1e-6  # in meters
    jcut_milling_depth = 3e-6  # in meters
    jcut_top_length = 12e-6
    ion_beam_field_of_view = 59.2e-6  # in meters
    # Setup
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_application_file(application_file)
    microscope.patterning.mode = "Parallel"
    microscope.patterning.clear_patterns()  # clear any existing patterns
    microscope.beams.ion_beam.horizontal_field_width.value = ion_beam_field_of_view
    # Create milling pattern
    # Right hand side of J-cut (long side)
    extra_bit = 3e-6
    center_x = +((jcut_top_length - jcut_trench_width) / 2)
    center_y = ((expected_lamella_depth - (extra_bit / 2)) / 2) * angle_correction_factor
    width = jcut_trench_width
    height =  (expected_lamella_depth + extra_bit) * angle_correction_factor
    depth = jcut_milling_depth
    jcut_lhs_pattern = microscope.patterning.create_rectangle(center_x, center_y, width, height, depth)



def mill_lamella_trenches(microscope, application_file="Si_Heidi"):
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_beam_type(2)  # ion beam default
    # INPUT PARAMETERS
    imaging_current = microscope.beams.ion_beam.beam_current.value  # ~20 pico-Amps for cryo yeast
    # milling_current = 7.6e-9  # in Amps (copper sample, milled with Argon)
    milling_current = 7.6e-9  # in Amps (cryo-yeast sample)
    ion_beam_field_of_view = 59.2e-6  # in meters
    milling_depth = 3e-6  # in meters
    trench_width = 15e-6  # in meters
    trench_height = 15e-6  # in meters
    lamella_thickness = 2e-6  # intended thickness of finished lamella
    # TODO: we need a bigger buffer size, exact value to be determined
    buffer = 0.5e-6  # the edges of the trenches are usually not exactly precise
    # Setup
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_application_file(application_file)
    microscope.patterning.mode = "Serial"
    microscope.patterning.clear_patterns()  # clear any existing patterns
    microscope.beams.ion_beam.horizontal_field_width.value = ion_beam_field_of_view
    # Add upper trench
    center_x = 0
    center_y = +(lamella_thickness / 2) + buffer + (trench_height / 2)
    width = trench_width
    height = trench_height
    depth = milling_depth
    milling_pattern_1 = microscope.patterning.create_cleaning_cross_section(
        center_x, center_y, width, height, depth)
    milling_pattern_1.scan_direction = "TopToBottom"
    # Add lower trench
    center_x = 0
    center_y = -((lamella_thickness / 2) + buffer + (trench_height / 2))
    width = trench_width
    height = trench_height
    depth = milling_depth
    milling_pattern_2 = microscope.patterning.create_cleaning_cross_section(
        center_x, center_y, width, height, depth)
    milling_pattern_2.scan_direction = "BottomToTop"
    # # Confirm before milling
    # user_input = input("")
    # if user_input == 'yes':
    #     microscope.beams.ion_beam.beam_current.value = milling_current
    #     microscope.patterning.run()
    #     microscope.beams.ion_beam.beam_current.value = imaging_current
    # else:
    #     "User did not confirm milling job, continue without milling."
    # # Cleanup
    # microscope.patterning.clear_patterns()
    return (milling_pattern_1, milling_pattern_2)


def mill_jcut(microscope, pretilt_degrees=27, application_file="Si_Alex"):
    # USER INPUT PARAMETERS
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_beam_type(2)  # ion beam default
    # milling_current = 2e-9  # ROOM TEMP copper sample, smaller milling current for J-cut
    milling_current = 0.74e-9 # CRYO YEAST sample
    jcut_angle = 6  # in degrees
    angle_correction_factor = np.sin(np.deg2rad(52 - jcut_angle))
    expected_lamella_depth = 5e-6  # in microns
    jcut_trench_width = 1e-6  # in meters
    jcut_milling_depth = 3e-6  # in meters
    jcut_top_length = 12e-6
    ion_beam_field_of_view = 59.2e-6  # in meters
    # Setup
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_application_file(application_file)
    microscope.patterning.mode = "Parallel"
    microscope.patterning.clear_patterns()  # clear any existing patterns
    microscope.beams.ion_beam.horizontal_field_width.value = ion_beam_field_of_view
    # Create milling patterns
    # Top bar of J-cut
    center_x = 0
    center_y = expected_lamella_depth * angle_correction_factor
    width = jcut_top_length
    height = jcut_trench_width
    depth = jcut_milling_depth
    jcut_top_pattern = microscope.patterning.create_rectangle(center_x, center_y, width, height, depth)
    # Left hand side of J-cut (long side)
    extra_bit = 3e-6
    center_x = -((jcut_top_length - jcut_trench_width) / 2)
    center_y = ((expected_lamella_depth - (extra_bit / 2)) / 2) * angle_correction_factor
    width = jcut_trench_width
    height =  (expected_lamella_depth + extra_bit) * angle_correction_factor
    depth = jcut_milling_depth
    jcut_lhs_pattern = microscope.patterning.create_rectangle(center_x, center_y, width, height, depth)
    # Right hand side of J-cut (short side)
    jcut_rightside_remaining = 1.5e-6  # in microns
    width = jcut_trench_width
    height = (expected_lamella_depth - jcut_rightside_remaining) * angle_correction_factor
    center_x = +((jcut_top_length - jcut_trench_width) / 2)
    center_y = jcut_rightside_remaining + (height / 2)
    depth = jcut_milling_depth
    jcut_rhs_pattern = microscope.patterning.create_rectangle(center_x, center_y, width, height, depth)

    # # Confirm before milling
    # user_input = input("")
    # if user_input == 'yes':
    #     microscope.beams.ion_beam.beam_current.value = milling_current
    #     microscope.patterning.run()
    #     microscope.beams.ion_beam.beam_current.value = imaging_current
    # else:
    #     "User did not confirm milling job, continue without milling."
    # # Cleanup
    # microscope.patterning.clear_patterns()
    # microscope.patterning.mode = "Serial"



# 130 micron field of view - better to fit lamella trench and fiducial in.


def main():
    # ASSUME
    # * The scan rotation of the microscope is zero
    # * The needle park position has been calibrated
    # * The sample is at eucentric height
    # * The sample is in focus
    # * The z-height has been linked accurately
    #
    # We start with the sample flat to the ion beam and the lamella already cut

    # USER INPUTS
    pretilt_degrees = 27
    # x_safety_buffer = ???  # in meters (needle safety buffer distance)
    # y_safety_buffer = ???  # in meters (needle safety buffer distance)
    # z_safety_buffer = ???

    ideal_z_gap = 50e-9  # ideally we want the needletip 50nm (almost touching)
    jcut_tilt_degrees = 6  # sample surface should be at this angle (6 degrees)

    # Setup
    microscope = initialize()
    needle = microscope.specimen.manipulator  # needle manipulator
    stage = microscope.specimen.stage  # sample stage

    # Assumptions we can write assert statements for
    assert stage.is_linked
    assert np.isclose(0, microscope.beams.electron_beam.scanning.rotation.value)
    assert np.isclose(0, microscope.beams.ion_beam.scanning.rotation.value)

    # Begin test
    flat_to_electron_beam(stage)
    flat_to_ion_beam(stage)
    mill_lamella_trenches(microscope)
    flat_to_ion_beam(stage)

    # Tilt sample so we are at the right angle for liftout/jcut
    # make flat to electron beam, then tilt to J-cut angle
    mill_jcut(stage, jcut_tilt_degrees, pretilt_degrees)
    # Adjust the lamella position (eucentric height isn't perfect, we need to do a correlation)

    # Move stage so the sample is flat to the electron beam

    # Insert needle (park position already calibrated by the user)
    needle.insert()
    park_position = needle.current_position

    # First step is to move -160 microns in z (blind moving)
    # The park position is always the same, we'll wind up with the needletip about 20 microns from the surface.
    stage_tilt = np.rad2deg(stage.current_position.t)
    z_move = z_corrected_needle(-160e-6, stage_tilt)
    needle.relative_move(z_move)
    # And we also move back a bit in x, just so the needle is never overlapping our target on the lamella
    x_move = x_corrected_needle(-10e-6)
    needle.relative_move(x_move)

    # Insert the Multichem
    multichem = microscope.gas.get_multichem()
    # multichem.insert()  # TODO: check this! - goes only to the elctron beam position :(
    # TODO: How to set the gas for the multichem (we want "Pt cryo" gas)

    # Take a picture
    electron_image = new_electron_image(microscope, settings=None)
    # USER INPUT - Click to mark needle tip and target position in the electron beam image.
    print("Please click the needle tip position")
    needletip_location = select_point(electron_image)
    print("Please click the lamella target position")
    target_location = select_point(electron_image)

    x_needletip_location = needletip_location[0]  # coordinates in x-y format
    y_needletip_location = needletip_location[1]  # coordinates in x-y format
    x_target_location = target_location[0]  # pixels, coordinates in x-y format
    y_target_location = target_location[1]  # pixels, coordinates in x-y format

    # Calculate the distance between the needle tip and the target.
    x_distance = x_target_location - x_needletip_location
    y_distance = y_target_location - y_needletip_location
    x_move = x_corrected_needle(x_distance)
    y_move = y_corrected_needle(y_distance, stage_tilt)

    # MOVEMENT IN Z
    # Take an ion beam image
    print("Taking a new ion beam image")
    ion_image = new_ion_image(microscope, settings=None)
    print("Pixelsize of ion beam image:", pixelsize_x)
    print("Please click the needle tip position")
    ion_needletip = select_point(ion_image)
    print("Please click the target position")
    ion_target = select_point(ion_image)
    print("Needletip location (ion beam):", ion_needletip)
    print("Target location (ion beam):", ion_target)
    # calculating Z
    z_distance = -(ion_target[1] - ion_needletip[1] / np.sin(np.deg2rad(52)))
    print("Z")
    print("Ion beam image, calculated z distance:", z_distance)

    # Move needle most of the way, minus some "safety buffer" distance
    import pdb; pdb.set_trace()
    x_move = x_corrected_needle(x_distance - x_safety_buffer)
    needle.relative_move(x_move)

    y_move = y_corrected_needle(y_distance - y_safety_buffer)
    needle.relative_move(y_move)

    z_move = z_corrected_needle(z_distance - z_safety_buffer)
    needle.relative_move(z_move)

    import pdb; pdb.set_trace()
    x_move2 = x_corrected_needle(x_safety_buffer)
    needle.relative_move(x_move2)

    y_move2 = y_corrected_needle(y_safety_buffer)
    needle.relative_move(y_move2)

    z_move2 = z_corrected_needle(z_safety_buffer - ideal_z_gap)
    needle.relative_move(z_move2)

    # Adjust the magnification of the electron beam image (we can increase this now the needle tip is closer)

    # Take a new electron beam image.

    # USER INPUT - Click to mark needle tip and target position in the electron beam image. (We may try and automate this step later on)

    # Calculate the remaining distance between the needle tip and the target.

    # Move the needle the remaining distance to the target, leaving some predefined gap in Z between the needle & target.

    # Run the platinum sputtering to weld the needle to the target. (May simulate this step during the test, if working at room temperature instead of cryo.)
    import pdb; pdb.set_trace()
    sputter_platinum(microscope, sputter_time=60)
    # Cut the lamella free on the right hand side
    mill_jcut_edge(microscope)
    # Retract the needle
    # Move sample stage to landing grid
    # landing grid position is flat to the ION BEAM (the landing posts stick straight up from the grid, so it's)
    # Now move -160 microns in z (blind moving)
    # The park position is always the same, we'll wind up with the needletip about 20 microns from the surface.
    stage_tilt = np.rad2deg(stage.current_position.t)
    z_move = z_corrected_needle(-160e-6, stage_tilt)
    needle.relative_move(z_move)
    # And we also move back a bit in x, just so the needle & lamella is not overlapping the target on the landing post
    x_move = x_corrected_needle(-20e-6)
    needle.relative_move(x_move)
    # Take a really nice, high res electron beam image, so you can see where to move the needle
    # Electron beam image with 500ns dwell time is just ok not great, 2us dwell time is really nice (but very slow ~1 minute to acquire)


if __name__ == "__main__":
    message = """
    Have you double-checked that:
    * The scan rotation of the microscope is zero
    * The needle park position has been calibrated
    * The sample is at eucentric height
    * The sample is in focus
    * The z-height has been linked accurately
    \nPlease answer yes/no\n
    """
    user_input = input(message)
    if user_input == 'yes':
        try:
            main()
        except KeyboardInterrupt:
            print("KeyboardInterrupt encountered, quitting program.")
    else:
        print("Ok, cancelling program.")
