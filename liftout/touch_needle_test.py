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
        print(event.ydata, event.xdata)
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


def sputter_platinum(microscope, sputter_time=20, *,
                     default_application_file="autolamella",
                     sputter_application_file="cryo_Pt_dep",
    ):
    """Sputter platinum over the sample.

    Parameters
    ----------
    sputter_time : int, optional
        Time in seconds for platinum sputtering. Default is 20 seconds.
    """
    # Setup
    original_active_view = microscope.imaging.get_active_view()
    microscope.imaging.set_active_view(1)  # the electron beam view
    microscope.patterning.clear_patterns()
    microscope.patterning.set_default_application_file(sputter_application_file)
    # Run sputtering
    start_x = 0
    start_y = 0
    end_x = 1e-6
    end_y = 1e-6
    depth = 2e-6
    microscope.patterning.create_line(start_x, start_y, end_x, end_y, depth)  # 1um, at zero in the FOV
    microscope.beams.electron_beam.blank()
    if microscope.patterning.state == "Idle":
        print('Sputtering with platinum for {} seconds...'.format(sputter_time))
        microscope.patterning.start()  # asynchronous patterning
    else:
        raise RuntimeError(
            "Can't sputter platinum, patterning state is not ready."
        )
    for i in tqdm(range(int(sputter_time))):
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
    logging.info("Sputtering finished.")


def actual_angle(stage, pretilt_degrees):
    """The actual angle of the sample surface, relative to the electron beam.

    Parameters
    ----------
    stage : microscope.specimen.stage
        AutoScript sample stage.
    pretilt_degrees : float
        Pre-tilt of sample holder, in degrees.

    Returns
    -------
    float
        The actual angle of the sample surface, in degrees.
    """
    reported_angle = np.rad2deg(stage.current_position.t)
    actual_angle = reported_angle + pretilt_degrees
    return actual_angle


def tilt(stage, target_angle, pretilt_degrees):
    """Tilt sample stage, taking into account any stage pre-tilt.

    Parameters
    ----------
    stage : microscope.specimen.stage
        Autoscript sample stage.
    target_angle : float
        Target angle for sample surface after stage tilt, in degrees.
    pretilt_degrees : float
        Pre-tilt of sample holder, in degrees.

    Returns
    -------
    angle_to_move : float
        In degrees.
        angle_to_move = target_angle - pretilt_degrees
    """
    angle_to_move = (target_angle - pretilt_degrees)
    stage.absolute_move(StagePosition(t=np.deg2rad(angle_to_move)))
    return angle_to_move


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
    pretilt_degrees = ???
    x_safety_buffer = ???  # in meters (needle safety buffer distance)
    y_safety_buffer = ???  # in meters (needle safety buffer distance)
    z_safety_buffer = ???
    ideal_z_gap = 50e-9  # ideally we want the needletip 50nm (almost touching)
    jcut_tilt_degrees = 6  # sample surface should be at this angle (6 degrees)
    needle_closest_distance = 50e-9  # ideally we want to leave a ~50nm gap

    # Setup
    microscope = initialize()
    needle = microscope.specimen.manipulator  # needle manipulator
    stage = microscope.specimen.stage  # sample stage

    # Assumptions we can write assert statements for
    assert stage.is_linked
    assert np.isclose(0, microscope.beams.electron_beam.scanning.rotation.value)
    assert np.isclose(0, microscope.beams.ion_beam.scanning.rotation.value)

    # Begin test
    # Tilt sample so we are at the right angle for liftout/jcut
    tilt(stage, jcut_tilt_degrees, pretilt_degrees)
    # Insert needle (park position already calibrated by the user)
    needle.insert()
    park_position = needle.current_position
    # Take a picture
    print("New electron beam image")
    electron_image = new_electron_image(microscope, settings=None)
    pixelsize_x = electron_image.metadata.binary_result.pixel_size.x
    pixelsize_y = electron_image.metadata.binary_result.pixel_size.y
    print("Electron beam pixelsize:", pixelsize_x)
    # USER INPUT - Click to mark needle tip and target position in the electron beam image.
    print("Please click the needle tip position")
    needletip_location = select_point(electron_image)
    x_needletip_location = needletip_location[0]  # coordinates in x-y format
    y_needletip_location = needletip_location[1]  # coordinates in x-y format
    print("Please click the lamella target position")
    target_location = select_point(electron_image)
    x_target_location = target_location[0]  # pixels, coordinates in x-y format
    y_target_location = target_location[1]  # pixels, coordinates in x-y format
    print("Needletip location:", needletip_location)
    print("Target location:", target_location)
    # Calculate the distance between the needle tip and the target.
    x_distance = pixelsize_x * (x_target_location - x_needletip_location)
    y_distance = pixelsize_y * (y_target_location - y_needletip_location)
    print("Estimated movement in X:", x_distance)
    print("Estimated movement in Y:", y_distance)

    # MOVEMENT IN Z
    # Take an ion beam image
    print("Taking a new ion beam image")
    ion_image = new_ion_image(microscope, settings=None)
    pixelsize_x = ion_image.metadata.binary_result.pixel_size.x
    pixelsize_y = ion_image.metadata.binary_result.pixel_size.y
    print("Pixelsize of ion beam image:", pixelsize_x)
    print("Please click the needle tip position")
    ion_needletip_location = select_point(ion_image)
    print("Please click the target position")
    ion_target_location = select_point(ion_image)
    print("Needletip location (ion beam):", ion_needletip_location)
    print("Target location (ion beam):", ion_target_location)
    # compare with results from electron image
    x_ion_distance = pixelsize_x * (ion_target_location[0] - ion_needletip_location[0])
    print("COMPARING CALCULATIONS")
    print("X")
    print("Electon image, calculated x distance:", x_distance)
    print("Ion beam image, calculated x distance:", x_ion_distance)
    n_pixels_in_y = ion_target_location[1] - ion_needletip_location[1]
    y_ion_distance = pixelsize_y * (n_pixels_in_y * np.cos(np.deg2rad(52)))
    print("Y")
    print("Electon image, calculated y distance:", y_distance)
    print("Ion beam image, calculated y distance:", y_ion_distance)

    # calculating Z
    n_pixels_in_y = ion_target_location[1] - ion_needletip_location[1]
    z_distance = pixelsize_y * (n_pixels_in_y * np.sin(np.deg2rad(52)))
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
    sputter_platinum(microscope, sputter_time=20)


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
