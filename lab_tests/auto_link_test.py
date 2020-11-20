import numpy as np


def initialize(ip_address='10.0.0.1'):
    """Initialize connection to FIBSEM microscope with Autoscript."""
    from autoscript_sdb_microscope_client import SdbMicroscopeClient

    microscope = SdbMicroscopeClient()
    microscope.connect(ip_address)
    return microscope


def linked_within_z_tolerance(microscope, expected_z=4e-3, tolerance=1e-4):
    """Check if the sample stage is linked and at the expected z-height.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    expected_z : float, optional
        Correct height for linked stage in z, in meters, by default 4e-3
    tolerance : float, optional
        Must be within this absolute tolerance of expected stage z height,
        in meters, by default 1e-4

    Returns
    -------
    bool
        Returns True if stage is linked and at the correct z-height.
    """
    # # Check the microscope stage is linked in z
    # if not microscope.specimen.stage.is_linked:
    #     return False
    # Check the microscope stage is at the correct height
    z_stage_height = microscope.specimen.stage.current_position.z
    if np.isclose(z_stage_height, expected_z, atol=tolerance):
        return True
    else:
        return False


def auto_link_stage(microscope, expected_z=4e-3, tolerance=50e-4):
    """Automatically focus and link sample stage z-height.

    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    expected_z : float, optional
        Correct height for linked stage in z, in meters, by default 4e-3
    tolerance : float, optional
        Must be within this absolute tolerance of expected stage z height,
        in meters, by default 1e-4
    """
    from autoscript_sdb_microscope_client.structures import StagePosition
    from liftout.acquire import autofocus

    counter = 0
    while not linked_within_z_tolerance(microscope):
        print(counter)
        if counter > 3:
            raise(UserWarning("Could not auto-link z stage height."))
            break
        # Focus and re-link z stage height
        autofocus(microscope)
        microscope.specimen.stage.link()
        microscope.specimen.stage.absolute_move(StagePosition(z=expected_z))
        counter += 1


if __name__ == "__main__":
    microscope = initialize()
    print("Is stage already linked?", linked_within_z_tolerance(microscope))
    print("Running automatic linking...")
    auto_link_stage(microscope)
    print("Is stage linked?", linked_within_z_tolerance(microscope))
    print("Finished.")
