import time

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.sdb_microscope.specimen._manipulator import Manipulator


def initialize(ip_address='10.0.0.1'):
    """Initialize connection to FIBSEM microscope with Autoscript."""
    microscope = MyMicroscope()
    microscope.connect(ip_address)
    return microscope


class MyMicroscope(SdbMicroscopeClient):
    def __init__(self, ip_address="10.0.0.1", autoconnect=True):
        super().__init__()
        if autoconnect:
            self.connect(ip_address)
        self.specimen.manipulator = MyManipulator()

    def sputter_platinum(self, sputter_time=20, *,
                         default_application_file="autolamella",
                         sputter_application_file="cryo_Pt_dep",
        ):
        """Sputter platinum over the sample.

        Parameters
        ----------
        sputter_time : float, optional
            Time in seconds for platinum sputtering. Default is 20 seconds.
        """
        # Setup
        original_active_view = self.imaging.get_active_view()
        self.imaging.set_active_view(1)  # the electron beam view
        self.patterning.clear_patterns()
        self.patterning.set_default_application_file(sputter_application_file)
        # Run sputtering
        self.patterning.create_line()  # 1um, at zero in the FOV
        self.beams.electron_beam.blank()
        if self.patterning.state == "Idle":
            print('Sputtering with platinum for {} seconds...'.format(sputter_time))
            self.patterning.start()  # asynchronous patterning
            time.sleep(sputter_time)
            self.patterning.stop()
        else:
            raise RuntimeError(
                "Can't sputter platinum, patterning state is not ready."
            )
        # Cleanup
        self.patterning.clear_patterns()
        self.beams.electron_beam.unblank()
        self.patterning.set_default_application_file(default_application_file)
        self.imaging.set_active_view(original_active_view)


class MyManipulator(Manipulator):
    """Needle for cryo-liftout."""
    def __init__(self):
        super().__init__()

    @property
    def parked_position(self):
        try:
            parked_position = self._parked_position
        except AttributeError:
            raise RuntimeError('')
        return parked_position

    def set_parked_position(self):
        self._parked_position(self.specimen.manipulator.current_position)


