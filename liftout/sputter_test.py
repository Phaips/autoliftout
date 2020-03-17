import time

from autoscript_sdb_microscope_client import SdbMicroscopeClient


class MyMicroscope(SdbMicroscopeClient):
    def __init__(self, ip_address="10.0.0.1", autoconnect=True):
        super().__init__()
        if autoconnect:
            self.connect(ip_address)

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


if __name__ == "__main__":
    SPUTTERING_TIME = 20  # number of seconds for platnium sputtering
    microscope = MyMicroscope()
    microscope.sputter_platinum(SPUTTERING_TIME)
