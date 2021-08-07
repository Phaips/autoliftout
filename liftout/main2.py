from enum import Enum
from liftout import utils
from liftout.fibsem import acquire
from liftout.fibsem.acquire import BeamType
from liftout.fibsem import movement
from autoscript_sdb_microscope_client.structures import *


class AutoLiftoutStatus(Enum):
    Setup = 0
    Milling = 1
    Liftout = 2
    Landing = 3
    Reset = 4
    Cleanup = 5
    Finished = 6


class AutoLiftout:
    def __init__(self, microscope, config_filename='../protocol_liftout.yml', run_name="run") -> None:

        self.save_path = utils.make_logging_directory(directory="log", prefix=run_name)  # TODO: fix pathing
        utils.configure_logging(save_path=self.save_path, log_filename='logfile_')
        self.current_status = AutoLiftoutStatus.Setup
        self.response = False

        self.settings = utils.load_config(config_filename)
        self.pretilt_degrees = 27
        # TODO: add pretilt_degrees to protocol
        self.microscope = microscope

        if self.microscope:
            movement.move_to_sample_grid(self.microscope, self.settings)

            acquire.autocontrast(self.microscope, BeamType.ELECTRON)
            acquire.autocontrast(self.microscope, BeamType.ION)

            self.image_settings = {'resolution': "1536x1024", 'dwell_time': 1e-6,
                                   'hfw': 2750e-6, 'brightness': None,
                                   'contrast': None, 'autocontrast': True,
                                   'save': True, 'label': 'grid',
                                   'beam_type': BeamType.ELECTRON,
                                   'save_path': self.save_path}

            self.eb_image = acquire.new_image(self.microscope, self.image_settings)