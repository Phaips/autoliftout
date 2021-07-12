import logging
import yaml
from pprint import pprint
import datetime
import time
import os
import sys
from enum import Enum

from liftout.main import *
# from liftout.user_input import load_config

########################################### TODO: REMOVE BELOW ###############################################################
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
    # settings_dict = _add_missing_keys(settings_dict)
    # settings_dict = _format_dictionary(settings_dict)
    # settings = Settings(**settings_dict)  # convert to python dataclass
    return settings_dict


def configure_logging(log_filename='logfile', log_level=logging.INFO):
    """Log to the terminal and to file simultaneously."""
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')#datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[
            logging.FileHandler(log_filename+timestamp+'.log'),
            logging.StreamHandler(),
        ])
    print(log_filename+timestamp+'.log')

class Storage():
    def __init__(self, DIR=''):
        self.DIR = DIR
        self.NEEDLE_REF_IMGS         = [] # dict()
        self.NEEDLE_WITH_SAMPLE_IMGS = [] # dict()
        self.LANDING_POSTS_REF       = []
        self.TRECHNING_POSITIONS_REF = []
        self.MILLED_TRENCHES_REF     = []
        self.liftout_counter = 0
        self.step_counter   = 0
        self.settings = ''
    def AddDirectory(self,DIR):
        self.DIR = DIR
    def NewRun(self, prefix='RUN'):
        self.__init__(self.DIR)
        if self.DIR == '':
            self.DIR = os.getcwd() # # dirs = glob.glob(saveDir + "/ALIGNED_*")        # nn = len(dirs) + 1
        stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')
        self.saveDir = self.DIR + '/' + prefix + '_' + stamp
        self.saveDir = self.saveDir.replace('\\', '/')
        os.mkdir(self.saveDir)
    def SaveImage(self, image, dir_prefix='', id=''):
        if len(dir_prefix) > 0:
            self.path_for_image = self.saveDir + '/'  + dir_prefix + '/'
        else:
            self.path_for_image = self.saveDir + '/'  + 'liftout%03d'%(self.liftout_counter) + '/'
        print(self.path_for_image)
        if not os.path.isdir( self.path_for_image ):
            print('creating directory')
            os.mkdir(self.path_for_image)
        self.fileName = self.path_for_image + 'step%02d'%(self.step_counter) + '_'  + id + '.tif'
        print(self.fileName)
        image.save(self.fileName)
# storage = Storage() # global variable

def initialize(ip_address='10.0.0.1'):
    """Initialize connection to FIBSEM microscope with Autoscript."""
    from autoscript_sdb_microscope_client import SdbMicroscopeClient

    microscope = SdbMicroscopeClient()
    microscope.connect(ip_address)
    return microscope

########################################### TODO: REMOVE ABOVE ###############################################################

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings


class AutoLiftoutStatus(Enum):
    Initialize = 0
    Setup = 1
    Milling = 2
    Liftout = 3
    Landing = 4
    Cleanup = 5
    Reset = 6
    Finished = 7

# TODO: logging and storage should be consistent and consolidated
class AutoLiftout:

    def __init__(self, config_filename) -> None:

        # initialise autoliftout
        configure_logging("log_")

        self.settings = load_config(config_filename)

        self.storage = Storage()
        self.storage.NewRun(prefix='test_run')
        self.storage.settings = self.settings

        self.microscope = initialize(self.settings["system"]["ip_address"])

        self.current_status = AutoLiftoutStatus.Initialize


    def _report_status(self):
        """Helper function for reporting liftout status"""
        print(f"\nCurrent Status: {self.current_status.name}")
        print(f"Liftout Counter: {self.storage.liftout_counter}")

    def setup(self):
        """ Initial setup of grid and selection for lamella and landing positions"""
        self.current_status = AutoLiftoutStatus.Setup
        self._report_status()
        autocontrast(self.microscope, beam_type=BeamType.ELECTRON)
        autocontrast(self.microscope, beam_type=BeamType.ION)

        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
        self.microscope.imaging.set_active_view(1)
        self.microscope.beams.electron_beam.horizontal_field_width.value = 2750e-6
        eb = self.microscope.imaging.grab_frame(image_settings)
        self.storage.SaveImage(eb, id='grid')
        self.storage.step_counter += 1

        if ask_user("Do you want to sputter the whole sample grid with platinum? yes/no: "):
            sputter_platinum_over_whole_grid(microscope)

        autocontrast(self.microscope, beam_type=BeamType.ELECTRON)
        self.microscope.beams.electron_beam.horizontal_field_width.value = 2750e-6
        eb = self.microscope.imaging.grab_frame(image_settings)
        self.storage.SaveImage(eb, id='grid_Pt_deposition')
        self.storage.step_counter += 1

        print("Please select the landing positions and check eucentric height manually.")
        self.landing_coordinates, self.original_landing_images = find_coordinates(self.microscope, name="landing position", move_stage_angle="landing")
        self.lamella_coordinates, self.original_trench_images  = find_coordinates(self.microscope, name="lamella",          move_stage_angle="trench")
        self.zipped_coordinates = list(zip(self.lamella_coordinates, self.landing_coordinates))
        self.storage.LANDING_POSTS_POS_REF = self.original_landing_images
        self.storage.LAMELLA_POS_REF       = self.original_trench_images

    def _get_fake_setup_data(self):
        self.zipped_coordinates = [[1, 1], [1, 1], [1, 1], [1, 1]]
        self.original_landing_images = [[1], [1], [1], [1]]
        self.original_trench_images = [[1], [1], [1], [1]]
    def run_liftout(self):

        self._report_status()
        self._get_fake_setup_data()

        # Start liftout for each lamella
        for i, (lamella_coord, landing_coord) in enumerate(self.zipped_coordinates):
            landing_reference_images      = self.original_landing_images[i]
            lamella_area_reference_images = self.original_trench_images[i]
            self.single_liftout(self.microscope, self.settings, landing_coord, lamella_coord, landing_reference_images, lamella_area_reference_images)
            self.storage.liftout_counter += 1
        self.current_status = AutoLiftoutStatus.Finished
        self._report_status()

    def single_liftout(self, microscope, settings, landing_coord, lamella_coord, original_landing_images, original_lamella_area_images):

        ### move to the previously stored position and correct the position using the reference images:
        microscope.specimen.stage.absolute_move(lamella_coord)
        # realign_sample_stage(microscope, image, template, beam_type=BeamType.ION, correct_z_height=False)
        # correct_stage_drift_using_reference_eb_images(microscope, original_lamella_area_images, plot=False)

        # mill
        self.current_status = AutoLiftoutStatus.Milling
        self._report_status()
        # mill_lamella(microscope, settings, confirm=False)

        # lift-out
        self.current_status = AutoLiftoutStatus.Liftout
        self._report_status()
        # liftout_lamella(microscope, settings)

        # land
        self.current_status = AutoLiftoutStatus.Landing
        self._report_status()
        # land_lamella(microscope, landing_coord, original_landing_images)

        # resharpen needle
        self.current_status = AutoLiftoutStatus.Reset
        self._report_status()
        # sharpen_needle(microscope)

# TODO: replace global instance of storage with class instance.... lots of work

config_filename = sys.argv[1]
# config_filename = "../liftout/protocol_liftout.yml"
# config_filename = r"\\ad.monash.edu\home\User007\prcle2\Documents\demarco\autoliftout\liftout\protocol_liftout.yml"
# autoliftout = AutoLiftout(config_filename)
# autoliftout.setup()
# autoliftout.run_liftout()



# pprint(autoliftout.settings)