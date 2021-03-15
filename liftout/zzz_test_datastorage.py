import sys, getopt, glob, os
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import math
import time
import scipy.ndimage as ndi
from scipy import fftpack, misc
from PIL import Image, ImageDraw, ImageFilter
from matplotlib.patches import Circle

import os, sys, glob
import datetime

PRETILT_DEGREES = 27

class BeamType(Enum):
    ION = 'ION'
    ELECTRON = 'ELECTRON'


stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')


# GLOBAL VARIABLE
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
        print('check if directory exists...')
        if not os.path.isdir( self.path_for_image ):
            print('creating directory')
            os.mkdir(self.path_for_image)
        self.fileName = self.path_for_image + 'step%02d'%(self.step_counter) + '_'  + id + '.tif'
        print(self.fileName)
        image.save(self.fileName)

storage = Storage()


def read_image(DIR, fileName, gaus_smooth=1):
    fileName = DIR + '/' + fileName
    image = Image.open(fileName)
    image = np.array(image)
    if image.shape[1] == 1536:
        image = image[0:1024, :]
    if image.shape[1] == 3072:
        image = image[0:2048, :]
    #image = ndi.filters.gaussian_filter(image, sigma=gaus_smooth) #median(imageTif, disk(1))
    return image

if __name__ == '__main__':
    if 1:
        storage.NewRun()
        print('TEST take image, save image')
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings
        from autoscript_sdb_microscope_client.structures import StagePosition
        ip_address = '10.0.0.1'
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        stage = microscope.specimen.stage
        image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)  # TODO: user input resolution, must match

        yy = input('MOVE TO 27 tilt: HIGH AND LOW RES, press Enter when ready...')
        microscope.beams.electron_beam.horizontal_field_width.value = 150e-6  # TODO: yaml use input
        microscope.beams.ion_beam.horizontal_field_width.value      = 150e-6  # TODO: yaml use input
        microscope.imaging.set_active_view(1)
        eb = microscope.imaging.grab_frame(image_settings)
        microscope.imaging.set_active_view(2)
        ib = microscope.imaging.grab_frame(image_settings)

        storage.SaveImage(eb, id='eb')
        storage.SaveImage(ib, id='ib')


    if 0:
        print('Storage test')
        DIR = r'Y:\Sergey\codes\HS auto lamella1\2021.02.15_trench_alignment'
        fileName_ib_tilt27 = r'001_I_T27_HFW_50um_1us_1536_1024_001.tif'
        fileName_ib_tilt37 = r'002_I_T37_HFW_50um_1us_1536_1024_002.tif'
        storage.AddDirectory(r'Y:\Sergey\codes\HS auto lamella1\01.15.2021_cross_corrolation_for_stage_rotation')
        storage.NewRun()




