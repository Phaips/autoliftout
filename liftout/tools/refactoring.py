from liftout import utils
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem import  acquire, movement

from liftout.fibsem.acquire import BeamType, ImageSettings, GammaSettings

settings = utils.load_config(r"C:\Users\Admin\Github\autoliftout\liftout\protocol_liftout.yml")
microscope = fibsem_utils.initialise_fibsem(ip_address=settings["system"]["ip_address"])
image_settings = {
    "resolution": settings["imaging"]["resolution"],
    "dwell_time": settings["imaging"]["dwell_time"],
    "hfw": settings["imaging"]["horizontal_field_width"],
    "autocontrast": True,
    "beam_type": BeamType.ION,
    "gamma": settings["gamma"],
    "save": False,
    "label": "test",
}

gamma_settings = GammaSettings(
    enabled = settings["gamma"]["correction"],
    min_gamma = settings["gamma"]["min_gamma"],
    max_gamma = settings["gamma"]["max_gamma"],
    scale_factor= settings["gamma"]["scale_factor"],
    threshold = settings["gamma"]["threshold"]
)

image_settings = ImageSettings(
    resolution = settings["imaging"]["resolution"],
    dwell_time = settings["imaging"]["dwell_time"],
    hfw =  settings["imaging"]["horizontal_field_width"],
    autocontrast = True,
    beam_type = BeamType.ION,
    gamma = gamma_settings,
    save = False,
    save_path = "",
    label = "test",
)


from pprint import pprint

pprint(gamma_settings)
pprint(image_settings)

# TODO: START_HERE: test that the new imaging works, then refactor main
img = acquire.new_image2(microscope, image_settings)