import logging
import os
from pathlib import Path

import numpy as np
import yaml
from fibsem import utils as fibsem_utils
from fibsem.patterning import MillingPattern
from fibsem.ui import utils as fibsem_ui
from PyQt5 import QtWidgets

from liftout.config import config
from liftout.structures import Lamella, Sample, create_experiment


def update_stage_label(label: QtWidgets.QLabel, lamella: Lamella):

    stage = lamella.current_state.stage
    status_colors = {
        "Initialisation": "gray",
        "Setup": "gold",
        "MillTrench": "coral",
        "MillJCut": "coral",
        "Liftout": "seagreen",
        "Landing": "dodgerblue",
        "Reset": "salmon",
        "Thinning": "mediumpurple",
        "Polishing": "cyan",
        "Finished": "silver",
        "Failure": "gray"
    }
    label.setText(f"Lamella {lamella._number:02d} \n{stage.name}")
    label.setStyleSheet(
        str(
            f"background-color: {status_colors[stage.name]}; color: white; border-radius: 5px"
        )
    )

def play_audio_alert(freq: int = 1000, duration: int = 500) -> None:
    import winsound
    winsound.Beep(freq, duration)

def load_configuration_from_ui(parent=None) -> dict:

    # load config
    logging.info(f"Loading configuration from file.")
    play_audio_alert()

    options = QtWidgets.QFileDialog.Options()
    config_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent,
        "Load Configuration",
        config.BASE_PATH,
        "Yaml Files (*.yml, *.yaml)",
        options=options,
    )

    if config_filename == "":
        raise ValueError("No protocol file was selected.")

    protocol = fibsem_utils.load_protocol(config_filename)
    
    return protocol

def setup_experiment_sample_ui(parent_ui):
    """Setup the experiment sample by either creating or loading a sample"""

    default_experiment_path = os.path.join(config.BASE_PATH, "log")
    default_experiment_name = "default_experiment"

    response = fibsem_ui.message_box_ui(
        title="AutoLiftout Startup", text="Do you want to load a previous experiment?"
    )

    # load experiment
    if response:
        print(f"{response}: Loading an existing experiment.")
        sample = load_experiment_ui(parent_ui, default_experiment_path)

    # new_experiment
    else:
        print(f"{response}: Starting new experiment.")
        #  TODO: enable selecting log directory in ui
        sample = create_experiment_ui(parent_ui, default_experiment_name)

    logging.info(f"Experiment {sample.name} loaded.")
    logging.info(f"{len(sample.positions)} lamella loaded from {sample.path}")

    return sample

def load_experiment_ui(parent, default_experiment_path: Path) -> Sample:

    # load_experiment
    experiment_path = QtWidgets.QFileDialog.getExistingDirectory(
        parent, "Choose Log Folder to Load", directory=default_experiment_path
    )
    # from fibsem.ui import utils as ui_utils
    # experiment_path = ui_utils.get_existing_directory(parent=parent, caption="Choose Log Folder to Load", directory=default_experiment_path)

    # if the user doesnt select a folder, start a new experiment
    # nb. should we include a check for invalid folders here too?
    if experiment_path == "":
        experiment_path = default_experiment_path

    sample_fname = os.path.join(experiment_path, "sample.yaml")
    sample = Sample.load(sample_fname)

    return sample

def create_experiment_ui(parent, default_experiment_name: str) -> Sample:
    # create_new_experiment
    experiment_name, okPressed = QtWidgets.QInputDialog.getText(
        parent,
        "New AutoLiftout Experiment",
        "Enter a name for your experiment:",
        QtWidgets.QLineEdit.Normal,
        default_experiment_name,
    )
    if not okPressed or experiment_name == "":
        experiment_name = default_experiment_name

    sample = create_experiment(experiment_name=experiment_name, path=None)

    return sample

def update_milling_protocol_ui(milling_pattern: MillingPattern, milling_stages: list, parent_ui=None):

    config_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent_ui,
        "Select Protocol File",
        config.BASE_PATH,
        "Yaml Files (*.yml, *.yaml)",
    )
    # from fibsem.ui import utils as ui_utils
    # config_filename, _ = ui_utils.open_existing_file_ui(parent = parent_ui, 
    #     caption = "Select Protocol File", directory = config.BASE_PATH, 
    #     filter_ext:="Yaml Files (*.yml, *.yaml)")  

    if config_filename == "":
        raise ValueError("No protocol file was selected.")

    protocol = fibsem_utils.load_protocol(config_filename)

    protocol_key = config.PATTERN_PROTOCOL_MAP[milling_pattern]

    if len(milling_stages) == 1:
        stage_settings = list(milling_stages.values())[0]
        protocol[protocol_key].update(stage_settings)

    else:
        stage_settings = list(milling_stages.values())[0]
        protocol[protocol_key].update(stage_settings)
        for i, stage_settings in enumerate(milling_stages.values()):
            protocol[protocol_key]["protocol_stages"][i].update(stage_settings)

    # save yaml file
    with open(config_filename, "w") as f:
        yaml.safe_dump(protocol, f)

    logging.info(f"Updated protocol: {config_filename}")
    # TODO: i dont think this updates the current protocol? need to refresh in that case





def create_overview_image(sample: Sample) -> np.ndarray:

    import scipy.ndimage as ndi

    PAD_PX = 10
    BASE_SHAPE = None

    vstack = None
    for i, lamella in enumerate(sample.positions.values()):

        hstack = None
        for fname in config.DISPLAY_REFERENCE_FNAMES:
            
            path = os.path.join(lamella.path, f"{fname}.tif")

            if os.path.exists(path):
                image = lamella.load_reference_image(fname).thumbnail
            else:
                image = np.zeros(shape=BASE_SHAPE)

            if BASE_SHAPE is None:
                BASE_SHAPE = image.data.shape

            image = np.pad(image.data, pad_width=PAD_PX)


            if hstack is None:
                hstack = image
            else:
                hstack = np.hstack([hstack, image])


        hstack = np.pad(hstack, pad_width=PAD_PX)
        if vstack is None:
            vstack = hstack
        else:
            vstack = np.vstack([vstack, hstack])
        
    vstack = vstack.astype(np.uint8)
    overview_image = ndi.median_filter(vstack, size=3)

    return overview_image





def get_completion_stats(sample: Sample) -> tuple:
    """Get the current completetion stats for lifout"""    
    from liftout.structures import AutoLiftoutStage
    n_stages = AutoLiftoutStage.Finished.value # init and failure dont count

    lam: Lamella
    active_lam = 0
    completed_stages = 0
    for lam in sample.positions.values():

        # dont count failure
        if lam.is_failure or lam.current_state.stage.value == 99:
            continue
        
        active_lam += 1
        completed_stages += lam.current_state.stage.value

    total_stages = n_stages * active_lam
    perc_complete = completed_stages / total_stages


    return n_stages, active_lam, completed_stages, total_stages, perc_complete
