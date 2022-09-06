import logging
import os
import winsound
from pathlib import Path

import scipy.ndimage as ndi
import yaml
from autoscript_sdb_microscope_client.structures import AdornedImage
from fibsem import utils as fibsem_utils
from fibsem.constants import METRE_TO_MILLIMETRE
from liftout.config import config
from liftout.patterning import MillingPattern
from liftout.sample import Lamella, Sample, create_experiment
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGridLayout, QLabel

from fibsem.ui import utils as fibsem_ui


###################
def draw_grid_layout(sample: Sample):
    gridLayout = QGridLayout()
    # TODO: refactor this better

    # Only add data is sample positions are added
    if not sample.positions:
        label = QLabel()
        label.setText("No Lamella have been selected. Press Setup to start.")
        gridLayout.addWidget(label)
        return gridLayout

    sample_images = [[] for _ in sample.positions]

    # initial, mill, jcut, liftout, land, reset, thin, polish
    config.DISPLAY_REFERENCE_FNAMES

    # headers
    headers = [
        "Sample No",
        "Position",
        "Reference",
        "Milling",
        "J-Cut",
        "Liftout",
        "Landing",
        "Reset",
        "Thinning",
        "Polishing",
    ]
    for j, title in enumerate(headers):
        label_header = QLabel()
        label_header.setText(title)
        label_header.setMaximumHeight(80)
        label_header.setStyleSheet(
            "font-family: Arial; font-weight: bold; font-size: 18px;"
        )
        label_header.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(label_header, 0, j)

    lamella: Lamella
    for i, lamella in enumerate(sample.positions.values()):

        # load the exemplar images for each sample
        qimage_labels = []
        for img_basename in config.DISPLAY_REFERENCE_FNAMES:
            fname = os.path.join(lamella.path, f"{img_basename}.tif")
            imageLabel = QLabel()
            imageLabel.setMaximumHeight(150)

            if os.path.exists(fname):
                adorned_image = lamella.load_reference_image(img_basename)

                def set_adorned_image_as_qlabel(
                    adorned_image: AdornedImage,
                    label: QLabel,
                    shape: tuple = (150, 150),
                ) -> QLabel:

                    image = QImage(
                        ndi.median_filter(adorned_image.data, size=3),
                        adorned_image.data.shape[1],
                        adorned_image.data.shape[0],
                        QImage.Format_Grayscale8,
                    )
                    label.setPixmap(QPixmap.fromImage(image).scaled(*shape))
                    label.setStyleSheet("border-radius: 5px")

                    return label

                imageLabel = set_adorned_image_as_qlabel(adorned_image, imageLabel)

            qimage_labels.append(imageLabel)

        sample_images[i] = qimage_labels

        # display information on grid
        row_id = i + 1

        # display lamella info
        label_sample = QLabel()
        label_sample.setText(
            f"""Sample {lamella._number:02d} \n{lamella._petname} \nStage: {lamella.current_state.stage.name}"""
        )
        label_sample.setStyleSheet("font-family: Arial; font-size: 12px;")
        label_sample.setMaximumHeight(150)
        gridLayout.addWidget(label_sample, row_id, 0)

        # display lamella position
        lamella_coordinates = lamella.lamella_state.absolute_position
        landing_coordinates = lamella.landing_state.absolute_position
        
        label_pos = QLabel()
        pos_text = f"""Pos: ({lamella_coordinates.x*METRE_TO_MILLIMETRE:.2f}, {lamella_coordinates.y*METRE_TO_MILLIMETRE:.2f}, {lamella_coordinates.z*METRE_TO_MILLIMETRE:.2f})\n"""
        if lamella.landing_selected:
            pos_text += f"""Land: ({landing_coordinates.x*METRE_TO_MILLIMETRE:.2f}, {landing_coordinates.y*METRE_TO_MILLIMETRE:.2f}, {landing_coordinates.z*METRE_TO_MILLIMETRE:.2f})\n"""

        label_pos.setText(pos_text)
        label_pos.setStyleSheet("font-family: Arial; font-size: 12px;")
        label_pos.setMaximumHeight(150)

        gridLayout.addWidget(label_pos, row_id, 1)

        # display exemplar images
        # ref, trench, jcut, liftout, land, reset, thin, polish
        gridLayout.addWidget(
            sample_images[i][0], row_id, 2, Qt.AlignmentFlag.AlignCenter
        )  # TODO: fix missing visual boxes
        gridLayout.addWidget(
            sample_images[i][1], row_id, 3, Qt.AlignmentFlag.AlignCenter
        )
        gridLayout.addWidget(
            sample_images[i][2], row_id, 4, Qt.AlignmentFlag.AlignCenter
        )
        gridLayout.addWidget(
            sample_images[i][3], row_id, 5, Qt.AlignmentFlag.AlignCenter
        )
        gridLayout.addWidget(
            sample_images[i][4], row_id, 6, Qt.AlignmentFlag.AlignCenter
        )
        gridLayout.addWidget(
            sample_images[i][5], row_id, 7, Qt.AlignmentFlag.AlignCenter
        )
        gridLayout.addWidget(
            sample_images[i][6], row_id, 8, Qt.AlignmentFlag.AlignCenter
        )
        gridLayout.addWidget(
            sample_images[i][7], row_id, 9, Qt.AlignmentFlag.AlignCenter
        )

    gridLayout.setRowStretch(9, 1)  # grid spacing
    return gridLayout

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
    winsound.Beep(freq, duration)

def load_configuration_from_ui(parent=None) -> dict:

    # load config
    logging.info(f"Loading configuration from file.")
    play_audio_alert()

    options = QtWidgets.QFileDialog.Options()
    config_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent,
        "Load Configuration",
        config.base_path,
        "Yaml Files (*.yml, *.yaml)",
        options=options,
    )

    if config_filename == "":
        raise ValueError("No protocol file was selected.")

    protocol = fibsem_utils.load_protocol(config_filename)
    
    return protocol

def setup_experiment_sample_ui(parent_ui):
    """Setup the experiment sample by either creating or loading a sample"""

    default_experiment_path = os.path.join(config.base_path, "log")
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
    if parent_ui:

        # update the ui
        parent_ui.label_experiment_name.setText(f"Experiment: {sample.name}")
        parent_ui.statusBar.showMessage(f"Experiment {sample.name} loaded.")
        parent_ui.statusBar.repaint()

    return sample

def load_experiment_ui(parent, default_experiment_path: Path) -> Sample:

    # load_experiment
    experiment_path = QtWidgets.QFileDialog.getExistingDirectory(
        parent, "Choose Log Folder to Load", directory=default_experiment_path
    )
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

    options = QtWidgets.QFileDialog.Options()
    config_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent_ui,
        "Select Protocol File",
        config.base_path,
        "Yaml Files (*.yml, *.yaml)",
        options=options,
    )

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

