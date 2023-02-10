
# Automated cryo-liftout for electron microscopy

`liftout` is a python package for automated liftout for cryo-lamella preparation.

## Getting Started

How to get started using autoliftout


### User Guide
Before starting please read the [User Guide](/UserGuide.md)

### Installation

Clone this repository:

```
git clone https://github.com/DeMarcoLab/autoliftout.git
```

Install dependencies and package

```bash
cd autoliftout
conda env create -f environment.yml
conda activate autoliftout
pip install -e .

```

### Install OpenFIBSEM dependency

Follow the instructions in [OpenFIBSEM](https://github.com/DeMarcoLab/fibsem) to install the fibsem dependencies.

**ThermoFisher Microscope**
Load Application Files

- Load the application files, autolamella.xml, and cryo_Pt_dep.xml into XtUI following the manufacturer instructions.
- These application files are used for milling and platinum deposition respectively. To provide your own, please edit the system.yaml file.

### Preparation

System File (system.yaml)

- Edit the system.yaml file to match your system configuration

Protocol File (protocol.yaml)

- Edit the protocol.yaml file to match your desired protocol

Preparing the Needle

- Mill the end of your manipulator so there is a 10um flat surface. TODO: image

Preparing the State
-  Save the lamella and landing states (Raw coordinates) in the protocol.yaml file.
- These are the initial positions for selecting lamella and landing positions.

### Running the Program
Once you have installed all the dependencies and completed the preparation steps, run:

```bash
conda activate autoliftout
autoliftout_ui
```

The AutoLiftout User Interface should appear on the screen, ready to start.
