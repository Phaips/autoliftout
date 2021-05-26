# Automated cryo-liftout for electron microscopy.

`liftout` is a python package for automated cryo-lamella preparation and
liftout.

## Citation
If you find this useful, please cite our work.

This project builds upon some of our earlier work on automated cryo-lamellae preparation, which you can find software for at https://github.com/DeMarcoLab/autolamella

## Software license
This software is released under the terms of the 3 clause BSD license.
There is NO WARRANTY either express or implied.
See [LICENSE](LICENSE) for details.

## Installation
See [INSTALLATION](INSTALLATION.md) for a more detailed guide.

* Ensure you have Python 3.6 available
* Install Autoscript (a commercial product from FEI)
and configure it for use with your FEI microscope
* Download the latest `liftout` release wheel from https://github.com/DeMarcoLab/autolamella/releases
* Pip install the wheel file (`.whl`) into your python environment

## Running the program
1. Create/edit the protocol file with details appropriate for your sample.
Protocols are YAML files with the format shown by `protocol_example.yml`
(see [USER_INPUT.md](USER_INPUT.md) for more details).
2. Activate the virtual environment where you have installed `autolamella` and
the dependencies (eg: if you are a conda user, open the Anaconda Prompt and
use "conda activate my-environment-name" or
"source activate my-environment-name", substituting the name of your own
virtual environment.)
3. Launch the program from the terminal by typing:
`liftout path/to/your_protocol.yml`
4. Follow the user prompts to interactively select new lamella locations,
before beginning the batch ion milling.

