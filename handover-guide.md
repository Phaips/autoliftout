# Handover guide

## Quick reference
Project page: https://github.com/DeMarcoLab/liftout

The conda environment we use in the lab is named "fibsem". You can activate this environment before running any code by opening the Anaconda Prompt and typing `conda activate fibsem`.

## Tools we use
* git: we use [git for version control](https://www.atlassian.com/git)
* python: the [Anaconda python distribution](https://www.anaconda.com/products/individual) is preferred
* conda: we use [conda virtual environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to keep python dependencies neatly separated for each project. You can use the `conda list` command to see what is installed in the current environment.

## Help with AutoScript
You can access the AutoScript help documentation from the start menu of a computer that has AutoScript installed.

From the start menu, you can search for and open:
* "Autoscript Reference Manual" - for the API documentation
* "Autoscript User Guide" or "Autoscript User Guide - Offline" - for examples and a user guide.

## Running a complated program
If you have a completed python script you'd like to run, first activate the "fibsem" environment and then

```
conda activate fibsem
python path/to/script.py
```

## Interactive computation
### IPython

```
conda activate fibsem
ipython
```

```python
from autoscript_sdb_microscope_client import SdbMicroscopeClient
microscope = SdbMicroscopeClient()
microscope.connect('10.0.0.1')
# ... your code here
```

### Breakpoints and debuggers

Alternatively, you can add a breakpoint to your script.
For example using the python debugger, you would add this line to your script:

```
import pdb; pdb.set_trace()
```

And then you would start the script as you usually would, and it will run until it hits the breakpoint:

```
conda activate fibsem
python path/to/script.py
```

If you're not familiar with using debuggers, you might like Nina Zakharenko's excellent tutorial: ["Goodbye print statement, hello debugger"](https://www.nnja.io/post/2020/pycon2020-goodbye-print-hello-debugger/)

## Software setup
Please see DEVELOPERS.md for a detailed *development installation* guide (i.e. using `pip install -e .`).

Please see INSTALLATION.md for a detailed release installation guide (i.e. pip installing from a wheel file).

## Creating new releases
The new release process is described in the last section of DEVELOPERS.md

In brief, once the code is finalized and tests added/pass you can create the release wheel file using:

```
conda activate fibsem
python setup.py bdist_wheel sdist
```
