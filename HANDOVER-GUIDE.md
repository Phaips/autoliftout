# Handover guide

## Quick reference
Project page: https://github.com/DeMarcoLab/liftout

The conda environment we use in the lab is named "fibsem". You can activate this environment before running any code by opening the Anaconda Prompt and typing `conda activate fibsem`.

Log in to the microscope PC on the "User" account.

Log in to the support PC "PFIB1" on the "Admin" account.

Quick start for interactive scripting: open the Anaconda Prompt on PFIB1 and type
```
cd MICROSCOPE\DeMarcoLab\liftout
conda activate fibsem
ipython
```
and when ipython starts, type
```
%load_ext autoreload
%autoreload 2
from liftout import *
microscope = initialize()
# ... whatever code you want to try next
```

The autoreload magic allows you to edit in Visual Studio Code and have it stay synchronized with ipython.

## Software tools we use
* git: we use [git for version control](https://www.atlassian.com/git)
* python: the [Anaconda python distribution](https://www.anaconda.com/products/individual) is preferred
* conda: we use [conda virtual environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to keep python dependencies neatly separated for each project. You can use the `conda list` command to see what is installed in the current environment.
* Visual Studio Code: you can use [Visual Studio Code](https://code.visualstudio.com/) or your preferred code editor.

### Python packages we use
* [numpy](https://numpy.org/): the [numpy API documentation here](https://numpy.org/doc/stable/reference/index.html)
* [scipy](https://www.scipy.org/): the [scipy API documentation here](https://www.scipy.org/docs.html)
* [scikit-image](https://scikit-image.org/): the [scikit-image documentation is here](https://scikit-image.org/docs/stable/)
* [matplotlib](https://matplotlib.org/): the [matplotlib documentation is here](https://matplotlib.org/3.3.3/contents.html)
* [click](https://click.palletsprojects.com/en/7.x/): for creating easy command line interfaces for our code
* The AutoScript python packages (a commercial product available from FEI). See INSTALLATION.md for more details.

## Help with AutoScript
You can access the AutoScript help documentation from the start menu of a computer that has AutoScript installed.

From the start menu, you can search for and open:
* "Autoscript Reference Manual" - for the API documentation
* "Autoscript User Guide" or "Autoscript User Guide - Offline" - for examples and a user guide.

Running Jupyter notebook cells in the "Autoscript User Guide" will execute those commands on the microscope.
Use the offline user guide if you don't want to accidentally move the microscopes stages/etc.

Autoscript uses several conventions:
* distances are always in units of METERS
* angles are always in units of RADIANS
* a positive rotation direction is CLOCKWISE

Remember you can always type one micron in units of meters like this: "1e-6"

## Running a complated program
If you have a completed python script you'd like to run, first activate the "fibsem" environment and then

```
conda activate fibsem
python path/to/script.py
```

We use 'entry points' when we install our python code (like autolamella, etc.)
which means that we give `python path/to/autolamella/main.py` an alias
`autolamella` that we can use from the command line, as a convenience.

## Interactive computation
### IPython
From the Anaconda Prompt, you can activate the fibsem environment and launch ipython:
```
conda activate fibsem
ipython
```

If you want to edit the code and keep your changes synchonised in ipython,
you can use the ipython autoreload magic. After ipython has started, type:
```
%load_ext autoreload
%autoreload 2
```

The first thing you will need to do is connect to the microscope,
then you can run whatever code you want:

```python
from autoscript_sdb_microscope_client import SdbMicroscopeClient
microscope = SdbMicroscopeClient()
microscope.connect('10.0.0.1')
# ... your code here
```

Or you can use the `initialize` convenience function to connect to the microscope:

```python
from liftout import *
microscope = initialize()
# ... your code here
```

### Breakpoints and debuggers

Alternatively, you can add a breakpoint to your script.
For example using the python debugger, you would add this line to your script:

```python
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
