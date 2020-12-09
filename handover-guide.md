# Handover guide

## Quick links
Project page: https://github.com/DeMarcoLab/liftout

## Tools we use
* git: we use [git for version control](https://www.atlassian.com/git)
* python: the [Anaconda python distribution](https://www.anaconda.com/products/individual) is preferred
* conda: we use [conda virtual environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to keep python dependencies neatly separated for each project

## Help with AutoScript
You can access the AutoScript help documentation from the start menu of a computer that has AutoScript installed.

From the start menu, you can search for and open:
* "Autoscript Reference Manual" - for the API documentation
* "Autoscript User Guide" or "Autoscript User Guide - Offline" - for examples and a user guide.

## Running a complated program
```
activate fibsem
autolamella path/to/protocol.yml
```

## Interactive computation
### IPython

```
activate fibsem
ipython
```

```python
from autoscript_sdb_microscope_client import SdbMicroscopeClient
microscope = SdbMicroscopeClient()
microscope.connect('10.0.0.1')
# ... your code here
```

### Breakpoints

`import pdb; pdb.set_trace()`

## Software setup
Please see INSTALLATION.md for a detailed installation guide.

##
