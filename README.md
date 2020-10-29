[![DOI](https://zenodo.org/badge/276416522.svg)](https://zenodo.org/badge/latestdoi/276416522)

# hdestimator --- history dependence estimator

The history dependence estimator tool provides a set of
routines that facilitate the estimation of history dependence in
neural spiking data, using estimators based on information-theoretical
measures, as presented in (Rudelt et al, in prep.).

A guide for how to use the tool can be found under
[docs/howto.pdf](docs/howto.pdf).


## Dependencies
- Python (>=3.2)
- h5py
- pyyaml
- numpy
- scipy
- mpmath
- matplotlib

### Optional Dependencies
- cython, for significantly faster running times


## Installation
Python packages can be installed either via your operating system's package manager or
using eg pip or conda.

* ubuntu: `sudo apt install python3-h5py python3-yaml python3-numpy python3-scipy python3-mpmath python3-matplotlib cython3`

* fedora: `sudo dnf install python3-h5py python3-pyyaml python3-numpy python3-scipy python3-mpmath python3-matplotlib python3-Cython`

* using pip: `pip install h5py pyyaml numpy scipy mpmath matplotlib cython3`

* using conda: `conda install h5py pyyaml numpy scipy mpmath matplotlib cython`


The repository can be cloned via ssh

```
git clone git@github.com:Priesemann-Group/hdestimator.git
```
or https
```
git clone https://github.com/Priesemann-Group/hdestimator.git
```

Once downloaded, you can change directory into the the repository's folder:

`cd hdestimator`

and you're ready to go!

### Recommended: Compile the Cython modules

From within the repository's base folder, change into the `src` directory:

`cd src`

There, compile the Cython modules:

`python3 setup.py build_ext --inplace`

If no errors occured (warnings are OK), you are all set.

### Windows users

Under Windows, you can use the tool eg through miniconda.

Install [miniconda for python3, 32bit](https://docs.conda.io/en/latest/miniconda.html).

To meet the dependencies to compile the Cython modules, download and install [Visual Studio](https://visualstudio.microsoft.com/downloads/).
There, select Desktop development with C++ and install
* MSVC v140 - VS 2015 C++ build tools (v14.00) (more recent versions probably work, too)
* Windows 10 SDK (10.0.18362.0)

Then compile the modules by running the commands as above.
