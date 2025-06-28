# Overview

This package contains algorithms to perform high-order wavefront sensing and control (HOWFSC) for the CGI instrument on the Nancy Grace Roman Space Telescope ("Roman"). It can be general use for other coronagraph instruments and testbeds. It requires the Engineering Exposure Time Calculator (EETC) package that assumes a CCD or EMCCD as the detector.

### Install cgi-howfsc
To install simply download the package, change directories into the downloaded folder, and run:

	pip install .

This will install the required packages (see below) except for `eetc`.  To install `eetc`, follow the installation instructions at [https://github.com/nasa-jpl/cgi-eetc](https://github.com/nasa-jpl/cgi-eetc).

## Algorithms and test scripts

The contents are divided as follows:
- `howfsc/control/`: algorithms for wavefront control.  This implements the electric field conjugation (EFC) algorithm, which is an iterated weighted-least-squares solve with variable regularization, using a Jacobian (first-order sensitivity matrix).
- `howfsc/model/`: algorithms for optical propagation model for CGI, suitable for use with several coronagraph configurations.
- `howfsc/sensing/`: algorithms for wavefront sensing; this implements the pairwise probing algorithm, which does a pixel-by-pixel solve for complex-valued electric field with support from the optical model
- `howfsc/util/`: general purpose algorithms used by `control`, `model`, and `sensing`, but not specific to these; many are also reused for calibration applications (in a different repository)
- `scripts/`: general purpose test scripts.

Of particular use is `scripts/nulltest_gitl.py`, which runs wavefront control for the NFOV configuration using the contents of `model`, `control`, `sensing`, and `util`.  Future updates will support other configurations.  The contents of this repository are not part of the algorithms themselves.

It supports five sample configurations in this repository:
 - WFOV: an annular SPC mask operating in CGI Band 4 (`widefov`)
 - SPEC: a bowtie-shaped SPC mask operating in CGI Band 3 (`spectroscopy`)
 - NFOV (v1): an asymmetric HLC mask operation in CGI Band 1 (`narrowfov`) which uses premade phase maps to create the DM shape.  This setting is not as realistic with respect to CGI behavior, but produces good constrast immediately.
 - NFOV (v2): an asymmetric HLC mask operation in CGI Band 1 (`nfov_dm`) which uses the DM model itself to create the mask-specific DM settings.  This is more realistic than narrowfov, but starts at a higher contrast and required more iterations to get down to a target contrast.  (This is also realistic.)
 - NFOV (v3): an asymmetric HLC mask operation in CGI Band 1 (`nfov_flat`) which starts with flat DMs.  This is also a realistic configuration, but representsa difficult, pessimistic case that is expected to converge poorly with a single Jacobian.

There are also scripts to compute Jacobians for each of these five configurations in `jactest.py` (single-process) or `jactest_mp.py` (multiprocessed).  `make_test_jacs.py` will precompute Jacobians for all five configurations, suitable for use with `scripts/nulltest_gitl.py` and other future scripts.  Note: `make_test_jacs.py` may take >12 hours to run, so run it well in advance of when you need the Jacobians for `scripts/nulltest_gitl.py`.

## Unit tests and documentation

All functions and classes have unit tests stored in the same directory as the module.  A file `foo/bar/XXX.py` will have its tests in `foo/bar/ut_XXX.py`.  The script `testsuite.py` will run tests: calling `python testsuite.py ut_XXX.py` will run the tests in a particular file, while `python testsuite.py` with no further arguments will run every test in the repository.

All functions have documentation implemented as docstrings, including function behavior, required and optional arguments, and returns.  The tests will verify the behavior documented in the docstring.

Static analysis on the contents of this repository is done with pylint and a custom rcfile in `rc_pylint`.  Usual call format from the base directory is `pylint -rn --rcfile=./rc_pylint path/to/file.py`.

## Required packages

Python 3.7 or later is required.  You must also have the following external packages:
 - [numpy](https://pypi.python.org/pypi/numpy)
 - [scipy](https://pypi.python.org/pypi/scipy)
 - [astropy](https://pypi.python.org/pypi/astropy)
 - [pyyaml](https://github.com/yaml/pyyaml)
 - [joblib](https://pypi.python.org/pypi/joblib)
 - [eetc](https://github.jpl.nasa.gov/WFIRST-CGI/eetc)

The development was done with a 3.7 Linux distribution of [Anaconda](https://www.anaconda.com/download/), including the above packages, and it has been tested on Windows with a 3.8 Windows distribution of Anaconda; it has not been tested against other versions.

If you wish to review the code with the same static analyzer as it was tested with, you will also need [pylint](https://pypi.python.org/pypi/pylint).

GIT LFS must be installed in order to correctly retrieve large non-text files, such as FITS files, which are stored with the repository.

## Copyright statement
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

## Authors

- Eric Cady (JPL)
- A J Eldorado Riggs (JPL)
- David Marx (JPL)
- Michael Bottom (University of Hawaii)
- Kevin Ludwick (University of Alabama, Huntsville)

Originally based on the hcim codebase developed for the JPL High-Contrast Imaging Testbed (HCIT) circa January 2018, authored by Brian Kern, Eric Cady, David Marx, Byoung-Joon Seo, Camilo Mejia Prada, Al Niessner, Felipe Fregoso, Brian Gordon, and Dwight Moody.