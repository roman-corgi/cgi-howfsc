# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
this script is an example of how to create a dm_crosstalk yaml file for use
in a CoronagraphMode cfgfile, such as howfsc/model/testdata/widefov/widefov.yaml

In this example, we create a yaml from the measured crosstalk for DM2, as given
by John Krist. This crosstalk is specifically for the (row, col) = [1, 0]
actuator offset.
"""

import os

import howfsc
from howfsc.model.dm_crosstalk import dm_crosstalk_fits_to_yaml


localpath = os.path.dirname(howfsc.__file__)
crosstalk_fn = os.path.join(localpath,
                            'model/testdata/ut/ut_DM2_crosstalk.fits')

# begin, put a copy of the crosstalk yaml in the path for each mode
for mode in ['narrowfov', 'spectroscopy', 'widefov', '', 'ut/testdata']:

    # the path and filename for the crosstalk yaml
    testdata_pn = os.path.join(localpath, 'model', 'testdata', mode)
    yaml_fn = os.path.join(testdata_pn, 'ut_DM2_crosstalk.yaml')

    # read and write
    dm_crosstalk_fits_to_yaml([crosstalk_fn,], [[1, 0]], yaml_fn)
