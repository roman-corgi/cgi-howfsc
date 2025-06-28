# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Load up a coronagraph mode for interactive use with the command line
"""

import os

import numpy as np

import howfsc
from howfsc.model.mode import CoronagraphMode

if __name__ == "__main__":

    # User params
    mode = 'widefov'#'narrowfov'#'spectroscopy'#'nfov_dm'#
    #dmlist = [50*np.ones((48, 48)), 50*np.ones((48, 48))]

    # Load coronagraph data
    if mode == 'widefov':
        cfgpath = os.path.join(os.path.dirname(
                               os.path.abspath(howfsc.__file__)),
                               'model', 'testdata', 'widefov', 'widefov.yaml')
        pass
    elif mode == 'narrowfov':
        cfgpath = os.path.join(os.path.dirname(
                           os.path.abspath(howfsc.__file__)),
                           'model', 'testdata', 'narrowfov', 'narrowfov.yaml')
        pass
    elif mode == 'spectroscopy':
        cfgpath = os.path.join(os.path.dirname(
                     os.path.abspath(howfsc.__file__)),
                     'model', 'testdata', 'spectroscopy', 'spectroscopy.yaml')
    elif mode == 'nfov_dm':
        cfgpath = os.path.join(os.path.dirname(
                        os.path.abspath(howfsc.__file__)),
                        'model', 'testdata', 'narrowfov', 'narrowfov_dm.yaml')
        pass
    else:
        raise Exception('Bad mode string')

    cfg = CoronagraphMode(cfgpath)

    # For a given DM setting (in dmlist, here), return a focal-plane intensity
    # image (NOT e-field) in each wavefront control band.
    imdhlist = []
    for sl in cfg.sl_list:
        edm0 = sl.eprop(cfg.initmaps)
        ely = sl.proptolyot(edm0)
        edh0 = sl.proptodh(ely)
        imdh = np.abs(edh0)**2
        imdhlist.append(imdh)
        pass
