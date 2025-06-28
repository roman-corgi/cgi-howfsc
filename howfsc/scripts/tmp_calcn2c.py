# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Cheap user script to manually build individual n2c arrays
"""

import time
import os

import howfsc
from howfsc.model.mode import CoronagraphMode
from howfsc.control.calcn2c import calcn2c

if __name__ == '__main__':
    cfgpath = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                           'model', 'testdata', 'narrowfov',
                           'narrowfov.yaml')
    cfg = CoronagraphMode(cfgpath)

    idx = 2
    nrow = 153
    ncol = 153

    t0 = time.time()
    n2c = calcn2c(cfg, idx, nrow, ncol, cfg.initmaps)
    t1 = time.time()
    print(str(t1-t0) + " seconds")
