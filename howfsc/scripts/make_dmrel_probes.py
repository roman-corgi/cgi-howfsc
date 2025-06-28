# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=line-too-long
"""
Functions to build a relative DM probe for unit test configurations
"""

import os
import argparse

import numpy as np
import astropy.io.fits as pyfits

import howfsc
from howfsc.model.mode import CoronagraphMode
from howfsc.util.prop_tools import efield, open_efield, make_dmrel_probe

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog='python make_dmrel_probes.py', description="Create a relative DM probe setting corresponding to a desired probe height for howfsc unit test configurations.  Run with 'python -i' to look at outputs (focal-plane intensities in npint0/npint1/npint2)")
    ap.add_argument('-t', '--target', default=1e-5, help="Target probe height, defaults to 1e-5.", type=float)
    ap.add_argument('--mode', default='nfov_dm', choices=['widefov', 'narrowfov', 'spectroscopy', 'nfov_dm', 'nfov_flat'], help="coronagraph mode from test data; must be one of 'widefov', 'narrowfov', 'nfov_dm', 'nfov_flat', or 'spectroscopy'.  Defaults to 'nfov_dm'.")
    ap.add_argument('--write', action='store_true', help="write FITS file outputs to the howfsc/model/testdata/ directory")
    args = ap.parse_args()

    # User params
    mode = args.mode
    _target = args.target
    write = args.write

    howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))

    # kwargs need to be tuned per mode--get area right, and most importantly
    # get center right.  If center if obscured, probe might be very large
    # artificially.  Need to look at probe images overlaid on pupil plane
    # obscurations (front + SP + Lyot).
    if mode == 'nfov_dm':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'narrowfov')
        cfgpath = os.path.join(modelpath, 'narrowfov_dm.yaml')
        _cfg = CoronagraphMode(cfgpath)

        kwargs = {'dact':46.3,
                  'xcenter':0,
                  'ycenter':-16,
                  'ximin':0,
                  'ximax':11,
                  'etamin':-11,
                  'etamax':11,
                  'lod_min':6,
                  'lod_max':9,
                  'maxiter':5,
                  }
        clocklist = [0, 0, 90]
        phaselist = [90, 0, 0]
        pass
    elif mode == 'narrowfov':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'narrowfov')
        cfgpath = os.path.join(modelpath, 'narrowfov.yaml')
        _cfg = CoronagraphMode(cfgpath)

        kwargs = {'dact':46.3,
                  'xcenter':0,
                  'ycenter':-16,
                  'ximin':0,
                  'ximax':11,
                  'etamin':-11,
                  'etamax':11,
                  'lod_min':6,
                  'lod_max':9,
                  'maxiter':5,
                  }

        clocklist = [0, 0, 90]
        phaselist = [90, 0, 0]
        pass
    elif mode == 'nfov_flat':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'narrowfov')
        cfgpath = os.path.join(modelpath, 'narrowfov_flat.yaml')
        _cfg = CoronagraphMode(cfgpath)

        kwargs = {'dact':46.3,
                  'xcenter':0,
                  'ycenter':-16,
                  'ximin':0,
                  'ximax':11,
                  'etamin':-11,
                  'etamax':11,
                  'lod_min':6,
                  'lod_max':9,
                  'maxiter':5,
                  }

        clocklist = [0, 0, 90]
        phaselist = [90, 0, 0]
        pass
    elif mode == 'widefov':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'widefov')
        cfgpath = os.path.join(modelpath, 'widefov.yaml')
        _cfg = CoronagraphMode(cfgpath)

        kwargs = {'dact':46.3,
                  'xcenter':14,
                  'ycenter':8,
                  'ximin':0,
                  'ximax':21,
                  'etamin':-21,
                  'etamax':21,
                  'lod_min':6,
                  'lod_max':9,
                  'maxiter':5,
                  }

        clocklist = [0, 0, 90]
        phaselist = [90, 0, 0]
        pass
    elif mode == 'spectroscopy':
        modelpath = os.path.join(howfscpath, 'model', 'testdata',
                                 'spectroscopy')
        cfgpath = os.path.join(modelpath, 'spectroscopy.yaml')
        _cfg = CoronagraphMode(cfgpath)

        kwargs = {'dact':46.3,
                  'xcenter':0,
                  'ycenter':-15,
                  'ximin':0,
                  'ximax':13,
                  'etamin':-10,
                  'etamax':10,
                  'lod_min':6,
                  'lod_max':9,
                  'maxiter':5,
                  }

        clocklist = [0, 0, 90]
        phaselist = [90, 0, 0]
        pass
    else:
        raise ValueError('Invalid mode name "' + str(mode) + '"')


    _dmlist = _cfg.initmaps
    _ind = 0
    _iopen = np.abs(open_efield(_cfg, _dmlist, _ind))**2
    _ipeak = np.max(_iopen)

    # cos
    dpout0 = make_dmrel_probe(cfg=_cfg, dmlist=_dmlist, target=_target,
                              clock=clocklist[0], phase=phaselist[0],
                              ind=_ind, **kwargs)
    _eplus = efield(_cfg, [_dmlist[0]+dpout0, _dmlist[1]], _ind)
    _eminus = efield(_cfg, [_dmlist[0]-dpout0, _dmlist[1]], _ind)
    _pampe = np.abs((_eplus - _eminus)/2j)
    npint0 = _pampe**2/_ipeak
    if write:
        fnbase = mode + '_dmrel_' + '{:.1e}'.format(_target) + '_cos.fits'
        fn = os.path.join(modelpath, fnbase)
        pyfits.writeto(fn, dpout0, overwrite=True)
        pass

    # sin 1
    dpout1 = make_dmrel_probe(cfg=_cfg, dmlist=_dmlist, target=_target,
                              clock=clocklist[1], phase=phaselist[1],
                              ind=_ind, **kwargs)
    _eplus = efield(_cfg, [_dmlist[0]+dpout1, _dmlist[1]], _ind)
    _eminus = efield(_cfg, [_dmlist[0]-dpout1, _dmlist[1]], _ind)
    _pampe = np.abs((_eplus - _eminus)/2j)
    npint1 = _pampe**2/_ipeak
    if write:
        fnbase = mode + '_dmrel_' + '{:.1e}'.format(_target) + '_sinlr.fits'
        fn = os.path.join(modelpath, fnbase)
        pyfits.writeto(fn, dpout1, overwrite=True)
        pass

    # sin 2
    dpout2 = make_dmrel_probe(cfg=_cfg, dmlist=_dmlist, target=_target,
                              clock=clocklist[2], phase=phaselist[2],
                              ind=_ind, **kwargs)
    _eplus = efield(_cfg, [_dmlist[0]+dpout2, _dmlist[1]], _ind)
    _eminus = efield(_cfg, [_dmlist[0]-dpout2, _dmlist[1]], _ind)
    _pampe = np.abs((_eplus - _eminus)/2j)
    npint2 = _pampe**2/_ipeak
    if write:
        fnbase = mode + '_dmrel_' + '{:.1e}'.format(_target) + '_sinud.fits'
        fn = os.path.join(modelpath, fnbase)
        pyfits.writeto(fn, dpout2, overwrite=True)
        pass

    pass
