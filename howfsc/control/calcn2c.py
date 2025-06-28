# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Tool to build n2clist objects from a model
"""

import os
import logging
from joblib import Parallel, delayed, cpu_count
import numpy as np

from howfsc.model.mode import CoronagraphMode
from howfsc.util.insertinto import insertinto
import howfsc.util.check as check

def calcn2c(cfg, idx, nrow, ncol, dmset_list):
    """
    Build an elementwise matrix to convert normalized intensity to contrast.

    Used to precompute n2clist elements required for howfsc_computation inputs.
    Grid spacing is one pixel, and arrays are centered as per the FFT
    convention (0 at center of odd-sized arrays, 0 just right of center for
    even-sized arrays) e.g. [-1, 0, 1] or [-2, -1, 0, 1] for lengths 3 or 4.

    To simplify computation, every pixel which was going to be excluded from
    dark hole calculations is set to NaN.  Every non-NaN pixel should generally
    be >= 1, where 1 corresponds to no PSF attenuation from mask edges, and
    >= 1 is a multiplier: if half the PSF is cut by a mask when centered at a
    pixel, you would expect the n2c value at that pixel to be 2.

    Note that it is possible that diffraction in specific optical
    configurations may cause the off-axis peak to be slightly enhanced, up to
    a couple percent.  But you should not be seeing values like 0.5 in an n2c
    matrix.

    Arguments:
     cfg: a CoronagraphMode object (i.e. optical model)
     idx: index indicating which wavelength channel to use.  Must be >= 0 and
      < len(cfg.sl_list).
     nrow: integer > 0 giving number of row pixels to calculate
     ncol: integer > 0 giving number of column pixels to calculate
     dmset_list: a list of DM settings, in voltage, currently applied to
      the DMs.  The DM sizes and ordering must match cfg.dmlist.

    Returns:
     2D nrow x ncol ndarray with NaNs for ignored pixels and rest generally
     >= 1

    """

    # Validate inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')
    check.scalar_integer(idx, 'idx', TypeError)
    if idx < 0 or idx >= len(cfg.sl_list):
        raise TypeError('idx must be mappable to index on cfg.sl_list')
    check.positive_scalar_integer(nrow, 'nrow', TypeError)
    check.positive_scalar_integer(ncol, 'ncol', TypeError)

    # Use cfg checker, going into cfg anyway
    cfg.sl_list[0]._check_dmset_list(dmset_list)

    # Wrangle model internals--the model code is not designed to be altered
    # after creation, so we'll step through ttgrid and then set back to the
    # original at the end.
    sl = cfg.sl_list[idx]
    orig_ttgrid = sl.epup.ttgrid.copy()
    dh = insertinto(sl.dh.e, (nrow, ncol))

    lrow = -(nrow//2) # centers a length nrow/col array as per FFT conv.
    lcol = -(ncol//2)

    # define worker function calculates each column
    def col_worker(j):
        flux_colj = np.zeros((nrow,))
        for k in range(nrow):
            # For efficiency, only populate pixels that are visible through the
            # dark hole mask anyway
            if dh[k, j] == 0:
                flux_colj[k] = np.nan
                continue

            # tip and tilt are in pixels, fortunately
            tip = lcol + j # tip moves cols ("x")
            tilt = lrow + k # tilt moves rows ("y")

            jk_ttgrid = sl.epup.get_ttgrid(tip, tilt)
            sl.ttph_up = np.exp(1j*2*np.pi
                                /(sl.epup.pixperpupil*sl.fs.pixperlod)
                                *jk_ttgrid)

            # propagate as usual
            edm0 = sl.eprop(dmset_list)
            ely = sl.proptolyot(edm0)
            edh0 = sl.proptodh(ely)
            flux_colj[k] = np.max(np.abs(edh0)**2)

        return flux_colj

    # calculate the columns with parallel processes, uses joblib tools,
    # see https://joblib.readthedocs.io/en/latest/parallel.html
    # set mkl threads to 1, for some reason, mkl.set_num_threads() doesn't
    # seem to work, but the environment variable does
    # it's possible the MKL_NUM_TRHEADS env var is already set, check first
    if not 'MKL_NUM_THREADS' in os.environ:
        do_reset_num_threads = True
        os.environ['MKL_NUM_THREADS'] = '1'
    else:
        do_reset_num_threads = False

    logging.info('MKL_NUM_THREADS = %s threads', os.environ['MKL_NUM_THREADS'])

    flux_list = Parallel(n_jobs=(max(cpu_count()//2-2, 1)), max_nbytes=None)(
        delayed(col_worker)(j) for j in range(ncol)
    )
    flux = np.array(flux_list).T

    if do_reset_num_threads:
        del os.environ['MKL_NUM_THREADS']

    # Get unocculted max
    edm0 = sl.eprop(dmset_list)
    ely = sl.proptolyot_nofpm(edm0)
    edh0 = sl.proptodh(ely)
    unocc_peak = np.max(np.abs(edh0)**2)

    n2c = 1/(flux/unocc_peak)

    # reset the object back to normal
    sl.epup.ttgrid = orig_ttgrid
    sl.ttph_up = np.exp(1j*2*np.pi
                        /(sl.epup.pixperpupil*sl.fs.pixperlod)
                        *sl.epup.ttgrid)

    return n2c
