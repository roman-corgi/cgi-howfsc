# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Utility functions for computing HOWFSC metrics, for monitoring and performance
evaluation
"""

import warnings

import numpy as np

import howfsc.util.check as check

def de_metrics(meas_old, model_old, meas_new, model_new):
    """
    Compute performance metrics between two electric field estimates from
    different iterations

    Tool to satisfy the following requirement:
    1133643 - Given both the measurement and the model electric field
    estimates in a CFAM filter over the dark hole from two different
    iterations, the CTC GSW shall compute the complex correlation, ΔE ratio,
    NI improvement ratio, and expected NI improvement ratio for the data for
    that filter.

    All arrays must be the same shape.

    Bad pixels are expected to be set to 0 (instead of NaN, which are used at
    other points in the HOWFSC processing pipeline).  Users may also zero
    pixels for other reasons (e.g. excluded from region of interest, blocked by
    mask); any point which is exactly 0 in both measured and model will be
    excluded from calculations.  Being 0 in only one of the two is *not*
    sufficient to be flagged for removal.

    It is not invalid to reuse the same electric field for old and new inputs,
    but certain metrics may be be available if so:
      - if models are the same, complex correlation cannot be calculated
      - the data is the same, complex correlation and delta-E ratio cannot be
        calculated

    Arguments:
     meas_old: 2D complex-valued ndarray with measured electric-field in each
      pixel.  This is the first of two iterations for comparison.  NOTE: if
      this input is all zeros, we won't be able to calculate NI improvement
      ratio or expected NI improvement ratio due to divide-by-zero issues.
     model_old: 2D complex-valued ndarray with model electric-field in each
      pixel.  This is the first of two iterations for comparison.
     meas_new: 2D complex-valued ndarray with measured electric-field in each
      pixel.  This is the second of two iterations for comparison.
     model_new: 2D complex-valued ndarray with model electric-field in each
      pixel.  This is the second of two iterations for comparison.

    Returns:
     dictionary with four keys:
      - 'CC' (complex correlation) => complex-valued scalar, ideally 1 + 0j
      - 'dE_rat' (delta-E ratio) => real-valued scalar, ideally 1
      - 'NI_rat' (NI improvement ratio) => real-valued scalar, 1.0 = perfect
        correction and 0 = no improvement
      - 'exp_NI_rat' (expected NI improvement ratio)  => real-valued scalar,
        1.0 = perfect correction and 0 = no improvement
     These values may also be None if the input data set is such that they
     cannot be calculated without a divide-by-zero error.  A warning will be
     issued in these cases.

    """

    # validate inputs
    check.twoD_array(meas_old, 'meas_old', TypeError)
    check.twoD_array(model_old, 'model_old', TypeError)
    check.twoD_array(meas_new, 'meas_new', TypeError)
    check.twoD_array(model_new, 'model_new', TypeError)

    shapeset = set()
    for m in [meas_old, model_old, meas_new, model_new]:
        shapeset.add(m.shape)
        pass
    if len(shapeset) != 1:
        raise TypeError('All arrays must be the same size')

    # If meas_old or meas_new is a nan, that's a bad pixel.
    # Should be no NaNs in models...but check them anyway, just in case.
    bp_old = np.logical_or(np.isnan(meas_old), np.isnan(model_old))
    bp_new = np.logical_or(np.isnan(meas_new), np.isnan(model_new))
    bp = np.logical_or(bp_old, bp_new)

    if bp_old.all():
        raise TypeError('All pixels in old iteration must not be bad')
    if bp_new.all():
        raise TypeError('All pixels in new iteration must not be bad')
    if bp.all():
        raise TypeError('At least one pixel must be good in both old and new')

    # _d = data, _m = model
    dE_d = meas_new - meas_old
    dE_m = model_new - model_old

    # Set flags to handle the case where someone put the same data in twice, or
    # things didn't change (e.g. two data collections at the same DM position
    # will have same model e-field, though maybe not same measured)
    #
    # This is primarily here to avert spurious divide-by-zero errors.
    datasame = True if (np.abs(dE_d) < np.finfo(float).eps).all() else False
    modelsame = True if (np.abs(dE_m) < np.finfo(float).eps).all() else False

    # Complex correlation
    if not datasame and not modelsame:
        CC = np.mean(dE_d[~bp]*dE_m[~bp].conj())
        CC /= np.sqrt(np.mean(np.abs(dE_m[~bp])**2))
        CC /= np.sqrt(np.mean(np.abs(dE_d[~bp])**2))
        pass
    else:
        CC = None
        if datasame and not modelsame:
            warnings.warn('Complex correlation not defined for identical ' +
                          'data between two iterations')
        elif modelsame and not datasame:
            warnings.warn('Complex correlation not defined for identical ' +
                          'model between two iterations')
        else:
            warnings.warn('Complex correlation not defined for identical ' +
                          'data and model between two iterations')
        pass


    # dE ratio
    if not datasame:
        dE_rat = np.sqrt(np.mean(np.abs(dE_m[~bp])**2))
        dE_rat /= np.sqrt(np.mean(np.abs(dE_d[~bp])**2))
        pass
    else:
        dE_rat = None
        warnings.warn('delta-E ratio not defined for identical data between ' +
                      'iterations')
        pass

    # NI improvement ratio and expected NI improvement ratio
    oldNI = np.mean(np.abs(meas_old[~bp])**2)
    if oldNI == 0:
        NI_rat = None
        warnings.warn('NI improvement ratio not defined when starting ' +
                      'intensity is zero')
        exp_NI_rat = None
        warnings.warn('expected NI improvement ratio not defined when ' +
                      'starting intensity is zero')
        pass
    else:
        NI_rat = 1 - np.mean(np.abs(meas_new[~bp])**2)/oldNI
        exp_NI_rat = 1 - np.mean(np.abs(meas_old[~bp] + dE_m[~bp])**2)/oldNI
        pass

    return {
        'CC':CC,
        'dE_rat':dE_rat,
        'NI_rat':NI_rat,
        'exp_NI_rat':exp_NI_rat,
    }
