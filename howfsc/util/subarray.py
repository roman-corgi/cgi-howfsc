# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Array manipulation for speeding up propagation routines
"""

import numpy as np

import howfsc.util.check as check

def centered_nonzero(arr):
    """
    Return the smallest centered slice of an array containing all non-zero
    points.

    This array will remain centered with respect to the original array, so no
    shifts or tilts need to be tracked.

    This function will retain even-odd sizing per axis to keep from having to
    maintain half-pixel shifts and introduce them as tilts later.  (If you're
    going to do that, you may as well do a tight fit and track the subregion
    center.)

    Arguments:
     arr: a 2D array to be truncated

    Returns:
     a sliced section of the original array (not a copy)

    """

    check.twoD_array(arr, 'arr', TypeError)

    sx, sy = arr.shape
    xind, yind = np.nonzero(arr)

    if len(xind) == 0: # xind and yind are same length; check only one
        return np.array([[]])

    cxind = _centerarray(sx, xind)
    cyind = _centerarray(sy, yind)
    return arr[cxind, cyind]


def _centerarray(s, ind):
    """
    Given an array size and an index list, turn it into a subregion slice

    Arguments:
     s: size of array along an axis
     ind: list of nonzero indices along that axis

    Returns:
     a slice object for the array subset

    """
    if s % 2 != 0:
        cind = ind - s//2 # center
        edge = max(abs(min(cind)), abs(max(cind))) # largest of neg and pos
        outind = slice(-edge + s//2, edge+1 + s//2)
        pass
    else:
        cind = ind - s//2
        edge = max(abs(min(cind)), abs(max(cind)+1))
        outind = slice(-edge + s//2, edge + s//2)
        pass
    return outind
