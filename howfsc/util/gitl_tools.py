# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Tools for manipulating data for the main GITL loop
"""

import warnings

import numpy as np

import howfsc.util.check as check

def validate_dict_keys(d, keys, custom_exception=TypeError):
    """
    Verify that a dictionary has exactly the set of input keys

    Will raise the exception in custom_exception if the set of keys in the
    dictionary is not an exact match (no extra, no missing).

    No returns.

    Arguments:
     d: input dictionary
     keys: set object, or object castable to set, containing keys

    Keyword Arguments:
     custom_exception: exception to raise in the case of key mismatch.
      Defaults to TypeError.

    """

    # Check inputs
    if not isinstance(d, dict):
        raise TypeError('d must be a dict')
    if not issubclass(custom_exception, Exception):
        raise TypeError('custom_exception must be descended from Exception')
    try:
        skeys = set(keys)
    except TypeError: # not castable to set
        raise TypeError('keys must be an object castable to a set') # reraise

    # Missing
    misskeys = skeys - set(d.keys())
    if misskeys != set():
        raise custom_exception('Missing keys in input config: ' +
                               str(misskeys))

    # Extra
    extrakeys = set(d.keys()) - skeys
    if extrakeys != set():
        raise custom_exception('Extra top-level keys in input file: + ' +
                               str(extrakeys))

    return


def param_order_to_list(nestlist):
    """
    Convert a list of lists, suitable for filling parameters, into a
    comprehensive list

    Each DM setting is applied with an unprobed image and a pair of probes, one
    positive and one negative.  Both positive and negative probes have a single
    FSW parameter between them for gain/exposure time/number of frames.  When
    collecting data, however, there is a telemetry value collected for each
    frame, which has a value each for positive and negative probes.  Example:

    param order: [[lam0 unprobed, lam0 probe0, lam0 probe1, lam0 probe2],
                  [lam1 unprobed, lam1 probe0, lam1 probe1, lam1 probe2],
                  [lam2 unprobed, lam2 probe0, lam2 probe1, lam2 probe2]]

    list from telem: [lam0 unprobed, lam0 probe0+, lam0 probe0-, lam0 probe1+,
                      lam0 probe1-, lam0 probe2+, lam0 probe2-, lam1 unprobed,
                      lam1 probe0+, lam1 probe0-, lam1 probe1+, lam1 probe1-,
                      lam1 probe2+, lam1 probe2-, lam2 unprobed, lam2 probe0+,
                      lam2 probe0-, lam2 probe1+, lam2 probe1-, lam2 probe2+,
                      lam2 probe2-]

    This function only reorders data; it does not alter type or content.  It
    assumes that the lists in param-order have the unprobed (i.e. not repeated)
    element as the first in the list.

    Arguments:
     nestlist: a list of lists following param-order.  Each sublist should be
      the same length and nonempty.

    Returns:
     a list in list-from-telem order. If len(nestlist) == A and
      len(nestlist[0]) == B, len(output) == A*(2*B-1)

    """
    # Check inputs
    try:
        A = len(nestlist)
    except TypeError: # not iterable
        raise TypeError('Outer structure of nestlist must be a list')

    try:
        B = len(nestlist[0])
    except TypeError: # not iterable
        raise TypeError('Inner structure of nestlist must be a list')
    if B == 0:
        raise TypeError('Inner lists must not be empty')

    for n in nestlist:
        try:
            tmpB = len(n)
        except TypeError: # not iterable
            raise TypeError('Inner structure of nestlist must be a list')
        if tmpB != B:
            raise TypeError('Inner lists of nestlist must all be the same ' +
                            'length')
        pass


    # Build list
    output = []
    for j in range(A):
        output.append(nestlist[j][0])
        for k in range(1, B):
            output.append(nestlist[j][k]) # +
            output.append(nestlist[j][k]) # -
        pass

    return output


def remove_subnormals(data):
    """
    Given an array, return an array of the same size that does not have any
    subnormal (aka denormalized) numbers.

    This is only necessary for floating-point numbers; integers cannot be
    subnormal.  This will also convert any negative zeros.

    CGI FSW cannot accept subnormal numbers--it will crash FSW--and so they
    need to be stripped out ahead of time by ground tools.  SSC may also do
    this, but better to plan to have it in GITL and prevent an accidental
    mistake if they do not.  (We're not sure if negative zeros would cause a
    crash, but it was easier to remove them than to find out.)

    This function will not touch/fix infinities or NaNs, only subnormals.

    Arguments:
     array: a 2D ndarray that may or may not have subnormal numbers

    Returns:
     an array of the same size, with no subnormal numbers.

    """
    # Check inputs
    check.twoD_array(data, 'data', TypeError)

    # Force subnormals (and negative 0) to 0
    tmp = data.copy()
    smallestnorm = np.finfo(tmp.dtype).tiny
    with warnings.catch_warnings(): # nans/infs cause RuntimeWarnings in > & <
        warnings.simplefilter('ignore')
        tmp[(tmp > -1*smallestnorm) & (tmp < smallestnorm)] = 0
        pass

    return tmp


def as_f32_normal(val):
    """
    Given a number, return a number that will not be subnormal (aka
    denormalize) when represented as a float32.

    This is only necessary for floating-point numbers; integers cannot be
    subnormal.  This will also convert any negative zeros.

    CGI FSW cannot accept subnormal numbers--it will crash FSW--and so they
    need to be stripped out ahead of time by ground tools.  SSC may also do
    this, but better to plan to have it in GITL and prevent an accidental
    mistake if they do not.  (We're not sure if negative zeros would cause a
    crash, but it was easier to remove them than to find out.)

    This function will not touch/fix infinities or NaNs, only subnormals.

    Arguments:
     val: a real scalar value that may or may not be subnormal

    Returns:
     a version of val that is no longer subnormal (as a floatt32) if it was

    """
    # Check inputs
    check.real_scalar(val, 'val', TypeError)

    if np.isinf(val) or np.isnan(val):
        return val

    # Force subnormals (and negative 0) to 0
    smallestnorm = np.finfo(np.float32).tiny
    if (val > -1*smallestnorm) and (val < smallestnorm):
        return 0
    return val
