# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions to prepare HOWFSC GITL frames for input to wavefront sensing
"""

import numpy as np

from howfsc.util import check

def extract_bp(im):
    """
    Extract a per-frame map of bad pixels from a HOWFSC GITL frame

    Bad pixels are stored as NaNs.  This function will not change the input
    array (no reason not to keep it as NaNs)

    Arguments:
     im: 2D array with some number of NaNs.

    Returns:
     boolean bad pixel map of the same size as im, with bad pixels as True to
     match the proc_cgi_frame convention

    """

    check.twoD_array(im, 'im', TypeError)

    return np.isnan(im)


def normalize(im, peakflux, exptime):
    """
    Normalize a HOWFSC GITL frame

    Used on cleaned data to convert to normalized intensity.  After this, the
    images should be able to be compared to model images, which will have the
    same normalization.

    The scale factor will be implemented as a division by peakflux and exptime.
    The intention is that we will measure the flux (counts/second) at the
    unocculted PSF peak in a separate photometric calibration, and then we
    convert this image into counts/sec by dividing by exptime, and then into
    a unitless normalization that = 1 at an unocculted PSF peak ("normalized
    intensity").

    Arguments:
     im: 2D array to scale
     peakflux: Counts per second at the peak of an unocculted PSF with the
      same CFAM filters as were used to collect the data in im.  Should be
      a real scalar > 0.
     exptime: Exposure time used when collecting the data in in.  Should be a
      real scalar > 0.

    Returns
     normalized intensity array of the same size as im

    """

    check.twoD_array(im, 'im', TypeError)
    check.real_positive_scalar(peakflux, 'peakflux', TypeError)
    check.real_positive_scalar(exptime, 'exptime', TypeError)

    return im/exptime/peakflux


def eval_c(nimlist, dhlist, n2clist):
    """
    Evaluate mean total contrast over control pixels which are not masked by
    bad pixel map

    This will use a fixed, precomputed array per filter to convert from
    normalized intensity (measured data scaled to have an unocculted peak = 1)
    to contrast (which includes the effect of mask edges in throughput for an
    off-axis source) by elementwise multiply.  Normalized intensity is
    otherwise used for wavefront sensing and wavefront control.

    These arrays, stored in n2clist, should in general be >= 1 at almost every
    pixel as they are used as multiplicative factors.  (They cannot be forced
    to be >= 1, as there are unusual but valid cases where very small regions
    might be < 1, so it falls to the user to make sure they are not using
    1/n2c arrays by mistake.)

    This works on a list of arrays, because the mean is evaluated across all
    wavelengths.  This does mean that even if you want to evaluate at a single
    wavelength, you still need to insert your data into one-element lists.

    Arguments:
     nimlist: list of 2D arrays of normalized intensities
     dhlist: list of 2D boolean arrays indicating which pixels are to be
      evaluated as part of wavefront control.  This list will be the same
      length as nimlist, and each element of the list should be an array of the
      same shape as the corresponding element in nimlist.  An array element
      will be True if that pixel is to be included in control and used as part
      of the mean.
     n2clist: list of 2D floating-point arrays giving the scale factor to
      convert from normalized intensity to contrast as a multiplier.  These
      correspond to the relative drop-off in flux from unity due to the
      presence of nearby coronagraphic masks.

      As an example, if the measured normalized intensity is 5e-9, but the
      presence of a mask edge is causing the flux to drop to 0.5x of usual,
      then the value in an array of n2clist will be 2, and the actual contrast
      will be 1e-8.

      This list will be the same length as nimlist, and each element of the
      list should be an array of the same shape as the corresponding element in
      nimlist.


    Returns:
     a single scalar contrast value

    """

    # Check inputs
    if not isinstance(nimlist, list):
        raise TypeError('nimlist must be a list')
    if not isinstance(dhlist, list):
        raise TypeError('dhlist must be a list')
    if not isinstance(n2clist, list):
        raise TypeError('n2clist must be a list')

    if len(nimlist) != len(dhlist):
        raise TypeError('nimlist and dhlist must be the same length')
    if len(nimlist) != len(n2clist):
        raise TypeError('nimlist and n2clist must be the same length')

    for index, nim in enumerate(nimlist):
        check.twoD_array(nim, 'nimlist[' + str(index) + ']', TypeError)
        pass

    for index, dh in enumerate(dhlist):
        check.twoD_array(dh, 'dhlist[' + str(index) + ']', TypeError)
        if dh.shape != nimlist[index].shape:
            raise TypeError('nimlist and dhlist elements must be the same ' +
                            'shape at corresponding indices. Problem at ' +
                            'index ' + str(index))
        if dh.dtype != bool:
            raise TypeError('dhlist arrays must be boolean')
        pass

    for index, n2c in enumerate(n2clist):
        check.twoD_array(n2c, 'n2clist[' + str(index) + ']', TypeError)
        if n2c.shape != nimlist[index].shape:
            raise TypeError('nimlist and n2clist elements must be the same ' +
                            'shape at corresponding indices. Problem at ' +
                            'index ' + str(index))
        pass


    counts = 0
    pixels = 0

    for nim, dh, n2c in zip(nimlist, dhlist, n2clist):
        bp = extract_bp(nim)
        control = np.logical_and(dh, ~bp)
        counts += np.sum(nim[control]*n2c[control])
        pixels += np.sum(control)
        pass

    if pixels == 0:
        raise ZeroDivisionError('No valid pixels in any frame')

    return counts/pixels
