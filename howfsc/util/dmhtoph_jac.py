# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions to create a surface profile from an array of actuator heights.
"""
import copy
from types import SimpleNamespace

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.interpolate

from . import check
from .insertinto import insertinto
from .math import ceil_even, ceil_odd

def dmhtoph_jac(nrow, ncol, pokerad, row, col, nact, inf_func, ppact_d,
                ppact_cx, ppact_cy, dx, dy, thact, flipx):
    """
    Produce a surface map for a single actuator poke.

    Takes a single actuator poke, in radians, and produces a surface in
    radians, optimized for Jacobian calculations

    This is used for the surface that premultiplies the propagation in the
    Jacobian poke calculations.

    This function should produce identical outputs to dmhtoph(), for the case
    where dmhtoph gets a dmin of all zeros except dmin[row, col] = pokerad.

    Arguments:
     nrow: number of rows in output array. Integer > 0.
     ncol: number of cols in output array. Integer > 0.
     pokerad: height of the DM poke, in radians.  This will be a real scalar.
     row: row index of poke (integer >= 0, < nact)
     col: column index of poke (integer >= 0, < nact)
     nact: number of actuators across the DM.  Should be an positve integer
      and match both ``dmin.shape[0]`` and ``dmin.shape[1]``.  This is not
      being taken directly from dmin to catch potential invalid inputs (as
      this function will happily work with whichever DM setting it gets).
     inf_func: 2D square array with the centered representation on what the
      facesheet deformation looks like for a single poke of unit height.  The
      function must be smaller than the DM (nact*ppact_d).   The edge size must
      be odd-sized so that the central pixel of the array coincides with the
      array peak.
     ppact_d: design pixels per actuator in ``inf_func``.  Expected to be a
      positive scalar integer. ``inf_func`` will in general cover several
      actuators, as influence functions are not perfectly localized;
      ``ppact_d`` is required to denote the underlying grid.  Must be sampled
      better than the camera (ppact_cx and ppact_cy).
     ppact_cx: pixels per actuator in the x-direction on the camera.  Expected
      to be a real positive scalar.  ``ppact_d`` will be downsampled to this.
       Measured by DM registration.
     ppact_cy: pixels per actuator in the y-direction on the camera.  Expected
      to be a real positive scalar.  ``ppact_d`` will be downsampled to this.
       Measured by DM registration.
     dx: number of pixels to shift the DM grid off center in the x-direction
      on the camera. Expected to be a real scalar.  Measured by DM
      registration.
     dy: number of pixels to shift the DM grid off center in the y-direction
      on the camera. Expected to be a real scalar.  Measured by DM
      registration.
     thact: number of degrees to rotate the grid in a counterclockwise
      direction about the center of the array.  Expected to be a real scalar.
      Measured by DM registration.
     flipx: boolean whether to flip the output in the x-direction, leaving the
      y-direction unchanged.  This will only be used if required to
      accommodate DM electronics being wired with a parity flip relative to
      the camera.

    Returns:
      a square, odd-sized array large enough to hold the entire interpolated
       surface of the DM along with at least one outer ring of zeros.
       Odd-sized array will force the output to be pixel-centered. It will be
       matched to camera orientation and represent a DM surface in radians.

    """
    # Check inputs
    check.real_scalar(pokerad, 'pokerad', TypeError)
    check.nonnegative_scalar_integer(row, 'row', TypeError)
    check.nonnegative_scalar_integer(col, 'col', TypeError)
    check.positive_scalar_integer(nact, 'nact', TypeError)
    check.positive_scalar_integer(ppact_d, 'ppact_d', TypeError)
    check.twoD_array(inf_func, 'inf_func', TypeError)
    check.real_positive_scalar(ppact_cx, 'ppact_cx', TypeError)
    check.real_positive_scalar(ppact_cy, 'ppact_cy', TypeError)
    check.real_scalar(dx, 'dx', TypeError)
    check.real_scalar(dy, 'dy', TypeError)
    check.real_scalar(thact, 'thact', TypeError)
    # No check on flipx since every Python object can be used for truth tests

    if row >= nact:
        raise ValueError('row must be < nact')
    if col >= nact:
        raise ValueError('col must be < nact')
    if (np.array(inf_func.shape) >= nact*ppact_d).any():
        raise TypeError('Influence function must be smaller than DM')
    if (ppact_d < ppact_cx) or (ppact_d < ppact_cy):
        raise TypeError('Design influence function must be sampled '
                        'better than camera')
    if inf_func.shape[0] != inf_func.shape[1]:
        raise TypeError('inf_func must be square')
    if inf_func.shape[0] % 2 != 1:
        raise TypeError('inf_func must be odd-sized')

    N = inf_func.shape[0]

    # doing balanced, not FFT convention, for this side of interpolation so
    # the peaks are exactly on integers where possible
    # also only do the exact area of the jac poke
    npix = (nact - 1)*ppact_d + N
    xyrow = (np.arange(N) + row*ppact_d - (npix-1)/2)/ppact_d
    xycol = (np.arange(N) + col*ppact_d - (npix-1)/2)/ppact_d
    interpolator = scipy.interpolate.RectBivariateSpline(
        xyrow, xycol, inf_func,
        bbox=[min(xyrow), max(xyrow), min(xycol), max(xycol)]
    )

    # Make output array for interpolation onto
    mppa = max(ppact_cx, ppact_cy)
    maxshift = max(np.abs(dx), np.abs(dy))
    # > sqrt(2) to cover 45deg rot
    nxyres = ceil_odd(np.ceil(np.sqrt(2)*(nact + N/ppact_d)*mppa) + 2*maxshift)

    # Use FFT convention for outputs as the rest of the repo is using this
    # convention. Fine since we're picking the output points.
    xout = (np.arange(nxyres) - nxyres//2)/ppact_cx
    yout = (np.arange(nxyres) - nxyres//2)/ppact_cy
    X, Y = np.meshgrid(xout, yout)

    # Do interpolation over a subarea as RectBivariateSpline extrapolates
    # edge points, which is not desired behavior
    interp_inds = np.logical_and(
        np.logical_and(X >= min(xycol), X <= max(xycol)),
        np.logical_and(Y >= min(xyrow), Y <= max(xyrow)))

    # Expects rows then columns, which in our convention is y then x
    sind = interpolator(Y[interp_inds], X[interp_inds], grid=False)
    s0 = np.zeros((nxyres, nxyres))
    s0[interp_inds] = sind

    # parity, rotation, translation
    if flipx:
        s0 = np.fliplr(s0)

    dmrot = scipy.ndimage.rotate(s0, -thact, reshape=False)
    surface = scipy.ndimage.shift(dmrot, [dy, dx])

    return pokerad * insertinto(surface, (nrow, ncol))


def dmhtoph_cropped_poke(pokerad, row, col, dm):
    """
    Generate the offcenter-cropped DM surface for a single specified actuator.

    For use in rapid Jacobian calculation.

    Requires the input dm object to be precomputed.

    Arguments:
     pokerad: floating-point height of the DM poke, in radians.
     row: row index of poke (integer >= 0, < nact).
     col: column index of poke (integer >= 0, < nact).
     dm: SimpleNamespace object containing influence function data.

    Returns:
     surfCrop : array_like
      An offcenter-cropped subarray of a full 2-D surface. This subarray
      should fully contain the influence function of one DM actuator.
     yxLowerLeft : tuple
      Tuple containing the (y, x) array coordinates at which to insert the
      surf_crop sub-array into a larger array of shape dmArrayShape.
     nSurf : int
      surfCrop goes back into a 2-D array of shape (nSurf, nSurf) at the
      position given by yxLowerLeft.

    """
    # Input checks
    check.real_scalar(pokerad, 'pokerad', TypeError)
    check.nonnegative_scalar_integer(row, 'row', TypeError)
    check.nonnegative_scalar_integer(col, 'col', TypeError)
    if not isinstance(dm, SimpleNamespace):
        raise TypeError('dmobj must be an instance of types.SimpleNamespace()')
    if row >= dm.nact:
        raise ValueError('row must be < nact')
    if col >= dm.nact:
        raise ValueError('col must be < nact')

    # Convert 2-D indices to 1-D for nact x nact array
    index1d = row*dm.nact + col

    deltax = dm.xOffsets[index1d]
    deltay = dm.yOffsets[index1d]
    surfCrop = pokerad * scipy.ndimage.shift(dm.infMaster, [deltay, deltax])

    yxLowerLeft = (dm.yLowerLeft[index1d], dm.xLowerLeft[index1d])

    return surfCrop, yxLowerLeft, dm.nSurf

def compute_master_inf_func(dmobj, reg_dict):
    """
    Pre-compute the resized, rotated influence function for the fast Jacobian
    calculation.

    Uses griddata instead of RectBivariateSpline because RectBivariateSpline
    was giving streak artifacts outside the influence function when it was
    laterally stretched. Since griddata was already being used, the z-rotation
    is also done with griddata instead of scipy.ndimage.rotate.

    Translation occurs in a separate function.

    Returns nothing. Modifies dmobj in place.

    Arguments:
     dmobj: an instance of types.Simplenamespace into which the
      outputs will be stored.
     reg_dict: dictionary containing the needed DM registration data

    Returns:
     None

    """
    # Input type checking
    if not isinstance(dmobj, SimpleNamespace):
        raise TypeError('dmobj must be an instance of types.SimpleNamespace()')
    if not isinstance(reg_dict, dict):
        raise TypeError('reg_dict must be of type dict')

    # Unpack the DM reg_dict dictionary
    ppact_cx = reg_dict['ppact_cx']
    ppact_cy = reg_dict['ppact_cy']
    ppact_d = reg_dict['ppact_d']
    dx = reg_dict['dx']
    dy = reg_dict['dy']
    inf_func = reg_dict['inf_func']
    nact = reg_dict['nact']
    thact = reg_dict['thact']
    flipx = reg_dict['flipx']

    thactRad = thact*np.pi/180

    # Compute minimum array width to contain full DM surface.
    # Forced to be odd.
    nInf0 = inf_func.shape[0]
    maxppa = max(ppact_cx, ppact_cy)
    maxshift = max(np.abs(dx), np.abs(dy))
    # min width factor of square needed to contain rotated inf func
    wsquare = np.abs(np.cos(thactRad)) + np.abs(np.sin(thactRad))
    nSurf = ceil_odd(wsquare*(nact + nInf0/ppact_d)*maxppa +
                     2*maxshift)

    pupil_dim = np.max(dmobj.pupil_shape)

    # set to minimum array width if necessary
    if nSurf < pupil_dim:
        nSurf = ceil_odd(pupil_dim)

    # Scale the array size of the master influence function
    n0 = inf_func.shape[0]
    x0 = (np.arange(n0) - (n0-1)/2)/ppact_d  # array centered coords
    n1 = ceil_odd(max(n0*ppact_cx/ppact_d, n0*ppact_cy/ppact_d))

    # Pad before rotating to contain the rotated inf func.
    # Pad to an odd-sized array so that the center of the influence
    # function stays on the center pixel.
    n1 *= wsquare
    n1 = ceil_odd(n1)

    # Fliplr if needed
    if flipx:
        values = inf_func.flatten()
    else:
        values = np.fliplr(inf_func).flatten()

    # Compute starting grid coordinates of influence function
    N = inf_func.shape[0]
    check.twoD_square_array(inf_func, 'inf_func', TypeError)
    if N % 2 != 1:
        raise ValueError('inf_func must have odd array dimensions')
    x0 = np.arange(-(N-1)/2, (N+1)/2)
    y0 = x0
    X0, Y0 = np.meshgrid(x0, y0)
    pointsIn = np.array([X0.flatten(), Y0.flatten()]).T

    # z-rotate output grid coordinates
    pointsOut = np.array([X0.flatten(), Y0.flatten()])
    if thact != 0:
        thactRad = np.radians(thact)
        rotMat = np.array([[np.cos(thactRad), np.sin(thactRad)],
                           [-np.sin(thactRad), np.cos(thactRad)]])
        for ii in range(N*N):
            pointOut = rotMat @ pointsOut[:, ii].reshape((2, 1))
            pointsOut[:, ii] = pointOut.flatten()

    # Laterally scale output grid coordinates
    xFac = (ppact_d/ppact_cx)
    yFac = (ppact_d/ppact_cy)
    pointsOut[0, :] *= xFac
    pointsOut[1, :] *= yFac

    pointsOut = pointsOut.T
    grid_z2 = scipy.interpolate.griddata(pointsIn, values, pointsOut,
                                         method='cubic', fill_value=0,
                                         rescale=False)
    infOut = grid_z2.reshape((N, N))
    infMaster = insertinto(infOut, (n1, n1))

    # Now pad to an even-sized array (only do after rotating).
    n1 = ceil_even(n1)
    infMaster = insertinto(infMaster, (n1, n1))

    # Compute the actuator center coordinates in units of pixels.

    # Scale
    xc1d = (np.arange(nact) - (nact-1)/2)*ppact_cx
    yc1d = (np.arange(nact) - (nact-1)/2)*ppact_cy
    XC, YC = np.meshgrid(xc1d, yc1d)
    xCenters = XC.flatten()
    yCenters = YC.flatten()

    # flip
    if flipx:
        xCenters *= -1

    # Rotate
    rotMat = np.array([[np.cos(thactRad), -np.sin(thactRad)],
                       [np.sin(thactRad), np.cos(thactRad)]])
    for ii, _ in enumerate(xCenters):
        xyRot = rotMat @ np.array([[xCenters[ii]], [yCenters[ii]]])
        xCenters[ii] = xyRot[0, 0]
        yCenters[ii] = xyRot[1, 0]

    # Translate after rotation to be consistent with dmhtoph()
    xCenters += dx
    yCenters += dy

    # Subarray details

    # Offsets of the actuator centers from the nearest pixel:
    xOffsets = xCenters - np.round(xCenters)
    yOffsets = yCenters - np.round(yCenters)

    # Compute lower-left pixel location in the full-sized array
    # where the cropped inf func gets inserted.
    xLowerLeft = np.round(xCenters).astype(int) + nSurf//2 - n1//2
    yLowerLeft = np.round(yCenters).astype(int) + nSurf//2 - n1//2

    # Store output data into an object
    dmobj.xLowerLeft = copy.copy(xLowerLeft)
    dmobj.yLowerLeft = copy.copy(yLowerLeft)
    dmobj.xOffsets = copy.copy(xOffsets)
    dmobj.yOffsets = copy.copy(yOffsets)
    dmobj.infMaster = copy.copy(infMaster)
    dmobj.nSurf = copy.copy(nSurf)
    dmobj.nact = copy.copy(nact)

    return None
