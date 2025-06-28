# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Compute SVD spectrum for user analysis

Based on implementations by Garreth Ruane and AJ Eldorado Riggs
"""

import numpy as np

from howfsc.model.mode import CoronagraphMode
from howfsc.util.insertinto import insertinto
import howfsc.util.check as check

def calc_svd_spectrum(jac, cfg, e0list):
    """
    Given a Jacobian and an electric field, compute how much power remains in
    each singular-value mode

    No per-pixel or per-dm weighting is included in this calculation.

    Arguments:
     jac: 3D real-valued DM Jacobian matrix, as produced by ``calcjacs()``.
      Size is 2 x Ndm x Npix.  jac is complex; real parts are stored in
      [0,:,:] and the imaginary parts in [1,:,:]
     cfg: CoronagraphMode object.  Should be the one used to calculate jac.
     e0list: list of 2D arrays of electric fields, one for each channel in the
      coronagraph configuration.  These will be complex-valued.   All must be
      the same shape.   Any bad pixels should be marked in each e0 as NaN.

    Returns:
     tuple with two values:
      - first is a list of singular values squared, normalized by the maximum
        singular value squared (so the largest value will always be 1).  These
        are ordered from largest to smallest. ('snorm')
      - second is a list of power per mode.  Order is the same as the first
        (goes from largest singular value to smallest).  ('iri')
     Spectrum may be plotted by plotting snorm on x-axis and iri on y-axis.

    """

    # Check inputs
    # cfg
    check.threeD_array(jac, 'jac', TypeError)
    check.real_array(jac, 'jac', TypeError)

    # jac
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')
    if jac.shape[0] != 2:
        raise TypeError('jac index 0 must have size 2')
    npix = np.sum([np.sum(sl.dh.e) for sl in cfg.sl_list])
    if jac.shape[2] != npix:
        raise TypeError('jac and cfg have inconsistent number of dark-hole ' +
                        'pixels: jac = ' + str(jac.shape[2]) + ', cfg = ' +
                        str(npix))
    nlam = len(cfg.sl_list)

    # e0list
    try:
        lene0list = len(e0list)
    except TypeError: # not iterable
        raise TypeError('e0list must be an iterable') # reraise
    if lene0list != nlam:
        raise TypeError('e0list must have one element for each wavelength')
    for index, e0 in enumerate(e0list):
        check.twoD_array(e0, f'e0list[{index}]', TypeError)
        pass
    nrow, ncol = e0list[0].shape
    for e0 in e0list:
        if e0.shape != (nrow, ncol):
            raise TypeError('All e0list elements must be the same shape')
        pass

    # Make 1D arrays mapped to jac
    dhlist = []
    for j in range(nlam):
        dh = insertinto(cfg.sl_list[j].dh.e, (nrow, ncol)).astype('bool')
        dhlist.append(dh)
        pass
    ndhpix = np.cumsum([0]+[np.sum(dh) for dh in dhlist])

    emeas = np.zeros((ndhpix[-1],)).astype('complex128')
    bpmeas = np.zeros((ndhpix[-1],)).astype('bool')
    for j in range(nlam):
        efield = e0list[j]
        badefield = np.isnan(efield)
        emeas[ndhpix[j]:ndhpix[j+1]] = efield[dhlist[j]]
        bpmeas[ndhpix[j]:ndhpix[j+1]] = badefield[dhlist[j]]

    # Handle the no-good case
    if bpmeas.all():
        raise ValueError('All pixels in dark hole regions are bad')

    # Rearrange into (2*Npix) x Ndm (or 1)
    rjac = np.zeros((jac.shape[0]*jac.shape[2], jac.shape[1]))
    for k in range(jac.shape[0]):
        rjac[k*jac.shape[2]:(k+1)*jac.shape[2], :] = jac[k, :, :].real.T
        pass
    pass
    re0 = np.vstack((np.real(emeas)[:, np.newaxis],
                     np.imag(emeas)[:, np.newaxis]))
    rbp = np.concatenate((bpmeas, bpmeas))

    u, s, _ = np.linalg.svd(rjac[~rbp, :], full_matrices=False)

    # transpose is fine because jac was real-valued and U will be as well.
    eri = u.T @ re0[~rbp]
    iri = np.abs(eri)**2

    alpha = np.max(s)**2 # use max sing val squared to match CGI normalization
    snorm = s**2/alpha

    return snorm, iri
