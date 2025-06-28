# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Perform one or more wavefront control steps and evaluate performance
"""

import numpy as np

import howfsc.util.check as check

from howfsc.control.cs import ControlStrategy, get_we0
from howfsc.model.mode import CoronagraphMode

class JTWJMap(object):
    """
    Object to hold all of the JTWJ matrices required to implement a control
    strategy.  More than one will be required if the control strategy uses more
    than one file in pixelweights.

    Arguments:
     cfg: a CoronagraphMode object (i.e. optical model)

     jac: 3D real-valued DM Jacobian array, as produced by calcjacs().
      Shape is 2 x ndm x npix.

     cstrat: a ControlStrategy object; this will be used to define the behavior
      of the wavefront control by setting the regularization, per-pixel
      weighting, multiplicative gain, and next-iteration probe height.  It will
      also contain information about fixed bad pixels.

     subcroplist: list of 4-tuples of (lower row, lower col,
      number of rows, number of columns), indicating where in a clean frame
      each PSF is taken.  All are integers; the first two must be >= 0 and the
      second two must be > 0.  This should have nlam elements.  This is a
      subset of the full croplist, which has ndm*nlam elements, but all values
      across each wavelength are the same.

      Note: in howfsc_computation, croplist is populated from telemetry.  When
      using this function, need to make sure these are the same.

    """
    def __init__(self, cfg, jac, cstrat, subcroplist):
        """
        Storage:
         - for each region in a cstrat, if the pw filename is new (not in
           storage dict)
          = get we0 using cfg, cstrat, subcroplist, iteration and contrast.
            Last two should be pulled from region 'first'/'low', since 'high'
            is  exclusive.  ('last' could have been used as well.)
          = compute jtwj using jac + we0
          = store in dict with fn as key and jtwj as value
         - if filename is not new (key in dict), skip; jtwj is the same

        """
        # Validate inputs
        if not isinstance(cfg, CoronagraphMode):
            raise TypeError('cfg must be a CoronagraphMode object')
        check.threeD_array(jac, 'jac', TypeError)
        if not isinstance(cstrat, ControlStrategy):
            raise TypeError('cstrat must be a ControlStrategy object')

        if not isinstance(subcroplist, list):
            raise TypeError('subcroplist must be a list')
        if len(subcroplist) != len(cfg.sl_list):
            raise TypeError('Number of crop regions does not match model')

        for index, crop in enumerate(subcroplist):
            if not isinstance(crop, tuple):
                raise TypeError('subcroplist[' + str(index) +
                                '] must be a tuple')
            if len(crop) != 4:
                raise TypeError('Each element of subcroplist must be a ' +
                                '4-tuple')
            check.nonnegative_scalar_integer(crop[0], 'subcroplist[' +
                                             str(index) + '][0]', TypeError)
            check.nonnegative_scalar_integer(crop[1], 'subcroplist[' +
                                             str(index) + '][1]', TypeError)
            check.positive_scalar_integer(crop[2], 'subcroplist[' +
                                          str(index) + '][2]', TypeError)
            check.positive_scalar_integer(crop[3], 'subcroplist[' +
                                          str(index) + '][3]', TypeError)
            pass

        # subcroplist is special--it connects directly to the physical detector
        # and is used, with fixedbp, to indicate which pixels in the HOWFSC
        # subregion are bad.  Everything else in the HOWFSC computation is
        # independent of the actual detector location.  Unfortunately, that
        # means we have to store the subcroplist we used here, so we can
        # compare it to the subcroplist data we get down in telemetry, and
        # flag if there is a mismatch.
        self.subcroplist = subcroplist
        # Don't save the others, though

        # Fill the internal dictionary using the input data
        self.jtwjs = dict()
        for r in cstrat.pixelweights_fn:
            if r.value not in self.jtwjs.keys():
                # Use first and low as they are inclusive; last would work, but
                # high is exclusive.
                we0 = get_we0(cfg, cstrat, subcroplist, r.first, r.low)
                jtwj = get_jtwj(jac, we0)
                self.jtwjs.update({r.value: jtwj})
                pass
            else:
                # if multiple regions are pointing at the same weighting file,
                # reuse that dictionary entry
                pass
            pass

        pass


    def retrieve_jtwj(self, cstrat, iteration, contrast):
        """
        Get the jtwj associated with a particular iteration and contrast

        Arguments:
         cstrat: a ControlStrategy object; this will be used to define the
          behavior of the wavefront control by setting the regularization,
          per-pixel weighting, multiplicative gain, and next-iteration probe
          height.  It will also contain information about fixed bad pixels.
         iteration: integer >= 1, giving iteration number for the iteration
          that is about to happen.  (Starting iteration, which used data
          preloaded on board, is iteration 0.)
         contrast: floating-point value >= 0.  Mean total contrast across all
          dark hole pixels in the iteration that just completed (i.e.
          'iteration' - 1).

        Returns:
         jtwj, an Ndm x Ndm matrix

        """
        # Validate inputs
        if not isinstance(cstrat, ControlStrategy):
            raise TypeError('cstrat must be a ControlStrategy object')
        check.positive_scalar_integer(iteration, 'iteration', TypeError)
        check.real_nonnegative_scalar(contrast, 'contrast', TypeError)

        pwfn = cstrat.get_pixelweights_fn(iteration, contrast)

        if pwfn in self.jtwjs:
            return self.jtwjs[pwfn]
        else:
            raise ValueError('Control strategy contains filename not found ' +
                             'in JTWJ object, suggests mismatch between ' +
                             'strategy used for precomputation and for ' +
                             'GITL computation.')
        pass



def get_jtwj(jac, we0):
    """
    Given a fixed per-pixel weighting vector W (which include a fixed bad
    pixel map) and a Jacobian J, compute J.T@W.T@W@J ('jtwj')

    Precomputation is important as jac is 2 x Ndm x Npix with Npix >> Ndm >> 1,
    which makes this an expensive computation to do per-iteration.  We will
    apply subsequent correction factors handle per-iteration modifications more
    quickly.  Note that jac[0] and jac[1] are transposes of a Jacobian, which
    will be Npix x Ndm.

    Do this in two parts, one real and one imaginary (RTWR + ITWI).  This is
    acceptable as if J were actually complex, the transpose would be a
    conjugate transpose.

    Arguments:
     jac: 3D real-valued DM Jacobian matrix, as produced by ``calcjacs()``.
      Size is 2 x Ndm x Npix.
     we0: weighting matrix of length Npix

    Returns:
     jtwj, an Ndm x Ndm matrix

    """

    # Check inputs
    check.oneD_array(we0, 'we0', TypeError)

    # jac should match calcjacs output, which is 3D
    # Note for future: jac is 3D as it's a 2D complex matrix being stored in
    # FITS files usually, and FITS can only handle real data.  If a different
    # format is ever adopted, this could drop to 2D and save some wrangling.
    check.threeD_array(jac, 'jac', TypeError)
    if jac.shape[0] != 2:
        raise TypeError('axis 0 of jac must be length 2 (real/imag)')
    if jac.shape[2] != len(we0):
        raise TypeError('jac and we0 must have the same number of pixels')

    # Do real and imag separately
    rjac = np.squeeze(jac[0, :, :]).copy() # copy so as to not change jac
    for j in range(rjac.shape[0]):
        rjac[j, :] *= we0
        pass

    jtwj = rjac @ rjac.T
    del rjac

    ijac = np.squeeze(jac[1, :, :]).copy()
    for j in range(ijac.shape[0]):
        ijac[j, :] *= we0
        pass
    jtwj += ijac @ ijac.T

    return jtwj
