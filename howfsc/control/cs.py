# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
File for control strategy class which is called by HOWFSC
"""

import numpy as np
import scipy.sparse

from howfsc.control.parse_cs import create_cs_lists, is_point_in_box
from howfsc.model.mode import CoronagraphMode
from howfsc.util.actlimits import maplimits, sparsefrommap
from howfsc.util.flat_tie import checktie
from howfsc.util.insertinto import insertinto
import howfsc.util.check as check

class ControlStrategy(object):
    """
    This class implements several methods to enable "control strategies",
    which allow wavefront control parameters to be tuned from 2D lookup tables
    which encode operator knowledge about HOWFSC performance
    """

    def __init__(self, fn):
        """
        Loads in seven lists of regions and a fixed-bad-pixel file from a
        YAML control strategy file.

        Raises CSException if the file is invalid.

        Arguments:
         fn: string containing path to a filename with a control strategy YAML
          file

        """

        self.regularization, self.pixelweights, self.pixelweights_fn, \
            self.dmmultgain, self.probeheight, \
            self.unprobedsnr, self.probedsnr, self.fixedbp \
            = create_cs_lists(fn, usefiles=True)

        pass


    def get_regularization(self, iteration, contrast):
        """
        Pulls a regularization value from a region stored in the control
        strategy.

        Raises a ValueError if there is no region which matches the supplied
        iteration value and mean total contrast.  The control strategy
        specification is checked on class instantiation, and specifies one and
        only one value for any valid iteration/contrast pair, so if this
        exception is ever thrown, it implies some uncaught software bug.

        Arguments:
         iteration: integer >= 1, giving iteration number for the iteration
          that is about to happen.  (Starting iteration, which used data
          preloaded on board, is iteration 0.)
         contrast: floating-point value >= 0.  Mean total contrast across all
          dark hole pixels in the iteration that just completed (i.e.
          'iteration' - 1).

        Returns:
         single floating-point value equal to log10 of the value relative to
          square of the largest singular value of the weighted Jacobian. (This
          turns out to be exactly what our least-squares solver is expecting.)

        """
        # Check inputs
        check.positive_scalar_integer(iteration, 'iteration', TypeError)
        check.real_nonnegative_scalar(contrast, 'contrast', TypeError)

        for r in self.regularization:
            if is_point_in_box(iteration, contrast, r):
                return r.value
            pass
        raise ValueError('iteration/contrast pair not found in ' +
                         'regularization')


    def get_pixelweights(self, iteration, contrast):
        """
        Pulls a list of matrices from a region stored in the control
        strategy.

        Raises a ValueError if there is no region which matches the supplied
        iteration value and mean total contrast.  The control strategy
        specification is checked on class instantiation, and specifies one and
        only one value for any valid iteration/contrast pair, so if this
        exception is ever thrown, it implies some uncaught software bug.

        Arguments:
         iteration: integer >= 1, giving iteration number for the iteration
          that is about to happen.  (Starting iteration, which used data
          preloaded on board, is iteration 0.)
         contrast: floating-point value >= 0.  Mean total contrast across all
          dark hole pixels in the iteration that just completed (i.e.
          'iteration' - 1).

        Returns:
         list of 2D arrays, one for each wavelength to be used with a
          CoronagraphMode

        """
        # Check inputs
        check.positive_scalar_integer(iteration, 'iteration', TypeError)
        check.real_nonnegative_scalar(contrast, 'contrast', TypeError)

        for r in self.pixelweights:
            if is_point_in_box(iteration, contrast, r):
                return r.value
            pass
        raise ValueError('iteration/contrast pair not found in ' +
                         'pixelweights')


    def get_pixelweights_fn(self, iteration, contrast):
        """
        Pulls a filename from a region stored in the control strategy.

        Want to save filenames because multiple regions may point at the same
        file and tracking this allows efficiencies elsewhere.

        Raises a ValueError if there is no region which matches the supplied
        iteration value and mean total contrast.  The control strategy
        specification is checked on class instantiation, and specifies one and
        only one value for any valid iteration/contrast pair, so if this
        exception is ever thrown, it implies some uncaught software bug.

        Arguments:
         iteration: integer >= 1, giving iteration number for the iteration
          that is about to happen.  (Starting iteration, which used data
          preloaded on board, is iteration 0.)
         contrast: floating-point value >= 0.  Mean total contrast across all
          dark hole pixels in the iteration that just completed (i.e.
          'iteration' - 1).

        Returns:
         list of 2D arrays, one for each wavelength to be used with a
          CoronagraphMode

        """
        # Check inputs
        check.positive_scalar_integer(iteration, 'iteration', TypeError)
        check.real_nonnegative_scalar(contrast, 'contrast', TypeError)

        for r in self.pixelweights_fn:
            if is_point_in_box(iteration, contrast, r):
                return r.value
            pass
        raise ValueError('iteration/contrast pair not found in ' +
                         'pixelweights_fn')



    def get_dmmultgain(self, iteration, contrast):
        """
        Pulls a scalar multiplier from a region stored in the control
        strategy.

        Raises a ValueError if there is no region which matches the supplied
        iteration value and mean total contrast.  The control strategy
        specification is checked on class instantiation, and specifies one and
        only one value for any valid iteration/contrast pair, so if this
        exception is ever thrown, it implies some uncaught software bug.

        Arguments:
         iteration: integer >= 1, giving iteration number for the iteration
          that is about to happen.  (Starting iteration, which used data
          preloaded on board, is iteration 0.)
         contrast: floating-point value >= 0.  Mean total contrast across all
          dark hole pixels in the iteration that just completed (i.e.
          'iteration' - 1).

        Returns:
         single floating-point value > 0.  Multiplying the change the DM
          setting, allowing it to act like a damping factor; 1 implies no
          change

        """
        # Check inputs
        check.positive_scalar_integer(iteration, 'iteration', TypeError)
        check.real_nonnegative_scalar(contrast, 'contrast', TypeError)

        for r in self.dmmultgain:
            if is_point_in_box(iteration, contrast, r):
                return r.value
            pass
        raise ValueError('iteration/contrast pair not found in ' +
                         'dmmultgain')


    def get_unprobedsnr(self, iteration, contrast):
        """
        Pulls a target unprobed SNR from a region stored in the control
        strategy.

        Raises a ValueError if there is no region which matches the supplied
        iteration value and mean total contrast.  The control strategy
        specification is checked on class instantiation, and specifies one and
        only one value for any valid iteration/contrast pair, so if this
        exception is ever thrown, it implies some uncaught software bug.

        Arguments:
         iteration: integer >= 1, giving iteration number for the iteration
          that is about to happen.  (Starting iteration, which used data
          preloaded on board, is iteration 0.)
         contrast: floating-point value >= 0.  Mean total contrast across all
          dark hole pixels in the iteration that just completed (i.e.
          'iteration' - 1).

        Returns:
         single floating-point value > 0

        """
        # Check inputs
        check.positive_scalar_integer(iteration, 'iteration', TypeError)
        check.real_nonnegative_scalar(contrast, 'contrast', TypeError)

        for r in self.unprobedsnr:
            if is_point_in_box(iteration, contrast, r):
                return r.value
            pass
        raise ValueError('iteration/contrast pair not found in ' +
                         'unprobedsnr')


    def get_probedsnr(self, iteration, contrast):
        """
        Pulls a target probed SNR from a region stored in the control
        strategy.

        Raises a ValueError if there is no region which matches the supplied
        iteration value and mean total contrast.  The control strategy
        specification is checked on class instantiation, and specifies one and
        only one value for any valid iteration/contrast pair, so if this
        exception is ever thrown, it implies some uncaught software bug.

        Arguments:
         iteration: integer >= 1, giving iteration number for the iteration
          that is about to happen.  (Starting iteration, which used data
          preloaded on board, is iteration 0.)
         contrast: floating-point value >= 0.  Mean total contrast across all
          dark hole pixels in the iteration that just completed (i.e.
          'iteration' - 1).

        Returns:
         single floating-point value > 0

        """
        # Check inputs
        check.positive_scalar_integer(iteration, 'iteration', TypeError)
        check.real_nonnegative_scalar(contrast, 'contrast', TypeError)

        for r in self.probedsnr:
            if is_point_in_box(iteration, contrast, r):
                return r.value
            pass
        raise ValueError('iteration/contrast pair not found in ' +
                         'probedsnr')



    def get_probeheight(self, iteration, contrast):
        """
        Pulls a probe height from a region stored in the control
        strategy.

        Raises a ValueError if there is no region which matches the supplied
        iteration value and mean total contrast.  The control strategy
        specification is checked on class instantiation, and specifies one and
        only one value for any valid iteration/contrast pair, so if this
        exception is ever thrown, it implies some uncaught software bug.

        Arguments:
         iteration: integer >= 1, giving iteration number for the iteration
          that is about to happen.  (Starting iteration, which used data
          preloaded on board, is iteration 0.)
         contrast: floating-point value >= 0.  Mean total contrast across all
          dark hole pixels in the iteration that just completed (i.e.
          'iteration' - 1).

        Returns:
         single floating-point value > 0 giving the desired mean probe
          amplitude across the region of modulation

        """
        # Check inputs
        check.positive_scalar_integer(iteration, 'iteration', TypeError)
        check.real_nonnegative_scalar(contrast, 'contrast', TypeError)

        for r in self.probeheight:
            if is_point_in_box(iteration, contrast, r):
                return r.value
            pass
        raise ValueError('iteration/contrast pair not found in ' +
                         'probeheight')


def get_wdm(cfg, dmlist, tielist):
    """
    Get components of per-actuator weighting matrix

    This can be used to tie and freeze actuators for neighbor rules and
    voltage limits, or exclude actuators for other reasons.

    wdm is computed internally as (I-F).  F covers two things:
     - It covers dead and tied actuators: dead actuators will have zeroed
       rows, and tied actuators will all be tied together. The information
       for this is precomputed.
     - F covers iteration-specific DM constraints; actuators that are too
       high or low will have zeroed rows, and neighbor-rule actuators will
       be tied together.
    F is produced by a specific separate function.  wdm is sparse.

    Arguments:
     cfg: a CoronagraphMode object
     dmlist: a list of ndarrays giving the current DM setting (list of 2D
      arrays of DM settings)
     tielist: a list of tiemaps (each a 2D array of integers, of the same
      size as a DM setting, which can take on the values 0, -1, or
      consecutive integers 1 -> N)

    Returns:
     sparse Ndm x Ndm array, with Ndm the total number of actuators

    """
    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode')

    try:
        if len(dmlist) != len(cfg.dmlist):
            raise TypeError('dmlist invalid length')
        for index, dm in enumerate(dmlist):
            check.twoD_array(dm, 'dm', TypeError)
            nact = cfg.dmlist[index].registration['nact']
            if dm.shape != (nact, nact):
                raise TypeError('Invalid DM shape in get_wdm')
            pass
        pass
    except TypeError: # not iterable
        raise TypeError('dmlist not a list in get_wdm input')

    try:
        if len(tielist) != len(cfg.dmlist):
            raise TypeError('tielist invalid length')
        for index, tie in enumerate(tielist):
            check.twoD_array(tie, 'tie', TypeError)
            if dmlist[index].shape != tie.shape:
                raise TypeError('dm shape must match tiemap')
            if not checktie(tie):
                raise ValueError('tie must have values 0, -1, or ' +
                                 'consecutive integers 1 -> N')
            pass
        pass
    except TypeError: # not iterable
        raise TypeError('tielist not a list in get_wdm input')

    # Tie and freeze actuators here
    limitlist = []
    for index, dm in enumerate(cfg.dmlist):
        limitlist.append(maplimits(dmlist[index], dm.dmvobj,
                                   tiemap=tielist[index]))
        pass
    F = sparsefrommap(limitlist, cfg)
    return scipy.sparse.eye(F.shape[0], F.shape[1], format='csr') - F


def get_we0(cfg, cstr, croplist, iteration, contrast):
    """
    Get per-pixel weighting

    In addition to weighting regions, it will also exclude any fixed bad pixels
     on EXCAM happen to fall in the control region for any wavelength.

    This function will also always apply an internal correction per wavelength
    to compensate for the fact the Jacobian actuator pokes are applied in
    radians, but the actual settings are applied in nm.  This does not need to
    be included in user-defined pixel weighting.  'Applied' here means the
    correction is incorporated into the we0 matrix, which is unitless; the
    radians-to-nm conversion is done elsewhere.  The scaling is done to place
    the reference (scale = 1) wavelength at the median of the wavelengths in
    the model.

    we0 produces 'amplitude' weighting, which weights the electric-field
    amplitude rather than the intensity.

    Arguments:
     cfg: CoronagraphMode object
     cstr: ControlStrategy object; pixelweights should have the same number of
      matrices as cfg has control wavelengths, and in the same order.  (The
      number will be checked; the order is outside the scope of this function.)
     croplist: list of 4-tuples of (lower row, lower col,
      number of rows, number of columns), indicating where in a clean frame
      each PSF is taken.  All are integers; the first two must be >= 0 and the
      second two must be > 0.  This should have the same number of elements as
      the model has wavelengths.
     iteration: integer >= 1, giving iteration number for the iteration
      that is about to happen.  (Starting iteration, which used data
      preloaded on board, is iteration 0.)
     contrast: floating-point value >= 0.  Mean total contrast across all
      dark hole pixels in the iteration that just completed (i.e.
      'iteration' - 1).

    Returns:
     a 1D array with the same number of pixels as are in all of the dark hole
      arrays (dh.e) in the SingleLambda objects in cfg (cfg.sl_list)

    """
    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode')
    if not isinstance(cstr, ControlStrategy):
        raise TypeError('cstr must be a ControlStrategy')

    if not isinstance(croplist, list):
        raise TypeError('croplist must be a list')
    if len(croplist) != len(cfg.sl_list):
        raise TypeError('Number of crop regions does not match model')

    for index, crop in enumerate(croplist):
        if not isinstance(crop, tuple):
            raise TypeError('croplist[' + str(index) + '] must be a tuple')
        if len(crop) != 4:
            raise TypeError('Each element of croplist must be a 4-tuple')
        check.nonnegative_scalar_integer(crop[0], 'croplist[' +
                                         str(index) + '][0]', TypeError)
        check.nonnegative_scalar_integer(crop[1], 'croplist[' +
                                         str(index) + '][1]', TypeError)
        check.positive_scalar_integer(crop[2], 'croplist[' +
                                      str(index) + '][2]', TypeError)
        check.positive_scalar_integer(crop[3], 'croplist[' +
                                      str(index) + '][3]', TypeError)
        pass


    check.positive_scalar_integer(iteration, 'iteration', TypeError)
    check.real_nonnegative_scalar(contrast, 'contrast', TypeError)

    pixelweights = cstr.get_pixelweights(iteration, contrast)
    if len(pixelweights) != len(cfg.sl_list):
        raise TypeError('Mismatch between ControlStrategy pixelweights ' +
                        'length and CoronagraphMode sl_list length')

    # Get "central wavelength"
    lamlist = [sl.lam for sl in cfg.sl_list]
    lamc = np.median(lamlist)

    # build we0 vector
    ndhpix = np.cumsum([0]+[len(sl.dh_inds) for sl in cfg.sl_list])
    we0 = np.ones((ndhpix[-1],))
    bp = np.zeros((ndhpix[-1],)).astype('bool')
    for index, p in enumerate(pixelweights):
        # if you set up your control strategy and coronagraph model self-
        # consistently, this should not be necessary, but just in case.  Output
        # will equal input if sizes are same
        iip = insertinto(p, cfg.sl_list[index].dh.e.shape)

        # get relevant subregion of fixedbp per wavelength
        nrow = croplist[index][2]
        ncol = croplist[index][3]

        slbp = np.ones((nrow, ncol)).astype('bool')
        chunk = cstr.fixedbp[croplist[index][0]:croplist[index][0]+nrow,
                             croplist[index][1]:croplist[index][1]+ncol]
        slbp[:chunk.shape[0], :chunk.shape[1]] = chunk

        # if you set up your coronagraph model and HOWFSC data collection self-
        # consistently, this should not be necessary, but just in case
        if (nrow, ncol) != cfg.sl_list[index].dh.e.shape:
            # insertinto handles all the shapes but only pads with zeros, so
            # use boolean tricks to get Trues.  Want True because any pad has
            # no info and must be a bad pixel in the map
            nslbp = ~slbp
            pnslbp = insertinto(nslbp, cfg.sl_list[index].dh.e.shape)
            slbp = (~pnslbp).astype('bool')
            pass

        dh = cfg.sl_list[index].dh.e.astype('bool')
        we0[ndhpix[index]:ndhpix[index+1]] = iip[dh]
        we0[ndhpix[index]:ndhpix[index+1]] *= lamc/cfg.sl_list[index].lam
        bp[ndhpix[index]:ndhpix[index+1]] = slbp[dh]

        pass

    # Zero out bad pixels
    we0[bp] = 0.

    return we0
