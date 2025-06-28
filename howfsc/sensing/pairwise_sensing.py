# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Collection of functions computing phase from sets of modulated intensity images
"""

import numpy as np

import howfsc.util.check as check

def get_coefficients(i0, iplus, iminus, phase_est):
    """
    Given an initial intensity image, the positive and negative
    probe intensity images, and the phase estimate of the probes:
    calculates the deltas and the matrix coefficients in Give'on, Kern, Shaklan
     2011, eqn's 8, 9.

    All arrays in the argument list below are the same shape and real-valued.

    Arguments:
     i0:         initial intensity image, a 2d array of real values
     iplus:      positive probe perturbation image, a 2d array of real values
     iminus:     negative probe perturbation image, a 2d array of real values
     phase_est:  the estimate phases of the perturbation, a 2d array of radians
                 this is estimated by propagating perturbations through an
                 optical model of the system
    Returns:
        delta:         the phase perturbation intensity difference image
                       (Iplus-Iminus)/2
        real_deltap:   the real part of the estimated phase perturbation
                       electric field
        imag_deltap:   the imaginary part of the phase perturbation electric
                       field
        badpixel:      a boolean image map of the bad pixels (True where bad)
    """
    # Check inputs
    input_list = [i0, iplus, iminus, phase_est]
    instr_list = ['i0', 'iplus', 'iminus', 'phase_est'] # strings for output

    for index, inp in enumerate(input_list):
        # All real 2D arrays of the same size
        check.twoD_array(inp, instr_list[index], TypeError)
        if input_list[0].shape != inp.shape:
            raise TypeError(instr_list[0] + ' and ' + instr_list[index]
                                    + ' must be the same size')
        if not (inp.imag == 0).all():
            raise TypeError(instr_list[index] + ' must be real')
        pass


    arg = (iplus + iminus)/2.0 - i0
    badpix = np.zeros(arg.shape, dtype='bool')
    nanpix = np.isnan(arg) # bad pixels are nan on input
    # boolean array flagging non-nan pixels that are problematic
    badpix[~nanpix] = (arg[~nanpix] <= 0)
    badpix[nanpix] = True # set nans after <= check to avoid RuntimeWarnings
    arg[badpix] = 0. # presetting keeps RuntimeWarnings from being thrown

    delta = (iplus - iminus)/2.0
    delta[badpix] = np.nan

    mag_deltap = np.sqrt(arg)
    mag_deltap[badpix] = np.nan

    real_deltap = mag_deltap*np.cos(phase_est)
    imag_deltap = mag_deltap*np.sin(phase_est)
    return delta, real_deltap, imag_deltap, badpix


def get_matrix(intensities, phases):
    """
    A function that takes the measured intensities and estimated phases and
    stacks them into a data cube form as in Giveon, Kern, Shaklan 2011 eqn's
    10.

    Arguments:
     intensities:    a 3d array of of real-valued image data.
                     The first element [0,:,:] is the initial intensity, and
                     subsequent arrays are the positive and negative phase
                     perturbations; ie 2N+1 arrays for N phase perturbations.
     phases:         a 3d array of phase data corresponding to the propagated
                     perturbations through the optical system, in radians

    Returns:
        deltas:         the phase perturbation intensity difference images
                        (Iplus-Iminus)/2
        real_deltaps:   the real part of the estimated phase perturbation
                        electric field
        imag_deltaps:   the imaginary part of the phase perturbation electric
                        field
        badpixels:      a boolean image map of the bad pixels (True where bad)
    """

    # Check inputs
    check.threeD_array(intensities, 'intensities', TypeError)
    check.threeD_array(phases, 'phases', TypeError)
    if not (intensities.imag == 0).all():
        raise TypeError('getmatrix() expects real-valued intensities')
    if not (phases.imag == 0).all():
        raise TypeError('getmatrix() expects real-valued phases')
    if (phases.shape[0]*2 + 1) != intensities.shape[0]:
        raise TypeError('There must be 2N+1 intensity measurements ' +
                                'where N = # of phase perturbations')

    i0 = intensities[0, :, :]

    deltas = np.zeros(phases.shape)
    real_deltaps = np.zeros(phases.shape)
    imag_deltaps = np.zeros(phases.shape)
    badpixels = np.full(phases.shape, False, dtype=bool)

    for i in range(phases.shape[0]):
        delta, real_deltap, imag_deltap, badpix = get_coefficients(i0=i0,
                                           iplus=intensities[2*i+1, :, :],
                                           iminus=intensities[2*i+2, :, :],
                                           phase_est=phases[i])
        deltas[i, :, :] = delta
        real_deltaps[i, :, :] = real_deltap
        imag_deltaps[i, :, :] = imag_deltap
        badpixels[i, :, :] = badpix
    return deltas, real_deltaps, imag_deltaps, badpixels


def solve_matrix(deltas, real_deltaps, imag_deltaps, badpixels, i0,
                 min_good_probes, eestclip, eestcondlim):
    """
    Solves Give'on, Kern, Shaklan 2011 eqn's 11.  Rearranges the datacubes to
    matrix form and uses a least-squares solver to solve it.

    First four below should be real-valued 3D arrays of the same size except
    badpixels, which is a 3D boolean array of the same size.

    Arguments:
     deltas:         the phase perturbation intensity difference images
                     (Iplus-Iminus)/2
     real_deltaps:   the real part of the estimated phase perturbation electric
                     field
     imag_deltaps:   the imaginary part of the phase perturbation electric
                     field
     badpixels:      a boolean image map of the bad pixels (True where bad)
     i0:             a 2D array of of real-valued image data with initial
                     intensity
     min_good_probes: an integer >= 2 giving the number of probe intensity
                      measurements that must be good at a pixel in order for
                      the estimation to be done at that pixel.  Given the
                      number of unknowns in the matrix solve, 2 good
                      measurements is mathematical minimum that can be
                      accepted.  Note that 'good' in this context means
                      not in badpixels as True AND does not produce an
                      estimate with negative intensity.  No other data quality
                      metric is applied.
     eestclip:        If the incoherent is less than -coherent*eestclip for an
                      e-field array element, that element will be marked as
                      bad. This permits incoherent estimates to be negative
                      (due to e.g. read noise) but caps how negative the
                      incoherent can be and how large the coherent can be.  In
                      particular, this mitigates spikes like the one seen in
                      PFR 218555.  Scalar value >= 0.  Setting to infinity
                      disables this check.
     eestcondlim:     The minimum condition number (ratio of smallest singular
                      value to largest in the least-square solve) which can be
                      accepted as a valid estimate.  As the matrix is N x 2 for
                      some N >= 2, there will only ever be two singular values.
                      A poorly-conditioned solution is close to rank 1, which
                      suggests the data is degenerate and not able to estimate
                      independent real and imaginary parts of the data.  >= 0.
                      Setting to 0 disables this check.

    Returns:
        realpart: the real part of the electric field, a 2d array
        imagpart: the imaginary part of the electric field, a 2d array
    """
    # Check inputs
    input_list = [deltas, real_deltaps, imag_deltaps, badpixels]
    instr_list = ['deltas', 'real_deltaps', 'imag_deltaps', 'badpixels']

    for index, inp in enumerate(input_list):
        # All real 3D arrays of the same size
        check.threeD_array(inp, instr_list[index], TypeError)
        if input_list[0].shape != inp.shape:
            raise TypeError(instr_list[0] + ' and ' + instr_list[index]
                                    + ' must be the same size')
        if not (inp.imag == 0).all(): # this will work on bools as well!
            raise TypeError(instr_list[index] + ' must be real')
        pass
    if badpixels.dtype != bool:
        raise TypeError('badpixels must be an array of booleans')
    check.twoD_array(i0, 'i0', TypeError)
    check.scalar_integer(min_good_probes, 'min_good_probes', TypeError)
    if min_good_probes < 2:
        raise ValueError('min_good_probes must be at least 2')
    check.real_nonnegative_scalar(eestcondlim, 'eestcondlim', TypeError)
    check.real_nonnegative_scalar(eestclip, 'eestclip', TypeError)

    realpart = np.zeros(deltas[0, :, :].shape)
    imagpart = np.zeros(deltas[0, :, :].shape)
    for ix, iy in np.ndindex(realpart.shape):
        #only use the part of the data where there aren't bad pixels
        good_inds = np.where(np.invert(badpixels[:, ix, iy]))[0]
        if len(good_inds) < min_good_probes:
            realpart[ix, iy] = np.nan
            imagpart[ix, iy] = np.nan
            continue

        deltacol = deltas[good_inds, ix, iy]

        #arrange into columnar format
        matrix = np.zeros((len(good_inds), 2))
        matrix[:, 0] = -2*imag_deltaps[good_inds, ix, iy]
        matrix[:, 1] = 2*real_deltaps[good_inds, ix, iy]

        sol, _, _, c = np.linalg.lstsq(matrix, deltacol, rcond=None)
        if c.min()/c.max() < eestcondlim:
            # flag as bad e-field if solve is ill-conditioned
            realpart[ix, iy] = np.nan
            imagpart[ix, iy] = np.nan
        else:
            # flag as
            icoh = sol[0]**2 + sol[1]**2
            iinc = i0[ix, iy] - icoh
            if iinc < -icoh*eestclip:
                # flag as bad e-field if coherent too large/incoh too negative
                realpart[ix, iy] = np.nan
                imagpart[ix, iy] = np.nan
            else:
                realpart[ix, iy] = sol[0]
                imagpart[ix, iy] = sol[1]

    return realpart, imagpart


def estimate_efield(intensities, phases,
                    min_good_probes=2, eestclip=np.inf, eestcondlim=0):
    """Estimates the electric field from the intensity measurements and phase
    perturbation measurements.  The general use function.

    intensities:    a 3d array of of real-valued image data.
                    The first element [0,:,:] is the initial intensity, and
                    subsequent arrays are the positive and negative phase
                    perturbations; ie 2N+1 arrays for N phase perturbations.
    phases:         a 3d array of phase data corresponding to the propagated
                    perturbations through the optical system, in radians.
                    Note the phase perturbations are assumed to be symmetric
                    +p, -p, so only the first of each pair of perturbations is
                    specified
    min_good_probes: an integer >= 2 giving the number of probe intensity
                      measurements that must be good at a pixel in order for
                      the estimation to be done at that pixel.  Given the
                      number of unknowns in the matrix solve, 2 good
                      measurements is mathematical minimum that can be
                      accepted.  Note that 'good' in this context means
                      not in badpixels as True AND does not produce an
                      estimate with negative intensity.  No other data quality
                      metric is applied.  Defaults to 2.
    eestclip:        If the incoherent is less than -coherent*eestclip for an
                      e-field array element, that element will be marked as
                      bad. This permits incoherent estimates to be negative
                      (due to e.g. read noise) but caps how negative the
                      incoherent can be and how large the coherent can be.  In
                      particular, this mitigates spikes like the one seen in
                      PFR 218555.  Scalar value >= 0.  Setting to infinity
                      disables this check.  Defaults to np.inf.
    eestcondlim:     The minimum condition number (ratio of smallest singular
                      value to largest in the least-square solve) which can be
                      accepted as a valid estimate.  As the matrix is N x 2 for
                      some N >= 2, there will only ever be two singular values.
                      A poorly-conditioned solution is close to rank 1, which
                      suggests the data is degenerate and not able to estimate
                      independent real and imaginary parts of the data.  >= 0.
                      Setting to 0 disables this check.  Defaults to 0.

    Returns:
      a 2d-array of complex electric field, with bad estimates as NaNs

    """

    # Check inputs
    check.threeD_array(intensities, 'intensities', TypeError)
    check.threeD_array(phases, 'phases', TypeError)
    if not (intensities.imag == 0).all():
        raise TypeError('getmatrix() expects real-valued intensities')
    if not (phases.imag == 0).all():
        raise TypeError('getmatrix() expects real-valued phases')
    if (phases.shape[0]*2 + 1) != intensities.shape[0]:
        raise TypeError('There must be 2N+1 intensity measurements ' +
                                'where N = # of phase perturbations')
    check.scalar_integer(min_good_probes, 'min_good_probes', TypeError)
    if min_good_probes < 2:
        raise ValueError('min_good_probes must be at least 2')
    check.real_nonnegative_scalar(eestcondlim, 'eestcondlim', TypeError)
    check.real_nonnegative_scalar(eestclip, 'eestclip', TypeError)

    deltas, real_deltaps, imag_deltaps, badpixels = get_matrix(intensities,
                                                               phases)
    realpart, imagpart = solve_matrix(deltas, real_deltaps, imag_deltaps,
                                      badpixels=badpixels,
                                      i0=intensities[0, :, :],
                                      min_good_probes=min_good_probes,
                                      eestcondlim=eestcondlim,
                                      eestclip=eestclip,
    )
    return realpart + 1j*imagpart
