# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Encode matrix inversion algorithms for testing
"""

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse import isspmatrix_csr

import howfsc.util.check as check
from howfsc.model.mode import CoronagraphMode

valid_methods = ['cholesky', 'qr', 'pcg'] # these should be all lowercase

class InversionException(Exception):
    pass


def jac_solve(jac, e0, beta, wdm, we0, bp, jtwj, method):
    """
    Solve weighted least squares problem (min ||WAx = Wb||_2) and return -x

    Actual problem is (wdm.T * jac.T * diag(we0)**2 * jac * wdm + lam * I) x
     = wdm.T*jac.T*diag(we0)**2*e0

    Inputs are specific to the Jacobian inversion problem

    The three solvers are:
     - Cholesky Decomposition ('cholesky')
     - QR Decomposition ('qr')
     - Preconditioned Conjugate Gradient ('pcg')

    Warning: The left-side matrix A must not be singular!
     Tikhonov regularization via the beta parameter guarantees that constraint
     except when beta = -np.inf (equivalent to not using regularization). In
     this case, it is possible for zero-parameters in the weightings or non-
     full column rank to have a JTJ matrix which is singular (and so not
     strictly positive definite).  In this case, the decomposition will fail.

     Since you're already on shaky ground if you disabled regularization, and
     since the computation required to check for this case exhaustively is
     comparable to the computation to do the Cholesky/QR in the first place,
     this function will raise an exception if it runs across this but will
     not check it during function load.

     Note also that matrices which are highly ill-conditioned may produce
     incorrect results (Cholesky is more sensitive to this than QR and PCG).
     As computing a condition number is also as computationally intensive as
     the actual inversion, and as Tikhonov regularization will prevent this
     unless beta is very small, this condition will not be explicitly checked.

    Arguments:
     jac: 3D real-valued DM Jacobian matrix, as produced by ``calcjacs()``.
      Size is 2 x Ndm x Npix.
     e0: 1D complex-valued array containing electric field information.  This
      will be internally broken into real and imaginary parts to match jac; do
      not do it in advance.
     beta: real-valued number giving the exponent of the regularization
      parameter (10^beta).  The regularization will also be scaled by the
      largest singular value of the weighted jac.T*jac, so beta = 0 adds that
      max singular value to all jtj diagonal elements.
     wdm: DM actuator transformation matrix, encoding any constraints on
      actuator motion.  Given as a single Ndm x Ndm sparse matrix.
     we0: Pixel actuator weighting vector.  Incorporates both control strategy
      weighting and any removal of individual fixed bad pixels.  Will be a 1D
      array of length Npix.
     bp: Image-specific bad-pixel vector.  Includes *only* time-varying bad
      pixels from data collection (e.g. cosmic rays) or image processing.  True
      corresponds to a bad pixel.  Will be a 1D array of length Npix.
     jtwj: Precalculated jac.T * diag(we0)**2 * jac; this includes all fixed
      bad pixels and all per-pixel weighting, but does not include any bad
      pixels that vary (e.g. from cosmic ray flux).  Should be a 2D Ndm x Ndm
      array.
     method: string with one of ['cholesky', 'qr', 'pcg'] for the underlying
      algorithm to use.  Case insensitive.

    Returns:
     DM setting as 1d array

    """

    # Check method
    try:
        mlist = [method.lower() == valid.lower() for valid in valid_methods]
        pass
    except AttributeError: # method is not string-like
        raise TypeError('method must be a string')
    if not any(mlist):
        raise TypeError('Invalid algorithm specification string')

    # Check beta
    check.real_scalar(beta, 'beta', TypeError)

    # Check wdm
    if not isspmatrix_csr(wdm):
        raise TypeError('wdm must be a CSR-format sparse matrix')
    if len(wdm.shape) != 2:
        raise TypeError('wdm must be a 2D array')

    # wdm is sparse; use workaround to account for older numpy version not
    # being able to use isrealobj() correctly on sparse matrices
    dtype = wdm.dtype
    if issubclass(dtype.type, np.complexfloating):
        wdmi = wdm.imag
        wdmi.eliminate_zeros() # in place
        if wdmi.nnz == 0: # secretly real
            wdm = wdm.real
            pass
        else:
            raise TypeError('wdm must be real-valued')
        pass

    # Check jtwj
    check.twoD_array(jtwj, 'jtwj', TypeError)
    if not np.isrealobj(jtwj):
        if (jtwj.imag == 0).all(): # secretly real
            jtwj = jtwj.real
            pass
        else:
            raise TypeError('jtwj elements must be ' +
                                     'real-valued')
        pass
    if wdm.shape != jtwj.shape:
        raise TypeError('wdm and jtwj should be the same ' +
                                 'shape')
    pass

    # jac should match calcjacs output, which is 3D
    # Note for future: jac is 3D as it's a 2D complex matrix being stored in
    # FITS files usually, and FITS can only handle real data.  If a different
    # format is ever adopted, this could drop to 2D and save some wrangling.
    try:
        if len(jac.shape) != 3:
            raise TypeError('jac must be a 3D array')
        pass
    except AttributeError: # not ndarray-like
        raise TypeError('jac must be an ndarray-type object')

    if np.isrealobj(jac):
        rjac = np.zeros((jac.shape[1], jac.shape[0]*jac.shape[2]))
        for k in range(jac.shape[0]):
            rjac[:, k*jac.shape[2]:(k+1)*jac.shape[2]] = jac[k, :, :]
            pass
        pass
    else:
        if (jac.imag == 0).all(): # secretly real
            rjac = np.zeros((jac.shape[1], jac.shape[0]*jac.shape[2]))
            for k in range(jac.shape[0]):
                rjac[:, k*jac.shape[2]:(k+1)*jac.shape[2]] = jac[k, :, :].real
                pass
            pass
        else:
            raise TypeError('jac must be real-valued')
        pass

    # e0 needs be a column vector of [all reals, all imags]
    e0 = np.array(e0)
    if len(e0.shape) == 1:
        e0 = e0[:, np.newaxis]
        ce0 = np.vstack((e0.real, e0.imag))
        pass
    elif len(e0.shape) == 2:
        if e0.shape[0] == 1:
            ce0 = np.vstack((e0.T.real, e0.T.imag))
            pass
        elif e0.shape[1] == 1:
            ce0 = np.vstack((e0.real, e0.imag))
            pass
        else:
            raise TypeError('e0 must be a 1D array, 2D array ' +
                                     'with one axis of size 1, or castable ' +
                                     'to 1D')
        pass
    else:
        raise TypeError('e0 must be a 1D array or 2D array ' +
                                 'with one axis of size 1, or castable ' +
                                 'to 1D')

    # we0 should be a 1D matrix, same weight for real and imag
    we0 = np.squeeze(np.array(we0))
    if len(we0.shape) != 1:
        raise TypeError('we0 must be a 1D array or n-D array ' +
                                 'with one axis not of size 1, or castable ' +
                                 'to 1D')
    if not np.isrealobj(we0):
        if (we0.imag == 0).all(): # secretly real
            we0 = we0.real
            pass
        else:
            raise TypeError('we0 must be real-valued')
        pass
    cwe0 = np.vstack((we0[:, np.newaxis], we0[:, np.newaxis])) # 2Npix x 1

    # bp should be a 1D matrix, and if real is bad, imag is bad
    bp = np.squeeze(np.array(bp))
    if len(bp.shape) != 1:
        raise TypeError('bp must be a 1D array or n-D array ' +
                                 'with one axis not of size 1, or castable ' +
                                 'to 1D')
    if not np.isrealobj(bp):
        if (bp.imag == 0).all(): # secretly real
            bp = bp.real
            pass
        else:
            raise TypeError('bp must be real-valued')
        pass

    cbp = np.vstack((bp[:, np.newaxis], bp[:, np.newaxis])).astype('bool')

    # Check matrix dimensions
    if wdm.shape[0] != rjac.shape[0]:
        raise TypeError('Shape mismatch in wdm and rjac: ' +
                                 str(wdm.shape) + ' vs. ' +  str(rjac.shape))
    if rjac.shape[1] != cwe0.shape[0]:
        raise TypeError('Shape mismatch in rjac and cwe0: ' +
                                 str(rjac.shape) + ' vs. ' +  str(cwe0.shape))
    if ce0.shape[0] != cwe0.shape[0]:
        raise TypeError('Shape mismatch in ce0 and cwe0: ' +
                                 str(ce0.shape) + ' vs. ' +  str(cwe0.shape))
    if cbp.shape[0] != cwe0.shape[0]:
        raise TypeError('Shape mismatch in cbp and cwe0: ' +
                                 str(cbp.shape) + ' vs. ' +  str(cwe0.shape))
    if wdm.shape[0] != wdm.shape[1]:
        raise TypeError('wdm must be square')

    #---------------------
    # Inversion algorithm
    #---------------------

    # Correct jtwj for any intermittent bad pixels
    bpinds = [i for i, b in enumerate(cbp) if b] # get indices of bad pix
    if bpinds != []:
        # Separate no-bad-pixel case so we don't get errors trying to create a
        # file to hold a zero-size array
        rowt = np.zeros((rjac.shape[0], len(bpinds)))
        for index, bpindex in enumerate(bpinds):
            rowt[:, index] = rjac[:, bpindex]*cwe0[bpindex]
            cwe0[bpindex] = 0 # for B calc
            pass
        PTP = np.dot(rowt, rowt.T)
        pass
    else:
        PTP = 0
        pass

    # Assemble left side; can't precompute as frozen/tied acts may change
    AA = wdm.T.dot(wdm.T.dot(jtwj-PTP).T)

    # Get max jac singular value squared as scaling factor for beta to match
    # SVD analysis
    mjsvs = scipy.sparse.linalg.eigsh(AA, k=1)[0][0].real
    lam = 10**(beta)*mjsvs
    for j in range(AA.shape[0]):
        AA[j, j] += lam # regularization on diagonal only
        pass

    # Do vector as innermost, then this reduces to matrix-vector
    # == wdm^T J^T W_e0^T W_e0 E
    # use wdm.T.dot so the outer matrix uses sparse methods
    BB = wdm.T.dot(np.dot(rjac, cwe0*cwe0*ce0))

    if method.lower() == 'cholesky':
        try:
            # Get triangular factor from Cholesky [lapack = ?potrf]
            c, low = scipy.linalg.cho_factor(AA)
            # Solve triangular [lapack = ?potrs]
            x = scipy.linalg.cho_solve((c, low), BB)
            pass
        except scipy.linalg.LinAlgError:
            # other things could happen in the future to handle this better
            # if we really care about pathological cases
            raise InversionException('Cholesky decomposition failed due to ' +
                                     'ill-conditioned matrix')
    elif method.lower() == 'qr':
        try:
            # Get Q/R, and multiply Q.T by BB (=CQ.T) [lapack = ?geqrf, ?ormqr]
            CQ, R = scipy.linalg.qr_multiply(AA, BB.T, mode='right')
            # Solve triangular R*x = Q.T*BB [lapack = ?trtrs]
            x = scipy.linalg.solve_triangular(R, CQ.T)
        except scipy.linalg.LinAlgError:
            # other things could happen in the future to handle this better
            # if we really care about pathological cases
            raise InversionException('QR decomposition failed due to ' +
                                     'ill-conditioned matrix')
        pass
    elif method.lower() == 'pcg':
        x = _pcg(AA, BB)
    else:
        # Should never reach this if we parsed our methods right at the start
        raise ValueError('Invalid algorithm at inversion time')

    # Want negative setting to take field back out
    return -x



def inv_to_dm(deltadm, cfg, dmlist):
    """
    Convert the result of a pseudoinversion into an absolute DM setting in
    volts

    Arguments:
     deltadm: output vector from ``jac_solve`` with change in DM heights in
      radians (1D vector of DM settings)
     cfg: CoronagraphMode configuration
     dmlist: a list of ndarrays giving the current DM setting (list of 2D
      arrays of DM settings)

    Returns:
     list of ndarrays giving the new DM setting, of the same format as dmlist

    """

    # Check cfg
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    # Define DM variables and use them to check deltadm/dmlist
    ndms = len(cfg.dmlist)
    ndmact = np.cumsum([0]+[d.registration['nact']**2 for d in cfg.dmlist])
    try:
        if len(dmlist) != ndms:
            raise TypeError('dmlist number of DMs must match cfg')
        pass
    except TypeError: # not iterable
        raise TypeError('dmlist must be a list')

    for index, dm in enumerate(dmlist):
        check.twoD_array(dm, 'dm', TypeError)
        if dm.shape != (cfg.dmlist[index].registration['nact'],
                        cfg.dmlist[index].registration['nact']):
            raise TypeError('dmlist DM sizes must match cfg')
        pass

    try:
        if deltadm.size != ndmact[-1]:
            raise TypeError('deltadm size must match ' +
                                     'number of actuators')
        pass
    except AttributeError: # .size not defined
        raise TypeError('deltadm must be an ndarray')

    if len(np.squeeze(deltadm).shape) != 1:
        raise TypeError('deltadm must be a 1D vector or an N-D ' +
                                 'vector with N-1 size-1 axes')

    # Get "central wavelength"
    lamlist = [sl.lam for sl in cfg.sl_list]
    lamc = np.median(lamlist)

    # Add in radians rather than in volts so if there are nonlinearities in
    # actuator gain, the conversion functions handle them.
    outdmlist = []
    for n in range(ndms):
        ddmarr = np.reshape(deltadm[ndmact[n]:ndmact[n+1]],
                             (cfg.dmlist[n].registration['nact'],
                              cfg.dmlist[n].registration['nact']))
        dm0 = cfg.dmlist[n].dmvobj.volts_to_dmh(dmlist[n], lamc)
        varr = cfg.dmlist[n].dmvobj.dmh_to_volts(dm0+ddmarr, lamc)
        outdmlist.append(varr)
        pass
    return outdmlist


def _pcg(AA, BB, tol=1.0e-5):
    """
    Implements the basic PCG solver from Appendix B3 of "An Introduction to
    the Conjugate Gradient Method Without the Agonizing Pain" (Shewchuk 1994)
    to solve AA*x = BB for x.

    Uses a basic diagonal preconditioner.

    Arguments:
     AA: Matrix, sized as set up in jac_solve
     BB: Vector, sized as set up in jac_solve

    Keyword Arguments:
     tol: tolerance on vector agreement; algorithm terminates when the 2-norm
      of the error on (AA*x - b) is a fraction tol of the 2-norm of the
      error on the starting guess.  Defaults to 1e-5, which is chosen to be a
      bit less than 1 LSB at 16 bits; anything better is likely washed out in
      DM discretization anyway.

    Returns:
     vector which solves AA*x = BB in a least-squares sense to within tol

    """

    # Skip checks; this is an internal function and only called on matrices
    # that were already checked for validity

    i = 0

    # initial guess
    x = np.zeros((AA.shape[1], 1))
    imax = AA.shape[0]

    # preconditioner
    M = np.diag(AA)
    Minv = np.zeros((AA.shape[1], 1))
    Minv[M != 0, 0] = 1./M[M != 0]

    # start.
    r = BB - np.dot(AA, x)
    d = Minv*r # "apply preconditioner", diag happens to be easy
    delnew = np.dot(r.T, d)
    del0 = delnew
    while (i < imax) and delnew > tol**2*del0:
        q = np.dot(AA, d)
        alpha = delnew/np.dot(d.T, q)
        x += alpha*d
        if np.mod(i, 50) == 0:
            r = BB - np.dot(AA, x)
            pass
        else:
            r -= alpha*q
            pass
        s = Minv*r
        delold = delnew
        delnew = np.dot(r.T, s)
        beta = delnew/delold
        d = s + beta*d
        i += 1
        pass

    return x
