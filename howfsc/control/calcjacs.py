# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Compute Jacobians for wavefront control."""

#import multiprocessing
import numpy as np
from joblib import Parallel, delayed
import scipy.sparse as sparse

import howfsc.util.check as check
from howfsc.util.insertinto import insertinto
from howfsc.model.singlelambda import SingleLambda
from howfsc.model.mask import DMFace
from howfsc.model.mode import CoronagraphMode

JAC_DTYPE = np.dtype('double')
MAX_CHUNK_BYTES = 2**31 # for multiprocessing.Pool, python 3.7 and earlier

class CalcJacsException(Exception):
    """Thin exception class for Jacobian calculation."""
    pass

def get_ndhpix(cfg):
    """
    returns the number of dark hole pixels for a given cfg.
    useful for allocating the jac array becauseit is the number of columns

    Arguments:
     cfg: a CoronagraphMode object defining the optical diffractive
      representations of instrument hardware.  Must contain a field called
      ``sl_list`` with a list of ``SingleLambda`` objects corresponding to
      each control channel, which specifies wavelength and off-axis offset.
      Must contain a field called ``dmlist`` with a list of ``DMFace`` objects
      which contain DM properties.

    Returns:
     ndhpix = number of of pixels in the control region
    """

    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    try:
        _ = [len(sl.dh_inds) for sl in cfg.sl_list]

    except TypeError:
        raise TypeError('cfg.sl_list must be iterable')

    except AttributeError:  # does not contain field to iterate on
        raise TypeError('cfg must contain sl_list atrribute, and sl must ' +
                        'contain dh_inds')

    return np.cumsum([0]+[len(sl.dh_inds) for sl in cfg.sl_list])

def calcjacs(cfg, ijlist, dm0list=None, jacmethod='normal', num_process=1):
    """
    Call either the single process or multiple processes version of calcjacs

    If a DM crosstalk matrix exists and is not zeros, calculate:
    jac = jac + crosstalk*(shift(jac))

    Arguments:
     cfg: a CoronagraphMode object defining the optical diffractive
      representations of instrument hardware.  Must contain a field called
      ``sl_list`` with a list of ``SingleLambda`` objects corresponding to
      each control channel, which specifies wavelength and off-axis offset.
      Must contain a field called ``dmlist`` with a list of ``DMFace`` objects
      which contain DM properties.
     ijlist: a 1D list of DM actuators to poke, as integers between 0 and
      (total # of acts -1).  May be taken from a set of 2 DM arrays using
      ``generate_ijlist()``.

    Keyword Arguments:
     dm0list: a list of 2D arrays defining the DM setting corresponding to the
      Epup in cfg, or None.  Defaults to None, which uses the .initmaps setting
      stored with the CoronagraphMode object.  initmaps contains the DM voltage
      setting at which the calibration data was taken, and this will be
      subtracted off in SingleLambda.proptodm(), so the None case is a Jacobian
      done at the same DM setting as the calibration.  These DMs are
      expected to be the same size as the DMs in cfg.dmlist, and will throw an
      error if otherwise.
     jacmethod: String specifying whether to compute the Jacobian the 'normal'
      way with full-sized arrays, or the 'fast' way with an offcenter-cropped
      subarray centered on the actuator being poked. The cropped propagation
      goes only from DM1 to the FPM; after that is the same for both methods.
      The only valid values are 'normal' and 'fast'.
     num_process: number of parallel processes to use for the calculation.
      Default = 1. Multiprocessing should be used in coordination with setting
      the number of mkl threads. Set the number of mkl threads before calling
      this function. To set the number of mkl threads use
      mkl.set_num_threads_local() or os.environ['MKL_NUM_THREADS']. See
      precomp.py howfsc_precomputation() for examples.

    Returns:
     a 2 x len(ijlist) x npixel Jacobian matrix J, with the real parts in
      [0,:,:] and the imaginary parts in [1,:,:].

      The real and imaginary split permits the Jacobian data to be stored
      directly in a FITS file, which cannot handle complex-valued data.
      Outside of this constraint, the split doesn't need to be maintained.

    """

    if num_process == 1:
        jac = calcjacs_sp(cfg, ijlist, dm0list, jacmethod)

    else:
        jac = calcjacs_mp(cfg, ijlist, dm0list, jacmethod, num_process)


    ### apply crosstalk
    #
    # Modify jac.
    #    Definition of crosstalk in Jacobian (e.g. offset = [io, jo]):
    #    J_xtalk[:, ij, :] = sl.propagate_to_dh(dmh(i,j)
    #                        + xtalk(i,j)*dmh(i+io,j+jo))
    #    we approximate with:
    #    J_xtalk[:, ij, :] = sl.pokeprop(ij) + xtalk[ij]*sl.pokeprop(i+io,j+jo)
    #    equals (where J[] is Jacobian with no crosstalk)
    #    J_xtalk[:, ij, :] = J[:,ij,:] + xtalk[ij]*J[:,(i+io,j+jo),:]
    #    in sparse matrix form:
    #    J_xtalk = HC_sparse @ J
    #
    # 1. for each DM trim HC_sparse to ijlist: if an actuator is missing
    #    from ijlist, then remove corresponding column and row
    #
    # 2. assemble block diagonal square matrix (N, N), where N = # of columns
    #    of the jacobian (= length of ijlist)
    #    block diagonal matrix:
    #    HC_ijlist = [ DM1 HC_sparse | zeros         ]
    #                [ zeros         | DM2 HC_sparse ]
    #
    # 3. Matrix multiply:
    #    jac_xtalk = HC_ijlist @ jac
    #
    # Note: only actuators in the ijlist are included in the crosstalk
    #    matrix. Actuators missing from the ijlist could affect the crosstalk
    #    calculation. For example, if the crosstalk for actuator (i, j) is
    #    from (i+1, j), and (i+1, j) is not in ijlist, then there is no
    #    crosstalk added to (i, j).

    # these dm index conversion functions are band-independent:
    get_dmind2d = cfg.sl_list[0].get_dmind2d

    # get dmind, j, j from ijlist
    dmnjk = np.array([get_dmind2d(dm_act_ij) for dm_act_ij in ijlist])
    # dmnjk.shape = (4608, 3)
    # dmnjk[:,0] = dmind # i.e. 0 or 1
    # dmnjk[:,1] = row
    # dmnjk[:,2] = col

    # assemble block diagonal crosstalk matrix HC_ijlist
    list_HC_sparse = []
    for idm, DM in enumerate(cfg.dmlist):
        # for each dm, trim HC_sparse, if it exists
        # slice for this dm:
        dmnjk_idm = dmnjk[dmnjk[:, 0] == idm, :]
        if DM.dmvobj.crosstalk.HC_sparse is None:
            # no crosstalk, set to identity
            list_HC_sparse.append(
                sparse.csc_matrix(sparse.eye(dmnjk_idm.shape[0])))
        else:
            # this eliminates the actuators not in the ijlist:
            k_idm = DM.dmvobj.crosstalk.k_diag(dmnjk_idm[:, 1],
                                               dmnjk_idm[:, 2])
            list_HC_sparse.append(
                DM.dmvobj.crosstalk.HC_sparse[k_idm, :][:, k_idm])

    # create the block diagonal crosstalk matrix from the list for each dm
    HC_ijlist = sparse.block_diag(list_HC_sparse, format='csc')

    # multiply real, imag parts of jac separately
    jac_xtalk = np.zeros(jac.shape)
    jac_xtalk[0, :, :] = HC_ijlist @ jac[0, :, :]
    jac_xtalk[1, :, :] = HC_ijlist @ jac[1, :, :]

    # point jac reference to new array
    jac = jac_xtalk

    ### end crosstalk section

    return jac

def calcjacs_mp(cfg, ijlist, dm0list=None, jacmethod='normal', num_process=1):
    """
    Compute the Jacobian for the specified coronagraph mode using
    multiprocessing

    This function creates a joint DM/peak Jacobian, which jointly minimizes
    the change in E_corr/E_pk when actuators are moved.  (It differs from a
    more conventional DM Jacobian in that it accounts for the loss in Strehl
    when light is moved out of the star peak, and thus the corresponding planet
    peak as well.)

    Arguments:
     cfg: a CoronagraphMode object defining the optical diffractive
      representations of instrument hardware.  Must contain a field called
      ``sl_list`` with a list of ``SingleLambda`` objects corresponding to
      each control channel, which specifies wavelength and off-axis offset.
      Must contain a field called ``dmlist`` with a list of ``DMFace`` objects
      which contain DM properties.
     ijlist: a 1D list of DM actuators to poke, as integers between 0 and
      (total # of acts -1).  May be taken from a set of 2 DM arrays using
      ``generate_ijlist()``.

    Keyword Arguments:
     dm0list: a list of 2D arrays defining the DM setting corresponding to the
      Epup in cfg, or None.  Defaults to None, which uses the .initmaps setting
      stored with the CoronagraphMode object.  initmaps contains the DM voltage
      setting at which the calibration data was taken, and this will be
      subtracted off in SingleLambda.proptodm(), so the None case is a Jacobian
      done at the same DM setting as the calibration.  These DMs are
      expected to be the same size as the DMs in cfg.dmlist, and will throw an
      error if otherwise.
     jacmethod: String specifying whether to compute the Jacobian the 'normal'
      way with full-sized arrays, or the 'fast' way with an offcenter-cropped
      subarray centered on the actuator being poked. The cropped propagation
      goes only from DM1 to the FPM; after that is the same for both methods.
      The only valid values are 'normal' and 'fast'.
     num_process: number of parallel processes to use for the calculation.
      Default = 1. Multiprocessing should be used in coordination with setting
      the number of mkl threads. Set the number of mkl threads before calling
      this function. To set the number of mkl threads use
      mkl.set_num_threads_local() or os.environ['MKL_NUM_THREADS']. See
      precomp.py howfsc_precomputation() for examples.

    Returns:
     a 2 x len(ijlist) x npixel Jacobian matrix J, with the real parts in
      [0,:,:] and the imaginary parts in [1,:,:].

      The real and imaginary split permits the Jacobian data to be stored
      directly in a FITS file, which cannot handle complex-valued data.
      Outside of this constraint, the split doesn't need to be maintained.

    """

    # since each parallel process calls calcjacs(), we rely on all the
    # parameter checks done in calcjacs(). Except ijlist must be iterable and
    # have length > 0 for splitting up the work
    try:
        _ = [ij for ij in ijlist]
    except TypeError:
        raise TypeError('ijlist must be iterable')

    if len(ijlist) < 1:
        raise TypeError('ijlist len < 1')

    # divvy up ijlist for num_process processes
    # for python 3.7 and earlier, max chunk size returned by calcjacs()
    # from a multiprocessing pool is 2^31 bytes
    # then max_ijproc_len for an input ijlist makes an output of
    # 2 (real,imag) * 8 byte double * num acuators.
    # We restrict the ijlist for each process to the lesser of:
    # max_ijproc_len or len(ijlist)/num_process.
    ndhpix = get_ndhpix(cfg)[-1] # number of pixels (columns) for jac
    max_ijproc_len = MAX_CHUNK_BYTES//(2*JAC_DTYPE.itemsize*ndhpix)
    ijproc_len = min(int(np.ceil(len(ijlist)/num_process)),
                     max_ijproc_len) # actuators
    num_ijproc = int(np.ceil(len(ijlist)/ijproc_len))
    list_ijproc = [ijlist[ip::num_ijproc] for ip in range(num_ijproc)]

    # test return jac chunk size < 2^31 bytes (necessary for python3.7 and
    # earlier)
    # this check should always pass because of the above "divvy up"
    for ijproc in list_ijproc:
        chunk_size_bytes = 2*len(ijproc)*JAC_DTYPE.itemsize*ndhpix
        if chunk_size_bytes > 2**31:
            raise ValueError('multiprocessing pool chunk size too large')

    def jac_worker(ijproc):
        # cfg, ij, dm0list, jacmethod
        return calcjacs_sp(cfg, ijproc, dm0list, jacmethod)

    jac_list = Parallel(n_jobs=num_process, max_nbytes=None)(
        delayed(jac_worker)(ij) for ij in list_ijproc
    )

    # allocate jac
    jac = np.zeros((2, len(ijlist), ndhpix), dtype='double')
    for jac_ij, ijproc in zip(jac_list, list_ijproc):
        jac[0, ijproc, :] = jac_ij[0]
        jac[1, ijproc, :] = jac_ij[1]

    return jac

def calcjacs_sp(cfg, ijlist, dm0list=None, jacmethod='normal'):
    """
    Compute the Jacobian for the specified coronagraph mode.

    This function creates a joint DM/peak Jacobian, which jointly minimizes
    the change in E_corr/E_pk when actuators are moved.  (It differs from a
    more conventional DM Jacobian in that it accounts for the loss in Strehl
    when light is moved out of the star peak, and thus the corresponding planet
    peak as well.)

    Arguments:
     cfg: a CoronagraphMode object defining the optical diffractive
      representations of instrument hardware.  Must contain a field called
      ``sl_list`` with a list of ``SingleLambda`` objects corresponding to
      each control channel, which specifies wavelength and off-axis offset.
      Must contain a field called ``dmlist`` with a list of ``DMFace`` objects
      which contain DM properties.
     ijlist: a 1D list of DM actuators to poke, as integers between 0 and
      (total # of acts -1).  May be taken from a set of 2 DM arrays using
      ``generate_ijlist()``.

    Keyword Arguments:
     dm0list: a list of 2D arrays defining the DM setting corresponding to the
      Epup in cfg, or None.  Defaults to None, which uses the .initmaps setting
      stored with the CoronagraphMode object.  initmaps contains the DM voltage
      setting at which the calibration data was taken, and this will be
      subtracted off in SingleLambda.proptodm(), so the None case is a Jacobian
      done at the same DM setting as the calibration.  These DMs are
      expected to be the same size as the DMs in cfg.dmlist, and will throw an
      error if otherwise.
     jacmethod: String specifying whether to compute the Jacobian the 'normal'
      way with full-sized arrays, or the 'fast' way with an offcenter-cropped
      subarray centered on the actuator being poked. The cropped propagation
      goes only from DM1 to the FPM; after that is the same for both methods.
      The only valid values are 'normal' and 'fast'.

    Returns:
     a 2 x len(ijlist) x npixel Jacobian matrix J, with the real parts in
      [0,:,:] and the imaginary parts in [1,:,:].

      The real and imaginary split permits the Jacobian data to be stored
      directly in a FITS file, which cannot handle complex-valued data.
      Outside of this constraint, the split doesn't need to be maintained.

    """
    # Check necessary cfg elements
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')
    try:
        for sl in cfg.sl_list:
            if not isinstance(sl, SingleLambda):
                raise TypeError('cfg.sl_list must be a list of '
                                'SingleLambda objects')

    except TypeError:  # cfg.sl_list not iterable
        raise TypeError('cfg.sl_list must be iterable')
    except AttributeError:  # does not contain field to iterate on
        raise TypeError('cfg must contain sl_list atrribute')

    try:
        for dm in cfg.dmlist:
            if not isinstance(dm, DMFace):
                raise TypeError('cfg.dmlist must be a list of '
                                'DMFace objects')

    except TypeError:  # cfg.dmlist not iterable
        raise TypeError('cfg.dmlist must be iterable')
    except AttributeError:  # does not contain field to iterate on
        raise TypeError('cfg must contain dmlist atrribute')

    # Check ijlist
    lastind = 0
    for dm in cfg.dmlist:
        lastind += dm.registration['nact']**2

    lastind -= 1  # to account for list indexing starting at 0, not 1

    try:
        for ij in ijlist:
            check.nonnegative_scalar_integer(ij, 'ij', TypeError)
            if ij > lastind:
                raise ValueError('1D DM index out of range')

    except TypeError:  # ijlist not iterable
        raise TypeError('ijlist must be iterable')

    # Check dm0list
    if dm0list is not None:
        try:
            for dm in dm0list:
                check.twoD_array(dm, 'dm', TypeError)

        except TypeError:  # dm0list not iterable
            raise TypeError('dm0list must be iterable')

        if len(dm0list) != len(cfg.dmlist):
            raise TypeError('dm0list must have same length '
                            'as cfg.dmlist')

        for index, dm in enumerate(dm0list):
            if dm.shape != (cfg.dmlist[index].registration['nact'],
                            cfg.dmlist[index].registration['nact']):
                raise TypeError('dm0list must have same sizes '
                                'as cfg.dmlist')

    # if it is None, we'll deal with it on a SingleLambda by SingleLambda basis
    # using that object's embedded .initmaps

    # Check jacmethod
    if not isinstance(jacmethod, str):
        raise TypeError('jacmethod must be a string')
    if jacmethod not in ('normal', 'fast'):
        raise ValueError("jacmethod must be 'normal' or 'fast'.")

    # 2 re/im x npixels x nactuators
    ndhpix = get_ndhpix(cfg)
    jac = np.zeros((2, len(ijlist), ndhpix[-1])).astype('float64')

    # Main jac loop
    for index, sl in enumerate(cfg.sl_list):
        if dm0list is None:
            dmset_list = cfg.initmaps

        else:
            dmset_list = dm0list

        sl.get_jac_precomp(dmset_list)
        if jacmethod == 'fast':
            sl.get_fast_jac_precomp()

        sl.inorm = sl.get_inorm(dmset_list)
        if sl.inorm == 0:
            raise CalcJacsException('sl.inorm == 0, model is not well-formed')

        # Compute non-poked field
        edm0 = sl.eprop(dmset_list)
        ely = sl.proptolyot(edm0)
        edh0 = sl.proptodh(ely)
        edhv0 = edh0.ravel()[sl.dh_inds]

        # Unpoked peak
        epk0 = np.mean(sl.proptolyot_nofpm(edm0))/np.sqrt(sl.inorm)
        if epk0 == 0:
            raise CalcJacsException('epk0 == 0, model is not well-formed')

        # Compute poked per actuator with peak jac included
        for ijind, dmind in enumerate(ijlist):

            if jacmethod == 'normal':

                edm = sl.pokeprop(dmind, dmset_list)

                # DM actuator Jacobian
                ely = sl.proptolyot(edm)
                edh = sl.proptodh(ely)
                edhv = edh.ravel()[sl.dh_inds]

                # Peak Jacobian
                epk = np.mean(sl.proptolyot_nofpm(edm))/np.sqrt(sl.inorm)

            elif jacmethod == 'fast':

                edm_full, edm, xy_lower_left, nSurf = \
                    sl.croppedpokeprop(dmind, dmset_list)

                # DM actuator Jacobian
                ely = sl.croppedproptolyot(edm, xy_lower_left, nSurf)
                edh = sl.proptodh(ely)
                edhv = edh.ravel()[sl.dh_inds]

                # Peak Jacobian
                # Don't need to use any larger than 'normal' uses
                edm_like0 = insertinto(edm_full, edm0.shape)
                epk = np.mean(sl.proptolyot_nofpm(edm_like0))/np.sqrt(sl.inorm)

            # Modify the DM jac in place to include peak jac effects
            # J_both = J_dm/epk0 - edh0/epk0^2 op J_pk
            # with op = outer product
            jac[0, ijind, ndhpix[index]:ndhpix[index+1]] = \
                (edhv/epk0 - edhv0/epk0**2*epk).real
            jac[1, ijind, ndhpix[index]:ndhpix[index+1]] = \
                (edhv/epk0 - edhv0/epk0**2*epk).imag

    return jac

def generate_ijlist(cfg, dmarrlist):
    """
    Produce a list of DM actuator indices used in the Jacobian.

    Given a configuration and a set of boolean maps, produce a list of
    indices suitable for defining the size of the DM axis of the Jacobian
    calculation.

    Arguments:
     cfg: a CoronagraphMode object defining the optical model of the system
     dmarrlist: a list of arrays, one for each element of cfg.dmlist and of the
      same sizes.

    Returns:
     a list of integer indices

    """
    # check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    try:
        if len(cfg.dmlist) != len(dmarrlist):
            raise TypeError('dmarrlist must have the same number '
                            'of DMs as cfg.dmlist')
        for index, dm in enumerate(dmarrlist):
            check.twoD_array(dm, 'dm', TypeError)
            if dm.shape != (cfg.dmlist[index].registration['nact'],
                            cfg.dmlist[index].registration['nact']):
                raise TypeError('DMs supplied must match cfg DM sizes')

    except TypeError:   # not iterable
        raise TypeError('dmarrlist must be an iterable')

    # loop DMs, adding previous array totals
    darr = np.zeros((0,)).astype('int')
    cumact = int(0)

    for index, dm in enumerate(dmarrlist):
        darr = np.concatenate((darr, np.flatnonzero(dm)+cumact))
        cumact += int(cfg.dmlist[index].registration['nact']**2)

    return list(darr)
