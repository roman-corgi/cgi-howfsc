# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Function to run precomputation activities
"""

import os
import logging
import time
import multiprocessing

import numpy as np

from howfsc.model.mode import CoronagraphMode

from howfsc.control.cs import ControlStrategy
from howfsc.control.calcjacs import calcjacs
from howfsc.control.calcjtwj import JTWJMap
from howfsc.control.calcn2c import calcn2c

import howfsc.util.check as check

valid_jacmethod = ['normal', 'fast']

def howfsc_precomputation(cfg, dmset_list, cstrat, subcroplist,
                          jacmethod='fast', do_n2clist=True,
                          num_process=None, num_threads=None):
    """
    Precomputes data inputs required for howfsc_computation which do not change
    from iteration to iteration.

    Run this once before starting GITL, as it provides several of the inputs.

    NOTE: if you only need to recalculate a Jacobian for relinearization
    purposes, run this with do_n2clist=False.  The n2clist data set is not
    going to change appreciably, and for most models that part of the
    calculation will be by far the slowest part.

    Arguments:
     cfg: a CoronagraphMode object (i.e. optical model)

     dmset_list: a list of DM settings, in voltage, currently applied to
      the DMs.  The DM sizes and ordering must match cfg.dmlist.

     cstrat: a ControlStrategy object; this will be used to define the behavior
      of the wavefront control by setting the regularization, per-pixel
      weighting, multiplicative gain, and next-iteration probe height.  It will
      also contain information about fixed bad pixels.

     subcroplist: list of 4-tuples of (lower row, lower col,
      number of rows, number of columns), indicating where in a clean frame
      each PSF is taken.  All are integers; the first two must be >= 0 and the
      second two must be > 0.  This should have the same number of elements in
      the list as cfg has wavelength channels.

    Keyword Arguments:
     jacmethod: one of either 'normal' or 'fast', to indicate which calculation
      path to follow.  Defaults to 'fast'.
     do_n2clist: either True or False.  If True, runs n2clist calculation,
      otherwise skips it.  Calculation for widefov case (worst case) may take
      upwards of 12 hours, so there may be cases where running this is not
      preferred.  Defaults to True.
     num_process: number of parallel processes to use when calculating the
      Jacobian. If None or not defined (default=None), then num_process is:
      os.environ['HOWFS_CALCJAC_NUM_PROCESS'], if defined, else no
      multiprocessing.  If value is 0 then
      num_process = multiprocessing.cpu_count()//2, which is the number
      of cores on computers with Intel CPUs.
     num_threads: the mkl library automatically uses multi-threading
      which causes bottle necks and results in slower performance when using
      parallel processes. With Intel CPUs with multiple cores, best performance
      is usually with parallel processing and num_threads = 1.
      If not defined (default=None), order of precedence is:
      os.environ['HOWFS_CALCJAC_NUM_THREADS'], if defined, else
      os.environ['MKL_NUM_THREADS'], if defined, else
      1, if num_process is defined and > 1, else
      num_threads is not set and the default OS threading is used. For most
      linux/Intel machines, the OS will use max number of threads.

      Note: if os.environ['MKL_NUM_THREADS'] is defined, this value sets
      number of threads for all computations, not just calcjacs(). The
      computation time of single process functions might be longer.

    Returns:
     - 3D real-valued DM Jacobian array, as produced by calcjacs(). Shape is
       2 x ndm x npix
     - JTWJMap object to hold all of the JTWJ matrices required to implement
       a control strategy
     - list of normalized-intensity-to-contrast matrices (n2clist).  List is
       same length as number of wavelengths in cfg; each array is a 2D ndarray
       with size (number of rows, number of cols) consistent with that tuple in
       subcroplist.  Note: if do_n2clist == False, this output will be arrays
       of ones with the same sizing as given above.

    """

    # Validate inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    # Use cfg checker, going into cfg anyway
    cfg.sl_list[0]._check_dmset_list(dmset_list)
    cfg.sl_list[0].check_nsurf_pupildim()

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

    check.string(jacmethod, 'jacmethod', TypeError)
    if jacmethod not in valid_jacmethod:
        raise ValueError('jacmethod input not in valid set')
    if not isinstance(do_n2clist, bool):
        raise TypeError('do_n2clist must be a bool')

    # set num_process, see doc string above for order of precedence
    if num_process is None:
        # default input argument is None
        num_process = int(os.environ.get('HOWFS_CALCJAC_NUM_PROCESS', 1))

    # check valid num_process
    check.nonnegative_scalar_integer(num_process, 'num_process', TypeError)

    if num_process == 0:
        num_process = multiprocessing.cpu_count()//2

    # set num_threads, see doc string above for order of precedence
    if num_threads is None:
        # default input argument is None
        # os.environ.get('key') returns None if 'key' doesn't exist
        num_threads = os.environ.get('HOWFS_CALCJAC_NUM_THREADS')

        if num_threads is None and os.environ.get('MKL_NUM_THREADS') is None \
           and num_process > 1:
            num_threads = 1

        # if os.environ['MKL_NUM_THREADS'] is defined, leave num_threads = None,
        # if num_process == 1, leave num_threads = None,

    # check valid num_threads
    if not num_threads is None:
        # environ variables are strings, e.g. '1'
        if isinstance(num_threads, str):
            # trap strings that are not integers
            try:
                num_threads = int(num_threads)
            except ValueError:
                raise TypeError('num_threads must be integer > 0')

        check.positive_scalar_integer(num_threads, 'num_threads', TypeError)

    #------------------
    # Part 1: Jacobian
    #------------------

    # Compute ijlist from configuration file
    ndmact = 0
    for d in cfg.dmlist:
        ndmact += d.registration['nact']**2
        pass
    ijlist = range(ndmact)

    # log start time and calc jac
    logging.info('Begin Jacobian calculation')
    t0 = time.time()

    # log number processes
    logging.info('multiprocessing using %d processes', num_process)

    # if defined, set num_threads locally and log
    if not num_threads is None:

        # local means we will unset after calcjacs
        bUnset_mkl_num_threads = True

        # if os.environ['MKL_NUM_THREADS'] already has a value, need to save it
        save_mkl_num_threads = os.environ['MKL_NUM_THREADS'] \
                               if 'MKL_NUM_THREADS' in os.environ else None

        # set
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        logging.info('setting os.environ[MKL_NUM_THREADS] %d threads',
                     num_threads)

    else:
        # nothing to unset
        bUnset_mkl_num_threads = False

        # if environment variable set exernally, do nothing just log
        if 'MKL_NUM_THREADS' in os.environ:
            logging.info('MKL_NUM_THREADS = %s threads',
                         os.environ['MKL_NUM_THREADS'])

    # single process or multiprocessing
    jac = calcjacs(cfg, ijlist, dmset_list, jacmethod, num_process)

    # unset or return mkl to outside value
    if bUnset_mkl_num_threads:
        logging.info('resetting nkl num threads to previous value')
        if save_mkl_num_threads is None:
            # environ variable didn't exist, remove it
            del os.environ['MKL_NUM_THREADS']

        else:
            # return environ varialbe to original value
            os.environ['MKL_NUM_THREADS'] = save_mkl_num_threads


    t1 = time.time()
    logging.info('Jacobian calculation complete: elapsed time = %f secs',
                 t1-t0)

    #---------------
    # Part 2: JTWJs
    #---------------

    logging.info('Begin JTWJ calculation')
    t0 = time.time()
    jtwj_map = JTWJMap(cfg, jac, cstrat, subcroplist)
    t1 = time.time()
    logging.info('All JTWJ calculations complete: elapsed time = %f secs',
                 t1-t0)

    #----------------------------
    # Part 3 (optional): n2clist
    #----------------------------

    if do_n2clist:
        logging.info('Begin n2clist calculation')
        t0 = time.time()
        n2clist = []
        for idx in range(len(cfg.sl_list)):
            nrow = subcroplist[idx][2]
            ncol = subcroplist[idx][3]
            n2c = calcn2c(cfg, idx, nrow, ncol, dmset_list)
            n2clist.append(n2c)
            pass
        t1 = time.time()
        logging.info('All n2clist calcs complete: elapsed time = %f secs',
                     t1-t0)
        pass

    else:
        logging.info('Skipping n2clist calculation, populate with ones')
        n2clist = []
        for idx in range(len(cfg.sl_list)):
            nrow = subcroplist[idx][2]
            ncol = subcroplist[idx][3]
            n2c = np.ones((nrow, ncol))
            n2clist.append(n2c)
            pass
        pass


    return jac, jtwj_map, n2clist
