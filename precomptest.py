# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Compute a Jacobian for the mode defined in cfg
"""

import time
import os
import argparse
import cProfile
import pstats
import logging
import numpy as np
import astropy.io.fits as fits

import howfsc
from howfsc.model.mode import CoronagraphMode
from howfsc.control.cs import ControlStrategy
from howfsc.precomp import howfsc_precomputation

JACMETHOD = 'fast'

if __name__ == "__main__":

    # setup for cmd line args
    ap = argparse.ArgumentParser(
        prog='python precomptest.py',
        description="Compute a full or partial Jacobian.  Stores the output in a FITS file, if a name is provided, and prints the time spent to do the calculation.  The filename must not be used already, as this script will not overwrite an existing file.  The output file will be a 2 x (num actuators) x (num pixels) 3D array. The first axis with 2 dimensions will be the real and imaginary parts of the Jacobian, as FITS files cannot store complex data directly."
    )

    ap.add_argument(
        '--mode', default='widefov',
        choices=['widefov', 'narrowfov', 'spectroscopy', 'nfov_dm', 'nfov_flat', 'smalljac'],
        help="coronagraph mode from test data; must be one of 'widefov', 'narrowfov', 'nfov_dm', 'nfov_flat', 'spectroscopy', or 'smalljac'"
    )

    ap.add_argument(
        '--num_process', default=None,
        help="number of processes, defaults to None, indicating no multiprocessing. If 0, then howfsc_precomputation() defaults to the available number of cores", type=int
    )

    ap.add_argument(
        '--num_threads', default=None,
        help="set mkl_num_threads to use for parallel processes for calcjacs(). If None (default), then do nothing (number of threads might also be set externally through environment variable 'MKL_NUM_THREADS' or 'HOWFS_CALCJAC_NUM_THREADS'",
        type=int
    )

    ap.add_argument(
        '--do_calcn2c', action='store_true',
        help='If present, include calcn2c() calculation'
    )

    ap.add_argument(
        '--profile', action='store_true',
        help='If present, runs the Python cProfile profiler on calcjacs and displays the top 20 howfsc/ contributors to cumulative time'
    )

    ap.add_argument(
        '--savefn', default=None, help="save jac as a fits file to given filename"
    )

    ap.add_argument(
        '--logfile', default=None,
        help="If present, absolute path to file location to log to."
    )

    args = ap.parse_args()

    # Set up logging
    if args.logfile:
        logging.basicConfig(filename=args.logfile, level=logging.INFO)

    else:
        logging.basicConfig(level=logging.INFO)

    start_time = time.time()
    logging.info('start time: %s', time.asctime())

    isprof = args.profile
    if isprof:
        pr = cProfile.Profile()

    # Load cfg
    if args.mode == 'widefov':
        cfgpath = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                               'model', 'testdata', 'widefov', 'widefov.yaml')
        cstratfile = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                                  'model', 'testdata', 'widefov',
                                  'cstrat_widefov_2weight.yaml')

    elif args.mode == 'narrowfov':
        cfgpath = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                               'model', 'testdata', 'narrowfov', 'narrowfov.yaml')
        cstratfile = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                                  'model', 'testdata', 'narrowfov',
                                  'cstrat_narrowfov_2weight.yaml')

    elif args.mode == 'spectroscopy':
        cfgpath = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                               'model', 'testdata', 'spectroscopy', 'spectroscopy.yaml')
        cstratfile = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                                  'model', 'testdata', 'spectroscopy',
                                  'cstrat_spectroscopy_2weight.yaml')

    elif args.mode == 'nfov_dm':
        cfgpath = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                               'model', 'testdata', 'narrowfov', 'narrowfov_dm.yaml')
        cstratfile = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                                  'model', 'testdata', 'narrowfov',
                                  'cstrat_nfov_dm_2weight.yaml')

    elif args.mode == 'nfov_flat':
        cfgpath = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                               'model', 'testdata', 'narrowfov', 'narrowfov_flat.yaml')
        cstratfile = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                                  'model', 'testdata', 'narrowfov',
                                  'cstrat_nfov_flat_2weight.yaml')

    elif args.mode == 'smalljac':
        cfgpath = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                               'model', 'testdata', 'ut', 'ut_smalljac.yaml')
        cstratfile = os.path.join(os.path.dirname(os.path.abspath(howfsc.__file__)),
                                  'control', 'testdata', 'ut_good_cs.yaml')


    else:
        # Should never reach this if choices=[..] works right
        raise Exception('Bad input for coronagraph mode')

    logging.info('reading cfg %s', cfgpath)

    cfg = CoronagraphMode(cfgpath)
    dmset_list = cfg.initmaps
    cstrat = ControlStrategy(cstratfile)
    subcroplist = [(436, 436, 153, 153)]*len(cfg.sl_list)

    # Main jac loop
    t0 = time.time()
    if isprof:
        pr.enable()

    jac, jtwjs, n2clist = howfsc_precomputation(cfg=cfg,
                                                dmset_list=dmset_list,
                                                cstrat=cstrat,
                                                subcroplist=subcroplist,
                                                jacmethod=JACMETHOD,
                                                do_n2clist=args.do_calcn2c,
                                                num_process=args.num_process,
                                                num_threads=args.num_threads)

    if isprof:
        pr.disable()

    t1 = time.time()
    logging.info('elapsed time = %.1f seconds', t1-t0)
    #print(str(t1-t0))

    if isprof:
        ps = pstats.Stats(pr)
        ps.sort_stats('cumtime').print_stats('howfsc', 20)

    if args.savefn:
        # jac shape is (2, N_dm_acts, M_pix)
        # jac[0, :, :] is real
        # jac[1, :, :] is imag
        fits.writeto(args.savefn, jac)
        if not n2clist is None:
            fits.writeto(os.path.splitext(args.savefn)[0] + '_n2clist.fits', np.array(n2clist))

    # compare result to reference for accuracy
    if not n2clist is None:
        ref_fn = 'precomptest_'+args.mode+'_reference_n2clist.fits'
        if os.path.exists(ref_fn):
            n2c_ref = fits.getdata(ref_fn)
            n2c_test = np.array(n2clist)

            try:
                err_max = np.max(np.abs(
                    n2c_ref[~np.isnan(n2c_ref)] - n2c_test[~np.isnan(n2c_test)]
                ))/np.sqrt(np.mean((n2c_ref[~np.isnan(n2c_ref)])**2))

                logging.info('n2clist normalized error from reference: %.3e', err_max)

            except ValueError:
                logging.error('n2clist shape %s does not match reference shape %s',
                              str(n2c_test.shape), str(n2c_ref.shape))
