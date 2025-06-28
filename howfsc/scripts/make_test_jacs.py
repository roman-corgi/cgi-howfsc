# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Script to build Jacobians for all five modes available for use with nulltest.py

Stores them by default in a directory (jacdata/) which will be built for the
purpose, and which nulltest.py expects.  Another directory may be provided via
command line argument if that directory can't work.

Calls jactest_mp to do this.  Might be a cleaner way to do, but works for now.
"""

import os
import argparse

import howfsc

if __name__ == "__main__":
    howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))
    jacpath = os.path.join(os.path.dirname(howfscpath), 'jacdata')

    # setup for cmd line args
    ap = argparse.ArgumentParser(prog='python make_test_jacs.py', description="Compute full Jacobians for all five modes that nulltest.py will accept.  These will have the name and location that nulltest.py (and other nulling tests) will expect.")
    ap.add_argument('-p', '--proc', default=1, help="number of processes, defaults to 1", type=int)
    ap.add_argument('-j', '--jacpath', default=jacpath, help="absolute path to write Jacobian files to", type=str)
    args = ap.parse_args()

    if args.proc <= 0:
        raise Exception('Number of processes must be a positive integer')

    mppath = os.path.join(howfscpath, 'scripts', 'jactest_mp.py')
    callstr = 'python3 ' + mppath + ' -p ' + str(args.proc) + ' '

    if not os.path.isdir(args.jacpath):
        os.mkdir(args.jacpath)
        pass

    for mode in ['widefov',
                 'narrowfov',
                 'spectroscopy',
                 'nfov_dm',
                 'nfov_flat']:
        print(mode)
        outfn = mode + '_jac_full.fits'
        outpath = os.path.join(args.jacpath, outfn)
        runstr = callstr + '-o ' + outpath + ' --mode ' + mode
        os.system(runstr)
        pass
