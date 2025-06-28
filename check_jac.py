# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Quick test script to verify the Jacobian output doesn't change"""

import os

import numpy as np
import astropy.io.fits as pyfits

if __name__ == "__main__":

    pf = pyfits.getdata

    print('Single-process')
    print('==============')

    os.system('rm -f jac_5actoutput_peak.fits')
    os.system('python3.7 jactest.py -n 5 -o jac_5actoutput_peak.fits')

    print('---------------')
    print('Peak Jacobian')
    j0p = pf('jac_5actoutput_peak.fits')
    uj0p = pf('ut_jac_5actoutput_peak.fits')
    print('Current jac: ' + str(j0p.max()))
    print('Unit test jac: ' + str(uj0p.max()))
    print('Max difference (should be 0 or numerical noise): '
          + str(np.max(j0p-uj0p)))

    print('Multiprocessed')
    print('==============')

    os.system('rm -f jac_5actoutput_peak.fits')
    os.system('python3.7 jactest_mp.py -n 5 -p 5 -o jac_5actoutput_peak.fits')

    print('---------------')
    print('Peak Jacobian')
    j0p = pf('jac_5actoutput_peak.fits')
    uj0p = pf('ut_jac_5actoutput_peak.fits')
    print('Current jac: ' + str(j0p.max()))
    print('Unit test jac: ' + str(uj0p.max()))
    print('Max difference (should be 0 or numerical noise): '
          + str(np.max(j0p-uj0p)))

    # Clean up after yourself
    os.system('rm -f jac_5actoutput_peak.fits')
