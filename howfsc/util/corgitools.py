import os

import astropy.io.fits as pyfits
import matplotlib.pylab as plt
import numpy as np

from howfsc.util.gitl_tools import param_order_to_list


def save_outputs(fileout, cfg, camlist, framelistlist, otherlist, measured_c):
    """
    Save the outputs of a HOWFSC run to disk.

    Arguments:
        fileout : string, path to the main output file by the nullingtest,
        containing all images of the last iteration.  The directory
        containing this file will be used to create subdirectories for each
        iteration's outputs.
        cfg: CoronagraphMode object used in the run.
        camlist : list of (camera, param_order) tuples, one per iteration.
        framelistlist : list of lists of 2D arrays, one list per iteration,
        one array per wavelength channel.
        otherlist : list of dictionaries, one per iteration.
        measured_c : list of floats, one per iteration, giving the measured
        contrast in the dark hole at that iteration.
    """
    outpath = os.path.dirname(fileout)

    # Plot measured_c vs iteration
    plt.figure()
    plt.plot(np.arange(len(measured_c)) + 1, measured_c, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Measured Contrast')
    plt.savefig(os.path.join(outpath, "contrast_vs_iteration.pdf"))
    plt.close()

    # Save measured_c to a csv file
    np.savetxt(os.path.join(outpath, "measured_contrast.csv"), np.array(measured_c), delimiter=",", header="Measured Contrast", comments="")

    # Create one subdirectory per iteration
    for i in range(len(framelistlist)):
        iterpath = os.path.join(outpath, f"iteration_{i+1:04d}")
        if not os.path.exists(iterpath):
            os.makedirs(iterpath)

    # Unprobed and probed images, in all wavelengths
    for i, flist in enumerate(framelistlist):
        hdr = pyfits.Header()
        hdr['NLAM'] = len(cfg.sl_list)
        hdr['ITER'] = i + 1
        prim = pyfits.PrimaryHDU(header=hdr)
        img = pyfits.ImageHDU(flist)
        prev = pyfits.ImageHDU(param_order_to_list(camlist[i][1]))
        hdul = pyfits.HDUList([prim, img, prev])
        fnout = os.path.join(outpath, f"iteration_{i+1:04d}", f"images.fits")
        hdul.writeto(fnout, overwrite=True)

    # Estimated E-fields at each wavelength
    efields = []
    for i, oitem in enumerate(otherlist):
        for n in range(len(cfg.sl_list)):
            efields.append(np.real(oitem[n]['meas_efield']))
            efields.append(np.imag(oitem[n]['meas_efield']))
        hdr = pyfits.Header()
        hdr['NLAM'] = len(cfg.sl_list)
        prim = pyfits.PrimaryHDU(header=hdr)
        img = pyfits.ImageHDU(efields)
        hdul = pyfits.HDUList([prim, img])
        fn, fe = os.path.splitext(fileout)
        fnout = os.path.join(outpath, f"iteration_{i+1:04d}", f"efield_estimations.fits")
        hdul.writeto(fnout, overwrite=True)
