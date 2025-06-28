# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Function to create a simulated GITL input frame for testing
"""

import numpy as np

from howfsc.model.mode import CoronagraphMode
from howfsc.util.insertinto import insertinto
import howfsc.util.check as check

def sim_gitlframe(cfg, dmlist, fixedbp, peakflux, exptime, crop, lind,
                  cleanrow=1024, cleancol=1024):
    """
    Create simulated GITL frames using the HOWFSC repo's optical model

    This is intended as a faster-running substitute for the higher-fidelity
    PROPER model contained in cgisim.

    Arguments:
     cfg: CoronagraphMode object
     dmlist: list of ndarrays, of the same size as the arrays expected by
      cfg.dmlist objects. These are DM voltages.  This should have the same
      number of DMs as the model.
     fixedbp: this is a fixed bad pixel map for a clean frame, and will be a 2D
      boolean array with the same size as a cleaned frame.
     peakflux: Counts per second at the peak of an unocculted PSF with the
      same CFAM filters as were used to collect the data in im.  Should be
      a real scalar > 0.
     exptime: Exposure time used when collecting the data in in.  Should be a
      real scalar > 0.
     crop: 4-tuple of (lower row, lower col, number of rows,
      number of columns), indicating where in a clean frame a PSF is taken.
      All are integers; the first two must be >= 0 and the second two must be
      > 0.
     lind: integer >= 0 indicating which wavelength channel in cfg to use.
      Must be < the length of cfg.sl_list.

    Keyword Arguments:
     cleanrow: Number of rows in a clean frame.  Integer > 0.  Defaults to
      1024, the number of active area rows on the EXCAM detector; under nominal
      conditions, there should be no reason to use anything else.
     cleancol: Number of columns in clean frame. Integer > 0.  Defaults to
      1024, the number of active area cols on the EXCAM detector; under nominal
      conditions, there should be no reason to use anything else.

    Returns:
     a 2D array of the shape defined by the last two elements of crop

    """

    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('Input model must be a CoronagraphMode object')

    if not isinstance(dmlist, list):
        raise TypeError('dmlist must be a list')
    if len(dmlist) != len(cfg.dmlist):
        raise TypeError('Number of DMs does not match model')
    for index, dm in enumerate(dmlist):
        check.twoD_array(dm, 'dm', TypeError)
        nact = cfg.dmlist[index].registration['nact']
        if dm.shape != (nact, nact):
            raise TypeError('DM dimensions do not match model')
        pass

    check.twoD_array(fixedbp, 'fixedbp', TypeError)
    if fixedbp.shape != (cleanrow, cleancol):
        raise TypeError('fixedbp must be the same size as a cleaned frame')
    if fixedbp.dtype != bool:
        raise TypeError('fixedbp must be boolean')

    check.real_positive_scalar(peakflux, 'peakflux', TypeError)
    check.real_positive_scalar(exptime, 'exptime', TypeError)

    if not isinstance(crop, tuple):
        raise TypeError('crop must be a tuple')
    if len(crop) != 4:
        raise TypeError('crop must be a 4-tuple')
    check.nonnegative_scalar_integer(crop[0], 'crop[0]', TypeError)
    check.nonnegative_scalar_integer(crop[1], 'crop[1]', TypeError)
    check.positive_scalar_integer(crop[2], 'crop[2]', TypeError)
    check.positive_scalar_integer(crop[3], 'crop[3]', TypeError)

    check.nonnegative_scalar_integer(lind, 'lind', TypeError)
    if lind >= len(cfg.sl_list):
        raise ValueError('lind must be < len(cfg.sl_list)')
    check.positive_scalar_integer(cleanrow, 'cleanrow', TypeError)
    check.positive_scalar_integer(cleancol, 'cleancol', TypeError)

    # Compute e-field
    edm = cfg.sl_list[lind].eprop(dmlist)
    ely = cfg.sl_list[lind].proptolyot(edm)
    edh = cfg.sl_list[lind].proptodh(ely)

    # convert to NI and upsize to cleaned-frame size
    idh = insertinto(np.abs(edh)**2, (cleanrow, cleancol))

    # apply fixed bad pixel map
    idh[fixedbp] = np.nan

    # apply normalization to get back to counts from NI
    idh *= peakflux*exptime

    # crop back down to GITL frame size
    nrow = crop[2]
    ncol = crop[3]
    frame = idh[crop[0]:crop[0]+nrow, crop[1]:crop[1]+ncol]

    return frame
