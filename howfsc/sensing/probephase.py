# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Function to compute phases for wavefront sensing
"""

import numpy as np

from howfsc.model.mode import CoronagraphMode
import howfsc.util.check as check

def probe_ap(cfg, dm1plus, dm1minus, dm2, lind):
    """
    Create the model-based probe phase and amp required by pairwise probing

    Use convention in Give'on, Kern and Shaklan 2011.

    Procedure:
    - load up a model, using index lind
    - compute the field with + probe settings (E+)
    - compute the field with - probe settings (E-)
    - compute deltaP = (E+ - E-)/2j
     = Turns out we don't need the 0 probe case here.
    - compute probe phase with angle(deltaP) and amp with abs(deltaP)
    See probephase.pdf for derivation

    Arguments:
     cfg: a CoronagraphMode object (optical model)
     dm1plus: a 2D DM array with size consistent with the optical model,
      representing the positive-probe DM1 absolute setting.  Units are volts.
     dm1minus: a 2D DM array with size consistent with the optical model,
      representing the negative-probe DM1 absolute setting.  Units. are volts.
     dm2: a 2D DM array with size consistent with the optical model,
      representing the DM2 absolute setting. (DM2 is not probed.)  Units are
      volts.
     lind: index of which of the cfg_list wavelengths to use. Must be an
      integer >= 0 and < len(cfg.sl_list).

    Returns:
     two 2D arrays, the first of amplitudes and the second of phases. Each
      array will be of size cfg.sl_list[lind].dh.e.shape.

    """

    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')
    check.twoD_array(dm1plus, 'dm1plus', TypeError)
    check.twoD_array(dm1minus, 'dm1minus', TypeError)
    check.twoD_array(dm2, 'dm2', TypeError)
    if dm1plus.shape != (cfg.dmlist[0].registration['nact'],
                         cfg.dmlist[0].registration['nact']):
        raise TypeError('dm1plus shape is inconsistent with cfg')
    if dm1minus.shape != (cfg.dmlist[0].registration['nact'],
                          cfg.dmlist[0].registration['nact']):
        raise TypeError('dm1minus shape is inconsistent with cfg')
    if dm2.shape != (cfg.dmlist[1].registration['nact'],
                     cfg.dmlist[1].registration['nact']):
        raise TypeError('dm2 shape is inconsistent with cfg')
    check.real_nonnegative_scalar(lind, 'lind', TypeError)
    if lind >= len(cfg.sl_list):
        raise TypeError('lind must be < len(cfg.sl_list)')

    # select wavelength
    sl = cfg.sl_list[lind]

    # +probe
    edm0 = sl.eprop([dm1plus, dm2])
    ely = sl.proptolyot(edm0)
    edh_plus = sl.proptodh(ely)

    # -probe
    edm0 = sl.eprop([dm1minus, dm2])
    ely = sl.proptolyot(edm0)
    edh_minus = sl.proptodh(ely)

    deltap = (edh_plus - edh_minus)/2j
    amp = np.abs(deltap)
    phase = np.angle(deltap)

    return amp, phase
