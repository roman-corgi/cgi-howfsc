# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for code to get probe phase estimates from the model.
"""
import unittest
import os

import numpy as np

import howfsc
from howfsc.model.mode import CoronagraphMode
from .probephase import probe_ap

specpath = os.path.join(os.path.dirname(
             os.path.abspath(howfsc.__file__)),
             'model', 'testdata', 'spectroscopy', 'spectroscopy.yaml')

nfovpath = os.path.join(os.path.dirname(
                os.path.abspath(howfsc.__file__)),
                'model', 'testdata', 'narrowfov', 'narrowfov_dm.yaml')

class TestProbePhase(unittest.TestCase):
    """
    x Test success
    x test cfg is right type
    x test dms are right shape
    x test outputs have right length and shape
    x test multiple numbers of wavelengths (spec vs nfov)
    check outputs are numerically correct
    """

    def setUp(self):
        # Make a very simple delta-dm
        self.ddm = np.zeros((48, 48))
        self.ddm[16, 16] = 1
        self.lind = 0

        pass


    def test_success(self):
        """Good inputs do not throw an error"""
        cfg_nfov = CoronagraphMode(nfovpath)
        dm1plus = cfg_nfov.initmaps[0] + self.ddm
        dm1minus = cfg_nfov.initmaps[0] - self.ddm
        dm2 = cfg_nfov.initmaps[1]
        probe_ap(cfg_nfov,
                   dm1plus,
                   dm1minus,
                   dm2,
                   self.lind)
        pass


    def test_invalid_cfg(self):
        """Invalid inputs caught as expected"""
        cfg_nfov = CoronagraphMode(nfovpath)
        dm1plus = cfg_nfov.initmaps[0] + self.ddm
        dm1minus = cfg_nfov.initmaps[0] - self.ddm
        dm2 = cfg_nfov.initmaps[1]

        for perr in [0, (5, 5), 'txt', None, dm2]:
            with self.assertRaises(TypeError):
                probe_ap(cfg=perr,
                           dm1plus=dm1plus,
                           dm1minus=dm1minus,
                           dm2=dm2,
                           lind=self.lind)
                pass
            pass
        pass


    def test_invalid_dms_type(self):
        """Invalid inputs caught as expected"""
        cfg_nfov = CoronagraphMode(nfovpath)
        dm1plus = cfg_nfov.initmaps[0] + self.ddm
        dm1minus = cfg_nfov.initmaps[0] - self.ddm
        dm2 = cfg_nfov.initmaps[1]

        for perr in [0, (5, 5), 'txt', None]:
            with self.assertRaises(TypeError):
                probe_ap(cfg=cfg_nfov,
                           dm1plus=perr,
                           dm1minus=dm1minus,
                           dm2=dm2,
                           lind=self.lind)
                pass
            pass

        for perr in [0, (5, 5), 'txt', None]:
            with self.assertRaises(TypeError):
                probe_ap(cfg=cfg_nfov,
                           dm1plus=dm1plus,
                           dm1minus=perr,
                           dm2=dm2,
                           lind=self.lind)
                pass
            pass

        for perr in [0, (5, 5), 'txt', None]:
            with self.assertRaises(TypeError):
                probe_ap(cfg=cfg_nfov,
                           dm1plus=dm1plus,
                           dm1minus=dm1minus,
                           dm2=perr,
                           lind=self.lind)
                pass
            pass
        pass


    def test_invalid_dms_size(self):
        """Invalid inputs caught as expected"""
        cfg_nfov = CoronagraphMode(nfovpath)
        dm1plus = cfg_nfov.initmaps[0] + self.ddm
        dm1minus = cfg_nfov.initmaps[0] - self.ddm
        dm2 = cfg_nfov.initmaps[1]

        dmshape = dm2.shape
        for perr in [np.zeros((dmshape[0]+1, dmshape[1])),
                     np.zeros((dmshape[0], dmshape[1]+1)),
                     np.zeros((dmshape[0]+1, dmshape[1]+1)),
                     ]:
            with self.assertRaises(TypeError):
                probe_ap(cfg=cfg_nfov,
                           dm1plus=perr,
                           dm1minus=dm1minus,
                           dm2=dm2,
                           lind=self.lind)
                pass
            pass

        for perr in [np.zeros((dmshape[0]+1, dmshape[1])),
                     np.zeros((dmshape[0], dmshape[1]+1)),
                     np.zeros((dmshape[0]+1, dmshape[1]+1)),
                     ]:
            with self.assertRaises(TypeError):
                probe_ap(cfg=cfg_nfov,
                           dm1plus=dm1plus,
                           dm1minus=perr,
                           dm2=dm2,
                           lind=self.lind)
                pass
            pass


        for perr in [np.zeros((dmshape[0]+1, dmshape[1])),
                     np.zeros((dmshape[0], dmshape[1]+1)),
                     np.zeros((dmshape[0]+1, dmshape[1]+1)),
                     ]:
            with self.assertRaises(TypeError):
                probe_ap(cfg=cfg_nfov,
                           dm1plus=dm1plus,
                           dm1minus=dm1minus,
                           dm2=perr,
                           lind=self.lind)
                pass
            pass
        pass


    def test_invalid_lind(self):
        """Verify invalid inputs caught"""
        cfg_nfov = CoronagraphMode(nfovpath)
        dm1plus = cfg_nfov.initmaps[0] + self.ddm
        dm1minus = cfg_nfov.initmaps[0] - self.ddm
        dm2 = cfg_nfov.initmaps[1]

        for perr in [1.5, 1j, None, 'txt', (0,), np.zeros((1,)), # not integer
                     -1, len(cfg_nfov.sl_list)]: # not in range
            with self.assertRaises(TypeError):
                probe_ap(cfg_nfov,
                         dm1plus,
                         dm1minus,
                         dm2,
                         perr)
            pass
        pass


    def test_correct_output_dims(self):
        """Check output dimension match documentation"""

        cfg = CoronagraphMode(nfovpath)
        dm1plus = cfg.initmaps[0] + self.ddm
        dm1minus = cfg.initmaps[0] - self.ddm
        dm2 = cfg.initmaps[1]
        amp, phase = probe_ap(cfg,
                              dm1plus,
                              dm1minus,
                              dm2,
                              self.lind)
        self.assertTrue(amp.shape == cfg.sl_list[self.lind].dh.e.shape)
        self.assertTrue(phase.shape == cfg.sl_list[self.lind].dh.e.shape)
        pass


    def test_probe_phase_measurement(self):
        """Check that the probe phase matches expectation"""

        # Make I0, I+, I- from exact
        # calc dp angle as above
        # calc dp abs from sqrt((I+ + I-)/2 - I0)
        # Get c[E0] from mode
        # calc I from |E0 +/- i*dp|^2
        # compare (averaging + nonlinearity will make this inexact)

        tol = 1e-6

        cfg_nfov = CoronagraphMode(nfovpath)

        cfg = cfg_nfov
        dm1 = cfg.initmaps[0]
        dm1plus = cfg.initmaps[0] + self.ddm
        dm1minus = cfg.initmaps[0] - self.ddm
        dm2 = cfg.initmaps[1]
        amp, phase = probe_ap(cfg,
                              dm1plus,
                              dm1minus,
                              dm2,
                              self.lind)

        sl = cfg.sl_list[self.lind]

        edm0 = sl.eprop([dm1, dm2])
        ely = sl.proptolyot(edm0)
        edh0 = sl.proptodh(ely)

        edm0 = sl.eprop([dm1plus, dm2])
        ely = sl.proptolyot(edm0)
        edh_plus = sl.proptodh(ely)
        Iplus = np.abs(edh_plus)**2

        edm0 = sl.eprop([dm1minus, dm2])
        ely = sl.proptolyot(edm0)
        edh_minus = sl.proptodh(ely)
        Iminus = np.abs(edh_minus)**2

        dp = amp*np.exp(1j*phase)
        dpIplus = np.abs(edh0 + 1j*dp)**2
        dpIminus = np.abs(edh0 - 1j*dp)**2

        self.assertTrue(np.max(np.abs(dpIplus - Iplus)) < tol)
        self.assertTrue(np.max(np.abs(dpIminus - Iminus)) < tol)
        pass



if __name__ == '__main__':
    unittest.main()
