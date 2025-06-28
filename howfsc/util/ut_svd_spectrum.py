# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Test suite for SVD-spectrum processing
"""

import unittest
import os
import copy

import numpy as np

from howfsc.model.mode import CoronagraphMode
from howfsc.util.svd_spectrum import calc_svd_spectrum

cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
               '..', 'model', 'testdata', 'ut', 'ut_smalljac.yaml')
cfg = CoronagraphMode(cfgpath)

class TestCalcSVDSpectrum(unittest.TestCase):
    """
    Tests for analytic tool to compute SVD spectral information from an
    iteration

    x good inputs work
    x bad inputs fail as expected
    x outputs have correct form, including case with a lot of bad pixels
    x one analytic case works as expected
    x bad e-field elements are handled correctly

    """

    def test_success(self):
        """Good inputs complete as expected without errors"""
        npix = int(np.sum([np.sum(sl.dh.e) for sl in cfg.sl_list]))
        ndm = np.sum([initmap.size for initmap in cfg.initmaps])
        nlam = len(cfg.sl_list)
        nrow, ncol = cfg.sl_list[0].dh.e.shape

        rng = np.random.default_rng(44444)
        jac = rng.random((2, ndm, npix))
        e0list = []
        for _ in range(nlam):
            e0 = rng.random((nrow, ncol)) + 1j*rng.random((nrow, ncol))
            e0list.append(e0)
            pass

        calc_svd_spectrum(jac, cfg, e0list)
        pass


    def test_exact(self):
        """analytic results retrieved as expected"""
        npix = 147
        ndm = 50
        tol = 1e-13

        # rjac: 2*npix X ndm
        u = np.eye(2*npix, ndm) # must be unitary
        v = np.eye(ndm) # must be unitary
        s = np.diag(4.0/np.arange(1, ndm+1)) # 4.0 just to make norm get used
        rjac = u @ s @ v.T
        jac = np.zeros((2, ndm, npix))
        jac[0, :, :] = rjac[:npix, :].T
        jac[1, :, :] = rjac[npix:, :].T

        e0 = np.zeros((7, 7), dtype='complex128')
        e1 = np.zeros((7, 7), dtype='complex128')
        i = 24 # 1D index at (3, 3)
        e1[3, 3] = 1 + 0j
        e0list = [e1, e0, e0]

        snorm, iri = calc_svd_spectrum(jac, cfg, e0list)

        # singular values as expected, including norm
        self.assertTrue(snorm[i] == 1/(i+1)**2) # not 4.0!
        # power only in the non-zero efield mode
        self.assertTrue(np.max(np.abs(iri[i] - 1)) < tol)
        for j, this_iri in enumerate(iri):
            if j == i:
                continue
            else:
                self.assertTrue(np.max(np.abs(this_iri)) < tol)
            pass
        pass


    def test_outputs_full_rank(self):
        """outputs have expected form"""
        npix = int(np.sum([np.sum(sl.dh.e) for sl in cfg.sl_list]))
        ndm = np.sum([initmap.size for initmap in cfg.initmaps])
        nlam = len(cfg.sl_list)
        nrow, ncol = cfg.sl_list[0].dh.e.shape

        rng = np.random.default_rng(13579)
        jac = rng.random((2, ndm, npix))
        e0list = []
        for _ in range(nlam):
            e0 = rng.random((nrow, ncol)) + 1j*rng.random((nrow, ncol))
            e0list.append(e0)
            pass

        snorm, iri = calc_svd_spectrum(jac, cfg, e0list)

        # Ordered largest to smallest, largest = 1
        self.assertTrue(snorm[0] == 1)
        self.assertTrue(snorm[0] == np.max(snorm))
        self.assertTrue((np.diff(snorm) <= 0).all())

        # Full-rank tests (true for this specific seed, so OK)
        self.assertTrue(len(snorm) == len(iri))
        self.assertTrue(len(snorm) == min(npix, ndm))

        pass


    def test_outputs_reduced_rank(self):
        """outputs have expected form"""
        npix = int(np.sum([np.sum(sl.dh.e) for sl in cfg.sl_list]))
        ndm = np.sum([initmap.size for initmap in cfg.initmaps])
        nlam = len(cfg.sl_list)
        nrow, ncol = cfg.sl_list[0].dh.e.shape

        rng = np.random.default_rng(13579)
        jac = rng.random((2, ndm, npix))
        e0list = []
        for _ in range(nlam):
            e0 = np.nan*np.ones((nrow, ncol))
            e0list.append(e0)
            pass
        e0list[0][0, 0] = 1

        snorm, iri = calc_svd_spectrum(jac, cfg, e0list)

        # Ordered largest to smallest, largest = 1
        self.assertTrue(snorm[0] == 1)
        self.assertTrue(snorm[0] == np.max(snorm))
        self.assertTrue((np.diff(snorm) <= 0).all())

        # reduced-rank tests
        self.assertTrue(len(snorm) == len(iri))
        self.assertTrue(len(snorm) < min(npix, ndm))
        self.assertTrue(len(snorm) == 2) # one for real, one for imag

        pass


    def test_bad_efield(self):
        """bad efield elements handled correctly"""
        tol = 1e-13

        npix = int(np.sum([np.sum(sl.dh.e) for sl in cfg.sl_list]))
        ndm = np.sum([initmap.size for initmap in cfg.initmaps])
        nrow, ncol = cfg.sl_list[0].dh.e.shape

        rng = np.random.default_rng(24681012)
        jac = rng.random((2, ndm, npix))

        localcfg = copy.deepcopy(cfg)
        localcfg.sl_list[0].dh.e[0, 0] = 0 # make corner excluded
        e0 = rng.random((nrow, ncol)) + 1j*rng.random((nrow, ncol))
        e1 = rng.random((nrow, ncol)) + 1j*rng.random((nrow, ncol))
        e2 = rng.random((nrow, ncol)) + 1j*rng.random((nrow, ncol))
        e0list = [e0, e1, e2]
        jac0 = jac[:, :, 1:].copy()

        snorm, iri = calc_svd_spectrum(jac0, localcfg, e0list)

        # Reinclude corner and suppress it with bad pixel
        localcfg.sl_list[0].dh.e[0, 0] = 1 # make corner included
        e0[0, 0] = np.nan
        e0list = [e0, e1, e2]

        snormmod, irimod = calc_svd_spectrum(jac, localcfg, e0list)

        self.assertTrue(len(snorm) == len(snormmod))
        self.assertTrue(len(iri) == len(irimod))
        self.assertTrue(np.max(np.abs(snorm - snormmod)) < tol)
        self.assertTrue(np.max(np.abs(iri - irimod)) < tol)

        pass


    def test_invalid_jac(self):
        """Invalid inputs caught as expected"""
        npix = int(np.sum([np.sum(sl.dh.e) for sl in cfg.sl_list]))
        ndm = np.sum([initmap.size for initmap in cfg.initmaps])
        nlam = len(cfg.sl_list)
        nrow, ncol = cfg.sl_list[0].dh.e.shape

        rng = np.random.default_rng(11111)
        e0list = []
        for _ in range(nlam):
            e0 = rng.random((nrow, ncol)) + 1j*rng.random((nrow, ncol))
            e0list.append(e0)
            pass

        perrlist = [
            None, 'txt', 1j, 0, # not arrays
            np.ones((2,)), np.ones((2, 2)), # not 3D
            1j*np.ones((2, 2, 2)), # not real
            np.ones((3, ndm, npix)),
            np.ones((2, ndm, ndm)), # incommensurate sizes
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                calc_svd_spectrum(jac=perr, cfg=cfg, e0list=e0list)
            pass
        pass


    def test_invalid_cfg(self):
        """Invalid inputs caught as expected"""
        npix = int(np.sum([np.sum(sl.dh.e) for sl in cfg.sl_list]))
        ndm = np.sum([initmap.size for initmap in cfg.initmaps])
        nlam = len(cfg.sl_list)
        nrow, ncol = cfg.sl_list[0].dh.e.shape

        rng = np.random.default_rng(11111)
        jac = rng.random((2, ndm, npix))
        e0list = []
        for _ in range(nlam):
            e0 = rng.random((nrow, ncol)) + 1j*rng.random((nrow, ncol))
            e0list.append(e0)
            pass

        perrlist = ['cfgfn', None, np.ones((5, 5)), 0]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                calc_svd_spectrum(jac=jac, cfg=perr, e0list=e0list)
            pass
        pass


    def test_invalid_e0list(self):
        """Invalid inputs caught as expected"""
        npix = int(np.sum([np.sum(sl.dh.e) for sl in cfg.sl_list]))
        ndm = np.sum([initmap.size for initmap in cfg.initmaps])

        rng = np.random.default_rng(11111)
        jac = rng.random((2, ndm, npix))

        perrlist = [
            None, 'txt', 1j, 0, # not lists
            [0, 0, 0], [np.ones((2,))]*3, # not lists of 2D arrays
            [np.eye(3), np.eye(4)], # list elements must be same size
            [np.eye(19), np.eye(19)], # incommensurate size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                calc_svd_spectrum(jac=jac, cfg=cfg, e0list=perr)
            pass
        pass


    def test_all_bad(self):
        """Handle case with all e-field inputs flagged as bad"""
        npix = int(np.sum([np.sum(sl.dh.e) for sl in cfg.sl_list]))
        ndm = np.sum([initmap.size for initmap in cfg.initmaps])
        nlam = len(cfg.sl_list)
        nrow, ncol = cfg.sl_list[0].dh.e.shape

        rng = np.random.default_rng(13579)
        jac = rng.random((2, ndm, npix))
        e0list = []
        for _ in range(nlam):
            e0 = np.nan*np.ones((nrow, ncol))
            e0list.append(e0)
            pass

        with self.assertRaises(ValueError):
            calc_svd_spectrum(jac=jac, cfg=cfg, e0list=e0list)
        pass



if __name__ == '__main__':
    unittest.main()
