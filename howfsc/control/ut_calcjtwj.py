# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Test suite for Jacobian product precomputation
"""

import unittest
import os

import numpy as np

from howfsc.control.cs import ControlStrategy
from howfsc.control.calcjacs import calcjacs, generate_ijlist

from howfsc.model.mode import CoronagraphMode

from .calcjtwj import get_jtwj, JTWJMap

# use small array and DM to keep Jacobian manageable
cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'model', 'testdata', 'ut', 'ut_smalljac.yaml')
cfg = CoronagraphMode(cfgpath)
jac = calcjacs(cfg, generate_ijlist(cfg, cfg.initmaps))

# Default control strategy unless we're testing something specific
cstratfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'testdata', 'ut_good_cs.yaml')
cstrat = ControlStrategy(cstratfile)


class TestJTWJMap(unittest.TestCase):
    """
    Tests for the JTWJ map class which handles JTWJ proliferation
    """

    def setUp(self):
        # use values for smalljac: 7x7 images, 4x4 DMs, nlam = 3
        self.nlam = 3
        self.nprobepair = 3
        self.ndm = 2*self.nprobepair + 1
        self.npix = 7
        self.nact = 4

        self.cfg = cfg
        self.jac = jac
        self.cstrat = cstrat

        self.subcroplist = []
        for _ in range(self.nlam):
            self.subcroplist.append((0, 0, self.npix, self.npix))
            pass
        pass


    def test_init_success(self):
        """Good inputs complete without any issues"""
        JTWJMap(cfg=self.cfg,
                jac=self.jac,
                cstrat=self.cstrat,
                subcroplist=self.subcroplist,
        )
        pass


    def test_retrieve_success(self):
        """Good inputs complete without any issues"""
        jm = JTWJMap(cfg=self.cfg,
                     jac=self.jac,
                     cstrat=self.cstrat,
                     subcroplist=self.subcroplist,
        )
        jm.retrieve_jtwj(cstrat=self.cstrat,
                         iteration=1,
                         contrast=1e-7,
        )
        pass


    def test_retrieve_output_size(self):
        """Output data matches expectation"""
        # 2DMs, each self.nact*self.nact
        ndmall = 2*self.nact*self.nact

        jm = JTWJMap(cfg=self.cfg,
                     jac=self.jac,
                     cstrat=self.cstrat,
                     subcroplist=self.subcroplist,
        )
        out = jm.retrieve_jtwj(cstrat=self.cstrat,
                         iteration=1,
                         contrast=1e-7,
        )
        self.assertTrue(out.shape == (ndmall, ndmall))
        pass


    def test_multiple_pixelweights(self):
        """Verify works as expected with more than one weighting file"""
        # use ut_good_cs_value_3, which will provide a weighting matrix of all
        # threes on iteration 5, contrast [1e-6, 1e-5), and all ones in all
        # other cases

        tol = 1e-13

        cstratfile3 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'testdata', 'ut_good_cs_value_3.yaml')
        cstrat3 = ControlStrategy(cstratfile3)

        jm = JTWJMap(cfg=self.cfg,
                     jac=self.jac,
                     cstrat=cstrat3,
                     subcroplist=self.subcroplist,
        )

        self.assertTrue(len(jm.jtwjs) == 2) # even though file has 5 entries

        # hit all four directions
        jtwj1a = jm.retrieve_jtwj(cstrat3, iteration=5, contrast=0.9e-6)
        jtwj1b = jm.retrieve_jtwj(cstrat3, iteration=5, contrast=1.1e-5)
        jtwj1c = jm.retrieve_jtwj(cstrat3, iteration=4, contrast=1e-6)
        jtwj1d = jm.retrieve_jtwj(cstrat3, iteration=6, contrast=1e-6)
        jtwj3 = jm.retrieve_jtwj(cstrat3, iteration=5, contrast=1e-6)

        # Weighting factor is applied as a X^2
        self.assertTrue(np.max(np.abs(jtwj3 - 3**2*jtwj1a)) < tol)
        self.assertTrue(np.max(np.abs(jtwj3 - 3**2*jtwj1b)) < tol)
        self.assertTrue(np.max(np.abs(jtwj3 - 3**2*jtwj1c)) < tol)
        self.assertTrue(np.max(np.abs(jtwj3 - 3**2*jtwj1d)) < tol)

        pass

    #---------------
    # Failure tests
    #---------------

    def test_init_invalid_cfg(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cstrat, # wrong type
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                JTWJMap(cfg=perr,
                        jac=self.jac,
                        cstrat=self.cstrat,
                        subcroplist=self.subcroplist,
                )
            pass
        pass


    def test_init_invalid_jac(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    np.ones((2,)), np.ones((self.nact, self.npix)), #wrong Ndim
                    np.ones((1, self.nact, self.npix)), # wrong dim size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                JTWJMap(cfg=self.cfg,
                        jac=perr,
                        cstrat=self.cstrat,
                        subcroplist=self.subcroplist,
                )
            pass
        pass


    def test_init_invalid_cstrat(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                JTWJMap(cfg=self.cfg,
                        jac=self.jac,
                        cstrat=perr,
                        subcroplist=self.subcroplist,
                )
            pass
        pass


    def test_init_invalid_subcroplist(self):
        """Invalid inputs caught as expected"""
        N = self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [(0, 0, self.npix, self.npix, 0)]*N, # wrong in size
                    [(0, 0, self.npix, self.npix)]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                JTWJMap(cfg=self.cfg,
                        jac=self.jac,
                        cstrat=self.cstrat,
                        subcroplist=perr,
                )
            pass
        pass


    def test_retrieve_invalid_cstrat(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    ]
        jm = JTWJMap(cfg=self.cfg,
                     jac=self.jac,
                     cstrat=self.cstrat,
                     subcroplist=self.subcroplist,
        )

        for perr in perrlist:
            with self.assertRaises(TypeError):
                jm.retrieve_jtwj(cstrat=perr,
                                 iteration=1,
                                 contrast=1e-7,
                )
            pass
        pass


    def test_retrieve_invalid_iteration(self):
        """Invalid inputs caught as expected"""
        perrlist = [0, -1, 1.5, 1j, None, 'txt', (5,)]
        jm = JTWJMap(cfg=self.cfg,
                     jac=self.jac,
                     cstrat=self.cstrat,
                     subcroplist=self.subcroplist,
        )

        for perr in perrlist:
            with self.assertRaises(TypeError):
                jm.retrieve_jtwj(cstrat=self.cstrat,
                                 iteration=perr,
                                 contrast=1e-7,
                )
            pass
        pass


    def test_retrieve_invalid_contrast(self):
        """Invalid inputs caught as expected"""
        perrlist = [-1, 1j, None, 'txt', (5,)]
        jm = JTWJMap(cfg=self.cfg,
                     jac=self.jac,
                     cstrat=self.cstrat,
                     subcroplist=self.subcroplist,
        )

        for perr in perrlist:
            with self.assertRaises(TypeError):
                jm.retrieve_jtwj(cstrat=self.cstrat,
                                 iteration=1,
                                 contrast=perr,
                )
            pass
        pass


    def test_retrieve_wrong_cs_in(self):
        """
        Check that giving the wrong control strategy as input will be caught
        if the relevant name is not present.

        This check is not foolproof.  If you have two control strategies that
        use the same set of pixelweights files defined over different regions,
        the checks in this tool will not catch it.
        """

        cstratfile3 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'testdata', 'ut_good_cs_value_3.yaml')
        cstrat3 = ControlStrategy(cstratfile3)

        jm = JTWJMap(cfg=self.cfg,
                     jac=self.jac,
                     cstrat=self.cstrat,
                     subcroplist=self.subcroplist,
        )
        with self.assertRaises(ValueError):
            # in this particular test file, iteration = 5 and contrast in
            # [1e-6, 1e-5) will activate a different file
            jm.retrieve_jtwj(cstrat=cstrat3,
                             iteration=5,
                             contrast=1e-6,
            )
        pass



class TestGetJTWJ(unittest.TestCase):
    """
    Test function to compute expensive intermediate terms
    """

    def setUp(self):
        self.ndm = 5
        self.npix = 7

        self.we0 = np.ones((self.npix,))
        self.jac = np.ones((2, self.ndm, self.npix))
        pass


    def test_success(self):
        """
        Nothing breaks when run with reasonable inputs
        """
        get_jtwj(self.jac, self.we0)
        pass


    def test_output_size(self):
        """
        Check output size matches documentation
        """
        jtwj = get_jtwj(self.jac, self.we0)
        self.assertTrue(jtwj.shape == (self.ndm, self.ndm))
        pass


    def test_jac_invalid(self):
        """
        Fails as expected given invalid input
        """
        badjaclist = [np.ones((2, self.ndm, 1)),
                      np.ones((1, self.ndm, self.npix)),
                      np.ones((1, 1))]

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None] + badjaclist:
            with self.assertRaises(TypeError):
                get_jtwj(perr, self.we0)
                pass
            pass
        pass


    def test_we0_invalid(self):
        """
        Fails as expected given invalid input
        """
        badwe0 = np.zeros((self.npix+1,)) # wrong size

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None, badwe0]:
            with self.assertRaises(TypeError):
                get_jtwj(self.jac, perr)
                pass
            pass
        pass

    def test_almost_all_bad(self):
        """
        Test specific known case where only one row is good
        """
        val = 5 # use != 1 so we check the square
        onegood = np.zeros_like(self.we0)
        onegood[-1] = val

        jtwj = get_jtwj(self.jac, onegood)

        lastrjac = self.jac[0, :, -1]
        lastijac = self.jac[1, :, -1]
        target = val**2*(np.outer(lastrjac, lastrjac) +
                         np.outer(lastijac, lastijac))

        self.assertTrue((jtwj == target).all())
        pass











if __name__ == '__main__':
    unittest.main()
