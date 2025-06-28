# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for HOWFSC precomputation
"""

import unittest
import os
import logging
import sys

import numpy as np

from howfsc.model.mode import CoronagraphMode

from howfsc.control.calcjtwj import JTWJMap
from howfsc.control.cs import ControlStrategy

from .precomp import howfsc_precomputation, valid_jacmethod

# Keep logger spam out of unit test results
# note that smalljac throws warnings with fast (but not normal)

if sys.platform.startswith('win'):
    logging.basicConfig(filename='NUL')
else:
    logging.basicConfig(filename='/dev/null')

# use small array and DM to keep Jacobian manageable
cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'model', 'testdata', 'ut', 'ut_smalljac.yaml')
cfg = CoronagraphMode(cfgpath)
npix = 7 # for smalljac

# Default control strategy unless we're testing something specific
cstratfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'control', 'testdata', 'ut_good_cs.yaml')
cstrat = ControlStrategy(cstratfile)

class TestHOWFSCPrecomputation(unittest.TestCase):
    """
    Tests for precomputation functionality

    This is function is a wrapper on calcjacs and calcjtwj.  Let them figure
    out if the numbers are the right numbers.  We'll check interfaces on this
    one.

    """

    def setUp(self):
        self.cfg = cfg
        self.nlam = len(cfg.sl_list)

        self.dmset_list = []
        self.ndmact = 0
        for d in cfg.dmlist:
            vmax = d.dmvobj.vmax
            vmin = d.dmvobj.vmin
            nact = d.registration['nact']
            self.dmset_list.append((vmax + vmin)/2*np.ones((nact, nact)))
            self.ndmact += nact**2
            pass

        self.ndhpix = 0
        for sl in cfg.sl_list:
            self.ndhpix += len(sl.dh_inds)
            pass

        self.cstrat = cstrat

        self.npix = npix
        self.subcroplist = []
        for _ in range(self.nlam):
            self.subcroplist.append((0, 0, self.npix, self.npix))
            pass

        pass


    def test_success(self):
        """Good inputs succeed without error"""
        howfsc_precomputation(
            cfg=self.cfg,
            dmset_list=self.dmset_list,
            cstrat=self.cstrat,
            subcroplist=self.subcroplist,
        )
        pass


    def test_fast_is_default_jacmethod(self):
        """Good inputs succeed without error"""
        jac0, _, _ = howfsc_precomputation(
            cfg=self.cfg,
            dmset_list=self.dmset_list,
            cstrat=self.cstrat,
            subcroplist=self.subcroplist,
        )

        jac1, _, _ = howfsc_precomputation(
            cfg=self.cfg,
            dmset_list=self.dmset_list,
            cstrat=self.cstrat,
            subcroplist=self.subcroplist,
            jacmethod='fast',
        )

        # should be identical if the default method is 'fast', as documented
        self.assertTrue((jac0 == jac1).all())

        pass


    def test_true_is_default_do_n2clist(self):
        """Good inputs succeed without error"""
        _, _, n2clist0 = howfsc_precomputation(
            cfg=self.cfg,
            dmset_list=self.dmset_list,
            cstrat=self.cstrat,
            subcroplist=self.subcroplist,
        )

        _, _, n2clist1 = howfsc_precomputation(
            cfg=self.cfg,
            dmset_list=self.dmset_list,
            cstrat=self.cstrat,
            subcroplist=self.subcroplist,
            do_n2clist=True,
        )

        _, _, n2clist2 = howfsc_precomputation(
            cfg=self.cfg,
            dmset_list=self.dmset_list,
            cstrat=self.cstrat,
            subcroplist=self.subcroplist,
            do_n2clist=False,
        )

        # Check ones outputs
        self.assertTrue(len(n2clist0) == len(n2clist2))
        for n2c in n2clist2:
            self.assertTrue((n2c == np.ones((self.npix, self.npix))).all())
            pass
        pass

        # Compare default/non-default outputs
        self.assertTrue(len(n2clist0) == len(n2clist1))
        for ind in range(len(n2clist0)):
            n0 = n2clist0[ind][~np.isnan(n2clist0[ind])]
            n1 = n2clist1[ind][~np.isnan(n2clist1[ind])]
            self.assertTrue((n0 == n1).all())
            pass
        pass


    def test_all_other_jacmethods_work(self):
        """Good inputs succeed without error"""
        for jacmethod in valid_jacmethod:
            if jacmethod == 'fast':
                continue
            else:
                howfsc_precomputation(
                    cfg=self.cfg,
                    dmset_list=self.dmset_list,
                    cstrat=self.cstrat,
                    subcroplist=self.subcroplist,
                    jacmethod=jacmethod,
                )
                pass
            pass
        pass


    def test_output_size(self):
        """Check outputs match spec"""
        jac, jtwj_map, n2clist = howfsc_precomputation(
            cfg=self.cfg,
            dmset_list=self.dmset_list,
            cstrat=self.cstrat,
            subcroplist=self.subcroplist,
        )

        self.assertTrue(jac.shape == (2, self.ndmact, self.ndhpix))
        self.assertTrue(isinstance(jtwj_map, JTWJMap))
        self.assertTrue(len(n2clist) == len(cfg.sl_list))
        for index, n2c in enumerate(n2clist):
            self.assertTrue(n2c.shape == (self.subcroplist[index][2],
                                          self.subcroplist[index][3]))
            pass
        pass


    def test_output_size_don2c_false(self):
        """Check outputs match spec"""
        _, _, n2clist = howfsc_precomputation(
            cfg=self.cfg,
            dmset_list=self.dmset_list,
            cstrat=self.cstrat,
            subcroplist=self.subcroplist,
            do_n2clist=False,
        )

        self.assertTrue(len(n2clist) == len(cfg.sl_list))
        for index, n2c in enumerate(n2clist):
            self.assertTrue(n2c.shape == (self.subcroplist[index][2],
                                          self.subcroplist[index][3]))
            pass
        pass


    def test_num_process_num_threads(self):
        """Check various combinations of num_process and num_threads for jac
           work and give same result
        """

        list_num_process = [None, 1, 4, 0]
        list_num_threads = [None, 1, 4]

        list_jac = []
        for num_process in list_num_process:
            for num_threads in list_num_threads:
                #     return jac, jtwj_map, n2clist
                jac, _, _ = howfsc_precomputation(cfg=self.cfg,
                                                  dmset_list=self.dmset_list,
                                                  cstrat=self.cstrat,
                                                  subcroplist=self.subcroplist,
                                                  num_process=num_process,
                                                  num_threads=num_threads,
                                                  do_n2clist=False,
                )
                list_jac.append(jac) # these are small_jac

        # are they all the same (do we need to allow for double precision eps?)
        jac0 = list_jac[0]
        for jac in list_jac[1:]:
            self.assertTrue((jac == jac0).all())

    #---------------
    # Failure tests
    #---------------

    def test_invalid_cfg(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cstrat, # wrong type
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                howfsc_precomputation(
                    cfg=perr,
                    dmset_list=self.dmset_list,
                    cstrat=self.cstrat,
                    subcroplist=self.subcroplist,
                )
            pass
        pass


    def test_invalid_dmset_list(self):
        """Invalid inputs caught as expected"""
        N = len(self.cfg.dmlist)
        nact0 = self.cfg.dmlist[0].registration['nact']
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [np.ones((nact0+1, nact0+1))]*N, # wrong in size
                    [d.dmvobj.flatmap for d in self.cfg.dmlist]
                     + [np.ones((nact0, nact0))], # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                howfsc_precomputation(
                    cfg=self,
                    dmset_list=perr,
                    cstrat=self.cstrat,
                    subcroplist=self.subcroplist,
                )
            pass
        pass



    def test_invalid_cstrat(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                howfsc_precomputation(
                    cfg=self.cfg,
                    dmset_list=self.dmset_list,
                    cstrat=perr,
                    subcroplist=self.subcroplist,
                )
            pass
        pass


    def test_invalid_subcroplist(self):
        """Invalid inputs caught as expected"""
        N = self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [(0, 0, self.npix, self.npix, 0)]*N, # wrong in size
                    [(0, 0, self.npix, self.npix)]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                howfsc_precomputation(
                    cfg=self.cfg,
                    dmset_list=self.dmset_list,
                    cstrat=self.cstrat,
                    subcroplist=perr,
                )
            pass
        pass


    def test_invalid_jacmethod_type(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, None, self.cfg, np.eye(3), # wrong type
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                howfsc_precomputation(
                    cfg=self.cfg,
                    dmset_list=self.dmset_list,
                    cstrat=self.cstrat,
                    subcroplist=self.subcroplist,
                    jacmethod=perr,
                )
            pass
        pass


    def test_invalid_jacmethod_value(self):
        """Invalid inputs caught as expected"""
        perrlist = ['not_a_valid_jacmethod'] # right type, wrong value

        for perr in perrlist:
            with self.assertRaises(ValueError): # <-- not a TypeError
                howfsc_precomputation(
                    cfg=self.cfg,
                    dmset_list=self.dmset_list,
                    cstrat=self.cstrat,
                    subcroplist=self.subcroplist,
                    jacmethod=perr,
                )
            pass
        pass


    def test_invalid_do_n2clist(self):
        """Invalid inputs caught as expected"""
        perrlist = [None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                howfsc_precomputation(
                    cfg=self.cfg,
                    dmset_list=self.dmset_list,
                    cstrat=self.cstrat,
                    subcroplist=self.subcroplist,
                    do_n2clist=perr,
                )
            pass
        pass

    def test_invalid_num_process(self):
        """Test invalid values for num_process and num_threads """

        list_invalid_num_process = [-1, 'a', [1, 2, 3], 3.4]

        for nperr in list_invalid_num_process:
            with self.assertRaises(TypeError):
                howfsc_precomputation(
                    cfg=self.cfg,
                    dmset_list=self.dmset_list,
                    cstrat=self.cstrat,
                    subcroplist=self.subcroplist,
                    num_process=nperr,
                )

    def test_invalid_num_threads(self):
        """Test invalid values for num_process and num_threads """

        list_invalid_num_threads = [0, -1, 'a', [1, 2, 3], 3.4]

        for nperr in list_invalid_num_threads:
            with self.assertRaises(TypeError):
                howfsc_precomputation(
                    cfg=self.cfg,
                    dmset_list=self.dmset_list,
                    cstrat=self.cstrat,
                    subcroplist=self.subcroplist,
                    num_threads=nperr,
                )


if __name__ == '__main__':
    unittest.main()
