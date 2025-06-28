# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for the control strategy checkout
"""
import unittest
from unittest.mock import patch
import os
import copy

import numpy as np
import scipy.sparse

from howfsc.model.mode import CoronagraphMode
from howfsc.util.insertinto import insertinto
from .cs import ControlStrategy, get_wdm, get_we0

cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'model', 'testdata', 'widefov', 'widefov.yaml')
cfg = CoronagraphMode(cfgpath)

class TestControlStrategy(unittest.TestCase):
    """
    Tests if the control strategy class validates as expected

    To add:
    bad inputs wdm
    successful readback wdm
    right value wdm

    """

    def setUp(self):
        localpath = os.path.dirname(os.path.abspath(__file__))
        self.fn = os.path.join(localpath, 'testdata', 'ut_good_cs.yaml')
        pass


    def test_success(self):
        """Class instantiates successful with valid input"""
        ControlStrategy(self.fn)
        pass


    def test_success_iteration_contrast(self):
        """
        All seven lookup tables which use iteration and contrast read back
        successfully for a few sample values
        """
        cstr = ControlStrategy(self.fn)

        # test end/not-end
        iclist = [(1, 0),
                  (1, 1e-6),
                  (5, 1e-6),
                  (5, 0),
                  ]

        for iteration, contrast in iclist:
            cstr.get_regularization(iteration, contrast)
            cstr.get_pixelweights(iteration, contrast)
            cstr.get_pixelweights_fn(iteration, contrast)
            cstr.get_dmmultgain(iteration, contrast)
            cstr.get_unprobedsnr(iteration, contrast)
            cstr.get_probedsnr(iteration, contrast)
            cstr.get_probeheight(iteration, contrast)
            pass
        pass


    def test_right_number_iteration_contrast(self):
        """
        Three lookup tables which use iteration and contrast read back
        the correct value from a YAML which sets that value only in a specific
        region
        """
        localpath = os.path.dirname(os.path.abspath(__file__))
        fn3 = os.path.join(localpath, 'testdata', 'ut_good_cs_value_3.yaml')
        cstr3 = ControlStrategy(fn3)

        iteration = 5
        contrast = 1e-6

        self.assertTrue(cstr3.get_regularization(iteration, contrast) == 3)
        self.assertTrue(cstr3.get_dmmultgain(iteration, contrast) == 3)
        self.assertTrue(cstr3.get_probeheight(iteration, contrast) == 3)
        self.assertTrue(cstr3.get_unprobedsnr(iteration, contrast) == 3)
        self.assertTrue(cstr3.get_probedsnr(iteration, contrast) == 3)

        pwlist = cstr3.get_pixelweights(iteration, contrast)
        for pw in pwlist:
            self.assertTrue((pw == 3).all())
        pass



    def test_invalid_iteration(self):
        """Invalid inputs caught"""
        cstr = ControlStrategy(self.fn)

        xlist = [0, -1, 1.5, 1j, None, 'txt', (5,)]
        contrast = 1e-6

        for x in xlist:
            with self.assertRaises(TypeError):
                cstr.get_regularization(x, contrast)
            with self.assertRaises(TypeError):
                cstr.get_pixelweights(x, contrast)
            with self.assertRaises(TypeError):
                cstr.get_pixelweights_fn(x, contrast)
            with self.assertRaises(TypeError):
                cstr.get_dmmultgain(x, contrast)
            with self.assertRaises(TypeError):
                cstr.get_unprobedsnr(x, contrast)
            with self.assertRaises(TypeError):
                cstr.get_probedsnr(x, contrast)
            with self.assertRaises(TypeError):
                cstr.get_probeheight(x, contrast)
            pass
        pass


    def test_invalid_contrast(self):
        """Invalid inputs caught"""
        cstr = ControlStrategy(self.fn)

        xlist = [-1, 1j, None, 'txt', (5,)]
        iteration = 5

        for x in xlist:
            with self.assertRaises(TypeError):
                cstr.get_regularization(iteration, x)
            with self.assertRaises(TypeError):
                cstr.get_pixelweights(iteration, x)
            with self.assertRaises(TypeError):
                cstr.get_pixelweights_fn(iteration, x)
            with self.assertRaises(TypeError):
                cstr.get_dmmultgain(iteration, x)
            with self.assertRaises(TypeError):
                cstr.get_unprobedsnr(iteration, x)
            with self.assertRaises(TypeError):
                cstr.get_probedsnr(iteration, x)
            with self.assertRaises(TypeError):
                cstr.get_probeheight(iteration, x)
            pass
        pass



class TestGetWdm(unittest.TestCase):
    """
    Test returning of components of per-actuator weighting matrix
    """

    def setUp(self):
        """
        Preload mode for repeated subsequent use
        """
        self.cfg = cfg
        self.dmlist = [np.zeros((dm.registration['nact'],)*2)
                       for dm in self.cfg.dmlist]
        self.tielist = [np.zeros_like(dm, dtype='int')
                        for dm in self.dmlist] # note *not* cfg.dmlist
        self.ndmact = np.array([dm.registration['nact']**2
                                for dm in cfg.dmlist]).sum()

        pass


    def test_success(self):
        """Good inputs return without issue"""
        get_wdm(self.cfg, self.dmlist, self.tielist)
        pass


    def test_invalid_cfg(self):
        """Verify invalid inputs caught as expected"""

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None]:
            with self.assertRaises(TypeError):
                get_wdm(perr, self.dmlist, self.tielist)
            pass
        pass


    def test_dmlist_input(self):
        """
        Fails as expected when given invalid input
        """
        baddmlist = [np.zeros((dm.registration['nact']+1,)*2)
                  for dm in self.cfg.dmlist]

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None, baddmlist]:
            with self.assertRaises(TypeError):
                get_wdm(self.cfg, perr, self.tielist)
            pass
        pass


    def test_tielist_input(self):
        """
        Fails as expected when given invalid input
        """
        badtielist = [np.zeros((dm.registration['nact']+1,)*2, dtype='int')
                      for dm in self.cfg.dmlist]

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None, badtielist]:
            with self.assertRaises(TypeError):
                get_wdm(self.cfg, self.dmlist, perr)
            pass
        pass


    def test_wdm_output_size(self):
        """
        Test output size is as expected (2D ndmact by ndmact array).
        """
        wdm = get_wdm(self.cfg, self.dmlist, self.tielist)
        self.assertTrue(wdm.shape == (self.ndmact, self.ndmact))
        pass


    def test_wdm_output_type(self):
        """
        Test output type is as expected (wdm sparse)
        """
        wdm = get_wdm(self.cfg, self.dmlist, self.tielist)
        self.assertTrue(scipy.sparse.isspmatrix_csr(wdm))
        pass


    def test_all_mid(self):
        """
        Test wdm when no actuators are violating any constraints and there are
        no ties
        """
        dmlist = [50*np.ones((dm.registration['nact'],)*2)
                  for dm in self.cfg.dmlist]
        wdm = get_wdm(self.cfg, dmlist, self.tielist)

        # More efficient to do != compare on sparse mats
        self.assertTrue((wdm != scipy.sparse.eye(wdm.shape[0],
                                                 wdm.shape[1])).nnz == 0)
        pass



    def test_all_low(self):
        """
        Test wdm freezing when all actuators are low
        """

        dmlist = [np.zeros((dm.registration['nact'],)*2)
                  for dm in self.cfg.dmlist]
        wdm = get_wdm(self.cfg, dmlist, self.tielist)

        # More efficient to do != compare on sparse mats
        self.assertTrue((wdm != scipy.sparse.csr_matrix(wdm.shape)).nnz == 0)
        pass


    def test_all_high(self):
        """
        Test wdm freezing when all actuators are high
        """
        dmlist = [100*np.ones((dm.registration['nact'],)*2)
                  for dm in self.cfg.dmlist]
        wdm = get_wdm(self.cfg, dmlist, self.tielist)

        # More efficient to do != compare on sparse mats
        self.assertTrue((wdm != scipy.sparse.csr_matrix(wdm.shape)).nnz == 0)
        pass


    def test_all_neighbor(self):
        """
        Test wdm tying when all actuators break neighbor rules
        """
        tol = 1e-13

        dmlist = []
        for dm in self.cfg.dmlist:
            checkerboard = 20*np.ones((dm.registration['nact'],)*2)
            checkerboard[0::2, 1::2] = 80.
            checkerboard[1::2, 0::2] = 80.
            dmlist.append(checkerboard)
            pass
        wdm = get_wdm(self.cfg, dmlist, self.tielist)

        nx = wdm.shape[0] # pylint: disable=unsubscriptable-object
        expect = np.ones((nx, nx)).astype('float')
        expect /= cfg.dmlist[0].registration['nact']**2
        expect[nx//2:nx, 0:nx//2] = 0.
        expect[0:nx//2, nx//2:nx] = 0.
        self.assertTrue(np.max(np.abs(wdm -  expect)) < tol)
        pass


class TestGetWe0(unittest.TestCase):
    """
    Test cases for per-pixel weighting vector (diagonal of diagonal matrix)
    """

    def setUp(self):
        self.cfg = cfg
        cstrpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'testdata', 'ut_good_cs.yaml')
        self.cstr = ControlStrategy(cstrpath)
        #436 = (1024 - 1024//2) - 153//2
        self.croplist = [(436, 436, 153, 153)]*len(self.cfg.sl_list)
        self.iteration = 5
        self.contrast = 1e-6
        pass


    def test_success(self):
        """Good inputs return successfully"""
        get_we0(self.cfg, self.cstr, self.croplist,
                self.iteration, self.contrast)
        pass


    def test_we0_output(self):
        """
        Test output is as expected (1D ndphix vector).
        """
        we0 = get_we0(self.cfg, self.cstr, self.croplist,
                      self.iteration, self.contrast)

        ndhpix = 0
        for sl in cfg.sl_list:
            ndhpix += np.sum(sl.dh.e)
            pass

        self.assertTrue(len(np.squeeze(we0).shape) == 1)
        self.assertTrue(we0.size == ndhpix)
        pass


    def test_we0_weights(self):
        """Test wavelengths are weighted as expected"""
        # jac pokes are 1 rad = lam/(2pi) nm, so the magnitude of the applied
        # correction in actuator height grows linearly with wavelength.
        # Compensate by deweighting inversely with wavelength.

        # start from uniform in data (should be in self.cstr)
        tol = 1e-13

        we0 = get_we0(self.cfg, self.cstr, self.croplist,
                      self.iteration, self.contrast)

        ndhpix = np.cumsum([0]+[len(sl.dh_inds) for sl in cfg.sl_list])
        wsum = np.zeros((ndhpix[-1],))
        for index, sl in enumerate(cfg.sl_list):
            # ratio
            wsum[ndhpix[index]:ndhpix[index+1]] = \
              we0[ndhpix[index]:ndhpix[index+1]]/(1/sl.lam)
            pass
        self.assertTrue((np.max(wsum) - np.min(wsum)) < tol)
        pass


    def test_all_bad(self):
        """
        Test filtering in we0 by telling it all pixels are bad
        """
        badcstr = copy.deepcopy(self.cstr)
        badcstr.fixedbp = np.ones_like(badcstr.fixedbp)

        we0 = get_we0(self.cfg, badcstr, self.croplist,
                      self.iteration, self.contrast)

        self.assertTrue((we0 == np.zeros_like(we0)).all())
        pass


    @patch('howfsc.control.cs.ControlStrategy.get_pixelweights')
    def test_ok_with_size_mismatch(self, mock_getpw):
        """
        verify that if the sizes of dh, cstr, and croplist are larger than
        expected, but don't crop the dark hole pixel set, there is no change in
        results
        """
        tmpcfg = copy.deepcopy(self.cfg)
        tmpcroplist = copy.deepcopy(self.croplist)

        # round up to 160 pixels across DH region, to avoid details of exact
        # cfg

        # dh size, pixelweights size, crop dimension
        # cycle through sizing each thing with each of these
        dpclist = [[180, 200, 220],
                   [180, 220, 200],
                   [200, 180, 220],
                   [200, 220, 180],
                   [220, 180, 200],
                   [220, 200, 180],
                   ]

        mock_getpw.side_effect = [[np.ones((160, 160))]*len(tmpcfg.sl_list)]
        we0 = get_we0(self.cfg, self.cstr, self.croplist,
                      self.iteration, self.contrast)

        for ds, ps, cs in dpclist:
            for index, sl in enumerate(tmpcfg.sl_list):
                sl.dh.e = insertinto(sl.dh.e, (ds, ds))
                tmpcroplist[index] = (tmpcroplist[index][0],
                                      tmpcroplist[index][1],
                                      cs,
                                      cs)
                pass

            mock_getpw.side_effect = [[np.ones((ps, ps))]*len(tmpcfg.sl_list)]

            tmpwe0 = get_we0(tmpcfg, self.cstr, tmpcroplist,
                             self.iteration, self.contrast)
            self.assertTrue((we0 == tmpwe0).all())
            pass
        pass



    def test_invalid_cfg(self):
        """Verify invalid inputs caught as expected"""

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None]:
            with self.assertRaises(TypeError):
                get_we0(perr, self.cstr, self.croplist,
                        self.iteration, self.contrast)
            pass
        pass


    def test_invalid_cstr(self):
        """Verify invalid inputs caught as expected"""

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None]:
            with self.assertRaises(TypeError):
                get_we0(self.cfg, perr, self.croplist,
                        self.iteration, self.contrast)
            pass
        pass


    def test_croplist_wrong_length(self):
        """Verify invalid inputs caught"""
        badcroplist = [(436, 436, 153, 153)]*(len(cfg.sl_list) + 1)

        with self.assertRaises(TypeError):
            get_we0(self.cfg, self.cstr, badcroplist,
                    self.iteration, self.contrast)
        pass


    def test_croplist_elements_wrong_type(self):
        """Verify invalid inputs caught"""
        xlist = [[np.ones((4,))]*len(cfg.sl_list),
                 [np.ones((4, 4))]*len(cfg.sl_list),
                 [np.ones((4, 4, 2))]*len(cfg.sl_list),
                 [1]*len(cfg.sl_list),
                 [None]*len(cfg.sl_list),
                 ['txt']*len(cfg.sl_list),
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_we0(self.cfg, self.cstr, x,
                        self.iteration, self.contrast)
            pass
        pass


    def test_croplist_elements_wrong_size(self):
        """Verify invalid inputs caught"""
        xlist = [[(436, 436, 153)]*len(cfg.sl_list),
                 [(436, 436, 153, 153, 0)]*len(cfg.sl_list),
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_we0(self.cfg, self.cstr, x,
                        self.iteration, self.contrast)
            pass
        pass



    def test_croplist_tuple_elements_wrong_type(self):
        """Verify invalid inputs caught"""
        xlist = [[(-1, 436, 153, 153)]*len(cfg.sl_list),
                 [(1.5, 436, 153, 153)]*len(cfg.sl_list),
                 [(436, -1, 153, 153)]*len(cfg.sl_list),
                 [(436, 1.5, 153, 153)]*len(cfg.sl_list),
                 [(436, 436, -1, 153)]*len(cfg.sl_list),
                 [(436, 436, 0, 153)]*len(cfg.sl_list),
                 [(436, 436, 1.5, 153)]*len(cfg.sl_list),
                 [(436, 436, -1, 153)]*len(cfg.sl_list),
                 [(436, 436, 0, 153)]*len(cfg.sl_list),
                 [(436, 436, 1.5, 153)]*len(cfg.sl_list),
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_we0(self.cfg, self.cstr, x,
                        self.iteration, self.contrast)
            pass
        pass


    def test_invalid_iteration(self):
        """Invalid inputs caught"""
        xlist = [0, -1, 1.5, 1j, None, 'txt', (5,)]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_we0(self.cfg, self.cstr, self.croplist,
                        x, self.contrast)
            pass
        pass


    def test_invalid_contrast(self):
        """Invalid inputs caught"""
        xlist = [-1, 1j, None, 'txt', (5,)]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_we0(self.cfg, self.cstr, self.croplist,
                        self.iteration, x)
            pass
        pass




if __name__ == '__main__':
    unittest.main()
