# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Test suite for code to adjust basis to accommodate DM voltage limitations
"""

import unittest
import os

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr

from howfsc.model.mode import CoronagraphMode
from .actlimits import sparsefrommap, maplimits, ActLimitException

cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'model', 'testdata', 'widefov', 'widefov.yaml')
cfg = CoronagraphMode(cfgpath)

class TestSparseFromMap(unittest.TestCase):
    """
    Check DM basis-change creation with voltage limits

    Use != instead of == to check sparse matrices for speed
    """

    def test_standard(self):
        """
        Returns just fine on a perfectly valid input
        """
        goodlim = [{'freeze':[0, 1, 2],
                   'link':[[3, 4, 5], [6, 7]]}] + [{'freeze':[], 'link':[[]]}]
        sparsefrommap(limitlist=goodlim, cfg=cfg)
        pass

    def test_iscsr(self):
        """Verify output is in CSR format"""
        goodlim = [{'freeze':[0, 1, 2],
                   'link':[[3, 4, 5], [6, 7]]}] + [{'freeze':[], 'link':[[]]}]
        F = sparsefrommap(limitlist=goodlim, cfg=cfg)
        self.assertTrue(isspmatrix_csr(F))

    def test_output_size(self):
        """Verify output size matches docs"""
        goodlim = [{'freeze':[0, 1, 2],
                   'link':[[3, 4, 5], [6, 7]]}] + [{'freeze':[], 'link':[[]]}]
        F = sparsefrommap(limitlist=goodlim, cfg=cfg)
        nactall = 0
        for dm in cfg.dmlist:
            nactall += dm.registration['nact']**2
            pass
        self.assertTrue(F.shape == (nactall, nactall))


    def test_limitlist_invalid(self):
        """
        Fails as expected with invalid list
        """
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None,
                     [{'foo', 'bar'}] + [{'freeze':[], 'link':[[]]}]]:
            with self.assertRaises(TypeError):
                sparsefrommap(limitlist=perr, cfg=cfg)
                pass
            pass
        pass


    def test_limitlist_short_keys(self):
        """
        Fails as expected with list not having correct keys
        """
        badlim = [[{'freeze':[0, 1, 2]}]  + [{'freeze':[], 'link':[[]]}],
                  [{'link':[[3, 4, 5]]}]  + [{'freeze':[], 'link':[[]]}]]

        for perr in badlim:
            with self.assertRaises(KeyError):
                sparsefrommap(limitlist=perr, cfg=cfg)
                pass
            pass
        pass


    def test_limitlist_short(self):
        """
        Fails as expected with limitlist too short
        """
        badlim = [{'freeze':[], 'link':[[]]}]

        with self.assertRaises(TypeError):
            sparsefrommap(limitlist=badlim, cfg=cfg)
            pass
        pass


    def test_duplicate_actuators(self):
        """
        Fails as expected if handed duplicate frozen and tied actuators
        """
        badlim = [{'freeze':[0, 1, 2],
                   'link':[[3, 4, 5], [0, 6]]}] + [{'freeze':[], 'link':[[]]}]

        with self.assertRaises(ActLimitException):
            sparsefrommap(limitlist=badlim, cfg=cfg)
            pass
        pass


    def test_cfg_invalid(self):
        """
        Fails as expected with invalid cfg
        """
        limitlist = [{'freeze':[], 'link':[[]]}]*2

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None]:
            with self.assertRaises(TypeError):
                sparsefrommap(limitlist=limitlist, cfg=perr)
                pass
            pass
        pass

    def test_same_index_ok(self):
        """
        Does not fail if the same index is frozen on one and tied on the other
        """
        oklim = [{'freeze':[0, 1],
                   'link':[[]]}] + [{'freeze':[], 'link':[[0, 1]]}]
        sparsefrommap(limitlist=oklim, cfg=cfg)


    def test_lo_hi_grp_DM1(self):
        """
        Behaves as expected when given some simple known inputs of all three
        types (too high, too low, breaking neighbor rules) on DM1
        """
        limitlist = [{'freeze':[0, 1],
                      'link':[[2, 3]]}] + [{'freeze':[],
                      'link':[[]]}]

        corner = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1-0.5, -0.5],
                           [0, 0, -0.5, 1-0.5]])
        corner = csr_matrix(corner)

        F = sparsefrommap(limitlist=limitlist, cfg=cfg)
        self.assertTrue((F[:4, :4] != corner).nnz == 0)
        self.assertTrue((F[4:, 4:] != csr_matrix(F[4:, 4:].shape)).nnz == 0)
        pass

    def test_lo_hi_grp_DM2(self):
        """
        Behaves as expected when given some simple known inputs of all three
        types (too high, too low, breaking neighbor rules) on DM2
        """
        limitlist = [{'freeze':[],
                      'link':[[]]}] + [{'freeze':[0, 1],
                      'link':[[2, 3]]}]

        corner = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1-0.5, -0.5],
                           [0, 0, -0.5, 1.-0.5]])
        corner = csr_matrix(corner)
        ndmact = np.cumsum([0]
                        +[dm.registration['nact']**2 for dm in cfg.dmlist])

        F = sparsefrommap(limitlist=limitlist, cfg=cfg)
        self.assertTrue((F[ndmact[1]:ndmact[1]+4,
                             ndmact[1]:ndmact[1]+4] != corner).nnz == 0)
        F1 = F[ndmact[1]+4:ndmact[2], ndmact[1]+4:ndmact[2]]
        self.assertTrue((F1 != csr_matrix(F1.shape)).nnz == 0)
        F2 = F[:ndmact[1], :ndmact[1]]
        self.assertTrue((F2 != csr_matrix(F2.shape)).nnz == 0)
        pass


    def test_multiple_ties(self):
        """
        Behaves as expected when given multiple sets of tied actuators
        """
        limitlist = [{'freeze':[],
                      'link':[[0, 1], [2, 3]]}] + [{'freeze':[],
                      'link':[[]]}]

        corner = np.array([[1-0.5, -0.5, 0, 0],
                           [-0.5, 1-0.5, 0, 0],
                           [0, 0, 1-0.5, -0.5],
                           [0, 0, -0.5, 1-0.5]])
        corner = csr_matrix(corner)

        F = sparsefrommap(limitlist=limitlist, cfg=cfg)
        self.assertTrue((F[:4, :4] != corner).nnz == 0)
        self.assertTrue((F[4:, 4:] !=
                         csr_matrix(F[4:, 4:].shape)).nnz == 0)
        pass


class TestMapLimits(unittest.TestCase):
    """
    Test for identifying which actuators to limit
    """

    def setUp(self):
        """Predefine variables for use"""
        self.dmvobj = cfg.dmlist[0].dmvobj
        nact = cfg.dmlist[0].registration['nact']
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vmid = (vmin + vmax)/2. # midpoint

        # no freeze/tie
        self.dmnom = vmid*np.ones((nact, nact))

        # much freeze/tie
        np.random.seed(5)
        vlo = vmid - vneighbor*2/3.
        vhi = vmid + vneighbor*2/3.
        rdm = (vlo + (vhi - vlo)*(np.random.random_sample((nact, nact))))
        lind = np.random.randint(0, nact**2, 20)
        hind = np.random.randint(0, nact**2, 20)
        rdm.ravel()[lind] = vmin
        rdm.ravel()[hind] = vmax
        rdm[0, 0] = vmin
        rdm[0, 1] = vmax # explicit case of freeze AND tie
        self.dmft = rdm

        # default tiemap
        self.tie = np.zeros((nact, nact), dtype='int')
        pass


    def test_standard(self):
        """
        Works when fed good input
        """
        maplimits(dmv=self.dmnom, dmvobj=self.dmvobj)
        pass


    def test_standard_tie(self):
        """
        Works when fed good input, including a valid tie
        """
        maplimits(dmv=self.dmnom, dmvobj=self.dmvobj, tiemap=self.tie)
        pass


    def test_dmv_invalid(self):
        """
        Fails as expected given bad DM input
        """
        nact = cfg.dmlist[0].registration['nact']
        baddm = [np.ones((nact+1, nact)),
                 np.ones((nact, nact+1)),
                 np.ones((nact,)),
                 np.ones((nact, nact, nact))]

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None] + baddm:
            with self.assertRaises(TypeError):
                maplimits(dmv=perr, dmvobj=self.dmvobj)
                pass
            pass
        pass


    def test_dmvobj_invalid(self):
        """
        Fails as expected given bad DM voltage spec
        """
        baddmvobj = [cfg]

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,), None] + baddmvobj:
            with self.assertRaises(TypeError):
                maplimits(dmv=self.dmnom, dmvobj=perr)
                pass
            pass
        pass


    def test_tie_invalid(self):
        """
        Fails as expected given invalid format
        """

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,),
                     np.ones((5,)), np.ones((5, 5, 2))]:
            with self.assertRaises(TypeError):
                maplimits(dmv=self.dmnom, dmvobj=self.dmvobj, tiemap=perr)
                pass
            pass
        pass


    def test_tie_not_in_tie_format(self):
        """
        Fails as expected given tie not meeting tie spec
        """
        tie = -2*np.ones_like(self.dmnom)
        with self.assertRaises(ValueError):
            maplimits(dmv=self.dmnom, dmvobj=self.dmvobj, tiemap=tie)
            pass
        pass


    def test_tie_wrong_size(self):
        """
        Fails as expected given dmv and tie of different sizes
        """

        tie = np.zeros((self.dmnom.shape[0]+1, self.dmnom.shape[1]+1))
        with self.assertRaises(TypeError):
            maplimits(dmv=self.dmnom, dmvobj=self.dmvobj, tiemap=tie)
            pass
        pass


    def test_tie_none(self):
        """Verify None is a valid input to tie"""
        maplimits(dmv=self.dmnom, dmvobj=self.dmvobj, tiemap=None)
        pass


    def test_tie_none_is_unconstrained(self):
        """Check None behaves like no constraints"""
        outn = maplimits(dmv=self.dmft, dmvobj=self.dmvobj, tiemap=None)
        outt = maplimits(dmv=self.dmft, dmvobj=self.dmvobj,
                         tiemap=np.zeros(self.dmft.shape, dtype='int'))
        self.assertTrue((outt['freeze'] == outn['freeze']).all())
        for index, grp in enumerate(outt['link']):
            self.assertTrue((grp == outn['link'][index]).all())
            pass
        pass


    def test_dead_unchanged(self):
        """Verify that dead actuators are not unfrozen in a nontrivial case"""
        tmptie = self.tie.copy()

        inds = [0, 1, 1000, 500, 501, 2303, 1994, 27, 2270, 1315, 8, 903]

        for ind in inds:
            tmptie.ravel()[ind] = -1
            pass

        out = maplimits(dmv=self.dmft, dmvobj=self.dmvobj, tiemap=tmptie)

        self.assertTrue(set(inds).issubset(set(out['freeze'])))

        pass


    def test_no_duplicate_actuators(self):
        """
        Test a case where actuators hit both caps and neighbor rules and ensure
        that we get no duplicates
        """
        limit = maplimits(dmv=self.dmft, dmvobj=self.dmvobj)

        # Check for duplicate indices (sets just make this efficient)
        fset = set(limit['freeze'])
        tset = set()
        for tie in limit['link']:
            tset.update(tie)
            pass
        self.assertTrue(not bool(fset & tset))
        pass


    def test_freeze_tie(self):
        """
        Test frozen *and* tied actuators produce expected result (all frozen)
        """

        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vcorner = self.dmvobj.vcorner
        nact = cfg.dmlist[0].registration['nact']

        dmtest = (vmax-max(vneighbor, vcorner))*np.ones((nact, nact))
        dmtest[1, 1] = vmax

        testset = set()
        for j in range(3):
            for k in range(3):
                testset.add((j, k))
                pass
            pass

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertEqual(limit['link'], [])
        flist = np.transpose(np.unravel_index(limit['freeze'], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_freeze_high(self):
        """High actuator freezes successfully"""
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vcorner = self.dmvobj.vcorner
        nact = cfg.dmlist[0].registration['nact']

        # use 1/2 NR so we don't tie
        dmtest = (vmax-max(vneighbor, vcorner)/2.)*np.ones((nact, nact))
        dmtest[1, 1] = vmax

        testset = set()
        testset.add((1, 1))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertEqual(limit['link'], [])
        flist = np.transpose(np.unravel_index(limit['freeze'], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_freeze_low(self):
        """Low actuator freezes successfully"""
        vmin = self.dmvobj.vmin
        vneighbor = self.dmvobj.vneighbor
        vcorner = self.dmvobj.vcorner
        nact = cfg.dmlist[0].registration['nact']

        # use 1/2 NR so we don't tie
        dmtest = (vmin+max(vneighbor, vcorner)/2.)*np.ones((nact, nact))
        dmtest[1, 1] = vmin

        testset = set()
        testset.add((1, 1))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertEqual(limit['link'], [])
        flist = np.transpose(np.unravel_index(limit['freeze'], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_tie_horizontal(self):
        """NR violations in horizontal become ties"""
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        dmtest = vmid*np.ones((nact, nact))
        dmtest[1, 1] = vmid + vneighbor/2.
        dmtest[1, 2] = vmid - vneighbor/2.

        testset = set()
        testset.add((1, 1))
        testset.add((1, 2))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertTrue(limit['freeze'].size == 0)
        self.assertNotEqual(limit['link'], [])
        self.assertTrue(len(limit['link']) == 1)

        flist = np.transpose(np.unravel_index(limit['link'][0], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_tie_vertical(self):
        """NR violations in horizontal become ties"""
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        dmtest = vmid*np.ones((nact, nact))
        dmtest[1, 1] = vmid + vneighbor/2.
        dmtest[2, 1] = vmid - vneighbor/2.

        testset = set()
        testset.add((1, 1))
        testset.add((2, 1))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertTrue(limit['freeze'].size == 0)
        self.assertNotEqual(limit['link'], [])
        self.assertTrue(len(limit['link']) == 1)

        flist = np.transpose(np.unravel_index(limit['link'][0], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_tie_diagonal_right(self):
        """NR violations in one diagonal become ties"""
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vcorner = self.dmvobj.vcorner
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        dmtest = vmid*np.ones((nact, nact))
        dmtest[1, 1] = vmid + vcorner/2.
        dmtest[2, 2] = vmid - vcorner/2.

        testset = set()
        testset.add((1, 1))
        testset.add((2, 2))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertTrue(limit['freeze'].size == 0)
        self.assertNotEqual(limit['link'], [])
        self.assertTrue(len(limit['link']) == 1)

        flist = np.transpose(np.unravel_index(limit['link'][0], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_tie_diagonal_left(self):
        """NR violations in other diagonal become ties"""
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vcorner = self.dmvobj.vcorner
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        dmtest = vmid*np.ones((nact, nact))
        dmtest[1, 1] = vmid + vcorner/2.
        dmtest[2, 0] = vmid - vcorner/2.

        testset = set()
        testset.add((1, 1))
        testset.add((2, 0))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertTrue(limit['freeze'].size == 0)
        self.assertNotEqual(limit['link'], [])
        self.assertTrue(len(limit['link']) == 1)

        flist = np.transpose(np.unravel_index(limit['link'][0], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_tie_vertical_edge(self):
        """NR violations in horizontal become ties for actuators along edge"""
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        dmtest = vmid*np.ones((nact, nact))
        dmtest[1, 0] = vmid + vneighbor/2.
        dmtest[2, 0] = vmid - vneighbor/2.

        testset = set()
        testset.add((1, 0))
        testset.add((2, 0))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertTrue(limit['freeze'].size == 0)
        self.assertNotEqual(limit['link'], [])
        self.assertTrue(len(limit['link']) == 1)

        flist = np.transpose(np.unravel_index(limit['link'][0], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_tie_multiple(self):
        """Tying is done correctly when all four types of tie are present"""
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vcorner = self.dmvobj.vcorner
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        dmtest = vmid*np.ones((nact, nact))
        dmtest[1, 1] = vmid + vcorner/2.
        dmtest[2, 0] = vmid - vcorner/2.
        dmtest[5, 1] = vmid + vcorner/2.
        dmtest[6, 2] = vmid - vcorner/2.
        dmtest[1, 5] = vmid + vneighbor/2.
        dmtest[2, 5] = vmid - vneighbor/2.
        dmtest[5, 5] = vmid + vneighbor/2.
        dmtest[5, 6] = vmid - vneighbor/2.

        sd1 = set()
        sd1.add((1, 1))
        sd1.add((2, 0))
        sd2 = set()
        sd2.add((5, 1))
        sd2.add((6, 2))
        sve = set()
        sve.add((1, 5))
        sve.add((2, 5))
        sho = set()
        sho.add((5, 5))
        sho.add((5, 6))
        listofsets = [sd1, sd2, sve, sho]

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertTrue(limit['freeze'].size == 0)
        self.assertNotEqual(limit['link'], [])
        self.assertTrue(len(limit['link']) == 4)

        for tlist in limit['link']:
            flist = np.transpose(np.unravel_index(tlist, (nact, nact)))

            fset = set()
            for f in flist:
                fset.add(tuple(f))
                pass
            self.assertTrue(fset in listofsets)
            pass
        pass


    def test_grow(self):
        """
        Test tied-group growth by adding in a chain of NR actuators which all
        connect along different axes but should be linked together by the
        _growgrp loop
        """
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        # keep these small so we don't accidentally freeze
        self.dmvobj.vneighbor = (vmax - vmin)/4.
        vneighbor = self.dmvobj.vneighbor
        self.dmvobj.vcorner = (vmax - vmin)/4.
        vcorner = self.dmvobj.vcorner
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        dmtest = vmid*np.ones((nact, nact))
        dmtest[0, 0] = vmid + vneighbor/2.
        dmtest[0, 1] = vmid - vneighbor/2.
        dmtest[1, 2] = vmid - vneighbor/2. + vcorner
        dmtest[2, 3] = vmid - vneighbor/2.
        dmtest[3, 3] = vmid + vneighbor/2.
        dmtest[4, 2] = vmid + vneighbor/2. - vcorner
        dmtest[5, 1] = vmid + vneighbor/2.
        dmtest[5, 0] = vmid - vneighbor/2.

        testset = set()
        testset.add((0, 0))
        testset.add((0, 1))
        testset.add((1, 2))
        testset.add((2, 3))
        testset.add((3, 3))
        testset.add((4, 2))
        testset.add((5, 1))
        testset.add((5, 0))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertTrue(limit['freeze'].size == 0)
        self.assertNotEqual(limit['link'], [])
        self.assertTrue(len(limit['link']) == 1)

        flist = np.transpose(np.unravel_index(limit['link'][0], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_doublegrow(self):
        """
        Test tied-group growth by feeding in a line of actuators on the same
        axis and verify that it expands to cover them all.  Force this to
        happen in growgrp by starting off with a horizontal NR and then making
        the rest vertical.
        """
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        dmtest = vmid*np.ones((nact, nact))
        dmtest[0, 0] = vmid + vneighbor/2.
        dmtest[0, 1] = vmid - vneighbor/2.
        dmtest[1, 1] = vmid + vneighbor/2. # 0,0 & 1,1 same, so no diag viol
        dmtest[1, 2] = vmid - vneighbor/2.
        dmtest[1, 3] = vmid + vneighbor/2.
        dmtest[1, 4] = vmid - vneighbor/2.
        dmtest[1, 5] = vmid + vneighbor/2.
        dmtest[1, 6] = vmid - vneighbor/2.

        testset = set()
        testset.add((0, 0))
        testset.add((0, 1))
        testset.add((1, 1))
        testset.add((1, 2))
        testset.add((1, 3))
        testset.add((1, 4))
        testset.add((1, 5))
        testset.add((1, 6))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertTrue(limit['freeze'].size == 0)
        self.assertNotEqual(limit['link'], [])
        self.assertTrue(len(limit['link']) == 1)

        flist = np.transpose(np.unravel_index(limit['link'][0], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_doublegrow_2group(self):
        """
        Test tied-group growth by feeding in a line of actuators on the same
        axis and verify that it expands to cover them all.  Force this to
        happen in growgrp by starting off with a horizontal NR and then making
        the rest vertical.

        Do this with two different groups to verify that multiple groups are
        being made correctly.
        """
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        dmtest = vmid*np.ones((nact, nact))
        dmtest[0, 0] = vmid + vneighbor/2.
        dmtest[0, 1] = vmid - vneighbor/2.
        dmtest[1, 1] = vmid + vneighbor/2. # 0,0 & 1,1 same, so no diag viol
        dmtest[1, 2] = vmid - vneighbor/2.
        dmtest[1, 3] = vmid + vneighbor/2.
        dmtest[1, 4] = vmid - vneighbor/2.
        dmtest[1, 5] = vmid + vneighbor/2.
        dmtest[1, 6] = vmid - vneighbor/2.

        dmtest[20, 0] = vmid + vneighbor/2.
        dmtest[20, 1] = vmid - vneighbor/2.
        dmtest[21, 1] = vmid + vneighbor/2. # 0,0 & 1,1 same, so no diag viol
        dmtest[21, 2] = vmid - vneighbor/2.
        dmtest[21, 3] = vmid + vneighbor/2.
        dmtest[21, 4] = vmid - vneighbor/2.
        dmtest[21, 5] = vmid + vneighbor/2.
        dmtest[21, 6] = vmid - vneighbor/2.

        testset = set()
        testset.add((0, 0))
        testset.add((0, 1))
        testset.add((1, 1))
        testset.add((1, 2))
        testset.add((1, 3))
        testset.add((1, 4))
        testset.add((1, 5))
        testset.add((1, 6))

        testset2 = set()
        testset2.add((20, 0))
        testset2.add((20, 1))
        testset2.add((21, 1))
        testset2.add((21, 2))
        testset2.add((21, 3))
        testset2.add((21, 4))
        testset2.add((21, 5))
        testset2.add((21, 6))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj)
        self.assertTrue(limit['freeze'].size == 0)
        self.assertNotEqual(limit['link'], [])
        self.assertTrue(len(limit['link']) == 2)

        # 1st group
        flist = np.transpose(np.unravel_index(limit['link'][0], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        # 2nd group
        flist = np.transpose(np.unravel_index(limit['link'][1], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset2, fset)

        pass


    def test_tie_grow(self):
        """
        Test tied-group growth by feeding in a line of actuators on the same
        axis and verify that it expands to cover them all.  Force this to
        happen in growgrp by starting off with a horizontal NR and then making
        the rest vertical.

        This one is combined with an additional nonlocal connection via the
        tie matrix
        """
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        tmptie = self.tie.copy()
        tmptie[1, 6] = 1
        tmptie[20, 20] = 1

        dmtest = vmid*np.ones((nact, nact))
        dmtest[0, 0] = vmid + vneighbor/2.
        dmtest[0, 1] = vmid - vneighbor/2.
        dmtest[1, 1] = vmid + vneighbor/2. # 0,0 & 1,1 same, so no diag viol
        dmtest[1, 2] = vmid - vneighbor/2.
        dmtest[1, 3] = vmid + vneighbor/2.
        dmtest[1, 4] = vmid - vneighbor/2.
        dmtest[1, 5] = vmid + vneighbor/2.
        dmtest[1, 6] = vmid - vneighbor/2.

        testset = set()
        testset.add((0, 0))
        testset.add((0, 1))
        testset.add((1, 1))
        testset.add((1, 2))
        testset.add((1, 3))
        testset.add((1, 4))
        testset.add((1, 5))
        testset.add((1, 6))
        testset.add((20, 20))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj, tiemap=tmptie)
        self.assertTrue(limit['freeze'].size == 0)
        self.assertNotEqual(limit['link'], [])
        self.assertTrue(len(limit['link']) == 1)

        flist = np.transpose(np.unravel_index(limit['link'][0], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_tie_freeze_grow(self):
        """
        Test tied-group growth by feeding in a line of actuators on the same
        axis and verify that it expands to cover them all.  Force this to
        happen in growgrp by starting off with a horizontal NR and then making
        the rest vertical.

        This one is combined with an additional nonlocal connection via the
        tie matrix, this time to an actuator which is high.
        """
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vcorner = self.dmvobj.vcorner
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        tmptie = self.tie.copy()
        tmptie[1, 6] = 1
        tmptie[47, 47] = 1

        dmtest = vmid*np.ones((nact, nact))
        dmtest[0, 0] = vmid + vneighbor/2.
        dmtest[0, 1] = vmid - vneighbor/2.
        dmtest[1, 1] = vmid + vneighbor/2. # 0,0 & 1,1 same, so no diag viol
        dmtest[1, 2] = vmid - vneighbor/2.
        dmtest[1, 3] = vmid + vneighbor/2.
        dmtest[1, 4] = vmid - vneighbor/2.
        dmtest[1, 5] = vmid + vneighbor/2.
        dmtest[1, 6] = vmid - vneighbor/2.
        dmtest[47, 47] = vmax
        # Keep (47, 47) from NR-linking locally though
        dmtest[47, 46] = max(vmax - vneighbor + 1, vmid)
        dmtest[46, 47] = max(vmax - vneighbor + 1, vmid)
        dmtest[46, 46] = max(vmax - vcorner + 1, vmid)

        dmtest[47, 45] = max(vmax - 2*vneighbor + 2, vmid)
        dmtest[45, 47] = max(vmax - 2*vneighbor + 2, vmid)
        dmtest[45, 45] = max(vmax - 2*vcorner + 2, vmid)
        dmtest[45, 46] = max(vmax - vneighbor - vcorner + 2, vmid)
        dmtest[46, 45] = max(vmax - vneighbor - vcorner + 2, vmid)

        testset = set()
        testset.add((0, 0))
        testset.add((0, 1))
        testset.add((1, 1))
        testset.add((1, 2))
        testset.add((1, 3))
        testset.add((1, 4))
        testset.add((1, 5))
        testset.add((1, 6))
        testset.add((47, 47))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj, tiemap=tmptie)
        self.assertEqual(limit['link'], [])
        flist = np.transpose(np.unravel_index(limit['freeze'], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_tie_freeze_grow_2link(self):
        """
        Test tied-group growth by feeding in a line of actuators on the same
        axis and verify that it expands to cover them all.  Force this to
        happen in growgrp by starting off with a horizontal NR and then making
        the rest vertical.

        This one is combined with an additional nonlocal connection via the
        tie matrix, this time to an actuator which is high, and a second which
        is entirely disconnected.
        """
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vcorner = self.dmvobj.vcorner
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        tmptie = self.tie.copy()
        tmptie[20, 20] = 1
        tmptie[20, 21] = 1
        tmptie[1, 6] = 2
        tmptie[47, 47] = 2

        dmtest = vmid*np.ones((nact, nact))
        dmtest[0, 0] = vmid + vneighbor/2.
        dmtest[0, 1] = vmid - vneighbor/2.
        dmtest[1, 1] = vmid + vneighbor/2. # 0,0 & 1,1 same, so no diag viol
        dmtest[1, 2] = vmid - vneighbor/2.
        dmtest[1, 3] = vmid + vneighbor/2.
        dmtest[1, 4] = vmid - vneighbor/2.
        dmtest[1, 5] = vmid + vneighbor/2.
        dmtest[1, 6] = vmid - vneighbor/2.
        dmtest[47, 47] = vmax
        # Keep (47, 47) from NR-linking locally though
        dmtest[47, 46] = max(vmax - vneighbor + 1, vmid)
        dmtest[46, 47] = max(vmax - vneighbor + 1, vmid)
        dmtest[46, 46] = max(vmax - vcorner + 1, vmid)

        dmtest[47, 45] = max(vmax - 2*vneighbor + 2, vmid)
        dmtest[45, 47] = max(vmax - 2*vneighbor + 2, vmid)
        dmtest[45, 45] = max(vmax - 2*vcorner + 2, vmid)
        dmtest[45, 46] = max(vmax - vneighbor - vcorner + 2, vmid)
        dmtest[46, 45] = max(vmax - vneighbor - vcorner + 2, vmid)

        testset = set()
        testset.add((0, 0))
        testset.add((0, 1))
        testset.add((1, 1))
        testset.add((1, 2))
        testset.add((1, 3))
        testset.add((1, 4))
        testset.add((1, 5))
        testset.add((1, 6))
        testset.add((47, 47))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj, tiemap=tmptie)
        self.assertTrue(len(limit['link']) == 1)
        self.assertTrue((limit['link'][0] == np.array([20*nact+20,
                                                       20*nact+21])).all())
        flist = np.transpose(np.unravel_index(limit['freeze'], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass


    def test_tie_dead_grow(self):
        """
        Test tied-group growth by feeding in a line of actuators on the same
        axis and verify that it expands to cover them all.  Force this to
        happen in growgrp by starting off with a horizontal NR and then making
        the rest vertical.

        This is combined with a dead flag on one of the actuators, which should
        freeze the whole set
        """
        vmin = self.dmvobj.vmin
        vmax = self.dmvobj.vmax
        vneighbor = self.dmvobj.vneighbor
        vmid = (vmin + vmax)/2. # midpoint
        nact = cfg.dmlist[0].registration['nact']

        tmptie = self.tie.copy()
        tmptie[1, 6] = -1

        dmtest = vmid*np.ones((nact, nact))
        dmtest[0, 0] = vmid + vneighbor/2.
        dmtest[0, 1] = vmid - vneighbor/2.
        dmtest[1, 1] = vmid + vneighbor/2. # 0,0 & 1,1 same, so no diag viol
        dmtest[1, 2] = vmid - vneighbor/2.
        dmtest[1, 3] = vmid + vneighbor/2.
        dmtest[1, 4] = vmid - vneighbor/2.
        dmtest[1, 5] = vmid + vneighbor/2.
        dmtest[1, 6] = vmid - vneighbor/2.

        testset = set()
        testset.add((0, 0))
        testset.add((0, 1))
        testset.add((1, 1))
        testset.add((1, 2))
        testset.add((1, 3))
        testset.add((1, 4))
        testset.add((1, 5))
        testset.add((1, 6))

        limit = maplimits(dmv=dmtest, dmvobj=self.dmvobj, tiemap=tmptie)
        self.assertEqual(limit['link'], [])
        flist = np.transpose(np.unravel_index(limit['freeze'], (nact, nact)))

        fset = set()
        for f in flist:
            fset.add(tuple(f))
            pass
        self.assertEqual(testset, fset)

        pass





if __name__ == '__main__':
    unittest.main()
