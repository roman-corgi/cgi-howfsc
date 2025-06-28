# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for calcn2c.py (normalized-intensity-to-contrast matrix calculator)
"""
import unittest
import os

import numpy as np

from howfsc.model.mode import CoronagraphMode
from howfsc.util.insertinto import insertinto
from .calcn2c import calcn2c

# use small array to keep n2c manageable
cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'model', 'testdata', 'ut', 'ut_smalljac.yaml')
cfg = CoronagraphMode(cfgpath)


class TestCalcN2C(unittest.TestCase):
    """
    Tests for normalized-intensity-to-contrast matrices
    """

    def setUp(self):
        self.cfg = cfg
        self.idx = 0
        self.nrow = 11
        self.ncol = 12
        self.dmset_list = cfg.initmaps
        pass


    def test_success(self):
        """valid inputs complete without error as expected"""
        calcn2c(
            self.cfg,
            self.idx,
            self.nrow,
            self.ncol,
            self.dmset_list,
        )
        pass


    def test_output_format_as_expected(self):
        """format meets description in docstring"""
        n2c = calcn2c(
            self.cfg,
            self.idx,
            self.nrow,
            self.ncol,
            self.dmset_list,
        )

        # shape
        self.assertTrue(n2c.shape, (self.nrow, self.ncol))
        # nans
        dh = insertinto(self.cfg.sl_list[self.idx].dh.e,
                        (self.nrow, self.ncol))
        self.assertTrue(((dh == 0) == (np.isnan(n2c))).all())
        # value (true for this particular config file)
        self.assertTrue((n2c[~np.isnan(n2c)] >= 1).all())

        pass


    def test_invalid_cfg(self):
        """valid inputs complete without error as expected"""
        perrlist = [0, None, 'txt', np.ones((5, 5)), cfg.sl_list[0]]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                calcn2c(
                    perr,
                    self.idx,
                    self.nrow,
                    self.ncol,
                    self.dmset_list,
                )
            pass
        pass


    def test_invalid_idx_type(self):
        """valid inputs complete without error as expected"""
        # things that aren't integers
        perrlist = [0.5, None, 'txt', np.ones((5, 5)), cfg.sl_list[0]]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                calcn2c(
                    self.cfg,
                    perr,
                    self.nrow,
                    self.ncol,
                    self.dmset_list,
                )
            pass
        pass


    def test_invalid_idx_contents(self):
        """valid inputs complete without error as expected"""
        # out-of-range integers
        perrlist = [-1, len(cfg.sl_list)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                calcn2c(
                    self.cfg,
                    perr,
                    self.nrow,
                    self.ncol,
                    self.dmset_list,
                )
            pass
        pass


    def test_invalid_nrow(self):
        """valid inputs complete without error as expected"""
        perrlist = [0, 0.5, None, 'txt', np.ones((5, 5)), cfg.sl_list[0]]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                calcn2c(
                    self.cfg,
                    self.idx,
                    perr,
                    self.ncol,
                    self.dmset_list,
                )
            pass
        pass


    def test_invalid_ncol(self):
        """valid inputs complete without error as expected"""
        perrlist = [0, 0.5, None, 'txt', np.ones((5, 5)), cfg.sl_list[0]]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                calcn2c(
                    self.cfg,
                    self.idx,
                    self.nrow,
                    perr,
                    self.dmset_list,
                )
            pass
        pass


    def test_invalid_dmsetlist(self):
        """valid inputs complete without error as expected"""
        # generic type mismatches
        perrlist = [0, 0.5, None, 'txt', np.ones((5, 5)), cfg.sl_list[0]]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                calcn2c(
                    self.cfg,
                    self.idx,
                    self.nrow,
                    self.ncol,
                    perr,
                )
            pass
        pass


    def test_invalid_dmsetlist_arrays(self):
        """valid inputs complete without error as expected"""
        # array mismatches
        perrlist = [[None, None],
                    [np.zeros((48, 48)), None],
                    [None, np.zeros((48, 48))],
                    [np.eye(49), np.eye(48)],
                    [np.eye(48), np.eye(47)],
                    [np.zeros((48, 48)), np.zeros((48, 47))],
                    [np.zeros((49, 48)), np.zeros((48, 48))],
                    [np.eye(48), np.eye(48), np.eye(48)],
                    [np.eye(48)],
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                calcn2c(
                    self.cfg,
                    self.idx,
                    self.nrow,
                    self.ncol,
                    perr,
                )
            pass
        pass




if __name__ == '__main__':
    unittest.main()
