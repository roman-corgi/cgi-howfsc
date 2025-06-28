# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# TODO: test for right keys in dms and in init
"""
Unit tests for CoronagraphMode class
"""

import unittest
import os

from howfsc.model.mode import CoronagraphMode
from howfsc.model.parse_mdf import MDFException

localpath = os.path.dirname(os.path.abspath(__file__))
utpath = os.path.join(localpath, 'testdata', 'ut')

class TestCoronagraphMode(unittest.TestCase):
    """
    Unit test suite for CoronagraphMode constructor

    Give yaml with extra data and verify it ignores it

    """

    def test_successful_read(self):
        """ No errors on correctly-formatted file. """
        fn = os.path.join(utpath, 'ut_parse_good.yaml')
        CoronagraphMode(fn)
        pass

    def test_unsuccessful_input(self):
        """Incorrectly formatted file fails as expected"""
        fn = os.path.join(utpath, 'ut_parse_2_extra.yaml')
        with self.assertRaises(MDFException):
            CoronagraphMode(fn)
        pass



if __name__ == '__main__':
    unittest.main()
