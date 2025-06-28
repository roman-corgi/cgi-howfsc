# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for the model definition file checkout
"""
import unittest
import os

import numpy as np

from .parse_mdf import MDFException
from .parse_mdf import check_mode_lists, validate_model_file

class TestMDFValidation(unittest.TestCase):
    """
    Tests if the HOWFSC optical model definition file validation tools
    validate as expected

    Check three expected returns (T/F/N) for validate function

    """
    def setUp(self):
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        pass


    def test_good_yaml(self):
        """Good YAML runs the lists to completion, with/without files"""
        fn = os.path.join(self.localpath, 'testdata', 'ut',
                          'ut_parse_good.yaml')

        check_mode_lists(fn, usefiles=True)
        check_mode_lists(fn, usefiles=False)
        pass


    def test_good_yaml_no_decpt(self):
        """
        Good YAML runs the lists to completion, with/without files, even
        despite sloppy handling of floating point numbers (sometimes they
        are parsed as strings if there's not an explicit decimal point)
        """
        fn = os.path.join(self.localpath, 'testdata', 'ut',
                          'ut_parse_good_no_decpt.yaml')

        check_mode_lists(fn, usefiles=True)
        check_mode_lists(fn, usefiles=False)
        pass


    def test_invalid_fn(self):
        """Invalid inputs caught"""

        xlist = [2, None, np.eye(3), (5, 5)]

        for x in xlist:
            with self.assertRaises(TypeError):
                check_mode_lists(x, usefiles=False)
            pass
        pass


    def test_invalid_usefiles(self):
        """Invalid inputs caught"""
        fn = os.path.join(self.localpath, 'testdata', 'ut',
                          'ut_parse_good.yaml')

        xlist = [2, None, np.eye(3), (5, 5), 'txt']

        for x in xlist:
            with self.assertRaises(TypeError):
                check_mode_lists(fn, x)
            pass
        pass


    def test_file_does_not_exist(self):
        """Verify missing file in fn caught as expected"""
        fn = 'does_not_exist'
        with self.assertRaises(MDFException):
            check_mode_lists(fn, False)
        pass


    def test_file_exists_but_is_not_yaml_type(self):
        """Non-YAML files caught as expected"""
        fn = os.path.join(self.localpath, 'testdata', 'ut',
                          'ut_not_yaml.fits')

        self.assertTrue(os.path.exists(fn))
        with self.assertRaises(MDFException):
            check_mode_lists(fn, False)
        pass


    def test_file_exists_but_is_not_valid_yaml(self):
        """Text files that don't meet the YAML spec are caught"""
        fn = os.path.join(self.localpath, 'testdata', 'ut',
                          'ut_not_valid_yaml.yaml')

        self.assertTrue(os.path.exists(fn))
        with self.assertRaises(MDFException):
            check_mode_lists(fn, False)
        pass


    def test_file_contains_deviations_from_spec(self):
        """contents of file are caught if they don't match the written spec"""
        fnsufflist = [
            'ut_parse_2_miss.yaml',
            'ut_parse_2_extra.yaml',
            'ut_parse_2a_miss.yaml',
            'ut_parse_2a_extra.yaml',
            'ut_parse_2ai_miss.yaml',
            'ut_parse_2ai_extra.yaml',
            'ut_parse_2ai1_type.yaml',
            'ut_parse_2ai1_val.yaml',
            'ut_parse_2ai2_type.yaml',
            'ut_parse_2ai3_miss.yaml',
            'ut_parse_2ai3_extra.yaml',
            'ut_parse_2ai3a_type.yaml',
            'ut_parse_2ai3b_type.yaml',
            'ut_parse_2ai3c_type.yaml',
            'ut_parse_2ai3d_type.yaml',
            'ut_parse_2ai3d_val.yaml',
            'ut_parse_2ai3e_type.yaml',
            'ut_parse_2ai3e_val.yaml',
            'ut_parse_2ai3f_type.yaml',
            'ut_parse_2ai3f_val.yaml',
            'ut_parse_2ai3g_type.yaml',
            'ut_parse_2ai3g_val.yaml',
            'ut_parse_2ai3h_type.yaml',
            'ut_parse_2ai3i_type.yaml',
            'ut_parse_2ai4_miss.yaml',
            'ut_parse_2ai4_extra.yaml',
            'ut_parse_2ai4a_type.yaml',
            'ut_parse_2ai4b_type.yaml',
            'ut_parse_2ai4b_vmax_vmin.yaml',
            'ut_parse_2ai4c_type.yaml',
            'ut_parse_2ai4c_val.yaml',
            'ut_parse_2ai4d_type.yaml',
            'ut_parse_2ai4d_val.yaml',
            'ut_parse_2ai4e_type.yaml',
            'ut_parse_2ai4e_val.yaml',
            'ut_parse_2ai4f_type.yaml',
            'ut_parse_2ai4g_type.yaml',
            'ut_parse_2ai4h_type.yaml',
            'ut_parse_2ai4i_type.yaml',
            'ut_parse_2b_miss.yaml',
            'ut_parse_2b_extra.yaml',
            'ut_parse_2bi_miss.yaml',
            'ut_parse_2bi_extra.yaml',
            'ut_parse_2bi1_type.yaml',
            'ut_parse_2c_miss.yaml',
            'ut_parse_2c_extra.yaml',
            'ut_parse_2c_wrongnums.yaml',
            'ut_parse_2ci_miss.yaml',
            'ut_parse_2ci_extra.yaml',
            'ut_parse_2ci1_type.yaml',
            'ut_parse_2ci1_val.yaml',
            'ut_parse_2ci2_miss.yaml',
            'ut_parse_2ci2_extra.yaml',
            'ut_parse_2ci2_combo.yaml',
            'ut_parse_2ci2a_afn.yaml',
            'ut_parse_2ci2a_pfn.yaml',
            'ut_parse_2ci2a_rfn.yaml',
            'ut_parse_2ci2a_ifn.yaml',
            'ut_parse_2ci2b_type.yaml',
            'ut_parse_2ci2b_val.yaml',
            'ut_parse_2ci2c_type.yaml',
            'ut_parse_2ci2d_type.yaml',
            'ut_parse_2ci3_miss.yaml',
            'ut_parse_2ci3_extra.yaml',
            'ut_parse_2ci3_combo.yaml',
            'ut_parse_2ci3a_afn.yaml',
            'ut_parse_2ci3a_pfn.yaml',
            'ut_parse_2ci3a_rfn.yaml',
            'ut_parse_2ci3a_ifn.yaml',
            'ut_parse_2ci3b_type.yaml',
            'ut_parse_2ci3b_val.yaml',
            'ut_parse_2ci4_miss.yaml',
            'ut_parse_2ci4_extra.yaml',
            'ut_parse_2ci4_combo.yaml',
            'ut_parse_2ci4a_afn.yaml',
            'ut_parse_2ci4a_pfn.yaml',
            'ut_parse_2ci4a_rfn.yaml',
            'ut_parse_2ci4a_ifn.yaml',
            'ut_parse_2ci4b_type.yaml',
            'ut_parse_2ci4b_val.yaml',
            'ut_parse_2ci4c_type.yaml',
            'ut_parse_2ci5_miss.yaml',
            'ut_parse_2ci5_extra.yaml',
            'ut_parse_2ci5_combo.yaml',
            'ut_parse_2ci5a_afn.yaml',
            'ut_parse_2ci5a_pfn.yaml',
            'ut_parse_2ci5a_rfn.yaml',
            'ut_parse_2ci5a_ifn.yaml',
            'ut_parse_2ci5b_type.yaml',
            'ut_parse_2ci5b_val.yaml',
            'ut_parse_2ci5bi_sizes.yaml',
            'ut_parse_2ci5c_type.yaml',
            'ut_parse_2ci5d_type.yaml',
            'ut_parse_2ci6_miss.yaml',
            'ut_parse_2ci6_extra.yaml',
            'ut_parse_2ci6_combo.yaml',
            'ut_parse_2ci6a_afn.yaml',
            'ut_parse_2ci6a_pfn.yaml',
            'ut_parse_2ci6a_rfn.yaml',
            'ut_parse_2ci6a_ifn.yaml',
            'ut_parse_2ci6b_type.yaml',
            'ut_parse_2ci6b_val.yaml',
            'ut_parse_2ci7_type.yaml',
            'ut_parse_2ci8_val.yaml',
            'ut_parse_2cii_order.yaml',
            'ut_parse_2ciii_repeat.yaml',
                      ]

        for fnsuff in fnsufflist:
            fn = os.path.join(self.localpath, 'testdata', 'ut', fnsuff)

            self.assertTrue(os.path.exists(fn))
            with self.assertRaises(MDFException):
                check_mode_lists(fn, False)
            with self.assertRaises(MDFException):
                check_mode_lists(fn, True)
            pass
        pass


    def test_strings_for_nonexistent_files(self):
        """
        Verify that filenames only throw errors when the files
        in the YAML don't exist if usefiles=True
        """
        fnsufflist = [
            'ut_parse_2ai3c_file.yaml',
            'ut_parse_2ai4f_file.yaml',
            'ut_parse_2ai4g_file.yaml',
            'ut_parse_2ai4h_file.yaml',
            'ut_parse_2ai4i_file.yaml',
            'ut_parse_2bi1_file.yaml',
            'ut_parse_2ci2a_file.yaml',
            'ut_parse_2ci3a_file.yaml',
            'ut_parse_2ci4a_file.yaml',
            'ut_parse_2ci5a_file.yaml',
            'ut_parse_2ci6a_file.yaml',
            'ut_parse_2ci7_file.yaml',
                      ]


        for fnsuff in fnsufflist:
            fn = os.path.join(self.localpath, 'testdata', 'ut', fnsuff)

            self.assertTrue(os.path.exists(fn))
            check_mode_lists(fn, False) # just returns, no exception
            with self.assertRaises(IOError):
                check_mode_lists(fn, True)
            pass
        pass


    def test_validation_outcomes(self):
        """
        Check that the three validation wrapper outcomes are produces as
        expected
        """

        # True if good
        fnsufflist = [
            'ut_parse_good.yaml',
            'ut_parse_good_no_decpt.yaml',
                      ]

        for fnsuff in fnsufflist:
            fn = os.path.join(self.localpath, 'testdata', 'ut', fnsuff)

            self.assertTrue(os.path.exists(fn))
            ret = validate_model_file(fn, usefiles=True, verbose=False)
            self.assertTrue(ret)
            pass

        # False if bad
        fnsufflist = [
            'ut_parse_2_miss.yaml',
            'ut_parse_2_extra.yaml',
            'ut_parse_2a_miss.yaml',
            ]

        for fnsuff in fnsufflist:
            fn = os.path.join(self.localpath, 'testdata', 'ut', fnsuff)

            self.assertTrue(os.path.exists(fn))
            ret = validate_model_file(fn, usefiles=True, verbose=False)
            self.assertFalse(ret)
            pass

        # None if it can't be evaluated with usefiles due to missing files
        fnsufflist = [
            'ut_parse_2ai3c_file.yaml',
            'ut_parse_2ai4f_file.yaml',
            ]

        for fnsuff in fnsufflist:
            fn = os.path.join(self.localpath, 'testdata', 'ut', fnsuff)

            self.assertTrue(os.path.exists(fn))
            ret = validate_model_file(fn, usefiles=True, verbose=False)
            self.assertTrue(ret is None)
            pass

        pass





if __name__ == '__main__':
    unittest.main()
