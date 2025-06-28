# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for precleaning steps
"""
import unittest

import numpy as np

from .preclean import extract_bp, normalize, eval_c

class TestExtractBP(unittest.TestCase):
    """
    Tests for extraction of per-frame bad pixel map
    """

    def test_outsize(self):
        """output size == input size"""

        sizetuplist = [(5, 8),
                       (1, 6),
                       (7, 1),
                       (1, 1)]

        for sizetup in sizetuplist:
            out = extract_bp(np.zeros(sizetup))
            self.assertTrue(out.shape == sizetup)
            pass
        pass


    def test_success_no_nan(self):
        """extracts correct map with no bad pixels"""

        im = np.zeros((5, 5))
        target = np.zeros((5, 5), dtype='bool')

        out = extract_bp(im)
        self.assertTrue((out == target).all())
        pass


    def test_success_some_nan(self):
        """extracts correct map with some bad pixels"""

        im = np.zeros((5, 5))
        im[1, 1] = np.nan
        im[2, 3] = np.nan
        target = np.zeros((5, 5), dtype='bool')
        target[1, 1] = True
        target[2, 3] = True

        out = extract_bp(im)
        self.assertTrue((out == target).all())
        pass


    def test_success_all_nan(self):
        """extracts correct map with all bad pixels"""

        im = np.nan*np.zeros((5, 5))
        target = np.ones((5, 5), dtype='bool')

        out = extract_bp(im)
        self.assertTrue((out == target).all())
        pass


    def test_invalid_im(self):
        """verify invalid values caught"""

        for im in [np.ones((3, 3, 3)), np.ones((3,)), 'txt', None, 0, 1j]:
            with self.assertRaises(TypeError):
                extract_bp(im)
            pass
        pass



class TestNormalize(unittest.TestCase):
    """
    Tests for flux normalization of a HOWFSC GITL frame

    check invalid inputs
    check exact values
    """

    def test_exact(self):
        """Verify normalization calculation as expected"""
        im = np.eye(3)
        peakflux = 2
        exptime = 8
        target = np.array([[1/16, 0, 0],
                           [0, 1/16, 0],
                           [0, 0, 1/16]])

        out = normalize(im, peakflux, exptime)
        self.assertTrue((out == target).all())
        pass


    def test_exact_nan(self):
        """Verify normalization calculation as expected with NaNs present"""
        im = np.eye(3)
        im[1, 0] = np.nan
        im[1, 1] = np.nan
        peakflux = 2
        exptime = 8
        target = np.array([[1/16, 0, 0],
                           [np.nan, np.nan, 0],
                           [0, 0, 1/16]])

        out = normalize(im, peakflux, exptime)
        good = ~np.isnan(out)
        self.assertTrue((out[good] == target[good]).all())
        # nan stay nans too
        self.assertTrue(np.isnan(out[~good]).all())
        pass


    def test_invalid_im(self):
        """verify invalid values caught"""
        peakflux = 2
        exptime = 8

        for x in [np.ones((3, 3, 3)), np.ones((3,)), 'txt', None, 0, 1j, -1]:
            with self.assertRaises(TypeError):
                normalize(x, peakflux, exptime)
            pass
        pass


    def test_invalid_peakflux(self):
        """verify invalid values caught"""
        im = np.ones((5, 5))
        exptime = 8

        for x in [np.ones((3, 3, 3)), np.ones((3,)), 'txt', None, 0, 1j, -1]:
            with self.assertRaises(TypeError):
                normalize(im, x, exptime)
            pass
        pass


    def test_invalid_exptime(self):
        """verify invalid values caught"""
        im = np.ones((5, 5))
        peakflux = 8

        for x in [np.ones((3, 3, 3)), np.ones((3,)), 'txt', None, 0, 1j, -1]:
            with self.assertRaises(TypeError):
                normalize(im, peakflux, x)
            pass
        pass


class TestEvalC(unittest.TestCase):
    """
    Tests for function to combine a mean total contrast from data frames
    """


    def setUp(self):
        # Mean should be 6e-8 without any changes
        self.target = 6e-8
        self.nimlist = [np.array([[1e-8, 2e-8], [4e-8, 8e-8]]),
                        np.array([[0.5e-8, np.nan], [np.nan, 0.0625e-8]])]
        self.dhlist = [np.array([[True, True], [True, False]]),
                       np.array([[True, True], [True, False]])]
        self.n2clist = [np.array([[6, 3], [1.5, 5]]),
                        np.array([[12, 7], [11, 13]])]
        pass


    def test_success(self):
        """Verify good inputs complete"""
        eval_c(self.nimlist, self.dhlist, self.n2clist)
        pass


    def test_len1_success(self):
        """Verify successful completion even when lists/arrays have dim 1"""
        nim = [np.array([[1e-8]])]
        dh = [np.array([[True]])]
        n2c = [np.array([[2]])]
        eval_c(nim, dh, n2c)
        pass


    def test_expected_results(self):
        """Verify results as expected"""
        tol = 1e-13
        out = eval_c(self.nimlist, self.dhlist, self.n2clist)
        self.assertTrue(np.max(np.abs(out - self.target)) < tol)
        pass


    def test_expected_results_len1(self):
        """Verify results as expected even when lists/arrays have dim 1"""
        tol = 1e-13

        # use docstring example
        nim = [np.array([[5e-9]])]
        dh = [np.array([[True]])]
        n2c = [np.array([[2]])]
        target = 1e-8

        out = eval_c(nim, dh, n2c)
        self.assertTrue(np.max(np.abs(out - target)) < tol)
        pass


    def test_not_lists(self):
        """verify invalud results caught"""
        xlist = [1, None, 'txt', (5, 5)]

        # nim
        for x in xlist:
            with self.assertRaises(TypeError):
                eval_c(x, self.dhlist, self.n2clist)
            pass

        # dh
        for x in xlist:
            with self.assertRaises(TypeError):
                eval_c(self.nimlist, x, self.n2clist)
            pass

        # n2c
        for x in xlist:
            with self.assertRaises(TypeError):
                eval_c(self.nimlist, self.dhlist, x)
            pass


    def test_wrong_list_length(self):
        """verify invalid results caught"""
        longlist = [np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))]
        longlistb = [np.ones((2, 2)).astype('bool'),
                     np.ones((2, 2)).astype('bool'),
                     np.ones((2, 2)).astype('bool')]
        shortlist = [np.ones((2, 2))]
        shortlistb = [np.ones((2, 2)).astype('bool')]

        # nim
        with self.assertRaises(TypeError):
            eval_c(longlist, self.dhlist, self.n2clist)
        with self.assertRaises(TypeError):
            eval_c(shortlist, self.dhlist, self.n2clist)


        # dh
        with self.assertRaises(TypeError):
            eval_c(self.nimlist, longlistb, self.n2clist)
        with self.assertRaises(TypeError):
            eval_c(self.nimlist, shortlistb, self.n2clist)

        # n2c
        with self.assertRaises(TypeError):
            eval_c(self.nimlist, self.dhlist, longlist)
        with self.assertRaises(TypeError):
            eval_c(self.nimlist, self.dhlist, shortlist)

        pass


    def test_lists_have_non_2d_arrays(self):
        """Verify invalid inputs caught"""
        test = [np.ones((2, 2)), np.ones((2,))]
        testb = [np.ones((2, 2)).astype('bool'),
                 np.ones((2,)).astype('bool')]

        # nim
        with self.assertRaises(TypeError):
            eval_c(test, self.dhlist, self.n2clist)

        # dh
        with self.assertRaises(TypeError):
            eval_c(self.nimlist, testb, self.n2clist)

        #n2c
        with self.assertRaises(TypeError):
            eval_c(self.nimlist, self.dhlist, test)

        pass


    def test_list_2d_arrays_are_wrong_size(self):
        """Verify invalid inputs caught"""
        test = [np.ones((2, 2)), np.ones((2, 3))]
        testb = [np.ones((2, 2)).astype('bool'),
                 np.ones((2, 3)).astype('bool')]

        # nim
        with self.assertRaises(TypeError):
            eval_c(test, self.dhlist, self.n2clist)

        # dh
        with self.assertRaises(TypeError):
            eval_c(self.nimlist, testb, self.n2clist)
        #n2c
        with self.assertRaises(TypeError):
            eval_c(self.nimlist, self.dhlist, test)

        pass


    def test_nonbool_inputs_to_bool_arrays(self):
        """Verify invalid inputs caught"""
        testb = [np.ones((2, 2)), np.ones((2, 2))]

        # dh
        with self.assertRaises(TypeError):
            eval_c(self.nimlist, testb, self.n2clist)

        pass


    def test_no_valid_pixels(self):
        """
        Test that giving no valid pixels correctly is caught as a divide-by-
        zero case.

        Test will have no valid pixels between dh and bp, but each is fine
        individually.
        """

        nimlist0 = [np.array([[1, np.nan], [np.nan, 1]]),
                    np.array([[1, np.nan], [np.nan, 1]])]
        dhlist0 = [np.array([[False, True], [True, False]]),
                   np.array([[False, True], [True, False]])]

        with self.assertRaises(ZeroDivisionError):
            eval_c(nimlist0, dhlist0, self.n2clist)

        pass






if __name__ == '__main__':
    unittest.main()
