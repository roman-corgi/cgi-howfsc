# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for subarray.py
"""
import unittest

import numpy as np

from .subarray import centered_nonzero

class TestCenteredNonzero(unittest.TestCase):
    """
    Unit test suite for centered_nonzero()
    """

    def test_size_combinations(self):
        """Verify even/odd maintained"""
        arrlist = [np.ones((5, 5)), np.ones((6, 6)), np.ones((6, 5)),
                   np.ones((5, 6)), np.ones((6, 1)), np.ones((1, 6))]
        arrlist.append(np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0]]))
        arrlist.append(np.array([[0, 0, 0],
                                 [0, 1, 1],
                                 [0, 0, 0]]))
        arrlist.append(np.array([[0, 0, 0, 0],
                                 [0, 1, 1, 0],
                                 [0, 0, 0, 0]]))
        arrlist.append(np.array([[0, 1, 0, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]]))

        for arr in arrlist:
            subarr = centered_nonzero(arr)
            self.assertTrue(subarr.shape[0] % 2 == arr.shape[0] % 2)
            self.assertTrue(subarr.shape[1] % 2 == arr.shape[1] % 2)
            pass
        pass


    def test_fullsize_combinations(self):
        """All of these should reproduce themselves"""
        arrlist = [np.eye(5), np.eye(6), np.ones((6, 5)), np.ones((5, 6)),
                   np.ones((1, 1)), np.ones((6, 1)), np.ones((1, 6))]
        arrlist.append(np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0]]))
        arrlist.append(np.array([[0, 0, 0],
                                 [0, 1, 1],
                                 [0, 1, 0]]))
        arrlist.append(np.array([[0, 0, 1, 1],
                                 [0, 1, 1, 0],
                                 [0, 0, 0, 0]]))
        arrlist.append(np.array([[0, 1, 0, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]]))

        for arr in arrlist:
            subarr = centered_nonzero(arr)
            self.assertTrue(subarr.shape == arr.shape)
            self.assertTrue((subarr == arr).all())
            pass
        pass

    def test_subarraying(self):
        """Verify a few point matrices get shrunk correctly"""
        arrlist = []
        outlist = []
        arrlist.append(np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]))
        outlist.append(np.array([[1]]))
        arrlist.append(np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0]]))
        outlist.append(np.array([[0],
                                 [1],
                                 [1]]))
        arrlist.append(np.array([[0, 0, 0],
                                 [0, 1, 1],
                                 [0, 0, 0]]))
        outlist.append(np.array([[0, 1, 1]]))
        arrlist.append(np.array([[0, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 0]]))
        outlist.append(np.array([[1, 0]]))
        arrlist.append(np.array([[0, 0, 0, 0],
                                 [0, 1, 1, 0],
                                 [0, 1, 1, 0]]))
        outlist.append(np.array([[0, 0],
                                 [1, 1],
                                 [1, 1]]))
        arrlist.append(np.array([[0, 0, 0, 0],
                                 [1, 1, 1, 0],
                                 [0, 0, 0, 0]]))
        outlist.append(np.array([[1, 1, 1, 0]]))

        for index, arr in enumerate(arrlist):
            subarr = centered_nonzero(arr)
            self.assertTrue((subarr == outlist[index]).all())
            pass
        pass

    def test_all_zeros(self):
        """Verify it handles the blank case"""
        blank = np.array([[]])

        arrlist = [np.zeros((3, 3)), np.zeros((6, 6)), np.zeros((7, 4)),
                   np.zeros((0, 0))]
        for arr in arrlist:
            subarr = centered_nonzero(arr)
            self.assertTrue((blank == subarr).all())
            pass
        pass

    def test_complex_input(self):
        """
        It should work correctly for real or complex numbers, either as zeros
        or nonzeros
        """

        arrlist = []
        outlist = []
        arrlist.append(np.array([[0j, 0j, 0j, 0j],
                                 [0j, 1j, 0j, 1j],
                                 [0j, 0j, 0j, 0j]]))
        outlist.append(np.array([[0j, 1j, 0j, 1j]]))
        arrlist.append(np.array([[0j, 0j, 0j, 0j],
                                 [0, 1, 0, 1],
                                 [0j, 0j, 0j, 0j]]))
        outlist.append(np.array([[0, 1, 0, 1]]))

        for index, arr in enumerate(arrlist):
            subarr = centered_nonzero(arr)
            self.assertTrue((subarr == outlist[index]).all())
            pass
        pass

    def test_arr_2darray(self):
        """Verify throws the correct Exception when handed incorrect inputs"""
        for perr in ['arr', [0, 1, 0], np.ones((5,)), 1]:
            with self.assertRaises(TypeError):
                centered_nonzero(arr=perr)
                pass
            pass
        pass



if __name__ == '__main__':
    unittest.main()
