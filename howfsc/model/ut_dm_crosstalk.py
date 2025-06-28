# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for dm_crosstalk.py
"""

import os
import unittest
import tempfile
import numpy as np
import astropy.io.fits as fits


from .dm_crosstalk import CDMCrosstalk
from .dm_crosstalk import dm_crosstalk_fits_to_yaml
from .dm_crosstalk import dm_crosstalk

TESTPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')

class TestCDmCrosstalk(unittest.TestCase):
    """
    Unit tests for CDMCrosstalk class constructor
    """
    def setUp(self):
        """Defaults for creating an object"""
        self.yaml_fn = os.path.join(TESTPATH, 'ut', 'ut_dm_crosstalk.yaml')
        self.list_xtalk_arrays = [
            fits.getdata(os.path.join(TESTPATH, 'ut', 'ut_dm_crosstalk.fits')),
        ]
        self.list_off_row_col = [[1, 0]]

    # Constructor
    def test_success(self):
        """successfully create object with either and both methods"""
        # using yaml
        CDMCrosstalk(yaml_fn=self.yaml_fn)
        # using list of arrays and offsets
        CDMCrosstalk(list_xtalk_arrays=self.list_xtalk_arrays,
                      list_off_row_col=self.list_off_row_col)
        # using both
        CDMCrosstalk(yaml_fn=self.yaml_fn,
                     list_xtalk_arrays=self.list_xtalk_arrays,
                     list_off_row_col=self.list_off_row_col)

    # for yaml_fn, constructor simply calls read_crosstalk_yaml(), tested below

    def test_invalid_list_xtalk(self):
        """test conformity of list inputs"""
        # number of offsets does not match number of xtalk arrays
        with self.assertRaises(ValueError):
            CDMCrosstalk(list_xtalk_arrays=self.list_xtalk_arrays,
                          list_off_row_col=self.list_off_row_col+[[0, 1]])
            CDMCrosstalk(list_xtalk_arrays=self.list_xtalk_arrays*2,
                          list_off_row_col=self.list_off_row_col)


        # if either one is not a list
        for notlist in [1, {'a':3}, 'abc', np.ones((48, 48))]:
            with self.assertRaises(TypeError):
                CDMCrosstalk(list_xtalk_arrays=self.list_xtalk_arrays,
                              list_off_row_col=notlist)
                CDMCrosstalk(list_xtalk_arrays=notlist,
                              list_off_row_col=self.list_off_row_col)

class TestAdd_xtalk_diagonal(unittest.TestCase):
    """
    Unit tets for class method add_xtalk_diagonal
    """
    def setUp(self):
        """Defaults for creating valid object"""

        self.yaml_fn = os.path.join(TESTPATH, 'ut', 'ut_dm_crosstalk.yaml')
        self.xtalk_array = fits.getdata(os.path.join(
            TESTPATH, 'ut', 'ut_dm_crosstalk.fits'))
        self.kdiag = 1
        self.nact = self.xtalk_array.shape[0]

        # add_xtalk_diagonal() assumes .HC_sparse has already been initialized
        self.crosstalk = CDMCrosstalk()
        self.crosstalk.init_HC_sparse(self.nact)

    def test_success(self):
        """
        test success for both empty and already-initialized crosstalk object
        """
        # test add to empty
        crosstalk = CDMCrosstalk()
        # must explicitly initialize HC_sparse before adding diagonal to empty
        crosstalk.init_HC_sparse(self.nact)
        crosstalk.add_xtalk_diagonal(self.xtalk_array.flatten(), self.kdiag)

        # test add new diagonal to existing
        crosstalk.add_xtalk_diagonal(self.xtalk_array.flatten(), self.kdiag+1)
        # verify there are two diagonals
        self.assertTrue(len(crosstalk.list_k_diag) == 2)


    def test_valid_xtalk(self):
        """test invalid xtalk array input raises error"""

        # not 1-d
        for xtalk in [np.ones((5, 5)), 1, 'abc']:
            with self.assertRaises(TypeError):
                self.crosstalk.add_xtalk_diagonal(xtalk, self.kdiag)

        # 1-d but not real
        for xtalk in [1j*np.ones((5,)), ]:
            with self.assertRaises(TypeError):
                self.crosstalk.add_xtalk_diagonal(xtalk, self.kdiag)

        # shape != (nact*nact,)
        with self.assertRaises(ValueError):
            # first create valid object to establish nact
            crosstalk = CDMCrosstalk(list_xtalk_arrays=[self.xtalk_array,],
                                      list_off_row_col=[[1, 1]])
            crosstalk.add_xtalk_diagonal(np.ones((crosstalk.nact+1,)),
                                         self.kdiag)


    def test_valid_k_diag(self):
        """k_diag scalar integer, in valid range"""

        # test type(k diag)
        for kdiag in [1j, 'abc', np.array([1, 2, 3])]:
            with self.assertRaises(TypeError):
                self.crosstalk.add_xtalk_diagonal(self.xtalk_array.flatten(),
                                                  kdiag)

        # test value of k diag
        for kdiag in [self.nact*self.nact, -self.nact*self.nact]:
            with self.assertRaises(ValueError):
                self.crosstalk.add_xtalk_diagonal(self.xtalk_array.flatten(),
                                                  kdiag)

    def test_invalid_HC_sparse(self):
        """
        add_xtalk_diagonal() assumes .HC_sparse has already been initialized
        """

        # un-initialized:
        crosstalk = CDMCrosstalk()
        with self.assertRaises(ValueError):
            crosstalk.add_xtalk_diagonal(self.xtalk_array.flatten(),
                                         self.kdiag)


class TestAdd_xtalk_array(unittest.TestCase):
    """Unit tets for class method add_xtalk_array """
    def setUp(self):
        """Defaults for creating valid object"""

        self.yaml_fn = os.path.join(TESTPATH, 'ut', 'ut_dm_crosstalk.yaml')
        self.xtalk_array = fits.getdata(os.path.join(
            TESTPATH, 'ut', 'ut_dm_crosstalk.fits'))
        self.off_i_j = [0, 1]
        self.crosstalk = CDMCrosstalk() # empty

    def test_success(self):
        """test successful add array for both initialized and empty cases"""

        # un-initialized
        crosstalk = CDMCrosstalk()
        crosstalk.add_xtalk_array(self.xtalk_array, self.off_i_j)

        # already initialized with an existing diagonal
        crosstalk = CDMCrosstalk(yaml_fn=self.yaml_fn)
        crosstalk.add_xtalk_array(self.xtalk_array, self.off_i_j)

    def test_invalid_array(self):
        """test invalid arrays raise errors"""

        for arr in [np.ones((5,)), 1j*np.ones((5, 5)), 1j, 5, 'abc',
                    np.ones((5, 4))]:
            with self.assertRaises(TypeError):
                self.crosstalk.add_xtalk_array(arr, self.off_i_j)

        # create an instance with a valid array,
        # then test a 2nd array of different size fails
        crosstalk = CDMCrosstalk(yaml_fn=self.yaml_fn)
        with self.assertRaises(ValueError):
            nact = crosstalk.nact
            crosstalk.add_xtalk_array(np.ones((nact-1, nact-1)), self.off_i_j)


    def test_invalid_off(self):
        """argument off_i_j must be a list of 2 integers"""

        for off in [np.ones((5,)), [1j, 1], 'abc', [1, 1, 1], np.ones((2, 2))]:
            with self.assertRaises(TypeError):
                self.crosstalk.add_xtalk_array(self.xtalk_array, off)

    def test_correct_diagonal(self):
        """check array is added to the correct diagonal"""

        self.crosstalk.add_xtalk_array(self.xtalk_array, self.off_i_j)
        k = self.crosstalk.k_diag(self.off_i_j[0], self.off_i_j[1])

        # k_diag is the last one added
        self.assertTrue(k == self.crosstalk.list_k_diag[-1])

        # get the diagonal from HC_sparse
        diag_data = self.crosstalk.HC_sparse.diagonal(k)

        # must equal flattened input array, and input array is
        # offset, see add_xtalk_diagonal()
        xtalk = self.xtalk_array.flatten()
        if k < 0:
            xtalk = xtalk[-k:]
        elif k > 0:
            xtalk = xtalk[:-k]

        # test
        self.assertTrue(np.all(diag_data == xtalk))


class TestWrite_crosstalk_yaml(unittest.TestCase):
    """
    Unit tests for class method write_crosstalk_yaml
    """
    # not sure what to test, except that write then read reproduces the object
    def setUp(self):
        """Defaults for creating valid object"""
        self.yaml_fn = os.path.join(TESTPATH, 'ut', 'ut_dm_crosstalk.yaml')
        self.crosstalk = CDMCrosstalk(self.yaml_fn)

    def testWriteRead(self):
        """test that write then read reproduces"""

        # create a temporary directory to write/read
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_yaml_fn = os.path.join(tmpdirname, 'tmp.yaml')
            self.crosstalk.write_crosstalk_yaml(tmp_yaml_fn)

            tmpcrosstalk = CDMCrosstalk(yaml_fn=tmp_yaml_fn)

            self.assertTrue(
                (self.crosstalk.HC_sparse - tmpcrosstalk.HC_sparse).nnz == 0
            )

        # on exit the context, the temporary directory and all its contents
        # are removed


class TestRead_crosstalk_yaml(unittest.TestCase):
    """
    Unit tests for class method read_crosstalk_yaml
    """
    def setUp(self):
        """Defaults for creating an object"""
        self.crosstalk = CDMCrosstalk() # empty crosstalk

    def test_yaml_keys(self):
        """test conformity to required form and keywords"""

        for bad_yaml in ['ut_dm_crosstalk_no_crosstalk.yaml',
                         'ut_dm_crosstalk_no_off_i.yaml',
                         'ut_dm_crosstalk_no_kdiag.yaml',
                         'ut_dm_crosstalk_no_off_j.yaml',
                         'ut_dm_crosstalk_no_nact.yaml',]:
            with self.assertRaises(ValueError):
                self.crosstalk.read_crosstalk_yaml(
                    os.path.join(TESTPATH, 'ut', bad_yaml))

    def test_yaml_2offsets(self):
        """test success read yaml with multiple diagonals"""

        self.crosstalk.read_crosstalk_yaml(
            os.path.join(TESTPATH, 'ut', 'ut_dm_crosstalk_2offsets.yaml'))

        # verify there are two diagonals
        self.assertTrue(len(self.crosstalk.list_k_diag) == 2)

    def test_yaml_badsize(self):
        """test size of crosstalk data is consistent with nact"""

        with self.assertRaises(ValueError):
            self.crosstalk.read_crosstalk_yaml(
                os.path.join(TESTPATH, 'ut', 'ut_dm_crosstalk_badsize.yaml'))

    def test_yaml_diffsize(self):
        """if yaml has multiple crosstalk arrays, they must have same size"""

        with self.assertRaises(ValueError):
            self.crosstalk.read_crosstalk_yaml(
                os.path.join(TESTPATH, 'ut', 'ut_dm_crosstalk_diffsize.yaml'))


class TestCrosstalk_forward_backward(unittest.TestCase):
    """Unit test for forward and backward application of crosstalk"""
    def setUp(self):
        """Defaults for creating valid object"""
        self.xtalk_array = fits.getdata(
            os.path.join(TESTPATH, 'ut', 'ut_DM2_crosstalk.fits')
        )
        self.ij_off = [1, 0]

        self.crosstalk = CDMCrosstalk(list_xtalk_arrays=[self.xtalk_array,],
                                       list_off_row_col=[self.ij_off])

        # read example dm command and dm actual
        # this specific example of command and actual are from John Krist
        # goes with crosstalk array saved in ut_DM2_crosstalk.fits
        self.command_dm2 = fits.getdata(
            os.path.join(TESTPATH, 'ut', 'ut_DM2_commanded_strokes_m.fits'))
        self.actual_dm2 = fits.getdata(
            os.path.join(TESTPATH, 'ut', 'ut_DM2_actual_strokes_m.fits'))

    def test_forward(self):
        """
        test forward application of dm actuator crosstalk is correct

        HC_sparse * command  == command + xtalk *
                   np.roll(np.roll(command, -offset_j, axis=1),
                           -offset_i, axis=0)

        note: np.roll(row) and np.roll(col) are commutative
        """

        # test that we get the same answer as John for the given example
        # saved command and actual are order 1e-9 meters, thus set atol
        actual_test = self.crosstalk.crosstalk_forward(self.command_dm2)
        self.assertTrue(np.allclose(actual_test, self.actual_dm2, atol=1e-10))

        # test that the sparse matrix multiplication gives the right
        # answer for any actuator offset. This tests that we always assign
        # to the correct diagonal. Explicit offset of the crosstalk
        # matrix using np.roll() is "truth".
        for ij_off in [[1, 0], [0, 1], [-1, 0], [0, -1],
                       [2, 1], [-1, 2], [-1, 1], [1, -2]]:
            # create dm crosstalk object & sparse matrix for this offset
            crosstalk = CDMCrosstalk(list_xtalk_arrays=[self.xtalk_array,],
                                     list_off_row_col=[ij_off])
            # and apply to command
            actual = crosstalk.crosstalk_forward(self.command_dm2)

            # definition of DM actuator crosstalk:
            command_roll = np.roll(np.roll(self.command_dm2, -ij_off[0],
                                           axis=0),
                                   -ij_off[1], axis=1)

            actual_roll = self.command_dm2 + self.xtalk_array * command_roll

            # check
            self.assertTrue(np.allclose(actual_roll, actual, atol=1e-10))

    def test_forward_backward(self):
        """test backward is inverse of foward"""

        # test for a bunch of offsets, see above test_forward()
        for ij_off in [[1, 0], [0, 1], [-1, 0], [0, -1],
                       [2, 1], [-1, 2], [-1, 1], [1, -2]]:

            # create dm crosstalk object & sparse matrix for this offset
            xtalk_tmp = CDMCrosstalk(list_xtalk_arrays=[self.xtalk_array,],
                                     list_off_row_col=[ij_off])

            # and apply to command
            actual = xtalk_tmp.crosstalk_forward(self.command_dm2)

            # and back again
            command_test = xtalk_tmp.crosstalk_backward(actual)

            self.assertTrue(np.allclose(command_test, self.command_dm2,
                                        atol=1e-10))

        # test cases of crosstalk in multiple directions (multiple diagonals)
        for ij_off in [[0, 1], [-1, 0], [0, -1], [1, 1]]:
            self.crosstalk.add_xtalk_array(self.xtalk_array, ij_off)

        # now HC_sparse has 5 diagonals
        # apply to command
        actual = self.crosstalk.crosstalk_forward(self.command_dm2)

        # and back again
        command_test = self.crosstalk.crosstalk_backward(actual)

        self.assertTrue(np.allclose(command_test, self.command_dm2,
                                    atol=1e-10))

class TestCrosstalk_fits_to_yaml(unittest.TestCase):
    """Unit test for writing crosstalk yaml from given fits"""
    def setUp(self):
        """Defaults for creating valid object"""
        self.list_xtalk_fits = [os.path.join(TESTPATH, 'ut_DM2_crosstalk.fits')]
        self.list_off = [[1, 0]]
        self.fd, self.yaml_save_fn = tempfile.mkstemp()
        self.yaml_correct_fn = os.path.join(TESTPATH, 'ut_dm_crosstalk.yaml')

    def testSuccess(self):
        """run the routine and check result matches"""
        obj_test = dm_crosstalk_fits_to_yaml(self.list_xtalk_fits,
                                             self.list_off,
                                             self.yaml_save_fn)

        # check that data in saved yaml matches correct and returned obj_test
        obj_correct = CDMCrosstalk(yaml_fn=self.yaml_correct_fn)
        obj_yaml_test = CDMCrosstalk(yaml_fn=self.yaml_save_fn)

        
        # check that all are the same
        self.assertTrue(
            (obj_test.HC_sparse - obj_correct.HC_sparse).nnz == 0
        )
        self.assertTrue(
            (obj_yaml_test.HC_sparse - obj_correct.HC_sparse).nnz == 0
        )

        self.assertTrue(obj_test.nact == obj_correct.nact)
        self.assertTrue(obj_yaml_test.nact == obj_correct.nact)

    def tearDown(self):
        """cleanup """
        os.close(self.fd)
        os.unlink(self.yaml_save_fn)


class TestCrosstalk_dm_crosstalk(unittest.TestCase):
    """Unit test for simple interface"""
    def setUp(self):
        """Defaults for creating valid object"""
        self.xtalk_fits = os.path.join(TESTPATH, 'ut_DM2_crosstalk.fits')
        self.config_yaml = os.path.join(TESTPATH, 'ut_crosstalk_config_basic.yaml')
        self.fd, self.yaml_save_fn = tempfile.mkstemp()

        self.yaml_correct_fn = os.path.join(TESTPATH, 'ut_dm_crosstalk.yaml')
        self.yaml_flipx_fn = os.path.join(TESTPATH, 'ut', 'ut_dm_crosstalk_flipx.yaml')
        
    def testSuccess(self):
        """run the routine and check result matches"""
        obj_test = dm_crosstalk(self.xtalk_fits,
                                self.config_yaml,
                                yaml_save_fn=self.yaml_save_fn)

        # check that data in saved yaml matches correct and returned obj_test
        obj_correct = CDMCrosstalk(yaml_fn=self.yaml_correct_fn)
        obj_yaml_test = CDMCrosstalk(yaml_fn=self.yaml_save_fn)

        # check that all are the same
        self.assertTrue(
            (obj_test.HC_sparse - obj_correct.HC_sparse).nnz == 0
        )
        self.assertTrue(
            (obj_yaml_test.HC_sparse - obj_correct.HC_sparse).nnz == 0
        )

        self.assertTrue(obj_test.nact == obj_correct.nact)
        self.assertTrue(obj_yaml_test.nact == obj_correct.nact)

    def testFlipX(self):
        """run with flip-x True and check against 'correct orientation'"""

        # this config has flip-x = True, which is actually the case for CGI
        config_yaml = os.path.join(TESTPATH, 'ut', 'ut_crosstalk_config_flipx.yaml')

        obj_test = dm_crosstalk(self.xtalk_fits,
                                config_yaml,
                                yaml_save_fn=self.yaml_save_fn)
        
        # check that data in saved yaml matches correct and returned obj_test
        obj_correct = CDMCrosstalk(yaml_fn=self.yaml_flipx_fn)
        obj_yaml_test = CDMCrosstalk(yaml_fn=self.yaml_save_fn)

        # check that all are the same
        self.assertTrue(
            (obj_test.HC_sparse - obj_correct.HC_sparse).nnz == 0
        )
        self.assertTrue(
            (obj_yaml_test.HC_sparse - obj_correct.HC_sparse).nnz == 0
        )

        self.assertTrue(obj_test.nact == obj_correct.nact)
        self.assertTrue(obj_yaml_test.nact == obj_correct.nact)

    def testRoll(self):
        """run with non-zero roll"""
        # this config has flip-x = True, which is actually the case for CGI
        config_yaml = os.path.join(TESTPATH, 'ut_crosstalk_config_rollrow.yaml')

        obj_test = dm_crosstalk(self.xtalk_fits,
                                config_yaml,
                                yaml_save_fn=self.yaml_save_fn)

        # read fits, manually apply roll and flip-x, create "truth" object
        fits_array = fits.getdata(os.path.join(TESTPATH, self.xtalk_fits))
        row_col = [1, 0]
        # apply flip-x
        fits_array = np.fliplr(fits_array)
        # apply roll
        fits_array = np.roll(fits_array, [1, 0], axis=(0, 1))
        obj_correct = CDMCrosstalk(list_xtalk_arrays=[fits_array],
                                   list_off_row_col=[row_col])

        # check that data in saved yaml matches correct and returned obj_test
        obj_yaml_test = CDMCrosstalk(yaml_fn=self.yaml_save_fn)

        # check that all are the same
        self.assertTrue(
            (obj_test.HC_sparse - obj_correct.HC_sparse).nnz == 0
        )
        self.assertTrue(
            (obj_yaml_test.HC_sparse - obj_correct.HC_sparse).nnz == 0
        )

        self.assertTrue(obj_test.nact == obj_correct.nact)
        self.assertTrue(obj_yaml_test.nact == obj_correct.nact)
        
    def tearDown(self):
        """cleanup """
        os.close(self.fd)
        os.unlink(self.yaml_save_fn)

if __name__ == '__main__':
    unittest.main()
