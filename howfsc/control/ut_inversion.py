# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#pylint: disable=unsubscriptable-object
"""
Unit tests for matrix pseudoinversion routines
"""

import unittest
import os

import numpy as np
from scipy.sparse import csr_matrix, eye as speye

from howfsc.control.inversion import jac_solve, inv_to_dm
from howfsc.control.inversion import valid_methods # list
from howfsc.model.mode import CoronagraphMode

cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'model', 'testdata', 'widefov', 'widefov.yaml')
cfg = CoronagraphMode(cfgpath)

def _jact2d(jac):
    """Make a 3D jac into a 2D for testing"""
    rjac = np.zeros((jac.shape[1], jac.shape[0]*jac.shape[2]))
    for k in range(jac.shape[0]):
        rjac[:, k*jac.shape[2]:(k+1)*jac.shape[2]] = jac[k, :, :]
        pass
    return rjac


class TestJacSolve(unittest.TestCase):
    """
    Test that getting the next DM setting via Cholesky/QR decomposition behaves
    as expected

    We'll use a handful of small arrays with known decompositions
    """

    def test_pure_inversion(self):
        """
        Test on an invertible square matrix with beta of -infinity
        and unit weights, which should be numerically-equivalent to inverting
        the matrix.
        """

        tol = 1e-11

        A = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])
        Ac = np.zeros((2, A.shape[1], A.shape[0]))
        Ac[0] = A.T # expects separated real and imaginary parts

        wdm = speye(4, format='csr')
        b = 10.*np.ones((4,))
        target = -np.ones((4, 1))

        jtwj = np.dot(_jact2d(Ac), _jact2d(Ac).T)

        for method in valid_methods:
            x = jac_solve(jac=Ac, e0=b, beta=-np.inf, wdm=wdm,
                          we0=np.ones((4,)), bp=np.zeros((4,)),
                          jtwj=jtwj, method=method)
            self.assertTrue((np.abs(x -  target) < tol).all())
            pass
        pass



    def test_no_weights(self):
        """
        Test on an invertible square matrix and unit weights

        Max singular value of A is 10, so beta = 0 => lam = 10

        """

        tol = 1e-11

        A = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])
        Ac = np.zeros((2, A.shape[1], A.shape[0]))
        Ac[0] = A.T # expects separated real and imaginary parts
        wdm = speye(4, format='csr')

        b = 10.*np.ones((4,))
        target = -0.5*np.ones((4, 1))

        jtwj = np.dot(_jact2d(Ac), _jact2d(Ac).T)

        for method in valid_methods:
            x = jac_solve(jac=Ac, e0=b, beta=0, wdm=wdm,
                          we0=np.ones((4,)), bp=np.zeros((4,)),
                          jtwj=jtwj, method=method)
            self.assertTrue((np.abs(x -  target) < tol).all())
            pass
        pass

    def test_wdm(self):
        """
        Test on an invertible square matrix and a DM weight matrix

        Use unitary matrix as weight and premultiply by transpose, so should
        get same results as unweighted case

        """

        tol = 1e-11

        A = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])
        Ac = np.zeros((2, A.shape[1], A.shape[0]))
        rot = np.array([[0, 1, 0, 0], [-1, 0, 0, 0],
                        [0, 0, 1, 0], [0, 0, 0, 1]])
        Ac[0] = np.dot(rot, A.T)

        wdm = csr_matrix(rot)

        b = 10.*np.ones((4,))
        target = -0.5*np.ones((4, 1))

        jtwj = np.dot(_jact2d(Ac), _jact2d(Ac).T)

        for method in valid_methods:
            x = jac_solve(jac=Ac, e0=b, beta=0, wdm=wdm,
                          we0=np.ones((4,)), bp=np.zeros((4,)),
                          jtwj=jtwj, method=method)
            self.assertTrue((np.abs(x -  target) < tol).all())
            pass
        pass


    def test_we0(self):
        """
        Test on an invertible square matrix and pixel weights

        using beta = -inf makes getting exact solution easier; nothing special

        """

        tol = 1e-10

        A = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])
        Ac = np.zeros((2, A.shape[1], A.shape[0]))
        Ac[0] = A.T # expects separated real and imaginary parts
        wdm = speye(4, format='csr')

        b = np.array([40, 30, 20, 10])
        target = -1.0*np.array([[-5], [5], [5], [5]])
        we0 = np.array([4, 3, 2, 1])

        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(Ac), np.diag(cwe0**2)),
                      _jact2d(Ac).T)

        for method in valid_methods:
            x = jac_solve(jac=Ac, e0=b, beta=-np.inf, wdm=wdm,
                          we0=we0, bp=np.zeros((4,)), jtwj=jtwj,
                          method=method)
            self.assertTrue((np.abs(x -  target) < tol).all())
            pass
        pass

    def test_bp(self):
        """
        Test on the we0 matrix, with an extra row and a garbage pixel which we
        zero out

        using beta = -inf makes getting exact solution easier; nothing special

        """

        tol = 1e-10

        A = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2],
                      [4, 1, 2, 3], [5, 0, 4, 5]])
        Ac = np.zeros((2, A.shape[1], A.shape[0]))
        Ac[0] = A.T # expects separated real and imaginary parts
        wdm = speye(4, format='csr')

        b = np.array([40, 30, 20, 10, 1000])
        target = -1.0*np.array([[-5], [5], [5], [5]])
        we0 = np.array([4, 3, 2, 1, 1])

        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(Ac), np.diag(cwe0**2)),
                      _jact2d(Ac).T)


        bp = np.array([0, 0, 0, 0, 1])
        for method in valid_methods:
            x = jac_solve(jac=Ac, e0=b, beta=-np.inf, wdm=wdm,
                          we0=we0, bp=bp, jtwj=jtwj, method=method)
            self.assertTrue((np.abs(x -  target) < tol).all())
            pass
        pass


    def test_wdm_zeros(self):
        """
        Test works as expected when wdm is not full rank but
        regularization present
        """

        tol = 1e-11

        A = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])
        Ac = np.zeros((2, A.shape[1], A.shape[0]))
        Ac[0] = A.T

        wdm = csr_matrix(np.eye(4) - np.diag([0, 1, 1, 1]))
        we0 = np.ones((4,))
        e0 = 3*np.ones((4,))
        beta = 0
        target = np.reshape(np.array([-0.5, 0, 0, 0]), (4, 1))

        bp = np.zeros((4,))
        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(Ac), np.diag(cwe0**2)),
                      _jact2d(Ac).T)


        for method in valid_methods:
            x = jac_solve(jac=Ac, e0=e0, beta=beta, wdm=wdm,
                          we0=we0, bp=bp, jtwj=jtwj, method=method)
            self.assertTrue((np.abs(x -  target) < tol).all())
            pass
        pass



    def test_we0_zeros(self):
        """
        Test works as expected when we0 contains zeros but
        regularization present
        """
        tol = 1e-11

        A = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])
        Ac = np.zeros((2, A.shape[1], A.shape[0]))
        Ac[0] = A.T

        wdm = speye(4, format='csr')
        we0 = np.array([1, 0, 0, 0])
        e0 = np.array([60, 10, 10, 10]) # last three should not matter
        beta = 0
        target = np.reshape(np.array([-1, -2, -3, -4]), (4, 1))

        bp = np.zeros((4,))
        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(Ac), np.diag(cwe0**2)),
                      _jact2d(Ac).T)

        for method in valid_methods:
            x = jac_solve(jac=Ac, e0=e0, beta=beta, wdm=wdm,
                          we0=we0, bp=bp, jtwj=jtwj, method=method)
            self.assertTrue((np.abs(x -  target) < tol).all())
            pass
        pass



    # Inputs
    def test_beta_real_scalar(self):
        """
        Verify beta format rejected bad inputs
        """
        jac = np.ones((2, 4, 5))
        e0 = np.ones((5,))
        wdm = speye(4, format='csr')
        we0 = np.ones((5,))
        method = valid_methods[0] # always should have at least 1 element

        bp = np.zeros_like(we0)
        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(jac), np.diag(cwe0**2)),
                      _jact2d(jac).T)

        for perr in ['txt', [0, 1], [0], 1j, np.zeros((1,)), np.eye(4), None]:
            with self.assertRaises(TypeError):
                jac_solve(jac=jac, e0=e0, beta=perr, wdm=wdm,
                          we0=we0, bp=bp, jtwj=jtwj, method=method)
                pass
            pass
        pass

    def test_jac_real_3D(self):
        """
        Verify jac format [2 x Ndm x Npix real ndarray] rejects bad inputs
        """
        e0 = np.ones((5,))
        beta = 0.
        wdm = speye(4, format='csr')
        we0 = np.ones((5,))
        method = valid_methods[0] # always should have at least 1 element

        bp = np.zeros_like(we0)
        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.ones((10, 4)).T,
                      np.dot(np.diag(cwe0**2), np.ones((10, 4))))

        for perr in [np.ones((4,)), np.ones((4, 4)), np.ones((2, 4, 5, 2)),
                     np.ones((4, 4, 2)), 1j*np.ones((2, 4, 5)),
                     np.ones((2, 5, 4)), 'txt', 1.0, range(4),
                     [np.ones((2, 4, 5))], None]:
            with self.assertRaises(TypeError):
                jac_solve(jac=perr, e0=e0, beta=beta,
                          wdm=wdm, we0=we0, bp=bp, jtwj=jtwj,
                          method=method)
                pass
            pass
        pass

    def test_e0_real_1D(self):
        """
        Verify e0 format [1D Npix complex ndarray] rejects bad inputs
        """
        jac = np.ones((2, 4, 5))
        beta = 0.
        wdm = speye(4, format='csr')
        we0 = np.ones((5,))
        method = valid_methods[0] # always should have at least 1 element

        bp = np.zeros_like(we0)
        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(jac), np.diag(cwe0**2)),
                      _jact2d(jac).T)

        for perr in [np.ones((2,)), np.ones((4,)), np.eye(5),
                     'txt', 1.0, None]:
            with self.assertRaises(TypeError):
                jac_solve(jac=jac, e0=perr, beta=beta,
                          wdm=wdm, we0=we0, bp=bp, jtwj=jtwj,
                          method=method)
                pass
            pass
        pass

    def test_wdm_2D_sparse(self):
        """
        Verify wdm format [Ndm x Ndm real ndarray] rejects bad inputs
        """
        jac = np.ones((2, 4, 5))
        e0 = np.ones((5,))
        beta = 0.
        we0 = np.ones((5,))
        method = valid_methods[0] # always should have at least 1 element

        bp = np.zeros_like(we0)
        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(jac), np.diag(cwe0**2)),
                      _jact2d(jac).T)

        badwdm = [(csr_matrix((4, 4)), np.eye(4)),
                  1j*speye(4),
                  np.eye(4),
                  csr_matrix((5, 5)),
                  ]

        for perr in [np.ones((4, 5)), np.ones((5, 4)), 1j*np.eye(4),
                     np.ones((4,)), np.ones((4, 4, 4)), 'txt', 1.0,
                     None] + badwdm:
            with self.assertRaises(TypeError):
                jac_solve(jac=jac, e0=e0, beta=beta,
                          wdm=perr, we0=we0, bp=bp, jtwj=jtwj,
                          method=method)
                pass
            pass
        pass


    def test_jtwj_real_2D(self):
        """
        Verify jtwj format [Ndm x Ndm real ndarray] rejects bad inputs
        """
        jac = np.ones((2, 4, 5))
        e0 = np.ones((5,))
        beta = 0.
        wdm = speye(4, format='csr')
        we0 = np.ones((5,))
        method = valid_methods[0] # always should have at least 1 element

        bp = np.zeros_like(we0)

        for perr in [np.ones((4, 5)), np.ones((5, 4)), 1j*np.eye(4),
                     np.ones((4,)), np.ones((4, 4, 4)), 'txt', 1.0,
                     None]:
            with self.assertRaises(TypeError):
                jac_solve(jac=jac, e0=e0, beta=beta,
                          wdm=wdm, we0=we0, bp=bp, jtwj=perr, method=method)
                pass
            pass
        pass


    def test_method_invalid(self):
        """
        Verify invalid method strings are excluded
        """
        jac = np.ones((2, 4, 5))
        e0 = np.ones((5,))
        beta = 0.
        wdm = speye(4, format='csr')
        we0 = np.ones((5,))

        bp = np.zeros_like(we0)
        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(jac), np.diag(cwe0**2)),
                      _jact2d(jac).T)

        for perr in ['does_not_exist', 1.0, None]:
            with self.assertRaises(TypeError):
                jac_solve(jac=jac, e0=e0, beta=beta,
                          wdm=wdm, we0=we0, bp=bp, jtwj=jtwj,
                          method=perr)
                pass
            pass
        pass


    def test_method_valid(self):
        """
        Verify valid method strings (i.e. mixed case) are included
        """
        jac = np.ones((2, 4, 5))
        e0 = np.ones((5,))
        beta = 0.
        wdm = speye(4, format='csr')
        we0 = np.ones((5,))

        bp = np.zeros_like(we0)
        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(jac), np.diag(cwe0**2)),
                      _jact2d(jac).T)

        for method in [valid_methods[0].upper(),
                       valid_methods[-1].capitalize()]:
            # should return without incident
            jac_solve(jac=jac, e0=e0, beta=beta, wdm=wdm,
                      we0=we0, bp=bp, jtwj=jtwj, method=method)
            pass
        pass


    def test_we0_real_1D(self):
        """
        Verify we0 format [1D Npix real ndarray] rejects bad inputs
        """
        jac = np.ones((2, 4, 5))
        e0 = np.ones((5,))
        beta = 0.
        wdm = speye(4, format='csr')
        method = valid_methods[0] # always should have at least 1 element

        bp = np.zeros((5,))
        jtwj = np.dot(_jact2d(jac), _jact2d(jac).T)

        for perr in [np.ones((2,)), np.ones((4,)), np.eye(5),
                     1j*np.ones((5,)), 'txt', 1.0, None]:
            with self.assertRaises(TypeError):
                jac_solve(jac=jac, e0=e0, beta=beta,
                          wdm=wdm, we0=perr, bp=bp, jtwj=jtwj,
                          method=method)
                pass
            pass
        pass


    def test_bp_real_1D(self):
        """
        Verify we0 format [1D Npix real ndarray] rejects bad inputs
        """
        jac = np.ones((2, 4, 5))
        e0 = np.ones((5,))
        beta = 0.
        wdm = speye(4, format='csr')
        we0 = np.ones((5,))
        method = valid_methods[0] # always should have at least 1 element

        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(jac), np.diag(cwe0**2)),
                      _jact2d(jac).T)

        for perr in [np.ones((2,)), np.ones((4,)), np.eye(5),
                     1j*np.ones((5,)), 'txt', 1.0, None]:
            with self.assertRaises(TypeError):
                jac_solve(jac=jac, e0=e0, beta=beta,
                          wdm=wdm, we0=we0, bp=perr, jtwj=jtwj,
                          method=method)
                pass
            pass
        pass


    def test_jac_solve_output(self):
        """
        Verify we0 format [1D Npix real ndarray] rejects bad inputs
        """
        ndm = 4
        jac = np.ones((2, ndm, 5))
        e0 = np.ones((5,))
        beta = 0.
        wdm = speye(ndm, format='csr')
        we0 = np.ones((5,))
        bp = np.zeros((5,))

        cwe0 = np.hstack((we0, we0))
        jtwj = np.dot(np.dot(_jact2d(jac), np.diag(cwe0**2)),
                      _jact2d(jac).T)

        for method in valid_methods:
            out = jac_solve(jac=jac, e0=e0, beta=beta,
                      wdm=wdm, we0=we0, bp=bp, jtwj=jtwj,
                      method=method)
            self.assertTrue(out.shape == (ndm, 1))
            pass
        pass



class TestInvToDM(unittest.TestCase):
    """
    Test getting a new absolute DM setting from the previous setting
    """

    def test_deltadm_and_dmlist_consistent_with_cfg(self):
        """
        Should pass if size matches config
        """
        sizelist = [dm.registration['nact'] for dm in cfg.dmlist]
        deltadm = np.ones(((np.array(sizelist)**2).sum(),))
        dmlist = [np.ones((sz, sz)) for sz in sizelist]

        inv_to_dm(deltadm, cfg, dmlist)
        pass

    def test_deltadm_inconsistent_with_cfg(self):
        """
        Should fail in expected way if deltadm size does not match config
        """
        sizelist = [dm.registration['nact'] for dm in cfg.dmlist]
        dmlist = [np.ones((sz, sz)) for sz in sizelist]

        for deltadm in [np.ones(((np.array(sizelist)**2).sum()+1,)),
                        dmlist]:
            with self.assertRaises(TypeError):
                inv_to_dm(deltadm, cfg, dmlist)
                pass
            pass
        pass

    def test_dmlist_inconsistent_with_cfg(self):
        """
        Should fail in expected way if dmlist size does not match config
        """
        sizelist = [dm.registration['nact'] for dm in cfg.dmlist]
        deltadm = np.ones(((np.array(sizelist)**2).sum(),))

        for dmlist in [[np.ones((sz, sz+1)) for sz in sizelist],
                        [np.ones((sz+1, sz)) for sz in sizelist],
                        [np.ones((sz, sz)) for sz in sizelist]
                          + [np.ones((2, 2))]]:
            with self.assertRaises(TypeError):
                inv_to_dm(deltadm, cfg, dmlist)
                pass
            pass
        pass


    def test_zero_deltadm(self):
        """
        If the deltadm is zero, the output should match the initial (dmlist)
        """
        sizelist = [dm.registration['nact'] for dm in cfg.dmlist]
        deltadm = np.zeros(((np.array(sizelist)**2).sum(),))
        dmlist = [np.ones((sz, sz)) for sz in sizelist]

        outdmlist = inv_to_dm(deltadm, cfg, dmlist)
        for index, outdm in enumerate(outdmlist):
            self.assertTrue(np.allclose(outdm, dmlist[index]))
            pass
        pass


    def test_output_shape(self):
        """
        Verify output shape matches doc
        """
        sizelist = [dm.registration['nact'] for dm in cfg.dmlist]
        deltadm = np.zeros(((np.array(sizelist)**2).sum(),))
        dmlist = [np.ones((sz, sz)) for sz in sizelist]

        outdmlist = inv_to_dm(deltadm, cfg, dmlist)
        self.assertTrue(len(outdmlist) == len(dmlist))
        for index, outdm in enumerate(outdmlist):
            self.assertTrue(outdm.shape == dmlist[index].shape)
            pass
        pass





if __name__ == '__main__':
    unittest.main()
