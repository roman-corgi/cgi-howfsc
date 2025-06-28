# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=unsubscriptable-object
"""Unit tests for dmhtoph.py."""

import unittest
import os
import types
import copy

import numpy as np
from astropy.io import fits

from .dmhtoph import dmhtoph
from .dmhtoph_jac import (
    dmhtoph_jac, dmhtoph_cropped_poke, compute_master_inf_func
)
from .insertinto import insertinto
from .math import ceil_odd

# set up some defaults
localpath = os.path.dirname(os.path.abspath(__file__))
fninf_pist = os.path.join(localpath, 'testdata',
                          'ut_influence_dm5v2_inffix.fits')

p = fits.getdata(fninf_pist)
ppa = 10  # correct value for this shape

# Test shapes
nact = 48
dmin0 = np.ones((nact, nact))  # all acts
dmin1 = np.ones((nact, nact))
dmin1[1:-1, 1:-1] = 0  # outer acts
dmin2 = np.zeros((nact, nact))
dmin2[0, 0] = 1  # one act

class TestDmhtophJac(unittest.TestCase):
    """Unit test suite for dmhtoph_jac()."""

    def test_outputs(self):
        """Check outputs are nrow x ncol."""
        for nrow, ncol in [(600, 600),
                           (600, 300),
                           (300, 600),
                           (601, 601)]:
            d0 = dmhtoph_jac(nrow=nrow, ncol=ncol,
                             pokerad=1, row=0, col=0,
                             nact=nact, inf_func=p, ppact_d=ppa,
                             ppact_cx=ppa-1, ppact_cy=ppa-1,
                             dx=0, dy=0, thact=0, flipx=False)
            self.assertTrue(d0.shape == (nrow, ncol))
            pass

        pass

    def test_range_success(self):
        """Verify the ends of the row/col range work."""
        dmhtoph_jac(nrow=601, ncol=601,
                    pokerad=1, row=0, col=0,
                    nact=nact, inf_func=p, ppact_d=ppa,
                    ppact_cx=ppa, ppact_cy=ppa,
                    dx=-ppa/2., dy=-ppa/2., thact=0, flipx=False)
        dmhtoph_jac(nrow=601, ncol=601,
                    pokerad=1, row=47, col=47,
                    nact=nact, inf_func=p, ppact_d=ppa,
                    ppact_cx=ppa, ppact_cy=ppa,
                    dx=-ppa/2., dy=-ppa/2., thact=0, flipx=False)
        dmhtoph_jac(nrow=601, ncol=601,
                    pokerad=1, row=47, col=0,
                    nact=nact, inf_func=p, ppact_d=ppa,
                    ppact_cx=ppa, ppact_cy=ppa,
                    dx=-ppa/2., dy=-ppa/2., thact=0, flipx=False)
        dmhtoph_jac(nrow=601, ncol=601,
                    pokerad=1, row=0, col=47,
                    nact=nact, inf_func=p, ppact_d=ppa,
                    ppact_cx=ppa, ppact_cy=ppa,
                    dx=-ppa/2., dy=-ppa/2., thact=0, flipx=False)
        pass


    def test_array_symmetry(self):
        """
        Verify that with no translation/rotation/scale and a symmetric input
        we get a symmetric output.   Use odd to be pixel-centered.
        """
        tol = 1e-13
        d0 = dmhtoph_jac(nrow=601, ncol=601, pokerad=1, row=23, col=23,
                         nact=47, inf_func=p, ppact_d=ppa,
                     ppact_cx=ppa-1, ppact_cy=ppa-1,
                     dx=0, dy=0, thact=0, flipx=False)
        self.assertTrue((np.abs(d0-np.fliplr(np.flipud(d0))) < tol).all())
        pass

    # Success tests
    def test_convolution_works(self):
        """Convolve one actuator and check it matches influence function."""
        tol = 1e-13
        N = p.shape[0]
        nrow = (nact - 1)*ppa + N
        ncol = (nact - 1)*ppa + N

        d0 = dmhtoph_jac(nrow=nrow, ncol=ncol,
                         pokerad=1, row=0, col=0,
                         nact=nact, inf_func=p, ppact_d=ppa,
                         ppact_cx=ppa, ppact_cy=ppa,
                         dx=0, dy=0, thact=0, flipx=False)
        self.assertTrue((np.abs(d0[:N, :N] - p) < tol).all())
        pass

    def test_output_real(self):
        """Verify output is real-valued."""
        for pcxy in [(10, 10), (4, 4), (4.5, 4), (4, 4.5), (4.5, 4.5)]:
            d0 = dmhtoph_jac(nrow=601, ncol=601,
                             pokerad=1, row=10, col=20,
                             nact=nact, inf_func=p, ppact_d=ppa,
                             ppact_cx=pcxy[0], ppact_cy=pcxy[1],
                             dx=0, dy=0, thact=0, flipx=False)
            self.assertTrue(np.isreal(d0).all())
            pass
        pass

    def test_flipx_flips_as_expected(self):
        """
        Verify DM flip flag matches theory expectation; odd sizing means
        arrays are symmetric about center.
        """
        dmin2a = np.zeros((nact-1, nact-1))
        dmin2a[0, -1] = 1
        tol = 1e-13

        d2 = dmhtoph_jac(nrow=601, ncol=601,
                         pokerad=1, row=0, col=0,
                         nact=nact-1, inf_func=p, ppact_d=ppa,
                         ppact_cx=ppa-1, ppact_cy=ppa-1,
                         dx=0, dy=0, thact=0, flipx=False)
        d2a = dmhtoph_jac(nrow=601, ncol=601,
                          pokerad=1, row=0, col=nact-2,
                          nact=nact-1, inf_func=p, ppact_d=ppa,
                          ppact_cx=ppa-1, ppact_cy=ppa-1,
                          dx=0, dy=0, thact=0, flipx=True)
        self.assertTrue((np.abs(d2a-d2) < tol).all())
        pass

    def test_thact_rotates_as_expected(self):
        """
        Verify DM rotate matches theory expectations; odd sizing means
        arrays are symmetric about center.

        Can't do asymmetry exactly with dmhtoph_jac, but use act not on
        centerline.
        """
        tol = 1e-13
        pokerad = 1

        # asymmetric pattern
        row = 0
        col = 1

        # 90 deg counterclockwise
        row90 = 1
        col90 = (nact-1)-1

        dF1 = dmhtoph_jac(nrow=601, ncol=601,
                          pokerad=pokerad, row=row, col=col,
                          nact=nact-1, inf_func=p, ppact_d=ppa,
                          ppact_cx=ppa-1, ppact_cy=ppa-1,
                          dx=0, dy=0, flipx=False, thact=90)

        dF2 = dmhtoph_jac(nrow=601, ncol=601,
                          pokerad=pokerad, row=row90, col=col90,
                          nact=nact-1, inf_func=p, ppact_d=ppa,
                          ppact_cx=ppa-1, ppact_cy=ppa-1,
                          dx=0, dy=0, flipx=False, thact=0)
        self.assertTrue((np.abs(dF1-dF2) < tol).all())
        pass

    def test_dxdy_shift_as_expected(self):
        """Verify DM x/y shift matches theory expectations."""
        # use roll to shift by integer amounts
        tol = 1e-13

        d0 = dmhtoph_jac(nrow=601, ncol=601,
                         pokerad=1, row=10, col=20,
                         nact=nact, inf_func=p, ppact_d=ppa,
                         ppact_cx=4, ppact_cy=4,
                         dx=0, dy=0, thact=0, flipx=False)

        xylist = [(5, 0), (0, 4), (-6, 0), (0, -5), (2, 3), (-4, 9)]
        biggest = np.max(np.abs(xylist)).astype('int')

        for xy in xylist:
            ddx = dmhtoph_jac(nrow=601, ncol=601,
                              pokerad=1, row=10, col=20,
                              nact=nact, inf_func=p, ppact_d=ppa,
                              ppact_cx=4, ppact_cy=4,
                              dx=xy[0], dy=xy[1], thact=0, flipx=False)
            rolld0 = np.roll(np.roll(d0, xy[0], axis=1), xy[1], axis=0)
            dim = (ddx.shape[0] - 2*biggest).astype('int')

            # roll moves data from one edge over to the other.  This data is
            # not necessarily correct for that location, so we'll trim the
            # edges that were rolled and check the center.
            trim_ddx = insertinto(ddx, (dim, dim))
            trim_rd0 = insertinto(rolld0, (dim, dim))

            self.assertTrue((np.abs(trim_rd0 - trim_ddx) < tol).all())
            pass
        pass

    # skip piston normalization, dmhtoph_jac can't do pistons

    def test_asymmetric_convolution_works(self):
        """
        Convolve one asymmetric actuator, shifted to be pixel-aligned with
        original corner, and check it matches influence function
        should ensure we don't flip/rotate inf function.
        """

        tol = 1e-13

        F = (p + np.roll(p, 4, axis=1)
             + np.roll(p, -4, axis=1)
             + np.roll(p, 2, axis=0)
             + np.roll(np.roll(p, 4, axis=1), 4, axis=0))/5.

        N = F.shape[0]
        nrow = (nact - 1)*ppa + N
        ncol = (nact - 1)*ppa + N

        d0 = dmhtoph_jac(nrow=nrow, ncol=ncol,
                         pokerad=1, row=0, col=0,
                         nact=nact, inf_func=F, ppact_d=ppa,
                         ppact_cx=ppa, ppact_cy=ppa,
                         dx=0, dy=0, thact=0, flipx=False)
        self.assertTrue((np.abs(d0[:N, :N] - F) < tol).all())
        pass

    #------------

    def test_dmhtoph_and_dmhtophjac_give_same_answers(self):
        """
        Verify the two functions produce identical results on a standard calc
        (should be no difference mathematical between the two).
        """
        tol = 1e-13

        row = 10
        col = 20
        pokerad = 1

        dmjac = np.zeros((nact, nact))
        dmjac[row, col] = pokerad
        d0 = dmhtoph(nrow=601, ncol=601,
                     dmin=dmjac, nact=nact, inf_func=p, ppact_d=ppa,
                     ppact_cx=ppa, ppact_cy=ppa,
                     dx=0, dy=0, thact=0, flipx=False)

        d0jac = dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=pokerad, row=row, col=col,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=ppa, ppact_cy=ppa,
                            dx=0, dy=0, thact=0, flipx=False)

        self.assertTrue(np.max(np.abs(d0 - d0jac)) < tol)
        pass

    # Failure tests
    def test_nrow_positive_scalar_integer(self):
        """Check number-of-actuator type valid."""
        for perr in [[], -48, 0, (48,), 48.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=perr, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=perr, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_ncol_positive_scalar_integer(self):
        """Check number-of-actuator type valid."""
        for perr in [[], -48, 0, (48,), 48.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=perr,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=perr, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass


    def test_pokerad_realscalar(self):
        """Check pokerad type valid."""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=perr, row=10, col=20,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=perr, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass


    def test_row_nonnegative_scalar_integer(self):
        """Check row type valid."""
        for perr in [[], -47, (47,), 47.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=perr, col=20,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass


    def test_row_smaller_than_nact(self):
        """Check row size valid."""
        for perr in [nact, nact+1]:
            with self.assertRaises(ValueError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=perr, col=20,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass


    def test_col_nonnegative_scalar_integer(self):
        """Check col type valid."""
        for perr in [[], -47, (47,), 47.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=perr,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_col_smaller_than_nact(self):
        """Check col size valid."""
        for perr in [nact, nact+1]:
            with self.assertRaises(ValueError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=perr,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_nact_positive_scalar_integer(self):
        """Check number-of-actuator type valid."""
        for nacte in [[], -48, 0, (48,), 48.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nacte, inf_func=p, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_inf_func_2Darray(self):
        """Check influence function array type valid."""
        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'inf']:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=perr, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_inf_func_square(self):
        """Check influence function array is square."""
        for perr in [np.ones((91, 90)), np.ones((90, 91))]:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=perr, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_inf_func_odd(self):
        """Check influence function array is odd-sized."""
        for perr in [np.ones((90, 90))]:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=perr, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_ppactd_realpositivescalar(self):
        """Check influence function scaling type valid."""
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=p, ppact_d=perr,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_ppactcx_realpositivescalar(self):
        """Check output X scaling type valid."""
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=perr, ppact_cy=4, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_ppactcy_realpositivescalar(self):
        """Check output Y scaling type valid."""
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=perr, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_dx_realscalar(self):
        """Check offset X scaling type valid."""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=perr, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_dy_realscalar(self):
        """Check offset Y scaling type valid."""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=perr, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_thact_realscalar(self):
        """Check rotation angle type valid."""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=p, ppact_d=ppa,
                            ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=perr,
                            flipx=False)
                pass
            pass
        pass

    def test_design_sampled_more_than_camera(self):
        """
        Should fail if your camera sampling is finer than your theoretical
        influence function (because you need a better sampling to downsample).
        """
        for ppactlist in [(10, 11, 11),
                          (10, 4, 11),
                          (10, 11, 4)]:
            pd, pcx, pcy = ppactlist
            with self.assertRaises(TypeError):
                dmhtoph_jac(nrow=601, ncol=601,
                            pokerad=1, row=10, col=20,
                            nact=nact, inf_func=p, ppact_d=pd,
                            ppact_cx=pcx, ppact_cy=pcy, dx=0, dy=0, thact=0,
                            flipx=False)
                pass
            pass
        pass

    def test_inf_func_smaller_than_DM_grid(self):
        """Check that the influence function fails if bigger than DMxo."""
        with self.assertRaises(TypeError):
            pwide = insertinto(p, (ppa*nact + 1, ppa*nact + 1))
            dmhtoph_jac(nrow=601, ncol=601,
                        pokerad=1, row=10, col=20,
                        nact=nact, inf_func=pwide, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
            pass
        pass


class TestDmhtophCroppedPoke(unittest.TestCase):
    """Unit test suite for dmhtoph_cropped_poke()."""

    def setUp(self):
        """Define repeatedly used values."""
        dmobj = types.SimpleNamespace()
        dmobj.nact = 48
        dmobj.nSurf = 1024
        dmobj.infMaster = insertinto(np.arange(49).reshape((7, 7)), (31, 31))
        dmobj.xLowerLeft = np.arange(dmobj.nact*dmobj.nact, dtype=int) % 30
        dmobj.yLowerLeft = np.arange(dmobj.nact*dmobj.nact, dtype=int) % 40
        dmobj.xOffsets = np.zeros(dmobj.nact*dmobj.nact)
        dmobj.yOffsets = np.zeros(dmobj.nact*dmobj.nact)
        self.row = 5
        self.col = 8
        self.index1d = self.row*dmobj.nact + self.col
        self.dx = 4
        self.dy = 7
        dmobj.xOffsets[self.index1d] = self.dx
        dmobj.yOffsets[self.index1d] = self.dy
        self.pokerad = 3
        self.dmobj = dmobj

    def test_output_shapes(self):
        """Test output shapes of dmhtoph_cropped_poke()."""
        surfCrop, yxLowerLeft, _ = dmhtoph_cropped_poke(
            self.pokerad, self.row, self.col, self.dmobj)

        self.assertTrue(len(yxLowerLeft) == 2)
        self.assertTrue(surfCrop.shape == self.dmobj.infMaster.shape)

    # Success Tests
    def test_against_dmhtoph_jac(self):
        """Verify accuracy against output of dmhtoph_jac()."""
        relTol = 1e-2
        row = 2
        col = 1
        pokerad = 3.1

        nact_ = 10
        ppact_cx = 7.6
        ppact_cy = 6.6
        dx = 6.2
        dy = -14.5
        inf_func = p
        ppact_d = 10

        reg_dict = {}
        reg_dict['ppact_cx'] = ppact_cx
        reg_dict['ppact_cy'] = ppact_cy
        reg_dict['ppact_d'] = ppact_d
        reg_dict['dx'] = dx
        reg_dict['dy'] = dy
        reg_dict['inf_func'] = p
        reg_dict['nact'] = nact_

        for thact in (-5.2, 20.3):
            for flipx in (False, True):

                reg_dict['thact'] = thact
                reg_dict['flipx'] = flipx

                dmobj = types.SimpleNamespace()
                dmobj.pupil_shape = (100, 100)
                compute_master_inf_func(dmobj, reg_dict)

                surfCrop, yxLowerLeft, nSurf = dmhtoph_cropped_poke(
                    pokerad, row, col, dmobj)
                nSubarray = surfCrop.shape[0]
                surfFast = np.zeros((nSurf, nSurf), dtype=float)
                surfFast[yxLowerLeft[0]:yxLowerLeft[0] + nSubarray,
                       yxLowerLeft[1]:yxLowerLeft[1] + nSubarray] = surfCrop

                nrow = nSurf
                ncol = nSurf
                surfNormal = dmhtoph_jac(nrow, ncol, pokerad, row, col, nact_,
                                         inf_func, ppact_d, ppact_cx, ppact_cy,
                                         dx, dy, thact, flipx)

                surfNormal = insertinto(surfNormal, (nSurf, nSurf))

                maxVal = np.max(surfNormal)
                maxDiff = np.max(np.abs(surfNormal - surfFast))
                maxNormError = maxDiff/maxVal

                self.assertTrue(maxNormError < relTol)

        pass

    def test_integer_shift(self):
        """Test dmhtoph_cropped_poke() against np.roll()."""
        tol = 10*np.finfo(float).eps

        surfCropExpected = (self.pokerad *
                            np.roll(self.dmobj.infMaster,
                                    (self.dy, self.dx),
                                    axis=(0, 1)))

        surfCrop, yxLowerLeft, nSurf = dmhtoph_cropped_poke(
            self.pokerad, self.row, self.col, self.dmobj)

        self.assertTrue(yxLowerLeft[0] == self.dmobj.yLowerLeft[self.index1d])
        self.assertTrue(yxLowerLeft[1] == self.dmobj.xLowerLeft[self.index1d])
        self.assertTrue(nSurf == self.dmobj.nSurf)
        self.assertTrue(np.max(np.abs(surfCrop - surfCropExpected)) < tol)

    # Failure tests
    def test_pokerad_realscalar(self):
        """Check pokerad type valid."""
        for pokerad in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph_cropped_poke(pokerad, self.row, self.col, self.dmobj)

    def test_row_nonnegative_scalar_integer(self):
        """Check row type valid."""
        for row in [[], -47, (47,), 47.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmhtoph_cropped_poke(self.pokerad, row, self.col, self.dmobj)

    def test_row_smaller_than_nact(self):
        """Check row size valid."""
        for row in [self.dmobj.nact, self.dmobj.nact+1]:
            with self.assertRaises(ValueError):
                dmhtoph_cropped_poke(self.pokerad, row, self.col, self.dmobj)

    def test_col_nonnegative_scalar_integer(self):
        """Check col type valid."""
        for col in [[], -47, (47,), 47.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmhtoph_cropped_poke(self.pokerad, self.row, col, self.dmobj)

    def test_col_smaller_than_nact(self):
        """Check col size valid."""
        for col in [nact, nact+1]:
            with self.assertRaises(ValueError):
                dmhtoph_cropped_poke(self.pokerad, self.row, col, self.dmobj)

    def test_dmobj_simple_namespace(self):
        """Check dmobj type valid."""
        for dmobj in [(1, 2), 1, 1j, 'asdf']:
            with self.assertRaises(TypeError):
                dmhtoph_cropped_poke(self.pokerad, self.row, self.col, dmobj)


class TestComputeMasterInfFunc(unittest.TestCase):
    """Unit test suite for compute_master_inf_func()."""

    def setUp(self):
        """Define reused variables."""
        self.reg_dict = {'ppact_cx': 6.37, 'ppact_cy': 6.46, 'ppact_d': ppa,
                         'dx': 0, 'dy': 0, 'inf_func': p, 'nact': 48,
                         'thact': 0, 'flipx': False}

    def test_output_attribute_existence(self):
        """Check that the object contains all the needed attributes."""
        dmobj = types.SimpleNamespace()
        dmobj.pupil_shape = (300, 300)
        compute_master_inf_func(dmobj, self.reg_dict)

        self.assertTrue(hasattr(dmobj, 'xLowerLeft'))
        self.assertTrue(hasattr(dmobj, 'yLowerLeft'))
        self.assertTrue(hasattr(dmobj, 'xOffsets'))
        self.assertTrue(hasattr(dmobj, 'yOffsets'))
        self.assertTrue(hasattr(dmobj, 'infMaster'))
        self.assertTrue(hasattr(dmobj, 'nSurf'))
        self.assertTrue(hasattr(dmobj, 'nact'))

    def test_output_shapes(self):
        """Check that the output object attributes are sized correctly."""
        dmobj = types.SimpleNamespace()
        dmobj.pupil_shape = (300, 300)
        compute_master_inf_func(dmobj, self.reg_dict)
        nact2 = self.reg_dict['nact']**2

        self.assertTrue(len(dmobj.infMaster.shape) == 2)
        self.assertTrue(dmobj.infMaster.shape[0] == dmobj.infMaster.shape[1])
        self.assertTrue(len(dmobj.xOffsets) == nact2)
        self.assertTrue(len(dmobj.yOffsets) == nact2)
        self.assertTrue(len(dmobj.xLowerLeft) == nact2)
        self.assertTrue(len(dmobj.yLowerLeft) == nact2)

    def test_subarray_indexing(self):
        """Make sure that subarray indices are never outside the full array."""
        reg_dict = copy.copy(self.reg_dict)
        ppact_vec = np.linspace(5.5, 7.5, 10)
        delta_vec = [0, 0.5, 2.6]
        theta_vec = [0, 10, -45]

        for ppact in ppact_vec:
            for delta in delta_vec:
                for theta in theta_vec:
                    dmobj = types.SimpleNamespace()
                    dmobj.pupil_shape = (300, 300)
                    reg_dict.update({'ppact_cx': ppact, 'ppact_cy': ppact,
                                     'dx': delta, 'dy': -delta,
                                     'theta': theta})
                    compute_master_inf_func(dmobj, reg_dict)

                    nCrop = dmobj.infMaster.shape[0]

                    self.assertTrue(np.all(dmobj.xLowerLeft >= 0))
                    self.assertTrue(np.all(dmobj.yLowerLeft >= 0))
                    self.assertTrue(
                        np.all(dmobj.xLowerLeft+nCrop <= dmobj.nSurf))
                    self.assertTrue(
                        np.all(dmobj.xLowerLeft+nCrop <= dmobj.nSurf))

    def test_rotation(self):
        """Test that the influence function is rotated as expected."""
        dmobj_a = types.SimpleNamespace()
        dmobj_a.pupil_shape = (300, 300)
        dmobj_b = types.SimpleNamespace()
        dmobj_b.pupil_shape = (300, 300)
        reg_dict = copy.copy(self.reg_dict)
        reg_dict.update({'ppact_cx': 10, 'ppact_cy': 10})

        inf_func = np.zeros((31, 31))
        inf_func[5:7, 1:-1] = 1
        inf_func[1:-1, 8:12] = 1
        reg_dict.update({'thact': 0, 'inf_func': inf_func})
        compute_master_inf_func(dmobj_a, reg_dict)

        reg_dict.update({'thact': 90})
        compute_master_inf_func(dmobj_b, reg_dict)

        nOdd = ceil_odd(dmobj_a.infMaster.shape[0])
        infMasterA = insertinto(dmobj_a.infMaster, (nOdd, nOdd))
        infMasterB = insertinto(dmobj_b.infMaster, (nOdd, nOdd))

        maxAbsDiff = np.max(np.abs(np.rot90(infMasterA, -1) - infMasterB))
        abs_tol = 1e-8
        self.assertTrue(maxAbsDiff < abs_tol)

    def test_flipx_infmaster(self):
        """Test that the influence function is flipped as expected."""
        dmobj_a = types.SimpleNamespace()
        dmobj_a.pupil_shape = (100, 100)
        dmobj_b = types.SimpleNamespace()
        dmobj_b.pupil_shape = (100, 100)
        reg_dict = copy.copy(self.reg_dict)
        reg_dict.update({'ppact_cx': 10, 'ppact_cy': 10})

        inf_func = np.zeros((31, 31))
        inf_func[5:7, 1:-1] = 1
        inf_func[1:-1, 8:12] = 1
        reg_dict.update({'flipx': False, 'inf_func': inf_func})
        compute_master_inf_func(dmobj_a, reg_dict)

        reg_dict.update({'flipx': True})
        compute_master_inf_func(dmobj_b, reg_dict)

        nOdd = ceil_odd(dmobj_a.infMaster.shape[0])
        infMasterA = insertinto(dmobj_a.infMaster, (nOdd, nOdd))
        infMasterB = insertinto(dmobj_b.infMaster, (nOdd, nOdd))

        maxAbsDiff = np.max(np.abs(np.fliplr(infMasterA) - infMasterB))
        abs_tol = 1e-8
        self.assertTrue(maxAbsDiff < abs_tol)

    def test_downsampling(self):
        """Test that the influence function is downsampled as expected."""
        dmobj = types.SimpleNamespace()
        dmobj.pupil_shape = (100, 100)
        reg_dict = copy.copy(self.reg_dict)
        reg_dict.update({'ppact_cx': 10/3, 'ppact_cy': 10/3})

        inf_func = np.ones((9, 9))
        inf_func = insertinto(inf_func, (21, 21))
        reg_dict.update({'inf_func': inf_func})
        compute_master_inf_func(dmobj, reg_dict)

        nOdd = ceil_odd(dmobj.infMaster.shape[0])
        infMaster = insertinto(dmobj.infMaster, (nOdd, nOdd))
        infMasterExpected = insertinto(np.ones((3, 3)), (nOdd, nOdd))

        maxAbsDiff = np.max(np.abs(infMaster - infMasterExpected))
        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(maxAbsDiff < abs_tol)

    # Failure tests
    def test_dmobj_simple_namespace(self):
        """Check dmobj type valid."""
        for dmobj in [(1, 2), 1, 1j, 'asdf']:
            with self.assertRaises(TypeError):
                compute_master_inf_func(dmobj, self.reg_dict)

    def test_reg_dict_type(self):
        """Check reg_dict type valid."""
        dmobj = types.SimpleNamespace()
        for reg_dict in [(1, 2), 1, 1j, 'asdf']:
            with self.assertRaises(TypeError):
                compute_master_inf_func(dmobj, reg_dict)


if __name__ == '__main__':
    unittest.main()
