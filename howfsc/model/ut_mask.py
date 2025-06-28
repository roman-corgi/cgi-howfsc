# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for the mask classes
"""

import unittest

import numpy as np

from howfsc.model.dmobj import DM
from .mask import ModelElement, PupilMask, FocalPlaneMask, Epup, DMFace, \
     LyotStop

class TestModelElement(unittest.TestCase):
    """
    Unit test suite for mask base class (ModelElement) constructor
    """

    # Constructor
    def test_construction(self):
        """Simple construction test"""
        ModelElement(lam=633e-9, e=np.ones((128, 128)))
        pass

    def test_lam_realpositivescalar(self):
        """Check wavelength type valid"""
        e = np.ones((128, 128))
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                ModelElement(lam=perr, e=e)
                pass
            pass
        pass

    def test_e_2darray(self):
        """Check electric field type valid"""
        lam = 633e-9
        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            with self.assertRaises(TypeError):
                ModelElement(lam=lam, e=perr)
                pass
            pass
        pass


class TestApplyMask(unittest.TestCase):
    """
    Unit test suite for ModelElement.applymask()
    """

    def test_correct_apply(self):
        """
        Check that a mask is applied correctly
        Values are right, and output and input are same size regardless of mask
        """
        e = 0.5*np.ones((3, 3))
        m = ModelElement(lam=633e-9, e=e)

        e0 = np.ones((5, 5))
        e1 = m.applymask(e0)

        etest = np.array([[0, 0, 0, 0, 0],
                          [0, 0.5, 0.5, 0.5, 0],
                          [0, 0.5, 0.5, 0.5, 0],
                          [0, 0.5, 0.5, 0.5, 0],
                          [0, 0, 0, 0, 0]])

        self.assertTrue((e1 == etest).all())
        pass


    def test_e0_2d_array(self):
        """Check electric field type valid"""
        m = ModelElement(lam=633e-9, e=np.ones((128, 128)))
        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            with self.assertRaises(TypeError):
                m.applymask(e0=perr)
                pass
            pass
        pass

    def test_e0_must_be_larger_than_e(self):
        """Fail if the input is undersized with respect to the mask """
        m = ModelElement(lam=633e-9, e=np.ones((128, 128)))
        for perr in [np.ones((127, 128)),
                     np.ones((128, 127)),
                     np.ones((127, 127))]:
            with self.assertRaises(TypeError):
                m.applymask(e0=perr)
                pass
            pass
        pass

    def test_output_size(self):
        """Check output size matches doc"""
        m = ModelElement(lam=633e-9, e=np.ones((128, 128)))
        for e0 in [np.ones((129, 128)), # should work for equal/larger sizes
                   np.ones((128, 129)),
                   np.ones((129, 129))]:
            out = m.applymask(e0=e0)
            self.assertTrue(out.shape == e0.shape)
            pass
        pass




class TestPupilMask(unittest.TestCase):
    """
    Unit test suite for PupilMask class
    """

    def test_inheritance(self):
        """PupilMask should inherit from ModelElement"""
        p = PupilMask(lam=633e-9, e=np.ones((128, 128)),
                      pixperpupil=386)
        self.assertTrue(isinstance(p, ModelElement))
        pass

    def test_check_plane(self):
        """Verify plane designation (only used for debugging)"""
        p = PupilMask(lam=633e-9, e=np.ones((128, 128)),
                      pixperpupil=386)
        self.assertTrue(p.plane == 'pupil')
        pass

    def test_pixperpupil_realpositivescalar(self):
        """Check scaling type valid"""
        lam = 633e-9
        e = np.ones((128, 128))
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                PupilMask(lam=lam, e=e, pixperpupil=perr)
                pass
            pass
        pass



class TestFocalPlaneMask(unittest.TestCase):
    """
    Unit test suite for FocalPlaneMask class
    """

    def test_inheritance(self):
        """FocalPlaneMask should inherit from ModelElement"""
        p = FocalPlaneMask(lam=633e-9, e=np.ones((128, 128)), isopen=True,
                           pixperlod=4.0)
        self.assertTrue(isinstance(p, ModelElement))
        pass

    def test_check_plane(self):
        """Verify plane designation (only used for debugging)"""
        p = FocalPlaneMask(lam=633e-9, e=np.ones((128, 128)), isopen=True,
                           pixperlod=4.0)
        self.assertTrue(p.plane == 'focal')
        pass

    def test_pixperlod_realpositivescalar(self):
        """Check scaling type valid"""
        lam = 633e-9
        e = np.ones((128, 128))
        isopen = True
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                FocalPlaneMask(lam=lam, e=e, isopen=isopen, pixperlod=perr)
                pass
            pass
        pass

    def test_isopen_both_cases(self):
        """
        anything can be interpreted as a boolean in Python, so let's make sure
        that both True and False work, and everything should follow
        """
        lam = 633e-9
        e = np.ones((128, 128))
        pixperlod = 4.
        for isopen in [True, False]:
            FocalPlaneMask(lam=lam, e=e, isopen=isopen, pixperlod=pixperlod)
            pass
        pass



class TestLyotStop(unittest.TestCase):
    """
    Unit test suite for LyotStop class
    """

    def test_inheritance(self):
        """LyotStop should inherit from PupilMask"""
        p = LyotStop(lam=633e-9, e=np.ones((128, 128)),
                     pixperpupil=386, tip=0, tilt=0)
        self.assertTrue(isinstance(p, PupilMask))
        pass


    def test_tip_realscalar(self):
        """Check tip type correct"""
        lam = 633e-9
        e = np.ones((128, 128))
        pixperpupil = 386
        tilt = 0
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                LyotStop(lam=lam, e=e, pixperpupil=pixperpupil,
                         tip=perr, tilt=tilt)
                pass
            pass
        pass


    def test_tilt_realscalar(self):
        """Check tilt type correct"""
        lam = 633e-9
        e = np.ones((128, 128))
        pixperpupil = 386
        tip = 0
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                LyotStop(lam=lam, e=e, pixperpupil=pixperpupil,
                         tip=tip, tilt=perr)
                pass
            pass
        pass


    def test_tiptilt_output_as_expected(self):
        """
        Check the grid covers the expected range
        """

        lam = 633e-9
        e = np.ones((128, 128))
        pixperpupil = 128
        tip = 7
        tilt = 5
        tol = 1e-13

        lyot = LyotStop(lam=lam, e=e, pixperpupil=pixperpupil,
                        tip=tip, tilt=tilt)

        gmin = -pixperpupil//2
        gmax = (pixperpupil-1)//2

        self.assertTrue(np.abs(lyot.ttgrid[0, 0] - (tip*gmin + tilt*gmin))
                        < tol)
        self.assertTrue(np.abs(lyot.ttgrid[0, -1] - (tip*gmax + tilt*gmin))
                        < tol)
        self.assertTrue(np.abs(lyot.ttgrid[-1, 0] - (tip*gmin + tilt*gmax))
                        < tol)
        self.assertTrue(np.abs(lyot.ttgrid[-1, -1] - (tip*gmax + tilt*gmax))
                        < tol)

        pass





class TestEpup(unittest.TestCase):
    """
    Unit test suite for Epup class
    """

    def test_inheritance(self):
        """Epup should inherit from ModelElement"""
        p = Epup(lam=633e-9, e=np.ones((128, 128)),
                 pixperpupil=386, tip=0, tilt=0)
        self.assertTrue(isinstance(p, ModelElement))
        pass

    def test_check_plane(self):
        """Verify plane designation (only used for debugging)"""
        p = Epup(lam=633e-9, e=np.ones((128, 128)),
                 pixperpupil=386, tip=0, tilt=0)
        self.assertTrue(p.plane == 'epup')
        pass

    def test_pixperpupil_realpositivescalar(self):
        """Check scaling type valid"""
        lam = 633e-9
        e = np.ones((128, 128))
        tip = 0
        tilt = 0
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                Epup(lam=lam, e=e, pixperpupil=perr, tip=tip, tilt=tilt)
                pass
            pass
        pass


    def test_tip_realscalar(self):
        """Check tip type correct"""
        lam = 633e-9
        e = np.ones((128, 128))
        pixperpupil = 386
        tilt = 0
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                Epup(lam=lam, e=e, pixperpupil=pixperpupil,
                     tip=perr, tilt=tilt)
                pass
            pass
        pass


    def test_tilt_realscalar(self):
        """Check tilt type correct"""
        lam = 633e-9
        e = np.ones((128, 128))
        pixperpupil = 386
        tip = 0
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                Epup(lam=lam, e=e, pixperpupil=pixperpupil,
                     tip=tip, tilt=perr)
                pass
            pass
        pass


    def test_tiptilt_output_as_expected(self):
        """
        Check the grid covers the expected range
        """

        lam = 633e-9
        e = np.ones((128, 128))
        pixperpupil = 128
        tip = 7
        tilt = 5
        tol = 1e-13

        epup = Epup(lam=lam, e=e, pixperpupil=pixperpupil,
                    tip=tip, tilt=tilt)

        gmin = -pixperpupil//2
        gmax = (pixperpupil-1)//2

        self.assertTrue(np.abs(epup.ttgrid[0, 0] - (tip*gmin + tilt*gmin))
                        < tol)
        self.assertTrue(np.abs(epup.ttgrid[0, -1] - (tip*gmax + tilt*gmin))
                        < tol)
        self.assertTrue(np.abs(epup.ttgrid[-1, 0] - (tip*gmin + tilt*gmax))
                        < tol)
        self.assertTrue(np.abs(epup.ttgrid[-1, -1] - (tip*gmax + tilt*gmax))
                        < tol)

        pass


    def test_tiptilt_output_grid_as_expected(self):
        """
        Check the grid covers the expected range when direct from get_ttgrid
        """

        lam = 633e-9
        e = np.ones((128, 128))
        pixperpupil = 128
        tip = 7
        tilt = 5
        tol = 1e-13

        epup = Epup(lam=lam, e=e, pixperpupil=pixperpupil,
                    tip=tip, tilt=tilt)

        gmin = -pixperpupil//2
        gmax = (pixperpupil-1)//2

        ttgrid = epup.get_ttgrid(tip, tilt)

        self.assertTrue(np.abs(ttgrid[0, 0] - (tip*gmin + tilt*gmin))
                        < tol)
        self.assertTrue(np.abs(ttgrid[0, -1] - (tip*gmax + tilt*gmin))
                        < tol)
        self.assertTrue(np.abs(ttgrid[-1, 0] - (tip*gmin + tilt*gmax))
                        < tol)
        self.assertTrue(np.abs(ttgrid[-1, -1] - (tip*gmax + tilt*gmax))
                        < tol)

        pass


    pass





class TestDMFace(unittest.TestCase):
    """
    Unit test suite for DMFace class
    """
    def setUp(self):
        """Store some nominals, this is not a test but is used by them"""
        self.dm_voltages = {'vmin':0,
                'vmax':100,
                'vneighbor':30.,
                'vcorner':30.,
                'vquant':110.0/2**16,
                'gainmap':4e-9*np.ones((48, 48)),
                'flatmap':np.zeros((48, 48)),
                'tiemap':np.zeros((48, 48)).astype('int')
                }
        self.dm_registration = {'nact':48,
                    'inf_func':np.ones((91, 91)),
                    'ppact_d':10,
                    'ppact_cx':4,
                    'ppact_cy':4,
                    'dx':0,
                    'dy':0,
                    'thact':0,
                    'flipx':False}
        self.dmvobj = DM(**self.dm_voltages)
        self.dmz = 0.0 # meters
        self.dmpitch = 1.0e-3 # meters per actuator


    def test_constructor(self):
        """Simple constructor test"""
        DMFace(self.dmz, self.dmpitch, self.dmvobj,
               self.dm_registration)
        pass

    def test_z_realscalar(self):
        """Check DM distance type valid"""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                DMFace(z=perr, pitch=self.dmpitch,
                       dmvobj=self.dmvobj, registration=self.dm_registration)
                pass
            pass
        pass

    def test_pitch_realpositivescalar(self):
        """Check DM pitch type valid"""
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                DMFace(z=self.dmz, pitch=perr,
                       dmvobj=self.dmvobj, registration=self.dm_registration)
                pass
            pass
        pass

    def test_dmvobj_DMclass(self):
        """Check DM voltage object class type valid"""
        for perr in [{'test':'test'}, [], 'text', 5.]:
            with self.assertRaises(TypeError):
                DMFace(z=self.dmz, pitch=self.dmpitch,
                       dmvobj=perr, registration=self.dm_registration)
                pass
            pass
        pass

    def test_missing_required_registration_keys(self):
        """Check missing registration keys raise an Exception"""
        treg = self.dm_registration.copy()
        del treg['nact']

        with self.assertRaises(KeyError):
            DMFace(z=self.dmz, pitch=self.dmpitch,
                   dmvobj=self.dmvobj, registration=treg)
            pass
        pass

    def test_extra_registration_keys(self):
        """Check extra registration keys raise an Exception"""
        treg = self.dm_registration.copy()
        treg['does_not_exist'] = 12

        with self.assertRaises(KeyError):
            DMFace(z=self.dmz, pitch=self.dmpitch,
                   dmvobj=self.dmvobj, registration=treg)
            pass
        pass

    ## Remove as currently no optional keys
    # def test_extra_optional_registration_keys_is_OK(self):
    #     """Check optional keys do NOT raise an Exception"""
    #     treg = self.dm_registration.copy()
    #
    #     DMFace(z=self.dmz, pitch=self.dmpitch,
    #            dmvobj=self.dmvobj, registration=treg)
    #     pass








if __name__ == '__main__':
    unittest.main()
