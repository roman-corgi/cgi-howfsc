# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for dmobj.py
"""

import unittest

import numpy as np

from .dmobj import DM

class TestDm(unittest.TestCase):
    """
    Unit tests for DM voltage base class constructor (DM)
    """
    def setUp(self):
        """Set some good defaults we can call and save space"""
        self.vmin = 0.
        self.vmax = 100.
        self.vneighbor = 50.
        self.vcorner = 75.
        self.vquant = 110.0/2**16
        self.gainmap = 1e-9*np.ones((48, 48))
        self.flatmap = np.zeros((48, 48))
        self.tiemap = np.zeros((48, 48)).astype('int')
        pass

    # Constructor
    def test_valid_vmin_vmax(self):
        """Verify valid ranges pass successfully"""
        for vmin, vmax in [(0, 100), (0., 50.), (-1, 1)]:
            DM(vmin=vmin, vmax=vmax,
               vneighbor=self.vneighbor, vcorner=self.vcorner,
               vquant=self.vquant, gainmap=self.gainmap, flatmap=self.flatmap,
               tiemap=self.tiemap, crosstalkfn=None)
            pass
        pass

    def test_invalid_vmin(self):
        """Check minimum voltage type valid"""
        for vmin in [1j, (0,), None, [], 'txt']:
            with self.assertRaises(TypeError):
                DM(vmin=vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap,
                   crosstalkfn=None)
            pass
        pass

    def test_invalid_vmax(self):
        """Check maximum voltage type valid"""
        for vmax in [1j, (0,), None, [], 'txt']:
            with self.assertRaises(TypeError):
                DM(vmin=self.vmin, vmax=vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap,
                   crosstalkfn=None)
            pass
        pass

    def test_invalid_vneighbor(self):
        """Check rectangular neighbor rule limit valid"""
        for vneighbor in [1j, (0,), -1., None, [], 'txt']:
            with self.assertRaises(TypeError):
                DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap,
                   crosstalkfn=None)
            pass
        pass

    def test_invalid_vcorner(self):
        """Check diagonal neighbor rule limit valid"""
        for vcorner in [1j, (0,), -1., None, [], 'txt']:
            with self.assertRaises(TypeError):
                DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap,
                   crosstalkfn=None)
            pass
        pass

    def test_invalid_vquant(self):
        """Check quantization step valid"""
        for vquant in [1j, (0,), -1., None, [], 'txt']:
            with self.assertRaises(TypeError):
                DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap,
                   crosstalkfn=None)
            pass
        pass

    def test_vmin_gteq_vmax(self):
        """
        Check behavior when minimum voltage is not strictly less than
        maximum voltage
        """
        for vmin, vmax in [(100, 0), (50, 50)]:
            with self.assertRaises(ValueError):
                DM(vmin=vmin, vmax=vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap,
                   crosstalkfn=None)
            pass
        pass

    def test_invalid_gainmap(self):
        """Check gainmap array type valid"""
        for gainmap in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap,
                   crosstalkfn=None)
            pass
        pass

    def test_invalid_flatmap_type(self):
        """Check flatmap array type valid"""
        for flatmap in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=flatmap, tiemap=self.tiemap,
                   crosstalkfn=None)
            pass
        pass

    def test_invalid_flatmap_range(self):
        """Check flatmap array range valid (don't flatten out of spec)"""
        for flatmap in [(self.vmin -1.)*np.ones((48, 48)),
                        (self.vmax +1.)*np.ones((48, 48))]:
            with self.assertRaises(TypeError):
                DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=flatmap, tiemap=self.tiemap,
                   crosstalkfn=None)
            pass
        pass


    def test_invalid_tiemap_type(self):
        """Check tiemap array type valid"""
        for tiemap in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=tiemap,
                   crosstalkfn=None)
            pass
        pass


    def test_invalid_tiemap_range(self):
        """Check tiemap array contents meet tiemap spec"""
        for tiemap in [(-3)*np.ones((48, 48)).astype('int'), # < -1
                        (1.5)*np.ones((48, 48)), # not int
                        2*np.eye(48).astype('int'), # non-consecutive groups
                        ]:
            with self.assertRaises(TypeError):
                DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=tiemap,
                   crosstalkfn=None)
            pass
        pass


class TestVoltsToDmh(unittest.TestCase):
    """
    Unit test suite for volts_to_dmh()
    """
    def setUp(self):
        """Set some good defaults we can call and save space"""
        self.vmin = 0.
        self.vmax = 100.
        self.vneighbor = 50.
        self.vcorner = 75.
        self.vquant = 110.0/2**16
        self.gainmap = 1e-9*np.ones((48, 48))
        self.flatmap = np.zeros((48, 48))
        self.tiemap = np.zeros((48, 48)).astype('int')
        self.dm = DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap)
        pass

    def test_correct_conversion(self):
        """Check volts converted to actuator height as expected"""
        lam = 500e-9
        volts = 50*np.ones((48, 48))
        dmh = self.dm.volts_to_dmh(volts=volts, lam=lam)
        self.assertTrue((dmh == 0.2*np.pi*np.ones((48, 48))).all())


    def test_invalid_volts(self):
        """Check voltage array type valid"""
        lam = 633e-9
        for volts in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                self.dm.volts_to_dmh(volts=volts, lam=lam)
                pass
            pass
        pass


    def test_invalid_lam(self):
        """Check wavelength type valid"""
        volts = 50*np.ones_like(self.dm.gainmap)
        for lam in [(5,), -8, 0, 1j, [], None, 'text', np.ones_like(volts)]:
            with self.assertRaises(TypeError):
                self.dm.volts_to_dmh(volts=volts, lam=lam)
                pass
            pass
        pass


    def test_volts_not_same_size_as_gainmap(self):
        """Check behavior when input DM size does not match gainmap size"""
        #2D array, but not the right one
        lam = 633e-9
        for volts in [np.ones((47, 48)),
                      np.ones((48, 47)),
                      np.ones((47, 47))]:
            with self.assertRaises(TypeError):
                self.dm.volts_to_dmh(volts=volts, lam=lam)
                pass
            pass
        pass


class TestDmhToVolts(unittest.TestCase):
    """
    Unit test suite for dmh_to_volts()
    """
    def setUp(self):
        """Set some good defaults we can call and save space"""
        self.vmin = 0.
        self.vmax = 100.
        self.vneighbor = 50.
        self.vcorner = 75.
        self.vquant = 110.0/2**16
        self.gainmap = 1e-9*np.ones((48, 48))
        self.flatmap = np.zeros((48, 48))
        self.tiemap = np.zeros((48, 48)).astype('int')
        self.dm = DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap)
        pass

    def test_correct_conversion(self):
        """Check actuator height converted to volts as expected"""
        lam = 500e-9
        dmh = 0.2*np.pi*np.ones((48, 48))
        volts = self.dm.dmh_to_volts(dmh=dmh, lam=lam)
        self.assertTrue((volts == 50*np.ones((48, 48))).all())


    def test_invalid_dmh(self):
        """Check actuator height array type valid"""
        lam = 633e-9
        for dmh in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                self.dm.dmh_to_volts(dmh=dmh, lam=lam)
                pass
            pass
        pass


    def test_invalid_lam(self):
        """Check wavelength type valid"""
        dmh = 0.2*np.pi*np.ones_like(self.dm.gainmap)
        for lam in [(5,), -8, 0, 1j, [], None, 'text', np.ones_like(dmh)]:
            with self.assertRaises(TypeError):
                self.dm.dmh_to_volts(dmh=dmh, lam=lam)
                pass
            pass
        pass


    def test_dmh_not_same_size_as_gainmap(self):
        """
        Check behavior when input actuator array does not match gainmap size
        """
        #2D array, but not the right one
        lam = 633e-9
        for dmh in [np.ones((47, 48)), np.ones((48, 47)), np.ones((47, 47))]:
            with self.assertRaises(TypeError):
                self.dm.dmh_to_volts(dmh=dmh, lam=lam)
                pass
            pass
        pass

class TestVoltsToDmPhys(unittest.TestCase):
    """
    Unit test suite for volts_to_dmphys()
    """
    def setUp(self):
        """Set some good defaults we can call and save space"""
        self.vmin = 0.
        self.vmax = 100.
        self.vneighbor = 50.
        self.vcorner = 75.
        self.vquant = 110.0/2**16
        self.gainmap = 1e-9*np.ones((48, 48))
        self.flatmap = np.zeros((48, 48))
        self.tiemap = np.zeros((48, 48)).astype('int')
        self.dm = DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap)
        pass

    def test_correct_conversion(self):
        """Check volts converted to actuator height as expected"""
        tol = 1e-13
        volts = 50*np.ones((48, 48))
        dmphys = self.dm.volts_to_dmphys(volts=volts)
        self.assertTrue(np.max(np.abs(dmphys - 50e-9*np.ones((48, 48)))) < tol)


    def test_invalid_volts(self):
        """Check voltage array type valid"""
        for volts in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                self.dm.volts_to_dmphys(volts=volts)
                pass
            pass
        pass

    def test_volts_not_same_size_as_gainmap(self):
        """Check behavior when input DM size does not match gainmap size"""
        #2D array, but not the right one
        for volts in [np.ones((47, 48)),
                      np.ones((48, 47)),
                      np.ones((47, 47))]:
            with self.assertRaises(TypeError):
                self.dm.volts_to_dmphys(volts=volts)
                pass
            pass
        pass

class TestDmPhysToVolts(unittest.TestCase):
    """
    Unit test suite for dmphys_to_volts()
    """
    def setUp(self):
        """Set some good defaults we can call and save space"""
        self.vmin = 0.
        self.vmax = 100.
        self.vneighbor = 50.
        self.vcorner = 75.
        self.vquant = 110.0/2**16
        self.gainmap = 1e-9*np.ones((48, 48))
        self.flatmap = np.zeros((48, 48))
        self.tiemap = np.zeros((48, 48)).astype('int')
        self.dm = DM(vmin=self.vmin, vmax=self.vmax,
                   vneighbor=self.vneighbor, vcorner=self.vcorner,
                   vquant=self.vquant, gainmap=self.gainmap,
                   flatmap=self.flatmap, tiemap=self.tiemap)
        pass

    def test_correct_conversion(self):
        """Check actuator height converted to volts as expected"""
        tol = 1e-13
        dmphys = 50e-9*np.ones((48, 48))
        volts = self.dm.dmphys_to_volts(dmphys=dmphys)
        self.assertTrue(np.max(np.abs(volts - 50*np.ones((48, 48)))) < tol)


    def test_invalid_dmphys(self):
        """Check actuator height array type valid"""
        for dmphys in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                self.dm.dmphys_to_volts(dmphys=dmphys)
                pass
            pass
        pass


    def test_dmphys_not_same_size_as_gainmap(self):
        """
        Check behavior when input actuator array does not match gainmap size
        """
        #2D array, but not the right one
        for dmphys in [np.ones((47, 48)),
                       np.ones((48, 47)),
                       np.ones((47, 47))]:
            with self.assertRaises(TypeError):
                self.dm.dmphys_to_volts(dmphys=dmphys)
                pass
            pass
        pass



class TestConstrainDM(unittest.TestCase):
    """
    Unit test suite for constrain_dm()

    Note since the code is mostly a wrapper for dmsmooth(), we'll lean on the
    comprehensive set of tests there to ensure that the constraint function
    returns a valid matrix.
    """
    def setUp(self):
        """Set some good defaults we can call and save space"""
        self.vmin = 0.
        self.vmax = 100.
        self.vneighbor = 50.
        self.vcorner = 75.
        self.vquant = 110.0/2**16
        self.gainmap = 1e-9*np.ones((48, 48))
        self.flatmap = np.zeros((48, 48))
        self.tiemap = np.zeros((48, 48)).astype('int')
        self.dm = DM(vmin=self.vmin, vmax=self.vmax,
                     vneighbor=self.vneighbor, vcorner=self.vcorner,
                     vquant=self.vquant, gainmap=self.gainmap,
                     flatmap=self.flatmap, tiemap=self.tiemap,
                     crosstalkfn=None)
        pass

    def test_output_size(self):
        """
        Check output size matches docs
        """
        vin = np.ones((48, 48))
        out = self.dm.constrain_dm(volts=vin)
        self.assertTrue(out.shape == vin.shape)
        pass

    def test_invalid_volts(self):
        """Check DM volt array type valid"""
        for volts in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                self.dm.constrain_dm(volts=volts)
                pass
            pass
        pass


    def test_complex_volts_no_imag(self):
        """
        Check the function still works if type is complex but input is
        purely real
        """
        vin = 101*np.ones((48, 48)).astype('complex128')
        vin[0, 0] = -1

        self.dm.constrain_dm(volts=vin)
        pass


    def test_complex_volts_imag(self):
        """
        Check function fails in expected way if type is complex and input
        has some imaginary part
        """
        vin = 101*np.ones((48, 48)).astype('complex128')
        vin[0, 0] = -1j

        with self.assertRaises(TypeError):
            self.dm.constrain_dm(volts=vin)
            pass
        pass






if __name__ == '__main__':
    unittest.main()
