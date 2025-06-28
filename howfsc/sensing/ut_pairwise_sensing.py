# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit test classes for pairwise probing
"""

import unittest
import numpy as np
from . import pairwise_sensing as ps

def allclose_w_nan(a, b):
    """checks if all elements in an array are the same or nans"""
    assert a.shape == b.shape
    return np.all(np.isclose(a, b) | np.isnan(a) | np.isnan(b))

def standard_efield():
    """a known uniform electric field"""
    return np.ones((2, 2))+1j*np.ones((2, 2))

def standard_data():
    """A standard dataset with an input electric field"""
    ET = standard_efield()
    I0 = np.abs(ET)**2

    Cpert1 = np.ones((2, 2))*0.01*1j
    Iplus1 = np.abs(ET + 1j*Cpert1)**2
    Iminus1 = np.abs(ET + -1j*Cpert1)**2
    phase1 = np.angle(Cpert1)

    Cpert2 = np.ones((2, 2))*0.01
    Iplus2 = np.abs(ET + 1j*Cpert2)**2
    Iminus2 = np.abs(ET + -1j*Cpert2)**2
    phase2 = np.angle(Cpert2)

    intensities = np.array([I0, Iplus1, Iminus1, Iplus2, Iminus2])
    phases = np.array([phase1, phase2])
    return intensities, phases

def standard_data_w_nans():
    """A standard dataset with an input electric field, with
    one nonresponsive probe pixel"""
    ET = standard_efield()
    I0 = np.abs(ET)**2

    Cpert1 = np.ones((2, 2))*0.01*1j
    Cpert1[1, 1] = 0
    Iplus1 = np.abs(ET + 1j*Cpert1)**2
    Iminus1 = np.abs(ET + -1j*Cpert1)**2
    phase1 = np.angle(Cpert1)

    Cpert2 = np.ones((2, 2))*0.01
    Cpert2[1, 1] = 0
    Iplus2 = np.abs(ET + 1j*Cpert2)**2
    Iminus2 = np.abs(ET + -1j*Cpert2)**2
    phase2 = np.angle(Cpert2)

    intensities = np.array([I0, Iplus1, Iminus1, Iplus2, Iminus2])
    phases = np.array([phase1, phase2])
    return intensities, phases



class Test_getmatrix(unittest.TestCase):
    """Tests the getmatrix function"""

    def setUp(self):
        """Preload test data"""
        self.intensities, self.phases = standard_data()
        self.intensities_w_nans, self.phases_w_nans = standard_data_w_nans()
        pass

    def test_invalid_intensities(self):
        """Fails when given invalid input"""
        for err in [1j, np.ones((2,)), np.eye(2), 1j*self.intensities,
                    None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.get_matrix(intensities=err, phases=self.phases)
                pass
            pass
        pass

    def test_invalid_phases(self):
        """Fails when given invalid input"""
        for err in [1j, np.ones((2,)), np.eye(2), 1j*self.phases,
                    None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.get_matrix(intensities=self.intensities, phases=err)
                pass
            pass
        pass

    def test_mismatch_in_number_of_slices(self):
        """Fails when # of intensity slices is not 1 + 2*# of phase slices"""
        slicedint = self.intensities[:-1, :, :]

        with self.assertRaises(TypeError):
            ps.get_matrix(intensities=slicedint, phases=self.phases)
            pass
        pass


    def test_standard_data_deltas(self):
        """
        For a standard data set without invalid pixels, verify output #0
        of getmatrix() matches analytic assumptions
        """
        self.assertTrue(
            allclose_w_nan(ps.get_matrix(self.intensities, self.phases)[0],
                                        np.array([np.full((2, 2), -0.02),
                                                  np.full((2, 2), 0.02)])))
        pass

    def test_standard_data_real_deltaps(self):
        """
        For a standard data set without invalid pixels, verify output #1
        of getmatrix() matches analytic assumptions
        """
        self.assertTrue(
            allclose_w_nan(ps.get_matrix(self.intensities, self.phases)[1],
                                        np.array([np.full((2, 2), 0.0),
                                                  np.full((2, 2), 0.01)])))
        pass

    def test_standard_data_imag_deltaps(self):
        """
        For a standard data set without invalid pixels, verify output #2
        of getmatrix() matches analytic assumptions
        """
        self.assertTrue(
            allclose_w_nan(ps.get_matrix(self.intensities, self.phases)[2],
                                        np.array([np.full((2, 2), 0.01),
                                                  np.full((2, 2), 0.0)])))
        pass

    def test_standard_data_badpix(self):
        """
        For a standard data set without invalid pixels, verify output #3
        of getmatrix() matches analytic assumptions
        """
        self.assertTrue(
            allclose_w_nan(ps.get_matrix(self.intensities, self.phases)[3],
                              np.array([np.full((2, 2), False, dtype=bool),
                                        np.full((2, 2), False, dtype=bool)])))
        pass

    def test_standard_data_wnans_deltas(self):
        """
        For a standard data set with nonphysical probe results at a pixel,
        verify output #0 of getmatrix() matches analytic assumptions
        """
        self.assertTrue(
            allclose_w_nan(
                ps.get_matrix(self.intensities_w_nans, self.phases_w_nans)[0],
                                        np.array([np.full((2, 2), -0.02),
                                                  np.full((2, 2), 0.02)])))
        pass

    def test_standard_data_wnans_real_deltaps(self):
        """
        For a standard data set with nonphysical probe results at a pixel,
        verify output #1 of getmatrix() matches analytic assumptions
        """
        self.assertTrue(
            allclose_w_nan(
                ps.get_matrix(self.intensities_w_nans, self.phases_w_nans)[1],
                                        np.array([np.full((2, 2), 0.0),
                                                  np.full((2, 2), 0.01)])))
        pass

    def test_standard_data_wnans_imag_deltaps(self):
        """
        For a standard data set with nonphysical probe results at a pixel,
        verify output #2 of getmatrix() matches analytic assumptions
        """
        self.assertTrue(
            allclose_w_nan(
                ps.get_matrix(self.intensities_w_nans, self.phases_w_nans)[2],
                                        np.array([np.full((2, 2), 0.01),
                                                  np.full((2, 2), 0.0)])))
        pass

    def test_standard_data_wnans_badpix(self):
        """
        For a standard data set with nonphysical probe results at a pixel,
        verify output #3 of getmatrix() matches analytic assumptions
        """
        answer = np.array([np.full((2, 2), False, dtype=bool),
                 np.full((2, 2), False, dtype=bool)])
        answer[:, 1, 1] = True
        self.assertTrue(
            allclose_w_nan(
                ps.get_matrix(self.intensities_w_nans, self.phases_w_nans)[3],
                answer))
        pass


class Test_getcoefficients(unittest.TestCase):
    """Tests the getcoefficients function"""

    def test_invalid_i0(self):
        """Fails when given invalid input"""
        for err in [1j, np.eye(2), None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.get_coefficients(i0=err,
                                    iplus=np.array([[1]]),
                                    iminus=np.array([[1]]),
                                    phase_est=np.array([[0]]))
                pass
            pass
        pass


    def test_invalid_iplus(self):
        """Fails when given invalid input"""
        for err in [1j, np.eye(2), None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.get_coefficients(i0=np.array([[0]]),
                                    iplus=err,
                                    iminus=np.array([[1]]),
                                    phase_est=np.array([[0]]))
                pass
            pass
        pass

    def test_invalid_iminus(self):
        """Fails when given invalid input"""
        for err in [1j, np.eye(2), None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.get_coefficients(i0=np.array([[0]]),
                                    iplus=np.array([[1]]),
                                    iminus=err,
                                    phase_est=np.array([[0]]))
                pass
            pass
        pass

    def test_invalid_phase_est(self):
        """Fails when given invalid input"""
        for err in [1j, np.eye(2), None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.get_coefficients(i0=np.array([[0]]),
                                    iplus=np.array([[1]]),
                                    iminus=np.array([[1]]),
                                    phase_est=err)
                pass
            pass
        pass



    def test_equal_intensities(self):
        """
        Verify that probes with 0 phase shift recover the expected intensities
        and a purely real probe
        """
        self.assertTrue(np.allclose(
                           ps.get_coefficients(i0=np.array([[0]]),
                                               iplus=np.array([[1]]),
                                               iminus=np.array([[1]]),
                                               phase_est=np.array([[0]])),
                           (np.array([[0]]),
                            np.array([[1]]),
                            np.array([[0]]),
                            np.array([[False]])), equal_nan=True))
        pass

    def test_equal_intensities_offset(self):
        """
        Verify that purely real probes not centered around zero e-field
        recover expected values
        """
        self.assertTrue(np.allclose(
                           ps.get_coefficients(i0=np.array([[2]]),
                                               iplus=np.array([[8]]),
                                               iminus=np.array([[0]]),
                                               phase_est=np.array([[0]])),
                           (np.array([[4]]),
                            np.array([[np.sqrt(2)]]),
                            np.array([[0]]),
                            np.array([[False]])), equal_nan=True))
        pass

    def test_equal_intensities_45(self):
        """
        Recover the pi/4 phase shift expected for equal intensities of probe
        """
        self.assertTrue(np.allclose(
                           ps.get_coefficients(i0=np.array([[0]]),
                                               iplus=np.array([[1]]),
                                               iminus=np.array([[1]]),
                                              phase_est=np.array([[np.pi/4]])),
                           (np.array([[0]]),
                            np.array([[np.sqrt(2)/2]]),
                            np.array([[np.sqrt(2)/2]]),
                            np.array([[False]])), equal_nan=True))
        pass

    def test_failed_probes(self):
        """
        These probes should not give reasonable numbers as the intensity of the
        probes is LOWER than the intensity of the image by itself
        """
        self.assertTrue(np.allclose(
                           ps.get_coefficients(i0=np.array([[1]]),
                                               iplus=np.array([[0.5]]),
                                               iminus=np.array([[0.5]]),
                                               phase_est=np.array([[0]])),
                           (np.array([[np.nan]]),
                            np.array([[np.nan]]),
                            np.array([[np.nan]]),
                            np.array([[True]])), equal_nan=True))
        pass


class Test_solve_matrix(unittest.TestCase):
    """Tests the solve_matrix function"""
    def setUp(self):
        """Preload test data"""
        self.E0 = standard_efield()
        self.intensities, self.phases = standard_data()
        self.intensities_w_nans, self.phases_w_nans = standard_data_w_nans()
        pass

    def test_invalid_deltas(self):
        """Fails when given invalid input"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = self.intensities[0, :, :]

        for err in [1j, np.eye(2), 1j*d, None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.solve_matrix(err, r, i, b, I, 2, np.inf, 0)
                pass
            pass
        pass

    def test_invalid_real_deltaps(self):
        """Fails when given invalid input"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = self.intensities[0, :, :]

        for err in [1j, np.eye(2), 1j*r, None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.solve_matrix(d, err, i, b, I, 2, np.inf, 0)
                pass
            pass
        pass

    def test_invalid_imag_deltaps(self):
        """Fails when given invalid input"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = self.intensities[0, :, :]

        for err in [1j, np.eye(2), 1j*i, None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.solve_matrix(d, r, err, b, I, 2, np.inf, 0)
                pass
            pass
        pass

    def test_invalid_i0(self):
        """Fails when given invalid input"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)

        for err in [1j, np.ones((2, 2, 2)), 1j*b, None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.solve_matrix(d, r, i, b, err, 2, np.inf, 0)
                pass
            pass
        pass

    def test_invalid_badpixels(self):
        """Fails when given invalid input"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = self.intensities[0, :, :]

        for err in [1j, np.eye(2), 1j*b, None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.solve_matrix(d, r, i, err, I, 2, np.inf, 0)
                pass
            pass
        pass


    def test_invalid_min_good_probes(self):
        """Fails when given invalid input"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = self.intensities[0, :, :]

        for err in [1j, np.ones((2,)), np.eye(2), 1j*self.intensities,
                    None, 'txt', [1], 0.5, -1.5]:
            with self.assertRaises(TypeError):
                ps.solve_matrix(d, r, i, b, I, err, np.inf, 0)
            pass
        pass


    def test_thresh_min_good_probes_at_2(self):
        """Need at least 2 good measurements to have rank for inversion"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = self.intensities[0, :, :]

        for err in [-1, 0, 1]:
            with self.assertRaises(ValueError):
                ps.solve_matrix(d, r, i, b, I, err, np.inf, 0)
            pass

        for err in [2, 3]:
            ps.solve_matrix(d, r, i, b, I, err, np.inf, 0)
            pass

        pass


    def test_nonbool_badpixels(self):
        """Fails when the input badpixels array is not booleans"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = self.intensities[0, :, :]
        b = b.astype(float)

        with self.assertRaises(TypeError):
            ps.solve_matrix(d, r, i, b, I, 2, np.inf, 0)
            pass
        pass

    def test_standard_data(self):
        """Matches input phase for a simple analytic checkout"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = self.intensities[0, :, :]
        answer = ps.solve_matrix(d, r, i, b, I, 2, np.inf, 0)
        self.assertTrue(allclose_w_nan(self.E0,
                                       answer[0] + 1j*answer[1]))
        pass

    def test_standard_data_w_nans(self):
        """
        Matches input phase on non-nan pixels for a simple analytic
        checkout
        """
        d, r, i, b = ps.get_matrix(self.intensities_w_nans, self.phases_w_nans)
        I = self.intensities[0, :, :]
        answer = ps.solve_matrix(d, r, i, b, I, 2, np.inf, 0)
        self.assertTrue(allclose_w_nan(self.E0,
                                       answer[0] + 1j*answer[1]))
        pass


    def test_eestcondlim_trip(self):
        """Verify eestcondlim functionality on rank-1"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        d[1, 0, 0] = d[0, 0, 0]
        r[1, 0, 0] = r[0, 0, 0]
        i[1, 0, 0] = i[0, 0, 0]
        b[1, 0, 0] = b[0, 0, 0]
        I = self.intensities[0, :, :]

        eestcondlim = 0.4 # anything > 0 is fine
        answer = ps.solve_matrix(d, r, i, b, I, 2, np.inf, eestcondlim)
        self.assertTrue(allclose_w_nan(self.E0,
                                       answer[0] + 1j*answer[1]))
        self.assertTrue(np.isnan(answer[0][0, 0]))
        self.assertTrue(np.isnan(answer[1][0, 0]))


    def test_eestcondlim_notrip(self):
        """Verify eestcondlim functionality on rank-2"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = self.intensities[0, :, :]

        eestcondlim = 0.4
        answer = ps.solve_matrix(d, r, i, b, I, 2, np.inf, eestcondlim)
        self.assertTrue(allclose_w_nan(self.E0,
                                       answer[0] + 1j*answer[1]))
        self.assertFalse(np.isnan(answer[0][0, 0]))
        self.assertFalse(np.isnan(answer[1][0, 0]))


    def test_eestclip_trip(self):
        """Verify eestclip functionality with bad intensity"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = 0*self.intensities[0, :, :]

        eestclip = 0.1 # anything < np.inf is fine
        answer = ps.solve_matrix(d, r, i, b, I, 2, eestclip, 0)
        self.assertTrue(allclose_w_nan(self.E0,
                                       answer[0] + 1j*answer[1]))
        self.assertTrue(np.isnan(answer[0][0, 0]))
        self.assertTrue(np.isnan(answer[1][0, 0]))


    def test_eestclip_notrip(self):
        """Verify eestclip functionality with good intensity"""
        d, r, i, b = ps.get_matrix(self.intensities, self.phases)
        I = self.intensities[0, :, :]

        eestclip = 0.1
        answer = ps.solve_matrix(d, r, i, b, I, 2, eestclip, 0)
        self.assertTrue(allclose_w_nan(self.E0,
                                       answer[0] + 1j*answer[1]))
        self.assertFalse(np.isnan(answer[0][0, 0]))
        self.assertFalse(np.isnan(answer[1][0, 0]))




class Test_estimate_efield(unittest.TestCase):
    """Tests the base estimation function on 2x2 images"""
    def setUp(self):
        """Preload test data"""
        self.E0 = standard_efield()
        self.intensities, self.phases = standard_data()
        self.intensities_w_nans, self.phases_w_nans = standard_data_w_nans()
        pass

    def test_invalid_intensities(self):
        """Fails when given invalid input"""
        for err in [1j, np.ones((2,)), np.eye(2), 1j*self.intensities,
                    None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.estimate_efield(intensities=err, phases=self.phases)
                pass
            pass
        pass

    def test_invalid_phases(self):
        """Fails when given invalid input"""
        for err in [1j, np.ones((2,)), np.eye(2), 1j*self.phases,
                    None, 'txt', [1]]:
            with self.assertRaises(TypeError):
                ps.estimate_efield(intensities=self.intensities, phases=err)
                pass
            pass
        pass


    def test_invalid_min_good_probes_type(self):
        """Fails when given invalid input"""
        for err in [1j, np.ones((2,)), np.eye(2), 1j*self.intensities,
                    None, 'txt', [1], 1.5, -1.5]:
            with self.assertRaises(TypeError):
                ps.estimate_efield(intensities=self.intensities,
                                   phases=self.phases,
                                   min_good_probes=err,
                )
                pass
            pass
        pass


    def test_invalid_min_good_probes_value(self):
        """Fails when given invalid input"""
        for err in [1, 0, -2]:
            with self.assertRaises(ValueError):
                ps.estimate_efield(intensities=self.intensities,
                                   phases=self.phases,
                                   min_good_probes=err,
                )
                pass
            pass
        pass


    def test_invalid_eestclip(self):
        """Fails when given invalid input"""
        for err in [1j, np.ones((2,)), np.eye(2), 1j*self.phases,
                    None, 'txt', [1], -1, -1.5]:
            with self.assertRaises(TypeError):
                ps.estimate_efield(intensities=self.intensities,
                                   phases=self.phases,
                                   eestclip=err,
                )
                pass
            pass
        pass


    def test_invalid_eestcondlim(self):
        """Fails when given invalid input"""
        for err in [1j, np.ones((2,)), np.eye(2), 1j*self.phases,
                    None, 'txt', [1], -1, -1.5]:
            with self.assertRaises(TypeError):
                ps.estimate_efield(intensities=self.intensities,
                                   phases=self.phases,
                                   eestcondlim=err,
                )
                pass
            pass
        pass

    def test_mismatch_in_number_of_slices(self):
        """Fails when # of intensity slices is not 1 + 2*# of phase slices"""
        slicedint = self.intensities[:-1, :, :]

        with self.assertRaises(TypeError):
            ps.estimate_efield(intensities=slicedint, phases=self.phases)
            pass
        pass


    def test_standard_data(self):
        """Match analytic e-field estimate"""
        self.assertTrue(allclose_w_nan(self.E0,
              ps.estimate_efield(self.intensities, self.phases)))


    def test_standard_data_w_nans(self):
        """Match analytic e-field estimate at non-nan points"""
        output = ps.estimate_efield(self.intensities_w_nans,
                                 self.phases_w_nans)
        self.assertTrue(allclose_w_nan(self.E0, output))
        self.assertTrue(np.isnan(output[1, 1])) # index in w_nans data


    def test_min_good_probes(self):
        """Verify min good probes functionality"""
        # it's OK to copy since it's already rank 2
        intensities3 = np.empty((7, 2, 2))
        intensities3[:5, :, :] = self.intensities
        intensities3[5:, :, :] = self.intensities[-2:, :, :]
        phases3 = np.empty((3, 2, 2))
        phases3[:2, :, :] = self.phases
        phases3[2:, :, :] = self.phases[-1:, :, :]

        output = ps.estimate_efield(intensities3, phases3, min_good_probes=3)

        self.assertTrue(allclose_w_nan(self.E0, output))
        self.assertTrue(~np.isnan(output).any())


    def test_eestcondlim(self):
        """Verify eestcondlim functionality"""
        # copy first probe over so rank is 1 instead of 2
        intensities = np.empty((5, 2, 2))
        intensities[:3, :, :] = self.intensities[:3, :, :]
        intensities[3:, :, :] = self.intensities[1:3, :, :]
        phases = np.empty((2, 2, 2))
        phases[0, :, :] = self.phases[0, :, :]
        phases[1, :, :] = self.phases[0, :, :]

        output = ps.estimate_efield(intensities, phases, eestcondlim=0.4)

        self.assertTrue(allclose_w_nan(self.E0, output))
        self.assertTrue(np.isnan(output).all())


    def test_eestclip(self):
        """Verify eestclip functionality"""
        # Set i0 to zero so coherent is always infinitely larger
        i0 = self.intensities[0, :, :]
        intensities = self.intensities.copy()
        intensities -= i0

        output = ps.estimate_efield(intensities, self.phases, eestclip=0.1)

        self.assertTrue(allclose_w_nan(self.E0, output))
        self.assertTrue(np.isnan(output).all())



if __name__ == '__main__':
    unittest.main()
