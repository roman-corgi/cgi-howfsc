# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for the control strategy checkout
"""
import unittest
import os

import numpy as np

from .parse_cs import Region, CSException
from .parse_cs import create_cs_lists, validate_control_strategy
from .parse_cs import _does_point_overlap_list, is_point_in_box, \
     _does_region_overlap_list, _do_boxes_overlap


class TestCSValidation(unittest.TestCase):
    """
    Tests if the control strategy validation tools validate as expected

    Check three expected returns (T/F/N) for validate function

    """
    def setUp(self):
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        pass


    def test_good_yaml(self):
        """Good YAML runs the lists to completion, with/without files"""
        fn = os.path.join(self.localpath, 'testdata',
                          'ut_good_cs.yaml')

        create_cs_lists(fn, usefiles=True)
        create_cs_lists(fn, usefiles=False)
        pass


    def test_good_yaml_no_decpt(self):
        """
        Good YAML runs the lists to completion, with/without files, even
        despite sloppy handling of floating point numbers (sometimes they
        are parsed as strings if there's not an explicit decimal point)
        """
        fn = os.path.join(self.localpath, 'testdata',
                          'ut_good_cs_no_decpt.yaml')

        create_cs_lists(fn, usefiles=True)
        create_cs_lists(fn, usefiles=False)
        pass


    def test_invalid_fn(self):
        """Invalid inputs caught"""

        xlist = [2, None, np.eye(3), (5, 5)]

        for x in xlist:
            with self.assertRaises(TypeError):
                create_cs_lists(x, usefiles=False)
            pass
        pass


    def test_invalid_usefiles(self):
        """Invalid inputs caught"""
        fn = os.path.join(self.localpath, 'testdata',
                          'ut_good_cs.yaml')

        xlist = [2, None, np.eye(3), (5, 5), 'txt']

        for x in xlist:
            with self.assertRaises(TypeError):
                create_cs_lists(fn, x)
            pass
        pass


    def test_file_does_not_exist(self):
        """Verify missing file in fn caught as expected"""
        fn = 'does_not_exist'
        with self.assertRaises(CSException):
            create_cs_lists(fn, False)
        pass


    def test_file_exists_but_is_not_yaml_type(self):
        """Non-YAML files caught as expected"""
        fn = os.path.join(self.localpath, 'testdata', 'ut_fixedbp.fits')

        self.assertTrue(os.path.exists(fn))
        with self.assertRaises(CSException):
            create_cs_lists(fn, False)
        pass


    def test_file_exists_but_is_not_valid_yaml(self):
        """Text files that don't meet the YAML spec are caught"""
        fn = os.path.join(self.localpath, 'testdata',
                          'ut_not_valid_yaml.yaml')

        self.assertTrue(os.path.exists(fn))
        with self.assertRaises(CSException):
            create_cs_lists(fn, False)
        pass


    def test_file_contains_deviations_from_spec(self):
        """contents of file are caught if they don't match the written spec"""
        fnsufflist = ['ut_too_many_top_level_keys.yaml',
                      'ut_too_few_top_level_keys.yaml',
                      'ut_not_dict_top_level.yaml',
                      'ut_reg_not_list_regions.yaml',
                      'ut_pix_not_list_regions.yaml',
                      'ut_dmm_not_list_regions.yaml',
                      'ut_pro_not_list_regions.yaml',
                      'ut_unp_not_list_regions.yaml',
                      'ut_prb_not_list_regions.yaml',
                      'ut_reg_empty_list.yaml',
                      'ut_pix_empty_list.yaml',
                      'ut_dmm_empty_list.yaml',
                      'ut_pro_empty_list.yaml',
                      'ut_unp_empty_list.yaml',
                      'ut_prb_empty_list.yaml',
                      'ut_reg_too_few_region_keys.yaml',
                      'ut_pix_too_few_region_keys.yaml',
                      'ut_dmm_too_few_region_keys.yaml',
                      'ut_pro_too_few_region_keys.yaml',
                      'ut_unp_too_few_region_keys.yaml',
                      'ut_prb_too_few_region_keys.yaml',
                      'ut_reg_too_many_region_keys.yaml',
                      'ut_pix_too_many_region_keys.yaml',
                      'ut_dmm_too_many_region_keys.yaml',
                      'ut_pro_too_many_region_keys.yaml',
                      'ut_unp_too_many_region_keys.yaml',
                      'ut_prb_too_many_region_keys.yaml',
                      'ut_reg_first_wrong_spec.yaml',
                      'ut_pix_first_wrong_spec.yaml',
                      'ut_dmm_first_wrong_spec.yaml',
                      'ut_pro_first_wrong_spec.yaml',
                      'ut_unp_first_wrong_spec.yaml',
                      'ut_prb_first_wrong_spec.yaml',
                      'ut_reg_last_wrong_spec.yaml',
                      'ut_pix_last_wrong_spec.yaml',
                      'ut_dmm_last_wrong_spec.yaml',
                      'ut_pro_last_wrong_spec.yaml',
                      'ut_unp_last_wrong_spec.yaml',
                      'ut_prb_last_wrong_spec.yaml',
                      'ut_reg_last_before_first.yaml',
                      'ut_pix_last_before_first.yaml',
                      'ut_dmm_last_before_first.yaml',
                      'ut_pro_last_before_first.yaml',
                      'ut_unp_last_before_first.yaml',
                      'ut_prb_last_before_first.yaml',
                      'ut_reg_low_wrong_spec.yaml',
                      'ut_pix_low_wrong_spec.yaml',
                      'ut_dmm_low_wrong_spec.yaml',
                      'ut_pro_low_wrong_spec.yaml',
                      'ut_unp_low_wrong_spec.yaml',
                      'ut_prb_low_wrong_spec.yaml',
                      'ut_reg_high_wrong_spec.yaml',
                      'ut_pix_high_wrong_spec.yaml',
                      'ut_dmm_high_wrong_spec.yaml',
                      'ut_pro_high_wrong_spec.yaml',
                      'ut_unp_high_wrong_spec.yaml',
                      'ut_prb_high_wrong_spec.yaml',
                      'ut_reg_high_before_low.yaml',
                      'ut_pix_high_before_low.yaml',
                      'ut_dmm_high_before_low.yaml',
                      'ut_pro_high_before_low.yaml',
                      'ut_unp_high_before_low.yaml',
                      'ut_prb_high_before_low.yaml',
                      'ut_reg_value_wrong_spec.yaml',
                      'ut_pix_value_wrong_spec.yaml',
                      'ut_dmm_value_wrong_spec.yaml',
                      'ut_pro_value_wrong_spec.yaml',
                      'ut_unp_value_wrong_spec.yaml',
                      'ut_prb_value_wrong_spec.yaml',
                      # no reg val check here; no > or < constraints
                      # no pixel val check here; done in separate test
                      'ut_dmm_value_wrong_value.yaml',
                      'ut_pro_value_wrong_value.yaml',
                      'ut_unp_value_wrong_value.yaml',
                      'ut_prb_value_wrong_value.yaml',
                      'ut_reg_regions_overlap.yaml',
                      'ut_pix_regions_overlap.yaml',
                      'ut_dmm_regions_overlap.yaml',
                      'ut_pro_regions_overlap.yaml',
                      'ut_unp_regions_overlap.yaml',
                      'ut_prb_regions_overlap.yaml',
                      'ut_reg_regions_do_not_cover.yaml',
                      'ut_pix_regions_do_not_cover.yaml',
                      'ut_dmm_regions_do_not_cover.yaml',
                      'ut_pro_regions_do_not_cover.yaml',
                      'ut_unp_regions_do_not_cover.yaml',
                      'ut_prb_regions_do_not_cover.yaml',
                      'ut_fix_wrong_spec.yaml',
                      ]

        for fnsuff in fnsufflist:
            fn = os.path.join(self.localpath, 'testdata', fnsuff)

            self.assertTrue(os.path.exists(fn))
            with self.assertRaises(CSException):
                create_cs_lists(fn, False)
            with self.assertRaises(CSException):
                create_cs_lists(fn, True)
            pass
        pass


    def test_strings_for_nonexistent_files(self):
        """
        Verify that pixelweights and fixedbp only throw errors when the files
        in the YAML don't exist if usefiles=True
        """
        fnsufflist = ['ut_pix_file_does_not_exist.yaml',
                      'ut_fix_file_does_not_exist.yaml',
                      ]


        for fnsuff in fnsufflist:
            fn = os.path.join(self.localpath, 'testdata', fnsuff)

            self.assertTrue(os.path.exists(fn))
            create_cs_lists(fn, False) # just returns, no exception
            with self.assertRaises(IOError):
                create_cs_lists(fn, True)
            pass
        pass


    def test_validation_outcomes(self):
        """
        Check that the three validation wrapper outcomes are produces as
        expected
        """

        # True if good
        fnsufflist = ['ut_good_cs.yaml',
                      'ut_good_cs_no_decpt.yaml'
                      ]

        for fnsuff in fnsufflist:
            fn = os.path.join(self.localpath, 'testdata', fnsuff)

            self.assertTrue(os.path.exists(fn))
            ret = validate_control_strategy(fn, usefiles=True, verbose=False)
            self.assertTrue(ret)
            pass

        # False if bad
        fnsufflist = ['ut_too_many_top_level_keys.yaml',
                      'ut_too_few_top_level_keys.yaml',
                      'ut_not_dict_top_level.yaml',
                      ]

        for fnsuff in fnsufflist:
            fn = os.path.join(self.localpath, 'testdata', fnsuff)

            self.assertTrue(os.path.exists(fn))
            ret = validate_control_strategy(fn, usefiles=True, verbose=False)
            self.assertFalse(ret)
            pass

        # None if it can't be evaluated with usefiles due to missing files
        fnsufflist = ['ut_pix_file_does_not_exist.yaml',
                      'ut_fix_file_does_not_exist.yaml',
                      ]

        for fnsuff in fnsufflist:
            fn = os.path.join(self.localpath, 'testdata', fnsuff)

            self.assertTrue(os.path.exists(fn))
            ret = validate_control_strategy(fn, usefiles=True, verbose=False)
            self.assertTrue(ret is None)
            pass

        pass



class TestBoxOverlapChecks(unittest.TestCase):
    """
    Tests if two regions overlap each other
    """

    def test_regions_do_not_overlap(self):
        """Verify nonoverlapping regions return False"""

        ref = Region(first=3, last=5, low=1e-7, high=1e-5, value=0)

        disjoint = Region(first=7, last=10, low=1e-9, high=1e-8, value=0)
        adjfirst = Region(first=1, last=2, low=1e-7, high=1e-5, value=0)
        adjlast = Region(first=6, last=8, low=1e-7, high=1e-5, value=0)
        adjlow = Region(first=3, last=5, low=1e-9, high=1e-7, value=0)
        adjhigh = Region(first=3, last=5, low=1e-5, high=1e-4, value=0)

        for r in [disjoint, adjfirst, adjlast, adjlow, adjhigh]:
            self.assertFalse(_do_boxes_overlap(ref, r))
            self.assertFalse(_do_boxes_overlap(r, ref))
            pass

        pass


    def test_regions_do_overlap(self):
        """Verify overlapping regions return True"""

        ref = Region(first=3, last=5, low=1e-7, high=1e-5, value=0)

        corner = Region(first=4, last=6, low=1e-6, high=1e-4, value=0)
        side = Region(first=4, last=4, low=1e-6, high=1e-4, value=0)
        cross = Region(first=4, last=4, low=1e-9, high=1e-4, value=0)
        overlay = Region(first=2, last=6, low=1e-9, high=1e-4, value=0)

        for r in [corner, side, cross, overlay]:
            self.assertTrue(_do_boxes_overlap(ref, r))
            self.assertTrue(_do_boxes_overlap(r, ref))
            pass
        pass


    def test_nonoverlap_list(self):
        """Verify lists of nonoverlapping regions return False"""

        ref = Region(first=3, last=5, low=1e-7, high=1e-5, value=0)

        disjoint = Region(first=7, last=10, low=1e-9, high=1e-8, value=0)
        adjfirst = Region(first=1, last=2, low=1e-7, high=1e-5, value=0)
        adjlast = Region(first=6, last=8, low=1e-7, high=1e-5, value=0)
        adjlow = Region(first=3, last=5, low=1e-9, high=1e-7, value=0)
        adjhigh = Region(first=3, last=5, low=1e-5, high=1e-4, value=0)
        corner = Region(first=4, last=6, low=1e-6, high=1e-4, value=0)

        rlist = [disjoint, adjfirst, adjlast, adjlow, adjhigh, corner]

        self.assertTrue(_does_region_overlap_list(ref, rlist))

        pass


    def test_overlap_list(self):
        """Verify lists with an overlapping region return True"""

        ref = Region(first=3, last=5, low=1e-7, high=1e-5, value=0)

        disjoint = Region(first=7, last=10, low=1e-9, high=1e-8, value=0)
        adjfirst = Region(first=1, last=2, low=1e-7, high=1e-5, value=0)
        adjlast = Region(first=6, last=8, low=1e-7, high=1e-5, value=0)
        adjlow = Region(first=3, last=5, low=1e-9, high=1e-7, value=0)
        adjhigh = Region(first=3, last=5, low=1e-5, high=1e-4, value=0)

        rlist = [disjoint, adjfirst, adjlast, adjlow, adjhigh]

        self.assertFalse(_does_region_overlap_list(ref, rlist))

        pass




class TestPointOverlapChecks(unittest.TestCase):
    """
    Tests if a point is in a box/range/list of boxes
    """

    def setUp(self):
        self.iter_low = 1
        self.first = 3
        self.iter_mid = 4
        self.last = 5
        self.iter_high = np.inf

        self.cont_low = 0
        self.low = 1e-7
        self.cont_mid = 1e-6
        self.high = 1e-5
        self.cont_high = np.inf

        self.value = 0
        self.testr = Region(first=self.first,
                            last=self.last,
                            low=self.low,
                            high=self.high,
                            value=self.value)

        self.r1 = Region(first=self.first,
                         last=self.first,
                         low=self.low,
                         high=self.cont_mid,
                         value=self.value)

        self.r2 = Region(first=self.first,
                         last=self.first,
                         low=self.cont_mid,
                         high=self.high,
                         value=self.value)

        self.r3 = Region(first=self.last,
                         last=self.last,
                         low=self.low,
                         high=self.cont_mid,
                         value=self.value)

        self.r4 = Region(first=self.last,
                         last=self.last,
                         low=self.cont_mid,
                         high=self.high,
                         value=self.value)

        pass


    def test_point_on_edge(self):
        """Edge points captured correctly"""

        # First is inclusive
        self.assertTrue(is_point_in_box(iteration=self.first,
                                        contrast=self.cont_mid,
                                        region=self.testr))

        # Last is inclusive
        self.assertTrue(is_point_in_box(iteration=self.last,
                                        contrast=self.cont_mid,
                                        region=self.testr))

        # Low is inclusive
        self.assertTrue(is_point_in_box(iteration=self.iter_mid,
                                        contrast=self.low,
                                        region=self.testr))

        # High is exclusive
        self.assertFalse(is_point_in_box(iteration=self.iter_mid,
                                         contrast=self.high,
                                         region=self.testr))

        pass


    def test_point_inside(self):
        """Interior point captured correctly"""
        self.assertTrue(is_point_in_box(iteration=self.iter_mid,
                                        contrast=self.cont_mid,
                                        region=self.testr))

        pass


    def test_point_outside(self):
        """Exterior point captured correctly"""

        iclist = [(self.iter_mid, self.cont_low),
                  (self.iter_mid, self.cont_high),
                  (self.iter_low, self.cont_mid),
                  (self.iter_high, self.cont_mid),
                  (self.iter_low, self.cont_low),
                  (self.iter_low, self.cont_high),
                  (self.iter_high, self.cont_low),
                  (self.iter_high, self.cont_high),
                  ]

        for iteration, contrast in iclist:
            self.assertFalse(is_point_in_box(iteration, contrast, self.testr))
            pass
        pass


    def test_nonoverlapping_region_list(self):
        """
        Check that a point that doesn't overlap a region list returns False
        """

        rlist = [self.r1, self.r2, self.r3, self.r4]

        iteration = self.iter_low
        contrast = self.cont_low

        self.assertFalse(_does_point_overlap_list(iteration, contrast, rlist))
        pass


    def test_overlapping_region_list(self):
        """
        Check that a point that does overlap a region list returns True
        """

        rlist = [self.r1, self.r2, self.r3, self.r4]

        iteration = self.last
        contrast = self.cont_mid

        self.assertTrue(_does_point_overlap_list(iteration, contrast, rlist))
        pass






if __name__ == '__main__':
    unittest.main()
