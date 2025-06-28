# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=unsubscriptable-object
"""
Unit tests for the main HOWFSC computation block
"""

import unittest
import copy
from unittest.mock import patch
import os
import logging
import warnings
import sys

import numpy as np

from eetc.excam_tools import EXCAMOptimizeException

from howfsc.util.loadyaml import loadyaml
from howfsc.observe import tech_demo_obs

# Keep logger spam out of unit test results
if sys.platform.startswith('win'):
    logging.basicConfig(filename='NUL')
else:
    logging.basicConfig(filename='/dev/null')

class TestTechDemoObs(unittest.TestCase):
    """
    Unit tests for routine to prepare camera settings for the upcoming
    observation

    success, with values that go down analog and PC branches
    invalid input
    form of output
    exact value test, if that's possible (might be too complex, do in
     functional).  Maybe scaling with different values for obs star? Maybe
     can mock out some of the internal functions?
    pcfrac/pc_ecount_max do the right things

    """

    def setUp(self):
        # set up undersized images
        self.nlam = 3
        self.npix = 4
        self.ndm = 7

        rng = np.random.default_rng(5551212)
        self.framelist = []
        self.prev_exptime_list = []
        self.n2clist = []

        for _ in range(self.nlam):
            for dummy in range(self.ndm):
                self.framelist.append(100*rng.random((self.npix, self.npix)))
                self.prev_exptime_list.append(10)
                pass
            pass


        hconffile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'control', 'testdata',
                                 'ut_observe_config_good.yaml')
        self.hconf = loadyaml(hconffile, custom_exception=TypeError)

        self.roi = np.ones((self.npix, self.npix), dtype=bool)
        self.whichstar = 'reference'
        self.pc_ecount_max = 0.1
        self.pcfrac = 0.025 # 2.5% (~2-sigma 1-sided)

        pass


    def test_success(self):
        """Good inputs return without incident"""
        tech_demo_obs(
            framelist=self.framelist,
            roi=self.roi,
            whichstar='reference',
            hconf=self.hconf,
            pc_ecount_max=self.pc_ecount_max,
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )

        tech_demo_obs(
            framelist=self.framelist,
            roi=self.roi,
            whichstar='target',
            hconf=self.hconf,
            pc_ecount_max=self.pc_ecount_max,
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )
        pass


    def test_success_mean_percentile(self):
        """Check valid scale/scale_bright options all complete"""

        this_hconf = self.hconf.copy()

        # 1
        this_hconf['excam']['scale_method'] = 'mean'
        this_hconf['excam']['scale_bright_method'] = 'mean'

        tech_demo_obs(
            framelist=self.framelist,
            roi=self.roi,
            whichstar='target',
            hconf=this_hconf,
            pc_ecount_max=self.pc_ecount_max,
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )

        # 2
        this_hconf['excam']['scale_method'] = 'mean'
        this_hconf['excam']['scale_bright_method'] = 'percentile'
        this_hconf['excam']['scale_bright_percentile'] = 99

        tech_demo_obs(
            framelist=self.framelist,
            roi=self.roi,
            whichstar='target',
            hconf=this_hconf,
            pc_ecount_max=self.pc_ecount_max,
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )

        # 3
        this_hconf['excam']['scale_method'] = 'percentile'
        this_hconf['excam']['scale_percentile'] = 70
        this_hconf['excam']['scale_bright_method'] = 'mean'

        tech_demo_obs(
            framelist=self.framelist,
            roi=self.roi,
            whichstar='target',
            hconf=this_hconf,
            pc_ecount_max=self.pc_ecount_max,
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )


        # 4
        this_hconf['excam']['scale_method'] = 'percentile'
        this_hconf['excam']['scale_percentile'] = 70
        this_hconf['excam']['scale_bright_method'] = 'percentile'
        this_hconf['excam']['scale_bright_percentile'] = 99

        tech_demo_obs(
            framelist=self.framelist,
            roi=self.roi,
            whichstar='target',
            hconf=this_hconf,
            pc_ecount_max=self.pc_ecount_max,
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )


    @patch('eetc.cgi_eetc.CGIEETC.calc_pc_exp_time')
    def test_outputs_pc_yes(self, mock_calcpc):
        """Outputs are as expected in shape/type"""
        mock_calcpc.side_effect = [(1, 2, 3, 4, 5)]
        out = tech_demo_obs(
            framelist=self.framelist,
            roi=self.roi,
            whichstar=self.whichstar,
            hconf=self.hconf,
            pc_ecount_max=np.inf, # CHANGED to force success on precheck
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )

        self.assertTrue('exptime' in out)
        self.assertTrue('gain' in out)
        self.assertTrue('obstype' in out)
        self.assertTrue(out['obstype'] == 'PC')
        pass


    @patch('eetc.cgi_eetc.CGIEETC.calc_pc_exp_time')
    def test_outputs_pc_no_optfail(self, mock_calcpc):
        """Outputs are as expected in shape/type"""
        mock_calcpc.side_effect = EXCAMOptimizeException
        out = tech_demo_obs(
            framelist=self.framelist,
            roi=self.roi,
            whichstar=self.whichstar,
            hconf=self.hconf,
            pc_ecount_max=np.inf, # CHANGED to force success on precheck
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )

        self.assertTrue('exptime' in out)
        self.assertTrue('gain' in out)
        self.assertTrue('obstype' in out)
        self.assertTrue(out['obstype'] == 'analog')
        pass


    @patch('eetc.cgi_eetc.CGIEETC.calc_pc_exp_time')
    def test_outputs_pc_no_checkfail(self, mock_calcpc):
        """Outputs are as expected in shape/type"""
        mock_calcpc.side_effect = [(1, 2, 3, 4, 5)]
        out = tech_demo_obs(
            framelist=self.framelist,
            roi=self.roi,
            whichstar=self.whichstar,
            hconf=self.hconf,
            pc_ecount_max=1e-200, # CHANGED to force failure on precheck
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )

        self.assertTrue('exptime' in out)
        self.assertTrue('gain' in out)
        self.assertTrue('obstype' in out)
        self.assertTrue(out['obstype'] == 'analog')
        pass



    @patch('eetc.cgi_eetc.CGIEETC.calc_pc_exp_time')
    @patch('eetc.cgi_eetc.CGIEETC.calc_flux_rate')
    def test_exact_value_opt_inputs(self, mock_calcflux, mock_calcpc):
        """Performs as expected for expected inputs"""
        tol = 1e-13

        nlam = 3
        ndm = 7
        npix = 4
        framelist = []
        prev_exptime_list = []
        for j in range(nlam*ndm):
            # each wavelength will be 0, 1, 2, ...
            framelist.append(j//ndm*np.ones((npix, npix)))
            prev_exptime_list.append(10)
            pass

        mock_calcflux.return_value = (1e7, 1e7)
        # expect scale = 1e-8 after this: mean(0, 1, 2)/1e7/10
        scale_targ = 1e-8
        # expect scale_bright = 2e-8 after this: max(0, 1, 2)/1e7/10
        scale_bright_targ = 2e-8

        # maintain original test behavior (mean and max)
        this_hconf = self.hconf.copy()
        this_hconf['excam']['scale_method'] = 'mean'
        this_hconf['excam']['scale_bright_method'] = 'percentile'
        this_hconf['excam']['scale_bright_percentile'] = 100

        # we don't see these numbers directly; use a mock to hack the outputs
        # into the actual function outputs
        def side_effect(sequence_name, snr, scale, scale_bright):
            return (1, scale, scale_bright, 1, 1)
        mock_calcpc.side_effect = side_effect

        out = tech_demo_obs(
            framelist=framelist,
            roi=self.roi,
            whichstar=self.whichstar,
            hconf=this_hconf,
            pc_ecount_max=np.inf, # set to ignore
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )

        # exptime is overwritten with scale
        self.assertTrue(np.max(np.abs(scale_targ - out['exptime'])) < tol)
        # gain is overwritten with scale_bright
        self.assertTrue(np.max(np.abs(scale_bright_targ - out['gain'])) < tol)
        pass


    @patch('eetc.cgi_eetc.CGIEETC.calc_exp_time')
    @patch('eetc.cgi_eetc.CGIEETC.calc_pc_exp_time')
    def test_pc_done_right(self, mock_calcpc, mock_calcanalog):
        """Verify the photon-counting parameters are handled correctly"""
        # use a successful mock return so that success/failure are purely on
        # the precheck
        mock_calcpc.side_effect = [(1, 2, 3, 4, 5)]
        mock_calcanalog.side_effect = [(1, 2, 3, 4, 5)]

        nlam = 3
        ndm = 7
        npix = 4
        framelist = []
        for j in range(nlam*ndm):
            # each wavelength will be 0, 1, 2, ...
            framelist.append(j//ndm*np.ones((npix, npix)))
            pass

        pc_ecount_max = 1.01 # frames are 0,1,2; 2/3 of frames are at or better
        out_succeed = tech_demo_obs(
            framelist=framelist,
            roi=self.roi,
            whichstar=self.whichstar,
            hconf=self.hconf,
            pc_ecount_max=pc_ecount_max,
            pcfrac=0.5, # 50% can exceed -> PC is ok
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )
        self.assertTrue(out_succeed['obstype'] == 'PC')

        out_failure = tech_demo_obs(
            framelist=framelist,
            roi=self.roi,
            whichstar=self.whichstar,
            hconf=self.hconf,
            pc_ecount_max=pc_ecount_max,
            pcfrac=0.25, # 25% can exceed -> PC is not ok (real value ~33%)
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )
        self.assertTrue(out_failure['obstype'] == 'analog')

        pass



    @patch('eetc.cgi_eetc.CGIEETC.calc_pc_exp_time')
    @patch('eetc.cgi_eetc.CGIEETC.calc_flux_rate')
    def test_works_with_nans_bp(self, mock_calcflux, mock_calcpc):
        """Verify works as expected with bad pixels"""
        # ignore the all-NaN slice warning, we're putting in all NaNs in a
        # slice intentionally in this test.
        warnings.filterwarnings('ignore',
                                category=RuntimeWarning,
                                module='howfsc.observe',
        )
        tol = 1e-13

        nlam = 3
        ndm = 7
        npix = 4
        framelist = []
        prev_exptime_list = []
        for j in range(nlam*ndm):
            # each wavelength will be 0, 1, 2, ...
            framelist.append(j//ndm*np.ones((npix, npix)))
            prev_exptime_list.append(10)
            pass
        framelist[(nlam-1)*ndm] *= np.nan # knock off the twos

        mock_calcflux.return_value = (1e7, 1e7)
        # expect scale = 0.5e-8 after this: mean(0, 1, NaN)/1e7/10
        scale_targ = 0.5e-8
        # expect scale_bright = 1e-8 after this: max(0, 1, NaN)/1e7/10
        scale_bright_targ = 1e-8

        # maintain original test behavior (mean and max)
        this_hconf = self.hconf.copy()
        this_hconf['excam']['scale_method'] = 'mean'
        this_hconf['excam']['scale_bright_method'] = 'percentile'
        this_hconf['excam']['scale_bright_percentile'] = 100

        # we don't see these numbers directly; use a mock to hack the outputs
        # into the actual function outputs
        def side_effect(sequence_name, snr, scale, scale_bright):
            return (1, scale, scale_bright, 1, 1)
        mock_calcpc.side_effect = side_effect

        out = tech_demo_obs(
            framelist=framelist,
            roi=self.roi,
            whichstar=self.whichstar,
            hconf=this_hconf,
            pc_ecount_max=np.inf, # set to ignore
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )

        # exptime is overwritten with scale
        self.assertTrue(np.max(np.abs(scale_targ - out['exptime'])) < tol)
        # gain is overwritten with scale_bright
        self.assertTrue(np.max(np.abs(scale_bright_targ - out['gain'])) < tol)
        pass


    @patch('eetc.cgi_eetc.CGIEETC.calc_pc_exp_time')
    @patch('eetc.cgi_eetc.CGIEETC.calc_flux_rate')
    def test_works_with_roi(self, mock_calcflux, mock_calcpc):
        """Verify works as expected with non-True roi"""
        tol = 1e-13

        nlam = 3
        ndm = 7
        npix = 4
        framelist = []
        prev_exptime_list = []
        for j in range(nlam*ndm):
            # each wavelength will be 0, 1, 2, ...
            framelist.append(j//ndm*np.ones((npix, npix)))
            prev_exptime_list.append(10)
            pass

        mock_calcflux.return_value = (1e7, 1e7)
        # expect scale = 0.5e-8 after this: mean(0, 1, 2)/1e7/10
        scale_targ = 1e-8
        # expect scale_bright = 1e-8 after this: max(0, 1, 2)/1e7/10
        scale_bright_targ = 2e-8

        # maintain original test behavior (mean and max)
        this_hconf = self.hconf.copy()
        this_hconf['excam']['scale_method'] = 'mean'
        this_hconf['excam']['scale_bright_method'] = 'percentile'
        this_hconf['excam']['scale_bright_percentile'] = 100

        # we don't see these numbers directly; use a mock to hack the outputs
        # into the actual function outputs
        def side_effect(sequence_name, snr, scale, scale_bright):
            return (1, scale, scale_bright, 1, 1)
        mock_calcpc.side_effect = side_effect

        # NEW this test
        roi = self.roi.copy()
        roi[0, 0] = False
        out = tech_demo_obs(
            framelist=framelist,
            roi=roi,
            whichstar=self.whichstar,
            hconf=this_hconf,
            pc_ecount_max=np.inf, # set to ignore
            pcfrac=self.pcfrac,
            nlam=self.nlam,
            prev_exptime_list=self.prev_exptime_list,
        )

        # exptime is overwritten with scale
        self.assertTrue(np.max(np.abs(scale_targ - out['exptime'])) < tol)
        # gain is overwritten with scale_bright
        self.assertTrue(np.max(np.abs(scale_bright_targ - out['gain'])) < tol)
        pass



    def test_invalid_framelist(self):
        """Invalid inputs caught as expected"""
        N = self.ndm*self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [np.ones((self.npix+1, self.npix+1))]*N, # wrong in size
                    [np.ones((self.npix, self.npix))]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                tech_demo_obs(
                    framelist=perr,
                    roi=self.roi,
                    whichstar=self.whichstar,
                    hconf=self.hconf,
                    pc_ecount_max=self.pc_ecount_max,
                    pcfrac=self.pcfrac,
                    nlam=self.nlam,
                    prev_exptime_list=self.prev_exptime_list,
                )
            pass
        pass


    def test_invalid_roi(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, # wrong type
                    np.ones((self.npix, self.npix)), # not bool
                    np.ones((self.npix+1, self.npix), dtype='bool'), # bad size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                tech_demo_obs(
                    framelist=self.framelist,
                    roi=perr,
                    whichstar=self.whichstar,
                    hconf=self.hconf,
                    pc_ecount_max=self.pc_ecount_max,
                    pcfrac=self.pcfrac,
                    nlam=self.nlam,
                    prev_exptime_list=self.prev_exptime_list,
                )
            pass
        pass


    def test_invalid_whichstar(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, None, np.ones((4, 4)), # wrong type
                    'some_other_string', # wrong string content
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                tech_demo_obs(
                    framelist=self.framelist,
                    roi=self.roi,
                    whichstar=perr,
                    hconf=self.hconf,
                    pc_ecount_max=self.pc_ecount_max,
                    pcfrac=self.pcfrac,
                    nlam=self.nlam,
                    prev_exptime_list=self.prev_exptime_list,
                )
            pass
        pass


    def test_invalid_hconf(self):
        """Invalid inputs caught as expected"""
        hconf = self.hconf

        hc2 = copy.deepcopy(hconf)
        hc2.popitem() # pop first level

        hc3a = copy.deepcopy(hconf)
        hc3a['overhead'].popitem() # pop second level
        hc3b = copy.deepcopy(hconf)
        hc3b['star'].popitem() # pop second level
        hc3c = copy.deepcopy(hconf)
        hc3c['excam'].popitem() # pop second level
        hc3d = copy.deepcopy(hconf)
        hc3d['hardware'].popitem() # pop second level
        hc3e = copy.deepcopy(hconf)
        hc3e['howfsc'].popitem() # pop second level
        hc3f = copy.deepcopy(hconf)
        hc3f['probe'].popitem() # pop second level

        hc4 = copy.deepcopy(hconf)
        hc4.update({'not_a_key':0})

        hc5a = copy.deepcopy(hconf)
        hc5a['overhead'].update({'not_a_key':0})
        hc5b = copy.deepcopy(hconf)
        hc5b['star'].update({'not_a_key':0})
        hc5c = copy.deepcopy(hconf)
        hc5c['excam'].update({'not_a_key':0})
        hc5d = copy.deepcopy(hconf)
        hc5d['hardware'].update({'not_a_key':0})
        hc5e = copy.deepcopy(hconf)
        hc5e['howfsc'].update({'not_a_key':0})
        hc5f = copy.deepcopy(hconf)
        hc5f['probe'].update({'not_a_key':0})

        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, # wrong type
                    hc2, hc3a, hc3b, hc3c, hc3d, hc3e, hc3f, # missing keys
                    hc4, hc5a, hc5b, hc5c, hc5d, hc5e, hc5f, # extra keys
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                tech_demo_obs(
                    framelist=self.framelist,
                    roi=self.roi,
                    whichstar=self.whichstar,
                    hconf=perr,
                    pc_ecount_max=self.pc_ecount_max,
                    pcfrac=self.pcfrac,
                    nlam=self.nlam,
                    prev_exptime_list=self.prev_exptime_list,
                )
            pass
        pass


    def test_invalid_pc_ecount_max(self):
        """Invalid inputs caught as expected"""
        perrlist = [-1, 0, 1j, None, np.ones((4, 4)), 'txt']

        for perr in perrlist:
            with self.assertRaises(TypeError):
                tech_demo_obs(
                    framelist=self.framelist,
                    roi=self.roi,
                    whichstar=self.whichstar,
                    hconf=self.hconf,
                    pc_ecount_max=perr,
                    pcfrac=self.pcfrac,
                    nlam=self.nlam,
                    prev_exptime_list=self.prev_exptime_list,
                )
            pass
        pass


    def test_invalid_pcfrac(self):
        """Invalid inputs caught as expected"""
        perrlist = [1j, None, np.ones((4, 4)), 'txt', # wrong type
                    -0.01, 1.01, # out of range

        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                tech_demo_obs(
                    framelist=self.framelist,
                    roi=self.roi,
                    whichstar=self.whichstar,
                    hconf=self.hconf,
                    pc_ecount_max=self.pc_ecount_max,
                    pcfrac=perr,
                    nlam=self.nlam,
                    prev_exptime_list=self.prev_exptime_list,
                )
            pass
        pass


    def test_invalid_nlam(self):
        """Invalid inputs caught as expected"""
        perrlist = [-1, 0, 1.5, 1j, None, np.ones((4, 4)), 'txt']

        for perr in perrlist:
            with self.assertRaises(TypeError):
                tech_demo_obs(
                    framelist=self.framelist,
                    roi=self.roi,
                    whichstar=self.whichstar,
                    hconf=self.hconf,
                    pc_ecount_max=self.pc_ecount_max,
                    pcfrac=self.pcfrac,
                    nlam=perr,
                    prev_exptime_list=self.prev_exptime_list,
                )
            pass
        pass


    def test_invalid_prev_exptime_list(self):
        """Invalid inputs caught as expected"""
        N = self.ndm*self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, # wrong type
                    [np.ones((self.npix,))]*N, # wrong inner type
                    [1]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                tech_demo_obs(
                    framelist=self.framelist,
                    roi=self.roi,
                    whichstar=self.whichstar,
                    hconf=self.hconf,
                    pc_ecount_max=self.pc_ecount_max,
                    pcfrac=self.pcfrac,
                    nlam=self.nlam,
                    prev_exptime_list=perr,
                )
            pass
        pass


    def test_invalid_hconf_excam(self):
        """Invalid inputs caught as expected"""
        this_hconf = self.hconf.copy()

        this_hconf['excam']['scale_method'] = 'foo'
        this_hconf['excam']['scale_bright_method'] = 'mean' # valid

        with self.assertRaises(TypeError):
            tech_demo_obs(
                framelist=self.framelist,
                roi=self.roi,
                whichstar='target',
                hconf=this_hconf,
                pc_ecount_max=self.pc_ecount_max,
                pcfrac=self.pcfrac,
                nlam=self.nlam,
                prev_exptime_list=self.prev_exptime_list,
            )

        this_hconf['excam']['scale_method'] = 'mean' # valid
        this_hconf['excam']['scale_bright_method'] = 'foo'

        with self.assertRaises(TypeError):
            tech_demo_obs(
                framelist=self.framelist,
                roi=self.roi,
                whichstar='target',
                hconf=this_hconf,
                pc_ecount_max=self.pc_ecount_max,
                pcfrac=self.pcfrac,
                nlam=self.nlam,
                prev_exptime_list=self.prev_exptime_list,
            )



if __name__ == '__main__':
    unittest.main()
