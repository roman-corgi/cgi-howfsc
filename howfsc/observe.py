# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Compute EXCAM settings for target and reference stars during the subsequent
observation.
"""

import logging
import warnings
import os

import numpy as np

import eetc
from eetc.cgi_eetc import CGIEETC
from eetc.excam_tools import EXCAMOptimizeException

from howfsc.control.nextiter import valid_method_list

from howfsc.util.gitl_tools import validate_dict_keys, as_f32_normal
from howfsc.gitl import (toplevel_keys, overhead_keys, star_keys,
                         excam_keys, hardware_keys, howfsc_keys, probe_keys)
import howfsc.util.check as check

log = logging.getLogger(__name__)

# Otherwise, you will get complaints if you have an all-NaN slice, which can
# happen if an unprobed frame gets lost
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='howfsc.observe')

eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))

def tech_demo_obs(framelist, roi, whichstar, hconf, pc_ecount_max, pcfrac,
                  nlam, prev_exptime_list):
    """
    Prepare camera settings for tech demo observations

    Satisfies requirement 1115539:
    ###
    Given:
    - a complete set of HOWFSC GITL frames for a CGI coronagraph mask
      configuration,
    - a set of pixels defining a region of interest,
    - a magnitude and stellar type for the star which provided the HOWFSC GITL
      frames
    - a magnitude and stellar type for the star to be observed
    - a photoelectron flux rate threshold for photon-counting, and an allowable
      percentage of pixels exceeding that rate

    the CTC GSW shall compute an EXCAM gain and exposure time, and a flag
    indicating whether the observation should be processed as analog or
    photon-counting.
    ###

    The actual function will require more inputs than this list, because it
    needs to recreate some of the GITL processing pipeline over the region of
    interest.

    Arguments:
     framelist: nlam*ndm list of HOWFSC GITL frames.  These are nrow x ncol
      intensity images as floating-point arrays, which have already been
      corrected for bias, dark and flat, and converted into units of
      photoelectrons.

      The order should match the order in which they are collected, which is
      to cycle the wavelengths as the outer loop, and the DM settings as the
      inner loop, starting with the unprobed image.  Example ordering:

      [lam0 unprobed, lam0 probe0+, lam0 probe0-, lam0 probe1+, lam0 probe1-,
       lam0, probe2+, lam0 probe2-, lam1 unprobed, lam1 probe0+, ...]

      framelist data will be sourced from HOWFSC GITL packets.  Note:
      unlike HOWFSC GITL iterations, only the unprobed data will be used here.

     roi: 2D array of booleans indicating which pixels to keep (True) and
      ignore (False) for the purposes of setting observation parameters.  The
      region of interest (ROI) might be the dark hole, or might be smaller if
      the observation should be targeting a particular region.  There is
      nothing preventing it from being larger, but pixels that were not nulled
      will tend to be brighter and raise the mean level in the ROI.  This array
      is the same size as all frames in framelist.

     whichstar: a string, either 'target' or 'reference'.  'reference' star
      will pull the star information from the same hconf elements used by
      HOWFSC, while 'target' will pull from a separate set of magnitude and
      stellar type data in hconf labeled with '_target'.

     hconf: dictionary of dictionaries.  Keys at the top level must match the
      set in toplevel_keys in gitl.py, while each subdictionary must have its
      own keys match the relevant list.  These contain scalar configuration
      parameters which do not change iteration-to-iteration.

     pc_ecount_max: scalar floating-point value > 0. Maximum photo-electron
      flux allowed in a pixel for photon counting.  Units of
      electrons/pixel/frame.

     pcfrac: fraction of pixels in roi which are allowed to exceed the
      flux-rate in pc_ecount_max.  Scalar floating point value >= 0 and <= 1.

     nlam: number of wavelengths (CFAM filters) used with the data set.
      Integer > 0.  Used to extract unprobed frames from the complete data set.

     prev_exptime_list: list of exposure times for each of the frames in
      framelist.  framelist data is averaged on board, so this will be the
      EXCAM exposure for a single frame during the data collection period that
      fed that frame.  This should have ndm*nlam elements.  prev_exptime_list
      data will be sourced from HOWFSC packets for ancillary GITL info.

    Returns:
     dictionary with three keys:
      - 'exptime' will be EXCAM exposure time per frame
      - 'gain' will be EXCAM gain
      - 'obstype' will be either 'PC' (if the observation data should be
        collected and processed as photon-counted data) or 'analog' (if the
        data should be collected and processed as analog data)

    """
    # Check inputs

    # nlam (do first, need for framelist)
    check.positive_scalar_integer(nlam, 'nlam', TypeError)

    # framelist
    try:
        lenflist = len(framelist)
    except TypeError: # not iterable
        raise TypeError('framelist must be an iterable') # reraise
    for index, gitlframe in enumerate(framelist):
        check.twoD_array(gitlframe, 'framelist[' + str(index) + ']', TypeError)
        pass
    nrow, ncol = framelist[0].shape
    for gitlframe in framelist:
        if gitlframe.shape != (nrow, ncol):
            raise TypeError('All frames in framelist must be the same size')
        pass

    ndm = lenflist // nlam
    if lenflist % ndm != 0: # did not divide evenly
        raise TypeError('Number of received frames not consistent with ' +
                        'number of model wavelengths and assumption of ' +
                        'identical probing per wavelength')
    if ndm % 2 != 1: # expect an odd number of DMs: N pairs + 1 unprobed
        raise TypeError('Expected odd number of DM1 settings, got even')

    # roi
    check.twoD_array(roi, 'roi', TypeError)
    if roi.dtype != 'bool':
        raise TypeError('roi must be an array of bools')
    if roi.shape != (nrow, ncol):
        raise TypeError('roi must be the same size as arrays in framelist')

    # whichstar
    if not isinstance(whichstar, str):
        raise TypeError('whichstar must be a string')
    if whichstar != 'reference' and whichstar != 'target':
        raise TypeError("whichstar must be 'target' or 'reference'")

    # hconf
    validate_dict_keys(hconf, toplevel_keys, custom_exception=TypeError)
    validate_dict_keys(hconf['overhead'], overhead_keys,
                       custom_exception=TypeError)
    validate_dict_keys(hconf['star'], star_keys,
                       custom_exception=TypeError)
    validate_dict_keys(hconf['excam'], excam_keys,
                       custom_exception=TypeError)
    validate_dict_keys(hconf['hardware'], hardware_keys,
                       custom_exception=TypeError)
    validate_dict_keys(hconf['howfsc'], howfsc_keys,
                       custom_exception=TypeError)
    validate_dict_keys(hconf['probe'], probe_keys,
                       custom_exception=TypeError)
    if hconf['excam']['scale_method'] not in valid_method_list:
        raise TypeError('scale_method in hconf not valid value')
    if hconf['excam']['scale_bright_method'] not in valid_method_list:
        raise TypeError('scale_bright_method in hconf not valid value')

    # pc_ecount_max
    check.real_positive_scalar(pc_ecount_max, 'pc_ecount_max', TypeError)

    # pcfrac
    check.real_scalar(pcfrac, 'pcfrac', TypeError)
    if pcfrac < 0 or pcfrac > 1:
        raise TypeError('pcfrac must be a scalar between 0 and 1 inclusive.')

    # prev_exptime_list
    try:
        lenpelist = len(prev_exptime_list)
    except TypeError: # not iterable
        raise TypeError('prev_exptime_list must be an iterable') # reraise
    for index, prev in enumerate(prev_exptime_list):
        check.real_positive_scalar(prev, 'prev_exptime_list[' + str(index) +
                                   ']', TypeError)
        pass
    if lenpelist != lenflist:
        raise TypeError('prev_exptime_list and framelist must contain the ' +
                        'same number of elements')


    #------------------
    # Begin processing
    #------------------

    # Build two exposure time objects, one for HOWFSC data and one for
    # observation (may or may not be the same star)
    log.info('Building exposure-time calculator classes')
    howfsc_eetc = CGIEETC(mag=hconf['star']['stellar_vmag'],
                       phot='v', # only using V-band magnitudes as a standard
                       spt=hconf['star']['stellar_type'],
                       pointer_path=os.path.join(eetc_path,
                                                 hconf['hardware']['pointer']),
    )
    log.info('HOWFSC evaluated with Vmag = %g, type = %s',
             hconf['star']['stellar_vmag'],
             hconf['star']['stellar_type'],
    )
    log.info('Observation star: %s', whichstar)
    if whichstar == 'reference':
        obs_eetc = CGIEETC(mag=hconf['star']['stellar_vmag'],
                           phot='v',
                           spt=hconf['star']['stellar_type'],
                           pointer_path=os.path.join(eetc_path,
                                                 hconf['hardware']['pointer']),
        )

        log.info('Observation evaluated with Vmag = %g, type = %s',
                 hconf['star']['stellar_vmag'],
                 hconf['star']['stellar_type'],
        )
        pass
    elif whichstar == 'target':
        obs_eetc = CGIEETC(mag=hconf['star']['stellar_vmag_target'],
                           phot='v',
                           spt=hconf['star']['stellar_type_target'],
                           pointer_path=os.path.join(eetc_path,
                                                 hconf['hardware']['pointer']),
        )
        log.info('Observation evaluated with Vmag = %g, type = %s',
                 hconf['star']['stellar_vmag_target'],
                 hconf['star']['stellar_type_target'],
        )
        pass
    else:
        # should never be able to reach here
        raise ValueError('whichstar has an unknown value past the input check')


    # Get measured flux rate from last measured data set
    # This is not the same as GITL!  GITL is producing normalized intensity
    #  (unitless), we want electrons/frame (per pixel, which is implicit)

    unprobedlist = [] # unprobed NI for contrast estimation
    brightlist = [] # list of non-CR brightest pixels to avoid saturation
    for j in range(nlam):
        log.info('Wavelength %d of %d', j+1, nlam)
        log.info('Use unprobed frame only')
        ind = j*ndm + 0 # unprobed is first in 0 to ndm-1 by convention

        log.info('Level 2b data in photo-e/pixel/frame')
        flux = framelist[ind].copy()
        brightlist.append(np.nanmax(flux))

        # nan the pixels we aren't using anyway.  Bad pixels are pre-NaN'ed, so
        # just do region of interest
        flux[~roi] = np.nan
        unprobedlist.append(flux)

        pass

    # use pc_ecount_max and pcfrac to precheck against photon-counting
    # conditions
    all_pixels = np.array(unprobedlist).ravel()
    pcfrac_level = np.nanpercentile(all_pixels, 100*(1 - pcfrac))
    isokforpc = (pcfrac_level <= pc_ecount_max)

    # Now that we're done using flux, convert to NI aka scale
    for j in range(nlam):
        _, peakflux = howfsc_eetc.calc_flux_rate(
            sequence_name=hconf['hardware']['sequence_list'][j],
        )
        ind = j*ndm + 0 # unprobed is first in 0 to ndm-1 by convention

        log.info('Divide by exposure time (sec/frame) to get photo-e/s')
        unprobedlist[j] /= prev_exptime_list[ind]
        brightlist[j] /= prev_exptime_list[ind]

        log.info('Divide by e-/s from star to get NI')
        unprobedlist[j] /= peakflux
        brightlist[j] /= peakflux
        pass

    if hconf['excam']['scale_method'] == 'mean':
        scale = np.nanmean(np.array(unprobedlist).ravel())
    elif hconf['excam']['scale_method'] == 'percentile':
        scale = np.nanpercentile(np.array(unprobedlist).ravel(),
                                 hconf['excam']['scale_percentile'])
    else: # should never reach here
        return Exception('reached end with an invalid method despite checks')

    if hconf['excam']['scale_bright_method'] == 'mean':
        scale_bright = np.nanmean(np.array(brightlist).ravel())
    elif hconf['excam']['scale_bright_method'] == 'percentile':
        scale_bright = np.nanpercentile(np.array(brightlist).ravel(),
                                 hconf['excam']['scale_bright_percentile'])
    else: # should never reach here
        return Exception('reached end with an invalid method despite checks')

    # Get new camera settings
    log.info('Observation camera settings from calculator')
    log.info('scale = %g, scale_bright = %g', scale, scale_bright)

    # overwrite defaults to enforce the single-frame case.  We want the gain
    # and exposure time that produce the best results in a single frame; we
    # don't want to require/account for the total number of frames, as that
    # will be partly determined by how the HOWFSC iterations went.  Since we're
    # also manually supplying a max PC count rate, make sure we use that too.
    obs_eetc.excam_config['Nmin'] = 1
    obs_eetc.excam_config['Nmax'] = 1
    obs_eetc.excam_config['pc_ecount_max'] = pc_ecount_max

    # try photon-counting first, go with analog if infeasible.  Prefer PC if
    # possible for data quality.  SNR=None implies we get the best SNR possible
    # per frame when combined with the Nmin/Nmax constraints above.
    if isokforpc:
        try:
            _, exptime, gain, _, _ = \
              obs_eetc.calc_pc_exp_time(
                  sequence_name=hconf['hardware']['sequence_observation'],
                  snr=None,
                  scale=scale,
                  scale_bright=scale_bright,
              )
            return {'exptime': as_f32_normal(exptime),
                    'gain': gain, 'obstype': 'PC'}
        except EXCAMOptimizeException: # not feasible
            pass

    # Do analog anyway if pc_ecount_max check failed, or if calc somehow failed
    _, exptime, gain, _, _ = \
      obs_eetc.calc_exp_time(
          sequence_name=hconf['hardware']['sequence_observation'],
          snr=None,
          scale=scale,
          scale_bright=scale_bright,
      )
    return {'exptime': as_f32_normal(exptime),
            'gain': gain, 'obstype': 'analog'}
