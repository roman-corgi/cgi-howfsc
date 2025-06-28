# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Script to write a YAML file
"""

import os

import yaml

import howfsc

# DM1
dm1_voltages = {
                'gainfn':'testdata/ut_DM1_gainmap.fits',
                'vmax':100,
                'vmin':0,
                'vneighbor':50.,
                'vcorner':75.,
                'vquant':110.0/2**16,
                'tiefn':'testdata/ut_DM1_tiemap.fits',
                'flatfn':'testdata/ut_DM1_flatmap.fits'}
dm1_registration = {
                    'dx':0,
                    'dy':0,
                    'inffn':'testdata/ut_influence_dm5v2_inffix.fits',
                    'nact':48,
                    'ppact_cx':8.3,
                    'ppact_cy':8.3,
                    'ppact_d':10,
                    'thact':0,
                    'flipx':False
                   }
dm1z = 0.0 # meters
dm1pitch = 1.0e-3 # meters per actuator
dm1 = {'pitch':dm1pitch,
       'registration':dm1_registration,
       'voltages':dm1_voltages,
       'z':dm1z,
       }

# DM2
dm2_voltages = {
                'gainfn':'testdata/ut_DM2_gainmap.fits',
                'vmax':100,
                'vmin':0,
                'vneighbor':50.,
                'vcorner':75.,
                'vquant':110.0/2**16,
                'tiefn':'testdata/ut_DM2_tiemap.fits',
                'flatfn':'testdata/ut_DM2_flatmap.fits'}
dm2_registration = {
                    'dx':0,
                    'dy':0,
                    'inffn':'testdata/ut_influence_dm5v2_inffix.fits',
                    'nact':48,
                    'ppact_cx':8.3,
                    'ppact_cy':8.3,
                    'ppact_d':10,
                    'thact':0,
                    'flipx':False
                    }

dm2z = 1.0 # meters
dm2pitch = 1.0e-3 # meters per actuator
dm2 = {'pitch':dm2pitch,
       'registration':dm2_registration,
       'voltages':dm2_voltages,
       'z':dm2z,
       }

dms = dict()
dms.update({'DM1':dm1,
            'DM2':dm2})

# Starting DMs
init = dict()
init.update({'DM1':{'dminit':'testdata/widefov/ut_DM1_init.fits'}})
init.update({'DM2':{'dminit':'testdata/widefov/ut_DM2_init.fits'}})

# Masks
pupil_diam_pix = 300
crit_sample_lam = 430e-9 # 2 pix/(lambda/D) here
# wavelength in m per "channel", which is a single hardware config for WF
# control data collection
channels = [797.5e-9,
            825.0e-9,
            852.5e-9]
sls = dict()

for index, channel in enumerate(channels):
    lam = channel
    pixperlod = 2.0*lam/float(crit_sample_lam)

    ch = dict()

    # dark hole area for wavefront control
    dh = dict()
    dh.update({'fn':'testdata/widefov/ut_dh_ch'+str(index)+'.fits',
               'isopen':False,
               'ppl':pixperlod})
    ch.update({'dh':dh})

    # starting field
    epup = dict()
    epup.update({'afn':'testdata/widefov/ut_epup_amp_ch'+str(index)+'.fits',
                 'isopen':False,
                 'pdp':pupil_diam_pix,
                 'pfn':'testdata/widefov/ut_epup_ph_ch'+str(index)+'.fits'})
    ch.update({'epup':epup})

    # focal plane mask (FPAM)
    fpm = dict()
    fpm.update({'afn':'testdata/widefov/ut_fpm_amp_ch'+str(index)+'.fits',
                'isopen':False,
                'pfn':'testdata/widefov/ut_fpm_ph_ch'+str(index)+'.fits',
                'ppl':pixperlod})
    ch.update({'fpm':fpm})

    # field stop (FSAM)
    fs = dict()
    fs.update({'afn':'testdata/widefov/ut_fs_amp_ch'+str(index)+'.fits',
               'isopen':False,
               'pfn':'testdata/widefov/ut_fs_ph_ch'+str(index)+'.fits',
               'ppl':pixperlod})
    ch.update({'fs':fs})

    # wavelength
    ch.update({'lam':lam})

    # Lyot stop (LSAM)
    lyot = dict()
    lyot.update({'afn':'testdata/widefov/ut_lyot_amp.fits',
                 'isopen':False,
                 'pdp':pupil_diam_pix,
                 'pfn':'testdata/widefov/ut_lyot_ph_ch'+str(index)+'.fits'})
    ch.update({'lyot':lyot})

    # shaped pupil (SPAM)
    sp = dict()
    sp.update({'afn':'testdata/widefov/ut_sp_amp.fits',
               'isopen':False,
               'pdp':pupil_diam_pix,
               'pfn':'testdata/widefov/ut_sp_ph_ch'+str(index)+'.fits'})
    ch.update({'sp':sp})

    # bulk tip/tilt
    ch.update({'tilt':0, 'tip':0})

    sls.update({index:ch})
    pass

howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))
yamlfn = os.path.join(howfscpath, 'model', 'testdata', 'widefov',
                      'widefov.yaml')
with open(yamlfn, 'w') as outfile:
    yaml.dump({'dms':dms, 'init':init, 'sls':sls},
              outfile,
              default_flow_style=False, # enforces block style
              sort_keys=False, # will dump in dict insertion order on Py3.7+
              )
