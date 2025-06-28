# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Class to handle conversions from voltage to actuator height.  Does not include
subsequent step to influence functions.
"""
import numpy as np

import howfsc.util.check as check

from howfsc.model.dm_crosstalk import CDMCrosstalk
from howfsc.util.constrain_dm import constrain_dm
from howfsc.util.flat_tie import checkflat, checktie
from howfsc.util.dmhtoph import dmh_to_volts, volts_to_dmh

class DM:
    """
    A basic object that encapsulates DM manipulation at the voltage level
    """
    def __init__(self, vmin, vmax, vneighbor, vcorner, vquant,
                 gainmap, flatmap, tiemap, crosstalkfn=None):
        # input type checks
        # vmin/vmax are high and low permitted voltages
        check.real_scalar(vmin, 'vmin', TypeError)
        check.real_scalar(vmax, 'vmax', TypeError)
        if vmax <= vmin:
            raise ValueError('vmin must be less than vmax')

        # Neighbor is permitted voltage between rectilinear actuators
        # Corner is permitted voltage between diagonal actuators
        check.real_positive_scalar(vneighbor, 'vneighbor', TypeError)
        check.real_positive_scalar(vcorner, 'vcorner', TypeError)

        # DAC LSB in volts
        check.real_positive_scalar(vquant, 'vquant', TypeError)

        # m/V per actuator (so for nm will be ~1e-9-ish)
        check.twoD_array(gainmap, 'gainmap', TypeError)

        # tiemap = integer array defining tied and dead actuators
        check.twoD_array(tiemap, 'tiemap', TypeError)
        if not checktie(tiemap):
            raise TypeError('tiemap not formatted correctly as a tiemap')

        # flatmap = voltage settings to make phase flat (zero-strain)
        check.twoD_array(flatmap, 'flatmap', TypeError)
        if not checkflat(flatmap, vmin, vmax, tiemap):
            raise TypeError('flatmap not formatted correctly as a flatmap')

        # crosstalkfn: yaml dm crosstalk file, or None
        # it is checked in CDMCrosstalk()

        # Direct inputs
        self.vmin = vmin
        self.vmax = vmax
        self.vneighbor = vneighbor
        self.vcorner = vcorner
        self.vquant = vquant
        self.gainmap = gainmap
        self.flatmap = flatmap
        self.tiemap = tiemap
        self.crosstalk = CDMCrosstalk(yaml_fn=crosstalkfn)


    def volts_to_dmh(self, volts, lam):
        """
        Convert voltages to poke heights in radians with a linear gainmap

        Gainmap is in ``self.gainmap``, and is in meters of poke height
        (surface) per volt.  (Most pokes will be nanometer scale, with large
        and small excursions to micron and picometer scales.)

        Arguments:
         volts: a 2D array of voltages, of the same size as ``self.gainmap``.
          This function by default does no input checking on the range of
          validity of voltages
         lam: wavelength of light to use for radian conversion, in meters

        Returns:
         2D array of poke heights in radians, of the same array size as volts

        """

        # Let dmhtoph handle the input validation
        dmh = volts_to_dmh(self.gainmap, volts, lam)

        # apply crosstalk and return
        return self.crosstalk.crosstalk_forward(dmh)

    def dmh_to_volts(self, dmh, lam):
        """
        Convert poke heights in radians to voltages with a linear gainmap

        ``gainmap`` is in meters of poke height (surface) per volt.  (Most
        pokes will be nanometer scale, with large and small excursions to
        micron and picometer scales.)

        Arguments:
         dmh: a 2D array of 'actual' poke heights, i.e. after applying
          crosstalk to the commanded poke heights.  This function by default
          does no input checking on the range of validity of heights, or of
          the output voltage array.  Units of radians.
         lam: wavelength of light to use for radian conversion, in meters

        Returns:
         2D array of voltages, of the same array size as dmh, equal to
          volts_command after accounting for crosstalk, such that
          volts_to_dmh(dmh_to_volts(dmh)) returns the same dmh is input here

        """

        # undo crosstalk
        dmh = self.crosstalk.crosstalk_backward(dmh)

        # Let dmhtoph handle the input validation, accounts for crosstalk
        volts = dmh_to_volts(self.gainmap, dmh, lam)

        return volts

    def volts_to_dmphys(self, volts):
        """
        Convert voltages to poke heights in meters with a linear gainmap

        Unlike radian conversion, we don't need a wavelength for physical units

        Gainmap is in ``self.gainmap``, and is in meters of poke height
        (surface) per volt.  (Most pokes will be nanometer scale, with large
        and small excursions to micron and picometer scales.)

        Arguments:
         volts: a 2D array of voltages, of the same size as ``self.gainmap``.
          This function by default does no input checking on the range of
          validity of voltages

        Returns:
         2D array of poke heights in meters, of the same array size as volts

        """

        check.twoD_array(volts, 'volts', TypeError)

        if volts.shape != self.gainmap.shape:
            raise TypeError('Array of voltages must be the same size ' + \
                              'as the gainmap')

        return self.crosstalk.crosstalk_forward(volts*self.gainmap)


    def dmphys_to_volts(self, dmphys):
        """
        Convert poke heights in meters to voltages with a linear gainmap

        Unlike radian conversion, we don't need a wavelength for physical units

        ``gainmap`` is in meters of poke height (surface) per volt.  (Most
        pokes will be nanometer scale, with large and small excursions to
        micron and picometer scales.)

        Arguments:
         dmphys: a 2D array of poke heights.  This function by default does no
          input checking on the range of validity of heights, or of the output
          voltage array.

        Returns:
         2D array of voltages, of the same array size as dmh

        """

        check.twoD_array(dmphys, 'dmphys', TypeError)

        if dmphys.shape != self.gainmap.shape:
            raise TypeError('Array of poke heights must be the same ' + \
                              'size as the gainmap')

        # remove crosstalk effect, then gain
        return self.crosstalk.crosstalk_backward(dmphys)/self.gainmap


    def constrain_dm(self, volts):
        """
        Given a DM setting, return one consistent with physical constraints

        Wrapper to howfsc.util.dmsmooth.dmsmooth(), which handles upper and
        lower constraints and neighbor rules.  May be expanded in the future to
        coupled and dead actuators (which in principle should not exist if DM
        meets requirements, but in practice may.)

        Arguments:
         volts: a 2D array of voltages.  This array must be a real-values
          object or, if complex, have no imaginary part

        Returns:
         a constrained 2D array of voltages

        """
        check.twoD_array(volts, 'volts', TypeError)

        if np.iscomplexobj(volts):
            if (volts.imag == np.zeros_like(volts.real)).all():
                volts = volts.real
                pass
            else:
                raise TypeError('volts must not have imaginary components')
            pass

        return constrain_dm(volts=volts,
                            flatmap=self.flatmap,
                            tie=self.tiemap,
                            vmax=self.vmax,
                            vlat=self.vneighbor,
                            vdiag=self.vcorner,
                            vquant=self.vquant)
