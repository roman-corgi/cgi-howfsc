# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
A set of classes to represent optical elements in a coronagraph model
"""

import numpy as np

from howfsc.util.insertinto import insertinto
from howfsc.util.subarray import centered_nonzero
import howfsc.util.check as check
from howfsc.model.dmobj import DM

class ModelElement(object):
    """
    A generic class for a coronagraph element in a plane (e.g. a mask)

    Assumed centered about center of pupil with sampling matched to camera.

    Arguments:
     lam: real positive scalar wavelength of representation in meters
     e: two-D array of mask representation

    """
    def __init__(self, lam, e):
        check.real_positive_scalar(lam, 'lam', TypeError)
        check.twoD_array(e, 'e', TypeError)
        self.lam = lam
        self.e = e
        pass


    def applymask(self, e0):
        """
        Apply the mask at a given plane to an input field

        Arguments:
         e0: 2D array, must be greater than or equal to ``self.e`` (unpadded)
          so the mask is not cut off

        Returns:
         matrix of the same size as e0

        """
        check.twoD_array(e0, 'e0', TypeError)
        if (e0.shape[0] < self.e.shape[0]) or (e0.shape[1] < self.e.shape[1]):
            raise TypeError('e0 must be larger or equal to than the ' + \
                                'mask in both dimensions')
        if self.e.shape == e0.shape: # save a call
            return self.e*e0
        else:
            return insertinto(self.e, e0.shape)*e0

    pass




class PupilMask(ModelElement):
    """
    An object representing a pupil-plane optical mask

    Assumed centered about center of pupil with sampling matched to camera.

    Arguments:
     lam: real positive scalar wavelength of representation in meters
     e: two-D array of mask representation
     pixperpupil: real positive scalar number of pixels across the defining
      pupil in this plane. Used to scale Fourier transforms for pupil ->
      focal plane propagation and back.

    """
    def __init__(self, lam, e, pixperpupil):
        super(PupilMask, self).__init__(lam, e)
        self.plane = 'pupil'
        check.real_positive_scalar(pixperpupil, 'pixperpupil', TypeError)
        self.pixperpupil = pixperpupil
        pass

    pass


class FocalPlaneMask(ModelElement):
    """
    An object representing a focal-plane optical mask upstream of the Lyot stop

    Assumed centered about center of PSF with sampling matched to camera.

    Arguments:
     lam: real positive scalar wavelength of representation in meters
     e: two-D array of mask representation
     isopen: boolean on whether the mask is completely open (``True``) or
      completely closed (``False``) outside the extent of the mask array.
      Other cases, like an infinite half-plane, are not currently supported; we
      do not expect any infinite half-planes in Roman CGI.
     pixperlod: pixels per lambda/D in this plane at wavelength ``lam``.

    """
    def __init__(self, lam, e, isopen, pixperlod):
        super(FocalPlaneMask, self).__init__(lam, e)
        self.plane = 'focal'
        check.real_positive_scalar(pixperlod, 'pixperlod', TypeError)
        self.pixperlod = pixperlod
        self.isopen = isopen # no check since anything can be used as a bool

        # Precompute MFT subregion on only centered, nonzero pixels
        if isopen:
            # include babinet term
            self.e = centered_nonzero(1-self.e)
        else:
            self.e = centered_nonzero(self.e)
        pass

    pass


class FieldStop(ModelElement):
    """
    An object representing a field stop

    Assumed centered about center of PSF with sampling matched to camera.

    Arguments:
     lam: real positive scalar wavelength of representation in meters
     e: two-D array of mask representation
     pixperlod: pixels per lambda/D in this plane at wavelength ``lam``.

    """
    def __init__(self, lam, e, pixperlod):
        super(FieldStop, self).__init__(lam, e)
        self.plane = 'focal'
        check.real_positive_scalar(pixperlod, 'pixperlod', TypeError)
        self.pixperlod = pixperlod

    pass


class LyotStop(PupilMask):
    """
    An object representing a Lyot stop in a Lyot-type coronagraph

    Called out as its own object as that's where we put any tip/tilt as seen
    on the camera.  Tip/tilt are post-FPM.

    Arguments:
     lam: real positive scalar wavelength of representation in meters
     e: two-D array of mask representation
     pixperpupil: real positive scalar number of pixels across the defining
      pupil in this plane. Used to scale Fourier transforms for pupil ->
      focal plane propagation and back.
     tip: wavefront tip offset, in EXCAM pixels.  Used to accommodate a star
      which is not exactly centered with respect to the grid.  Assumed a real
      scalar.  tip = +1 moves one pixel along the positive x-axis (columms).
     tilt: wavefront tilt offset, in EXCAM pixels.  Used to accommodate a star
      which is not exactly centered with respect to the grid.  Assumed a real
      scalar.  tilt = +1 moves one pixel along the positive y-axis (rows).

    """
    def __init__(self, lam, e, pixperpupil, tip, tilt):
        super(LyotStop, self).__init__(lam, e, pixperpupil)
        check.real_scalar(tip, 'tip', TypeError)
        check.real_scalar(tilt, 'tilt', TypeError)

        self.tip = tip
        self.tilt = tilt

        # Make grids in pixel coordinates so pixperpupil can be used to
        # normalize
        xx, yy = np.meshgrid(np.arange(self.e.shape[0])-self.e.shape[0]//2,
                             np.arange(self.e.shape[1])-self.e.shape[0]//2)

        self.ttgrid = xx*tip + yy*tilt

        pass

    pass



class Epup(PupilMask):
    """
    An object representing the pupil-plane field prior to the DMs and
    coronagraph.

    Assumed centered about center of pupil with sampling matched to camera.

    Tip/tilt are pre-FPM.

    Arguments:
     lam: real positive scalar wavelength of representation in meters
     e: two-D array of mask representation
     pixperpupil: real positive scalar number of pixels across the defining
      pupil in this plane. Used to scale Fourier transforms for pupil ->
      focal plane propagation and back.
     tip: wavefront tip offset, in EXCAM pixels.  Used to accommodate a star
      which is not exactly centered with respect to the FPM.  Assumed a real
      scalar.  tip = +1 moves one pixel along the positive x-axis (columms).
     tilt: wavefront tilt offset, in EXCAM pixels.  Used to accommodate a star
      which is not exactly centered with respect to the FPM.  Assumed a real
      scalar.  tilt = +1 moves one pixel along the positive y-axis (rows).

    """
    def __init__(self, lam, e, pixperpupil, tip, tilt):
        super(Epup, self).__init__(lam, e, pixperpupil)
        self.plane = 'epup'

        check.real_scalar(tip, 'tip', TypeError)
        check.real_scalar(tilt, 'tilt', TypeError)

        self.tip = tip
        self.tilt = tilt

        self.ttgrid = self.get_ttgrid(self.tip, self.tilt)

        pass


    def get_ttgrid(self, tip, tilt):
        """
        Get the grid (ttgrid) used to change star tip/tilt.

        Note that this function does not change any model internals.

        Arguments:
         tip: wavefront tip offset, in EXCAM pixels.  Used to accommodate a
          star which is not exactly centered with respect to the FPM.  Assumed
          a real scalar.  tip = +1 moves one pixel along the positive x-axis
          (columms).
         tilt: wavefront tilt offset, in EXCAM pixels.  Used to accommodate a
          star which is not exactly centered with respect to the FPM.  Assumed
          a real scalar.  tilt = +1 moves one pixel along the positive y-axis
          (rows).

        """

        check.real_scalar(tip, 'tip', TypeError)
        check.real_scalar(tilt, 'tilt', TypeError)

        # Make grids in pixel coordinates so pixperpupil can be used to
        # normalize
        xx, yy = np.meshgrid(np.arange(self.e.shape[0])-self.e.shape[0]//2,
                             np.arange(self.e.shape[1])-self.e.shape[0]//2)

        return xx*tip + yy*tilt

    pass


class DMFace(object):
    """
    An object representing the field change from reflection off of a
    deformable mirror

    Assumed centered about center of pupil with sampling matched to camera.

    Arguments:
     z: real scalar distance of DM from pupil, in meters
     pitch: real positive scalar actuator pitch (spacing between actuators),
      in meters per actuator
     dmvobj: a ``DM`` object which covers the voltage behavior of the
      physical DM
     registration: a dictionary object specifying the relationship between
      the location of the actuator grid and the center of the image.  These
      parameters are fed directly into ``dmhtoph()``.  Required keys are
      ``'nact'``, ``'inf_func'``, ``'ppact_d'``, ``'ppact_cx'``,
      ``'ppact_cy'``, ``'dx'``, ``'dy'``, ``'thact'``, ``'flipx'``.
      See ``dmhtoph()`` documentation for further definitions.

    """
    def __init__(self, z, pitch, dmvobj, registration):
        # Check all inputs before assigning
        check.real_scalar(z, 'z', TypeError)
        check.real_positive_scalar(pitch, 'pitch', TypeError)
        if not isinstance(dmvobj, DM):
            raise TypeError('dmvobj must be DM object')

        # Make sets of keys and check membership
        reqkeys = {'nact', 'inf_func', 'ppact_d', 'ppact_cx', 'ppact_cy',
                   'dx', 'dy', 'thact', 'flipx'}
        optkeys = set() # no optional right now
        try:
            misskeys = reqkeys - set(registration.keys())
            extrakeys = set(registration.keys()) - reqkeys - optkeys
        except AttributeError:
            raise TypeError('Registration parameters must be ' + \
                                'dictionary-like')
        if misskeys != set():
            raise KeyError('Missing required registration parameters ' \
                           + str(list(misskeys)))
        if extrakeys != set():
            raise KeyError('Received unexpected registration ' + \
                           'parameters ' + str(list(extrakeys)))

        # Use the same input checks dmhtoph does at first read
        check.positive_scalar_integer(registration['nact'],
                                      'nact', TypeError)
        check.positive_scalar_integer(registration['ppact_d'], 'ppact_d',
                                                 TypeError)
        check.twoD_array(registration['inf_func'], 'inf_func', TypeError)
        check.real_positive_scalar(registration['ppact_cx'], 'ppact_cx',
                                               TypeError)
        check.real_positive_scalar(registration['ppact_cy'], 'ppact_cy',
                                               TypeError)
        check.real_scalar(registration['dx'], 'dx', TypeError)
        check.real_scalar(registration['dy'], 'dy', TypeError)
        check.real_scalar(registration['thact'], 'thact', TypeError)
        # No check on flipx since every Python object can be used for
        # truth tests

        # ok we're done finally
        self.z = z
        self.pitch = pitch
        self.dmvobj = dmvobj
        self.registration = registration

        # derived parameters
        # we expect obliquity to foreshorten in one direction
        self.pixpermeter = max(self.registration['ppact_cx'],
                               self.registration['ppact_cy'])/self.pitch


        pass

    pass
