# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Collection of routines to freeze actuators passing limits and to tie ones that
are too far apart ("neighbor-rule violations")
"""

import numpy as np
from scipy.sparse import lil_matrix

from howfsc.model.mode import CoronagraphMode
from howfsc.model.dmobj import DM
from howfsc.util.flat_tie import checktie
import howfsc.util.check as check

class ActLimitException(Exception):
    """Thin class for exceptions in DM limit identification"""
    pass


def sparsefrommap(limitlist, cfg):
    """
    Given lists of low, high, and tied actuators for each DM, create a sparse
    matrix which freezes the low and high and ties the neighbor-rule groups.

    Arguments:
     limitlist: a list of dictionaries, each of which should contain the keys
      'freeze' and 'link'.  The contents of 'freeze' must be a list of indices,
      and the contents of 'link' must be a list of lists of indices.  If an
      index appears in freeze, it must not appear in tie, and vice versa. This
      list should be of the same length as ``cfg.dmlist``. A dictionary of
      this form and obeying these constraints will be output by
      ``maplimits()`` for each DM.
     cfg: CoronagraphMode object

    Returns:
     2D sparse CSR-format square array with total number of actuators on each
      side (e.g. for 2 48*48 DMs, this will be 4608x4608).

    """

    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('Invalid cfg in wdmfrommap()')

    try:
        if len(limitlist) != len(cfg.dmlist):
            raise TypeError('limitlist invalid length')
        pass
    except TypeError: # not iterable
        raise TypeError('dmlist not a list')

    try:
        for index, limit in enumerate(limitlist):
            if 'freeze' not in limit.keys():
                raise KeyError('Missing frozen actuators on DM'
                               + str(index))
            if 'link' not in limit.keys():
                raise KeyError('Missing tied actuators on DM'
                               + str(index))
            pass
        pass
    except AttributeError: # not a dict, doesn't have keys()
        raise TypeError('Limit list element not a dictionary')

    # Check for duplicate indices per DM (sets just make this efficient)
    for limit in limitlist:
        fset = set(limit['freeze'])
        tset = set()
        for tie in limit['link']:
            tset.update(tie)
            pass
        pass
        if bool(fset & tset): # empty set -> False, only if sets disjoint
            raise ActLimitException('Indices in freeze and tie must not be ' +
                                    'in both')
        pass

    # Only include sparse modifiers
    # Most efficient to build as lil and convert to csr for matrix ops
    ndmact = np.cumsum([0]
              +[dm.registration['nact']**2 for dm in cfg.dmlist]).astype('int')
    sparse_constraints = lil_matrix((ndmact[-1], ndmact[-1]))

    for index, limit in enumerate(limitlist):
        # One for diagonal elements (zero after subtract from identity)
        freeze = np.array(limit['freeze']).astype('int')
        sparse_constraints[ndmact[index]+freeze,
                           ndmact[index]+freeze] = 1.

        # Safe to set groups after zeroing as maplimits() will not return
        # indices in more than one group
        # Tweak to match I-F formulation
        for grp in limit['link']:
            for i in grp:
                sparse_constraints[ndmact[index]+np.array(i).astype('int'),
                    ndmact[index]+np.array(grp).astype('int')] = -1./len(grp)
                pass
            for i in grp:
                sparse_constraints[ndmact[index]+np.array(i).astype('int'),
                    ndmact[index]+np.array(i).astype('int')] += 1.
            pass
        pass
    sparse_constraints = sparse_constraints.tocsr()
    return sparse_constraints





def maplimits(dmv, dmvobj, tiemap=None):
    """
    Create lists of actuators to freeze and link for a single DM for HOWFSC.

    This includes an input matrix (tiemap) which indicates which actuators have
    fixed electrical constraints (dead or driven together)

    This map incorporates 5 effects:
     - dead actuators are frozen
     - actuators at or below low-voltage constraints are frozen
     - actuators at or above high-voltage constraints are frozen
     - electrically-connected actuators groups are linked together to move
       together
     - actuators at the boundaries of neighbor rule violations (both lateral
      and diagonal) are linked togther to move together
    Since HOWFSC computes changes in DM settings, constraining them to move
    together is equivalent to forcing their delta-DM settings to be the same.
    It doesn't force the underlying absolute voltages to be the same.

    Does not fix neighbor rule violations or cap violations! It only identifies
     ones that do violate; the handling is elsewhere.

    Regions can be marked as candidates for either freezing or linking, or
    neither. They will never be a member of both, and if an actuator would be
    in both, it is frozen.  Indices will be stored in ndarrays.

    For each of x, y, diag, other diag:
     find all the violations, pick the first one, and check it for adjacent
     violations until you run out nearby acts.  Go to the next one.  Repeat
     until x is done, then y, then diag, then other diag.

    Based on original HCIT implementation by Brian Kern.

    Arguments:
     dmv: array with target voltage
     dmvobj: DM object containing voltage information for a specific DM

    Keyword Arguments:
     tiemap: a 2D array of integers, of the same size as dmv, which can take
      on the values 0, -1, or consecutive integers 1 -> N, or None.  If not
      None, these values will encode dead and electrically-connected actuators
      (dead = -1, connected = 1, 2, etc., neither = 0).  If None, this function
      will behave as though there are no electrical constraints.

      The tiemap encodes actuators that not only move together, but also must
      have the same absolute voltage.  Generally it represents physical
      hardware constraints.

    Returns:
     dictionary with two keys: ['freeze', 'link'], which are lists of
      indices of actuators which are low/high (and need to be frozen) and
      groups that are outside the neighbor-rule limit (and need to be link)

    """

    ngval = 0 # value for "no group"
    deadval = -1 # special code for dead actuators
    invalid = -2 # value not otherwise permitted, must be < ngval

    # Check inputs
    check.twoD_array(dmv, 'dmv', TypeError)
    if not isinstance(dmvobj, DM):
        raise TypeError('Invalid dmvobj in maplimits()')
    if dmv.shape != dmvobj.flatmap.shape:
        raise TypeError('dmv shape must match flatmap')

    if tiemap is not None:
        check.twoD_array(tiemap, 'tiemap', TypeError)
        if dmv.shape != tiemap.shape:
            raise TypeError('dmv shape must match tiemap')
        if not checktie(tiemap):
            raise ValueError('tiemap must have values 0, -1, or consecutive ' +
                             'integers 1 -> N')
        pass
    else:
        tiemap = ngval*np.ones(dmv.shape, dtype='int')
        pass

    ddm = dmv - dmvobj.flatmap

    margin = 2*dmvobj.vquant # use 2LSB to match dmsmooth implementation

    # Call it a match if it's within margin to avoid rounding issues
    vlo = dmvobj.vmin + margin
    vhi = dmvobj.vmax - margin
    vnb = dmvobj.vneighbor - margin
    vco = dmvobj.vcorner - margin

    # freeze if *at* threshold as well as beyond, so we can fix values to
    # a threshold elsewhere and they stick.
    mlohi = np.logical_or(dmv <= vlo, dmv >= vhi)

    # freeze dead actuators
    mlohi = np.logical_or(mlohi, tiemap == deadval)

    delx = ddm[:, 1:] - ddm[:, :-1] # x neighbors
    dely = ddm[1:, :] - ddm[:-1, :] # y neighbors
    delxy = ddm[1:, 1:] - ddm[:-1, :-1] # +x+y neighbors
    delyx = ddm[1:, :-1] - ddm[:-1, 1:] # -x+y neighbors

    mx, my = [np.abs(d) > vnb for d in [delx, dely]]
    mxy, myx = [np.abs(d) > vco for d in [delxy, delyx]]
    ms = mx, my, mxy, myx # pack for passing to _growgrp compactly

    # igrp is a map of group members. 0 is no group, 1+ by int are individual
    # groups, -1 is the special group of dead actuators.  -2 is used by
    # igrploop only to get through first call to while() since it will always
    # fail
    #
    # Note that _growgrp changes igrp in place.

    igrp = ngval*np.ones(tiemap.shape, dtype='int')
    ig = ngval
    igrploop = invalid*np.ones(igrp.shape, dtype='int')
    while (igrp != igrploop).any():
        igrploop = igrp.copy()
        mnog = igrp == ngval
        mnew = np.logical_and(mx, np.logical_and(mnog[:, 1:], mnog[:, :-1]))
        # new groups that contain x neighbors, as (row, col) tuples
        # from the difference matrix mx
        wnew = np.transpose(np.nonzero(mnew))
        if wnew.size > 0:
            firstw = wnew[0]
            ig += 1
            igrp[firstw[0], firstw[1]] = ig
            igrp[firstw[0], firstw[1]+1] = ig
            _growgrp(ig, igrp, ms)
            pass
        pass

    igrploop = invalid*np.ones(igrp.shape)
    while (igrp != igrploop).any():
        igrploop = igrp.copy()
        mnog = igrp == ngval
        mnew = np.logical_and(my, np.logical_and(mnog[1:, :], mnog[:-1, :]))
        # new groups that contain y neighbors, as (row, col) tuples
        # from the difference matrix my
        wnew = np.transpose(np.nonzero(mnew))
        if wnew.size > 0:
            firstw = wnew[0]
            ig += 1
            igrp[firstw[0], firstw[1]] = ig
            igrp[firstw[0]+1, firstw[1]] = ig
            _growgrp(ig, igrp, ms)
            pass
        pass

    igrploop = invalid*np.ones(igrp.shape)
    while (igrp != igrploop).any():
        igrploop = igrp.copy()
        mnog = igrp == ngval
        mnew = np.logical_and(mxy,
                              np.logical_and(mnog[1:, 1:], mnog[:-1, :-1]))
        # new groups that contain +x+y neighbors, as (row, col) tuples
        # from the difference matrix mxy
        wnew = np.transpose(np.nonzero(mnew))
        if wnew.size > 0:
            firstw = wnew[0]
            ig += 1
            igrp[firstw[0], firstw[1]] = ig
            igrp[firstw[0]+1, firstw[1]+1] = ig
            _growgrp(ig, igrp, ms)
            pass
        pass

    igrploop = invalid*np.ones(igrp.shape)
    while (igrp != igrploop).any():
        igrploop = igrp.copy()
        mnog = igrp == ngval
        mnew = np.logical_and(myx,
                              np.logical_and(mnog[1:, :-1], mnog[:-1, 1:]))
        # new groups that contain -x+y neighbors, as (row, col) tuples
        # from the difference matrix myx
        wnew = np.transpose(np.nonzero(mnew))
        if wnew.size > 0:
            firstw = wnew[0]
            ig += 1
            igrp[firstw[0], firstw[1]+1] = ig
            igrp[firstw[0]+1, firstw[1]] = ig
            _growgrp(ig, igrp, ms)
            pass
        pass

    # Combine tiemap with neighbor-rule-only map at the very end
    filt = tiemap.copy()
    filt[filt == deadval] = 0 # don't try to treat dead as linked together here
    fgrps = set(filt.ravel()) - {ngval}

    for f in fgrps:
        inds = (filt == f)
        igrp_num = set(igrp[inds].ravel()) - {ngval}

        # Pick a new number not used in igrp, and set 1) all igrp indices that
        # match the tiemap group (inds) to that number and 2) all igrp members
        # that share a number with a element in igrp[inds], to spread out and
        # catch nonlocal linkages
        nextval = np.max(igrp)+1
        for n in igrp_num:
            igrp[igrp == n] = nextval
            pass
        igrp[inds] = nextval
        pass

    # Separate low and high for freezing; rest to be linked
    igrplohi = np.unique(igrp[mlohi])
    for i in igrplohi: # if a group contains any low or high limits,
                     # freeze whole group
        if i > ngval:
            mlohi[igrp == i] = True
            igrp[igrp == i] = ngval # If it's frozen, don't link it together
            pass
        pass

    freeze = np.flatnonzero(mlohi)
    link = [] # list of lists
    igrp_num = set(igrp.ravel()) - {ngval}
    for i in igrp_num:
        igrplist = np.flatnonzero(igrp == i)
        if igrplist.size != 0: # strip empty arrays left after freeze
            link.append(igrplist)
            pass
        pass
    return {'freeze':freeze, 'link':link}


def _growgrp(ig, igrp, ms):
    """
    Given a seed group of neighbor-rule-violating actuators, expand that group
    until it includes all NR violations linked to that seed group.

    This function modifies the igrp array in place! As such it returns nothing.

    Arguments:
     ig: group index, will be an integer >= 0
     igrp: 2D array of integers of the same size as the DM.  This matrix
      assigns a group number to each actuator (integer >= 0) or -1 if that
      actuator is not part of a group.  Actuators in a group all move together.
     ms: a tuple with four arrays of neighbor-rule violators -> mx (horizontal)
      my (vertical), mxy (diagonal), myx (other diagonal)

    """
    mx, my, mxy, myx = ms

    # Loop through and check all left/right/up/down/diagonal neighbors to
    # existing group to see if they are neighbor-rule violators with group
    # members.  If they are, expand the group around them and recheck to see
    # if the expansion picked up new violators.  If not, call it done.
    igrplast = np.zeros(igrp.shape, dtype='bool')
    while (igrplast != igrp).any():
        igrplast = igrp.copy()

        # horizontal
        ig0 = igrp == ig
        mnew = np.logical_and(mx, np.logical_or(ig0[:, 1:], ig0[:, :-1]))
        wnew = np.transpose(np.nonzero(mnew))
        for w in wnew:
            igrp[w[0], w[1]] = ig
            igrp[w[0], w[1]+1] = ig
            pass

        # vertical
        ig0 = igrp == ig
        mnew = np.logical_and(my, np.logical_or(ig0[1:, :], ig0[:-1, :]))
        wnew = np.transpose(np.nonzero(mnew))
        for w in wnew:
            igrp[w[0], w[1]] = ig
            igrp[w[0]+1, w[1]] = ig
            pass

        # one diagonal
        ig0 = igrp == ig
        mnew = np.logical_and(mxy, np.logical_or(ig0[1:, 1:], ig0[:-1, :-1]))
        wnew = np.transpose(np.nonzero(mnew))
        for w in wnew:
            igrp[w[0], w[1]] = ig
            igrp[w[0]+1, w[1]+1] = ig
            pass

        # other diagonal
        ig0 = igrp == ig
        mnew = np.logical_and(myx, np.logical_or(ig0[1:, :-1], ig0[:-1, 1:]))
        wnew = np.transpose(np.nonzero(mnew))
        for w in wnew:
            igrp[w[0], w[1]+1] = ig
            igrp[w[0]+1, w[1]] = ig
            pass
        pass
    return
