# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Documentation for the various status codes that HOWFSC can return

Each is tied 1:1 to an Exception class, except for nominal (no error)
"""

status_codes = {
    # standard output if nothing when wrong
    'nominal': 0,

    # unknown exception not matching any expected failure mode
    'Exception': 1,

    #------------
    # Bad inputs
    #------------

    # TypeError: invalid input data type
    'TypeError': 2,

    # ValueError: invalid input data range. note: real and positive/nonnegative
    # is handled as part of "type"
    'ValueError': 3,

    #--------------------------------------------------
    # Potential real-time algorithmic or data problems
    #--------------------------------------------------

    # (ConstrainDMException in constrain_dm.py) Raised when the code to
    # constrain the DM setting (enforcing tied and capped actuators) hits its
    # maximum iteration limit.  Not necessarily a bug, could be a somewhat
    # pathological DM setting that needs investigation why it hit the limit.
    'ConstrainDMException': 4,

    # (InversionException) Raised when the linear solver fails.  This could
    # happen during GITL and would be a bad thing.  Likely cause is
    # insufficient regularization in the control strategy.
    'InversionException': 5,

    # (ZeroDivisionError) unexpected divide by zero.  Known raise in
    # preclean.py if there are no good pixels to use to compute average
    # contrast
    'ZeroDivisionError': 6,

    #--------------------------------------
    # Potential file/precomputation issues
    #--------------------------------------

    # (KeyError) missing dictionary key; will also throw for extra keys in
    # cases where that matters.  Known raise in mask.py, mode.py, parse_cs.py
    # for YAML loads
    'KeyError': 7,

    # (IOError) thrown when a file could not be loaded (by load.py,
    # darkhole.py, mode.py, parse_cs.py)
    'IOError': 8,

    # (CalcJacsException) Raised when the model produces zero outputs for
    # things that shouldn't be zero. Suggests a poorly formed model (bad inputs
    # defining it).  Won't be raised by GITL as Jacobians are not built in
    # real time.
    'CalcJacsException': 9,

    # (CSException) Raised when the control strategy file is inconsistent with
    # the CS file specification.  Shouldn't be raised in real-time if the file
    # has been pre-validated; file path issues will raise an IOError.
    'CSException': 10,

    # (MDFException) Raised when the HOWFSC optical model definition file is
    # inconsistent with the MDF file specification.  Shouldn't be raised in
    # real-time if the file has been pre-validated.
    'MDFException': 11,

    #--------------------------------------------------
    # Rare should-never-happen software bug conditions
    #--------------------------------------------------

    # (ActLimitException in actlimits.py) Thrown when an actuator is marked to
    # be both tied and frozen.  Should not be thrown unless there is a genuine
    # bug.
    'ActLimitException': 12,

    # (CheckException in check.py) Thrown when invalid options are given to a
    # data type checker.  Distinct from TypeError so we can tell when failures
    # are due to the data type or the code to check the data type.  Should not
    # be thrown unless there is a genuine bug, as this implies the code is
    # wrong.
    'CheckException': 13,

    # (SingleLambdaException) Raised when 1) a class element is None and
    # should have been defined, or 2) logic falls through.  These are all
    # "should never happen" events.  Should not be thrown unless there is a
    # genuine bug.
    'SingleLambdaException': 14,

    #----------------------------------------------------
    # Exceptions and warnings imported/derived from eetc
    #----------------------------------------------------

    # (EXCAMOptimizeException) Raised when eetc is not able to find any
    # usable set of camera parameters, or when the underlying constraints
    # are fundamentally incompatible, e.g. star is too bright to avoid
    # saturation even at the shortest possible exposure time.  The two-layer
    # optimization means this should be an incredibly rare event, and likely
    # indicates something wrong with the inputs to eetc.
    'EXCAMOptimizeException': 15,

    # (ThptToolsException) Raised when a throughput in a band is greater than
    # 100% or less than 0%.  Checked to be sure interpolator artifacts are
    # caught.  Extremely difficult to raise, as all throughput curves are
    # checked at load to be in-bounds.
    'ThptToolsException': 16,

    # LowerThanExpectedSNR status is used when the first optimizer is unable to
    # find a set of camera settings that meet the SNR target, and instead has
    # selected settings that will give you the best SNR you can get.
    #
    # NOTE: This is *NOT* an exception!  This is a warning that the performance
    # may not be as good as hoped due the physical constraints, but you
    # certainly can choose to go ahead with the iteration anyway.
    'LowerThanExpectedSNR': 17,

    }
