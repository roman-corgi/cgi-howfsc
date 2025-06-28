# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Script to test phase estimation via pairwise probing
"""

import numpy as np
import howfsc.sensing.pairwise_sensing as ps


def fake_probe(amp, size):
    """
    Generates a probe of random phase and known amplitude.
    A matrix of shape (size x size)
    """
    return amp*np.exp(1j*np.random.random((size, size)))

def fake_probe_randamp(size):
    """
    Generates a probe of random phase and random amp.
    A matrix of shape (size x size)
    """
    amp = np.random.random((size, size))
    return amp*np.exp(1j*np.random.random((size, size)))

def generate_initial_efield(size):
    """
    A random electric field of (size x size)
    This field is the "truth", ie, the one we are trying to recover
    """
    return np.random.random((size, size)) + 1j*np.random.random((size, size))

def efield_to_intensity(x):
    """Converts a complex electric field to an intensity"""
    return np.abs(x)**2

def generate_intensity_images(actual_efield, probelist):
    """
    Takes an electric field, adds probes to it via eq'n 6 of
    Give'on, Kern, Shaklan; and converts the sum fields to intensities.
    Returns:
    """
    outputlist = []
    outputlist.append(efield_to_intensity(actual_efield))
    for probe in probelist:
        #eq'n 6 of giveon, kern, shaklan
        outputlist.append(efield_to_intensity(actual_efield + 1j*probe))
    return outputlist

def generate_probes(probe_amp=None, n_probes=2, size=None):
    """
    Generates a positive and negative probe given an amplitude.
    These are C[E_t Delta psi] in eqn 6 of giveon, kern, shaklan 2011,
    Returns: electric fields of probes (typically estimated in practice)
    """
    assert n_probes >= 2, "Need two or more probes"
    output = []
    for _ in range(n_probes):
        probe = fake_probe(amp=probe_amp, size=size)
        #probe = fake_probe_randamp(size=size)
        output.append(probe)
        output.append(-1*probe)
    return output

def runall(actual_efield, probe_amp, n_probes):
    """
    Given an electric field, a probe amplitude, and number of probes, estimates
    the electric field using the algorithm in Giv'eon, Kern, Shaklan.
    """
    #probes only implemented for squares for now.  Does not affect recovery
    #algorithms
    assert actual_efield.shape[0] == actual_efield.shape[1], \
      " currently implemented only for square arrays"
    size = actual_efield.shape[0]
    plist = generate_probes(probe_amp=probe_amp,
                            n_probes=n_probes,
                            size=size)
    intensities = np.array(generate_intensity_images(actual_efield, plist))
    #estimate of the positive phase probes
    phases = np.array([np.angle(x) for x in plist[0::2]])

    complex_estimate = ps.estimate_efield(intensities, phases)[0]
    return complex_estimate


if __name__ == "__main__":
    np.random.seed(42)
    _probe_amp = 0.1
    _n_probes = 2
    _size = 10 #ie size x size image

    _actual_efield = generate_initial_efield(_size)
    _complex_estimate = runall(_actual_efield, _probe_amp, _n_probes)

    print("Test passed?")
    print(np.allclose(_complex_estimate, _actual_efield))
