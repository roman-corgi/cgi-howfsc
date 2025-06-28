# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Compares the results of pairwise probing to the saved electric fields from
loadprobe.py
"""

import matplotlib.pyplot as plt
import astropy.io.fits as pf
import numpy as np

import howfsc.sensing.pairwise_sensing as ps

def realdiff(phasediff):
    """phase wrapping to -pi, pi"""
    return (phasediff + np.pi) % (2*np.pi)-np.pi

if __name__ == "__main__":
    print("Run this after running loadprobe.py")

    real_e0 = pf.open('real_e0.fits')[0].data
    imag_e0 = pf.open('imag_e0.fits')[0].data
    true_phase = np.angle(real_e0+1j*imag_e0)

    intensitycube = pf.open('intensitycube.fits')[0].data
    phasecube = pf.open('phasecube.fits')[0].data

    sensed_efield, badpixels = ps.estimate_efield(intensitycube, phasecube)
    sensed_phase = np.angle(sensed_efield)

    diff = sensed_phase-true_phase
    data = [true_phase, sensed_phase, realdiff(diff)]
    titles = ['True phase (rad)', 'Sensed phase (rad)', 'Difference (rad)']

    fig = plt.figure(figsize=(8, 4))
    columns = 3
    rows = 1
    for i in range(1, columns*rows+1):
        print(i)
        fig.add_subplot(rows, columns, i)
        im = plt.imshow(data[i-1], cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.title(titles[i-1])
        plt.axis('off')
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        if i != 3:
            #cb.remove()
            plt.draw()
    plt.show()
