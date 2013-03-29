import sys
import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_sdss_corrected_spectra
from astroML.datasets import fetch_sdss_spectrum
from astroML.datasets.tools import query_plate_mjd_fiber, TARGET_GALAXY_RED

primtarget=TARGET_GALAXY_RED
zlim=(0.42, 0.55)

plate, mjd, fiber = query_plate_mjd_fiber(20, primtarget, zlim[0], zlim[1])
print plate,mjd,fiber

for plate_n, mjd_n, fiber_n in zip(plate, mjd, fiber):
    print plate_n, mjd_n, fiber_n

    spec = fetch_sdss_spectrum(plate_n, mjd_n, fiber_n)    
    l = plt.plot(spec.wavelength(), spec.spectrum)
    c = l[0].get_color()

plt.xlim(3000, 10000)

plt.show()
