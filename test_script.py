import numpy as np
import matplotlib.pyplot as plt

from astroML.datasets import fetch_sdss_spectrum
from astroML.datasets.tools import query_plate_mjd_fiber, TARGET_GALAXY_RED

from specanalysis import SpecMeanAggregator


primtarget=TARGET_GALAXY_RED
zlim=(0.42, 0.55)

plate, mjd, fiber = query_plate_mjd_fiber(20, primtarget, zlim[0], zlim[1])

agg = SpecMeanAggregator()
lam = agg.lam

for plate_n, mjd_n, fiber_n in zip(plate, mjd, fiber):
    spec = fetch_sdss_spectrum(plate_n, mjd_n, fiber_n)
                 
    print spec.z
    agg.add_spec(spec)

    specI, dspecI = agg.regrid_spec(spec)
    plt.plot(lam, specI)
    
spec, dspec = agg.reduce()

plt.plot(lam, spec, '-k')
#plt.plot(lam, dspec, '-b')

plt.fill_between(lam, spec - dspec, spec + dspec,
                 color='#AAAAAA', alpha=0.3)

plt.xlim(3000, 6500)
plt.ylim(0, 10)

plt.show()
