import sys

import numpy as np
import matplotlib.pyplot as plt

from astroML.datasets import fetch_sdss_spectrum
from astroML.datasets.tools import query_plate_mjd_fiber, TARGET_GALAXY_RED

from specanalysis import SpecMeanAggregator


primtarget=TARGET_GALAXY_RED
zlim=(0.2, 0.6)

plate, mjd, fiber = query_plate_mjd_fiber(100, primtarget, zlim[0], zlim[1])

agg = SpecMeanAggregator()
lam = agg.lam

zdist = []

for plate_n, mjd_n, fiber_n in zip(plate, mjd, fiber):
    sys.stdout.write("{0}.{1}.{2}         \r".format(plate_n, mjd_n, fiber_n))
    sys.stdout.flush()
    spec = fetch_sdss_spectrum(plate_n, mjd_n, fiber_n)
    zdist.append(spec.z)
    agg.add_spec(spec)

sys.stdout.write('\n')
    
spec, dspec = agg.reduce()

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(lam, spec, '-k')
#ax[0].fill_between(lam, spec - dspec, spec + dspec,
#                   color='#AAAAAA', alpha=0.3)

ax[0].set_xlim(2500, 7500)
ax[0].set_ylim(0, 0.7)
ax[0].set_xlabel(r'$\lambda')
ax[0].set_ylabel('normalized flux')

ax[1].hist(agg.redshifts, bins=20, histtype='stepfilled', alpha=0.3)
ax[1].set_xlabel('z')
ax[1].set_ylabel('N(z)')

plt.show()
