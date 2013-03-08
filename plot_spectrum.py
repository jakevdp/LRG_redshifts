import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

from astroML.datasets import fetch_sdss_spectrum
from astroML.datasets.tools import query_plate_mjd_fiber, TARGET_GALAXY_RED


def interpolate_with_error(x, y, dy, x_new, fill_value=np.nan):
    out_of_bounds = (x_new < x[0]) | (x_new > x[-1])
    i_x_new = np.searchsorted(x, x_new)
    i_x_new = i_x_new.clip(1, len(x) - 1).astype(int)
    lo = i_x_new - 1
    hi = i_x_new

    x_lo = x[lo]
    x_hi = x[hi]
    y_lo = y[lo]
    y_hi = y[hi]
    dy_lo = dy[lo]
    dy_hi = dy[hi]

    fact = (x_new - x_lo) / (x_hi - x_lo)

    y_new = y_lo + (y_hi - y_lo) * fact    
    dy_new = np.sqrt(dy_lo ** 2 + (dy_hi ** 2 + dy_lo ** 2) * abs(fact))

    y_new[out_of_bounds] = 0
    dy_new[out_of_bounds] = np.inf

    return y_new, dy_new


class SpecAggregator(object):
    def __init__(self, coeff0=3.5, coeff1=1E-4, N=5E3):
        self.lam = 10 ** (coeff0 + coeff1 * np.arange(N))
        self.count = np.zeros(N, dtype=int)
        self.numerator = np.zeros(N, dtype=int)
        self.denominator = np.zeros(N, dtype=float)

    def add_spec(self, spectrum):
        spec = spectrum.hdulist[0].data[0]
        dspec = np.ones_like(spec)

        dspec = spectrum.hdulist[0].data[2]
        dspec[dspec == 0] = np.inf
        w = 1. / dspec / dspec

        norm = np.dot(spec, w) / np.sum(w)

        lam = spectrum.restframe().wavelength()
        spec_I, dspec_I = interpolate_with_error(lam,
                                                 spec / norm,
                                                 dspec / norm,
                                                 self.lam)
        w = 1. / (dspec_I ** 2)

        self.count += (w > 0)
        self.numerator += spec_I * w
        self.denominator += w

    def reduce(self):
        return (self.numerator / self.denominator,
                np.sqrt(self.count) / self.denominator)


if __name__ == '__main__':
    # Query for red galaxies
    plate, mjd, fiber = query_plate_mjd_fiber(20,
                                              primtarget=TARGET_GALAXY_RED)

    agg = SpecAggregator()

    for (p, m, f) in zip(plate, mjd, fiber):
        spec = fetch_sdss_spectrum(p, m, f,
                                   data_home='.') # download to this directory
                 
        print spec.z
        agg.add_spec(spec)
    
    lam = agg.lam
    spec, dspec = agg.reduce()
    plt.plot(lam, spec, '-k')
    plt.fill_between(lam, spec - dspec, spec + dspec,
                     color='#AAAAAA', alpha=0.3)

    plt.show()
    
