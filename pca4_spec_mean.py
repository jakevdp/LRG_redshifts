import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

from astroML.datasets import fetch_sdss_spectrum
from astroML.datasets.tools import query_plate_mjd_fiber, TARGET_GALAXY_RED

#############################################################
#### modified query part from 'plot_spectrum.py' Jakevdp ####
#############################################################

def interpolate_with_error(x, y, dy, x_new, fill_value=np.nan):
    """Linear Interpolation, keeping track of error propagation

    Parameters
    ----------
    x, y, dy : arrays
        locations, values, and errors of input points
    x_new : array
        locations of points to be interpolated
    fill_value : array
        value to use outside the domain of x

    Returns
    -------
    y_new, dy_new : arrays
        values and errors at the interpolated locations
    """
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
    def __init__(self, coeff0=3.5, coeff1=1E-4, N=4E3):
        self.lam = 10 ** (coeff0 + coeff1 * np.arange(N))
        self.count = np.zeros(N, dtype=int)
        self.numerator = np.zeros(N, dtype=int)
        self.denominator = np.zeros(N, dtype=float)

    def add_spec(self, spectrum):
        spec = spectrum.hdulist[0].data[0]
        dspec = spectrum.hdulist[0].data[2]
        dspec[dspec == 0] = np.inf
        w = 1. / dspec / dspec

        lam = spectrum.restframe().wavelength()
        spec_I, dspec_I = interpolate_with_error(lam,
                                                 spec,
                                                 dspec,
                                                 self.lam)
        w = 1. / (dspec_I ** 2)

        # compute norm by matching new spectrum to current mean,
        # accounting for measurement errors
        if np.sum(self.count) == 0:
            norm = np.dot(spec_I, w) / np.sum(w)
        else:
            mu = self.numerator / self.denominator
            dmu2 = self.count / (self.denominator ** 2)
            
            mu[self.count == 0] = 0
            dmu2[self.count == 0] = np.inf
            sig2 = dmu2 + dspec_I ** 2

            num = spec_I * spec_I / sig2
            denom = spec_I * mu / sig2
            norm = num.sum() / denom.sum()

        self.count += (w > 0)
        self.numerator += norm * spec_I * w
        self.denominator += norm * norm * w

    def reduce(self):
        return (self.numerator / self.denominator,
                np.sqrt(self.count) / self.denominator)


if __name__ == '__main__':

    #This part is modified from 'plot_spectrum.py' but got a different error message.(see extended description.)

    # Query for red galaxies
    
    primtarget=TARGET_GALAXY_RED
    zlim=(0.42, 0.55)

    plate, mjd, fiber = query_plate_mjd_fiber(20, primtarget, zlim[0], zlim[1])

    agg = SpecAggregator()

    for plate_n, mjd_n, fiber_n in zip(plate, mjd, fiber):
        spec = fetch_sdss_spectrum(plate_n, mjd_n, fiber_n)
                 
        print spec.z
        agg.add_spec(spec)
    
    lam = agg.lam
    spec, dspec = agg.reduce()
    plt.plot(lam, spec, '-k')
    plt.fill_between(lam, spec - dspec, spec + dspec,
                     color='#AAAAAA', alpha=0.3)

    plt.show()