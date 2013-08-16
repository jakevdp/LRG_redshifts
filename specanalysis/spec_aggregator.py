import numpy as np

from .tools import interpolate_with_error

class SpecMeanAggregator(object):
    """Spectrum Aggregator to Calculate the Mean

    This class takes a series of SDSS FITS spectra, and aggregates them
    in order to compute the (weighted) mean.
    """
    def __init__(self, coeff0=3.5, coeff1=1E-4, N=4E3):
        self.lam = 10 ** (coeff0 + coeff1 * np.arange(N))
        self.num = 0
        self.num_sq = 0
        self.denom = 0

    def regrid_spec(self, spectrum):
        spec = spectrum.hdulist[0].data[0]
        dspec = spectrum.hdulist[0].data[2]
        dspec[dspec == 0] = np.inf

        lam = spectrum.restframe().wavelength()
        return interpolate_with_error(lam, spec, dspec, self.lam)

    def add_spec(self, spectrum):
        spec_I, dspec_I = self.regrid_spec(spectrum)
        w = 1. / (dspec_I ** 2)

        self.num += w * spec_I
        self.num_sq += (w * spec_I) ** 2
        self.denom += w

    def reduce(self):
        return (self.num / self.denom,
                np.sqrt(self.num ** 2 - self.num_sq) / self.denom)
