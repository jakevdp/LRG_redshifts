import warnings
import numpy as np

from .tools import interpolate_with_error

class SpecMeanAggregator(object):
    """Spectrum Aggregator to Calculate the Mean

    This class takes a series of SDSS FITS spectra, and aggregates them
    in order to compute the (weighted) mean.
    """
    def __init__(self, coeff0=3.4, coeff1=1E-4, N=5E3):
        self.lam = 10 ** (coeff0 + coeff1 * np.arange(N))
        self.num = 0
        self.num_sq = 0
        self.denom = 0
        self.err2 = 0
        self.redshifts = []

    def _regrid_spec(self, spectrum):
        """
        Given a spectrum object, shift to the rest frame and regrid on
        the internal binning.
        """
        spec = spectrum.hdulist[0].data[0]
        dspec = spectrum.hdulist[0].data[2]
        dspec[dspec == 0] = np.inf

        lam = spectrum.restframe().wavelength()
        return interpolate_with_error(lam, spec, dspec, self.lam)

    def _find_normalization(self, spec, dspec):
        """Compute the normalization factor of the interpolated spectrum"""
        mu, dmu = self.reduce()
        w = 1. / (dmu ** 2 + dspec ** 2)

        num = (mu * spec * w).sum()
        denom = (spec * spec * w).sum()
        return num / denom

    def add_spec(self, spectrum):
        spec_I, dspec_I = self._regrid_spec(spectrum)
        N = self._find_normalization(spec_I, dspec_I)
        spec_I *= N
        dspec_I *= N
        w = 1. / (dspec_I ** 2)

        self.num += w * spec_I
        self.num_sq += (w * spec_I) ** 2
        self.denom += w
        self.err2 += dspec_I ** 2
        self.redshifts.append(spectrum.z)

    def reduce(self):
        if len(self.redshifts) == 0:
            mu = np.ones(len(self.lam))
            dmu = np.ones(len(self.lam))
        else:
            # there will likely be some 0/0 entries:
            # suppress warnings and take care of these later
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu = self.num / self.denom
                dmu2 = (self.err2 / self.denom
                        + self.num_sq / self.denom
                        - (self.num ** 2) / self.denom ** 2)
                #dmu2[dmu2 < 0] = 0
                dmu = np.sqrt(dmu2)

            i_nan = (self.denom == 0)
            mu[i_nan] = 0
            dmu[i_nan] = np.inf

        return mu, dmu
