import sys
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from numpy import *
from astroML.datasets import fetch_sdss_corrected_spectra
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import normalize
from astroML.datasets import sdss_corrected_spectra

#-----Modified PCA---------#
data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()
X = sdss_corrected_spectra.reconstruct_spectra(data) #(4000,1000)

#--------data.shape--------# 
labels = data['spec_cln']  #(4000,)
X0 = X[(labels == 0)]      #(1,1000) all spectra with label 0
labels[0] = labels[labels == 0] #()
#np.transpose(X0).shape  #(1000,)
mu = data['mu']          #(1000,)
evals=data['evals']      #(1000,)
NII=data['log_NII_Ha']   #(4000,)
OIII=data['log_OIII_Hb'] #(4000,)
mask=data['mask']        #(4000,1000) True, False
index=data['lineindex_cln'] #(4000,)
evecs=data['evecs']     #(1000,1000)
zerr=data['zerr']       #(4000,)
norms=data['norms']     #(4000,)
cfs=data['coeffs']      #(4000,10)
cf1=data['coeff1']      #()
cf0=data['coeff0']      #() 
z=data['z']             #(4000,)
#-----------------------#

#-----normalizing spectra-----#
#X = normalize(X)

#-----Randomized PCA-----#
rpca = RandomizedPCA(n_components=4, random_state=0) #n_component=(4000,n)
X_proj = rpca.fit_transform(X)
X_proj.shape   #(4000,4)

#-----plot 4 eigenspectra-----#
for i in range(4): #1~4 number of eigenspectra
    l = pl.plot(rpca.components_[i] + 0.7 * i)    
    c = l[0].get_color()
    pl.text(600, -0.24 + 0.7 * i, "component %i" % (i + 1), color=c)

#-----plot mean spectrum-----#
l = pl.plot(rpca.mean_ - 3)
c = l[0].get_color()
pl.text(600, 10, "Mean Spectrum", color=c)

#------------------------------#
pl.xlim(0, 1000)
pl.ylim(-1, 22)
#pl.xlabel('wavelength (Angstroms)')
#pl.ylabel('scaled flux + offset')
pl.title('Mean Spectrum and Eigen-spectra')
pl.show()
