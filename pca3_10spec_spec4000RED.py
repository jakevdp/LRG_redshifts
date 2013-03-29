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

#download data using 'compute_sdss_pca.py'
#https://github.com/astroML/astroML/blob/master/examples/datasets/compute_sdss_pca.py

#--------------data------------------#
data = np.load('spec4000RED.npz')    #
#-----------data.shape---------------#
evals=data['evals']                  #(100,)
log_NII_Ha=data['log_NII_Ha']        #(100,)
log_OIII_Hb=data['log_OIII_Hb']      #(100,)
lineindex_cln=data['lineindex_cln']  #(100,)
spectra=data['spectra']              #(100, 1000)
mask=data['mask']                    #(100, 1000)
evecs=data['evecs']                  #(100, 1000)
zerr=data['zerr']                    #(100,)
mu=data['mu']                        #(1000,)
norms=data['norms']                  #(100,)
spec_cln = data['spec_cln']          #(100,)
coeffs=data['coeffs']                #(100, 10)
coeff1=data['coeff1']                #()
coeff0=data['coeff0']                #()
z=data['z']                          #(100,)
#------------------------------------#

#-----reconstruct-----#
X = sdss_corrected_spectra.reconstruct_spectra(data)  #(100, 1000)
#X = normalize(X) #if needed
rpca = RandomizedPCA(n_components=10, random_state=0)
X_proj = rpca.fit_transform(X)       #(100,10)

#-----multiplot at once-----#
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0.05)

for i in range(10):
    ax = fig.add_subplot(521 + i)
    ax.plot(rpca.components_[i], '-k', )
    if i < 8:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax.set_xlabel('Spectra')
    #ax.set_ylabel('PCA')

plt.show()

#-----reference-----#
##Plots
#Corrected Spectra
#http://astroml.github.com/examples/datasets/plot_corrected_spectra.html
#Projection Spectra
#http://astroml.github.com/book_figures/chapter7/fig_PCA_components.html
#Spectra Decompositions
#http://astroml.github.com/book_figures/chapter7/fig_spec_decompositions.html
