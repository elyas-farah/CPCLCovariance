import pymaster as nmt
import numpy as np
import healpy as hp
import wget
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.integrate import simpson

import matplotlib
import glass

from utils2 import *
from tqdm import tqdm
import gc
import concurrent.futures

nside_mask = 256

npix = hp.nside2npix(nside_mask)

sel = np.ones(npix)

# theta, phi = hp.pix2ang(nside_mask, np.arange(npix))

# sel = np.sin(theta)

# sel[theta > np.pi / 2] = 0
# sel[phi > np.pi/3*2 ] = 0
cl_extern = np.load("/home/s59efara_hpc/covariance/test_cl_kappadm.npy")
ell_extern = np.arange(len(cl_extern[:,0,0]))






nside = 256
nell = 30
lmin = 1

edges = np.unique(np.geomspace(lmin,3*nside - 1,nell).astype(int))

# edges= np.arange(0, 3*nside + 1)

# cls_prediction_list = []
# cls_fixed_field_list = []
# cls_field_variance_list = []

# var_prediction_list = []
# var_fixed_field_list = []
# var_field_variance_list = []

result1_list = []
result2_list = []




def f1(i, j, cat_size, spin, spin2):
    leff_fixed_auto, cl_mean_rnd_fixed_auto, cl_prediction_fixed_auto, covmat_fixed_auto, analytic_cov_fixed_auto, field_variance_cov_fixed_auto, Nf_list, Nf_var, cls_auto  = generate_mocks(spin=spin, spin2=spin2, cat_sizes=cat_size,
                                                                             mode="fixed",
                                                                             nsims=100,
                                                                             nell=nell,
                                                                             lmin = lmin,
                                                                             i_tomo=i,
                                                                            j_tomo=j,
                                                                             nside=nside,
                                                                             cl_extern=cl_extern,
                                                                             ell_extern=ell_extern,
                                                                             sel=sel,
                                                                             edges = edges)
    return [leff_fixed_auto, cl_mean_rnd_fixed_auto[0], np.diag(covmat_fixed_auto[0]), Nf_list, Nf_var]

def f2(i, j, cat_size, spin, spin2):
    leff_fixed_auto, cl_mean_rnd_fixed_auto, cl_prediction_fixed_auto, covmat_fixed_auto, analytic_cov_fixed_auto, field_variance_cov_fixed_auto,Nf_list, Nf_var, cls_auto  = generate_mocks(spin=spin, spin2=spin2, cat_sizes=cat_size,
                                                                             mode="field_variance",
                                                                             nsims=100,
                                                                             nell=nell,
                                                                             lmin = lmin,
                                                                             i_tomo=i,
                                                                            j_tomo=j,
                                                                             nside=nside,
                                                                             cl_extern=cl_extern,
                                                                             ell_extern=ell_extern,
                                                                             sel=sel,
                                                                             edges = edges)
    return [leff_fixed_auto, cl_mean_rnd_fixed_auto[0], np.diag(covmat_fixed_auto[0]), Nf_list, Nf_var]




import time


n_sources_list = np.array([1000, 3000, 5000], dtype=int)    
    
    
def g1(n_sources):
    n_sources_list = [n_sources]
    return f1(0, 0, n_sources_list, 0,0)

def g2(n_sources):
    n_sources_list = [n_sources]
    return f2(0, 0, n_sources_list, 0,0)

begin = time.time()
with concurrent.futures.ThreadPoolExecutor() as excuter:
    result1 = list(excuter.map(g1, n_sources_list))
    result2 = list(excuter.map(g2, n_sources_list))
    
for k in result1:
    result1_list.append(k)
    
for k in result2:
    result2_list.append(k)
        
end = time.time()

print(end - begin)


np.save('/home/s59efara_hpc/covariance/data_sets/varied_numbers_compare_fixed_field_wide_bins.npy', result1_list)
np.save('/home/s59efara_hpc/covariance/data_sets/varied_numbers_compare_field_variance_wide_bins.npy', result2_list)