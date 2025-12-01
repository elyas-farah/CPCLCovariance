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

nside_mask = 512

npix = hp.nside2npix(nside_mask)

sel = np.ones(npix)

# theta, phi = hp.pix2ang(nside_mask, np.arange(npix))

# sel = np.sin(theta)

# sel[theta > np.pi / 2] = 0
# sel[phi > np.pi/3*2 ] = 0
cl_extern = np.load("/home/s59efara_hpc/covariance/test_cl_kappadm.npy")
ell_extern = np.arange(len(cl_extern[:,0,0]))






nside = 512
nell = 15
lmin = 20

# cls_prediction_list = []
# cls_fixed_field_list = []
# cls_field_variance_list = []

# var_prediction_list = []
# var_fixed_field_list = []
# var_field_variance_list = []

result1_list = []
result2_list = []




def f1(i, j, cat_size, spin, spin2):
    leff_variance_auto, cl_mean_rnd_variance_auto, cl_prediction_variance_auto, covmat_variance_auto, analytic_cov_variance_auto, field_variance_cov_variance_auto, cls_auto = generate_mocks(spin=spin, spin2=spin2, cat_sizes=cat_size,
                                                                             mode="field_variance",
                                                                             nsims=100,
                                                                             nell=nell,
                                                                             lmin = lmin, 
                                                                            i_tomo=i,
                                                                            j_tomo=j,
                                                                             nside=nside,
                                                                             cl_extern=cl_extern,
                                                                             ell_extern=ell_extern,
                                                                             sel=sel)
    return [cl_prediction_variance_auto[0], cl_mean_rnd_variance_auto[0], np.diag(covmat_variance_auto[0]), analytic_cov_variance_auto[0]]


def f2(i, j, cat_size, spin, spin2):
    leff_fixed_auto, cl_mean_rnd_fixed_auto, cl_prediction_fixed_auto, covmat_fixed_auto, analytic_cov_fixed_auto, field_variance_cov_fixed_auto,cls_auto  = generate_mocks(spin=spin, spin2=spin2, cat_sizes=cat_size,
                                                                             mode="fixed",
                                                                             nsims=100,
                                                                             nell=nell,
                                                                             lmin = lmin,
                                                                             i_tomo=i,
                                                                            j_tomo=j,
                                                                             nside=nside,
                                                                             cl_extern=cl_extern,
                                                                             ell_extern=ell_extern,
                                                                             sel=sel)
    return [cl_mean_rnd_fixed_auto[0], np.diag(covmat_fixed_auto[0])]
import time


    
    
    
def g1(bin_str):
    i, j = int(bin_str[0]), int(bin_str[1])
    if i == j and i == 0:
        spin = 0
        spin2 = None
        cat_size = [int(5e4)]
    elif i == j and i > 0:
        spin=2
        spin2 = None
        cat_size = [int(1e6)]
    elif i > 0 and j > 0:
        spin = 2
        spin2 = 2
        cat_size = [[int(5e4), int(1e6)]]
    elif i==0 and j>0:
        spin = 0
        spin2 = 2
        cat_size = [[int(5e4), int(1e6)]]
    
    elif j==0 and i>0:
        spin = 2
        spin2 = 0
        cat_size = [[int(1e6), int(5e4)]]

    # print(spin, spin2)   
    # print(i, j, cat_size)
    
    return f1(i, j, cat_size, spin, spin2)


def g2(bin_str):
    i, j = int(bin_str[0]), int(bin_str[1])
    
    if i == j and i == 0:
        spin = 0
        spin2 = None
        cat_size = [int(5e4)]
    elif i == j and i > 0:
        spin=2
        spin2 = None
        cat_size = [int(1e6)]
    elif i > 0 and j > 0:
        spin = 2
        spin2 = 2
        cat_size = [[int(5e4), int(1e6)]]
    elif i==0 and j>0:
        spin = 0
        spin2 = 2
        cat_size = [[int(5e4), int(1e6)]]
    
    elif j==0 and i>0:
        spin = 2
        spin2 = 0
        cat_size = [[int(1e6), int(5e4)]]
    # print(i, j, cat_size)
    
    
    return f2(i, j, cat_size, spin, spin2) 


bins_list = []    
for i in range(4):
    for j in range(i + 1):
        # print(i, j)
        bins_list.append(f"{i}{j}")


begin = time.time()
with concurrent.futures.ThreadPoolExecutor() as excuter:
    result1 = list(excuter.map(g1, bins_list))
    result2 = list(excuter.map(g2, bins_list))
    
for k in result1:
    result1_list.append(k)
for k in result2:
    result2_list.append(k)
        
end = time.time()

print(end - begin)


np.save('/home/s59efara_hpc/covariance/data_sets/variance_pred_results_lensing1.npy', result1_list)
np.save('/home/s59efara_hpc/covariance/data_sets/fixed_field_results_lensing1.npy', result2_list)    
    
    
''' for j in range(i + 1):
        print(i, j, 'done')
        leff_variance_auto, cl_mean_rnd_variance_auto, cl_prediction_variance_auto, covmat_variance_auto, analytic_cov_variance_auto, field_variance_cov_variance_auto, cls_auto = generate_mocks(cat_sizes=cat_size,
                                                                             mode="field_variance",
                                                                             nsims=100,
                                                                             nell=nell,
                                                                             lmin = lmin, 
                                                                            i_tomo=i,
                                                                            j_tomo=j,
                                                                             nside=nside,
                                                                             cl_extern=cl_extern,
                                                                             ell_extern=ell_extern,
                                                                             sel=sel)
        
        
        leff_fixed_auto, cl_mean_rnd_fixed_auto, cl_prediction_fixed_auto, covmat_fixed_auto, analytic_cov_fixed_auto, field_variance_cov_fixed_auto,cls_auto  = generate_mocks(cat_sizes=cat_size,
                                                                             mode="fixed",
                                                                             nsims=100,
                                                                             nell=nell,
                                                                             lmin = lmin,
                                                                             i_tomo=i,
                                                                            j_tomo=j,
                                                                             nside=nside,
                                                                             cl_extern=cl_extern,
                                                                             ell_extern=ell_extern,
                                                                             sel=sel)
        leff_random_auto, cl_mean_rnd_random_auto, cl_prediction_random_auto, covmat_random_auto, analytic_cov_random_auto, field_variance_cov_random_auto,cls_auto = generate_mocks(cat_sizes=cat_size,
                                                                             mode="random",
                                                                             nsims=100,
                                                                             i_tomo=i,
                                                                            j_tomo=j,
                                                                             exponent=1,
                                                                             nside=nside)
        cls_prediction_list.append(cl_prediction_variance_auto[0])
        cls_fixed_field_list.append(cl_mean_rnd_variance_auto[0])
        cls_field_variance_list.append(cl_mean_rnd_fixed_auto[0])

        var_prediction_list.append(np.diag(analytic_cov_fixed_auto[0]))
        var_fixed_field_list.append(np.diag(covmat_fixed_auto[0]))
        var_field_variance_list.append(np.diag(covmat_variance_auto[0]))
        gc.collect()

np.save('cls_pred.npy', cls_prediction_list)
np.save('cls_fixed_field.npy', cls_fixed_field_list)
np.save('cls_field_variance.npy', cls_field_variance_list)

np.save('sigma_pred.npy', var_prediction_list)
np.save('sigma_fixed_field.npy', var_fixed_field_list)
np.save('sigma_field_variance.npy', var_field_variance_list)

np.save('ells.npy', leff_fixed_auto)
'''