import jax
import jax.numpy as jnp
import jax.lax as lax
import matplotlib.pyplot as plt
import plothist


from scipy.integrate import simpson
import numpy as np
import pymaster as nmt
import healpy as hp
import gc
import time
from tqdm import tqdm


from utils2 import *



def build_binning_matrix(edges, lmax):
    nbins = len(edges) - 1
    lmin = edges[0]
    B = np.zeros((nbins, lmax - lmin))
    ell_eff = np.zeros(nbins)
    for b in range(nbins):
        ell_min, ell_max = edges[b], edges[b + 1]
        ells_in_bin = np.arange(ell_min, ell_max)
        Δℓ = len(ells_in_bin)
        if Δℓ > 0:
            B[b, ells_in_bin - lmin] = 1.0 / Δℓ
            ell_eff[b] = np.mean(ells_in_bin)
    return B, ell_eff


def pad_binning_matrix(B, lmax_full, lmin=0):
    nbins, ncols = B.shape
    B_padded = np.zeros((nbins, lmin + ncols))
    B_padded[:, :ncols] = B
    return B_padded


deg = jnp.pi / 180.0

def cos_theta_pair(gl_i, gb_i, gl_j, gb_j):
    return (
        jnp.sin(gb_i) * jnp.sin(gb_j)
        + jnp.cos(gb_i) * jnp.cos(gb_j) * jnp.cos(gl_i - gl_j)
    )

# This funciton applies the operation from the ith index one-by-one to the jth index.
cos_theta_row = jax.vmap(
    cos_theta_pair,
    in_axes=(None, None, 0, 0)
)


# This function applies the 
cos_theta_matrix = jax.vmap(
    cos_theta_row,
    in_axes=(0, 0, None, None)
)

@jax.jit
def compute_cos_theta(gl, gb):
    return cos_theta_matrix(gl * deg, gb * deg,
                             gl * deg, gb * deg)
    
n_sources = [10000]
nside = 256
npix = hp.nside2npix(nside)
sel = np.ones(npix)
nside = 256
nell = 30
lmin = 2
lmax = 3*nside - 1
edges = np.unique(np.geomspace(lmin,3*nside - 1,nell).astype(int))
# edges = np.arange(lmin, 3*nside + 1)
cl_DMDM = np.load("/home/s59efara_hpc/covariance/test_cl_kappadm.npy")[:, 0 ,0]
alm = hp.synalm(cls=cl_DMDM, lmax = lmax)
map = hp.alm2map(alm, nside=nside, lmax=lmax)



variance_list = []

for n_samples in n_sources:
    print('sampling on a sphere')
    (gl ,gb),  _ = get_pos(n_samples, 'random', sel, nside)
    
    
    cos_theta = compute_cos_theta(gl, gb)
    
    
    
    
    print('calculating the mode coupling matrix')
    ipix = hp.ang2pix(nside, gl, gb, lonlat=True)
    w = np.ones_like(ipix)
    # beware of the binning scheme and the values to be added to the  
    b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
    f_vals = map[ipix] - np.mean(map[ipix])
    f_nmt = nmt.NmtFieldCatalog(positions = [gl, gb], weights=w, field = f_vals, lmax=b.lmax, lonlat=True)
    wasp = nmt.NmtWorkspace.from_fields(f_nmt, f_nmt, b)
    
    
    Sl_coupled = nmt.compute_coupled_cell(f_nmt, f_nmt) # the coupled, noise subtracted power spectrum 
    Nf = f_nmt.Nf
    Sl = wasp.decouple_cell(Sl_coupled)
    ells = b.get_effective_ells()
    
    
    
    mcm = jnp.asarray(wasp.get_coupling_matrix())
    mcm_inv = np.linalg.inv(mcm)
    var_f = np.var(f_vals)
    print('calculating Paij')
    # Recurrence: (l+1) P_{l+1} = (2l+1)x P_l - l P_{l-1}
    def body_fn(carry, l):
        P_lm1, P_l = carry
        P_lp1 = ((2*l + 1)*cos_theta*P_l - l*P_lm1) / (l + 1)
        return (P_l, P_lp1), P_lp1

    # Initialize P_0 = 1, P_1 = x
    carry_init = (jnp.ones_like(cos_theta), cos_theta)

    # Run scan from l=1 to Lmax-1
    _, P_all = lax.scan(body_fn, carry_init, jnp.arange(2, edges[-1]))
    P_all = jnp.concatenate([
        jnp.ones_like(cos_theta)[None, :],   # P_0
        cos_theta[None, :],                  # P_1
        P_all                        # P_2 ... P_Lmax
    ], axis=0)
    
    full_ells = jnp.arange(0, edges[-1])
    
    Sl_unbinned = b.unbin_cell(Sl)[0]
    signal_corr = jnp.einsum('l, lij->ij', (2*full_ells + 1) *Sl_unbinned, P_all)/4./np.pi
    
    unit_matrix = np.zeros_like(cos_theta)
    np.fill_diagonal(unit_matrix, 1)
    
    
    field_variance = jnp.sum((2*full_ells + 1)*Sl_unbinned)/4./np.pi
    noise_variance = var_f - field_variance
    
    
    
    Nw = np.sum(w*w)/4./np.pi
    Nl = Nw*noise_variance
    

    # noise_corr = jnp.einsum('l, lij->ij', (2*full_ells + 1) *Nl, P_all)/4./np.pi
    noise_corr = unit_matrix*noise_variance
    full_field_corr =  signal_corr + noise_corr
    del noise_corr, signal_corr
    
    
    w_i_j = w[None, :]*w[:, None]
    np.fill_diagonal(w_i_j, 0)
    
    
    Binning, ell_eff = build_binning_matrix(edges, edges[-1])
    Binning_matrix_padded = pad_binning_matrix(Binning, lmax, lmin)
    Binning_matrix_padded = jnp.array(Binning_matrix_padded)

    
    
    # intermediate_step = jnp.einsum('ij, aij->aij', w_i_j, P_all)
    print('summing terms')
    # direct_sum = 2*jnp.einsum('aij, bkm, ik, jm->ab', intermediate_step, intermediate_step, full_field_corr, full_field_corr)/(4*np.pi)**2
    
    
    direct_sum = 2*jnp.einsum('ij, aij, km, bkm, ik, jm->ab', w_i_j, P_all, w_i_j, P_all, full_field_corr, full_field_corr)/(4*np.pi)**2
    del P_all, full_field_corr
   

    # direct_sum = 2*jnp.einsum('ij, aij, km, bkm, ik, jm->ab', w_i_j, P_all, w_i_j, P_all, full_field_corr, full_field_corr)/(4*np.pi)**2
    print('Applying MCM')
    term1 = mcm_inv @ direct_sum @ mcm_inv.T
    del direct_sum
    print('trimming the array')
    # term1 = term1[lmin:, lmin:]
    print('applying the binning scheme')
    term1_binned = Binning_matrix_padded @ term1 @ Binning_matrix_padded.T
    del term1, Binning_matrix_padded, mcm_inv, w_i_j
    variance_list.append(jnp.diag(term1_binned))
    
    
    gc.collect()

np.save('/home/s59efara_hpc/covariance/data_sets/analytical_cov_10000.npy', variance_list)

data = np.load('/home/s59efara_hpc/covariance/data_sets/varied_numbers_compare_fixed_field_wide_bins_large_numbers.npy')

ells = b.get_effective_ells()



color = ['r', 'b', 'g']

# fig, ax = plt.subplots(figsize = (12, 8))

for i in range(len(n_sources)):
    ls, cls, sigma, Nf, _ = data[i]
    
    plt.loglog(ls, sigma, color = color[i], label = n_sources[i])
    plt.loglog(ells, variance_list[i], ls = '--', color = color[i])
    
    

plt.axhline(-1, ls = '--', color = 'black', label = 'analytical')
plt.axhline(-1, color = 'black', label = 'simulation')


plt.xlabel(r'$\ell$')
plt.ylabel(r'$Cov_{\ell \ell}$')
plt.legend()
plt.tight_layout()
plt.savefig('figs/full_covariance_plot_large_numbers.png', dpi = 300)
