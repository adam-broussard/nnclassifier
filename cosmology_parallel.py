
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import numpy as np
import os
import pandas as pd
from pandas import read_csv
from tqdm import tqdm
from astropy.io import fits
import astropy.units as u
from glob import glob
import filtersim
from sklearn.neighbors import KernelDensity
import pyccl as ccl
from astropy.coordinates import SkyCoord
import multiprocessing
# from cosmology import get_c_ells


def make_bins(rundir = './tpzruns/MatchCOSMOS2015/', bin_edges = np.arange(0, 1.61, 0.2), nnc_boundary = 0.8):

    df = read_csv(rundir + 'cosmo_info.dat')

    binned_info = df[['ra', 'dec', 'cosmos_photz', 'tpz_photz']][df.c_nnc >= nnc_boundary]
    bins = np.searchsorted(bin_edges, binned_info.tpz_photz)-1 # Gives the left edge of each object's assigned bin
    binned_info['zbin'] = bins

    return binned_info



def get_c_ells(obs = True, bin_edges = np.arange(0, 1.61, 0.2), nnc_boundary = 0.8, f_sky = 4.848e-5, ell = np.arange(2,2000)):

    cache_stub = './cache/cosmology/'

    if len(glob(cache_stub + f'{nnc_boundary}_{f_sky}*.dat')) == 3:
        ells = np.loadtxt(cache_stub + f'{nnc_boundary}_{f_sky}_ells.dat')
        C_ells = np.loadtxt(cache_stub + f'{nnc_boundary}_{f_sky}_C_ells.dat')
        bin_pairs = np.loadtxt(cache_stub + f'{nnc_boundary}_{f_sky}_binpairs.dat')

        return ells, C_ells, bin_pairs

    else:

        cat = make_bins(bin_edges = bin_edges, nnc_boundary = nnc_boundary)

        cosmo = ccl.Cosmology(Omega_c = 0.27, Omega_b = 0.045, h = 0.67, sigma8 = 0.83, n_s = 0.96)

        z = np.arange(min(bin_edges), max(bin_edges), 0.01)
        z_center = z[:-1] + (np.diff(z)/2.)

        C_ells = np.ones((len(bin_edges)-1, len(bin_edges)-1, len(ell)))*np.nan # All values initialized to nan

        for thisbin1 in tqdm(range(len(bin_edges)-1)):

            for thisbin2 in tqdm(range(thisbin1, len(bin_edges)-1)):

                if sum(cat.zbin == thisbin1) != 0 and sum(cat.zbin == thisbin2) != 0:

                    Nz1 = np.histogram(cat.cosmos_photz[cat.zbin == thisbin1], bins = z, density = True)[0]

                    cluster_tracer1 = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(z_center,Nz1), bias=(z_center,1.5*np.ones(len(z_center))))
                    if thisbin1 == thisbin2: 
                        cluster_tracer2 = cluster_tracer1
                    else:
                        # If the bins aren't the same, make a new tracer for the second bin
                        Nz2 = np.histogram(cat.cosmos_photz[cat.zbin == thisbin2], bins = z, density = True)[0]
                        cluster_tracer2 = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(z_center,Nz2), bias=(z_center,1.5*np.ones(len(z_center))))

                    C_ells[thisbin1][thisbin2] = ccl.angular_cl(cosmo, cluster_tracer1, cluster_tracer2, ell)

                    if thisbin1 == thisbin2 and obs:

                        # Add shot noise

                        n = np.sum(cat.zbin == thisbin1)/(4*np.pi*f_sky)

                        C_ells[thisbin1][thisbin2] += 1./n

        C_ells = C_ells.T.reshape(len(ell),-1) # Reorders the C_ell array so that axes are [ells, bin1, bin2] and then flattens it to [ells, binpair]
        hasdata = np.all(np.isfinite(C_ells.T), axis = 1)
        C_ells = C_ells.T[hasdata].T # Get rid of all bin pairs that have no data

        i, j = np.meshgrid(range(len(bin_edges)-1), range(len(bin_edges)-1))
        # bin_pairs = np.array(list(map(str,zip(i.flatten(), j.flatten()))))[hasdata]
        bin_pairs = np.array(list(zip(i.flatten(), j.flatten())))[hasdata]

        np.savetxt(cache_stub + f'{nnc_boundary}_{f_sky}_ells.dat', ell)
        np.savetxt(cache_stub + f'{nnc_boundary}_{f_sky}_C_ells.dat', C_ells)
        np.savetxt(cache_stub + f'{nnc_boundary}_{f_sky}_binpairs.dat', bin_pairs)

        return get_c_ells(obs = obs, bin_edges = bin_edges, nnc_boundary = nnc_boundary, f_sky = f_sky, ell = ell)




def set_entries(x, all_C_ells, all_ell, all_bin_pairs, delta_ell, f_sky):

    C_ell_1 = all_C_ells[x]
    ell_1 = all_ell[x]
    zbin_1 = all_bin_pairs[x]

    i, j = zbin_1
            
    i_ind = np.any(all_bin_pairs == i, axis = 1)
    j_ind = np.any(all_bin_pairs == j, axis = 1)

    ell_ind = (all_ell == ell_1)

    this_cov = np.zeros(len(all_C_ells))

    for y, (C_ell_2, ell_2, zbin_2) in enumerate(zip(all_C_ells[x:], all_ell[x:], all_bin_pairs[x:])):

        if ell_1 == ell_2:

            # If both bin numbers are equal (i.e., both 0), we can end up selecting any entry with a 0 in it
            # rather than just the ones with BOTH 0, so this takes care of that edge case.  Could probably be
            # written better...

            m, n = zbin_2
            m_ind = np.any(all_bin_pairs == m, axis = 1)
            n_ind = np.any(all_bin_pairs == n, axis = 1)

            if i == m:
                C_im = all_C_ells[(np.sum(all_bin_pairs == i, axis = 1) == 2) & ell_ind]
            else:
                C_im = all_C_ells[i_ind & m_ind & ell_ind]

            if i == n:
                C_in = all_C_ells[(np.sum(all_bin_pairs == i, axis = 1) == 2) & ell_ind]
            else:
                C_in = all_C_ells[i_ind & n_ind & ell_ind]

            if j == m:
                C_jm = all_C_ells[(np.sum(all_bin_pairs == j, axis = 1) == 2) & ell_ind]
            else:
                C_jm = all_C_ells[j_ind & m_ind & ell_ind]

            if j == n:
                C_jn = all_C_ells[(np.sum(all_bin_pairs == j, axis = 1) == 2) & ell_ind]
            else:
                C_jn = all_C_ells[j_ind & n_ind & ell_ind]

            this_cov[x+y] = ((2*ell_1 + 1) * delta_ell * f_sky)**-1 * ((C_im*C_jn) + (C_in*C_jm)) # From Takada and Jain 2009 (or 2004, nix the 2 in the numerator)
            # this_cov[x+y] = ((2*ell_1 + 1) * delta_ell * f_sky)**-1 * (C_im + C_jn) * (C_in + C_jm)

    return this_cov



def calculate_covariance_matrix_parallel(nnc_boundary = 0.8, delta_ell = 11, f_sky = 4.848e-5):

    if not delta_ell % 2:
        # Prevents the center of an ell bin from being a half
        raise Exception('delta_ell must be odd.')

    ell, C_ells, bin_pairs = get_c_ells(f_sky = f_sky, nnc_boundary = nnc_boundary)

    indices = np.arange(len(ell))

    if len(indices)%delta_ell:
        add_on = 1
    else:
        add_on = 0

    # effective_indices2 = np.arange(int(delta_ell/2.), len(ell), delta_ell)
    effective_indices = np.array_split(indices, int(len(indices)/delta_ell) + add_on)

    # effective_ells2 = ell[effective_indices2]
    # effective_C_ells2 = C_ells[effective_indices2]

    effective_ells = np.hstack([np.mean(ell[this_ind], axis = 0) for this_ind in effective_indices])
    effective_C_ells = np.vstack([np.mean(C_ells[this_ind], axis = 0) for this_ind in effective_indices])

    all_ell = np.hstack([effective_ells,] * effective_C_ells.shape[1])
    all_C_ells = effective_C_ells.ravel()
    all_bin_pairs = np.vstack([[this_bin_pair,] * effective_C_ells.shape[0] for this_bin_pair in bin_pairs])

    pool = multiprocessing.Pool()
    # cov_shared = multiprocessing.Array(ctypes.c_double, len(all_C_ells)**2)

    args = list(zip(range(len(all_C_ells)), [all_C_ells,]*len(all_C_ells), [all_ell,]*len(all_C_ells), [all_bin_pairs,]*len(all_C_ells), [delta_ell,]*len(all_C_ells), [f_sky,]*len(all_C_ells)))

    cov_rows = pool.starmap(set_entries, args, chunksize = 200)
    cov = np.vstack(cov_rows)
    cov = np.maximum(cov, cov.T)

    pool.close() # Stop additional processes from going to the pool and tell processes to exit when they're done
    pool.join() # # Wait for the exit signal

    return cov, all_ell, all_C_ells, all_bin_pairs