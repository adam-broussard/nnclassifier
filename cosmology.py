import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import numpy as np
from numpy import linalg
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

biasx = np.array([0.57, 0.68, 0.91, 1.26]) 
biasy = np.array([1.25, 1.35, 1.53, 1.85]) # Approximated from https://arxiv.org/pdf/1912.08209.pdf Fig 21


def make_cosmology_info_file(rundir = './tpzruns/MatchCOSMOS2015/'):

    sigmacut = 0.1

    cosmos = fits.open('./COSMOS2015/COSMOS2015_Laigle+_v1.1.fits')[1].data
    # spec = fits.open('./HSC/HSC_wide_clean_pdr2.fits')[1].data
    phot = fits.open('./HSC/HSC_wide_clean_pdr2_phot_depth_Unltd_COSMOS.fits')[1].data
    cosmos = cosmos[(cosmos.TYPE == 0) & (cosmos.ZP_2 < 0)]

    cosmos_cat = SkyCoord(ra = cosmos.ALPHA_J2000*u.degree, dec = cosmos.DELTA_J2000*u.degree)
    phot_cat = SkyCoord(ra = phot.ra*u.degree, dec = phot.dec * u.degree)

    idx, d2d, _ = cosmos_cat.match_to_catalog_sky(phot_cat) # Produces a list of length cosmos_cat of indices matching to phot_cat objects

    # Find the phot_cat objects that are closest to the cosmos_cat objects and only keep those

    unique_idx = []

    for this_match_idx in np.unique(idx):

        matches = np.where(idx == this_match_idx)[0] # Get the list of matching objects
        unique_idx.append(matches[np.argmin(d2d[matches])]) # Pick the one with the smallest distance

    unique_idx = np.array(unique_idx) # unique_idx contains the indices of COSMOS objects that have "correct" matches (closest to the spec catalog object)
    phot_idx = idx[unique_idx]

    cosmos_photz = cosmos.ZPDF[unique_idx] # ZPDF for median of the PDF, ZMINCHI2 for minimum chi2 (-99 for <3 bands)
    cosmos_photz_sigma = (cosmos.ZPDF_H68-cosmos.ZPDF_L68)[unique_idx] # ZPDF_H68-ZPDF_L68 for ZPDF; CHI2BEST for ZMINCHI2

    good_inds = (cosmos_photz >= 0) & (cosmos_photz_sigma < sigmacut) & (d2d.to('arcsecond').value[unique_idx] < 1) & (cosmos.ip_MAG_APER2[unique_idx] < 25.) & (cosmos.ZP_2[unique_idx] < 0) & (cosmos.NBFILT[unique_idx] >= 25)

    cosmos_photz = cosmos_photz[good_inds]
    phot_idx = phot_idx[good_inds]

    nnc_results = np.loadtxt(glob(rundir + '*/results_application.dat')[0])

    if len(nnc_results) != len(cosmos_photz):
        raise Exception(f'Mismatched numbers of objects between COSMOS2015 and NNC results.\n COSMOS contains {len(cosmos)} while NNC contains {len(nnc_results)}.')

    # ra = phot_cat.ra[phot_idx]
    # dec = phot_cat.dec[phot_idx]

    ra = cosmos.ALPHA_J2000[unique_idx[good_inds]]
    dec = cosmos.DELTA_J2000[unique_idx[good_inds]]
    hsc_id = phot.object_id[phot_idx]
    cosmos_id = cosmos.NUMBER[unique_idx[good_inds]]
    tpz_photz, tpz_zconf, tpz_zerr = np.loadtxt(rundir + 'output/results/tpzrun.1.mlz', usecols = [2,4,6], unpack = True) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

    # df = pd.DataFrame(data = [hsc_id, cosmos_id, ra, dec, nnc_results, cosmos_photz, tpz_photz, tpz_zconf, tpz_zerr], columns = ['hsc_id', 'cosmos_id', 'ra', 'dec', 'nnc_results', 'cosmos_photz', 'tpz_photz', 'tpz_zconf', 'tpz_zerr'])
    df = pd.DataFrame(data = {'hsc_id':hsc_id, 'cosmos_id':cosmos_id, 'ra':ra, 'dec':dec, 'c_nnc':nnc_results, 'cosmos_photz':cosmos_photz, 'tpz_photz':tpz_photz, 'tpz_zconf':tpz_zconf, 'tpz_zerr':tpz_zerr})
    df.to_csv(rundir + 'cosmo_info.dat', index=False, index_label=False)






def plot_n_z(rundir = './tpzruns/MatchCOSMOS2015/', bin_edges = np.linspace(0,1.5,9), nnc_boundary = 0., plot_bin_edges = False):

    cat = make_bins(rundir = rundir, bin_edges = bin_edges, nnc_boundary = nnc_boundary)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    if plot_bin_edges:

        for this_bin_edge in bin_edges:
            sp.plot([this_bin_edge,this_bin_edge], [0,1], color = '0.8', transform = sp.get_xaxis_transform())

    zbins = np.linspace(0,1.6,50)

    for x, this_bin_num in enumerate(sorted(np.unique(cat.zbin))):

        hist = np.histogram(cat.cosmos_photz[cat.zbin == this_bin_num], bins = zbins)[0]
        sp.step(zbins[:-1], hist, label = f'{x+1}')

    sp.text(0.02, 0.98, '$C_{TPZ} = ' + f'{nnc_boundary}$', fontsize = 18, ha = 'left', va = 'top', transform = sp.transAxes)

    sp.legend(loc = 'upper right')

    sp.set_xlabel('Redshfit (z)')
    sp.set_ylabel('N')




def plot_c_ells(obs = False, bin_edges = np.linspace(0,1.5,9), nnc_boundary = 0., f_sky = 4.848e-5):

    z = np.arange(min(bin_edges), max(bin_edges), 0.01)


    cat = make_bins(bin_edges = bin_edges, nnc_boundary = nnc_boundary)
    ells, C_ells, bin_pairs = get_c_ells(obs = obs, bin_edges = bin_edges, nnc_boundary = nnc_boundary, f_sky = f_sky)

    # bin_type = 'tophat'
    # sub_in = 3
    # cat = make_bins_diagnostic(bin_type = bin_type, sub_in = sub_in, bin_edges = bin_edges, nnc_boundary = nnc_boundary)
    # ells, C_ells, bin_pairs = get_c_ells_diagnostic(bin_type = bin_type, sub_in = sub_in, obs = obs, bin_edges = bin_edges, nnc_boundary = nnc_boundary, f_sky = f_sky)

    fig = plt.figure(figsize = (10,10))

    dim = 9

    for (x, y), this_C_ell in zip(bin_pairs, C_ells.T):

            this_sp = fig.add_subplot(dim, dim, x+1 + y*dim)

            this_sp.plot(ells, this_C_ell)

            this_sp.text(0.02, 0.02, r'$C_{\ell,' +  f'[{x}, {y}]' + '}$', ha = 'left', va = 'bottom', transform = this_sp.transAxes)

            this_sp.set_xscale('log')
            this_sp.set_yscale('log')

            this_sp.set_ylim(10**-8, 10**-2)

            if x != 0:
                this_sp.set_yticklabels([])
            if y != dim-1:
                this_sp.set_xticklabels([])

    gs = mpl.gridspec.GridSpec(nrows = 10*dim, ncols = 10*dim)
    if dim%2 == 1:
        gridloc = gs[0:int((dim-1)/2)*10 - 2,int((dim+1)/2)*10+2:int(dim)*10]
    else:
        gridloc = gs[0:int(dim/2),int(dim/2):dim]
    hist_sp = fig.add_subplot(gridloc)

    for this_bin in sorted(np.unique(bin_pairs.ravel())):

        hist_sp.hist(cat.cosmos_photz[cat.zbin == this_bin], histtype = 'step', bins = z, label = str(this_bin))

    hist_sp.text(0.02, 0.98, '$C_{TPZ} = ' + f'{nnc_boundary}$', fontsize = 12, ha = 'left', va = 'top', transform = hist_sp.transAxes)
    hist_sp.set_xlabel('Redshift (z)')
    hist_sp.set_ylabel('N')
    hist_sp.legend(loc = 'upper right', fontsize = 12)

    fig.text(0.5, 0.03, r'$\ell$', fontsize = 20)

    plt.subplots_adjust(wspace = 0, hspace = 0)




def make_bins_diagnostic(sub_in = None, nnc_boundary = 0., bin_edges = np.arange(0, 1.61, .2), bin_type = 'tophat', rundir = './tpzruns/MatchCOSMOS2015/'):

    np.random.seed(100)

    if bin_type == 'tophat':

        cosmos_photz = np.random.random(size = 150000)*1.6
        bins = np.searchsorted(bin_edges, cosmos_photz)-1
    
    elif bin_type == 'gaussian':

        bin_centers = bin_edges[:-1] + np.diff(bin_edges)
        cosmos_photz = np.hstack([np.random.normal(this_bin_center, scale = 0.15, size = 30000) for this_bin_center in bin_centers])
        bins = np.hstack([[bin_number,]*30000 for bin_number in range(len(bin_centers))])
    
    elif bin_type == 'overlapping_tophat':

        cosmos_photz = np.hstack([np.random.random(size = 25000)*1.6,]*(len(bin_edges)-1))
        bins = np.hstack([[bin_number,]*25000 for bin_number in range(len(bin_edges)-1)])

    if sub_in != None:

        df = read_csv(rundir + 'cosmo_info.dat')

        binned_info = df[['cosmos_photz', 'tpz_photz']][df.c_nnc >= nnc_boundary]
        remove_inds = bins == sub_in
        bins = np.delete(bins, remove_inds)
        cosmos_photz = np.delete(cosmos_photz, remove_inds)

        databins = np.searchsorted(bin_edges, binned_info.tpz_photz)-1 # Gives the left edge of each object's assigned bin
        cosmos_photz = np.append(cosmos_photz, binned_info.cosmos_photz[databins == sub_in])
        bins = np.append(bins, databins[databins == sub_in])

    return pd.DataFrame(data = {'cosmos_photz':cosmos_photz, 'zbin':bins})




def make_bins(rundir = './tpzruns/MatchCOSMOS2015/', bin_edges = np.linspace(0,1.5,9), nnc_boundary = 0.):

    df = read_csv(rundir + 'cosmo_info.dat')

    binned_info = df[['cosmos_photz', 'tpz_photz']][df.c_nnc >= nnc_boundary]
    bins = np.searchsorted(bin_edges, binned_info.tpz_photz)-1 # Gives the left edge of each object's assigned bin
    binned_info['zbin'] = bins

    return binned_info


def make_bins_optimized(rundir = './tpzruns/MatchCOSMOS2015/', bin_edges = np.linspace(0,1.5,9), nnc_boundary = 0.):

    pass



def calculate_contamination_fraction(rundir = './tpzruns/MatchCOSMOS2015/', bin_edges = np.linspace(0,1.5,9), nnc_boundary = 0.):

    cat = make_bins(rundir = rundir, bin_edges = bin_edges, nnc_boundary = nnc_boundary)

    good_galaxies = 0

    for bin_number, bin_lo, bin_hi in zip(np.arange(len(bin_edges)-1), bin_edges[:-1], bin_edges[1:]):

        good_galaxies += np.sum((cat.cosmos_photz > bin_lo) & (cat.cosmos_photz <= bin_hi) & (cat.zbin == bin_number))

    contam_frac = 1-(good_galaxies/float(len(cat)))

    return contam_frac





def get_c_ells_diagnostic(bin_type = 'tophat', sub_in = None, nnc_boundary = 0., obs = True, f_sky = 4.848e-5, ells = np.arange(2,2000), bin_edges = np.linspace(0,1.5,9) ):

    cat = make_bins_diagnostic(sub_in = sub_in, nnc_boundary = nnc_boundary, bin_edges = bin_edges, bin_type = bin_type)

    cosmo = ccl.Cosmology(Omega_c = 0.27, Omega_b = 0.045, h = 0.67, sigma8 = 0.83, n_s = 0.96)

    z = np.arange(min(bin_edges), max(bin_edges), 0.01)
    z_center = z[:-1] + (np.diff(z)/2.)

    C_ells = np.ones((len(bin_edges)-1, len(bin_edges)-1, len(ells)))*np.nan # All values initialized to nan

    for thisbin1 in tqdm(range(len(bin_edges)-1)):

        for thisbin2 in tqdm(range(thisbin1, len(bin_edges)-1)):

            if sum(cat.zbin == thisbin1) != 0 and sum(cat.zbin == thisbin2) != 0:

                Nz1 = np.histogram(cat.cosmos_photz[cat.zbin == thisbin1], bins = z, density = True)[0]

                cluster_tracer1 = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_center,Nz1), bias=(z_center,1 + 0.84*z_center)) # Bias comes from Nicola 2020 Fig 21 https://arxiv.org/pdf/1912.08209.pdf
                if thisbin1 == thisbin2: 
                    cluster_tracer2 = cluster_tracer1
                else:
                    # If the bins aren't the same, make a new tracer for the second bin
                    Nz2 = np.histogram(cat.cosmos_photz[cat.zbin == thisbin2], bins = z, density = True)[0]
                    cluster_tracer2 = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_center,Nz2), bias=(z_center,1 + 0.84*z_center)) # Bias comes from Nicola 2020 Fig 21 https://arxiv.org/pdf/1912.08209.pdf

                C_ells[thisbin1][thisbin2] = ccl.angular_cl(cosmo, cluster_tracer1, cluster_tracer2, ells)

                if thisbin1 == thisbin2 and obs:

                    # Add shot noise

                    n = np.sum(cat.zbin == thisbin1)/(4*np.pi*f_sky)

                    C_ells[thisbin1][thisbin2] += 1./n

    C_ells = C_ells.T.reshape(len(ells),-1) # Reorders the C_ell array so that axes are [ells, bin1, bin2] and then flattens it to [ells, binpair]
    hasdata = np.all(np.isfinite(C_ells.T), axis = 1)
    C_ells = C_ells.T[hasdata].T # Get rid of all bin pairs that have no data

    i, j = np.meshgrid(range(len(bin_edges)-1), range(len(bin_edges)-1))
    # bin_pairs = np.array(list(map(str,zip(i.flatten(), j.flatten()))))[hasdata]
    bin_pairs = np.array(list(zip(i.flatten(), j.flatten())))[hasdata]


    return ells.astype(int), C_ells, bin_pairs.astype(int)






def get_c_ells(obs = True, bin_edges = np.linspace(0,1.5,9), nnc_boundary = 0., f_sky = 4.848e-5, ell = np.arange(2,2000), force_compute = False):

    cache_stub = './cache/cosmology/'

    if obs:
        obs_tag = 'obs'
    else:
        obs_tag = 'noobs'

    if len(glob(cache_stub + f'{nnc_boundary:.3f}_{len(bin_edges)-1}_{f_sky}_{obs_tag}*.dat')) == 3 and not force_compute:
        ells = np.loadtxt(cache_stub + f'{nnc_boundary:.3f}_{len(bin_edges)-1}_{f_sky}_{obs_tag}_ells.dat')
        C_ells = np.loadtxt(cache_stub + f'{nnc_boundary:.3f}_{len(bin_edges)-1}_{f_sky}_{obs_tag}_C_ells.dat')
        bin_pairs = np.loadtxt(cache_stub + f'{nnc_boundary:.3f}_{len(bin_edges)-1}_{f_sky}_{obs_tag}_binpairs.dat')

        return ells.astype(int), C_ells, bin_pairs.astype(int)

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

                    cluster_tracer1 = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(z_center,Nz1), bias=(z_center,1 + 0.84*z_center)) # Bias comes from Nicola 2020 Fig 21 https://arxiv.org/pdf/1912.08209.pdf
                    if thisbin1 == thisbin2: 
                        cluster_tracer2 = cluster_tracer1
                    else:
                        # If the bins aren't the same, make a new tracer for the second bin
                        Nz2 = np.histogram(cat.cosmos_photz[cat.zbin == thisbin2], bins = z, density = True)[0]
                        cluster_tracer2 = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(z_center,Nz2), bias=(z_center,1 + 0.84*z_center)) # Bias comes from Nicola 2020 Fig 21 https://arxiv.org/pdf/1912.08209.pdf

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

        np.savetxt(cache_stub + f'{nnc_boundary:.3f}_{len(bin_edges)-1}_{f_sky}_{obs_tag}_ells.dat', ell)
        np.savetxt(cache_stub + f'{nnc_boundary:.3f}_{len(bin_edges)-1}_{f_sky}_{obs_tag}_C_ells.dat', C_ells)
        np.savetxt(cache_stub + f'{nnc_boundary:.3f}_{len(bin_edges)-1}_{f_sky}_{obs_tag}_binpairs.dat', bin_pairs)

        return get_c_ells(obs = obs, bin_edges = bin_edges, nnc_boundary = nnc_boundary, f_sky = f_sky, ell = ell)






def calculate_covariance_matrix_parallel_diagnostic(bin_type = 'tophat', sub_in = None, nnc_boundary = 0., delta_ell = 11, f_sky = 4.848e-5):

    if not delta_ell % 2:
        # Prevents the center of an ell bin from being a half
        raise Exception('delta_ell must be odd.')

    effective_ells, effective_C_ells, effective_bin_pairs = build_data_vector_diagnostic(bin_type = bin_type, sub_in = sub_in, nnc_boundary = nnc_boundary, delta_ell = delta_ell, f_sky=f_sky, obs = True)

    pool = multiprocessing.Pool()
    # cov_shared = multiprocessing.Array(ctypes.c_double, len(all_C_ells)**2)

    args = list(zip(range(len(effective_C_ells)), [effective_C_ells,]*len(effective_C_ells), [effective_ells,]*len(effective_C_ells), [effective_bin_pairs,]*len(effective_C_ells), [delta_ell,]*len(effective_C_ells), [f_sky,]*len(effective_C_ells)))

    cov_rows = pool.starmap(set_entries, args, chunksize = 200)
    cov = np.vstack(cov_rows)
    cov = np.maximum(cov, cov.T)

    pool.close() # Stop additional processes from going to the pool and tell processes to exit when they're done
    pool.join() # # Wait for the exit signal

    return cov






def calculate_covariance_matrix(nnc_boundary = 0., delta_ell = 11, f_sky = 4.848e-5):

    ell, C_ells, bin_pairs = get_c_ells(f_sky = f_sky, nnc_boundary = nnc_boundary)

    effective_indices = np.arange(int(delta_ell/2.), len(ell), delta_ell)

    effective_ells = ell[effective_indices]
    effective_C_ells = C_ells[effective_indices]

    all_ells = np.hstack([effective_ells,] * effective_C_ells.shape[1])
    all_C_ells = effective_C_ells.ravel()
    all_bin_pairs = np.vstack([[this_bin_pair,] * effective_C_ells.shape[0] for this_bin_pair in bin_pairs])

    # cov = np.ones((len(all_C_ells), len(all_C_ells)))*np.nan # All values initialized to nan

    # cov = lil_matrix((len(all_C_ells), len(all_C_ells)), dtype = np.float32)

    cov = np.zeros((len(all_C_ells), len(all_C_ells)))

    for x, (C_ell_1, ell_1, zbin_1) in enumerate(tqdm(zip(all_C_ells, all_ells, all_bin_pairs), total = len(all_C_ells))):
        i, j = zbin_1
                
        i_ind = np.any(all_bin_pairs == i, axis = 1)
        j_ind = np.any(all_bin_pairs == j, axis = 1)
        
        ell_ind = (all_ells == ell_1)

        for y, (C_ell_2, ell_2, zbin_2) in enumerate(zip(all_C_ells[x:], all_ells[x:], all_bin_pairs[x:])):

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

                cov[x,x+y] = ((2*ell_1 + 1) * delta_ell * f_sky)**-1 * ((C_im*C_jn) + (C_in*C_jm)) # From Takada and Jain 2009 (or 2004, nix the 2 in the numerator)

    cov = np.maximum(cov, cov.T)

    return cov, all_ells, all_C_ells, all_bin_pairs



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




def build_data_vector_diagnostic(bin_type = 'tophat', sub_in = None, nnc_boundary = 0., delta_ell = 11, f_sky = 4.848e-5, obs = True):

    ell, C_ells, bin_pairs = get_c_ells_diagnostic(bin_type = bin_type, sub_in = sub_in, f_sky = f_sky, nnc_boundary = nnc_boundary, obs = obs)

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

    all_ell = np.hstack([[this_ell,] * effective_C_ells.shape[1] for this_ell in effective_ells])
    all_C_ells = effective_C_ells.ravel()
    all_bin_pairs = np.vstack([bin_pairs,] * effective_C_ells.shape[0])

    return all_ell, all_C_ells, all_bin_pairs




def build_data_vector(nnc_boundary = 0., bin_edges = np.linspace(0,1.5,9), delta_ell = 11, f_sky = 4.848e-5, obs = True):

    ell, C_ells, bin_pairs = get_c_ells(f_sky = f_sky, bin_edges = bin_edges, nnc_boundary = nnc_boundary, obs = obs)

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

    all_ell = np.hstack([[this_ell,] * effective_C_ells.shape[1] for this_ell in effective_ells])
    all_C_ells = effective_C_ells.ravel()
    all_bin_pairs = np.vstack([bin_pairs,] * effective_C_ells.shape[0])

    return all_ell, all_C_ells, all_bin_pairs





def calculate_covariance_matrix_parallel(nnc_boundary = 0., bin_edges = np.linspace(0,1.5,9), delta_ell = 11, f_sky = 4.848e-5):

    if not delta_ell % 2:
        # Prevents the center of an ell bin from being a half
        raise Exception('delta_ell must be odd.')

    effective_ells, effective_C_ells, effective_bin_pairs = build_data_vector(nnc_boundary = nnc_boundary, bin_edges = bin_edges, delta_ell = delta_ell, f_sky=f_sky, obs = True)

    pool = multiprocessing.Pool()
    # cov_shared = multiprocessing.Array(ctypes.c_double, len(all_C_ells)**2)

    args = list(zip(range(len(effective_C_ells)), [effective_C_ells,]*len(effective_C_ells), [effective_ells,]*len(effective_C_ells), [effective_bin_pairs,]*len(effective_C_ells), [delta_ell,]*len(effective_C_ells), [f_sky,]*len(effective_C_ells)))

    cov_rows = pool.starmap(set_entries, args, chunksize = 200)
    cov = np.vstack(cov_rows)
    cov = np.maximum(cov, cov.T)

    pool.close() # Stop additional processes from going to the pool and tell processes to exit when they're done
    pool.join() # # Wait for the exit signal

    return cov




def calculate_snr_diagnostic(bin_type = 'tophat', sub_in = None, nnc_boundary = 0., bin_edges = np.linspace(0,1.5,9), delta_ell = 11, f_sky = 4.848e-5):

    cov = calculate_covariance_matrix_parallel_diagnostic(bin_type = bin_type, sub_in = sub_in, nnc_boundary = nnc_boundary, bin_edges = bin_edges, delta_ell=delta_ell, f_sky = f_sky)
    effective_ells, effective_C_ells, effective_bin_pairs = build_data_vector_diagnostic(bin_type = bin_type, sub_in = sub_in, nnc_boundary = nnc_boundary, bin_edges = bin_edges, delta_ell = delta_ell, f_sky=f_sky, obs = False)

    try:
        return (effective_C_ells @ np.linalg.inv(cov)) @ effective_C_ells
    except:
        # print(f'Covariance matrix with nnc_boundary {nnc_boundary}, delta_ell {delta_ell}, and f_sky {f_sky} is singluar.  Returning nan.')
        return np.nan





def calculate_snr(nnc_boundary = 0., bin_edges = np.linspace(0,1.5,9), delta_ell = 11, f_sky = 4.848e-5, parallel = True):

    if parallel:
        cov = calculate_covariance_matrix_parallel(nnc_boundary = nnc_boundary, bin_edges = bin_edges, delta_ell=delta_ell, f_sky = f_sky)
        effective_ells, effective_C_ells, effective_bin_pairs = build_data_vector(nnc_boundary = nnc_boundary, bin_edges = bin_edges, delta_ell = delta_ell, f_sky=f_sky, obs = False)
    else:
        cov = calculate_covariance_matrix(nnc_boundary = nnc_boundary, bin_edges = bin_edges, delta_ell=delta_ell, f_sky = f_sky)
        effective_ells, effective_C_ells, effective_bin_pairs = build_data_vector(nnc_boundary = nnc_boundary, bin_edges = bin_edges, delta_ell = delta_ell, f_sky=f_sky, obs = False)
    try:
        return np.sqrt((effective_C_ells @ np.linalg.inv(cov)) @ effective_C_ells)
    except:
        # print(f'Covariance matrix with nnc_boundary {nnc_boundary}, delta_ell {delta_ell}, and f_sky {f_sky} is singluar.  Returning nan.')
        return np.nan





def plot_contam_frac(nnc_boundaries = np.arange(0,1,0.01), nbins = np.arange(4,13)):

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    cmap = mpl.cm.get_cmap('turbo')
    norm = mpl.colors.Normalize(vmin=min(nbins), vmax=max(nbins))

    for this_bin_num in tqdm(nbins):

        contam_frac_list = []

        for this_nnc_boundary in tqdm(nnc_boundaries):

            contam_frac_list.append(calculate_contamination_fraction(nnc_boundary = this_nnc_boundary, bin_edges = np.linspace(0,1.5,this_bin_num)))

        sp.plot(nnc_boundaries, contam_frac_list, c = 'k', linewidth = 3, zorder = 1)
        sp.plot(nnc_boundaries, contam_frac_list, label = f'{this_bin_num}', c = cmap(norm(this_bin_num)), linewidth = 2, zorder = 2)

    sp.set_xlabel('$C_{NNC}$')
    sp.set_ylabel('Contamination Fraction')

    sp.set_ylim(0,0.6)

    sp.legend(fontsize = 16)






def plot_snr_contam_frac(require_contam_frac = 0.3, nnc_boundaries = np.arange(0,1,.01), force_compute = False):

    cache_file = './cache/plot_snr_contam_frac.dat'

    if not os.path.isfile(cache_file) or force_compute:

        nbins_list = []

        for this_nnc_boundary in tqdm(nnc_boundaries):

            nbins = 4
            next_nbins = 4
            contam_frac = calculate_contamination_fraction(bin_edges = np.linspace(0,1.5,4), nnc_boundary = this_nnc_boundary)
            next_contam_frac = 0.
            while next_contam_frac < require_contam_frac and next_nbins <= 12:
                
                # If the contamination fraction from adding a bin is still less than the threshold, 
                # update the previous contamination fraction and number of bins and step forward

                contam_frac = next_contam_frac
                nbins = next_nbins

                next_nbins = nbins + 1
                next_contam_frac = calculate_contamination_fraction(bin_edges = np.linspace(0,1.5,next_nbins), nnc_boundary = this_nnc_boundary)

            nbins_list.append(nbins)

        nbins_list = np.array(nbins_list)

        plot_inds = np.zeros(len(nbins_list), dtype = bool)

        for this_bin_num in np.unique(nbins_list):

            this_ind = np.where(nbins_list == this_bin_num)[0]

            plot_inds[this_ind[::2]] = True
            plot_inds[this_ind[-1]] = True

        snr = []

        for this_nbins, this_nnc_boundary in tqdm(zip(nbins_list[plot_inds], nnc_boundaries[plot_inds]), total = sum(plot_inds)):

            if this_nbins > 16:
                delta_ell = 31
            elif this_nbins > 8:
                delta_ell = 21
            else:
                delta_ell = 11

            try:
                snr.append(calculate_snr(nnc_boundary = this_nnc_boundary, bin_edges = np.linspace(0,1.5,this_nbins), delta_ell = delta_ell))
            except:
                snr.append(np.nan)

        snr = np.array(snr)


        xvals = []
        yvals = []
        bin_num = []

        xvals.append(nnc_boundaries[0])
        yvals.append(snr[0])
        bin_num.append(nbins_list[0])

        for x, this_bin_num in enumerate(np.unique(nbins_list[plot_inds])):

            bin_inds = nbins_list[plot_inds] == this_bin_num
            bin_num.append(this_bin_num)
            if np.sum(bin_inds) > 1:
                point_ind = np.argmax(snr[bin_inds])
                xvals.append(nnc_boundaries[plot_inds][bin_inds][point_ind])
                yvals.append(snr[bin_inds][point_ind])
            else:
                xvals.append(nnc_boundaries[plot_inds][bin_inds])
                yvals.append(snr[bin_inds])

        breakpoint()

        xvals.append(nnc_boundaries[plot_inds][np.where(np.isfinite(snr))[0][-1]])
        yvals.append(snr[np.where(np.isfinite(snr))[0][-1]])
        bin_num.append(nbins_list[plot_inds][np.where(np.isfinite(snr))[0][-1]])

        np.savetxt(cache_file, np.vstack((xvals, yvals, bin_num)).T)

    else:

        xvals, yvals, bin_num = np.loadtxt(cache_file, unpack = True)
        bin_num = bin_num.astype(int)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    points = sp.scatter(xvals[1:-1], yvals[1:-1], c = bin_num[1:-1], s = 75, cmap = 'turbo', linewidth = 1, ec = 'k')
    points2 = sp.scatter(xvals[[0,-1]], yvals[[0,-1]], c = 'k', s = 75, marker = 'D', linewidth = 1, ec = 'k')

    fig.canvas.draw()
    colors = points.get_facecolors()
    # colors2 = points2.get_facecolors()

    # sp.text(xvals[0], yvals[0]-.75, f'{bin_num[0]}', ha = 'left' , va = 'top', fontsize = 18, c = colors2[0], alpha = 0.5)
    # sp.text(xvals[-1], yvals[-1]-.75, f'{bin_num[-1]}', ha = 'right', va = 'top', fontsize = 18, c = colors2[1], alpha = 0.5)

    for thisx, thisy, this_bin_num, this_color in zip(xvals[1:-1], yvals[1:-1], bin_num[1:-1], colors):

        if this_bin_num == 11:
            sp.text(thisx, thisy+.75, f'{this_bin_num}', ha = 'center', va = 'bottom', fontsize = 18, c = this_color)
        else:
            sp.text(thisx, thisy-.75, f'{this_bin_num}', ha = 'center', va = 'top', fontsize = 18, c = this_color)

    sp_extent = sp.get_position().extents
    cbar_ax = fig.add_axes([sp_extent[-2], sp_extent[1], 0.03, sp_extent[-1] - sp_extent[1]])
    fig.colorbar(points, cax = cbar_ax)
    cbar_ax.set_ylabel('$N_{Bins}$')

    # For plotting lines

    # for x, this_bin_num in enumerate(np.unique(nbins_list[plot_inds])):

    #     bin_inds = nbins_list[plot_inds] == this_bin_num
    #     if np.sum(bin_inds) > 1:
    #         sp.plot(nnc_boundaries[plot_inds][bin_inds], snr[bin_inds], label = f'{this_bin_num} Bins')
    #     else:
    #         sp.scatter(nnc_boundaries[plot_inds][bin_inds], snr[bin_inds], color = f'C{x}', label = f'{this_bin_num} Bins')

    # for this_bin_num in np.unique(nbins_list[plot_inds])[:-1]:

    #     ind1 = np.where(nbins_list[plot_inds] == this_bin_num)[0][-1]
    #     ind2 = ind1 + 1
    #     nncb1 = nnc_boundaries[plot_inds][ind1]
    #     nncb2 = nnc_boundaries[plot_inds][ind2]
        
    #     sp.plot([nncb1, nncb2], [snr[ind1], snr[ind2]], color = 'k')


    # sp.plot(nnc_boundaries[plot_inds], snr)
    sp.set_xlabel('$C_{NNC}$')
    sp.set_ylabel('SNR')

    sp.text(0.02, 0.02, f'Contamination Fraction $ < {require_contam_frac}$', fontsize = 20, ha = 'left', va = 'bottom', transform = sp.transAxes)

    sp.set_xlim(-0.05,1.05)
    sp.set_ylim(0,22)











