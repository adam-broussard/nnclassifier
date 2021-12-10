import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from pandas import read_csv
import matplotlib.patheffects as pe
from glob import glob
from astropy.io import fits
import filtersim
import os
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


def read_feature_file(fname):

    with open(fname, 'r') as readfile:
        column_names = [thisname for thisname in readfile.readline()[:-1].split(' ') if (thisname != '#') and (thisname != '')]

    trainfile = read_csv(fname, header = None, comment = '#', delimiter = '\s+', names = column_names)

    return trainfile


def read_tpz_output(fname):

    return read_csv(fname, header = None, comment = '#', delimiter = '\s+', names = ['ztrue', 'zmode0', 'zmean1', 'zConf0', 'zConf1', 'err0', 'err1'])



def plot_residual(run_dir1 = './tpzruns/SpecCOSMOS2015/', run_dir2 = './tpzruns/MatchCOSMOS2015/', nn_identifier = 'nnc_', zmin = 0, zmax = 1.5):
    
    if run_dir1[-1] != '/':
        run_dir1 += '/'

    nnc_folders1 = glob(run_dir1 + '*' + nn_identifier + '*/')

    if len(nnc_folders1) == 1:
        nnc_folder1 = nnc_folders1[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')
        for thisfolder in nnc_folders1:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    # nnr_folder1 = nnc_folder1.replace('nnc', 'nnr').replace('_corr', '')

    # features_file = read_csv(glob(run_dir + '*.nnc_test')[0], delimiter = '\s+', comment = '#') 
    # nnc_results = np.loadtxt(nnc_folder + 'results_test.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    features_file1 = read_csv(glob(run_dir1 + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results1 = np.loadtxt(nnc_folder1 + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    # nnr_results1 = np.loadtxt(nnr_folder1 + 'results_application.dat')

    zspec1 = features_file1['specz'].to_numpy()
    zphot1 = features_file1['zphot'].to_numpy()
    zphot_err1 = features_file1['zerr'].to_numpy()
    zphot_conf1 = features_file1['zconf'].to_numpy()

    residual1 = ((zphot1 - zspec1)/(1+zspec1))[(zspec1 > zmin) & (zspec1 < zmax)]


    if run_dir2[-1] != '/':
        run_dir2 += '/'

    nnc_folders2 = glob(run_dir2 + '*' + nn_identifier + '*/')

    if len(nnc_folders2) == 1:
        nnc_folder2 = nnc_folders2[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')
        for thisfolder in nnc_folders2:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    # features_file = read_csv(glob(run_dir + '*.nnc_test')[0], delimiter = '\s+', comment = '#') 
    # nnc_results = np.loadtxt(nnc_folder + 'results_test.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    features_file2 = read_csv(glob(run_dir2 + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results1 = np.loadtxt(nnc_folder2 + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    # nnr_results1 = np.loadtxt(nnr_folder2 + 'results_application.dat')

    zspec2 = features_file2['specz'].to_numpy()
    zphot2 = features_file2['zphot'].to_numpy()
    zphot_err2 = features_file2['zerr'].to_numpy()
    zphot_conf2 = features_file2['zconf'].to_numpy()

    residual2 = ((zphot2 - zspec2)/(1+zspec2))[(zspec2 > zmin) & (zspec2 < zmax)]

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    h = np.histogram2d(residual1, residual2, range = ((-1.0,1.5),(-1.0,1.5)), bins = 100)[0]
    sp.imshow(np.log10(h).T, origin = 'lower', extent = (-1.0,1.5,-1.0,1.5), cmap = 'plasma_r')

    sp.text(0.02, 0.98, '$%.2f < z < %.2f$' % (zmin, zmax), fontsize = 24, ha = 'left', va = 'top', transform = sp.transAxes)

    sp.set_xlabel('Test Set Residual\n(Spec COSMOS)')
    sp.set_ylabel('Test Set Residual\n(Match COSMOS)')

    sp.set_xlim(-1.0,1.5)
    sp.set_ylim(-1.0,1.5)




def plot_residual_hists(run_dir1 = './tpzruns/SpecCOSMOS2015/', run_dir2 = './tpzruns/MatchCOSMOS2015/', nn_identifier = 'nnc_', zmin = 0, zmax = 1.5):
    

    cosmos = fits.open('./COSMOS2015/COSMOS2015_Laigle+_v1.1.fits')[1].data
    spec = fits.open('./HSC/HSC_wide_clean_pdr2.fits')[1].data
    cosmos = cosmos[(cosmos.TYPE == 0) & (cosmos.ZP_2 < 0)]

    cosmos_cat = SkyCoord(ra = cosmos.ALPHA_J2000*u.degree, dec = cosmos.DELTA_J2000*u.degree)
    spec_cat = SkyCoord(ra = spec.ra*u.degree, dec = spec.dec * u.degree)

    idx, d2d, _ = cosmos_cat.match_to_catalog_sky(spec_cat) # Produces a list of length cosmos_cat of indices matching to spec_cat objects

    # Find the spec_cat objects that are closest to the cosmos_cat objects and only keep those

    unique_idx = []

    for this_match_idx in np.unique(idx):

        matches = np.where(idx == this_match_idx)[0] # Get the list of matching objects
        unique_idx.append(matches[np.argmin(d2d[matches])]) # Pick the one with the smallest distance

    unique_idx = np.array(unique_idx) # unique_idx contains the indices of COSMOS objects that have "correct" matches (closest to the spec catalog object)
    spec_idx = idx[unique_idx]

    specz = spec.specz_redshift[spec_idx]
    photz = cosmos.ZPDF[unique_idx] # ZPDF for median of the PDF, ZMINCHI2 for minimum chi2 (-99 for <3 bands)
    photz_sigma = (cosmos.ZPDF_H68-cosmos.ZPDF_L68)[unique_idx] # ZPDF_H68-ZPDF_L68 for ZPDF; CHI2BEST for ZMINCHI2

    good_inds = (photz >= 0) & (photz_sigma < 0.1) & (d2d.to('arcsecond').value[unique_idx] < 1) & (cosmos.ip_MAG_APER2[unique_idx] < 25.) & (cosmos.ZP_2[unique_idx] < 0) & (cosmos.NBFILT[unique_idx] >= 25)

    specz = specz[good_inds]
    photz = photz[good_inds]

    residual_cosmos = (photz-specz)/(1+specz)



    if run_dir1[-1] != '/':
        run_dir1 += '/'

    nnc_folders1 = glob(run_dir1 + '*' + nn_identifier + '*/')

    if len(nnc_folders1) == 1:
        nnc_folder1 = nnc_folders1[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')
        for thisfolder in nnc_folders1:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    # nnr_folder1 = nnc_folder1.replace('nnc', 'nnr').replace('_corr', '')

    # features_file = read_csv(glob(run_dir + '*.nnc_test')[0], delimiter = '\s+', comment = '#') 
    # nnc_results = np.loadtxt(nnc_folder + 'results_test.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    features_file1 = read_csv(glob(run_dir1 + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results1 = np.loadtxt(nnc_folder1 + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    # nnr_results1 = np.loadtxt(nnr_folder1 + 'results_application.dat')

    zspec1 = features_file1['specz'].to_numpy()
    zphot1 = features_file1['zphot'].to_numpy()
    zphot_err1 = features_file1['zerr'].to_numpy()
    zphot_conf1 = features_file1['zconf'].to_numpy()

    residual1 = ((zphot1 - zspec1)/(1+zspec1))[(zspec1 > zmin) & (zspec1 < zmax)]


    if run_dir2[-1] != '/':
        run_dir2 += '/'

    nnc_folders2 = glob(run_dir2 + '*' + nn_identifier + '*/')

    if len(nnc_folders2) == 1:
        nnc_folder2 = nnc_folders2[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')
        for thisfolder in nnc_folders2:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    # features_file = read_csv(glob(run_dir + '*.nnc_test')[0], delimiter = '\s+', comment = '#') 
    # nnc_results = np.loadtxt(nnc_folder + 'results_test.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    features_file2 = read_csv(glob(run_dir2 + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results1 = np.loadtxt(nnc_folder2 + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    # nnr_results1 = np.loadtxt(nnr_folder2 + 'results_application.dat')

    zspec2 = features_file2['specz'].to_numpy()
    zphot2 = features_file2['zphot'].to_numpy()
    zphot_err2 = features_file2['zerr'].to_numpy()
    zphot_conf2 = features_file2['zconf'].to_numpy()

    residual2 = ((zphot2 - zspec2)/(1+zspec2))[(zspec2 > zmin) & (zspec2 < zmax)]

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    sp.hist(residual1, histtype = 'step', range = (-1.5,1.5), bins = 50, density = True, label = 'SpecCOSMOS')
    sp.hist(residual2, histtype = 'step', range = (-1.5,1.5), bins = 50, density = True, label = 'MatchCOSMOS')
    sp.hist(residual_cosmos, histtype = 'step', range = (-1.5,1.5), bins = 50, density = True, label = 'COSMOS2015')

    # xgrid = np.linspace(-1.5,1.55, 10000)
    # gaussian = lambda sigma, mu: (sigma * np.sqrt(2*np.pi))**-1 * np.exp(-.5 * (xgrid - mu)**2/sigma**2)

    # sp.plot(xgrid, gaussian(np.std(residual1), np.mean(residual1)), color = 'C0', linewidth = 2.5, label = r'$\sigma = {:.3f}, \mu = {:.3f}$'.format(np.std(residual1), np.mean(residual1)))
    # sp.plot(xgrid, gaussian(np.std(residual2), np.mean(residual2)), color = 'C1', linewidth = 2.5, label = r'$\sigma = {:.3f}, \mu = {:.3f}$'.format(np.std(residual2), np.mean(residual2)))
    # sp.plot(xgrid, gaussian(np.std(residual_cosmos), np.mean(residual_cosmos)), color = 'C2', linewidth = 2.5, label = r'$\sigma = {:.3f}, \mu = {:.3f}$'.format(np.std(residual_cosmos), np.mean(residual_cosmos)))


    # sp.plot(xgrid, gaussian(1.4826*np.median(np.abs(residual1)), np.mean(residual1)), color = 'C0', linewidth = 2.5, alpha = 0.5, linestyle = '--', label = r'$NMAD = {:.3f}, \mu = {:.3f}$'.format(1.4826*np.median(np.abs(residual1)), np.mean(residual1)))
    # sp.plot(xgrid, gaussian(1.4826*np.median(np.abs(residual2)), np.mean(residual2)), color = 'C1', linewidth = 2.5, alpha = 0.5, linestyle = '--', label = r'$NMAD = {:.3f}, \mu = {:.3f}$'.format(1.4826*np.median(np.abs(residual2)), np.mean(residual2)))
    # sp.plot(xgrid, gaussian(1.4826*np.median(np.abs(residual_cosmos)), np.mean(residual_cosmos)), color = 'C2', linewidth = 2.5, alpha = 0.5, linestyle = '--', label = r'$NMAD = {:.3f}, \mu = {:.3f}$'.format(1.4826*np.median(np.abs(residual_cosmos)), np.mean(residual_cosmos)))

    # sp.legend(loc = 'upper right', fontsize = 12)

    sp.set_xlabel('Residual')
    sp.set_ylabel('Density')

    sp.set_xlim(-1.5,1.5)



def plot_zphot_zspec(run_dir = './tpzruns/default_run_small/', nn_identifier = 'nnc_', nn_boundary = 0.0, fit_err_boundary = np.inf, zconf_boundary = 0, nn_correction = False, plot_percentile_lines = True):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')
        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')

    # features_file = read_csv(glob(run_dir + '*.nnc_test')[0], delimiter = '\s+', comment = '#') 
    # nnc_results = np.loadtxt(nnc_folder + 'results_test.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()
    zphot_err = features_file['zerr'].to_numpy()
    zphot_conf = features_file['zconf'].to_numpy()

    if nn_boundary > 0:
        # Read in nn_classifier fit file
        lim_inds = nnc_results > nn_boundary
        zspec = zspec[lim_inds]
        zphot = zphot[lim_inds]
        zphot_err = zphot_err[lim_inds]
        zphot_conf = zphot_conf[lim_inds]
        nnr_results = nnr_results[lim_inds]

    if nn_correction:
        # Read in nn_regressor fit file and subtract it from the zphots
        zphot = zphot - nnr_results


    if np.isfinite(fit_err_boundary) or zconf_boundary > 0:

        lim_inds = (zphot_err < fit_err_boundary) & (zphot_conf > zconf_boundary)
        zspec = zspec[lim_inds]
        zphot = zphot[lim_inds]
        zphot_err = zphot_err[lim_inds]
        zphot_conf = zphot_conf[lim_inds]        



    error = np.abs(zphot - zspec)/(1+zspec)

    if plot_percentile_lines:
        plot_percentiles = [25, 50, 75, 95]
        line_lo = []
        line_hi = []
        xgrid = np.linspace(0,1.5,1000)

        for this_percentile in plot_percentiles:

            ind = np.argmin(np.abs(np.percentile(error, this_percentile) - error))
            line_lo.append(xgrid - (1+xgrid)*error[ind])
            line_hi.append(xgrid + (1+xgrid)*error[ind])

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    h = np.histogram2d(zspec, zphot, range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp.imshow(np.log10(h).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    if plot_percentile_lines:
        for thisline_lo, thisline_hi, this_percentile in zip(line_lo, line_hi, plot_percentiles):
            sp.plot(xgrid, thisline_lo, color = 'k', label = '%i'%this_percentile)
            sp.plot(xgrid, thisline_hi, color = 'k')

    sp.set_xlabel('$z_{spec}$')
    sp.set_ylabel('$z_{phot}$')

    txt1 = sp.text(0.98, 0.02, '$P_{25}=%.3f$\n$P_{50}=%.3f$\n$P_{75}=%.3f$\n$P_{95}=%.3f$\n' % tuple(np.percentile(error, [25,50,75,95])) + 
                                '$NMAD=%.3f$\n' % (1.4826*np.median(error)) +
                                r'$\sigma=' + '%.3f$' % np.std(error), 
                                color = 'white', fontsize = 24, ha = 'right', va = 'bottom', transform = sp.transAxes)
    txt1.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

    txt2 = sp.text(0.02, 0.98, '$N=%i$' % len(zphot), color = 'white', fontsize = 30, ha = 'left', va = 'top', transform = sp.transAxes)
    txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

    sp.set_xlim(0,1.5)
    sp.set_ylim(0,1.5)


def plot_zphot_zspec_nnc2(run_dir = './tpzruns/default_run_small/', nn_identifier = 'nnc2_', nn_boundary = 0.0, fit_err_boundary = np.inf, zconf_boundary = 0, nn_correction = False, plot_percentile_lines = True):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')
        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc2', 'nnr').replace('_corr', '')

    # features_file = read_csv(glob(run_dir + '*.nnc_test')[0], delimiter = '\s+', comment = '#') 
    # nnc_results = np.loadtxt(nnc_folder + 'results_test.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()
    zphot_err = features_file['zerr'].to_numpy()
    zphot_conf = features_file['zconf'].to_numpy()

    if nn_boundary > 0:
        # Read in nn_classifier fit file
        lim_inds = nnc_results > nn_boundary
        zspec = zspec[lim_inds]
        zphot = zphot[lim_inds]
        zphot_err = zphot_err[lim_inds]
        zphot_conf = zphot_conf[lim_inds]
        nnr_results = nnr_results[lim_inds]

    if nn_correction:
        # Read in nn_regressor fit file and subtract it from the zphots
        zphot = zphot - nnr_results


    if np.isfinite(fit_err_boundary) or zconf_boundary > 0:

        lim_inds = (zphot_err < fit_err_boundary) & (zphot_conf > zconf_boundary)
        zspec = zspec[lim_inds]
        zphot = zphot[lim_inds]
        zphot_err = zphot_err[lim_inds]
        zphot_conf = zphot_conf[lim_inds]        



    error = np.abs(zphot - zspec)/(1+zspec)

    if plot_percentile_lines:
        plot_percentiles = [25, 50, 75, 95]
        line_lo = []
        line_hi = []
        xgrid = np.linspace(0,1.5,1000)

        for this_percentile in plot_percentiles:

            ind = np.argmin(np.abs(np.percentile(error, this_percentile) - error))
            line_lo.append(xgrid - (1+xgrid)*error[ind])
            line_hi.append(xgrid + (1+xgrid)*error[ind])

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    h = np.histogram2d(zspec, zphot, range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp.imshow(np.log10(h).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    if plot_percentile_lines:
        for thisline_lo, thisline_hi, this_percentile in zip(line_lo, line_hi, plot_percentiles):
            sp.plot(xgrid, thisline_lo, color = 'k', label = '%i'%this_percentile)
            sp.plot(xgrid, thisline_hi, color = 'k')

    sp.set_xlabel('$z_{spec}$')
    sp.set_ylabel('$z_{phot}$')

    txt1 = sp.text(0.98, 0.02, '$P_{25}=%.3f$\n$P_{50}=%.3f$\n$P_{75}=%.3f$\n$P_{95}=%.3f$\n' % tuple(np.percentile(error, [25,50,75,95])) + 
                                '$NMAD=%.3f$\n' % (1.4826*np.median(error)) +
                                r'$\sigma=' + '%.3f$' % np.std(error), 
                                color = 'white', fontsize = 24, ha = 'right', va = 'bottom', transform = sp.transAxes)
    txt1.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

    txt2 = sp.text(0.02, 0.98, '$N=%i$' % len(zphot), color = 'white', fontsize = 30, ha = 'left', va = 'top', transform = sp.transAxes)
    txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

    sp.set_xlim(0,1.5)
    sp.set_ylim(0,1.5)



def plot_zphot_zspec_delete_later(run_dir = './tpzruns/default_run/', nn_identifier = 'nnc_', nn_boundary = 0.0, fit_err_boundary = np.inf, zconf_boundary = 0, nn_correction = False, plot_percentile_lines = False):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')
        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')

    # features_file = read_csv(glob(run_dir + '*.nnc_test')[0], delimiter = '\s+', comment = '#') 
    # nnc_results = np.loadtxt(nnc_folder + 'results_test.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()
    zphot_err = features_file['zerr'].to_numpy()
    zphot_conf = features_file['zconf'].to_numpy()

    if nn_boundary > 0:
        # Read in nn_classifier fit file
        lim_inds = nnc_results > nn_boundary
        zspec = zspec[lim_inds]
        zphot = zphot[lim_inds]
        zphot_err = zphot_err[lim_inds]
        zphot_conf = zphot_conf[lim_inds]
        nnr_results = nnr_results[lim_inds]

    if nn_correction:
        # Read in nn_regressor fit file and subtract it from the zphots
        zphot = zphot - nnr_results


    if np.isfinite(fit_err_boundary) or zconf_boundary > 0:

        lim_inds = (zphot_err < fit_err_boundary) & (zphot_conf > zconf_boundary)
        zspec = zspec[lim_inds]
        zphot = zphot[lim_inds]
        zphot_err = zphot_err[lim_inds]
        zphot_conf = zphot_conf[lim_inds]        



    error = np.abs(zphot - zspec)/(1+zspec)
    twosigma_outlier_frac = sum(np.abs(zphot - zspec) > 2 * zphot_err)/float(len(zphot_err))

    # if plot_percentile_lines:
    #     plot_percentiles = [25, 50, 75, 95]
    #     line_lo = []
    #     line_hi = []
    #     xgrid = np.linspace(0,1.5,1000)

    #     for this_percentile in plot_percentiles:

    #         ind = np.argmin(np.abs(np.percentile(error, this_percentile) - error))
    #         line_lo.append(xgrid - (1+xgrid)*error[ind])
    #         line_hi.append(xgrid + (1+xgrid)*error[ind])

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    h = np.histogram2d(zspec, zphot, range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp.imshow(np.log10(h).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    if plot_percentile_lines:
        for thisline_lo, thisline_hi, this_percentile in zip(line_lo, line_hi, plot_percentiles):
            sp.plot(xgrid, thisline_lo, color = 'k', label = '%i'%this_percentile)
            sp.plot(xgrid, thisline_hi, color = 'k')

    sp.set_xlabel('$z_{spec}$')
    sp.set_ylabel('$z_{phot}$')

    txt1 = sp.text(0.98, 0.02, '$NMAD=%.3f$\n' % (1.4826*np.median(error)) +
                                r'$2\sigma$ Outlier Frac$ = %.3f$' % twosigma_outlier_frac + '\n'
                                r'$\sigma=' + '%.3f$' % np.std(error), 
                                color = 'white', fontsize = 24, ha = 'right', va = 'bottom', transform = sp.transAxes)
    txt1.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

    txt2 = sp.text(0.02, 0.98, '$N=%i$' % len(zphot), color = 'white', fontsize = 30, ha = 'left', va = 'top', transform = sp.transAxes)
    txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

    sp.set_xlim(0,1.5)
    sp.set_ylim(0,1.5)






def plot_nn_population_fraction(run_dir = './tpzruns/default_run/', nn_identifier = 'nnc_'):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    percentiles = [float(np.sum(nnc_results >= this_nn_boundary))/len(nnc_results) for this_nn_boundary in np.arange(0,1,0.01)]

    sp.plot(np.arange(0,1,0.01), percentiles)

    sp.set_xlabel('NN Boundary')
    sp.set_ylabel('Population Fraction')



def plot_pipeline_results(run_dir = './tpzruns/default_run/', nn_identifier = 'nnc_', nn_boundary = 0.65, zconf_boundary = 0., plot_percentile_lines = True, gz_split = False, gz_splitval = None):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()
    zconf = features_file['zconf'].to_numpy()

    if gz_split != False:
        gz = features_file['g'].to_numpy() - features_file['z'].to_numpy()
        if gz_splitval == None:
            gz_splitval = np.nanmedian(gz)
            print('gz Median: %.2f' % gz_splitval)
        
        if gz_split == 'red':
            gz_colorlim_inds = gz > gz_splitval

        elif gz_split == 'blue':
            gz_colorlim_inds = gz < gz_splitval

        zspec = zspec[gz_colorlim_inds]
        zphot = zphot[gz_colorlim_inds]
        zconf = zconf[gz_colorlim_inds]
        nnc_results = nnc_results[gz_colorlim_inds]
        nnr_results = nnr_results[gz_colorlim_inds]

    nnc_lim_inds = (nnc_results > nn_boundary)
    zconf_lim_inds = (zconf > zconf_boundary)
    all_lim_inds = nnc_lim_inds & zconf_lim_inds


    fig = plt.figure(figsize = (18,6))
    sp1 = fig.add_subplot(131)
    sp2 = fig.add_subplot(132)
    sp3 = fig.add_subplot(133)

    plot1 = np.histogram2d(zspec[zconf_lim_inds], zphot[zconf_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp1.imshow(np.log10(plot1).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot2 = np.histogram2d(zspec[zconf_lim_inds], (zphot - nnr_results)[zconf_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp2.imshow(np.log10(plot2).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot3 = np.histogram2d(zspec[nnc_lim_inds], (zphot - nnr_results)[nnc_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp3.imshow(np.log10(plot3).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    xdata = [zspec, zspec, zspec[nnc_lim_inds]]
    ydata = [zphot, zphot - nnr_results, (zphot-nnr_results)[nnc_lim_inds]]
    error = []

    if plot_percentile_lines:
        plot_percentiles = [25, 50, 75, 95]
        xgrid = np.linspace(0,1.5,1000)

        for thissubplot, thisx, thisy in zip([sp1, sp2, sp3], xdata, ydata):

            thiserror = np.abs(thisy - thisx)/ (1 + thisx)
            error.append(thiserror[np.isfinite(thiserror)])

            for this_percentile in plot_percentiles:

                ind = np.argmin(np.abs(np.percentile(thiserror, this_percentile) - thiserror))
                thissubplot.plot(xgrid, xgrid - (1+xgrid)*thiserror[ind], color = 'k')
                thissubplot.plot(xgrid, xgrid + (1+xgrid)*thiserror[ind], color = 'k')


    for thissubplot, thiserror in zip([sp1, sp2, sp3], error):

        txt1 = thissubplot.text(0.98, 0.02, '$P_{25}=%.3f$\n$P_{50}=%.3f$\n$P_{75}=%.3f$\n$P_{95}=%.3f$\n' % tuple(np.percentile(thiserror, [25,50,75,95])) + 
                                    '$NMAD=%.3f$\n' % (1.4826*np.median(thiserror)) +
                                    r'$\sigma=' + '%.3f$' % np.std(thiserror), 
                                    color = 'white', fontsize = 18, ha = 'right', va = 'bottom', transform = thissubplot.transAxes)
        txt1.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

        txt2 = thissubplot.text(0.02, 0.98, '$N=%i$' % len(thiserror), color = 'white', fontsize = 20, ha = 'left', va = 'top', transform = thissubplot.transAxes)
        txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

        thissubplot.set_xlim(0,1.5)
        thissubplot.set_ylim(0,1.5)

    fig.text(0.5, 0., '$z_{spec}$', fontsize = 30)
    sp1.set_ylabel('$z_{phot}$', fontsize = 30)

    sp2.set_yticklabels([])
    sp3.set_yticklabels([])
    plt.subplots_adjust(wspace = 0)








def plot_all_results_nonnr(run_names = ['SpecSpec', 'SpecCOSMOS2015', 'COSMOSCOSMOS', 'MatchCOSMOS2015'], nn_identifier = 'nnc_', sample_fraction = 1./3., plot_percentile_lines = False):
    
    run_dirs = ['./tpzruns/' + thisname + '/' for thisname in run_names]
    names = [r'Spec $\rightarrow$ Spec Case', r'Spec $\rightarrow$ COSMOS2015 Case', r'COSMOS2015 $\rightarrow$ COSMOS2015 Case', r'Match $\rightarrow$ COSMOS2015 Case']

    fig, subplots = plt.subplots(len(run_names), 4, figsize = (24, 8*len(run_names)))
    plt.subplots_adjust(wspace = 0, hspace = 0.22)

    for this_run_dir, thisgroupname, (sp1, sp2, sp3, sp4) in zip(run_dirs, names, subplots):

        nnc_folders = glob(this_run_dir + '*' + nn_identifier + '*0/')

        if len(nnc_folders) == 1:
            nnc_folder = nnc_folders[0]
        elif len(nnc_folders) == 0:
            print('There are no folders matching the nn_identifier.')
            return None
        else:
            print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

            for thisfolder in nnc_folders:
                print(thisfolder)
            return None

        features_file = read_csv(glob(this_run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

        zspec = features_file['specz'].to_numpy()
        zphot = features_file['zphot'].to_numpy()
        zconf = features_file['zconf'].to_numpy()
        zphot_err = features_file['zerr'].to_numpy()

        # zconf_lim_inds = (zconf > zconf_boundary)
        zphot_err_boundary = np.sort(zphot_err)[int(len(zphot_err)*sample_fraction)]
        zphot_err_lim_inds = zphot_err < zphot_err_boundary
        # Find the nn_boundary that gives the same number of objects
        nn_boundary = np.sort(nnc_results)[::-1][sum(zphot_err_lim_inds)]
        nnc_lim_inds = (nnc_results > nn_boundary)
        # Find the zphot_err that gives the same number of objects
        # fake_zphot_err = zphot_err + np.random.random(len(zphot_err))*10**-4  # Add a small random number to zphot_err because values are not unique
        # zphot_err_boundary = np.sort(fake_zphot_err)[sum(zconf_lim_inds)]
        # zphot_err_lim_inds = (fake_zphot_err < zphot_err_boundary)
        fake_zconf = zconf + np.random.random(len(zconf))* 10**-4
        zconf_boundary = np.sort(fake_zconf)[::-1][sum(zphot_err_lim_inds)]
        zconf_lim_inds = (fake_zconf > zconf_boundary)

        plot1 = np.histogram2d(zspec, zphot, range = ((0,1.5),(0,1.5)), bins = 50)[0]
        sp1.imshow(np.log10(plot1).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

        plot2 = np.histogram2d(zspec[zphot_err_lim_inds], zphot[zphot_err_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
        sp2.imshow(np.log10(plot2).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

        plot3 = np.histogram2d(zspec[zconf_lim_inds], zphot[zconf_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
        sp3.imshow(np.log10(plot3).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

        plot4 = np.histogram2d(zspec[nnc_lim_inds], zphot[nnc_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
        sp4.imshow(np.log10(plot4).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

        xdata = [zspec, zspec[zphot_err_lim_inds], zspec[zconf_lim_inds], zspec[nnc_lim_inds]]
        ydata = [zphot, zphot[zphot_err_lim_inds], zphot[zconf_lim_inds], zphot[nnc_lim_inds]]
        error = []

        # if plot_percentile_lines:
        plot_percentiles = [25, 50, 75, 95]
        xgrid = np.linspace(0,1.5,1000)

        for thissubplot, thisx, thisy in zip([sp1, sp2, sp3, sp4], xdata, ydata):

            thiserror = np.abs(thisy - thisx)/ (1 + thisx)
            error.append(thiserror[np.isfinite(thiserror)])

            if plot_percentile_lines:

                for this_percentile in plot_percentiles:

                    ind = np.argmin(np.abs(np.percentile(thiserror, this_percentile) - thiserror))
                    thissubplot.plot(xgrid, xgrid - (1+xgrid)*thiserror[ind], color = 'k')
                    thissubplot.plot(xgrid, xgrid + (1+xgrid)*thiserror[ind], color = 'k')

        plotnames = ['', r'$\sigma_{TPZ}<' + '%.2f$\n'%zphot_err_boundary, '$zConf>%.2f$\n' % zconf_boundary, '$C_{NNC} > %.2f$\n' % nn_boundary]

        for thissubplot, thiserror, thisname in zip([sp1, sp2, sp3, sp4], error, plotnames):

            if plot_percentile_lines:
                percentile_text = '$P_{25}=%.3f$\n$P_{50}=%.3f$\n$P_{75}=%.3f$\n$P_{95}=%.3f$\n' % tuple(np.percentile(thiserror, [25,50,75,95]))
            else:
                percentile_text = ''

            txt1 = thissubplot.text(0.98, 0.02, percentile_text + 
                                        '$f_{out}=%.3f$\n' % (np.sum(thiserror>0.15)/float(len(thiserror))) +
                                        '$NMAD=%.3f$\n' % (1.4826*np.median(thiserror)) +
                                        r'$\sigma_z=' + '%.3f$' % np.std(thiserror), 
                                        color = 'k', fontsize = 18, ha = 'right', va = 'bottom', weight = 'normal', transform = thissubplot.transAxes)
            txt1.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

            txt2 = thissubplot.text(0.02, 0.98, thisname + '$N=%i$' % len(thiserror), color = 'k', fontsize = 20, ha = 'left', va = 'top', weight = 'normal', transform = thissubplot.transAxes)
            txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

            thissubplot.set_xlim(0,1.5)
            thissubplot.set_ylim(0,1.5)
        
        sp1.set_ylabel('$z_{phot}$', fontsize = 26)
        sp2.set_yticklabels([])
        sp3.set_yticklabels([])
        sp4.set_yticklabels([])

        fig.canvas.draw()

        for this_sp in [sp1, sp2, sp3]:
            labels = [item.get_text() for item in this_sp.get_xticklabels()]
            labels[-1] = ' '
            this_sp.set_xticklabels(labels)

        xcoord, ybottom, _, height = sp3.get_position().bounds
        ytop = ybottom + height

        fig.text(xcoord, ybottom-0.015, '$z_{spec}$', fontsize = 26, ha = 'center', va = 'top')
        fig.text(xcoord, ytop+0.004, thisgroupname, weight = 'normal', fontsize = 30, ha = 'center', va = 'bottom')



def plot_all_train_test_nonnr(run_names = ['SpecCOSMOS2015', 'COSMOSCOSMOS', 'MatchCOSMOS2015'], nn_identifier = 'nnc_'):
    
    run_dirs = ['./tpzruns/' + thisname + '/' for thisname in run_names]
    names = [r'Spec $\rightarrow$ COSMOS2015 Case', r'COSMOS2015 $\rightarrow$ COSMOS2015', r'Match $\rightarrow$ COSMOS2015 Case']

    fig, subplots = plt.subplots(len(run_names), 2, figsize = (9, 7*len(run_names)))
    plt.subplots_adjust(wspace = 0, hspace = 0.23)


    for this_run_dir, thisgroupname, (sp1, sp2) in zip(run_dirs, names, subplots):

        nnc_folders = glob(this_run_dir + '*' + nn_identifier + '*0/')

        if len(nnc_folders) == 1:
            nnc_folder = nnc_folders[0]
        elif len(nnc_folders) == 0:
            print('There are no folders matching the nn_identifier.')
            return None
        else:
            print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

            for thisfolder in nnc_folders:
                print(thisfolder)
            return None

        app_feature_file = read_csv(glob(this_run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        train_feature_file = read_csv(glob(this_run_dir + '*.nnc_train')[0], delimiter = '\s+', comment = '#').append(read_csv(glob(this_run_dir + '*.tpz_train')[0], delimiter = '\s+', comment = '#'))

        i_train = train_feature_file['i'].to_numpy()
        gz_train = train_feature_file['g'].to_numpy() - train_feature_file['z'].to_numpy()
        i_app = app_feature_file['i'].to_numpy()
        gz_app = app_feature_file['g'].to_numpy() - app_feature_file['z'].to_numpy()

        plot1 = np.histogram2d(gz_train, i_train, range = ((-4,9),(13,28)), bins = 50)[0]
        sp1.imshow(np.log10(plot1).T, origin = 'lower', extent = ((-4,9,13,28)), cmap = 'YlGnBu')

        plot2 = np.histogram2d(gz_app, i_app, range = ((-4,9),(13,28)), bins = 50)[0]
        sp2.imshow(np.log10(plot2).T, origin = 'lower', extent = ((-4,9,13,28)), cmap = 'YlGnBu')
        
        for thissubplot, data_size in zip([sp1,sp2], [len(gz_train), len(gz_app)]):
            txt = thissubplot.text(0.02, 0.98, '$N=%i$' % data_size, color = 'k', fontsize = 20, ha = 'left', va = 'top', weight = 'normal', transform = thissubplot.transAxes)
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

        sp1.set_ylabel('$i$', fontsize = 24)
        sp2.set_yticklabels([])

        sp1.text(0.02, 0.02, 'Training', color = 'k', fontsize = 20, ha = 'left', va = 'bottom', transform = sp1.transAxes)
        sp2.text(0.98, 0.02, 'Test', color = 'k', fontsize = 20, ha = 'right', va = 'bottom', transform = sp2.transAxes)

        fig.canvas.draw()

        labels = [item.get_text() for item in sp2.get_xticklabels()]
        labels[-1] = ' '
        sp2.set_xticklabels(labels)

        xcoord, ybottom, _, height = sp2.get_position().bounds
        ytop = ybottom + height

        fig.text(xcoord, ybottom-0.015, '$(g-z)$', fontsize = 24, ha = 'center', va = 'top')
        fig.text(xcoord, ytop+0.004, thisgroupname, weight = 'normal', fontsize = 30, ha = 'center', va = 'bottom')












def plot_pipeline_results_nonnr(run_dir = './tpzruns/default_run/', nn_identifier = 'nnc_', zphot_err_boundary = 0.07, plot_percentile_lines = False, gz_split = False, gz_splitval = None):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*0/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    
    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()
    zconf = features_file['zconf'].to_numpy()
    zphot_err = features_file['zerr'].to_numpy()

    if gz_split != False:
        gz = features_file['g'].to_numpy() - features_file['z'].to_numpy()
        if gz_splitval == None:
            gz_splitval = np.nanmedian(gz)
            print('gz Median: %.2f' % gz_splitval)
        
        if gz_split == 'red':
            gz_colorlim_inds = gz > gz_splitval

        elif gz_split == 'blue':
            gz_colorlim_inds = gz < gz_splitval

        zspec = zspec[gz_colorlim_inds]
        zphot = zphot[gz_colorlim_inds]
        zconf = zconf[gz_colorlim_inds]
        nnc_results = nnc_results[gz_colorlim_inds]

    # zconf_lim_inds = (zconf > zconf_boundary)
    zphot_err_lim_inds = zphot_err < zphot_err_boundary
    # Find the nn_boundary that gives the same number of objects
    nn_boundary = np.sort(nnc_results)[::-1][sum(zphot_err_lim_inds)]
    nnc_lim_inds = (nnc_results > nn_boundary)
    # Find the zphot_err that gives the same number of objects
    # fake_zphot_err = zphot_err + np.random.random(len(zphot_err))*10**-4  # Add a small random number to zphot_err because values are not unique
    # zphot_err_boundary = np.sort(fake_zphot_err)[sum(zconf_lim_inds)]
    # zphot_err_lim_inds = (fake_zphot_err < zphot_err_boundary)
    fake_zconf = zconf + np.random.random(len(zconf))* 10**-4
    zconf_boundary = np.sort(fake_zconf)[::-1][sum(zphot_err_lim_inds)]
    zconf_lim_inds = (fake_zconf > zconf_boundary)

    fig = plt.figure(figsize = (24,6))
    sp1 = fig.add_subplot(141)
    sp2 = fig.add_subplot(142)
    sp3 = fig.add_subplot(143)
    sp4 = fig.add_subplot(144)
    # sp3 = fig.add_subplot(122)

    plot1 = np.histogram2d(zspec, zphot, range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp1.imshow(np.log10(plot1).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot2 = np.histogram2d(zspec[zphot_err_lim_inds], zphot[zphot_err_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp2.imshow(np.log10(plot2).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot3 = np.histogram2d(zspec[zconf_lim_inds], zphot[zconf_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp3.imshow(np.log10(plot3).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot4 = np.histogram2d(zspec[nnc_lim_inds], zphot[nnc_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp4.imshow(np.log10(plot4).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    xdata = [zspec, zspec[zphot_err_lim_inds], zspec[zconf_lim_inds], zspec[nnc_lim_inds]]
    ydata = [zphot, zphot[zphot_err_lim_inds], zphot[zconf_lim_inds], zphot[nnc_lim_inds]]
    error = []

    # if plot_percentile_lines:
    plot_percentiles = [25, 50, 75, 95]
    xgrid = np.linspace(0,1.5,1000)

    for thissubplot, thisx, thisy in zip([sp1, sp2, sp3, sp4], xdata, ydata):

        thiserror = np.abs(thisy - thisx)/ (1 + thisx)
        error.append(thiserror[np.isfinite(thiserror)])

        if plot_percentile_lines:

            for this_percentile in plot_percentiles:

                ind = np.argmin(np.abs(np.percentile(thiserror, this_percentile) - thiserror))
                thissubplot.plot(xgrid, xgrid - (1+xgrid)*thiserror[ind], color = 'k')
                thissubplot.plot(xgrid, xgrid + (1+xgrid)*thiserror[ind], color = 'k')

    plotnames = ['', r'$\sigma_{TPZ} < ' + '%.2f$\n' % zphot_err_boundary, '$zConf > %.2f$\n' % zconf_boundary, '$C_{TPZ} > %.2f$\n' % nn_boundary]

    for thissubplot, thiserror, thisname in zip([sp1, sp2, sp3, sp4], error, plotnames):

        if plot_percentile_lines:
            percentile_text = '$P_{25}=%.3f$\n$P_{50}=%.3f$\n$P_{75}=%.3f$\n$P_{95}=%.3f$\n' % tuple(np.percentile(thiserror, [25,50,75,95]))
        else:
            percentile_text = ''

        txt1 = thissubplot.text(0.98, 0.02, percentile_text + 
                                    '$f_{out}=%.3f$\n' % (np.sum(thiserror>0.1)/float(len(thiserror))) +
                                    '$NMAD=%.3f$\n' % (1.4826*np.median(thiserror)) +
                                    r'$\sigma_z=' + '%.3f$' % np.std(thiserror), 
                                    color = 'k', fontsize = 18, ha = 'right', va = 'bottom', transform = thissubplot.transAxes)
        txt1.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

        txt2 = thissubplot.text(0.02, 0.98, thisname + '$N=%i$' % len(thiserror), color = 'k', fontsize = 20, ha = 'left', va = 'top', transform = thissubplot.transAxes)
        txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

        thissubplot.set_xlim(0,1.5)
        thissubplot.set_ylim(0,1.5)

    fig.text(0.5, 0.05, '$z_{spec}$', fontsize = 30, weight = 'normal')
    sp1.set_ylabel('$z_{phot}$', fontsize = 30, weight = 'normal')

    sp2.set_yticklabels([])
    sp3.set_yticklabels([])
    sp4.set_yticklabels([])

    plt.subplots_adjust(wspace = 0)
    fig.canvas.draw()

    for this_sp in [sp1, sp2, sp3]:
        this_sp_ticks = [thistick.get_text() for thistick in this_sp.get_xticklabels()]
        this_sp_ticks[-1] = ''
        this_sp.set_xticklabels(this_sp_ticks)





def plot_pipeline_results_nonnr_del_later(run_dir = './tpzruns/MatchCOSMOS2015/', nn_identifier = 'nnc_', zphot_err_boundary = 0.07, plot_percentile_lines = False, gz_split = False, gz_splitval = None):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*0/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    
    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()
    zconf = features_file['zconf'].to_numpy()
    zphot_err = features_file['zerr'].to_numpy()

    if gz_split != False:
        gz = features_file['g'].to_numpy() - features_file['z'].to_numpy()
        if gz_splitval == None:
            gz_splitval = np.nanmedian(gz)
            print('gz Median: %.2f' % gz_splitval)
        
        if gz_split == 'red':
            gz_colorlim_inds = gz > gz_splitval

        elif gz_split == 'blue':
            gz_colorlim_inds = gz < gz_splitval

        zspec = zspec[gz_colorlim_inds]
        zphot = zphot[gz_colorlim_inds]
        zconf = zconf[gz_colorlim_inds]
        nnc_results = nnc_results[gz_colorlim_inds]

    # zconf_lim_inds = (zconf > zconf_boundary)
    zphot_err_lim_inds = zphot_err < zphot_err_boundary
    # Find the nn_boundary that gives the same number of objects
    nn_boundary = np.sort(nnc_results)[::-1][sum(zphot_err_lim_inds)]
    nnc_lim_inds = (nnc_results > nn_boundary)
    # Find the zphot_err that gives the same number of objects
    # fake_zphot_err = zphot_err + np.random.random(len(zphot_err))*10**-4  # Add a small random number to zphot_err because values are not unique
    # zphot_err_boundary = np.sort(fake_zphot_err)[sum(zconf_lim_inds)]
    # zphot_err_lim_inds = (fake_zphot_err < zphot_err_boundary)
    fake_zconf = zconf + np.random.random(len(zconf))* 10**-4
    zconf_boundary = np.sort(fake_zconf)[::-1][sum(zphot_err_lim_inds)]
    zconf_lim_inds = (fake_zconf > zconf_boundary)

    fig = plt.figure(figsize = (17,6))
    sp1 = fig.add_subplot(131)
    sp2 = fig.add_subplot(132)
    sp3 = fig.add_subplot(133)
    # sp3 = fig.add_subplot(122)

    plot1 = np.histogram2d(zspec, zphot, range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp1.imshow(np.log10(plot1).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot2 = np.histogram2d(zspec[zphot_err_lim_inds], zphot[zphot_err_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp2.imshow(np.log10(plot2).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot3 = np.histogram2d(zspec[nnc_lim_inds], zphot[nnc_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp3.imshow(np.log10(plot3).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    xdata = [zspec, zspec[zphot_err_lim_inds], zspec[nnc_lim_inds]]
    ydata = [zphot, zphot[zphot_err_lim_inds], zphot[nnc_lim_inds]]
    error = []

    # if plot_percentile_lines:
    plot_percentiles = [25, 50, 75, 95]
    xgrid = np.linspace(0,1.5,1000)

    for thissubplot, thisx, thisy in zip([sp1, sp2, sp3], xdata, ydata):

        thiserror = np.abs(thisy - thisx)/ (1 + thisx)
        error.append(thiserror[np.isfinite(thiserror)])

        if plot_percentile_lines:

            for this_percentile in plot_percentiles:

                ind = np.argmin(np.abs(np.percentile(thiserror, this_percentile) - thiserror))
                thissubplot.plot(xgrid, xgrid - (1+xgrid)*thiserror[ind], color = 'k')
                thissubplot.plot(xgrid, xgrid + (1+xgrid)*thiserror[ind], color = 'k')

    plotnames = ['', r'$\sigma_{TPZ} < ' + '%.2f$\n' % zphot_err_boundary, '$C_{TPZ} > %.2f$\n' % nn_boundary]

    for thissubplot, thiserror, thisname in zip([sp1, sp2, sp3], error, plotnames):

        if plot_percentile_lines:
            percentile_text = '$P_{25}=%.3f$\n$P_{50}=%.3f$\n$P_{75}=%.3f$\n$P_{95}=%.3f$\n' % tuple(np.percentile(thiserror, [25,50,75,95]))
        else:
            percentile_text = ''

        txt1 = thissubplot.text(0.98, 0.02, percentile_text + 
                                    '$f_{out}=%.3f$\n' % (np.sum(thiserror>0.1)/float(len(thiserror))) +
                                    '$NMAD=%.3f$\n' % (1.4826*np.median(thiserror)) +
                                    r'$\sigma_z=' + '%.3f$' % np.std(thiserror), 
                                    color = 'k', fontsize = 18, ha = 'right', va = 'bottom', transform = thissubplot.transAxes)
        txt1.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

        txt2 = thissubplot.text(0.02, 0.98, thisname + '$N=%i$' % len(thiserror), color = 'k', fontsize = 20, ha = 'left', va = 'top', transform = thissubplot.transAxes)
        txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

        thissubplot.set_xlim(0,1.5)
        thissubplot.set_ylim(0,1.5)

    fig.text(0.5, 0.02, '$z_{spec}$', fontsize = 30, weight = 'normal')
    fig.text(0.5, 0.93, r'Match $\rightarrow$ COSMOS2015', fontsize = 35, weight = 'normal', ha = 'center')
    sp1.set_ylabel('$z_{phot}$', fontsize = 30, weight = 'normal')

    sp2.set_yticklabels([])
    sp3.set_yticklabels([])

    plt.subplots_adjust(wspace = 0)
    fig.canvas.draw()

    for this_sp in [sp1, sp2]:
        this_sp_ticks = [thistick.get_text() for thistick in this_sp.get_xticklabels()]
        this_sp_ticks[-1] = ''
        this_sp.set_xticklabels(this_sp_ticks)





def plot_pipeline_results_nnc2(run_dir = './tpzruns/default_run/', nn_identifier = 'nnc2_', nn_boundary = 0.9, zconf_boundary = 0., plot_percentile_lines = True, gz_split = False, gz_splitval = None):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc2', 'nnr').replace('_corr', '')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    good_inds = (np.all(np.isfinite(nnc_results), axis = 1) & np.isfinite(nnr_results))

    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    zconf = features_file['zconf'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    nnr_results = nnr_results[good_inds]

    nnc_class = np.argmax(nnc_results, axis = 1)

    zphot_nnc = np.copy(zphot)
    zphot_nnc[nnc_class == 1] = (zphot - nnr_results)[nnc_class == 1] #NNR photoz is better
    goodfit_conf = np.maximum(*nnc_results.T)

    if gz_split != False:
        gz = features_file['g'].to_numpy() - features_file['z'].to_numpy()
        if gz_splitval == None:
            gz_splitval = np.nanmedian(gz)
            print('gz Median: %.2f' % gz_splitval)
        
        if gz_split == 'red':
            gz_colorlim_inds = gz > gz_splitval

        elif gz_split == 'blue':
            gz_colorlim_inds = gz < gz_splitval

        zspec = zspec[gz_colorlim_inds]
        zphot = zphot[gz_colorlim_inds]
        zconf = zconf[gz_colorlim_inds]
        nnc_results = nnc_results[gz_colorlim_inds]
        nnr_results = nnr_results[gz_colorlim_inds]

    nnc_lim_inds = (goodfit_conf > nn_boundary)
    zconf_lim_inds = (zconf > zconf_boundary)

    fig = plt.figure(figsize = (18,6))
    sp1 = fig.add_subplot(131)
    sp2 = fig.add_subplot(132)
    sp3 = fig.add_subplot(133)

    plot1 = np.histogram2d(zspec[zconf_lim_inds], zphot[zconf_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp1.imshow(np.log10(plot1).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot2 = np.histogram2d(zspec[zconf_lim_inds], (zphot - nnr_results)[zconf_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp2.imshow(np.log10(plot2).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot3 = np.histogram2d(zspec[nnc_lim_inds], zphot_nnc[nnc_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp3.imshow(np.log10(plot3).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    xdata = [zspec, zspec, zspec[nnc_lim_inds]]
    ydata = [zphot, zphot - nnr_results, zphot_nnc[nnc_lim_inds]]
    error = []

    # if plot_percentile_lines:
    plot_percentiles = [25, 50, 75, 95]
    xgrid = np.linspace(0,1.5,1000)

    for thissubplot, thisx, thisy in zip([sp1, sp2, sp3], xdata, ydata):

        thiserror = np.abs(thisy - thisx)/ (1 + thisx)
        error.append(thiserror[np.isfinite(thiserror)])

        if plot_percentile_lines:

            for this_percentile in plot_percentiles:

                ind = np.argmin(np.abs(np.percentile(thiserror, this_percentile) - thiserror))
                thissubplot.plot(xgrid, xgrid - (1+xgrid)*thiserror[ind], color = 'k')
                thissubplot.plot(xgrid, xgrid + (1+xgrid)*thiserror[ind], color = 'k')


    for thissubplot, thiserror in zip([sp1, sp2, sp3], error):

        if plot_percentile_lines:
            percentile_text = '$P_{25}=%.3f$\n$P_{50}=%.3f$\n$P_{75}=%.3f$\n$P_{95}=%.3f$\n' % tuple(np.percentile(thiserror, [25,50,75,95]))
        else:
            percentile_text = ''

        txt1 = thissubplot.text(0.98, 0.02, percentile_text + 
                                    '$R_{out}=%.2f$\n' % (np.sum(thiserror<0.1)) +
                                    '$NMAD=%.3f$\n' % (1.4826*np.median(thiserror)) +
                                    r'$\sigma=' + '%.3f$' % np.std(thiserror), 
                                    color = 'white', fontsize = 18, ha = 'right', va = 'bottom', transform = thissubplot.transAxes)
        txt1.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

        txt2 = thissubplot.text(0.02, 0.98, '$N=%i$' % len(thiserror), color = 'white', fontsize = 20, ha = 'left', va = 'top', transform = thissubplot.transAxes)
        txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

        thissubplot.set_xlim(0,1.5)
        thissubplot.set_ylim(0,1.5)

    fig.text(0.5, 0., '$z_{spec}$', fontsize = 30)
    sp1.set_ylabel('$z_{phot}$', fontsize = 30)

    sp2.set_yticklabels([])
    sp3.set_yticklabels([])
    plt.subplots_adjust(wspace = 0)



def plot_pipeline_results_nnc4(run_dir = './tpzruns/default_run/', nn_identifier = 'nnc4_', nn_boundary = 0.65, zconf_boundary = 0., plot_percentile_lines = True, gz_split = False, gz_splitval = None):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc4', 'nnr').replace('_corr', '')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    nnc_class = np.argmax(nnc_results, axis = 1)
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()
    zconf = features_file['zconf'].to_numpy()

    zphot_nnc = np.copy(zphot)
    zphot_nnc[(nnc_class == 2) | (nnc_class == 3)] = (zphot - nnr_results)[(nnc_class == 2) | (nnc_class == 3)] #NNR photoz is better
    goodfit_conf = np.maximum(nnc_results.T[1], nnc_results.T[3])
    is_goodfit = ((nnc_class == 1) | (nnc_class == 3)) & (goodfit_conf > nn_boundary)
    # use_nnr = (nnc_class == 2) | (nnc_class == 3)

    if gz_split != False:
        gz = features_file['g'].to_numpy() - features_file['z'].to_numpy()
        if gz_splitval == None:
            gz_splitval = np.nanmedian(gz)
            print('gz Median: %.2f' % gz_splitval)
        
        if gz_split == 'red':
            gz_colorlim_inds = gz > gz_splitval

        elif gz_split == 'blue':
            gz_colorlim_inds = gz < gz_splitval

        zspec = zspec[gz_colorlim_inds]
        zphot = zphot[gz_colorlim_inds]
        zconf = zconf[gz_colorlim_inds]
        is_goodfit = is_goodfit[gz_colorlim_inds]
        nnr_results = nnr_results[gz_colorlim_inds]

    # nnc_lim_inds = (nnc_results > nn_boundary)
    zconf_lim_inds = (zconf > zconf_boundary)
    # all_lim_inds = nnc_lim_inds & zconf_lim_inds


    fig = plt.figure(figsize = (18,6))
    sp1 = fig.add_subplot(131)
    sp2 = fig.add_subplot(132)
    sp3 = fig.add_subplot(133)

    plot1 = np.histogram2d(zspec[zconf_lim_inds], zphot[zconf_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp1.imshow(np.log10(plot1).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot2 = np.histogram2d(zspec[zconf_lim_inds], (zphot - nnr_results)[zconf_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp2.imshow(np.log10(plot2).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot3 = np.histogram2d(zspec[is_goodfit], zphot_nnc[is_goodfit], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp3.imshow(np.log10(plot3).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    xdata = [zspec, zspec, zspec[is_goodfit]]
    ydata = [zphot, zphot - nnr_results, zphot_nnc[is_goodfit]]
    error = []

    if plot_percentile_lines:
        plot_percentiles = [25, 50, 75, 95]
        xgrid = np.linspace(0,1.5,1000)

        for thissubplot, thisx, thisy in zip([sp1, sp2, sp3], xdata, ydata):

            thiserror = np.abs(thisy - thisx)/ (1 + thisx)
            error.append(thiserror[np.isfinite(thiserror)])

            for this_percentile in plot_percentiles:

                ind = np.argmin(np.abs(np.percentile(thiserror, this_percentile) - thiserror))
                thissubplot.plot(xgrid, xgrid - (1+xgrid)*thiserror[ind], color = 'k')
                thissubplot.plot(xgrid, xgrid + (1+xgrid)*thiserror[ind], color = 'k')


    for thissubplot, thiserror in zip([sp1, sp2, sp3], error):

        txt1 = thissubplot.text(0.98, 0.02, '$P_{25}=%.3f$\n$P_{50}=%.3f$\n$P_{75}=%.3f$\n$P_{95}=%.3f$\n' % tuple(np.percentile(thiserror, [25,50,75,95])) + 
                                    '$NMAD=%.3f$\n' % (1.4826*np.median(thiserror)) +
                                    r'$\sigma=' + '%.3f$' % np.std(thiserror), 
                                    color = 'white', fontsize = 18, ha = 'right', va = 'bottom', transform = thissubplot.transAxes)
        txt1.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

        txt2 = thissubplot.text(0.02, 0.98, '$N=%i$' % len(thiserror), color = 'white', fontsize = 20, ha = 'left', va = 'top', transform = thissubplot.transAxes)
        txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

        thissubplot.set_xlim(0,1.5)
        thissubplot.set_ylim(0,1.5)

    fig.text(0.5, 0., '$z_{spec}$', fontsize = 30)
    sp1.set_ylabel('$z_{phot}$', fontsize = 30)

    sp2.set_yticklabels([])
    sp3.set_yticklabels([])
    plt.subplots_adjust(wspace = 0)




def plot_trainsize_var(outlier_val = 0.15, target_val = np.arange(0.02, 0.1, 0.01)):

    test_results = sorted(glob('./tpzruns/MatchCOSMOS2015_TrainMod/*/output/results/tpzrun.0.mlz'))
    app_results = sorted(glob('./tpzruns/MatchCOSMOS2015_TrainMod/*/output/results/tpzrun.1.mlz'))

    trainfrac = np.array([float((this_file[:-27].split('/')[-2]))/100. for this_file in test_results]) * 147435
    colors = [mpl.cm.RdYlBu(int(place)) for place in (target_val/max(target_val) * mpl.cm.RdYlBu.N)]

    sigma_nfrac = []
    outlier_nfrac = []
    nmad_nfrac = []

    for testfile, appfile in tqdm(zip(test_results, app_results), total = len(test_results)):

        run_dir = testfile[:-27]
        nnc_folder = run_dir + 'nnc_epoch1000/'

        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

        good_inds = np.isfinite(nnc_results)

        zspec = features_file['specz'].to_numpy()[good_inds]
        zphot = features_file['zphot'].to_numpy()[good_inds]
        zphot_err = features_file['zerr'].to_numpy()[good_inds]
        zconf = features_file['zconf'].to_numpy()[good_inds]
        nnc_results = nnc_results[good_inds]
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
        
        nnsorted_predictions, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, z_err, z_err_sigmas)))))

        tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

        n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

        if len(nnsorted_zerr) > 10000:
            inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
        else:
            inds = np.arange(len(nnsorted_zerr))

        # sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
        # nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
        # outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

        sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
        nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
        outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))

        sigma_nfrac.append(np.interp(target_val, sigma_nn[::-1], n_frac[::-1]))
        outlier_nfrac.append(np.interp(target_val, outlier_nn[::-1], n_frac[::-1]))
        nmad_nfrac.append(np.interp(target_val, nmad_nn[::-1], n_frac[::-1]))

    sigma_nfrac = np.array(sigma_nfrac).T
    outlier_nfrac = np.array(outlier_nfrac).T
    nmad_nfrac = np.array(nmad_nfrac).T


    fig = plt.figure(figsize = (24, 8))
    sp_outlier = fig.add_subplot(131)
    sp_nmad = fig.add_subplot(132)
    sp_sigma = fig.add_subplot(133)

    for this_metric, this_sp in zip(['outlier', 'NMAD', 'sigma'], [sp_outlier, sp_nmad, sp_sigma]):


        if this_metric == 'sigma':
            score_y = sigma_nfrac
            score_label = r'$\sigma_z$'
        elif this_metric == 'outlier':
            score_y = outlier_nfrac
            score_label = '$f_{out}$'
        elif this_metric == 'NMAD':
            score_y = nmad_nfrac
            score_label = 'NMAD'

        this_sp.text(0.98, 0.98, score_label, fontsize = 24, ha = 'right', va = 'top', transform = this_sp.transAxes)

        for thisy, thiscolor, thistarget in zip(score_y, colors, target_val):

            if this_metric != 'NMAD' or thistarget <= 0.05:
                this_sp.plot(trainfrac, thisy, color = thiscolor, linewidth = 2, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
                thistxt = this_sp.text(trainfrac[-1] + 5000, thisy[-1],'%.2f' % thistarget, color = thiscolor, fontsize = 16, ha = 'left')
                thistxt.set_path_effects([pe.withStroke(linewidth=1, foreground='k')])

    plt.subplots_adjust(wspace = 0.)



    sp_nmad.set_xlabel(r'Training Sample Size')
    sp_outlier.set_ylabel('f$_{sample}$')
    sp_outlier.set_xlim(0, 1.15 * 147435)
    sp_nmad.set_xlim(0, 1.15 * 147435)
    sp_sigma.set_xlim(0, 1.15 * 147435)

    sp_outlier.set_ylim(0, 1)
    sp_nmad.set_ylim(0, 1)
    sp_sigma.set_ylim(0, 1)

    sp_nmad.set_yticklabels([])
    sp_sigma.set_yticklabels([])

    fig.canvas.draw()

    sp_outlier_ticks = [thistick.get_text() for thistick in sp_outlier.get_xticklabels()]
    sp_outlier_ticks[-2] = ''
    sp_outlier.set_xticklabels(sp_outlier_ticks)

    sp_nmad_ticks = [thistick.get_text() for thistick in sp_nmad.get_xticklabels()]
    sp_nmad_ticks[-2] = ''
    sp_nmad.set_xticklabels(sp_nmad_ticks)



def plot_trainsize_var_old(outlier_val = 0.1, metric = 'sigma', abs_num = True):

    test_results = sorted(glob('./tpzruns/MatchCOSMOS2015_TrainMod/*/output/results/tpzrun.0.mlz'))
    app_results = sorted(glob('./tpzruns/MatchCOSMOS2015_TrainMod/*/output/results/tpzrun.1.mlz'))

    fig = plt.figure(figsize = (12, 8))
    gs = GridSpec(3, 4, figure=fig)
    sp_nmad = fig.add_subplot(gs[0,0])
    sp_outlier = fig.add_subplot(gs[1,0])
    sp_sigma = fig.add_subplot(gs[2,0])
    sp_score = fig.add_subplot(gs[:,1:])

    score_x = []
    score_y = []

    trainfrac = np.array([float((this_file[:-27].split('/')[-2]))/100. for this_file in test_results])
    colors = [mpl.cm.RdYlBu(int(place)) for place in (trainfrac * mpl.cm.RdYlBu.N)]

    sigma_score = []
    outlier_score = []
    nmad_score = []

    for testfile, appfile, thiscolor in tqdm(zip(test_results, app_results, colors), total = len(test_results)):

        run_dir = testfile[:-27]
        nnc_folder = run_dir + 'nnc_epoch1000/'

        trainfrac = float((run_dir.split('/')[-2]))/100.
        if abs_num:
            trainfrac = trainfrac * 147435
        score_x.append(trainfrac)

        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

        # z_tpz, zconf, zerr = np.loadtxt(directory + 'output/results/' + filename + '.0.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        good_inds = np.isfinite(nnc_results)

        zspec = features_file['specz'].to_numpy()[good_inds]
        zphot = features_file['zphot'].to_numpy()[good_inds]
        zphot_err = features_file['zerr'].to_numpy()[good_inds]
        zconf = features_file['zconf'].to_numpy()[good_inds]
        nnc_results = nnc_results[good_inds]
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
        
        nnsorted_predictions, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, z_err, z_err_sigmas)))))

        tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

        n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

        if len(nnsorted_zerr) > 10000:
            inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
        else:
            inds = np.arange(len(nnsorted_zerr))

        sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
        nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
        outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

        sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
        nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
        outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))

        # sp_nmad.plot(n_frac, nmad_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_nmad.plot(n_frac, nmad_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        # sp_outlier.plot(n_frac, outlier_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_outlier.plot(n_frac, outlier_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        # sp_sigma.plot(n_frac, sigma_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_sigma.plot(n_frac, sigma_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)


        sp_nmad.plot(n_frac, nmad_nn, color = thiscolor, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
        sp_outlier.plot(n_frac, outlier_nn, color = thiscolor, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
        sp_sigma.plot(n_frac, sigma_nn, color = thiscolor, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])

        sigma_score.append(np.interp(0.33, n_frac[::-1], sigma_nn[::-1]))
        outlier_score.append(np.interp(0.33, n_frac[::-1], outlier_nn[::-1]))
        nmad_score.append(np.interp(0.33, n_frac[::-1], nmad_nn[::-1]))
    
    if metric == 'sigma':
        sp_sigma.plot([.3,.3], [0,1], color = 'k', linestyle = '--', transform = sp_sigma.get_xaxis_transform())
        score_y = sigma_score
        score_label = r'$\sigma$'
    elif metric == 'outlier':
        sp_outlier.plot([.3,.3], [0,1], color = 'k', linestyle = '--', transform = sp_outlier.get_xaxis_transform())
        score_y = outlier_score
        score_label = 'Outlier Frac'
    elif metric == 'NMAD':
        sp_nmad.plot([.3,.3], [0,1], color = 'k', linestyle = '--', transform = sp_nmad.get_xaxis_transform())
        score_y = nmad_score
        score_label = 'NMAD'

    sp_score.plot(score_x, score_y, color = 'k', linewidth = 2, zorder = 1)
    [sp_score.scatter(thisx, thisy, s = 50, zorder = 2, c = thiscolor, edgecolor = 'k') for thisx, thisy, thiscolor in zip(score_x, score_y, colors)]

    plt.subplots_adjust(wspace = 0, hspace = 0)
    sp_sigma.set_xlabel('N$_{frac}$')
    sp_nmad.set_ylabel('NMAD', fontsize = 16)
    sp_outlier.set_ylabel('Outlier Fraction', fontsize = 16)
    # sp_outlier.set_ylabel(r'Outlier Fraction $\left( \frac{\Delta z}{1+z} > ' + '%.2f' % outlier_val + r'\right)$', fontsize = 16)
    sp_sigma.set_ylabel(r'$\sigma$', fontsize = 16)

    sp_score.yaxis.tick_right()
    sp_score.yaxis.set_label_position('right')
    sp_score.set_ylabel(score_label)
    if not abs_num:
        sp_score.set_xlabel('N$_{frac}$')
    else:
        sp_score.set_xlabel('Training Sample Size')

    print('Boundaries:')
    print(score_x)
    print('')

    for metric_name, this_metric in zip(['sigma', 'outlier', 'NMAD'], [sigma_score, outlier_score, nmad_score]):
        print(metric_name + ':')
        print(this_metric)
        print('')





def plot_acc_var(outlier_val = 0.15, target_val = np.arange(0.02, 0.1, 0.01)):

    test_results = sorted(glob('./tpzruns/MatchCOSMOS2015_0*/output/results/tpzrun.0.mlz'))
    app_results = sorted(glob('./tpzruns/MatchCOSMOS2015_0*/output/results/tpzrun.1.mlz'))

    colors = [mpl.cm.winter_r(int(place)) for place in np.linspace(0, mpl.cm.winter_r.N, len(test_results))]

    accuracy = []
    sigma_nfrac = []
    outlier_nfrac = []
    nmad_nfrac = []

    for testfile, appfile in tqdm(zip(test_results, app_results), total = len(test_results)):

        run_dir = testfile[:-27]
        nnc_folder = run_dir + 'nnc_epoch1000/'

        acc_cutoff = float((run_dir.split('/')[-2])[-2:])/100.
        accuracy.append(acc_cutoff)

        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

        good_inds = np.isfinite(nnc_results)

        zspec = features_file['specz'].to_numpy()[good_inds]
        zphot = features_file['zphot'].to_numpy()[good_inds]
        zphot_err = features_file['zerr'].to_numpy()[good_inds]
        zconf = features_file['zconf'].to_numpy()[good_inds]
        nnc_results = nnc_results[good_inds]
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
        
        nnsorted_predictions, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, z_err, z_err_sigmas)))))

        tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

        n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

        if len(nnsorted_zerr) > 10000:
            inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
        else:
            inds = np.arange(len(nnsorted_zerr))

        # sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
        # nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
        # outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

        sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
        nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
        outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))
    
        sigma_nfrac.append(np.interp(target_val, sigma_nn[::-1], n_frac[::-1]))
        outlier_nfrac.append(np.interp(target_val, outlier_nn[::-1], n_frac[::-1]))
        nmad_nfrac.append(np.interp(target_val, nmad_nn[::-1], n_frac[::-1]))

    sigma_nfrac = np.array(sigma_nfrac).T
    outlier_nfrac = np.array(outlier_nfrac).T
    nmad_nfrac = np.array(nmad_nfrac).T

    fig = plt.figure(figsize = (24, 8))
    sp_outlier = fig.add_subplot(131)
    sp_nmad = fig.add_subplot(132)
    sp_sigma = fig.add_subplot(133)

    for this_metric, this_sp in zip(['outlier', 'NMAD', 'sigma'], [sp_outlier, sp_nmad, sp_sigma]):


        if this_metric == 'sigma':
            score_y = sigma_nfrac
            score_label = r'$\sigma_z$'
        elif this_metric == 'outlier':
            score_y = outlier_nfrac
            score_label = '$f_{out}$'
        elif this_metric == 'NMAD':
            score_y = nmad_nfrac
            score_label = 'NMAD'

        this_sp.text(0.98, 0.98, score_label, fontsize = 24, ha = 'right', va = 'top', transform = this_sp.transAxes)

        for thisy, thiscolor, thistarget in zip(score_y, colors, target_val):

            if this_metric != 'NMAD' or thistarget <= 0.05:
                this_sp.plot(accuracy, thisy, color = thiscolor, linewidth = 2, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
                thistxt = this_sp.text(accuracy[-1] + 0.005, thisy[-1],'%.2f' % thistarget, color = thiscolor, fontsize = 16, ha = 'left')
                thistxt.set_path_effects([pe.withStroke(linewidth=1, foreground='k')])

    plt.subplots_adjust(wspace = 0.)


    sp_nmad.set_xlabel(r'NNC Training Boundary $\left(\frac{\Delta z}{1+z}\right)$')
    sp_outlier.set_ylabel('f$_{sample}$')
    sp_outlier.set_xlim(0.035, 0.168)
    sp_nmad.set_xlim(0.035, 0.168)
    sp_sigma.set_xlim(0.035, 0.168)

    sp_outlier.set_ylim(0, 1)
    sp_nmad.set_ylim(0, 1)
    sp_sigma.set_ylim(0, 1)

    sp_nmad.set_yticklabels([])
    sp_sigma.set_yticklabels([])



def plot_acc_var_old(outlier_val = 0.15, metric = 'sigma'):

    test_results = sorted(glob('./tpzruns/MatchCOSMOS2015_0*/output/results/tpzrun.0.mlz'))
    app_results = sorted(glob('./tpzruns/MatchCOSMOS2015_0*/output/results/tpzrun.1.mlz'))

    fig = plt.figure(figsize = (12, 8))
    gs = GridSpec(3, 4, figure=fig)
    sp_nmad = fig.add_subplot(gs[0,0])
    sp_outlier = fig.add_subplot(gs[1,0])
    sp_sigma = fig.add_subplot(gs[2,0])
    sp_score = fig.add_subplot(gs[:,1:])

    colors = [mpl.cm.winter_r(int(place)) for place in np.linspace(0, mpl.cm.winter_r.N, len(test_results))]

    score_x = []
    sigma_score = []
    outlier_score = []
    nmad_score = []

    for testfile, appfile, thiscolor in tqdm(zip(test_results, app_results, colors), total = len(test_results)):

        run_dir = testfile[:-27]
        nnc_folder = run_dir + 'nnc_epoch1000/'

        acc_cutoff = float((run_dir.split('/')[-2])[-2:])/100.
        score_x.append(acc_cutoff)

        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

        # z_tpz, zconf, zerr = np.loadtxt(directory + 'output/results/' + filename + '.0.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        good_inds = np.isfinite(nnc_results)

        zspec = features_file['specz'].to_numpy()[good_inds]
        zphot = features_file['zphot'].to_numpy()[good_inds]
        zphot_err = features_file['zerr'].to_numpy()[good_inds]
        zconf = features_file['zconf'].to_numpy()[good_inds]
        nnc_results = nnc_results[good_inds]
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
        
        nnsorted_predictions, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, z_err, z_err_sigmas)))))

        tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

        n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

        if len(nnsorted_zerr) > 10000:
            inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
        else:
            inds = np.arange(len(nnsorted_zerr))

        sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
        nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
        outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

        sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
        nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
        outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))

        # sp_nmad.plot(n_frac, nmad_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_nmad.plot(n_frac, nmad_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        # sp_outlier.plot(n_frac, outlier_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_outlier.plot(n_frac, outlier_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        # sp_sigma.plot(n_frac, sigma_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_sigma.plot(n_frac, sigma_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        sp_nmad.plot(n_frac, nmad_nn, color = thiscolor, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
        sp_outlier.plot(n_frac, outlier_nn, color = thiscolor, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
        sp_sigma.plot(n_frac, sigma_nn, color = thiscolor, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
    
        sigma_score.append(np.interp(0.33, n_frac[::-1], sigma_nn[::-1]))
        outlier_score.append(np.interp(0.33, n_frac[::-1], outlier_nn[::-1]))
        nmad_score.append(np.interp(0.33, n_frac[::-1], nmad_nn[::-1]))

    if metric == 'sigma':
        sp_sigma.plot([.3,.3], [0,1], color = 'k', linestyle = '--', transform = sp_sigma.get_xaxis_transform())
        score_y = sigma_score
        score_label = r'$\sigma$'
    elif metric == 'outlier':
        sp_outlier.plot([.3,.3], [0,1], color = 'k', linestyle = '--', transform = sp_outlier.get_xaxis_transform())
        score_y = outlier_score
        score_label = 'Outlier Frac'
    elif metric == 'NMAD':
        sp_nmad.plot([.3,.3], [0,1], color = 'k', linestyle = '--', transform = sp_nmad.get_xaxis_transform())
        score_y = nmad_score
        score_label = 'NMAD'

    sp_score.plot(score_x, score_y, color = 'k', linewidth = 2, zorder = 1)
    [sp_score.scatter(thisx, thisy, s = 50, zorder = 2, c = thiscolor, edgecolor = 'k') for thisx, thisy, thiscolor in zip(score_x, score_y, colors)]

    plt.subplots_adjust(wspace = 0, hspace = 0)
    sp_sigma.set_xlabel('N$_{frac}$')
    sp_nmad.set_ylabel('NMAD', fontsize = 16)
    sp_outlier.set_ylabel('Outlier Fraction', fontsize = 16)
    # sp_outlier.set_ylabel(r'Outlier Fraction $\left( \frac{\Delta z}{1+z} > ' + '%.2f' % outlier_val + r'\right)$', fontsize = 16)
    sp_sigma.set_ylabel(r'$\sigma$', fontsize = 16)

    sp_score.yaxis.tick_right()
    sp_score.yaxis.set_label_position('right')
    sp_score.set_ylabel(score_label)
    sp_score.set_xlabel(r'NNC Boundary [$\Delta z/(1+z)$]')

    sp_nmad.set_xticklabels([])
    sp_outlier.set_xticklabels([])

    print('Boundaries:')
    print(score_x)
    print('')

    for metric_name, this_metric in zip(['sigma', 'outlier', 'NMAD'], [sigma_score, outlier_score, nmad_score]):
        print(metric_name + ':')
        print(this_metric)
        print('')




def plot_acc_var2(outlier_val = 0.15, metric = 'sigma', target_val = np.arange(0.03, 0.1, 0.01)):

    test_results = sorted(glob('./tpzruns/MatchCOSMOS2015_0*/output/results/tpzrun.0.mlz'))
    app_results = sorted(glob('./tpzruns/MatchCOSMOS2015_0*/output/results/tpzrun.1.mlz'))

    fig = plt.figure(figsize = (8, 8))
    sp = fig.add_subplot(111)

    colors = [mpl.cm.winter_r(int(place)) for place in np.linspace(0, mpl.cm.winter_r.N, len(test_results))]

    accuracy = []
    sigma_nfrac = []
    outlier_nfrac = []
    nmad_nfrac = []

    for testfile, appfile in tqdm(zip(test_results, app_results), total = len(test_results)):

        run_dir = testfile[:-27]
        nnc_folder = run_dir + 'nnc_epoch1000/'

        acc_cutoff = float((run_dir.split('/')[-2])[-2:])/100.
        accuracy.append(acc_cutoff)

        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

        good_inds = np.isfinite(nnc_results)

        zspec = features_file['specz'].to_numpy()[good_inds]
        zphot = features_file['zphot'].to_numpy()[good_inds]
        zphot_err = features_file['zerr'].to_numpy()[good_inds]
        zconf = features_file['zconf'].to_numpy()[good_inds]
        nnc_results = nnc_results[good_inds]
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
        
        nnsorted_predictions, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, z_err, z_err_sigmas)))))

        tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

        n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

        if len(nnsorted_zerr) > 10000:
            inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
        else:
            inds = np.arange(len(nnsorted_zerr))

        # sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
        # nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
        # outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

        sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
        nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
        outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))
    
        sigma_nfrac.append(np.interp(target_val, sigma_nn[::-1], n_frac[::-1]))
        outlier_nfrac.append(np.interp(target_val, outlier_nn[::-1], n_frac[::-1]))
        nmad_nfrac.append(np.interp(target_val, nmad_nn[::-1], n_frac[::-1]))

    sigma_nfrac = np.array(sigma_nfrac).T
    outlier_nfrac = np.array(outlier_nfrac).T
    nmad_nfrac = np.array(nmad_nfrac).T

    if metric == 'sigma':
        score_y = sigma_nfrac
        score_label = r'$\sigma$'
    elif metric == 'outlier':
        score_y = outlier_nfrac
        score_label = '$f_{out}$'
    elif metric == 'NMAD':
        score_y = nmad_nfrac
        score_label = 'NMAD'

    for thisy, thiscolor, thistarget in zip(score_y, colors, target_val):

        sp.plot(accuracy, thisy, color = thiscolor, linewidth = 2, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
        thistxt = sp.text(accuracy[-1] + 0.005, thisy[-1], score_label + ' = %.2f' % thistarget, color = thiscolor, fontsize = 16, ha = 'left')
        thistxt.set_path_effects([pe.withStroke(linewidth=1, foreground='k')])

    sp.set_xlabel(r'NNC Training Boundary $\left(\frac{\Delta z}{1+z}\right)$')
    sp.set_ylabel('N$_{frac}$')
    sp.set_xlim(0.035, 0.185)





def plot_nnr_comparison(nonnr_dir = './tpzruns/MatchCOSMOS2015/', nnr_dir = './tpzruns/MatchCOSMOS2015_nnr/', outlier_val = 0.15):

    fig = plt.figure(figsize = (24,8))

    sp_nmad = fig.add_subplot(131)
    sp_outlier = fig.add_subplot(132)
    sp_sigma = fig.add_subplot(133)


    for run_dir, thiscolor, thisname in zip([nonnr_dir, nnr_dir], ['C0', 'C2'], ['NNC', 'NNR+NNC']):

        if thisname[:3] == 'NNR':
            nnc_folder = run_dir + 'nnc_epoch1000_corr/'
        else:
            nnc_folder = run_dir + 'nnc_epoch1000/'

        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

        good_inds = np.isfinite(nnc_results)

        zspec = features_file['specz'].to_numpy()[good_inds]
        zphot = features_file['zphot'].to_numpy()[good_inds]
        zphot_err = features_file['zerr'].to_numpy()[good_inds]
        zconf = features_file['zconf'].to_numpy()[good_inds]
        nnc_results = nnc_results[good_inds]
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
        
        nnsorted_predictions, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, z_err, z_err_sigmas)))))

        tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

        n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

        if len(nnsorted_zerr) > 10000:
            inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
        else:
            inds = np.arange(len(nnsorted_zerr))

        sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
        nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
        outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

        sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
        nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
        outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))

        # sp_nmad.plot(n_frac, nmad_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_nmad.plot(n_frac, nmad_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        # sp_outlier.plot(n_frac, outlier_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_outlier.plot(n_frac, outlier_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        # sp_sigma.plot(n_frac, sigma_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_sigma.plot(n_frac, sigma_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        sp_nmad.plot(nmad_nn, n_frac,  color = thiscolor, label = thisname)
        sp_outlier.plot(outlier_nn, n_frac,  color = thiscolor, label = thisname)
        sp_sigma.plot(sigma_nn, n_frac,  color = thiscolor, label = thisname)


    plt.subplots_adjust(wspace = 0)
    sp_nmad.set_ylabel('$f_{sample}$')
    sp_nmad.set_xlabel('NMAD')
    sp_outlier.set_xlabel(r'Outlier Fraction $\left( \frac{\Delta z}{1+z} > ' + '%.2f' % outlier_val + r'\right)$')
    sp_sigma.set_xlabel(r'$\sigma_z$')

    sp_nmad.set_ylim(0,1)
    sp_outlier.set_ylim(0,1)
    sp_sigma.set_ylim(0,1)

    sp_outlier.set_yticklabels([])
    sp_sigma.set_yticklabels([])

    # sp_nmad.set_xlim(0,1)
    # sp_outlier.set_xlim(0,1)
    # sp_sigma.set_xlim(0,1)

    sp_nmad.legend(loc = 'upper left', fontsize = 20)




def plot_triplet_comparison(trip_dir = './tpzruns/MatchCOSMOS2015/', notrip_dir = './tpzruns/MatchCOSMOS2015_notrip/', outlier_val = 0.15):

    fig = plt.figure(figsize = (24,8))

    sp_nmad = fig.add_subplot(131)
    sp_outlier = fig.add_subplot(132)
    sp_sigma = fig.add_subplot(133)


    for run_dir, thiscolor, thisname in zip([notrip_dir, trip_dir], ['C4', 'C0'], ['Triplets', 'No Triplets']):

        nnc_folder = run_dir + 'nnc_epoch1000/'

        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

        good_inds = np.isfinite(nnc_results)

        zspec = features_file['specz'].to_numpy()[good_inds]
        zphot = features_file['zphot'].to_numpy()[good_inds]
        zphot_err = features_file['zerr'].to_numpy()[good_inds]
        zconf = features_file['zconf'].to_numpy()[good_inds]
        nnc_results = nnc_results[good_inds]
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
        
        nnsorted_predictions, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, z_err, z_err_sigmas)))))

        tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

        n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

        if len(nnsorted_zerr) > 10000:
            inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
        else:
            inds = np.arange(len(nnsorted_zerr))

        sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
        nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
        outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

        sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
        nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
        outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))

        # sp_nmad.plot(n_frac, nmad_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_nmad.plot(n_frac, nmad_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        # sp_outlier.plot(n_frac, outlier_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_outlier.plot(n_frac, outlier_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        # sp_sigma.plot(n_frac, sigma_tpz, label = r'TPZ $\sigma$', color = 'k', alpha = (np.log10(trainfrac)+2)/2)
        # sp_sigma.plot(n_frac, sigma_nn, label = 'NNC', color = 'C0', alpha = (np.log10(trainfrac)+2)/2)

        sp_nmad.plot(nmad_nn, n_frac,  color = thiscolor, label = thisname)
        sp_outlier.plot(outlier_nn, n_frac,  color = thiscolor, label = thisname)
        sp_sigma.plot(sigma_nn, n_frac,  color = thiscolor, label = thisname)


    plt.subplots_adjust(wspace = 0)
    sp_nmad.set_ylabel('$f_{sample}$')
    sp_nmad.set_xlabel('NMAD')
    sp_outlier.set_xlabel(r'Outlier Fraction $\left( \frac{\Delta z}{1+z} > ' + '%.2f' % outlier_val + r'\right)$')
    sp_sigma.set_xlabel(r'$\sigma_z$')

    sp_nmad.set_ylim(0,1)
    sp_outlier.set_ylim(0,1)
    sp_sigma.set_ylim(0,1)

    sp_outlier.set_yticklabels([])
    sp_sigma.set_yticklabels([])

    # sp_nmad.set_xlim(0,1)
    # sp_outlier.set_xlim(0,1)
    # sp_sigma.set_xlim(0,1)

    sp_nmad.legend(loc = 'upper left', fontsize = 20)






def plot_app_roc(run_dir = './tpzruns/median_shift/', nn_identifier = 'nnc_'):

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()

    is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

    sorted_predictions, sorted_is_goodfit = np.array(list(zip(*sorted(zip(nnc_results, is_goodfit)))))
    sorted_is_goodfit = sorted_is_goodfit.astype(bool)

    tpr = (np.cumsum(sorted_is_goodfit[::-1])/float(np.sum(sorted_is_goodfit)))[::-1]
    fpr = (np.cumsum(~sorted_is_goodfit[::-1])/float(np.sum(~sorted_is_goodfit)))[::-1]

    auc = -np.trapz(tpr, fpr)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    sp.plot(fpr, tpr)
    sp.text(0.98, 0.02, '$AUC = %.2f$' % auc, fontsize = 20, ha = 'right', va = 'bottom')
    sp.set_xlabel('False Positive Rate')
    sp.set_ylabel('True Positive Rate')
    sp.set_xlim(0,1)
    sp.set_ylim(0,1)
    for cutoff in np.arange(0.1,1,.1):
        thisx = np.interp(cutoff, sorted_predictions, fpr)
        thisy = np.interp(cutoff, sorted_predictions, tpr)
        sp.annotate('%.1f\n(%.2f, %.2f)'%(cutoff, thisx, thisy), (thisx, thisy), (thisx+.05, thisy- 0.05), arrowprops = dict(arrowstyle = '-', ec = 'k'))



def plot_app_roc_nnc2(run_dir = './tpzruns/default_run/', nn_identifier = 'nnc2_'):

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc2', 'nnr').replace('_corr', '')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    good_inds = (np.all(np.isfinite(nnc_results), axis = 1) & np.isfinite(nnr_results))

    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    nnr_results = nnr_results[good_inds]

    nnc_class = np.argmax(nnc_results, axis = 1)

    zphot_nnc = np.copy(zphot)
    zphot_nnc[nnc_class == 1] = (zphot - nnr_results)[nnc_class == 1] #NNR photoz is better
    goodfit_conf = np.maximum(*nnc_results.T)

    is_goodfit = (np.abs(zphot_nnc - zspec) < (1. + zspec)*0.02)

    sorted_predictions, sorted_is_goodfit = np.array(list(zip(*sorted(zip(goodfit_conf, is_goodfit)))))
    sorted_is_goodfit = sorted_is_goodfit.astype(bool)

    tpr = (np.cumsum(sorted_is_goodfit[::-1])/float(np.sum(sorted_is_goodfit)))[::-1]
    fpr = (np.cumsum(~sorted_is_goodfit[::-1])/float(np.sum(~sorted_is_goodfit)))[::-1]

    auc = -np.trapz(tpr, fpr)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    sp.plot(fpr, tpr)
    sp.text(0.98, 0.02, '$AUC = %.2f$' % auc, fontsize = 20, ha = 'right', va = 'bottom')
    sp.set_xlabel('False Positive Rate')
    sp.set_ylabel('True Positive Rate')
    sp.set_xlim(0,1)
    sp.set_ylim(0,1)
    for cutoff in np.arange(0.1,1,.1):
        thisx = np.interp(cutoff, sorted_predictions, fpr)
        thisy = np.interp(cutoff, sorted_predictions, tpr)
        sp.annotate('%.1f\n(%.2f, %.2f)'%(cutoff, thisx, thisy), (thisx, thisy), (thisx+.05, thisy- 0.05), arrowprops = dict(arrowstyle = '-', ec = 'k'))



def plot_app_precision_recall(run_dir = './tpzruns/median_shift/', nn_identifier = 'nnc_'):

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()

    is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

    sorted_predictions, sorted_is_goodfit = np.array(list(zip(*sorted(zip(nnc_results, is_goodfit)))))
    sorted_is_goodfit = sorted_is_goodfit.astype(bool)

    tpr = (np.cumsum(sorted_is_goodfit[::-1])/float(np.sum(sorted_is_goodfit)))[::-1]
    fpr = (np.cumsum(~sorted_is_goodfit[::-1])/float(np.sum(~sorted_is_goodfit)))[::-1]

    precision = (np.cumsum(sorted_is_goodfit[::-1])/(np.arange(len(sorted_is_goodfit[::-1]), dtype = float) + 1))[::-1]
    recall = tpr

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    auc = -np.trapz(precision, recall)
    sp.text(0.98, 0.02, '$AUC = %.2f$' % auc, fontsize = 20, ha = 'right', va = 'bottom', transform = sp.transAxes)
    sp.plot(recall, precision)
    sp.set_xlabel('Recall')
    sp.set_ylabel('Precision')
    sp.set_xlim(0,1)
    sp.set_ylim(0,1)
    for cutoff in np.arange(0.1,1,.1):
        thisx = np.interp(cutoff, sorted_predictions, recall)
        thisy = np.interp(cutoff, sorted_predictions, precision)
        sp.annotate('%.1f\n(%.2f, %.2f)'%(cutoff, thisx, thisy), (thisx, thisy), (thisx-.05, thisy- 0.05), arrowprops = dict(arrowstyle = '-', ec = 'k'))




def plot_zfrac(run_dir = './tpzruns/MatchCOSMOS2015/', nn_identifier = 'nnc_', ztype = 'phot', zbin_edges = np.arange(0, 1.5, 0.1)):

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc', 'nnr')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
    training_features = read_csv(glob(run_dir + '*.nnc_train')[0], delim_whitespace = True, comment = '#')
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    # nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    good_inds = np.isfinite(nnc_results)

    zspec = features_file['specz'].to_numpy()[good_inds]
    zspec_training = training_features['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    # nnr_results = nnr_results[good_inds]

    # fig = plt.figure(figsize = (8,8))
    # sp = fig.add_subplot(111)

    fig, (sp1, sp2) = plt.subplots(2,1,gridspec_kw = {'height_ratios':[4,1]}, figsize = (8,10))

    x_vals = zbin_edges[:-1] + np.diff(zbin_edges)

    if ztype == 'phot':
        zvals = zphot
    elif ztype == 'spec':
        zvals = zspec

    for x, this_nn_boundary in enumerate(np.append(np.arange(0.1, 1, 0.1), np.array([0.95]))):

        thiscurve = np.array([sum((zvals >= thisbin_lo) & (zvals < thisbin_hi) & (nnc_results > this_nn_boundary))/float(sum((zvals >= thisbin_lo) & (zvals < thisbin_hi))) for thisbin_lo, thisbin_hi in zip(zbin_edges[:-1], zbin_edges[1:])])

        good_inds = np.isfinite(thiscurve)

        thisx = x_vals[good_inds]
        thisy = thiscurve[good_inds]

        sp1.plot(thisx, thisy, color = 'k', linewidth = 4)
        sp1.plot(thisx, thisy, color = plt.cm.RdYlBu(int(float(x)*plt.cm.RdYlBu.N/9.)), linewidth = 3, label = '%.1f'%this_nn_boundary)

        thistxt = sp1.text(thisx[-1] + 0.1, thisy[-1], '%g' % this_nn_boundary, color = plt.cm.RdYlBu(int(float(x)*plt.cm.RdYlBu.N/9.)), fontsize = 16, ha = 'left')
        thistxt.set_path_effects([pe.withStroke(linewidth=1, foreground='k')])

    sp2.hist(zspec_training, histtype = 'step', bins = zbin_edges, label = '$z_{train}$', density = True)
    sp2.hist(zphot, histtype = 'step', bins = zbin_edges, label = '$z_{test,TPZ}$', density = True)

    sp2.legend(loc = 'upper right')

    sp2.set_xlabel('Redshift (z)')
    sp2.set_ylabel('N/N$_{tot}$')
    sp1.set_ylabel('$f_{out}$')
    
    sp1.set_xticklabels([])
    plt.subplots_adjust(hspace = 0)

    sp1.set_xlim(0, 1.7)
    sp2.set_xlim(0, 1.7)



def plot_z_dist(run_dir = './tpzruns/MatchCOSMOS2015/', nn_identifier = 'nnc_'):

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc', 'nnr')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    # nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    good_inds = np.isfinite(nnc_results)

    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    h_phot, bin_edges = np.histogram(zphot, range = (0, 1.5), bins = 70, normed = True)
    h_spec, bin_edges = np.histogram(zspec, range = (0, 1.5), bins = 70, normed = True)

    sp.step(bin_edges[:-1], h_phot, label = '$z_{TPZ}$')
    sp.step(bin_edges[:-1], h_spec, label = '$z_{COSMOS2015}$')

    sp.set_xlabel('Redshift (z)')
    sp.set_ylabel('$N/N_{tot}$')
    sp.legend(loc = 'upper right')



def plot_zfrac_nnc2(run_dir = './tpzruns/cosmos_closest_match_07/', nn_identifier = 'nnc2_', zbin_edges = np.append(np.arange(0, 1.5, 0.1), 1.5)):

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc2', 'nnr').replace('_corr', '')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    good_inds = (np.all(np.isfinite(nnc_results), axis = 1) & np.isfinite(nnr_results))

    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    nnr_results = nnr_results[good_inds]

    nnc_class = np.argmax(nnc_results, axis = 1)

    zphot_nnc = np.copy(zphot)
    zphot_nnc[nnc_class == 1] = (zphot - nnr_results)[nnc_class == 1] #NNR photoz is better
    goodfit_conf = np.maximum(*nnc_results.T)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    x_vals = zbin_edges[:-1] + np.diff(zbin_edges)

    for x, this_nn_boundary in enumerate(np.arange(0, 1, 0.1)):

        thiscurve = [sum((zphot_nnc >= thisbin_lo) & (zphot_nnc < thisbin_hi) & (goodfit_conf > this_nn_boundary))/float(sum((zphot_nnc >= thisbin_lo) & (zphot_nnc < thisbin_hi))) for thisbin_lo, thisbin_hi in zip(zbin_edges[:-1], zbin_edges[1:])]

        sp.plot(x_vals, thiscurve, color = 'k', linewidth = 4)
        sp.plot(x_vals, thiscurve, color = plt.cm.RdYlBu(int(float(x)*plt.cm.RdYlBu.N/10.)), linewidth = 3, label = '%.1f'%this_nn_boundary)

    sp.set_xlabel('Redshift (z)')
    sp.set_ylabel('$N_{frac}$')




def plot_sigma_nfrac_comparison_threepanel_nncboundary(run_dir = './tpzruns/cosmos_closest_match_07/', nn_identifier = 'nnc_', err_type = 'tpz', bpz_appdir = None, outlier_val = 0.1, include_nnr = False, show_zconf = True):
    

    # FIRST PANEL

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*0/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    if bpz_appdir != None:
        if bpz_appdir [-1] != ['/']:
            bpz_appdir += '/'

        bpz_nnc_folders = glob(bpz_appdir + '*' + nn_identifier + '*/')

        if len(bpz_nnc_folders) == 1:
            bpz_nnc_folder = bpz_nnc_folders[0]
            if include_nnr:
                bpz_nnr_folder = bpz_nnc_folder.replace('nnc', 'nnr').replace('_corr', '')
            else:
                bpz_nnr_folder = bpz_nnc_folder.replace('nnc', 'nnr')
        else:
            print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

            for thisfolder in bpz_nnc_folders:
                print(thisfolder)
            # [print(thisfolder) for thisfolder in nnc_folders]
            return None
        bpz_features_file = read_csv(glob(bpz_appdir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        bpz_nnc_results = np.loadtxt(bpz_nnc_folder + 'results_application.dat')
        
        if include_nnr:
            bpz_nnr_results = np.loadtxt(bpz_nnr_folder + 'results_application.dat')
            bpz_good_inds = (np.isfinite(bpz_nnc_results) & np.isfinite(bpz_nnr_results))
        else:
            bpz_good_inds = np.isfinite(bpz_nnc_results)

        if include_nnr:
            bpz_nnr_results = bpz_nnr_results[bpz_good_inds]

        bpz_zspec = bpz_features_file['specz'].to_numpy()[bpz_good_inds]
        bpz_zphot = bpz_features_file['zphot'].to_numpy()[bpz_good_inds]
        bpz_zphot_err = bpz_features_file['zerr'].to_numpy()[bpz_good_inds]
        bpz_nnc_results = bpz_nnc_results[bpz_good_inds]
        bpz_is_goodfit = (np.abs(bpz_zphot - bpz_zspec) < (1. + bpz_zspec)*0.02)
        bpz_z_err = np.abs(bpz_zphot - bpz_zspec) / (1. + bpz_zspec)
        bpz_z_err_sigmas = np.abs(bpz_zphot - bpz_zspec)/bpz_zphot_err
        bpz_nnsorted_predictions, bpz_nnsorted_is_goodfit, bpz_nnsorted_zerr, bpz_nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(bpz_nnc_results, bpz_is_goodfit, bpz_z_err, bpz_z_err_sigmas)))))
        bpzsorted_predictions, bpzsorted_zerr, bpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(bpz_zphot_err, bpz_z_err, bpz_z_err_sigmas))))))

        bpz_nnsorted_is_goodfit = bpz_nnsorted_is_goodfit.astype(bool)
        bpz_n_frac = np.arange(len(bpz_nnsorted_predictions), dtype = float)[::-1]/len(bpz_nnsorted_predictions)

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    if include_nnr:
        nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')
        nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')
        good_inds = (np.isfinite(nnc_results) & np.isfinite(nnr_results))
        nnr_results = nnr_results[good_inds]
    else:
        good_inds = np.isfinite(nnc_results)


    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    zconf = features_file['zconf'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    if err_type == 'tpz':
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
    elif err_type == 'nnr':
        z_err = np.abs(zphot - nnr_results - zspec)/(1+zspec)
        z_err_sigmas = np.abs(zphot - nnr_results - zspec)/zphot_err
    
    is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

    nnsorted_predictions, nnsorted_is_goodfit, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, is_goodfit, z_err, z_err_sigmas)))))
    nnsorted_is_goodfit = nnsorted_is_goodfit.astype(bool)

    tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

    n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

    if len(nnsorted_zerr) > 10000:
        inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
    else:
        inds = np.arange(len(nnsorted_zerr))

    sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
    nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
    outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

    sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
    nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
    outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))

    if show_zconf:

        zconfsorted_predictions, zconfsorted_zerr, zconfsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(zconf, z_err, z_err_sigmas)))))
        sigma_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([np.std(zconfsorted_zerr[x:]) for x in inds]))
        nmad_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([1.4826*np.median(zconfsorted_zerr[x:]) for x in inds]))
        outlier_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([np.sum(zconfsorted_zerr[x:] >=outlier_val)/float(len(zconfsorted_zerr) - x) for x in inds]))

    fig = plt.figure(figsize = (24,8))

    sp_nmad = fig.add_subplot(131)
    sp_outlier = fig.add_subplot(132)
    sp_sigma = fig.add_subplot(133)

    if bpz_appdir != None:
        sigma_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([np.std(bpz_nnsorted_zerr[x:]) for x in inds]))
        nmad_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([1.4826*np.median(bpz_nnsorted_zerr[x:]) for x in inds]))
        outlier_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([np.sum(bpz_nnsorted_zerr[x:] >= outlier_val)/float(len(bpz_nnsorted_zerr) - x) for x in inds]))
        sigma_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([np.std(bpzsorted_zerr[x:]) for x in inds]))
        nmad_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([1.4826*np.median(bpzsorted_zerr[x:]) for x in inds]))
        outlier_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([np.sum(bpzsorted_zerr[x:] >= outlier_val)/float(len(bpzsorted_zerr) - x) for x in inds]))

        sp_nmad.plot(nmad_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')
        sp_outlier.plot(outlier_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')
        sp_sigma.plot(sigma_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')

        sp_nmad.plot(nmad_bpz, bpz_n_frac, label = r'BPZ $\sigma$', color = 'pink')
        sp_outlier.plot(outlier_bpz, bpz_n_frac, label = r'BPZ $\sigma$', color = 'pink')
        sp_sigma.plot(sigma_bpz, bpz_n_frac, label = r'BPZ $\sigma$', color = 'pink')

    if show_zconf:
        sp_nmad.plot(nmad_zconf, n_frac, label = r'TPZ zConf', color = '0.5')
        sp_outlier.plot(outlier_zconf, n_frac, label = r'TPZ zConf', color = '0.5')
        sp_sigma.plot(sigma_zconf, n_frac, label = r'TPZ zConf', color = '0.5')

    if bpz_appdir == None:
        nnclabel = 'NNC'
    else:
        nnclabel = 'TPZ NNC'

    sp_nmad.plot(nmad_tpz, n_frac, label = r'TPZ $\sigma$', color = 'k')
    sp_nmad.plot(nmad_nn, n_frac, label = nnclabel, color = 'C0')

    sp_outlier.plot(outlier_tpz, n_frac, label = r'TPZ $\sigma$', color = 'k')
    sp_outlier.plot(outlier_nn, n_frac, label = nnclabel, color = 'C0')

    sp_sigma.plot(sigma_tpz, n_frac, label = r'TPZ $\sigma$', color = 'k')
    sp_sigma.plot(sigma_nn, n_frac, label = nnclabel, color = 'C0')




    # for cutoff in np.arange(0.1,1,.1):
    #     thisx = np.interp(cutoff, nnsorted_predictions, n_frac)
    #     thisy_nmad = np.interp(cutoff, nnsorted_predictions, nmad_nn)
    #     thisy_sigma = np.interp(cutoff, nnsorted_predictions, sigma_nn)
    #     sp_nmad.annotate('%.1f'%(cutoff), (thisx, thisy_nmad), (thisx-.05, thisy_nmad+ 0.05), arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
    #     sp_sigma.annotate('%.1f'%(cutoff), (thisx, thisy_sigma), (thisx-.05, thisy_sigma+ 0.05), arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')

    for cutoff in np.arange(0.1,1,.1):
        thisy = np.interp(cutoff, nnsorted_predictions, n_frac)
        thisx_nmad = np.interp(cutoff, nnsorted_predictions, nmad_nn)
        thisx_sigma = np.interp(cutoff, nnsorted_predictions, sigma_nn)
        thisx_outlier = np.interp(cutoff, nnsorted_predictions, outlier_nn)
        if all(np.isfinite([thisx_nmad, thisx_sigma, thisx_outlier])):
            sp_nmad.annotate('%.1f'%(cutoff), (thisx_nmad, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_sigma.annotate('%.1f'%(cutoff), (thisx_sigma, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_outlier.annotate('%.1f'%(cutoff), (thisx_outlier, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')


    # sp_nmad.set_xlim(0,max(np.hstack((nmad_nn, nmad_tpz))))
    # sp_outlier.set_xlim(0,max(np.hstack((outlier_nn, outlier_tpz))))
    # sp_sigma.set_xlim(0,max(np.hstack((sigma_nn, sigma_tpz))))

    sp_nmad.set_xlim(0, np.maximum(nmad_nn[0], nmad_tpz[0]))
    sp_outlier.set_xlim(0, np.maximum(outlier_nn[0], outlier_tpz[0]))
    sp_sigma.set_xlim(0, np.maximum(sigma_nn[0], sigma_tpz[0]))

    sp_nmad.set_ylabel('N$_{frac}$')
    sp_nmad.set_xlabel('NMAD')
    sp_outlier.set_xlabel(r'Outlier Fraction $\left( \frac{\Delta z}{1+z} > ' + '%.2f' % outlier_val + r'\right)$')
    sp_sigma.set_xlabel(r'$\sigma$')

    sp_nmad.set_ylim(0,1)
    sp_outlier.set_ylim(0,1)
    sp_sigma.set_ylim(0,1)

    sp_outlier.set_yticklabels([])
    sp_sigma.set_yticklabels([])

    # sp_nmad.set_xlim(0,1)
    # sp_outlier.set_xlim(0,1)
    # sp_sigma.set_xlim(0,1)

    sp_nmad.legend(loc = 'upper left', fontsize = 20)

    plt.subplots_adjust(wspace = 0, hspace = 0)




def plot_sigma_nfrac_comparison_threepanel_nnrcompare(run_dir = './tpzruns/cosmos_closest_match_07/', nn_identifier = 'nnc_', err_type = 'tpz', bpz_appdir = None, outlier_val = 0.1, include_nnr = False, show_zconf = True):
    

    # FIRST PANEL

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*0/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    if bpz_appdir != None:
        if bpz_appdir [-1] != ['/']:
            bpz_appdir += '/'

        bpz_nnc_folders = glob(bpz_appdir + '*' + nn_identifier + '*/')

        if len(bpz_nnc_folders) == 1:
            bpz_nnc_folder = bpz_nnc_folders[0]
            if include_nnr:
                bpz_nnr_folder = bpz_nnc_folder.replace('nnc', 'nnr').replace('_corr', '')
            else:
                bpz_nnr_folder = bpz_nnc_folder.replace('nnc', 'nnr')
        else:
            print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

            for thisfolder in bpz_nnc_folders:
                print(thisfolder)
            # [print(thisfolder) for thisfolder in nnc_folders]
            return None
        bpz_features_file = read_csv(glob(bpz_appdir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        bpz_nnc_results = np.loadtxt(bpz_nnc_folder + 'results_application.dat')
        
        if include_nnr:
            bpz_nnr_results = np.loadtxt(bpz_nnr_folder + 'results_application.dat')
            bpz_good_inds = (np.isfinite(bpz_nnc_results) & np.isfinite(bpz_nnr_results))
        else:
            bpz_good_inds = np.isfinite(bpz_nnc_results)

        if include_nnr:
            bpz_nnr_results = bpz_nnr_results[bpz_good_inds]

        bpz_zspec = bpz_features_file['specz'].to_numpy()[bpz_good_inds]
        bpz_zphot = bpz_features_file['zphot'].to_numpy()[bpz_good_inds]
        bpz_zphot_err = bpz_features_file['zerr'].to_numpy()[bpz_good_inds]
        bpz_nnc_results = bpz_nnc_results[bpz_good_inds]
        bpz_is_goodfit = (np.abs(bpz_zphot - bpz_zspec) < (1. + bpz_zspec)*0.02)
        bpz_z_err = np.abs(bpz_zphot - bpz_zspec) / (1. + bpz_zspec)
        bpz_z_err_sigmas = np.abs(bpz_zphot - bpz_zspec)/bpz_zphot_err
        bpz_nnsorted_predictions, bpz_nnsorted_is_goodfit, bpz_nnsorted_zerr, bpz_nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(bpz_nnc_results, bpz_is_goodfit, bpz_z_err, bpz_z_err_sigmas)))))
        bpzsorted_predictions, bpzsorted_zerr, bpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(bpz_zphot_err, bpz_z_err, bpz_z_err_sigmas))))))

        bpz_nnsorted_is_goodfit = bpz_nnsorted_is_goodfit.astype(bool)
        bpz_n_frac = np.arange(len(bpz_nnsorted_predictions), dtype = float)[::-1]/len(bpz_nnsorted_predictions)

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    if include_nnr:
        nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')
        nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')
        good_inds = (np.isfinite(nnc_results) & np.isfinite(nnr_results))
        nnr_results = nnr_results[good_inds]
    else:
        good_inds = np.isfinite(nnc_results)


    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    zconf = features_file['zconf'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    if err_type == 'tpz':
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
    elif err_type == 'nnr':
        z_err = np.abs(zphot - nnr_results - zspec)/(1+zspec)
        z_err_sigmas = np.abs(zphot - nnr_results - zspec)/zphot_err
    
    is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

    nnsorted_predictions, nnsorted_is_goodfit, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, is_goodfit, z_err, z_err_sigmas)))))
    nnsorted_is_goodfit = nnsorted_is_goodfit.astype(bool)

    tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

    n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

    if len(nnsorted_zerr) > 10000:
        inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
    else:
        inds = np.arange(len(nnsorted_zerr))

    sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
    nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
    outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

    sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
    nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
    outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))

    if show_zconf:

        zconfsorted_predictions, zconfsorted_zerr, zconfsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(zconf, z_err, z_err_sigmas)))))
        sigma_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([np.std(zconfsorted_zerr[x:]) for x in inds]))
        nmad_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([1.4826*np.median(zconfsorted_zerr[x:]) for x in inds]))
        outlier_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([np.sum(zconfsorted_zerr[x:] >=outlier_val)/float(len(zconfsorted_zerr) - x) for x in inds]))

    fig = plt.figure(figsize = (24,8))

    sp_nmad = fig.add_subplot(131)
    sp_outlier = fig.add_subplot(132)
    sp_sigma = fig.add_subplot(133)

    if bpz_appdir != None:
        sigma_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([np.std(bpz_nnsorted_zerr[x:]) for x in inds]))
        nmad_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([1.4826*np.median(bpz_nnsorted_zerr[x:]) for x in inds]))
        outlier_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([np.sum(bpz_nnsorted_zerr[x:] >= outlier_val)/float(len(bpz_nnsorted_zerr) - x) for x in inds]))
        sigma_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([np.std(bpzsorted_zerr[x:]) for x in inds]))
        nmad_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([1.4826*np.median(bpzsorted_zerr[x:]) for x in inds]))
        outlier_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([np.sum(bpzsorted_zerr[x:] >= outlier_val)/float(len(bpzsorted_zerr) - x) for x in inds]))

        sp_nmad.plot(nmad_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')
        sp_outlier.plot(outlier_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')
        sp_sigma.plot(sigma_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')

        sp_nmad.plot(nmad_bpz, bpz_n_frac, label = r'BPZ $\sigma$', color = 'pink')
        sp_outlier.plot(outlier_bpz, bpz_n_frac, label = r'BPZ $\sigma$', color = 'pink')
        sp_sigma.plot(sigma_bpz, bpz_n_frac, label = r'BPZ $\sigma$', color = 'pink')

    if show_zconf:
        sp_nmad.plot(nmad_zconf, n_frac, label = r'TPZ zConf', color = '0.5')
        sp_outlier.plot(outlier_zconf, n_frac, label = r'TPZ zConf', color = '0.5')
        sp_sigma.plot(sigma_zconf, n_frac, label = r'TPZ zConf', color = '0.5')

    if bpz_appdir == None:
        nnclabel = 'NNC'
    else:
        nnclabel = 'TPZ NNC'

    sp_nmad.plot(nmad_tpz, n_frac, label = r'TPZ $\sigma$', color = 'k')
    sp_nmad.plot(nmad_nn, n_frac, label = nnclabel, color = 'C0')

    sp_outlier.plot(outlier_tpz, n_frac, label = r'TPZ $\sigma$', color = 'k')
    sp_outlier.plot(outlier_nn, n_frac, label = nnclabel, color = 'C0')

    sp_sigma.plot(sigma_tpz, n_frac, label = r'TPZ $\sigma$', color = 'k')
    sp_sigma.plot(sigma_nn, n_frac, label = nnclabel, color = 'C0')




    # for cutoff in np.arange(0.1,1,.1):
    #     thisx = np.interp(cutoff, nnsorted_predictions, n_frac)
    #     thisy_nmad = np.interp(cutoff, nnsorted_predictions, nmad_nn)
    #     thisy_sigma = np.interp(cutoff, nnsorted_predictions, sigma_nn)
    #     sp_nmad.annotate('%.1f'%(cutoff), (thisx, thisy_nmad), (thisx-.05, thisy_nmad+ 0.05), arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
    #     sp_sigma.annotate('%.1f'%(cutoff), (thisx, thisy_sigma), (thisx-.05, thisy_sigma+ 0.05), arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')

    for cutoff in np.arange(0.1,1,.1):
        thisy = np.interp(cutoff, nnsorted_predictions, n_frac)
        thisx_nmad = np.interp(cutoff, nnsorted_predictions, nmad_nn)
        thisx_sigma = np.interp(cutoff, nnsorted_predictions, sigma_nn)
        thisx_outlier = np.interp(cutoff, nnsorted_predictions, outlier_nn)
        if all(np.isfinite([thisx_nmad, thisx_sigma, thisx_outlier])):
            sp_nmad.annotate('%.1f'%(cutoff), (thisx_nmad, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_sigma.annotate('%.1f'%(cutoff), (thisx_sigma, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_outlier.annotate('%.1f'%(cutoff), (thisx_outlier, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')


    # sp_nmad.set_xlim(0,max(np.hstack((nmad_nn, nmad_tpz))))
    # sp_outlier.set_xlim(0,max(np.hstack((outlier_nn, outlier_tpz))))
    # sp_sigma.set_xlim(0,max(np.hstack((sigma_nn, sigma_tpz))))

    sp_nmad.set_xlim(0, np.maximum(nmad_nn[0], nmad_tpz[0]))
    sp_outlier.set_xlim(0, np.maximum(outlier_nn[0], outlier_tpz[0]))
    sp_sigma.set_xlim(0, np.maximum(sigma_nn[0], sigma_tpz[0]))

    sp_nmad.set_ylabel('N$_{frac}$')
    sp_nmad.set_xlabel('NMAD')
    sp_outlier.set_xlabel(r'Outlier Fraction $\left( \frac{\Delta z}{1+z} > ' + '%.2f' % outlier_val + r'\right)$')
    sp_sigma.set_xlabel(r'$\sigma$')

    sp_nmad.set_ylim(0,1)
    sp_outlier.set_ylim(0,1)
    sp_sigma.set_ylim(0,1)

    sp_outlier.set_yticklabels([])
    sp_sigma.set_yticklabels([])

    # sp_nmad.set_xlim(0,1)
    # sp_outlier.set_xlim(0,1)
    # sp_sigma.set_xlim(0,1)

    sp_nmad.legend(loc = 'upper left', fontsize = 20)

    plt.subplots_adjust(wspace = 0, hspace = 0)



def plot_sigma_nfrac_comparison_threepanel_triplets(run_dir = './tpzruns/cosmos_closest_match_07/', nn_identifier = 'nnc_', err_type = 'tpz', bpz_appdir = None, outlier_val = 0.1, include_nnr = False, show_zconf = True):
    

    # FIRST PANEL

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*0/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    if bpz_appdir != None:
        if bpz_appdir [-1] != ['/']:
            bpz_appdir += '/'

        bpz_nnc_folders = glob(bpz_appdir + '*' + nn_identifier + '*/')

        if len(bpz_nnc_folders) == 1:
            bpz_nnc_folder = bpz_nnc_folders[0]
            if include_nnr:
                bpz_nnr_folder = bpz_nnc_folder.replace('nnc', 'nnr').replace('_corr', '')
            else:
                bpz_nnr_folder = bpz_nnc_folder.replace('nnc', 'nnr')
        else:
            print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

            for thisfolder in bpz_nnc_folders:
                print(thisfolder)
            # [print(thisfolder) for thisfolder in nnc_folders]
            return None
        bpz_features_file = read_csv(glob(bpz_appdir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        bpz_nnc_results = np.loadtxt(bpz_nnc_folder + 'results_application.dat')
        
        if include_nnr:
            bpz_nnr_results = np.loadtxt(bpz_nnr_folder + 'results_application.dat')
            bpz_good_inds = (np.isfinite(bpz_nnc_results) & np.isfinite(bpz_nnr_results))
        else:
            bpz_good_inds = np.isfinite(bpz_nnc_results)

        if include_nnr:
            bpz_nnr_results = bpz_nnr_results[bpz_good_inds]

        bpz_zspec = bpz_features_file['specz'].to_numpy()[bpz_good_inds]
        bpz_zphot = bpz_features_file['zphot'].to_numpy()[bpz_good_inds]
        bpz_zphot_err = bpz_features_file['zerr'].to_numpy()[bpz_good_inds]
        bpz_nnc_results = bpz_nnc_results[bpz_good_inds]
        bpz_is_goodfit = (np.abs(bpz_zphot - bpz_zspec) < (1. + bpz_zspec)*0.02)
        bpz_z_err = np.abs(bpz_zphot - bpz_zspec) / (1. + bpz_zspec)
        bpz_z_err_sigmas = np.abs(bpz_zphot - bpz_zspec)/bpz_zphot_err
        bpz_nnsorted_predictions, bpz_nnsorted_is_goodfit, bpz_nnsorted_zerr, bpz_nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(bpz_nnc_results, bpz_is_goodfit, bpz_z_err, bpz_z_err_sigmas)))))
        bpzsorted_predictions, bpzsorted_zerr, bpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(bpz_zphot_err, bpz_z_err, bpz_z_err_sigmas))))))

        bpz_nnsorted_is_goodfit = bpz_nnsorted_is_goodfit.astype(bool)
        bpz_n_frac = np.arange(len(bpz_nnsorted_predictions), dtype = float)[::-1]/len(bpz_nnsorted_predictions)

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    if include_nnr:
        nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')
        nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')
        good_inds = (np.isfinite(nnc_results) & np.isfinite(nnr_results))
        nnr_results = nnr_results[good_inds]
    else:
        good_inds = np.isfinite(nnc_results)


    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    zconf = features_file['zconf'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    if err_type == 'tpz':
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
    elif err_type == 'nnr':
        z_err = np.abs(zphot - nnr_results - zspec)/(1+zspec)
        z_err_sigmas = np.abs(zphot - nnr_results - zspec)/zphot_err
    
    is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

    nnsorted_predictions, nnsorted_is_goodfit, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, is_goodfit, z_err, z_err_sigmas)))))
    nnsorted_is_goodfit = nnsorted_is_goodfit.astype(bool)

    tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

    n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

    if len(nnsorted_zerr) > 10000:
        inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
    else:
        inds = np.arange(len(nnsorted_zerr))

    sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
    nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
    outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

    sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
    nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
    outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))

    if show_zconf:

        zconfsorted_predictions, zconfsorted_zerr, zconfsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(zconf, z_err, z_err_sigmas)))))
        sigma_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([np.std(zconfsorted_zerr[x:]) for x in inds]))
        nmad_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([1.4826*np.median(zconfsorted_zerr[x:]) for x in inds]))
        outlier_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([np.sum(zconfsorted_zerr[x:] >=outlier_val)/float(len(zconfsorted_zerr) - x) for x in inds]))

    fig = plt.figure(figsize = (24,8))

    sp_nmad = fig.add_subplot(131)
    sp_outlier = fig.add_subplot(132)
    sp_sigma = fig.add_subplot(133)

    if bpz_appdir != None:
        sigma_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([np.std(bpz_nnsorted_zerr[x:]) for x in inds]))
        nmad_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([1.4826*np.median(bpz_nnsorted_zerr[x:]) for x in inds]))
        outlier_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([np.sum(bpz_nnsorted_zerr[x:] >= outlier_val)/float(len(bpz_nnsorted_zerr) - x) for x in inds]))
        sigma_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([np.std(bpzsorted_zerr[x:]) for x in inds]))
        nmad_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([1.4826*np.median(bpzsorted_zerr[x:]) for x in inds]))
        outlier_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([np.sum(bpzsorted_zerr[x:] >= outlier_val)/float(len(bpzsorted_zerr) - x) for x in inds]))

        sp_nmad.plot(nmad_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')
        sp_outlier.plot(outlier_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')
        sp_sigma.plot(sigma_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')

        sp_nmad.plot(nmad_bpz, bpz_n_frac, label = r'BPZ $\sigma$', color = 'pink')
        sp_outlier.plot(outlier_bpz, bpz_n_frac, label = r'BPZ $\sigma$', color = 'pink')
        sp_sigma.plot(sigma_bpz, bpz_n_frac, label = r'BPZ $\sigma$', color = 'pink')

    if show_zconf:
        sp_nmad.plot(nmad_zconf, n_frac, label = r'TPZ zConf', color = '0.5')
        sp_outlier.plot(outlier_zconf, n_frac, label = r'TPZ zConf', color = '0.5')
        sp_sigma.plot(sigma_zconf, n_frac, label = r'TPZ zConf', color = '0.5')

    if bpz_appdir == None:
        nnclabel = 'NNC'
    else:
        nnclabel = 'TPZ NNC'

    sp_nmad.plot(nmad_tpz, n_frac, label = r'TPZ $\sigma$', color = 'k')
    sp_nmad.plot(nmad_nn, n_frac, label = nnclabel, color = 'C0')

    sp_outlier.plot(outlier_tpz, n_frac, label = r'TPZ $\sigma$', color = 'k')
    sp_outlier.plot(outlier_nn, n_frac, label = nnclabel, color = 'C0')

    sp_sigma.plot(sigma_tpz, n_frac, label = r'TPZ $\sigma$', color = 'k')
    sp_sigma.plot(sigma_nn, n_frac, label = nnclabel, color = 'C0')




    # for cutoff in np.arange(0.1,1,.1):
    #     thisx = np.interp(cutoff, nnsorted_predictions, n_frac)
    #     thisy_nmad = np.interp(cutoff, nnsorted_predictions, nmad_nn)
    #     thisy_sigma = np.interp(cutoff, nnsorted_predictions, sigma_nn)
    #     sp_nmad.annotate('%.1f'%(cutoff), (thisx, thisy_nmad), (thisx-.05, thisy_nmad+ 0.05), arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
    #     sp_sigma.annotate('%.1f'%(cutoff), (thisx, thisy_sigma), (thisx-.05, thisy_sigma+ 0.05), arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')

    for cutoff in np.arange(0.1,1,.1):
        thisy = np.interp(cutoff, nnsorted_predictions, n_frac)
        thisx_nmad = np.interp(cutoff, nnsorted_predictions, nmad_nn)
        thisx_sigma = np.interp(cutoff, nnsorted_predictions, sigma_nn)
        thisx_outlier = np.interp(cutoff, nnsorted_predictions, outlier_nn)
        if all(np.isfinite([thisx_nmad, thisx_sigma, thisx_outlier])):
            sp_nmad.annotate('%.1f'%(cutoff), (thisx_nmad, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_sigma.annotate('%.1f'%(cutoff), (thisx_sigma, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_outlier.annotate('%.1f'%(cutoff), (thisx_outlier, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')


    # sp_nmad.set_xlim(0,max(np.hstack((nmad_nn, nmad_tpz))))
    # sp_outlier.set_xlim(0,max(np.hstack((outlier_nn, outlier_tpz))))
    # sp_sigma.set_xlim(0,max(np.hstack((sigma_nn, sigma_tpz))))

    sp_nmad.set_xlim(0, np.maximum(nmad_nn[0], nmad_tpz[0]))
    sp_outlier.set_xlim(0, np.maximum(outlier_nn[0], outlier_tpz[0]))
    sp_sigma.set_xlim(0, np.maximum(sigma_nn[0], sigma_tpz[0]))

    sp_nmad.set_ylabel('N$_{frac}$')
    sp_nmad.set_xlabel('NMAD')
    sp_outlier.set_xlabel(r'Outlier Fraction $\left( \frac{\Delta z}{1+z} > ' + '%.2f' % outlier_val + r'\right)$')
    sp_sigma.set_xlabel(r'$\sigma$')

    sp_nmad.set_ylim(0,1)
    sp_outlier.set_ylim(0,1)
    sp_sigma.set_ylim(0,1)

    sp_outlier.set_yticklabels([])
    sp_sigma.set_yticklabels([])

    # sp_nmad.set_xlim(0,1)
    # sp_outlier.set_xlim(0,1)
    # sp_sigma.set_xlim(0,1)

    sp_nmad.legend(loc = 'upper left', fontsize = 20)

    plt.subplots_adjust(wspace = 0, hspace = 0)





def plot_sigma_nfrac_comparison_threepanel(run_dir = './tpzruns/MatchCOSMOS2015/', nn_identifier = 'nnc_', err_type = 'tpz', bpz_appdir = None, outlier_val = 0.15, include_nnr = False, show_zconf = True, show_sn = False, train = False):
    

    # FIRST PANEL

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*0/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    if bpz_appdir != None:
        if bpz_appdir [-1] != ['/']:
            bpz_appdir += '/'

        bpz_nnc_folders = glob(bpz_appdir + '*' + nn_identifier + '*/')

        if len(bpz_nnc_folders) == 1:
            bpz_nnc_folder = bpz_nnc_folders[0]
            if include_nnr:
                bpz_nnr_folder = bpz_nnc_folder.replace('nnc', 'nnr').replace('_corr', '')
            else:
                bpz_nnr_folder = bpz_nnc_folder.replace('nnc', 'nnr')
        else:
            print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

            for thisfolder in bpz_nnc_folders:
                print(thisfolder)
            # [print(thisfolder) for thisfolder in nnc_folders]
            return None
        bpz_features_file = read_csv(glob(bpz_appdir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        bpz_nnc_results = np.loadtxt(bpz_nnc_folder + 'results_application.dat')
        
        if include_nnr:
            bpz_nnr_results = np.loadtxt(bpz_nnr_folder + 'results_application.dat')
            bpz_good_inds = (np.isfinite(bpz_nnc_results) & np.isfinite(bpz_nnr_results))
        else:
            bpz_good_inds = np.isfinite(bpz_nnc_results)

        if include_nnr:
            bpz_nnr_results = bpz_nnr_results[bpz_good_inds]

        bpz_zspec = bpz_features_file['specz'].to_numpy()[bpz_good_inds]
        bpz_zphot = bpz_features_file['zphot'].to_numpy()[bpz_good_inds]
        bpz_zphot_err = bpz_features_file['zerr'].to_numpy()[bpz_good_inds]
        bpz_nnc_results = bpz_nnc_results[bpz_good_inds]
        bpz_is_goodfit = (np.abs(bpz_zphot - bpz_zspec) < (1. + bpz_zspec)*0.02)
        bpz_z_err = np.abs(bpz_zphot - bpz_zspec) / (1. + bpz_zspec)
        bpz_z_err_sigmas = np.abs(bpz_zphot - bpz_zspec)/bpz_zphot_err
        bpz_nnsorted_predictions, bpz_nnsorted_is_goodfit, bpz_nnsorted_zerr, bpz_nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(bpz_nnc_results, bpz_is_goodfit, bpz_z_err, bpz_z_err_sigmas)))))
        bpzsorted_predictions, bpzsorted_zerr, bpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(bpz_zphot_err, bpz_z_err, bpz_z_err_sigmas))))))

        bpz_nnsorted_is_goodfit = bpz_nnsorted_is_goodfit.astype(bool)
        bpz_n_frac = np.arange(len(bpz_nnsorted_predictions), dtype = float)[::-1]/len(bpz_nnsorted_predictions)

    if not train:
        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    else:
        features_file = read_csv(glob(run_dir + '*.nnc_train')[0], delimiter = '\s+', comment = '#') 
        nnc_results = np.loadtxt(nnc_folder + 'results_train.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    if include_nnr:
        nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')
        nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')
        good_inds = (np.isfinite(nnc_results) & np.isfinite(nnr_results))
        nnr_results = nnr_results[good_inds]
    else:
        good_inds = np.isfinite(nnc_results)


    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    zconf = features_file['zconf'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    if err_type == 'tpz':
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_sigmas = np.abs(zphot - zspec)/zphot_err
    elif err_type == 'nnr':
        z_err = np.abs(zphot - nnr_results - zspec)/(1+zspec)
        z_err_sigmas = np.abs(zphot - nnr_results - zspec)/zphot_err
    
    is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

    nnsorted_predictions, nnsorted_is_goodfit, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(nnc_results, is_goodfit, z_err, z_err_sigmas)))))
    nnsorted_is_goodfit = nnsorted_is_goodfit.astype(bool)

    tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

    n_frac = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)

    if len(nnsorted_zerr) > 10000:
        inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
    else:
        inds = np.arange(len(nnsorted_zerr))

    sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
    nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
    outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >=outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))

    sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
    nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
    outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))

    if show_zconf:

        zconfsorted_predictions, zconfsorted_zerr, zconfsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(zconf, z_err, z_err_sigmas)))))
        sigma_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([np.std(zconfsorted_zerr[x:]) for x in inds]))
        nmad_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([1.4826*np.median(zconfsorted_zerr[x:]) for x in inds]))
        outlier_zconf = np.interp(np.arange(len(zconfsorted_zerr)), inds, np.array([np.sum(zconfsorted_zerr[x:] >=outlier_val)/float(len(zconfsorted_zerr) - x) for x in inds]))

    if show_sn:

        mags = features_file[['g', 'r', 'i', 'z', 'y']].to_numpy()
        mag_errs = features_file[['eg', 'er', 'ei', 'ez', 'ey']].to_numpy()
        fluxes = np.power(10., (mags + 48.6)/(-2.5))
        flux_errs = -3.34407e-20 * np.exp(-0.921034 * mags)
        sn = np.sqrt(np.sum((fluxes/flux_errs)**2, axis = 1))

        snsorted_predictions, snsorted_zerr, snsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(sn, z_err, z_err_sigmas)))))
        sigma_sn = np.interp(np.arange(len(snsorted_zerr)), inds, np.array([np.std(snsorted_zerr[x:]) for x in inds]))
        nmad_sn = np.interp(np.arange(len(snsorted_zerr)), inds, np.array([1.4826*np.median(snsorted_zerr[x:]) for x in inds]))
        outlier_sn = np.interp(np.arange(len(snsorted_zerr)), inds, np.array([np.sum(snsorted_zerr[x:] >=outlier_val)/float(len(snsorted_zerr) - x) for x in inds]))


    fig = plt.figure(figsize = (24,8))

    sp_outlier = fig.add_subplot(131)
    sp_nmad = fig.add_subplot(132)
    sp_sigma = fig.add_subplot(133)

    if bpz_appdir != None:
        sigma_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([np.std(bpz_nnsorted_zerr[x:]) for x in inds]))
        nmad_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([1.4826*np.median(bpz_nnsorted_zerr[x:]) for x in inds]))
        outlier_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([np.sum(bpz_nnsorted_zerr[x:] >= outlier_val)/float(len(bpz_nnsorted_zerr) - x) for x in inds]))
        sigma_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([np.std(bpzsorted_zerr[x:]) for x in inds]))
        nmad_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([1.4826*np.median(bpzsorted_zerr[x:]) for x in inds]))
        outlier_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([np.sum(bpzsorted_zerr[x:] >= outlier_val)/float(len(bpzsorted_zerr) - x) for x in inds]))

        sp_nmad.plot(nmad_bpznn, bpz_n_frac, label = r'$C_{NNC,BPZ}$', color = 'r')
        sp_outlier.plot(outlier_bpznn, bpz_n_frac, label = r'$C_{NNC,BPZ}$', color = 'r')
        sp_sigma.plot(sigma_bpznn, bpz_n_frac, label = r'$C_{NNC,BPZ}$', color = 'r')

        sp_nmad.plot(nmad_bpz, bpz_n_frac, label = r'$\sigma_{BPZ}$', color = 'pink')
        sp_outlier.plot(outlier_bpz, bpz_n_frac, label = r'$\sigma_{BPZ}$', color = 'pink')
        sp_sigma.plot(sigma_bpz, bpz_n_frac, label = r'$\sigma_{BPZ}$', color = 'pink')

    if show_zconf:
        sp_nmad.plot(nmad_zconf, n_frac, label = r'zConf', color = '0.5')
        sp_outlier.plot(outlier_zconf, n_frac, label = r'zConf', color = '0.5')
        sp_sigma.plot(sigma_zconf, n_frac, label = r'zConf', color = '0.5')

    if show_sn:
        sp_nmad.plot(nmad_sn, n_frac, label = 'S/N', color = 'C1')
        sp_outlier.plot(outlier_sn, n_frac, label = 'S/N', color = 'C1')
        sp_sigma.plot(sigma_sn, n_frac, label = 'S/N', color = 'C1')


    if bpz_appdir == None:
        nnclabel = '$C_{NNC}$'
    else:
        nnclabel = '$C_{NNC,TPZ}$'

    sp_nmad.plot(nmad_tpz, n_frac, label = r'$\sigma_{TPZ}$', color = 'k')
    sp_nmad.plot(nmad_nn, n_frac, label = nnclabel, color = 'C0')

    sp_outlier.plot(outlier_tpz, n_frac, label = r'$\sigma_{TPZ}$', color = 'k')
    sp_outlier.plot(outlier_nn, n_frac, label = nnclabel, color = 'C0')

    sp_sigma.plot(sigma_tpz, n_frac, label = r'$\sigma_{TPZ}$', color = 'k')
    sp_sigma.plot(sigma_nn, n_frac, label = nnclabel, color = 'C0')




    # for cutoff in np.arange(0.1,1,.1):
    #     thisx = np.interp(cutoff, nnsorted_predictions, n_frac)
    #     thisy_nmad = np.interp(cutoff, nnsorted_predictions, nmad_nn)
    #     thisy_sigma = np.interp(cutoff, nnsorted_predictions, sigma_nn)
    #     sp_nmad.annotate('%.1f'%(cutoff), (thisx, thisy_nmad), (thisx-.05, thisy_nmad+ 0.05), arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
    #     sp_sigma.annotate('%.1f'%(cutoff), (thisx, thisy_sigma), (thisx-.05, thisy_sigma+ 0.05), arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')

    for cutoff in np.arange(0.1,1,.1):
        thisy = np.interp(cutoff, nnsorted_predictions, n_frac)
        thisx_nmad = np.interp(cutoff, nnsorted_predictions, nmad_nn)
        thisx_sigma = np.interp(cutoff, nnsorted_predictions, sigma_nn)
        thisx_outlier = np.interp(cutoff, nnsorted_predictions, outlier_nn)
        if all(np.isfinite([thisx_nmad, thisx_sigma, thisx_outlier])):
            sp_nmad.annotate('%.1f'%(cutoff), (thisx_nmad, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_sigma.annotate('%.1f'%(cutoff), (thisx_sigma, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_outlier.annotate('%.1f'%(cutoff), (thisx_outlier, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')


    # sp_nmad.set_xlim(0,max(np.hstack((nmad_nn, nmad_tpz))))
    # sp_outlier.set_xlim(0,max(np.hstack((outlier_nn, outlier_tpz))))
    # sp_sigma.set_xlim(0,max(np.hstack((sigma_nn, sigma_tpz))))

    sp_nmad.set_xlim(0, np.maximum(nmad_nn[0], nmad_tpz[0]))
    sp_outlier.set_xlim(0, np.maximum(outlier_nn[0], outlier_tpz[0]))
    sp_sigma.set_xlim(0, np.maximum(sigma_nn[0], sigma_tpz[0]))

    sp_outlier.set_ylabel('$f_{sample}$')
    sp_nmad.set_xlabel('NMAD')
    sp_outlier.set_xlabel(r'Outlier Fraction $\left( \frac{\Delta z}{1+z} > ' + '%.2f' % outlier_val + r'\right)$')
    sp_sigma.set_xlabel(r'$\sigma_z$')

    sp_nmad.set_ylim(0,1)
    sp_outlier.set_ylim(0,1)
    sp_sigma.set_ylim(0,1)

    sp_nmad.set_yticklabels([])
    sp_sigma.set_yticklabels([])

    # sp_nmad.set_xlim(0,1)
    # sp_outlier.set_xlim(0,1)
    # sp_sigma.set_xlim(0,1)

    sp_outlier.legend(loc = 'lower right', fontsize = 20)

    plt.subplots_adjust(wspace = 0, hspace = 0)



def plot_sigma_nfrac_comparison_threepanel_nnc2(run_dir = './tpzruns/default_run/', nn_identifier = 'nnc2_', bpz_appdir = None, outlier_val = 0.1, tpz_only = False, nnr_only = False):
    

    # FIRST PANEL

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc2', 'nnr').replace('_corr', '')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    good_inds = (np.all(np.isfinite(nnc_results), axis = 1) & np.isfinite(nnr_results))

    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    nnr_results = nnr_results[good_inds]

    if tpz_only:
        nnc_class = np.zeros(len(nnc_results))
    elif nnr_only:
        nnc_class = np.ones(len(nnc_results))
    else:
        nnc_class = np.argmax(nnc_results, axis = 1)

    zphot_nnc = np.copy(zphot)
    zphot_nnc[nnc_class == 1] = (zphot - nnr_results)[nnc_class == 1] #NNR photoz is better
    goodfit_conf = np.maximum(*nnc_results.T)
    # is_goodfit = ((nnc_class == 1) | (nnc_class == 3)) & (goodfit_conf > nn_boundary)
    # use_nnr = (nnc_class.T[2] | nnc_class.T[3])

    z_err = np.abs(zphot - zspec)/(1+zspec)
    z_err_sigmas = np.abs(zphot - zspec)/zphot_err

    z_err_nnc = np.abs(zphot_nnc - zspec)/(1+zspec)

    # if err_type == 'tpz':
    #     z_err = np.abs(zphot - zspec) / (1. + zspec)
    #     z_err_sigmas = np.abs(zphot - zspec)/zphot_err
    # elif err_type == 'nnr':
    #     z_err = np.abs(zphot - nnr_results - zspec)/(1+zspec)
    #     z_err_sigmas = np.abs(zphot - nnr_results - zspec)/zphot_err
    
    # is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

    nnsorted_predictions, nnsorted_zerr = np.array(list(zip(*sorted(zip(goodfit_conf, z_err_nnc)))))

    tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

    n_frac_nn = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)
    n_frac_tpz = np.arange(len(tpzsorted_predictions), dtype = float)[::-1]/len(tpzsorted_predictions)

    if len(nnsorted_zerr) > 10000:
        inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
    else:
        inds = np.arange(len(nnsorted_zerr))

    sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
    nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
    outlier_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.sum(nnsorted_zerr[x:] >= outlier_val)/float(len(nnsorted_zerr) - x) for x in inds]))

    sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
    nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
    outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.sum(tpzsorted_zerr[x:] >= outlier_val)/float(len(tpzsorted_zerr) - x) for x in inds]))


    if bpz_appdir != None:
        if bpz_appdir [-1] != ['/']:
            bpz_appdir += '/'

        bpz_nnc_folders = glob(bpz_appdir + '*' + nn_identifier + '*/')

        if len(bpz_nnc_folders) == 1:
            bpz_nnc_folder = bpz_nnc_folders[0]
            bpz_nnr_folder = bpz_nnc_folder.replace('nnc2', 'nnr').replace('_corr', '')
        else:
            print('There are multiple folders matching the bpz_nn_identifier.  Please pick one and try again:')

            for thisfolder in bpz_nnc_folders:
                print(thisfolder)
            # [print(thisfolder) for thisfolder in nnc_folders]
            return None
        bpz_features_file = read_csv(glob(bpz_appdir + '*.nn_app')[0], delimiter = '\s+', comment = '#')
        bpz_nnc_results = np.loadtxt(bpz_nnc_folder + 'results_application.dat')
        bpz_nnr_results = np.loadtxt(bpz_nnr_folder + 'results_application.dat')

        bpz_good_inds = (np.isfinite(bpz_nnc_results) & np.isfinite(bpz_nnr_results))

        bpz_zspec = bpz_features_file['specz'].to_numpy()[bpz_good_inds]
        bpz_zphot = bpz_features_file['zphot'].to_numpy()[bpz_good_inds]
        bpz_zphot_err = bpz_features_file['zerr'].to_numpy()[bpz_good_inds]
        bpz_nnc_results = bpz_nnc_results[bpz_good_inds]
        bpz_nnr_results = bpz_nnr_results[bpz_good_inds]
        bpz_is_goodfit = (np.abs(bpz_zphot - bpz_zspec) < (1. + bpz_zspec)*0.02)
        bpz_z_err = np.abs(bpz_zphot - bpz_zspec) / (1. + bpz_zspec)
        bpz_z_err_sigmas = np.abs(bpz_zphot - bpz_zspec)/bpz_zphot_err
        bpz_nnsorted_predictions, bpz_nnsorted_is_goodfit, bpz_nnsorted_zerr, bpz_nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(bpz_nnc_results, bpz_is_goodfit, bpz_z_err, bpz_z_err_sigmas)))))
        bpzsorted_predictions, bpzsorted_zerr, bpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(bpz_zphot_err, bpz_z_err, bpz_z_err_sigmas))))))

        bpz_nnsorted_is_goodfit = bpz_nnsorted_is_goodfit.astype(bool)
        bpz_n_frac = np.arange(len(bpz_nnsorted_predictions), dtype = float)[::-1]/len(bpz_nnsorted_predictions)

    fig = plt.figure(figsize = (24,8))

    sp_nmad = fig.add_subplot(131)
    sp_outlier = fig.add_subplot(132)
    sp_sigma = fig.add_subplot(133)

    sp_nmad.plot(nmad_nn, n_frac_nn, label = r'NNC', color = 'C0')
    sp_nmad.plot(nmad_tpz, n_frac_tpz, label = r'TPZ', color = 'k')

    sp_outlier.plot(outlier_nn, n_frac_nn, label = r'NNC', color = 'C0')
    sp_outlier.plot(outlier_tpz, n_frac_tpz, label = r'TPZ', color = 'k')

    sp_sigma.plot(sigma_nn, n_frac_nn, label = r'NNC', color = 'C0')
    sp_sigma.plot(sigma_tpz, n_frac_tpz, label = r'TPZ', color = 'k')

    for cutoff in np.arange(0.1,1,.1):
        thisy = np.interp(cutoff, nnsorted_predictions, n_frac_nn)
        thisx_nmad = np.interp(cutoff, nnsorted_predictions, nmad_nn)
        thisx_sigma = np.interp(cutoff, nnsorted_predictions, sigma_nn)
        thisx_outlier = np.interp(cutoff, nnsorted_predictions, outlier_nn)
        if all(np.isfinite([thisx_nmad, thisx_sigma, thisx_outlier])):
            sp_nmad.annotate('%.1f'%(cutoff), (thisx_nmad, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_sigma.annotate('%.1f'%(cutoff), (thisx_sigma, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_outlier.annotate('%.1f'%(cutoff), (thisx_outlier, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')

    sp_nmad.set_xlim(0, np.maximum(nmad_nn[0], nmad_tpz[0]))
    sp_outlier.set_xlim(0, np.maximum(outlier_nn[0], outlier_tpz[0]))
    sp_sigma.set_xlim(0, np.maximum(sigma_nn[0], sigma_tpz[0]))

    if bpz_appdir != None:
        sigma_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([np.std(bpz_nnsorted_zerr[x:]) for x in inds]))
        nmad_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([1.4826*np.median(bpz_nnsorted_zerr[x:]) for x in inds]))
        outlier_bpznn = np.interp(np.arange(len(bpz_nnsorted_zerr)), inds, np.array([np.sum(bpz_nnsorted_zerr[x:] >= outlier_val)/float(len(bpz_nnsorted_zerr) - x) for x in inds]))
        sigma_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([np.std(bpzsorted_zerr[x:]) for x in inds]))
        nmad_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([1.4826*np.median(bpzsorted_zerr[x:]) for x in inds]))
        outlier_bpz = np.interp(np.arange(len(bpzsorted_zerr)), inds, np.array([np.sum(bpzsorted_zerr[x:] >= outlier_val)/float(len(bpzsorted_zerr) - x) for x in inds]))

        sp_nmad.plot(nmad_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')
        sp_outlier.plot(outlier_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')
        sp_sigma.plot(sigma_bpznn, bpz_n_frac, label = r'BPZ NNC', color = 'r')

        sp_nmad.plot(nmad_bpz, bpz_n_frac, label = r'BPZ', color = 'pink')
        sp_outlier.plot(outlier_bpz, bpz_n_frac, label = r'BPZ', color = 'pink')
        sp_sigma.plot(sigma_bpz, bpz_n_frac, label = r'BPZ', color = 'pink')

    sp_nmad.set_ylabel('N$_{frac}$')
    sp_nmad.set_xlabel('NMAD')
    sp_outlier.set_xlabel(r'Outlier Fraction $\left( \frac{\Delta z}{1+z} > ' + '%.2f' % outlier_val + r'\right)$')
    sp_sigma.set_xlabel(r'$\sigma$')

    sp_nmad.set_ylim(0,1)
    sp_outlier.set_ylim(0,1)
    sp_sigma.set_ylim(0,1)

    sp_outlier.set_yticklabels([])
    sp_sigma.set_yticklabels([])

    sp_nmad.legend(loc = 'upper left', fontsize = 20)

    plt.subplots_adjust(wspace = 0, hspace = 0)


def plot_sigma_nfrac_comparison_threepanel_nnc4(run_dir = './tpzruns/default_run/', nn_identifier = 'nnc4_', goodfit_conf_lim = 0):
    

    # FIRST PANEL

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc4', 'nnr').replace('_corr', '')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    good_inds = (np.all(np.isfinite(nnc_results), axis = 1) & np.isfinite(nnr_results))

    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    nnr_results = nnr_results[good_inds]

    nnc_results = nnc_results/np.sum(nnc_results, axis = 1).reshape(-1,1)

    nnc_class = np.argmax(nnc_results, axis = 1)

    zphot_nnc = np.copy(zphot)
    zphot_nnc[(nnc_class == 2) | (nnc_class == 3)] = (zphot - nnr_results)[(nnc_class == 2) | (nnc_class == 3)] #NNR photoz is better
    goodfit_conf = np.maximum(nnc_results.T[1], nnc_results.T[3])
    # is_goodfit = ((nnc_class == 1) | (nnc_class == 3)) & (goodfit_conf > nn_boundary)
    use_nnr = (nnc_class.T[2] | nnc_class.T[3])

    z_err = np.abs(zphot - zspec)/(1+zspec)
    z_err_sigmas = np.abs(zphot - zspec)/zphot_err

    z_err_nnc = np.abs(zphot_nnc - zspec)/(1+zspec)
    z_err_sigmas_nnc = np.abs(zphot_nnc - zspec)/zphot_err

    # if err_type == 'tpz':
    #     z_err = np.abs(zphot - zspec) / (1. + zspec)
    #     z_err_sigmas = np.abs(zphot - zspec)/zphot_err
    # elif err_type == 'nnr':
    #     z_err = np.abs(zphot - nnr_results - zspec)/(1+zspec)
    #     z_err_sigmas = np.abs(zphot - nnr_results - zspec)/zphot_err
    
    # is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

    goodfit_lim_inds = goodfit_conf > goodfit_conf_lim

    nnsorted_predictions, nnsorted_zerr, nnsorted_zerr_sigmas = np.array(list(zip(*sorted(zip(goodfit_conf[goodfit_lim_inds], z_err_nnc[goodfit_lim_inds], z_err_sigmas_nnc[goodfit_lim_inds])))))

    tpzsorted_predictions, tpzsorted_zerr, tpzsorted_zerr_sigmas = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err, z_err_sigmas))))))

    n_frac_nn = np.arange(len(nnsorted_predictions), dtype = float)[::-1]/len(nnsorted_predictions)
    n_frac_tpz = np.arange(len(tpzsorted_predictions), dtype = float)[::-1]/len(tpzsorted_predictions)

    if len(nnsorted_zerr) > 10000:
        inds = np.linspace(0, len(nnsorted_zerr), 10000, dtype = int)
    else:
        inds = np.arange(len(nnsorted_zerr))

    sigma_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([np.std(nnsorted_zerr[x:]) for x in inds]))
    nmad_nn = np.interp(np.arange(len(nnsorted_zerr)), inds, np.array([1.4826*np.median(nnsorted_zerr[x:]) for x in inds]))
    outlier_nn = np.interp(np.arange(len(nnsorted_zerr_sigmas)), inds, np.array([np.sum(nnsorted_zerr_sigmas[x:] >= 2)/float(len(nnsorted_zerr_sigmas) - x) for x in inds]))

    sigma_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([np.std(tpzsorted_zerr[x:]) for x in inds]))
    nmad_tpz = np.interp(np.arange(len(tpzsorted_zerr)), inds, np.array([1.4826*np.median(tpzsorted_zerr[x:]) for x in inds]))
    outlier_tpz = np.interp(np.arange(len(tpzsorted_zerr_sigmas)), inds, np.array([np.sum(tpzsorted_zerr_sigmas[x:] >= 2)/float(len(tpzsorted_zerr_sigmas) - x) for x in inds]))

    fig = plt.figure(figsize = (24,8))

    sp_nmad = fig.add_subplot(131)
    sp_outlier = fig.add_subplot(132)
    sp_sigma = fig.add_subplot(133)

    sp_nmad.plot(nmad_nn, n_frac_nn, label = r'NNC', color = 'C0')
    sp_nmad.plot(nmad_tpz, n_frac_tpz, label = r'TPZ', color = 'k')

    sp_outlier.plot(outlier_nn, n_frac_nn, label = r'NNC', color = 'C0')
    sp_outlier.plot(outlier_tpz, n_frac_tpz, label = r'TPZ', color = 'k')

    sp_sigma.plot(sigma_nn, n_frac_nn, label = r'NNC', color = 'C0')
    sp_sigma.plot(sigma_tpz, n_frac_tpz, label = r'TPZ', color = 'k')

    for cutoff in np.arange(0.1,1,.1):
        thisy = np.interp(cutoff, nnsorted_predictions, n_frac_nn)
        thisx_nmad = np.interp(cutoff, nnsorted_predictions, nmad_nn)
        thisx_sigma = np.interp(cutoff, nnsorted_predictions, sigma_nn)
        thisx_outlier = np.interp(cutoff, nnsorted_predictions, outlier_nn)
        if all(np.isfinite([thisx_nmad, thisx_sigma, thisx_outlier])):
            sp_nmad.annotate('%.1f'%(cutoff), (thisx_nmad, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_sigma.annotate('%.1f'%(cutoff), (thisx_sigma, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')
            sp_outlier.annotate('%.1f'%(cutoff), (thisx_outlier, thisy), (-20, 20), textcoords = 'offset pixels', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')

    sp_nmad.set_xlim(0, np.maximum(nmad_nn[0], nmad_tpz[0]))
    sp_outlier.set_xlim(0, np.maximum(outlier_nn[0], outlier_tpz[0]))
    sp_sigma.set_xlim(0, np.maximum(sigma_nn[0], sigma_tpz[0]))

    sp_nmad.set_ylabel('N$_{frac}$')
    sp_nmad.set_xlabel('NMAD')
    sp_outlier.set_xlabel(r'$2\sigma_{TPZ}$ Outlier Fraction')
    sp_sigma.set_xlabel(r'$\sigma$')

    sp_nmad.set_ylim(0,1)
    sp_outlier.set_ylim(0,1)
    sp_sigma.set_ylim(0,1)

    sp_outlier.set_yticklabels([])
    sp_sigma.set_yticklabels([])

    sp_nmad.legend(loc = 'upper left', fontsize = 20)

    plt.subplots_adjust(wspace = 0, hspace = 0)



def plot_sigma_nfrac_comparison_double(run_dir = './tpzruns/default_run/', zerr_quantity = 'sigma', nn_identifier = 'nnc_'):
    

    # FIRST PANEL

    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    good_inds = (np.isfinite(nnc_results) & np.isfinite(nnr_results))

    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    zphot_err = features_file['zerr'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    nnr_results = nnr_results[good_inds]
    z_err = np.abs(zphot - zspec) / (1. + zspec)
    z_err_nnr = np.abs(zphot - nnr_results - zspec)/(1+zspec)
    
    is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

    sorted_predictions, sorted_is_goodfit, sorted_z_err, sorted_z_err_nnr = np.array(list(zip(*sorted(zip(nnc_results, is_goodfit, z_err, z_err_nnr)))))
    sorted_is_goodfit = sorted_is_goodfit.astype(bool)
    sorted_zphoterr_predictions, sorted_zphoterr_z_err = np.array(list(zip(*reversed(sorted(zip(zphot_err, z_err))))))

    n_frac = np.arange(len(sorted_predictions), dtype = float)[::-1]/len(sorted_predictions)

    if zerr_quantity == 'sigma':
        if len(sorted_z_err) > 10000:
            inds = np.linspace(0, len(sorted_z_err), 10000, dtype = int)
        else:
            inds = np.arange(len(sorted_z_err))
        width = np.interp(np.arange(len(sorted_z_err)), inds, np.array([np.std(sorted_z_err[x:]) for x in inds]))
        width_zphoterr = np.interp(np.arange(len(sorted_zphoterr_z_err)), inds, np.array([np.std(sorted_zphoterr_z_err[x:]) for x in inds]))
        # width_nnr = np.interp(np.arange(len(sorted_z_err)), inds, np.array([np.std(sorted_z_err_nnr[x:]) for x in inds]))
        label = r'$\sigma \left(\frac{\Delta z}{1+z}\right)$'
    elif zerr_quantity == 'nmad':
        if len(sorted_z_err) > 10000:
            inds = np.linspace(0, len(sorted_z_err), 10000, dtype = int)
        else:
            inds = np.arange(len(sorted_z_err))
        width = np.interp(np.arange(len(sorted_z_err)), inds, np.array([1.4826*np.median(sorted_z_err[x:]) for x in inds]))
        width_zphoterr = np.interp(np.arange(len(sorted_zphoterr_z_err)), inds, np.array([1.4826*np.median(sorted_zphoterr_z_err[x:]) for x in inds]))
        # width_nnr = np.interp(np.arange(len(sorted_z_err)), inds, np.array([1.4826*np.median(sorted_z_err_nnr[x:]) for x in inds]))
        label = 'NMAD'

    fig = plt.figure(figsize = (8,8))

    sp = fig.add_subplot(111)
    # sp_04 = fig.add_subplot(122)

    if os.path.isdir(run_dir[:-1] + '_04/'):
        sp.plot(n_frac, width, label = r'$NNC \rightarrow 0.02$')
    else:
        sp.plot(n_frac, width, label = r'NNC')
    for cutoff in np.arange(0.1,1,.1):
        thisx = np.interp(cutoff, sorted_predictions, n_frac)
        thisy = np.interp(cutoff, sorted_predictions, width)
        sp.annotate('%.1f'%(cutoff), (thisx, thisy), (thisx-.05, thisy+ 0.005), arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')

    sp.plot(n_frac, width_zphoterr, label = r'Uncertainty Cut', color = 'k')
    # for cutoff in np.arange(0.1,1,.1):
    #     thisx = np.interp(cutoff, sorted_predictions, n_frac)
    #     thisy = np.interp(cutoff, sorted_predictions, width)
    #     sp.annotate('%.1f'%(cutoff), (thisx, thisy), (thisx-.05, thisy+ 0.005), arrowprops = dict(arrowstyle = '-', ec = 'C0'), color = 'C0', fontsize = 20, fontweight = 'normal')


    # sp.plot(n_frac, width_nnr)
    sp.set_ylim(0,max(width))
    sp.set_xlim(0,1)

    run_dir = run_dir[:-1] + '_04/'

    if os.path.isdir(run_dir):

        nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

        if len(nnc_folders) == 1:
            nnc_folder = nnc_folders[0]
        else:
            print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

            for thisfolder in nnc_folders:
                print(thisfolder)
            # [print(thisfolder) for thisfolder in nnc_folders]
            return None

        nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')

        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
        # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
        nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

        good_inds = (np.isfinite(nnc_results) & np.isfinite(nnr_results))

        zspec = features_file['specz'].to_numpy()[good_inds]
        zphot = features_file['zphot'].to_numpy()[good_inds]
        nnc_results = nnc_results[good_inds]
        nnr_results = nnr_results[good_inds]
        z_err = np.abs(zphot - zspec) / (1. + zspec)
        z_err_nnr = np.abs(zphot - nnr_results - zspec)/(1+zspec)
        
        is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

        sorted_predictions, sorted_is_goodfit, sorted_z_err, sorted_z_err_nnr = np.array(list(zip(*sorted(zip(nnc_results, is_goodfit, z_err, z_err_nnr)))))
        sorted_is_goodfit = sorted_is_goodfit.astype(bool)

        n_frac = np.arange(len(sorted_predictions), dtype = float)[::-1]/len(sorted_predictions)

        if zerr_quantity == 'sigma':
            if len(sorted_z_err) > 10000:
                inds = np.linspace(0, len(sorted_z_err), 10000, dtype = int)
            else:
                inds = np.arange(len(sorted_z_err))
            width = np.interp(np.arange(len(sorted_z_err)), inds, np.array([np.std(sorted_z_err[x:]) for x in inds]))
            # width_nnr = np.interp(np.arange(len(sorted_z_err)), inds, np.array([np.std(sorted_z_err_nnr[x:]) for x in inds]))
        elif zerr_quantity == 'nmad':
            if len(sorted_z_err) > 10000:
                inds = np.linspace(0, len(sorted_z_err), 10000, dtype = int)
            else:
                inds = np.arange(len(sorted_z_err))
            width = np.interp(np.arange(len(sorted_z_err)), inds, np.array([1.4826*np.median(sorted_z_err[x:]) for x in inds]))
            # width_nnr = np.interp(np.arange(len(sorted_z_err)), inds, np.array([1.4826*np.median(sorted_z_err_nnr[x:]) for x in inds]))

        sp.plot(n_frac, width, label = r'$NNC \rightarrow 0.04$')
        for cutoff in np.arange(0.1,1,.1):
            thisx = np.interp(cutoff, sorted_predictions, n_frac)
            thisy = np.interp(cutoff, sorted_predictions, width)
            sp.annotate('%.1f'%(cutoff), (thisx, thisy), (thisx+.05, thisy- 0.005), arrowprops = dict(arrowstyle = '-', ec = 'C1'), color = 'C1', fontsize = 20, fontweight = 'normal')

    # sp_04.plot(n_frac, width_nnr)
    # sp_04.set_ylim(0,max(width))
    # sp_04.set_xlim(0,1)
    # sp_04.set_yticklabels([])

    sp.set_xlabel('N$_{frac}$')
    sp.set_ylabel(label)
    sp.legend(loc = 'upper left', fontsize = 20)

    plt.subplots_adjust(wspace = 0, hspace = 0)





def plot_sigma_nfrac_comparison(run_dir = './tpzruns/default_run/', zerr_quantity = 'sigma', nn_identifier = 'nnc_'):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')

    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')
    # nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]
    nnr_results = np.loadtxt(nnr_folder + 'results_application.dat')

    good_inds = (np.isfinite(nnc_results) & np.isfinite(nnr_results))

    zspec = features_file['specz'].to_numpy()[good_inds]
    zphot = features_file['zphot'].to_numpy()[good_inds]
    nnc_results = nnc_results[good_inds]
    nnr_results = nnr_results[good_inds]
    z_err = np.abs(zphot - zspec) / (1. + zspec)
    z_err_nnr = np.abs(zphot - nnr_results - zspec)/(1+zspec)
    
    is_goodfit = (np.abs(zphot - zspec) < (1. + zspec)*0.02)

    sorted_predictions, sorted_is_goodfit, sorted_z_err, sorted_z_err_nnr = np.array(list(zip(*sorted(zip(nnc_results, is_goodfit, z_err, z_err_nnr)))))
    sorted_is_goodfit = sorted_is_goodfit.astype(bool)

    n_frac = np.arange(len(sorted_predictions), dtype = float)[::-1]/len(sorted_predictions)

    if zerr_quantity == 'sigma':
        if len(sorted_z_err) > 10000:
            inds = np.linspace(0, len(sorted_z_err), 10000, dtype = int)
        else:
            inds = np.arange(len(sorted_z_err))
        width = np.interp(np.arange(len(sorted_z_err)), inds, np.array([np.std(sorted_z_err[x:]) for x in inds]))
        width_nnr = np.interp(np.arange(len(sorted_z_err)), inds, np.array([np.std(sorted_z_err_nnr[x:]) for x in inds]))
        label = r'$\sigma$'
    elif zerr_quantity == 'nmad':
        if len(sorted_z_err) > 10000:
            inds = np.linspace(0, len(sorted_z_err), 10000, dtype = int)
        else:
            inds = np.arange(len(sorted_z_err))
        width = np.interp(np.arange(len(sorted_z_err)), inds, np.array([1.4826*np.median(sorted_z_err[x:]) for x in inds]))
        width_nnr = np.interp(np.arange(len(sorted_z_err)), inds, np.array([1.4826*np.median(sorted_z_err_nnr[x:]) for x in inds]))
        label = 'NMAD'

    fig = plt.figure(figsize = (8,8))

    sp_err = fig.add_subplot(221)
    sp_both = fig.add_subplot(222)
    sp_frac = fig.add_subplot(224)

    sp_err.plot(sorted_predictions, width)
    sp_err.plot(sorted_predictions, width_nnr)
    sp_both.plot(n_frac, width)
    sp_both.plot(n_frac, width_nnr)
    sp_frac.plot(n_frac, sorted_predictions)

    sp_frac.set_xlabel('Population Fraction')
    sp_frac.set_ylabel('NNC')
    sp_err.set_xlabel('NNC')
    sp_err.set_ylabel(label)

    sp_both.set_xticklabels([])
    sp_both.set_yticklabels([])

    sp_err.set_xlim(0,1)
    sp_frac.set_ylim(0,1)

    sp_err.set_ylim(0, max(width))
    sp_frac.set_xlim(0,1)
    sp_both.set_ylim(0,max(width))
    sp_both.set_xlim(0,1)

    plt.subplots_adjust(wspace = 0, hspace = 0)








def plot_pipeline_results_delete_later(run_dir = './tpzruns/median_shift/', nn_identifier = 'nnc_', nn_boundary = 0.65, plot_percentile_lines = True):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')
        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    nnr_folder = nnc_folder.replace('nnc', 'nnr').replace('_corr', '')

    features_file = read_csv(glob(run_dir + '*.nnc_test')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_test.dat')
    nnr_results = np.loadtxt(nnr_folder + 'results_test.dat')[np.loadtxt(glob(run_dir + '*.nnc_test_inds')[0], usecols = [1], dtype = int)]

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()

    nnc_lim_inds = nnc_results > nn_boundary


    fig = plt.figure(figsize = (18,6))
    sp1 = fig.add_subplot(131)
    sp2 = fig.add_subplot(132)
    sp3 = fig.add_subplot(133)

    plot1 = np.histogram2d(zspec, zphot, range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp1.imshow(np.log10(plot1).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot2 = np.histogram2d(zspec, zphot - nnr_results, range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp2.imshow(np.log10(plot2).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    plot3 = np.histogram2d(zspec[nnc_lim_inds], zphot[nnc_lim_inds], range = ((0,1.5),(0,1.5)), bins = 50)[0]
    sp3.imshow(np.log10(plot3).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'plasma_r')

    xdata = [zspec, zspec, zspec[nnc_lim_inds]]
    ydata = [zphot, zphot - nnr_results, zphot[nnc_lim_inds]]
    error = []

    if plot_percentile_lines:
        plot_percentiles = [68, 95]
        xgrid = np.linspace(0,1.5,1000)

        for thissubplot, thisx, thisy in zip([sp1, sp2, sp3], xdata, ydata):

            thiserror = np.abs(thisy - thisx)/ (1+thisx)
            error.append(thiserror)

            for this_percentile in plot_percentiles:

                ind = np.argmin(np.abs(np.percentile(thiserror, this_percentile) - thiserror))
                thissubplot.plot(xgrid, xgrid - (1+xgrid)*thiserror[ind], color = 'k')
                thissubplot.plot(xgrid, xgrid + (1+xgrid)*thiserror[ind], color = 'k')


    for thissubplot, thiserror in zip([sp1, sp2, sp3], error):

        # txt1 = thissubplot.text(0.98, 0.02, '$P_{68}=%.3f$\n$P_{95}=%.3f$\n' % tuple(np.percentile(thiserror, [68,95])) + 
        #                             '$NMAD=%.3f$\n' % (1.4826*np.median(thiserror)) +
        #                             r'$\sigma=' + '%.3f$' % np.std(thiserror), 
        #                             color = 'white', fontsize = 18, ha = 'right', va = 'bottom', transform = thissubplot.transAxes)
        # txt1.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

        txt2 = thissubplot.text(0.02, 0.98, '$N=%i$' % len(thiserror), color = 'white', fontsize = 20, ha = 'left', va = 'top', transform = thissubplot.transAxes)
        txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='k')])

        thissubplot.set_xlim(0,1.5)
        thissubplot.set_ylim(0,1.5)

    fig.text(0.5, 0., 'True Distance', fontsize = 30)
    sp1.set_ylabel('Fit Distance', fontsize = 30)

    sp2.set_yticklabels([])
    sp3.set_yticklabels([])

    sp1ticks = [thistick.get_text() for thistick in sp1.get_xticklabels()]
    sp1ticks[-1] = ''
    sp2ticks = [thistick.get_text() for thistick in sp2.get_xticklabels()]
    sp2ticks[-1] = ''

    sp1.set_xticklabels(sp1ticks)
    sp2.set_xticklabels(sp2ticks)


    plt.subplots_adjust(wspace = 0)











def plot_z_stats(stat = 'sigma68', run_dir = './tpzruns/default_run/', data_name = 'default', nn_identifier = '_epoch1000_corr', nn_lim = 0.65, zlo = 0., zhi = 1.5, bins = 15, plot_des_reference = True):

    with open(run_dir + data_name + '.nntest', 'r') as readfile:
        column_names = [thisname for thisname in readfile.readline()[:-1].split(' ') if (thisname != '#') and (thisname != '')]
    testfile = read_csv(run_dir + data_name  + '.nntest', header = None, comment = '#', delimiter = '\s+', names = column_names)

    specz = testfile['specz']
    photz = testfile['zphot']
    photz_err = testfile['zerr']
    # nnr_result = np.loadtxt(run_dir + 'nnr' + nn_identifier + '/results_test.dat')
    nnc_result = np.loadtxt(run_dir + 'nnc' + nn_identifier + '/results_test.dat')

    bin_edges = np.linspace(zlo, zhi, bins+1)

    deltaz = photz - specz
    deltaz_avg = np.average(deltaz)
    deltaz_50 = np.median(deltaz)
    deltaz_sigma = np.std(deltaz)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    bin_centers = bin_edges[:-1] + 0.5*np.diff(bin_edges)

    if plot_des_reference:
        if stat == 'sigma68':
            sp.plot([0,1],[0.12,0.12], color = 'k', linestyle = '--', transform = sp.get_yaxis_transform())
        elif stat == 'out2sigma':
            sp.plot([0,1],[0.1,0.1], color = 'k', linestyle = '--', transform = sp.get_yaxis_transform())
        elif stat == 'out3sigma':
            sp.plot([0,1],[0.015,0.015], color = 'k', linestyle = '--', transform = sp.get_yaxis_transform())

    for thisnnlim in [0., nn_lim]:

        statlist = []

        for binlo, binhi in zip(bin_edges[:-1], bin_edges[1:]):

            inds = (photz > binlo) & (photz < binhi) & (nnc_result > thisnnlim)

            if sum(inds) > 0:

                if stat == 'sigma68':
                    # Half-width of the interval around deltaz_50 containing 68 percent of the galaxies; should be less than 0.12
                    statnum = np.percentile(abs((deltaz-deltaz_50)[inds]), 68)
                    ylabel = r'$\sigma_{68}$'

                elif stat == 'out2sigma':
                    # Fraction of galaxies with abs(deltaz - deltaz_avg) > 2 * deltaz_sigma; should be less than 0.1
                    statnum = sum(np.abs((deltaz - deltaz_avg)[inds]) > 2*deltaz_sigma)/float(len(inds))
                    ylabel = r'out$_{2\sigma}$'

                elif stat == 'out3sigma':
                    # Fraction of galaxies with abs(deltaz - deltaz_avg) > 3 * deltaz_sigma; should be less than 0.015
                    statnum = sum(np.abs((deltaz - deltaz_avg)[inds]) > 3*deltaz_sigma)/float(len(inds))
                    ylabel = r'out$_{3\sigma}$'

                statlist.append(statnum)

            else:
                statlist.append(np.nan)

        sp.scatter(bin_centers, statlist, label = 'NNC > %.2f' % thisnnlim)


    sp.set_xlabel('$z_{phot}$')
    sp.set_ylabel(ylabel)




def plot_color_color_selection(run_dir = './tpzruns/MatchCOSMOS2015/', color1bands = 'ri', color2bands = 'gr', nn_identifier = 'nnc_', sample_fraction = 0.333):

        nnc_folders = glob(run_dir + '*' + nn_identifier + '*0/')

        if len(nnc_folders) == 1:
            nnc_folder = nnc_folders[0]
        elif len(nnc_folders) == 0:
            print('There are no folders matching the nn_identifier.')
            return None
        else:
            print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

            for thisfolder in nnc_folders:
                print(thisfolder)
            return None

        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

        # zspec = features_file['specz'].to_numpy()
        # zphot = features_file['zphot'].to_numpy()
        # zconf = features_file['zconf'].to_numpy()
        # zphot_err = features_file['zerr'].to_numpy()

        color1mag1 = features_file[color1bands[0]].to_numpy()
        color1mag2 = features_file[color1bands[1]].to_numpy()
        color2mag1 = features_file[color2bands[0]].to_numpy()
        color2mag2 = features_file[color2bands[1]].to_numpy()

        color1 = color1mag1 - color1mag2
        color2 = color2mag1 - color2mag2

        nn_boundary = np.sort(nnc_results)[::-1][int(len(nnc_results)*sample_fraction)]
        nnc_lim_inds = (nnc_results > nn_boundary)

        goodhist = np.histogram2d(color1[nnc_lim_inds], color2[nnc_lim_inds], range = ((-1,3),(-1,3)), bins = 100)[0]
        badhist  = np.histogram2d(color1[~nnc_lim_inds], color2[~nnc_lim_inds], range = ((-1,3),(-1,3)), bins = 100)[0]

        fig = plt.figure(figsize = (15,8))
        sp1 = fig.add_subplot(121)
        sp2 = fig.add_subplot(122)

        sp1.imshow(np.log10(goodhist).T, origin = 'lower', extent = (-1,3,-1,3), cmap = 'YlGnBu')
        sp2.imshow(np.log10(badhist).T, origin = 'lower', extent = (-1,3,-1,3), cmap = 'YlGnBu')

        plt.subplots_adjust(wspace = 0.)

        sp1.text(0.02, 0.98, '$C_{NNC} > %.2f$' % nn_boundary, ha = 'left', va = 'top', fontsize = 24, transform = sp1.transAxes)
        sp2.text(0.02, 0.98, r'$C_{NNC} \leq ' + '%.2f$' % nn_boundary, ha = 'left', va = 'top', fontsize = 24, transform = sp2.transAxes)

        sp2.set_yticklabels([])

        fig.text(0.5, 0.02, '({}-{})'.format(*color1bands), fontsize = 24)
        sp1.set_ylabel('({}-{})'.format(*color2bands), fontsize = 24)



def plot_color_mag_selection(run_dir = './tpzruns/MatchCOSMOS2015/', colorbands = 'gr', magband = 'r', nn_identifier = 'nnc_', sample_fraction = 0.333):

        nnc_folders = glob(run_dir + '*' + nn_identifier + '*0/')

        if len(nnc_folders) == 1:
            nnc_folder = nnc_folders[0]
        elif len(nnc_folders) == 0:
            print('There are no folders matching the nn_identifier.')
            return None
        else:
            print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

            for thisfolder in nnc_folders:
                print(thisfolder)
            return None

        features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
        nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

        # zspec = features_file['specz'].to_numpy()
        # zphot = features_file['zphot'].to_numpy()
        # zconf = features_file['zconf'].to_numpy()
        # zphot_err = features_file['zerr'].to_numpy()

        colormag1 = features_file[colorbands[0]].to_numpy()
        colormag2 = features_file[colorbands[1]].to_numpy()
        mag = features_file[magband].to_numpy()

        color = colormag1 - colormag2

        nn_boundary = np.sort(nnc_results)[::-1][int(len(nnc_results)*sample_fraction)]
        nnc_lim_inds = (nnc_results > nn_boundary)

        goodhist = np.histogram2d(mag[nnc_lim_inds], color[nnc_lim_inds], range = ((18,26),(-1,3)), bins = 100)[0]
        badhist  = np.histogram2d(mag[~nnc_lim_inds], color[~nnc_lim_inds], range = ((18,26),(-1,3)), bins = 100)[0]

        fig = plt.figure(figsize = (15,8))
        sp1 = fig.add_subplot(121)
        sp2 = fig.add_subplot(122)

        sp1.imshow(np.log10(goodhist).T, origin = 'lower', extent = (18,26,-1,3), cmap = 'YlGnBu', aspect = 'auto')
        sp2.imshow(np.log10(badhist).T, origin = 'lower', extent = (18,26,-1,3), cmap = 'YlGnBu', aspect = 'auto')

        plt.subplots_adjust(wspace = 0.)

        sp1.text(0.02, 0.98, '$C_{NNC} > %.2f$' % nn_boundary, ha = 'left', va = 'top', fontsize = 24, transform = sp1.transAxes)
        sp2.text(0.02, 0.98, r'$C_{NNC} \leq ' + '%.2f$' % nn_boundary, ha = 'left', va = 'top', fontsize = 24, transform = sp2.transAxes)

        sp2.set_yticklabels([])

        sp1.set_ylabel('({}-{})'.format(*colorbands), fontsize = 24)
        fig.text(0.5, 0.02, magband, fontsize = 24)





def plot_colormag_split(ishift = -2.4, gzshift = 0.34, run_dir = './tpzruns/median_shift/', phot_file = './HSC/HSC_wide_clean_pdr2_phot_depth_1M.fits'):

    col_names = ['specz','g','r','i','z','y','gr','ri','iz','zy','gri','riz','izy','eg','er','ei','ez','ey','egr','eri','eiz','ezy','egri','eriz','eizy']

    tpz_train_file = read_csv(glob(run_dir + '*.tpz_train')[0], names = col_names, comment = '#', delimiter = '\s+')
    tpz_test_file = read_csv(glob(run_dir + '*.tpz_test')[0], names = col_names, comment = '#', delimiter = '\s+')
    phot_file = fits.open(phot_file)[1].data

    gz_spec = np.append(tpz_train_file['g'].to_numpy() - tpz_train_file['z'].to_numpy(), tpz_test_file['g'].to_numpy() - tpz_test_file['z'].to_numpy())
    i_spec = np.append(tpz_train_file['i'].to_numpy(), tpz_test_file['i'].to_numpy())

    gz_phot = phot_file['g_cmodel_mag'] - phot_file['z_cmodel_mag']
    i_phot = phot_file['i_cmodel_mag']

    fig = plt.figure(figsize = (24,4))
    sp1 = fig.add_subplot(161)
    sp2 = fig.add_subplot(162)
    sp3 = fig.add_subplot(163)
    sp4 = fig.add_subplot(164)
    sp5 = fig.add_subplot(165)
    sp6 = fig.add_subplot(166)

    spechist, x_edges, y_edges = np.histogram2d(gz_spec, i_spec, range = ((-4,9),(13,28)), bins = 100)
    phothist = np.histogram2d(gz_phot, i_phot, range = ((-4,9),(13,28)), bins = 100)[0]

    sp1.imshow(np.log10(spechist).T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')
    sp2.imshow(np.log10(phothist).T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')

    x_shift = 0
    y_shift = 0

    if gzshift > 0:
        x_shift = np.argmin(np.abs(x_edges - min(x_edges) - gzshift))
    if gzshift < 0:
        x_shift = np.argmin(np.abs(x_edges - max(x_edges) - gzshift))-len(y_edges)
    if ishift > 0:
        y_shift = np.argmin(np.abs(y_edges - min(y_edges) - ishift))
    if ishift < 0:
        y_shift = np.argmin(np.abs(y_edges - max(y_edges) - ishift))-len(y_edges)

    eval_ratiohist = spechist/(spechist + phothist)

    sp3.imshow(eval_ratiohist.T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')

    eval_ratiohist = np.roll(eval_ratiohist, x_shift, axis = 0)
    eval_ratiohist = np.roll(eval_ratiohist, y_shift, axis = 1)

    if x_shift > 0:
        eval_ratiohist[:x_shift] = np.nan
    if x_shift < 0:
        eval_ratiohist[x_shift:] = np.nan
    if y_shift > 0:
        eval_ratiohist[:,:y_shift] = np.nan
    if y_shift < 0:
        eval_ratiohist[:,y_shift:] = np.nan

    # eval_ratiohist = np.nan_to_num(eval_ratiohist, nan = 0.)

    sp4.imshow(eval_ratiohist.T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')

    train_hist = np.histogram2d(tpz_train_file['g'] - tpz_train_file['z'], tpz_train_file['i'], range = ((-4,9),(13,28)), bins = 100)[0]
    test_hist = np.histogram2d(tpz_test_file['g'] - tpz_test_file['z'], tpz_test_file['i'], range = ((-4,9),(13,28)), bins = 100)[0]

    sp5.imshow(np.log10(train_hist).T, cmap = 'inferno_r', origin = 'lower', extent = (-4,9,13,28))
    sp6.imshow(np.log10(test_hist).T, cmap = 'inferno_r', origin = 'lower', extent = (-4,9,13,28))

    sp5.text(0.02, 0.98, '%i'%len(tpz_train_file['g']), transform = sp5.transAxes, ha = 'left', va = 'top')
    sp6.text(0.02, 0.98, '%i'%len(tpz_test_file['g']), transform = sp6.transAxes, ha = 'left', va = 'top')

    sp2.set_yticklabels([])
    sp3.set_yticklabels([])
    sp4.set_yticklabels([])
    sp5.set_yticklabels([])
    sp6.set_yticklabels([])

    plt.subplots_adjust(wspace = 0)

    fig.text(0.5, 0., 'g-z', fontsize = 24)
    sp1.set_ylabel('i', fontsize = 24)

    sp1.set_title('Spec', fontsize = 20)
    sp2.set_title('Phot', fontsize = 20)
    sp3.set_title('Spec/(Spec+Phot)', fontsize = 20)
    sp4.set_title('Shift [%.2f, %.2f]' % (gzshift, ishift), fontsize = 20)
    sp5.set_title('Train', fontsize = 20)
    sp6.set_title('Test', fontsize = 20)






def plot_colormag_match(run_data = './tpzruns/closest_match/match.tpz_app', phot_file = './HSC/HSC_wide_clean_pdr2_phot_depth_1M.fits', spec_file = './HSC/HSC_wide_clean_pdr2.fits'):

    g, i, z = np.loadtxt(run_data, usecols = [1,3,4], unpack = True)
    data_phot = fits.open(phot_file)[1].data
    data_spec = fits.open(spec_file)[1].data

    fig = plt.figure(figsize = (18,6))
    sp1 = fig.add_subplot(131)
    sp2 = fig.add_subplot(132)
    sp3 = fig.add_subplot(133)

    spec_hist = np.histogram2d(data_spec.g_cmodel_mag - data_spec.z_cmodel_mag,data_spec.i_cmodel_mag, range = ((-4,9),(13,28)), bins = 100)[0]
    phot_hist = np.histogram2d(data_phot.g_cmodel_mag - data_phot.z_cmodel_mag,data_phot.i_cmodel_mag, range = ((-4,9),(13,28)), bins = 100)[0]
    draw_hist = np.histogram2d(g-z, i, range = ((-4,9),(13,28)), bins = 100)[0]

    sp1.imshow(np.log10(spec_hist).T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')
    sp2.imshow(np.log10(phot_hist).T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')
    sp3.imshow(np.log10(draw_hist).T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')

    sp1.set_title('Spec')
    sp2.set_title('Phot')
    sp3.set_title('Drawn')

    sp2.set_xlabel('(g-z)')
    sp1.set_ylabel('i')







def plot_colormag_split_delete_later(ishift = -3.62, gzshift = 0.62, run_dir = './tpzruns/median_shift/', phot_file = './HSC/HSC_wide_clean_pdr2_phot_depth_1M.fits'):

    col_names = ['specz','g','r','i','z','y','gr','ri','iz','zy','gri','riz','izy','eg','er','ei','ez','ey','egr','eri','eiz','ezy','egri','eriz','eizy']

    tpz_train_file = read_csv(glob(run_dir + '*.tpz_train')[0], names = col_names, comment = '#', delimiter = '\s+')
    tpz_test_file = read_csv(glob(run_dir + '*.tpz_test')[0], names = col_names, comment = '#', delimiter = '\s+')
    phot_file = fits.open(phot_file)[1].data

    gz_spec = np.append(tpz_train_file['g'].to_numpy() - tpz_train_file['z'].to_numpy(), tpz_test_file['g'].to_numpy() - tpz_test_file['z'].to_numpy())
    i_spec = np.append(tpz_train_file['i'].to_numpy(), tpz_test_file['i'].to_numpy())

    gz_phot = phot_file['g_cmodel_mag'] - phot_file['z_cmodel_mag']
    i_phot = phot_file['i_cmodel_mag']

    fig = plt.figure(figsize = (24,4))
    sp1 = fig.add_subplot(161)
    sp2 = fig.add_subplot(162)
    sp3 = fig.add_subplot(163)
    sp4 = fig.add_subplot(164)
    sp5 = fig.add_subplot(165)
    sp6 = fig.add_subplot(166)

    spechist, x_edges, y_edges = np.histogram2d(gz_spec, i_spec, range = ((-4,9),(13,28)), bins = 100)
    phothist = np.histogram2d(gz_phot, i_phot, range = ((-4,9),(13,28)), bins = 100)[0]

    sp1.imshow(np.log10(spechist).T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')
    sp2.imshow(np.log10(phothist).T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')

    x_shift = 0
    y_shift = 0

    if gzshift > 0:
        x_shift = np.argmin(np.abs(x_edges - min(x_edges) - gzshift))
    if gzshift < 0:
        x_shift = np.argmin(np.abs(x_edges - max(x_edges) - gzshift))-len(y_edges)
    if ishift > 0:
        y_shift = np.argmin(np.abs(y_edges - min(y_edges) - ishift))
    if ishift < 0:
        y_shift = np.argmin(np.abs(y_edges - max(y_edges) - ishift))-len(y_edges)

    eval_ratiohist = spechist/(spechist + phothist)

    sp3.imshow(eval_ratiohist.T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')

    eval_ratiohist = np.roll(eval_ratiohist, x_shift, axis = 0)
    eval_ratiohist = np.roll(eval_ratiohist, y_shift, axis = 1)

    if x_shift > 0:
        eval_ratiohist[:x_shift] = np.nan
    if x_shift < 0:
        eval_ratiohist[x_shift:] = np.nan
    if y_shift > 0:
        eval_ratiohist[:,:y_shift] = np.nan
    if y_shift < 0:
        eval_ratiohist[:,y_shift:] = np.nan

    # eval_ratiohist = np.nan_to_num(eval_ratiohist, nan = 0.)

    sp4.imshow(eval_ratiohist.T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')

    train_hist = np.histogram2d(tpz_train_file['g'] - tpz_train_file['z'], tpz_train_file['i'], range = ((-4,9),(13,28)), bins = 100)[0]
    test_hist = np.histogram2d(tpz_test_file['g'] - tpz_test_file['z'], tpz_test_file['i'], range = ((-4,9),(13,28)), bins = 100)[0]

    sp5.imshow(np.log10(train_hist).T, cmap = 'inferno_r', origin = 'lower', extent = (-4,9,13,28))
    sp6.imshow(np.log10(test_hist).T, cmap = 'inferno_r', origin = 'lower', extent = (-4,9,13,28))

    sp1.text(0.02, 0.98, '$N=%i$'%len(i_spec), fontsize = 14, transform = sp1.transAxes, ha = 'left', va = 'top')
    sp2.text(0.02, 0.98, '$N=%i$'%len(i_phot), fontsize = 14, transform = sp2.transAxes, ha = 'left', va = 'top')

    sp5.text(0.02, 0.98, '$N=%i$'%len(tpz_train_file['g']), fontsize = 14, transform = sp5.transAxes, ha = 'left', va = 'top')
    sp6.text(0.02, 0.98, '$N=%i$'%len(tpz_test_file['g']), fontsize = 14, transform = sp6.transAxes, ha = 'left', va = 'top')

    sp2.set_yticklabels([])
    sp3.set_yticklabels([])
    sp4.set_yticklabels([])
    sp5.set_yticklabels([])
    sp6.set_yticklabels([])

    plt.subplots_adjust(wspace = 0)

    fig.text(0.5, 0., 'Spectral Slope', fontsize = 24)
    sp1.set_ylabel('Brightness', fontsize = 24)

    sp1.set_title('Truth', fontsize = 20)
    sp2.set_title('Application', fontsize = 20)
    sp3.set_title('Truth/(Truth+App)', fontsize = 20)
    sp4.set_title('Shift [%.2f, %.2f]' % (gzshift, ishift), fontsize = 20)
    sp5.set_title('Train', fontsize = 20)
    sp6.set_title('Test', fontsize = 20)

    sp1.set_ylim(28,13)
    sp2.set_ylim(28,13)
    sp3.set_ylim(28,13)
    sp4.set_ylim(28,13)
    sp5.set_ylim(28,13)
    sp6.set_ylim(28,13)





def plot_colormag_phot_spec_cosmos(color = 'gz', mag = 'i', phot_file = './HSC/HSC_wide_clean_pdr2_phot_depth_1M.fits', spec_file = './HSC/HSC_wide_clean_pdr2.fits', cosmos_match_file = './tpzruns/MatchCOSMOS2015/tpzrun.tpz_app'):

    data_phot = fits.open(phot_file)[1].data
    data_spec = fits.open(spec_file)[1].data
    cosmos_file = read_csv(cosmos_match_file, delim_whitespace = True)

    cosmos_colorband1 = cosmos_file[color[0]]
    cosmos_colorband2 = cosmos_file[color[1]]
    cosmos_magband = cosmos_file[mag]

    fig = plt.figure(figsize = (18,7))
    sp1 = fig.add_subplot(131)
    sp2 = fig.add_subplot(132)
    sp3 = fig.add_subplot(133)

    spec_hist = np.histogram2d(data_spec[f'{color[0]}_cmodel_mag'] - data_spec[f'{color[1]}_cmodel_mag'],data_spec[f'{mag}_cmodel_mag'], range = ((-4,9),(13,28)), bins = 100)[0]
    phot_hist = np.histogram2d(data_phot[f'{color[0]}_cmodel_mag'] - data_phot[f'{color[1]}_cmodel_mag'],data_phot[f'{mag}_cmodel_mag'], range = ((-4,9),(13,28)), bins = 100)[0]
    cosmos_hist = np.histogram2d(cosmos_colorband1 - cosmos_colorband2, cosmos_magband, range = ((-4,9),(13,28)), bins = 100)[0]

    sp1.imshow(np.log10(spec_hist).T, extent = (-4,9,13,28), cmap = 'YlGnBu', origin = 'lower')
    sp2.imshow(np.log10(phot_hist).T, extent = (-4,9,13,28), cmap = 'YlGnBu', origin = 'lower')
    sp3.imshow(np.log10(cosmos_hist).T, extent = (-4,9,13,28), cmap = 'YlGnBu', origin = 'lower')

    sp1.text(0.98, 0.02, 'Spec', ha = 'right', va = 'bottom', transform = sp1.transAxes, fontsize = 28)
    sp2.text(0.98, 0.02, 'Phot', ha = 'right', va = 'bottom', transform = sp2.transAxes, fontsize = 28)
    sp3.text(0.98, 0.02, 'COSMOS2015', ha = 'right', va = 'bottom', transform = sp3.transAxes, fontsize = 28)

    # print(np.std((data_spec.g_cmodel_mag - data_spec.z_cmodel_mag)[(data_spec.g_cmodel_mag-data_spec.z_cmodel_mag > -4) & (data_spec.g_cmodel_mag-data_spec.z_cmodel_mag < 9)]))
    # print(np.std((data_phot.g_cmodel_mag - data_phot.z_cmodel_mag)[(data_phot.g_cmodel_mag-data_phot.z_cmodel_mag > -4) & (data_phot.g_cmodel_mag-data_phot.z_cmodel_mag < 9)]))
    # print(np.std((g-z)[(g-z > -4) & (g-z < 9)]))

    # print(np.percentile(data_spec.g_cmodel_mag - data_spec.z_cmodel_mag, [16, 84]))
    # print(np.percentile(data_phot.g_cmodel_mag - data_phot.z_cmodel_mag, [16, 84]))
    # print(np.percentile(g-z, [16, 84]))

    sp2.set_yticklabels([])
    sp3.set_yticklabels([])

    plt.subplots_adjust(wspace = 0.)

    sp1.text(0.02, 0.98, '$N = %i$' % 307193, fontsize = 24, ha = 'left', va = 'top', transform = sp1.transAxes)
    sp2.text(0.02, 0.98, '$N = %i$' % len(data_phot[f'{color[0]}_cmodel_mag']), fontsize = 24, ha = 'left', va = 'top', transform = sp2.transAxes)
    sp3.text(0.02, 0.98, '$N = %i$' % len(cosmos_magband), fontsize = 24, ha = 'left', va = 'top', transform = sp3.transAxes)

    sp2.set_xlabel(f'({color[0]}-{color[1]})')
    sp1.set_ylabel(mag)


def plot_colormag_spec_cosmos(spec_file = './HSC/HSC_wide_clean_pdr2.fits', cosmos_match_file = './tpzruns/MatchCOSMOS2015/tpzrun.tpz_app'):

    data_spec = fits.open(spec_file)[1].data
    g, i, z = read_csv(cosmos_match_file, delim_whitespace = True, usecols = [1,3,4]).values.T

    fig = plt.figure(figsize = (15,9))
    sp1 = fig.add_subplot(121)
    sp2 = fig.add_subplot(122)

    spec_hist = np.histogram2d(data_spec.g_cmodel_mag - data_spec.z_cmodel_mag,data_spec.i_cmodel_mag, range = ((-4,9),(13,28)), bins = 100)[0]
    cosmos_hist = np.histogram2d(g - z, i, range = ((-4,9),(13,28)), bins = 100)[0]

    sp1.imshow(np.log10(spec_hist).T, extent = (-4,9,13,28), cmap = 'YlGnBu', origin = 'lower')
    sp2.imshow(np.log10(cosmos_hist).T, extent = (-4,9,13,28), cmap = 'YlGnBu', origin = 'lower')

    sp1.text(0.98, 0.02, 'Spec', ha = 'right', va = 'bottom', transform = sp1.transAxes, fontsize = 28)
    sp1.text(0.02, 0.98, '$N = %i$' % len(data_spec.g_cmodel_mag), ha = 'left', va = 'top', transform = sp1.transAxes, fontsize = 24)
    sp2.text(0.98, 0.02, 'COSMOS2015', ha = 'right', va = 'bottom', transform = sp2.transAxes, fontsize = 28)
    sp2.text(0.02, 0.98, '$N = %i$' % len(g), ha = 'left', va = 'top', transform = sp2.transAxes, fontsize = 24)

    print(np.std((g-z)[(g-z > -4) & (g-z < 9)]))
    print(np.percentile(g-z, [16, 84]))
    
    sp2.set_yticklabels([])

    plt.subplots_adjust(wspace = 0.)

    fig.text(0.5, 0, '(g-z)', fontsize = 24)
    sp1.set_ylabel('i', fontsize = 24)




def plot_colormag_constructed_training(phot_file = './HSC/HSC_wide_clean_pdr2_phot_depth_1M.fits', spec_file = './HSC/HSC_wide_clean_pdr2.fits', run_file = './tpzruns/closest_match/match.tpz_app'):

    g, i, z = np.loadtxt(run_file, usecols = [1,3,4], unpack = True)
    data_phot = fits.open(phot_file)[1].data

    fig = plt.figure(figsize = (16,10))
    sp1 = fig.add_subplot(121)
    sp2 = fig.add_subplot(122)

    phot_hist = np.histogram2d(data_phot.g_cmodel_mag - data_phot.z_cmodel_mag,data_phot.i_cmodel_mag, range = ((-4,9),(13,28)), bins = 100)[0]
    draw_hist = np.histogram2d(g-z, i, range = ((-4,9),(13,28)), bins = 100)[0]

    sp1.imshow(np.log10(phot_hist).T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')
    sp2.imshow(np.log10(draw_hist).T, extent = (-4,9,13,28), cmap = 'inferno_r', origin = 'lower')

    sp1.text(0.98, 0.02, 'Phot', ha = 'right', va = 'bottom', transform = sp1.transAxes, fontsize = 28)
    sp2.text(0.98, 0.02, 'Constructed', ha = 'right', va = 'bottom', transform = sp2.transAxes, fontsize = 28)


    fig.text(0.5, 0, '(g-z)', fontsize = 26)
    sp1.set_ylabel('i', fontsize = 26)

    plt.subplots_adjust(wspace = 0.)
    sp2.set_yticklabels([])





def plot_color_dist(spec_file = './HSC/HSC_wide_clean_pdr2.fits', phot_file = './HSC/HSC_wide_clean_pdr2_phot_depth_Unltd_COSMOS.fits'):

    spec = fits.open(spec_file)[1].data
    phot = fits.open(phot_file)[1].data

    gz_spec = spec['g_cmodel_mag'] - spec['z_cmodel_mag']
    gz_phot = phot['g_cmodel_mag'] - phot['z_cmodel_mag']

    gz_spec = gz_spec[np.isfinite(gz_spec)]
    gz_phot = gz_phot[np.isfinite(gz_phot)]

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    n_spec, bins_spec, patches_spec = sp.hist(gz_spec, histtype = 'step', range = (-2, 5), bins = 100, density = True, label = 'Spec')
    n_phot, bins_phot, patches_phot = sp.hist(gz_phot, histtype = 'step', range = (-2, 5), bins = 100, density = True, label = 'Phot')

    mode_spec = np.sum(bins_spec[np.argmax(n_spec) + np.array([0,1])])/2.
    mode_phot = np.sum(bins_phot[np.argmax(n_phot) + np.array([0,1])])/2.

    # sp.set_xlim(-2,5)

    sp.text(0.02, 0.98, r'$\sigma = ' + '%.2f$' % np.std(gz_spec) + '\n$Med=%.2f$' % np.median(gz_spec) + '\n$Avg=%.2f$' % np.mean(gz_spec[np.isfinite(gz_spec)]) + '\n$Mode=%.2f$'%mode_spec, fontsize = 18, color = 'C0', ha = 'left', va = 'top', transform = sp.transAxes)
    sp.text(0.98, 0.98, r'$\sigma = ' + '%.2f$' % np.std(gz_phot) + '\n$Med=%.2f$' % np.median(gz_phot) + '\n$Avg=%.2f$' % np.mean(gz_phot[np.isfinite(gz_phot)]) + '\n$Mode=%.2f$'%mode_phot, fontsize = 18, color = 'C1', ha = 'right', va = 'top', transform = sp.transAxes)

    sp.set_xlabel('g-z')
    sp.set_ylabel('Norm Freq')



def plot_mag_dist(spec_file = './HSC/HSC_wide_clean_pdr2.fits', phot_file = './HSC/HSC_wide_clean_pdr2_phot_depth_1M.fits'):

    spec = fits.open(spec_file)[1].data
    phot = fits.open(phot_file)[1].data

    i_spec = spec['i_cmodel_mag']
    i_phot = phot['i_cmodel_mag']

    i_spec = i_spec[np.isfinite(i_spec)]
    i_phot = i_phot[np.isfinite(i_phot)]

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    n_spec, bins_spec, patches_spec = sp.hist(i_spec, histtype = 'step', range = (20, 27), bins = 100, density = True, label = 'Spec')
    n_phot, bins_phot, patches_phot = sp.hist(i_phot, histtype = 'step', range = (20, 27), bins = 100, density = True, label = 'Phot')

    mode_spec = np.sum(bins_spec[np.argmax(n_spec) + np.array([0,1])])/2.
    mode_phot = np.sum(bins_phot[np.argmax(n_phot) + np.array([0,1])])/2.

    sp.text(0.02, 0.98, r'$\sigma = ' + '%.2f$' % np.std(i_spec) + '\n$Med=%.2f$' % np.median(i_spec) + '\n$Avg=%.2f$' % np.mean(i_spec[np.isfinite(i_spec)]) + '\n$Mode=%.2f$'%mode_spec, fontsize = 18, color = 'C0', ha = 'left', va = 'top', transform = sp.transAxes)
    sp.text(0.98, 0.98, r'$\sigma = ' + '%.2f$' % np.std(i_phot) + '\n$Med=%.2f$' % np.median(i_phot) + '\n$Avg=%.2f$' % np.mean(i_phot[np.isfinite(i_phot)]) + '\n$Mode=%.2f$'%mode_phot, fontsize = 18, color = 'C1', ha = 'right', va = 'top', transform = sp.transAxes)

    sp.set_xlabel('i')
    sp.set_ylabel('Norm Freq')




def diagnostic_mag_color_comparison(xval = 'i-z', yval = 'g'):

    gal_only = fits.open('./HSC/use_to_replace_later/HSC_wide_clean_pdr2_phot_depth_1M.fits')[1].data
    gal_plus = fits.open('./HSC/archive/HSC_wide_clean_pdr2_phot_depth_1M.fits')[1].data

    x_bands = xval.split('-')
    y_bands = yval.split('-')

    xdata_gal = np.array([gal_only['%s_cmodel_mag' % thisband] for thisband in x_bands])
    ydata_gal = np.array([gal_only['%s_cmodel_mag' % thisband] for thisband in y_bands])

    xdata_galplus = np.array([gal_plus['%s_cmodel_mag' % thisband] for thisband in x_bands])
    ydata_galplus = np.array([gal_plus['%s_cmodel_mag' % thisband] for thisband in y_bands])


    fig = plt.figure(figsize = (18, 6))
    sp1 = fig.add_subplot(131)
    sp2 = fig.add_subplot(132)
    sp3 = fig.add_subplot(133)

    if len(x_bands) == 2:
        xdata_gal = np.diff(xdata_gal, axis = 0).squeeze()
        xdata_galplus = np.diff(xdata_galplus, axis = 0).squeeze()
        x_range = (-2,2)
    else:
        xdata_gal = xdata_gal.squeeze()
        xdata_galplus = xdata_galplus.squeeze()
        x_range = (18,28)

    if len(y_bands) == 2:
        ydata_gal = np.diff(ydata_gal, axis = 0).squeeze()
        ydata_galplus = np.diff(ydata_galplus, axis = 0).squeeze()
        y_range = (-2,2)
    else:
        ydata_gal = ydata_gal.squeeze()
        ydata_galplus = ydata_galplus.squeeze()
        y_range = (18,28)

    hist_gal = np.histogram2d(xdata_gal, ydata_gal, bins = 100, range = (x_range, y_range))[0]
    hist_galplus = np.histogram2d(xdata_galplus, ydata_galplus, bins = 100, range = (x_range, y_range))[0]

    im2 = sp2.imshow(np.log10(hist_galplus).T, origin = 'lower', extent = x_range+y_range, cmap = 'inferno_r', aspect = 'auto')
    im1 = sp1.imshow(np.log10(hist_gal).T, origin = 'lower', extent = x_range+y_range, cmap = 'inferno_r', aspect = 'auto')
    im3 = sp3.imshow(np.log10(np.abs(hist_galplus - hist_gal)).T, origin = 'lower', extent = x_range+y_range, cmap = 'inferno_r', aspect = 'auto')
    im1.set_clim(im2.get_clim())
    im3.set_clim(im2.get_clim())

    sp2.set_yticklabels([])
    sp3.set_yticklabels([])

    sp2.set_xlabel(xval)
    sp1.set_ylabel(yval)

    sp1.text(0.02, 0.98, 'Gal+', fontsize = 30, ha = 'left', va = 'top', transform = sp1.transAxes)
    sp2.text(0.02, 0.98, 'Gal', fontsize = 30, ha = 'left', va = 'top', transform = sp2.transAxes)
    sp3.text(0.02, 0.98, 'log(abs(Diff))', fontsize = 30, ha = 'left', va = 'top', transform = sp3.transAxes)

    plt.subplots_adjust(wspace = 0)








def example_spec(redshift = [0.3, 0.8]):

    import photosim

    colors = ['#ae0a9d', '#002A78', '#00cc66', '#f1c40f', '#F78102', '#941919']

    lsst_filters = filtersim.filtersim()
    phot = photosim.photosim()

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    filterclone = sp.twinx()

    filter_waves = [lsst_filters.wavelength[thiskey] for thiskey in lsst_filters.keys]
    filter_resp = [lsst_filters.response[thiskey] for thiskey in lsst_filters.keys]
    filter_names = ['u', 'g', 'r', 'i', 'z', 'y']

    for x, (thisname, thiswave, thisresp) in enumerate(zip(filter_names, filter_waves, filter_resp)):

        filterclone.plot(thiswave/10, thisresp, color = 'k', linewidth = 3.5)
        filterclone.plot(thiswave/10, thisresp, color = colors[x], label = thisname)
        txt = filterclone.text(np.trapz(thiswave * thisresp, x = thiswave)/np.trapz(thisresp, x = thiswave)/10, max(thisresp) + .1, thisname, color = colors[x], fontsize = 16, ha = 'center')
        txt.set_path_effects([pe.withStroke(linewidth=1, foreground='k')])

    # Spectrum 1

    wave, spec = phot.find_spectrum(tage = 2., sfh_params = {'tau': -.4}, sfh_type = 6, peraa = True, Av = 1.0)
    wave = wave*(1+redshift[0])
    sp.plot(wave/10, np.log10(spec), color = 'steelblue', zorder = 1, alpha = 1.0)

    photo_x, photo_y, _ = lsst_filters.get_photometry(wave, spec, input_flam = True)
    photo_y = photo_y * 3e10/(photo_x**2.)
    sp.scatter(photo_x/10, np.log10(photo_y), s = 75, edgecolor = 'k', linewidth = 0.5, facecolor = colors, zorder = 2)

    sp.text(0.02, 0.98, '$Distance = %i Mly$' % cosmo.comoving_distance(redshift[0]).to('Mlyr').value, ha = 'left', va = 'top', fontsize = 24, transform = sp.transAxes, color = 'steelblue')

    # Spectrum 2

    wave, spec = phot.find_spectrum(tage = 2., sfh_params = {'tau': -.4}, sfh_type = 6, peraa = True, Av = 1.0)
    wave = wave*(1+redshift[1])
    spec = spec/2.
    sp.plot(wave/10, np.log10(spec), color = 'crimson', zorder = 1, alpha = 1.0)

    photo_x, photo_y, _ = lsst_filters.get_photometry(wave, spec, input_flam = True)
    photo_y = photo_y * 3e10/(photo_x**2.)
    sp.scatter(photo_x/10, np.log10(photo_y), s = 75, edgecolor = 'k', linewidth = 0.5, facecolor = colors, zorder = 2)

    sp.text(0.98, 0.98, '$Distance = %i Mly$' % cosmo.comoving_distance(redshift[1]).to('Mlyr').value, ha = 'right', va = 'top', fontsize = 24, transform = sp.transAxes, color = 'crimson')

    sp.set_xlabel(r'Wavelength (nm)')
    filterclone.set_ylabel('Filter Sensitivity')
    filterclone.set_xscale('log')
    filterclone.set_xlim(10**2.35, 10**3.35)
    filterclone.set_ylim(0, 3)
    sp.set_ylabel('log(Flux)')
    avgy = np.average(np.log10(photo_y))
    sp.set_ylim(avgy-3, avgy+3)




def cosmos2015_diagnostic(sigmacut = 0.1):

    cosmos = fits.open('./COSMOS2015/COSMOS2015_Laigle+_v1.1.fits')[1].data
    spec = fits.open('./HSC/HSC_wide_clean_pdr2.fits')[1].data
    cosmos = cosmos[(cosmos.TYPE == 0) & (cosmos.ZP_2 < 0)]

    cosmos_cat = SkyCoord(ra = cosmos.ALPHA_J2000*u.degree, dec = cosmos.DELTA_J2000*u.degree)
    spec_cat = SkyCoord(ra = spec.ra*u.degree, dec = spec.dec * u.degree)

    idx, d2d, _ = cosmos_cat.match_to_catalog_sky(spec_cat) # Produces a list of length cosmos_cat of indices matching to spec_cat objects

    # Find the spec_cat objects that are closest to the cosmos_cat objects and only keep those

    unique_idx = []

    for this_match_idx in np.unique(idx):

        matches = np.where(idx == this_match_idx)[0] # Get the list of matching objects
        unique_idx.append(matches[np.argmin(d2d[matches])]) # Pick the one with the smallest distance

    unique_idx = np.array(unique_idx) # unique_idx contains the indices of COSMOS objects that have "correct" matches (closest to the spec catalog object)
    spec_idx = idx[unique_idx]

    specz = spec.specz_redshift[spec_idx]
    photz = cosmos.ZPDF[unique_idx] # ZPDF for median of the PDF, ZMINCHI2 for minimum chi2 (-99 for <3 bands)
    photz_sigma = (cosmos.ZPDF_H68-cosmos.ZPDF_L68)[unique_idx] # ZPDF_H68-ZPDF_L68 for ZPDF; CHI2BEST for ZMINCHI2

    good_inds = (photz >= 0) & (photz_sigma < sigmacut) & (d2d.to('arcsecond').value[unique_idx] < 1) & (cosmos.ip_MAG_APER2[unique_idx] < 25.) & (cosmos.ZP_2[unique_idx] < 0) & (cosmos.NBFILT[unique_idx] >= 25) & (photz <= 1.5) & (specz <= 1.5)

    print('Post-selection: %i' % sum(good_inds))

    specz = specz[good_inds]
    photz = photz[good_inds]

    photz_err = (photz-specz)/(1+specz)

    hist = np.histogram2d(specz, photz, range = ((0,1.5),(0,1.5)), bins = 100)[0]

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    sp.imshow(np.log10(hist).T, origin = 'lower', extent = (0,1.5,0,1.5), cmap = 'inferno_r')

    sp.text(0.98, 0.02,'$f_{out}=%.3f$\n' % (np.sum(np.abs(photz_err)>0.15)/float(len(photz_err))) +
                            '$NMAD=%.3f$\n' % (1.4826*np.median(np.abs(photz_err))) +
                            r'$\sigma=' + '%.3f$' % np.std(photz_err), 
                            fontsize = 25, ha = 'right', va = 'bottom', transform = sp.transAxes)
    sp.text(0.02, 0.98, '$N = %i$'%len(specz), fontsize = 25, ha = 'left', va = 'top', transform = sp.transAxes)

    sp.set_xlabel('$z_{spec}$')
    sp.set_ylabel('$z_{COSMOS2015}$')



def plot_diagnostic_colormag_blob(run_dir = './tpzruns/closest_match/', nn_identifier = 'nnc_', zconf_boundary = 0.5):
    
    if run_dir[-1] != '/':
        run_dir += '/'

    nnc_folders = glob(run_dir + '*' + nn_identifier + '*0/')

    if len(nnc_folders) == 1:
        nnc_folder = nnc_folders[0]
    else:
        print('There are multiple folders matching the nn_identifier.  Please pick one and try again:')

        for thisfolder in nnc_folders:
            print(thisfolder)
        # [print(thisfolder) for thisfolder in nnc_folders]
        return None

    
    features_file = read_csv(glob(run_dir + '*.nn_app')[0], delimiter = '\s+', comment = '#') 
    nnc_results = np.loadtxt(nnc_folder + 'results_application.dat')

    zspec = features_file['specz'].to_numpy()
    zphot = features_file['zphot'].to_numpy()
    zconf = features_file['zconf'].to_numpy()
    zphot_err = features_file['zerr'].to_numpy()

    zconf_lim_inds = (zconf > zconf_boundary)
    # Find the nn_boundary that gives the same number of objects
    nn_boundary = np.sort(nnc_results)[::-1][sum(zconf_lim_inds)]
    nnc_lim_inds = (nnc_results > nn_boundary)
    # Find the zphot_err that gives the same number of objects
    fake_zphot_err = zphot_err + np.random.random(len(zphot_err))*10**-4  # Add a small random number to zphot_err because values are not unique
    zphot_err_boundary = np.sort(fake_zphot_err)[sum(zconf_lim_inds)]
    zphot_err_lim_inds = (fake_zphot_err < zphot_err_boundary)

    blob_inds = nnc_lim_inds & (zphot > 0.4) & (zphot < 0.8) & (zspec < 0.35) & (zphot > 0.)

    g = features_file['g'].to_numpy()
    z = features_file['z'].to_numpy()
    i = features_file['i'].to_numpy()

    gz = g-z

    fig = plt.figure(figsize = (18,6))
    sp1 = fig.add_subplot(131)
    sp2 = fig.add_subplot(132)
    sp3 = fig.add_subplot(133)

    h1 = np.histogram2d(gz[blob_inds], i[blob_inds], range = ((-5,10),(14,28)), bins = 50)[0]
    sp1.imshow(np.log10(h1).T, origin = 'lower', extent = (-5,10,14,28), cmap = 'plasma_r', aspect = 'auto')
    sp1.text(0.02, 0.98, '$N = %i$' % sum(blob_inds), ha = 'left', va = 'top', transform = sp1.transAxes, fontsize = 20)

    rand_inds = np.random.choice(np.arange(len(blob_inds)), size = sum(blob_inds), replace = False)
    h2 = np.histogram2d(gz[rand_inds], i[rand_inds], range = ((-5,10),(14,28)), bins = 50)[0]
    sp2.imshow(np.log10(h2).T, origin = 'lower', extent = (-5,10,14,28), cmap = 'plasma_r', aspect = 'auto')
    sp2.text(0.02, 0.98, '$N = %i$' % len(rand_inds), ha = 'left', va = 'top', transform = sp2.transAxes, fontsize = 20)

    h3 = np.histogram2d(gz[~blob_inds], i[~blob_inds], range = ((-5,10),(14,28)), bins = 50)[0]
    sp3.imshow(np.log10(h3).T, origin = 'lower', extent = (-5,10,14,28), cmap = 'plasma_r', aspect = 'auto')
    sp3.text(0.02, 0.98, '$N = %i$' % sum(~blob_inds), ha = 'left', va = 'top', transform = sp3.transAxes, fontsize = 20)

    # sp.scatter(gz, i, s = 3, color = 'k')

    fig.text(0.5, 0., '(g-z)', fontsize = 26)
    sp1.set_ylabel('i', fontsize = 26)




# def colormag_triple():



def paper_plots():

    try:
        plot_all_results_nonnr()
        plt.savefig('./Figures/paperfigs/all_results_nonnr.pdf', bbox_inches = 'tight')
        plt.close()
    except:
        print('Could not generate figure "all_results_nonnr".')

    try:
        plot_all_train_test_nonnr()
        plt.savefig('./Figures/paperfigs/all_train_test_nonnr.pdf', bbox_inches = 'tight')
        plt.close()
    except:
        print('Could not generate figure "all_train_test_nonnr".')

    try:
        plot_colormag_phot_spec_cosmos()
        plt.savefig('./Figures/paperfigs/colormag_phot_spec_cosmos.pdf', bbox_inches = 'tight')
        plt.close()
    except:
        print('Could not generate figure "colormag_phot_spec_cosmos".')

    try:
        cosmos2015_diagnostic()
        plt.savefig('./Figures/paperfigs/cosmos2015_diagnostic.pdf', bbox_inches = 'tight')
        plt.close()
    except:
        print('Could not generate figure "cosmos2015_diagnostic".')

    try:
        plot_sigma_nfrac_comparison_threepanel()
        plt.savefig('./Figures/paperfigs/sigma_nfrac_comparison_threepanel.pdf', bbox_inches = 'tight')
        plt.close()
    except:
        print('Could not generate figure "sigma_nfrac_comparison_threepanel".')

    try:
        plot_sigma_nfrac_comparison_threepanel(bpz_appdir = '/home/adam/Research/pz_pdf/pz/BPZ/HSC_app_cosmos/')
        plt.savefig('./Figures/paperfigs/sigma_nfrac_comparison_threepanel.pdf', bbox_inches = 'tight')
        plt.close()
    except:
        print('Could not generate figure "sigma_nfrac_comparison_threepanel".')

    try:
        plot_z_dist()
        plt.savefig('./Figures/paperfigs/z_dist.pdf', bbox_inches = 'tight')
        plt.close()
    except:
        print('Could not generate figure "z_dist".')

    try:
        plot_zfrac()
        plt.savefig('./Figures/paperfigs/z_frac.pdf', bbox_inches = 'tight')
        plt.close()
    except:
        print('Could not generate figure "z_frac".')