import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from pandas import read_csv
from tqdm import tqdm
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from mlz.ml_codes import *
from glob import glob
import filtersim
from sklearn.neighbors import KernelDensity
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize_scalar
import matplotlib.patheffects as PathEffects





def hsc_select(inputfp = 'HSC/HSC_wide_clean_pdr2.fits', size = None, unique = True, maglim = None, sn_lim = 10., zlim = 1.5, notzero = True, output_mags = True):

    np.random.seed(0)

    data = fits.open(inputfp)[1].data
    ID = data['object_id']
    pz = data['pz_best_miz']
    sz = data['specz_redshift']
    indices = np.arange(len(ID))

    keys_orig = ['g_cmodel_mag', 'g_cmodel_magsigma', 'r_cmodel_mag', 'r_cmodel_magsigma', 'i_cmodel_mag', 'i_cmodel_magsigma', 'z_cmodel_mag', 
    'z_cmodel_magsigma', 'y_cmodel_mag', 'y_cmodel_magsigma', 'g_cmodel_flux', 'g_cmodel_fluxsigma', 'r_cmodel_flux', 'r_cmodel_fluxsigma', 'i_cmodel_flux', 'i_cmodel_fluxsigma', 
    'z_cmodel_flux', 'z_cmodel_fluxsigma', 'y_cmodel_flux', 'y_cmodel_fluxsigma']
    keys = [''.join(thiskey.split('_cmodel')) for thiskey in keys_orig]

    data_table = {}
    for oldkey, newkey in zip(keys_orig,keys):
        data_table[newkey] = data[oldkey]


    specz_lim = np.where((sz >0.) & (sz < 9) & data['specz_flag_homogeneous'])[0]
        
    for thiskey in keys:
        data_table[thiskey] = data_table[thiskey][specz_lim]

    indices = indices[specz_lim]
    ID = ID[specz_lim]
    pz = pz[specz_lim]
    sz = sz[specz_lim]

    if unique:

        lim_indices = np.where(np.unique(ID, return_counts = True)[1] == 1)[0]
        for thiskey in keys:
            data_table[thiskey] = data_table[thiskey][lim_indices]

        indices = indices[lim_indices]
        ID = ID[lim_indices]
        pz = pz[lim_indices]
        sz = sz[lim_indices]



    if maglim != None:
        if not hasattr(maglim, '__iter__'):
            maglim = [maglim,]*5
        lim_indices = np.where((data_table['g_mag'] < maglim[0]) & (data_table['r_mag'] < maglim[1]) & (data_table['i_mag'] < maglim[2]) & (data_table['z_mag'] < maglim[3]) & (data_table['y_mag'] < maglim[4]))[0]

        for thiskey in keys:
            data_table[thiskey] = data_table[thiskey][lim_indices]

        indices = indices[lim_indices]
        ID = ID[lim_indices]
        pz = pz[lim_indices]
        sz = sz[lim_indices]


    if sn_lim != None:
        total_sn = np.sqrt((data_table['g_flux']/data_table['g_fluxsigma'])**2 + (data_table['r_flux']/data_table['r_fluxsigma'])**2 + 
            (data_table['i_flux']/data_table['i_fluxsigma'])**2 + (data_table['z_flux']/data_table['z_fluxsigma'])**2 + (data_table['y_flux']/data_table['y_fluxsigma'])**2)
        lim_indices = np.where(total_sn > sn_lim)

        for thiskey in keys:
            data_table[thiskey] = data_table[thiskey][lim_indices]

        indices = indices[lim_indices]
        ID = ID[lim_indices]
        pz = pz[lim_indices]
        sz = sz[lim_indices]


    if zlim != None:
        lim_indices = np.where(sz < zlim)[0]

        for thiskey in keys:
            data_table[thiskey] = data_table[thiskey][lim_indices]

        indices = indices[lim_indices]
        ID = ID[lim_indices]
        pz = pz[lim_indices]
        sz = sz[lim_indices]

    if notzero:

        lim_indices = np.where(sz > 0.01)[0]
        
        for thiskey in keys:
            data_table[thiskey] = data_table[thiskey][lim_indices]

        indices = indices[lim_indices]
        ID = ID[lim_indices]
        pz = pz[lim_indices]
        sz = sz[lim_indices]

    if output_mags:

        data = np.vstack((data_table['g_mag'], data_table['r_mag'], data_table['i_mag'], data_table['z_mag'], data_table['y_mag'])).T
        errs = np.vstack((data_table['g_magsigma'], data_table['r_magsigma'], data_table['i_magsigma'], data_table['z_magsigma'], data_table['y_magsigma'])).T
        goodinds = ~np.isnan(data).any(axis=1)


    else:

        data = np.vstack((data_table['g_flux'], data_table['r_flux'], data_table['i_flux'], data_table['z_flux'], data_table['y_flux'])).T
        errs = np.vstack((data_table['g_fluxsigma'], data_table['r_fluxsigma'], data_table['i_fluxsigma'], data_table['z_fluxsigma'], data_table['y_fluxsigma']))        

    indices = indices[goodinds]
    ID = ID[goodinds]
    pz = pz[goodinds]
    sz = sz[goodinds]
    data = data[goodinds]
    errs = errs[goodinds]

    if size != None:

        selections = np.random.choice(np.arange(len(ID)), size = size, replace = False)
    
        indices = indices[selections]
        ID = ID[selections]
        pz = pz[selections]
        sz = sz[selections]
        data = data[selections]
        errs = errs[selections]


    return indices, ID, pz, sz, data, errs




# def write_feature_file(savedir = './tpzruns/ORIGINAL/', filename = 'default', trainsplit = 0.8, i_maglim = None, gz_colorlim = None, zlim = 1.5, sn_lim = 10., notzero = True, nn_train = False, nn_fit = False, size = None):

#     np.random.seed(1456)

#     if not os.path.isdir(savedir):

#         os.makedirs(savedir)

#     indices, ID, _, sz, mags, errs = hsc_select(size = size, zlim = zlim, sn_lim = sn_lim, notzero = notzero)

#     g, r, i, z, y = mags.T
#     g_err, r_err, i_err, z_err, y_err = errs.T

#     features = np.copy(mags)
#     feature_errs = np.copy(errs)

#     colors = np.array([g-r, r-i, i-z, z-y]).T
#     colors_errs = np.array([g_err+r_err, r_err+i_err, i_err+z_err, z_err+y_err]).T

#     features = np.hstack((features, colors))
#     feature_errs = np.hstack((feature_errs, colors_errs))

#     triplets = np.array([(g-r) - (r-i), (r-i) - (i-z), (i-z) - (z-y)]).T
#     triplets_errs = np.array([(g_err+r_err) + (r_err+i_err), (r_err+i_err) + (i_err+z_err), (i_err+z_err) + (z_err+y_err)]).T

#     features = np.hstack((features, triplets))
#     feature_errs = np.hstack((feature_errs, triplets_errs))

#     all_inds = np.arange(features.shape[0])

#     if i_maglim == None and gz_colorlim == None:

#         shuffle_inds = np.arange(features.shape[0])
#         np.random.shuffle(shuffle_inds)
#         train_inds = sorted(shuffle_inds[:int(trainsplit*len(shuffle_inds))])
#         test_inds = sorted(shuffle_inds[int(trainsplit*len(shuffle_inds)):])

#     elif gz_colorlim != None and i_maglim != None:
#         train_inds = np.where((g-z > gz_colorlim) & (i < i_maglim))[0]
#         test_inds = np.where((g-z <= gz_colorlim) & (i >= i_maglim))[0]

#     elif gz_colorlim != None:
#         train_inds = np.where((g-z > gz_colorlim))[0]
#         test_inds = np.where((g-z <= gz_colorlim))[0]

#     elif i_maglim != None:
#         train_inds = np.where((i < i_maglim))[0]
#         test_inds = np.where((i >= i_maglim))[0]

#     if nn_train:


#         if os.path.isfile(savedir + filename + '.nnvalidate') or os.path.isfile(savedir + filename + '.nntrain'):
#             resp = input('Are you sure you want to overwrite files in ' + savedir + '?')
#             if resp != 'y':
#                 return None

#         tpz_features = np.loadtxt(savedir + 'output/results/' + filename + '.0.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr
#         features = np.hstack((features, tpz_features))

#         header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy'
#         fmt_str = ' %.5f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3

#         np.savetxt(savedir + filename + '.nntrain', np.hstack((sz[train_inds].reshape(-1,1), features[train_inds], feature_errs[train_inds])), header = header, fmt = fmt_str)
#         np.savetxt(savedir + filename + '.nnvalidate', np.hstack((sz[test_inds].reshape(-1,1), features[test_inds], feature_errs[test_inds])), header = header, fmt = fmt_str)


#     elif nn_fit:

#         if os.path.isfile(savedir + filename + '.nnfit'):
#             resp = input('Are you sure you want to overwrite files in ' + savedir + '?')
#             if resp != 'y':
#                 return None

#         tpz_features = np.loadtxt(savedir + 'output/results/' + filename + '.0.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr
#         features = np.hstack((features, tpz_features))

#         header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy'
#         fmt_str = ' %.5f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3

#         np.savetxt(savedir + filename + '.nnfit', np.hstack((sz[all_inds].reshape(-1,1), features[all_inds], feature_errs[all_inds])), header = header, fmt = fmt_str)            

#     else:


#         if os.path.isfile(savedir + filename + '.test') or os.path.isfile(savedir + filename + '.train') or os.path.isfile(savedir + filename + '.train_inds') or os.path.isfile(savedir + filename + '.train_inds'):
#             resp = input('Are you sure you want to overwrite files in ' + savedir + '? ')
#             if resp != 'y':
#                 return None

#         header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
#         fmt_str = ' %.5f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

#         np.savetxt(savedir + filename + '.train', np.hstack((sz[train_inds].reshape(-1,1), features[train_inds], feature_errs[train_inds])), header = header, fmt = fmt_str)
#         np.savetxt(savedir + filename + '.test', np.hstack((sz[all_inds].reshape(-1,1), features[all_inds], feature_errs[all_inds])), header = header, fmt = fmt_str)
#         np.savetxt(savedir + filename + '.test_inds', test_inds, header = 'Test indices for galaxies in the fit catalog', fmt = '%i')
#         np.savetxt(savedir + filename + '.train_inds', train_inds, header = 'Train indices for galaxies in the fit catalog', fmt = '%i')





def plot_colormag(inputfp = 'HSC/HSC_wide_clean_pdr2.fits', sn_lim = 10.):

    if 'phot_depth' not in inputfp:
        indices, ID, _, sz, mags, errs = hsc_select(inputfp = inputfp, sn_lim = sn_lim)

        i = mags.T[2]
        g = mags.T[0]
        z = mags.T[3]

        gz = g-z

    else:
        photfits = fits.open(inputfp)[1].data
        i = photfits['i_cmodel_mag']
        gz = photfits['g_cmodel_mag'] - photfits['z_cmodel_mag']
        total_sn = np.sqrt((photfits['g_cmodel_flux']/photfits['g_cmodel_fluxsigma'])**2 + (photfits['r_cmodel_flux']/photfits['r_cmodel_fluxsigma'])**2 + 
            (photfits['i_cmodel_flux']/photfits['i_cmodel_fluxsigma'])**2 + (photfits['z_cmodel_flux']/photfits['z_cmodel_fluxsigma'])**2 + (photfits['y_cmodel_flux']/photfits['y_cmodel_fluxsigma'])**2)
        i = i[total_sn > sn_lim]
        gz = gz[total_sn > sn_lim]

    h = np.histogram2d(gz, i, range = ((-3,8),(14,28)), bins = 100)[0]

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    sp.imshow(np.log10(h.T), extent = (-3,8,14,28), origin = 'lower', interpolation = 'nearest', cmap = 'inferno_r')

    sp.set_xlabel('g-z')
    sp.set_ylabel('i')



def plot_colormap_ratio(ishift = -2, gzshift = 0, specfile = 'HSC/HSC_wide_clean_pdr2.fits', photfile = 'HSC/HSC_wide_clean_pdr2_phot_depth_1M.fits'):

    specfits = fits.open(specfile)[1].data
    photfits = fits.open(photfile)[1].data

    i_spec = specfits['i_cmodel_mag']
    gz_spec = specfits['g_cmodel_mag'] - specfits['z_cmodel_mag']
    i_phot = photfits['i_cmodel_mag']
    gz_phot = photfits['g_cmodel_mag'] - photfits['z_cmodel_mag']

    spechist = np.histogram2d(gz_spec, i_spec, range = ((-4,9),(13,28)), bins = 100)[0]
    phothist = np.histogram2d(gz_phot, i_phot, range = ((-4,9),(13,28)), bins = 100)[0]

    ratiohist = spechist/(spechist + phothist)

    fig = plt.figure(figsize = (12,8))
    sp = fig.add_subplot(111)

    img = sp.imshow(ratiohist.T, extent = (-4 + gzshift,9 + gzshift,13 + ishift,28 + ishift), origin = 'lower', cmap = 'RdYlBu')
    plt.colorbar(img)

    sp.set_xlabel('(g-z)')
    sp.set_ylabel('i')




def ratio_test_train_split(indices, ishift = -2, gzshift = 0, savedir = './tpzruns/ORIGINAL/', filename = 'default', specfile = 'HSC/HSC_wide_clean_pdr2.fits', photfile = 'HSC/HSC_wide_clean_pdr2_phot_depth_1M.fits'):

    np.random.seed(1456)

    if not os.path.isdir(savedir):

        os.makedirs(savedir)

    specfits = fits.open(specfile)[1].data[indices]
    photfits = fits.open(photfile)[1].data

    i_spec = specfits['i_cmodel_mag']
    gz_spec = specfits['g_cmodel_mag'] - specfits['z_cmodel_mag']
    i_phot = photfits['i_cmodel_mag']
    gz_phot = photfits['g_cmodel_mag'] - photfits['z_cmodel_mag']

    spechist, x_edges, y_edges = np.histogram2d(gz_spec, i_spec, range = ((-4,9),(13,28)), bins = 100)
    phothist= np.histogram2d(gz_phot, i_phot, range = ((-4,9),(13,28)), bins = 100)[0]

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

    # shift x and y

    eval_ratiohist = np.roll(eval_ratiohist, x_shift, axis = 0)
    eval_ratiohist = np.roll(eval_ratiohist, y_shift, axis = 1)

    if x_shift > 0:
        eval_ratiohist[:x_shift] = 0.
    if x_shift < 0:
        eval_ratiohist[x_shift:] = 0.
    if y_shift > 0:
        eval_ratiohist[:,:y_shift] = 0.
    if y_shift < 0:
        eval_ratiohist[:,y_shift:] = 0.

    eval_ratiohist = np.nan_to_num(eval_ratiohist, nan = 0.)

    train_numbers = np.minimum(np.random.poisson(spechist * eval_ratiohist), spechist).astype(int)
    # val_numbers = (spechist - train_numbers).astype(int)

    # fig = plt.figure(figsize = (8,8))
    # plt.imshow(np.log10(train_numbers).T, origin = 'lower', cmap = 'inferno_r', extent = (-4,9,13,28))

    # fig = plt.figure(figsize = (8,8))
    # plt.imshow(np.log10(val_numbers).T, origin = 'lower', cmap = 'inferno_r', extent = (-4,9,13,28))

    train_inds = []
    test_inds = []

    for i, (xmin, xmax) in enumerate(tqdm(zip(x_edges[:-1], x_edges[1:]), total = len(x_edges)-1)):
        for j, (ymin, ymax) in enumerate(zip(y_edges[:-1], y_edges[1:])):

            scrambled = np.copy(indices[(gz_spec > xmin) & (gz_spec < xmax) & (i_spec > ymin) & (i_spec < ymax)])
            np.random.shuffle(scrambled)

            # these_inds = spec_inds[(gz_spec > xmin) & (gz_spec < xmax) & (i_spec > ymin) & (i_spec < ymax)]
            # scrambled = np.random.choice(these_inds, size = len(these_inds), replace = False)

            train_inds = train_inds + list(scrambled[:train_numbers[i][j]])
            test_inds = test_inds + list(scrambled[train_numbers[i][j]:])

    return train_inds, test_inds



def default_test_train_split(indices, i_maglim = None, gz_colorlim = None, splitfrac = 0.5, specfile = './HSC/HSC_wide_clean_pdr2.fits'):

    data = fits.open(specfile)[1].data[indices]
    i = data['i_cmodel_mag']
    gz = data['g_cmodel_mag'] - data['z_cmodel_mag']

    if i_maglim == None and gz_colorlim == None:
    
        print('Splitting test and train data by specified ratio %.2f' % splitfrac)

        new_indices = np.copy(indices)
        np.random.shuffle(new_indices)

        split_index = int(splitfrac*len(new_indices))

        train_indices = new_indices[:split_index]
        test_indices = new_indices[split_index:]

    else:

        print('Splitting train and test data specified by color-magnitude limits.')

        data = fits.open(specfile)[1].data[indices]
        i = data['i_cmodel_mag']
        gz = data['g_cmodel_mag'] - data['z_cmodel_mag']

        if gz_colorlim != None and i_maglim != None:
            train_indices = indices[(gz > gz_colorlim) & (i < i_maglim)]
            test_indices = indices[(gz <= gz_colorlim) & (i >= i_maglim)]

        elif gz_colorlim != None:
            train_indices = indices[(gz > gz_colorlim)]
            test_indices = indices[(gz <= gz_colorlim)]

        elif i_maglim != None:
            train_indices = indices[(i < i_maglim)]
            test_indices = indices[(i >= i_maglim)]

    return train_indices, test_indices



def closest_matches(indices, specfile, photfile = './HSC/HSC_wide_clean_pdr2_phot_depth_Unltd_COSMOS.fits', flux_units = 'linear', sn_lim = 10., train_frac_lim = 1.):
    
    # app_specz, app_features_app_feature_errs = closest_matches(indices, inputfp)

    data_spec = fits.open(specfile)[1].data[indices]
    if photfile == 'MATCHED_COSMOS':
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

        photz = cosmos.ZPDF[unique_idx] # ZPDF for median of the PDF, ZMINCHI2 for minimum chi2 (-99 for <3 bands)
        photz_sigma = (cosmos.ZPDF_H68-cosmos.ZPDF_L68)[unique_idx] # ZPDF_H68-ZPDF_L68 for ZPDF; CHI2BEST for ZMINCHI2

        good_inds = (photz >= 0) & (photz_sigma < sigmacut) & (d2d.to('arcsecond').value[unique_idx] < 1) & (cosmos.ip_MAG_APER2[unique_idx] < 25.) & (cosmos.ZP_2[unique_idx] < 0) & (cosmos.NBFILT[unique_idx] >= 25)

        photz = photz[good_inds]
        phot_idx = phot_idx[good_inds]

        data_phot = phot[phot_idx]

        mags = np.vstack((phot['g_cmodel_mag'], phot['r_cmodel_mag'], phot['i_cmodel_mag'], phot['z_cmodel_mag'], phot['y_cmodel_mag'])).T[phot_idx]
        mag_errs = np.vstack((phot['g_cmodel_magsigma'], phot['r_cmodel_magsigma'], phot['i_cmodel_magsigma'], phot['z_cmodel_magsigma'], phot['y_cmodel_magsigma'])).T[phot_idx]

    else:
        data_phot = fits.open(photfile)[1].data

    # print('Remove this later')

    good_phot_inds = np.where(np.isfinite(data_phot.g_cmodel_flux) & np.isfinite(data_phot.r_cmodel_flux) & np.isfinite(data_phot.i_cmodel_flux) & np.isfinite(data_phot.z_cmodel_flux) & np.isfinite(data_phot.y_cmodel_flux) & 
        np.isfinite(data_phot.g_cmodel_fluxsigma) & np.isfinite(data_phot.r_cmodel_fluxsigma) & np.isfinite(data_phot.i_cmodel_fluxsigma) & np.isfinite(data_phot.z_cmodel_fluxsigma) & np.isfinite(data_phot.y_cmodel_fluxsigma) & 
        np.isfinite(data_phot.g_cmodel_mag) & np.isfinite(data_phot.r_cmodel_mag) & np.isfinite(data_phot.i_cmodel_mag) & np.isfinite(data_phot.z_cmodel_mag) & np.isfinite(data_phot.y_cmodel_mag) & 
        np.isfinite(data_phot.g_cmodel_magsigma) & np.isfinite(data_phot.r_cmodel_magsigma) & np.isfinite(data_phot.i_cmodel_magsigma) & np.isfinite(data_phot.z_cmodel_magsigma) & np.isfinite(data_phot.y_cmodel_magsigma))

    data_phot = data_phot[good_phot_inds]

    if train_frac_lim != 1.:
        indices = np.arange(len(data_phot))
        np.random.shuffle(indices)
        indices = np.sort(indices[:int(len(indices)*train_frac_lim)])
        data_phot = data_phot[indices]

    if flux_units == 'linear':
        modifier = lambda x: x
    elif flux_units == 'log':
        modifier = np.log10

    g_spec = modifier(data_spec['g_cmodel_flux'])
    r_spec = modifier(data_spec['r_cmodel_flux'])
    i_spec = modifier(data_spec['i_cmodel_flux'])
    z_spec = modifier(data_spec['z_cmodel_flux'])
    y_spec = modifier(data_spec['y_cmodel_flux'])

    g_phot = modifier(data_phot['g_cmodel_flux'])
    r_phot = modifier(data_phot['r_cmodel_flux'])
    i_phot = modifier(data_phot['i_cmodel_flux'])
    z_phot = modifier(data_phot['z_cmodel_flux'])
    y_phot = modifier(data_phot['y_cmodel_flux'])

    vectors_spec = np.vstack((g_spec, r_spec, i_spec, z_spec, y_spec)).T
    norm_spec = np.sqrt(np.sum(vectors_spec**2, axis = 1))
    unit_vectors_spec = vectors_spec / norm_spec.reshape(-1,1)

    vectors_phot = np.vstack((g_phot, r_phot, i_phot, z_phot, y_phot)).T
    norm_phot = np.sqrt(np.sum(vectors_phot**2, axis = 1))
    unit_vectors_phot = vectors_phot / norm_phot.reshape(-1,1)

    match_index = []
    match_norm = []

    for this_vec, this_norm in tqdm(zip(unit_vectors_phot, norm_phot), total = len(norm_phot)):

        closest_ind = np.argmax(unit_vectors_spec @ this_vec)
        match_index.append(closest_ind)
        match_norm.append(this_norm/norm_spec[closest_ind])

    match_norm = np.array(match_norm)

    rescaled_matched_spec = vectors_spec[match_index] * match_norm.reshape(-1,1) 

    specz = data_spec['specz_redshift'][match_index]
    
    if flux_units == 'linear':
        rescaled_matched_spec_errs = np.vstack((data_spec.g_cmodel_fluxsigma, data_spec.r_cmodel_fluxsigma, data_spec.i_cmodel_fluxsigma, data_spec.z_cmodel_fluxsigma, data_spec.y_cmodel_fluxsigma)).T[match_index] * match_norm.reshape(-1,1)
    else:
        pass
        # rescaled_matched_spec_errs = np.vstack()

    flux_errs = np.vstack((data_phot.g_cmodel_fluxsigma, data_phot.r_cmodel_fluxsigma, data_phot.i_cmodel_fluxsigma, data_phot.z_cmodel_fluxsigma, data_phot.y_cmodel_fluxsigma)).T
    mag_errs = np.vstack((data_phot.g_cmodel_magsigma, data_phot.r_cmodel_magsigma, data_phot.i_cmodel_magsigma, data_phot.z_cmodel_magsigma, data_phot.y_cmodel_magsigma)).T

    differential_error = np.nan_to_num(np.sqrt(flux_errs**2 - rescaled_matched_spec_errs**2))

    noisy_rescaled_matched_spec = np.random.normal(rescaled_matched_spec, differential_error)
    # print('DO NOT LEAVE THIS; TESTING PURPOSES ONLY')
    # noisy_rescaled_matched_spec = np.random.normal(rescaled_matched_spec, 0)

    total_sn = np.sqrt(np.sum((noisy_rescaled_matched_spec/flux_errs)**2, axis = 1))
    lim_indices = np.where((total_sn > sn_lim) & np.all(noisy_rescaled_matched_spec > 0, axis = 1))

    # Turn fluxes back into magnitudes

    specz = specz[lim_indices]
    g, r, i, z, y = -2.5*np.log10((noisy_rescaled_matched_spec[lim_indices]*10**-9)/3631).T # noisy_rescaled_matched_spec is in nano Janskys
    g_err, r_err, i_err, z_err, y_err = mag_errs[lim_indices].T

    mags = np.vstack((g,r,i,z,y)).T
    mag_errs = np.vstack((g_err,r_err,i_err,z_err,y_err)).T

    colors = np.array([g-r, r-i, i-z, z-y]).T
    color_errs = np.array([g_err+r_err, r_err+i_err, i_err+z_err, z_err+y_err]).T

    triplets = np.array([(g-r) - (r-i), (r-i) - (i-z), (i-z) - (z-y)]).T
    triplet_errs = np.array([(g_err+r_err) + (r_err+i_err), (r_err+i_err) + (i_err+z_err), (i_err+z_err) + (z_err+y_err)]).T

    features = np.hstack((mags, colors, triplets))
    feature_errs = np.hstack((mag_errs, color_errs, triplet_errs))

    return specz, features, feature_errs




def cosmos_closest_matches(indices, specfile, cosmos_file = './HSC/HSC_wide_clean_pdr2_phot_depth_Unltd_COSMOS.fits', flux_units = 'linear', sn_lim = 10., sigmacut = 0.1):
    
    data_spec = fits.open(specfile)[1].data[indices]

    # Get COSMOS2015 matches

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

    photz = cosmos.ZPDF[unique_idx] # ZPDF for median of the PDF, ZMINCHI2 for minimum chi2 (-99 for <3 bands)
    photz_sigma = (cosmos.ZPDF_H68-cosmos.ZPDF_L68)[unique_idx] # ZPDF_H68-ZPDF_L68 for ZPDF; CHI2BEST for ZMINCHI2

    good_inds = (photz >= 0) & (photz_sigma < sigmacut) & (d2d.to('arcsecond').value[unique_idx] < 1) & (cosmos.ip_MAG_APER2[unique_idx] < 25.) & (cosmos.ZP_2[unique_idx] < 0) & (cosmos.NBFILT[unique_idx] >= 25)

    photz = photz[good_inds]
    phot_idx = phot_idx[good_inds]

    good_phot_inds = np.where(np.isfinite(phot.g_cmodel_flux[phot_idx]) & np.isfinite(phot.r_cmodel_flux[phot_idx]) & np.isfinite(phot.i_cmodel_flux[phot_idx]) & np.isfinite(phot.z_cmodel_flux[phot_idx]) & np.isfinite(phot.y_cmodel_flux[phot_idx]) & 
        np.isfinite(phot.g_cmodel_fluxsigma[phot_idx]) & np.isfinite(phot.r_cmodel_fluxsigma[phot_idx]) & np.isfinite(phot.i_cmodel_fluxsigma[phot_idx]) & np.isfinite(phot.z_cmodel_fluxsigma[phot_idx]) & np.isfinite(phot.y_cmodel_fluxsigma[phot_idx]) & 
        np.isfinite(phot.g_cmodel_mag[phot_idx]) & np.isfinite(phot.r_cmodel_mag[phot_idx]) & np.isfinite(phot.i_cmodel_mag[phot_idx]) & np.isfinite(phot.z_cmodel_mag[phot_idx]) & np.isfinite(phot.y_cmodel_mag[phot_idx]) & 
        np.isfinite(phot.g_cmodel_magsigma[phot_idx]) & np.isfinite(phot.r_cmodel_magsigma[phot_idx]) & np.isfinite(phot.i_cmodel_magsigma[phot_idx]) & np.isfinite(phot.z_cmodel_magsigma[phot_idx]) & np.isfinite(phot.y_cmodel_magsigma[phot_idx]))

    if flux_units == 'linear':
        modifier = lambda x: x
    elif flux_units == 'log':
        modifier = np.log10

    cosmos_fluxes = np.vstack((phot['g_cmodel_flux'], phot['r_cmodel_flux'], phot['i_cmodel_flux'], phot['z_cmodel_flux'], phot['y_cmodel_flux'])).T[phot_idx]
    cosmos_flux_errs = np.vstack((phot['g_cmodel_fluxsigma'], phot['r_cmodel_fluxsigma'], phot['i_cmodel_fluxsigma'], phot['z_cmodel_fluxsigma'], phot['y_cmodel_fluxsigma'])).T[phot_idx]

    cosmos_fluxes = cosmos_fluxes[good_phot_inds]
    cosmos_flux_errs = cosmos_flux_errs[good_phot_inds]

    g_cosmos, r_cosmos, i_cosmos, z_cosmos, y_cosmos = cosmos_fluxes.T
    g_err_cosmos, r_err_cosmos, i_err_cosmos, z_err_cosmos, y_err_cosmos = cosmos_flux_errs.T

    g_spec = modifier(data_spec['g_cmodel_flux'])
    r_spec = modifier(data_spec['r_cmodel_flux'])
    i_spec = modifier(data_spec['i_cmodel_flux'])
    z_spec = modifier(data_spec['z_cmodel_flux'])
    y_spec = modifier(data_spec['y_cmodel_flux'])

    vectors_spec = np.vstack((g_spec, r_spec, i_spec, z_spec, y_spec)).T
    norm_spec = np.sqrt(np.sum(vectors_spec**2, axis = 1))
    unit_vectors_spec = vectors_spec / norm_spec.reshape(-1,1)

    vectors_cosmos = np.vstack((g_cosmos, r_cosmos, i_cosmos, z_cosmos, y_cosmos)).T
    norm_cosmos = np.sqrt(np.sum(vectors_cosmos**2, axis = 1))
    unit_vectors_cosmos = vectors_cosmos / norm_cosmos.reshape(-1,1)

    match_index = []
    match_norm = []

    for this_vec, this_norm in tqdm(zip(unit_vectors_cosmos, norm_cosmos), total = len(norm_cosmos)):

        closest_ind = np.argmax(unit_vectors_spec @ this_vec)
        match_index.append(closest_ind)
        match_norm.append(this_norm/norm_spec[closest_ind])

    match_norm = np.array(match_norm)

    rescaled_matched_spec = vectors_spec[match_index] * match_norm.reshape(-1,1) 

    specz = data_spec['specz_redshift'][match_index]
    
    if flux_units == 'linear':
        rescaled_matched_spec_errs = np.vstack((data_spec.g_cmodel_fluxsigma, data_spec.r_cmodel_fluxsigma, data_spec.i_cmodel_fluxsigma, data_spec.z_cmodel_fluxsigma, data_spec.y_cmodel_fluxsigma)).T[match_index] * match_norm.reshape(-1,1)
    else:
        pass
        # rescaled_matched_spec_errs = np.vstack()

    # flux_errs = np.vstack((data_phot.g_cmodel_fluxsigma, data_phot.r_cmodel_fluxsigma, data_phot.i_cmodel_fluxsigma, data_phot.z_cmodel_fluxsigma, data_phot.y_cmodel_fluxsigma)).T
    mag_errs = np.vstack((phot.g_cmodel_magsigma, phot.r_cmodel_magsigma, phot.i_cmodel_magsigma, phot.z_cmodel_magsigma, phot.y_cmodel_magsigma)).T[phot_idx]

    differential_error = np.nan_to_num(np.sqrt(cosmos_flux_errs**2 - rescaled_matched_spec_errs**2))

    noisy_rescaled_matched_spec = np.random.normal(rescaled_matched_spec, differential_error)
    # print('DO NOT LEAVE THIS; TESTING PURPOSES ONLY')
    # noisy_rescaled_matched_spec = np.random.normal(rescaled_matched_spec, 0)

    total_sn = np.sqrt(np.sum((noisy_rescaled_matched_spec/cosmos_flux_errs)**2, axis = 1))
    lim_indices = np.where((total_sn > sn_lim) & np.all(noisy_rescaled_matched_spec > 0, axis = 1))

    # Turn fluxes back into magnitudes

    specz = specz[lim_indices]
    g, r, i, z, y = -2.5*np.log10((noisy_rescaled_matched_spec[lim_indices]*10**-9)/3631).T # noisy_rescaled_matched_spec is in nano Janskys
    g_err, r_err, i_err, z_err, y_err = mag_errs[lim_indices].T

    mags = np.vstack((g,r,i,z,y)).T
    mag_errs = np.vstack((g_err,r_err,i_err,z_err,y_err)).T

    colors = np.array([g-r, r-i, i-z, z-y]).T
    color_errs = np.array([g_err+r_err, r_err+i_err, i_err+z_err, z_err+y_err]).T

    triplets = np.array([(g-r) - (r-i), (r-i) - (i-z), (i-z) - (z-y)]).T
    triplet_errs = np.array([(g_err+r_err) + (r_err+i_err), (r_err+i_err) + (i_err+z_err), (i_err+z_err) + (z_err+y_err)]).T

    features = np.hstack((mags, colors, triplets))
    feature_errs = np.hstack((mag_errs, color_errs, triplet_errs))

    return specz, features, feature_errs








def get_features(indices, inputfp = './HSC/HSC_wide_clean_pdr2.fits'):

    np.random.seed(4132)

    data = fits.open(inputfp)[1].data[indices]

    specz = data['specz_redshift'] + (np.random.random(len(data['specz_redshift']))*10**-5)

    mags = np.vstack((data['g_cmodel_mag'], data['r_cmodel_mag'], data['i_cmodel_mag'], data['z_cmodel_mag'], data['y_cmodel_mag'])).T
    mag_errs = np.vstack((data['g_cmodel_magsigma'], data['r_cmodel_magsigma'], data['i_cmodel_magsigma'], data['z_cmodel_magsigma'], data['y_cmodel_magsigma'])).T

    g, r, i, z, y = mags.T
    g_err, r_err, i_err, z_err, y_err = mag_errs.T

    colors = np.array([g-r, r-i, i-z, z-y]).T
    color_errs = np.array([g_err+r_err, r_err+i_err, i_err+z_err, z_err+y_err]).T

    triplets = np.array([(g-r) - (r-i), (r-i) - (i-z), (i-z) - (z-y)]).T
    triplet_errs = np.array([(g_err+r_err) + (r_err+i_err), (r_err+i_err) + (i_err+z_err), (i_err+z_err) + (z_err+y_err)]).T

    features = np.hstack((mags, colors, triplets))
    feature_errs = np.hstack((mag_errs, color_errs, triplet_errs))

    return specz, features, feature_errs




def generate_input_files(directory, filename = 'tpzrun', inputfp = 'HSC/HSC_wide_clean_pdr2.fits', tpz_nnr_nnc_split = np.array([0.333, 0.666]), nn_train_val_split = 0.85, closest_match_split = False, overall_split = 0.8, ishift = 0., gzshift = 0., gz_colorlim = None, i_maglim = None, no_nnr = True, train_frac_lim = 1., subrun = False):


    if no_nnr and all(tpz_nnr_nnc_split == np.array([0.333, 0.666])):
        tpz_nnr_nnc_split = np.array([0.5, 0.5])

    if not os.path.isfile(directory + filename + '.train_inds') and not os.path.isfile(directory + filename + '.test_inds'):

        print('Generating fresh indexing files in %s' % directory)

        indices = hsc_select(inputfp = inputfp, zlim = 1.5, sn_lim = 10., notzero = True)[0] # indices contains the indices of objects that are good for training/testing/whatever

        if train_frac_lim != 1.:
            np.random.shuffle(indices)
            indices = np.sort(indices[:int(len(indices)*train_frac_lim)])

        if closest_match_split:
            train_inds = indices
        elif not (ishift == 0 and gzshift == 0):
            train_inds, test_inds = ratio_test_train_split(indices, ishift = ishift, gzshift = gzshift, savedir = directory, filename = filename, specfile = inputfp)
        else:
            train_inds, test_inds = default_test_train_split(indices, i_maglim = i_maglim, gz_colorlim = gz_colorlim, splitfrac = overall_split, specfile = inputfp)

        scrambled_train = np.arange(len(train_inds))
        np.random.shuffle(scrambled_train)

        tpz_indicator = np.zeros(len(train_inds), dtype = int)
        nnr_indicator = np.zeros(len(train_inds), dtype = int)
        nnc_indicator = np.zeros(len(train_inds), dtype = int)

        tpz_split, nnr_split, nnc_split = np.split(scrambled_train, (tpz_nnr_nnc_split * len(scrambled_train)).astype(int))

        tpz_indicator[tpz_split] = 1
        nnr_indicator[nnr_split] = 1
        nnc_indicator[nnc_split] = 1

        np.savetxt(directory + filename + '.train_inds', np.vstack((train_inds, tpz_indicator, nnr_indicator, nnc_indicator)).T, header = 'train_inds  tpz_train  nnr_train  nnc_train', fmt = '%i  %i  %i  %i')
        if not closest_match_split:
            np.savetxt(directory + filename + '.app_inds', test_inds, header = 'test_inds', fmt = '%i')

        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not os.path.isfile(directory + filename + '.tpz_train') and not os.path.isfile(directory + filename + '.tpz_test'):

        print('Generating fresh TPZ input files in %s' % directory)

        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', unpack = True, dtype = int)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        tpz_train_inds = train_inds[tpz_indicator]
        tpz_test_inds = train_inds[nnr_indicator | nnc_indicator]

        train_specz, train_features, train_feature_errs = get_features(tpz_train_inds, inputfp = inputfp)
        test_specz, test_features, test_feature_errs = get_features(tpz_test_inds, inputfp = inputfp)

        header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

        np.savetxt(directory + filename + '.tpz_train', np.hstack((train_specz.reshape(-1,1), train_features, train_feature_errs)), header = header, fmt = fmt)
        np.savetxt(directory + filename + '.tpz_test', np.hstack((test_specz.reshape(-1,1), test_features, test_feature_errs)), header = header, fmt = fmt)

        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    
    elif (os.path.isfile(directory + 'output/results/' + filename + '.0.mlz') and not 
            (os.path.isfile(directory + filename + '.nnr_train') or 
                os.path.isfile(directory + filename + '.nnr_test') or 
                os.path.isfile(directory + filename + '.nnr_validate'))):

        print('Generating fresh NNR input files in %s' % directory)

        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', dtype = int, unpack = True)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        train_inds_scrambled = np.copy(train_inds[nnr_indicator])
        np.random.shuffle(train_inds_scrambled)

        nnr_train_inds, nnr_validate_inds = np.split(train_inds_scrambled, [int(nn_train_val_split*len(train_inds_scrambled))])
        nnr_train_inds.sort()
        nnr_validate_inds.sort()

        np.savetxt(directory + filename + '.nnr_train_inds', nnr_train_inds, fmt = '%i')
        np.savetxt(directory + filename + '.nnr_val_inds', nnr_validate_inds, fmt = '%i')

        nnr_test_inds = train_inds[nnc_indicator]

        nnr_train_specz, nnr_train_features, nnr_train_feature_errs = get_features(nnr_train_inds, inputfp = inputfp)
        nnr_validate_specz, nnr_validate_features, nnr_validate_feature_errs = get_features(nnr_validate_inds, inputfp = inputfp)
        nnr_test_specz, nnr_test_features, nnr_test_feature_errs = get_features(nnr_test_inds, inputfp = inputfp)

        tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.0.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        nnr_indicator_trans = nnr_indicator[~tpz_indicator]
        nnc_indicator_trans = nnc_indicator[~tpz_indicator]

        nnr_train_tpz_features, nnr_validate_tpz_features = np.split(tpz_features[nnr_indicator_trans], [int(nn_train_val_split*np.sum(nnr_indicator_trans))])
        nnr_test_tpz_features = tpz_features[nnc_indicator_trans]

        nnr_train_features_lo = nnr_train_features - nnr_train_feature_errs
        nnr_validate_features_lo = nnr_validate_features - nnr_validate_feature_errs
        nnr_test_features_lo = nnr_test_features - nnr_test_feature_errs
        nnr_train_features_hi = nnr_train_features + nnr_train_feature_errs
        nnr_validate_features_hi = nnr_validate_features + nnr_validate_feature_errs
        nnr_test_features_hi = nnr_test_features + nnr_test_feature_errs
        nnr_train_features_frac = nnr_train_feature_errs/nnr_train_features
        nnr_validate_features_frac = nnr_validate_feature_errs/nnr_validate_features
        nnr_test_features_frac = nnr_test_feature_errs/nnr_test_features

        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(directory + filename + '.nnr_train', np.hstack((nnr_train_specz.reshape(-1,1), nnr_train_features, nnr_train_tpz_features, nnr_train_feature_errs, nnr_train_features_lo, nnr_train_features_hi, nnr_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnr_validate', np.hstack((nnr_validate_specz.reshape(-1,1), nnr_validate_features, nnr_validate_tpz_features, nnr_validate_feature_errs, nnr_validate_features_lo, nnr_validate_features_hi, nnr_validate_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnr_test', np.hstack((nnr_test_specz.reshape(-1,1), nnr_test_features, nnr_test_tpz_features, nnr_test_feature_errs, nnr_test_features_lo, nnr_test_features_hi, nnr_test_features_frac)), comments = '', header = header, fmt = fmt)

        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)



    elif (os.path.isfile(directory + 'output/results/' + filename + '.0.mlz') and not 
            (os.path.isfile(directory + filename + '.nnc_train') or 
                os.path.isfile(directory + filename + '.nnc_test') or 
                os.path.isfile(directory + filename + '.nnc_validate'))):

        print('Generating fresh NNC input files in %s' % directory)
        
        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', dtype = int, unpack = True)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        nnr_test_inds = train_inds[nnc_indicator]
        nnr_test_specz, nnr_test_features, nnr_test_feature_errs = get_features(nnr_test_inds, inputfp = inputfp)
        nnr_test_tpz_features = read_csv(directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#')[['zphot', 'zconf', 'zerr']].to_numpy()

        translation_inds = np.arange(len(nnr_test_inds))
        np.random.shuffle(translation_inds)

        nnc_train_inds, nnc_validate_inds = np.split(nnr_test_inds[translation_inds], {int(nn_train_val_split*len(nnr_test_inds))})
        train_trans, val_trans = np.split(translation_inds, [int(nn_train_val_split*len(nnr_test_inds))])

        np.savetxt(directory + filename + '.nnc_train_inds', nnc_train_inds, fmt = '%i')
        np.savetxt(directory + filename + '.nnc_val_inds', nnc_validate_inds, fmt = '%i')

        np.savetxt(directory + filename + '.nnc_train_inds', np.vstack((nnc_train_inds, train_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')
        np.savetxt(directory + filename + '.nnc_validate_inds', np.vstack((nnc_validate_inds, val_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')

        nnc_train_features = nnr_test_features[train_trans]
        nnc_validate_features = nnr_test_features[val_trans]
        
        nnc_train_feature_errs = nnr_test_feature_errs[train_trans]
        nnc_validate_feature_errs = nnr_test_feature_errs[val_trans]
        
        nnc_train_specz = nnr_test_specz[train_trans]
        nnc_validate_specz = nnr_test_specz[val_trans]
        
        nnc_train_tpz_features = nnr_test_tpz_features[train_trans]
        nnc_validate_tpz_features = nnr_test_tpz_features[val_trans]
        
        nnc_train_features_lo = nnc_train_features - nnc_train_feature_errs
        nnc_validate_features_lo = nnc_validate_features - nnc_validate_feature_errs
        nnc_train_features_hi = nnc_train_features + nnc_train_feature_errs
        nnc_validate_features_hi = nnc_validate_features + nnc_validate_feature_errs
        nnc_train_features_frac = nnc_train_feature_errs/nnc_train_features
        nnc_validate_features_frac = nnc_validate_feature_errs/nnc_validate_features
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(directory + filename + '.nnc_train', np.hstack((nnc_train_specz.reshape(-1,1), nnc_train_features, nnc_train_tpz_features, nnc_train_feature_errs, nnc_train_features_lo, nnc_train_features_hi, nnc_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnc_validate', np.hstack((nnc_validate_specz.reshape(-1,1), nnc_validate_features, nnc_validate_tpz_features, nnc_validate_feature_errs, nnc_validate_features_lo, nnc_validate_features_hi, nnc_validate_features_frac)), comments = '', header = header, fmt = fmt)
        # np.savetxt(directory + filename + '.nnc_test', np.hstack((nnc_test_specz.reshape(-1,1), nnc_test_features, nnc_test_tpz_features, nnc_test_feature_errs, nnc_test_features_lo, nnc_test_features_hi, nnc_test_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not os.path.isfile(directory + filename + '.tpz_app'):

        print('Generating TPZ application set files...')

        if closest_match_split:
            indices = hsc_select(inputfp = inputfp, zlim = 1.5, sn_lim = 10., notzero = True)[0]
            app_specz, app_features, app_feature_errs = closest_matches(indices, inputfp, photfile = 'MATCHED_COSMOS')
        else:
            app_inds = np.loadtxt(directory + filename + '.app_inds', dtype = int)
            app_specz, app_features, app_feature_errs = get_features(app_inds, inputfp = inputfp)

        header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

        np.savetxt(directory + filename + '.tpz_app', np.hstack((app_specz.reshape(-1,1), app_features, app_feature_errs)), header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not os.path.isfile(directory + filename + '.nn_app') and os.path.isfile(directory + 'output/results/' + filename + '.1.mlz'):

        print('Generating NN application set files...')

        if closest_match_split:
            indices = [[0], range(1,13), range(13,25)]
            app_specz, app_features, app_feature_errs = [np.loadtxt(directory + filename + '.tpz_app', usecols = thisindex) for thisindex in indices]
        else:
            app_inds = np.loadtxt(directory + filename + '.app_inds', dtype = int)
            app_specz, app_features, app_feature_errs = get_features(app_inds, inputfp = inputfp)

        app_tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.1.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        app_features_lo = app_features - app_feature_errs
        app_features_hi = app_features + app_feature_errs
        app_features_frac = app_feature_errs/app_features
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(directory + filename + '.nn_app', np.hstack((app_specz.reshape(-1,1), app_features, app_tpz_features, app_feature_errs, app_features_lo, app_features_hi, app_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not subrun:
        print('All of these files already exist.  Please delete the input files before continuing.')




def generate_input_files_forirene(directory, filename = 'tpzrun', inputfp = 'HSC/HSC_wide_clean_pdr2.fits', tpz_nnr_nnc_split = np.array([0.333, 0.666]), nn_train_val_split = 0.85, overall_split = 0.8, no_nnr = True, train_frac_lim = 1., subrun = False):

    # directory: The location of the run where you want to save all the files
    # filesname: The name of all the files for this run
    # inputfp: The file path of the input catalog
    # tpz_nnr_nnc_split: The split of the data for TPZ/NNR/NNC training; default is in thirds (or in halves if there is no NNR)
    # nn_train_val_split: Fraction of objects assigned to each NN for training that will be used for training (as opposed to validation)
    # overall_split: The fraction of the input catalog that will be used for training TPZ/NNR/NNC.  The remaining objects will be the "application set"
    # train_frac_lim: Fraction of training sample you would like to use.  Should typically be 1. 

    # NOTE TO IRENE
    # Don't forget, you will need to replace default_test_train_split with your own function for splitting the indices into training and application indices.  It should split them using overall_split
    # Write a function called get_features that takes a list of indices as an argument and outputs (specz, photometric_features, photometric_feature_errs) for those objects.  photometric_features (and errs) should have shape NxM where N is the number of objects and M is the number of features
    # Write a function to replace hsc_select that returns all the indices of objects that are acceptable for training/fitting.  No NaNs, negative magnitudes, etc.  You may want to apply a signal to noise cut, etc...

    # Here is the run order:
    # 1) Generate Files (will make indexing files and TPZ train/test/application files)
    # 2) Run TPZ (mpiexec -n <NUMCORES> runmlz <filename>.inputs)
    # 3) Run TPZ on application set (mpiexec -n <NUMCORES> runMLZ --modify testfile=<filename>.tpz_app --no_train <filename>.inputs)
    # 4) Generate Files (will make NNR and NNC train/test/application files)
    # 5) Train NNC (nnc = nnclass.nn_classifier(run_folder = <directory>, data_name = <filename>))
    # 6) Run NNC on application set (nnc.fit_app())
    # At this point, your important outputs will be in the .nn_app file (for the TPZ features) and the results_application.dat file within the NNC's output folder (which has all the NNC confidence values)

    #If there isn't a NNR running, just set the default split to 50/50

    if no_nnr and all(tpz_nnr_nnc_split == np.array([0.333, 0.666])):
        tpz_nnr_nnc_split = np.array([0.5, 0.5])

    # =======================
    # GENERATE INDEXING FILES
    # =======================

    if not os.path.isfile(directory + filename + '.train_inds') and not os.path.isfile(directory + filename + '.test_inds'):

        print('Generating fresh indexing files in %s' % directory)

        # indices should contain the integer indices of objects in the catalog that are good for training/testing/etc.
        indices = hsc_select(inputfp = inputfp, zlim = 1.5, sn_lim = 10., notzero = True)[0] # REPLACE

        if train_frac_lim != 1.:
            np.random.shuffle(indices)
            indices = np.sort(indices[:int(len(indices)*train_frac_lim)])

        # Here, it has split indices into two lists, which are the OVERALL train and test indices (for the whole pipeline, not a single element; this is essentially the application set).
        train_inds, test_inds = default_test_train_split(indices, i_maglim = i_maglim, gz_colorlim = gz_colorlim, splitfrac = overall_split, specfile = inputfp) # REPLACE

        # Shuffle the training indices and assign objects as training objects for TPZ, NNR, and NNC

        scrambled_train = np.arange(len(train_inds))
        np.random.shuffle(scrambled_train)

        tpz_indicator = np.zeros(len(train_inds), dtype = int)
        nnr_indicator = np.zeros(len(train_inds), dtype = int)
        nnc_indicator = np.zeros(len(train_inds), dtype = int)

        tpz_split, nnr_split, nnc_split = np.split(scrambled_train, (tpz_nnr_nnc_split * len(scrambled_train)).astype(int))

        tpz_indicator[tpz_split] = 1
        nnr_indicator[nnr_split] = 1
        nnc_indicator[nnc_split] = 1

        # Save some files that have flags for what trains on what

        np.savetxt(directory + filename + '.train_inds', np.vstack((train_inds, tpz_indicator, nnr_indicator, nnc_indicator)).T, header = 'train_inds  tpz_train  nnr_train  nnc_train', fmt = '%i  %i  %i  %i')
        np.savetxt(directory + filename + '.app_inds', test_inds, header = 'test_inds', fmt = '%i')

        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    # ==================
    # GENERATE TPZ FILES
    # ==================

    elif not os.path.isfile(directory + filename + '.tpz_train') and not os.path.isfile(directory + filename + '.tpz_test'):

        print('Generating fresh TPZ input files in %s' % directory)

        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', unpack = True, dtype = int)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        tpz_train_inds = train_inds[tpz_indicator]
        tpz_test_inds = train_inds[nnr_indicator | nnc_indicator]  # The TPZ test indices are all the objects in the pipeline training set that won't be used by the NNC or NNR

        # Make a function called 'get_features' that returns the object specz's, all relevant photometric features, and their errors
        train_specz, train_features, train_feature_errs = get_features(tpz_train_inds, inputfp = inputfp)
        test_specz, test_features, test_feature_errs = get_features(tpz_test_inds, inputfp = inputfp)

        header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

        # Save the TPZ training and test set files - after this step, you're going to have to run TPZ before anything else will happen

        np.savetxt(directory + filename + '.tpz_train', np.hstack((train_specz.reshape(-1,1), train_features, train_feature_errs)), header = header, fmt = fmt)
        np.savetxt(directory + filename + '.tpz_test', np.hstack((test_specz.reshape(-1,1), test_features, test_feature_errs)), header = header, fmt = fmt)

        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    # ==================
    # GENERATE NNR FILES
    # ==================

    elif (os.path.isfile(directory + 'output/results/' + filename + '.0.mlz') and not 
            (os.path.isfile(directory + filename + '.nnr_train') or 
                os.path.isfile(directory + filename + '.nnr_test') or 
                os.path.isfile(directory + filename + '.nnr_validate'))):

        # This section likely isn't needed since we aren't running the NNR, but I don't want to accidentally break the rest and it just generates a few extra files

        print('Generating fresh NNR input files in %s' % directory)

        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', dtype = int, unpack = True)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        #Scramble the training objects for the NNR to produce its training and validation files

        train_inds_scrambled = np.copy(train_inds[nnr_indicator])
        np.random.shuffle(train_inds_scrambled)

        nnr_train_inds, nnr_validate_inds = np.split(train_inds_scrambled, [int(nn_train_val_split*len(train_inds_scrambled))])
        nnr_train_inds.sort()
        nnr_validate_inds.sort()

        np.savetxt(directory + filename + '.nnr_train_inds', nnr_train_inds, fmt = '%i')
        np.savetxt(directory + filename + '.nnr_val_inds', nnr_validate_inds, fmt = '%i')

        nnr_test_inds = train_inds[nnc_indicator]

        # Get the features for the correct indices

        nnr_train_specz, nnr_train_features, nnr_train_feature_errs = get_features(nnr_train_inds, inputfp = inputfp)
        nnr_validate_specz, nnr_validate_features, nnr_validate_feature_errs = get_features(nnr_validate_inds, inputfp = inputfp)
        nnr_test_specz, nnr_test_features, nnr_test_feature_errs = get_features(nnr_test_inds, inputfp = inputfp)

        # Load the TPZ outputs because the NNR needs them for training

        tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.0.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        nnr_indicator_trans = nnr_indicator[~tpz_indicator]
        nnc_indicator_trans = nnc_indicator[~tpz_indicator]

        # Build the features

        nnr_train_tpz_features, nnr_validate_tpz_features = np.split(tpz_features[nnr_indicator_trans], [int(nn_train_val_split*np.sum(nnr_indicator_trans))])
        nnr_test_tpz_features = tpz_features[nnc_indicator_trans]

        nnr_train_features_lo = nnr_train_features - nnr_train_feature_errs
        nnr_validate_features_lo = nnr_validate_features - nnr_validate_feature_errs
        nnr_test_features_lo = nnr_test_features - nnr_test_feature_errs
        nnr_train_features_hi = nnr_train_features + nnr_train_feature_errs
        nnr_validate_features_hi = nnr_validate_features + nnr_validate_feature_errs
        nnr_test_features_hi = nnr_test_features + nnr_test_feature_errs
        nnr_train_features_frac = nnr_train_feature_errs/nnr_train_features
        nnr_validate_features_frac = nnr_validate_feature_errs/nnr_validate_features
        nnr_test_features_frac = nnr_test_feature_errs/nnr_test_features

        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        # Save all the NNR files

        np.savetxt(directory + filename + '.nnr_train', np.hstack((nnr_train_specz.reshape(-1,1), nnr_train_features, nnr_train_tpz_features, nnr_train_feature_errs, nnr_train_features_lo, nnr_train_features_hi, nnr_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnr_validate', np.hstack((nnr_validate_specz.reshape(-1,1), nnr_validate_features, nnr_validate_tpz_features, nnr_validate_feature_errs, nnr_validate_features_lo, nnr_validate_features_hi, nnr_validate_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnr_test', np.hstack((nnr_test_specz.reshape(-1,1), nnr_test_features, nnr_test_tpz_features, nnr_test_feature_errs, nnr_test_features_lo, nnr_test_features_hi, nnr_test_features_frac)), comments = '', header = header, fmt = fmt)

        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    # ==================
    # GENERATE NNC FILES
    # ==================

    elif (os.path.isfile(directory + 'output/results/' + filename + '.0.mlz') and not 
            (os.path.isfile(directory + filename + '.nnc_train') or 
                os.path.isfile(directory + filename + '.nnc_test') or 
                os.path.isfile(directory + filename + '.nnc_validate'))):

        print('Generating fresh NNC input files in %s' % directory)
        
        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', dtype = int, unpack = True)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        # Read in the TPZ features from the NNR test set fiel

        nnr_test_inds = train_inds[nnc_indicator]
        nnr_test_specz, nnr_test_features, nnr_test_feature_errs = get_features(nnr_test_inds, inputfp = inputfp)
        nnr_test_tpz_features = read_csv(directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#')[['zphot', 'zconf', 'zerr']].to_numpy()

        translation_inds = np.arange(len(nnr_test_inds))
        np.random.shuffle(translation_inds)

        # Split the training set in to training and validation

        nnc_train_inds, nnc_validate_inds = np.split(nnr_test_inds[translation_inds], {int(nn_train_val_split*len(nnr_test_inds))})
        train_trans, val_trans = np.split(translation_inds, [int(nn_train_val_split*len(nnr_test_inds))])

        np.savetxt(directory + filename + '.nnc_train_inds', nnc_train_inds, fmt = '%i')
        np.savetxt(directory + filename + '.nnc_val_inds', nnc_validate_inds, fmt = '%i')

        np.savetxt(directory + filename + '.nnc_train_inds', np.vstack((nnc_train_inds, train_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')
        np.savetxt(directory + filename + '.nnc_validate_inds', np.vstack((nnc_validate_inds, val_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')

        # Build the features that will go in the train and validate files

        nnc_train_features = nnr_test_features[train_trans]
        nnc_validate_features = nnr_test_features[val_trans]
        
        nnc_train_feature_errs = nnr_test_feature_errs[train_trans]
        nnc_validate_feature_errs = nnr_test_feature_errs[val_trans]
        
        nnc_train_specz = nnr_test_specz[train_trans]
        nnc_validate_specz = nnr_test_specz[val_trans]
        
        nnc_train_tpz_features = nnr_test_tpz_features[train_trans]
        nnc_validate_tpz_features = nnr_test_tpz_features[val_trans]
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 

        # Save the NNC training and validation files

        np.savetxt(directory + filename + '.nnc_train', np.hstack((nnc_train_specz.reshape(-1,1), nnc_train_features, nnc_train_tpz_features, nnc_train_feature_errs, nnc_train_features_lo, nnc_train_features_hi, nnc_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnc_validate', np.hstack((nnc_validate_specz.reshape(-1,1), nnc_validate_features, nnc_validate_tpz_features, nnc_validate_feature_errs, nnc_validate_features_lo, nnc_validate_features_hi, nnc_validate_features_frac)), comments = '', header = header, fmt = fmt)
        # np.savetxt(directory + filename + '.nnc_test', np.hstack((nnc_test_specz.reshape(-1,1), nnc_test_features, nnc_test_tpz_features, nnc_test_feature_errs, nnc_test_features_lo, nnc_test_features_hi, nnc_test_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    # ==================================
    # GENERATE TPZ APPLICATION SET FILES
    # ==================================

    elif not os.path.isfile(directory + filename + '.tpz_app'):

        print('Generating TPZ application set files...')

        app_inds = np.loadtxt(directory + filename + '.app_inds', dtype = int)
        app_specz, app_features, app_feature_errs = get_features(app_inds, inputfp = inputfp)

        header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

        # Generate the TPZ application set

        np.savetxt(directory + filename + '.tpz_app', np.hstack((app_specz.reshape(-1,1), app_features, app_feature_errs)), header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    # ==========================================
    # GENERATE NNC AND NNR APPLICATION SET FILES
    # ==========================================

    elif not os.path.isfile(directory + filename + '.nn_app') and os.path.isfile(directory + 'output/results/' + filename + '.1.mlz'):

        print('Generating NN application set files...')

        app_inds = np.loadtxt(directory + filename + '.app_inds', dtype = int)
        app_specz, app_features, app_feature_errs = get_features(app_inds, inputfp = inputfp)

        app_tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.1.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy '
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3

        # Generate the NN application set

        np.savetxt(directory + filename + '.nn_app', np.hstack((app_specz.reshape(-1,1), app_features, app_tpz_features, app_feature_errs, app_features_lo, app_features_hi, app_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not subrun:
        print('All of these files already exist.  Please delete the input files before continuing.')





def generate_input_files_cosmos_only(directory, filename = 'tpzrun', inputfp = 'HSC/HSC_wide_clean_pdr2.fits', tpz_nnr_nnc_split = np.array([0.333, 0.666]), nn_train_val_split = 0.85, train_type = 'split', overall_split = 0.8, ishift = 0., gzshift = 0., gz_colorlim = None, i_maglim = None, no_nnr = True, subrun = False):

    #split_type can be 'split' (split the spec sample), 'cosmos' (use the COSMOS photometric sample to generate a training sample), or 'all' (use all of the spec sample for training)

    if no_nnr:
        tpz_nnr_nnc_split = np.array([0.5, 0.5])

    if not os.path.isfile(directory + filename + '.train_inds') and (no_nnr or not os.path.isfile(directory + filename + '.test_inds')):

        print('Generating fresh indexing files in %s' % directory)

        indices = hsc_select(inputfp = inputfp, zlim = 1.5, sn_lim = 10., notzero = True)[0]

        if train_type == 'cosmos':
            if not os.path.isfile(directory + filename + '.closest_matches'):
                cm_specz, cm_features, cm_feature_errs = cosmos_closest_matches(indices, inputfp)
            else:
                load_indices = [[0], range(1,13), range(13,25)]
                cm_specz, cm_features, cm_feature_errs = [read_csv(directory + filename + '.closest_matches', delim_whitespace = True, usecols = thisindex).values for thisindex in load_indices]

            header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
            fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12
            np.savetxt(directory + filename + '.closest_matches', np.hstack((cm_specz.reshape(-1,1), cm_features, cm_feature_errs)), header = header, fmt = fmt, comments = '')
            # new_indices = np.copy(indices)
            # np.random.shuffle(new_indices)
            # split_index = int(overall_split*len(new_indices))
            # train_inds = new_indices[:split_index]
            # test_inds = new_indices[split_index:]
            train_inds = np.arange(len(cm_specz))
        elif train_type == 'split' and not (ishift == 0 and gzshift == 0):
            train_inds, test_inds = ratio_test_train_split(indices, ishift = ishift, gzshift = gzshift, savedir = directory, filename = filename, specfile = inputfp, photfile = 'HSC/HSC_wide_clean_pdr2_phot_depth_Unltd_COSMOS.fits')
            # train_inds= np.concatenate(ratio_test_train_split(indices, ishift = ishift, gzshift = gzshift, savedir = directory, filename = filename, specfile = inputfp))
        elif train_type == 'split':
            train_inds, test_inds = default_test_train_split(indices, i_maglim = i_maglim, gz_colorlim = gz_colorlim, splitfrac = overall_split, specfile = inputfp)
            # train_inds = np.concatenate(default_test_train_split(indices, i_maglim = i_maglim, gz_colorlim = gz_colorlim, splitfrac = overall_split, specfile = inputfp))
        elif train_type == 'all':
            train_inds = indices

        scrambled_train = np.arange(len(train_inds))
        np.random.shuffle(scrambled_train)

        tpz_indicator = np.zeros(len(train_inds), dtype = int)
        nnr_indicator = np.zeros(len(train_inds), dtype = int)
        nnc_indicator = np.zeros(len(train_inds), dtype = int)

        tpz_split, nnr_split, nnc_split = np.split(scrambled_train, (tpz_nnr_nnc_split * len(scrambled_train)).astype(int))

        tpz_indicator[tpz_split] = 1
        nnr_indicator[nnr_split] = 1
        nnc_indicator[nnc_split] = 1

        np.savetxt(directory + filename + '.train_inds', np.vstack((train_inds, tpz_indicator, nnr_indicator, nnc_indicator)).T, header = 'train_inds  tpz_train  nnc_train', fmt = '%i  %i  %i  %i')
        if train_type == 'split':
            np.savetxt(directory + filename + '.test_inds', test_inds, header = 'test_inds', fmt = '%i')

        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, train_type, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, subrun = True)



    elif not os.path.isfile(directory + filename + '.tpz_train') and not os.path.isfile(directory + filename + '.tpz_test'):

        print('Generating fresh TPZ input files in %s' % directory)

        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', unpack = True, dtype = int)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        tpz_train_inds = train_inds[tpz_indicator]
        tpz_test_inds = train_inds[nnr_indicator | nnc_indicator]

        if train_type == 'all' or train_type == 'split':

            train_specz, train_features, train_feature_errs = get_features(tpz_train_inds, inputfp = inputfp)
            test_specz, test_features, test_feature_errs = get_features(tpz_test_inds, inputfp = inputfp)

        else:

            indices = [[0], range(1,13), range(13,25)]
            cm_specz, cm_features, cm_feature_errs = [read_csv(directory + filename + '.closest_matches', delim_whitespace = True, usecols = thisindex).values for thisindex in indices]
            train_specz = cm_specz[tpz_train_inds]
            train_features = cm_features[tpz_train_inds]
            train_feature_errs = cm_feature_errs[tpz_train_inds]
            test_specz = cm_specz[tpz_test_inds]
            test_features = cm_features[tpz_test_inds]
            test_feature_errs = cm_feature_errs[tpz_test_inds]

        header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

        np.savetxt(directory + filename + '.tpz_train', np.hstack((train_specz.reshape(-1,1), train_features, train_feature_errs)), header = header, fmt = fmt)
        np.savetxt(directory + filename + '.tpz_test', np.hstack((test_specz.reshape(-1,1), test_features, test_feature_errs)), header = header, fmt = fmt)

        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, train_type, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, subrun = True)

    
    elif (not no_nnr and 
            os.path.isfile(directory + 'output/results/' + filename + '.0.mlz') and not 
            (os.path.isfile(directory + filename + '.nnr_train') or 
                os.path.isfile(directory + filename + '.nnr_test') or 
                os.path.isfile(directory + filename + '.nnr_validate'))):

        print('Generating fresh NNR input files in %s' % directory)

        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', dtype = int, unpack = True)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        train_inds_scrambled = np.copy(train_inds[nnr_indicator])
        np.random.shuffle(train_inds_scrambled)

        nnr_train_inds, nnr_validate_inds = np.split(train_inds_scrambled, [int(nn_train_val_split*len(train_inds_scrambled))])
        nnr_train_inds.sort()
        nnr_validate_inds.sort()

        np.savetxt(directory + filename + '.nnr_train_inds', nnr_train_inds, fmt = '%i')
        np.savetxt(directory + filename + '.nnr_val_inds', nnr_validate_inds, fmt = '%i')

        nnr_test_inds = train_inds[nnc_indicator]

        if train_type == 'cosmos':
            indices = [[0], range(1,13), range(13,25)]
            cm_specz, cm_features, cm_feature_errs = [read_csv(directory + filename + '.closest_matches', delim_whitespace = True, usecols = thisindex).values for thisindex in indices]
            nnr_train_specz = cm_specz[nnr_train_inds]
            nnr_train_features = cm_features[nnr_train_inds]
            nnr_train_feature_errs = cm_feature_errs[nnr_train_inds]
            nnr_test_specz = cm_specz[nnr_test_inds]
            nnr_test_features = cm_features[nnr_test_inds]
            nnr_test_feature_errs = cm_feature_errs[nnr_test_inds]
            nnr_validate_specz = cm_specz[nnr_validate_inds]
            nnr_validate_features = cm_features[nnr_validate_inds]
            nnr_validate_feature_errs = cm_feature_errs[nnr_validate_inds]
        else:
            nnr_train_specz, nnr_train_features, nnr_train_feature_errs = get_features(nnr_train_inds, inputfp = inputfp)
            nnr_validate_specz, nnr_validate_features, nnr_validate_feature_errs = get_features(nnr_validate_inds, inputfp = inputfp)
            nnr_test_specz, nnr_test_features, nnr_test_feature_errs = get_features(nnr_test_inds, inputfp = inputfp)

        tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.0.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        nnr_indicator_trans = nnr_indicator[~tpz_indicator]
        nnc_indicator_trans = nnc_indicator[~tpz_indicator]

        nnr_train_tpz_features, nnr_validate_tpz_features = np.split(tpz_features[nnr_indicator_trans], [int(nn_train_val_split*np.sum(nnr_indicator_trans))])
        nnr_test_tpz_features = tpz_features[nnc_indicator_trans]

        nnr_train_features_lo = nnr_train_features - nnr_train_feature_errs
        nnr_validate_features_lo = nnr_validate_features - nnr_validate_feature_errs
        nnr_test_features_lo = nnr_test_features - nnr_test_feature_errs
        nnr_train_features_hi = nnr_train_features + nnr_train_feature_errs
        nnr_validate_features_hi = nnr_validate_features + nnr_validate_feature_errs
        nnr_test_features_hi = nnr_test_features + nnr_test_feature_errs
        nnr_train_features_frac = nnr_train_feature_errs/nnr_train_features
        nnr_validate_features_frac = nnr_validate_feature_errs/nnr_validate_features
        nnr_test_features_frac = nnr_test_feature_errs/nnr_test_features

        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(directory + filename + '.nnr_train', np.hstack((nnr_train_specz.reshape(-1,1), nnr_train_features, nnr_train_tpz_features, nnr_train_feature_errs, nnr_train_features_lo, nnr_train_features_hi, nnr_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnr_validate', np.hstack((nnr_validate_specz.reshape(-1,1), nnr_validate_features, nnr_validate_tpz_features, nnr_validate_feature_errs, nnr_validate_features_lo, nnr_validate_features_hi, nnr_validate_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnr_test', np.hstack((nnr_test_specz.reshape(-1,1), nnr_test_features, nnr_test_tpz_features, nnr_test_feature_errs, nnr_test_features_lo, nnr_test_features_hi, nnr_test_features_frac)), comments = '', header = header, fmt = fmt)

        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, train_type, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, subrun = True)



    elif (os.path.isfile(directory + 'output/results/' + filename + '.0.mlz') and not 
            (os.path.isfile(directory + filename + '.nnc_train') or 
                os.path.isfile(directory + filename + '.nnc_test') or 
                os.path.isfile(directory + filename + '.nnc_validate'))):

        print('Generating fresh NNC input files in %s' % directory)
        
        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', dtype = int, unpack = True)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        if not no_nnr:
            nnr_test_inds = train_inds[nnc_indicator]
            nnr_test_specz = read_csv(directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#', usecols = [0]).to_numpy()
            nnr_test_features = read_csv(directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#', usecols = range(1,13)).to_numpy()
            nnr_test_feature_errs = read_csv(directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#', usecols = range(16,28)).to_numpy()
            nnr_test_tpz_features = read_csv(directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#')[['zphot', 'zconf', 'zerr']].to_numpy()

            translation_inds = np.arange(len(nnr_test_inds))
            np.random.shuffle(translation_inds)

            nnc_train_inds, nnc_validate_inds = np.split(nnr_test_inds[translation_inds], {int(nn_train_val_split*len(nnr_test_inds))})
            train_trans, val_trans = np.split(translation_inds, [int(nn_train_val_split*len(nnr_test_inds))])

            np.savetxt(directory + filename + '.nnc_train_inds', np.vstack((nnc_train_inds, train_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')
            np.savetxt(directory + filename + '.nnc_validate_inds', np.vstack((nnc_validate_inds, val_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')

        else:
            train_inds_scrambled = np.copy(train_inds[nnc_indicator])
            np.random.shuffle(train_inds_scrambled)

            nnc_train_inds, nnc_validate_inds = np.split(train_inds_scrambled, [int(nn_train_val_split*len(train_inds_scrambled))])
            nnc_train_inds.sort()
            nnc_validate_inds.sort()
           
            np.savetxt(directory + filename + '.nnc_train_inds', nnc_train_inds, fmt = '%i')
            np.savetxt(directory + filename + '.nnc_validate_inds', nnc_validate_inds, fmt = '%i')

        if not no_nnr:

            nnc_train_features = nnr_test_features[train_trans]
            nnc_validate_features = nnr_test_features[val_trans]
            
            nnc_train_feature_errs = nnr_test_feature_errs[train_trans]
            nnc_validate_feature_errs = nnr_test_feature_errs[val_trans]

            nnc_train_specz = nnr_test_specz[train_trans]
            nnc_validate_specz = nnr_test_specz[val_trans]
        
            nnc_train_tpz_features = nnr_test_tpz_features[train_trans]
            nnc_validate_tpz_features = nnr_test_tpz_features[val_trans]

        else:

            tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.0.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr
            nnc_indicator_trans = nnc_indicator[~tpz_indicator]
            nnc_train_tpz_features, nnc_validate_tpz_features = np.split(tpz_features[nnc_indicator_trans], [int(nn_train_val_split*np.sum(nnc_indicator_trans))])

            if train_type == 'cosmos':
                indices = [[0], range(1,13), range(13,25)]
                cm_specz, cm_features, cm_feature_errs = [read_csv(directory + filename + '.closest_matches', delim_whitespace = True, usecols = thisindex).values for thisindex in indices]
                nnc_train_specz = cm_specz[nnc_train_inds]
                nnc_train_features = cm_features[nnc_train_inds]
                nnc_train_feature_errs = cm_feature_errs[nnc_train_inds]
                # nnc_test_specz = cm_specz[nnc_test_inds]
                # nnc_test_features = cm_features[nnc_test_inds]
                # nnc_test_feature_errs = cm_feature_errs[nnc_test_inds]
                nnc_validate_specz = cm_specz[nnc_validate_inds]
                nnc_validate_features = cm_features[nnc_validate_inds]
                nnc_validate_feature_errs = cm_feature_errs[nnc_validate_inds]

            else:
                nnc_train_specz, nnc_train_features, nnc_train_feature_errs = get_features(nnc_train_inds, inputfp = inputfp)
                nnc_validate_specz, nnc_validate_features, nnc_validate_feature_errs = get_features(nnc_validate_inds, inputfp = inputfp)
                # nnc_test_specz, nnc_test_features, nnc_test_feature_errs = get_features(nnc_test_inds, inputfp = inputfp)
            
        nnc_train_features_lo = nnc_train_features - nnc_train_feature_errs
        nnc_validate_features_lo = nnc_validate_features - nnc_validate_feature_errs
        nnc_train_features_hi = nnc_train_features + nnc_train_feature_errs
        nnc_validate_features_hi = nnc_validate_features + nnc_validate_feature_errs
        nnc_train_features_frac = nnc_train_feature_errs/nnc_train_features
        nnc_validate_features_frac = nnc_validate_feature_errs/nnc_validate_features
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(directory + filename + '.nnc_train', np.hstack((nnc_train_specz.reshape(-1,1), nnc_train_features, nnc_train_tpz_features, nnc_train_feature_errs, nnc_train_features_lo, nnc_train_features_hi, nnc_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnc_validate', np.hstack((nnc_validate_specz.reshape(-1,1), nnc_validate_features, nnc_validate_tpz_features, nnc_validate_feature_errs, nnc_validate_features_lo, nnc_validate_features_hi, nnc_validate_features_frac)), comments = '', header = header, fmt = fmt)
        # np.savetxt(directory + filename + '.nnc_test', np.hstack((nnc_test_specz.reshape(-1,1), nnc_test_features, nnc_test_tpz_features, nnc_test_feature_errs, nnc_test_features_lo, nnc_test_features_hi, nnc_test_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, train_type, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, subrun = True)

    elif not os.path.isfile(directory + filename + '.tpz_app'):

        print('Generating TPZ application set files...')

        if train_type == 'all' or train_type == 'cosmos':
            
            cosmos_photz, cosmos_features, cosmos_feature_errs = get_cosmos_features()

            header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
            fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

            np.savetxt(directory + filename + '.tpz_app', np.hstack((cosmos_photz.reshape(-1,1), cosmos_features, cosmos_feature_errs)), header = header, fmt = fmt)
            generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, train_type, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)

            # indices = hsc_select(inputfp = inputfp, zlim = 1.5, sn_lim = 10., notzero = True)[0]
            # app_specz, app_features, app_feature_errs = closest_matches(indices, inputfp, photfile = './HSC/HSC_wide_clean_pdr2_phot_depth_Unltd_COSMOS.fits')
        else:
            test_inds = np.loadtxt(directory + filename + '.test_inds', dtype = int)
            test_specz, test_features, test_feature_errs = get_features(test_inds, inputfp = inputfp)

            header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
            fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

            np.savetxt(directory + filename + '.tpz_app', np.hstack((test_specz.reshape(-1,1), test_features, test_feature_errs)), header = header, fmt = fmt)
            generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, train_type, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, subrun = True)

    elif not os.path.isfile(directory + filename + '.nn_app') and os.path.isfile(directory + 'output/results/' + filename + '.1.mlz'):

        print('Generating NN application set files...')

        if train_type == 'all' or train_type == 'cosmos':
            indices = [[0], range(1,13), range(13,25)]
            test_specz, test_features, test_feature_errs = [np.loadtxt(directory + filename + '.tpz_app', usecols = thisindex) for thisindex in indices]
        else:
            test_inds = np.loadtxt(directory + filename + '.test_inds', dtype = int)
            test_specz, test_features, test_feature_errs = get_features(test_inds, inputfp = inputfp)

        test_tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.1.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        test_features_lo = test_features - test_feature_errs
        test_features_hi = test_features + test_feature_errs
        test_features_frac = test_feature_errs/test_features
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(directory + filename + '.nn_app', np.hstack((test_specz.reshape(-1,1), test_features, test_tpz_features, test_feature_errs, test_features_lo, test_features_hi, test_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, train_type, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, subrun = True)

    elif not subrun:
        print('All of these files already exist.  Please delete the input files before continuing.')





def generate_input_files_cosmos(directory, filename = 'tpzrun', inputfp = 'HSC/HSC_wide_clean_pdr2.fits', tpz_nnr_nnc_split = np.array([0.333, 0.666]), nn_train_val_split = 0.85, closest_match_split = False, closest_match_test = False, cosmos2015_train = False, overall_split = 0.8, ishift = 0., gzshift = 0., gz_colorlim = None, i_maglim = None, no_nnr = True, train_frac_lim = 1., subrun = False):


    if no_nnr and all(tpz_nnr_nnc_split == np.array([0.333, 0.666])):
        tpz_nnr_nnc_split = np.array([0.5, 0.5])


    if not os.path.isfile(directory + filename + '.train_inds') and not os.path.isfile(directory + filename + '.test_inds'):

        print('Generating fresh indexing files in %s' % directory)

        indices = hsc_select(inputfp = inputfp, zlim = 1.5, sn_lim = 10., notzero = True)[0]

        if train_frac_lim != 1. and not closest_match_split and not cosmos2015_train:
            np.random.shuffle(indices)
            indices = np.sort(indices[:int(len(indices)*train_frac_lim)])

        if closest_match_split:
            if not os.path.isfile(directory + filename + '.closest_matches'):
                cm_specz, cm_features, cm_feature_errs = closest_matches(indices, inputfp, 'MATCHED_COSMOS', train_frac_lim = train_frac_lim) # Note we change the phot file here to be only those objects that are in the COSMOS field, but we include ALL of them
            else:
                load_indices = [[0], range(1,13), range(13,25)]
                cm_specz, cm_features, cm_feature_errs = [read_csv(directory + filename + '.closest_matches', delim_whitespace = True, usecols = thisindex).values for thisindex in load_indices]

            if closest_match_test:
                indices = np.arange(len(cm_specz))

            header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
            fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12
            np.savetxt(directory + filename + '.closest_matches', np.hstack((cm_specz.reshape(-1,1), cm_features, cm_feature_errs)), header = header, fmt = fmt, comments = '')
            new_indices = np.copy(indices)
            np.random.shuffle(new_indices)
            split_index = int(overall_split*len(new_indices))
            if closest_match_test:
                train_inds = np.sort(new_indices[:split_index])
                test_inds = np.sort(new_indices[split_index:])
            else:
                train_inds = np.arange(len(cm_specz))
        elif cosmos2015_train:
            cosmos_photz, cosmos_features, cosmos_feature_errs = get_cosmos_features()
            indices = np.arange(len(cosmos_photz))
            new_indices = np.copy(indices)
            np.random.shuffle(new_indices)
            split_index = int(overall_split*len(new_indices))
            train_inds = np.sort(new_indices[:split_index])
            test_inds = np.sort(new_indices[split_index:])
        elif not (ishift == 0 and gzshift == 0):
            # train_inds, test_inds = ratio_test_train_split(indices, ishift = ishift, gzshift = gzshift, savedir = directory, filename = filename, specfile = inputfp)
            train_inds= np.concatenate(ratio_test_train_split(indices, ishift = ishift, gzshift = gzshift, savedir = directory, filename = filename, specfile = inputfp))
        else:
            # train_inds, test_inds = default_test_train_split(indices, i_maglim = i_maglim, gz_colorlim = gz_colorlim, splitfrac = overall_split, specfile = inputfp)
            train_inds = np.concatenate(default_test_train_split(indices, i_maglim = i_maglim, gz_colorlim = gz_colorlim, splitfrac = overall_split, specfile = inputfp))

        scrambled_train = np.arange(len(train_inds))
        np.random.shuffle(scrambled_train)

        tpz_indicator = np.zeros(len(train_inds), dtype = int)
        nnr_indicator = np.zeros(len(train_inds), dtype = int)
        nnc_indicator = np.zeros(len(train_inds), dtype = int)

        tpz_split, nnr_split, nnc_split = np.split(scrambled_train, (tpz_nnr_nnc_split * len(scrambled_train)).astype(int))

        tpz_indicator[tpz_split] = 1
        nnr_indicator[nnr_split] = 1
        nnc_indicator[nnc_split] = 1

        np.savetxt(directory + filename + '.train_inds', np.vstack((train_inds, tpz_indicator, nnr_indicator, nnc_indicator)).T, header = 'train_inds  tpz_train  nnr_train  nnc_train', fmt = '%i  %i  %i  %i')
        if closest_match_test or cosmos2015_train:
            np.savetxt(directory + filename + '.app_inds', test_inds, header = 'test_inds', fmt = '%i')

        generate_input_files_cosmos(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, closest_match_test, cosmos2015_train, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not os.path.isfile(directory + filename + '.tpz_train') and not os.path.isfile(directory + filename + '.tpz_test'):

        print('Generating fresh TPZ input files in %s' % directory)

        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', unpack = True, dtype = int)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        tpz_train_inds = train_inds[tpz_indicator]
        tpz_test_inds = train_inds[nnr_indicator | nnc_indicator]

        if closest_match_split:
            indices = [[0], range(1,13), range(13,25)]
            cm_specz, cm_features, cm_feature_errs = [read_csv(directory + filename + '.closest_matches', delim_whitespace = True, usecols = thisindex).values for thisindex in indices]
            train_specz = cm_specz[tpz_train_inds]
            train_features = cm_features[tpz_train_inds]
            train_feature_errs = cm_feature_errs[tpz_train_inds]
            test_specz = cm_specz[tpz_test_inds]
            test_features = cm_features[tpz_test_inds]
            test_feature_errs = cm_feature_errs[tpz_test_inds]
        elif cosmos2015_train:
            cosmos_photz, cosmos_features, cosmos_feature_errs = get_cosmos_features()
            train_specz = cosmos_photz[tpz_train_inds]
            train_features = cosmos_features[tpz_train_inds]
            train_feature_errs = cosmos_feature_errs[tpz_train_inds]
            test_specz = cosmos_photz[tpz_test_inds]
            test_features = cosmos_features[tpz_test_inds]
            test_feature_errs = cosmos_feature_errs[tpz_test_inds]
        else:
            train_specz, train_features, train_feature_errs = get_features(tpz_train_inds, inputfp = inputfp)
            test_specz, test_features, test_feature_errs = get_features(tpz_test_inds, inputfp = inputfp)

        header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

        np.savetxt(directory + filename + '.tpz_train', np.hstack((train_specz.reshape(-1,1), train_features, train_feature_errs)), header = header, fmt = fmt)
        np.savetxt(directory + filename + '.tpz_test', np.hstack((test_specz.reshape(-1,1), test_features, test_feature_errs)), header = header, fmt = fmt)

        generate_input_files_cosmos(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, closest_match_test, cosmos2015_train, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    
    elif (os.path.isfile(directory + 'output/results/' + filename + '.0.mlz') and not 
            (os.path.isfile(directory + filename + '.nnr_train') or 
                os.path.isfile(directory + filename + '.nnr_test') or 
                os.path.isfile(directory + filename + '.nnr_validate'))):

        print('Generating fresh NNR input files in %s' % directory)

        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', dtype = int, unpack = True)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        train_inds_scrambled = np.copy(train_inds[nnr_indicator])
        np.random.shuffle(train_inds_scrambled)

        nnr_train_inds, nnr_validate_inds = np.split(train_inds_scrambled, [int(nn_train_val_split*len(train_inds_scrambled))])
        nnr_train_inds.sort()
        nnr_validate_inds.sort()

        np.savetxt(directory + filename + '.nnr_train_inds', nnr_train_inds, fmt = '%i')
        np.savetxt(directory + filename + '.nnr_val_inds', nnr_validate_inds, fmt = '%i')

        nnr_test_inds = train_inds[nnc_indicator]

        if closest_match_split:
            indices = [[0], range(1,13), range(13,25)]
            cm_specz, cm_features, cm_feature_errs = [read_csv(directory + filename + '.closest_matches', delim_whitespace = True, usecols = thisindex).values for thisindex in indices]
            nnr_train_specz = cm_specz[nnr_train_inds]
            nnr_train_features = cm_features[nnr_train_inds]
            nnr_train_feature_errs = cm_feature_errs[nnr_train_inds]
            nnr_test_specz = cm_specz[nnr_test_inds]
            nnr_test_features = cm_features[nnr_test_inds]
            nnr_test_feature_errs = cm_feature_errs[nnr_test_inds]
            nnr_validate_specz = cm_specz[nnr_validate_inds]
            nnr_validate_features = cm_features[nnr_validate_inds]
            nnr_validate_feature_errs = cm_feature_errs[nnr_validate_inds]
        elif cosmos2015_train:
            cosmos_photz, cosmos_features, cosmos_feature_errs = get_cosmos_features()
            nnr_train_specz = cosmos_photz[nnr_train_inds]
            nnr_train_features = cosmos_features[nnr_train_inds]
            nnr_train_feature_errs = cosmos_feature_errs[nnr_train_inds]
            nnr_test_specz = cosmos_photz[nnr_test_inds]
            nnr_test_features = cosmos_features[nnr_test_inds]
            nnr_test_feature_errs = cosmos_feature_errs[nnr_test_inds]
            nnr_validate_specz = cosmos_photz[nnr_validate_inds]
            nnr_validate_features = cosmos_features[nnr_validate_inds]
            nnr_validate_feature_errs = cosmos_feature_errs[nnr_validate_inds]
        else:
            nnr_train_specz, nnr_train_features, nnr_train_feature_errs = get_features(nnr_train_inds, inputfp = inputfp)
            nnr_validate_specz, nnr_validate_features, nnr_validate_feature_errs = get_features(nnr_validate_inds, inputfp = inputfp)
            nnr_test_specz, nnr_test_features, nnr_test_feature_errs = get_features(nnr_test_inds, inputfp = inputfp)

        tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.0.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        nnr_indicator_trans = nnr_indicator[~tpz_indicator]
        nnc_indicator_trans = nnc_indicator[~tpz_indicator]

        nnr_train_tpz_features, nnr_validate_tpz_features = np.split(tpz_features[nnr_indicator_trans], [int(nn_train_val_split*np.sum(nnr_indicator_trans))])
        nnr_test_tpz_features = tpz_features[nnc_indicator_trans]

        nnr_train_features_lo = nnr_train_features - nnr_train_feature_errs
        nnr_validate_features_lo = nnr_validate_features - nnr_validate_feature_errs
        nnr_test_features_lo = nnr_test_features - nnr_test_feature_errs
        nnr_train_features_hi = nnr_train_features + nnr_train_feature_errs
        nnr_validate_features_hi = nnr_validate_features + nnr_validate_feature_errs
        nnr_test_features_hi = nnr_test_features + nnr_test_feature_errs
        nnr_train_features_frac = nnr_train_feature_errs/nnr_train_features
        nnr_validate_features_frac = nnr_validate_feature_errs/nnr_validate_features
        nnr_test_features_frac = nnr_test_feature_errs/nnr_test_features

        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(directory + filename + '.nnr_train', np.hstack((nnr_train_specz.reshape(-1,1), nnr_train_features, nnr_train_tpz_features, nnr_train_feature_errs, nnr_train_features_lo, nnr_train_features_hi, nnr_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnr_validate', np.hstack((nnr_validate_specz.reshape(-1,1), nnr_validate_features, nnr_validate_tpz_features, nnr_validate_feature_errs, nnr_validate_features_lo, nnr_validate_features_hi, nnr_validate_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnr_test', np.hstack((nnr_test_specz.reshape(-1,1), nnr_test_features, nnr_test_tpz_features, nnr_test_feature_errs, nnr_test_features_lo, nnr_test_features_hi, nnr_test_features_frac)), comments = '', header = header, fmt = fmt)

        generate_input_files_cosmos(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, closest_match_test, cosmos2015_train, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)



    elif (os.path.isfile(directory + 'output/results/' + filename + '.0.mlz') and not 
            (os.path.isfile(directory + filename + '.nnc_train') or 
                os.path.isfile(directory + filename + '.nnc_test') or 
                os.path.isfile(directory + filename + '.nnc_validate'))):

        print('Generating fresh NNC input files in %s' % directory)
        
        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', dtype = int, unpack = True)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        nnr_test_inds = train_inds[nnc_indicator]

        # nnr_test_specz, nnr_test_features, nnr_test_feature_errs = get_features(nnr_test_inds, inputfp = inputfp)
        nnr_test_specz = read_csv(directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#', usecols = [0]).to_numpy()
        nnr_test_features = read_csv(directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#', usecols = range(1,13)).to_numpy()
        nnr_test_feature_errs = read_csv(directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#', usecols = range(16,28)).to_numpy()
        nnr_test_tpz_features = read_csv(directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#')[['zphot', 'zconf', 'zerr']].to_numpy()

        translation_inds = np.arange(len(nnr_test_inds))
        np.random.shuffle(translation_inds)

        nnc_train_inds, nnc_validate_inds = np.split(nnr_test_inds[translation_inds], {int(nn_train_val_split*len(nnr_test_inds))})
        train_trans, val_trans = np.split(translation_inds, [int(nn_train_val_split*len(nnr_test_inds))])

        np.savetxt(directory + filename + '.nnc_train_inds', nnc_train_inds, fmt = '%i')
        np.savetxt(directory + filename + '.nnc_val_inds', nnc_validate_inds, fmt = '%i')

        np.savetxt(directory + filename + '.nnc_train_inds', np.vstack((nnc_train_inds, train_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')
        np.savetxt(directory + filename + '.nnc_validate_inds', np.vstack((nnc_validate_inds, val_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')

        nnc_train_features = nnr_test_features[train_trans]
        nnc_validate_features = nnr_test_features[val_trans]
        
        nnc_train_feature_errs = nnr_test_feature_errs[train_trans]
        nnc_validate_feature_errs = nnr_test_feature_errs[val_trans]
        
        nnc_train_specz = nnr_test_specz[train_trans]
        nnc_validate_specz = nnr_test_specz[val_trans]
        
        nnc_train_tpz_features = nnr_test_tpz_features[train_trans]
        nnc_validate_tpz_features = nnr_test_tpz_features[val_trans]
        
        nnc_train_features_lo = nnc_train_features - nnc_train_feature_errs
        nnc_validate_features_lo = nnc_validate_features - nnc_validate_feature_errs
        nnc_train_features_hi = nnc_train_features + nnc_train_feature_errs
        nnc_validate_features_hi = nnc_validate_features + nnc_validate_feature_errs
        nnc_train_features_frac = nnc_train_feature_errs/nnc_train_features
        nnc_validate_features_frac = nnc_validate_feature_errs/nnc_validate_features
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(directory + filename + '.nnc_train', np.hstack((nnc_train_specz.reshape(-1,1), nnc_train_features, nnc_train_tpz_features, nnc_train_feature_errs, nnc_train_features_lo, nnc_train_features_hi, nnc_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnc_validate', np.hstack((nnc_validate_specz.reshape(-1,1), nnc_validate_features, nnc_validate_tpz_features, nnc_validate_feature_errs, nnc_validate_features_lo, nnc_validate_features_hi, nnc_validate_features_frac)), comments = '', header = header, fmt = fmt)
        # np.savetxt(directory + filename + '.nnc_test', np.hstack((nnc_test_specz.reshape(-1,1), nnc_test_features, nnc_test_tpz_features, nnc_test_feature_errs, nnc_test_features_lo, nnc_test_features_hi, nnc_test_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files_cosmos(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, closest_match_test, cosmos2015_train, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not os.path.isfile(directory + filename + '.tpz_app'):

        print('Generating TPZ application set files...')

        if closest_match_test:
            app_inds = np.loadtxt(directory + filename + '.app_inds', dtype = int)
            indices = [[0], range(1,13), range(13,25)]
            cm_specz, cm_features, cm_feature_errs = [read_csv(directory + filename + '.closest_matches', delim_whitespace = True, usecols = thisindex).values for thisindex in indices]
            app_specz, app_features, app_feature_errs = (cm_specz[app_inds], cm_features[app_inds], cm_feature_errs[app_inds])

        elif cosmos2015_train:
            cosmos_specz, cosmos_features, cosmos_feature_errs = get_cosmos_features()
            app_inds = np.loadtxt(directory + filename + '.app_inds', dtype = int)
            app_specz = cosmos_specz[app_inds]
            app_features = cosmos_features[app_inds]
            app_feature_errs = cosmos_feature_errs[app_inds]
        else:

            app_specz, app_features, app_feature_errs = get_cosmos_features()

        header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

        np.savetxt(directory + filename + '.tpz_app', np.hstack((app_specz.reshape(-1,1), app_features, app_feature_errs)), header = header, fmt = fmt)
        generate_input_files_cosmos(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, closest_match_test, cosmos2015_train, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not os.path.isfile(directory + filename + '.nn_app') and os.path.isfile(directory + 'output/results/' + filename + '.1.mlz'):

        print('Generating NN application set files...')

        indices = [[0], range(1,13), range(13,25)]
        app_specz, app_features, app_feature_errs = [np.loadtxt(directory + filename + '.tpz_app', usecols = thisindex) for thisindex in indices]

        app_tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.1.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        app_features_lo = app_features - app_feature_errs
        app_features_hi = app_features + app_feature_errs
        app_features_frac = app_feature_errs/app_features
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(directory + filename + '.nn_app', np.hstack((app_specz.reshape(-1,1), app_features, app_tpz_features, app_feature_errs, app_features_lo, app_features_hi, app_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files_cosmos(directory, filename, inputfp, tpz_nnr_nnc_split, nn_train_val_split, closest_match_split, closest_match_test, cosmos2015_train, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not subrun:
        print('All of these files already exist.  Please delete the input files before continuing.')



def get_cosmos_features(sigmacut = 0.1):

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

    photz = cosmos.ZPDF[unique_idx] # ZPDF for median of the PDF, ZMINCHI2 for minimum chi2 (-99 for <3 bands)
    photz_sigma = (cosmos.ZPDF_H68-cosmos.ZPDF_L68)[unique_idx] # ZPDF_H68-ZPDF_L68 for ZPDF; CHI2BEST for ZMINCHI2

    good_inds = (photz >= 0) & (photz_sigma < sigmacut) & (d2d.to('arcsecond').value[unique_idx] < 1) & (cosmos.ip_MAG_APER2[unique_idx] < 25.) & (cosmos.ZP_2[unique_idx] < 0) & (cosmos.NBFILT[unique_idx] >= 25)

    photz = photz[good_inds]
    phot_idx = phot_idx[good_inds]

    mags = np.vstack((phot['g_cmodel_mag'], phot['r_cmodel_mag'], phot['i_cmodel_mag'], phot['z_cmodel_mag'], phot['y_cmodel_mag'])).T[phot_idx]
    mag_errs = np.vstack((phot['g_cmodel_magsigma'], phot['r_cmodel_magsigma'], phot['i_cmodel_magsigma'], phot['z_cmodel_magsigma'], phot['y_cmodel_magsigma'])).T[phot_idx]

    g, r, i, z, y = mags.T
    g_err, r_err, i_err, z_err, y_err = mag_errs.T

    colors = np.array([g-r, r-i, i-z, z-y]).T
    color_errs = np.array([g_err+r_err, r_err+i_err, i_err+z_err, z_err+y_err]).T

    triplets = np.array([(g-r) - (r-i), (r-i) - (i-z), (i-z) - (z-y)]).T
    triplet_errs = np.array([(g_err+r_err) + (r_err+i_err), (r_err+i_err) + (i_err+z_err), (i_err+z_err) + (z_err+y_err)]).T

    features = np.hstack((mags, colors, triplets))
    feature_errs = np.hstack((mag_errs, color_errs, triplet_errs))

    return photz, features, feature_errs





def process():

    generate_input_files('./tpzruns/SpecSpec/')
    # generate_input_files('./tpzruns/ShiftedSpecSpec/', ishift = 20.65-23.92, gzshift = 1.81-1.23)
    generate_input_files('./tpzruns/SpecMatch/', closest_match_split = True)
    generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_TrainMod/001/', closest_match_split = True, train_frac_lim = .01)
    generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_TrainMod/05/', closest_match_split = True, train_frac_lim = .05)
    generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_TrainMod/010/', closest_match_split = True, train_frac_lim = .10)
    generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_TrainMod/020/', closest_match_split = True, train_frac_lim = .20)
    generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_TrainMod/040/', closest_match_split = True, train_frac_lim = .40)
    generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_TrainMod/060/', closest_match_split = True, train_frac_lim = .60)
    generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_TrainMod/080/', closest_match_split = True, train_frac_lim = .80)
    generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015/', closest_match_split = True)
    os.system('ln -sf /home/adam/Research/DESC_EELG/tpzruns/MatchCOSMOS2015/tpzrun.closest_matches ./tpzruns/MatchCOSMOS2015_nnr/')
    os.system('ln -sf /home/adam/Research/DESC_EELG/tpzruns/MatchCOSMOS2015/tpzrun.closest_matches ./tpzruns/MatchCOSMOS2015_notrip/')
    os.system('ln -sf /home/adam/Research/DESC_EELG/tpzruns/MatchCOSMOS2015/tpzrun.closest_matches ./tpzruns/MatchCOSMOS2015_004/')
    os.system('ln -sf /home/adam/Research/DESC_EELG/tpzruns/MatchCOSMOS2015/tpzrun.closest_matches ./tpzruns/MatchCOSMOS2015_005/')
    os.system('ln -sf /home/adam/Research/DESC_EELG/tpzruns/MatchCOSMOS2015/tpzrun.closest_matches ./tpzruns/MatchCOSMOS2015_006/')
    os.system('ln -sf /home/adam/Research/DESC_EELG/tpzruns/MatchCOSMOS2015/tpzrun.closest_matches ./tpzruns/MatchCOSMOS2015_008/')
    os.system('ln -sf /home/adam/Research/DESC_EELG/tpzruns/MatchCOSMOS2015/tpzrun.closest_matches ./tpzruns/MatchCOSMOS2015_009/')
    os.system('ln -sf /home/adam/Research/DESC_EELG/tpzruns/MatchCOSMOS2015/tpzrun.closest_matches ./tpzruns/MatchCOSMOS2015_010/')
    generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_nnr/', closest_match_split = True, no_nnr = False)
    generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_notrip/', closest_match_split = True)
    # generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_004/', closest_match_split = True)
    # generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_005/', closest_match_split = True)
    # generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_006/', closest_match_split = True)
    # generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_008/', closest_match_split = True)
    # generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_009/', closest_match_split = True)
    # generate_input_files_cosmos('./tpzruns/MatchCOSMOS2015_010/', closest_match_split = True)
    generate_input_files_cosmos('./tpzruns/MatchMatch/', closest_match_split = True, closest_match_test = True)

    # cd ./SpecSpec/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../SpecMatch/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../MatchMatch/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../MatchCOSMOS2015_TrainMod/001/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../005/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../010/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../020/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../040/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../060/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../080/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../
    # cd ../MatchCOSMOS2015/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../MatchCOSMOS2015_nnr/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../MatchCOSMOS2015_notrip/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../MatchCOSMOS2015_04/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../MatchCOSMOS2015_05/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../MatchCOSMOS2015_06/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../MatchCOSMOS2015_07/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../MatchCOSMOS2015_08/
    # mpiexec -n 10 runMLZ tpzrun.inputs
    # cd ../




    # cd ./SpecSpec/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../SpecMatch/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../MatchMatch/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../MatchCOSMOS2015_TrainMod/001/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../005/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../010/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../020/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../040/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../060/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../080/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../
    # cd ../MatchCOSMOS2015/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../MatchCOSMOS2015_nnr/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../MatchCOSMOS2015_notrip/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../MatchCOSMOS2015_04/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../MatchCOSMOS2015_05/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../MatchCOSMOS2015_06/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../MatchCOSMOS2015_08/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../MatchCOSMOS2015_09/
    # mpiexec -n 10 runMLZ --modify testfile=tpzrun.tpz_app --no_train tpzrun.inputs 
    # cd ../


    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/SpecSpec/')
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/SpecMatch/')
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchMatch/')
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_TrainMod/001/')
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_TrainMod/005/')
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_TrainMod/010/')
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_TrainMod/020/')
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_TrainMod/040/')
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_TrainMod/060/')
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_TrainMod/080/')
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015/')
    # nnc.fit_app()
    # # nnr = nnclass.nn_regressor(run_folder = './tpzruns/MatchCOSMOS2015_nnr/')
    # # nnr.fit_app()
    # # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_nnr/', correct_pz = True)
    # # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_notrip/', phot_feature_list = ['i', 'gr', 'ri', 'iz', 'zy'])
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_004/', acc_cutoff = 0.04)
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_005/', acc_cutoff = 0.05)
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_006/', acc_cutoff = 0.06)
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_008/', acc_cutoff = 0.08)
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_009/', acc_cutoff = 0.09)
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_010/', acc_cutoff = 0.10)
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_011/', acc_cutoff = 0.11)
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_012/', acc_cutoff = 0.12)
    # nnc.fit_app()
    # nnc = nnclass.nn_classifier(run_folder = './tpzruns/MatchCOSMOS2015_013/', acc_cutoff = 0.13)
    # nnc.fit_app()


