import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
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


goodss_bands = ['VIMOS_U', 'CTIO_U', 'IRAC_CH1', 'IRAC_CH2', 'IRAC_CH3', 'IRAC_CH4', 'WFC3_F160W', 'WFC3_F125W', 'ACS_F850LP', 'ACS_F606W', 'ACS_F775W', 'ACS_F814W', 'ACS_F435W', 'ISAAC_KS']





def combine_cats(directory = './tpzruns/uvcandels/', outfile = 'uvcandels_combined.cat'):

    if os.path.isfile(directory + outfile):
        os.remove(directory + outfile)

    catalogs = [read_csv(thiscat, delim_whitespace = True) for thiscat in sorted(glob(directory + '*.cat'))]
    combined_catalog = pd.concat(catalogs, ignore_index = True, sort = False)
    catalog_names = [thiscat.split('_')[-1].split('.')[0].upper() for thiscat in sorted(glob(directory + '*.cat'))]

    catalog_flag = []

    for this_cat_name, thiscat in zip(catalog_names, catalogs):

        catalog_flag += [this_cat_name,]*len(thiscat)

    combined_catalog.insert(1, 'field', catalog_flag)

    combined_catalog.to_csv(directory + outfile, index = False, index_label = False, sep = ' ', na_rep = 'NaN')
    







def generate_input_files(directory = './tpzruns/uvcandels/', filename = 'tpzrun', inputfp = ['uvcandels_goodss.cat'], band_names = goodss_bands, tpz_nnr_nnc_split = np.array([0.333, 0.666]), nn_train_val_split = 0.85, closest_match_split = False, closest_match_test = False, cosmos2015_train = False, overall_split = 0.8, ishift = 0., gzshift = 0., gz_colorlim = None, i_maglim = None, no_nnr = True, train_frac_lim = 1., subrun = False):


    if no_nnr and all(tpz_nnr_nnc_split == np.array([0.333, 0.666])):
        tpz_nnr_nnc_split = np.array([0.5, 0.5])


    if not os.path.isfile(directory + filename + '.train_inds') and not os.path.isfile(directory + filename + '.test_inds'):

        print('Generating fresh indexing files in %s' % directory)

        specz, features, feature_errs, feature_names = get_features([directory + thisfile for thisfile in inputfp], band_names)

        indices = np.arange(len(specz))
        new_indices = np.copy(indices)
        np.random.shuffle(new_indices)
        split_index = int(overall_split*len(new_indices))
        train_inds = np.sort(new_indices[:split_index])
        test_inds = np.sort(new_indices[split_index:])

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
        np.savetxt(directory + filename + '.app_inds', test_inds, header = 'test_inds', fmt = '%i')

        generate_input_files(directory, filename, inputfp, band_names, tpz_nnr_nnc_split, nn_train_val_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not os.path.isfile(directory + filename + '.tpz_train') and not os.path.isfile(directory + filename + '.tpz_test'):

        print('Generating fresh TPZ input files in %s' % directory)

        train_inds, tpz_indicator, nnr_indicator, nnc_indicator = np.loadtxt(directory + filename + '.train_inds', unpack = True, dtype = int)

        tpz_indicator = tpz_indicator.astype(bool)
        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        tpz_train_inds = train_inds[tpz_indicator]
        tpz_test_inds = train_inds[nnr_indicator | nnc_indicator]

        specz, features, feature_errs, feature_names = get_features([directory + thisfile for thisfile in inputfp], band_names)

        train_specz = specz[tpz_train_inds]
        train_features = features[tpz_train_inds]
        train_feature_errs = feature_errs[tpz_train_inds]

        test_specz = specz[tpz_test_inds]
        test_features = features[tpz_test_inds]
        test_feature_errs = feature_errs[tpz_test_inds]

        header = 'specz ' + ' '.join(feature_names) + ' ' + ' '.join(['e_' + thisfeature for thisfeature in feature_names])
        fmt = ' %.10f' + ' %.5f'*features.shape[1] + ' %.5e' * feature_errs.shape[1]

        np.savetxt(directory + filename + '.tpz_train', np.hstack((train_specz.reshape(-1,1), train_features, train_feature_errs)), header = header, fmt = fmt)
        np.savetxt(directory + filename + '.tpz_test', np.hstack((test_specz.reshape(-1,1), test_features, test_feature_errs)), header = header, fmt = fmt)

        generate_input_files(directory, filename, inputfp, band_names, tpz_nnr_nnc_split, nn_train_val_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    
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

        specz, features, feature_errs, feature_names = get_features([directory + thisfile for thisfile in inputfp], band_names)

        nnr_train_specz = specz[nnr_train_inds]
        nnr_train_features = features[nnr_train_inds]
        nnr_train_feature_errs = feature_errs[nnr_train_inds]

        nnr_test_specz = specz[nnr_test_inds]
        nnr_test_features = features[nnr_test_inds]
        nnr_test_feature_errs = feature_errs[nnr_test_inds]

        nnr_validate_specz = specz[nnr_validate_inds]
        nnr_validate_features = features[nnr_validate_inds]
        nnr_validate_feature_errs = feature_errs[nnr_validate_inds]

        tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.0.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        nnr_indicator_trans = nnr_indicator[~tpz_indicator]
        nnc_indicator_trans = nnc_indicator[~tpz_indicator]

        nnr_train_tpz_features, nnr_validate_tpz_features = np.split(tpz_features[nnr_indicator_trans], [int(nn_train_val_split*np.sum(nnr_indicator_trans))])
        nnr_test_tpz_features = tpz_features[nnc_indicator_trans]

        # nnr_train_features_lo = nnr_train_features - nnr_train_feature_errs
        # nnr_validate_features_lo = nnr_validate_features - nnr_validate_feature_errs
        # nnr_test_features_lo = nnr_test_features - nnr_test_feature_errs
        # nnr_train_features_hi = nnr_train_features + nnr_train_feature_errs
        # nnr_validate_features_hi = nnr_validate_features + nnr_validate_feature_errs
        # nnr_test_features_hi = nnr_test_features + nnr_test_feature_errs
        # nnr_train_features_frac = nnr_train_feature_errs/nnr_train_features
        # nnr_validate_features_frac = nnr_validate_feature_errs/nnr_validate_features
        # nnr_test_features_frac = nnr_test_feature_errs/nnr_test_features

        header = 'specz ' + ' '.join(feature_names) + ' zphot zconf zerr ' + ' '.join(['e_' + thisfeature for thisfeature in feature_names])
        fmt = ' %.10f' + ' %.5f'*features.shape[1] + ' %.5e'*3 + ' %.5e' * feature_errs.shape[1] 

        np.savetxt(directory + filename + '.nnr_train', np.hstack((nnr_train_specz.reshape(-1,1), nnr_train_features, nnr_train_tpz_features, nnr_train_feature_errs)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnr_validate', np.hstack((nnr_validate_specz.reshape(-1,1), nnr_validate_features, nnr_validate_tpz_features, nnr_validate_feature_errs)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnr_test', np.hstack((nnr_test_specz.reshape(-1,1), nnr_test_features, nnr_test_tpz_features, nnr_test_feature_errs)), comments = '', header = header, fmt = fmt)

        generate_input_files(directory, filename, inputfp, band_names, tpz_nnr_nnc_split, nn_train_val_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)



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
        
        # nnc_train_features_lo = nnc_train_features - nnc_train_feature_errs
        # nnc_validate_features_lo = nnc_validate_features - nnc_validate_feature_errs
        # nnc_train_features_hi = nnc_train_features + nnc_train_feature_errs
        # nnc_validate_features_hi = nnc_validate_features + nnc_validate_feature_errs
        # nnc_train_features_frac = nnc_train_feature_errs/nnc_train_features
        # nnc_validate_features_frac = nnc_validate_feature_errs/nnc_validate_features
        
        header = 'specz ' + ' '.join(band_names) + ' zphot zconf zerr ' + ' '.join(['e_' + thisfeature for thisfeature in band_names])
        fmt = ' %.10f' + ' %.5f'*nnc_train_features.shape[1] + ' %.5e'*3 + ' %.5e' * nnc_train_feature_errs.shape[1] 

        np.savetxt(directory + filename + '.nnc_train', np.hstack((nnc_train_specz.reshape(-1,1), nnc_train_features, nnc_train_tpz_features, nnc_train_feature_errs)), comments = '', header = header, fmt = fmt)
        np.savetxt(directory + filename + '.nnc_validate', np.hstack((nnc_validate_specz.reshape(-1,1), nnc_validate_features, nnc_validate_tpz_features, nnc_validate_feature_errs)), comments = '', header = header, fmt = fmt)
        # np.savetxt(directory + filename + '.nnc_test', np.hstack((nnc_test_specz.reshape(-1,1), nnc_test_features, nnc_test_tpz_features, nnc_test_feature_errs, nnc_test_features_lo, nnc_test_features_hi, nnc_test_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, band_names, tpz_nnr_nnc_split, nn_train_val_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not os.path.isfile(directory + filename + '.tpz_app'):

        print('Generating TPZ application set files...')

        specz, features, feature_errs, feature_names = get_features([directory + thisfile for thisfile in inputfp], band_names)
        app_inds = np.loadtxt(directory + filename + '.app_inds', dtype = int)
        app_specz = specz[app_inds]
        app_features = features[app_inds]
        app_feature_errs = feature_errs[app_inds]

        header = 'specz ' + ' '.join(feature_names) + ' ' + ' '.join(['e_' + thisfeature for thisfeature in feature_names])
        fmt = ' %.10f' + ' %.5f'*features.shape[1] + ' %.5e' * feature_errs.shape[1]

        np.savetxt(directory + filename + '.tpz_app', np.hstack((app_specz.reshape(-1,1), app_features, app_feature_errs)), header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, band_names, tpz_nnr_nnc_split, nn_train_val_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not os.path.isfile(directory + filename + '.nn_app') and os.path.isfile(directory + 'output/results/' + filename + '.1.mlz'):

        print('Generating NN application set files...')

        indices = [[0], range(1,13), range(13,25)]
        app_specz, app_features, app_feature_errs = [np.loadtxt(directory + filename + '.tpz_app', usecols = thisindex) for thisindex in indices]

        app_tpz_features = np.loadtxt(directory + 'output/results/' + filename + '.1.mlz', usecols = [2,4,6]) # 1,3,5 corresponds to zmode; 2,4,6 corresponds to zmean; is z, zconf, zerr

        app_features_lo = app_features - app_feature_errs
        app_features_hi = app_features + app_feature_errs
        app_features_frac = app_feature_errs/app_features
        
        header = 'specz ' + ' '.join(band_names) + ' zphot zconf zerr ' + ' '.join(['e_' + thisfeature for thisfeature in band_names])
        fmt = ' %.10f' + ' %.5f'*app_features.shape[1] + ' %.5e'*3 + ' %.5e' * app_feature_errs.shape[1] 

        np.savetxt(directory + filename + '.nn_app', np.hstack((app_specz.reshape(-1,1), app_features, app_tpz_features, app_feature_errs)), comments = '', header = header, fmt = fmt)
        generate_input_files(directory, filename, inputfp, band_names, tpz_nnr_nnc_split, nn_train_val_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, no_nnr, train_frac_lim, subrun = True)

    elif not subrun:
        print('All of these files already exist.  Please delete the input files before continuing.')




def get_features(inputfp, feature_names, spec_flag = 2):

    all_data = pd.concat([read_csv(thisfile, delim_whitespace = True) for thisfile in inputfp], ignore_index=True)

    fluxes = all_data[[thisfeature + '_FLUX' for thisfeature in feature_names]]
    flux_errs = all_data[[thisfeature + '_FLUXERR' for thisfeature in feature_names]]
    
    inds = (all_data.Z_S > 0) & (all_data.Z_S_FLAG >= spec_flag) & (np.all(fluxes > 0, axis = 1)) & (np.all(np.isfinite(fluxes), axis = 1))

    all_data = all_data[inds]
    fluxes = fluxes[inds].to_numpy()
    flux_errs = flux_errs[inds].to_numpy()

    mags = -2.5*np.log10(fluxes) + 23.9
    mag_errs = (-2.5*np.log10(fluxes+flux_errs) + 23.9) - (-2.5*np.log10(fluxes - flux_errs) + 23.9)

    specz = all_data.Z_S

    # g, r, i, z, y = mags.T
    # g_err, r_err, i_err, z_err, y_err = mag_errs.T

    # colors = np.array([g-r, r-i, i-z, z-y]).T
    # color_errs = np.array([g_err+r_err, r_err+i_err, i_err+z_err, z_err+y_err]).T

    # triplets = np.array([(g-r) - (r-i), (r-i) - (i-z), (i-z) - (z-y)]).T
    # triplet_errs = np.array([(g_err+r_err) + (r_err+i_err), (r_err+i_err) + (i_err+z_err), (i_err+z_err) + (z_err+y_err)]).T

    # features = np.hstack((mags, colors, triplets))
    # feature_errs = np.hstack((mag_errs, color_errs, triplet_errs))

    # return specz, features, feature_errs

    return specz.values, mags, mag_errs, feature_names



def analyze_bands(inputfp = './tpzruns/uvcandels/uvcandels_goodsn.cat', specz_only = False):

    data = read_csv(inputfp, delimiter = '\s+')

    if specz_only:
        data = data[data.Z_S > 0]

    fluxes = data[[thisfeature for thisfeature in data.columns if thisfeature[-5:] == '_FLUX']]

    flux_names = fluxes.columns

    flux_order = []
    total_frac = []

    print(f'Total number of galaxies: {len(fluxes)}\n\n')

    for x in range(len(flux_names)):

        fracs = []

        for thiscolumnname in flux_names:
            if thiscolumnname in flux_order:
                fracs.append(-1)
            else:
                this_set = fluxes[flux_order + [thiscolumnname]]
                if this_set.shape[1] > 1:
                    fracs.append(np.sum(np.all(this_set > -90, axis = 1))/float(len(this_set)))
                else:
                    try:
                        fracs.append(np.sum(this_set > -90)/float(len(this_set)))
                    except:
                        breakpoint()


        thisband = flux_names[np.argmax(fracs)]
        thisfrac = fracs[np.argmax(fracs)]
        flux_order.append(thisband)
        total_frac.append(thisfrac)

        if len(total_frac) == 1:
            print(((thisband + ':').ljust(25) + '%.3f' % thisfrac) + ''.rjust(8) + (('%i' % int(len(fluxes) * thisfrac)).rjust(8)))
        else:
            print((thisband + ':').ljust(25) + ('%.3f' % thisfrac) + ('%.3f' % (total_frac[-1] - total_frac[-2])).rjust(8) + (('%i' % int(len(fluxes) * thisfrac)).rjust(8)))
