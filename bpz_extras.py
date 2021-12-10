import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import filtersim
import photosim
from pandas import read_csv
from tqdm import tqdm
import shelve
from glob import glob
from matplotlib import patheffects
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u


phot = photosim.photosim()
odin_phot = photosim.photosim(odin_filters = True)
filters = filtersim.filtersim()

BPZ_dir = '/home/adam/Research/pz_pdf/pz/BPZ/'
stage1dir = 'stage1/'
stage2dir = 'stage2/'
stage3dir = 'stage3/'


current_stagedir = stage3dir



def runbpz(directory = current_stagedir, infile = 'templatecat.bpz_train', logfile = False, interp = 0, spectra = 'adam_spec_dusty.list', sed_dir = BPZ_dir + 'SED/adam_templates/adamtemp/', probs = 'no', prior = 'hdfn_gen_adamtemp', madau = 'yes', new_ab = 'no', zmax = 4.005, odds = 0.68, use_old_z = False):
# def runbpz(directory = 'HSC_train_cosmos', infile = 'templatecat.bpz_train', logfile = False, interp = 2, spectra = 'CWWSB4.list', sed_dir = BPZ_dir + 'SED/', probs = 'no', prior = 'hdfn_gen', madau = 'yes', new_ab = 'no', zmax = 4.005, odds = 0.68):

    directory = BPZ_dir + directory

    bpzfp = directory+infile[:-3]+'bpz'

    if os.path.isfile(bpzfp):
        bpz_file = open(bpzfp, 'r')
        all_lines = bpz_file.readlines()
        bpz_file.close()

        old_madau = [s for s in all_lines if 'MADAU' in s][0].split('=')[-1][:-1]

        if old_madau != madau:
            # ASK IF YOU WOULD LIKE TO REGENERATE AB FILES
            new_ab = 'yes'

    if use_old_z:
        output = 'modified.bpz'
        command = 'python3 ' + BPZ_dir + 'bpz_tcorr_fits_py3.py {} -INTERP {} -SPECTRA {} -PROBS {} -PRIOR {} -MADAU {} -NEW_AB {} -ZMAX {} -ODDS {} -SED_DIR {} -OUTPUT {} -GET_Z {}'.format(infile, interp, spectra, probs, prior, madau, new_ab, zmax, odds, sed_dir, output, 'no')
    else:
        command = 'python3 ' + BPZ_dir + 'bpz_tcorr_fits_py3.py {} -INTERP {} -SPECTRA {} -PROBS {} -PRIOR {} -MADAU {} -NEW_AB {} -ZMAX {} -ODDS {} -SED_DIR {}'.format(infile, interp, spectra, probs, prior, madau, new_ab, zmax, odds, sed_dir)

    current_directory = os.getcwd()
    os.chdir(directory)
    if logfile:
        command += ' > ./log.txt'
    
    os.system(command)
    os.chdir(current_directory)




def hsc_convert(inputfp = 'HSC/HSC_wide_clean.fits', outputfp = 'HSC/', size = None, unique = True, maglim = None, sn_lim = 10., zlim = None):

    if outputfp[-1] != '/':
        outputfp += '/'

    outputfp = BPZ_dir + outputfp

    if 'phot_depth' in inputfp:
        hscfile = fits.open(inputfp)[1].data
        ID = hscfile.object_id
        index = np.arange(len(ID))
        pz = np.ones(len(ID))*-1
        sz = pz
        mags = np.vstack((hscfile.g_cmodel_mag, hscfile.r_cmodel_mag, hscfile.i_cmodel_mag, hscfile.z_cmodel_mag, hscfile.y_cmodel_mag)).T
        errs = np.vstack((hscfile.g_cmodel_magsigma, hscfile.r_cmodel_magsigma, hscfile.i_cmodel_magsigma, hscfile.z_cmodel_magsigma, hscfile.y_cmodel_magsigma)).T

        good_inds = np.all(np.isfinite(mags) & np.isfinite(errs), axis = 1)
        ID = ID[good_inds]
        index = index[good_inds]
        pz = pz[good_inds]
        sz = sz[good_inds]
        mags = mags[good_inds]
        errs = errs[good_inds]

        if sn_lim:

            fluxes = np.vstack((hscfile.g_cmodel_flux, hscfile.r_cmodel_flux, hscfile.i_cmodel_flux, hscfile.z_cmodel_flux, hscfile.y_cmodel_flux)).T[index]
            flux_errs = np.vstack((hscfile.g_cmodel_fluxsigma, hscfile.r_cmodel_fluxsigma, hscfile.i_cmodel_fluxsigma, hscfile.z_cmodel_fluxsigma, hscfile.y_cmodel_fluxsigma)).T[index]

            total_sn = np.sqrt(np.sum(fluxes**2/flux_errs**2, axis = 1))
            lim_indices = total_sn > sn_lim

            ID = ID[lim_indices]
            index = index[lim_indices]
            pz = pz[lim_indices]
            sz = sz[lim_indices]
            mags = mags[lim_indices]
            errs = errs[lim_indices]
            mags = mags.T
            errs = errs.T


    else:
        index, ID, pz, sz, mags, errs = hsc_select(inputfp, size, unique, maglim, sn_lim, zlim)

    # np.savetxt(outputfp + 'index.dat', np.vstack((ID, pz, sz, index)).T, fmt = '  %i  %.3f  %.3f  %i', header = 'id, photoz, specz, hsc_index')

    # Save to file

    with open(outputfp + 'index.dat', 'w') as writefile:

        writefile.write('#  id  photoz  specz  hsc_index\n')

        for thisentry in zip(ID, pz, sz, index):

            writefile.write('  %i  %.3f  %.3f  %i\n' % thisentry)

    with open(outputfp + 'templatecat.cat', 'w') as writefile:

        writefile.write('#  id  f_g  e_g  f_r  e_r  f_i  e_i  f_z  e_z  f_y  e_y\n')

        for thisentry in zip(ID, mags[0], errs[0], mags[1], errs[1], mags[2], errs[2], mags[3], errs[3], mags[4], errs[4]):

            writefile.write(('  %i' + '  %.5f'*10 + '\n') % thisentry)


    # writetable = np.vstack((ID, mags[0], errs[0], mags[1], errs[1], mags[2], errs[2], mags[3], errs[3], mags[4], errs[4])).T
    # np.savetxt(outputfp + 'templatecat.cat', writetable, fmt = '  %i' + '  %.5f'*10, header = '  id  f_g  e_g  f_r  e_r  f_i  e_i  f_z  e_z  f_y  e_y')


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



def closest_matches(indices, specfile, photfile = './HSC/HSC_wide_clean_pdr2_phot_depth_1M.fits', flux_units = 'linear', sn_lim = 10.):
    
    # app_specz, app_features_app_feature_errs = closest_matches(indices, inputfp)

    data_spec = fits.open(specfile)[1].data[indices]
    data_phot = fits.open(photfile)[1].data

    # print('Remove this later')

    good_phot_inds = np.where(np.isfinite(data_phot.g_cmodel_flux) & np.isfinite(data_phot.r_cmodel_flux) & np.isfinite(data_phot.i_cmodel_flux) & np.isfinite(data_phot.z_cmodel_flux) & np.isfinite(data_phot.y_cmodel_flux) & 
        np.isfinite(data_phot.g_cmodel_fluxsigma) & np.isfinite(data_phot.r_cmodel_fluxsigma) & np.isfinite(data_phot.i_cmodel_fluxsigma) & np.isfinite(data_phot.z_cmodel_fluxsigma) & np.isfinite(data_phot.y_cmodel_fluxsigma) & 
        np.isfinite(data_phot.g_cmodel_mag) & np.isfinite(data_phot.r_cmodel_mag) & np.isfinite(data_phot.i_cmodel_mag) & np.isfinite(data_phot.z_cmodel_mag) & np.isfinite(data_phot.y_cmodel_mag) & 
        np.isfinite(data_phot.g_cmodel_magsigma) & np.isfinite(data_phot.r_cmodel_magsigma) & np.isfinite(data_phot.i_cmodel_magsigma) & np.isfinite(data_phot.z_cmodel_magsigma) & np.isfinite(data_phot.y_cmodel_magsigma))

    data_phot = data_phot[good_phot_inds]

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



def generate_input_files(train_directory = BPZ_dir + 'HSC_specz/', app_directory = BPZ_dir + 'HSC_photz/', filename = 'templatecat', inputfp = 'HSC/HSC_wide_clean_pdr2.fits', nnr_nnc_split = np.array([0.5]), nn_train_val_split = 0.85, closest_match_split = False, overall_split = 0.8, ishift = 0., gzshift = 0., gz_colorlim = None, i_maglim = None, subrun = False):


    if not os.path.isfile(train_directory + filename + '.train_inds'):

        print('Generating fresh indexing files in %s' % train_directory)

        indices = hsc_select(inputfp = inputfp, zlim = 1.5, sn_lim = 10., notzero = True)[0]

        if closest_match_split:
            train_inds = indices
        elif not (ishift == 0 and gzshift == 0):
            train_inds, test_inds = ratio_test_train_split(indices, ishift = ishift, gzshift = gzshift, specfile = inputfp)
        else:
            train_inds, test_inds = default_test_train_split(indices, i_maglim = i_maglim, gz_colorlim = gz_colorlim, splitfrac = overall_split)

        scrambled_train = np.arange(len(train_inds))
        np.random.shuffle(scrambled_train)

        nnr_indicator = np.zeros(len(train_inds), dtype = int)
        nnc_indicator = np.zeros(len(train_inds), dtype = int)

        nnr_split, nnc_split = np.split(scrambled_train, (nnr_nnc_split * len(scrambled_train)).astype(int))

        nnr_indicator[nnr_split] = 1
        nnc_indicator[nnc_split] = 1

        np.savetxt(train_directory + filename + '.train_inds', np.vstack((train_inds, nnr_indicator, nnc_indicator)).T, header = 'train_inds  nnr_train  nnc_train', fmt = '%i  %i  %i')
        if not closest_match_split:
            np.savetxt(app_directory + filename + '.app_inds', test_inds, header = 'test_inds', fmt = '%i')

        generate_input_files(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)


    elif not os.path.isfile(train_directory + filename + '.bpz_train'):

        print('Generating fresh BPZ input files in %s' % train_directory)

        train_inds, nnr_indicator, nnc_indicator = np.loadtxt(train_directory + filename + '.train_inds', unpack = True, dtype = int)

        train_specz, train_features, train_feature_errs = get_features(train_inds, inputfp = inputfp)

        header = 'index g e_g r e_r i e_i z e_z y e_y'
        fmt = ' %i' + ' %.5f'*24

        features_and_errs = np.empty((train_features.shape[0], train_features.shape[1] + train_feature_errs.shape[1]), dtype=float)
        features_and_errs[:,0::2] = train_features
        features_and_errs[:,1::2] = train_feature_errs

        np.savetxt(train_directory + filename + '.bpz_train', np.hstack((train_inds.reshape(-1,1), features_and_errs)), header = header, fmt = fmt)

        generate_input_files(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)

    
    elif (os.path.isfile(train_directory + filename + '.bpz') and not 
            (os.path.isfile(train_directory + filename + '.nnr_train') or 
                os.path.isfile(train_directory + filename + '.nnr_test') or 
                os.path.isfile(train_directory + filename + '.nnr_validate'))):

        print('Generating fresh NNR input files in %s' % train_directory)

        train_inds, nnr_indicator, nnc_indicator = np.loadtxt(train_directory + filename + '.train_inds', dtype = int, unpack = True)

        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        train_inds_scrambled = np.copy(train_inds[nnr_indicator])
        np.random.shuffle(train_inds_scrambled)

        nnr_train_inds, nnr_validate_inds = np.split(train_inds_scrambled, [int(nn_train_val_split*len(train_inds_scrambled))])
        nnr_train_inds.sort()
        nnr_validate_inds.sort()

        np.savetxt(train_directory + filename + '.nnr_train_inds', nnr_train_inds, fmt = '%i')
        np.savetxt(train_directory + filename + '.nnr_val_inds', nnr_validate_inds, fmt = '%i')

        nnr_test_inds = train_inds[nnc_indicator]

        nnr_train_specz, nnr_train_features, nnr_train_feature_errs = get_features(nnr_train_inds, inputfp = inputfp)
        nnr_validate_specz, nnr_validate_features, nnr_validate_feature_errs = get_features(nnr_validate_inds, inputfp = inputfp)
        nnr_test_specz, nnr_test_features, nnr_test_feature_errs = get_features(nnr_test_inds, inputfp = inputfp)

        temp = np.loadtxt(train_directory + filename + '.bpz', usecols = [1,2,3,5], unpack = True) # 1,2,3,5 corresponds to z_best, z_best_min, z_best_max, odds
        bpz_features = np.vstack((temp[0], temp[3], temp[2] - temp[1])).T

        # breakpoint()

        nnr_train_bpz_features, nnr_validate_bpz_features = np.split(bpz_features[nnr_indicator], [int(nn_train_val_split*np.sum(nnr_indicator))])
        nnr_test_bpz_features = bpz_features[nnc_indicator]

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

        np.savetxt(train_directory + filename + '.nnr_train', np.hstack((nnr_train_specz.reshape(-1,1), nnr_train_features, nnr_train_bpz_features, nnr_train_feature_errs, nnr_train_features_lo, nnr_train_features_hi, nnr_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(train_directory + filename + '.nnr_validate', np.hstack((nnr_validate_specz.reshape(-1,1), nnr_validate_features, nnr_validate_bpz_features, nnr_validate_feature_errs, nnr_validate_features_lo, nnr_validate_features_hi, nnr_validate_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(train_directory + filename + '.nnr_test', np.hstack((nnr_test_specz.reshape(-1,1), nnr_test_features, nnr_test_bpz_features, nnr_test_feature_errs, nnr_test_features_lo, nnr_test_features_hi, nnr_test_features_frac)), comments = '', header = header, fmt = fmt)

        generate_input_files(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)



    elif (os.path.isfile(train_directory + filename + '.bpz') and not 
            (os.path.isfile(train_directory + filename + '.nnc_train') or 
                os.path.isfile(train_directory + filename + '.nnc_test') or 
                os.path.isfile(train_directory + filename + '.nnc_validate'))):

        print('Generating fresh NNC input files in %s' % train_directory)
        
        train_inds, nnr_indicator, nnc_indicator = np.loadtxt(train_directory + filename + '.train_inds', dtype = int, unpack = True)

        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        nnr_test_inds = train_inds[nnc_indicator]
        nnr_test_specz, nnr_test_features, nnr_test_feature_errs = get_features(nnr_test_inds, inputfp = inputfp)
        nnr_test_bpz_features = read_csv(train_directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#')[['zphot', 'zconf', 'zerr']].to_numpy()

        translation_inds = np.arange(len(nnr_test_inds))
        np.random.shuffle(translation_inds)

        nnc_train_inds, nnc_validate_inds = np.split(nnr_test_inds[translation_inds], {int(nn_train_val_split*len(nnr_test_inds))})
        train_trans, val_trans = np.split(translation_inds, [int(nn_train_val_split*len(nnr_test_inds))])

        np.savetxt(train_directory + filename + '.nnc_train_inds', nnc_train_inds, fmt = '%i')
        np.savetxt(train_directory + filename + '.nnc_val_inds', nnc_validate_inds, fmt = '%i')

        np.savetxt(train_directory + filename + '.nnc_train_inds', np.vstack((nnc_train_inds, train_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')
        np.savetxt(train_directory + filename + '.nnc_validate_inds', np.vstack((nnc_validate_inds, val_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')

        nnc_train_features = nnr_test_features[train_trans]
        nnc_validate_features = nnr_test_features[val_trans]
        
        nnc_train_feature_errs = nnr_test_feature_errs[train_trans]
        nnc_validate_feature_errs = nnr_test_feature_errs[val_trans]
        
        nnc_train_specz = nnr_test_specz[train_trans]
        nnc_validate_specz = nnr_test_specz[val_trans]
        
        nnc_train_bpz_features = nnr_test_bpz_features[train_trans]
        nnc_validate_bpz_features = nnr_test_bpz_features[val_trans]
        
        nnc_train_features_lo = nnc_train_features - nnc_train_feature_errs
        nnc_validate_features_lo = nnc_validate_features - nnc_validate_feature_errs
        nnc_train_features_hi = nnc_train_features + nnc_train_feature_errs
        nnc_validate_features_hi = nnc_validate_features + nnc_validate_feature_errs
        nnc_train_features_frac = nnc_train_feature_errs/nnc_train_features
        nnc_validate_features_frac = nnc_validate_feature_errs/nnc_validate_features
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(train_directory + filename + '.nnc_train', np.hstack((nnc_train_specz.reshape(-1,1), nnc_train_features, nnc_train_bpz_features, nnc_train_feature_errs, nnc_train_features_lo, nnc_train_features_hi, nnc_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(train_directory + filename + '.nnc_validate', np.hstack((nnc_validate_specz.reshape(-1,1), nnc_validate_features, nnc_validate_bpz_features, nnc_validate_feature_errs, nnc_validate_features_lo, nnc_validate_features_hi, nnc_validate_features_frac)), comments = '', header = header, fmt = fmt)
        # np.savetxt(train_directory + filename + '.nnc_test', np.hstack((nnc_test_specz.reshape(-1,1), nnc_test_features, nnc_test_tpz_features, nnc_test_feature_errs, nnc_test_features_lo, nnc_test_features_hi, nnc_test_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)



    elif not os.path.isfile(app_directory + filename + '.bpz_app'):

        print('Generating BPZ application set files...')

        if closest_match_split:
            indices = hsc_select(inputfp = inputfp, zlim = 1.5, sn_lim = 10., notzero = True)[0]
            app_specz, app_features, app_feature_errs = closest_matches(indices, inputfp)
        else:
            app_inds = np.loadtxt(app_directory + filename + '.app_inds', dtype = int)
            app_specz, app_features, app_feature_errs = get_features(app_inds, inputfp = inputfp)

        header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

        header2 = 'index g e_g r e_r i e_i z e_z y e_y'
        fmt2 = ' %i' + ' %.5f'*10

        mags = app_features[:,:5]
        mag_errs = app_feature_errs[:,:5]

        mags_and_errs = np.empty((mags.shape[0], mags.shape[1] + mag_errs.shape[1]), dtype=float)
        mags_and_errs[:,0::2] = mags
        mags_and_errs[:,1::2] = mag_errs

        good_inds = np.all(np.isfinite(mags_and_errs), axis = 1)

        mags_and_errs = mags_and_errs[good_inds]
        app_specz = app_specz[good_inds]
        app_features = app_features[good_inds]
        app_feature_errs = app_feature_errs[good_inds]

        np.savetxt(app_directory + filename + '.app_info', np.hstack((app_specz.reshape(-1,1), app_features, app_feature_errs)), header = header, fmt = fmt)
        np.savetxt(app_directory + filename + '.bpz_app', np.hstack((np.arange(len(mags_and_errs)).reshape(-1,1), mags_and_errs)), header = header2, fmt = fmt2)
        generate_input_files(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)



    elif not os.path.isfile(train_directory + filename + '.nn_app') and os.path.isfile(app_directory + filename + '.bpz'):

        print('Generating NN application set files...')

        if closest_match_split:
            indices = [[0], range(1,13), range(13,25)]
            app_specz, app_features, app_feature_errs = [np.loadtxt(app_directory + filename + '.app_info', usecols = thisindex) for thisindex in indices]
        else:
            app_inds = np.loadtxt(app_directorydirectory + filename + '.app_inds', dtype = int)
            app_specz, app_features, app_feature_errs = get_features(app_inds, inputfp = inputfp)

        temp = np.loadtxt(app_directory + filename + '.bpz', usecols = [1,2,3,5], unpack = True) # 1,2,3,5 corresponds to z_best, z_best_min, z_best_max, odds
        bpz_features = np.vstack((temp[0], temp[3], temp[2] - temp[1])).T

        app_features_lo = app_features - app_feature_errs
        app_features_hi = app_features + app_feature_errs
        app_features_frac = app_feature_errs/app_features
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(train_directory + filename + '.nn_app', np.hstack((app_specz.reshape(-1,1), app_features, bpz_features, app_feature_errs, app_features_lo, app_features_hi, app_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)

    elif not subrun:
        print('All of these files already exist.  Please delete the input files before continuing.')



def generate_input_files_cosmos(train_directory = BPZ_dir + 'HSC_train_cosmos/', app_directory = BPZ_dir + 'HSC_app_cosmos/', filename = 'templatecat', inputfp = 'HSC/HSC_wide_clean_pdr2.fits', nnr_nnc_split = np.array([0.5]), nn_train_val_split = 0.85, closest_match_split = True, overall_split = 0.8, ishift = 0., gzshift = 0., gz_colorlim = None, i_maglim = None, subrun = False):


    if not os.path.isfile(train_directory + filename + '.train_inds'):

        print('Generating fresh indexing files in %s' % train_directory)

        indices = hsc_select(inputfp = inputfp, zlim = 1.5, sn_lim = 10., notzero = True)[0]

        if closest_match_split:
            if not os.path.isfile(train_directory + filename + '.closest_matches'):
                cm_specz, cm_features, cm_feature_errs = closest_matches(indices, inputfp)
            else:
                load_indices = [[0], range(1,13), range(13,25)]
                cm_specz, cm_features, cm_feature_errs = [np.loadtxt(train_directory + filename + '.closest_matches', usecols = thisindex) for thisindex in load_indices]

            header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
            fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12
            np.savetxt(train_directory + filename + '.closest_matches', np.hstack((cm_specz.reshape(-1,1), cm_features, cm_feature_errs)), header = header, fmt = fmt)
            # new_indices = np.copy(indices)
            # np.random.shuffle(new_indices)
            # split_index = int(overall_split*len(new_indices))
            # train_inds = new_indices[:split_index]
            # test_inds = new_indices[split_index:]
            train_inds = np.arange(len(cm_specz))
        elif not (ishift == 0 and gzshift == 0):
            # train_inds, test_inds = ratio_test_train_split(indices, ishift = ishift, gzshift = gzshift, savedir = directory, filename = filename, specfile = inputfp)
            train_inds= np.concatenate(ratio_test_train_split(indices, ishift = ishift, gzshift = gzshift, savedir = train_directory, filename = filename, specfile = inputfp))
        else:
            # train_inds, test_inds = default_test_train_split(indices, i_maglim = i_maglim, gz_colorlim = gz_colorlim, splitfrac = overall_split, specfile = inputfp)
            train_inds = np.concatenate(default_test_train_split(indices, i_maglim = i_maglim, gz_colorlim = gz_colorlim, splitfrac = overall_split, specfile = inputfp))

        scrambled_train = np.arange(len(train_inds))
        np.random.shuffle(scrambled_train)

        nnr_indicator = np.zeros(len(train_inds), dtype = int)
        nnc_indicator = np.zeros(len(train_inds), dtype = int)

        nnr_split, nnc_split = np.split(scrambled_train, (nnr_nnc_split * len(scrambled_train)).astype(int))

        nnr_indicator[nnr_split] = 1
        nnc_indicator[nnc_split] = 1

        np.savetxt(train_directory + filename + '.train_inds', np.vstack((train_inds, nnr_indicator, nnc_indicator)).T, header = 'train_inds  nnr_train  nnc_train', fmt = '%i  %i  %i')

        generate_input_files_cosmos(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)


    elif not os.path.isfile(train_directory + filename + '.bpz_train'):

        print('Generating fresh BPZ input files in %s' % train_directory)

        train_inds, nnr_indicator, nnc_indicator = np.loadtxt(train_directory + filename + '.train_inds', unpack = True, dtype = int)

        if closest_match_split:
            indices = [[0], range(1,13), range(13,25)]
            cm_specz, cm_features, cm_feature_errs = [np.loadtxt(train_directory + filename + '.closest_matches', usecols = thisindex) for thisindex in indices]
            train_specz = cm_specz[train_inds]
            train_features = cm_features[train_inds]
            train_feature_errs = cm_feature_errs[train_inds]
        else:
            train_specz, train_features, train_feature_errs = get_features(train_inds, inputfp = inputfp)

        header = 'index g e_g r e_r i e_i z e_z y e_y'
        fmt = ' %i' + ' %.5f'*24

        features_and_errs = np.empty((train_features.shape[0], train_features.shape[1] + train_feature_errs.shape[1]), dtype=float)
        features_and_errs[:,0::2] = train_features
        features_and_errs[:,1::2] = train_feature_errs

        np.savetxt(train_directory + filename + '.bpz_train', np.hstack((train_inds.reshape(-1,1), features_and_errs)), header = header, fmt = fmt)

        generate_input_files_cosmos(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)

    
    elif (os.path.isfile(train_directory + filename + '.bpz') and not 
            (os.path.isfile(train_directory + filename + '.nnr_train') or 
                os.path.isfile(train_directory + filename + '.nnr_test') or 
                os.path.isfile(train_directory + filename + '.nnr_validate'))):

        print('Generating fresh NNR input files in %s' % train_directory)

        train_inds, nnr_indicator, nnc_indicator = np.loadtxt(train_directory + filename + '.train_inds', dtype = int, unpack = True)

        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        train_inds_scrambled = np.copy(train_inds[nnr_indicator])
        np.random.shuffle(train_inds_scrambled)

        nnr_train_inds, nnr_validate_inds = np.split(train_inds_scrambled, [int(nn_train_val_split*len(train_inds_scrambled))])
        nnr_train_inds.sort()
        nnr_validate_inds.sort()

        np.savetxt(train_directory + filename + '.nnr_train_inds', nnr_train_inds, fmt = '%i')
        np.savetxt(train_directory + filename + '.nnr_val_inds', nnr_validate_inds, fmt = '%i')

        nnr_test_inds = train_inds[nnc_indicator]

        if closest_match_split:
            indices = [[0], range(1,13), range(13,25)]
            cm_specz, cm_features, cm_feature_errs = [np.loadtxt(train_directory + filename + '.closest_matches', usecols = thisindex) for thisindex in indices]
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

        temp = np.loadtxt(train_directory + filename + '.bpz', usecols = [1,2,3,5], unpack = True) # 1,2,3,5 corresponds to z_best, z_best_min, z_best_max, odds
        bpz_features = np.vstack((temp[0], temp[3], temp[2] - temp[1])).T

        # breakpoint()

        nnr_train_bpz_features, nnr_validate_bpz_features = np.split(bpz_features[nnr_indicator], [int(nn_train_val_split*np.sum(nnr_indicator))])
        nnr_test_bpz_features = bpz_features[nnc_indicator]

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

        np.savetxt(train_directory + filename + '.nnr_train', np.hstack((nnr_train_specz.reshape(-1,1), nnr_train_features, nnr_train_bpz_features, nnr_train_feature_errs, nnr_train_features_lo, nnr_train_features_hi, nnr_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(train_directory + filename + '.nnr_validate', np.hstack((nnr_validate_specz.reshape(-1,1), nnr_validate_features, nnr_validate_bpz_features, nnr_validate_feature_errs, nnr_validate_features_lo, nnr_validate_features_hi, nnr_validate_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(train_directory + filename + '.nnr_test', np.hstack((nnr_test_specz.reshape(-1,1), nnr_test_features, nnr_test_bpz_features, nnr_test_feature_errs, nnr_test_features_lo, nnr_test_features_hi, nnr_test_features_frac)), comments = '', header = header, fmt = fmt)

        generate_input_files_cosmos(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)



    elif (os.path.isfile(train_directory + filename + '.bpz') and not 
            (os.path.isfile(train_directory + filename + '.nnc_train') or 
                os.path.isfile(train_directory + filename + '.nnc_test') or 
                os.path.isfile(train_directory + filename + '.nnc_validate'))):

        print('Generating fresh NNC input files in %s' % train_directory)
        
        train_inds, nnr_indicator, nnc_indicator = np.loadtxt(train_directory + filename + '.train_inds', dtype = int, unpack = True)

        nnr_indicator = nnr_indicator.astype(bool)
        nnc_indicator = nnc_indicator.astype(bool)

        nnr_test_inds = train_inds[nnc_indicator]
        nnr_test_specz = read_csv(train_directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#', usecols = [0]).to_numpy()
        nnr_test_features = read_csv(train_directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#', usecols = range(1,13)).to_numpy()
        nnr_test_feature_errs = read_csv(train_directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#', usecols = range(16,28)).to_numpy()
        nnr_test_bpz_features = read_csv(train_directory + filename  + '.nnr_test', delimiter = '\s+', comment = '#')[['zphot', 'zconf', 'zerr']].to_numpy()


        translation_inds = np.arange(len(nnr_test_inds))
        np.random.shuffle(translation_inds)

        nnc_train_inds, nnc_validate_inds = np.split(nnr_test_inds[translation_inds], {int(nn_train_val_split*len(nnr_test_inds))})
        train_trans, val_trans = np.split(translation_inds, [int(nn_train_val_split*len(nnr_test_inds))])

        np.savetxt(train_directory + filename + '.nnc_train_inds', nnc_train_inds, fmt = '%i')
        np.savetxt(train_directory + filename + '.nnc_val_inds', nnc_validate_inds, fmt = '%i')

        np.savetxt(train_directory + filename + '.nnc_train_inds', np.vstack((nnc_train_inds, train_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')
        np.savetxt(train_directory + filename + '.nnc_validate_inds', np.vstack((nnc_validate_inds, val_trans)).T, fmt = '%i  %i', header = 'HSC_ind  NNR_ind')

        nnc_train_features = nnr_test_features[train_trans]
        nnc_validate_features = nnr_test_features[val_trans]
        
        nnc_train_feature_errs = nnr_test_feature_errs[train_trans]
        nnc_validate_feature_errs = nnr_test_feature_errs[val_trans]
        
        nnc_train_specz = nnr_test_specz[train_trans]
        nnc_validate_specz = nnr_test_specz[val_trans]
        
        nnc_train_bpz_features = nnr_test_bpz_features[train_trans]
        nnc_validate_bpz_features = nnr_test_bpz_features[val_trans]
        
        nnc_train_features_lo = nnc_train_features - nnc_train_feature_errs
        nnc_validate_features_lo = nnc_validate_features - nnc_validate_feature_errs
        nnc_train_features_hi = nnc_train_features + nnc_train_feature_errs
        nnc_validate_features_hi = nnc_validate_features + nnc_validate_feature_errs
        nnc_train_features_frac = nnc_train_feature_errs/nnc_train_features
        nnc_validate_features_frac = nnc_validate_feature_errs/nnc_validate_features
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(train_directory + filename + '.nnc_train', np.hstack((nnc_train_specz.reshape(-1,1), nnc_train_features, nnc_train_bpz_features, nnc_train_feature_errs, nnc_train_features_lo, nnc_train_features_hi, nnc_train_features_frac)), comments = '', header = header, fmt = fmt)
        np.savetxt(train_directory + filename + '.nnc_validate', np.hstack((nnc_validate_specz.reshape(-1,1), nnc_validate_features, nnc_validate_bpz_features, nnc_validate_feature_errs, nnc_validate_features_lo, nnc_validate_features_hi, nnc_validate_features_frac)), comments = '', header = header, fmt = fmt)
        # np.savetxt(train_directory + filename + '.nnc_test', np.hstack((nnc_test_specz.reshape(-1,1), nnc_test_features, nnc_test_tpz_features, nnc_test_feature_errs, nnc_test_features_lo, nnc_test_features_hi, nnc_test_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files_cosmos(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)



    elif not os.path.isfile(app_directory + filename + '.bpz_app'):

        print('Generating BPZ application set files...')

        cosmos_photz, cosmos_features, cosmos_feature_errs = get_cosmos_features()

        header = 'specz g r i z y gr ri iz zy gri riz izy eg er ei ez ey egr eri eiz ezy egri eriz eizy'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12

        header2 = 'index g e_g r e_r i e_i z e_z y e_y'
        fmt2 = ' %i' + ' %.5f'*10

        mags = cosmos_features[:,:5]
        mag_errs = cosmos_feature_errs[:,:5]

        mags_and_errs = np.empty((mags.shape[0], mags.shape[1] + mag_errs.shape[1]), dtype=float)
        mags_and_errs[:,0::2] = mags
        mags_and_errs[:,1::2] = mag_errs

        good_inds = np.all(np.isfinite(mags_and_errs), axis = 1)

        mags_and_errs = mags_and_errs[good_inds]
        cosmos_photz = cosmos_photz[good_inds]
        cosmos_features = cosmos_features[good_inds]
        cosmos_feature_errs = cosmos_feature_errs[good_inds]

        # np.savetxt(app_directory + filename + '.app_info', np.hstack((app_specz.reshape(-1,1), app_features, app_feature_errs)), header = header, fmt = fmt)
        np.savetxt(app_directory + filename + '.app_info', np.hstack((cosmos_photz.reshape(-1,1), cosmos_features, cosmos_feature_errs)), header = header, fmt = fmt)
        np.savetxt(app_directory + filename + '.bpz_app', np.hstack((np.arange(len(mags_and_errs)).reshape(-1,1), mags_and_errs)), header = header2, fmt = fmt2)

        generate_input_files_cosmos(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)



    elif not os.path.isfile(train_directory + filename + '.nn_app') and os.path.isfile(app_directory + filename + '.bpz'):

        print('Generating NN application set files...')

        if closest_match_split:
            indices = [[0], range(1,13), range(13,25)]
            app_specz, app_features, app_feature_errs = [np.loadtxt(app_directory + filename + '.app_info', usecols = thisindex) for thisindex in indices]
        else:
            app_inds = np.loadtxt(app_directory + filename + '.app_inds', dtype = int)
            app_specz, app_features, app_feature_errs = get_features(app_inds, inputfp = inputfp)

        temp = np.loadtxt(app_directory + filename + '.bpz', usecols = [1,2,3,5], unpack = True) # 1,2,3,5 corresponds to z_best, z_best_min, z_best_max, odds
        bpz_features = np.vstack((temp[0], temp[3], temp[2] - temp[1])).T

        app_features_lo = app_features - app_feature_errs
        app_features_hi = app_features + app_feature_errs
        app_features_frac = app_feature_errs/app_features
        
        header = 'specz g r i z y gr ri iz zy gri riz izy zphot zconf zerr eg er ei ez ey egr eri eiz ezy egri eriz eizy g_lo r_lo i_lo z_lo y_lo gr_lo ri_lo iz_lo zy_lo gri_lo riz_lo izy_lo g_hi r_hi i_hi z_hi y_hi gr_hi ri_hi iz_hi zy_hi gri_hi riz_hi izy_hi g_frac r_frac i_frac z_frac y_frac gr_frac ri_frac iz_frac zy_frac gri_frac riz_frac izy_frac'
        fmt = ' %.10f' + ' %.5f'*5 + ' %.5e' * 7 + ' %.5e'*12 + ' %.5e'*3 + ' %.5e'*12 + ' %.5e'*12 + ' %.5e'*12 

        np.savetxt(train_directory + filename + '.nn_app', np.hstack((app_specz.reshape(-1,1), app_features, bpz_features, app_feature_errs, app_features_lo, app_features_hi, app_features_frac)), comments = '', header = header, fmt = fmt)
        generate_input_files_cosmos(train_directory, app_directory, filename, inputfp, nnr_nnc_split, nn_train_val_split, closest_match_split, overall_split, ishift, gzshift, gz_colorlim, i_maglim, subrun = True)

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




def gen_hsc_filters(input_dir = './HSC/filters/', output_dir = './HSC/filters/Processed/'):

    filter_files = glob(input_dir + 'wHSC-*')
    quantum_eff = np.loadtxt(input_dir + 'qe_ccd_HSC.txt', unpack = True)
    window_transmittance = np.loadtxt(input_dir + 'throughput_win.txt', unpack = True)
    popt2_transmittance = np.loadtxt(input_dir + 'throughput_popt2.txt', unpack = True)
    atmos_transmittance = np.loadtxt(input_dir + 'lsst_atmos_std.dat', unpack = True)
    atmos_transmittance[0] *= 10.

    for thisfile in filter_files:
        filter_name = thisfile[-5]

        tempwave, tempthrough = np.loadtxt(thisfile, unpack = True)
        
        for wave, through in [quantum_eff, window_transmittance, popt2_transmittance, atmos_transmittance]:

            interp_through = np.interp(tempwave, wave, through)
            tempthrough = tempthrough * interp_through

        np.savetxt(output_dir + thisfile.split('/')[-1][:-4] + '_processed.res', np.vstack((tempwave, tempthrough)).T, fmt = '  %.2f  %.8f')










def color_covariance(color = 'u-g', tempfile = BPZ_dir + 'SED/adam_spec_dusty.list', AB_dir = BPZ_dir + 'AB/', chi2mask = None):

    # Add redshift <= 6 limitf

    band1_name = color[0]
    band2_name = color[-1]

    with open(tempfile, 'r') as readfile:

        all_lines = readfile.readlines()

    tempnames = [thisline[:-1] for thisline in all_lines]

    band1 = [np.loadtxt(AB_dir + thistemp[:-3] + 'DC2LSST_' + band1_name + '.AB', usecols = [1]) for thistemp in tempnames]
    band2 = [np.loadtxt(AB_dir + thistemp[:-3] + 'DC2LSST_' + band2_name + '.AB', usecols = [1]) for thistemp in tempnames]
    zgrid = np.loadtxt(AB_dir + tempnames[0][:-3] + 'DC2LSST_' + band1_name + '.AB', usecols = [0])

    colors = np.array(band1) - np.array(band2)

    fig = plt.figure(figsize = (10,8))
    sp = fig.add_subplot(111)

    # cmap = plt.cm.jet
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # newcmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    # bounds = np.linspace(-1,1,11)
    # norm = mpl.colors.BoundaryNorm(bounds, newcmap.N)

    img = sp.imshow(np.corrcoef(colors), origin = 'lower', cmap = 'plasma')
    if chi2mask != None:
        mask = np.zeros()
        maskimg = sp.imshow(mask, origin = 'lower', cmap = 'Greys', vmin = 0, vmax = 1)

    plt.colorbar(img)

    sp.set_title(color)




def plot_templates(spectra = 'adam_spec.list'):

    specdir = BPZ_dir + 'SED/'

    readfile = open(specdir + spectra, 'r')

    lines = readfile.readlines()

    readfile.close()

    files = [line[:-1] for line in lines] # Remove newline characters

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    plasma = plt.get_cmap('plasma')

    for x, file in enumerate(files):

        waves, spec = np.loadtxt(specdir + file, unpack = True)

        sp.plot(waves, spec, linewidth = 2, color = plasma(int(float(x)*plasma.N/len(files))), label = file[:-4])

    sp.set_xlabel(r'Wavelength ($\AA$)')
    sp.set_ylabel('Flux Density ')

    sp.set_xscale('log')

    sp.legend()
    sp.set_xlim(10**3, 10**5)
    sp.set_ylim(0,16)





def colorcolortest(bpzfp = BPZ_dir + 'adam/eazytest.bpz', color_name1 = 'r-i', color_name2 = 'u-g', interp_z = 0.3, legend = False):

    AB_dir = BPZ_dir + 'AB/'

    bpz_file = open(bpzfp, 'r')
    all_lines = bpz_file.readlines()
    bpz_file.close()

    sed_dir = [s for s in all_lines if 'SED_DIR' in s][0].split('=')[-1][:-1]
    spectrafile = sed_dir + [s for s in all_lines if 'SPECTRA' in s][0].split('=')[-1][:-1]
    catfile = bpzfp + 'adam/' + [s for s in all_lines if 'INPUT' in s][0].split('=')[-1][:-1]
    interp = int([s for s in all_lines if 'INTERP' in s][0].split('=')[-1][:-1])

    with open(spectrafile, 'r') as rf:
        temp_spec_files = rf.readlines()

    color1 = []
    color2 = []

    template_names = [thisfile[:-5] for thisfile in temp_spec_files]

    for thisname in template_names:

        # Color 1

        z1, flux1 = np.loadtxt(AB_dir + thisname + '.DC2LSST_' + color_name1.split('-')[0] + '.AB', unpack = True)
        z2, flux2 = np.loadtxt(AB_dir + thisname + '.DC2LSST_' + color_name1.split('-')[1] + '.AB', unpack = True)

        interp1 = np.interp(interp_z, z1, flux1)
        interp2 = np.interp(interp_z, z2, flux2)

        color1.append(-2.5*np.log10(interp1/interp2))

        # Color 2

        z1, flux1 = np.loadtxt(AB_dir + thisname + '.DC2LSST_' + color_name2.split('-')[0] + '.AB', unpack = True)
        z2, flux2 = np.loadtxt(AB_dir + thisname + '.DC2LSST_' + color_name2.split('-')[1] + '.AB', unpack = True)

        interp1 = np.interp(interp_z, z1, flux1)
        interp2 = np.interp(interp_z, z2, flux2)

        color2.append(-2.5*np.log10(interp1/interp2))

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    for temp, tc1, tc2 in zip(template_names, color1, color2):
        sp.scatter(tc1, tc2, s=np.linspace(50, 10, len(interp_z)), label = temp)
    
    sp.set_xlabel(color_name1)
    sp.set_ylabel(color_name2)

    if legend:
        sp.legend(loc = 'lower right')





def colortest(color_name = 'u-g', bpzfp = BPZ_dir + 'adam/eazytest.bpz', interp_z = 0.3, legend = False):

    AB_dir = BPZ_dir + 'AB/'

    bpz_file = open(bpzfp, 'r')
    all_lines = bpz_file.readlines()
    bpz_file.close()

    sed_dir = [s for s in all_lines if 'SED_DIR' in s][0].split('=')[-1][:-1]
    spectrafile = sed_dir + [s for s in all_lines if 'SPECTRA' in s][0].split('=')[-1][:-1]
    catfile = bpzfp + 'adam/' + [s for s in all_lines if 'INPUT' in s][0].split('=')[-1][:-1]
    interp = int([s for s in all_lines if 'INTERP' in s][0].split('=')[-1][:-1])

    with open(spectrafile, 'r') as rf:
        temp_spec_files = rf.readlines()

    color = []

    template_names = [thisfile[:-5] for thisfile in temp_spec_files]

    for thisname in template_names:

        z1, flux1 = np.loadtxt(AB_dir + thisname + '.DC2LSST_' + color_name.split('-')[0] + '.AB', unpack = True)
        z2, flux2 = np.loadtxt(AB_dir + thisname + '.DC2LSST_' + color_name.split('-')[1] + '.AB', unpack = True)

        # interp1 = np.interp(interp_z, z1, flux1)
        # interp2 = np.interp(interp_z, z2, flux2)

        color.append(-2.5*np.log10(flux1/flux2))
        redshift = z1

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    for temp, thiscolor in zip(template_names, color):
        sp.plot(redshift, thiscolor, linewidth = 2, label = temp)
    
    sp.set_xlabel('Redshift (z)')
    sp.set_ylabel(color_name)

    if legend:
        sp.legend(loc = 'lower right')


def calc_a(flux1, flux2):

    return sum(flux1*flux2)/sum(flux2**2)


def calc_b(flux1, flux2):

    return np.sqrt(sum((flux1 - (calc_a(flux1, flux2) * flux2))**2)/sum(flux1**2))



def gen_sparse_template_spectra(b_lim = 0.04, info_folder = None):

    # Less
    # tau = np.append(-np.logspace(-2,1,3), np.logspace(-2,1,3)[::-1])
    # ages = 10.**np.linspace(-2,1,4)
    # Z = np.array([1.0])
    # dust = np.array([0.05, 1.0])

    # Normal
    # tau = np.append(-np.logspace(-2,1,10), np.logspace(-2,1,10)[::-1])
    # ages = 10.**np.linspace(-2,1,7)
    # Z = np.array([0.2,1.0])
    # dust = np.array([0.05, 1.0])

    # More
    tau = np.append(-np.logspace(-2,1,10), np.logspace(-2,1,10)[::-1])
    ages = 10.**np.linspace(-2,1,7)
    Z = np.arange(0.2,1.01,0.4)
    dust = np.arange(0.05, 1.06, 0.2)

    waves = None
    l_lambda_list = []
    l_lambda_list_norm = [] # This has to exist because otherwise, there are overflow errors when calculating similarity parameters

    arrsize = len(Z) * len(tau) * len(ages) * len(dust)

    params = []

    for thisZ in Z:
        for thistau in tau:
            for thisage in ages:
                for thisdust in dust:
                    params.append((thisZ, thistau, thisage, thisdust))

    params = np.array(params, dtype = [('Z', float), ('tau', float), ('age', float), ('dust', float)])
    
    if info_folder and os.path.isdir(info_folder):
        paramfile = info_folder + 'params.dat'
        read_params = np.loadtxt(paramfile)
        if len(read_params) == len(params):
            if all(read_params[:,0] == params['Z']) and all(read_params[:,1] == params['tau']) and all(read_params[:,2] == params['age']) and all(read_params[:,3] == params['dust']):

                tempnums = []
                analognums = []

                for thisfile in sorted(glob(info_folder + '*_*.dat')):

                    tempnums.append(int(thisfile.split('_')[-1].split('.')[0]))
                    analognums.append(np.loadtxt(thisfile, ndmin = 1).astype(int))

                return tempnums, analognums, params

    print('\nGenerating Initial Spectral Pool...')

    for thisZ in tqdm(Z):
        for thistau in tqdm(tau):
            for thisage in ages:
                for thisdust in dust:

                    waves, l_lambda = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, peraa = True, Av = thisdust, emline = True)
                    l_lambda_list_norm.append(l_lambda/max(l_lambda))
                    l_lambda_list.append(l_lambda)
    
    l_lambda_list_norm = np.array(l_lambda_list_norm)

    # Method from Kriek+2011 https://ui.adsabs.harvard.edu/abs/2011ApJ...743..168K/abstract

    b_array = -np.ones((arrsize, arrsize))

    print('\nCalculating Analogs...')

    for x, lum1 in enumerate(tqdm(l_lambda_list_norm)):
        for y, lum2 in zip(range(x,len(l_lambda_list_norm)),l_lambda_list_norm[x:]):
            lum1_trimmed = lum1[(waves>1700) & (waves < 12000)]
            lum2_trimmed = lum2[(waves>1700) & (waves < 12000)]

            a_xy = sum(lum1_trimmed*lum2_trimmed)/sum(lum2_trimmed**2)
            b_array[x][y] = np.sqrt(sum((lum1_trimmed - (a_xy * lum2_trimmed))**2)/sum(lum1_trimmed**2))
    
            if x != y:
                b_array[y][x] = b_array[x][y]

    b_analogs = b_array<b_lim

    print('\nCulling Analogs...')

    analognums = []
    tempnums = []
    orig_tempnums = np.arange(arrsize)

    while len(b_analogs) > 0:

        analog_counts = np.sum(b_analogs, axis = 1)

        max_ind = np.where(analog_counts == max(analog_counts))[0]

        if len(max_ind) > 1:
            max_ind = np.random.choice(max_ind)
        else:
            max_ind = max_ind[0]

        analog_inds = np.where(b_analogs[max_ind])[0]
        analognums.append(orig_tempnums[analog_inds[analog_inds != max_ind]]) # Add all the analogs except the kept template to the analog list
        tempnums.append(orig_tempnums[max_ind])

        b_analogs = np.delete(b_analogs, analog_inds, axis = 0)
        b_analogs = np.delete(b_analogs, analog_inds, axis = 1)
        orig_tempnums = np.delete(orig_tempnums, analog_inds)



    return tempnums, analognums, params




def save_sparse_templates(templatefolder = BPZ_dir + 'SED/sparsetemplates/', tempfile = 'sparse_spec.list', b_lim = 0.04, norm = True, median = False, interp_emission_lines = False):

    tempnums, analognums, params = gen_sparse_template_spectra(b_lim, info_folder = templatefolder + 'analog_info/')

    names = []

    for thisage, thistau, thisZ, thisdust, thisanalogs in zip(params['age'][tempnums], params['tau'][tempnums], params['Z'][tempnums], params['dust'][tempnums], analognums):

        basename = 'adamtemp_{:.3f}Gyr_{:.3f}tau_{:.1f}Zsol_{:.2f}Av'.format(thisage, thistau, thisZ, thisdust)

        if interp_emission_lines:
            thisname = basename + '_noemline'
        elif median:
            thisname = basename + '_med'
        else:
            thisname = basename

        thisname = thisname + '.sed'

        names.append(thisname)

        # =====================
        #  SPECTRUM OPERATIONS
        # =====================

        if interp_emission_lines:
            emline_indices = [7,31,32,49,50,62] # Lya, OII3727, OII3730, OIII4963, OIII5007, Ha
            emline_widths = [75, 10, 10, 10, 10, 10]

            # waves_el, l_lambda_el = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, peraa = True, Av = thisdust, emline = True)
            waves_nel, l_lambda_nel = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, peraa = True, Av = thisdust, emline = False)
            emline_waves = phot.starpop.emline_wavelengths[emline_indices]

            for thiswave, thiswidth in zip(emline_waves, emline_widths):
                smooth_index = np.where(np.abs(thiswave - waves_nel) <= thiswidth)[0]
                l_lambda_nel[smooth_index] = np.interp(waves_nel[smooth_index], waves_nel[smooth_index[[0,-1]]], l_lambda_nel[smooth_index[[0,-1]]])


            # same = np.where((l_lambda_el-l_lambda_nel)==0)[0]
            # samewaves = waves_el[same]
            # sameflux = l_lambda_el[same]

            # interp_flux = np.interp(waves_nel, samewaves, sameflux)
            waves = waves_nel
            l_lambda = l_lambda_nel

        else:
            waves, l_lambda = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, peraa = True, Av = thisdust, emline = True)

        if norm:
            if not median and not interp_emission_lines:
                l_lambda = l_lambda/max(l_lambda)*10.
            else:
                w, l = np.loadtxt(templatefolder + basename + '.sed', unpack = True)
                l_lambda = l_lambda * l[0]/l_lambda[0]

        if median:

            l_nu = l_lambda * waves**2
            findmed = np.vectorize(lambda wavelength: np.median(np.interp(np.arange(wavelength-75., wavelength+75.), waves, l_nu))) 
            med_l_nu = findmed(waves)
            med_l_lambda = med_l_nu / waves**2
            l_lambda = med_l_lambda

        # if len(names) == 1:
        #     print(max(l_lambda))

        np.savetxt(templatefolder + thisname, np.vstack((waves, l_lambda)).T, fmt = '%e   %.10e')

    if not median and not interp_emission_lines:

        if os.path.isfile(templatefolder + tempfile):

            template_file = open(templatefolder+tempfile, 'r')
            all_lines = template_file.readlines()
            template_file.close()
            all_lines = [thisline for thisline in all_lines if 'adamtemp' not in thisline]
            all_lines = all_lines + [thisname + '\n' for thisname in names]
            with open(templatefolder+tempfile, 'w') as writefile:
                writefile.write(''.join(all_lines))
        else:
            with open(templatefolder + tempfile, 'w') as writefile:
                for thistemplate in names:
                    writefile.write(thistemplate)
                    writefile.write('\n')


        if os.path.isdir(templatefolder + 'analog_info/'):
            for thisfile in glob(templatefolder + 'analog_info/*'):
                os.remove(thisfile)
        else:
            os.mkdir(templatefolder + 'analog_info/')

        for x, (thistempnum, theseanalogs) in enumerate(zip(tempnums, analognums)):

            np.savetxt(templatefolder + 'analog_info/%03i_%04i.dat' %(x, thistempnum), theseanalogs, fmt = '%04i', header = 'Analog numbers for spectrum %04i (template %03i)' %(thistempnum, x))

        np.savetxt(templatefolder + 'analog_info/params.dat', params.T, header = 'Parameters for original spectrum pool that was pared down to this one\nZ  tau  age  A_v')




def plot_analogs(templatenum, spec_folder = BPZ_dir + 'SED/sparsetemplates/'):

    templatenums, analognums, params = gen_sparse_template_spectra(info_folder = spec_folder + 'analog_info/')

    Z = params[templatenums[templatenum]]['Z']
    age = params[templatenums[templatenum]]['age']
    tau = params[templatenums[templatenum]]['tau']
    dust = params[templatenums[templatenum]]['dust']

    filename = spec_folder + 'adamtemp_{:.3f}Gyr_{:.3f}tau_{:.1f}Zsol_{:.2f}Av.sed'.format(age, tau, Z, dust)

    wave, l_lambda = np.loadtxt(filename, unpack = True)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    if len(analognums[templatenum]) > 0:
        for thisanalog in tqdm(analognums[templatenum]):
            thisZ = params[thisanalog]['Z']
            thisage = params[thisanalog]['age']
            thistau = params[thisanalog]['tau']
            thisdust = params[thisanalog]['dust']

            thiswave, thisflux = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, peraa = True, Av = thisdust, emline = True)
            a_xy = sum(l_lambda*thisflux)/sum(thisflux**2)

            sp.plot(thiswave, a_xy * thisflux, c = '0.5', alpha = 0.2)

    sp.plot(wave, l_lambda, c = 'k')

    sp.set_xscale('log')
    sp.set_yscale('log')

    sp.set_xlabel(r'Wavelength ($\AA$)')
    sp.set_ylabel('Flux (Arbitrary Units)')

    sp.set_xlim(3000, 12000)
    sp.set_ylim(10**-3, 10**2)

















def gen_template_spectra_dusty(templatefolder = BPZ_dir + 'SED/adamtemp/', tempfile = 'adam_spec_dusty.list', norm = True, emline = True, median = False, interp_emission_lines = False):

    tau = np.append(-np.logspace(-2,1,10), np.logspace(-2,1,10)[::-1])
    ages = 10.**np.linspace(-2,1,7)
    Z = np.array([0.2,1.0])
    dust = np.array([0.05, 1.0])

    # names = ['adamtemp_{:.3f}Gyr_{:.3f}tau_{:.1f}Zsol.sed'.format(thisage, thistau, thisZ) for thisZ in Z for thistau in tau for thisage in ages if ((thistau<0 and thistau >= -1.) and thisage < 1) or (thistau > 0) or (thistau < -1.)]
    names = []

    for thisZ in tqdm(Z):
        for thistau in tqdm(tau):
            for thisage in ages:
                for thisdust in dust:
                    # if ((thistau<0 and thistau >= -1.) and thisage < 1) or (thistau > 0) or (thistau < -1.): # Strongly rising SFHs get unphysically bright after doing so for long periods of time
                
                    basename = 'adamtemp_{:.3f}Gyr_{:.3f}tau_{:.1f}Zsol_{:.2f}Av'.format(thisage, thistau, thisZ, thisdust)
                    if emline:
                        thisname = basename
                    else:
                        thisname = basename + '_noemline'

                    if median:
                        thisname = basename + '_med'
                    else:
                        thisname = basename

                    thisname = thisname + '.sed'

                    names.append(thisname)


                    # =====================
                    #  SPECTRUM OPERATIONS
                    # =====================

                    if interp_emission_lines:
                        emline_indices = [7,31,32,49,50,62] # Lya, OII3727, OII3730, OIII4963, OIII5007, Ha
                        emline_widths = [75, 10, 10, 10, 10, 10]

                        # waves_el, l_lambda_el = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, peraa = True, Av = thisdust, emline = True)
                        waves_nel, l_lambda_nel = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, peraa = True, Av = thisdust, emline = False)
                        emline_waves = phot.starpop.emline_wavelengths[emline_indices]

                        for thiswave, thiswidth in zip(emline_waves, emline_widths):
                            smooth_index = np.where(np.abs(thiswave - waves_nel) <= thiswidth)[0]
                            l_lambda_nel[smooth_index] = np.interp(waves_nel[smooth_index], waves_nel[smooth_index[[0,-1]]], l_lambda_nel[smooth_index[[0,-1]]])


                        # same = np.where((l_lambda_el-l_lambda_nel)==0)[0]
                        # samewaves = waves_el[same]
                        # sameflux = l_lambda_el[same]

                        # interp_flux = np.interp(waves_nel, samewaves, sameflux)
                        waves = waves_nel
                        l_lambda = l_lambda_nel

                    else:
                        waves, l_lambda = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, peraa = True, Av = thisdust, emline = emline)

                    if norm:
                        if emline and not median and not interp_emission_lines:
                            l_lambda = l_lambda/max(l_lambda)*10.
                        else:
                            w, l = np.loadtxt(templatefolder + basename + '.sed', unpack = True)
                            l_lambda = l_lambda * l[0]/l_lambda[0]

                    if median:

                        l_nu = l_lambda * waves**2
                        findmed = np.vectorize(lambda wavelength: np.median(np.interp(np.arange(wavelength-75., wavelength+75.), waves, l_nu))) 
                        med_l_nu = findmed(waves)
                        med_l_lambda = med_l_nu / waves**2
                        l_lambda = med_l_lambda

                    if len(names) == 1:
                        print(max(l_lambda))


                    np.savetxt(templatefolder + thisname, np.vstack((waves, l_lambda)).T, fmt = '%e   %.10e')

    if emline and not median and not interp_emission_lines:

        if os.path.isfile(templatefolder + tempfile):

            template_file = open(templatefolder+tempfile, 'r')
            all_lines = template_file.readlines()
            template_file.close()
            all_lines = [thisline for thisline in all_lines if 'adamtemp' not in thisline]
            all_lines = all_lines + [thisname + '\n' for thisname in names]
            with open(templatefolder+tempfile, 'w') as writefile:
                writefile.write(''.join(all_lines))
        else:
            with open(templatefolder + tempfile, 'w') as writefile:
                for thistemplate in names:
                    writefile.write(thistemplate)
                    writefile.write('\n')


def gen_template_spectra(templatefolder = BPZ_dir + 'SED/adamtemp/', tempfile = 'adam_spec.list', emline = True, norm = True):

    tau = np.append(-np.logspace(-2,1,10), np.logspace(-2,1,10)[::-1])
    ages = 10.**np.linspace(-2,1,7)
    Z = np.array([0.2,1.0])

    # names = ['adamtemp_{:.3f}Gyr_{:.3f}tau_{:.1f}Zsol.sed'.format(thisage, thistau, thisZ) for thisZ in Z for thistau in tau for thisage in ages if ((thistau<0 and thistau >= -1.) and thisage < 1) or (thistau > 0) or (thistau < -1.)]
    names = []

    for thisZ in tqdm(Z):
        for thistau in tqdm(tau):
            for thisage in ages:
                # if ((thistau<0 and thistau >= -1.) and thisage < 1) or (thistau > 0) or (thistau < -1.): # Strongly rising SFHs get unphysically bright after doing so for long periods of time
            
                thisname = 'adamtemp_{:.3f}Gyr_{:.3f}tau_{:.1f}Zsol'.format(thisage, thistau, thisZ)
                if emline:
                    thisname = thisname + '.sed'
                else:
                    thisname = thisname + '_noemline.sed'

                names.append(thisname)

                waves, l_lambda = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, peraa = True, emline = emline)
                if norm:
                    l_lambda = l_lambda/max(l_lambda)*10.

                np.savetxt(templatefolder + thisname, np.vstack((waves, l_lambda)).T, fmt = '%e   %.10e')

    if emline:

        if os.path.isfile(templatefolder + tempfile):

            template_file = open(templatefolder+tempfile, 'r')
            all_lines = template_file.readlines()
            template_file.close()
            all_lines = [thisline for thisline in all_lines if 'adamtemp' not in thisline]
            all_lines = all_lines + [thisname + '\n' for thisname in names]
            with open(templatefolder+tempfile, 'w') as writefile:
                writefile.write(''.join(all_lines))
        else:
            with open(templatefolder + tempfile, 'w') as writefile:
                for thistemplate in names:
                    writefile.write(thistemplate)
                    writefile.write('\n')





def gen_cat_odin(rand_catsize = 10000, redshift_lims = (2., 5.), cat_output = 'ODIN/', magnorm = 18., Avmin = 0.0, Avmax = 1.0):

    cat_output = BPZ_dir + cat_output

    if rand_catsize:
        np.random.seed(100)

        redshift = np.random.random(rand_catsize)*(redshift_lims[1]-redshift_lims[0]) + redshift_lims[0]
        tau = np.random.random(rand_catsize)*(3)-2 * (2*(np.random.random(rand_catsize) >= 0.5)-1) # Second expression gives a random sign
        Z = np.random.choice(np.array([0.2, 1.0]), size = rand_catsize)
        dust = np.random.random(rand_catsize) * (Avmax - Avmin) + Avmin
        ages = 10**(np.random.random(rand_catsize)*3-2)

    else:
        ages, tau, Z, dust = [this.ravel() for this in np.meshgrid(10.**np.linspace(-2,1,7), np.append(-np.logspace(-2,1,10), np.logspace(-2,1,10)[::-1]), np.array([0.2,1.0]), np.array([0.05, 1.]))]
        # dust = np.random.random_sample(size = len(tau)) * (Avmax-Avmin) + Avmin
        redshift = np.average(redshift_lims)
        # dust = dust.reshape((len(Z), len(tau), len(ages)))

    if not hasattr(redshift, '__iter__'):
        redshift = np.array([redshift,]*len(tau))
    elif len(redshift) != len(tau):
        print('WARNING, REDSHIFT LIST IS NOT CORRECT LENGTH')


    phot_wave = []
    phot_flux = []
    phot_err = []

    for gal_id, (thisZ, thistau, thisage, thisdust, thisredshift) in enumerate(tqdm(zip(Z, tau, ages, dust, redshift), total = len(Z))):

        waves, l_nu = odin_phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, Av = thisdust, redshift = thisredshift)

        shifted_wavelengths, spec_flux = odin_phot.redshift(thisredshift, waves, l_nu)

        shifted_wavelengths_normed, spec_flux_normed = odin_phot.filters.mag_norm(shifted_wavelengths, spec_flux, magnorm)

        # phot_wave, phot_flux, phot_err = odin_phot.filters.get_photometry(shifted_wavelengths, spec_flux, output_mags = True, dmag_err = True)
        temp_phot_wave, temp_phot_flux, temp_phot_err = odin_phot.filters.get_photometry(shifted_wavelengths_normed, spec_flux_normed, output_mags = True, dmag_err = True)

        phot_wave.append(temp_phot_wave)
        phot_flux.append(temp_phot_flux)
        phot_err.append(temp_phot_err)

    phot_wave = np.array(phot_wave)
    phot_flux = np.array(phot_flux)
    phot_err = np.array(phot_err)

    odincat_header = 'id ' + ''.join(['f_%s e_%s ' % (thiskey, thiskey) for thiskey in odin_phot.filters.keys])
    not_odincat_header = 'id ' + ''.join(['f_%s e_%s ' % (thiskey, thiskey) for thiskey in odin_phot.filters.keys if thiskey[0] != 'N'])

    pairs = [(thisflux, thiserr) for thisflux, thiserr in zip(phot_flux.T, phot_err.T)]
    odincat = np.vstack((np.arange(len(phot_flux)), *[item for thispair in pairs for item in thispair])).T

    not_odin_inds = [0] + sorted(list(np.ravel(np.where(~odin_phot.filters.is_odin_filter)[0] * 2 + np.array([0,1]).reshape(-1,1) + 1)))
    not_odincat = odincat.T[not_odin_inds].T

    np.savetxt(cat_output + 'odin_filters.cat', odincat, header = odincat_header, fmt = '%03i ' + '%.5f '*(odincat.shape[1]-1))
    np.savetxt(cat_output + 'no_odin_filters.cat', not_odincat, header = not_odincat_header, fmt = '%03i ' + '%.5f '*(not_odincat.shape[1]-1))

    with open(cat_output + 'index.dat', 'w') as indexfile:

        for x, (thisage, thistau, thisZ, thisAv, thisredshift) in enumerate(zip(ages, tau, Z, dust, redshift)):
            indexfile.write('  {:03}{:10.3f}{:10.3f}{:10.1f}{:10.2f}{:10.2f}'.format(x, thisage, thistau, thisZ, thisAv, thisredshift) + '\n')



def gen_norm_test(lam_start = 3000, lam_end = 10000, fwhm = 50, redshift = 3.12, runfp = '/home/adam/Research/pz_pdf/pz/BPZ/norm_test/'):
    gen_madau_test(lam_start, lam_end, fwhm, redshift, runfp)


def gen_madau_test(lam_start = 1000, lam_end = 4000, fwhm = 50, redshift = 3.12, runfp = '/home/adam/Research/pz_pdf/pz/BPZ/madau_test/'):

    # Make test filters

    baseline_lam = np.linspace(-fwhm, 2*fwhm, 150)
    baseline_throughput = np.zeros(baseline_lam.shape)
    baseline_throughput[50:100] = 1.
    filter_waves = np.arange(lam_start, lam_end, fwhm)
    filternum = len(filter_waves)
    filter_names = []

    for x, this_lam in enumerate(filter_waves.reshape(-1,1) + baseline_lam):

        fname = '%i_%i_%i_%03i_TESTFILT.res' % (lam_start, lam_end, fwhm, x)
        filter_names.append(fname)
        np.savetxt('/home/adam/Research/pz_pdf/pz/BPZ/FILTER/' + fname, np.vstack((this_lam, baseline_throughput)).T)


    # Make test catalog

    temp_lam, temp_l_lam = odin_phot.find_spectrum(10, peraa = True)
    obs_lam, obs_l_nu = odin_phot.find_spectrum(10, redshift = redshift)
    shifted_obs_lam, shifted_obs_l_nu = odin_phot.redshift(redshift, obs_lam, obs_l_nu)

    shifted_obs_lam_normed, shifted_obs_f_nu_normed = odin_phot.filters.mag_norm(shifted_obs_lam, shifted_obs_l_nu, 18.)

    # Get photometry

    c = 3.e10 # speed of light in cm/s
    phot_wave = filter_waves
    phot_err = np.ones(len(baseline_lam))*0.001

    f_lambda = shifted_obs_f_nu_normed * c/(shifted_obs_lam_normed**2.)

    interp_x = np.arange(500, 11000, 0.1) # The x-axis of the spectrum

    phot_flux = []

    # Loop through each of the LSST filters
    for this_filt_wave in filter_waves.reshape(-1,1) + baseline_lam:

        interp_y = np.interp(interp_x, shifted_obs_lam_normed, f_lambda) # The interpolated y-axis of the spectrum (in f_lambda)

        # Find the filter curve values at the same x values as interp_x
        filter_interp_y = np.interp(interp_x, this_filt_wave, baseline_throughput) 

        phot_flux.append(np.trapz(filter_interp_y * interp_y * interp_x, x = interp_x)/np.trapz(filter_interp_y * c / interp_x, x = interp_x))

    phot_mags = np.array(-2.5*np.log10(phot_flux) -48.6)

    cat_header = 'id ' + ''.join(['f_%03i e_%03i ' % (x,x) for x in range(filternum)])

    pairs = [(thisflux, thiserr) for thisflux, thiserr in zip(phot_mags, phot_err)]
    cat = np.vstack([np.hstack((idnum, *[item for thispair in pairs for item in thispair])) for idnum in range(2)])

    np.savetxt(runfp + 'test.cat', cat, header = cat_header, fmt = '%03i ' + '%.5f '*(cat.shape[1]-1))




    # Make column file

    filter_numbers = [thisname.split('_')[2] for thisname in filter_names]
    with open(runfp + 'test.columns', 'w') as writefile:
        writefile.write('# Filter            columns  AB/Vega  zp_error  zp_offset\n')

        for x, thisname in enumerate(filter_names):
            writefile.write((thisname.split('.')[0]).ljust(30) + ('%i,%i' % (2*(x+1), 2*(x+1)+1)).ljust(9)  + 'AB'.ljust(9) + '0.01'.ljust(10) + '0.00\n')

        writefile.write('M_0'.ljust(30) + '2\n')
        writefile.write('ID'.ljust(30) + '1\n')

    # Generate flat spectrum and spec file

    # spec_wave = np.logspace(2.5, 6, 5000)
    # spec_flux = np.ones(len(spec_wave))

    # np.savetxt('/home/adam/Research/pz_pdf/pz/BPZ/SED/adam_templates/test/flatspec.sed', np.vstack((spec_wave, spec_flux)).T, fmt = '%.2f  %.2f')
    # with open('/home/adam/Research/pz_pdf/pz/BPZ/SED/adam_templates/test/test.list', 'w') as writefile:
    #     writefile.write('flatspec.sed\n')

    np.savetxt('/home/adam/Research/pz_pdf/pz/BPZ/SED/adam_templates/test/blue_with_break.sed', np.vstack((temp_lam, temp_l_lam/max(temp_l_lam))).T, fmt = '%.2f  %.4e')
    with open('/home/adam/Research/pz_pdf/pz/BPZ/SED/adam_templates/test/test.list', 'w') as writefile:
        writefile.write('blue_with_break.sed\n')




def etau_madau(lam_obs, z):

    # Modified by Adam Broussard to fix errors

    A = [3.6e-3, 1.7e-3, 1.2e-3, 9.3e-4]
    lyman_lam = [1216., 1026, 973, 950]

    tau = np.zeros(len(lam_obs))

    for this_A, this_lyman in zip(A, lyman_lam):
        inds = lam_obs < this_lyman*(1+z)
        tau[inds] += this_A * (lam_obs[inds] / this_lyman)**3.46

    # tau[lam_obs/(1+z) < 912.] = 0

    xc = lam_obs[lam_obs/(1+z) < 912.] / 912.
    xem = 1+z

    tau[lam_obs/(1+z) < 912.] += (0.25 * xc**3 * (xem**0.46 - xc**0.46) + 
        9.4 * xc**1.5 * (xem**0.18 - xc**0.18) - 
        0.7 * xc**3 * (xc**-1.32 - xem**-1.32) - 
        0.023 * (xem**1.68 - xc**1.68))

        # if this_lyman*(1+z) > lam_obs:
        #     total += this_A * (lam_obs / this_lyman)**3.46

    tau = np.clip(tau,0,700)

    tau[lam_obs < lam_obs[np.argmax(tau)]] = max(tau)

    return np.exp(-tau)



def plot_norm_test(true_z = 3.12, plot_true_spec = True):
    return plot_madau_test(true_z, plot_true_spec, runfp = '/home/adam/Research/pz_pdf/pz/BPZ/norm_test/', lam_start = 3000, lam_end = 10000)


def plot_madau_test(true_z = 3.12, plot_true_spec = True, runfp = '/home/adam/Research/pz_pdf/pz/BPZ/madau_test/', lam_start = 1000, lam_end = 4000, fwhm = 50):

    # fit_template = int(self.t_b[gal_id])
    # fit_z = self.z_b[gal_id]
    # true_z = self.z_in[gal_id]

    # photometry_waves = np.array([self.filters.wavelength_centers[thiskey] for thiskey in self.filters.keys])
    # temp_photometry = self.fit_flux[gal_id]
    # obs_photometry = self.obs_flux[gal_id]
    # tempwave = self.tempwave[fit_template-1]*(1+fit_z)
    # temp_fnu = self.tempflux[fit_template-1] * self.fit_norm[gal_id] * tempwave**2 / 3e18

    baseline_lam = np.linspace(-fwhm, 2*fwhm, 150)
    baseline_throughput = np.zeros(baseline_lam.shape)
    baseline_throughput[50:100] = 1.
    filter_waves = np.arange(lam_start, lam_end, fwhm)
    filternum = len(filter_waves)
    photometry_waves = (filter_waves.reshape(-1,1) + baseline_lam).T[75]

    _, _, fit_z, _, fit_norm, *fluxes = np.loadtxt(runfp + 'test.flux_comparison')[0]
    temp_photometry = np.array(fluxes[:int(len(fluxes)/3)]) * 10**-19.44
    obs_photometry = np.array(fluxes[int(len(fluxes)/3):int(2*len(fluxes)/3)]) * 10**-19.44
    print(obs_photometry[0]/temp_photometry[0])
    temp_photometry = temp_photometry * obs_photometry[0]/temp_photometry[0]

    # orig_obs_photometry = np.power(10., (np.loadtxt('/home/adam/Research/pz_pdf/pz/BPZ/madau_test/test.cat')[0][1:][::2] + 48.6)/(-2.5))
    # obs_flux_errs = fluxes[120:]

    tempwave, tempflam = np.loadtxt('/home/adam/Research/pz_pdf/pz/BPZ/SED/adam_templates/test/blue_with_break.sed', unpack = True)
    temp_fnu = tempflam * fit_norm * tempwave**2 / 3e18 * 10**-19.44 * (1+fit_z)**2
    temp_fnu_abs = temp_fnu * etau_madau(tempwave*(1+fit_z), fit_z)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    sp.plot(np.array([1216., 1216.])*(1+fit_z), [0,1], color = 'C1', linestyle = '--', transform = sp.get_xaxis_transform())
    sp.plot(np.array([1216., 1216.])*(1+true_z), [0,1], color = 'C1', transform = sp.get_xaxis_transform())

    sp.plot(tempwave*(1+fit_z), temp_fnu, color = 'k', alpha = 0.5, zorder = 1)
    sp.plot(tempwave*(1+fit_z), temp_fnu_abs, color = 'k', zorder = 1)

    # sp.plot(self.tempwave[fit_template]*(1+fit_z), self.tempflux[fit_template] * self.fit_norm[gal_id] * tempwave**2 / 3e18, color = 'g')

    if plot_true_spec:
        waves, l_nu = odin_phot.find_spectrum(10, redshift = true_z)
        shifted_wavelengths, spec_flux = odin_phot.redshift(true_z, waves, l_nu)
        shifted_wavelengths_normed, spec_flux_normed = odin_phot.filters.mag_norm(shifted_wavelengths, spec_flux, 18.)
        # temp_phot_wave, temp_phot_flux, temp_phot_err = odin_phot.filters.get_photometry(shifted_wavelengths_normed, spec_flux_normed, output_mags = True, dmag_err = True)

        sp.plot(shifted_wavelengths_normed, spec_flux_normed, color = 'r', zorder = 2)

        waves, l_nu = odin_phot.find_spectrum(10, redshift = 0)
        shifted_wavelengths, spec_flux = odin_phot.redshift(true_z, waves, l_nu)
        norm = spec_flux_normed[300] / spec_flux[300]
        # temp_phot_wave, temp_phot_flux, temp_phot_err = odin_phot.filters.get_photometry(shifted_wavelengths_normed, spec_flux_normed, output_mags = True, dmag_err = True)

        sp.plot(shifted_wavelengths_normed, spec_flux * norm, color = 'pink', zorder = 2)

    filter_lo = 25
    filter_hi = 25
    # sp.errorbar(photometry_waves, [0.1,] * sum(~self.filters.is_odin_filter), fmt = 'None', xerr = [filter_lo, filter_hi], ecolor = 'C0', transform = sp.get_xaxis_transform())
    # sp.errorbar(photometry_waves[self.filters.is_odin_filter], [0.13,] * sum(self.filters.is_odin_filter), fmt = 'None', xerr = [filter_lo[self.filters.is_odin_filter], filter_hi[self.filters.is_odin_filter]], ecolor = 'C0', transform = sp.get_xaxis_transform())

    # sp.scatter(photometry_waves, orig_obs_photometry, color = 'b', s = 50, zorder = 5, alpha = 0.5)
    sp.scatter(photometry_waves, temp_photometry, color = 'k', s = 75, zorder = 4)
    sp.scatter(photometry_waves, obs_photometry, color = 'None', edgecolors = 'r', s = 125, zorder = 3)

    sp.text(0.02, 0.98, '$z_{true} = %.2f$\n$z_{phot} = %.2f$' % (true_z, fit_z), fontsize = 20, ha = 'left', va = 'top', transform = sp.transAxes)

    sp.set_xlim(3000, 10000)
    # sp.set_ylim(np.median(temp_photometry)*10**-2, np.median(temp_photometry)*10**3)
    sp.set_yscale('log')

    sp.set_xlabel('Observed Wavelength (Angstrom)')
    sp.set_ylabel('Flux Density')

    return spec_flux_normed[500]/temp_fnu[500]






def gen_template_cat_randomdust(redshift = .3, templates_fp = 'SED/adam_spec_dusty.list', cat_output = stage1dir + 'templatecat.cat', magnorm = 18., Avmin = 0.0, Avmax = 1.0):

    cat_output = BPZ_dir + cat_output
    templates_fp = BPZ_dir + templates_fp

    tau = np.append(-np.logspace(-2,1,10), np.logspace(-2,1,10)[::-1])
    ages = np.linspace(.01,10,5)
    Z = np.array([0.2,1.0])
    dust = np.random.random_sample(size = len(tau)*len(ages)*len(Z)) * (Avmax-Avmin) + Avmin
    dust = dust.reshape((len(Z), len(tau), len(ages)))

    ordered_tau = []
    ordered_ages = []
    ordered_Z = []
    ordered_Av = []

    if not hasattr(redshift, '__iter__'):
        redshift = np.array([redshift,]*(len(tau)*len(ages)*len(Z)))
    elif len(redshift) != len(tau)*len(ages)*len(Z):
        print('WARNING, REDSHIFT LIST IS NOT CORRECT LENGTH')

    with open(cat_output, 'w') as writefile:

        writefile.write('# ')
        writefile.write('id'.rjust(4))
        writefile.write('  ')

        for x in range(len(phot.filters.keys)):

            writefile.write(('f_' + phot.filters.keys[x]).ljust(15))
            writefile.write(('e_' + phot.filters.keys[x]).ljust(15))

        writefile.write('\n')

        gal_id = 0

        for i, thisZ in enumerate(tqdm(Z)):
            for j, thistau in enumerate(tqdm(tau)):
                for k, thisage in enumerate(ages):
                    
                    thisdust = dust[i][j][k]
                    waves, l_nu = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, Av = thisdust)

                    thisredshift = redshift[gal_id]
                    shifted_wavelengths, spec_flux = phot.redshift(thisredshift, waves, l_nu)

                    shifted_wavelengths_normed, spec_flux_normed = phot.filters.mag_norm(shifted_wavelengths, spec_flux, magnorm)

                    # phot_wave, phot_flux, phot_err = phot.filters.get_photometry(shifted_wavelengths, spec_flux, output_mags = True, dmag_err = True)
                    phot_wave, phot_flux, phot_err = phot.filters.get_photometry(shifted_wavelengths_normed, spec_flux_normed, output_mags = True, dmag_err = True)

                    writefile.write('   %03i' % (gal_id) )
                    writefile.write('  ')

                    for this_phot_flux, this_phot_err in zip(phot_flux, phot_err):

                        writefile.write(('%.5f' % this_phot_flux).ljust(15))
                        writefile.write(('%.5f' % this_phot_err).ljust(15))

                    writefile.write('\n')

                    ordered_tau.append(thistau)
                    ordered_ages.append(thisage)
                    ordered_Z.append(thisZ)
                    ordered_Av.append(thisdust)

                    gal_id += 1


    with open('/'.join(cat_output.split('/')[:-1]) + '/index.dat', 'w') as indexfile:

        for x, (thisage, thistau, thisZ, thisAv, thisredshift) in enumerate(zip(ordered_ages, ordered_tau, ordered_Z, ordered_Av, redshift)):
            indexfile.write('  {:03}{:10.3f}{:10.3f}{:10.1f}{:10.2f}{:10.2f}'.format(x, thisage, thistau, thisZ, thisAv, thisredshift) + '\n')


def gen_template_cat_dusty(redshift = .3, templates_fp = 'SED/adam_spec_dusty.list', cat_output = stage1dir + 'templatecat.cat', magnorm = 18.):

    templates_fp = BPZ_dir + templates_fp
    cat_output = BPZ_dir + cat_output

    files = sorted(glob(templates_fp))

    template_file = open(templates_fp, 'r')
    all_lines = template_file.readlines()
    template_file.close()
    files = ['/'.join(templates_fp.split('/')[:-1]) + '/' + thisline[:-1] for thisline in all_lines if 'adamtemp' in thisline]


    if not hasattr(redshift, '__iter__'):
        redshift = np.array([redshift,]*len(files))

    with open(cat_output, 'w') as writefile:

        writefile.write('# ')
        writefile.write('id'.rjust(4))
        writefile.write('  ')

        for x in range(len(phot.filters.keys)):

            writefile.write(('f_' + phot.filters.keys[x]).ljust(15))
            writefile.write(('e_' + phot.filters.keys[x]).ljust(15))

        writefile.write('\n')

        for x, (thisfile, thisredshift) in enumerate(tqdm(zip(files, redshift))):

            thiswave, this_l_lambda = np.loadtxt(thisfile, unpack = True)
            this_lnu = this_l_lambda * thiswave**2/(3e18)
            shifted_wavelengths, spec_flux = phot.redshift(thisredshift, thiswave, this_lnu)

            shifted_wavelengths_normed, spec_flux_normed = phot.filters.mag_norm(shifted_wavelengths, spec_flux, magnorm)

            # phot_wave, phot_flux, phot_err = phot.filters.get_photometry(shifted_wavelengths, spec_flux, output_mags = True, dmag_err = True)
            phot_wave, phot_flux, phot_err = phot.filters.get_photometry(shifted_wavelengths_normed, spec_flux_normed, output_mags = True, dmag_err = True)

            writefile.write('   %03i' % x)
            writefile.write('  ')

            for this_phot_flux, this_phot_err in zip(phot_flux, phot_err):

                writefile.write(('%.5f' % this_phot_flux).ljust(15))
                writefile.write(('%.5f' % this_phot_err).ljust(15))

            writefile.write('\n')


    with open('/'.join(cat_output.split('/')[:-1]) + '/index.dat', 'w') as indexfile:

        names = [thisfile.split('/')[-1] for thisfile in files]
        ages = np.array([float(thisname.split('_')[-4][:-3]) for thisname in names])
        tau = np.array([float(thisname.split('_')[-3][:-3]) for thisname in names])
        Z = np.array([float(thisname.split('_')[-2][:-4]) for thisname in names])
        Av = np.array([float(thisname.split('_')[-1][:-6]) for thisname in names])

        for x, (thisage, thistau, thisZ, thisAv, thisredshift) in enumerate(zip(ages, tau, Z, Av, redshift)):
            indexfile.write('  {:03}{:10.3f}{:10.3f}{:10.1f}{:10.2f}{:10.2f}'.format(x, thisage, thistau, thisZ, thisAv, thisredshift) + '\n')


def gen_linear_cat_dusty(redshift = .3, templates_fp = 'SED/adam_spec_dusty.list', cat_output = stage2dir + 'templatecat.cat', magnorm = 18.):

    cat_output = BPZ_dir + cat_output
    templates_fp = BPZ_dir + templates_fp

    tau = np.append(-np.logspace(-2,1,10), np.logspace(-2,1,10)[::-1])
    ages = np.linspace(.01,10,5)
    Z = np.array([0.2,1.0])
    dust = np.array([0.05, 1.0])

    ordered_tau = []
    ordered_ages = []
    ordered_Z = []
    ordered_Av = []

    if not hasattr(redshift, '__iter__'):
        redshift = np.array([redshift,]*(len(tau)*len(ages)*len(Z)*len(dust)))
    elif len(redshift) != len(tau)*len(ages)*len(Z):
        print('WARNING, REDSHIFT LIST IS NOT CORRECT LENGTH')

    with open(cat_output, 'w') as writefile:

        writefile.write('# ')
        writefile.write('id'.rjust(4))
        writefile.write('  ')

        for x in range(len(phot.filters.keys)):

            writefile.write(('f_' + phot.filters.keys[x]).ljust(15))
            writefile.write(('e_' + phot.filters.keys[x]).ljust(15))

        writefile.write('\n')

        gal_id = 0

        for thisZ in tqdm(Z):
            for thistau in tqdm(tau):
                for thisage in ages:
                    for thisdust in dust:
                        # if ((thistau<0 and thistau >= -1.) and thisage < 1) or (thistau > 0) or (thistau < -1.): # Strongly rising SFHs get unphysically bright after doing so for long periods of time

                        waves, l_nu = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, Av = thisdust)

                        thisredshift = redshift[gal_id]
                        shifted_wavelengths, spec_flux = phot.redshift(thisredshift, waves, l_nu)

                        shifted_wavelengths_normed, spec_flux_normed = phot.filters.mag_norm(shifted_wavelengths, spec_flux, magnorm)

                        # phot_wave, phot_flux, phot_err = phot.filters.get_photometry(shifted_wavelengths, spec_flux, output_mags = True, dmag_err = True)
                        phot_wave, phot_flux, phot_err = phot.filters.get_photometry(shifted_wavelengths_normed, spec_flux_normed, output_mags = True, dmag_err = True)

                        writefile.write('   %03i' % (gal_id) )
                        writefile.write('  ')

                        for this_phot_flux, this_phot_err in zip(phot_flux, phot_err):

                            writefile.write(('%.5f' % this_phot_flux).ljust(15))
                            writefile.write(('%.5f' % this_phot_err).ljust(15))

                        writefile.write('\n')

                        ordered_tau.append(thistau)
                        ordered_ages.append(thisage)
                        ordered_Z.append(thisZ)
                        ordered_Av.append(thisdust)

                        gal_id += 1


    with open('/'.join(cat_output.split('/')[:-1]) + '/index.dat', 'w') as indexfile:

        for x, (thisage, thistau, thisZ, thisAv, thisredshift) in enumerate(zip(ordered_ages, ordered_tau, ordered_Z, ordered_Av, redshift)):
            indexfile.write('  {:03}{:10.3f}{:10.3f}{:10.1f}{:10.2f}{:10.2f}'.format(x, thisage, thistau, thisZ, thisAv, thisredshift) + '\n')


def gen_template_cat(redshift = .3, templates_fp = 'SED/adam_spec.list', cat_output = stage1dir + '/templatecat.cat', magnorm = 18.):

    cat_output = BPZ_dir + cat_output
    templates_fp = BPZ_dir + templates_fp

    files = sorted(glob(templates_fp))

    template_file = open(templates_fp, 'r')
    all_lines = template_file.readlines()
    template_file.close()
    files = ['/'.join(templates_fp.split('/')[:-1]) + '/' + thisline[:-1] for thisline in all_lines if 'adamtemp' in thisline]


    if not hasattr(redshift, '__iter__'):
        redshift = np.array([redshift,]*len(files))

    with open(cat_output, 'w') as writefile:

        writefile.write('# ')
        writefile.write('id'.rjust(4))
        writefile.write('  ')

        for x in range(len(phot.filters.keys)):

            writefile.write(('f_' + phot.filters.keys[x]).ljust(15))
            writefile.write(('e_' + phot.filters.keys[x]).ljust(15))

        writefile.write('\n')

        for x, (thisfile, thisredshift) in enumerate(tqdm(zip(files, redshift))):

            thiswave, this_l_lambda = np.loadtxt(thisfile, unpack = True)
            this_lnu = this_l_lambda * thiswave**2/(3e18)
            shifted_wavelengths, spec_flux = phot.redshift(thisredshift, thiswave, this_lnu)

            shifted_wavelengths_normed, spec_flux_normed = phot.filters.mag_norm(shifted_wavelengths, spec_flux, magnorm)

            # phot_wave, phot_flux, phot_err = phot.filters.get_photometry(shifted_wavelengths, spec_flux, output_mags = True, dmag_err = True)
            phot_wave, phot_flux, phot_err = phot.filters.get_photometry(shifted_wavelengths_normed, spec_flux_normed, output_mags = True, dmag_err = True)

            writefile.write('   %03i' % x)
            writefile.write('  ')

            for this_phot_flux, this_phot_err in zip(phot_flux, phot_err):

                writefile.write(('%.5f' % this_phot_flux).ljust(15))
                writefile.write(('%.5f' % this_phot_err).ljust(15))

            writefile.write('\n')


    with open('/'.join(cat_output.split('/')[:-1]) + '/index.dat', 'w') as indexfile:

        names = [thisfile.split('/')[-1] for thisfile in files]
        ages = np.array([float(thisname.split('_')[-3][:-3]) for thisname in names])
        tau = np.array([float(thisname.split('_')[-2][:-3]) for thisname in names])
        Z = np.array([float(thisname.split('_')[-1][:-8]) for thisname in names])

        for x, (thisage, thistau, thisZ, thisredshift) in enumerate(zip(ages, tau, Z, redshift)):
            indexfile.write('  {:03}{:10.3f}{:10.3f}{:10.1f}{:10.2f}'.format(x, thisage, thistau, thisZ, thisredshift) + '\n')


def gen_linear_cat(redshift = .3, templates_fp = 'SED/adam_spec.list', cat_output = stage2dir + 'templatecat.cat', magnorm = 18.):

    cat_output = BPZ_dir + cat_output
    templates_fp = BPZ_dir + templates_fp

    tau = np.append(-np.logspace(-2,1,10), np.logspace(-2,1,10)[::-1])
    ages = np.linspace(.01,10,5)
    Z = np.array([0.2,1.0])

    ordered_tau = []
    ordered_ages = []
    ordered_Z = []

    if not hasattr(redshift, '__iter__'):
        redshift = np.array([redshift,]*(len(tau)*len(ages)*len(Z)))
    elif len(redshift) != len(tau)*len(ages)*len(Z):
        print('WARNING, REDSHIFT LIST IS NOT CORRECT LENGTH')

    with open(cat_output, 'w') as writefile:

        writefile.write('# ')
        writefile.write('id'.rjust(4))
        writefile.write('  ')

        for x in range(len(phot.filters.keys)):

            writefile.write(('f_' + phot.filters.keys[x]).ljust(15))
            writefile.write(('e_' + phot.filters.keys[x]).ljust(15))

        writefile.write('\n')

        gal_id = 0

        for i, thisZ in enumerate(tqdm(Z)):
            for j, thistau in enumerate(tqdm(tau)):
                for k, thisage in enumerate(ages):
                    # if ((thistau<0 and thistau >= -1.) and thisage < 1) or (thistau > 0) or (thistau < -1.): # Strongly rising SFHs get unphysically bright after doing so for long periods of time

                    waves, l_nu = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau})

                    thisredshift = redshift[gal_id]
                    shifted_wavelengths, spec_flux = phot.redshift(thisredshift, waves, l_nu)

                    shifted_wavelengths_normed, spec_flux_normed = phot.filters.mag_norm(shifted_wavelengths, spec_flux, magnorm)

                    # phot_wave, phot_flux, phot_err = phot.filters.get_photometry(shifted_wavelengths, spec_flux, output_mags = True, dmag_err = True)
                    phot_wave, phot_flux, phot_err = phot.filters.get_photometry(shifted_wavelengths_normed, spec_flux_normed, output_mags = True, dmag_err = True)

                    writefile.write('   %03i' % (gal_id) )
                    writefile.write('  ')

                    for this_phot_flux, this_phot_err in zip(phot_flux, phot_err):

                        writefile.write(('%.5f' % this_phot_flux).ljust(15))
                        writefile.write(('%.5f' % this_phot_err).ljust(15))

                    writefile.write('\n')

                    ordered_tau.append(thistau)
                    ordered_ages.append(thisage)
                    ordered_Z.append(thisZ)

                    gal_id += 1


    with open('/'.join(cat_output.split('/')[:-1]) + '/index.dat', 'w') as indexfile:

        for x, (thisage, thistau, thisZ, thisredshift) in enumerate(zip(ordered_ages, ordered_tau, ordered_Z, redshift)):
            indexfile.write('  {:03}{:10.3f}{:10.3f}{:10.1f}{:10.2f}'.format(x, thisage, thistau, thisZ, thisredshift) + '\n')



def hsc_z_cum_dist(inputfp = 'HSC/HSC_wide_clean.fits', sn_lim = 30., unique = True, zlim = 1.5):

    _, _, _, sz, _, _ = hsc_select(inputfp, None, unique, None, sn_lim, zlim)

    cumdistx = np.sort(sz)
    cumdisty = np.arange(len(sz))/float(len(sz))

    return cumdistx, cumdisty



def hsc_select(inputfp = 'HSC/HSC_wide_clean.fits', size = None, unique = True, maglim = None, sn_lim = 10., zlim = 1.5, notzero = True):

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


    specz_lim = np.where((sz >0.) & (sz < 9))[0]
        
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


    mags = np.vstack((data_table['g_mag'], data_table['r_mag'], data_table['i_mag'], data_table['z_mag'], data_table['y_mag']))
    errs = np.vstack((data_table['g_magsigma'], data_table['r_magsigma'], data_table['i_magsigma'], data_table['z_magsigma'], data_table['y_magsigma']))
        
    goodinds = ~np.isnan(mags).any(axis=0) & ~np.isnan(errs).any(axis=0)

    mags = mags[:,goodinds]
    errs = errs[:,goodinds]

    indices = indices[goodinds]
    ID = ID[goodinds]
    pz = pz[goodinds]
    sz = sz[goodinds]
    
    if size != None:

        selections = np.random.choice(np.arange(len(ID)), size = size, replace = False)
        indices = indices[selections]
        ID = ID[selections]
        pz = pz[selections]
        sz = sz[selections]
        mags = mags[:,selections]
        errs = errs[:,selections]

    return indices, ID, pz, sz, mags, errs



def gen_hsclike_cat(size = 1000, templates_fp = 'SED/adam_spec_dusty.list', cat_output = '/HSC_like/templatecat.cat', magnorm = 18.):
    
    templates_fp = BPZ_dir + templates_fp
    cat_output = BPZ_dir + cat_output

    template_file = open(templates_fp, 'r')
    all_lines = template_file.readlines()
    template_file.close()
    files = ['/'.join(templates_fp.split('/')[:-1]) + '/' + thisline[:-1] for thisline in all_lines if 'adamtemp' in thisline]

    names = [thisfile.split('/')[-1] for thisfile in files]
    ages = np.array([float(thisname.split('_')[-4][:-3]) for thisname in names])
    tau = np.array([float(thisname.split('_')[-3][:-3]) for thisname in names])
    Z = np.array([float(thisname.split('_')[-2][:-4]) for thisname in names])
    Av = np.array([float(thisname.split('_')[-1][:-6]) for thisname in names])

    cumdist_x, cumdist_y = hsc_z_cum_dist()

    redshift = np.interp(np.random.random_sample(size = size), cumdist_y, cumdist_x)

    # =======================
    # Generate the templates
    # =======================    


    template_lnu = []
    template_waves = None

    for thisZ in tqdm(Z):
        for thistau in tqdm(tau):
            for thisage in ages:
                for thisAv in Av:
                    waves, l_nu = phot.find_spectrum(tage = thisage, metallicity = thisZ, sfh_type = 6, sfh_params = {'tau': thistau}, Av = thisdust)

                    templates_lnu.append(l_nu)
                    if not hasattr(template_waves, '__iter__'):
                        template_waves = waves

    for thisredshift in redshift:
        pass

        # Choose template based on a reference magnitude and mag variance?












class result:
    def __init__(self, runfp = current_stagedir, resultfile =  'templatecat.bpz'):

        runfp = BPZ_dir + runfp

        self.bpz_fp = runfp + resultfile
        self.runfp = runfp

        bpz_file = open(self.bpz_fp, 'r')
        all_lines = bpz_file.readlines()
        bpz_file.close()

        self.params = {}

        for line in all_lines:
            if '=' in line:
                key, value = line.split('=')
                key = key.replace('#', '')
                value = value[:-1]

                try:
                    value = int(value)
                except:
                    try:
                        value = float(value)
                    except:
                        pass

                self.params[key] = value

        template_file = open(self.params['SED_DIR'] + self.params['SPECTRA'], 'r')
        all_lines = template_file.readlines()
        template_file.close()
        self.template_name = [thisline.split('/')[-1][:-1] for thisline in all_lines]
        self.template_id = np.arange(0, len(self.template_name)) + 1
        self.template_id_interp = np.linspace(0, len(self.template_name), len(self.template_name) + (len(self.template_name)-1) * self.params['INTERP']) + 1


        indexfile = runfp + 'index.dat'

        if 'dusty' in self.params['SPECTRA']:
            self.id, self.age, self.tau, self.Z, self.Av, self.z_in = np.loadtxt(indexfile, unpack = True)
        else:
            self.id, self.age, self.tau, self.Z, self.z_in = np.loadtxt(indexfile, unpack = True)
            self.Av = np.zeros(len(self.id))
        self.id = self.id.astype(int)

        self.bpz_id, self.z_b, self.z_b_min, self.z_b_max, self.t_b, self.odds, self.z_ml, self.t_ml, self.chisq, self.m_0 = np.loadtxt(self.bpz_fp, unpack = True)

        self.agevals = np.array(sorted(np.unique(self.age)))
        self.tauvals = self.tausort(np.unique(self.tau))
        self.Zvals = np.array(sorted(np.unique(self.Z)))


    def tausort(self, arr):

        neg = list(reversed(sorted(arr[arr < 0])))
        pos = list(reversed(sorted(arr[arr >= 0])))

        return np.array(neg+pos)


    def plot_results_z(self):

        fig = plt.figure(figsize = (16,8))
        sp1 = fig.add_subplot(121)
        sp2 = fig.add_subplot(122)

        agegrid, taugrid = np.meshgrid(self.agevals, self.tauvals)

        cmap = plt.cm.coolwarm
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[127] = 'mediumseagreen'
        cmaplist[128] = 'mediumseagreen'
        newcmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        bounds = np.linspace(-.4,.4,10)
        norm = mpl.colors.BoundaryNorm(bounds, newcmap.N)

        for metallicity, sp in zip(np.unique(self.Z), [sp1, sp2]):

            grid = np.empty((len(self.tauvals), len(np.unique(self.agevals))))
            grid.fill(np.nan)
            antigrid = np.empty(grid.shape)
            antigrid.fill(np.nan)

            thismetallicity = np.where(self.Z == metallicity)

            for thisage, thistau, input_z, best_z in zip(self.age[thismetallicity], self.tau[thismetallicity], self.z_in[thismetallicity], self.z_b[thismetallicity]):

                grid[np.where((agegrid == thisage) & (taugrid == thistau))] = (input_z - best_z) / (1.+input_z)

            antigrid[np.isnan(grid)] = 0.8

            background = sp.imshow(antigrid, origin = 'lower', cmap = mpl.cm.binary, extent = [min(self.agevals), max(self.agevals), min(self.tauvals), max(self.tauvals)], aspect = 'auto', vmin = 0, vmax = 1)

            img = sp.imshow(grid, origin = 'lower', cmap = newcmap, norm = norm, extent = [min(self.agevals), max(self.agevals), min(self.tauvals), max(self.tauvals)], aspect = 'auto')

            # img = sp.pcolormesh(agegrid, taugrid, grid)

            
            yticks = np.linspace(min(self.tauvals), max(self.tauvals), len(self.tauvals)+1)
            yticks += .5*(yticks[1]-yticks[0])
            sp.set_yticks(yticks)
            sp.set_ylim(min(self.tauvals), max(self.tauvals))

            xticks = np.linspace(min(self.agevals), max(self.agevals), len(self.agevals)+1)
            xticks += .5*(xticks[1]-xticks[0])
            sp.set_xticks(xticks)
            sp.set_xlim(min(self.agevals), max(self.agevals))

            txt = sp.text(0.02, 0.02, r'$Z/Z_\odot={:.1f}$'.format(metallicity), transform = sp.transAxes, fontsize = 40, color = 'white')
            txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='k')])
        
        sp1.set_yticklabels(self.tauvals)
        sp2.set_yticklabels([])

        sp1.set_xticklabels(['%.2f' % thisnum for thisnum in np.log10(self.agevals)])
        sp2.set_xticklabels(['%.2f' % thisnum for thisnum in np.log10(self.agevals)])

        fig.subplots_adjust(wspace = 0.)
        fig.text(0.5, 0.04, 'Log(Age/Gyr)', fontsize = 20)

        # cbar = fig.colorbar(img, ax = sp2)
        cbax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
        cb = mpl.colorbar.ColorbarBase(cbax, cmap = newcmap, norm = norm, 
            spacing = 'proportional', ticks = bounds, boundaries = bounds, format = '%.2f')
        fig.text(.98, 0.5, r'$(z_\mathrm{in}-z_\mathrm{b})/(1+z_\mathrm{in})$', fontsize = 20, rotation = 'vertical', ha = 'left', va = 'center')
        sp1.set_ylabel(r'$\tau$ (Gyr)', fontsize = 20)




    def plot_results_chi2(self, vmin = 0, vmax = 3):

        fig = plt.figure(figsize = (16,8))
        sp1 = fig.add_subplot(121)
        sp2 = fig.add_subplot(122)

        norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)

        agegrid, taugrid = np.meshgrid(self.agevals, self.tauvals)

        # cmap = plt.cm.coolwarm
        # cmaplist = [cmap(i) for i in range(cmap.N)]
        # cmaplist[127] = 'mediumseagreen'
        # cmaplist[128] = 'mediumseagreen'
        # newcmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        # bounds = np.linspace(-.4,.4,10)
        # norm = mpl.colors.BoundaryNorm(bounds, newcmap.N)

        for metallicity, sp in zip(np.unique(self.Z), [sp1, sp2]):

            grid = np.empty((len(self.tauvals), len(np.unique(self.agevals))))
            grid.fill(np.nan)
            antigrid = np.empty(grid.shape)
            antigrid.fill(np.nan)

            thismetallicity = np.where(self.Z == metallicity)

            for thisage, thistau, chisq in zip(self.age[thismetallicity], self.tau[thismetallicity], self.chisq[thismetallicity]):

                grid[np.where((agegrid == thisage) & (taugrid == thistau))] = np.log10(chisq)

            # antigrid[np.isnan(grid)] = 0.8

            # background = sp.imshow(antigrid, origin = 'lower', cmap = mpl.cm.binary, extent = [min(self.agevals), max(self.agevals), min(self.tauvals), max(self.tauvals)], aspect = 'auto', vmin = 0, vmax = 1)

            img = sp.imshow(grid, origin = 'lower', cmap = 'summer', extent = [min(self.agevals), max(self.agevals), min(self.tauvals), max(self.tauvals)], norm = norm, aspect = 'auto')

            # img = sp.pcolormesh(agegrid, taugrid, grid)

            
            yticks = np.linspace(min(self.tauvals), max(self.tauvals), len(self.tauvals)+1)
            yticks += .5*(yticks[1]-yticks[0])
            sp.set_yticks(yticks)
            sp.set_ylim(min(self.tauvals), max(self.tauvals))

            xticks = np.linspace(min(self.agevals), max(self.agevals), len(self.agevals)+1)
            xticks += .5*(xticks[1]-xticks[0])
            sp.set_xticks(xticks)
            sp.set_xlim(min(self.agevals), max(self.agevals))

            txt = sp.text(0.02, 0.02, r'$Z/Z_\odot={:.1f}$'.format(metallicity), transform = sp.transAxes, fontsize = 40, color = 'white')
            txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='k')])
        
        sp1.set_yticklabels(self.tauvals)
        sp2.set_yticklabels([])

        sp1.set_xticklabels(['%.2f' % thisnum for thisnum in np.log10(self.agevals)])
        sp2.set_xticklabels(['%.2f' % thisnum for thisnum in np.log10(self.agevals)])

        fig.subplots_adjust(wspace = 0.)
        fig.text(0.5, 0.04, 'Log(Age/Gyr)', fontsize = 20)

        # cbar = fig.colorbar(img, ax = sp2)
        cbax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
        # cb = mpl.colorbar.ColorbarBase(cbax, cmap = newcmap, norm = norm, 
        #     spacing = 'proportional', ticks = bounds, boundaries = bounds, format = '%.2f')
        cb = mpl.colorbar.ColorbarBase(cbax, cmap = 'summer', format = '%.2f', norm = norm)
        fig.text(.98, 0.5, r'$\log_{10}(\chi^2)$', fontsize = 20, rotation = 'vertical', ha = 'left', va = 'center')
        sp1.set_ylabel(r'$\tau$ (Gyr)', fontsize = 20)


    def plot_prior(self, galid):

        shelve_file = self.runfp + self.params['INPUT'].split('.')[0] + '.full_probs'

        # sed_dir = self.params['SED_DIR']
        # spectrafile = self.params['SPECTRA']
        # catfile = self.params['INPUT']
        # interp = self.params['INTERP']

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        data_file = shelve.open(shelve_file)
        data = data_file[str(galid).zfill(3)]
        data_file.close()
        zbins = data[0]
        priors = data[1]

        [sp.plot(zbins, thisprior*(1.+.1*x), label = '{:03}'.format(thistemp_id)) for x, (thisprior, thistemp_id) in enumerate(zip(priors.T[::self.params['INTERP']+1], self.template_id))]

        sp.set_xlabel('Redshift (z)')
        sp.set_ylabel('Prior')



    def plot_chi2(self, xaxis = 'age'):

        if xaxis == 'age':
            xlabel = 'Age'
            xvals = self.agevals
            xdata = self.age

        elif xaxis == 'tau':
            xlabel = r'$\tau$'
            xvals = self.tauvals
            xdata = self.tau

        data = []

        for value in xvals:

            data.append(self.chisq[xdata == value])

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        sp.boxplot(data)

        sp.set_xticklabels(xvals)

        sp.set_xlabel(xlabel)
        sp.set_ylabel(r'$\chi^2$')


    def plot_template_scatter(self):

        if self.runfp != stage1dir:
            print('This plot only works for stage1, bud')
        
        else:

            fig = plt.figure(figsize = (8,8))
            sp = fig.add_subplot(111)

            sp.scatter(range(len(self.t_b)), self.t_b)

            sp.set_xlabel('Input Template')
            sp.set_ylabel('Fit Template')


    def plot_dz(self):

        dz = (self.z_b - self.z_in)/(1+self.z_in)

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        sp.scatter(self.id[self.Av > .5], dz[self.Av > .5], c = 'r', marker = '.')
        sp.scatter(self.id[self.Av < .5], dz[self.Av < .5], c = 'b', marker = '.')

        sp.set_xlabel('Template ID')
        sp.set_ylabel(r'$\frac{z_b - z_{in}}{1+z_{in}}$')




class result_odin:
    def __init__(self, runfp = 'ODIN/', resultfile =  'odin_filters.bpz', err_mult = 10., eel_bands = 1, minimum_excursion = 0.00):

        runfp = BPZ_dir + runfp

        self.bpz_fp = runfp + resultfile
        self.runfp = runfp

        self.filters = odin_phot.filters

        self.eel_bands = eel_bands
        self.err_mult = err_mult
        self.minimum_excursion = minimum_excursion

        bpz_file = open(self.bpz_fp, 'r')
        all_lines = bpz_file.readlines()
        bpz_file.close()

        self.g_mag, self.r_mag, self.i_mag, self.z_mag, self.y_mag = np.loadtxt(self.bpz_fp[:-3] + 'cat', usecols = [1,3,5,7,9], unpack = True)

        self.params = {}

        for line in all_lines:
            if '=' in line:
                key, value = line.split('=')
                key = key.replace('#', '')
                value = value[:-1]

                try:
                    value = int(value)
                except:
                    try:
                        value = float(value)
                    except:
                        pass

                self.params[key] = value

        flux_compare_file = runfp + resultfile.split('.')[0] + '.flux_comparison'

        convert_to_cgs = 3.63134249641159e-20

        self.fit_norm = np.loadtxt(flux_compare_file, usecols = [4]) * convert_to_cgs
        self.fit_flux = np.loadtxt(flux_compare_file, usecols = np.arange(5,5+len(self.filters.keys))) * convert_to_cgs
        self.obs_flux = np.loadtxt(flux_compare_file, usecols = np.arange(5+len(self.filters.keys),5+2*len(self.filters.keys))) * convert_to_cgs
        self.obs_flux_err = np.loadtxt(flux_compare_file, usecols = np.arange(5+2*len(self.filters.keys),5+3*len(self.filters.keys))) * convert_to_cgs

        template_file = open(self.params['SED_DIR'] + self.params['SPECTRA'], 'r')
        all_lines = template_file.readlines()
        template_file.close()
        self.template_name = np.array([thisline.split('/')[-1][:-1] for thisline in all_lines])
        self.template_id = np.arange(0, len(self.template_name)) + 1
        self.template_id_interp = np.linspace(0, len(self.template_name)-1, len(self.template_name) + (len(self.template_name)-1) * self.params['INTERP']) + 1

        pz_T = np.loadtxt(self.runfp + self.params['PROBS_LITE'], unpack = True)[1:]
        self.pz = np.array(pz_T).T
        with open(self.runfp + self.params['PROBS_LITE'], 'r') as readfile:
            firstline = readfile.readline()

        beg, end, sep = firstline.split('(')[-1].split(')')[0].split(',')
        self.pz_z = np.arange(float(beg), float(end), float(sep))

        indexfile = runfp + 'index.dat'

        self.id = np.loadtxt(indexfile, usecols = [0], dtype = int)

        self.id, self.age, self.tau, self.Z, self.Av, self.z_in = np.loadtxt(indexfile, unpack = True)
        self.id = self.id.astype(int)

        self.bpz_id = np.loadtxt(self.bpz_fp, usecols = [0], dtype = int)
        self.z_b, self.z_b_min, self.z_b_max, self.t_b, self.odds, self.z_ml, self.t_ml, self.chisq, self.m_0 = np.loadtxt(self.bpz_fp, unpack = True, usecols = np.arange(1,10))
        
        # self.t_b = self.t_b.astype(int)


        # self.sigmafile = './cache/' + self.runfp.split('/')[-2] + '.eel'
        # self.excursionfile = self.sigmafile[:-3] + 'me'

        # if not os.path.isfile(self.sigmafile):
        #     sigmaconf, multiplicative_excursion = self.get_eels(np.arange(len(self.id)))
        #     np.savetxt(self.sigmafile, sigmaconf, fmt = '%.8e '*5)
        #     np.savetxt(self.excursionfile, multiplicative_excursion, fmt = '%.2f '*5)
  
        # self.sigmaconf = np.loadtxt(self.sigmafile)
        # self.multiplicative_excursion = np.loadtxt(self.excursionfile)
        # self.set_eels(err_mult, eel_bands, minimum_excursion)
        # # Note this also excludes galaxies with z<0.4
        # # From Zhou+2020 https://arxiv.org/pdf/2001.06018.pdf
        # self.islrg = (self.z_mag > 20.41) & (self.r_mag - self.z_mag > (self.z_mag - 17.18)/2.) & (self.r_mag - self.z_mag > 0.9) & ((self.r_mag - self.z_mag > 1.15) | (self.g_mag - self.r_mag > 1.65))

        # maxsigma = np.max(self.sigmaconf, axis = 1)
        # self.eelstat = np.log10(maxsigma)
        # self.eelstat[maxsigma < 1] = self.eelstat[maxsigma < 1]/3.


        # ==============================================
        #  Read in Various Versions of Best Fit Spectra
        # ==============================================

        print('Loading Templates...')

        self.tempflux = []
        self.tempwave = []
        self.tempflux_ne = []
        self.tempwave_ne = []
        self.tempflux_med = []
        self.tempwave_med = []

        for thisname in tqdm(self.template_name):

            temp = np.loadtxt(self.params['SED_DIR'] + thisname, unpack = True)
            temp_ne = np.loadtxt(self.params['SED_DIR'] + thisname[:-4] + '_noemline.sed', unpack = True)
            temp_med = np.loadtxt(self.params['SED_DIR'] + thisname[:-4] + '_med.sed', unpack = True)

            self.tempwave.append(temp[0])
            self.tempflux.append(temp[1])
            self.tempwave_ne.append(temp_ne[0])
            self.tempflux_ne.append(temp_ne[1])
            self.tempwave_med.append(temp_med[0])
            self.tempflux_med.append(temp_med[1])

        self.tempwave = np.array(self.tempwave)
        self.tempflux = np.array(self.tempflux)
        self.tempwave_ne = np.array(self.tempwave_ne)
        self.tempflux_ne = np.array(self.tempflux_ne)
        self.tempwave_med = np.array(self.tempwave_med)
        self.tempflux_med = np.array(self.tempflux_med)





    def get_eels(self, gal_index):

        data = fits.open('HSC/HSC_wide_clean.fits')[1].data

        if not hasattr(gal_index, '__iter__'):
            gal_index = [gal_index]


        sigma = []
        multiplicative_excursion = []


        for x, (thisid, thisredshift, thistemplate, thishsc_index, thisfitnorm) in enumerate(tqdm(zip(self.id[gal_index], self.z_b[gal_index], self.template_name[self.t_b[gal_index].astype(int) -1], self.hsc_index[gal_index], self.fit_norm[gal_index]), total = len(gal_index))):

            # Get the HSC fluxes and flux errors so we know the fractional bump needed from the emission lines

            fluxes = np.array([data[thishsc_index][band + 'cmodel_flux'] for band in ['g', 'r', 'i', 'z', 'y']])
            flux_errs = np.array([data[thishsc_index][band + 'cmodel_flux_err'] for band in ['g', 'r', 'i', 'z', 'y']])
            err_fraction = flux_errs/fluxes

            # Find the flux density in each filter when there are emission lines
            wave, spec = np.loadtxt(self.params['SED_DIR'] + thistemplate, unpack = True)
            spec_fnu = (1+thisredshift)**2 * thisfitnorm * spec * wave**2 / 3e18
            shifted_wavelengths, _ = phot.redshift(thisredshift, wave, spec_fnu)
            phot_wave, phot_flux = self.filters.get_photometry(shifted_wavelengths, spec_fnu, output_mags = False)

            # Do the same for the case with no emission lines
            wave_ne, spec_ne = np.loadtxt(self.params['SED_DIR'] + thistemplate[:-4] + '_noemline.sed', unpack = True)
            spec_ne_fnu = (1+thisredshift)**2 * thisfitnorm * spec_ne * wave_ne**2 / 3e18
            shifted_wavelengths_ne, _ = phot.redshift(thisredshift, wave_ne, spec_ne_fnu)
            phot_wave_ne, phot_flux_ne = self.filters.get_photometry(shifted_wavelengths_ne, spec_ne_fnu, output_mags = False)

            # sigma.append(((phot_flux/phot_flux_ne) - 1)/err_fraction)
            sigma.append((phot_flux-phot_flux_ne)/flux_errs)
            multiplicative_excursion.append(phot_flux/phot_flux_ne)



        return np.array(sigma), np.array(multiplicative_excursion)




    def set_eels(self, err_mult, eel_bands, minimum_excursion):

        self.err_mult = err_mult
        self.eel_bands = eel_bands
        self.minimum_excursion = minimum_excursion

        if eel_bands == 1:
            self.iseel = np.array([any((thisgalsigma > err_mult) & (thisgal_me > 1. + minimum_excursion)) for (thisgalsigma, thisgal_me) in zip(self.sigmaconf, self.multiplicative_excursion)])
        else:
            num_eel_bands = np.sum(((self.sigmaconf > err_mult) & (self.multiplicative_excursion > 1. + minimum_excursion)), axis = 1)
            self.iseel = num_eel_bands >= eel_bands


    def plot_fit(self, gal_id, plot_true_spec = True, subplot = None):

        fit_template = int(self.t_b[gal_id])
        fit_z = self.z_b[gal_id]
        true_z = self.z_in[gal_id]

        photometry_waves = np.array([self.filters.wavelength_centers[thiskey] for thiskey in self.filters.keys])
        temp_photometry = self.fit_flux[gal_id]
        obs_photometry = self.obs_flux[gal_id]
        tempwave = self.tempwave[fit_template-1]*(1+fit_z)
        temp_fnu = self.tempflux[fit_template-1] * self.fit_norm[gal_id] * tempwave**2 / 3e18

        if subplot == None:
            fig = plt.figure(figsize = (8,8))
            sp = fig.add_subplot(111)
        else:
            sp = subplot

        sp.plot(np.array([1216., 1216.])*(1+fit_z), [0,1], color = 'C1', linestyle = '--', transform = sp.get_xaxis_transform())
        sp.plot(np.array([1216., 1216.])*(1+true_z), [0,1], color = 'C1', transform = sp.get_xaxis_transform())

        if fit_template != 0:
            sp.plot(tempwave, temp_fnu, color = 'k', zorder = 1)

        # sp.plot(self.tempwave[fit_template]*(1+fit_z), self.tempflux[fit_template] * self.fit_norm[gal_id] * tempwave**2 / 3e18, color = 'g')

        if plot_true_spec:
            waves, l_nu = odin_phot.find_spectrum(tage = self.age[gal_id], metallicity = self.Z[gal_id], sfh_type = 6, sfh_params = {'tau': self.tau[gal_id]}, Av = self.Av[gal_id], redshift = self.z_in[gal_id])
            shifted_wavelengths, spec_flux = odin_phot.redshift(self.z_in[gal_id], waves, l_nu)
            shifted_wavelengths_normed, spec_flux_normed = odin_phot.filters.mag_norm(shifted_wavelengths, spec_flux, 18.)
            # temp_phot_wave, temp_phot_flux, temp_phot_err = odin_phot.filters.get_photometry(shifted_wavelengths_normed, spec_flux_normed, output_mags = True, dmag_err = True)

            sp.plot(shifted_wavelengths_normed, spec_flux_normed, color = 'r', zorder = 2)

        filter_lo = np.array([self.filters.wavelength_centers[thiskey] - self.filters.halfmax_lo[thiskey] for thiskey in self.filters.keys])
        filter_hi = np.array([self.filters.halfmax_hi[thiskey] - self.filters.wavelength_centers[thiskey] for thiskey in self.filters.keys])
        sp.errorbar(photometry_waves[~self.filters.is_odin_filter], [0.1,] * sum(~self.filters.is_odin_filter), fmt = 'None', xerr = [filter_lo[~self.filters.is_odin_filter], filter_hi[~self.filters.is_odin_filter]], ecolor = 'C0', transform = sp.get_xaxis_transform())
        sp.errorbar(photometry_waves[self.filters.is_odin_filter], [0.13,] * sum(self.filters.is_odin_filter), fmt = 'None', xerr = [filter_lo[self.filters.is_odin_filter], filter_hi[self.filters.is_odin_filter]], ecolor = 'C0', transform = sp.get_xaxis_transform())

        for this_wave, thiskey, is_odin_filter in zip(photometry_waves, self.filters.keys, self.filters.is_odin_filter):

            if is_odin_filter:
                offset = 0.14
            else:
                offset = 0.11

            sp.text(this_wave, offset, thiskey, fontsize = 16, color = 'C0', ha = 'center', va = 'bottom', transform = sp.get_xaxis_transform())

        if fit_template != 0:
            sp.scatter(photometry_waves, temp_photometry, color = 'k', s = 75, zorder = 4)
        sp.scatter(photometry_waves, obs_photometry, color = 'None', edgecolors = 'r', s = 125, zorder = 3)

        sp.text(0.02, 0.98, '$z_{true} = %.2f$\n$z_{phot} = %.2f$' % (true_z, fit_z), fontsize = 20, ha = 'left', va = 'top', transform = sp.transAxes)

        sp.set_xlim(3000, 10000)
        sp.set_ylim(np.median(temp_photometry)*10**-2, np.median(temp_photometry)*10**3)
        sp.set_yscale('log')

        sp.set_xlabel('Observed Wavelength (Angstrom)')
        sp.set_ylabel('Flux Density')


    def plot_all_fits(self):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        for this_id in tqdm(range(len(self.z_in))):
            if self.t_b[this_id]!=0:
                try:
                    self.plot_fit(this_id, plot_true_spec=True, subplot=sp)
                except:
                    print(this_id)
                    break
                plt.savefig('./Figures/ODIN_fits/%05i.png' % this_id, bbox_inches = 'tight')
                sp.cla()

        plt.close()



    def plot_prior(self, galid):

        shelve_file = self.runfp + self.params['INPUT'].split('.')[0] + '.full_probs'

        # sed_dir = self.params['SED_DIR']
        # spectrafile = self.params['SPECTRA']
        # catfile = self.params['INPUT']
        # interp = self.params['INTERP']

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        data_file = shelve.open(shelve_file)
        data = data_file[str(galid).zfill(3)]
        data_file.close()
        zbins = data[0]
        priors = data[1]

        [sp.plot(zbins, thisprior*(1.+.1*x), label = '{:03}'.format(thistemp_id)) for x, (thisprior, thistemp_id) in enumerate(zip(priors.T[::self.params['INTERP']+1], self.template_id))]

        sp.set_xlabel('Redshift (z)')
        sp.set_ylabel('Prior')






class result_hsc:
    def __init__(self, runfp = 'HSC10_extra_res/', resultfile =  'templatecat.bpz', err_mult = 10., eel_bands = 1, minimum_excursion = 0.00):

        runfp = BPZ_dir + runfp

        self.bpz_fp = runfp + resultfile
        self.runfp = runfp

        self.eel_bands = eel_bands
        self.err_mult = err_mult
        self.minimum_excursion = minimum_excursion

        bpz_file = open(self.bpz_fp, 'r')
        all_lines = bpz_file.readlines()
        bpz_file.close()

        self.g_mag, self.r_mag, self.i_mag, self.z_mag, self.y_mag = np.loadtxt(self.bpz_fp[:-3] + 'cat', usecols = [1,3,5,7,9], unpack = True)

        self.params = {}

        for line in all_lines:
            if '=' in line:
                key, value = line.split('=')
                key = key.replace('#', '')
                value = value[:-1]

                try:
                    value = int(value)
                except:
                    try:
                        value = float(value)
                    except:
                        pass

                self.params[key] = value

        flux_compare_file = runfp + resultfile.split('.')[0] + '.flux_comparison'

        convert_to_cgs = 3.63134249641159e-20

        self.fit_norm = np.loadtxt(flux_compare_file, usecols = [4]) * convert_to_cgs
        self.fit_flux = np.loadtxt(flux_compare_file, usecols = np.arange(5,10)) * convert_to_cgs
        self.obs_flux = np.loadtxt(flux_compare_file, usecols = np.arange(10,15)) * convert_to_cgs
        self.obs_flux_err = np.loadtxt(flux_compare_file, usecols = np.arange(15,20)) * convert_to_cgs

        template_file = open(self.params['SED_DIR'] + self.params['SPECTRA'], 'r')
        all_lines = template_file.readlines()
        template_file.close()
        self.template_name = np.array([thisline.split('/')[-1][:-1] for thisline in all_lines])
        self.template_id = np.arange(0, len(self.template_name)) + 1
        self.template_id_interp = np.linspace(0, len(self.template_name)-1, len(self.template_name) + (len(self.template_name)-1) * self.params['INTERP']) + 1

        pz_T = np.loadtxt(self.runfp + self.params['PROBS_LITE'], unpack = True)[1:]
        self.pz = np.array(pz_T).T
        with open(self.runfp + self.params['PROBS_LITE'], 'r') as readfile:
            firstline = readfile.readline()

        beg, end, sep = firstline.split('(')[-1].split(')')[0].split(',')
        self.pz_z = np.arange(float(beg), float(end), float(sep))

        indexfile = runfp + 'index.dat'

        self.id = np.loadtxt(indexfile, usecols = [0], dtype = int)

        self.z_hsc, self.z_spec, self.hsc_index = np.loadtxt(indexfile, unpack = True, usecols = [1,2,3])
        self.id = self.id.astype(int)
        self.hsc_index = self.hsc_index.astype(int)

        self.bpz_id = np.loadtxt(self.bpz_fp, usecols = [0], dtype = int)
        self.z_b, self.z_b_min, self.z_b_max, self.t_b, self.odds, self.z_ml, self.t_ml, self.chisq, self.m_0 = np.loadtxt(self.bpz_fp, unpack = True, usecols = np.arange(1,10))
        
        self.t_b = self.t_b.astype(int)

        self.filters = filtersim.odin_filtersim()

        self.sigmafile = './cache/' + self.runfp.split('/')[-2] + '.eel'
        self.excursionfile = self.sigmafile[:-3] + 'me'

        if not os.path.isfile(self.sigmafile):
            sigmaconf, multiplicative_excursion = self.get_eels(np.arange(len(self.id)))
            np.savetxt(self.sigmafile, sigmaconf, fmt = '%.8e '*5)
            np.savetxt(self.excursionfile, multiplicative_excursion, fmt = '%.2f '*5)
  
        self.sigmaconf = np.loadtxt(self.sigmafile)
        self.multiplicative_excursion = np.loadtxt(self.excursionfile)
        self.set_eels(err_mult, eel_bands, minimum_excursion)
        # Note this also excludes galaxies with z<0.4
        # From Zhou+2020 https://arxiv.org/pdf/2001.06018.pdf
        self.islrg = (self.z_mag > 20.41) & (self.r_mag - self.z_mag > (self.z_mag - 17.18)/2.) & (self.r_mag - self.z_mag > 0.9) & ((self.r_mag - self.z_mag > 1.15) | (self.g_mag - self.r_mag > 1.65))

        maxsigma = np.max(self.sigmaconf, axis = 1)
        # maxsigma[maxsigma < 1] = maxsigma[maxsigma < 1]/3.
        self.eelstat = np.log10(maxsigma)
        self.eelstat[maxsigma < 1] = self.eelstat[maxsigma < 1]/3.
        # self.eelstat[maxsigma < 1] = self.eelstat[maxsigma < 1]/3. - 1
        # self.eelstat[maxsigma >= 1] = (self.eelstat[maxsigma >= 1] - 1) * 2.


        # ==============================================
        #  Read in Various Versions of Best Fit Spectra
        # ==============================================

        print('Loading Templates...')

        self.tempflux = []
        self.tempwave = []
        self.tempflux_ne = []
        self.tempwave_ne = []
        self.tempflux_med = []
        self.tempwave_med = []

        for thisname in tqdm(self.template_name):

            temp = np.loadtxt(self.params['SED_DIR'] + thisname, unpack = True)
            temp_ne = np.loadtxt(self.params['SED_DIR'] + thisname[:-4] + '_noemline.sed', unpack = True)
            temp_med = np.loadtxt(self.params['SED_DIR'] + thisname[:-4] + '_med.sed', unpack = True)

            self.tempwave.append(temp[0])
            self.tempflux.append(temp[1])
            self.tempwave_ne.append(temp_ne[0])
            self.tempflux_ne.append(temp_ne[1])
            self.tempwave_med.append(temp_med[0])
            self.tempflux_med.append(temp_med[1])

        self.tempwave = np.array(self.tempwave)
        self.tempflux = np.array(self.tempflux)
        self.tempwave_ne = np.array(self.tempwave_ne)
        self.tempflux_ne = np.array(self.tempflux_ne)
        self.tempwave_med = np.array(self.tempwave_med)
        self.tempflux_med = np.array(self.tempflux_med)



    def set_eels(self, err_mult, eel_bands, minimum_excursion):

        self.err_mult = err_mult
        self.eel_bands = eel_bands
        self.minimum_excursion = minimum_excursion

        if eel_bands == 1:
            self.iseel = np.array([any((thisgalsigma > err_mult) & (thisgal_me > 1. + minimum_excursion)) for (thisgalsigma, thisgal_me) in zip(self.sigmaconf, self.multiplicative_excursion)])
        else:
            num_eel_bands = np.sum(((self.sigmaconf > err_mult) & (self.multiplicative_excursion > 1. + minimum_excursion)), axis = 1)
            self.iseel = num_eel_bands >= eel_bands


    def save_template_plots(self, output_dir = './Figures/Templates/'):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        for tempnum in self.template_id-1:

            sp.plot(self.tempwave[tempnum], self.tempflux[tempnum])
            sp.set_title('%3i' % (tempnum+1))
            sp.set_xlabel(r'Wavelength ($\AA$)')
            sp.set_ylabel('Flux')

            thisage = float(self.template_name[tempnum].split('_')[1][:-3])
            thistau = float(self.template_name[tempnum].split('_')[2][:-3])
            thisZ = float(self.template_name[tempnum].split('_')[3][:-4])
            thisdust = float(self.template_name[tempnum].split('_')[4][:-6])


            sp.text(0.02, 0.98, 'Lya: %i\nHa: %i\n[OIII4963]: %i\n[OIII5007]: %i\n[OII]: %i' % tuple(self.get_equivalent_width_new(tempnum, tempindex = True)[0]), fontsize = 14, transform = sp.transAxes, ha = 'left', va = 'top')
            sp.text(0.98, 0.98, 'Age: %.3f Gyr\n' % thisage + r'$\tau$:%.3f' % thistau + '\nZ: %.1f\n' % thisZ + '$A_V$: %.1f' % thisdust, fontsize = 14, transform = sp.transAxes, ha = 'right', va = 'top')

            sp.set_xscale('log')
            sp.set_yscale('log')

            sp.set_xlim(10**3, 10**4.5)
            sp.set_ylim(10**-4, 10**2)

            plt.savefig(output_dir + '%03i.png' % (tempnum+1), bbox_inches = 'tight')

            sp.cla()

        plt.close()



    def save_pz_plots(self, output_dir = './Figures/pofz/'):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        for x in range(len(self.id)):

            sp.plot(self.pz_z, self.pz[x])
            sp.plot([self.z_spec[x],]*2, [-1,1], color = 'r')
            sp.plot([self.z_b[x],]*2, [-1,1], color = 'b')
            sp.plot([self.z_b_min[x],]*2, [-1,1], color = 'b', linestyle = '--', alpha = 0.5)
            sp.plot([self.z_b_max[x],]*2, [-1,1], color = 'b', linestyle = '--', alpha = 0.5)

            sp.set_title('%3i' % (x))
            sp.set_xlabel('Redshift (z)')
            sp.set_ylabel('p(z)')

            if self.iseel[x]:

                sp.text(0.98, 0.02, 'EEL', color = 'r', fontsize = 40, transform = sp.transAxes, ha = 'right', va = 'bottom')

            accuracy = (self.z_b[x]-self.z_spec[x])/(1.+self.z_spec[x])
            precision = (self.z_b_max[x]-self.z_b_min[x])/(1. + self.z_b[x])

            sp.text(0.98, 0.98, r'$\frac{z_\mathrm{B}-z_\mathrm{spec}}{1+z_\mathrm{spec}}=%.2f$' % accuracy + '\n' 
                + r'$\frac{\Delta z_\mathrm{B}}{1+z_\mathrm{B}}=%.2f$' % precision, 
                fontsize = 30, transform = sp.transAxes, ha = 'right', va = 'top')

            sp.set_ylim(-0.1, .3)
            sp.set_xlim(-0.1, 4.2)
            plt.savefig(output_dir + '%03i.png' % (x), bbox_inches = 'tight')

            sp.cla()

        plt.close()



    def plot_precision_accuracy(self, accuracy_abs = False):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        accuracy = (self.z_b-self.z_spec)/(1.+self.z_spec)
        precision = (self.z_b_max-self.z_b_min)/(2. * (1. + self.z_b))

        if accuracy_abs:
            accuracy = np.abs(accuracy)

        sp.scatter(precision[np.logical_not(self.iseel)], accuracy[np.logical_not(self.iseel)], color = 'k', marker = '.')
        sp.scatter(precision[self.iseel], accuracy[self.iseel], color = 'r', marker = '.')

        sp.set_xlabel(r'$\frac{\Delta z_\mathrm{B}}{1+z_\mathrm{B}}$')
        sp.set_ylabel(r'$\frac{z_\mathrm{B}-z_\mathrm{spec}}{1+z_\mathrm{spec}}$')






    def z_scatter(self, odds_lim = 0.95):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        sp.scatter(self.z_hsc[self.odds >= odds_lim], self.z_b[self.odds >= odds_lim], marker = ',', s = 1)

        if odds_lim != 0:

            sp.text(0.98, 0.02, r'odds$ \geq %.2f$' % odds_lim, fontsize = 24, transform = sp.transAxes, ha = 'right', va = 'bottom')

        sp.set_xlabel(r'$z_\mathrm{HSC}$')
        sp.set_ylabel(r'$z_\mathrm{B}$')

        sp.set_xlim(0,3)
        sp.set_ylim(0,3)


    def z_hist(self, odds_lim = 0.95, chisqlim = 10., deltazlim = np.inf):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        dz = (self.z_b_max - self.z_b_min)

        bpz_data = (self.z_b - self.z_spec)/(1 + self.z_spec)
        lim_indices = np.where((self.odds >= odds_lim) & (np.isfinite(bpz_data)) & (self.chisq <= chisqlim) & (dz <= deltazlim))
        hsc_data = (self.z_hsc - self.z_spec)/(1 + self.z_spec)

        sp.hist(bpz_data[lim_indices], histtype = 'step', bins = 40, range = (-1, 1), label = 'BPZ', density = True)
        sp.hist(hsc_data, histtype = 'step', bins = 40, range = (-1,1), label = 'HSC', density = True)

        sp.set_xlabel(r'$\frac{z_\mathrm{phot}-z_\mathrm{spec}}{1+z_\mathrm{spec}}$')
        sp.set_ylabel('Frequency')

        sp.text(0.02, 0.90, r'$\sigma_{BPZ} = ' + '%.2f$' % np.std(bpz_data[lim_indices]), fontsize = 20, ha = 'left', va = 'top', transform = sp.transAxes)
        sp.text(0.02, 0.84, r'$\sigma_{HSC} = ' + '%.2f$' % np.std(hsc_data), fontsize = 20, ha = 'left', va = 'top', transform = sp.transAxes)


        if odds_lim != 0:

            sp.text(0.02, 0.98, r'odds$ \geq %.2f$' % odds_lim, fontsize = 24, transform = sp.transAxes, ha = 'left', va = 'top')

        sp.legend()


    def z_compare(self, odds_lim = 0.95, chisqlim = 10., deltazlim = np.inf):
        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        dz = (self.z_b_max - self.z_b_min)

        lim_indices = np.where((self.odds >= odds_lim) & (self.chisq <= chisqlim) & (dz <= deltazlim))[0]

        tau_ID = self.t_b.astype(int)
        temps = [self.template_name[np.where(thisID == self.template_id)[0][0]] for thisID in tau_ID]
        taus = np.array([float(thistemp.split('_')[2][:-3]) if 'adamtemp' in thistemp else np.nan for thistemp in temps])
        taus = taus[lim_indices]

        # sp.scatter(self.z_hsc[lim_indices], self.z_spec[lim_indices], marker = '.', label = r'$z_\mathrm{HSC}$', color = 'k')
        # # sp.errorbar(self.z_b[lim_indices], self.z_spec[lim_indices], xerr = np.vstack(((self.z_b - self.z_b_min)[lim_indices], (self.z_b_max - self.z_b)[lim_indices])), ecolor = taus, cmap = 'RdYlBu', fmt = 'none')
        # pts = sp.scatter(self.z_b[lim_indices], self.z_spec[lim_indices], marker = '.', label = r'$z_\mathrm{B}$', cmap = 'RdYlBu', c = taus)
        
        sp.scatter(self.z_hsc[lim_indices], self.z_spec[lim_indices], marker = '.', label = r'$z_\mathrm{HSC}$', color = 'C1')
        # sp.errorbar(self.z_b[lim_indices], self.z_spec[lim_indices], xerr = np.vstack(((self.z_b - self.z_b_min)[lim_indices], (self.z_b_max - self.z_b)[lim_indices])), ecolor = 'C1',fmt = 'none')
        sp.scatter(self.z_b[lim_indices], self.z_spec[lim_indices], marker = '.', label = r'$z_\mathrm{B}$', color = 'C0')

        # cbax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
        # cb = mpl.colorbar.ColorbarBase(cbax, cmap = newcmap, norm = norm, 
        #     spacing = 'proportional', ticks = bounds, boundaries = bounds, format = '%.2f')
        # cb = mpl.colorbar.ColorbarBase(cbax, cmap = 'RdYlBu', format = '%.2f', norm = norm)


        if odds_lim != 0:

            sp.text(0.98, 0.02, r'odds$ \geq %.2f$' % odds_lim, fontsize = 24, transform = sp.transAxes, ha = 'right', va = 'bottom')

        sp.set_xlabel(r'$z_\mathrm{phot}$')
        sp.set_ylabel(r'$z_\mathrm{spec}$')

        sp.legend(loc = 'upper left')

        sp.set_xlim(0,3)
        sp.set_ylim(0,3)


    def plot_likelihood(self, galnum, plotsum = True):

        shelve_file = self.runfp + self.params['INPUT'].split('.')[0] + '.full_probs'

        # sed_dir = self.params['SED_DIR']
        # spectrafile = self.params['SPECTRA']
        # catfile = self.params['INPUT']
        # interp = self.params['INTERP']

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        data_file = shelve.open(shelve_file)
        data = data_file[str(self.bpz_id[galnum])]
        data_file.close()
        zbins = data[0]
        likelihoods = data[2]

        if plotsum:
            sumlikelihoods = np.sum(likelihoods, axis = 1)
            sp.plot(zbins, sumlikelihoods, color = 'k')
        
        [sp.plot(zbins, thislikelihood, label = '{:03}'.format(thistemp_id)) for x, (thislikelihood, thistemp_id) in enumerate(zip(likelihoods.T, self.template_id_interp))]
        # [sp.plot(zbins, thislikelihood, label = '{:03}'.format(thistemp_id)) for x, (thislikelihood, thistemp_id) in enumerate(zip(likelihoods.T[::self.params['INTERP']+1], self.template_id))]

        sp.set_xlabel('Redshift (z)')
        sp.set_ylabel('P(z)')


    def plot_likelihood_lim(self, plotnum = 10, reslo = 0., reshi = 0.1):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        residual = (self.z_b - self.z_spec)/(1 + self.z_spec)
        lim_indices = np.where((residual > reslo) & (residual < reshi))[0]

        selections = lim_indices[np.random.choice(np.arange(len(lim_indices)), size = plotnum, replace = False)]
        odds = self.odds[selections]

        shelve_file = self.runfp + self.params['INPUT'].split('.')[0] + '.full_probs'
        data_file = shelve.open(shelve_file)
        data = [data_file[str(thisid)] for thisid in self.bpz_id[selections]]
        data_file.close()

        for thisodds, thisgalaxy in zip(odds, data):

            zbins = thisgalaxy[0]
            prior = np.sum(thisgalaxy[2], axis = 1)

            sp.plot(zbins, prior, color = plt.cm.plasma(int(thisodds*plt.cm.RdYlBu.N/max(odds))))

        sp.set_xlabel('z')
        sp.set_ylabel('P(z)')

        return odds


    def get_eels(self, gal_index):

        data = fits.open('HSC/HSC_wide_clean.fits')[1].data

        if not hasattr(gal_index, '__iter__'):
            gal_index = [gal_index]


        sigma = []
        multiplicative_excursion = []


        for x, (thisid, thisredshift, thistemplate, thishsc_index, thisfitnorm) in enumerate(tqdm(zip(self.id[gal_index], self.z_b[gal_index], self.template_name[self.t_b[gal_index].astype(int) -1], self.hsc_index[gal_index], self.fit_norm[gal_index]), total = len(gal_index))):

            # Get the HSC fluxes and flux errors so we know the fractional bump needed from the emission lines

            fluxes = np.array([data[thishsc_index][band + 'cmodel_flux'] for band in ['g', 'r', 'i', 'z', 'y']])
            flux_errs = np.array([data[thishsc_index][band + 'cmodel_flux_err'] for band in ['g', 'r', 'i', 'z', 'y']])
            err_fraction = flux_errs/fluxes

            # Find the flux density in each filter when there are emission lines
            wave, spec = np.loadtxt(self.params['SED_DIR'] + thistemplate, unpack = True)
            spec_fnu = (1+thisredshift)**2 * thisfitnorm * spec * wave**2 / 3e18
            shifted_wavelengths, _ = phot.redshift(thisredshift, wave, spec_fnu)
            phot_wave, phot_flux = self.filters.get_photometry(shifted_wavelengths, spec_fnu, output_mags = False)

            # Do the same for the case with no emission lines
            wave_ne, spec_ne = np.loadtxt(self.params['SED_DIR'] + thistemplate[:-4] + '_noemline.sed', unpack = True)
            spec_ne_fnu = (1+thisredshift)**2 * thisfitnorm * spec_ne * wave_ne**2 / 3e18
            shifted_wavelengths_ne, _ = phot.redshift(thisredshift, wave_ne, spec_ne_fnu)
            phot_wave_ne, phot_flux_ne = self.filters.get_photometry(shifted_wavelengths_ne, spec_ne_fnu, output_mags = False)

            # sigma.append(((phot_flux/phot_flux_ne) - 1)/err_fraction)
            sigma.append((phot_flux-phot_flux_ne)/flux_errs)
            multiplicative_excursion.append(phot_flux/phot_flux_ne)



        return np.array(sigma), np.array(multiplicative_excursion)


    def plot_eel(self, gal_index, subplot = None):

        if subplot == None:

            fig = plt.figure(figsize = (8,8))
            sp = fig.add_subplot(111)

        else:

            sp = subplot

        tempwave, tempflam = np.loadtxt(self.params['SED_DIR'] + self.template_name[int(self.t_b[gal_index]) - 1], unpack = True)
        tempfnu = (1+self.z_b[gal_index])**2 * self.fit_norm[gal_index] * tempflam * tempwave**2 / 3e18

        tempwave_r, _ = phot.redshift(self.z_b[gal_index], tempwave, tempfnu)
        # tempwave_r = tempwave
        sp.plot(tempwave_r, tempfnu, color = 'C0')

        tempwave_ne, tempflam_ne = np.loadtxt(self.params['SED_DIR'] + self.template_name[int(self.t_b[gal_index]) - 1][:-4] + '_noemline.sed', unpack = True)
        # ne_norm = self.fit_norm[gal_index]*tempflam[0]/tempflam_ne[0]
        ne_norm = self.fit_norm[gal_index]
        tempfnu_ne = (1+self.z_b[gal_index])**2 * ne_norm * tempflam_ne * tempwave_ne**2 / 3e18

        tempwave_ne_r, _ = phot.redshift(self.z_b[gal_index], tempwave_ne, tempfnu_ne)
        # tempwave_ne_r = tempwave_ne

        sp.plot(tempwave_ne_r, tempfnu_ne, color = 'C1')

        filter_wave = [self.filters.wavelength_centers[thiskey] for thiskey in self.filters.keys]

        sp.scatter(filter_wave, self.fit_flux[gal_index], c = 'C0')

        filterlo = [self.filters.wavelength_centers[key] - self.filters.halfmax_lo[key] for key in self.filters.keys]
        filterhi = [self.filters.halfmax_hi[key] - self.filters.wavelength_centers[key] for key in self.filters.keys]

        sp.errorbar(filter_wave, self.obs_flux[gal_index], yerr = self.obs_flux_err[gal_index], xerr = [filterlo, filterhi], mfc = 'k', ecolor = 'k', linestyle = 'None', marker = 'o')

        # sp.scatter(filter_wave, self.obs_flux[gal_index], c = 'C1')

        phot_wave, phot_flux= self.filters.get_photometry(tempwave_r, tempfnu, output_mags = False)
        phot_wave_ne, phot_flux_ne= self.filters.get_photometry(tempwave_ne_r, tempfnu_ne, output_mags = False)
        
        sp.scatter(filter_wave, phot_flux_ne + self.err_mult*self.obs_flux_err[gal_index], marker = '_', c = 'r')

        sp.scatter(phot_wave, phot_flux, linewidths = 2, edgecolor = 'C0', c = 'None', s = 100)
        sp.scatter(phot_wave_ne, phot_flux_ne, linewidths = 2, edgecolor = 'C1', c = 'None', s = 100)

        arrow_x_ind = np.where(self.sigmaconf[gal_index] > self.err_mult)[0]
        sigma = self.sigmaconf[gal_index][arrow_x_ind]

        for filtkey, thisx, thissigma in zip(self.filters.keys[arrow_x_ind], (phot_wave[arrow_x_ind]-3000.)/8000., sigma):

            sp.arrow(thisx, 0.94, 0, -0.05, length_includes_head=True,head_width=0.015, head_length=0.02, transform = sp.transAxes, color = 'k')
            # EW = self.get_equivalent_width_new(gal_index)
            sp.text(thisx, 0.95, '%.2f' % thissigma + r'$\sigma$', ha = 'center', va = 'bottom', fontsize = 12, transform = sp.transAxes)

        emline_waves = np.array([1216., 6563., 4980., 3727.])*(1.+self.z_b[gal_index])
        # emline_names = ['lya', 'ha', 'oiii1', 'oiii2', 'oii']
        EW = self.get_equivalent_width_new(gal_index)[0]
        EW_weighted = self.get_equivalent_width_new(gal_index, weight_by_transmission = True)[0]

        EW[2]+=EW[3]
        EW_weighted[2]+= EW_weighted[3]
        EW = np.delete(EW, 3)
        EW_weighted = np.delete(EW_weighted,3)

        for line_wave, this_ew, this_ew_weighted in zip((emline_waves-3000.)/8000., EW, EW_weighted):
            if line_wave >0 and line_wave < 1:
                sp.arrow(line_wave, 0.15, 0, 0.05, length_includes_head=True,head_width=0.015, head_length=0.02, transform = sp.transAxes, color = 'C0')
                sp.text(line_wave, 0.14, '%i EW\n%i EW$_{W}$' % (this_ew, this_ew_weighted), ha = 'center', va = 'top', fontsize = 12, transform = sp.transAxes)

            

        sp.text(0.02, 0.98, 'z=%.2f' % self.z_b[gal_index], fontsize = 30, ha = 'left', va = 'top', transform = sp.transAxes)

        if self.iseel[gal_index]:
            sp.text(0.98, 0.02, 'EEL', fontsize = 30, ha = 'right', va = 'bottom', color = 'b', transform = sp.transAxes)

        elif self.islrg[gal_index]:
            sp.text(0.98, 0.02, 'LRG', fontsize = 30, ha = 'right', va = 'bottom', color = 'r', transform = sp.transAxes)

        sp.set_xlabel(r'Wavelength ($\AA$)')      
        sp.set_ylabel(r'$F_\nu$')
        sp.set_yscale('log')
        sp.set_xlim(3000, 11000)
        sp.set_ylim(self.fit_flux[gal_index][2]/100., self.fit_flux[gal_index][2]*100.)




    def save_eel_plots(self, output_dir = './Figures/hsc_fits/', eels_only = False):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        if eels_only:
            plotlist = np.where(self.iseel)[0]
        else:
            plotlist = range(len(self.id))

        for thiseel in plotlist:

            self.plot_eel(thiseel, sp)

            accuracy = (self.z_b[thiseel] - self.z_spec[thiseel])/(1+self.z_spec[thiseel])

            if np.abs(accuracy) > 0.1:
                plt.savefig(output_dir + 'badfit/%03i.png' % thiseel, bbox_inches = 'tight')
            else:
                plt.savefig(output_dir + 'goodfit/%03i.png' % thiseel, bbox_inches = 'tight')

            sp.cla()

        plt.close()



    # def get_equivalent_width(self, wave_no_el, f_nu_no_el, wave_el, f_nu_el, filter_transmission, filter_wavelength):
    #     # f_lam_no_el = f_nu_no_el * 3e18 / wavelength**2 # f_lambda in erg/s/cm^2/A
    #     # return flux_el / f_lam_no_el

    #     transfunc = lambda wavelength: np.interp(wavelength, filter_wavelength, filter_transmission, left = 0., right = 0.)

    #     f_lam_no_el = f_nu_no_el * 3e18 / wave_no_el**2 # f_lambda in erg/s/cm^2/A
    #     f_lam_el = f_nu_el * 3e18 / wave_el**2 # f_lambda in erg/s/cm^2/A

    #     if all([wave1 == wave2 for wave1, wave2 in zip(wave_no_el, wave_el)]):

    #         filtered_flux_no_el = f_lam_no_el * transfunc(wave_no_el)
    #         filtered_flux_el = f_lam_el * transfunc(wave_el)

    #         nonzero = np.where((filtered_flux_no_el != 0) & (filtered_flux_el != 0))

    #         EW = -np.trapz(1-(filtered_flux_el[nonzero]/filtered_flux_no_el[nonzero]), x = wave_el[nonzero])

    #     else:
    #         pass

    #     return EW






    def get_equivalent_width_new(self, gal_index, widths = {'lya': 75., 'ha':10., 'oiii1': 10., 'oiii2': 10., 'oii':15.}, weight_by_transmission = False, tempindex = False):

        emline_waves = {'lya': 1216., 'ha':6563., 'oiii1': 4963., 'oiii2': 5007., 'oii':3727.}

        # EW = {'lya': [], 'ha':[], 'oiii1': [], 'oiii2': [], 'oii':[]}

        EW = [[],[],[],[],[]]

        if not hasattr(gal_index, '__iter__'):
            gal_index = [gal_index]

        if tempindex:
            indices = self.template_id[gal_index] - 1
        else:
            indices = self.t_b[gal_index] - 1

        for x, thiskey in enumerate(emline_waves.keys()):

            for thisindex in indices:

                wave_ind = np.where(np.abs(emline_waves[thiskey] - self.tempwave[thisindex]) <= widths[thiskey])

                this_ew = np.trapz((self.tempflux[thisindex][wave_ind] - self.tempflux_ne[thisindex][wave_ind]) / self.tempflux_med[thisindex][wave_ind], x = self.tempwave[thisindex][wave_ind]*(1.+self.z_b[thisindex]))

                if weight_by_transmission:

                    max_transmission = 0.

                    for thisfilter in self.filters.keys:

                        thiswave = self.filters.wavelength[thisfilter]
                        thisresp = self.filters.response[thisfilter]

                        interp_transmission = np.interp(emline_waves[thiskey], thiswave, thisresp)

                        if interp_transmission > max_transmission:
                            max_transmission = interp_transmission

                    this_ew = this_ew * max_transmission

                EW[x].append(this_ew)

        return np.array(EW).T




    def plot_sigma_deltaz(self, yaxis = 'precision', nbins = 4, quantiles = True):

        maxsigma = np.max(self.sigmaconf, axis = 1)
        eel_filter = np.argmax(self.sigmaconf, axis = 1)
        # all_EW = self.get_equivalent_width_new(np.arange(len(self.id)))
        # EW = all_EW[[np.arange(len(self.id)), eel_filter]]

        generic = np.logical_not(self.iseel & self.islrg)

        if yaxis == 'precision':
            ydata_eel = (self.z_b_max[self.iseel] - self.z_b_min[self.iseel])/(2. * (1. + self.z_b[self.iseel]))
            ydata_lrg = (self.z_b_max[self.islrg] - self.z_b_min[self.islrg])/(2. * (1. + self.z_b[self.islrg]))
            ydata_generic = (self.z_b_max[generic] - self.z_b_min[generic])/(2. * (1. + self.z_b[generic]))
            ylabel = r'$\frac{\Delta z_\mathrm{B}}{1+z_\mathrm{B}}$'
            ylims = (-0.01, 1.)

            ydata_eel = np.log10(1 + ydata_eel)
            ydata_generic = np.log10(1 + ydata_generic)
            ydata_lrg = np.log10(1 + ydata_lrg)

        elif yaxis == 'accuracy':
            ydata_eel = (self.z_b[self.iseel] - self.z_spec[self.iseel])/(1+self.z_spec[self.iseel])
            ydata_lrg = (self.z_b[self.islrg] - self.z_spec[self.islrg])/(1+self.z_spec[self.islrg])
            ydata_generic = (self.z_b[generic] - self.z_spec[generic])/(1+self.z_spec[generic])
            ylabel = r'$\frac{z_\mathrm{B} - z_\mathrm{spec}}{1+z_\mathrm{spec}}$'
            ylims = (-0.5, 0.5)

            ydata_eel = np.log10(1 + ydata_eel)
            ydata_generic = np.log10(1 + ydata_generic)
            ydata_lrg = np.log10(1 + ydata_lrg)

        elif yaxis == 'precisionsigmas':
            ydata_eel = (self.z_b[self.iseel] - self.z_spec[self.iseel])/(self.z_b_max[self.iseel] - self.z_b_min[self.iseel])
            ydata_lrg = (self.z_b[self.islrg] - self.z_spec[self.islrg])/(self.z_b_max[self.islrg] - self.z_b_min[self.islrg])
            ydata_generic = (self.z_b[generic] - self.z_spec[generic])/(self.z_b_max[generic] - self.z_b_min[generic])
            ylabel = 'Inacurracy in Precision Sigmas'
            ylims = (-10,10)

        # xdata_eel = np.log10(maxsigma[self.iseel])
        # xdata_generic = np.log10(maxsigma[generic])

        xdata_eel = self.eelstat[self.iseel]
        xdata_lrg = self.eelstat[self.islrg]
        xdata_generic = self.eelstat[generic]

        good_eel = np.where(np.isfinite(xdata_eel))
        good_lrg = np.where(np.isfinite(xdata_lrg))
        good_generic = np.where(np.isfinite(xdata_generic))
        xdata_eel = xdata_eel[good_eel]
        xdata_lrg = xdata_lrg[good_lrg]
        xdata_generic = xdata_generic[good_generic]
        ydata_eel = ydata_eel[good_eel]
        ydata_lrg = ydata_lrg[good_lrg]
        ydata_generic = ydata_generic[good_generic]

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        plt.subplots_adjust(wspace = 0)

        sp.scatter(xdata_generic, ydata_generic, marker = '.', color = '0.7')
        sp.scatter(xdata_eel, ydata_eel, marker = '.', color = 'b')
        sp.scatter(xdata_lrg, ydata_lrg, marker = '.', color = 'r')

        if nbins > 0:

            bin_edges, sigma, nmad, avg, quartiles = self.bin_describe(np.hstack((xdata_eel, xdata_noteel)),np.hstack((ydata_eel, ydata_noteel)), quantiles = quantiles)

            _, sigma_eel, nmad_eel, avg_eel, quartiles_eel = self.bin_describe(xdata_eel, ydata_eel, bin_edges = bin_edges, quantiles = quantiles)
            _, sigma_lrg, nmad_lrg, avg_lrg, quartiles_lrg = self.bin_describe(xdata_lrg, ydata_lrg, bin_edges = bin_edges, quantiles = quantiles)
            _, sigma_generic, nmad_generic, avg_generic, quartiles_generic = self.bin_describe(xdata_generic, ydata_generic, bin_edges = bin_edges, quantiles = quantiles)

            for binlo, thissigma, thisnmad, thisavg, thisquartiles in zip(bin_edges[:-1], sigma, nmad, avg, quartiles):

                sp.text(binlo+.25, 1.3, r'$\sigma=%.2f$' % thissigma + '\n' 
                    + r'$\mathrm{NMAD} = %.2f$' % thisnmad + '\n'
                    + r'$\bar{y}=%.2f$' % thisavg + '\n'
                    + r'$q_{0}=%.2f$' % thisquartiles[0] + '\n'
                    + r'$q_{25}=%.2f$' % thisquartiles[1] + '\n'
                    + r'$q_{50}=%.2f$' % thisquartiles[2] + '\n'
                    + r'$q_{75}=%.2f$' % thisquartiles[3] + '\n'
                    + r'$q_{100}=%.2f$' % thisquartiles[4] + '\n', 
                    fontsize = 12, ha = 'left', va = 'top')

            print('======')
            print(' EELS ')
            print('======\n')
            print('')

            for binnum, (binlo, binhi, thissigma, thisnmad, thisavg, thisquartiles) in enumerate(zip(bin_edges[:-1], bin_edges[1:], sigma_eel, nmad_eel, avg_eel, quartiles_eel)):

                print('BIN%i' % (binnum+1) + ': {%.2f, %.2f]' % (binlo, binhi))
                print('------------------')

                print('sig: %.2f' % thissigma)
                print('NMAD: %.2f' % thisnmad)
                print('avg: %.2f' % thisavg)
                print('Quart: [%.2f, %.2f, %.2f, %.2f, %.2f]\n' % tuple(thisquartiles))


            print('======')
            print(' LRGs ')
            print('======\n')
            print('')

            for binnum, (binlo, binhi, thissigma, thisnmad, thisavg, thisquartiles) in enumerate(zip(bin_edges[:-1], bin_edges[1:], sigma_lrg, nmad_lrg, avg_lrg, quartiles_lrg)):

                print('BIN%i' % (binnum+1) + ': {%.2f, %.2f]' % (binlo, binhi))
                print('------------------')

                print('sig: %.2f' % thissigma)
                print('NMAD: %.2f' % thisnmad)
                print('avg: %.2f' % thisavg)
                print('Quart: [%.2f, %.2f, %.2f, %.2f, %.2f]\n' % tuple(thisquartiles))

            print('\n\n=========')
            print(    ' Generic ')
            print(    '=========\n')

            for binnum, (binlo, binhi, thissigma, thisnmad, thisavg, thisquartiles) in enumerate(zip(bin_edges[:-1], bin_edges[1:], sigma_generic, nmad_generic, avg_generic, quartiles_generic)):

                print('BIN%i' % (binnum+1) + ': {%.2f, %.2f]' % (binlo, binhi))
                print('------------------')

                print('sig: %.2f' % thissigma)
                print('NMAD: %.2f' % thisnmad)
                print('avg: %.2f' % thisavg)
                print('Quart: [%.2f, %.2f, %.2f, %.2f, %.2f]\n' % tuple(thisquartiles))


            # print('\n\n==========')
            # print(' ALL DATA ')
            # print('==========\n')

            # for binnum, (binlo, binhi, thissigma, thisnmad, thisavg, thisquartiles) in enumerate(zip(bin_edges[:-1], bin_edges[1:], sigma, nmad, avg, quartiles)):

            #     print('BIN%i' % (binnum+1) + ': {%.2f, %.2f]' % (binlo, binhi))
            #     print('------------------')

            #     print('sig: %.2f' % thissigma)
            #     print('NMAD: %.2f' % thisnmad)
            #     print('avg: %.2f' % thisavg)
            #     print('Quart: [%.2f, %.2f, %.2f, %.2f, %.2f]\n' % tuple(thisquartiles))


            for edge in bin_edges:

                sp.plot([edge, edge], [-10, 10], color = 'r', linewidth = 2)


        # sp.set_xlabel(xlabel)
        fig.text(0.5, 0., r'EEL Statistic', ha = 'center', va = 'bottom', fontsize = 24)
        sp.set_ylabel(ylabel)

        sp.set_ylim(ylims)




    def plot_ew_deltaz(self, yaxis = 'precision', weight_by_transmission = True, nbins = 4, quantiles = True):


        maxsigma = np.max(self.sigmaconf, axis = 1)
        eel_filter = np.argmax(self.sigmaconf, axis = 1)
        all_EW = self.get_equivalent_width_new(np.arange(len(self.id)), weight_by_transmission = weight_by_transmission)
        EW = all_EW[(np.arange(len(self.id)), np.argmax(all_EW, axis = 1))]

        if weight_by_transmission:
            xlabel = r'Weighted Equivalent Width ($\AA$)'
        else:
            xlabel = r'Equivalent Width ($\AA$)'

        noteel = np.logical_not(self.iseel)

        if yaxis == 'precision':
            ydata_eel = (self.z_b_max[self.iseel] - self.z_b_min[self.iseel])/(2. * (1. + self.z_b[self.iseel]))
            ydata_noteel = (self.z_b_max[noteel] - self.z_b_min[noteel])/(2. * (1. + self.z_b[noteel]))
            ylabel = r'$\frac{\Delta z_\mathrm{B}}{1+z_\mathrm{B}}$'

        elif yaxis == 'accuracy':
            ydata_eel = (self.z_b[self.iseel] - self.z_spec[self.iseel])/(1+self.z_spec[self.iseel])
            ydata_noteel = (self.z_b[noteel] - self.z_spec[noteel])/(1+self.z_spec[noteel])
            ylabel = r'$\frac{z_\mathrm{B} - z_\mathrm{spec}}{1+z_\mathrm{spec}}$'

        xdata_eel = np.log10(EW[self.iseel])
        xdata_noteel = np.log10(EW[noteel])
        # EW_eel = EW[self.iseel]
        # EW_noteel = EW[noteel]

        good_eel = np.where(np.isfinite(xdata_eel))
        good_noteel = np.where(np.isfinite(xdata_noteel))
        xdata_eel = xdata_eel[good_eel]
        xdata_noteel = xdata_noteel[good_noteel]
        ydata_eel = ydata_eel[good_eel]
        ydata_noteel = ydata_noteel[good_noteel]

        # ydata_eel = np.log10(1 + ydata_eel)
        # ydata_noteel = np.log10(1 + ydata_noteel)

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        plt.subplots_adjust(wspace = 0)

        sp.scatter(xdata_noteel, ydata_noteel, color = '0.7')
        sp.scatter(xdata_eel, ydata_eel, edgecolors = 'k', c = np.log10(maxsigma[self.iseel]), cmap = 'plasma')

        if nbins > 0:

            bin_edges, sigma, nmad, avg, quartiles = self.bin_describe(np.hstack((xdata_eel, xdata_noteel)),np.hstack((ydata_eel, ydata_noteel)), quantiles = quantiles)

            _, sigma_eel, nmad_eel, avg_eel, quartiles_eel = self.bin_describe(xdata_eel, ydata_eel, bin_edges = bin_edges, quantiles = quantiles)
            _, sigma_noteel, nmad_noteel, avg_noteel, quartiles_noteel = self.bin_describe(xdata_noteel, ydata_noteel, bin_edges = bin_edges, quantiles = quantiles)

            for binlo, thissigma, thisnmad, thisavg, thisquartiles in zip(bin_edges[:-1], sigma, nmad, avg, quartiles):

                sp.text(binlo+.25, 1.3, r'$\sigma=%.2f$' % thissigma + '\n' 
                    + r'$\mathrm{NMAD} = %.2f$' % thisnmad + '\n'
                    + r'$\bar{y}=%.2f$' % thisavg + '\n'
                    + r'$q_{0}=%.2f$' % thisquartiles[0] + '\n'
                    + r'$q_{25}=%.2f$' % thisquartiles[1] + '\n'
                    + r'$q_{50}=%.2f$' % thisquartiles[2] + '\n'
                    + r'$q_{75}=%.2f$' % thisquartiles[3] + '\n'
                    + r'$q_{100}=%.2f$' % thisquartiles[4] + '\n', 
                    fontsize = 12, ha = 'left', va = 'top')

            print('======')
            print(' EELS ')
            print('======\n')
            print('')

            for binnum, (binlo, binhi, thissigma, thisnmad, thisavg, thisquartiles) in enumerate(zip(bin_edges[:-1], bin_edges[1:], sigma_eel, nmad_eel, avg_eel, quartiles_eel)):

                print('BIN%i' % (binnum+1) + ': {%.2f, %.2f]' % (binlo, binhi))
                print('------------------')

                print('sig: %.2f' % thissigma)
                print('NMAD: %.2f' % thisnmad)
                print('avg: %.2f' % thisavg)
                print('Quart: [%.2f, %.2f, %.2f, %.2f, %.2f]\n' % tuple(thisquartiles))

            print('\n\n==========')
            print(' NON-EELS ')
            print('==========\n')

            for binnum, (binlo, binhi, thissigma, thisnmad, thisavg, thisquartiles) in enumerate(zip(bin_edges[:-1], bin_edges[1:], sigma_noteel, nmad_noteel, avg_noteel, quartiles_noteel)):

                print('BIN%i' % (binnum+1) + ': {%.2f, %.2f]' % (binlo, binhi))
                print('------------------')

                print('sig: %.2f' % thissigma)
                print('NMAD: %.2f' % thisnmad)
                print('avg: %.2f' % thisavg)
                print('Quart: [%.2f, %.2f, %.2f, %.2f, %.2f]\n' % tuple(thisquartiles))


            # print('\n\n==========')
            # print(' ALL DATA ')
            # print('==========\n')

            # for binnum, (binlo, binhi, thissigma, thisnmad, thisavg, thisquartiles) in enumerate(zip(bin_edges[:-1], bin_edges[1:], sigma, nmad, avg, quartiles)):

            #     print('BIN%i' % (binnum+1) + ': {%.2f, %.2f]' % (binlo, binhi))
            #     print('------------------')

            #     print('sig: %.2f' % thissigma)
            #     print('NMAD: %.2f' % thisnmad)
            #     print('avg: %.2f' % thisavg)
            #     print('Quart: [%.2f, %.2f, %.2f, %.2f, %.2f]\n' % tuple(thisquartiles))

            for edge in bin_edges:

                sp.plot([edge, edge], [-1, 1.5], color = 'r', linewidth = 2)                

        # sp.set_xlabel(xlabel)
        fig.text(0.5, 0., xlabel, ha = 'center', va = 'bottom', fontsize = 24)
        sp.set_ylabel(ylabel)

        sp.set_ylim(-1,1.5)
        sp.set_xlim(-11, 4)



    def bin_describe(self, xdata, ydata, bin_edges = None, nbins = 4, quantiles = True):

        if not hasattr(bin_edges, '__iter__'):
            if quantiles:
                bin_edges = np.quantile(xdata, np.linspace(0, 1, nbins+1))
            else:
                bin_edges = np.histogram(xdata, bins = nbins)[1]

        sigma = []
        nmad = []
        avg = []
        quartiles = []

        for lowedge, hiedge in zip(bin_edges[:-1], bin_edges[1:]):
            subset = ydata[(xdata > lowedge) & (xdata <= hiedge)]

            if len(subset)>0:

                sigma.append(np.std(subset))
                nmad.append(np.median(np.abs(subset - np.median(subset)))*1.4826)
                avg.append(np.average(subset))
                quartiles.append(np.quantile(subset, [0, .25, .50, .75, 1.00]))

            else:
                sigma.append(np.nan)
                nmad.append(np.nan)
                avg.append(np.nan)
                quartiles.append(np.array([np.nan,]*5))

        return bin_edges, np.array(sigma), np.array(nmad), np.array(avg), np.array(quartiles)



    def plot_precision_accuracy(self):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        noteel = np.logical_not(self.iseel)

        precision_eel = (self.z_b_max[self.iseel] - self.z_b_min[self.iseel])/(2. * (1. + self.z_b[self.iseel]))
        precision_noteel = (self.z_b_max[noteel] - self.z_b_min[noteel])/(2. * (1. + self.z_b[noteel]))
        sp.set_xlabel(r'$\frac{\Delta z_\mathrm{B}}{1+z_\mathrm{B}}$')

        precision_eel = np.log10(1 + precision_eel)
        precision_noteel = np.log10(1 + precision_noteel)

        # elif yaxis == 'accuracy':
        #     ydata_eel = (self.z_b[self.iseel] - self.z_spec[self.iseel])/(1+self.z_spec[self.iseel])
        #     ydata_noteel = (self.z_b[noteel] - self.z_spec[noteel])/(1+self.z_spec[noteel])
        #     ylabel = r'$\frac{z_\mathrm{B} - z_\mathrm{spec}}{1+z_\mathrm{spec}}$'
        #     ylims = (-0.5, 0.5)

        #     ydata_eel = np.log10(1 + ydata_eel)
        #     ydata_noteel = np.log10(1 + ydata_noteel)

        # elif yaxis == 'precisionsigmas':
        #     ydata_eel = (self.z_b[self.iseel] - self.z_spec[self.iseel])/(self.z_b_max[self.iseel] - self.z_b_min[self.iseel])
        #     ydata_noteel = (self.z_b[noteel] - self.z_spec[noteel])/(self.z_b_max[noteel] - self.z_b_min[noteel])
        #     ylabel = 'Inacurracy in Precision Sigmas'
        #     ylims = (-10,10)

        xdata_eel = np.log10(maxsigma[self.iseel])
        xdata_noteel = np.log10(maxsigma[noteel])

        good_eel = np.where(np.isfinite(xdata_eel))
        good_noteel = np.where(np.isfinite(xdata_noteel))
        xdata_eel = xdata_eel[good_eel]
        xdata_noteel = xdata_noteel[good_noteel]
        ydata_eel = ydata_eel[good_eel]
        ydata_noteel = ydata_noteel[good_noteel]







    def plot_deltaz(self, xaxis = 'sigma', err_mult = 5.):

        maxsigma = np.max(self.sigmaconf, axis = 1)
        eel_filter = np.argmax(self.sigmaconf, axis = 1)

        if xaxis == 'sigma':
            xdata = maxsigma
            xlabel = r'EEL Confidence ($\sigma$)'

        elif xaxis == 'ew':

            # EW = []


            # for gal_index in range(len(self.id)):
            #     tempwave, tempflam = np.loadtxt(self.params['SED_DIR'] + self.template_name[int(self.t_b[gal_index]) - 1], unpack = True)
            #     tempfnu = (1+self.z_b[gal_index])**2 * self.fit_norm[gal_index] * tempflam * tempwave**2 / 3e18

            #     tempwave_r, _ = phot.redshift(self.z_b[gal_index], tempwave, tempfnu)
            #     # tempwave_r = tempwave

            #     tempwave_ne, tempflam_ne = np.loadtxt(self.params['SED_DIR'] + self.template_name[int(self.t_b[gal_index]) - 1][:-4] + '_noemline.sed', unpack = True)
            #     # ne_norm = self.fit_norm[gal_index]*tempflam[0]/tempflam_ne[0]
            #     ne_norm = self.fit_norm[gal_index]
            #     tempfnu_ne = (1+self.z_b[gal_index])**2 * ne_norm * tempflam_ne * tempwave_ne**2 / 3e18

            #     tempwave_ne_r, _ = phot.redshift(self.z_b[gal_index], tempwave_ne, tempfnu_ne)
            #     # tempwave_ne_r = tempwave_ne
            
            #     # eel_ind = np.where(phot_flux > phot_flux_ne + err_mult*self.obs_flux_err[gal_index])[0]

            #     filtkey = self.filters.keys[eel_filter[gal_index]]

            #     EW.append(self.get_equivalent_width(tempwave_ne_r, tempfnu_ne, tempwave_r, tempfnu, self.filters.norm_response[filtkey], self.filters.wavelength[filtkey]))

            all_EW = self.get_equivalent_width_new(np.arange(len(self.id)))

            EW = all_EW[[np.arange(len(self.id)), eel_filter]]


            xdata = np.array(EW)
            xlabel = r'EW ($\AA$)'

        elif xaxis == 'sn':

            # total_sn = np.sqrt((data_table['g_flux']/data_table['g_flux_err'])**2 + (data_table['r_flux']/data_table['r_flux_err'])**2 + 
            # (data_table['i_flux']/data_table['i_flux_err'])**2 + (data_table['z_flux']/data_table['z_flux_err'])**2 + (data_table['y_flux']/data_table['y_flux_err'])**2)

            total_sn = np.sqrt(np.sum((self.obs_flux/self.obs_flux_err)**2, axis = 1))
            xdata = total_sn
            xlabel = 'Total S/N'

        elif xaxis == 'g-i':

            catfile = self.runfp + self.params['INPUT']
            g, i = np.loadtxt(catfile, usecols = [1,5], unpack = True)
            xdata = g - i
            xlabel = 'g-i'

        elif xaxis == 'g-z':

            catfile = self.runfp + self.params['INPUT']
            g,z = np.loadtxt(catfile, usecols = [1,7], unpack = True)
            xdata = g-z
            xlabel = 'g-z'

        elif xaxis == 'chisq':

            chisq = self.chisq

            xdata = chisq
            xlabel = r'$\chi^2$'

        elif xaxis == 'flux':
            pass




        noteel = np.logical_not(self.iseel)
        ydata_eel = (self.z_b[self.iseel] - self.z_spec[self.iseel])/(1+self.z_spec[self.iseel])
        ydata_noteel = (self.z_b[noteel] - self.z_spec[noteel])/(1+self.z_spec[noteel])
        xdata_eel = xdata[self.iseel]
        xdata_noteel = xdata[noteel]

        # ydata_eel = np.log10(1 + ydata_eel)
        # ydata_noteel = np.log10(1 + ydata_noteel)

        fig = plt.figure(figsize = (16,8))
        sp1 = fig.add_subplot(121)
        sp2 = fig.add_subplot(122)

        plt.subplots_adjust(wspace = 0)

        sp1.scatter(xdata_noteel, ydata_noteel, marker = '.')
        sp2.scatter(xdata_eel, ydata_eel, marker = '.')

        # sp.set_xlabel(xlabel)
        fig.text(0.5, 0., xlabel, ha = 'center', va = 'bottom', fontsize = 24)
        # sp1.set_ylabel(r'$\log_{10}\left( 1+ \frac{\Delta z}{1+z}\right)$')
        sp1.set_ylabel(r'$\frac{\Delta z}{1+z}$')
        sp2.set_yticklabels([])

        sp1.text(0.98, 0.98, 'Non-EEL', fontsize = 24, va = 'top', ha = 'right', transform = sp1.transAxes)
        sp2.text(0.98, 0.98, 'EEL', fontsize = 24, va = 'top', ha = 'right', transform = sp2.transAxes)

        sp1.set_ylim(-1,1.5)
        sp2.set_ylim(-1,1.5)




    def plot_eel_zconf(self, zcompare = 'bpz'):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        noteels = np.logical_not(self.iseel)

        if zcompare == 'bpz':
            z_eel = self.z_b[self.iseel]
            z_noteel = self.z_b[noteels]
            dz_eel = self.z_b_max[self.iseel]-self.z_b_min[self.iseel]
            dz_noteel = self.z_b_max[noteels]-self.z_b_min[noteels]
            xtext = r'$z_\mathrm{BPZ}$'
            ytext = r'$\Delta z_\mathrm{BPZ}$'
        elif zcompare == 'specz':
            z_eel = self.z_spec[self.iseel]
            z_noteel = self.z_spec[noteels]
            dz_eel = self.z_b[self.iseel] - self.z_spec[self.iseel]
            dz_noteel = self.z_b[noteels] - self.z_spec[noteels]
            xtext = r'$z_\mathrm{spec}$'
            ytext = r'$z_\mathrm{BPZ} - z_\mathrm{spec}$'


        sp.scatter(z_eel, dz_eel, c = 'b', marker = '.', label = 'EEL')
        sp.scatter(z_noteel, dz_noteel, c = 'r', marker = '.', label = 'non-EEL')

        sp.set_xlabel(xtext)
        sp.set_ylabel(ytext)


    def plot_eel_zconf_hist(self, zcompare = 'bpz'):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        noteels = np.logical_not(self.iseel)

        if zcompare == 'bpz':
            z_eel = self.z_b[self.iseel]
            z_noteel = self.z_b[noteels]
            dz_eel = self.z_b_max[self.iseel]-self.z_b_min[self.iseel]
            dz_noteel = self.z_b_max[noteels]-self.z_b_min[noteels]
            xtext = r'$\frac{z_\mathrm{hi} - z_\mathrm{lo}}{z_\mathrm{BPZ}}$'
            histrange = (0,2)
        elif zcompare == 'specz':
            z_eel = self.z_spec[self.iseel]
            z_noteel = self.z_spec[noteels]
            dz_eel = self.z_b[self.iseel] - self.z_spec[self.iseel]
            dz_noteel = self.z_b[noteels] - self.z_spec[noteels]
            xtext = r'$\frac{z_\mathrm{BPZ} - z_\mathrm{spec}}{z_\mathrm{spec}}$'
            histrange = (-1,1)


        sp.hist(dz_eel/z_eel, histtype = 'step', range = histrange, bins = 40, color = 'b')
        sp.hist(dz_noteel/z_noteel, histtype = 'step', range = histrange, bins = 40, color = 'r')

        sp.set_xlabel(xtext)
        sp.set_ylabel('Frequency')        

