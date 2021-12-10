import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import filtersim
from glob import glob
from astropy.cosmology import Planck15 as cosmo
from os import system
from os.path import isfile
from os.path import isdir
from os import makedirs
from tqdm import tqdm
from pandas import read_csv


with open('/home/adam/Research/fsps/src/sps_vars.f90', 'r') as readfile:
    lines = readfile.readlines()
    if ('MILES 0' in lines[8]) or ('MIST 0' in lines[14]) or ('BPASS 1' in lines[21]):
        system('fsps_mm')

import fsps


font = {'family':'Roboto', 'weight':'light'}

Lsun = 3.848e33 # erg/s
c_angstroms = 3e8 * 10**10 # in AA/s


dirlist = ['./cache/neb_em', './cache/neb_noem', './cache/noneb_noem']
if not all(map(isdir, dirlist)):
    for directory in dirlist:
        makedirs(directory)


class photosim:

    def __init__(self, inputfolder = './adam_synth_spectra/', odin_filters = False, narrow_lya = True):

        if odin_filters:
            self.filters = filtersim.odin_filtersim()
        else:
            self.filters = filtersim.filtersim()

        self.starpop = None
        self.narrow_lya = narrow_lya


    def redshift(self, redshift, wavelength, l_nu):

        if redshift == 0:
            return wavelength, l_nu
        else:

            new_wavelength = wavelength * (1.+redshift)
            flux = l_nu * (1+redshift) / (4*np.pi*np.power(cosmo.luminosity_distance(redshift).to('cm').value,2))

            return new_wavelength, flux


    def find_spectrum(self, tage, metallicity = 1., imf_type = 1, sfh_type = 3, sfh_params = {'sfr':np.array([1.,1.]), 't':np.array([0,13])}, dust_type = 2, Av = 0, emline = True, nebcont = True, increase_ssp = True, delay_csf = False, peraa = False, redshift = 0.):

        # Cache and retreive spectra

        # fname_params = (tage, metallicity, imf_type, sfh_type, dust_type)

        # fname = '%.5f_%.2f_%i_%i_%i.spec' % fname_params

        # fstub = './cache/'

        # if emline and nebcont:
        #     fstub = fstub + 'neb_em/'
        # elif nebcont:https://stackoverflow.com/questions/3394835/use-of-args-and-kwargs
        #     fstub = fstub + 'neb_noem/'
        # elif not emline and not nebcont:
        #     fstub = fstub + 'noneb_noem/'

        # if isfile(fstub + fname):

        #     waves, l_nu = np.loadtxt(fstub + fname, unpack = True)

        # else:


        if sfh_type == 6:

            time = np.linspace(0, 10, 250)
            sfr = np.exp(time/sfh_params['tau'])
            if not 't' in sfh_params.keys():
                sfh_params['t'] = time
            if not 'sfr' in sfh_params.keys():
                sfh_params['sfr'] = sfr

            return self.find_spectrum(tage, metallicity = metallicity, imf_type = imf_type, sfh_type = 3, sfh_params = sfh_params, dust_type = dust_type, emline = emline, nebcont = nebcont, peraa = peraa, Av = Av, redshift = redshift)

            
        if self.starpop == None:
            self.starpop = fsps.StellarPopulation(zcontinuous=1, add_neb_emission = True, nebemlineinspec = emline, add_neb_continuum = nebcont,
                imf_type = imf_type, dust_type = dust_type, sfh = sfh_type, logzsol = np.log10(metallicity), tage = tage, gas_logz = np.log10(metallicity), dust2 = Av/1.08574, zred = redshift, redshift_colors = redshift > 0, add_igm_absorption = redshift > 0)
        else:
            self.starpop.params['imf_type'] = imf_type
            self.starpop.params['logzsol'] = np.log10(metallicity)
            self.starpop.params['gas_logz'] = np.log10(metallicity)
            self.starpop.params['dust_type'] = dust_type
            self.starpop.params['sfh'] = sfh_type
            self.starpop.params['nebemlineinspec'] = emline
            self.starpop.params['add_neb_continuum'] = nebcont
            self.starpop.params['dust2'] = Av/1.08574
            self.starpop.params['zred'] = redshift
            self.starpop.params['redshift_colors'] = redshift > 0
            self.starpop.params['add_igm_absorption'] = redshift > 0

        if sfh_type == 3:
            # Form stars at 1Msun/yr for 1Gyr, then spike to 200Msun/yr
            # starpop.set_tabular_sfh(np.array([0,0.999,1,1.1]), np.array([1,1,10,10]))
            self.starpop.set_tabular_sfh(sfh_params['t'], sfh_params['sfr'])
            if delay_csf:
                tage = tage + 1.

        elif sfh_type == 1:

            if not 'tau' in sfh_params.keys():
                sfh_params['tau'] = 1. 

            #Set everything to zeros if they don't already exist

            for thiskey in ['const', 'sf_start', 'sf_trunc', 'tburst', 'fburst']:
                if thiskey not in sfh_params.keys():
                    sfh_params[thiskey] = 0.

            for thiskey in list(sfh_params.keys()):
                self.starpop.params[thiskey] = sfh_params[thiskey]
       
        waves, lum = self.starpop.get_spectrum(tage = tage, peraa = peraa)

        # peraa doesn't work for python FSPS
        # if peraa:
        #     lum = lum * c_angstroms / (waves**2)

        lum = lum * Lsun

        # np.savetxt(fstub + fname, np.vstack((waves, lum)).T, fmt = '%e   %.10e')

        if self.narrow_lya:
            pass

        if increase_ssp and sfh_type == 0:
            lum = lum * 10.**7

        return waves, lum

        


    def gencat(self, cat_input = './EAZY_runs/cat_input.param', cat_output = './EAZY_runs/cat.dat', output_mags = False, dmag_err = False):


        gal_id, redshift, age, sfh, metal, imf, emline, nebcont = read_csv(cat_input,
            header = None, comment = '#', delimiter = '\s+').values.T

        sfh_type = np.zeros(len(sfh))
        imf_type = np.zeros(len(imf))

        imf_type[imf == 'CHABRIER'] = 1
        imf_type[imf == 'KROUPA'] = 2

        sfh_type[sfh == 'CSF'] = 3

        with open(cat_output, 'w') as writefile:

            writefile.write('# ')
            writefile.write('id'.rjust(4))
            writefile.write('  ')

            for x in range(len(self.filters.keys)):

                writefile.write(('f_' + self.filters.keys[x]).ljust(15))
                writefile.write(('e_' + self.filters.keys[x]).ljust(15))

            # writefile.write('z_spec')

            writefile.write('\n')

            for x in tqdm(range(len(gal_id))):

                wavelengths, spec_l_nu = self.find_spectrum(age[x], metal[x], imf_type[x], sfh_type[x], 
                    emline = emline[x], nebcont = nebcont[x])
                spec_l_lambda = spec_l_nu * (wavelengths**2.) / 3.e10 

                shifted_wavelengths, spec_flux = self.redshift(redshift[x], wavelengths, spec_l_nu)

                phot_wave, phot_flux, phot_err = self.filters.get_photometry(shifted_wavelengths, spec_flux, output_mags = output_mags, dmag_err = dmag_err)

                writefile.write('   %03i' % gal_id[x])
                writefile.write('  ')

                if output_mags:

                    for y in range(len(phot_wave)):

                        writefile.write(('%.5f' % phot_flux[y]).ljust(15))
                        writefile.write(('%.5f' % phot_err[y]).ljust(15))


                else:

                    for y in range(len(phot_wave)):

                        writefile.write(('%.6e' % phot_flux[y]).ljust(15))
                        writefile.write(('%.6e' % phot_err[y]).ljust(15))

                # writefile.write('-1.000')
                writefile.write('\n')





    def save_temp(self, sfhtype = 'CSF', metallicity = 1.0, time = 0.001, renorm = True, savefp = '/home/adam/Research/eazy-photoz/templates/AdamTemps/'):

        fname = savefp + sfhtype +'_%iMyr_Z_%.1f.dat' % (time*1000, metallicity)

        if sfhtype == 'SSP':
            starpop = fsps.StellarPopulation(zcontinuous=1, add_neb_emission = True, imf_type = 1, dust_type = 2, 
                sfh = 0, logzsol = np.log10(metallicity)) # chabrier IMF and Calzetti dust
        elif sfhtype == 'CSF':
            starpop = fsps.StellarPopulation(zcontinuous=1, add_neb_emission = True, imf_type = 1, dust_type = 2, 
                sfh = 3, logzsol = np.log10(metallicity))
            starpop.set_tabular_sfh(np.array([0,1]), np.array([1,1]))

        wavelength, l_nu = self.find_spectrum(starpop, time)

        if renorm:
            l_nu = 2 * l_nu / max(l_nu) # Templates in EAZY seem to be normalized so they peak around 2, so maybe this will help?

        with open(fname, 'w') as writefile:

            for x in range(len(wavelength)):
                writefile.write(('%.5e' % wavelength[x]).ljust(20))
                writefile.write('%.5e' % l_nu[x] + '\n')


