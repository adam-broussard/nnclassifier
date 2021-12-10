import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from glob import glob

noNewLine = '\x1b[1A\x1b[1M'
font = {'family':'Roboto', 'weight':'light'}

class filtersim:

    def __init__(self, filter_fp = './lsst_filts/*.dat', manualkeys = ['u', 'g', 'r', 'i', 'z', 'y']):

        filterfiles = glob(filter_fp)

        self.wavelength = {}
        self.response = {}
        self.norm_response = {}

        for fname in filterfiles:

            tempwave, tempresp = np.loadtxt(fname, unpack = True)
            self.wavelength[fname.split('/')[-1][-5]] = tempwave*10.
            self.response[fname.split('/')[-1][-5]] = tempresp

            self.norm_response[fname.split('/')[-1][-5]] = tempresp/np.trapz(tempresp, x = tempwave*10.)

        if manualkeys:
            self.keys = np.array(manualkeys)
        else:
            self.keys = array(list(self.wavelength.keys()))

        self.wavelength_centers = {}

        for index, key in enumerate(self.keys):
            
            self.wavelength_centers[key] = np.trapz(self.norm_response[key] * self.wavelength[key], x = self.wavelength[key])

        # From the LSST Sience book pg 21 https://arxiv.org/pdf/0912.0201.pdf
        # Units are erg/s/cm^2/Hz

        self.error = {'u':np.power(10., (26.3 + 48.6)/(-2.5)),
                        'g':np.power(10., (27.5 + 48.6)/(-2.5)),
                        'r':np.power(10., (27.7 + 48.6)/(-2.5)),
                        'i':np.power(10., (27.0 + 48.6)/(-2.5)),
                        'z':np.power(10., (26.2 + 48.6)/(-2.5)),
                        'y':np.power(10., (24.9 + 48.6)/(-2.5))}

        self.error_mags = {'u':26.3, 'g':27.5, 'r':27.7, 'i':27.0, 'z':26.2, 'y':24.9}



    def flux_to_abmag(self, flux):

        # Converts erg/s/cm^2/Hz to AB magnitudes

        return -2.5*np.log10(flux) -48.6

    def abmag_to_flux(self, mag):

        return np.power(10., (mag + 48.6)/(-2.5))



    def plot_filters(self):

        fig = plt.figure(figsize = (8,8))
        sp = fig.add_subplot(111)

        colors = [mpl.cm.RdYlBu_r(int(place)) for place in np.linspace(0, mpl.cm.RdYlBu_r.N, len(self.keys))]

        for index, key in enumerate(self.keys):

            sp.plot(self.wavelength[key], self.response[key], color = 'k', linewidth = 4, zorder = index)
            sp.plot(self.wavelength[key], self.response[key], color = colors[index], linewidth = 2, label = key, zorder = index)

        # sp.set_xscale('log')
        sp.set_ylim(-0.05, 1.0)
        sp.legend(loc = 'upper right', fontsize = 16)

        sp.set_xlabel(r'$\lambda\,\,(\mathrm{\AA})$', fontdict = font, fontsize = 24)
        sp.set_ylabel('Throughput', fontdict = font, fontsize = 24)
        sp.set_title('LSST Throughput Curves', fontdict = font, fontsize = 28)



    def get_photometry(self, wavelength, flux_density, wavestep = 0.1, output_mags = False, dmag_err = False, input_flam = False):

        # This is slightly different from the above because LSST filters are photon count filters

        # flux_density needs to be in f_nu units

        c = 3.e10 # speed of light in cm/s

        if input_flam:
            f_lambda = flux_density
        else:
            f_lambda = flux_density * c/(wavelength**2.)

        wavelengths = [self.wavelength_centers[key] for key in self.keys]
        phot = []

        interp_x = np.arange(3000, 12000, wavestep) # The x-axis of the spectrum

        # Loop through each of the LSST filters
        for index, key in enumerate(self.keys):

            interp_y = np.interp(interp_x, wavelength, f_lambda) # The interpolated y-axis of the spectrum (in f_lambda)

            # Find the filter curve values at the same x values as interp_x
            filter_interp_y = np.interp(interp_x, self.wavelength[key], self.norm_response[key]) 

            phot.append(np.trapz(filter_interp_y * interp_y * interp_x, x = interp_x)/np.trapz(filter_interp_y * c / interp_x, x = interp_x))

            phot_err = [self.error[key] for key in self.keys]

        wavelengths = np.array(wavelengths)
        phot = np.array(phot)
        phot_err = np.array(phot_err)

        if output_mags:

            if dmag_err:
                #dmag_err is the difference between phot (as a magnitude) and phot-phot_err (as a magnitude)
                phot_err = np.abs(self.flux_to_abmag(phot) - self.flux_to_abmag(abs(phot - phot_err)))
                nans = np.where(np.isnan(phot_err))[0]
            else:
                phot_err = self.flux_to_abmag(phot_err)

            phot = self.flux_to_abmag(phot)

            if len(nans) > 0:
                onesigma = [self.error[self.keys[thisnan]] / 5. for thisnan in nans] # These are five sigma detection limits, so divide by 5
                phot[nans] = 99.
                phot_err[nans] = self.flux_to_abmag(onesigma)

        return wavelengths, phot, phot_err



    def output_eazy_filters(self, fname = './EAZY_runs/Inputs/lsst.filters.res'):

        with open(fname, 'w') as writefile:

            for index, key in enumerate(self.keys):

                writefile.write(str(len(self.wavelength[key])).ljust(10) + 'LSST ' + key 
                    + '-band Filter; lam_ctr = %.2f\n' % self.wavelength_centers[key])

                for x in range(len(self.wavelength[key])):

                    writefile.write('%6i' % x)
                    writefile.write('  %.2f'.ljust(12) % self.wavelength[key][x])
                    writefile.write('%.4f\n' % self.response[key][x])


    def output_filter_translate(self, fname = './EAZY_runs/Inputs/zphot.translate'):

        with open(fname, 'w') as writefile:

            for index, key in enumerate(self.keys):

                writefile.write('f_' + key + '  F%i\n' % (index+1))
                writefile.write('e_' + key + '  E%i\n' % (index+1))



    def mag_norm(self, waves, f_nu, norm_mag = 25.):

        phot_wave, phot, phot_err = self.get_photometry(waves, f_nu, output_mags = True, dmag_err = True)

        # -2.5*(log10(fluxnorm) + log10(flux)) - 48.6 = norm_mag
        # dmag = norm_mag - phot
        # norm_flux = 10**(-2.5*dmag)

        dmag = norm_mag - phot
        dmag_single = dmag[np.argmin(phot)] # Pull the brightest magnitude and use it to set the normalization
        flux_norm = 10**(dmag_single/-2.5)

        return waves, f_nu*flux_norm




class hsc_filtersim:

    def __init__(self, filter_fp = './HSC/filters/Processed/*processed.res', manualkeys = ['g', 'r2', 'i2', 'z', 'Y']):

        filterfiles = glob(filter_fp)

        self.wavelength = {}
        self.response = {}
        self.norm_response = {}

        for fname in filterfiles:

            tempwave, tempresp = np.loadtxt(fname, unpack = True)
            tempresp[tempresp<0] = 0.
            self.wavelength[fname.split('/')[-1][5:-14]] = tempwave
            self.response[fname.split('/')[-1][5:-14]] = tempresp

            self.norm_response[fname.split('/')[-1][5:-14]] = tempresp/np.trapz(tempresp, x = tempwave)

        if manualkeys:
            self.keys = np.array(manualkeys)
        else:
            self.keys = array(list(self.wavelength.keys()))

        self.wavelength_centers = {}
        self.halfmax_lo_ind = {}
        self.halfmax_hi_ind = {}
        self.halfmax_lo = {}
        self.halfmax_hi = {}

        for index, key in enumerate(self.keys):
            
            self.wavelength_centers[key] = np.trapz(self.norm_response[key] * self.wavelength[key], x = self.wavelength[key])
            thismax = np.max(self.response[key])
            self.halfmax_lo_ind[key] = np.where(self.response[key] - thismax/2. > 0)[0][0]
            self.halfmax_hi_ind[key] = np.where(self.response[key] - thismax/2. > 0)[0][-1]
            self.halfmax_lo[key] = self.wavelength[key][self.halfmax_lo_ind[key]]
            self.halfmax_hi[key] = self.wavelength[key][self.halfmax_hi_ind[key]]


    def get_photometry(self, wavelength, flux_density, wavestep = 0.1, output_mags = False, input_flam = False):

        # This is slightly different from the above because LSST filters are photon count filters

        # flux_density needs to be in f_nu units

        c = 3.e10 # speed of light in cm/s

        if not input_flam:

            f_lambda = flux_density * c/(wavelength**2.)

        else:

            f_lambda = flux_density

        wavelengths = [self.wavelength_centers[key] for key in self.keys]
        phot = []

        interp_x = np.arange(3000, 12000, wavestep) # The x-axis of the spectrum

        # Loop through each of the LSST filters
        for index, key in enumerate(self.keys):

            interp_y = np.interp(interp_x, wavelength, f_lambda) # The interpolated y-axis of the spectrum (in f_lambda)

            # Find the filter curve values at the same x values as interp_x
            filter_interp_y = np.interp(interp_x, self.wavelength[key], self.norm_response[key]) 

            phot.append(np.trapz(filter_interp_y * interp_y * interp_x, x = interp_x)/np.trapz(filter_interp_y * c / interp_x, x = interp_x))

        wavelengths = np.array(wavelengths)
        phot = np.array(phot)

        if output_mags:

            phot = self.flux_to_abmag(phot)


        return wavelengths, phot


    def flux_to_abmag(self, flux):

        # Converts erg/s/cm^2/Hz to AB magnitudes

        return -2.5*np.log10(flux) -48.6

    def abmag_to_flux(self, mag):

        return np.power(10., (mag + 48.6)/(-2.5))





class odin_filtersim:

    def __init__(self, hsc_filter_fp = './HSC/filters/Processed/*_processed.res', odin_filter_fp = './ODIN/Processed/*.dat', manualkeys = ['g', 'r2', 'i2', 'z', 'Y', 'N420', 'N501', 'N673']):

        hsc_filterfiles = glob(hsc_filter_fp)
        odin_filterfiles = sorted(glob(odin_filter_fp))

        self.wavelength = {}
        self.response = {}
        self.norm_response = {}

        self.is_odin_filter = np.array([thisfile in odin_filterfiles for thisfile in hsc_filterfiles + odin_filterfiles])

        for fname in hsc_filterfiles:

            tempwave, tempresp = np.loadtxt(fname, unpack = True)
            tempresp[tempresp<0] = 0.
            self.wavelength[fname.split('/')[-1][5:-14]] = tempwave
            self.response[fname.split('/')[-1][5:-14]] = tempresp

            self.norm_response[fname.split('/')[-1][5:-14]] = tempresp/np.trapz(tempresp, x = tempwave)

        for x, fname in enumerate(odin_filterfiles):

            tempwave, tempresp = np.loadtxt(fname, unpack = True)
            tempresp[tempresp<0] = 0.
            tempwave = tempwave * 10.
    
            if manualkeys:
                thiskey = manualkeys[x+len(hsc_filterfiles)]
            else:
                thiskey = fname.split('/')[-1].split('.')[0]
    
            self.wavelength[thiskey] = tempwave
            self.response[thiskey] = tempresp
            self.norm_response[thiskey] = tempresp/np.trapz(tempresp, x = tempwave)

        if manualkeys:
            self.keys = np.array(manualkeys)
        else:
            self.keys = array(list(self.wavelength.keys()))

        self.wavelength_centers = {}
        self.halfmax_lo_ind = {}
        self.halfmax_hi_ind = {}
        self.halfmax_lo = {}
        self.halfmax_hi = {}

        for index, key in enumerate(self.keys):
            
            self.wavelength_centers[key] = np.trapz(self.norm_response[key] * self.wavelength[key], x = self.wavelength[key])
            thismax = np.max(self.response[key])
            self.halfmax_lo_ind[key] = np.where(self.response[key] - thismax/2. > 0)[0][0]
            self.halfmax_hi_ind[key] = np.where(self.response[key] - thismax/2. > 0)[0][-1]
            self.halfmax_lo[key] = self.wavelength[key][self.halfmax_lo_ind[key]]
            self.halfmax_hi[key] = self.wavelength[key][self.halfmax_hi_ind[key]]


        self.error = {'g':np.power(10., (27.8 + 48.6)/(-2.5)),
                'r2':np.power(10., (27.4 + 48.6)/(-2.5)),
                'i2':np.power(10., (27.1 + 48.6)/(-2.5)),
                'z':np.power(10., (26.6 + 48.6)/(-2.5)),
                'Y':np.power(10., (25.6 + 48.6)/(-2.5)),
                'N420':np.power(10., (25.9 + 48.6)/(-2.5)),
                'N501':np.power(10., (26.0 + 48.6)/(-2.5)),
                'N673':np.power(10., (25.8 + 48.6)/(-2.5))}


    def get_photometry(self, wavelength, flux_density, wavestep = 0.1, output_mags = False, dmag_err = False, input_flam = False):

        # This is slightly different from the above because LSST filters are photon count filters

        # flux_density needs to be in f_nu units

        c = 3.e10 # speed of light in cm/s

        if input_flam:
            f_lambda = flux_density
        else:
            f_lambda = flux_density * c/(wavelength**2.)

        wavelengths = [self.wavelength_centers[key] for key in self.keys]
        phot = []

        interp_x = np.arange(3000, 12000, wavestep) # The x-axis of the spectrum

        # Loop through each of the LSST filters
        for index, key in enumerate(self.keys):

            interp_y = np.interp(interp_x, wavelength, f_lambda) # The interpolated y-axis of the spectrum (in f_lambda)

            # Find the filter curve values at the same x values as interp_x
            filter_interp_y = np.interp(interp_x, self.wavelength[key], self.norm_response[key]) 

            phot.append(np.trapz(filter_interp_y * interp_y * interp_x, x = interp_x)/np.trapz(filter_interp_y * c / interp_x, x = interp_x))

            phot_err = [self.error[key] for key in self.keys]

        wavelengths = np.array(wavelengths)
        phot = np.array(phot)
        phot_err = np.array(phot_err)

        if output_mags:

            if dmag_err:
                #dmag_err is the difference between phot (as a magnitude) and phot-phot_err (as a magnitude)
                phot_err = np.abs(self.flux_to_abmag(phot) - self.flux_to_abmag(abs(phot - phot_err)))
                nans = np.where(np.isnan(phot_err))[0]
            else:
                phot_err = self.flux_to_abmag(phot_err)

            phot = self.flux_to_abmag(phot)

            if len(nans) > 0:
                onesigma = [self.error[self.keys[thisnan]] / 5. for thisnan in nans] # These are five sigma detection limits, so divide by 5
                phot[nans] = 99.
                phot_err[nans] = self.flux_to_abmag(onesigma)

        return wavelengths, phot, phot_err


    def flux_to_abmag(self, flux):

        # Converts erg/s/cm^2/Hz to AB magnitudes

        return -2.5*np.log10(flux) -48.6

    def abmag_to_flux(self, mag):

        return np.power(10., (mag + 48.6)/(-2.5))


    def mag_norm(self, waves, f_nu, norm_mag = 25.):

        phot_wave, phot, phot_err = self.get_photometry(waves, f_nu, output_mags = True, dmag_err = True)

        # -2.5*(log10(fluxnorm) + log10(flux)) - 48.6 = norm_mag
        # dmag = norm_mag - phot
        # norm_flux = 10**(-2.5*dmag)

        dmag = norm_mag - phot
        dmag_single = dmag[np.argmin(phot)] # Pull the brightest magnitude and use it to set the normalization
        flux_norm = 10**(dmag_single/-2.5)

        return waves, f_nu*flux_norm