"""
Report section for noise diagnostics.

Classes
-------
NoiseReport
    Noise report section.

Notes
-----
This incorporates functionality that was previously in noisespecs.py.

It also includes the updated functionality from the Laliotis et al. analysis (August 2024).

This version is trying to implement some things to reduce imcom-related correlations, including:

#. only FFTing the interior postage stamp region (throwing out padded regions)

#. convolving noise images with a window function before FFTing

We have changed the clipping to use the full unique region ``[bdpad:L+bdpad,bdpad:L+bdpad]`` in each image.

"""

import sys
import os
import healpy
from astropy.io import fits
from astropy import wcs
from os.path import exists
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.colors as colors

from scipy import ndimage
from collections import namedtuple
from skimage.filters import window
import galsim
import json
import re
import numpy as np

from ..config import Settings
from .report import ReportSection

RomanFilters = ['W146', 'F184', 'H158', 'J129', 'Y106', 'Z087', 'R062', 'PRSM', 'DARK', 'GRSM', 'K213']

AreaArray = [22085, 4840, 7340, 7111, 7006, 6635, 9011, 0,0,0, 4654]
# in cm^2; 0 for prism/dark/grism as these are not imaging elements

# Set up a named tuple for the results that will contain relevant information
PspecResults = namedtuple(
            'PspecResults', 'ps_image ps_image_err npix k ps_2d noiselayer'
)

class NoiseReport(ReportSection):
    """
    The noise section of the report.

    Inherits from pyimcom.diagnostics.report.ReportSection. Overrides build.
    
    """

    def build(self, nblockmax=100, m_ab=23.9, bin_flag=1, alpha=0.9, tarfiles=True):
        """
        Builds the noise section of the report.

        Parameters
        ----------
        nblockmax : int, optional
            Maximum size of mosaic to build.
        m_ab : float, optional
            Scaling magnitude (not currently used).
        bin_flag : int, optional
            Whether to bin? (1 = bin 8x8, 0 = do not bin)
        alpha : float, optional
            Tukey window width for noise power spectrum.
        tarfiles : bool, optional
            Generate a tarball of the data files?

        Returns
        -------
        None

        """

        # pulled the reference magnitude back up here
        # also the binning flag (1 = bin 8x8, 0 = do not bin)

        self.nblock = min(nblockmax, self.cfg.nblock)
        self.psfiles = [] # will keep appending so all the power spectrum files with all information in their names is in here
        self.suffix = '' # added to block output file

        self.tex += '\\section{Injected noise layers section}\n\n'

        # example output slabs for white, 1/f, and lab noise
        self.outslab = [None,None,None]

        # there are several sets of files to build here
        self.build_noisespec(m_ab,bin_flag,alpha)
        self.average_spectra(bin_flag)

        # make one example figure, the 2D power spectrum
        self.gen_overview_fig()

        # add variances
        filter = Settings.RomanFilters[self.cfg.use_filter][0]
        for k in range(len(self.orignames)):
            with fits.open(self.datastem + '_' + filter + self.suffix + '_ps_avg.fits') as f:
                self.data += 'LAYER{:02d}'.format(k) + ' ' + '{:24s}'.format(self.orignames[k])\
                             + ' {:11.5E}\n'.format(np.average(f[0].data[k,:,:])/self.s_out**2)

        # tarball the files if requested
        if tarfiles:
            tarfile_head, tarfile_tail = os.path.split(self.datastem + '_blockps' + self.suffix + '.tar')
            lf = []
            for f in self.psfiles:
                pdir,pname = os.path.split(f)
                lf += [pname]
            pwd = os.getcwd()
            os.chdir(pdir)
            proc = subprocess.run(['tar', '--create', '--file='+tarfile_tail] + lf, capture_output=True)
            print('tar output -->\n', proc.stdout)
            print('errors -->\n', proc.stderr)
            for f in lf: os.remove(f)
            os.chdir(pwd)

    # --- noisespec --- #

    def build_noisespec(self, m_ab, bin_flag, alpha):
        """
        Computes the noise power spectrum.

        Parameters
        ----------
        m_ab : float
            Reference star brightness (not used)
        bin_flag : int, optional
            Whether to bin? (1 = bin 8x8, 0 = do not bin)
        alpha : float, optional
            Tukey window width for noise power spectrum.

        Returns
        -------
        str
            Status string ('Completed').

        """

        pars = [] # list of parameters, replaces global construction when wrapped

        #Set useful constants for te lab noise data
        tfr = 3.08 #sec
        gain = 1.458 #electrons/DN
        ABstd = 3.631e-20 #erg/cm^2
        h_erg = 6.62607015e-27 #erg/Hz
        h_jy = h_erg*1e29 #microJy*cm^2*s
        s_in = Settings.pixscale_native * 648000./np.pi #arcsec # updated to refer back to Settings
        B0 = 0.38 #e/px/s, background estimate
        t_exp = 139.8 #s

        area = AreaArray[self.cfg.use_filter]
        filter = Settings.RomanFilters[self.cfg.use_filter][0]

        # extra background to add for lab noise
        B1 = 0.
        if filter=='K': B1 = 4.65

        # which blocks?
        is_first = True
        for iby in range(self.nblock):
            for ibx in range(self.nblock):

                blockid = '{:s}_{:02d}_{:02d}'.format(filter,ibx,iby)
                if alpha>0:
                    win=True
                    blockid += '_alpha_'+str(alpha)
                else:
                    win=False
                if bin_flag==0: blockid += '_nobin'

                # loop over blocks
                infile = self.infile(ibx,iby)

                if not exists(infile):
                    return None

                # the first time
                if is_first:
                    is_first=False

                    with fits.open(infile) as f:
                        n = np.shape(f[0].data)[-1] # size of output images
                        config = ''
                        for g in f['CONFIG'].data['text'].tolist(): config += g+' '
                        configStruct = json.loads(config)
                        configdata = f['CONFIG'].data

                    blocksize = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) * float(configStruct['OUTSIZE'][2]) / 3600. *np.pi/180 #block size in radians
                    L = self.L = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) # side length in px
                    # snap to nearest multiple of 16
                    L = (L//16)*16

                    self.s_out = s_out = float(configStruct['OUTSIZE'][2]) # in arcsec

                    # size of padding region around each edge (in px)
                    bdpad = int(configStruct['OUTSIZE'][1]) * int(configStruct['PAD'])

                    # figure out which noise layers are there
                    layers = [''] + configStruct['EXTRAINPUT']
                    print('# Layers:', layers)
                    noiselayers = {}
                    for i in range(len(layers)):
                        m = re.match(r'^whitenoise(\d+)$', layers[i])
                        if m:
                            noiselayers[str(m[0])] = i
                            whitenoisekey = str(m[0])
                        m = re.match(r'^1fnoise(\d+)$', layers[i])
                        if m:
                            noiselayers[str(m[0])] = i
                        m = re.match(r'^labnoise$', layers[i])
                        if m:
                            noiselayers[str(m[0])] = i
                    print('# Noise Layers (format is layer:use_slice): ', noiselayers)

                print('# Running file: ' + infile, 'whitenoisekey =', whitenoisekey)

                # mean coverage
                with fits.open(infile) as f:
                    mean_coverage = np.mean(np.sum(np.where(f['INWEIGHT'].data[0, :, :, :] > 0, 1, 0), axis=0)[2:-2, 2:-2])

                if bin_flag==0:
                    ps2d_all = np.zeros((L, L, len(noiselayers)))
                    ps1d_all = np.zeros((L//2, 4, len(noiselayers)))
                elif bin_flag==1:
                    ps2d_all = np.zeros((L//8, L//8, len(noiselayers)))
                    ps1d_all = np.zeros((L//16, 4, len(noiselayers)))
                else:
                    raise Exception('Error: bin flag must be 0 (no binning) or 1 (8x8 binning)')

                i_layer = 0
                for noiselayer in noiselayers:
                    use_slice = noiselayers[noiselayer]
                    with fits.open(infile) as f:
                        indata = np.copy(f[0].data[0, use_slice, bdpad:L+bdpad, bdpad:L+bdpad]).astype(np.float32)
                    nradbins = L//16 # Number of radial bins is side length div. into 8 from binning and then (floor) div. by 2.
                                     # Note that with the new clipping this is L again, the Aug. 2024 clipping needed (L-bdpad)
                    if bin_flag==0: nradbins *= 8

                    norm = (L / s_out )** 2 # normalization factor to get power spectrum in units of [image units]^2 * arcsec^2
                    # special normalization for the lab data
                    m= re.search(r'lab', noiselayer)
                    if m:
                        norm_LN = (s_in**2)*area*tfr/(h_jy*gain) #factor to convert LN from flux DN/fr/s to intensity microJy/arcsec^2
                        if filter == 'K':
                            with fits.open(infile) as f:
                                wndata = np.copy(f[0].data[0, noiselayers[whitenoisekey], bdpad:L+bdpad, bdpad:L+bdpad]).astype(np.float32)
                            wndata*=np.sqrt((B1-B0)/t_exp)*tfr/gain #convert WN to DN/fr
                            indata+=wndata #add to lab noise
                        indata = indata/norm_LN

                    powerspectrum = NoiseReport.get_powerspectra(indata, L, norm, nradbins, use_slice=use_slice, bin_flag=bin_flag, win=win, alpha=alpha)
                        # norm and nradbins now required arguments; use_slice is needed to get the correct format in the output file
                    ps2d_all[:,:,i_layer] = powerspectrum.ps_2d
                    ps1d_all[:, 0, i_layer] = powerspectrum.k
                    ps1d_all[:, 1, i_layer] = powerspectrum.ps_image
                    ps1d_all[:,2, i_layer] = powerspectrum.ps_image_err
                    ps1d_all[:, 3, i_layer] = powerspectrum.noiselayer

                    i_layer+=1

                # Reshape things for fits files
                ps2d_all = np.transpose(ps2d_all, (2, 0, 1))
                #print('# TRANSPOSED ps2d shape:', np.shape(ps2d_all))
                # reshape P1D's:
                ps1d_all = np.transpose(ps2d_all, (2,0,1)).reshape(-1,np.shape(ps2d_all)[1]).T
                #print('# TRANSPOSED ps1d shape:', np.shape(ps1d_all))

                # Save power spectra data in a fits file
                # Two HDUs: Primary contains 2D spectrum, second is a table with 1D spectrum and MC values
                hdu_ps2d = fits.PrimaryHDU(ps2d_all)
                hdr = hdu_ps2d.header
                hdr['INSTEM'] = infile[:-11] # updated from original script
                hdr['MEANCOVG'] = mean_coverage
                hdr['LAYERKEY'] = str(noiselayers)
                hdr['NLAYERS'] = (len(noiselayers), 'Number of layers with noise')
                key_layer2 = ['']*(1 + len(configStruct['EXTRAINPUT']))
                for d in noiselayers.keys():
                    d_ = noiselayers[d]
                    key_layer2[d_]=d
                key_layer = []
                self.orignames = []
                for k in range(len(key_layer2)):
                    if key_layer2[k]!='':
                        key_layer.append(key_layer2[k])
                        self.orignames.append(configStruct['EXTRAINPUT'][k-1])
                del key_layer2
                for il in range(len(noiselayers)):
                    key_ = 'LAYER{:02d}'.format(il)
                    hdr[key_] = (key_layer[il], 'Noise layer '.format(il))
                    if key_layer[il][:10]=='whitenoise': self.outslab[0]=il
                    if key_layer[il][:7]=='1fnoise': self.outslab[1]=il
                    if key_layer[il][:8]=='labnoise': self.outslab[2]=il
                del key_
                hdr['AREAUNIT'] = 'arcsec**2'

                col1 = fits.Column(name='Wavenumber', format='E', array=ps1d_all[:,0])
                col2 = fits.Column(name='Power', format='E', array=ps1d_all[:,1])
                col3 = fits.Column(name='Error', format='E', array=ps1d_all[:,2])
                col4 = fits.Column(name='NoiseLayerID', format='I', array=ps1d_all[:,3])
                p1d_cols = fits.ColDefs([col1, col2, col3, col4])
                hdu_ps1d = fits.BinTableHDU.from_columns(p1d_cols, name='P1D_TABLE')

                hdu_config = fits.BinTableHDU(data=configdata, name='CONFIG')
                hdul = fits.HDUList([hdu_ps2d, hdu_config, hdu_ps1d])
                self.psfiles.append(self.datastem + '_' + blockid + '_ps.fits')
                hdul.writeto(self.psfiles[-1], overwrite=True)
                print('# Results saved to ', self.datastem, '_', blockid, '_ps.fits')

        self.suffix = blockid[7:]
        self.noiselayers = noiselayers # save this for reference later
        return 'Completed'

    ## Utility functions below here ##

    @staticmethod
    def measure_power_spectrum(noiseframe, L, norm=1., bin=True, win=True, alpha=0.9):
        """
        Measure the 2D power spectrum of image.

        Parameters
        ----------
        noiseframe : np.array
            The 2D input image to measure the power spectrum of.
            In this case, a noise frame from the simulations
        L : int
            The length of the FFT (must be a multiple of 8).
        norm : float, optional
            The normalization to use (power spectrum is |FFT|^2/norm).
        bin : bool, optional
            Whether to bin the 2D spectrum.
            Default=True, bins spectrum into L/8 x L/8 image.
            Potential extra rows are cut off.
        win : bool, optional
            Whether to convolve the noise frame with a Tukey window function.
        alpha : float, optional
            Tukey window parameter.

        Returns
        -------
        np.array
            The 2D power spectrum of the image.

        """

        # get the window function and its normalization
        if win:
            w = window(('tukey', alpha), (np.shape(noiseframe)))
            norm = norm * np.average(w**2)
            noiseframe = noiseframe * w

        fft = np.fft.fftshift(np.fft.fft2(noiseframe))
        ps = ((np.abs(fft)) ** 2) / norm
        if bin:
            #print('# 2D spectrum is 8x8 binned\n')
            binned_ps = np.average(np.reshape(ps, ( L//8, 8, L//8, 8)), axis = (1,3))
            #print('# Binned PS has shape ', np.shape(binned_ps))
            return binned_ps
        else:
            return ps

    @staticmethod
    def _get_wavenumbers(window_length, num_radial_bins):
        """
        Calculate wavenumbers for the input image.

        Parameters
        ----------
        window_length : int
            The length of one axis of the image.
        num_radial_bins: int
            Number of radial bins the image should be averaged into.

        Returns
        -------
        kmean : np.array
            1D array of the wavenumbers for the image

        """

        k = np.fft.fftshift(np.fft.fftfreq(window_length))
        kx, ky = np.meshgrid(k, k)
        k = np.sqrt(kx ** 2 + ky ** 2)
        k, kmean, kerr = NoiseReport.azimuthal_average(k, num_radial_bins)

        return kmean

    @staticmethod
    def azimuthal_average(image, num_radial_bins):
        """
        Compute radial profile of image.

        Parameters
        ----------
        image : np.array
            Input image, 2D.
        num_radial_bins : int
            Number of radial bins in profile.

        Returns
        -------
        r : np.array
            Value of radius at each point
        radial_mean : np.array
            Mean intensity within each annulus. Main result
        radial_err : np.array
            Standard error on the mean: sigma / sqrt(N).

        """

        ny, nx = image.shape
        yy, xx = np.mgrid[:ny, :nx]
        center = np.array(image.shape) / 2

        r = np.hypot(xx - center[1], yy - center[0])
        rbin = (num_radial_bins * r / r.max()).astype(int)

        radial_mean = ndimage.mean(
            image, labels=rbin, index=np.arange(1, rbin.max() + 1))
        radial_stddev = ndimage.standard_deviation(
            image, labels=rbin, index=np.arange(1, rbin.max() + 1))
        npix = ndimage.sum(np.ones_like(image), labels=rbin,
                           index=np.arange(1, rbin.max() + 1))

        radial_err = radial_stddev / np.sqrt(npix)
        return r, radial_mean, radial_err

    @staticmethod
    def get_powerspectra(noiseframe, L, norm, num_radial_bins, use_slice=-1, bin_flag=1, win=True, alpha=0.9):
        """
        Calculate the azimuthally-averaged 1D power spectrum of the image.

        Parameters
        ----------
        noiseframe: np.array
            The 2D input image to be averaged over.
        L : int
            Length of FFT (must be a multiple of 8).
        norm : float
            Normalization of |FFT|^2->power spectrum.
        num_radial_bins : int
            Number of bins, should match bin number in get_wavenumbers
        use_slice : int, optional
            Noise slice number used.
        bin_flag : int, optional
            Binning? (1=yes, 0=no).

        Returns
        -------
        results : collection.namedtuple
            Power spectrum results.

        """

        noise = noiseframe.copy()
        if bin_flag==0:
            ps_2d = NoiseReport.measure_power_spectrum(noise, L, norm=norm, bin=False, win=win, alpha=alpha)
        else:
            ps_2d = NoiseReport.measure_power_spectrum(noise, L, norm=norm, bin=True, win=win, alpha=alpha)
        ps_r, ps_1d, ps_image_err = NoiseReport.azimuthal_average(ps_2d, num_radial_bins)
        wavenumbers = NoiseReport._get_wavenumbers(noise.shape[0], num_radial_bins)
        npix = np.product(noiseframe.shape)
        comment = [use_slice] * num_radial_bins

        # consolidate results
        results = PspecResults(ps_image=ps_1d,
                               ps_image_err=ps_image_err,
                               npix=npix,
                               k=wavenumbers,
                               ps_2d = ps_2d,
                               noiselayer=comment
                               )
        return results

    # --- average_spectra --- #

    def average_spectra(self, bin_flag):
        """
        Averages together all the power spectra in one band.

        Parameters
        ----------
        bin_flag
            Whether to bin? (1 = bin 8x8, 0 = do not bin)

        Returns
        -------
        None

        """

        for iblock in range(self.nblock**2):
            ibx = iblock%self.nblock
            iby = iblock//self.nblock

            infile = self.psfiles[iblock]
            print(ibx,iby,infile); sys.stdout.flush()

            # extract information from the header of the first file
            if iblock==0:
                with fits.open(infile) as f:
                    n = np.shape(f['PRIMARY'])[0]
                    l = (f['P1D_TABLE'].data).shape[0]
                    total_2D = np.zeros( np.shape(np.transpose(f['PRIMARY'].data, (1, 2, 0))) )
                    total_1D = np.zeros( (l, 4) )
                    header = np.copy(f['PRIMARY'].header)

            if not exists(infile):
                continue

            with fits.open(infile) as f:
                indata_2D = np.copy(np.transpose(f['PRIMARY'].data, (1, 2, 0))).astype(np.float32)
                indata_1D = np.copy(f['P1D_TABLE'].data)

            for k in range(0, n):
                total_2D[:, :, k] += indata_2D[:, :, k]
            for k in range(0, l):
                for m in range(0,4):
                    total_1D[k, m] += indata_1D[k][m]

        for k in range(0, n):
            total_2D[:, :, k] = total_2D[:, :, k] / self.nblock**2
        total_1D = total_1D/self.nblock**2

        hdu1 = fits.PrimaryHDU(np.transpose(total_2D, (2, 0, 1)))
        hdr = hdu1.header

        with fits.open(self.psfiles[0]) as g:
            copykey = ['INSTEM', 'LAYERKEY', 'NLAYERS']
            for il in range(len(self.noiselayers)):
                copykey.append('LAYER{:02d}'.format(il))
            copykey.append('AREAUNIT')
            for key in copykey:
                hdr[key] = g[0].header[key]

        col1 = fits.Column(name='Wavenumber', format='E', array=total_1D[:,0])
        col2 = fits.Column(name='Power', format='E', array=total_1D[:,1])
        col3 = fits.Column(name='Error', format='E', array=total_1D[:,2])
        col4 = fits.Column(name='NoiseLayerID', format='I', array=total_1D[:,3])
        p1d_cols = fits.ColDefs([col1, col2, col3, col4])
        hdu_ps1d = fits.BinTableHDU.from_columns(p1d_cols, name='P1D_TABLE')

        hdul = fits.HDUList([hdu1, hdu_ps1d])
        filter = Settings.RomanFilters[self.cfg.use_filter][0]
        outfile = self.datastem + '_' + filter + self.suffix + '_ps_avg.fits'
        hdul.writeto(outfile, overwrite = True)
        print('# Average power spectrum saved to '+outfile )

    # --- figures --- #
    def gen_overview_fig(self):
        """
        Makes a simple overview figure.


        Returns
        -------
        str
            File name of the figure written.

        """

        filter = Settings.RomanFilters[self.cfg.use_filter][0]
        print(self.outslab)

        matplotlib.rcParams.update({'font.size': 10})
        F = plt.figure(figsize=(9,5.5))
        ntypes = ['white', '1/f', 'lab']
        vmax = [.01,.3,.05]
        pos = ['Left', 'Center', 'Right']
        um = .5/self.s_out
        unit_ = ['arcsec$^2$', 'arcsec$^2$', '$\mu$Jy$^2$/arcsec$^2$']
        for k in range(3):
            if self.outslab[k] is not None:
                S = F.add_subplot(1,3,k+1)
                S.set_title('Power spectrum: ' + ntypes[k] + ' noise\n' + unit_[k], usetex=True)
                S.set_xlabel('u [cycles/arcsec]')
                S.set_ylabel('v [cycles/arcsec]')
                with fits.open(self.datastem + '_' + filter + self.suffix + '_ps_avg.fits') as f:
                    im = S.imshow(f[0].data[self.outslab[k],:,:], cmap='gnuplot', aspect=1, interpolation='nearest', origin='lower',
                          extent = (-um,um,-um,um), norm=colors.LogNorm(vmin=vmax[k]/300.,vmax=vmax[k]*1.0000001,clip=True))
                F.colorbar(im, location='bottom')
        outfile = self.datastem + '_' + filter + self.suffix + '_3panel.pdf'
        F.set_tight_layout(True)
        F.savefig(outfile)
        plt.close(F)

        # the caption
        self.tex += '\\begin{figure}\n'
        self.tex += '\\includegraphics[width=6.5in]{' + self.datastem_from_dir + '_' + filter + self.suffix + '_3panel.pdf}\n'
        self.tex += '\\caption{\\label{fig:noise3panel}The 2D power spectra of the noise realizations.\n'
        for k in range(3):
            self.tex += ' {\em ' + pos[k] + ' panel} (' + ntypes[k] + ' noise): '
            if self.outslab[k] is not None:
                self.tex += 'layer {:d} (in output file), name='.format(self.outslab[k]) + '{\\tt ' + self.orignames[self.outslab[k]] + '}.'
            else:
                self.tex += 'not run.'
            self.tex += ' \n'
        self.tex += '}\n\\end{figure}\n\n'

        self.tex += 'The noise power spectra are shown in Fig.~\\ref{fig:noise3panel}.\n'

        return outfile
