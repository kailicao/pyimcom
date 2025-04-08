## imports

## File based on https://github.com/kailicao/pyimcom/blob/8fcf7892498ea0b0160d45db33772eb1f7d7a119/historical/OpenUniverse2024/genpsf.py
## Generate input PSF for PyIMCOM. Modified to allow for SEDs different from the default flat SED.

from astropy.table import Table
from astropy.time import Time
import os
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import galsim
import galsim.roman as roman
import numpy as np
import matplotlib.pyplot as plt
from astropy import wcs

import sys
import numpy
import scipy
from scipy.ndimage import gaussian_filter
import galsim
from roman_imsim.utils import roman_utils
from astropy.io import fits

from datetime import datetime
import pytz



def get_psf_fits(obs_data, obsid, outdir, oversample_factor = 8, sed_type = 'flat' , stamp_size = 512, sed = None, eff_const = False , normalize = True):
   
    filter_ =  obsdata[obsid][5]
    roman_filters = roman.getBandpasses(AB_zeropoint=True)
    bpass = roman_filters[filter_]
    constant = 1
    if eff_const:
        constant = bpass.effective_wavelength
    
    st_model = galsim.DeltaFunction()

    if sed_type == 'flat':
        st_model = st_model*galsim.SED(lambda x:constant, 'nm', 'fphotons')
    if sed_type == 'lin':
        st_model = st_model*galsim.SED(lambda x:x, 'nm', 'fphotons')
    if sed_type == 'quad':
        st_model = st_model*galsim.SED(lambda x: x**2, 'nm', 'fphotons')
    if sed_type == 'real':
        assert (sed is not None)
        st_model = st_model*sed
    
    if sed_type == 'flat_lk':
        st_model = st_model*galsim.SED(galsim.LookupTable([100, 2600], [constant,constant], interpolant='linear'),
                              wave_type='nm', flux_type='fphotons')
    if sed_type == 'lin_lk':
        st_model = st_model*galsim.SED(galsim.LookupTable([100, 2600], [100,2600], interpolant='linear'),
                              wave_type='nm', flux_type='fphotons')
    
    if normalize:
        st_model = st_model.withFlux(1.,bpass)
    
    
    # put some information in the Primary HDU
    mainhdu = fits.PrimaryHDU()
    mainhdu.header['CFORMAT'] = 'Legendre basis'
    mainhdu.header['PORDER'] = (1, 'bivariate polynomial order')
    mainhdu.header['ABSCISSA'] =  ('u=(x-2044.5)/2044, v=(y-2044.5)/2044', 'x,y start at 1')
    mainhdu.header['NCOEF'] = (4, '(PORDER+1)**2')
    mainhdu.header['SEQ'] = 'for n=0..PORDER { for m=0..PORDER { coef P_m(u) P_n(v) }}'
    mainhdu.header['OBSID'] = obsid
    mainhdu.header['NSCA'] = 18
    mainhdu.header['OVSAMP'] = 8
    mainhdu.header['SIMRUN'] = 'Theta -> OSC'
    hdulist = [mainhdu]

    # make each layer
    for sca in range(1,19):
        util = roman_utils('was.yaml',image_name='Roman_WAS_simple_model_' + filter_+'_{:d}_{:d}.fits.gz'.format(obsid,sca))
        out_psf = np.zeros((6,stamp_size,stamp_size))
        x_ = [   1.,4088.,   1.,4088.,2044.5, 500.]
        y_ = [   1.,   1.,4088.,4088.,2044.5,1000.]
        for j in range(5):
            x_[j] = .9*x_[j]+.1*2044.5
            y_[j] = .9*y_[j]+.1*2044.5
            psf = util.getPSF(x_[j],y_[j],8)
            psf_image = galsim.Convolve(st_model, galsim.Transform(psf,jac=8*numpy.identity(2)))
            stamp = galsim.Image(stamp_size,stamp_size,wcs=util.wcs)
            arr = psf_image.drawImage(util.bpass,image=stamp,wcs=util.wcs,method='no_pixel')
            out_psf[j,:,:] = arr.array
                    
        out_psf_coef = np.zeros((5,stamp_size,stamp_size))
        out_psf_coef[0,:,:] = out_psf[4,:,:]
        out_psf_coef[1,:,:] = (out_psf[1,:,:] + out_psf[3,:,:] - out_psf[0,:,:] - out_psf[2,:,:])/2./(4087*.9) * 2044
        out_psf_coef[2,:,:] = (out_psf[2,:,:] + out_psf[3,:,:] - out_psf[0,:,:] - out_psf[1,:,:])/2./(4087*.9) * 2044
        out_psf_coef[3,:,:] = (out_psf[0,:,:] + out_psf[3,:,:] - out_psf[1,:,:] - out_psf[2,:,:])/4./(4087*.45)**2 * 2044**2

        # convolution
        sig_mtf = 0.3279*8 # 8 for oversampled pixels
        for j in range(4):
            out_psf_coef[j,:,:] = 0.17519*gaussian_filter(out_psf_coef[j,:,:], 0.4522*sig_mtf, truncate=7.0)\
                                 +0.53146*gaussian_filter(out_psf_coef[j,:,:], 0.8050*sig_mtf, truncate=7.0)\
                                 +0.29335*gaussian_filter(out_psf_coef[j,:,:], 1.4329*sig_mtf, truncate=7.0)

        
        hdu = fits.ImageHDU(out_psf_coef[:4,128:-128,128:-128].astype(np.float32))
        hdu.header['OBSID'] = obsid
        hdu.header['SCA'] = sca
        
        hdulist.append(hdu)
    fits.HDUList(hdulist).writeto(outdir +'/psf_polyfit_{:d}.fits'.format(obsid), overwrite=True)


# get obs data and only use ones were we have preview images from
file_name = '../RomanWAS_preview/Roman_WAS_obseq_11_1_23.fits'

with fits.open(file_name) as myf:
    obsdata = myf[1].data
    obscols = myf[1].columns
    n_obs_tot = len(obsdata.field(0))


from os import listdir
imagedir = '../RomanWAS_preview/images/simple/H158/'
obsid_list = np.array(listdir(imagedir )).astype(int) #list of obs ids in the preview data



## Run psf generation

# parameters
sed_type = 'flat'
sed = None
eff_const = False
normalize = True
outdir = '../RomanWAS_preview/input_psf/psf_flatsed'

## parelelize code for speed

from concurrent.futures import ProcessPoolExecutor

def process_obsid(obsid):
    try:
        print(f"Processing obsid: {obsid}")  # For debugging
        get_psf_fits(obsdata, obsid, outdir, sed_type=sed_type, sed =sed, normalize = normalize, eff_const = eff_const)
    except Exception as e:
        print(f"Error processing obsid {obsid}: {e}")

def parallelize_get_psf_fits(obsid_list, outdir, obsdata):

    with ProcessPoolExecutor() as executor:
        executor.map(process_obsid, obsid_list)

parallelize_get_psf_fits(obsid_list, outdir, obsdata)

