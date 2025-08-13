"""
Noise spectra

Usage: python noisespecs.py <filter> <input prefix> <outstem>

input file name is ``<input prefix><filter>_DD_DD_map.fits``

"""

import sys
import numpy
import healpy
from astropy.io import fits
from astropy import wcs
from os.path import exists
import matplotlib.pyplot as plt
from scipy import ndimage
from collections import namedtuple
import galsim
import json
import re
import numpy as np

nblock = 48
nstart = 0

if len(sys.argv) < 4:
    print('Missing an argument. Usage format is python noisespecs.py <filter> <input prefix> <outstem>\n')
    exit()

# Determine filter to use
filter = sys.argv[1];
nblockuse = 250

if filter=='Y': 
  filtername='Y106'
  area= 7006 #cm^2
if filter=='J': 
  filtername='J129'
  area= 7111
if filter=='H': 
  filtername='H158'
  area= 7340
if filter=='F': 
  filtername='F184'
  area= 4840
if filter == 'W':
  filtername = 'W146'
  area = 22085
if filter == 'K':
  filtername = 'K213'
  area = 4654


# prefix and suffix
in1 = sys.argv[2]
outstem = sys.argv[3]

#Set useful constants
tfr = 3.08 #sec
gain = 1.458 #electrons/DN
ABstd = 3.631*10**(-20) #erg/cm^2
h = 6.626*10**(-27) #erg/Hz
m_ab = 23.9 #sample mag for PS
s_in = 0.11 #arcsec

#Loop through all the blocks
for iblock in range(nstart,nstart+nblockuse):

  j = iblock
  ibx = j%nblock; iby = j//nblock

  #Combine in1 with block ID to get input file and block label
  blockid = '{:s}_{:02d}_{:02d}'.format(filter,ibx,iby)
  infile = in1 + blockid + '_map.fits'

 # extract information from the header of the first file
  if iblock==nstart:
    with fits.open(infile) as f:

      n = numpy.shape(f[0].data)[-1] # size of output images

      config = ''
      for g in f['CONFIG'].data['text'].tolist(): config += g+' '
      configStruct = json.loads(config)
      configdata = f['CONFIG'].data

      mean_coverage = np.mean(np.sum(np.where(f['INWEIGHT'].data[0, :, :, :] > 0, 1, 0), axis=0)[2:-2, 2:-2])

      blocksize = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) * float(configStruct['OUTSIZE'][2]) / 3600. *numpy.pi/180 #block size in radians
      L = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) + 2*int(configStruct['OUTSIZE'][1])* int(configStruct['PAD']) # side length in px

      s_out = float(configStruct['OUTSIZE'][2]) # in arcsec
      force_scale = .40/s_out # in output pixels

      # padding region around the edge
      bdpad = int(configStruct['OUTSIZE'][1]) * int(configStruct['PAD'])

      # figure out which noise layers are there
      layers = [''] + configStruct['EXTRAINPUT']
      print('# Layers:', layers)
      noiselayers = {}
      
      for i in range(len(layers)):
        m = re.match(r'^whitenoise(\d+)$', layers[i])
        if m:
          noiselayers[str(m[0])] = i
        m = re.match(r'^1fnoise(\d+)$', layers[i])
        if m:
          noiselayers[str(m[0])] = i
        m = re.match(r'^labnoise$', layers[i])
        if m:
          noiselayers[str(m[0])] = i
              

    print('# Noise Layers (format is layer:use_slice): ', noiselayers)


  if not exists(infile): 
    continue

  print('# Running file: ' + infile +'\n')
  
  ps2d_all = np.zeros((L//8, L//8, len(noiselayers)))
  ps1d_all = np.zeros((L//16, 4, len(noiselayers)))

  i_layer = 0 # index to track where i am in the dictionary
  for noiselayer in noiselayers:
  
      print('# Noise layer: ', noiselayer)
  
      use_slice = noiselayers[noiselayer]

      f = fits.open(infile)
      indata = np.copy(f[0].data[0, use_slice, :, :]).astype(np.float32)
      f.close()

      nradbins = L//16 # Number of radial bins is side length div. into 8 from binning and then (floor) div. by 2.

      m= re.search(r'white', noiselayer)
      if m:
          norm = (L * (s_in/s_out) )** 2
      m= re.search(r'1f', noiselayer)
      if m:
          norm = (L * (s_in/s_out)) ** 2
      m= re.search(r'lab', noiselayer)
      if m:
          norm = tfr/gain * ABstd/h * area * 10**(-0.4*m_ab) * s_out**2
      
      def measure_power_spectrum(noiseframe, bin=True):
          """
          Measure the 2D power spectrum of image.

          Parameters
          ----------
          noiseframe : np.array
              2D array; the input image to measure the power spectrum of.
          bin : bool, optional
              Whether to bin the 2D spectrum.
              Default=True, bins spectrum into L/8 x L/8 image.
              (Potential extra rows are cut off.)

          Returns
          -------
          np.array
              The 2D power spectrum of the image.

          """
          
          fft = np.fft.fftshift(np.fft.fft2(noiseframe))
          ps = ((np.abs(fft)) ** 2) / norm
          if bin:
              print('# 2D spectrum is 8x8 binned\n')
              binned_ps = np.average(np.reshape(ps, (L//8, 8, L//8, 8)), axis = (1,3))
              print('# Binned PS has shape ', np.shape(binned_ps))
              return binned_ps
          else:
              return ps
      
      
      def _get_wavenumbers(window_length, num_radial_bins=nradbins):
          """
          Calculate wavenumbers for the input image.

          Parameters
          ----------
          window_length : int
              The length of one axis of the image.
          num_radial_bins : int, optional
              The number of radial bins the image should be averaged into.

          Returns
          -------
          kmean : np.array
              The wavenumbers for the image (1D), length `num_radial_bins`.

          """

          k = np.fft.fftshift(np.fft.fftfreq(window_length))
          kx, ky = np.meshgrid(k, k)
          k = np.sqrt(kx ** 2 + ky ** 2)
          k, kmean, kerr = azimuthal_average(k, num_radial_bins)
            
          return kmean
      
      
      def azimuthal_average(image, num_radial_bins=nradbins):
          """
          Compute radial profile of image.
      
          Parameters
          ----------
          image : np.array
              Input image (2D).
          num_radial_bins : int
              Number of radial bins in profile.
      
          Returns
          -------
          r : np.array
              Value of radius at each point (1D, length `num_radial_bins`).
          radial_mean : np.array
              Mean intensity within each annulus. Main result. 1D, length `num_radial_bins`.
          radial_err : np.array
              Standard error on the mean: sigma / sqrt(N). 1D, length `num_radial_bins`.

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
     
      # Set up a named tuple for the results that will contain relevant information
      PspecResults = namedtuple(
          'PspecResults', 'ps_image ps_image_err npix k ps_2d noiselayer'
      )
      
      
      def get_powerspectra(noiseframe, num_radial_bins=nradbins):
          """

          Calculate the azimuthally-averaged 1D power spectrum of the image.

          Parameters
          ----------
          noiseframe : np.array
              the input image to be averaged over (2D)
          num_radial_bins : int
              Number of bins, should match bin number in get_wavenumbers.
      
          Returns
          -------
          results : collections.namedtuple
              Contains the keys 'ps_image', 'ps_image_err', 'npix', 'k', 'ps_2d', 'noiselayer'.

          """
      
          noise = noiseframe.copy()
      
          ps_2d = measure_power_spectrum(noise, bin=True)
      
          ps_r, ps_1d, ps_image_err = azimuthal_average(ps_2d, num_radial_bins)
      
          wavenumbers = _get_wavenumbers(noise.shape[0], num_radial_bins)
      
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
      
      powerspectrum = get_powerspectra(indata)
      
      
      ps2d_all[:,:,i_layer] = powerspectrum.ps_2d
      
      ps1d_all[:, 0, i_layer] = powerspectrum.k
      ps1d_all[:, 1, i_layer] = powerspectrum.ps_image
      ps1d_all[:,2, i_layer] = powerspectrum.ps_image_err
      ps1d_all[:, 3, i_layer] = powerspectrum.noiselayer
            
      i_layer+=1
      
      



  # Reshape things for fits files
  ps2d_all = np.transpose(ps2d_all, (2, 0, 1))
  print('# TRANSPOSED ps2d shape:', np.shape(ps2d_all))
  ps1d_all = np.concatenate((ps1d_all[:, :, 0].reshape(-1, ps1d_all.shape[1]),
                                  ps1d_all[:, :, 1].reshape(-1, ps1d_all.shape[1])), axis=0)
  print('# TRANSPOSED ps1d shape:', np.shape(ps1d_all))

  # Save power spectra data in a fits file
  # Two HDUs: Primary contains 2D spectrum, second is a table with 1D spectrum and MC values
  hdu_ps2d = fits.PrimaryHDU(ps2d_all)
  hdr = hdu_ps2d.header
  hdr['INSTEM'] = in1
  hdr['MEANCOVG'] = mean_coverage
  hdr['LAYERKEY'] = str(noiselayers)
        
  col1 = fits.Column(name='Wavenumber', format='E', array=ps1d_all[:,0])
  col2 = fits.Column(name='Power', format='E', array=ps1d_all[:,1])
  col3 = fits.Column(name='Error', format='E', array=ps1d_all[:,2])
  col4 = fits.Column(name='NoiseLayerID', format='I', array=ps1d_all[:,3])
  p1d_cols = fits.ColDefs([col1, col2, col3, col4])
  hdu_ps1d = fits.BinTableHDU.from_columns(p1d_cols, name='P1D_TABLE')
  

  hdu_config = fits.BinTableHDU(data=configdata, name='CONFIG')
  hdul = fits.HDUList([hdu_ps2d, hdu_config, hdu_ps1d])
  hdul.writeto(outstem + blockid + '_ps.fits', overwrite=True)
  print('# Results saved to ', outstem, blockid, '_ps.fits')
