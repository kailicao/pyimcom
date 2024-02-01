# usage: python noisespecs.py <filter> <noisetype> <input prefix> <outstem> 
# input file name is <input prefix><filter>_DD_DD_map.fits

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

#Check input format
if len(sys.argv) < 5:
    print('Missing an argument. Usage format is python noisespecs.py <filter> <noisetype> <input prefix> <outstem>\n')
    exit()

# Determine filter to use
filter = sys.argv[1]; nblockuse = 500

if filter=='Y': 
  filtername='Y106'
  area= 5915 #cm^2
if filter=='J': 
  filtername='J129'
  area= 6051
if filter=='H': 
  filtername='H158'
  area= 5978
if filter=='F': 
  filtername='F184'
  area= 3929

# Determine type of noise to use (white or 1/f)
if sys.argv[2] == 'white' or sys.argv[2] == 'w' or sys.argv[2] == 'White' or sys.argv[2] == 'W':
    noisetype = 'W'
elif sys.argv[2] == 'f' or sys.argv[2] == '1f' or sys.argv[2] == 'F' or sys.argv[2] == '1F':
    noisetype = 'F'
elif sys.argv[2] == 'l' or sys.argv[2] == 'lab' or sys.argv[2] == 'L' or sys.argv[2] == 'Lab':
    noisetype = 'L'

# prefix and suffix
in1 = sys.argv[3]
outstem = sys.argv[4]

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
  label = blockid + '_' + noisetype
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
      configdata = np.append(configdata, np.array([('    "MEANCOVG:" '+str(mean_coverage))], dtype=configdata.dtype))

      blocksize = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) * float(configStruct['OUTSIZE'][2]) / 3600. *numpy.pi/180 # radians

      s_out = float(configStruct['OUTSIZE'][2]) # in arcsec
      force_scale = .40/s_out # in output pixels

      # padding region around the edge
      bdpad = int(configStruct['OUTSIZE'][1]) * int(configStruct['PAD'])

      # figure out which layer we want
      layers = [''] + configStruct['EXTRAINPUT']
      print('# Layers:', layers)
      for i in range(len(layers))[::-1]:
        if noisetype == 'W':
            m = re.match(r'^whitenoise(\d+)$', layers[i])
            if m:
              use_slice = i
        elif noisetype == 'F':
            m = re.match(r'^1fnoise(\d+)$', layers[i])
            if m:
              use_slice = i
        elif noisetype == 'L':
            m = re.match(r'^labnoise$', layers[i])
            if m:
              use_slice = i

#  print('# Analyzing ', noisetype, ' using layer', use_slice, ', output pix =', s_out, ' arcsec,   n=', n)


  if not exists(infile): 
    continue

  print('# Running file: ' + infile +'\n')

  f = fits.open(infile)
  indata = np.copy(f[0].data[0, use_slice, :, :]).astype(np.float32)
  f.close()

  L = indata.shape[0] #side length of blocks
  nradbins = L//16 # Number of radial bins is side length div. into 8 from binning and then (floor) div. by 2.
  
  norm = tfr/gain * ABstd/h * area * 10**(-0.4*m_ab) * s_out**2
  
  def measure_power_spectrum(noiseframe, bin=True):
      """
      Measure the 2D power spectrum of image.
      :param noiseframe: 2D ndarray
       the input image to measure the power spectrum of.
       in this case, a noise frame from the simulations
      :param bin: True/False
       Whether to bin the 2D spectrum.
       Default=True, bins spectrum into L/8 x L/8 image. 
               (Potential extra rows are cut off.)
      :return: 2D ndarray, ps
       the 2D power spectrum of the image.
      """
      noiseframe = noiseframe/norm
      fft = np.fft.fftshift(np.fft.fft2(noiseframe))
      ps = ((np.abs(fft)) ** 2) / ((L * (s_in/s_out) ** 2))
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
      :param window_length: integer
       the length of one axis of the image.
      :param num_radial_bins: integer
       number of radial bins the image should be averaged into
      :return: 1D np array, kmean
       the wavenumbers for the image
      """
      k = np.fft.fftshift(np.fft.fftfreq(window_length))
      kx, ky = np.meshgrid(k, k)
      k = np.sqrt(kx ** 2 + ky ** 2)
      k, kmean, kerr = azimuthal_average(k, num_radial_bins)
  
      # print('k shape: ', k.shape)
  
      return kmean
  
  
  def azimuthal_average(image, num_radial_bins=nradbins):
      """
      Compute radial profile of image.
  
      Parameters
      ----------
      image : 2D ndarray
          Input image.
      num_radial_bins : int
          Number of radial bins in profile.
  
      Returns
      -------
      r : ndarray
          Value of radius at each point
      radial_mean : ndarray
          Mean intensity within each annulus. Main result
      radial_err : ndarray
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
 
  # Set up a named tuple for the results that will contain relevant information
  PspecResults = namedtuple(
      'PspecResults', 'ps_image ps_image_err npix k ps_2d'
  )
  
  
  def get_powerspectra(noiseframe, num_radial_bins=nradbins):
      """
      Calculate the azimuthally-averaged 1D power spectrum of the image
      :param noiseframe: 2D ndarray
          the input image to be averaged over
      :param num_radial_bins: number of bins, should match bin number in get_wavenumbers
  
      :return: named tuple, results
      """
  
      noise = noiseframe.copy()
  
      ps_2d = measure_power_spectrum(noise, bin=True)
  
      ps_r, ps_1d, ps_image_err = azimuthal_average(ps_2d, num_radial_bins)
  
      wavenumbers = _get_wavenumbers(noise.shape[0], num_radial_bins)
  
      npix = np.product(noiseframe.shape)

      # consolidate results
      results = PspecResults(ps_image=ps_1d,
                             ps_image_err=ps_image_err,
                             npix=npix,
                             k=wavenumbers,
                             ps_2d = ps_2d
                             )
  
      return results
  
  powerspectrum = get_powerspectra(indata)
  ps_data = np.column_stack((powerspectrum.k, powerspectrum.ps_image))

# Save power spectra data in a fits file
# Two HDUs: Primary contains 2D spectrum, second is a table with 1D spectrum and MC values

#  print('# K shape: ', powerspectrum.k.shape)
#  print('# 1D image shape: ', powerspectrum.ps_image.shape)
#  print('# Image Err shape: ', powerspectrum.ps_image_err.shape)

  hdu_ps2d = fits.PrimaryHDU(powerspectrum.ps_2d)
  col1 = fits.Column(name='Wavenumber', format='E', array=powerspectrum.k)
  col2 = fits.Column(name='Power', format='E', array=powerspectrum.ps_image)
  col3 = fits.Column(name='Error', format='E', array=powerspectrum.ps_image_err)
  p1d_cols = fits.ColDefs([col1, col2, col3])
  hdu_ps1d = fits.BinTableHDU.from_columns(p1d_cols, name='P1D_TABLE')
  hdu_config = fits.BinTableHDU(data=configdata, name='CONFIG')
  hdul = fits.HDUList([ hdu_ps2d, hdu_config, hdu_ps1d])
  hdul.writeto(outstem + label + 'ps.fits', overwrite=True)
  print('# Results saved to ', outstem, label, 'ps.fits')
