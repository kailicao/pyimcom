"""
Program to average together all the power spectra in one band.

Usage format is python average_spectra.py <filter> <inpath> <outpath>

KL: add MC binning to this?

"""

import sys
import numpy as np
from astropy.io import fits
from os.path import exists

band = sys.argv[1]
inpath = sys.argv[2]
outpath = sys.argv[3]

nblock = 36
nstart = 0

if len(sys.argv) < 4:
    print('Missing an argument. Usage format is python noisespecs.py <filter> <input prefix> <outstem>\n')
    exit()

nblockuse = nblock**2

#Loop through all the blocks
for iblock in range(nstart,nstart+nblockuse):

  j = iblock
  ibx = j%nblock; iby = j//nblock

  #Combine in1 with block ID to get input file and block label
  blockid = '{:s}_{:02d}_{:02d}'.format(band,ibx,iby)
  infile = inpath + blockid + '_ps.fits'

 # extract information from the header of the first file
  if iblock==nstart:
    with fits.open(infile) as f:

      n = np.shape(f['PRIMARY'])[0]
      l = (f['P1D_TABLE'].data).shape[0]

      total_2D = np.zeros( np.shape(np.transpose(f['PRIMARY'].data, (1, 2, 0))) )
      total_1D = np.zeros( (l, 4) )
     
      header = np.copy(f['PRIMARY'].header)
      
  if not exists(infile):
    continue
    
  f = fits.open(infile)
  indata_2D = np.copy(np.transpose(f['PRIMARY'].data, (1, 2, 0))).astype(np.float32)
  indata_1D = np.copy(f['P1D_TABLE'].data)
  f.close()

  for k in range(0, n):
    total_2D[:, :, k] += indata_2D[:, :, k]
    
  for k in range(0, l):
    for m in range(0,4):
        total_1D[k, m] += indata_1D[k][m]
    
for k in range(0, n):
    total_2D[:, :, k] = total_2D[:, :, k] / nblockuse

total_1D= total_1D/nblockuse
    
hdu1 = fits.PrimaryHDU(np.transpose(total_2D, (2, 0, 1)))
hdr = hdu1.header

col1 = fits.Column(name='Wavenumber', format='E', array=total_1D[:,0])
col2 = fits.Column(name='Power', format='E', array=total_1D[:,1])
col3 = fits.Column(name='Error', format='E', array=total_1D[:,2])
col4 = fits.Column(name='NoiseLayerID', format='I', array=total_1D[:,3])
p1d_cols = fits.ColDefs([col1, col2, col3, col4])
hdu_ps1d = fits.BinTableHDU.from_columns(p1d_cols, name='P1D_TABLE')

hdul = fits.HDUList([hdu1, hdu_ps1d])
hdul.writeto(band+'_ps_avg.fits', overwrite = True)
print('# Average Y band power spectrum saved to '+band+'_ps_avg.fits' )
