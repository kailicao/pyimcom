# code to construct dynamic range estimates from block files
#

# usage:
# python dyrange.py <pathstem>

nscale=1
# nscale=10 # this line is for bug compensation --- will remove it later

import warnings
import sys
import numpy
import healpy
from astropy.io import fits
from astropy import wcs
from os.path import exists
import json
import re
from .outimage_utils.helper import HDU_to_bels

# Inputs
pathstem = sys.argv[1]

# ... and radius (needs to be integer <bd)
rpix = 50

# initialize table of pixel values
vals = []
for j in range(rpix):
  vals += [numpy.zeros((0,))]

is_first = True

nblockmax = 512 # maximum number of blocks to consider

# histogram initialization
N_noise = 100
d_noise = .02
countnoise = numpy.zeros((N_noise,2))
countnoise[:,0] = d_noise*numpy.linspace(.5,N_noise-.5,N_noise)
tnoise = 0.
tnoise_gt = 0.
N_neff = 100
d_neff = .1
countneff = numpy.zeros((N_neff,2))
countneff[:,0] = d_neff*numpy.linspace(.5,N_neff-.5,N_neff)
tneff = 0.
tneff_gt = 0.

# now loop over the blocks
for iby in range(nblockmax):
  for ibx in range(nblockmax):
    infile = pathstem + '_{:02d}_{:02d}.fits'.format(ibx,iby)
    if not exists(infile): continue

    # if this is the first block we find, get the configuration file
    if is_first:
      is_first = False
      with fits.open(infile) as f:
        config = ''
        for g in f['CONFIG'].data['text'].tolist(): config += g+' '
        configStruct = json.loads(config)

        blocksize = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) * float(configStruct['OUTSIZE'][2]) / 3600. *numpy.pi/180 # radians
        rs = 1.5*blocksize/numpy.sqrt(2.) # search radius

        # padding region around the edge
        bd = int(configStruct['OUTSIZE'][1]) * int(configStruct['PAD'])

        # figure out which layer we want
        layers = [''] + configStruct['EXTRAINPUT']
        for i in range(len(layers))[::-1]:
          m = re.match(r'^nstar(\d+),', layers[i])
          if m:
            framenumber = i
            res = int(m.group(1))
        print('# using layer', framenumber, 'resolution', res)
        print('# rs=', rs)

    # now we know this file exists
    with fits.open(infile) as f:
      n = numpy.shape(f[0].data)[-1]
      mywcs = wcs.WCS(f[0].header)
      starmap = f[0].data[0,framenumber,:,:]

      # now extract histogram information
      try:
        sigma_ = 10**(-.5*HDU_to_bels(f['SIGMA'])*f['SIGMA'].data[0,bd:-bd,bd:-bd]) # noise standard deviation in units of input noise
        for j in range(N_noise):
          countnoise[j,1] = countnoise[j,1] + numpy.count_nonzero(numpy.logical_and(sigma_/d_noise>=j, sigma_/d_noise<j+1))
        tnoise = tnoise + numpy.size(sigma_)
        tnoise_gt = tnoise_gt + numpy.count_nonzero(sigma_>=d_noise*N_noise)
      except:
        warnings.warn('No valid noise frame: '+infile)

      try:
        neff_ = 10**(HDU_to_bels(f['EFFCOVER'])*f['EFFCOVER'].data[0,bd:-bd,bd:-bd]*nscale) # effective coverage
        for j in range(N_neff):
          countneff[j,1] = countneff[j,1] + numpy.count_nonzero(numpy.logical_and(neff_/d_neff>=j, neff_/d_neff<j+1))
        tneff = tneff + numpy.size(neff_)
        tneff_gt = tneff_gt + numpy.count_nonzero(neff_>=d_neff*N_neff)
      except:
        warnings.warn('No valid coverage frame: '+infile)

    # identify which HEALpix positions we have
    ra_cent, dec_cent = mywcs.all_pix2world([(n-1)/2], [(n-1)/2], [0.], [0.], 0, ra_dec_order=True)
    ra_cent = ra_cent[0]; dec_cent = dec_cent[0]
    vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
    qp = healpy.query_disc(2**res, vec, rs, nest=False)
    ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=False, lonlat=True)
    npix = len(ra_hpix)
    x, y, z1, z2 = mywcs.all_world2pix(ra_hpix, dec_hpix, numpy.zeros((npix,)), numpy.zeros((npix,)), 0)
    xi = numpy.rint(x).astype(numpy.int16); yi = numpy.rint(y).astype(numpy.int16)
    grp = numpy.where(numpy.logical_and(numpy.logical_and(xi>=bd,xi<n-bd),numpy.logical_and(yi>=bd,yi<n-bd)))
    ra_hpix = ra_hpix[grp]
    dec_hpix = dec_hpix[grp]
    x = x[grp]
    y = y[grp]
    npix = len(x)

    print('# read grid postion:', (ibx, iby), n, 'number of HEALPix pixels =', npix)
    sys.stdout.flush()

    # extract profile around each object
    x_, y_ = numpy.meshgrid(range(n),range(n))
    for ipix in range(npix):
      r = numpy.floor(numpy.sqrt((x_-x[ipix])**2 + (y_-y[ipix])**2)).astype(numpy.int16)
      for j in range(rpix):
        vals[j] = numpy.concatenate((vals[j], starmap[r==j]))

for j in range(rpix):
  outst = '{:3d} {:8d}'.format(j, numpy.size(vals[j]))
  for q in [1,5,25,50,75,95,99]: outst += ' {:12.5E}'.format(numpy.percentile(vals[j],q))
  print(outst)

# save histograms
numpy.savetxt(sys.argv[2]+'_sqrtS_hist.dat', countnoise, header=' {:11.5E} {:9.6f}'.format(numpy.amax(countnoise[:,1]), 100*tnoise_gt/tnoise))
numpy.savetxt(sys.argv[2]+'_neff_hist.dat', countneff, header=' {:11.5E} {:9.6f}'.format(numpy.amax(countneff[:,1]), 100*tneff_gt/tneff))
