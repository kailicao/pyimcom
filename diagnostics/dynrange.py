# code to construct dynamic range estimates from block files
#

# usage:
# python dyrange.py <pathstem>

import sys
import numpy
import healpy
from astropy.io import fits
from astropy import wcs
from os.path import exists
import json
import re

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

# now loop over the blocks
for iby in range(nblockmax):
  for ibx in range(nblockmax):
    infile = pathstem + '_{:02d}_{:02d}_map.fits'.format(ibx,iby)
    if not exists(infile): continue

    # if this is the first block we find, get the configuration file
    if is_first:
      is_first = False
      with fits.open(infile) as f:
        config = ''
        for g in f['CONFIG'].data['text'].tolist(): config += g+' '
        configStruct = json.loads(config)

        blocksize = int(configStruct['OUTSIZE'][1]) * float(configStruct['OUTSIZE'][2]) / 3600. *numpy.pi/180 # radians
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

    # now we know this file exists
    with fits.open(infile) as f:
      n = numpy.shape(f[0].data)[-1]
      mywcs = wcs.WCS(f[0].header)
      starmap = f[0].data[0,framenumber,:,:]

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
