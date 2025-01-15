# usage: python truthcats.py <filter> <input prefix> <outstem>
# input file name is <input prefix><filter>_DD_DD_map.fits (block files)

import sys
import numpy
import healpy
from astropy.io import fits
from astropy import wcs
from astropy.table import Table, Column
from os.path import exists
import galsim
import json
import re
from .outimage_utils.helper import HDU_to_bels

bd = 40 # padding size
bd2 = 8

nblockmax = 100 # maximum
ncol = 22
nstart = 0

filter = sys.argv[1]

if filter=='Y': filtername='Y106'
if filter=='J': filtername='J129'
if filter=='H': filtername='H158'
if filter=='F': filtername='F184'

pos = numpy.zeros((1,ncol))
image = numpy.zeros((1,bd*2-1,bd*2-1))

# prefix and suffix
in1 = sys.argv[2]
outstem = sys.argv[3]

outfile_g = outstem + '_TruthCat_{:s}.fits'.format(filter)

for iblock in range(nstart,nblockmax**2):

  j = iblock
  ibx = j%nblockmax; iby = j//nblockmax

  infile = in1 + '{:s}_{:02d}_{:02d}_map.fits'.format(filter,ibx,iby)

  # extract information from the header of the first file
  if iblock==nstart:
    with fits.open(infile) as f:

      n = numpy.shape(f[0].data)[-1] # size of output images

      config = ''
      for g in f['CONFIG'].data['text'].tolist(): config += g+' '
      configStruct = json.loads(config)

      blocksize = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) * float(configStruct['OUTSIZE'][2]) / 3600. *numpy.pi/180 # radians
      rs = 1.5*blocksize/numpy.sqrt(2.) # search radius
      n2 = int(configStruct['OUTSIZE'][1])  # will be used for coverage

      outscale = float(configStruct['OUTSIZE'][2]) # in arcsec
      force_scale = .40/outscale # in output pixels

      # padding region around the edge
      bdpad = int(configStruct['OUTSIZE'][1]) * int(configStruct['PAD'])

      # figure out which layer we want
      layers = [''] + configStruct['EXTRAINPUT']
      use_layers={}
      print('# All EXTRAINPUT layers:', layers)
      for i in range(len(layers))[::-1]:
        m = re.match(r'^gs\S*$', layers[i])
        if m:
          use_layers[str(m.group(0))]=i  # KL: later note: use re.split to separate at commas
        m = re.match(r'^ns\S*$', layers[i])
        if m:
          use_layers[str(m.group(0))]=i

  if not exists(infile): continue
  with fits.open(infile) as f:
    mywcs = wcs.WCS(f[0].header)
    wt = numpy.sum(numpy.where(f['INWEIGHT'].data[0,:,:,:]>0.01, 1, 0), axis=0)

  resolutions=[]
  for layer in use_layers.keys():
    params = re.split(r',',layer)
    m = re.search(r'(\D*)(\d*)',params[0])
    if m:
      res=m.group(2)

    if res not in resolutions:

      ra_cent, dec_cent = mywcs.all_pix2world([(n-1)/2], [(n-1)/2], [0.], [0.], 0, ra_dec_order=True)
      ra_cent = ra_cent[0]; dec_cent = dec_cent[0]
      vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
      qp = healpy.query_disc(2**res, vec, rs, nest=False)
      ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=False, lonlat=True)
      npix = len(ra_hpix)
      x, y, z1, z2 = mywcs.all_world2pix(ra_hpix, dec_hpix, numpy.zeros((npix,)), numpy.zeros((npix,)), 0)
      xi = numpy.rint(x).astype(numpy.int16); yi = numpy.rint(y).astype(numpy.int16)
      grp = numpy.where(numpy.logical_and(numpy.logical_and(xi>=bdpad,xi<n-bdpad),numpy.logical_and(yi>=bdpad,yi<n-bdpad)))
      ra_hpix = ra_hpix[grp]
      dec_hpix = dec_hpix[grp]
      x = x[grp]
      y = y[grp]
      npix = len(x)

      newpos = numpy.zeros((npix,ncol))
      xi = numpy.rint(x).astype(numpy.int16)
      yi = numpy.rint(y).astype(numpy.int16)

      # Initiate table
      t = Table()
      t['Block'] = [r'{:02d}_{:02d}'.format(ibx,iby)] * npix
      t['Layer'] = [layer] * npix
      # Position information
      t['ra_hpix'] = ra_hpix
      t['dec_hpix'] = dec_hpix
      t['ibx'] = ibx
      t['iby'] = iby
      t['x'] = x
      t['y'] = y
      t['xi'] = xi
      t['yi'] = yi
      t['dx=x-xi'] = dx = x-xi
      t['dy=y-yi'] = dy = y-yi

      resolutions.append(res)

    else:
      # if res is in resolutions i need to figure out how to get the correct table?
      # KL work on this bit later, for now keep going as though things are defined

    # parse params for arguments
    # also get defaults for when there arent

    # if layer is a gs:
      # if layer is a galaxy:
        # truthcat = GalSimInject.genobj(12*4**res, ipix, 'exp1', seed)
      # if layer is a star:
        # truthcat = GalSimInject ???
    # if layer is a ns:
        # truthcat = croutine thing

  for k in range(npix):


    # flush
    sys.stdout.flush()
    # end galaxy loop

  pos = numpy.concatenate((pos, newpos), axis=0)
  image = numpy.concatenate((image, newimage), axis=0)

pos = pos[1:,:]
image = image[1:,:,:]

fits.HDUList([fits.PrimaryHDU(image.astype(numpy.float32))]).writeto(outfile_g, overwrite=True)

numpy.savetxt(outstem + '_StarCat_{:s}.txt'.format(filter), pos,
  header = ' {:14.8E}'.format(numpy.median(newpos[:,13])))

for fy in range(20,81):
  print('{:2d} {:8.6f} {:8.6f}'.format(fy, fhist[fy]/numpy.sum(fhist), numpy.sum(fhist[:fy+1])/numpy.sum(fhist)))
