# usage: python starcube_nonoise.py <filter> <input prefix> <outstem>
# input file name is <input prefix><filter>_DD_DD_map.fits

import sys
import numpy
import healpy
from astropy.io import fits
from astropy import wcs
from os.path import exists
import galsim
import json
import re
from outimage_utils.helper import HDU_to_bels

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

outfile_g = outstem + '_StarCat_galsim_{:s}.fits'.format(filter)

fhist = numpy.zeros((81,),dtype=numpy.uint32)

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
      print('#', layers)
      for i in range(len(layers))[::-1]:
        m = re.match(r'^gsstar(\d+)$', layers[i])
        if m:
          use_slice = i
          res = int(m.group(1))
      print('# using layer', use_slice, 'resolution', res, 'output pix =', outscale, 'arcsec   n=',n)
      print('# rs=', rs)

  if not exists(infile): continue
  with fits.open(infile) as f:
    mywcs = wcs.WCS(f[0].header)
    map = f[0].data[0,use_slice,:,:]
    wt = numpy.sum(numpy.where(f['INWEIGHT'].data[0,:,:,:]>0.01, 1, 0), axis=0)
    fmap = f['FIDELITY'].data[0,:,:].astype(numpy.float32) * HDU_to_bels(f['FIDELITY'])/.1 # convert to dB
    fmap = numpy.floor(fmap).astype(numpy.int16) # and round to integer
    for fy in range(81): fhist[fy] += numpy.count_nonzero(fmap[bdpad:-bdpad,bdpad:-bdpad]==fy)

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
  # position information
  newpos[:,0] = ra_hpix
  newpos[:,1] = dec_hpix
  newpos[:,2] = ibx
  newpos[:,3] = iby
  newpos[:,4] = x
  newpos[:,5] = y
  newpos[:,6] = xi
  newpos[:,7] = yi
  newpos[:,8] = dx = x-xi
  newpos[:,9] = dy = y-yi

  newimage = numpy.zeros((npix,bd*2-1,bd*2-1))
  print(iblock, infile, npix)
  for k in range(npix):
    newimage[k,:,:] = map[yi[k]+1-bd:yi[k]+bd,xi[k]+1-bd:xi[k]+bd]

    # PSF shape
    try:
      moms = galsim.Image(newimage[k,:,:]).FindAdaptiveMom()
    except:
      continue

    newpos[k,10] = moms.moments_amp
    newpos[k,11] = moms.moments_centroid.x-bd-dx[k]
    newpos[k,12] = moms.moments_centroid.y-bd-dy[k]
    newpos[k,13] = moms.moments_sigma
    newpos[k,14] = moms.observed_shape.g1
    newpos[k,15] = moms.observed_shape.g2

    # higher moments
    x_,y_ = numpy.meshgrid(numpy.array(range(1,bd*2)) - moms.moments_centroid.x, numpy.array(range(1,bd*2)) - moms.moments_centroid.y)
    e1 = moms.observed_shape.e1
    e2 = moms.observed_shape.e2
    Mxx = moms.moments_sigma**2 * (1+e1) / numpy.sqrt(1-e1**2-e2**2)
    Myy = moms.moments_sigma**2 * (1-e1) / numpy.sqrt(1-e1**2-e2**2)
    Mxy = moms.moments_sigma**2 * e2 / numpy.sqrt(1-e1**2-e2**2)
    D = Mxx*Myy-Mxy**2
    zeta = D*(Mxx+Myy+2*numpy.sqrt(D))
    u_ = ( (Myy+numpy.sqrt(D))*x_ - Mxy*y_ )/zeta**0.5
    v_ = ( (Mxx+numpy.sqrt(D))*y_ - Mxy*x_ )/zeta**0.5
    wti = newimage[k,:,:] * numpy.exp(-0.5*(u_**2+v_**2))
    newpos[k,16] = numpy.sum(wti*(u_**4-v_**4))/numpy.sum(wti)
    newpos[k,17] = 2*numpy.sum(wti*(u_**3*v_+u_*v_**3))/numpy.sum(wti)

    # moments with forced scale length
    wti2 = newimage[k,:,:] * numpy.exp(-0.5*(x_**2+y_**2)/force_scale**2)
    newpos[k,18] = numpy.sum(wti2*(x_**2-y_**2))/numpy.sum(wti2)/force_scale**2
    newpos[k,19] = numpy.sum(wti2*(2*x_*y_))/numpy.sum(wti2)/force_scale**2

    # fidelity
    newpos[k,20] = numpy.mean(fmap[yi[k]+1-bd2:yi[k]+bd2,xi[k]+1-bd2:xi[k]+bd2])

    # coverage
    newpos[k,21] = wt[yi[k]//n2,xi[k]//n2]

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
