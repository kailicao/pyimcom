# usage: python starcube2.py <filter> <input prefix> <outstem>
# input file name is <input prefix><filter>_DD_DD_map.fits
# this code takes the map.fits files and analyzes the star and galaxy grid layers

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

# Padding size
bd2 = 8

# Definitions
gain = 1.458  # e/DN, determined from solid-waffle for SCA 21814
t_fr = 3.08  # s/frame in flight mode, including overheads
t_exp = 139.8 #s/exp, according to https://arxiv.org/pdf/2303.08750
F_AB = 3.631*10**9 # microJy
AB_std = 3.631 * 10 ** (-20)  # erg/s/Hz/cm^2 = 1 Jy
h = 662.6  # microJy cm^2 s
m_AB = 24  # median galaxy magnitude in IR
nblock = 36
s_in = 0.11 # input pixel scale
B0 = 0.38 #e/px/s, bakground estimate

nblockmax = nblock**2
ncol = 36
nstart = 0

filter = sys.argv[1]
in1 = sys.argv[2]
outstem = sys.argv[3]

if filter=='Y':
  filtername='Y106'
  area= 7006 # cm^2
  B1 = 0.29 # minimum background in e/px/s (https://roman.gsfc.nasa.gov/science/WFI_technical.html)
  include_thermal_background = False
if filter=='J':
  filtername='J129'
  area= 7111
  B1 = 0.28
  include_thermal_background = False
if filter=='H':
  filtername='H158'
  area= 7340
  B1 = 0.3
  include_thermal_background = False
if filter=='F':
  filtername='F184'
  area= 4840
  B1 = 0.32
  include_thermal_background = False
if filter == 'W':
  filtername = 'W146'
  area = 22085
  B1 = 1.83
  include_thermal_background = False
if filter == 'K':
  filtername = 'K213'
  area = 4654
  B1 = 4.65
  include_thermal_background = True


pos = numpy.zeros((1, ncol))
# columns: ra,dec,ibx,iby,x,y,xi,yi,dx,dy, [A,xc,yc,s,g1,g2,M40-M04,M31+M13,Mxx-Myy_forced, Mxy_forced]star,gal, fid, coverage, [g1_noise, g2_noise]star,gal

#image = numpy.zeros((1, bd * 2 - 1, bd * 2 - 1))
#imageB = numpy.zeros((1, bd * 2 - 1, bd * 2 - 1))

outfile_star = outstem + 'LNStarCat_{:s}.fits'.format(filter)
outfile_gal = outstem + 'LNExtCat_{:s}.fits'.format(filter)

print('# Outfiles: ', outfile_star, outfile_gal)

fhist = numpy.zeros((81,), dtype=numpy.uint32)

for iblock in range(nstart, nstart + nblockmax ** 2):

    j = iblock
    ibx = j % nblockmax;
    iby = j // nblockmax

    infile = in1 + '{:s}_{:02d}_{:02d}_map.fits'.format(filter, ibx, iby)

    # extract information from the header of the first file
    if iblock == nstart:
        with fits.open(infile) as f:

            n = numpy.shape(f[0].data)[-1]  # size of output images

            config = ''
            for g in f['CONFIG'].data['text'].tolist(): config += g + ' '
            configStruct = json.loads(config)

            blocksize = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) * float(
                configStruct['OUTSIZE'][2]) / 3600. * numpy.pi / 180  # radians
            rs = 1.5 * blocksize / numpy.sqrt(2.)  # search radius

            s_out = float(configStruct['OUTSIZE'][2])  # in arcsec
            force_scale = .40 / s_out # in output pixels

            # padding region around the edge
            bd = int(configStruct['OUTSIZE'][1])
            bdpad = bd * int(configStruct['PAD'])

            # figure out which layer we want
            layers = [''] + configStruct['EXTRAINPUT']
            print('#', layers)
            for i in range(len(layers))[::-1]:
                m = re.match(r'^gsstar(\d+)$', layers[i])
                if m:
                    star_slice = i
                    res = int(m.group(1))
                m = re.match(r'^labnoise', layers[i])
                if m:
                    LN_slice = i
                m = re.match(r'^whitenoise10', layers[i])
                if m:
                    WN_slice = i
                m = re.match(r'^gsext(\d+)', layers[i])
                if m:
                    # match the ext obj grid w/o shear
                    if 'shear=' not in layers[i]:
                        ext_slice = i
                        
            print('# star layer', star_slice, 'extobj layer', ext_slice, 'resolution', res)
            print('# rs=', rs, 'output pix =', s_out, 'arcsec   n=', n, ' thermal bkgnd: ', include_thermal_background)
            image = numpy.zeros((1, bd * 2 - 1, bd * 2 - 1))
            imageB = numpy.zeros((1, bd * 2 - 1, bd * 2 - 1))

    if not exists(infile): continue
    with fits.open(infile) as f:
        mywcs = wcs.WCS(f[0].header)

        # Read in layers
        LN_map = f[0].data[0, LN_slice, :, :]
        star_map = f[0].data[0, star_slice, :, :]
        gal_map = f[0].data[0, ext_slice, :, :]
        WN_map = f[0].data[0, WN_slice, :, :]
        wt = numpy.sum(numpy.where(f['INWEIGHT'].data[0, :, :, :] > 0.01, 1, 0), axis=0)
        fmap = f['FIDELITY'].data[0, :, :].astype(numpy.float32) * HDU_to_bels(f['FIDELITY']) / .1  # convert to dB
        fmap = numpy.floor(fmap).astype(numpy.int16)  # and round to integer
        for fy in range(81): fhist[fy] += numpy.count_nonzero(fmap[bdpad:-bdpad, bdpad:-bdpad] == fy)

    ra_cent, dec_cent = mywcs.all_pix2world([(n - 1) / 2], [(n - 1) / 2], [0.], [0.], 0, ra_dec_order=True)
    ra_cent = ra_cent[0];
    dec_cent = dec_cent[0]
    vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
    qp = healpy.query_disc(2 ** res, vec, rs, nest=False)
    ra_hpix, dec_hpix = healpy.pix2ang(2 ** res, qp, nest=False, lonlat=True)
    npix = len(ra_hpix)
    x, y, z1, z2 = mywcs.all_world2pix(ra_hpix, dec_hpix, numpy.zeros((npix,)), numpy.zeros((npix,)), 0)
    xi = numpy.rint(x).astype(numpy.int16);
    yi = numpy.rint(y).astype(numpy.int16)
    grp = numpy.where(
        numpy.logical_and(numpy.logical_and(xi >= bd, xi < n - bd), numpy.logical_and(yi >= bd, yi < n - bd)))
    ra_hpix = ra_hpix[grp]
    dec_hpix = dec_hpix[grp]
    x = x[grp]
    y = y[grp]
    npix = len(x)

    newpos = numpy.zeros((npix, ncol))
    xi = numpy.rint(x).astype(numpy.int16)
    yi = numpy.rint(y).astype(numpy.int16)
    # position information
    newpos[:, 0] = ra_hpix
    newpos[:, 1] = dec_hpix
    newpos[:, 2] = ibx
    newpos[:, 3] = iby
    newpos[:, 4] = x
    newpos[:, 5] = y
    newpos[:, 6] = xi
    newpos[:, 7] = yi
    newpos[:, 8] = dx = x - xi
    newpos[:, 9] = dy = y - yi

    newimage_star = numpy.zeros((npix, bd * 2 - 1, bd * 2 - 1))
    newimage_gal = numpy.zeros((npix, bd * 2 - 1, bd * 2 - 1))

    print('# starting object loop for: ', iblock, infile, npix)
    for k in range(npix):
        # conversions to physical units
        if include_thermal_background:
            thisnoiseimage = (
                LN_map[yi[k] + 1 - bd : yi[k] + bd, xi[k] + 1 - bd : xi[k] + bd] /
                ((t_fr * area * s_in**2) / (gain * h)) +
                WN_map[yi[k] + 1 - bd : yi[k] + bd, xi[k] + 1 - bd : xi[k] + bd] *
                np.sqrt((B1 - B0) / t_exp)
            )  # units microJy/arcsec^2
        else:
            thisnoiseimage = (
                LN_map[yi[k] + 1 - bd:yi[k] + bd, xi[k] + 1 - bd:xi[k] + bd] /
                ((t_fr * area * s_in**2 ) / (gain * h) )
            ) # units microJy/arcsec^2
                        
        newimage_star[k, :, :] = star_map[yi[k] + 1 - bd:yi[k] + bd, xi[k] + 1 - bd:xi[k] + bd] * (
                        F_AB * 10**(-0.4*m_AB) / (s_in**2) ) # microJy/arcsec^2
        newimage_gal[k, :, :] = gal_map[yi[k] + 1 - bd:yi[k] + bd, xi[k] + 1 - bd:xi[k] + bd] * (
                        F_AB * 10**(-0.4*m_AB) / (s_in**2) ) #microJy/arcsec^2
                        
        try:

            # PSF shape + moments : STAR
            moms = galsim.Image(newimage_star[k, :, :]).FindAdaptiveMom()
            newpos[k, 10] = moms.moments_amp
            newpos[k, 11] = moms.moments_centroid.x - bd - dx[k]
            newpos[k, 12] = moms.moments_centroid.y - bd - dy[k]
            newpos[k, 13] = moms.moments_sigma
            newpos[k, 14] = moms.observed_shape.g1
            newpos[k, 15] = moms.observed_shape.g2
            
            # Higher moments FOR STARS
            x_, y_ = numpy.meshgrid(numpy.array(range(1, bd * 2)) - moms.moments_centroid.x,
                                    numpy.array(range(1, bd * 2)) - moms.moments_centroid.y)
            e1 = moms.observed_shape.e1
            e2 = moms.observed_shape.e2
            Mxx = moms.moments_sigma ** 2 * (1 + e1) / numpy.sqrt(1 - e1 ** 2 - e2 ** 2)
            Myy = moms.moments_sigma ** 2 * (1 - e1) / numpy.sqrt(1 - e1 ** 2 - e2 ** 2)
            Mxy = moms.moments_sigma ** 2 * e2 / numpy.sqrt(1 - e1 ** 2 - e2 ** 2)
            D = Mxx * Myy - Mxy ** 2
            zeta = D * (Mxx + Myy + 2 * numpy.sqrt(D))
            u_ = ((Myy + numpy.sqrt(D)) * x_ - Mxy * y_) / zeta ** 0.5
            v_ = ((Mxx + numpy.sqrt(D)) * y_ - Mxy * x_) / zeta ** 0.5
            wti = newimage_star[k, :, :] * numpy.exp(-0.5 * (u_ ** 2 + v_ ** 2))
            newpos[k, 16] = numpy.sum(wti * (u_ ** 4 - v_ ** 4)) / numpy.sum(wti)
            newpos[k, 17] = 2 * numpy.sum(wti * (u_ ** 3 * v_ + u_ * v_ ** 3)) / numpy.sum(wti)

            # moments with forced scale length
            wti2 = newimage_star[k, :, :] * numpy.exp(-0.5 * (x_ ** 2 + y_ ** 2) / force_scale ** 2)
            newpos[k, 18] = numpy.sum(wti2 * (x_ ** 2 - y_ ** 2)) / numpy.sum(wti2) / force_scale ** 2
            newpos[k, 19] = numpy.sum(wti2 * (2 * x_ * y_)) / numpy.sum(wti2) / force_scale ** 2
            
            # PSF shape + moments : EXTENDED SOURCES
            moms = galsim.Image(newimage_gal[k, :, :]).FindAdaptiveMom()
            newpos[k, 20] = moms.moments_amp
            newpos[k, 21] = moms.moments_centroid.x - bd - dx[k]
            newpos[k, 22] = moms.moments_centroid.y - bd - dy[k]
            newpos[k, 23] = moms.moments_sigma
            newpos[k, 24] = moms.observed_shape.g1
            newpos[k, 25] = moms.observed_shape.g2
            
            # Higher moments EXTENDED SOURCES
            x_, y_ = numpy.meshgrid(numpy.array(range(1, bd * 2)) - moms.moments_centroid.x,
                                    numpy.array(range(1, bd * 2)) - moms.moments_centroid.y)
            e1 = moms.observed_shape.e1
            e2 = moms.observed_shape.e2
            Mxx = moms.moments_sigma ** 2 * (1 + e1) / numpy.sqrt(1 - e1 ** 2 - e2 ** 2)
            Myy = moms.moments_sigma ** 2 * (1 - e1) / numpy.sqrt(1 - e1 ** 2 - e2 ** 2)
            Mxy = moms.moments_sigma ** 2 * e2 / numpy.sqrt(1 - e1 ** 2 - e2 ** 2)
            D = Mxx * Myy - Mxy ** 2
            zeta = D * (Mxx + Myy + 2 * numpy.sqrt(D))
            u_ = ((Myy + numpy.sqrt(D)) * x_ - Mxy * y_) / zeta ** 0.5
            v_ = ((Mxx + numpy.sqrt(D)) * y_ - Mxy * x_) / zeta ** 0.5
            wti = newimage_star[k, :, :] * numpy.exp(-0.5 * (u_ ** 2 + v_ ** 2))
            newpos[k, 26] = numpy.sum(wti * (u_ ** 4 - v_ ** 4)) / numpy.sum(wti)
            newpos[k, 27] = 2 * numpy.sum(wti * (u_ ** 3 * v_ + u_ * v_ ** 3)) / numpy.sum(wti)

            # moments with forced scale length
            wti2 = newimage_gal[k, :, :] * numpy.exp(-0.5 * (x_ ** 2 + y_ ** 2) / force_scale ** 2)
            newpos[k, 28] = numpy.sum(wti2 * (x_ ** 2 - y_ ** 2)) / numpy.sum(wti2) / force_scale ** 2
            newpos[k, 29] = numpy.sum(wti2 * (2 * x_ * y_)) / numpy.sum(wti2) / force_scale ** 2
            
            # fidelity
            newpos[k, 30] = numpy.mean(fmap[yi[k] + 1 - bd2:yi[k] + bd2, xi[k] + 1 - bd2:xi[k] + bd2])

            # coverage
            newpos[k, 31] = wt[yi[k] // bd, xi[k] // bd]

            # NOISE BIAS MEASUREMENTS: Stars, then Extended Sources
            try:
                sc = 1.
                # Star
                moms_noise_zero = galsim.Image(newimage_star[k, :, :]).FindAdaptiveMom()
                moms_noise_positive = galsim.Image(newimage_star[k, :, :] + thisnoiseimage).FindAdaptiveMom()
                moms_noise_negative = galsim.Image(newimage_star[k, :, :] - thisnoiseimage).FindAdaptiveMom()
                newpos[k, 32] = (
                                            moms_noise_positive.observed_shape.g1 + moms_noise_negative.observed_shape.g1) / 2. - moms_noise_zero.observed_shape.g1
                newpos[k, 33] = (
                                            moms_noise_positive.observed_shape.g2 + moms_noise_negative.observed_shape.g2) / 2. - moms_noise_zero.observed_shape.g2
               
                # Galaxy
                moms_noise_zero = galsim.Image(newimage_gal[k, :, :]).FindAdaptiveMom()
                moms_noise_positive = galsim.Image(newimage_gal[k, :, :] + thisnoiseimage).FindAdaptiveMom()
                moms_noise_negative = galsim.Image(newimage_gal[k, :, :] - thisnoiseimage).FindAdaptiveMom()
                newpos[k, 34] = (
                                            moms_noise_positive.observed_shape.g1 + moms_noise_negative.observed_shape.g1) / 2. - moms_noise_zero.observed_shape.g1
                newpos[k, 35] = (
                                            moms_noise_positive.observed_shape.g2 + moms_noise_negative.observed_shape.g2) / 2. - moms_noise_zero.observed_shape.g2
                
            except:
                try:
                    print('BACKUP-0.5    {:d},{:d}  coverage={:2d}'.format(iblock, k, int(newpos[k, 23])))
                    sc = numpy.sqrt(10.)
                    # Star
                    moms_noise_zero = galsim.Image(newimage_star[k, :, :]).FindAdaptiveMom()
                    moms_noise_positive = galsim.Image(newimage_star[k, :, :] + thisnoiseimage / sc).FindAdaptiveMom()
                    moms_noise_negative = galsim.Image(newimage_star[k, :, :] - thisnoiseimage / sc).FindAdaptiveMom()
                    newpos[k, 32] = (
                                                moms_noise_positive.observed_shape.g1 + moms_noise_negative.observed_shape.g1) / 2. - moms_noise_zero.observed_shape.g1
                    newpos[k, 33] = (
                                                moms_noise_positive.observed_shape.g2 + moms_noise_negative.observed_shape.g2) / 2. - moms_noise_zero.observed_shape.g2
                    newpos[k, 32:34] *= sc ** 2
                    
                    # Galaxy
                    moms_noise_zero = galsim.Image(newimage_gal[k, :, :]).FindAdaptiveMom()
                    moms_noise_positive = galsim.Image(newimage_gal[k, :, :] + thisnoiseimage / sc).FindAdaptiveMom()
                    moms_noise_negative = galsim.Image(newimage_gal[k, :, :] - thisnoiseimage / sc).FindAdaptiveMom()
                    newpos[k, 34] = (
                                                moms_noise_positive.observed_shape.g1 + moms_noise_negative.observed_shape.g1) / 2. - moms_noise_zero.observed_shape.g1
                    newpos[k, 35] = (
                                                moms_noise_positive.observed_shape.g2 + moms_noise_negative.observed_shape.g2) / 2. - moms_noise_zero.observed_shape.g2
                    newpos[k, 34:36] *= sc ** 2
                    
                    
                except:
                    try:
                        print('BACKUP-1.0    {:d},{:d}  coverage={:2d}'.format(iblock, k, int(newpos[k, 23])))
                        sc = 10.
                        moms_noise_zero = galsim.Image(newimage_star[k, :, :]).FindAdaptiveMom()
                        moms_noise_positive = galsim.Image(newimage_star[k, :, :] + thisnoiseimage / sc).FindAdaptiveMom()
                        moms_noise_negative = galsim.Image(newimage_star[k, :, :] - thisnoiseimage / sc).FindAdaptiveMom()
                        newpos[k, 32] = (
                                                    moms_noise_positive.observed_shape.g1 + moms_noise_negative.observed_shape.g1) / 2. - moms_noise_zero.observed_shape.g1
                        newpos[k, 33] = (
                                                    moms_noise_positive.observed_shape.g2 + moms_noise_negative.observed_shape.g2) / 2. - moms_noise_zero.observed_shape.g2
                        newpos[k, 32:34] *= sc ** 2
                        
                        # Galaxy
                        moms_noise_zero = galsim.Image(newimage_gal[k, :, :]).FindAdaptiveMom()
                        moms_noise_positive = galsim.Image(newimage_gal[k, :, :] + thisnoiseimage / sc).FindAdaptiveMom()
                        moms_noise_negative = galsim.Image(newimage_gal[k, :, :] - thisnoiseimage / sc).FindAdaptiveMom()
                        newpos[k, 34] = (
                                                    moms_noise_positive.observed_shape.g1 + moms_noise_negative.observed_shape.g1) / 2. - moms_noise_zero.observed_shape.g1
                        newpos[k, 35] = (
                                                    moms_noise_positive.observed_shape.g2 + moms_noise_negative.observed_shape.g2) / 2. - moms_noise_zero.observed_shape.g2
                        newpos[k, 34:36] *= sc ** 2
                        
                    except:
                        print('ERROR {:d},{:d}  coverage={:2d}'.format(iblock, k, int(newpos[k, 31])))
                        pass
        except Exception as e:
            print('exception:', e, ', object number: ', k)
            continue
        # end galaxy loop

    pos = numpy.concatenate((pos, newpos), axis=0)
    image = numpy.concatenate((image, newimage_star), axis=0)
    imageB = numpy.concatenate((image, newimage_gal), axis=0)

pos = pos[1:, :]
image = image[1:, :, :]
imageB = image[1:, :, :]

fits.HDUList([fits.PrimaryHDU(image.astype(numpy.float32))]).writeto(outfile_star, overwrite=True)
fits.HDUList([fits.PrimaryHDU(imageB.astype(numpy.float32))]).writeto(outfile_gal, overwrite=True)

numpy.savetxt(outstem + 'LNAllCat_{:s}.txt'.format(filter), pos,
              header=' {:14.8E} {:14.8E}'.format(numpy.median(newpos[:, 13]), numpy.median(newpos[:, 23])) )
#numpy.savetxt(outstem + 'LNGalCat_{:s}.txt'.format(filter), pos,
#              header=' {:14.8E}'.format(numpy.median(newpos[:, 23])))

              
print(pos[:, -1])
