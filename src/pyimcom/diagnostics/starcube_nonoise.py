"""
Code to generate a catalog of injected stars and their properties in the coadded image.

Functions
---------
gen_starcube_nonoise
    Extracts the noiseless star cube and the moments.

"""

import sys
import numpy as np
import healpy
from astropy.io import fits
from astropy import wcs
from os.path import exists
import galsim
import json
import re

from .outimage_utils.helper import HDU_to_bels
from ..config import Config

def gen_starcube_nonoise(infile_fcn, outstem, nblockmax=100):
    """
    Extracts the noiseless star cube and the moments.

    Arguments
    ---------
    infile_fcn : function
        A function that returns the input file path given block (ix,iy).
    outstem : str
        Output file stem.
    nblockmax : int, default=100
        Maximum number of blocks on each axis of a mosaic.

    Returns
    -------
    output : dict
        The return parameters dictionary contains two strings:
        key 'STARCAT' points to the file name for the star catalog, and
        key 'FIDHIST' points to the file name for the fidelity histogram.

    """

    output = {
        'STARCAT': None
    }

    bd = 40 # padding size
    bd2 = 8 # fidelity extraction size

    # if needed, shrink the padding size
    try:
        configStruct = Config(infile_fcn(0,0), inmode='block')
        n2_ = configStruct.n2
        print('# n2 =', n2_)
        if n2_<bd: bd=n2_
    except:
        pass

    ncol = 22 # number of columns in star catalog

    pos = np.zeros((1,ncol)) # prototype object (will be stripped at the end)
    image = np.zeros((1,bd*2-1,bd*2-1), dtype=np.float32)

    outfile_g = outstem + '_StarCat_galsim.fits'
    fhist = np.zeros((81,),dtype=np.uint32)

    is_first = True
    for ibx in range(nblockmax):
        for iby in range(nblockmax):
            try:
                infile = infile_fcn(ibx,iby)
            except:
                continue

            # extract information from the header of the first file
            if is_first:
                with fits.open(infile) as f:

                    n = np.shape(f[0].data)[-1] # size of output images

                    config = ''
                    for g in f['CONFIG'].data['text'].tolist(): config += g+' '
                    configStruct = json.loads(config)

                    blocksize = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) * float(configStruct['OUTSIZE'][2]) / 3600. *np.pi/180 # radians
                    rs = 1.5*blocksize/np.sqrt(2.) # search radius
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

                is_first = False # don't do this again

            if not exists(infile): continue
            # get WCS
            with fits.open(infile) as f:
                mywcs = wcs.WCS(f[0].header)
                map = f[0].data[0,use_slice,:,:]
                wt = np.sum(np.where(f['INWEIGHT'].data[0,:,:,:]>0.01, 1, 0), axis=0)
                fmap = f['FIDELITY'].data[0,:,:].astype(np.float32) * HDU_to_bels(f['FIDELITY'])/.1 # convert to dB
                fmap = np.floor(fmap).astype(np.int16) # and round to integer
                for fy in range(81): fhist[fy] += np.count_nonzero(fmap[bdpad:-bdpad,bdpad:-bdpad]==fy)

            print(infile, use_slice, res); sys.stdout.flush()

            # extract HEALPix pixels with the stars
            ra_cent, dec_cent = mywcs.all_pix2world([(n-1)/2], [(n-1)/2], [0.], [0.], 0, ra_dec_order=True)
            ra_cent = ra_cent[0]; dec_cent = dec_cent[0]
            vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
            qp = healpy.query_disc(2**res, vec, rs, nest=False)
            ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=False, lonlat=True)
            npix = len(ra_hpix)
            x, y, z1, z2 = mywcs.all_world2pix(ra_hpix, dec_hpix, np.zeros((npix,)), np.zeros((npix,)), 0)
            xi = np.rint(x).astype(np.int16); yi = np.rint(y).astype(np.int16)
            grp = np.where(np.logical_and(np.logical_and(xi>=bdpad,xi<n-bdpad),np.logical_and(yi>=bdpad,yi<n-bdpad)))
            ra_hpix = ra_hpix[grp]
            dec_hpix = dec_hpix[grp]
            x = x[grp]
            y = y[grp]
            npix = len(x)

            newpos = np.zeros((npix,ncol))
            xi = np.rint(x).astype(np.int16)
            yi = np.rint(y).astype(np.int16)
            # position information
            dx = x-xi; dy = y-yi
            newpos[:,0] = ra_hpix
            newpos[:,1] = dec_hpix
            newpos[:,2] = ibx
            newpos[:,3] = iby
            newpos[:,4] = x
            newpos[:,5] = y
            newpos[:,6] = xi
            newpos[:,7] = yi
            newpos[:,8] = dx
            newpos[:,9] = dy

            newimage = np.zeros((npix,bd*2-1,bd*2-1))
            print(ibx, iby, infile, npix)
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
                x_,y_ = np.meshgrid(np.array(range(1,bd*2)) - moms.moments_centroid.x, np.array(range(1,bd*2)) - moms.moments_centroid.y)
                e1 = moms.observed_shape.e1
                e2 = moms.observed_shape.e2
                Mxx = moms.moments_sigma**2 * (1+e1) / np.sqrt(1-e1**2-e2**2)
                Myy = moms.moments_sigma**2 * (1-e1) / np.sqrt(1-e1**2-e2**2)
                Mxy = moms.moments_sigma**2 * e2 / np.sqrt(1-e1**2-e2**2)
                D = Mxx*Myy-Mxy**2
                zeta = D*(Mxx+Myy+2*np.sqrt(D))
                u_ = ( (Myy+np.sqrt(D))*x_ - Mxy*y_ )/zeta**0.5
                v_ = ( (Mxx+np.sqrt(D))*y_ - Mxy*x_ )/zeta**0.5
                wti = newimage[k,:,:] * np.exp(-0.5*(u_**2+v_**2))
                newpos[k,16] = np.sum(wti*(u_**4-v_**4))/np.sum(wti)
                newpos[k,17] = 2*np.sum(wti*(u_**3*v_+u_*v_**3))/np.sum(wti)

                # moments with forced scale length
                wti2 = newimage[k,:,:] * np.exp(-0.5*(x_**2+y_**2)/force_scale**2)
                newpos[k,18] = np.sum(wti2*(x_**2-y_**2))/np.sum(wti2)/force_scale**2
                newpos[k,19] = np.sum(wti2*(2*x_*y_))/np.sum(wti2)/force_scale**2

                # fidelity
                newpos[k,20] = np.mean(fmap[yi[k]+1-bd2:yi[k]+bd2,xi[k]+1-bd2:xi[k]+bd2])

                # coverage
                newpos[k,21] = wt[yi[k]//n2,xi[k]//n2]

                # flush
                sys.stdout.flush()
                # end loop over simulated stars

            pos = np.concatenate((pos, newpos), axis=0)
            image = np.concatenate((image, newimage.astype(np.float32)), axis=0)
            # end block loop

    # strip fictitious first object
    pos = pos[1:,:]
    image = image[1:,:,:]

    fits.HDUList([fits.PrimaryHDU(image)]).writeto(outfile_g, overwrite=True)

    ofile = outstem + '_StarCat.txt'
    np.savetxt(ofile, pos, header = ' {:14.8E}'.format(np.median(newpos[:,13])))
    output['STARCAT'] = ofile

    # fidelity histogram
    ofile = outstem + '_fidHist.txt'
    with open(ofile, "w") as f:
        for fy in range(20,81):
            f.write('{:2d} {:8.6f} {:8.6f}'.format(fy, fhist[fy]/np.sum(fhist), np.sum(fhist[:fy+1])/np.sum(fhist)))
    output['FIDHIST'] = ofile

    return output
