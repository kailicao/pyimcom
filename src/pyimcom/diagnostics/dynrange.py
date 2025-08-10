# code to construct dynamic range estimates from block files
#

nscale=1
# nscale=10 # this line is for bug compensation --- will remove it later

import warnings
import sys
import numpy as np
import healpy
from astropy.io import fits
from astropy import wcs
from os.path import exists
import json
import re
from .outimage_utils.helper import HDU_to_bels

def gen_dynrange_data(inpath, outstem, rpix_try=50, nblockmax=100):
    """Takes files from inpath and writes histograms to outstem.
    inpath should be a function that takes ix and iy and returns a file name

    Optional input parameters:
    rpix_try : int = radius over which to compute profiles
        (needs to be an integer less than the padding, will truncate)
    nblockmax : maximum block number to consider (default 100, only need to
        reduce this if you want to make a report generate faster for testing)

    The output files are:

    Histograms:
        outstem +'_sqrtS_hist.dat': histogram of noise amplification factor sqrtS
        outstem +'_neff_hist.dat': histogram of effective exposure number
    Both of these have a header that indicates the fraction of data that is off scale high.

    dynamic range file:
        outstem +'_dynrange.dat': table of percentiles of noisy star images
    (Columns are radius and [1,5,25,50,75,95,99] percentiles)

    Returns a dictionary of which files were successfully generated:
    output[key] = filename (if successful), None (if not successful)
    Current file keys: 'SQRTS', 'NEFF', 'DYNRANGE'
    Header information is in keys 'SQRTS_HEADER' and 'NEFF_HEADER'
    Number of blocks read is in 'COUNTBLOCK'

    """

    # initialization of output
    output = {
        'SQRTS': None,
        'NEFF': None,
        'DYNRANGE': None,
        'COUNTBLOCK': 0
    }

    # initialize table of pixel values
    vals = []
    for j in range(rpix_try):
      vals += [np.zeros((0,))]

    is_first = True

    # histogram initialization
    N_noise = 100
    d_noise = .02
    countnoise = np.zeros((N_noise,2))
    countnoise[:,0] = d_noise*np.linspace(.5,N_noise-.5,N_noise)
    tnoise = 0.
    tnoise_gt = 0.
    N_neff = 100
    d_neff = .1
    countneff = np.zeros((N_neff,2))
    countneff[:,0] = d_neff*np.linspace(.5,N_neff-.5,N_neff)
    tneff = 0.
    tneff_gt = 0.

    # now loop over the blocks
    for iby in range(nblockmax):
        for ibx in range(nblockmax):
            # get the input file (if it exists -- otherwise keep going)
            try:
                infile = inpath(ibx,iby)
                if not exists(infile): continue
            except:
                continue

            # if this is the first block we find, get the configuration file
            if is_first:
                is_first = False
                config = ''
                with fits.open(infile) as f:
                    for g in f['CONFIG'].data['text'].tolist(): config += g+' '
                configStruct = json.loads(config)

                blocksize = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) * float(configStruct['OUTSIZE'][2]) / 3600. *np.pi/180 # radians
                rs = 1.5*blocksize/np.sqrt(2.) # search radius

                # padding region around the edge
                bd = int(configStruct['OUTSIZE'][1]) * int(configStruct['PAD'])
                rpix = min(rpix_try,bd-1)

                # figure out which layer we want
                layers = [''] + configStruct['EXTRAINPUT']
                framenumber = 0
                res = 9
                nstarlayer = {}
                for i in range(len(layers))[::-1]:
                    m = re.match(r'^nstar(\d+),([^,]+),([^,]+),([^,]+)$', layers[i])
                    if m:
                        framenumber = i
                        res = int(m.group(1))
                        nstarlayer = {
                            'RESOLUTION': res,
                            'FLUX': float(m.group(2)),
                            'BACKGROUND': float(m.group(3)),
                            'SEED': int(m.group(4))
                        }
                print('# using layer', framenumber, 'resolution', res)
                print('# rs=', rs)

            # now we know this file exists
            with fits.open(infile) as f:
                n = np.shape(f[0].data)[-1]
                mywcs = wcs.WCS(f[0].header)
                starmap = f[0].data[0,framenumber,:,:]

                # now extract histogram information
                try:
                    sigma_ = 10**(-.5*HDU_to_bels(f['SIGMA'])*f['SIGMA'].data[0,bd:-bd,bd:-bd]) # noise standard deviation in units of input noise
                    for j in range(N_noise):
                        countnoise[j,1] = countnoise[j,1] + np.count_nonzero(np.logical_and(sigma_/d_noise>=j, sigma_/d_noise<j+1))
                    tnoise = tnoise + np.size(sigma_)
                    tnoise_gt = tnoise_gt + np.count_nonzero(sigma_>=d_noise*N_noise)
                except:
                    warnings.warn('No valid noise frame: '+infile)

                try:
                    neff_ = 10**(HDU_to_bels(f['EFFCOVER'])*f['EFFCOVER'].data[0,bd:-bd,bd:-bd]*nscale) # effective coverage
                    for j in range(N_neff):
                        countneff[j,1] = countneff[j,1] + np.count_nonzero(np.logical_and(neff_/d_neff>=j, neff_/d_neff<j+1))
                    tneff = tneff + np.size(neff_)
                    tneff_gt = tneff_gt + np.count_nonzero(neff_>=d_neff*N_neff)
                except:
                    warnings.warn('No valid coverage frame: '+infile)

                # identify which HEALpix positions we have
                ra_cent, dec_cent = mywcs.all_pix2world([(n-1)/2], [(n-1)/2], [0.], [0.], 0, ra_dec_order=True)
                ra_cent = ra_cent[0]; dec_cent = dec_cent[0]
                vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
                qp = healpy.query_disc(2**res, vec, rs, nest=False)
                ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=False, lonlat=True)
                npix = len(ra_hpix)
                x, y, z1, z2 = mywcs.all_world2pix(ra_hpix, dec_hpix, np.zeros((npix,)), np.zeros((npix,)), 0)
                xi = np.rint(x).astype(np.int16); yi = np.rint(y).astype(np.int16)
                grp = np.where(np.logical_and(np.logical_and(xi>=bd,xi<n-bd),np.logical_and(yi>=bd,yi<n-bd)))
                ra_hpix = ra_hpix[grp]
                dec_hpix = dec_hpix[grp]
                x = x[grp]; xi=xi[grp]
                y = y[grp]; yi=yi[grp]
                npix = len(x)

                print('# read grid postion:', (ibx, iby), n, 'number of HEALPix pixels =', npix)
                # complain if it didn't find a star
                for ipix in range(npix):
                    if starmap[yi[ipix],xi[ipix]]<1000:
                        print('block', ibx,iby, 'star', ipix, 'pos', xi[ipix],yi[ipix], 'val', starmap[yi[ipix],xi[ipix]])
                sys.stdout.flush()

                # extract profile around each object
                x_, y_ = np.meshgrid(range(n),range(n))
                for ipix in range(npix):
                    r = np.floor(np.sqrt((x_-x[ipix])**2 + (y_-y[ipix])**2)).astype(np.int16)
                    for j in range(rpix):
                        vals[j] = np.concatenate((vals[j], starmap[r==j]))

            output['COUNTBLOCK'] += 1

    outst = ''
    for j in range(rpix):
        outst += '{:3d} {:8d}'.format(j, np.size(vals[j]))
        for q in [1,5,25,50,75,95,99]: outst += ' {:12.5E}'.format(np.percentile(vals[j],q))
        outst += '\n'
    ofile = outstem+'_dynrange.dat'
    with open(ofile, "w") as fn:
        fn.write(outst)
    if framenumber>0: output['DYNRANGE'] = ofile

    # save histograms
    ofile = outstem+'_sqrtS_hist.dat'
    np.savetxt(ofile, countnoise, header=' {:11.5E} {:9.6f}'.format(np.amax(countnoise[:,1]), 100*tnoise_gt/tnoise))
    output['SQRTS'] = ofile
    output['SQRTS_HEADER'] = (np.amax(countnoise[:,1]), 100*tnoise_gt/tnoise)
    ofile = outstem+'_neff_hist.dat'
    np.savetxt(ofile, countneff, header=' {:11.5E} {:9.6f}'.format(np.amax(countneff[:,1]), 100*tneff_gt/tneff))
    output['NEFF'] = ofile
    output['NEFF_HEADER'] = (np.amax(countneff[:,1]), 100*tneff_gt/tneff)

    output['NSTARLAYER'] = nstarlayer

    return output

if __name__ == "__main__":
    def fn(ibx,iby):
        return sys.argv[1] + '_{:02d}_{:02d}.fits'.format(ibx,iby)

    output = gen_dynrange_data(fn,sys.argv[2])
    print('**', output)
