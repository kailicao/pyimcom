# usage: python truthcats.py <filter> <input prefix> <outstem>
# input file name is <input prefix><filter>_DD_DD_map.fits (block files)

import sys
import numpy
import healpy
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.table import Table, vstack
from os.path import exists
from layer import GalSimInject, GridInject
import json
import re

bd = 40  # padding size
bd2 = 8

nblockmax = 100  # maximum
ncol = 22
nstart = 0

filter = sys.argv[1]

if filter == 'Y': filtername = 'Y106'
if filter == 'J': filtername = 'J129'
if filter == 'H': filtername = 'H158'
if filter == 'F': filtername = 'F184'
if filter == 'K': filtername = 'K213'

# prefix and suffix
in1 = sys.argv[2]
outstem = sys.argv[3]
outfile_g = outstem + '_TruthCat_{:s}.fits'.format(filter)

fullTables = {}

for iblock in range(nstart, nblockmax ** 2):

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
            n2 = int(configStruct['OUTSIZE'][1])  # will be used for coverage

            outscale = float(configStruct['OUTSIZE'][2])  # in arcsec
            force_scale = .40 / outscale  # in output pixels

            # padding region around the edge
            bdpad = int(configStruct['OUTSIZE'][1]) * int(configStruct['PAD'])

            # figure out which layer we want
            layers = [''] + configStruct['EXTRAINPUT']
            use_layers = {}
            print('# All EXTRAINPUT layers:', layers)
            for i in range(len(layers))[::-1]:
                m = re.match(r'^gs\S*$', layers[i])
                if m:
                    use_layers[str(m.group(0))] = i  # KL: later note: use re.split to separate at commas
                m = re.match(r'^ns\S*$', layers[i])
                if m:
                    use_layers[str(m.group(0))] = i

    if not exists(infile): continue

    with fits.open(infile) as f:
        mywcs = wcs.WCS(f[0].header)

    resolutionTables = {}  # re-initiate these to empty for each block, since we need to redo coords per block

    for layerName in use_layers.keys():

        params = re.split(r',', layerName)
        m = re.search(r'(\D*)(\d*)', params[0])
        if m:
            res = m.group(2)
            this_res = str(res)

        if 'TRUTH'+str(res) not in fullTables.keys:  # This will only happen one time
            fullTables['TRUTH'+str(res)] = []

        if this_res not in resolutionTables.keys:  # This will happen every time we start a new block

            resolutionTables[this_res]=None

            # Calculate the coordinate information for this block
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
            grp = numpy.where(numpy.logical_and(numpy.logical_and(xi >= bdpad, xi < n - bdpad),
                                                numpy.logical_and(yi >= bdpad, yi < n - bdpad)))
            ra_hpix = ra_hpix[grp]
            dec_hpix = dec_hpix[grp]
            ipix = qp[grp]
            x = x[grp]
            y = y[grp]
            npix = len(x)

            newpos = numpy.zeros((npix, ncol))
            xi = numpy.rint(x).astype(numpy.int16)
            yi = numpy.rint(y).astype(numpy.int16)

            # Initiate table
            blockTable = Table()
            blockTable['Block'] = [r'{:02d}_{:02d}'.format(ibx, iby)] * npix
            blockTable['Layer'] = [layerName] * npix
            blockTable['Res_hpix'] = res
            # Position information
            blockTable['ra_hpix'] = ra_hpix
            blockTable['dec_hpix'] = dec_hpix
            blockTable['ipix'] = ipix
            blockTable['ibx'] = ibx
            blockTable['iby'] = iby
            blockTable['x'] = x
            blockTable['y'] = y
            blockTable['xi'] = xi
            blockTable['yi'] = yi
            blockTable['dx=x-xi'] = dx = x - xi
            blockTable['dy=y-yi'] = dy = y - yi

            resolutionTables[this_res]=blockTable

        # default params
        seed = 4096
        shear = None

        if 'gsext' in layerName:
            for param in params:
                m = re.match(r'seed=(\d*)', param)
                if m: seed = m.group(1)
                m = re.match(r'shear=(\S*)', param)
                if m: shear = m.group(1)

            truthcat = GalSimInject.genobj(12 * 4 ** res, ipix, 'exp1', seed)

            if shear is not None:
                g_i = truthcat['g'][0, :] + truthcat['g'][1, :] * 1j
                q_i = (1 - numpy.absolute(g_i)) / (1 + numpy.absolute(g_i))

                apply_shear = re.split(r':', shear)
                g_t = apply_shear[0] + apply_shear[1] * 1j
                q_t = (1 - numpy.absolute(g_t)) / (1 + numpy.absolute(g_t))

                g_f = (g_i + g_t) / (1 + numpy.conj(g_t) * g_i)  # transformations
                r_f = truthcat['sersic']['r'][:] * numpy.sqrt(q_t / q_i)
                truthcat['g'][0, :] = g_f.real  # update the catalog
                truthcat['g'][1, :] = g_f.imag
                truthcat['sersic']['r'][:] = r_f

            # Include results in the table (for gsext objects)
            resolutionTables[this_res]['sersic_r_' + layerName] = truthcat['sersic']['r'][:] # this needs to be able to get the right table
            resolutionTables[this_res]['g1_' + layerName] = truthcat['g'][0, :]
            resolutionTables[this_res]['g2_' + layerName] = truthcat['g'][1, :]


        elif 'gsfdstar' in layerName:
            for param in params:
                m = re.match(r'[^a-zA-Z]+', param)
                if m:
                    fdm_amp = m.group(0)
            resolutionTables[this_res]['fdm_amp_' + layerName] = fdm_amp

        elif 'nstar' in layerName:
            # truthcat = croutine thing
            args = re.split(r',', params)
            tot_int = float(args[0])
            bg = float(args[1])
            q = int(args[2])
            ns_ipix, ns_xsca, ns_ysca, ns_rapix, ns_decpix = GridInject.generate_star_grid(int(res), mywcs)
            resolutionTables[this_res]['ipix' + layerName] = ns_ipix
            resolutionTables[this_res]['xsca' + layerName] = ns_xsca
            resolutionTables[this_res]['ysca' + layerName] = ns_ysca
            resolutionTables[this_res]['rapix' + layerName] = ns_rapix
            resolutionTables[this_res]['decpix' + layerName] = ns_decpix

            # inside layer loop

    # inside block loop
    print('TABLE INFO:')
    print('N_COLS:', len(resolutionTables[0].colnames))
    if iblock==nstart:
        print('BLOCK_INFO:',resolutionTables[0].info)
    # At this point all the layers have been added, the block table is complete
    for key1 in resolutionTables.keys:
        for key2 in fullTables.keys:
            if key1 in key2:
                fullTables[key2] = vstack([fullTables[key2],resolutionTables[key1]])



    # flush
    sys.stdout.flush()

# Make the fits file
phdu = fits.PrimaryHDU(np.zeros((2,2)))
phdu.header['COMMENT'] = 'This is a trivial HDU. Truth tables for injected objects at various HEALPix resolutions' \
                 ' are contained in the following table HDUs.'
hdul = fits.HDUList([phdu])

for key in fullTables.keys:
    hdu = fits.BinTableHDU(data=fullTables[key])
    hdu.header['RESOLUTION:'] = key
    hdu.header['FILTER:'] = filtername
    hdu.header['IN_BLOCK_PATH'] = infile
    hdul.append(hdu)

output_filename = 'multi_table.fits'
hdul.writeto(output_filename, overwrite=True)
print(f"FITS file '{output_filename}' with multiple tables created.")

