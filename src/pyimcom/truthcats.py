"""
Truth catalog generation (needed since some of the injected objects have random parameters).

Functions
---------
gen_truthcats
    Generates a truth catalog.

"""

import json
import re
import sys
import time
from os.path import exists

import healpy
import numpy
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Column, Table, vstack

from .coadd import Block
from .config import Config, Settings
from .layer import GalSimInject


def gen_truthcats(pars):
    """Generates a truth catalog and writes it to a FITS file.

    Parameters
    ----------
    pars : list
        A 4-entry list of:
        * name : str or None
        * filter : str or int
        * input prefix : str
        * outstem : str

    Returns
    -------
    None

    [<name>, <filter>, <input prefix>, <outstem>]

    Notes
    -----
    If "name" is not None, then reads WCS from the output file.
    If "name" is None, then the WCS is generated internally. In this case, only the
    starting block output file needs to exist.

    The "filter" should either be a letter ('F','H','J',...) or an integer (0,1,2,...) designation.
    If it is a letter, then you should still include it in the input file prefix.

    If the outstem is '', then the output file name is generated according to the configuration file.

    """

    t0 = time.time()

    # bd = 40  # padding size
    # bd2 = 8

    nblockmax = 100  # maximum
    # ncol = 22 # <-- legacy number of columns
    nstart = 0

    filter = pars[1]

    if isinstance(filter, int):
        filtername = Settings.RomanFilters[filter]
        filter = filtername[0]
        filter_ = ""
    else:
        if filter == "Y":
            filtername = "Y106"
        if filter == "J":
            filtername = "J129"
        if filter == "H":
            filtername = "H158"
        if filter == "F":
            filtername = "F184"
        if filter == "K":
            filtername = "K213"
        filter_ = filter

    # prefix and suffix
    in1 = pars[2]
    outstem = pars[3]
    if pars[3]:
        outfile_g = outstem + f"TruthCat_{filter:s}.fits"

    fullTables = {}

    for iblock in range(nstart, nblockmax**2):
        j = iblock
        ibx = j % nblockmax
        iby = j // nblockmax

        # make sure that we can read either the legacy (_map.fits) or new (.fits) file names.
        infile = in1 + f"{filter_:s}_{ibx:02d}_{iby:02d}_map.fits"
        if not exists(infile):
            infile = in1 + f"{filter_:s}_{ibx:02d}_{iby:02d}.fits"
        if not exists(infile):
            continue

        print(f"INFILE: {infile}")

        # extract information from the header of the first file
        if iblock == nstart:
            infile_ = infile  # save the first input file in the FITS header
            with fits.open(infile) as f:
                n = numpy.shape(f[0].data)[-1]  # size of output images

                config = ""
                for g in f["CONFIG"].data["text"].tolist():
                    config += g + " "
                configStruct = json.loads(config)

                blocksize = (
                    int(configStruct["OUTSIZE"][0])
                    * int(configStruct["OUTSIZE"][1])
                    * float(configStruct["OUTSIZE"][2])
                    / 3600.0
                    * numpy.pi
                    / 180
                )  # radians
                rs = 1.5 * blocksize / numpy.sqrt(2.0)  # search radius
                # n2 = int(configStruct["OUTSIZE"][1])  # will be used for coverage

                # outscale = float(configStruct["OUTSIZE"][2])  # in arcsec
                # force_scale = .40 / outscale  # in output pixels <-- not used

                # padding region around the edge
                bdpad = int(configStruct["OUTSIZE"][1]) * int(configStruct["PAD"])

                # figure out which layer we want
                layers = [""] + configStruct["EXTRAINPUT"]
                use_layers = {}
                print("# All EXTRAINPUT layers:", layers)
                for i in range(len(layers))[::-1]:
                    m = re.match(r"^gs\S*$", layers[i])
                    if m:
                        use_layers[str(m.group(0))] = i  # KL: later note: use re.split to separate at commas
                    m = re.match(r"^ns\S*$", layers[i])
                    if m:
                        use_layers[str(m.group(0))] = i

                CFG = Config(config)
                CFG.tempfile = None  # the temporary directory for coaddition might not exist anymore
                if not pars[3]:
                    outfile_g = CFG.outstem + "_TruthCat.fits"

        if pars[0] is not None:
            with fits.open(infile) as f:
                mywcs = wcs.WCS(f[0].header)
        else:
            B = Block(cfg=CFG, this_sub=ibx * CFG.nblock + iby, run_coadd=False)
            B.parse_config()
            mywcs = B.outwcs

        # re-initiate these to empty for each block, since we need to redo coords per block
        resolutionTables = {}

        for layerName in use_layers:
            print("LAYERNAME: ", layerName)

            params = re.split(r",", layerName)
            print("PARAMS:", params)
            m = re.search(r"(\D*)(\d*)", params[0])
            if m:
                res = int(m.group(2))
                this_res = str(res)

            if "TRUTH" + str(res) not in fullTables:  # This will only happen one time
                fullTables["TRUTH" + str(res)] = []

            # This will happen when we start a new block if we don't have
            # a layer at this resolution yet. So this part of the code should handle
            # anything generic to that HEALPix resolution (RA/Dec, position in the block, etc.)
            # that doesn't depend on *what* object was injected there.
            if this_res not in resolutionTables:
                resolutionTables[this_res] = None

                # Calculate the coordinate information for this block
                if mywcs.pixel_n_dim == 4:
                    ra_cent, dec_cent = mywcs.all_pix2world(
                        [(n - 1) / 2], [(n - 1) / 2], [0.0], [0.0], 0, ra_dec_order=True
                    )
                else:
                    ra_cent, dec_cent = mywcs.all_pix2world(
                        [(n - 1) / 2], [(n - 1) / 2], 0, ra_dec_order=True
                    )
                ra_cent = ra_cent[0]
                dec_cent = dec_cent[0]
                vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
                qp = healpy.query_disc(2**res, vec, rs, nest=False)
                ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=False, lonlat=True)
                npix = len(ra_hpix)
                if mywcs.pixel_n_dim == 4:
                    x, y, z1, z2 = mywcs.all_world2pix(
                        ra_hpix, dec_hpix, numpy.zeros((npix,)), numpy.zeros((npix,)), 0
                    )
                else:
                    x, y = mywcs.all_world2pix(ra_hpix, dec_hpix, 0)
                xi = numpy.rint(x).astype(numpy.int16)
                yi = numpy.rint(y).astype(numpy.int16)
                grp = numpy.where(
                    numpy.logical_and(
                        numpy.logical_and(xi >= bdpad, xi < n - bdpad),
                        numpy.logical_and(yi >= bdpad, yi < n - bdpad),
                    )
                )
                ra_hpix = ra_hpix[grp]
                dec_hpix = dec_hpix[grp]
                ipix = qp[grp]
                x = x[grp]
                y = y[grp]
                npix = len(x)

                xi = numpy.rint(x).astype(numpy.int16)
                yi = numpy.rint(y).astype(numpy.int16)

                # orientation angle (computed by finite difference @ +/- 1 arcsec)
                if mywcs.pixel_n_dim == 4:
                    xPP, yPP, z1, z2 = mywcs.all_world2pix(
                        ra_hpix, dec_hpix + 1.0 / 3600.0, numpy.zeros((npix,)), numpy.zeros((npix,)), 0
                    )
                    xMM, yMM, z1, z2 = mywcs.all_world2pix(
                        ra_hpix, dec_hpix - 1.0 / 3600.0, numpy.zeros((npix,)), numpy.zeros((npix,)), 0
                    )
                else:
                    xPP, yPP = mywcs.all_world2pix(ra_hpix, dec_hpix + 1.0 / 3600.0, 0)
                    xMM, yMM = mywcs.all_world2pix(ra_hpix, dec_hpix - 1.0 / 3600.0, 0)
                pa_hpix = np.arctan2(xPP - xMM, yPP - yMM) * 180.0 / numpy.pi
                pa_hpix -= 360.0 * np.floor(pa_hpix / 360.0)

                # Initiate table
                blockTable = Table(
                    {
                        "Block": Column([rf"{ibx:02d}_{iby:02d}"] * npix, dtype="U5"),
                        "Layer": Column([layerName] * npix, dtype="U160"),
                        "Res_hpix": Column([res] * npix, dtype="int64"),
                        "ra_hpix": Column(ra_hpix, dtype="float64"),
                        "dec_hpix": Column(dec_hpix, dtype="float64"),
                        "pa_hpix": Column(pa_hpix, dtype="float64"),
                        "ipix": Column(ipix, dtype="int64"),
                        "ibx": Column([np.int16(ibx)] * npix, dtype="int16"),
                        "iby": Column([np.int16(iby)] * npix, dtype="int16"),
                        "x": Column(x, dtype="float64"),
                        "y": Column(y, dtype="float64"),
                        "xi": Column(np.int32(xi), dtype="int32"),
                        "yi": Column(np.int32(yi), dtype="int32"),
                        "dx": Column(x - xi, dtype="float64"),
                        "dy": Column(y - yi, dtype="float64"),
                    }
                )

                resolutionTables[this_res] = blockTable

            # The following code should get executed for each layer, and will generate
            # more columns if need be. That depends on the type of layer.

            # default params
            seed = 4096
            shear = None

            icase = f"{use_layers[layerName]:d}"  # identifying integer for this layer

            # ## extended object layer ##
            if "gsext" in layerName:
                for param in params:
                    m = re.match(r"seed=(\d*)", param)
                    if m:
                        seed = int(m.group(1))
                    m = re.match(r"shear=(\S*)", param)
                    if m:
                        shear = m.group(1)

                truthcat = GalSimInject.genobj(12 * 4**res, ipix, "exp1", seed)

                if shear is not None:
                    g_i = truthcat["g"][0, :] + truthcat["g"][1, :] * 1j
                    q_i = (1 - numpy.absolute(g_i)) / (1 + numpy.absolute(g_i))

                    apply_shear = re.split(r":", shear)
                    g_t = float(apply_shear[0]) + float(apply_shear[1]) * 1j
                    q_t = (1 - numpy.absolute(g_t)) / (1 + numpy.absolute(g_t))

                    g_f = (g_i + g_t) / (1 + numpy.conj(g_t) * g_i)  # transformations
                    r_f = truthcat["sersic"]["r"][:] * numpy.sqrt(q_t / q_i)

                    truthcat["g"][0, :] = g_f.real  # update the catalog
                    truthcat["g"][1, :] = g_f.imag
                    truthcat["sersic"]["r"][:] = r_f

                # Include results in the table (for gsext objects)
                resolutionTables[this_res].add_column(
                    Column(truthcat["sersic"]["r"][:], dtype="float64", name="sersic_r_L" + icase)
                )
                resolutionTables[this_res].add_column(
                    Column(truthcat["g"][0, :], dtype="float64", name="g1_L" + icase)
                )
                resolutionTables[this_res].add_column(
                    Column(truthcat["g"][1, :], dtype="float64", name="g2_L" + icase)
                )

            # ## field-dependent amplitude star layer ##
            elif "gsfdstar" in layerName:
                for param in params:
                    m = re.match(r"[^a-zA-Z]+", param)
                    if m:
                        fdm_amp = m.group(0)
                resolutionTables[this_res].add_column(
                    Column([fdm_amp] * len(ipix), dtype="float64", name="fdm_amp_L" + icase)
                )

            # elif 'nstar' in layerName:
            #     args = params
            #     ns_ipix, ns_xsca, ns_ysca, ns_rapix, ns_decpix = GridInject.generate_star_grid(res, mywcs)
            #     resolutionTables[this_res]['ipix' + layerName] = ns_ipix
            #     resolutionTables[this_res]['xsca' + layerName] = ns_xsca
            #     resolutionTables[this_res]['ysca' + layerName] = ns_ysca
            #     resolutionTables[this_res]['rapix' + layerName] = ns_rapix
            #     resolutionTables[this_res]['decpix' + layerName] = ns_decpix

            # inside layer loop

        # inside block loop
        # At this point all the layers have been added, the block table is complete
        for key1 in resolutionTables:
            for key2 in fullTables:
                if key1 in key2:
                    fullTables[key2] = vstack([fullTables[key2], resolutionTables[key1]])

        # flush
        sys.stdout.flush()

    # Make the fits file
    phdu = fits.PrimaryHDU(np.zeros((2, 2)))
    phdu.header["COMMENT"] = (
        "This is a trivial HDU. Truth tables for injected objects at various HEALPix resolutions"
        " are contained in the following table HDUs."
    )
    hdul = fits.HDUList([phdu])

    for key in fullTables:
        hdu = fits.BinTableHDU(data=fullTables[key])
        hdu.name = key
        hdu.header["RESOLUTI"] = key
        hdu.header["FILTER"] = filtername
        hdu.header["INBLKPTH"] = infile_
        for i in range(len(layers)):
            this_keyword = f"LYR{i:d}"
            hdu.header[this_keyword] = (layers[i], f"name of layer {i:d}")

        # remove TNULL since it has the potential to cause some problems.
        for kw in hdu.header.keys():  # easier to read this version # noqa: SIM118
            if kw[:5] == "TNULL":
                hdu.header.remove(kw)

        hdul.append(hdu)

    hdul.writeto(outfile_g, overwrite=True)
    print(f"FITS file '{outfile_g}' with truth tables created.")
    print(f"Time elapsed: {time.time() - t0}")


def gen_truthcats_from_cfg(cfg):
    """Usage from configuration file."""
    gen_truthcats([None, cfg.use_filter, cfg.outstem, None])


# stand-alone usage
if __name__ == "__main__":
    # gen_truthcats(sys.argv)
    gen_truthcats([None, 1, "/fs/scratch/PCON0003/cond0007/itertest2-out/itertest2_F", "pyimcom/temp/test1"])
