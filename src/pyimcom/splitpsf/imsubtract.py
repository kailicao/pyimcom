"""
Routines for implementing the image subtraction step in PSF wing removal.

Functions
---------
pltshow
    Helper to determine where to save a plot.
get_wcs
    Extracts the World Coordinate System from a cached file.
run_imsubtract
    Main workflow for image subtraction step.

"""

# from astropy.wcs import WCS
import os
import re
import sys
import time

import asdf
import matplotlib
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from scipy.signal.windows import tukey

# local imports
from ..config import Config
from ..utils import compareutils
from ..wcsutil import PyIMCOM_WCS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def pltshow(plt, display, pars={}):
    """
    Where to save a plot.

    Parameters
    ----------
    plt : matplotlib.pyplot
        The pyplot module to use for plotting.
    display : str or None
        Sends to file (if string), screen (None), or nowhere (if '/dev/null')
    pars : dict, optional
        Parameters for saving the file.
        Must be provided if a file is requested.

    Returns
    -------
    None

    Notes
    -----
    The `pars` dictionary contains the keys:
    * 'type' : str, currently only supports 'window'
    * 'obsid' : int, observation ID
    * 'sca' : int, SCA number
    * 'ix' : int, x block index
    * 'iy' : int, y block index

    """

    if display is None:
        plt.show()
        return

    if display == "/dev/null":
        return

    # if we get here, we need to save the file
    if pars["type"].lower() == "window":
        obsid = pars["obsid"]
        sca = pars["sca"]
        ix = pars["ix"]
        iy = pars["iy"]
        plt.savefig(display + f"_{obsid}_{sca}_{ix:02d}_{iy:02d}.png")


def get_wcs(cachefile):
    """
    Gets the WCS from a cached FITS file.

    If a gwcs is used, finds the attached ASDF file and reads that.

    Parameters
    ----------
    cachefile : str
        Name of the cached file.

    Returns
    -------
    pyimcom.wcsutils.PyIMCOM_WCS
        The World Coordinate System of the cached file.

    """

    with fits.open(cachefile) as hdul:
        if "WCSTYPE" in hdul[1].header and hdul[1].header["WCSTYPE"][:4].lower() == "gwcs":
            with asdf.open(cachefile[:-5] + "_wcs.asdf") as f2:
                return PyIMCOM_WCS(f2["wcs"])
        return PyIMCOM_WCS(hdul["SCIWCS"].header)


def get_wcs_from_infile(infile):
    """
    #### I need to add the documentation for this

    """
    g = infile[0].header
    block_wcs = SlicedLowLevelWCS(WCS(g), slices=[0, 0, slice(0, g["NAXIS2"]), slice(0, g["NAXIS1"])])

    return block_wcs


def run_imsubtract(config_file, display=None):
    """
    Main routine to run imsubtract.

    Parameters
    ----------
    config_file : str
        Location of a configuration file.
    display : str or None, optional
        Display location for intermediate steps.

    Notes
    -----
    There are several options for `display`:

    * `display` = None : print to screen
    * `display` = '/dev/null' : don't save
    * `display` = any other string : save to ``display+f'_{obsid}_{sca}_{ix:02d}_{iy:02d}.png'``

    """

    # load the file using Config and get information
    cfgdata = Config(config_file)

    info = cfgdata.inlayercache
    block_path = cfgdata.outstem
    ra = cfgdata.ra * (np.pi / 180)  # convert to radians
    dec = cfgdata.dec * (np.pi / 180)  # convert to radians
    lonpole = cfgdata.lonpole * (np.pi / 180)  # convert to radians
    nblock = cfgdata.nblock
    n1 = cfgdata.n1  # number of postage stamps
    n2 = cfgdata.n2  # size of single run
    postage_pad = cfgdata.postage_pad  # postage stamp padding
    dtheta_deg = cfgdata.dtheta
    blocksize_rad = n1 * n2 * dtheta_deg * (np.pi) / 180  # convert to radians
    # print(ra, dec, lonpole, nblock, n1, n2, dtheta_deg)

    # separate the path from the inlayercache info
    m = re.search(r"^(.*)\/(.*)", info)
    if m:
        path = m.group(1)
        exp = m.group(2)
    # print(path, exp)

    # create empty list of exposures
    exps = []

    # find all the fits files and add them to the list
    for _, _, files in os.walk(path):
        for file in files:
            if file.startswith(exp) and file.endswith(".fits") and file[-6].isdigit():
                exps.append(file)
    print("list of files:", exps)

    # move to the directory with the files
    os.chdir(path)

    # loop over the list of observation pair files (for each SCA)
    for exp in exps:
        # get SCA and obsid
        m2 = re.search(r"(\w*)_0*(\d*)_(\d*).fits", exp)
        if m2:
            obsid = int(m2.group(2))
            sca = int(m2.group(3))
        print("OBSID: ", obsid, "SCA: ", sca)

        # inlayercache data --- changed to context manager structure
        with fits.open(exp) as hdul:
            # read in the input image, I
            I_input = np.copy(hdul[0].data)  # this is I # noqa: F841

        # get wcs information from fits file (or asdf if indicated)
        sca_wcs = get_wcs(exp)

        # results from splitpsf
        # read in the kernel
        hdul2 = fits.open(f"{info}.psf/psf_{obsid:d}.fits")
        K = np.copy(hdul2[sca + hdul2[0].header["KERSKIP"]].data)
        # get the number of pixels on the axis
        axis_num = K.shape[1]
        # get the oversampling factor
        oversamp = hdul2[0].header["OVSAMP"]
        hdul2.close()

        # get the kernel size
        s_in_rad = 0.11 * np.pi / (180 * 3600)  # convert arcsec to radians
        ker_size = axis_num / oversamp * s_in_rad
        print("kernel size: ", ker_size)

        # define pad
        pad = ker_size / 2  # at least half of the kernel size in native pixels
        # convert to x, y, z using wcs coords (center of SCA)
        x, y, z, p = compareutils.getfootprint(sca_wcs, pad)
        v = np.array([x, y, z])

        # convert to x', y', z'
        # define coordinates and transformation matrix
        ex = np.array([np.sin(ra), -np.cos(ra), 0])
        ey = np.array([-np.cos(ra) * np.sin(dec), -np.sin(dec) * np.sin(ra), np.cos(dec)])
        ez = np.array([-np.cos(dec) * np.cos(ra), -np.cos(dec) * np.sin(ra), -np.sin(dec)])
        T = np.array([ex, ey, ez])

        # perform transformation and define individual values
        v_p = np.matmul(T, v)
        x_p = v_p[0]
        y_p = v_p[1]
        z_p = v_p[2]

        # define the rotation matrix, coefficient, and additional vector
        rot = np.array([[-np.cos(lonpole), -np.sin(lonpole)], [np.sin(lonpole), -np.cos(lonpole)]])
        coeff = 2 / (1 - z_p)
        v_convert = np.array([x_p, y_p])

        # convert to eta and xi (block coordinates)
        block_coords = coeff * np.matmul(rot, v_convert)
        xi = block_coords[0]  # noqa: F841
        eta = block_coords[1]  # noqa: F841

        # find theta in original coordinates, convert to block coordinates
        theta = (
            2 * np.arctan(np.sqrt(p / (2 - p)))
            + blocksize_rad / np.sqrt(2)
            + np.sqrt(2) * pad
            + ker_size / np.sqrt(2)
        ) * coeff
        theta_block = theta / blocksize_rad
        print("theta in units of blocks: ", theta_block)
        # sigma = (nblock*blocksize_rad)/np.sqrt(2)    # I don't think I need these for the grid method
        # theta_max = theta * (1+(sigma**2)/4)

        # add theta to this set of coords
        block_coords = np.append(block_coords, theta)

        # convert the units of this coordinate system to blocks
        block_coords_blocks = block_coords / blocksize_rad

        # find the center of SCA relative to the bottom left of the mosaic
        SCA_coords = block_coords_blocks.copy()
        SCA_coords[:2] += nblock / 2  # take only the xi and eta directions

        # find the blocks the SCA covers
        side = np.arange(nblock) + 0.5
        xx, yy = np.meshgrid(side, side)
        distance = np.hypot(xx - SCA_coords[0], yy - SCA_coords[1])
        in_SCA = np.where(distance <= theta_block)
        block_list = np.stack((in_SCA[1], in_SCA[0]), axis=-1)
        # print(SCA_coords, block_list)
        # print('>', blocksize_rad, xi, eta, v)
        print("list of blocks: \n", block_list)

        # loop over the blocks in the list
        count = 0  # noqa: F841
        for ix, iy in block_list:
            print("BLOCK: ", ix, iy)

            # open the block info
            hdul3 = fits.open(block_path + f"_{ix:02d}_{iy:02d}.fits")
            block_data = np.copy(hdul3[0].data)
            block_wcs = get_wcs_from_infile(hdul3)
            hdul3.close()

            # determine the length of one axis of the block
            block_length = block_data.shape[-1]  # length in output pixels
            overlap = n2 * postage_pad  # size of one overlap region due to postage stamp
            a1 = 4 * overlap / (block_length - 1)  # percentage of region to have window function taper
            # the '-1' is due to scipy's convention on alpha that the denominator is the distance from the
            # first to the last point, so 1 less than the length
            window = tukey(block_length, alpha=a1)
            # apply window function to block data
            block = block_data[0] * window  # noqa: F841

            # check the window function
            plt.plot(np.arange(len(window)), window, color="indigo")
            plt.axvline(block_length - 1, c="mediumpurple")
            plt.axvline(block_length - overlap - 1, c="mediumpurple")
            plt.axvline(block_length - 2 * overlap - 1, c="mediumpurple")
            plt.xlim(block_length - 3 * overlap, block_length + overlap)
            plt.plot(block_length - 2, window[block_length - 2], c="darkmagenta", marker="o")
            plt.plot(
                block_length - 2 * overlap, window[block_length - 2 * overlap], c="darkmagenta", marker="o"
            )
            plt.plot(block_length - overlap, window[block_length - overlap], c="blueviolet", marker="o")
            plt.plot(
                block_length - overlap - 2, window[block_length - overlap - 2], c="blueviolet", marker="o"
            )
            pltshow(plt, display, {"type": "window", "obsid": obsid, "sca": sca, "ix": ix, "iy": iy})
            print(
                window[block_length - 2],
                window[block_length - 2 * overlap],
                window[block_length - 2] + window[block_length - 2 * overlap],
            )
            print(
                window[block_length - overlap],
                window[block_length - overlap - 2],
                window[block_length - overlap] + window[block_length - overlap - 2],
            )

            # find the 'Bounding Box' in SCA coordinates
            # create mesh grid for output block
            block_arr = np.arange(block_length)
            x_out, y_out = np.meshgrid(block_arr, block_arr)
            # convert to ra and dec using block wcs
            ra_sca, dec_sca = block_wcs.pixel_to_world_values(x_out, y_out, 0)
            # print(ra_sca.shape, dec_sca.shape)
            print("ra, dec: ", ra_sca[0::2663, 0::2663], dec_sca[0::2663, 0::2663])
            # convert into coordinates in the SCA
            x_in, y_in = sca_wcs.all_world2pix(ra_sca, dec_sca, 0)
            print("x_in, y_in: ", x_in[0::2663, 0::2663], y_in[0::2663, 0::2663])
            # get the bounding box from the max and min values
            left = np.floor(np.min(x_in))
            right = np.ceil(np.max(x_in))
            bottom = np.floor(np.min(y_in))
            top = np.ceil(np.max(y_in))
            # create the bounding box mesh grid, with ovsamp
            # determine side lengths of the box
            width = oversamp * (right - left) + 2
            height = oversamp * (top - bottom) + 2
            # create arrays for meshgrid
            x = np.linspace(left, right, width)
            y = np.linspace(bottom, top, height)
            bb_x, bb_y = np.meshgrid(x, y)


if __name__ == "__main__":
    """Calling program is here.

    python3 -m pyimcom.splitpsf.imsubtract <config> [<output images>]
    (uses plt.show() if output stem not specified; output image directory is relative to cache file)

    """

    start = time.time()
    # get the json file
    config_file = sys.argv[1]

    display = "/dev/null"
    if len(sys.argv) > 2:
        display = sys.argv[2]
    run_imsubtract(config_file, display=display)

    end = time.time()
    elapsed = end - start
    print(f"Execution time: {elapsed:.4f} seconds.")
