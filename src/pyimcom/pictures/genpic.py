"""
Tools to make a mosaic picture.

Functions
---------
resolve_bounds
    Turn boundary list into tuple.
get_config
    Utility to get the configuration from a FITS file..
cmapscale
    Color mapping.
make_picture_1band
    Turns a single-band mosaic into an image file.

"""

import numpy as np
import sys
import os
from astropy.io import fits
from PIL import Image
from matplotlib import cm

from ..config import Config

def resolve_bounds(bounds, nblock):
    """
    Turns bounds object into a tuple of ymin,ymax,xmin,xmax restricted to `nblock` x `nblock` region.

    Parameters
    ----------
    bounds : list of int
        Length 4: [ymin,ymax,xmin,xmax]
    nblock : int
        The mosaic size in blocks.

    Returns
    -------
    ymin : int
        Bottom side of mosaic (inclusive).
    ymax : int
        Top side of mosaic (exclusive).
    xmin : int
        Left side of mosaic (inclusive).
    xmax : int
        Right side of mosaic (exclusive).

    """

    def check1(ymin,ymax,xmin,xmax):
        return ymin>=0 and ymax<=nblock and xmin>=0 and xmax<=nblock and ymax>ymin and xmax>xmin

    # None is the whole block
    if bounds is None:
        return 0,nblock,0,nblock

    # list-like object
    if type(bounds) is list:
        ymin = int(bounds[0])
        ymax = (int(bounds[1])+nblock-1)%nblock +1
        xmin = int(bounds[2])
        xmax = (int(bounds[3])+nblock-1)%nblock +1

        if check1(ymin,ymax,xmin,xmax):
            return ymin,ymax,xmin,xmax
        else:
            raise Exception("genpic.resolve_bounds: Invalid bounds")

    # by default, take the whole block
    return 0,nblock,0,nblock

def get_config(fn1):
    """
    Utility to get the configuration.

    Parameters
    ----------
    fn1 : str
        File name.

    Returns
    -------
    pyimcom.config.Config
        The configuration used to generate `fn1`.

    """

    cf = ''
    with fits.open(fn1) as f:
        for line in f['CONFIG'].data['text']:
            cf += line + "\n"
    cfg = Config(cf)
    return cfg

# color mapping, input --> output on 0-255 scale
def cmapscale(inarray, srange, cmap=None, stretch='asinh'):
    """
    Color mapping for making a display image.

    Parameters
    ----------
    inarray : np.array of float
        The array to be mapped. Shape (ny,nx).
    srange : (float, float)
        The minimum and maximum values to be represented (values beyond this will saturate).
    cmap : str or None, default=None
        If string, uses that color scale; if None, makes a black and white image.
    stretch : str, default='asinh'
        The stretch. Current options are 'linear' and 'asinh'.

    Returns
    -------
    np.array of uint8
        Either a grayscale 2D array shape = (ny,nx) if cmap is None;
        or an RGB 3D array shape = (ny,nx,3) if cmap is not None.

    """

    (lsmin,lsmax) = srange

    medarray = np.clip(inarray,lsmin,lsmax)
    if stretch=='asinh':
        outarray = (np.arcsinh(medarray/np.abs(lsmin)) - np.arcsinh(-1)) / (np.arcsinh(lsmax/np.abs(lsmin)) - np.arcsinh(-1))
    elif stretch=='linear':
        outarray = (medarray-lsmin)/(lsmax-lsmin)
    else:
        raise Exception('Unrecognized stretch type: '+stretch)
    outarray = np.clip(outarray,0,1)

    # black & white
    if cmap is None:
        return(np.clip(np.rint(255*outarray),0,255).astype(np.uint8))
    # color
    return((getattr(cm,cmap)(outarray)*255).astype(np.uint8)[:,:,:3])


def make_picture_1band(fn, outfile, layer=0, bounds=None, binning=1, cmap=None, srange=(-8.,600.), stretch='asinh'):
    """
    Writes a mosaic image from a set of IMCOM output files.

    Parameters
    ----------
    fn : str
        File stem (without the _DD_DD.fits).
    outfile : str
        Output file name.
    layer : int, default=0
        Which image layer to use (default is the Science layer).
    bounds : list or None, default=None
        Boundary of the output image. If a list, should be [ymin,ymax,xmin,xmax], to indicate the range
        xmin<=x<xmax, ymin<=y<ymax. If None (default), draws the whole mosaic.
    binning : int, default=1
        Binning relative to the FITS images. Larger binning reduces size of output image.
        The default is 1, corresponding to native resolution.
    cmap : str or None, default=None
        Color map (uses matplotlib names; None -> black & white).
    srange : (float, float), default=(-8.,600.)
        Minimum and maximum of the color scale.
    stretch : str, default='asinh'
        Stretch type (currently: 'asinh' or 'linear')

    Returns
    -------
    None

    """

    bw = (cmap is None)

    # get the configuration
    cfg = get_config(fn+'_00_00.fits')
    nint = cfg.n1*cfg.n2
    pad = cfg.n2*cfg.postage_pad

    # check that this can be binned
    if nint%binning>0:
        raise Exception("genpic.make_picture_1band: can't bin {:d} in groups of {:d}".format(nint, binning))

    # and the subregion
    ymin,ymax,xmin,xmax = resolve_bounds(bounds, cfg.nblock)
    cube = np.zeros(((ymax-ymin)*nint//binning,(xmax-xmin)*nint//binning,(1 if bw else 3)), dtype=np.uint8)

    # now read the files
    for ix in range(xmax-xmin):
        for iy in range(ymax-ymin):
            fname = fn+'_{:02d}_{:02d}.fits'.format(ix+xmin,iy+ymin)
            if os.path.exists(fname):
                with fits.open(fname) as f:
                    print(pad, np.shape(f[0].data), fname)
                    D = np.mean(f[0].data[0,layer,pad:-pad,pad:-pad].reshape((nint//binning,binning,nint//binning,binning)), axis=(1,3))
                    if bw:
                        cube[iy*nint//binning:(iy+1)*nint//binning,ix*nint//binning:(ix+1)*nint//binning,0] = cmapscale(D, srange, cmap=cmap, stretch=stretch)
                    else:
                        cube[iy*nint//binning:(iy+1)*nint//binning,ix*nint//binning:(ix+1)*nint//binning,:] = cmapscale(D, srange, cmap=cmap, stretch=stretch)

    if bw:
        Image.fromarray(cube[::-1,:,0]).save(outfile)
    else:
        Image.fromarray(cube[::-1,:,:]).save(outfile)
