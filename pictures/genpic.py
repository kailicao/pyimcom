import numpy as np
import sys
import os
from astropy.io import fits
from PIL import Image
from matplotlib import cm

from ..config import Config

def resolve_bounds(bounds, nblock):
    """Turns bounds object into a tuple of ymin,ymax,xmin,xmax"""

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
    """Utility to get the configuration file."""

    cf = ''
    with fits.open(fn1) as f:
        for line in f['CONFIG'].data['text']:
            cf += line + "\n"
    cfg = Config(cf)
    return cfg

# color mapping, input --> output on 0-255 scale
def cmapscale(inarray, srange, cmap=None, stretch='asinh'):

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
    """Writes a mosaic image from a set of IMCOM output files.

    Inputs:
        fn = file stem (without the _DD_DD.fits)
        outfile = output file name
        layer = which image layer to use
        bounds = boundary of the output image (default = whole mosaic)
        binning = binning relative to the FITS images (reduces size of output image)
        cmap = color map (uses matplotlib names; None -> black & white)
        srange = color scale (min,max)
        stretch = stretch type (current: asinh, linear)
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
