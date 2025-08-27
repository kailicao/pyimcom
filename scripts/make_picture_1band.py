"""
Stand-alone script to make a display PNG image from a mosaic.

Calling format::

   python -m make_picture_1band /fs/scratch/PCON0003/cond0007/itertest2-out/itertest2_F\
       xstart ystart n out.png

if the output images are in ``/fs/scratch/PCON0003/cond0007/itertest2-out/itertest2_F_DD_DD.fits``
(where ``DD`` = x & y block indices)
and you want to combine n x n blocks starting at (xstart,ystart).

"""

import sys

import numpy as np
from astropy.io import fits
from PIL import Image
from pyimcom.config import Config

# extract the file name (without the '_DD_DD.fits')
fn = sys.argv[1]
cf = ""
with fits.open(sys.argv[1] + "_00_00.fits") as f:
    for line in f["CONFIG"].data["text"]:
        cf += line + "\n"
cfg = Config(cf)

nint = cfg.n1 * cfg.n2
pad = cfg.n2 * cfg.postage_pad

# subregions
xstart = int(sys.argv[2])
ystart = int(sys.argv[3])
n = int(sys.argv[4])

if xstart < 0 or ystart < 0 or xstart + n > cfg.nblock or ystart + n > cfg.nblock:
    print("Error: invalid subregion")
    exit()

print(cfg.nblock, xstart, ystart, n, nint, pad)

cube = np.zeros((n * nint, n * nint, 3), dtype=np.uint8)

# scale


#        'ra', 'dec', 'nblock', 'n1', 'n2', 'dtheta', 'Nside',  # SECTION III
#        'fade_kernel', 'postage_pad', 'pad_sides', 'stoptile',  # SECTION IV


# color mapping, input --> output on 0-255 scale
def cmapscale(inarray, lsmin=-8.0, lsmax=600.0):
    """Color mapping: asinh scale."""

    medarray = np.clip(inarray, lsmin, lsmax)
    outarray = (np.arcsinh(medarray / np.abs(lsmin)) - np.arcsinh(-1)) / (
        np.arcsinh(lsmax / np.abs(lsmin)) - np.arcsinh(-1)
    )
    return np.clip(np.rint(255 * outarray), 0, 255).astype(np.uint8)


for ix in range(n):
    for iy in range(n):
        fname = sys.argv[1] + f"_{ix + xstart:02d}_{iy + ystart:02d}.fits"
        print(fname)
        print(iy * nint, (iy + 1) * nint, ix * nint, (ix + 1) * nint)
        with fits.open(fname) as f:
            cube[iy * nint : (iy + 1) * nint, ix * nint : (ix + 1) * nint, 1] = cmapscale(
                f[0].data[0, 0, pad:-pad, pad:-pad]
            )

Image.fromarray(cube[::-1, :, :]).save(sys.argv[5])
