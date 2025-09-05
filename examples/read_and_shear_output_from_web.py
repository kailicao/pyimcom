"""Example of reading a mosaic remotely and shearing it."""

import sys
import time

import numpy as np
from pyimcom.meta import distortimage

t0 = time.time()

# Read in a group of blocks. This example uses several features of MetaMosaic:
#
# * The file name is in a regular expression format, with the extension and block coordinates
#   (here: ix=18, iy=14) at the end after the '^' character.
#
# * A bounding box [12<=ix<24, 12<=iy<24) is provided for which blocks are available.
#
# * The "extpix" keyword tells us to extend the mosaic 800 pixels from the block edge (reading
#   padding as needed from neighboring files).
#
in1 = distortimage.MetaMosaic(
    "https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/roman/preview/RomanWAS/images/"
    "coadds/H158/Row{1:02d}/prod_H_{0:02d}_{1:02d}^_18_14_map.fits",
    bbox=[12, 24, 12, 24],
    extpix=800,
    verbose=True,
)

# do some masking
in1.mask_fidelity_cut(40)  # mask pixels with U/C>1e-4
in1.mask_noise_cut(-3)  # mask pixels with noise>-3dB

# get the center
ra_, dec_ = in1.wcs.all_pix2world(in1.Nside / 2 - 0.5, in1.Nside / 2 - 0.5, 0)

# mask a 3x3 grid of "stars" -- just for demo
ra_mask, dec_mask = np.meshgrid(ra_ + np.linspace(-1, 1, 3) / 120.0, dec_ + np.linspace(-1, 1, 3) / 120.0)
ra_mask = ra_mask.ravel()
dec_mask = dec_mask.ravel()
radius_mask = 5.0 / 3600.0
print(ra_mask, dec_mask, radius_mask)
sys.stdout.flush()
in1.mask_caps(ra_mask, dec_mask, radius_mask)

# and we'll do one off to the side for good measure
dec_mask += 4.0 / 120.0
radius_mask = np.linspace(1.0, 3.0, len(ra_mask)) / 3600.0
print(ra_mask, dec_mask, radius_mask)
sys.stdout.flush()
in1.mask_caps(ra_mask, dec_mask, radius_mask)

in1.to_file("test-ex.fits")  # write the block + extension
print("&&", time.time() - t0)
sys.stdout.flush()

# Extract the unsheared image
# I_noshear is the object you want if you are going to just pull the raw pixels from the
# PyIMCOM outputs and then use another tool to do shearing.
I_noshear = in1.origimage(2800, select_layers=[0, 4, 6])
distortimage.shearimage_to_fits(I_noshear, "test-ns.fits", layers=None, overwrite=True)
print(I_noshear.keys())
print(I_noshear["pars"])
print(I_noshear["ref"])
print("&&", time.time() - t0)
sys.stdout.flush()

# Make a sheared image, 2800x2800, at original scale.
# This is an example of PyIMCOM Meta module applying a 3% shear.
I_shear = in1.shearimage(
    2800,
    jac=[[0.97, 0], [0, 1.03]],
    psfgrow=1.06,
    oversamp=1.0,
    Rsearch=3.5,
    select_layers=[0, 4, 6],
    verbose=True,
)
distortimage.shearimage_to_fits(I_shear, "test-sh.fits", layers=None, overwrite=True)
print("leakage=", I_shear["pars"]["UMAX"], "noise=", I_shear["pars"]["SMAX"])
print(I_shear.keys())
print(I_shear["pars"])
print(I_shear["ref"])
print("&&", time.time() - t0)
sys.stdout.flush()
del I_shear
