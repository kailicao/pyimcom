********************
PyIMCOM + Meta Tools
********************

First attempt at meta-ing images in PyIMCOM
===========================================

This is a simple module for doing meta-like operations on PyIMCOM coadds. I intend to add to it as necessary.

*Note: this version only works if PyIMCOM is used with the Gaussian output PSF, otherwise an error is returned.*

The algorithm is actually a nested version of IMCOM itself, except that because it takes input on a regular grid with analytically defined system 
matrices, it is much faster. The core interpolation function, ``ginterp.InterpMatrix``, takes a grid of samples with a function that has been assumed to 
be smoothed by a Gaussian. (The given FWHM in pixels is important to not amplify noise.) It can be thought of as an alternative to common interpolation 
schemes such as bi-linear, cubic, etc., but optimized for fields with a Gaussian PSF. The algorithm can implement smoothing by a Gaussian by including 
this in the target PSF (thus it is included in the IMCOM B-matrix). Then the ``ginterp.MultiInterp`` function can be used to interpolate onto a 
regularly spaced output grid with arbitrary affine paramterization.

The application to PyIMCOM outputs is in ``distortimage``, which contains the ``MetaMosaic`` class. This can be constructed from a PyIMCOM output file, 
and the information needed to build the portion of the mosaic needed is extracted from the FITS headers. A ``MetaMosaic`` contains up to a 3x3 grid of 
blocks (except that boundaries are masked). The ``to_file`` method saves that portion of a mosaic to a file, which might be useful for testing; but most 
users of this module will want the ``shearimage`` method, which constructs a sheared version of the mosaic with specified affine transformation, and 
with the PSF enlarged by a specified factor (usually a bit greater than 1). This is in the form of a Python dictionary, but it can be written to a FITS 
file using the ``shearimage_to_fits`` function.

Stand-alone usage
-----------------

It is possible to call this module in a few different ways. A simple one is to put a file ``test.py`` in the ``pyimcom/`` directory such as::

  import numpy as np
  from .meta import distortimage
  mosaic = distortimage.MetaMosaic('/fs/scratch/PCON0003/cond0007/anl-run-out/prod_H_24_13_map.fits', verbose=True)
  rot = 30*np.pi/180.
  im = mosaic.shearimage(3200, jac=[[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]], psfgrow=1.05, oversamp=1.)
  distortimage.shearimage_to_fits(im, 'xdist.fits', overwrite=True)

(you'll have to replace the path to the PyIMCOM output file you want) and then call it from one level up via::

  python3 -m pyimcom.test
