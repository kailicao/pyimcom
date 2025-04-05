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

It is possible to call this module in a few different ways. A simple one is the following code, which rotates the image by 30 degrees and grows the PSF by a factor of 1.05::

  import numpy as np
  from pyimcom.meta import distortimage
  mosaic = distortimage.MetaMosaic('/fs/scratch/PCON0003/cond0007/anl-run-out/prod_H_24_13_map.fits', verbose=True)
  rot = 30*np.pi/180.
  im = mosaic.shearimage(3200, jac=[[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]], psfgrow=1.05, oversamp=1.)
  distortimage.shearimage_to_fits(im, 'xdist.fits', overwrite=True)

This will work either with the original ``.fits`` files, or with the ``.cpr.fits.gz`` files.

Detailed usage information
****************************

There are several options available and points to keep in mind when working with the ``distortimage.MetaMosaic`` class.

Constructor
============

Construction of a mosaic is in principle quite simple: it is built from a PyIMCOM output file, with the optional ``verbose`` keyword::

   mosaic = distortimage.MetaMosaic('mymosaic_24_13.fits', verbose=True)

The above call will make a ``MetaMosaic`` from block x=24, y=13. This contains information from the group of 3x3 blocks centered at the indicated block, i.e., the above call will also load data from (23,12), (23,13), (23,14), (24,12), (24,14), (25,12), (25,13), and (25,14) in  addition to (25,13). If a block is at the edge of the mosaic, then fewer than 9 blocks may be loaded.

The constructor can work with either the raw ``.fits`` files, or with the compressed ``.cpr.fits.gz`` files (see `PyIMCOM compression <../compress/compress_README.rst>`_).

The configuration that generated an object is accessible as an attribute, e.g., you may get the pixel scale in the coated image by asking for ``mosaic.cfg.dtheta``, or the additional layers from ``mosaic.cfg.extrainput``.

Masking an image
==================

*under construction*

Shearing an image
==================

A sheared image dictionary is generated from the ``shearimage`` method, e.g.::

  im = mosaic.shearimage(3200,
    jac=[[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]],
    psfgrow=1.05, oversamp=1.)

The arguments provided are:

+--------------------------+------------------------------------------------+
| Argument                 |        Description                             |
+==========================+================================================+
| ``N``, *integer*         | Size of the output image (shape will be (N,N)).|
| (required).              | This is in output (post-shearing) pixels.      |
+--------------------------+------------------------------------------------+
| ``jac``, *2x2 matrix* or | Shear matrix to apply. This should be a numpy  |
| None (default)           | array of shape (2,2); if None is provided, this|
|                          | defaults to the identity (no transformation).  |
|                          | The convention is that if this matrix is A,    |
|                          | then the input and output coordinates are      |
|                          | related by  x_in[i] = sum_j A[i,j] x_out[j].   |
|                          | So to shear by gamma_1 = +0.01 and kappa=0, you|
|                          | should use jac=[[0.99,0.],[0.,1.01]].          |
+--------------------------+------------------------------------------------+
| ``psfgrow``, *float*     | Factor by which to grow the PSF. Usually in    |
| (default=1.)             | meta-type operations, you want this to be a    |
|                          | little bit bigger than 1. To be "stable", i.e.,|
|                          | to not be de-convolving the PSF on any axis,   |
|                          | you should make this at least as large as the  |
|                          | reciprocal of the smallest singular value of   |
|                          | jac.                                           |
+--------------------------+------------------------------------------------+
| ``oversamp``, *float*    | Factor by which to up-sample relative to the   |
| (default=1.)             | pixel scale of the coated image. The default is|
|                          | 1. (do not change the pixel scale), but there  |
|                          | are cases where you might choose to increase   |
|                          | the sampling. For example, if your analysis    |
|                          | code uses lower-order interpolation, you should|
|                          | make sure that the image you generate is       |
|                          | **very** oversampled. If the coadded image is  |
|                          | 2x oversampled relative to the coadded PSF, and|
|                          | you want a 6x oversampled image for your       |
|                          | analysis, you can set oversamp=3.              |
+--------------------------+------------------------------------------------+
| ``fidelity_min``, *float*| Minimum fidelity (in dB), below which a pixel  |
| (default=30.)            | will be masked before shearing.                |
+--------------------------+------------------------------------------------+
| ``Rsearch``, *float*.    | Search radius (in coadded pixels) when building|
| (default=6.)             | the interpolation kernel.                      |
+--------------------------+------------------------------------------------+
| ``verbose``, *bool*      | Print extra outputs.                           |
| (default=False)          |                                                |
+--------------------------+------------------------------------------------+


The output dictionary has the following keys:

- ``image``: The image as a 3D numpy array (layer, y, x)
- ``mask``: The mask as a 2D numpy array (y,x)
- ``wcs``: The WCS of the output image (if you have implemented a shear, then the WCS is also sheared: the RA and Dec of an object in the WCS corresponds to its true position)
- ``pars``: A dictionary of parameters associated with the sheared image (including provenance data and the applied shear)
- ``layers``: The names of the layers (copied from the ``extrainput`` used to generate the mosaic.)

Writing to a file
====================

There is a simple function to write a sheared image dictionary to disk::

  pyimcom.meta.shearimage_to_fits(im, fname, layers=None, overwrite=False)

Here:

- ``im`` is the dictionary containing the image;
- ``fname`` is the file to write to (should have a .fits or .fits.gz extension);
- ``layers`` is either a list of the numerical indices of the layers to write or None (which writes *all* the layers);
- ``overwrite`` is a boolean indicating whether to overwrite the file.

The output fits file contains the 3D image cube (primary HDU); and the 2D mask (``MASK`` HDU). The primary HDU contains the WCS and the parameters (``im.pars``, re-written as FITS keyword/value pairs).
