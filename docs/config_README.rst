PyIMCOM Configuration files
###########################

This is a guide to the use of PyIMCOM configuration files.

Managing the configuration
**************************

Configurations can be built, edited, and saved.

Building a configuration
========================

PyIMCOM runs are controlled by the ``config.Config`` class.
There are several ways to set up the configuration:

- The most common way is to read from a JSON file: you may run something like::

   cfg = config.Config('my_config.json')

- A string that is not a valid path name but could be interpreted as a JSON file can also be used as an argument for the configuration::

   str = '{"OBSFILE": "myobs.fits",\n'
   str += '"INDATA": ["input_data", "anlsim"],\n'
   ...
   str += '"SMAX": 0.5}\n'
   cfg = config.Config(str)

- You may also build a configuration file from an IMCOM output file, thus allowing you to reconstruct the configuration that produced it::

   cfg = config.Config('test_F_00_00.fits', inmode='block')

  **Warning**: If you do this, it will reconstruct the *entire* configuration file, including the input and output file paths on the system that generated the file. If you simply want to do this so that the grid spacings, choices of input layers, etc. are accessible from your process, this is fine. But if you are going to run the co-adds, you will want to edit the configuration file so that the file paths correspond to your platform, and (if on the same platform) you don't unintentionally overwrite the outputs.

- You can input a configuration in interactive mode so that the user types the configuration directly in the terminal::

   cfg = config.Config(None)

  This is useful for testing, but we don't expect many users will want to do production runs this way.

Editing a configuration
=======================

A configuration file can be edited by setting an element of the configuration. This might be useful if you want to run a configuration that was the same as that which generated another file, but with the inputs/outputs changed. It is also useful if there are paths that are only determined at run time. For example, at the Ohio Supercomputer Center a job gets access to the local disk on each node but the path name is provided as an environment variable ``$TMPDIR``; in this case, you could set PyIMCOM's temporary file storage via::

   cfg.tempfile = os.getenv('TMPDIR') + '/temp'

If you change the numerical parameters, then you will have to update the derived parameters via a call to the configuration file::

   cfg()

Saving a configuration
======================

The usual way of saving a configuration is with the ``to_file`` method. For example, to save to a file::

   cfg.to_file('my_config.json')

or to save to a string and print to the terminal::

   str = cfg.to_file(None)
   print(str)

Contents of the configuration file
**********************************

The configuration file has 8 sections, which we describe in turn. Since the configuration is stored internally as a Python dictionary it does not matter whether the keywords in the input file are ordered by section or interspersed. Optional arguments are marked with a \*.

Input files
===========

This section covers where PyIMCOM looks for input data.

OBSFILE: The table of observations
--------------------------------------------------

PyIMCOM needs a table of all observations that it should search over. This is in the format of a FITS binary table. The path to the file is determined by the ``OBSFILE`` keyword::

   "OBSFILE": "/users/PCON0003/cond0007/imcom/pyimcom/anlsim/Roman_WAS_obseq_11_1_23.fits"

The current format for the FITS file is that HDU #1 is a binary table with columns including:

- ``date`` : Modified Julian Date of the observation
- ``exptime`` : Exposure time (in seconds)
- ``ra`` : Right ascension of the WFI origin (in degrees, J2000)
- ``dec`` : Declination of the WFI origin (in degrees, J2000)
- ``pa`` : Position angle of the WFI +Y direction (in degrees, J2000)
- ``filter`` : Filter code, integer. Convention is 0 (W146), 1 (F184), 2 (H158), 3 (J129), 4 (Y106), 5 (Z087), 6 (R062), 7 (prism), 8 (dark), 9 (grism), 10 (K213). Note that PyIMCOM is for imaging so 7, 8, and 9 will be ignored.

The coordinate and filter information is needed for PyIMCOM to select the appropriate exposures for use in each block. (It is not used for precision alignment of the images: for that, PyIMCOM uses the WCS provided in the input files.) The date information is not used right now, but it could be used in the future to build co-adds in deep fields that correspond to certain ranges of epochs.

INDATA: The input observations
-----------------------------------

This has two entries containing the directory to find input images and the format::

   "INDATA": [
      "/fs/scratch/PCON0003/cond0007/anl-run-in-prod/stitch",
      "anlsim"]

PyIMCOM looks for any files in the input directory with the correct format. If your input files are spread across multiple directories or file systems (there is a lot of data!), then for now we recommend making symbolic links.

The second value is the format. Right now the valid formats are:

- ``dc2_imsim`` : The Roman + Rubin Data Challenge 2 simulation format (`Troxel et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.2044T/abstract>`_)

- ``anlsim`` : The Open Universe "simple" image format (`IPAC site <https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/overview.html>`_; `Open Universe paper <https://ui.adsabs.harvard.edu/abs/2025arXiv250105632O/abstract>`_)

More input formats will be added as needed to support further simulations and Roman data analysis.


FILTER: Filter choice
---------------------------

This is a simple integer for the filter to coadd::

   "FILTER": 4

Convention is 0 (W146), 1 (F184), 2 (H158), 3 (J129), 4 (Y106), 5 (Z087), 6 (R062), and 10 (K213). (7, 8, and 9 are codes for the prism, dark, and grism, and are not supported.)

INPSF: Input PSF files
-----------------------------

This is a list containing a directory, PSF format, and oversampling factor::

    "INPSF": [
       "/fs/scratch/PCON0003/cond0007/anl-run-in-prod/psf_vlarge",
       "anlsim",
       8]

The above example looks for PSFs in the directory ``/fs/scratch/PCON0003/cond0007/anl-run-in-prod/psf_vlarge``; has PSF format type ``anlsim``; and the PSF is oversampled relative to native pixels by a factor of 8.

The valid PSF formats are the same as the input data formats in the `INDATA <INDATA: The input observations>`_ keyword. Most of the time, you will want to use the same format as in ``INDATA``, but this is not enforced.

PSFSPLIT: Splitting the PSF\*
-------------------------------

*Experimental feature; optional*

This keyword is optional (it defaults to ``None``). If provided, it means that the ``pyimcom.splitpsf`` module has been used to split the PSF into long- and short-range parts, which will (eventually) allow for iterative cleaning of the long-range part of the PSF. An example is::

    "PSFSPLIT": [6.0, 10.0, 0.01]

This directs ``pyimcom.splitpsf`` to split the PSF so that the short-range part goes smoothly to zero from 6 to 10 pixels, with a regularization parameter for the long-range part of epsilon=0.01.

**Comment**: The PSF splitting tool is under development: it runs, but the iterative cleaning of the long-range PSF is not yet implemented. So if you use the current version, you won't achieve the improvements that we ultimately expect.

Masks and layers
==================

PMASK: Permanent mask\*
------------------------

This provides a permanent mask file::

       "PMASK": "/users/PCON0003/cond0007/imcom/pyimcom/anlsim/permanent_mask_ft_231228.fits"

The mask is a FITS file with an integer-type primary HDU consisting of a 18x4088x4088 (18 SCAs, 4088 rows, 4088 columns) data cube. Nonzero values indicate that a pixel should be permanently masked.

If not provided, defaults to no permanent mask.

CMASK: Cosmic ray mask fraction\*
------------------------------------

This specifies a cosmic ray rate per pixel that should be randomly masked::

    "CMASK": 0.00077

This is useful in simulations to explore what a cosmic ray mask does to downstream processing (e.g., shape measurements of galaxies where some input pixels were masked). It is not something that we expect to apply to real data (since then we will have a tool that masks the pixels that were really hit).

The simulation masks a 3x3 region around each hit. The random number generator is configured to produce the same masks if the simulation is re-run or if another mosaic is built that uses the same SCA.

The default (0) is to not implement a cosmic ray mask.

EXTRAINPUT: Additional layers\*
-----------------------------------

This allows the user to specify a list of additional layers (suites of input images) to run through PyIMCOM with the same coaddition matrix **T**. The "science" layer is layer 0, and if N additional layers are specified then the output FITS images are data cubes with N+1 frames along axis -3. The defailt is no additional layers. An example usage is::

    "EXTRAINPUT": [
        "labnoise",
        "gsstar14",
        "nstar14,2e5,86,3",
        "gstrstar14",
        "gsfdstar14,0.05",
        "gsext14,seed=100",
        "gsext14,seed=100,shear=.02:0",
        "gsext14,seed=100,shear=-.01:0.017320508075688773",
        "gsext14,seed=100,shear=-.01:-0.017320508075688773",
        "1fnoise9",
        "whitenoise10",
        "whitenoise11",
        "whitenoise12",
        "whitenoise13"
    ]

In this example, the "science" image (always present) is layer 0; "labnoise" is layer 1; "gsstar14" is layer 2, etc.

The ``pyimcom.layers`` module contains instructions for building each of the different layers, and additional options will be added in the future. The currently supported options include:

- ``truth`` : This layer is the true (no noise or saturation, but including PSF) image, if supported by that input format. Clearly this is only available for the simulations.

- ``whitenoise`` n : This layer generates white noise. The trailing integer n controls the random number generator seed. If the same n is used in another mosaic, then each observation ID + SCA will produce the same noise realization. The normalization is mean 0 and variance 1 in each pixel; if you are interested in other normalizations, you can appropriately re-scale the output.

- ``1fnoise`` n : This layer generates 1/f noise in each readout channel, with striping along the fast-read direction. The trailing integer n controls the random number generator seed. If the same n is used in another mosaic, then each observation ID + SCA will produce the same noise realization. The normalization is mean 0 and variance 1 per logarithmic range in frequency; if you are interested in other normalizations, you can appropriately re-scale the output.

- ``labnoise`` : This layer is a "real" dark from ground testing that has been matched to the corresponding observation ID + SCA. See `Laliotis et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024PASP..136l4506L/abstract>`_.

- ``skyerr`` : This is the realization of sky noise provided by the simulation (only available for the ``dc2_imsim`` input format).

- ``cstar`` n : This is a grid of ideal point sources with unit flux at HEALPix resolution n (i.e., 12\*4^n pixels). They are drawn according to the PSF provided using PyIMCOM's internal routines.

- ``nstar`` n,f,s,p : This is a grid of noisy point sources at HEALPix resolution n (i.e., 12\*4^n pixels). The flux (in electrons) is given by f; the sky background (which is included in the Poisson noise but subtracted from the layer) is given by s; and the random number generator seed is p. So ``nstar14,2e5,86,3`` will generate stars on a resolution 14 grid, with flux 2e5 electrons, with Poisson noise from 86 electrons per pixel, and use random seed p=3.

- ``gsstar`` n : This is a grid of ideal point sources with unit flux at HEALPix resolution n (i.e., 12\*4^n pixels). They are drawn using GalSim (as opposed to ``cstar`` n, which does the same thing but internally).

- ``gstrstar`` n : This is a grid of transient point sources with unit flux at HEALPix resolution n (i.e., 12\*4^n pixels), drawn using GalSim. It is used to test what happens in PyIMCOM if a source is present in one pass but not the other. Point sources in even-numbered pixels are drawn if WFI +Y is pointed north (±90°) and those in odd-numbered pixels are drawn if WFI +Y is pointed south (±90°).

- ``gsfdstar`` n,A : This is a grid of point sources with field-dependent flux at HEALPix resolution n (i.e., 12\*4^n pixels). The flux is 1+A(x^2+y^2)/R^2, where R is the radius of the focal plane and (x,y) are the focal plane coordinates. So for example, ``gsfdstar14,0.05`` will generate stars on a resolution 14 grid, with flux ranging from 1 at the field center up to 1.05 at the corners. This was used as a test of how large-scale flat field errors or field-dependent bandpass terms multiplying an object's SED would impact PyIMCOM coadded images.

- ``gsext`` n\+ : This is a grid of extended objects drawn by GalSim at HEALPix resolution n (i.e., 12\*4^n pixels). It can contain multiple arguments. The current version makes exponential profile objects, with half-light radius logarithmically distributed between 0.125 and 0.5 arcsec, and shapes (g_1,g_2) uniformly distributed in the circle \|g\|\<0.5. The arguments are comma-delimited, e.g., ``gsext14,seed=100,shear=.02:0``. Legal arguments include:

  - ``seed=`` p : The random seed p to use (you can generate the same galaxies in multiple layers by using the same seed).

  - ``rot=`` theta : Rotate by the angle theta (in degrees, counterclockwise as seen by the observer) after drawing it but before any shear is applied. The ``rot=90`` option is commonly used in simulations to partially cancel shape noise.

  - ``shear=`` g_1:g_2 : Shears the galaxy by the indicated amount, in coordinates where g_1 is the East-West direction and g_2 is the Northeast-Southwest direction.

LABNOISETHRESHOLD: Mask based on a laboratory dark\*
-------------------------------------------------------

*Optional; only valid if* ``labnoise`` *is in one of the layers.*

This masks a pixel if the laboratory noise field is above some threshold. It is useful for studying the impact of correlated noise but removing large features such as hot pixels. The value specified is the clipping threshold::

   "LABNOISETHRESHOLD": 3.0

What area to coadd
===================

This section contains geometrical information on the output mosaic, including the information needed to build the output WCS. The stereographic (``STG``) projection around the mosaic center is used, since it has zero shear distortion and smaller magnification distortion than the commonly used gnomonic (``TAN``) projection.

CTR: Projection center
------------------------

This gives the RA (first) and Dec (second) of the projection center of the mosaic (in degrees, J2000)::

    "CTR": [9.55, -44.1]

LONPOLE: Rotating the mosaic\*
---------------------------------

*Optional; default is North pointing up*

This is the same as the ``LONPOLE`` FITS keyword (see the `standard <https://ui.adsabs.harvard.edu/abs/2002A%26A...395.1077C/abstract>`_). A value of 180° corresponds to North being up; 270° corresponds to East being up; 0° corresponds to South being up; and 90° corresponds to West being up. Other values are allowed, for example the following has the same center as the case above, but with "up" being 60° East of North::

    "LONPOLE": 240.0

BLOCK: Size of the mosaic
------------------------------

The mosaic is a square array of blocks. So to make a 12x12 array, we use::

    "BLOCK": 12

The projection center is the same as the mosaic center, so here it would be at the corners of blocks (5,5), (6,5), (5,6), and (6,6).

OUTSIZE: Pixel, postage stamp, and block dimensions
----------------------------------------------------

This controls the size of a block; for example: ::

    "OUTSIZE": [
        72,
        40,
        0.0425]

This case has an output pixel size of 0.0425 arcsec. Each postage stamp to coadd is 40x40 output pixels, so has a size of 40x0.0425 = 1.7 arcsec. Then each block is 72x72 postage stamps, so the block size is 72x17=122.4 arcsec.

**Important**: Because of the way PSF computations are saved (every 2x2 postage stamps), the number of postage stamps per block (72 in the above example) must be even.

More about postage stamps
==============================

FADE: Transition pixels\*
----------------------------

This controls the number of transition pixels around each postage stamp where it "fades away" while the next postage stamp "fades in". This ensures a smooth transition from one postage stamp to the next, even though they are constructed from different sets of input pixels. So for example, to set the number of transition pixels around each postage stamp to 2::

       "FADE": 2

Specifying this is optional; it defaults to 3.

A truncated sine function is used for the weights, so that the total weight is always 1.

PAD: Padding postage stamps\*
-------------------------------

*Optional; no padding if not set.*

This tells PyIMCOM to compute an additional set of postage stamps around each block. So to compute 1 postage stamp around the edge::

      "PAD": 1

For most applications where you may be interested in sources near the edge of a block, having a padding postage stamp is recommended. If there are n\_2 pixels per postage stamps, ``FADE`` =k, and ``PAD`` =P, then there will be 2(kn\_2-P) output pixels that are exactly the same between one block and the next. 

PADSIDES: Which padding stamps to compute\*
----------------------------------------------

*Optional; defaults to "auto".*

This tells PyIMCOM which padding stamps to compute. If you want to compute a stand-alone block without doing any additional post-processing (this is probably most applications), you should use ``"all"``::

   "PADSIDES": "all"

If you do not select ``"all"``, then unfinished postage stamps will be left as zeroes and you will want to fill them in later.
Other options include:

- ``"auto"`` :  Compute only the "new" postage stamps (does not re-compute postage stamps that are in other blocks, so that they can be copied in post-processing).

- ``"none"`` : Do not compute the padding postage stamps.

- ``[BTLR]+`` : You may specify bottom, top, left, or right with a string containing one or more of these characters (e.g., ``"BRL"`` to compute bottom, right, and left padding stamps, but not the top).

STOP: Compute only a portion of the block\*
----------------------------------------------

*Optional*

If specified and positive, halts coaddition after the specified number of postage stamps have been coadded (``STOP=0`` is ignored). So the following will compute only the first 148 postage stamps and then stop::

    "STOP": 148

This will give you an output block that has the bottom filled in, but then the rest will be empty. This is mostly useful during de-bugging: you might want to run only, say, 1/4 of a block so that your modification -> re-run -> analysis -> next modification cycle is shorter.

What and where to output
===========================

OUTMAPS: Which maps to save\*
-------------------------------

*Optional; default is to save everything.*

This is a string with a capital letter for each possible output we want to save, e.g. to save U, S, T, and N maps::

   "OUTMAPS": "USTN"

The outputs choices (and the names of the extension HDUs they generate in the coadded image FITS files) are rescaled versions of:

- ``U`` [``FIDELITY``]: This is the PSF leakage map, the square norm of the difference between the PSF of an output pixel and the target PSF: 1/(U_alpha/C).

- ``S`` [``SIGMA``]: The noise amplification map, or sum of squares of weights of each input pixel that went into each output pixel: 1/\Sigma_\alpha.

- ``K`` [``KAPPA``]: The Lagrange multiplier 1/\kappa_\alpha used for that pixel that specifies where on the Pareto front PyIMCOM chose. Larger values reduce noise at the expense of more PSF leakage.

- ``T`` [``INWTSUM``]: The total weight over input pixels, ∑_i T_{alpha i}. Expect this to be close to but not exactly 1.

- ``N`` [``EFFCOVER``]: The effective coverage n\_\{eff,alpha\} as defined in `Cao et al. (2025) <https://ui.adsabs.harvard.edu/abs/2024arXiv241005442C/abstract>`_. This is like a number of exposures, but because of partial weights it need not be an integer.

Specific conventions for U, S, K, and T are as described in `Rowe et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJ...741...46R/abstract>`_.

**Convention note**: To save space, the output images are stored as 16-bit integers on a logarithmic scale. The ``UNIT`` keyword in the output images specifies the scale in bels with an SI prefix: for example, if it is ``0.2mB``, then the quantity x is stored as (log_{10}x)/(0.2*1e-3). You can get the original quantity back with the code::

   from pyimcom.diagnostics.outimage_utils.helper import HDU_to_bels
   x = 10**(HDU_to_bels(hdu)*hdu.data)

OUT: Output file location
---------------------------

This is a stem for the output coadded image file locations. An example would be::

    "OUT": "/fs/scratch/PCON0003/cond0007/itertest2-out/itertest2_F"

which generates output file ``/fs/scratch/PCON0003/cond0007/itertest2-out/itertest2_F_02_00.fits`` for block (2,0).

TEMPFILE: Temporary storage for a block\*
----------------------------------------------

*Optional; does not use virtual memory if not given.*

This is a stem for temporary files during coaddition (e.g., virtual memory). This can be specified in the configuration file::

   "TEMPFILE": "/tmp/my_pyimcom_run"

On some platforms, including the Ohio Supercomputer Center, a process only finds out the path for local storage on its node at runtime, so you can't use the hard-coded ``TEMPFILE`` in the configuration. In that case, the script that calls PyIMCOM should, after loading a configuration, find out which directory it is supposed to use. On OSC, this is provided by the ``$TMPDIR`` environment variable, so after loading the configuration in Python you would modify it::

   config_file = sys.argv[1]
   cfg = Config(config_file)
   cfg.tempfile = os.getenv('TMPDIR') + '/temp'

INLAYERCACHE: Temporary storage for a mosaic
--------------------------------------------------

This is also a stem for file storage::

   "INLAYERCACHE": "/fs/scratch/PCON0003/cond0007/itertest2-out/cache/in_F"

The difference is that this is common to the *whole mosaic*, and in particular it should be on a disk (usually a scratch disk) that remains after a process finishes and is only cleared after the entire mosaic is finished. It is primarily used to store input layers so that they do not need to be re-computed. If specified, when a block draws the layers corresponding to a given observation ID/SCA, it saves those in the ``INLAYERCACHE``. For example, in the above case, layers drawn for  observation ID 14746, SCA 16 are stored in a FITS cube at ``/fs/scratch/PCON0003/cond0007/itertest2-out/cache/in_F_00014746_16.fits``, and the pixel mask is stored at ``/fs/scratch/PCON0003/cond0007/itertest2-out/cache/in_F_00014746_16_mask.fits``. Later blocks will detect that these files have been generated, and load them instead of re-generating them.

**Comment**: The ``INLAYERCACHE`` stem is also being used to store data for some experimental features, so we expect to make more use of it in the future. Therefore even though PyIMCOM will run without it right now, we expect that most users will need it in the future and we recommend treating it as required.

Target output PSFs
=====================

NOUT: Number of output PSFs\*
------------------------------------------------

*Deprecated*

The number of output PSFs to generate simultaneously. The default is 1. So nothing will change if you write::

    "NOUT": 1

**Comment**: When this option was first introduced, the linear algebra engine in PyIMCOM required an eigendecomposition of the **A** matrix (see `Rowe et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011ApJ...741...46R/abstract>`_). This was extremely slow, but once it was done it was relatively fast to generate multiple output PSFs at the same time. In the Cholesky decomposition approach, the advantage of doing multiple PSFs at once is much smaller, and the memory burden of handling all the outputs simultaneously has been found to slow things down. It is also not compatible (for math reasons, not coding reasons) with the PSF splitting that we plan to implement for Roman.

OUTPSF: Output PSF type
----------------------------

This sets the type of target output PSF::

   "OUTPSF": "GAUSSIAN"

The options are:

- ``GAUSSIAN`` : This output PSF is a simple Gaussian, with a 1 sigma width set by the ``EXTRASMOOTH`` keyword. (This is the baseline for the Roman PIT analysis.)

- ``AIRYOBSC`` : An obscured Airy PSF (diffraction pattern from an annular aperture), convolved with optional extra smoothing. (This was used in `Hirata et al. 2024 <https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.2533H/abstract>`_.)

- ``AIRYUNOSBC`` : An unobscured Airy PSF (diffraction pattern from a circular aperture), convolved with optional extra smoothing.

In the latter two cases, the ``EXTRASMOOTH`` keyword determines the extra smoothing.

EXTRASMOOTH: Output PSF Gaussian component
-----------------------------------------------

This sets the Gaussian width (if ``OUTPSF`` is ``GAUSSIAN``) or the extra smoothing (if ``OUTPSF`` is one of the diffractive options). The width is the 1 sigma width (or rms per axis) in units of undistorted input pixels (i.e., 0.11 arcsec). So for example::

   "EXTRASMOOTH": 0.9767200703312219

will give a Gaussian with a 1 sigma width of 0.97672x0.11 = 0.10744 arcsec.

Building linear systems
==========================

NPIXPSF: Size of PSF inner product arrays\*
----------------------------------------------

*Optional; default is 48.*

This is the size of the arrays used to compute PSF inner products in native pixels (should be an even integer). So to set this to 42 native pixels or 42x0.11 = 4.64 arcsec, you may write::

    "NPIXPSF": 42

The default of 48 is recommended for most Roman uses for now based on experience with the DC2 and OpenUniverse simulations.

**Comment**: If you use the PSF splitting, then we know rigorously that ``NPIXPSF`` \>4(1+alpha)R\_{out}, where R\_{out} is the outer radius of the PSF and alpha is the geometric distortion (i.e., true pixel size is 0.11(1+alpha) arcsec), is sufficient. So this will be the plan for production runs on Roman data.

PSFCIRC: Apply a circular cutout to the PSF\*
-----------------------------------------------

*Experimental feature; default=False*


PSFNORM: Rescale thet normalization of the PSF\*
--------------------------------------------------

*Experimental feature; default=False*

AMPPEN: Apply an additional penalty in the PSF leakage for low-frequency modes in the PSF that do not match the target\*
---------------------------------------------------------------------------------------------------------------------------------------------

*Deprecated, default = [0,0]*

This was used in the first Roman IMCOM run `Hirata et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.2533H/abstract>`_. We have since assessed that it is not needed.

FLATPEN: Apply an additional penalty in the PSF leakage if the input images do not receive equal weight\*
------------------------------------------------------------------------------------------------------------------------

*Deprecated; default = 0*

This was used in the first Roman IMCOM run `Hirata et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.2533H/abstract>`_. We have since assessed that it is not needed.

INPAD: Selection of input pixels
--------------------------------------

This sets the acceptance radius (in arcsec) around the postage stamp to select input pixels. See Fig. 4c of `Hirata et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.2533H/abstract>`_. It is a single keyword/value::

    "INPAD": 1.05

Solving linear systems
===========================

LAKERNEL: Setting the linear algebra kernel
-----------------------------------------------------------------

This sets the linear algebra kernel used to solve for the coaddition weights **T** in terms of the system matrices **A** and **B**::

    "LAKERNEL": "Cholesky"

The choices are (see `Cao et al. (2025) <https://ui.adsabs.harvard.edu/abs/2024arXiv241005442C/abstract>`_):

- ``Eigen`` : This does an eigendecomposition of **A**, as initially suggested in `Rowe et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJ...741...46R/abstract>`_.

- ``Cholesky`` : This does a Cholesky decomposition of **A**, and is much faster if only a few values of kappa are needed (or if one can interpolate between them, see `Cao et al. 2025 <https://ui.adsabs.harvard.edu/abs/2024arXiv241005442C/abstract>`_ §3.1).

- ``Iterative`` : This does a conjugate-gradient solution of the linear system rather than a decomposition of **A**. It has intermediate speed.

- ``Empirical`` : This is a fast approximation that does not actually solve the linear system. It does not reach PyIMCOM's potential for accuracy, but since it is fast it can be useful for testing whether your interfaces to PyIMCOM are working.

The ``Iterative`` kernel uses two additional keywords (these are the default values)::

   "ITERTOL": 1.5e-3
   "ITERMAX": 30

The ``Empirical`` kernel takes an additional keyword that allows one to turn off the U_alpha/C and Sigma_alpha computations (default is false, but turning this on makes the code very fast)::

   "EMPIRNQC": false

KAPPAC: Lagrange multiplier array
-----------------------------------------------------------------

This sets the grid of Lagrange multipliers for Cholesky and Iterative methods. It is a list in ascending order::

       "KAPPAC": [0.0002]

If one value is given (as in the above case), a single choice of kappa_alpha is used. If multiple values are given, then interpolation is used to approximate the Pareto front (see `Cao et al. 2025 <https://ui.adsabs.harvard.edu/abs/2024arXiv241005442C/abstract>`_ §3.1).

UCMIN and SMAX: Bounding the search space of the Lagrange multiplier
-----------------------------------------------------------------------

These values describe the bounds in leakage U_alpha/C and noise Sigma_alpha metrics::

    "UCMIN": 1e-06,
    "SMAX": 0.5

The "normal" behavior of PyIMCOM (except in Empirical mode, or if only a single value is given in ``KAPPAC``) is to find the best possible leakage performance subject to the noise constraint; but if the noise can be reduced below ``SMAX`` at U_alpha/C = ``UCMIN`` then it switches to this behavior.

