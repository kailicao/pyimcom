Input format options
######################

The PyIMCOM framework supports several input format options. We generally add more when simulations or data products become available in a new format. The current choices are:

* ``dc2sim``: Data Challenge 2 simulation, as used in `Troxel et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.2801T/abstract>`_.

* ``anlsim``: OpenUniverse 2024 simulation, as used in `OpenUniverse (2025) <https://arxiv.org/abs/2501.05632>`_.

* ``L2_2506``: Roman Level 2-like format as of summer 2025, with some exceptions.

Some more specifications on input files are below. Note that in file names, ``filter`` is a 4-character code (e.g., ``H158``), and ``obsid`` and ``sca`` are integers without leading 0's.

Science images & World Coordinate System
===========================================

The input science images are as specified in the ``INDATA`` configuration keyword. The file names/formats depend on the input format:

* ``dc2sim``:

  - File name: ``simple/dc2_{filter}_{obsid}_{sca}.fits``

  - Format: FITS file, 2D array in ``'SCI''`` HDU, in electrons per pixel per exposure. The sky level is specified as the ``'SKY_MEAN'`` keyword in the ``'SCI'`` header. (In this simulation, it was taken to be constant.) The WCS is given in the FITS header; TAN-SIP projection is supported.

* ``anlsim``:

  - File name: ``simple/Roman_WAS_simple_model_{filter}_{obsid}_{sca}.fits``

  - Format: FITS file, 2D array in ``'SCI''`` HDU, in electrons per pixel per exposure. The sky level is specified as the ``'SKY_MEAN'`` keyword in the ``'SCI'`` header. (In this simulation, it was taken to be constant.) The WCS is given in the FITS header; TAN-SIP projection is supported.

* ``L2_2506``:

  - File name: ``sim_L2_{filter}_{obsid}_{sca}.asdf``

  - Format: ASDF file, 2D array in the ``['roman']['data']`` branch of the tree. The WCS is a ``gwcs`` object in the ``['roman']['meta']['wcs']`` branch of the tree.

The file name broker for science images is the ``pyimcom.layer._get_sca_imagefile`` function.

Point spread functions
==========================

The PSF format is specified in the ``'INPSF'`` configuration keyword. The file names/format on the input format:

* ``dc2sim``:

  - File name: ``dc2_psf_{obsid}.fits``

  - Format: N x N 2D FITS images, oversampled, pixel tophat not included. The PSFs at the different SCAs are in different HDUs (1 ... 18). The PSF is centered at the array center (half integer pixel value if N is even). There is no support for spatial variation of the PSF across the SCA in this format.

* ``anlsim`` and ``L2_2506``:

  - File name: ``psf_polyfit_{obsid}.fits``

  - Format: Legendre polynomial cube, Ncoef x N x N 3D FITS image, oversampled, pixel tophat not included. The PSFs at the different SCAs are in different HDUs (1 ... 18). The PSF is centered at the array center (half integer pixel value if N is even). The spatial variation across the SCA is described in terms of 2D Legendre polynomials; the ``data[i,:,:]`` slice of the image corresponds to the ``i`` th basis function.

    The basis functions are 2D Legendre polynomials are defined in terms of the re-scaled SCA coordinates ``u=(x-2044.5)/2044, v=(y-2044.5)/2044``. There are Ncoef=(p+1)**2 coefficients, in the order ``P_0(u)P_0(v) ... P_p(u)P_0(v), P_0(u)P_1(v) ...P_p(u)P_1(v), ... P_0(u)P_p(v) ... P_p(u)P_p(v)``.

Reading PSFs (including selecting the file name and format) occurs in the ``pyimcom.coadd.InImage.get_psf_pos`` function.

Laboratory noise realizations
=================================

It is possible to feed laboratory noise images into PyIMCOM as additional layers using the ``'labnoise'`` option in ``EXTRAINPUT``. The file names/format on the input format:

* ``dc2sim``, ``anlsim``:

  - File name: ``labnoise/slope_{obsid}_{sca}.fits``

  - Format: 2D FITS image in the primary HDU, either 4088x4088 or 4096x4096 (including reference pixels, which are stripped).

* ``L2_2506``:

  - File name: ``labnoise/slope_{obsid}_{sca}.fits``

  - Format: 2D FITS image in the primary HDU, either 4088x4088 or 4096x4096 (including reference pixels, which are stripped). The input image is in the *Detector* frame, but ``L2_2506`` data products are in the *Science* frame, so the flip is applied when the data is read in.

The file name broker for lab noise images is the ``pyimcom.layer._get_sca_imagefile`` function.

Simulated noise realizations
================================

In the early days of PyIMCOM development, simulated (white and 1/f) noise realizations were generated internally. This is still possible, but it is *also* now possible to feed in noise images from an external simulator (most likely ``romanimpreprocess``). The file names/format on the input format:

* ``L2_2506``:

  - File name: ``sim_L2_{filter}_{obsid}_{sca}_noise.asdf``

  - Format: ASDF file. The tree contains branch ``['config']['NOISE']['LAYER']``, which is a list of the noise layers generated, e.g., ``['RS2C1', 'RS2C2', 'RS0']``. That list is composed of unique strings used by ``romanimpreprocess``. The noise data itself is a 3D image the ``['noise']`` branch, so e.g. the 4088 x 4088 noise image for 'RS0' (element 2 in the list) is in ``tree['noise'][2,:,:]``.

The file name broker for lab noise images is the ``pyimcom.layer._get_sca_imagefile`` function.

Masks
==========

Starting with the summer 2025 run, we are able to pass an externally generated mask into PyIMCOM. The file names/format on the input format:

* ``L2_2506``:

  - File name: ``sim_L2_{filter}_{obsid}_{sca}_mask.fits``

  - Format: 4088 x 4088 FITS image in the ``'MASK'`` HDU; a non-zero value indicates that that pixel should be masked.

The file name broker for mask images is the ``pyimcom.layer._get_sca_imagefile`` function, and the implementation of reading it is in ``pyimcom.layer.Mask.load_mask_from_maskfile``.
