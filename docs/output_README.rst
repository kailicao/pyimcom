Reading PyIMCOM output files
############################

There are several options for reading PyIMCOM output files. From "lowest" to "highest" level interaction:

* FITS readers: The PyIMCOM output files conform to the FITS standard (including WCS headers), and can be read with any standard FITS reader.

* Compressed files have their non-science layers scrambled. They can be read using the ``pyimcom.compress.compressutils.ReadFile`` utility, which can be used just like ``astropy.io.fits.open`` (including the context management)::

    from pyimcom.compress.compressutils import ReadFile
    # here fout is the file name that we want to read
    with ReadFile(fout) as f:
       x = f[0].data
       ...

  The other tools listed below are built on top of ``ReadFile``.

* The ``pyimcom.analysis.OutImage`` method builds one output block from the FITS file::

    from pyimcom.analysis import OutImage
    # here fout is the file name that we want to read
    my_block = OutImage(fout)
    # extracts a layer: 'SCI' is the science image, or you can use other layer names
    sci_image = im.get_coadded_layer('SCI')
    # extracts a metadata map
    fidelity_map = im.get_output_map('FIDELITY') # options are: 'FIDELITY', 'SIGMA', 'KAPPA', 'INTWTSUM', 'EFFCOVER'

* The ``pyimcom.meta.distortimage.MetaMosaic`` class is the highest-level interface and constructs a sub-mosaic from the 3x3 set of blocks centered on the specified file. It can be subarrayed, sheared, masked, etc.::

    from pyimcom.meta import distortimage
    # here fout is the file name that we want to read
    mosaic = distortimage.MetaMosaic(fout)
    im = mosaic.noshearimage(4000) # 4000x4000 subarray

  Detailed instructions for this class are on the `Meta Readme <meta_README.rst>`_.
