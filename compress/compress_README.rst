PyIMCOM Compression module
############################

This is the compression module for PyIMCOM outputs. It carries out both lossy and lossless compression operations.

Overview
**********

There are two fundamental operations: compression and decompression. Logically, compression comes first, but because of the many varied uses of PyIMCOM coadds, we expect more users will need to implement decompression.

Compression
=============

Compression operations are carried out with the ``pyimcom.compress.compressutils.CompressedOutput`` class. This is initialized with the name of the file you want to compress. Then various compressions can be done, and the output written to a file. A standard code snippet to compress file ``fname`` and write the output to ``fout`` might be::

   from pyimcom.compress.compressutils import CompressedOutput
   with CompressedOutput(fname) as f:
      for j in range(1,len(f.cfg.extrainput)):
         if f.cfg.extrainput[j][:6].lower()=='gsstar' or f.cfg.extrainput[j][:5].lower()=='cstar'\
               or f.cfg.extrainput[j][:8].lower()=='gstrstar' or f.cfg.extrainput[j][:8].lower()=='gsfdstar':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -1./64., 'VMAX': 7./64., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:5].lower()=='nstar':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -1500., 'VMAX': 10500., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:5].lower()=='gsext':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -1./64., 'VMAX': 7./64., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:8].lower()=='labnoise':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -5, 'VMAX': 5, 'BITKEEP': 16, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:10].lower()=='whitenoise':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -8, 'VMAX': 8, 'BITKEEP': 16, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:7].lower()=='1fnoise':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -32, 'VMAX': 32, 'BITKEEP': 16, 'DIFF': True, 'SOFTBIAS': -1})
      f.to_file(fout, overwrite=True)

The configuration used to generate file ``fname`` appears as ``f.cfg``. So this script loops through the layers in the file (except the 0th one, which is the science layer), checks whether their names start with the indicated prefixes, and if layer ``j`` has that prefix then it compresses the layer with the given scheme and parameters. Finally, the compressed structure is written to ``fout`` with the ``to_file`` method.

Metadata on the compression scheme (needed to decompress the data) is stored in a new ASCII Table in the compressed file (the ``CPRESS`` HDU). The possible schemes and parameters are described in more detail below.

It is recommended to use the ``.cpr.fits.gz`` suffix for a compressed PyIMCOM file, e.g., ``itertest2_F_10_04.fits`` compresses to ``itertest2_F_10_04.cpr.fits.gz``. This is not enforced by ``compressutils``, but some analysis tools recognize compressed files by this convention. The .gzipped file can be unzipped (it is a valid .fits.gz file!) but we expect this won't be common; if you need the uncompressed file, you will probably just decompress it.

Decompression
==================

To decompress file ``fout`` to a standard fits file ``frec``, you can run::

   from pyimcom.compress.compressutils import ReadFile
   ReadFile(fout).writeto(frec, overwrite=True)

The ``ReadFile`` function returns an ``astropy.io.fits.HDUList`` object, so standard reading of the headers and data is then possible, e.g., you could write::

   from pyimcom.compress.compressutils import ReadFile
   with ReadFile(fout) as f:
      x = f[0].data
      ...

Compression schemes
*********************

There are several compression schemes offered.

I24B
======

The ``I24B`` compression scheme is the most common. For each layer, it generates two HDUs. The compressed data cube is a 3D ``uint8`` array with a name starting with ``HSHX`` and ending with the hexadecimal code for that layer, e.g., ``HSHX000D`` is the compressed cube for layer 13 (=0XD). There is also a binary table (in this case: ``HSHV000D``) containing values that overflowed the minimum and maximum of the compression range (this is common if, e.g., the image to be compressed is mostly dark sky, but there are a few bright objects, and you don't want to set the compression scale to handle the brightest pixels). The naming scheme will fail and the algorithm will refuse to compress if your layer number goes past 0XFFFF=65535, but in practical cases you won't need that many layers.

The ``pars`` argument controls parameters to be passed to the I24B algorithm. It controls the following steps in order (for compression; note that decompression is the opposite):

.. list-table:: **I24B steps**
   :widths: 20 20 60
   :header-rows: 1

   * - Step
     - Keywords
     - Operation

   * - Rescaling
     - ``VMIN``, ``VMAX``, ``ALPHA``, ``BITKEEP``
     - Stretches the data so that the range from ``VMIN`` to ``VMAX`` is compressed to ``0`` to ``2**BITKEEP-1``. This may be linear (``ALPHA``\ =1, default) or stretched with another power law (e.g., ``ALPHA``\ =0.5 for square-root stretch). Values that fall off the end of the range will be saved to the overflow table.

   * - Differencing
     - ``DIFF``
     - If True, saves the difference of successive pixels rather than the pixels themselves. This is carried out in arithmetic mod ``2**BITKEEP``.

   * - Biasing
     - ``SOFTBIAS``
     - If \>0, adds this as a bias to prevent slight negative numbers from rolling over to all 1's. If =0 (default), this is skipped. If ``SOFTBIAS`` is -1, re-maps in order of absolute value (i.e., 0,-1,1,-2,2,-3,3,... are mapped to 0,1,2,3,4,5,6,...) so that all small values are close to zero.

   * - Repacking
     - ``REORDER``
     - The integers are repackaged into a ``uint8`` array, with the 8 least significant bits in ``array[0,:,:]``, then the next 8 in ``array[1,:,:]``, and (if ``BITKEEP``\>16) the next 8 in ``array[2,:,:]``. If ``REORDER`` is True (default), then every bit is repacked so that the 0x1 bits fill the first eighth of ``array[0,:,:]``, then the 0x2 bits fill the next eighth, etc. If ``REORDER`` is False, then the repacking is done only at the byte level.

   * - Gzipping
     - *none*
     - The file is gzipped; the previous operations are usually optimized so that long runs of all-0 or all-1 bits (which gzip efficiently) appear together.

These steps are controlled by the following keywords. The default is given, unless a value is required (no default):

.. list-table:: **I24B keywords**
   :widths: 20 20 60
   :header-rows: 1

   * - Keyword
     - Default
     - Meaning
   * - ``VMIN``
     - required
     - Minimum value to compress into the integer image.
   * - ``VMAX``
     - required
     - Maximum value to compress into the integer image.
   * - ``ALPHA``
     - 1\.
     - Power-law stretch of the compression scale (1 is a linear stretch, but values \<1 are useful if you want to emphasize features near the bottom range of the scale, e.g., 0.5 is a square root stretch).
   * - ``BITKEEP``
     - 24
     - Number of bits to keep in the integer image. The maximum is the default of 24. Reducing this will reduce your file size (especially since least significant bits tend to be noisy and not compress well), but you could introduce quantization biases if it is too small.
   * - ``DIFF``
     - False
     - Whether to save differences of pixels (True) or the pixels themselves (False).
   * - ``SOFTBIAS``
     - 0
     - If positive, this is a bias to prevent numbers near 0 from rolling over to produce lots of 1's in the binary encoding. So, e.g., ``SOFTBIAS`` of 64 will bias -30 up to 34. The special value of ``SOFTBIAS``\=-1 is used to take all numbers with small absolute values and map them to near zero. Often ``SOFTBIAS=-1`` is useful with ``DIFF=True``.
   * - ``REORDER``
     - True
     - Pack bits (instead of bytes) in order from least significant to most significant. This is recommended for most uses.
