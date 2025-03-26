Workflow to run PyIMCOM
#############################

This is a quick guide for the jobs you will want to submit to build a mosaic in PyIMCOM. To build a mosaic, you will need:

- The configuration file (for production runs, this is usually an external JSON file). See `configuration files <config_README.rst>`_ for the format.

- The input images. The format should be as indicated in the ``INDATA`` keyword in the configuration file.

- Metadata, including (at least) the PSFs and WCS. (In some formats, the WCS is in the FITS headers of the input images.)

- Additional layer information (e.g., laboratory darks) if you are going to be requesting those as layers to coadd.

Overall workflow
******************

There are several steps in the workflow:

#. [*If needed*] Put the input files to the appropriate input directories (either actually moving them or building symbolic links). PyIMCOM will search for files in the ``INDATA`` directory given.

#. [*Optional*] If ``INLAYERCACHE`` is set, you may choose to generate the input layers first. This is recommended if you are on a standard high-performance computing cluster.

#. Run the co-addition (this is the most time-consuming step).

#. [*Recommended*] Generate a diagnostic report on the outputs.

(We don't have the PSF splitting in this workflow yet, but we will!)

We describe each of these in turn below.

Input data
===================

The input data may be in one of the following formats (more may be added in the future). Science images and lab darks are in subdirectories of the directory given by the ``INDATA`` keyword (first value); PSFs are in the directory given by the ``INPSF`` keyword (first value). The formats are: 

- ``dc2_sim`` : Roman + Rubin Data Challenge 2 format (`Troxel et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.2801T/abstract>`_). The WCS is part of the FITS header for the images, and the PSF images were provided separately.

- ``anlsim`` : OpenUniverse 2024 simulation format (`Open Universe 2024 <https://ui.adsabs.harvard.edu/abs/2025arXiv250105632O/abstract>`_). The WCS is part of the FITS header for the images. The PSF Legendre data cubes have to be constructed separately from the main data release; see instructions `here <historical/OpenUniverse2024/genpsf.py>`_.

  - Science image example: \<indata directory\> ``/simple/Roman_WAS_simple_model_F184_1591_8.fits`` (filter F184, observation ID 1591, SCA 8)

  - Lab noise example: \<indata directory\> ``/labnoise/slope_1511_12.fits`` (observation ID 1511, SCA 12)

  - PSF example: \<indata directory\> ``/psf_polyfit_5434.fits`` (observation ID 5434, all SCAs)

If your platform doesn't support putting all those files in one directory, we recommend symbolic links as a workaround.

Generating input layers
===========================

*Optional: only used if you have turned on INLAYERCACHE.*

If ``INLAYERCACHE`` is set, then PyIMCOM will cache all of the input layers it generates itself (e.g., grids of injected objects, synthetic noise fields ...). The ``INLAYERCACHE`` keyword is a stem: if you set it to ``mydir/myprefix``, then all of the cached files will appear in ``mydir`` (or subdirectories) with a file name that starts with ``myprefix``.
When a PyIMCOM process finds it needs one of these files, it first searches for it: if it is there then it simply reads from disk, otherwise it computes it and then saves to disk.

You can use less computing time if you generate all the input layers first, and then do the coaddition. A simple way to do this if your configuration file is ``config_file`` and you are coadding blocks ``j1`` through ``j2`` (of ``BLOCK**2`` : recall this is the number of blocks in a square mosaic) is a script like ``gen1.py``::

   import sys
   from pyimcom.config import Config
   from pyimcom.coadd import Block

   cfg = Config(sys.argv[1])
   cfg.stoptile=4
   block = Block(cfg=cfg, this_sub=int(sys.argv[2]))

(the ``stoptile`` tells PyIMCOM not to run through the coalition of the whole block) and then you can write a Perl script like::

   for $j ($j1..$j2) {
       system "python3 -m gen1.py config $j > log-gen.$j";
   }

You could generate all of the input layers in series with ``$j1=0`` and ``$j2=BLOCK**2-1``. However, more commonly you will want to parallelize. As a rough guide, an SCA is 636 arcsec along the diagonal. So if you are generating blocks with a size of, say, 100x100 arcsec, and you are doing 36x36 blocks (BLOCK=36), then row ``i`` of the mosaic is guaranteed to use SCAs that overlap with row ``i+8``; so you can break the mosaic into "strips" of 8=ceil(636/100)+1 rows (or 8x36=288 blocks) and then each run those in parallel. So in that case, you could run 0-287, 288-575, 576-863, 864-1151, and 1152-1295 (the last block).

Running the coaddition
==============================

This is the most time-consuming step. In principle, all the blocks can be run in parallel. So you could run a script like ``run_coaddition.py``::

   import sys
   from pyimcom.config import Config
   from pyimcom.coadd import Block
   from pyimcom.truthcats import gen_truthcats_from_cfg

   cfg = Config(sys.argv[1])
   block = Block(cfg=cfg, this_sub=int(sys.argv[2]))
   if int(sys.argv[2])==0: gen_truthcats_from_cfg(cfg)

and then call::

   python3 -m run_coaddition.py config $j > log-coadd.$j

You could put this in a loop in a Perl script, but on the Ohio Supercomputer Center we found it is faster to just make a job array and run a separate job for each block.

The "truth catalog" for the layers that PyIMCOM draws only needs to be generated once in a given mosaic. The above script generates it with the first block.

Generating the diagnostic report
=====================================

You can generate a diagnostic report by running the diagnostics module *after* the coaddition has finished::

   python3 -m pyimcom.diagnostics.run outdir/outstem_00_00.fits writeupdir/writeupstem

Here ``outdir/outstem_00_00.fits`` is one of the pyIMCOM output files, and documentation will be put into the directory ``writeupdir`` with file names that begin with ``writeupstem``. A PDF file will be generated with the suffix ``_main.pdf`` (in this case: ``writeupdir/writeupstem_main.pdf``). The LaTeX file (in this case: ``writeupdir/writeupstem_main.tex``), and data including figures that have to be ``\include``\ d (in the folder ``writeupdir/writeupstem_data``) are generated.

You need to provide a different ``writeupdir/writeupstem`` if you are running a different mosaic **or different band** so as to avoid collisions.

If you want to add more diagnostic reports, see the `instructions <../diagnostics/README.rst>`_.

