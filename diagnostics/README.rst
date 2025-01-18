======================
Diagnostics in PyIMCOM
======================

This directory contains tools to generate diagnostics for a PyIMCOM run. The diagnostics should run in a single call on a UNIX-based system, and produce a PDF report (along with associated LaTeX files).

Process for Generating a Report
-------------------------------

**Initializing a new report**

The fundamental object is the ``ValidationReport`` class in the ``diagnostics.report`` module. This is built from a PyIMCOM output file (any block file in a mosaic may be used), and a "stem" for the output files is also given. So for example, one may write::

  from pyimcom.report import ValidationReport
  rpt=ValidationReport(
      'itertest2_F_00_00.fits',
      'out/report1',
      clear_all=True)

The arguments indicate the following:

* We read the configuration from the file ``itertest2_F_00_00.fits`` (other files from the same mosaic will be read if needed)

* Outputs are directed to ``out/report1*``. Some files are sent to that directory (e.g., the output report goes to ``out/report1_main.pdf``), but data (including figures referenced in the LaTeX) are sent to the directory ``out/report1_data/``.

* The ``clear_all`` argument is optional; if True then the LaTeX files, figures, data files, etc. are removed and re-built from scratch. (*Default*: False).

The ``ValidationReport`` class has attributes containing, among other things, the PyIMCOM configuration (``rpt.cfg`` in the above example) extracted from the FITS headers, and information on the input and output directories.

----

**Generating sections**

The ``ReportSection`` class in the ``diagnostics.report`` module generates a report. *This is a base class and is not going to be used much by itself, rather each section of the report will be implemented as a subclass.* A report section is initialized by calling it from the report::

  from pyimcom.report import ReportSection
  sec = ReportSection(rpt)

The report section has an ``infile`` function that can provide the name of a file with a given (*x*, *y*) block coordinate::

  my_file = sec.infile(4,6) # returns location of block (4,6)

You almost certainly will never need to override this. It has a ``build`` method that is intended to be overridden in a subclass. The ``build`` method can be of the form::

  sec.build(nblockmax=8) # nblockmax is for testing only

The ``nblockmax`` argument should be used to only use the lower-left corner of the mosaic if the number of blocks on a side (BLOCK parameter in the configuration file) is larger than this. So it will make a report generate faster when you are de-bugging. But we don't anticipate using it in production.

There are three attributes that need to be generated in the ``build`` method (there are initializations when the section is created):

* ``tex``: The LaTeX code associated with the section. This may reference figures that are generated during the build and are in the data directory (``out/report1_data/`` in the case above).

* ``data``: Data that will be returned as plaintext in the LaTeX report (see below; it could also be extracted by script from the ``.tex`` file if you ever want to make a statistical summary of the reports).

* ``result``: A one-line summary (e.g., "PASS" or "FAIL") from that report section, if applicable. (*Default*: "N/A")

You can add your section to the report::

  rpt.addsections([sec]) # adds one section
  rpt.addsections([sec1,sec2]) # adds two sections, sec1 and sec2

Comments:

* PDFLaTeX is used to build the report, so plots in ``.eps`` format should be converted to ``.pdf`` (this can be done with a system call to ``eps2eps`` followed by ``epstopdf``; note that skipping ``eps2eps`` can result in low-resolution output).

----

**Compiling the report:**

You can compile the report via::

  rpt.compile(ntimes=3)

The variable ``ntimes`` (optional, *default* = 2) controls the number of times the LaTeX is called (if you do this only once, then the links won't update).

The output report contains a header/summary section at the beginning; a section for each report (including figures, if applicable); and an appendix with the configuration file.

Currently Available Reports
---------------------------

*Note*: We will update this section as more types of reports are added.

.. list-table:: **Reports**
  :widths: 33 67

  * - Name
    - Description
  * - ``ReportSection``
    - *(base class)*
  * - ``MosaicImage``
    - Thumbnail of the mosaic
  * - ``SimulatedStar``
    - 1-point statistics of simulated stars

