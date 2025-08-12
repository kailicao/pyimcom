"""
PyIMCOM diagnostic report base classes.

Classes
-------
ReportSection
    Section of the validataion report.
ValidationReport
    An entire validation report, encompassing multiple sections.

"""

import numpy as np
import sys
import os
import subprocess
from astropy.io import fits
from datetime import datetime

from ..config import Config, Settings
from ..diagnostics.outimage_utils.helper import HDU_to_bels

class ReportSection():
    """
    Contains a report section.

    Parameters
    ----------
    rpt : pyimcom.diagnostics.report.ValidationReport
        The report that this section belongs to.

    Attributes
    ----------
    tex : str
        LaTeX source.
    data : str
        Machine readable data block, to be embedded in the report.
    result : str
        Result of test (N/A for not completed).

    Methods
    -------
    __init__
        Constructor
    infile
        Construct the input file name for a given block.
    build
        Build this report section.

    Notes
    -----
    The build method can be redefined in inherited classes, but should keep the signature.
    The optional nblockmax is used to restrict to a maximum size nblockmax x nblockmax sub-mosaic
    (this is mostly used for testing if you want to see if your section compiles without waiting
    for everything to run).

    """

    def __init__(self, rpt):
        """Constructor."""

        self.stem = rpt.stem
        self.LegacyName = rpt.LegacyName
        self.cfg = rpt.cfg
        self.dstem = rpt.dstem
        self.datadir = rpt.datadir
        self.datastem = rpt.datastem
        self.datastem_from_dir = rpt.datastem_from_dir

        self.tex = '\n'+'%'*72+'\n'
        self.data = ''
        self.result = 'N/A'

    def infile(self, in_x, in_y):
        """
        Get the input file name for a given block.

        Parameters
        ----------
        in_x : int
            Horizontal position of the block.
        in_y : int
            Vertical position of the block.

        Returns
        -------
        str
            File name.

        """

        if in_x>=0 and in_x<self.cfg.nblock and in_y>=0 and in_y<self.cfg.nblock:
           in_fname = self.stem + '_{:02d}_{:02d}'.format(in_x,in_y)
           if self.LegacyName: in_fname += '_map'
           in_fname += '.fits'
           return(in_fname)
        else:
           raise Exception("pyimcom.validation.report.ReportSection: infile: block selection out of range")

    def build(self, nblockmax=100):
        """
        Builds report section. Intended to be overwritten.

        Parameters
        ----------
        nblockmax : int
            Maximum block size allowed.

        Returns
        -------
        None

        """

        self.tex += '\\section{Base class section}\nHello world.\n'
        self.data += 'HI, I AM A BOT. HOORAY!\n'

class ValidationReport():
    """
    Contains the validation report.

    Parameters
    ----------
    fname : str
        Block file name.
    dstem : str
        Stem for output files.
    clear_all : bool, default=False
        Removes existing files.

    Attributes
    ----------
    tex : dict of str
        LaTeX source, dictionary with keys 'preamble', 'head', 'body', 'appendix', and 'end'.
    stem: str
        Stem for the input files.
    LegacyName : bool
        Legacy naming convention for the input files with _map.fits? Should be False in the future.
    cfg : pyimcom.config.Config
        The configuration used to generate the mosaic.
    dstem : str
        Stem for output files.
    datadir : str
        Directory for associated data files.
    datastem : str
        Stem for associated datafiles.
    datastem_from_dir : str
        Stem for datafiles, local from compilation directory (for inclusion in LaTeX).

    Methods
    -------
    __init__
        Constructor.
    addsections
        Add sections to the report.
    texout
        Generate LaTeX output.
    writeto
        Writes to a file.
    compile
        Compile the LaTeX source into a PDF.

    """

    def __init__(self, fname, dstem, clear_all=False):
        """Constructor."""

        with fits.open(fname) as f:
            c = f['CONFIG'].data['text']
            n = len(c)
            cf = ''
            for j in range(n):
                cf += c[j]+'\n'
            self.nlayer = np.shape(f['PRIMARY'].data)[-3]
            self.im_dtype = f['PRIMARY'].data.dtype
        self.cfg = Config(cf)
        self.dstem = dstem

        # if the data directory doesn't exist yet, make it
        self.datadir = dstem + '_data'
        if not os.path.exists(self.datadir):
            os.mkdir(self.datadir)
        head, tail = os.path.split(self.dstem)
        self.datastem = self.datadir + '/' + tail
        self.datastem_from_dir = tail + '_data/' + tail
        # ... and remove files if desired
        if clear_all:
            rmlist = []
            n = len(tail)
            for rfname in os.listdir(head):
                if rfname[:n+1] == tail+'_':
                    if not os.path.isdir(head+'/'+rfname):
                        rmlist.append(head+'/'+rfname)
            for rfname in os.listdir(self.datadir):
                if rfname[:n+1] == tail+'_':
                    if not os.path.isdir(self.datadir+'/'+rfname):
                        rmlist.append(self.datadir+'/'+rfname)
            print('Removing:', rmlist)
            for rf in rmlist: os.remove(rf)

        # get the file coordinates
        self.LegacyName = False
        self.stem = fname[:-11]; tail = fname[-11:]
        if fname[-9:]=='_map.fits':
             self.LegacyName = True
             self.stem = fname[:-15]; tail = fname[-15:]

        # LaTeX skeleton
        self.tex = {
            'preamble': '\\documentclass[11pt]{article}\n',
            'head': '\\begin{document}\n\\title{IMCOM Validation report}\n\date{' + datetime.now().strftime("%B %d, %Y") + '}\n',
            'body': '\n',
            'appendix': '\\appendix\n\n',
            'end': '\\end{document}\n'
        }

        # put in margins
        self.tex['preamble'] += '\\setlength{\\hoffset}{0pt}\n'
        self.tex['preamble'] += '\\setlength{\\voffset}{0pt}\n'
        self.tex['preamble'] += '\\setlength{\\topmargin}{-23pt}\n'
        self.tex['preamble'] += '\\setlength{\\headheight}{12pt}\n'
        self.tex['preamble'] += '\\setlength{\\headsep}{23pt}\n'
        self.tex['preamble'] += '\\setlength{\\oddsidemargin}{0pt}\n'
        self.tex['preamble'] += '\\setlength{\\textheight}{648pt}\n'
        self.tex['preamble'] += '\\setlength{\\textwidth}{468pt}\n'

        # packages
        self.tex['preamble'] += '\\usepackage{graphicx}\n'
        self.tex['preamble'] += '\\usepackage{rotating}\n'

        # put in title & summary
        self.tex['head'] += '\\maketitle\n\\tableofcontents\n'
        self.tex['head'] += '\n\\section{Summary}\n'
        self.tex['head'] += '\nThis is a report on the IMCOM run in ' + Settings.RomanFilters[self.cfg.use_filter] + ' band centered at:\n'
        self.tex['head'] += '\\begin{verbatim}RA = ' + '{:8.4f}'.format(self.cfg.ra) + '    DEC = ' + '{:8.4f}'.format(self.cfg.dec) + '\\end{verbatim}\n'
        self.tex['head'] += 'The tests returned the following results.\n\n'

        # appendix on configuration file
        self.tex['appendix'] += '\\section{Configuration file}\n\\label{app:config}\n{\\scriptsize\n\\begin{verbatim}\n'
        self.tex['appendix'] += self.cfg.to_file(None)
        self.tex['appendix'] += '\\end{verbatim}}\n\n'

    def addsections(self, sectionlist):
        """
        Add the list of sections to the report.

        Parameters
        ----------
        sectionlist : list of pyimcom.diagnostics.report.ReportSection
            Sections to add to the report.

        Returns
        -------
        None

        """

        for section in sectionlist:
            # header section
            thisresult = '{:16s}'.format(type(section).__name__[:16]) +':' + section.result
            usechar = None
            for xchar in ['+', '`', '|', '@', '$', '*']:
                if xchar not in thisresult:
                    usechar = xchar
            if usechar is not None:
                self.tex['head'] += '\\noindent\\begin{verbatim}\n' + thisresult + '\\end{verbatim}'
                # '\\noindent\\verb' + usechar + thisresult + usechar + '\n'
                # '\\noindent\\begin{verbatim}\n' + thisresult + '\n\\end{verbatim}\n'
                #
            else:
                raise Exception("Can't print"+thisresult+"in LaTeX verbatim mode")
            self.tex['head'] += '\n'

            # body sections
            self.tex['body'] += '\n' + section.tex + '\n'
            self.tex['body'] += '\\begin{verbatim}\n$$$START ' + type(section).__name__ + '\n';
            self.tex['body'] += section.data
            self.tex['body'] += '\n$$$END ' + type(section).__name__ + '\n\\end{verbatim}\n'

    def texout(self):
        """Returns the entire LaTeX file as one string."""

        return self.tex['preamble'] + self.tex['head'] + self.tex['body'] + self.tex['appendix'] + self.tex['end']

    def writeto(self):
        """Write the LaTeX file."""

        # clear logfiles here
        for ending in ['aux', 'log', 'toc']:
            fn = self.dstem + '_main.' + ending
            if os.path.exists(fn): os.remove(fn)
        with open(self.dstem + '_main.tex', "w") as f:
            f.write(self.texout())

    def compile(self, ntimes=2):
        """
        Compile the LaTeX into a PDF.

        Parameters
        ----------
        ntimes : int, default=2
            Number of times to compile (may have to run twice to get all the references).

        Returns
        -------
        None

        """

        self.writeto()
        pwd = os.getcwd()
        head, tail = os.path.split(self.dstem)
        os.chdir(head)
        for k in range(ntimes):
            try:
                print('compiling from', os.getcwd())
                self.compileproc = subprocess.run(['pdflatex', '-interaction=nonstopmode', tail+'_main.tex'], capture_output=True)
            except:
                print("ERROR *** LaTeX failed to compile! ***\n"); sys.stdout.flush()
        os.chdir(pwd)
