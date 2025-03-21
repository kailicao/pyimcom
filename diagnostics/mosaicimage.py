# Mosaic image

import numpy as np

from .report import ReportSection
from ..pictures.genpic import make_picture_1band

class MosaicImage(ReportSection):
    """Builds the simulated star section of the report."""

    def build(self, nblockmax=100):
        # which blocks to take
        n = min(nblockmax, self.cfg.nblock)
        ns = self.cfg.n1*self.cfg.n2
        j = int(n*ns/1999.99999)
        while ns%j!=0:
            j -= 1
        print('binning=', j, 'nside=', ns, 'tot=', n*ns)

        # make the image itself
        make_picture_1band(
            self.stem,
            self.datastem + '_mosaic.png',
            bounds=[0,n,0,n],
            binning=j,
            srange=(-10.,1000.),
            stretch='asinh')

        self.tex += '\\section{Mosaic image}\n'
        self.tex += '\\begin{figure}\n\\includegraphics[width=6.5in]{' + self.datastem_from_dir + '_mosaic.png}\n'
        self.tex += '\\caption{\\label{fig:MosaicImage1}The mosaic (PNG binned $'+str(j)+'\\times'+str(j)+'$).\n'
        self.tex += 'The image is ' + '{:7.5f}'.format(n*self.cfg.n1*self.cfg.n2*self.cfg.dtheta) + ' degrees across.}\n'
        self.tex += '\\end{figure}\n\n'
        self.tex += 'The mosaic image is shown in Fig.~\\ref{fig:MosaicImage1}.\n\n'

        self.data += 'N = {:2d}, BIN = {:3d}\nIMAGEFILE = {:s}\n'.format(n, j, self.datastem_from_dir + '_mosaic.png')
