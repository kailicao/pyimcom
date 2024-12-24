# Simulated star diagnostics report section

import numpy as np
import sys
import os
import subprocess
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.colors as colors

from ..config import Settings
from .report import ReportSection
from .dynrange import gen_dynrange_data
from .starcube_nonoise import gen_starcube_nonoise

class SimulatedStar(ReportSection):
    """Builds the simulated star section of the report."""

    def build(self, nblockmax=100):

        nblock = min(nblockmax, self.cfg.nblock)

        # The runs
        output = gen_dynrange_data(self.infile, self.datastem+'_SimulatedStar', nblockmax=nblockmax)
        print(output)
        output2 = gen_starcube_nonoise(self.infile, self.datastem+'_SimulatedStar', nblockmax=nblockmax)
        print(output2)

        # Make figures
        newpath = os.path.dirname(os.path.abspath(__file__))
        proc = subprocess.run(['perl', newpath+'/starplot_diagnostic.pl', self.datastem], capture_output=True)
        print('STDOUT', proc.stdout)
        print('STDERR', proc.stderr)
        if proc.returncode:
            print('Error in starplot_diagnostic.pl')
        try:
            proc = subprocess.run(['epstopdf', self.datastem+'_SimulatedStar_all.eps', self.datastem+'_SimulatedStar_all.pdf'], capture_output=True)
            print('STDOUT', proc.stdout)
            print('STDERR', proc.stderr)
            if proc.returncode:
                raise Exception("Conversion error, epstopdf")
        except:
            print("Couldn't do conversion")

        # extract outputs (this is one way to get data back from a Perl script)
        outdict = {}
        checkfile = self.datastem+'_SimulatedStar_outputs.txt'
        print("3", checkfile); sys.stdout.flush()
        if os.path.exists(checkfile):
            print("4", checkfile); sys.stdout.flush()
            with open(checkfile, "r") as f:
                returntext = f.readlines()
            print(returntext)
            a = returntext[0].split()
            b = returntext[1].split()
            print(a); print(b)
            for i,j in zip(a,b):
                outdict[i]=j
            print(outdict)

        # The TeX output
        self.tex += '\\section{Simulated star section}\nThe following data were generated, based on ' + str(output['COUNTBLOCK']) + ' blocks:\n'
        self.tex += '\\begin{list}{$\\bullet$}{}\n'
        if output['SQRTS'] is not None:
            self.tex += '\\item The {\\tt SQRTS} (noise amplification) histogram.\n'
        if output['NEFF'] is not None:
            self.tex += '\\item The {\\tt NEFF} (effective coverage) histogram.\n'
        if output['DYNRANGE'] is not None:
            self.tex += '\\item The {\\tt DYNRANGE} file, listing the profiles of simulated stars.\n'
            self.tex += 'The simulated stars have flux ${:f}$ electrons, background per pixel ${:f}$ e/pix, and random number seed index ${:d}$.\n'.format(
                        output['NSTARLAYER']['FLUX'], output['NSTARLAYER']['BACKGROUND'], output['NSTARLAYER']['SEED'])
        self.tex += '\\end{list}\n'
        # describe the figure
        self.tex += '\\begin{sidewaysfigure}\n\\includegraphics[height=6in]{' + self.datastem_from_dir + '_SimulatedStar_all.pdf}\n'
        self.tex += '\\caption{\\label{fig:SimulatedStarALL}The profiles of noisy injected stars (top left); histograms of the coverage $n_{\\rm eff}$ and noise amplification'
        self.tex += '$\\sqrt{S}$ (top center \\& right); and adaptive size, adaptive ellipticity, and fixed-window ellipticity (bottom, left to right).}\n'
        self.tex += '\\end{sidewaysfigure}\n'
        self.tex += 'These results are displayed in Fig.~\\ref{fig:SimulatedStarALL}.\n'

        # whisker plot
        RR = self.cfg.sigmatarget * Settings.pixscale_native / (self.cfg.dtheta*np.pi/180)
        print(RR)
        bandnames = Settings.RomanFilters[self.cfg.use_filter]
        print(bandnames)

        grid__ = np.array(range(nblock))
        X,Y = np.meshgrid(grid__,grid__)
        sigma = np.zeros((nblock,nblock))
        g1 = np.zeros((nblock,nblock))
        g2 = np.zeros((nblock,nblock))

        # load information
        c1 = c2 = 0
        data = np.loadtxt(self.datastem+'_SimulatedStar_StarCat.txt')
        for ix in range(nblock):
            for iy in range(nblock):
                ind = np.logical_and(data[:,2]==ix, data[:,3]==iy)
                c2 += np.count_nonzero(ind)
                ind = np.logical_and(ind, data[:,21]>0) # with coverage
                c1 += np.count_nonzero(ind)
                dd = data[ind,:]
                sigma[iy,ix] = np.mean(dd[:,13])
                g1[iy,ix] = np.mean(dd[:,14])
                g2[iy,ix] = np.mean(dd[:,15])
        print(c2,c1,1-c1/c2)
        outdict['PCT_NOT_COVERED'] = (1-c1/c2)*100

        # now make plot
        matplotlib.rcParams.update({'font.size': 10})
        F = plt.figure(figsize=(8,6.5))
        S = F.add_subplot(1,1,1)
        S.set_title('PSF residuals, '+bandnames)
        S.set_xlabel('block x')
        S.set_ylabel('block y')
        im = S.imshow(sigma/RR-1., cmap='RdBu_r', aspect=1, interpolation='nearest', origin='lower',
        norm=colors.SymLogNorm(3e-4, vmin=-.004, vmax=.004))
        phi = np.arctan2(g2,g1)/2.
        g = np.sqrt(g1**2+g2**2)
        v = g/5e-4
        im2 = S.quiver(X,Y,v*np.cos(phi),v*np.sin(phi), headaxislength=0, headlength=0, headwidth=0, pivot='mid', scale=1., scale_units='xy', angles='xy')
        F.colorbar(im, orientation='vertical')
        F.set_tight_layout(True)
        F.savefig(self.datastem+'_SimulatedStar_etmap.pdf')
        plt.close(F)
        # ... and describe it
        self.tex += '\\begin{figure}\n\\includegraphics[width=6in]{' + self.datastem_from_dir + '_SimulatedStar_etmap.pdf}\n'
        self.tex += '\\caption{\\label{fig:SimulatedStar_etmap}PSF size error (diverging color scale) and ellipticity (whiskers).'
        self.tex += ' A whisher length of 1 block corresponds to an ellipticity of $g=5\\times 10^{-4}$.}\n\\end{figure}\n\n'
        self.tex += 'A map of the PSF size and ellipticity errors from the noiseless simulated stars is shown in Fig.~\\ref{fig:SimulatedStar_etmap}.\n'

        # put the key results in the data section
        # format is: name of result, value
        for i in outdict.keys():
            self.data += '{:19s} '.format(i) + str(outdict[i]) + '\n'
