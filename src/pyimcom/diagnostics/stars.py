"""
Simulated star diagnostics report section.

Classes
-------
SimulatedStar
    Report section for the simulated stars.

Functions
---------
_starplot_diagnostic
    Makes the multipanel figure for simulated star diagnostics.

"""

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
plt.rcParams['text.usetex'] = True

from ..config import Settings
from .report import ReportSection
from .dynrange import gen_dynrange_data
from .starcube_nonoise import gen_starcube_nonoise

def _starplot_diagnostic(datastem):
    """
    Generates a multi-panel figure and associated data for the injected star diagnostics.

    Parameters
    ----------
    datastem : str
        Stem for the data location.

    Returns
    -------
    dict
        Report information.

    Notes
    -----
    Reads the files::

    * `datastem` + ``_SimulatedStar_StarCat.txt``
    * `datastem` + ``_SimulatedStar_dynrange.dat``
    * `datastem` + ``_SimulatedStar_sqrtS_hist.dat``
    * `datastem` + ``_SimulatedStar_neff_hist.dat``

    Writes the files::

    * `datastem` + ``_SimulatedStar_etmap.pdf``

    This replaces a perl + gnuplot script that was used in the OpenUniverse 2024 run.

    """

    ## Generate data ##

    # this is to send back as part of the report
    outdict = {}

    # pull star catalog data
    # alldata is a 2D array with number of rows equal to the number of stars
    with open(datastem+'_SimulatedStar_StarCat.txt', 'r') as f:
        lines = f.readlines()
    alldata = np.loadtxt(datastem+'_SimulatedStar_StarCat.txt')
    N = np.shape(alldata)[0]
    print(N, 'stars'); sys.stdout.flush()
    # RMS ellipticity and size
    sizemed = float(lines[0].split()[-1]) # this was passed in the header
    evar = np.mean(alldata[:,14]**2+alldata[:,15]**2)
    ewinvar = np.mean(alldata[:,18]**2+alldata[:,19]**2)
    sizerrvar = np.mean((alldata[:,13]-sizemed)**2)

    ## Make plots ##
    matplotlib.rcParams.update({'font.size': 10})
    F = plt.figure(figsize=(13.5,9.))

    ## Dynamic range plot ##
    S = F.add_subplot(2,3,1)
    data = np.loadtxt(datastem+'_SimulatedStar_dynrange.dat')
    xmax = data[-1][0]
    ymin = min(data[0][-1], -30.)
    ymax = max(data[-1][2], 50000.)
    S.set_xlabel(r'radius ($s_{\rm out}$)')
    S.set_ylabel(r'intensity ($e/s_{\rm in}^2$/p)')
    S.set_title('Star profiles')
    S.set_xlim(0,xmax)
    S.set_ylim(ymin,ymax)
    S.set_yscale('symlog', linthresh=abs(ymin))
    S.grid(True, color='g', linestyle='-', linewidth=.25)
    j = int(np.floor(xmax/4))
    S.set_xticks(np.linspace(0,4*j,j+1))
    ylist = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0]
    ylist_label = ['-100'] + ['']*8 + ['-10', '0']
    for p in range(1,5):
        ylist.append(10**p)
        ylist_label.append('{:d}'.format(10**p))
        for q in range(2,10):
            ylist.append(q*10**p)
            ylist_label.append('')
    y = []; yl = []
    for y_, yl_ in zip(ylist, ylist_label):
        if y_>=ymin and y_<=ymax:
            y.append(y_); yl.append(yl_)
    print(y)
    print(yl)
    sys.stdout.flush()
    S.set_yticks(y,yl)
    S.text(0.5*xmax, 0.3*ymax, f'N={N}', color='k')
    S.text(0.5*xmax, 0.1*ymax, r'1,5,25,50,75,95,99 pctiles', color='k')
    S.plot(data[:,0],data[:,5],'k-')
    for j in [2,3,4,6,7,8]:
        S.plot(data[:,0],data[:,j],'b:')

    ## Ellipticity plots ##
    S = F.add_subplot(2,3,5)
    S.set_xlabel('Fidelity (dB)')
    S.set_ylabel(r'$|g_{\rm out}|$')
    S.set_title('Ellipticities')
    S.set_xlim(30,80)
    S.set_ylim(1e-5,1e-2)
    S.set_yscale('log')
    S.grid(True, color='#808080', linestyle='-', linewidth=.25)
    S.set_xticks(np.array(range(30,81,5)))
    y = []; yl = []
    for i in [-5,-4,-3,-2]:
        y.append(10**i)
        yl.append(r'$10^{' + '{:d}'.format(i) + r'}$')
        for c in range(2,10):
            y.append(c*10**i)
            yl.append('')
    S.set_yticks(y,yl)
    esig = evar**0.5
    outdict['RMS_ELLIP_ADAPT'] = esig
    S.text(55, 0.05, 'rms = {:11.5E}'.format(esig), color='k')
    S.scatter(alldata[:,20],np.hypot(alldata[:,14],alldata[:,15]),s=0.1,color='#2020ff',marker='+')

    ## Windowed ellipticity ##
    S = F.add_subplot(2,3,6)
    S.set_xlabel('Fidelity (dB)')
    S.set_ylabel(r'$|e_{\rm out}|$ (0.4$"$ window)')
    S.set_title('Windowed ellipticity')
    S.set_xlim(30,80)
    S.set_ylim(1e-5,1e-2)
    S.set_yscale('log')
    S.grid(True, color='#808080', linestyle='-', linewidth=.25)
    S.set_xticks(np.array(range(30,81,5)))
    y = []; yl = []
    for i in [-5,-4,-3,-2,-1]:
        y.append(10**i)
        yl.append(r'$10^{' + '{:d}'.format(i) + r'}$')
        if i==-1: break
        for c in range(2,10):
            y.append(c*10**i)
            yl.append('')
    S.set_yticks(y,yl)
    esig = ewinvar**0.5
    outdict['RMS_ELLIP_FIXEDWIN'] = esig
    S.text(55, 0.05, 'rms = {:11.5E}'.format(esig), color='k')
    S.scatter(alldata[:,20],np.hypot(alldata[:,18],alldata[:,19]),s=0.1,color='#2020ff',marker='+')

    # size deviation from median
    S = F.add_subplot(2,3,4)
    S.set_xlabel('Fidelity (dB)')
    S.set_ylabel(r'$(\sigma-\sigma_{\rm med})/\sigma_{\rm med}$')
    S.set_xlim(30,80)
    S.set_ylim(1e-5,1e-2)
    S.set_yscale('log')
    S.set_title('Size errors')
    S.grid(True, color='#808080', linestyle='-', linewidth=.25)
    S.set_xticks(np.array(range(30,81,5)))
    y = []; yl = []
    for i in [-5,-4,-3,-2,-1]:
        y.append(10**i)
        yl.append(r'$10^{' + '{:d}'.format(i) + r'}$')
        if i==-1: break
        for c in range(2,10):
            y.append(c*10**i)
            yl.append('')
    S.set_yticks(y,yl)
    outdict['RMS_SIZE_ERR'] = sizerrvar**0.5/sizemed
    S.text(55, 0.018, 'rms = {:11.5E}'.format(outdict['RMS_SIZE_ERR']), color='k')
    outdict['MED_SIZE'] = sizemed
    S.text(55, 0.012, r'$\sigma_{\rm med}/s_{\rm out}=$' + '{:.5f}'.format(sizemed), color='k')
    ds = alldata[:,13]/sizemed-1.
    S.scatter(alldata[:,20],np.clip(ds,1e-49,None),s=0.2,color='#00a040',marker='1',label='positive')
    S.scatter(alldata[:,20],np.clip(-ds,1e-49,None),s=0.2,color='#e06000',marker='2',label='negative')
    S.legend(loc='upper right')

    # sqrtS histogram
    S = F.add_subplot(2,3,3)
    S.set_title('Noise histogram')
    S.set_xlim(0,2)
    S.set_xticks(np.linspace(0,2,11))
    with open(datastem + '_SimulatedStar_sqrtS_hist.dat', 'r') as f:
        args__ = f.readlines()[0].split()[1:]
        ymax   = float(args__[0])
        pc     = float(args__[1])
    S.set_ylim(0.9, 1.25*ymax)
    S.set_yscale('log')
    S.set_xlabel(r'$S^{1/2}$')
    S.set_ylabel('counts')
    y = []; yl = []
    for i in range(9):
        y.append(10**i)
        yl.append(r'$10^{' + '{:d}'.format(i) + r'}$')
        for c in range(2,10):
            y.append(c*10**i)
            yl.append('')
            if c*10**i>ymax: break
        if 10**i>ymax: break
    S.set_yticks(y,yl)
    S.grid(True, color='g', linestyle='-', linewidth=.25)
    S.text(1.2, 0.4*ymax, '{:.3f}'.format(pc) + r'\% at $>$2', color='k')
    outdict['PCT_NOISE_GT2'] = pc
    sdata = np.loadtxt(datastem + '_SimulatedStar_sqrtS_hist.dat')
    S.bar(sdata[:,0], sdata[:,1], width=0.7*(sdata[1,0]-sdata[0,0]), align='center', facecolor='#406000', edgecolor='#0000a0')

    # coverage histogram
    S = F.add_subplot(2,3,2)
    S.set_title('Coverage histogram')
    S.set_xlim(0,10)
    S.set_xticks(np.linspace(0,10,11))
    with open(datastem + '_SimulatedStar_neff_hist.dat', 'r') as f:
        args__ = f.readlines()[0].split()[1:]
        ymax   = float(args__[0])
        pc     = float(args__[1])
    S.set_ylim(0.9, 1.25*ymax)
    S.set_yscale('log')
    S.set_xlabel(r'effective $N_{\rm exp}$')
    S.set_ylabel('counts')
    y = []; yl = []
    for i in range(9):
        y.append(10**i)
        yl.append(r'$10^{' + '{:d}'.format(i) + r'}$')
        for c in range(2,10):
            y.append(c*10**i)
            yl.append('')
            if c*10**i>ymax: break
        if 10**i>ymax: break
    S.set_yticks(y,yl)
    S.grid(True, color='g', linestyle='-', linewidth=.25)
    S.text(6.5, 0.4*ymax, '{:.3f}'.format(pc) + r'\% at $>$10', color='k')
    ndata = np.loadtxt(datastem + '_SimulatedStar_neff_hist.dat')
    S.bar(ndata[:,0], ndata[:,1], width=0.7*(ndata[1,0]-ndata[0,0]), align='center', facecolor='#406000', edgecolor='#0000a0')

    # finish up figure
    F.set_tight_layout(True)
    F.savefig(datastem+'_SimulatedStar_all.pdf')
    plt.close(F)

    return outdict

class SimulatedStar(ReportSection):
    """
    Builds the simulated star section of the report.

    Inherits from pyimcom.diagnostics.report.ReportSection, overrides the build method.

    """

    def build(self, nblockmax=100):

        nblock = min(nblockmax, self.cfg.nblock)

        # The runs
        output = gen_dynrange_data(self.infile, self.datastem+'_SimulatedStar', nblockmax=nblockmax)
        print(output)
        output2 = gen_starcube_nonoise(self.infile, self.datastem+'_SimulatedStar', nblockmax=nblockmax)
        print(output2)

        # Make figures
        outdict = _starplot_diagnostic(self.datastem)
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
        self.tex += '\\caption{\\label{fig:SimulatedStarALL}The profiles of noisy injected stars (top left); '
        self.tex += 'histograms of the coverage $n_{\\rm eff}$ and noise amplification'
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
        v = g/1e-4
        v = np.where(v<1, v, np.log(v)+1)**.5/2.
        g50 = np.nanpercentile(g,50)
        g90 = np.nanpercentile(g,90)
        outdict['BLK_SIMSTAR_G50PCT'] = g50
        outdict['BLK_SIMSTAR_G90PCT'] = g90
        print('percentiles of the binned median:', g50, g90)
        im2 = S.quiver(X,Y,v*np.cos(phi),v*np.sin(phi), headaxislength=0, headlength=0, headwidth=0, width=.03/nblock, pivot='mid', scale=1., scale_units='xy', angles='xy')
        F.colorbar(im, orientation='vertical')
        F.set_tight_layout(True)
        F.savefig(self.datastem+'_SimulatedStar_etmap.pdf')
        plt.close(F)
        # ... and describe it
        self.tex += '\\begin{figure}\n\\includegraphics[width=6in]{' + self.datastem_from_dir + '_SimulatedStar_etmap.pdf}\n'
        self.tex += '\\caption{\\label{fig:SimulatedStar_etmap}PSF size error (diverging color scale) and ellipticity (whiskers).'
        self.tex += ' Whisher lengths are on a square root scale: 0.5 blocks for $g=10^{-4}$; 1.0 blocks for $g=4\\times 10^{-4}$; etc.'
        self.tex += ' so that the ellipticity relates to the moment of inertia of the line.}\n\\end{figure}\n\n'
        self.tex += 'A map of the PSF size and ellipticity errors from the noiseless simulated stars is shown in Fig.~\\ref{fig:SimulatedStar_etmap}.\n'

        # put the key results in the data section
        # format is: name of result, value
        for i in outdict.keys():
            self.data += '{:19s} '.format(i) + str(outdict[i]) + '\n'
