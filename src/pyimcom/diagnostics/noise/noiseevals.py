# Python script for calculating block-wise evaluation criteria: noise amplification and 2D equivalent width
# Relies on power spectra outputs from noisespecs.py
# Usage format: python noiseevals.py <filter> <instem> <outstem>

import sys
import numpy
from astropy.io import fits
from os.path import exists
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import CubicSpline
import scipy.integrate as integrate
from collections import namedtuple
import json
import re
import numpy as np
import datetime

print('This run started: ', datetime.datetime.now())

nblock = 48
nstart = 0

# # Check input format
if len(sys.argv) < 4:
    print('Missing an argument. Usage format is python noisespecs.py <filter> <input prefix> <outstem>\n')
    exit()

# Determine filter to use
filter = sys.argv[1];
nblockuse = 5

if filter == 'Y':
    filtername = 'Y106'
    area = 5915  # cm^2
    wavelength = 1.06*10**-6 #m
if filter == 'J':
    filtername = 'J129'
    area = 6051
    wavelength = 1.29*10**-6
if filter == 'H':
    filtername = 'H158'
    area = 5978
    wavelength = 1.58*10**-6
if filter == 'F':
    filtername = 'F184'
    area = 3929
    wavelength = 1.84*10**-6

# prefix and suffix
in1 = sys.argv[2]
outstem = sys.argv[3]

print('# Filter: ', filter, ', instem: ', in1, ' outstem: ', outstem)
    
outf = open(outstem+filtername+'_results.txt', 'w+')
outf.write("Block ID    NoiseLayer  EW_2D   Az_anisotropy \n")
outf.close()

# Set useful constants
tfr = 3.08  # sec
gain = 1.458  # electrons/DN
s_in = 0.11  # arcsec
D = 2.37 #m, telescope diameter
th_fwhm = 2.25 # for now...
sigma = th_fwhm/np.sqrt((8*np.log(2)))


# Loop through all the blocks
for iblock in range(nstart, nstart + nblockuse):

    j = iblock
    ibx = j % nblock;
    iby = j // nblock

    # Combine in1 with block ID to get input file and block label
    blockid = '{:s}_{:02d}_{:02d}'.format(filter, ibx, iby)
    infile = in1 + blockid + '_ps.fits'
    
    # extract information from the header of the first file
    if iblock == nstart:
        with fits.open(infile) as f:

            n = numpy.shape(f[0].data)[-1]  # size of 2D PS images

            config = ''
            for g in f['CONFIG'].data['text'].tolist(): config += g + ' '
            configStruct = json.loads(config)
            configdata = f['CONFIG'].data

            blocksize = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) * float(
                configStruct['OUTSIZE'][2]) / 3600. * numpy.pi / 180  # radians
            L = int(configStruct['OUTSIZE'][0]) * int(configStruct['OUTSIZE'][1]) + 2*int(configStruct['OUTSIZE'][1])* int(configStruct['PAD']) # side length in px
            
            s_out = float(configStruct['OUTSIZE'][2])  # in arcsec
            # padding region around the edge
            bdpad = int(configStruct['OUTSIZE'][1]) * int(configStruct['PAD'])

            # figure out which noise layers are there
            layers = [''] + configStruct['EXTRAINPUT']
            print('# Layers:', layers)
            noiselayers = []
          
            for i in range(len(layers)):
              m = re.match(r'^whitenoise(\d+)$', layers[i])
              if m:
                noiselayers.append(str(m[0]))
              m = re.match(r'^1fnoise(\d+)$', layers[i])
              if m:
                noiselayers.append(str(m[0]))
              m = re.match(r'^labnoise$', layers[i])
              if m:
                noiselayers.append(str(m[0]))
 



    if not exists(infile):
        continue
        
    print('# Running block: ', blockid)
    f = fits.open(infile)
    ps_2d = np.copy(f['PRIMARY'].data.astype(np.float32))
    ps_1d = np.copy(f['P1D_TABLE'].data)
    f.close()
    
    print('# Noise Layers: ', noiselayers)

    
    nradbins = L//16

    def az_anisotropy(p1d, noisetype, N_phi, N_x, nmcbins=5):
        """
        Calculate the azimuthal anisotropy of a 1D power spectrum.

        Inputs:
        p1d: numpy array containing power, wavenumber
        noisetype: white, 1f
        nmcbind: number of mean coverage bins. default=5

        Returns: Ratio of measured vs. predicted 1D power spectrum
        """
        p1d_x = p1d[0, :]
        p1d_y = p1d[1, :]
        
        print('# Cubic spline integration step')
        p1d_cs = CubicSpline(p1d_x, p1d_y)
        p1d_int = p1d_cs.integrate(np.min(p1d_x), np.max(p1d_x))
        print('# Actual P1d int: ', p1d_int)
        
        # Depending on the type of noise being evaluated, define the analytical power spectrum P(u)
        # See arxiv 2303.08750 (Yamamoto et al, 2023) Appendix A for derivation
        m=re.search(r'white', noisetype)
        if m:
        # k = u * (2 * np.pi)
            print('# White Noise theoretical PS')
            def P(u):
                return (2 * np.pi * u) * (1 / nmcbins) * s_in**2 * \
                np.exp(((1 / 12 * s_in ** 2 - sigma ** 2) * (u * (2 * np.pi)) ** 2) + (
                    1 / 1920 * s_in ** 4 * (u * (2 * np.pi)) ** 4))
                    
        m=re.search(r'1f', noisetype)
        if m:
            print('# 1/f Noise theoretical PS')
            def P_1D(f):
                return np.where(np.logical_and(1 / N_x ** 2 <= np.abs(f), np.abs(f) <= 1 / 2), 1 / (2 * np.abs(f)), 0.)

            def P(u):
                P = 0
                for K_phi in range(1, N_phi - 1):
                    phi = 2 * np.pi * K_phi / N_phi
                    j_min = int(-N_x / 2 - np.amax(u))
                    j_max = int(N_x / 2 + np.amax(u))
                    j = np.arange(j_min, j_max)
                    P = 1 / N_phi * np.sum( P_1D( (s_in * u * np.sin(phi) - j) / N_x) * \
                             (np.sinc(N_x * s_in * u * np.cos(phi) + j - s_in * u * np.sin(phi))) ** 2 * \
                             (s_in**2 / nmcbins) * np.exp(((1 / 12 * s_in ** 2 - sigma ** 2) * (u * (2 * np.pi)) ** 2) + (1 / 1920 * s_in ** 4 * (u * (2 * np.pi)) ** 4)))
                return P
                        
        m=re.search(r'lab', noisetype)
        if m:
            print('# Lab noise field- no theoretical version')
            pass
                
        #print('# Integrating theory power spectrum start: ', datetime.datetime.now())
        exp_int, err = integrate.quad(lambda u: P(u), np.min(p1d_x), np.max(p1d_x))
        #print('# Integrating theory power spectrum end: ', datetime.datetime.now())
        return p1d_int/exp_int
        
    def ew_2d(p2d):
        """
        Measure the "equivalent width" of the + sign feature w/r to the background of the 2D power spectrum.

        Input: p2d: a 2D numpy array power spectrum

        Output: "equivalent width" of a continuum-depth + sign feature with the same flux as the measured + sign
        """

        L = np.shape(p2d)[1]
        R = np.rint((2/np.pi)*np.sqrt(-np.log(10**-3)*np.log(2))*s_out*L/th_fwhm)
        area_2d = L**2 - (2 * R)**2 #px^2
        area_1d = 2 * (L - (2 * R))
        
        print('# 2D area: ', area_2d, ' 1D area: ', area_1d, ' radius: ', R)

        # Find the flux in the spikes (along central axes only for now)
        mask_spike = np.copy(p2d)
        for ipx in range(0, L):
            for jpx in range(0, L):
                if ((ipx >= L//2-R) and (ipx <= L//2+R)) or ((jpx >= L//2-R) and (jpx <= L//2+R)):
                    mask_spike[ipx,jpx]=True
                else:
                    mask_spike[ipx, jpx]=False
        F_spike = np.sum(p2d*mask_spike)
                
        #Find the total flux in the continuum
        mask_cont = np.copy(p2d)
        for ipx in range(0,L):
            if (ipx < L//2-R) or (ipx > L//2+R):
                for jpx in range(0, L):
                    mask_cont[ipx, jpx]=True
            elif (L//2-R <= ipx <= L//2+R):
                for jpx in range(0, L):
                  if not (L//2-R <= jpx <= L//2+R):
                    mask_cont[ipx, jpx]=True
        F_continuum = np.sum(p2d*mask_cont)
        
        print('# F spike and F cont: ', F_spike, F_continuum )

        return (area_2d/area_1d)*(F_spike/F_continuum)


    def eval_results(blockid, p1d, p2d):
        """
        Calculate the two criteria for the given block's power spectra and return them in one results tuple
        p1d: 1d power spectrum from input file
        p2d: array of 2d power spectra
        """
        
        outf = open(outstem+filtername+'_results.txt', 'a')
        p2d = np.transpose(p2d, (2, 1, 0))
        
        for layer in noiselayers:
            print('# Noise layer: ', layer)
            
            layer_ind = layers.index(layer)
            print('# Layer index: ', layer_ind)
                  
            ps_1d_i = np.zeros((nradbins, 2))
            for row in range(len(p1d)):
                if p1d[row][3]==layer_ind:
                    ps_1d_i[row-noiselayers.index(layer)*nradbins][0] = p1d[row][0]
                    ps_1d_i[row-noiselayers.index(layer)*nradbins][1] = p1d[row][1]
        
            ps_2d_i = p2d[:,:,noiselayers.index(layer)]

            print('# P2D shape: ', ps_2d_i.shape, ' ps1d shape: ', ps_1d_i.shape)

            
            EW_2D = ew_2d(ps_2d_i)
            print('# EW 2d: ', EW_2D)
            az_an = az_anisotropy(ps_1d_i, layer, 8192, 128, nmcbins=1)
            print('# Az Anisotropy: ', az_an)
                        
            outf.write(blockid + "  " + layer + "   " + str(EW_2D) + "   " + str(az_an) + "\n")

        outf.close()


    eval_results(blockid, ps_1d, ps_2d)
    print('# Results written out to ' + outstem + filtername + '_results.txt')
