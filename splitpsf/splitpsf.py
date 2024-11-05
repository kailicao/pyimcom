import sys
import numpy as np
from astropy.io import fits
import scipy
import scipy.signal
from scipy.ndimage import gaussian_filter
from scipy.special import roots_legendre, eval_legendre
from astropy import wcs
import copy
import json
import os

from ..config import Settings

class SplitPSF:

   @staticmethod
   def Window_integratedBlackman(x):
      """Integrated Blackman window.

      Returns 1 if x>1; 0 if x<-1; interpolates between these.

      x is a numpy array.
      """

      alpha = 0.08
      return np.where(x>=1, 1., np.where(x<=-1, 0.,
        0.5*(x+1) + (0.5*np.sin(np.pi*x) + alpha/4*np.sin(2*np.pi*x))/((1-alpha)*np.pi)
        ))

   @staticmethod
   def Window_2D_integratedBlackman(n, r1, r2):
      """2D version of integrated Blackman.

      Inputs:
         n = side length of array
         r1 = inner radius of filter (pixels)
         r2 = outer radius of filter (pixels)

      Returns:
        arr = 2D image of filter, center at ((n-1)/2,(n-1)/2)
      """

      X_ = np.linspace((1-n)/2.,(n-1)/2.,n)
      xx,yy = np.meshgrid(X_,X_)
      r = np.sqrt(xx**2+yy**2)
      arr = SplitPSF.Window_integratedBlackman(-1. + 2./(r2-r1)*(r2-r))
      return arr

   @staticmethod
   def Truncate_2D_integratedBlackman(n, m):
      """2D version of integrated Blackman.

      Inputs:
         n = side length of array
         m = number of pixels to truncate at the side

      Returns:
        arr = 2D image of filter, center at ((n-1)/2,(n-1)/2)
      """

      if m==0: return np.ones((n,n)) # special case

      X_ = np.ones((n,))
      X_[:m] = SplitPSF.Window_integratedBlackman(np.linspace(-1.,1.,m+2))[1:-1]
      X_[-m:] = X[m-1::-1]
      return np.outer(X_,X_)

   @staticmethod
   def padft2d(arr):
      """2D pad and Fourier transform for a square array"""
      n = np.shape(arr)[1]
      arr_double = np.zeros((2*n,2*n), dtype=arr.dtype)
      arr_double[:n,:n] = arr
      arr_double = np.roll(arr_double, -(n//2), axis=0)
      arr_double = np.roll(arr_double, -(n//2), axis=1)
      ft = np.fft.fftshift(np.fft.fft2(arr_double.astype(np.complex128)))
      return ft[n//2:-(n//2), n//2:-(n//2)]

   @staticmethod
   def tophatfilter(inArray, tophatwidth):
      """Smooth 3D array in each of the last 2 planes with a tophat of the given width"""

      npad = int(np.ceil(tophatwidth))
      npad += (4-npad) % 4  # make a multiple of 4
      (nplane, ny, nx) = np.shape(inArray)
      nyy = ny+npad*2
      nxx = nx+npad*2
      outArray = np.zeros((nplane, nyy, nxx))
      outArray[:,npad:-npad, npad:-npad] = inArray
      outArrayFT = np.fft.fft2(outArray)

      # convolution
      uy = np.linspace(0, nyy-1, nyy)/nyy
      uy = np.where(uy > .5, uy-1, uy)
      ux = np.linspace(0, nxx-1, nxx)/nxx
      ux = np.where(ux > .5, ux-1, ux)
      s = np.sinc(ux[None, :]*tophatwidth)*np.sinc(uy[:, None]*tophatwidth)
      outArrayFT = outArrayFT * s[None,:,:]

      outArray = np.real(np.fft.ifft2(outArrayFT))
      if npad>0: outArray = outArray[:,npad:-npad,npad:-npad]
      return outArray

   @staticmethod
   def gauss_deconv(arr, C, eps=1e-3):
      """Deconvolve a Gaussian, matrix C is 2x2 covariance (in pixel units), epsilon=cutoff"""

      n = np.shape(arr)[1]
      arr_double = np.zeros((2*n,2*n), dtype=arr.dtype)
      arr_double[:n,:n] = arr
      ft = np.fft.fft2(arr_double.astype(np.complex128))
      u_ = np.linspace(0,2*n-1,2*n)/(2*n)
      u_[n:] = u_[n:]-1
      u,v = np.meshgrid(u_,u_)
      GaussWin = np.exp(-2*np.pi**2*(C[0,0]*u**2+C[1,1]*v**2+2*C[0,1]*u*v))
      ft = ft*GaussWin/(GaussWin**2 + eps**2)
      W = np.fft.ifft2(ft).real.astype(arr.dtype)
      return W[:n,:n]

   @staticmethod
   def gauss_stamp(n, C):
      """Makes nxn array of a Gaussian with given covariance, centered at the image center.

      n should be even. Covariance is in pixel units.
      """

      X_ = np.linspace((1-n)/2.,(n-1)/2.,n)
      xx,yy = np.meshgrid(X_,X_)
      detC = C[0,0]*C[1,1]-C[0,1]**2
      iC = np.array([[C[1,1], -C[0,1]],[-C[0,1],C[0,0]]])/detC
      return np.exp(-.5*(iC[0,0]*xx**2 + iC[1,1]*yy**2) - iC[0,1]*xx*yy)/(2*np.pi*np.sqrt(detC))

   def __init__(self, psfcube, wcs_, pars):
      """Class constructor to generate a split PSF from a Legendre cube file.

      Inputs:
         psfcube = a PSF data cube in Legendre polynomial format. Each slice is a PSF image,
            that should be multiplied by P_m(u_) P_n(v_) where flatten((n,m)) = range((order+1)**2)
            where (u_, v_) are the coordinates on the SCA

         wcs_ = WCS associated with the image. if None, then no distortion

         pars = dictionary of parameters. The choices (and defaults if not specified) are:
            ref_pixscale = 0.11 (arcsec) -> reference native pixel scale (without distortion)
            oversamp = 8 -> oversampling of PSF relative to native scale
            tophat_in = False -> pixel tophat already included
            smallstamp_size = [side length of psfcube] -> size of small PSF postage stamp, in units of the oversampled pixels
            nside = 4088 -> SCA side length
            r_in = 4.0 -> inner cut radius in native pixels
            r_out = 9.0 -> inner cut radius in native pixels
            sigmaGamma = 1.0 -> 1 sigma scale length of the desired output PSF, in reference input pixels
            eps = 0.02 -> regularization parameter in Gaussian deconvolution of the PSF wings
            m_trunc = 0 -> truncation window width at edge of PSF postage stamp (in units of oversampled pixels)

      A constraint is that PSF input and output sizes are even numbers of oversampled pixels, with (0,0)
      at the center of the array.
      """

      self.ref_pixscale = 0.11
      if 'ref_pixscale' in pars: self.ref_pixscale = pars['ref_pixscale']
      self.oversamp = 8
      if 'oversamp' in pars: self.oversamp = pars['oversamp']
      self.tophat_in = False
      if 'tophat_in' in pars: self.tophat_in = pars['tophat_in']
      self.smallstamp_size = self.largestamp_size = np.shape(psfcube)[1]
      if 'smallstamp_size' in pars: self.smallstamp_size = pars['smallstamp_size']
      self.nside = 4088
      if 'nside' in pars: self.nside = pars['nside']
      self.r_in = 4.0
      if 'r_in' in pars: self.r_in = pars['r_in']
      self.r_out = 9.0
      if 'r_out' in pars: self.r_out = pars['r_out']
      self.sigmaGamma = 1.0
      if 'sigmaGamma' in pars: self.sigmaGamma = pars['sigmaGamma']
      self.eps = 0.02
      if 'eps' in pars: self.eps = pars['eps']
      self.m_trunc = 0
      if 'm_trunc' in pars: self.m_trunc = pars['m_trunc']

      if self.tophat_in:
         self.psfcube = np.copy(psfcube) # copy ensures the same reference behavior in both casees
      else:
         self.psfcube = SplitPSF.tophatfilter(psfcube, self.oversamp)

      self.wcs_ = wcs_

      # Get Legendre order
      self.npoly = np.shape(psfcube)[0]
      self.lorder = 0
      while (self.lorder+1)**2<self.npoly: self.lorder += 1

      # Checks
      if self.smallstamp_size%2!=0 or self.largestamp_size%2!=0: raise "SplitPSF requires even dimension"
      if (self.lorder+1)**2!=self.npoly: raise "SplitPSF Legendre polynomial dimension error"

   def build(self):
      """Builds the short/long range decomposition for this SplitPSF."""

      # Long/short range split
      W = SplitPSF.Window_2D_integratedBlackman(self.largestamp_size, self.oversamp*self.r_in, self.oversamp*self.r_out)
      ntrim = (self.largestamp_size-self.smallstamp_size)//2
      self.smallpsf = W[None,:,:]*self.psfcube
      if ntrim>0: self.smallpsf = self.smallpsf[:,ntrim:-ntrim,ntrim:-ntrim]
      resid = self.psfcube*(1-W)[None,:,:]*SplitPSF.Truncate_2D_integratedBlackman(self.largestamp_size, self.m_trunc)[None,:,:]

      # select grid points for the conversion.
      # we want int_{-1}^{+1} int_{-1}^{+1} dx dy f(x,y) approx sum_i w_i f(x_i,y_i)
      # wg, xg, yg are numpy arrays of length self.npoly
      (xLegendre,wLegendre) = roots_legendre(self.lorder+1)
      xg, yg = np.meshgrid(xLegendre,xLegendre)
      xg = xg.flatten(); yg = yg.flatten()
      wg = np.outer(wLegendre,wLegendre).flatten()

      # The covariance matrix of the desired Gaussian Gamma (can be done outside the for loop)
      var_ref = (self.oversamp * self.sigmaGamma)**2

      # Now do the de-projection in each grid cell.
      self.K_Legendre = np.zeros((self.npoly, self.largestamp_size, self.largestamp_size))
      self.K_real = np.zeros((self.npoly, self.largestamp_size, self.largestamp_size))
      self.zeta_real = np.zeros((self.npoly, self.largestamp_size, self.largestamp_size))
      self.Cov = np.zeros((self.npoly, 2, 2))
      for i in range(self.npoly):
         if self.wcs_ is None:
            self.Cov[i,:,:] = var_ref*np.identity(2)
         else:
            compute_point_pix = [self.nside/2.*(1+xg[i]), self.nside/2.*(1+yg[i])]
            globalpos = self.wcs_.all_pix2world(np.array([compute_point_pix]), 0)[0]
            jac = wcs.utils.local_partial_pixel_derivatives(self.wcs_, *compute_point_pix)
            g = np.array([[np.cos(globalpos[1]*np.pi/180)**2,0],[0,1]]) # metric tensor
            self.Cov[i,:,:] = var_ref * np.linalg.inv(jac.T@g@jac) * (self.ref_pixscale/3600)**2

         # get Legendre polynomial weights for this point (length self.npoly)
         lpw = np.outer(eval_legendre(range(self.lorder+1),yg[i]), eval_legendre(range(self.lorder+1),xg[i])).flatten()

         locLRP = np.einsum('a,aij->ij',lpw,resid)
         self.K_real[i,:,:] = SplitPSF.gauss_deconv(locLRP, self.Cov[i,:,:], eps=self.eps)
         self.zeta_real[i,:,:] = locLRP - scipy.signal.convolve(self.K_real[i,:,:], SplitPSF.gauss_stamp(self.largestamp_size, self.Cov[i,:,:]), mode='same', method='fft')

         # Convert back to Legendre space --- do this with the current coefficients
         self.K_Legendre += wg[i] * np.tensordot(lpw, self.K_real[i,:,:], axes=0)

      # end for i

      # normalize the Legendre coefficients, i.e. multiply by (2l_x+1)/2 * (2l_y+1)/2
      l_ = np.array(range(self.lorder))+.5
      lnorm = np.outer(l_,l_).flatten()
      self.K_Legendre = self.K_Legendre * lnorm[:,None,None]

def split_psf_to_fits(psf_file, wcs_format, pars, outfile):
   """Computes split PSFs from an input PSF file.

   Inputs:
      psf_file = the PSF Legendre polynomial file as input (FITS file, primary and then 1 HDU per SCA)
      wcs_format = WCS file format. should be able to generate a file with the SCA header in the path
         wcs_format.format(sca) (sca = 1..18, inclusive)
         missing files will have 'None' WCS (ignore distortion)
      pars = PSF splitting parameters
      outfile = output file for the PSF.

   The format of the file written is as follows:
   """

   psf_hdulist = fits.open(psf_file)

   # Generate the primary HDU
   prim = fits.PrimaryHDU()
   prim.header['FROMFILE'] = psf_file
   for copykeys in ['CFORMAT', 'PORDER', 'ABSCISSA', 'NCOEF', 'SEQ', 'OBSID', 'NSCA', 'OVSAMP', 'SIMRUN']: 
      if copykeys in psf_hdulist[0].header:
         prim.header[copykeys] = psf_hdulist[0].header[copykeys]
         prim.header.comments[copykeys] = psf_hdulist[0].header.comments[copykeys]
   if 'NSCA' in psf_hdulist[0].header:
      nsca = int(psf_hdulist[0].header['NSCA'])
   else:
      nsca = len(psf_hdulist)-1
      prim.header['NSCA'] = (nsca, 'from input file')
   prim.header['GSSKIP'] = (nsca, 'number of HDUs to skip for short range PSF')
   prim.header['KERSKIP'] = (2*nsca, 'number of HDUs to skip for Kernel')
   savezeta = False
   if 'SAVEZETA' in pars:
      if pars['SAVEZETA']:
         prim.header['ZETASKIP'] = (3*nsca, 'number of HDUs to skip for zeta')
         savezeta = True
   prim.header['COMMENT'] = 'SplitPSF file. Original PSF in HDUs {:d}..{:d}'.format(1,nsca)
   prim.header['COMMENT'] = 'Short range PSF in HDUs {:d}..{:d}'.format(nsca+1,2*nsca)
   prim.header['COMMENT'] = 'Long-range kernel in HDUs {:d}..{:d}'.format(2*nsca+1,3*nsca)

   # build the HDUs for each SCA
   shortrangepsfs = []
   kernels = []
   zetas = []
   zetamax = np.zeros((nsca,))
   truewcs = np.zeros((nsca,), dtype=np.bool_)
   Kint = np.zeros((nsca,))
   K2int = np.zeros((nsca,))
   for isca in range(1,nsca+1):
      try:
         with fits.open(wcs_format.format(isca)) as f:
            this_wcs_ = wcs.WCS(f['SCI'].header)
         prim.header['INWCS{:02d}'.format(isca)] = wcs_format.format(isca)
      except:
         prim.header['INWCS{:02d}'.format(isca)] = '/dev/null'
         this_wcs_ = None

      sp = SplitPSF(psf_hdulist[isca].data.astype(np.float64), this_wcs_, pars)
      sp.build()

      # make the 'short range' image HDU
      x = fits.ImageHDU(sp.smallpsf.astype(np.float32))
      x.header['IMTYPE'] = 'Short range PSF'
      x.header['SCA'] = isca
      shortrangepsfs += [x]

      # make the 'kernel' HDU
      y = fits.ImageHDU(sp.K_Legendre.astype(np.float32))
      y.header['IMTYPE'] = 'Kernel K'
      y.header['SCA'] = isca
      if this_wcs_ is None:
         y.header['TRUEWCS'] = (False, 'No WCS available, ignored distortion')
         truewcs[isca-1] = False
      else:
         y.header['TRUEWCS'] = (True, 'Used WCS from file')
         truewcs[isca-1] = True
      zetamax[isca-1] = np.amax(np.abs(sp.zeta_real))
      y.header['MAXZETA'] = (zetamax[isca-1], 'maximum error |zeta|')
      Kint[isca-1] = np.sum(sp.K_Legendre[0,:,:])/sp.oversamp**2
      K2int[isca-1] = np.sum(sp.K_Legendre[0,:,:]**2)/sp.oversamp**2
      y.header['KINT'] = (Kint[isca-1], 'integral of K kernel')
      y.header['K2INT'] = (K2int[isca-1], 'integral of K^2 (native pix^-2)')
      kernels += [y]

      # the 'zeta' HDU (not currently written)
      z = fits.ImageHDU(sp.zeta_real.astype(np.float32))
      zetas += [z]

      del sp

   # report global worst zeta
   prim.header['MAXZETA'] = np.amax(zetamax)

   if savezeta:
      prim.header['SAVEZETA'] = True
   else:
      prim.header['SAVEZETA'] = False
      zetas = []

   hdulist = fits.HDUList([prim] + psf_hdulist[1:nsca+1] + shortrangepsfs + kernels + zetas)
   hdulist.writeto(outfile, overwrite=True)

   psf_hdulist.close()

   # tell the user which T/F values there were in the WCS
   print('WCS:',truewcs)
   print('zetamax:',zetamax)
   print('Kint:',Kint)
   print('K2int:',K2int)

# ### MAIN DRIVER ### #

if __name__ == "__main__":

   """Call with python3 -m pyimcom.splitpsf [config_file]"""

   # Extract the information we need from the config file
   with open(sys.argv[1]) as f:
      cfg_dict = json.load(f)
   print("Configuration file:\n")
   print(cfg_dict)
   print('')

   if not ('INLAYERCACHE' in cfg_dict): raise "Couldn't find INLAYERCACHE."

   # get target PSF properties
   if cfg_dict['OUTPSF'] != 'GAUSSIAN': raise "SplitPSF currently only works for Gaussians."
   sigma = float(cfg_dict['EXTRASMOOTH'])
   print('PSF sigma (input pixels) -->', sigma)

   # get number of rows
   with fits.open(cfg_dict['OBSFILE']) as f:
      Nobs = f[1].header['NAXIS2']
      filters_obs = f[1].data['filter']
   print(Nobs, 'observations to search')
   print(filters_obs)

   # extract oversampling factor
   ovsamp = float(cfg_dict['INPSF'][2])
   print('Input PSFs are {:f}x oversampled'.format(ovsamp))

   # extract PSF splitting parameters
   r1 = float(cfg_dict['PSFSPLIT'][0])
   r2 = float(cfg_dict['PSFSPLIT'][1])
   epsilon = float(cfg_dict['PSFSPLIT'][2])
   print(r1,r2,epsilon)

   # decide on stamp size; multiple of 8, must include r2 radius
   smallstampsize = int(np.ceil(r2*ovsamp*2 + 4))
   smallstampsize += 8-smallstampsize%8
   print('chosen stamp size = ', smallstampsize)

   # where to put the files
   targetdir = cfg_dict['INLAYERCACHE'] + '.psf'
   try:
      os.mkdir(targetdir)
      print('made directory -->', targetdir)
   except OSError as error:
      print("Couldn't make directory", targetdir, ':', error)

   use_filter = Settings.RomanFilters[int(cfg_dict['FILTER'])]
   print('selecting from filter', use_filter)
   print('')

   count = 0
   for iobs in range(Nobs):

      # different file name options depending on the simulation type
      if cfg_dict['INPSF'][1] == 'anlsim':
         psf_filename = '/psf_polyfit_{:d}.fits'.format(iobs)
      else:
         raise "unrecognized input data type"
      #
      # and for science data (just used here for WCS)
      #
      if cfg_dict['INDATA'][1] == 'anlsim':
         sci_filename = cfg_dict['INDATA'][0] + '/simple/Roman_WAS_simple_model_{:s}_{:d}_'.format(filters_obs[iobs], iobs) + '{:d}.fits'
      else:
         raise "unrecognized input data type"

      psf_file = cfg_dict['INPSF'][0] + psf_filename
      if os.path.exists(psf_file) and filters_obs[iobs]==use_filter:

         # Need to transfer this file
         outfile = targetdir + '/psf_{:d}.fits'.format(iobs)
         print('{:8d}/{:8d} found, file is at '.format(iobs,Nobs) + psf_file, '-->', outfile)
         print('   sci in =', sci_filename)
         split_psf_to_fits(psf_file, sci_filename, {
            'smallstamp_size': smallstampsize,
            'sigmaGamma': sigma,
            'r_in': r1,
            'r_out': r2,
            'eps': epsilon,
            'SAVEZETA': False
         }, outfile)
         # <-- 'SAVEZETA': True is for diagnostics/figures only. The zeta HDUs are not actually needed for the calculation,
         # and you might want to keep it off to save space.

         sys.stdout.flush()
         count = count+1
         print('exposure counter =', count)
         print('')
         # if count==1: exit() # <-- for testing: exit after one file
