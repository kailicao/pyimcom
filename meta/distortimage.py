import numpy as np
import os
from astropy.io import fits
from astropy import wcs
from . import ginterp

from ..config import Config, Settings
from ..diagnostics.outimage_utils.helper import HDU_to_bels

class MetaMosaic:
    """
    Contains mosaic information for use in meta operations.

    Methods
    -------
    __init__ : Constructor.

    to_file : Writes the mosaic object to a FITS file.
    shearimage : Generate a sheared image.

    """

    def __init__(self, fname, verbose=False):
        """Build from an example file.
        """

        with fits.open(fname) as f:
            c = f['CONFIG'].data['text']
            n = len(c)
            cf = ''
            for j in range(n):
                cf += c[j]+'\n'
            self.nlayer = np.shape(f['PRIMARY'].data)[-3]
            self.im_dtype = f['PRIMARY'].data.dtype
        self.cfg = Config(cf)

        # get the file coordinates
        self.LegacyName = False
        self.stem = fname[:-11]; tail = fname[-11:]
        if fname[-9:]=='_map.fits':
             self.LegacyName = True
             self.stem = fname[:-15]; tail = fname[-15:]
        self.ix = int(tail[1:3])
        self.iy = int(tail[4:6])

        # build maps
        self.Nside = 3*self.cfg.n1*self.cfg.n2
        self.in_image = np.zeros((self.nlayer, self.Nside, self.Nside), dtype=self.im_dtype)
        self.in_fidelity = np.zeros((self.Nside, self.Nside), dtype=np.float32)
        self.in_noise = np.zeros((self.Nside, self.Nside), dtype=np.float32)

        # Load the data. A lot of logic in here to pull out which portions we need
        # (including handling of boundary effects)
        xpad = [self.ix==0, self.ix==self.cfg.nblock-1]
        ypad = [self.iy==0, self.iy==self.cfg.nblock-1]
        for dx in range(-1,2):
            cx = self.cfg.n1*self.cfg.n2*(1+dx) - self.cfg.postage_pad*self.cfg.n2
            sxmin = self.cfg.postage_pad*self.cfg.n2
            sxmax = sxmin + self.cfg.n1*self.cfg.n2
            if xpad[0]: sxmin -= self.cfg.postage_pad*self.cfg.n2
            if xpad[1]: sxmax += self.cfg.postage_pad*self.cfg.n2
            if cx+sxmin<0: sxmin=-cx
            if cx+sxmax>self.Nside: sxmax=self.Nside-cx
            for dy in range(-1,2):
                cy = self.cfg.n1*self.cfg.n2*(1+dy) - self.cfg.postage_pad*self.cfg.n2
                symin = self.cfg.postage_pad*self.cfg.n2
                symax = symin + self.cfg.n1*self.cfg.n2
                if ypad[0]: symin -= self.cfg.postage_pad*self.cfg.n2
                if ypad[1]: symax += self.cfg.postage_pad*self.cfg.n2
                if cy+symin<0: symin=-cy
                if cy+symax>self.Nside: symax=self.Nside-cy

                # Now get this input image if it is within the mosaic
                in_x = self.ix+dx
                in_y = self.iy+dy
                if in_x>=0 and in_x<self.cfg.nblock and in_y>=0 and in_y<self.cfg.nblock:
                    in_fname = self.stem + '_{:02d}_{:02d}'.format(in_x,in_y)
                    if self.LegacyName: in_fname += '_map'
                    in_fname += '.fits'
                    if verbose:
                        print('IN {:2d},{:2d} [{:4d}:{:4d},{:4d}:{:4d}] offset x={:5d} y={:5d}'.format(in_x, in_y, symin, symax, sxmin, sxmax, cx, cy))
                        print('  <<', in_fname)

                    with fits.open(in_fname) as f:
                        # the map
                        self.in_image[:,symin+cy:symax+cy,sxmin+cx:sxmax+cx] = f['PRIMARY'].data[0,:,symin:symax,sxmin:sxmax]
                        # fidelity, converted to dB
                        self.in_fidelity[symin+cy:symax+cy,sxmin+cx:sxmax+cx] = f['FIDELITY'].data[0,symin:symax,sxmin:sxmax].astype(np.float32)\
                            * HDU_to_bels(f['FIDELITY'])/.1
                        # noise, converted to dB
                        self.in_noise[symin+cy:symax+cy,sxmin+cx:sxmax+cx] = f['SIGMA'].data[0,symin:symax,sxmin:sxmax].astype(np.float32)\
                            * HDU_to_bels(f['SIGMA'])/.1

    def to_file(self, fname):
        """Writes the input mosaic images to a FITS file.
        """

        # generate the WCS
        outwcs = wcs.WCS(naxis=2)
        outwcs.wcs.crpix = [.5 - self.cfg.Nside*(self.ix-1-self.cfg.nblock//2), .5 - self.cfg.Nside*(self.iy-1-self.cfg.nblock//2)]
        outwcs.wcs.cdelt = [-self.cfg.dtheta, self.cfg.dtheta]
        outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        outwcs.wcs.crval = [self.cfg.ra, self.cfg.dec]

        # make the HDUs
        primary = fits.PrimaryHDU(self.in_image, header=outwcs.to_header())
        fidelity = fits.ImageHDU(self.in_fidelity, header=outwcs.to_header())
        noise = fits.ImageHDU(self.in_noise, header=outwcs.to_header())

        primary.header['SOURCE'] = 'pyimcom.meta.distortimage.MetaMosaic.to_file'
        primary.header['IMTYPE'] = '3x3 block, undistorted'
        fidelity.header['UNIT'] = 'dB'
        noise.header['UNIT'] = 'dB'

        fits.HDUList([primary, fidelity, noise]).writeto(fname, overwrite=True)

    def shearimage(self, N, jac=None, psfgrow=1., oversamp=1., fidelity_min=30., Rsearch=6., verbose=False):
        """Generates a sheared image and its WCS.

        Inputs:
        N = size of the output image (shape will be (N,N))
        jac = 2x2 Jacobian for transformation (None defaults to the identity)
        psfgrow = factor (in linear scale) by which to grow the PSF
        oversamp = up-sampling factor (e.g., 1 = preserve pixel scale)
        fidelity_min = fidelity cut (in dB) for which pixels to use
        Rsearch = search radius in interpolation
        verbose = talk to STDOUT

        Returns:
        im = image dictionary containing:
            im['image'] = image cube (3D)
            im['mask'] = image mask (2D, Boolean, True=masked)
            im['wcs'] = WCS object (appropriate for a FITS file)
            im['pars'] = parameter dictionary (can be turned into a FITS header)
            im['layers'] = list of layers

        Comments:
        The sense of jac is that the *output* is related to the *input* by:
        d{input coords} = J d{output coords}

        Assumes a Gaussian PSF.

        """

        # check PSF type
        if self.cfg.outpsf != 'GAUSSIAN':
            raise ValueError('shearimage: only works on GAUSSIAN, received '+self.cfg.outpsf)

        # Figure out the geometrical mapping. First the scale:
        if jac is None:
            J = np.identity(2)
        else:
            J = np.asarray(jac)
        J_orig = np.copy(J)
        J = J/oversamp
        scale = self.cfg.dtheta
        # ... and now the projection center in block coordinates (Q)
        Q_orig = np.asarray([self.cfg.nblock/2 - self.ix - .5, self.cfg.nblock/2 - self.iy - .5]) * self.cfg.n1 * self.cfg.n2
        Q_new = np.linalg.solve(J,Q_orig)
        xref = np.round(Q_new[0] + 1e-7)+.5 + N/2 # rounds to nearest half integer for even N
        yref = np.round(Q_new[1] + 1e-7)+.5 + N/2

        # origin position in the input array
        opos = J@np.asarray([1-xref,1-yref]) # recall the lower-left corner is (1,1) in FITS
        opos[0] += (self.cfg.nblock/2 - self.ix + 1) * self.cfg.n1 * self.cfg.n2 - .5
        opos[1] += (self.cfg.nblock/2 - self.iy + 1) * self.cfg.n1 * self.cfg.n2 - .5
        # the "-0.5" is because the lower-left corner of the pixel is at (-.5,-.5)
        # and have the +1 since the lower-left corner of the image is in the block (ix-1,iy-1)

        # generate the WCS
        outwcs = wcs.WCS({
            'CTYPE1': "RA---STG",
            'CUNIT1': 'deg',
            'CRPIX1': xref,
            'NAXIS1': N,
            'CTYPE2': "DEC--STG",
            'CUNIT2': 'deg',
            'CRPIX2': yref,
            'NAXIS2': N,
            'CD1_1': -J[0,0]*scale,
            'CD1_2': -J[0,1]*scale,
            'CD2_1':  J[1,0]*scale,
            'CD2_2':  J[1,1]*scale,
            'CRVAL1': self.cfg.ra,
            'CRVAL2': self.cfg.dec
        })

        #input mask
        inmask = self.in_fidelity<fidelity_min

        # get smearing information
        sigma = self.cfg.sigmatarget * Settings.pixscale_native * (180./np.pi) / self.cfg.dtheta
            # recall: pixscale_native is in radians, but dtheta is in degrees, hence the conversion
        dCov = sigma**2 * ( psfgrow**2*J_orig@J_orig.T - np.identity(2) )
        C = [dCov[0,0], dCov[0,1], dCov[1,1]]

        if verbose:
            print('Q_orig', Q_orig)
            print('Q_new', Q_new)
            print('opos', opos)
            print('sigmatarget', self.cfg.sigmatarget, 'dtheta', self.cfg.dtheta, 'pixscale_native', Settings.pixscale_native)
            print('sigma', sigma)
            print('C', C)

        image, mask, Umax, Smax = ginterp.MultiInterp(self.in_image, inmask, (N,N), opos, J, Rsearch, sigma*np.sqrt(8*np.log(2)), C) 
            # could add kappa, deweight

        # SVD of the Jacobian
        z = J_orig[0,0]+J_orig[1,1]+1j*(J_orig[1,0]-J_orig[0,1])
        cpd = np.abs(z)
        apx = np.angle(z)
        z = J_orig[0,0]-J_orig[1,1]+1j*(J_orig[1,0]+J_orig[0,1])
        cmd = np.abs(z)
        amx = np.angle(z)
        Eig1 = (cpd+cmd)/2.
        Eig2 = (cpd-cmd)/2.
        alpha = (apx+amx)/2.
        mu = 1./(Eig1*Eig2)
        eta = -np.log(Eig1/Eig2)
        eta1 = eta*np.cos(2*alpha)
        eta2 = eta*np.sin(2*alpha)
        g1 = np.tanh(eta/2.)*np.cos(2*alpha)
        g2 = np.tanh(eta/2.)*np.sin(2*alpha)
        conv = 1. - (Eig1+Eig2)/2.

        pardict = {
            'STEM': (self.stem, 'stem for file name'),
            'BLOCKX': (self.ix, 'x block index'),
            'BLOCKY': (self.iy, 'y block index'),
            'UMAX': (Umax, 'interp - max leakage (square norm)'),
            'SMAX': (Smax, 'interp - max noise (square norm)'),
            'JXX': (J_orig[0,0], 'Jacobian x_in, x_out'),
            'JXY': (J_orig[0,1], 'Jacobian x_in, y_out'),
            'JYX': (J_orig[1,0], 'Jacobian y_in, x_out'),
            'JYY': (J_orig[1,1], 'Jacobian y_in, y_out'),
            'COVXX': (C[0], 'smoothing covariance xx'),
            'COVXY': (C[1], 'smoothing covariance xy'),
            'COVYY': (C[2], 'smoothing covariance yy'),
            'SIGMAOUT': (self.cfg.sigmatarget * Settings.pixscale_native * (180./np.pi) * 3600 * psfgrow, 'arcsec'),
            'PIXSCALE': (self.cfg.dtheta*3600/oversamp, 'arcsec'),
            'OVERSAMP': (oversamp, 'oversampling implemented in shearimage'),
            'MU': (mu, 'amplification applied'),
            'ETA1': (eta1, 'shear component 1'),
            'ETA2': (eta2, 'shear component 2'),
            'JROTATE': (apx, 'rotation angle, CCW in-->out, radians'),
            'G1': (g1, 'reduced shear component 1'),
            'G2': (g2, 'reduced shear component 2'),
            'CONV': (conv, 'convergence kappa')
        }

        return {'image':image, 'mask':mask, 'wcs':outwcs, 'pars':pardict, 'layers':self.cfg.extrainput}

def shearimage_to_fits(im, fname, layers=None, overwrite=False):
    """utility to save a shearimage dictionary a FITS file"""

    # which layers to use?
    nlayer = np.shape(im['image'])[-2]
    use_layers = layers
    if layers is None:
        use_layers = range(nlayer)
    use_layers = np.asarray(use_layers).astype(np.int16)

    H1 = fits.PrimaryHDU(im['image'][use_layers,:,:], header=im['wcs'].to_header())
    H1.header['SOURCE'] = 'pyimcom.meta.distortimage.shearimage_to_fits'
    for p in im['pars'].keys():
        H1.header[p] = im['pars'][p]
    for q in range(len(use_layers)):
        qst = 'LAYER{:03d}'.format(q+1)
        st = im['layers'][q]
        if st is None: st = '__SCI__'
        H1.header[qst] = (st, 'layer {:3d} here, was {:3d} in original'.format(use_layers[q]+1, q+1))
    H2 = fits.ImageHDU(im['mask'].astype(np.uint8))
    fits.HDUList([H1,H2]).writeto(fname, overwrite=overwrite)
