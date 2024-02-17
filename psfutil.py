'''
Utilities for PSFs and system matrices.

Classes
-------
OutPSF : Simple target output PSF models.
PSFGrp : Group of PSFs attached to either an InStamp or a Block.
PSFOvl : Overlap between two PSFGrp instances or a PSFGrp instance and itself.
SysMatA : System matrix A attached to a coadd.Block instance.
SysMatB : System matrix B attached to a coadd.Block instance.

'''

import numpy as np
from scipy.special import jv
from astropy import wcs
import matplotlib.pyplot as plt

try:
    from mkl_fft import _numpy_fft as numpy_fft
except:
    import numpy.fft as numpy_fft

from .config import Settings as Stn
from pyimcom_croutines import iD5512C, iD5512C_sym, gridD5512C


class OutPSF:
    '''
    Simple target output PSF models (for testing or outputs).

    Methods
    -------
    psf_gaussian (staticmethod) : Gaussian spot.
    psf_simple_airy (staticmethod) : Airy spot.
    psf_cplx_airy (staticmethod) : somewhat messier Airy function.

    '''

    @staticmethod
    def psf_gaussian(n: int, sigmax: float, sigmay: float) -> np.array:
        '''
        Gaussian spot, n x n, given sigma, centered
        (useful for testing)

        Parameters
        ----------
        n : int
        sigmax : float
        sigmay : float

        Returns
        -------
        np.array, shape : (n, n)

        '''

        y, x = np.mgrid[(1-n)/2/sigmay:(n-1)/2/sigmay:n*1j,
                        (1-n)/2/sigmax:(n-1)/2/sigmax:n*1j]

        I = np.exp(-0.5 * (np.square(x) + np.square(y)))\
            / (2.0 * np.pi * sigmax * sigmay)
        del y, x

        return I

    @staticmethod
    def psf_simple_airy(n: int, ldp: float, obsc: float = 0.0,
                        tophat_conv: float = 0.0, sigma: float = 0.0) -> np.array:
        '''
        Airy spot, n x n, with lambda/D = ldp pixels,
        and convolved with a tophat (square, full width tophat_conv)
        and Gaussian (sigma)
        and linear obscuration factor (obsc)

        Parameters
        ----------
        n : int
        ldp : int
            lambda/D = ldp pixels.
        obsc : float, optional. The default is 0.0.
        tophat_conv : float, optional. The default is 0.0.
        sigma : float, optional. The default is 0.0.

        Returns
        -------
        np.array, shape : (n, n)

        result is centered on (n-1)/2, (n-1)/2 (so on a pixel if
        n is odd and a corner if n is even)

        normalized to *sum* to unity if analytically extended

        '''

        # figure out pad size -- want to get to at least a tophat width and 6 sigmas
        kp = 1 + int(np.ceil(tophat_conv + 6*sigma))
        npad = n + 2*kp

        y, x = np.mgrid[(1-npad)/2:(npad-1)/2:npad*1j,
                        (1-npad)/2:(npad-1)/2:npad*1j]
        r = np.sqrt(np.square(x) + np.square(y)) / ldp  # r in units of ldp

        # make Airy spot
        I = np.square(             jv(0, np.pi*r)      + jv(2, np.pi*r)
                      - obsc**2 * (jv(0, np.pi*r*obsc) + jv(2, np.pi*r*obsc))) \
            / (4.0*ldp**2 * (1-obsc**2)) * np.pi
        del y, x, r

        # now convolve
        It = numpy_fft.rfft2(I)
        uxa = np.linspace(0, 1-1/npad, npad); uxa[-(npad//2):] -= 1
        ux = np.tile(uxa[None, :npad//2+1], (npad, 1))
        uy = np.tile(uxa[:, None], (1, npad//2+1))
        It *= np.exp(-2.0*np.pi**2 * (np.square(ux*sigma) + np.square(uy*sigma))) \
            * np.sinc(ux*tophat_conv) * np.sinc(uy*tophat_conv)
        I = numpy_fft.irfft2(It, s=(npad, npad))
        del It, uxa, ux, uy

        return I[kp:-kp, kp:-kp]

    @staticmethod
    def psf_cplx_airy(n: int, ldp: float, tophat_conv: float = 0.0,
                      sigma: float = 0.0, features: int = 0) -> np.array:
        '''
        somewhat messier Airy function with a few diffraction features printed on
        'features' is an integer that can be added. everything is band limited

        Parameters
        ----------
        n : int
        ldp : float
            lamda/D = ldp pixels.
        tophat_conv : float, optional. The default is 0.0.
        sigma : float, optional. The default is 0.0.
        features : int, optional. The default is 0.

        Returns
        -------
        np.array, shape : (n, n)

        '''

        # figure out pad size -- want to get to at least a tophat width and 6 sigmas
        kp = 1 + int(np.ceil(tophat_conv + 6*sigma))
        npad = n + 2*kp

        y, x = np.mgrid[(1-npad)/2:(npad-1)/2:npad*1j,
                        (1-npad)/2:(npad-1)/2:npad*1j]
        r = np.sqrt(np.square(x) + np.square(y)) / ldp  # r in units of ldp
        phi = np.arctan2(y, x)

        # make modified Airy spot
        L1 = 0.8
        L2 = 0.01
        f = L1 * L2 * 4.0/np.pi
        II = jv(0, np.pi*r) + jv(2, np.pi*r)
        for t in range(6):
            II -= f * np.sinc(L1*r * np.cos(phi + t*np.pi/6.0)) \
                    * np.sinc(L2*r * np.sin(phi + t*np.pi/6.0))
        I = II**2 / (4.0*ldp**2 * (1-6*f)) * np.pi
        del r, phi, II

        if features & 1:  # features % 2 == 1
            rp = np.sqrt(np.square(x-1*ldp) + np.square(y+2*ldp)) / (2.0*ldp)
            II = jv(0, np.pi*rp) + jv(2, np.pi*rp)
            I *= 0.8
            I += 0.2 * II**2 / (4.0*(2.0*ldp)**2) * np.pi
            del rp, II
        del y, x

        if features & 2:  # (features//2) % 2 == 1
            Icopy = np.copy(I)
            I         *= 0.85
            I[:-8, :] += 0.15 * Icopy[8:, :]
            del Icopy

        if features & 4:  # (features//4) % 2 == 1
            Icopy = np.copy(I)
            I           *= 0.8
            I[ :-4, :-4] += 0.1 * Icopy[4:,   4:]
            I[4:,   :-4] += 0.1 * Icopy[ :-4, 4:]
            del Icopy

        # now convolve
        It = numpy_fft.rfft2(I)
        uxa = np.linspace(0, 1-1/npad, npad); uxa[-(npad//2):] -= 1
        ux = np.tile(uxa[None, :npad//2+1], (npad, 1))
        uy = np.tile(uxa[:, None], (1, npad//2+1))
        It *= np.exp(-2.0*np.pi**2 * (np.square(ux*sigma) + np.square(uy*sigma))) \
            * np.sinc(ux*tophat_conv) * np.sinc(uy*tophat_conv)
        I = numpy_fft.irfft2(It, s=(npad, npad))
        del It, uxa, ux, uy

        return I[kp:-kp, kp:-kp]


class PSFGrp:
    '''
    Group of PSFs.

    Either a group of input PSFs attached to an coadd.InStamp instance
        or a group of output PSFs attached to a coadd.Block instance.

    Methods
    -------
    setup (classmethod) : Set up class attributes.
    __init__ : Constructor.
    visualize_psf (staticmethod) : Visualize a PSF.
    _sample_psf : Sample a single PSF or a set of PSFs.
    _build_inpsfgrp : Build a group of input PSFs.
    _get_outpsf (staticmethod) : Get an output PSF specified by configuration.
    _build_outpsfgrp : Build a group of output PSFs.
    accel_pad_and_rfft2 (staticmethod) : Zero-padding and rfft2.
    clear: Free up memory space.

    '''

    # some default settings, will be overwritten by PSFGrp.setup
    nsamp = 383
    nc = nsamp // 2  # 191
    nfft = 768

    @classmethod
    def setup(cls, npixpsf: int = 48, oversamp: int = 8) -> None:
        '''
        Set up class attributes.

        Parameters
        ----------
        npixpsf : int, optional
            Size of PSF postage stamp in native pixels. The default is 48.
        oversamp : int, optional
            PSF oversampling factor relative to native pixel scale. The default is 8.

        Returns
        -------
        None.

        '''

        cls.oversamp = oversamp
        cls.nsamp = npixpsf * oversamp - 1  # 383 by default
        # sampling matrix has shape (..., nsamp, nsamp)
        cls.nc = cls.nsamp // 2  # 191 by default

        # unrotated grid of sampling positions
        # never modified (but frequently referred to) after initialization
        cls.xyo = np.mgrid[(1-cls.nsamp)/2:(cls.nsamp-1)/2:cls.nsamp*1j,
                           (1-cls.nsamp)/2:(cls.nsamp-1)/2:cls.nsamp*1j]
        # dsample (sampling rate of overlap matrices) has been fixed to 1.0.

        # nfft0div8 = 2**(int(np.ceil(np.log2(cls.nsamp-0.5)))-2)
        # cls.nfft = nfft0div8 * int(np.ceil(2*cls.nsamp/nfft0div8))  # 1280 by default
        cls.nfft = npixpsf * oversamp * 2  # 768 by default

    def __init__(self, in_or_out: bool = True,
                 inst: 'coadd.InStamp' = None, blk: 'coadd.Block' = None,
                 verbose: bool = False, visualize: bool = False) -> None:
        '''
        Constructor.

        Parameters
        ----------
        in_or_out : bool, optional
            True if input PSF group, False if output PSF group. The default is True.
        inst : coadd.InStamp, must be provided when in_or_out is True
        blk : coadd.Block, must be provided when in_or_out is False
        visualize : bool, optional
            Whether to visualize PSFs and sampling positions. The default is False.

        Returns
        -------
        None.

        '''

        self.in_or_out = in_or_out

        if in_or_out:  # input PSF group
            assert inst is not None, 'inst must be specified for an input PSF group'
            if verbose:
                print(f'--> building input PSFGrp for InStamp {(inst.j_st, inst.i_st)}',
                      '@', inst.blk.timer(), 's')
            self.inst = inst
            self._build_inpsfgrp(visualize)
            # this produces the following instance attributes:
            # use_inimage, idx_blk2grp, idx_grp2blk, n_psf, psf_arr

        else:  # output PSF group
            assert blk is not None, 'block must be specified for an output PSF group'
            if verbose:
                print(f'--> building output PSFGrp for Block {blk.this_sub=}', '@', blk.timer(), 's')
            self.blk = blk
            self._build_outpsfgrp(visualize)
            # this produces the following instance attributes:
            # n_psf, psf_arr

        self.psf_rft = PSFGrp.accel_pad_and_rfft2(self.psf_arr)
        del self.psf_arr  # remove the PSFs since they are only used for FFT purposes

    @staticmethod
    def visualize_psf(psf_: np.array, xyco: np.array,
                      xctr: float, yctr: float) -> None:
        '''
        Visualize a single PSF, together with sampling positions.

        Parameters
        ----------
        psf_ : np.array, shape : (ny, nx)
            A single PSF.
        xyco : np.array
            Sampling positions, with (0, 0) at the PSF center.
        xctr, yctr : float, float
            Position of the PSF center.

        Returns
        -------
        None.

        '''

        plt.imshow(np.log10(psf_), origin='lower')
        plt.colorbar()
        plt.scatter(xyco[0].ravel()+xctr, xyco[1].ravel()+yctr, s=1e-5)
        plt.show()

    def _sample_psf(self, idx: int, psf: np.array, distort_mat: np.array = None,
                    visualize: bool = False) -> None:
        '''
        Perform interpolations to sample a single PSF or a set of PSFs.

        Parameters
        ----------
        idx : int
            Index of the single PSF (int); sample a set of PSFs at the same time.
        psf : np.array, shape : (n_psf, ny, nx)
            PSF(s) to be sampled.
        distort_mat : np.array, shape : (2, 2)
            Distortion matrix of the original input PSF.
            The default is None, meaning not to rotate the sampling positions.
        visualize : bool, optional
            Whether to visualize PSFs and sampling positions. The default is False.

        Returns
        -------
        None.

        '''

        ny, nx = np.shape(psf)[-2:]
        xctr = (nx-1) / 2.0
        yctr = (ny-1) / 2.0

        # get sampling positions
        if distort_mat is None:
            xyco = PSFGrp.xyo
        else:  # succinct way to rotate
            rotate_mat = np.linalg.inv(distort_mat)
            xyo_ = np.flip(np.moveaxis(PSFGrp.xyo, 0, -1), axis=-1)
            xyco = np.moveaxis(np.flip(xyo_ @ rotate_mat.T, axis=-1), -1, 0)
            del rotate_mat, xyo_

        if visualize:
            if idx is not None:
                PSFGrp.visualize_psf(psf, xyco, xctr, yctr)
            else:  # sampling a group of output PSFs at the same time
                for psf_ in psf:
                    PSFGrp.visualize_psf(psf_, xyco, xctr, yctr)

        if idx is not None:
            out_arr = np.zeros((1, PSFGrp.nsamp**2))
            iD5512C(psf.reshape((1, ny, nx)),
                    xyco[1].ravel()+xctr, xyco[0].ravel()+yctr, out_arr)
            self.psf_arr[idx] = out_arr.reshape((PSFGrp.nsamp, PSFGrp.nsamp))
            del out_arr

        else:  # sample a group of output PSFs at the same time
            out_arr = np.zeros((self.n_psf, PSFGrp.nsamp**2))
            iD5512C(psf, xyco[1].ravel()+xctr, xyco[0].ravel()+yctr, out_arr)
            self.psf_arr = out_arr.reshape((self.n_psf, PSFGrp.nsamp, PSFGrp.nsamp))
            del out_arr

    def _build_inpsfgrp(self, visualize: bool = False) -> None:
        '''
        Build a group of input PSFs.

        Parameters
        ----------
        visualize : bool, optional
            Whether to visualize PSFs and sampling positions. The default is False.

        Returns
        -------
        None.

        '''

        # whether to use each of the input images
        self.use_inimage = np.zeros((self.inst.blk.n_inimage,), dtype=bool)
        for dj_st in range(2):
            for di_st in range(2):
                this_inst = self.inst.blk.instamps[self.inst.j_st+dj_st][self.inst.i_st+di_st]
                self.use_inimage = np.logical_or(self.use_inimage, this_inst.pix_count.astype(bool))
                del this_inst

        # conversion between block index and group index of input images
        self.idx_blk2grp = np.full((self.use_inimage.shape[0],), 255, dtype=np.uint8)
        self.idx_grp2blk = np.full((self.use_inimage.shape[0],), 255, dtype=np.uint8)

        self.n_psf = 0
        for idx_b, use_this in enumerate(self.use_inimage):
            if use_this:
                self.idx_blk2grp[idx_b] = self.n_psf
                self.idx_grp2blk[self.n_psf] = idx_b
                self.n_psf += 1

        blk = self.inst.blk  # shortcut
        psf_compute_point = blk.outwcs.all_pix2world(np.array([self.inst.psf_compute_point_pix]), 0)[0]
        dWdp_out = wcs.utils.local_partial_pixel_derivatives(blk.outwcs, *self.inst.psf_compute_point_pix)
        print('INPUT/PSF computation at RA={:8.4f}, Dec={:8.4f}'.format(*psf_compute_point), end='; ')
        # print(' --> partial derivatives, ', dWdp_out)

        self.psf_arr = np.zeros((self.n_psf, PSFGrp.nsamp, PSFGrp.nsamp))
        for idx in range(self.n_psf):
            # print('PSF info -->', self.idx_grp2blk[idx], end=' ')
            this_psf, distort_matrice = self.inst.blk.inimages[self.idx_grp2blk[idx]].\
                get_psf_and_distort_mat(psf_compute_point, dWdp_out)            
            self._sample_psf(idx, this_psf, distort_matrice, visualize=visualize)
            if visualize:
                print(f'The above PSF is from InImage {(self.inst.blk.inimages[self.idx_grp2blk[idx]].idsca)}',
                      f'at the upper right corner of InStamp {(self.inst.j_st, self.inst.i_st)}')
        print('using input exposures:', [self.idx_grp2blk[idx] for idx in range(self.n_psf)])

    @staticmethod
    def _get_outpsf(outpsf: str = 'AIRYOBSC', extrasmooth: float = 0.0,
                    use_filter: int = 4) -> np.array:
        '''
        Get an output PSF specified by configuration.

        Parameters
        ----------
        outpsf : str, optional
            Target output PSF type. The default is 'AIRYOBSC'.
        extrasmooth : float, optional
            Target output PSF extra smearing. The default is 0.0.
        use_filter : int, optional
            Which filter.

        Returns
        -------
        np.array, shape : (PSFGrp.nsamp+1, PSFGrp.nsamp+1)
            Output PSF specified by configuration.

        '''

        if outpsf == 'GAUSSIAN':
            return OutPSF.psf_gaussian(PSFGrp.nsamp+1,
                extrasmooth * PSFGrp.oversamp, extrasmooth * PSFGrp.oversamp)

        elif outpsf == 'AIRYOBSC':
            return OutPSF.psf_simple_airy(PSFGrp.nsamp+1,
                Stn.QFilterNative[use_filter] * PSFGrp.oversamp,
                obsc=Stn.obsc, tophat_conv=0.0, sigma=extrasmooth * PSFGrp.oversamp)

        elif outpsf == 'AIRYUNOBSC':
            return OutPSF.psf_simple_airy(PSFGrp.nsamp+1,
                Stn.QFilterNative[use_filter] * PSFGrp.oversamp,
                obsc=0.0, tophat_conv=0.0, sigma=extrasmooth * PSFGrp.oversamp)

        else:
            raise RuntimeError('Error: target output PSF type')


    def _build_outpsfgrp(self, visualize: bool = False) -> None:
        '''
        Build a group of output PSFs.

        What output PSFs to include is set by the configuration file.

        Parameters
        ----------
        visualize : bool, optional
            Whether to visualize PSFs and sampling positions. The default is False.

        Returns
        -------
        None.

        '''

        cfg = self.blk.cfg  # shortcut

        self.n_psf = cfg.n_out
        psf_orig = np.zeros((self.n_psf, PSFGrp.nsamp+1, PSFGrp.nsamp+1))
        psf_orig[0] = PSFGrp._get_outpsf(cfg.outpsf, cfg.sigmatarget, cfg.use_filter)

        if self.n_psf > 1:
            for j_out in range(1, self.n_out):
                psf_orig[j_out] = PSFGrp._get_outpsf(
                    cfg.outpsf_extra[j_out-1], cfg.sigmatarget_extra[j_out-1], cfg.use_filter)

        self._sample_psf(None, psf_orig)  # this produces self.psf_arr

        # save the output PSFs to a file
        # OutputPSFHDU = fits.HDUList([fits.PrimaryHDU()] + [fits.ImageHDU(x) for x in psf_orig])
        # for i in range(self.n_psf):
        #     OutputPSFHDU[i+1].header['WHICHPSF'] = 'Output {:d}'.format(i)
        # OutputPSFHDU.writeto(self.blk.outstem+'_outputpsfs.fits', overwrite=True)
        # del OutputPSFHDU

    @staticmethod
    def accel_pad_and_rfft2(psf_arr: np.array) -> np.array:
        '''
        Accelerated version of zero-padding and rfft2.

        For nsamp = 537 and nfft = 1280, this saves
        about 20% of the time compared to simply using rfft2.

        Diagram of the steps involved here:
                                             +---+      +---+
                                             |   | fft  |***|
        +---+ pad  +-------+ rfft +---+ pad  |   | ===> |***|
        |***| ===> |***    | ===> |***| ===> |***|      |***|
        +---+      +-------+      +---+      +---+      +---+
        psf_arr     pad_m1                   pad_m2      res
        Note: m1 stands for axis=-1, and m2 stands for axis=-2. 

        Parameters
        ----------
        psf_arr : np.array, shape : (..., PSFGrp.nsamp, PSFGrp.nsamp)
            PSF array to be padded and Fourier transformed.

        Returns
        -------
        np.array, shape : (..., PSFGrp.nfft, PSFGrp.nfft//2+1)
            Real Fourier transform results of psf_arr.

        '''

        n_arr = psf_arr.shape[0]
        pad_m1 = np.zeros((n_arr, PSFGrp.nsamp, PSFGrp.nfft))
        pad_m2 = np.zeros((n_arr, PSFGrp.nfft, PSFGrp.nfft//2+1), dtype=np.complex128)

        pad_m1[:, :, :PSFGrp.nsamp] = psf_arr
        pad_m2[:, :PSFGrp.nsamp, :] = numpy_fft.rfft(pad_m1, axis=-1)
        res = numpy_fft.fft(pad_m2, axis=-2)
        del pad_m1, pad_m2

        return res

    def clear(self, verbose: bool = False) -> None:
        '''
        Free up memory space.

        Use this in addition to the default __del__ method to make sure that
        the PSF array is removed when it is no longer used.

        Returns
        -------
        None.

        '''

        if self.in_or_out:
            if verbose:
                print(f'--> clearing input PSFGrp attached to InStamp {(self.inst.j_st, self.inst.i_st)}',
                      '@', self.inst.blk.timer(), 's')
            # the following instance attributes should not be removed
            # as they will be referred to in PSFOvl.__call__:
            # self.use_inimage, self.idx_blk2grp, self.idx_grp2blk
        del self.n_psf, self.psf_rft


class PSFOvl:
    '''
    Overlap between two PSFGrp instances or a PSFGrp instance and itself.

    Methods
    -------
    setup (classmethod) : Set up class attribute.
    __init__ : Constructor.
    _idx_square2triangle : Convert a 2-d square index to a 1-d triangle index.
    accel_irfft2_and_extract (staticmethod) : irfft2 and extraction.
    _build_psfovl : Build the PSF overlap array.
    visualize_psfovl : Visualize the PSF overlap array.

    __call__ : Wrapper for the C interpolators.
    _call_ii_cross : Interpolations in the input-input cross-overlap case.
    _call_io_cross : Interpolations in the input-output cross-overlap case.
    _call_ii_self : Interpolations in the input-input self-overlap case.
    clear: Free up memory space.

    '''

    # default setting, will be overwritten by PSFOvl.setup
    flat_penalty = 1e-7

    @classmethod
    def setup(cls, flat_penalty: float = 1e-7) -> None:
        '''
        Set up class attribute.

        Parameters
        ----------
        flat_penalty : float, optional
            Amount by which to penalize having different contributions
            to the output from different input images. The default is 1e-7.

        Returns
        -------
        None.

        '''

        cls.flat_penalty = flat_penalty

    def __init__(self, psfgrp1: PSFGrp, psfgrp2: PSFGrp = None,
                 verbose: bool = False) -> None:
        '''
        Constructor.

        Parameters
        ----------
        psfgrp1 : PSFGrp
            The first PSFGrp instance.
        psfgrp2 : PSFGrp, optional
            The second PSFGrp instance.
            The default is None, indicating the self-overlap of psfgrp1.

        Returns
        -------
        None.

        '''

        self.grp1 = psfgrp1
        self.grp2 = psfgrp2  # None if self-overlap

        if verbose:
            if psfgrp2 is not None:  # cross-overlap
                if psfgrp2.in_or_out:  # input-input cross-overlap
                    print(f'--> building input-input PSFOvl for InStamp {(psfgrp1.inst.j_st, psfgrp1.inst.i_st)}',
                          f'and InStamp {(psfgrp2.inst.j_st, psfgrp2.inst.i_st)}', '@', psfgrp1.inst.blk.timer(), 's')
                else:  # input-output cross-overlap
                    print(f'--> building input-output PSFOvl for InStamp {(psfgrp1.inst.j_st, psfgrp1.inst.i_st)}',
                          f'and Block this_sub={psfgrp2.blk.this_sub}', '@', psfgrp1.inst.blk.timer(), 's')

            else:  # self-overlap
                if psfgrp1.in_or_out:  # input self-overlap
                    print(f'--> building input-self PSFOvl for InStamp {(psfgrp1.inst.j_st, psfgrp1.inst.i_st)}',
                          '@', psfgrp1.inst.blk.timer(), 's')
                else:  # output self-overlap
                    print(f'--> building output-self PSFOvl for Block this_sub={psfgrp1.blk.this_sub}',
                          '@', psfgrp1.blk.timer(), 's')

        self._build_psfovl()  # this produces self.ovl_arr

        # scale conversion for interpolations in PSF overlap arrays
        dtheta = psfgrp1.inst.blk.cfg.dtheta if psfgrp1.in_or_out else psfgrp1.blk.cfg.dtheta
        self.dscale = (Stn.pixscale_native/Stn.arcsec) / (dtheta*3600) / PSFGrp.oversamp

    def _idx_square2triangle(self, idx1: int, idx2: int) -> int:
        '''
        Convert a 2-d square index to a 1-d triangle index.

        The motivation is that, in the input self-overlap case,
        the overlap between psf_arr[idx2] and psf_arr[idx1] is simply
        the flipped version of the overlap between psf_arr[idx1] and psf_arr[idx2].
        Therefore, we only need to compute the latter, and the results
        are stored in a np.array of shape (n_psf*(n_psf+1)//2, nsamp, nsamp).

        For example, in the n_psf == 3 case:
         \idx2  0   1   2
        idx1  +---+---+---+
          0   | 0 | 1 | 2 |
              +---+---+---+
          1   |   | 3 | 4 |
              +---+---+---+
          2   |   |   | 5 |
              +---+---+---+

        Parameters
        ----------
        idx1, idx2 : int, int
            2-d square index.

        Returns
        -------
        int
            1-d triangle index.

        '''

        assert idx1 <= idx2, 'the two indices must satisfy idx1 <= idx2'
        return (2*self.grp1.n_psf-idx1+1)*idx1//2 + idx2-idx1

    @staticmethod
    def accel_irfft2_and_extract(ovl_rft: np.array) -> np.array:
        '''
        Accelerated version of irfft2 and extraction.

        This acceleration is based on the fact that we only need
        less than (1/2) x (1/2) of the irfft2 results.
        For nsamp = 537 and nfft = 1280, this saves about 40%
        of the time compared to simply using irfft2 and ifftshift.

        Note: Because of the ifftshift part, this function is *not*
        the inverse function of PSFGrp.accel_pad_and_rfft2.

        Diagram of the steps involved here (ext. stands for extraction):
        +---+      +---+
        |***| ifft |***|
        |***| ===> |***| ext. +---+ irfft+-------+ ext. +---+
        |***|      |***| ===> |***| ===> |*******| ===> |***|
        +---+      +---+      +---+      +-------+      +---+
        ovl_rft    ift_m2     ovl_m2      ift_m1        ovl_m1
        Note: m2 stands for axis=-2, and m1 stands for axis=-1. 

        Parameters
        ----------
        ovl_rft : np.array, shape : (..., PSFGrp.nfft, PSFGrp.nfft//2+1)
            Real Fourier transform of the PSF overlap array we want.

        Returns
        -------
        np.array, shape : (..., nsamp, nsamp)
            The PSF overlap array we want.

        '''

        n_arr = ovl_rft.shape[0]
        ovl_m2 = np.zeros((n_arr, PSFGrp.nsamp, PSFGrp.nfft//2+1), dtype=np.complex128)
        ovl_m1 = np.zeros((n_arr, PSFGrp.nsamp, PSFGrp.nsamp))
        nc = PSFGrp.nc  # shortcut

        ift_m2 = numpy_fft.ifft(ovl_rft, axis=-2)
        ovl_m2[:,   :nc, :] = ift_m2[:, -nc:    , :]
        ovl_m2[:, nc:  , :] = ift_m2[:,    :nc+1, :]
        del ift_m2

        ift_m1 = numpy_fft.irfft(ovl_m2, axis=-1, n=PSFGrp.nfft)
        ovl_m1[:, :,   :nc] = ift_m1[:, :, -nc:    ]
        ovl_m1[:, :, nc:  ] = ift_m1[:, :,    :nc+1]
        del ovl_m2, ift_m1

        return ovl_m1

    def _build_psfovl(self) -> None:
        '''
        Build the PSF overlap array.

        Returns
        -------
        None.

        '''

        if self.grp2 is not None:  # cross-overlap
            self.ovl_arr = np.zeros((self.grp1.n_psf, self.grp2.n_psf, PSFGrp.nsamp, PSFGrp.nsamp))

            for idx in range(self.grp1.n_psf):
                ovl_rft = self.grp1.psf_rft[idx] * self.grp2.psf_rft.conjugate()
                self.ovl_arr[idx] = PSFOvl.accel_irfft2_and_extract(ovl_rft)
                del ovl_rft

        elif self.grp1.in_or_out:  # input self-overlap
            n_psf = self.grp1.n_psf  # shortcut
            self.ovl_arr = np.zeros((n_psf*(n_psf+1)//2, PSFGrp.nsamp, PSFGrp.nsamp))

            for idx in range(n_psf):
                start = self._idx_square2triangle(idx, idx)
                ovl_rft = self.grp1.psf_rft[idx] * self.grp1.psf_rft[idx:].conjugate()
                self.ovl_arr[start:start+n_psf-idx] = PSFOvl.accel_irfft2_and_extract(ovl_rft)
                del ovl_rft

        else:  # output self-overlap
            ovl_rft = self.grp1.psf_rft * self.grp1.psf_rft.conjugate()
            # we do not need overlaps between different output PSFs
            self.ovl_arr = PSFOvl.accel_irfft2_and_extract(ovl_rft)
            del ovl_rft

            # extract C value(s)
            self.outovlc = self.ovl_arr[:, PSFGrp.nc, PSFGrp.nc]
            del self.ovl_arr  # move this to PSFOvl.clear
            # if the entire output self-overlap array is needed

    def visualize_psfovl(self) -> None:
        '''
        Visualize the PSF overlap array.

        Returns
        -------
        None.

        '''

        if self.grp2 is not None:  # cross-overlap
            n_psf1, n_psf2 = self.ovl_arr.shape[:2]  # shortcuts
            fig, axs = plt.subplots(
                n_psf1, n_psf2, figsize=(3.2*min(n_psf2, 4), 2.4*min(n_psf1, 4)))

            for idx1 in range(n_psf1):
                for idx2 in range(n_psf2):
                    if n_psf1 > 1:
                        if n_psf2 > 1: ax = axs[idx1, idx2]
                        else: ax = axs[idx1]
                    else:
                        if n_psf2 > 1: ax = axs[idx2]
                        else: ax = axs

                    im = ax.imshow(np.log10(self.ovl_arr[idx1, idx2]), origin='lower')
                    plt.colorbar(im)

            plt.show()
            del fig, axs
        
        else:  # self overlap
            n_psf = self.grp1.n_psf  # shortcut

            if n_psf == 1:
                plt.imshow(np.log10(self.ovl_arr[0]), origin='lower')
                plt.colorbar()
                plt.show()

            else:
                fig, axs = plt.subplots(
                    n_psf, n_psf, figsize=(3.2*min(n_psf, 4), 2.4*min(n_psf, 4)))

                for idx1 in range(n_psf):
                    for idx2 in range(n_psf):
                        if idx2 < idx1:
                            axs[idx1, idx2].axis('off')
                            continue

                        im = axs[idx1, idx2].imshow(np.log10(
                            self.ovl_arr[self._idx_square2triangle(idx1, idx2)]), origin='lower')
                        plt.colorbar(im)

                plt.show()
                del fig, axs

    def __call__(self, st1: 'coadd.InStamp',
                 st2: 'coadd.InStamp, coadd.OutStamp, or None' = None,
                 visualize: bool = False) -> np.array:
        '''
        Wrapper for the C interpolators.

        Parameters
        ----------
        st1 : coadd.InStamp
        st2 : coadd.InStamp, coadd.OutStamp, or None
            The default is None, indicating diagonal blocks of the A matrix.
        visualize : bool, optional
            Whether to visualize overlap arrays and sampling positions. The default is False.

        Returns
        -------
        np.array, of which the shape depends on
        the nature of this PSFOvl instance and the input

        '''

        if self.grp2 is not None:  # cross-overlap
            if self.grp2.in_or_out:  # input-input cross-overlap
                return self._call_ii_cross(st1, st2, visualize)
            else:  # input-output cross-overlap
                return self._call_io_cross(st1, st2, visualize)

        else:  # input self-overlap
            assert self.grp1.in_or_out, f'{self.grp1.in_or_out=} in a self-overlap case'
            # this method never deals with output self-overlap!
            return self._call_ii_self(st1, st2, visualize)

    def _call_ii_cross(self, st1: 'coadd.InStamp', st2: 'coadd.InStamp',
                       visualize: bool = False) -> np.array:
        '''
        Interpolations in the input-input cross-overlap case.

        Parameters
        ----------
        st1 : coadd.InStamp
        st2 : coadd.InStamp
        visualize : bool, optional
            Whether to visualize overlap arrays and sampling positions. The default is False.

        Returns
        -------
        np.array, shape : (st1.pix_cumsum[-1], st2.pix_cumsum[-1])

        '''

        res = np.zeros((st1.pix_cumsum[-1], st2.pix_cumsum[-1]))
        ddx = st1.x_val[:, None] - st2.x_val[None, :]
        ddx /= self.dscale; ddx += PSFGrp.nc
        ddy = st1.y_val[:, None] - st2.y_val[None, :]
        ddy /= self.dscale; ddy += PSFGrp.nc

        n_psf1, n_psf2 = self.ovl_arr.shape[:2]
        if visualize:
            fig, axs = plt.subplots(n_psf1, n_psf2, figsize=(6.4*n_psf2, 4.8*n_psf1))
        n_in = (n_psf1 * n_psf2) ** 0.5  # for flat_penalty

        for j_im in range(st1.blk.n_inimage):
            if st1.pix_count[j_im] == 0: continue
            for i_im in range(st2.blk.n_inimage):
                if st2.pix_count[i_im] == 0: continue

                slice_ = np.s_[st1.pix_cumsum[j_im]:st1.pix_cumsum[j_im+1],
                               st2.pix_cumsum[i_im]:st2.pix_cumsum[i_im+1]]

                if visualize:
                    if n_psf1 > 1:
                        if n_psf2 > 1: ax = axs[self.grp1.idx_blk2grp[j_im], self.grp2.idx_blk2grp[i_im]]
                        else: ax = axs[self.grp1.idx_blk2grp[j_im]]
                    else:
                        if n_psf2 > 1: ax = axs[self.grp2.idx_blk2grp[i_im]]
                        else: ax = axs

                    im = ax.imshow(np.log10(self.ovl_arr[
                        self.grp1.idx_blk2grp[j_im], self.grp2.idx_blk2grp[i_im]]), origin='lower')
                    plt.colorbar(im)
                    ax.scatter(ddx[slice_].ravel(), ddy[slice_].ravel(), s=1e-2)

                out_arr = np.zeros((1, st1.pix_count[j_im] * st2.pix_count[i_im]))
                iD5512C(self.ovl_arr[self.grp1.idx_blk2grp[j_im], self.grp2.idx_blk2grp[i_im]].\
                        reshape((1, PSFGrp.nsamp, PSFGrp.nsamp)),
                        ddx[slice_].ravel(), ddy[slice_].ravel(), out_arr)

                res[slice_] = out_arr.reshape((st1.pix_count[j_im], st2.pix_count[i_im]))
                del out_arr

                # flat penalty
                if PSFOvl.flat_penalty != 0.0:
                    res[slice_] -= PSFOvl.flat_penalty / n_in
                    if j_im == i_im: res[slice_] += PSFOvl.flat_penalty
                del slice_

        if visualize:
            plt.show()
            del fig, axs

        del ddx, ddy

        return res

    def _call_io_cross(self, st1: 'coadd.InStamp', st2: 'coadd.OutStamp',
                       visualize: bool = False) -> np.array:
        '''
        Interpolations in the input-output cross-overlap case.

        Parameters
        ----------
        st1 : coadd.InStamp
        st2 : coadd.OutStamp
        visualize : bool, optional
            Whether to visualize overlap arrays and sampling positions. The default is False.

        Returns
        -------
        np.array, shape : either (self.grp2.n_psf, n_outpix, st1.pix_cumsum[-1])
                              or (self.grp2.n_psf, n_outpix, selection.shape[0])

        '''

        x_val_, y_val_ = st1.x_val, st1.y_val
        pix_count_ = st1.pix_count
        pix_cumsum_ = st1.pix_cumsum
        n_outpix = np.prod(st2.yx_val.shape[-2:])

        selection = st2.selections[(st1.j_st-st2.j_st+1) * 3 + (st1.i_st-st2.i_st+1)]
        if selection is None:
            res = np.zeros((self.grp2.n_psf, n_outpix, st1.pix_cumsum[-1]))
        else:
            x_val_ = x_val_[selection]
            y_val_ = y_val_[selection]
            pix_cumsum_ = np.searchsorted(selection, st1.pix_cumsum)
            pix_count_ = np.diff(pix_cumsum_)
            res = np.zeros((self.grp2.n_psf, n_outpix, selection.shape[0]))

        ddx = x_val_[:, None] - st2.yx_val[None, 1, 0, :]
        ddx /= self.dscale; ddx += PSFGrp.nc
        ddy = y_val_[:, None] - st2.yx_val[None, 0, :, 0]
        ddy /= self.dscale; ddy += PSFGrp.nc

        if visualize:
            n_psf1, n_psf2 = self.ovl_arr.shape[:2]
            fig, axs = plt.subplots(n_psf1, n_psf2, figsize=(6.4*n_psf2, 4.8*n_psf1))

        for i_psf in range(self.grp2.n_psf):
            for j_im in range(st1.blk.n_inimage):
                if st1.pix_count[j_im] == 0: continue

                if visualize:
                    if n_psf2 == 1:
                        if n_psf1 > 1: ax = axs[self.grp1.idx_blk2grp[j_im]]
                        else: ax = axs
                    else:
                        if n_psf1 > 1: ax = axs[self.grp1.idx_blk2grp[j_im], i_psf]
                        else: ax = axs[i_psf]

                    im = ax.imshow(np.log10(self.ovl_arr[self.grp1.idx_blk2grp[j_im], i_psf]), origin='lower')
                    plt.colorbar(im)
                    ddx_ = st1.x_val[:, None] - st2.yx_val[1, ::10, ::10].ravel()[None, :]
                    ddx_ /= self.dscale; ddx_ += PSFGrp.nc
                    ddy_ = st1.y_val[:, None] - st2.yx_val[0, ::10, ::10].ravel()[None, :]
                    ddy_ /= self.dscale; ddy_ += PSFGrp.nc
                    ax.scatter(ddx_.ravel(), ddy_.ravel(), s=1e-2, c='r')
                    del ddx_, ddy_

                out_arr = np.zeros((pix_count_[j_im], n_outpix))
                gridD5512C(self.ovl_arr[self.grp1.idx_blk2grp[j_im], i_psf],
                           ddx[pix_cumsum_[j_im]:pix_cumsum_[j_im+1], :],
                           ddy[pix_cumsum_[j_im]:pix_cumsum_[j_im+1], :], out_arr)
                res[i_psf, :, pix_cumsum_[j_im]:pix_cumsum_[j_im+1]] = out_arr.T
                del out_arr

        if visualize:
            plt.show()
            del fig, axs

        del x_val_, y_val_, pix_count_, pix_cumsum_
        del selection, ddx, ddy

        return res

    def _call_ii_self(self, st1: 'coadd.InStamp', st2: 'coadd.InStamp',
                      visualize: bool = False) -> np.array:
        '''
        Interpolations in the input-input self-overlap case.

        Note that the self-overlap of an input PSF group can be used to compute
        either the block-diagonal submatrices for a single input stamp
            or the submatrices for a pair of input stamps within a 2x2 group.

        Parameters
        ----------
        st1 : coadd.InStamp
        st2 : coadd.InStamp
            The default is None, indicating diagonal blocks of the A matrix.
        visualize : bool, optional
            Whether to visualize overlap arrays and sampling positions. The default is False.

        Returns
        -------
        np.array, shape : (st1.pix_cumsum[-1], st2.pix_cumsum[-1])

        '''

        same_inst = st2 is None
        if same_inst: st2 = st1
        res = np.zeros((st1.pix_cumsum[-1], st2.pix_cumsum[-1]))
        ddx = st1.x_val[:, None] - st2.x_val[None, :]
        ddy = st1.y_val[:, None] - st2.y_val[None, :]

        if visualize:
            for arr in [ddx, ddy]:
                plt.imshow(arr, origin='upper')
                plt.colorbar()
                plt.show()

        ddx /= self.dscale; ddx += PSFGrp.nc
        ddy /= self.dscale; ddy += PSFGrp.nc

        if visualize:
            n_psf = self.grp1.n_psf  # shortcut
            fig, axs = plt.subplots(n_psf, n_psf, figsize=(6.4*n_psf, 4.8*n_psf))

        for j_im in range(st1.blk.n_inimage):
            if st1.pix_count[j_im] == 0: continue
            for i_im in range(0 if not same_inst else j_im, st2.blk.n_inimage):
                if st2.pix_count[i_im] == 0: continue

                if j_im <= i_im:
                    ovl_arr_ = self.ovl_arr[self._idx_square2triangle(
                        self.grp1.idx_blk2grp[j_im], self.grp1.idx_blk2grp[i_im])]
                else:
                    ovl_arr_ = np.flip(self.ovl_arr[self._idx_square2triangle(
                        self.grp1.idx_blk2grp[i_im], self.grp1.idx_blk2grp[j_im])])

                slice_ = np.s_[st1.pix_cumsum[j_im]:st1.pix_cumsum[j_im+1],
                               st2.pix_cumsum[i_im]:st2.pix_cumsum[i_im+1]]

                if visualize:
                    ax = axs[self.grp1.idx_blk2grp[j_im], self.grp1.idx_blk2grp[i_im]] if n_psf > 1 else axs
                    im = ax.imshow(np.log10(ovl_arr_), origin='lower')
                    plt.colorbar(im)
                    ax.scatter(ddx[slice_].ravel(), ddy[slice_].ravel(), s=1e-2)

                out_arr = np.zeros((1, st1.pix_count[j_im] * st2.pix_count[i_im]))
                interpolator = iD5512C_sym if same_inst and j_im == i_im else iD5512C
                interpolator(ovl_arr_.reshape((1, PSFGrp.nsamp, PSFGrp.nsamp)),
                             ddx[slice_].ravel(), ddy[slice_].ravel(), out_arr)

                res[slice_] = out_arr.reshape((st1.pix_count[j_im], st2.pix_count[i_im]))

                # flat penalty
                if PSFOvl.flat_penalty != 0.0:
                    res[slice_] -= PSFOvl.flat_penalty / self.grp1.n_psf
                    if j_im == i_im: res[slice_] += PSFOvl.flat_penalty

                if same_inst and j_im < i_im:
                    res[st2.pix_cumsum[i_im]:st2.pix_cumsum[i_im+1],
                        st1.pix_cumsum[j_im]:st1.pix_cumsum[j_im+1]] = res[slice_].T
                del out_arr, slice_

        if visualize:
            if n_psf > 1:
                for j_im in range(st1.blk.n_inimage):
                    if st1.pix_count[j_im] == 0: continue
                    for i_im in range(j_im):
                        if st2.pix_count[i_im] == 0: continue
                        axs[self.grp1.idx_blk2grp[j_im], self.grp1.idx_blk2grp[i_im]].axis('off')

            plt.show()
            del fig, axs

        del ddx, ddy

        return res

    def clear(self, verbose: bool = False) -> None:
        '''
        Free up memory space.

        Use this in addition to the default __del__ method to make sure that
        the huge PSF overlap array is removed when it is no longer used.

        Returns
        -------
        None.

        '''

        if verbose:
            if self.grp2 is not None and not self.grp2.in_or_out:
                print(f'--> clearing input-output PSFOvl for InStamp {(self.grp1.inst.j_st, self.grp1.inst.i_st)}',
                      f'and Block this_sub={self.grp2.blk.this_sub}', '@', self.grp1.inst.blk.timer(), 's')

        del self.grp1, self.grp2, self.ovl_arr


class SysMatA:
    '''
    System matrix A attached to a coadd.Block instance.

    The symtem matrix A is defined in Rowe+ 2011 Equation (17).

    Methods
    -------
    __init__ : Constructor.
    ji_st2psf (staticmethod) : Convert InStamp index to PSFGrp index.
    shift_ji_st (staticmethod) : Tool function for tuple addition.
    iisubmat_dist (staticmethod) : Calculate the index for iisubmats_ref.
    _compute_iisubmats : Make input-input PSFOvl and compute A submatrices.
    get_iisubmat : Return a requested A submatrix.

    '''

    def __init__(self, blk: 'coadd.Block') -> None:
        '''
        Constructor.

        Parameters
        ----------
        blk : coadd.Block
            The Block instance to which this SysMatA instance is attached.

        Returns
        -------
        None.

        '''

        self.blk = blk

        # dictionary of A submatrices; ii stands for input-input
        self.iisubmats = {}
        # reference counts for the A submatrices
        # see the static method iisubmat_dist defined below for why 13
        self.iisubmats_ref = np.zeros((blk.cfg.n1P+2, blk.cfg.n1P+2, 13), dtype=np.uint8)

    @staticmethod
    def ji_st2psf(ji_st: (int, int)) -> (int, int):
        '''
        Convert InStamp index to PSFGrp index.

        More precisely, convert index of the InStamp of interest
        to that of its neighbor to which the PSFGrp in that 2x2 group is attached.
        For example, any element of {(0, 0), (0, 1), (1, 0), (1, 1)} -> (0, 0).

        Parameters
        ----------
        ji_st : (int, int)
            Index of the InStamp of interest.

        Returns
        -------
        (int, int)
            Index of the InStamp which harbors the PSFGrp.

        '''

        return tuple(ji >> 1 << 1 for ji in ji_st)

    @staticmethod
    def shift_ji_st(ji_st: (int, int), dji_st: (int, int)) -> (int, int):
        '''
        Tool function for tuple addition.

        For our purposes, this function shifts an InStamp index
        in each direction by either 0 or 1.

        Parameters
        ----------
        ji_st : (int, int)
            InStamp index to be shifted.
        dji_st : (int, int)
            Shift in InStamp index.

        Returns
        -------
        (int, int)
            Shifted InStamp index.

        '''

        return (ji_st[0]+dji_st[0], ji_st[1]+dji_st[1])

    @staticmethod
    def iisubmat_dist(ji_st1: (int, int), ji_st2: (int, int)) -> (int, int, int):
        '''
        Calculate the index for iisubmats_ref.

        Specifically, calculate the "distance" between two input postage stamps
        in order to store the reference counts in the 3-d array iisubmats_ref
        using indices (ji_st1[0], ji_st1[1], dist), which this static method returns.

        The "distance" is defined as follows:
        +---+---+---+---+---+
        | 8 | 9 | 10| 11| 12|
        +---+---+---+---+---+
        | 3 | 4 | 5 | 6 | 7 |
        +---+---+---+---+---+
        |   |   | 0*| 1 | 2 |
        +---+---+---+---+---+
        where "*" denotes ji_st1. Note that ji_st1 must precede ji_st2.
        If the "distance" is out of range, this function returns None.

        Parameters
        ----------
        ji_st1 : (int, int)
            Index of the first InStamp.
        ji_st2 : (int, int)
            Index of the second InStamp.

        Returns
        -------
        (int, int, int)
            Index of the first InStamp combined with the "distance,"
            which can be directly used as the index of iisubmats_ref;
        or None
            when the "distance" is out of range.

        '''

        # this should never happen by design
        assert ji_st1 <= ji_st2, f'{ji_st1=} should precede {ji_st2=}'

        dj_st = ji_st2[0] - ji_st1[0]
        if not 0 <= dj_st <= 2: return None
        di_st = ji_st2[1] - ji_st1[1]
        if abs(di_st) > 2: return None
        if dj_st == 0 and di_st < 0: return None

        dist = dj_st*5 + di_st+2 - 2
        return (*ji_st1, dist)

    def _compute_iisubmats(self, ji_st1: (int, int), ji_st2: (int, int),
                           sim_mode: bool = False, verbose: bool = False) -> None:
        '''
        Make input-input PSFOvl and compute A submatrices.

        This method is only called in SysMatA.get_iisubmat.
        When an A submatrix is requested from an OutStamp but not in iisubmats,
        get_iisubmat calls this method, which makes the input-input PSFOvl,
        computes all the needed A submatrices which are supposed to be computed
        using that particular PSFOvl instance, and clears the PSFOvl.

        Note that removing the huge PSF overlap arrays is important!
        Otherwise pyimcom would demand much more memory.

        To determine whether a specific system submatrix is needed,
        a Block instance first runs the postage coaddition framework
        in sim_mode to count references to all possible system submatrices.

        Parameters
        ----------
        ji_st1 : (int, int)
            Index of the first InStamp.
        ji_st2 : (int, int)
            Index of the second InStamp.
        sim_mode : bool, optional
            Whether to count references without actually computing submatrices.
            The default is False.

        Returns
        -------
        None.

        '''

        # identify InStamp instance(s) to which the input PSFGrp instance(s) is(are) attached
        ji_psf1 = SysMatA.ji_st2psf(ji_st1)
        ji_psf2 = SysMatA.ji_st2psf(ji_st2)

        psfgrp1 = self.blk.instamps[ji_psf1[0]][ji_psf1[1]].get_inpsfgrp(sim_mode)
        if ji_psf1 != ji_psf2:
            psfgrp2 = self.blk.instamps[ji_psf2[0]][ji_psf2[1]].get_inpsfgrp(sim_mode)
        else:
            psfgrp2 = None

        if not sim_mode:
            # make the input-input PSFOvl instance
            iipsfovl = PSFOvl(psfgrp1, psfgrp2)

        counter = 0

        # all 4 possible InStamp's in the 2x2 group of psfgrp1
        for dji_st1 in range(4):
            ji_st1_ = SysMatA.shift_ji_st(ji_psf1, divmod(dji_st1, 2))

            # all 4 possible InStamp's in the 2x2 group of psfgrp2
            for dji_st2 in range(4):
                ji_st2_ = SysMatA.shift_ji_st(ji_psf2, divmod(dji_st2, 2))

                # SysMatA.iisubmat_dist requires the first index to precede the second
                ji_st_pair = (ji_st1_, ji_st2_) if ji_st1_ <= ji_st2_ else (ji_st2_, ji_st1_)
                ji_dist = SysMatA.iisubmat_dist(*ji_st_pair)

                # don't compute this A submatrix if the distance
                # between these two InStamp's is out of range,
                # or if this A submatrix will never be used
                if ji_dist is None or (not sim_mode and self.iisubmats_ref[ji_dist] == 0): continue

                if ji_st_pair not in self.iisubmats:
                    if sim_mode:
                        self.iisubmats[ji_st_pair] = None
                    else:
                        my_submat = iipsfovl(self.blk.instamps[ji_st1_[0]][ji_st1_[1]],
                                             self.blk.instamps[ji_st2_[0]][ji_st2_[1]] \
                                             if ji_st1_ != ji_st2_ else None)

                        # although ji_psf1 always precedes ji_psf2, sometimes ji_st2_ precedes ji_st1_
                        # for example, ji_psf1 = (0, 0), ji_psf2 = (0, 2), ji_st1_ = (1, 0), ji_st2_ = (0, 2)
                        # in such (relatively rare) cases, we need to transpose the submatrix
                        if ji_st1_ <= ji_st2_:
                            self.iisubmats[ji_st_pair] = my_submat
                        else:
                            self.iisubmats[ji_st_pair] = my_submat.T

                    counter += 1

        if not sim_mode:  # remove the large PSFGrp and PSFOvl arrays and declare victory
            if self.blk.instamps[ji_psf1[0]][ji_psf1[1]].inpsfgrp_ref == 0:
                psfgrp1.clear()
            if ji_psf1 != ji_psf2 and self.blk.instamps[ji_psf2[0]][ji_psf2[1]].inpsfgrp_ref == 0:
                psfgrp2.clear()
            del psfgrp1, psfgrp2

            iipsfovl.clear(); del iipsfovl
            if verbose:
                print(f'--> finished computing {counter} input-input submatrices', '@', self.blk.timer(), 's')

    def get_iisubmat(self, ji_st1: (int, int), ji_st2: (int, int),
                     sim_mode: bool = False, ji_st_out: (int, int) = None) -> np.array:
        '''
        Return the requested A submatrix.

        This is the only public interface of this class.
        For the sim_mode, see the docstring of SysMatA._compute_iisubmats.

        Parameters
        ----------
        ji_st1 : (int, int)
            Index of the first InStamp.
        ji_st2 : (int, int)
            Index of the second InStamp.
        sim_mode : bool, optional
            Whether to count references without actually computing submatrices.
            The default is False.
        ji_st_out : (int, int)
            Index of the OutStamp. Needed for virtual memory.

        Returns
        -------
        np.array, shape : (st1.pix_cumsum[-1], st2.pix_cumsum[-1])
            The requested A submatrix.

        '''

        assert ji_st1 <= ji_st2, f'{ji_st1=} should precede {ji_st2=}'
        ji_dist = SysMatA.iisubmat_dist(ji_st1, ji_st2)
        assert ji_dist is not None, f'distance between InStamps {ji_st1} and {ji_st2} is out of range'

        if sim_mode:
            self.iisubmats_ref[ji_dist] += 1
            if (ji_st1, ji_st2) not in self.iisubmats:
                self._compute_iisubmats(ji_st1, ji_st2, sim_mode)
            return

        if (ji_st1, ji_st2) not in self.iisubmats:
            # load virtual memory when available
            fname = 'iisubmat_' + '_'.join(f'{ji:02d}' for ji in ji_st1 + ji_st2) + '.npy'
            fpath = self.blk.cache_dir / fname
            if fpath.exists():
                # print(f'VIRMEM: loading {fname}')
                self.iisubmats[(ji_st1, ji_st2)] = np.load(str(fpath))
                fpath.unlink(); del fname, fpath
            else:
                self._compute_iisubmats(ji_st1, ji_st2, sim_mode)
        arr = self.iisubmats[(ji_st1, ji_st2)]

        self.iisubmats_ref[ji_dist] -= 1
        if self.iisubmats_ref[ji_dist] == 0:
            # remove this A submatrix from iisubmats if it will never be referred to again
            del self.iisubmats[(ji_st1, ji_st2)]
        elif ji_st_out is not None and (ji_st_out[0] % 2 == 0) \
            and (ji_st_out[1] == min(ji_st1[1], ji_st2[1]) + 1):
            # save virtual memory when needed
            fname = 'iisubmat_' + '_'.join(f'{ji:02d}' for ji in ji_st1 + ji_st2) + '.npy'
            fpath = self.blk.cache_dir / fname  #; print(f'VIRMEM: saving {fname}')
            with open(str(fpath), 'wb') as f: np.save(f, arr)
            del self.iisubmats[(ji_st1, ji_st2)], fname, fpath

        return arr


class SysMatB:
    '''
    System matrix B attached to a coadd.Block instance.

    The symtem matrix B is defined in Rowe+ 2011 Equation (18).
    IMPORTANT: The -2 coefficient is NOT included in this program.

    Methods
    -------
    __init__ : Constructor.
    get_iosubmat : Return a requested B submatrix.

    '''

    def __init__(self, blk: 'coadd.Block') -> None:
        '''
        Constructor.

        Parameters
        ----------
        blk : coadd.Block
            The Block instance to which this SysMatB instance is attached.

        Returns
        -------
        None.

        '''

        self.blk = blk

        # dictionary of input-output PSFOvl's; io stands for input-output
        # in the case of the system matrix B, storing PSFOvl instances
        # is more economical than storing submatrices (which are never reused)
        self.iopsfovls = {}
        # reference counts for the input-output PSFOvl's
        self.iopsfovls_ref = np.zeros((blk.cfg.n1P//2+1, blk.cfg.n1P//2+1), dtype=np.uint8)

    def get_iosubmat(self, ji_st_in: (int, int), ji_st_out: (int, int),
                     sim_mode: bool = False) -> np.array:
        '''
        Return a requested B submatrix.

        Note that this is the courterpart of the combination of
        SysMatA.get_iisubmat and the _compute_iisubmats method behind it,
        since SysMatB is much simpler than SysMatA.

        For the sim_mode, see the docstring of SysMatA._compute_iisubmats.

        Parameters
        ----------
        ji_st_in : (int, int)
            Index of the InStamp.
        ji_st_out : (int, int)
            Index of the OutStamp.
        sim_mode : bool, optional
            Whether to count references without actually computing submatrices.
            The default is False.

        Returns
        -------
        np.array, shape : same as PSFOvl._call_io_cross
            The requested B submatrix.

        '''

        assert max(abs(ji_st_in[0] - ji_st_out[0]), abs(ji_st_in[1] - ji_st_out[1])) <= 1, \
            f'distance between InStamp {ji_st_in} and OutStamp {ji_st_out} is out of range'

        # identify InStamp instance to which the input PSFGrp instance is attached
        ji_st_inpsf = SysMatA.ji_st2psf(ji_st_in)
        # since iopsfovls_ref is an "shrunk" array, we need a conversion here
        inpsf_key = tuple(ji >> 1 for ji in ji_st_inpsf)

        if sim_mode: self.iopsfovls_ref[inpsf_key] += 1
        if inpsf_key not in self.iopsfovls:
            # get the input PSFGrp array and make the input-output PSFOvl instance
            inpsfgrp = self.blk.instamps[ji_st_inpsf[0]][ji_st_inpsf[1]].get_inpsfgrp(sim_mode)
            self.iopsfovls[inpsf_key] = PSFOvl(inpsfgrp, self.blk.outpsfgrp) if not sim_mode else None

            # remove the input PSFGrp array if it will never be referred to again
            if not sim_mode and self.blk.instamps[ji_st_inpsf[0]][ji_st_inpsf[1]].inpsfgrp_ref == 0:
                inpsfgrp.clear()
            del inpsfgrp
        if sim_mode: return

        self.iopsfovls_ref[inpsf_key] -= 1
        iosubmat = self.iopsfovls[inpsf_key](self.blk.instamps [ji_st_in [0]][ji_st_in [1]],
                                             self.blk.outstamps[ji_st_out[0]][ji_st_out[1]])

        # clear this input-output PSFOvl from iopsfovls if it will never be referred to again
        if self.iopsfovls_ref[inpsf_key] == 0:
            self.iopsfovls[inpsf_key].clear(); del self.iopsfovls[inpsf_key]
        return iosubmat
