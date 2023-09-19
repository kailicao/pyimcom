import numpy as np
from astropy.io import fits
from scipy.special import jv
import matplotlib.pyplot as plt

from config import Settings as Stn
from pyimcom_croutines import iD5512C, iD5512C_sym, gridD5512C


class OutPSF:
    '''Simple PSF models (for testing or outputs).
    '''

    @staticmethod  # NOT TESTED
    def psf_gaussian(n, sigmax, sigmay):
        '''Gaussian spot, n x n, given sigma, centered
        (useful for testing)
        '''

        y, x = np.mgrid[(1-n)/2:(n-1)/2:n*1j,
                        (1-n)/2:(n-1)/2:n*1j]

        I = np.exp(-.5*(x**2/sigmax**2+y**2/sigmay**2)) / \
            (2.*np.pi*sigmax*sigmay)

        return I

    @staticmethod
    def psf_simple_airy(n, ldp, obsc=0., tophat_conv=0., sigma=0.):
        '''Airy spot, n x n, with lamda/D = ldp pixels,
        and convolved with a tophat (square, full width tophat_conv)
        and Gaussian (sigma)
        and linear obscuration factor (obsc)

        result is centered on (n-1)/2,(n-1)/2 (so on a pixel if
        n is odd and a corner if n is even)

        normalized to *sum* to unity if analytically extended
        '''

        # figure out pad size -- want to get to at least a tophat width and 6 sigmas
        kp = 1 + int(np.ceil(tophat_conv + 6*sigma))
        npad = n + 2*kp

        y, x = np.mgrid[(1-npad)/2:(npad-1)/2:npad*1j,
                        (1-npad)/2:(npad-1)/2:npad*1j]
        r = np.sqrt(x**2+y**2) / ldp  # r in units of ldp

        # make Airy spot
        I = (jv(0, np.pi*r)+jv(2, np.pi*r)
             - obsc**2*(jv(0, np.pi*r*obsc) + jv(2, np.pi*r*obsc))
             )**2 / (4.*ldp**2*(1-obsc**2)) * np.pi

        # now convovle
        It = np.fft.rfft2(I)
        uxa = np.linspace(0, 1-1/npad, npad)
        uxa[-(npad//2):] -= 1
        ux = np.tile(uxa[None, :npad//2+1], (npad, 1))
        uy = np.tile(uxa[:, None], (1, npad//2+1))
        It *= np.exp(-2*np.pi**2*sigma**2*(ux**2+uy**2)) * \
            np.sinc(ux*tophat_conv) * np.sinc(uy*tophat_conv)
        I = np.real(np.fft.irfft2(It))

        return I[kp:-kp, kp:-kp]

    @staticmethod  # NOT TESTED
    def psf_cplx_airy(n, ldp, tophat_conv=0., sigma=0., features=0):
        '''somewhat messier Airy function with a few diffraction features printed on
        'features' is an integer that can be added. everything is band limited
        '''

        # figure out pad size -- want to get to at least a tophat width and 6 sigmas
        kp = 1 + int(np.ceil(tophat_conv + 6*sigma))
        npad = n + 2*kp

        y, x = np.mgrid[(1-npad)/2:(npad-1)/2:npad*1j,
                        (1-npad)/2:(npad-1)/2:npad*1j]
        r = np.sqrt(x**2+y**2) / ldp  # r in units of ldp
        phi = np.arctan2(y, x)

        # make modified Airy spot
        L1 = .8
        L2 = .01
        f = L1*L2*4./np.pi
        II = jv(0, np.pi*r)+jv(2, np.pi*r)
        for t in range(6):
            II -= f*np.sinc(L1*r*np.cos(phi+t*np.pi/6.)) * \
                np.sinc(L2*r*np.sin(phi+t*np.pi/6.))
        I = II**2 / (4.*ldp**2*(1-6*f)) * np.pi

        if features % 2 == 1:
            rp = np.sqrt((x-1*ldp)**2+(y+2*ldp)**2) / 2. / ldp
            II = jv(0, np.pi*rp)+jv(2, np.pi*rp)
            I = .8*I + .2*II**2 / (4.*(2.*ldp)**2) * np.pi

        if (features//2) % 2 == 1:
            Icopy = np.copy(I)
            I *= .85
            I[:-8, :] += .15*Icopy[8:, :]

        if (features//4) % 2 == 1:
            Icopy = np.copy(I)
            I *= .8
            I[:-4, :-4] += .1*Icopy[4:, 4:]
            I[4:, :-4] += .1*Icopy[:-4, 4:]

        # now convolve
        It = np.fft.rfft2(I)
        uxa = np.linspace(0, 1-1/npad, npad)
        uxa[-(npad//2):] -= 1
        ux = np.tile(uxa[None, :npad//2+1], (npad, 1))
        uy = np.tile(uxa[:, None], (1, npad//2+1))
        It *= np.exp(-2*np.pi**2*sigma**2*(ux**2+uy**2)) * \
            np.sinc(ux*tophat_conv) * np.sinc(uy*tophat_conv)
        I = np.real(np.fft.irfft2(It))

        return I[kp:-kp, kp:-kp]


class PSFGrp:
    '''Either a group of input PSFs attached to an InStamp instance
    or a group of output PSFs attached to a Block instance.
    '''

    dsample = 1.0  # 0.5  # sampling rate of overlap matrices
    npixpsf = 64  # size of PSF postage stamp in native pixels
    oversamp = 8  # will be set by Config.inpsf_oversamp
    # nsample = 2 * npixpsf * oversamp + 15  # 1039
    nsample = npixpsf * oversamp + 15  # 527
    # sampling matrix has shape (..., nsample, nsample)

    nc = nsample // 2  # 519 -> 263
    kpad = 5
    ns2 = nsample + 2 * kpad  # 1049 -> 537

    # unrotated grid of sampling positions
    # never modified (but frequently used) after initialization
    xyo = np.mgrid[(1-ns2)/2*dsample:(ns2-1)/2*dsample:ns2*1j,
                   (1-ns2)/2*dsample:(ns2-1)/2*dsample:ns2*1j]

    # p = 0  # pad size, pyimcom_interface.py line 189

    nfft0div8 = 2**(int(np.ceil(np.log2(ns2-0.5)))-2)
    nfft = nfft0div8 * int(np.ceil(2*ns2/nfft0div8))

    def __init__(self, in_or_out: bool = True,
                 inst=None, blk=None, visualize: bool = False):
        self.in_or_out = in_or_out  # True if input PSF group, False if output PSF group
        self.visualize = visualize  # whether to visualize PSFs and sampling positions

        if in_or_out:
            assert inst is not None, 'inst must be specified for an input PSF group'
            print(f'building input PSFGrp for InStamp {(inst.j_st, inst.i_st)}',
                  '@', inst.blk.timer(), 's')
            self.inst = inst
            self.build_inpsfgrp()
        else:
            assert blk is not None, 'block must be specified for an output PSF group'
            print(f'building output PSFGrp for Block {blk.this_sub=}', '@', blk.timer(), 's')
            self.blk = blk
            self.build_outpsfgrp()

        self.perform_rfts()
        # if in_or_out:
        #     print(f'finished', '@', inst.blk.timer(), 's')
        # else:
        #     print(f'finished', '@', blk.timer(), 's')

    @staticmethod
    def visualize_psf(psf: np.ndarray, xyco: np.ndarray, xctr: float, yctr: float):
        plt.imshow(np.log10(psf), origin='lower')
        plt.colorbar()
        plt.scatter(xyco[0].ravel()+xctr, xyco[1].ravel()+yctr, s=1e-5)
        plt.show()

    def sample_psf(self, idx: int, psf: np.ndarray,
                   distort_mat: np.ndarray = None, print_shapes: bool = False):
        '''Perform the interpolation to sample a single input/output PSF.

        Corresponding to furry-parakeet/pyimcom_interface.py lines 191 to 226.
        print_shapes: print shapes of C routine arguments for debugging purposes.
        '''

        ny, nx = np.shape(psf)[-2:]
        xctr = (nx-1) / 2.0
        yctr = (ny-1) / 2.0

        # xyco: sampling positions
        if distort_mat is None:
            xyco = PSFGrp.xyo
        else:  # succinct way to rotate
            rotate_mat = np.linalg.inv(distort_mat)
            xyo_ = np.flip(np.moveaxis(PSFGrp.xyo, 0, -1), axis=-1)
            xyco = np.moveaxis(np.flip(xyo_ @ rotate_mat.T, axis=-1), -1, 0)
            del rotate_mat, xyo_

            # TypeError: In-place matrix multiplication is not (yet) supported.
            # xyco = PSFGrp.xyo.copy()
            # view = np.flip(np.moveaxis(xyco, 0, -1), axis=-1)
            # view = view @ rotate_mat.T
            # del rotate_mat, view

        if self.visualize:
            if idx is not None:
                PSFGrp.visualize_psf(psf, xyco, xctr, yctr)
            else:  # sampling a group of output PSFs at the same time
                for psf_ in psf:
                    PSFGrp.visualize_psf(psf_, xyco, xctr, yctr)

        if idx is not None:
            out_arr = np.zeros((1, PSFGrp.ns2**2))
            if print_shapes:
                # print('iD5512C in sample_psf', (1, ny+2*PSFGrp.p, nx+2*PSFGrp.p),
                print('iD5512C in sample_psf', (1, ny, nx),
                      xyco[1].ravel().shape, xyco[0].ravel().shape, out_arr.shape)
            # iD5512C(np.pad(psf, PSFGrp.p).reshape((1, ny+2*PSFGrp.p, nx+2*PSFGrp.p)),
            iD5512C(psf.reshape((1, ny, nx)),
                    xyco[1].ravel()+xctr, xyco[0].ravel()+yctr, out_arr)
            self.psf_arr[idx] = out_arr.reshape((PSFGrp.ns2, PSFGrp.ns2))
            del out_arr

        else:  # sampling a group of output PSFs at the same time
            out_arr = np.zeros((self.n_psf, PSFGrp.ns2**2))
            if print_shapes:
                # print('iD5512C in sample_psf', (self.n_psf, ny+2*PSFGrp.p, nx+2*PSFGrp.p),
                print('iD5512C in sample_psf', (self.n_psf, ny, nx),
                      xyco[1].ravel().shape, xyco[0].ravel().shape, out_arr.shape)
            # iD5512C(np.pad(psf, pad_width=((0, 0), (PSFGrp.p, PSFGrp.p), (PSFGrp.p, PSFGrp.p))),
            #         xyco[1].ravel()+xctr, xyco[0].ravel()+yctr, out_arr)
            iD5512C(psf, xyco[1].ravel()+xctr, xyco[0].ravel()+yctr, out_arr)
            self.psf_arr = out_arr.reshape((self.n_psf, PSFGrp.ns2, PSFGrp.ns2))
            del out_arr

    def build_inpsfgrp(self):
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
        # print(self.use_inimage, self.idx_blk2grp, self.idx_grp2blk, self.n_psf, sep='\n')

        self.psf_arr = np.zeros((self.n_psf, PSFGrp.ns2, PSFGrp.ns2))
        for idx in range(self.n_psf):
            self.sample_psf(idx, *self.inst.blk.inimages[self.idx_grp2blk[idx]].\
                            get_psf_and_distort_mat(self.inst.j_st, self.inst.i_st))
            if self.visualize:
                print(f'The above PSF is from InImage {(self.inst.blk.inimages[self.idx_grp2blk[idx]].idsca)}',
                      f'at the upper right corner of InStamp {(self.inst.j_st, self.inst.i_st)}')

    def build_outpsfgrp(self):
        self.n_psf = 1  # to be set by self.blk.cfg
        psf_orig = np.zeros((self.n_psf, PSFGrp.npixpsf * PSFGrp.oversamp,
                             PSFGrp.npixpsf * PSFGrp.oversamp))

        for idx in range(self.n_psf):
            psf_orig[idx] = OutPSF.psf_simple_airy(PSFGrp.npixpsf * PSFGrp.oversamp,
                Stn.QFilterNative[self.blk.cfg.use_filter] * PSFGrp.oversamp,
                obsc=Stn.obsc, tophat_conv=0.0, sigma=self.blk.cfg.sigmatarget * PSFGrp.oversamp)
        self.sample_psf(None, psf_orig)
        # self.psf_arr.shape: (self.n_psf, PSFGrp.ns2, PSFGrp.ns2)

        # save the output PSFs to a file
        OutputPSFHDU = fits.HDUList([fits.PrimaryHDU()] + [fits.ImageHDU(x) for x in psf_orig])
        for i in range(self.n_psf):
            OutputPSFHDU[i+1].header['WHICHPSF'] = 'Output {:d}'.format(i)
        OutputPSFHDU.writeto(self.blk.cfg.outstem+'_outputpsfs.fits', overwrite=True)
        del OutputPSFHDU

    @staticmethod
    def accel_pad_and_rfft2(psf_arr: np.array):
        '''Accelerated version for zero-padding and rfft2.
        '''

        n_arr = psf_arr.shape[0]
        # m1 stands for axis=-1, m2 stands for axis=-2. 
        pad_m1 = np.zeros((n_arr, PSFGrp.ns2,  PSFGrp.nfft))
        pad_m2 = np.zeros((n_arr, PSFGrp.nfft, PSFGrp.nfft//2+1), dtype=np.complex128)

        pad_m1[:, :, :PSFGrp.ns2] = psf_arr
        pad_m2[:, :PSFGrp.ns2, :] = np.fft.rfft(pad_m1, axis=-1)
        del pad_m1
        return np.fft.fft(pad_m2, axis=-2)

    def perform_rfts(self):
        if self.n_psf <= Stn.fft_max_n_arr:
            self.psf_rft = PSFGrp.accel_pad_and_rfft2(self.psf_arr)
        else:  # perform rfts in batches
            self.psf_rft = np.zeros((self.n_psf, PSFGrp.nfft, PSFGrp.nfft//2+1), dtype=np.complex128)
            n_batch = int(np.ceil(self.n_psf / Stn.fft_max_n_arr))
            for i_b in range(n_batch):
                self.psf_rft[i_b*Stn.fft_max_n_arr:(i_b+1)*Stn.fft_max_n_arr] =\
                PSFGrp.accel_pad_and_rfft2(self.psf_arr[i_b*Stn.fft_max_n_arr:(i_b+1)*Stn.fft_max_n_arr])
        del self.psf_arr


class PSFOvl:
    '''Overlap between two PSFGrp instances or a PSFGrp instance and itself.
    '''

    # for slicing psf overlap arrays
    s_lower = PSFGrp.nfft//2-PSFGrp.nc    #  761 -> 377
    s_upper = PSFGrp.nfft//2+PSFGrp.nc+1  # 1800 -> 904

    flat_penalty = 1e-6

    # The lines below have been superseded by self.dscale:
    # in_stamp_dscale = Stn.pixscale_native/Stn.arcsec
    # s_in = Stn.pixscale_native/Stn.arcsec
    # s = PSFGrp.dsample/PSFGrp.oversamp*s_in  # this needs a better name!
    # scale of the PSF overlap arrays in arcsecs

    def __init__(self, psfgrp1: PSFGrp, psfgrp2: PSFGrp = None):
        self.grp1 = psfgrp1
        self.grp2 = psfgrp2  # None if self-overlap

        if psfgrp2 is not None:
            if psfgrp2.in_or_out:
                print(f'building input-input PSFOvl for InStamp {(psfgrp1.inst.j_st, psfgrp1.inst.i_st)}',
                      f'and InStamp {(psfgrp2.inst.j_st, psfgrp2.inst.i_st)}', '@', psfgrp1.inst.blk.timer(), 's')
            else:
                print(f'building input-output PSFOvl for InStamp {(psfgrp1.inst.j_st, psfgrp1.inst.i_st)}',
                      f'and Block this_sub={psfgrp2.blk.this_sub}', '@', psfgrp1.inst.blk.timer(), 's')

        else:
            if psfgrp1.in_or_out:
                print(f'building input-self PSFOvl for InStamp {(psfgrp1.inst.j_st, psfgrp1.inst.i_st)}',
                      '@', psfgrp1.inst.blk.timer(), 's')
            else:
                print(f'building output-self PSFOvl for Block {psfgrp1.blk.this_sub=}',
                      '@', psfgrp1.blk.timer(), 's')

        self.build_psfovl()

        # scale conversion for interpolations in PSF overlap arrays
        self.dscale = PSFGrp.dsample/PSFGrp.oversamp * Stn.pixscale_native/Stn.arcsec
        if psfgrp1.in_or_out:
            self.dscale /= psfgrp1.inst.blk.cfg.dtheta*3600
            self.n_inimage = psfgrp1.inst.blk.n_inimage
        else:
            self.dscale /= psfgrp1.blk.cfg.dtheta*3600
            self.n_inimage = psfgrp1.blk.n_inimage

        # if psfgrp2 is not None:
        #     print('finished', '@', psfgrp1.inst.blk.timer(), 's')
        # else:
        #     if psfgrp1.in_or_out:
        #         print('finished', '@', psfgrp1.inst.blk.timer(), 's')
        #     else:
        #         print('finished', '@', psfgrp1.blk.timer(), 's')

    def idx_square2triangle(self, idx1: int, idx2: int):
        '''For the self-overlap of an input PSF group,
        convert a 2-d square index to a 1-d triangle index
        e.g., for self.grp1.n_psf == 3:
        +---+---+---+
        | 0 | 1 | 2 |
        +---+---+---+
        |   | 3 | 4 |
        +---+---+---+
        |   |   | 5 |
        +---+---+---+
        '''

        assert idx1 <= idx2, 'the two indices must satisfy idx1 <= idx2'
        return (2*self.grp1.n_psf-idx1+1)*idx1//2 + idx2-idx1

    @staticmethod
    def accel_irfft2_and_extract(ovl_rft: np.array):
        '''Accelerated version for irfft2 and extracting,
        based on the fact that we only need about (2/5) x (2/5) of the irfft2 results.
        '''

        n_arr = ovl_rft.shape[0]
        # m2 stands for axis=-2, m1 stands for axis=-1. 
        ovl_m2 = np.zeros((n_arr, PSFGrp.nsample, PSFGrp.nfft//2+1), dtype=np.complex128)
        ovl_m1 = np.zeros((n_arr, PSFGrp.nsample, PSFGrp.nsample))
        nc = PSFGrp.nc  # shortcut

        ift_m2 = np.fft.ifft(ovl_rft, axis=-2)
        ovl_m2[:,   :nc, :] = ift_m2[:, -nc:    , :]
        ovl_m2[:, nc:  , :] = ift_m2[:,    :nc+1, :]
        del ift_m2

        ift_m1 = np.fft.irfft(ovl_m2, axis=-1, n=PSFGrp.nfft)
        ovl_m1[:, :,   :nc] = ift_m1[:, :, -nc:    ]
        ovl_m1[:, :, nc:  ] = ift_m1[:, :,    :nc+1]
        del ovl_m2, ift_m1

        return ovl_m1

    def build_psfovl(self):
        if self.grp2 is not None:  # cross-overlap
            self.ovl_arr = np.zeros((self.grp1.n_psf, self.grp2.n_psf, PSFGrp.nsample, PSFGrp.nsample))

            if self.grp2.n_psf <= Stn.fft_max_n_arr:
                for idx in range(self.grp1.n_psf):
                    ovl_rft = self.grp1.psf_rft[idx] * self.grp2.psf_rft.conjugate()
                    self.ovl_arr[idx] = PSFOvl.accel_irfft2_and_extract(ovl_rft)
                    del ovl_rft

            else:  # perform irfts in batches
                n_batch = int(np.ceil(self.grp2.n_psf / Stn.fft_max_n_arr))
                for idx in range(self.grp1.n_psf):
                    for i_b in range(n_batch):
                        ovl_rft = self.grp1.psf_rft[idx] * self.grp2.psf_rft\
                        [i_b*Stn.fft_max_n_arr:(i_b+1)*Stn.fft_max_n_arr].conjugate()
                        self.ovl_arr[idx, i_b*Stn.fft_max_n_arr:(i_b+1)*Stn.fft_max_n_arr] =\
                        PSFOvl.accel_irfft2_and_extract(ovl_rft)
                        del ovl_rft

        elif self.grp1.in_or_out:  # self-overlap of an input PSF group
            n_psf = self.grp1.n_psf  # shortcut
            self.ovl_arr = np.zeros((n_psf*(n_psf+1)//2, PSFGrp.nsample, PSFGrp.nsample))

            for idx in range(n_psf):
                start = self.idx_square2triangle(idx, idx)
                # print(idx, start, start+n_psf-idx)

                if n_psf <= Stn.fft_max_n_arr:
                    ovl_rft = self.grp1.psf_rft[idx] * self.grp1.psf_rft[idx:].conjugate()
                    self.ovl_arr[start:start+n_psf-idx] = PSFOvl.accel_irfft2_and_extract(ovl_rft)
                    del ovl_rft

                else:  # perform irfts in batches
                    n_batch = int(np.ceil((n_psf-idx) / Stn.fft_max_n_arr))
                    for i_b in range(n_batch):
                        ovl_rft = self.grp1.psf_rft[idx] * self.grp1.psf_rft\
                        [idx+i_b*Stn.fft_max_n_arr:idx+(i_b+1)*Stn.fft_max_n_arr].conjugate()
                        self.ovl_arr[start+i_b*Stn.fft_max_n_arr:
                                     start+min((i_b+1)*Stn.fft_max_n_arr, n_psf-idx)] =\
                        PSFOvl.accel_irfft2_and_extract(ovl_rft)
                        del ovl_rft

        else:  # self-overlap of an output PSF group
            n_psf = self.grp1.n_psf  # shortcut

            if n_psf <= Stn.fft_max_n_arr:
                ovl_rft = self.grp1.psf_rft * self.grp1.psf_rft.conjugate()
                self.ovl_arr = PSFOvl.accel_irfft2_and_extract(ovl_rft)
                del ovl_rft

            else:  # perform irfts in batches, not tested yet
                self.ovl_arr = np.zeros((n_psf, PSFGrp.nsample, PSFGrp.nsample))

                n_batch = int(np.ceil((n_psf) / Stn.fft_max_n_arr))
                for i_b in range(n_batch):
                    ovl_rft = self.grp1.psf_rft * self.grp1.psf_rft\
                    [i_b*Stn.fft_max_n_arr:(i_b+1)*Stn.fft_max_n_arr].conjugate()
                    self.ovl_arr[i_b*Stn.fft_max_n_arr:(i_b+1)*Stn.fft_max_n_arr] =\
                    PSFOvl.accel_irfft2_and_extract(ovl_rft)
                    del ovl_rft

            # extract C value(s)
            self.outovlc = self.ovl_arr[:, PSFGrp.nc, PSFGrp.nc]
            del self.ovl_arr  # remove this if the entire overlap array is needed

    def visualize_ovls(self):
        if self.grp2 is not None:
            n_psf1, n_psf2 = self.ovl_arr.shape[:2]
            fig, axs = plt.subplots(n_psf1, n_psf2,
                                    figsize=(3.2*min(n_psf2, 4), 2.4*min(n_psf1, 4)))

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
        
        else:  # self-overlap of a PSF group
            n_psf = self.grp1.n_psf  # shortcut

            if n_psf == 1:
                plt.imshow(np.log10(self.ovl_arr[0]), origin='lower')
                plt.colorbar()
                plt.show()

            else:
                fig, axs = plt.subplots(n_psf, n_psf,
                                        figsize=(3.2*min(n_psf, 4), 2.4*min(n_psf, 4)))

                for idx1 in range(n_psf):
                    for idx2 in range(n_psf):
                        if idx2 < idx1:
                            axs[idx1, idx2].axis('off')
                            continue

                        im = axs[idx1, idx2].imshow(np.log10(
                            self.ovl_arr[self.idx_square2triangle(idx1, idx2)]), origin='lower')
                        plt.colorbar(im)

                plt.show()
                del fig, axs

    def __call__(self, st1: 'InStamp', st2: 'InStamp, OutStamp, or None' = None,
                 visualize: bool = False, print_shapes: bool = False):
        '''Wrapper for the C interpolators.

        st1: must be InStamp.
        st2: can be either InStamp, OutStamp, or None (for self-overlap).
        visualize: whether to visualize overlap arrays and sampling positions.
        print_shapes: print shapes of C routine arguments for debugging purposes.
        '''

        if self.grp2 is not None:
            if self.grp2.in_or_out:  # overlap between two input PSF groups
                return self.call_ii_cross(st1, st2, visualize, print_shapes)
            else:  # overlap between an input PSF group and the output PSF group
                return self.call_io_cross(st1, st2, visualize, print_shapes)
        else:  # self-overlap of an input PSF group
            assert self.grp1.in_or_out, f'{self.grp1.in_or_out=}'
            return self.call_ii_self(st1, st2, visualize, print_shapes)

    def call_ii_cross(self, st1: 'InStamp', st2: 'InStamp',
                      visualize: bool = False, print_shapes: bool = False):
        res = np.zeros((st1.pix_cumsum[-1], st2.pix_cumsum[-1]))
        ddx = st1.x_val[:, None] - st2.x_val[None, :]
        ddx /= self.dscale; ddx += PSFGrp.nc
        ddy = st1.y_val[:, None] - st2.y_val[None, :]
        ddy /= self.dscale; ddy += PSFGrp.nc

        if visualize:
            n_psf1, n_psf2 = self.ovl_arr.shape[:2]
            fig, axs = plt.subplots(n_psf1, n_psf2, figsize=(6.4*n_psf2, 4.8*n_psf1))

        # print(f'{(st1.j_st, st1.i_st)=}', f'{st1.pix_count=}')
        # print(f'{(st2.j_st, st2.i_st)=}', f'{st2.pix_count=}')

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
                if print_shapes:
                    print('iD5512C in call_ii_cross', (1, PSFGrp.nsample, PSFGrp.nsample),
                          ddx[slice_].ravel().shape, ddy[slice_].ravel().shape, out_arr.shape)
                # print(f'{self.ovl_arr.shape=}')
                # print(f'{j_im=}', f'{self.grp1.idx_blk2grp=}')
                # print(f'{i_im=}', f'{self.grp2.idx_blk2grp=}')
                iD5512C(self.ovl_arr[self.grp1.idx_blk2grp[j_im], self.grp2.idx_blk2grp[i_im]].\
                        reshape((1, PSFGrp.nsample, PSFGrp.nsample)),
                    ddx[slice_].ravel(), ddy[slice_].ravel(), out_arr)

                res[slice_] = out_arr.reshape((st1.pix_count[j_im], st2.pix_count[i_im]))
                del out_arr

                # flat penalty
                if j_im == i_im:
                    res[slice_] += PSFOvl.flat_penalty
                else:
                    res[slice_] -= PSFOvl.flat_penalty / self.n_inimage / 2.

        if visualize:
            plt.show()
            del fig, axs

        return res

    def call_io_cross(self, st1: 'InStamp', st2: 'OutStamp',
                      visualize: bool = False, print_shapes: bool = False):
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
                if print_shapes:
                    print('gridD5512C in call_io_cross', self.ovl_arr[self.grp1.idx_blk2grp[j_im], i_psf].shape,
                          ddx[pix_cumsum_[j_im]:pix_cumsum_[j_im+1], :].shape,
                          ddy[pix_cumsum_[j_im]:pix_cumsum_[j_im+1], :].shape, out_arr.shape)
                gridD5512C(self.ovl_arr[self.grp1.idx_blk2grp[j_im], i_psf],
                           ddx[pix_cumsum_[j_im]:pix_cumsum_[j_im+1], :],
                           ddy[pix_cumsum_[j_im]:pix_cumsum_[j_im+1], :], out_arr)
                res[i_psf, :, pix_cumsum_[j_im]:pix_cumsum_[j_im+1]] = out_arr.T
                del out_arr

        if visualize:
            plt.show()
            del fig, axs

        return res

    def call_ii_self(self, st1: 'InStamp', st2: 'InStamp',
                     visualize: bool = False, print_shapes: bool = False):
        '''The self-overlap of an input PSF group can be used to
        compute either the sub-matrices for a single input stamp
        or the sub-matrices for a pair of input stamps within a 2x2 group.
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
            # print(n_psf)
            fig, axs = plt.subplots(n_psf, n_psf, figsize=(6.4*n_psf, 4.8*n_psf))

        for j_im in range(st1.blk.n_inimage):
            if st1.pix_count[j_im] == 0: continue
            for i_im in range(0 if not same_inst else j_im, st2.blk.n_inimage):
                if st2.pix_count[i_im] == 0: continue

                if j_im <= i_im:
                    ovl_arr_ = self.ovl_arr[self.idx_square2triangle(
                        self.grp1.idx_blk2grp[j_im], self.grp1.idx_blk2grp[i_im])]
                else:
                    ovl_arr_ = np.flip(self.ovl_arr[self.idx_square2triangle(
                        self.grp1.idx_blk2grp[i_im], self.grp1.idx_blk2grp[j_im])])

                slice_ = np.s_[st1.pix_cumsum[j_im]:st1.pix_cumsum[j_im+1],
                               st2.pix_cumsum[i_im]:st2.pix_cumsum[i_im+1]]

                if visualize:
                    # print(j_im, i_im, self.grp1.idx_blk2grp[j_im], self.grp1.idx_blk2grp[i_im])
                    ax = axs[self.grp1.idx_blk2grp[j_im], self.grp1.idx_blk2grp[i_im]] if n_psf > 1 else axs
                    im = ax.imshow(np.log10(ovl_arr_), origin='lower')
                    plt.colorbar(im)
                    ax.scatter(ddx[slice_].ravel(), ddy[slice_].ravel(), s=1e-2)

                out_arr = np.zeros((1, st1.pix_count[j_im] * st2.pix_count[i_im]))
                interpolator = iD5512C_sym if same_inst and j_im == i_im else iD5512C
                if print_shapes:
                    print('iD5512C(_sym) in call_ii_self', (1, PSFGrp.nsample, PSFGrp.nsample),
                          ddx[slice_].ravel().shape, ddy[slice_].ravel().shape, out_arr.shape)
                interpolator(ovl_arr_.reshape((1, PSFGrp.nsample, PSFGrp.nsample)),
                             ddx[slice_].ravel(), ddy[slice_].ravel(), out_arr)

                res[slice_] = out_arr.reshape((st1.pix_count[j_im], st2.pix_count[i_im]))
                # flat penalty
                if j_im == i_im:
                    res[slice_] += PSFOvl.flat_penalty
                else:
                    res[slice_] -= PSFOvl.flat_penalty / self.n_inimage / 2.

                if same_inst and j_im < i_im:
                    res[st2.pix_cumsum[i_im]:st2.pix_cumsum[i_im+1],
                        st1.pix_cumsum[j_im]:st1.pix_cumsum[j_im+1]] = res[slice_].T
                del out_arr

        if visualize:
            if n_psf > 1:
                for j_im in range(st1.blk.n_inimage):
                    if st1.pix_count[j_im] == 0: continue
                    for i_im in range(j_im):
                        if st2.pix_count[i_im] == 0: continue
                        axs[self.grp1.idx_blk2grp[j_im], self.grp1.idx_blk2grp[i_im]].axis('off')

            plt.show()
            del fig, axs

        return res

    def clear(self):
        '''Use this in addition to the default __del__ method
        to make sure that references to PSFGrp instances are removed,
        and more importantly the the huge overlap array (self.ovl_arr).
        '''

        if self.grp2 is not None and not self.grp2.in_or_out:
            print(f'clearing input-output PSFOvl for InStamp {(self.grp1.inst.j_st, self.grp1.inst.i_st)}',
                  f'and Block this_sub={self.grp2.blk.this_sub}', '@', self.grp1.inst.blk.timer(), 's')

        del self.grp1, self.grp2, self.ovl_arr


class SysMatA:
    '''System matrix A attached to a Block instance.
    '''

    def __init__(self, blk):
        self.blk = blk

        self.iisubmats = {}  # ii stands for input-input
        self.iisubmats_ref = np.zeros((blk.cfg.n1+2, blk.cfg.n1+2, 13), dtype=np.uint8)

    @staticmethod
    def ji_st2psf(ji_st: (int, int)) -> (int, int):
        '''Convert ji_st to that of the corresponding psfovl,
        e.g., (0, 0), (0, 1), (1, 0), (1, 1) -> (0, 0).
        '''

        return tuple(ji >> 1 << 1 for ji in ji_st)

    @staticmethod
    def shift_ji_st(in_st: (int, int), din_st: (int, int)):
        '''Tool function for tuple addition.
        '''

        return (in_st[0]+din_st[0], in_st[1]+din_st[1])

    @staticmethod
    def iisubmat_dist(ji_st1: (int, int), ji_st2: (int, int)) -> (int, int, int):
        '''Calculate the "distance" between two input postage stamps
        to store the reference counts in the 3-d array self.iisubmats_ref
        using indices (ji_st1[0], ji_st1[1], dist), which this static method returns.

        The "distance" is defined as follows:
        +---+---+---+---+---+
        | 8 | 9 | 10| 11| 12|
        +---+---+---+---+---+
        | 3 | 4 | 5 | 6 | 7 |
        +---+---+---+---+---+
        |   |   | 0*| 1 | 2 |
        +---+---+---+---+---+
        where "*" denotes ji_st1.
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

    def compute_iisubmats(self, ji_st1: (int, int), ji_st2: (int, int), sim_mode: bool = False):
        '''sim_mode: count references, no actual iisubmats computed.
        '''

        ji_psf1 = SysMatA.ji_st2psf(ji_st1)
        ji_psf2 = SysMatA.ji_st2psf(ji_st2)

        psfgrp1 = self.blk.instamps[ji_psf1[0]][ji_psf1[1]].get_inpsfgrp(sim_mode)
        if ji_psf1 != ji_psf2:
            psfgrp2 = self.blk.instamps[ji_psf2[0]][ji_psf2[1]].get_inpsfgrp(sim_mode)
        else:
            psfgrp2 = None
        if not sim_mode:
            iipsfovl = PSFOvl(psfgrp1, psfgrp2)

        counter = 0
        # all 4 possible ji_st1's using the same iipsfovl
        for dji_st1 in range(4):
            ji_st1_ = SysMatA.shift_ji_st(ji_psf1, divmod(dji_st1, 2))

            # all 4 possible ji_st2's using the same iipsfovl
            for dji_st2 in range(4):
                ji_st2_ = SysMatA.shift_ji_st(ji_psf2, divmod(dji_st2, 2))

                ji_st_pair = (ji_st1_, ji_st2_) if ji_st1_ <= ji_st2_ else (ji_st2_, ji_st1_)
                ji_dist = SysMatA.iisubmat_dist(*ji_st_pair)
                # print(ji_st_pair, ji_dist, self.iisubmats_ref[ji_dist] if ji_dist is not None else '')
                if ji_dist is None or (not sim_mode and self.iisubmats_ref[ji_dist] == 0): continue
                if ji_st_pair not in self.iisubmats:
                    if sim_mode:
                        self.iisubmats[ji_st_pair] = None
                    else:
                        my_submat = iipsfovl(self.blk.instamps[ji_st1_[0]][ji_st1_[1]],
                                             self.blk.instamps[ji_st2_[0]][ji_st2_[1]] \
                                             if ji_st1_ != ji_st2_ else None)
                        if ji_st1_ <= ji_st2_:
                            self.iisubmats[ji_st_pair] = my_submat
                        else:
                            self.iisubmats[ji_st_pair] = my_submat.T
                        # print(f'{ji_psf1=}', f'{ji_st1_=}', f'{ji_psf2=}', f'{ji_st2_=}', ji_st1_ <= ji_st2_)

                    # The code below is incorrect in some rare cases.
                    # self.iisubmats[ji_st_pair] = None if sim_mode else iipsfovl(
                    #     self.blk.instamps[ji_st_pair[0][0]][ji_st_pair[0][1]],
                    #     self.blk.instamps[ji_st_pair[1][0]][ji_st_pair[1][1]] \
                    #     if ji_st_pair[0] != ji_st_pair[1] else None)

                    counter += 1

        if not sim_mode:
            iipsfovl.clear(); del iipsfovl
            print(f'finished computing {counter} input-input sub-matrices', '@', self.blk.timer(), 's')

    def get_iisubmat(self, ji_st1: (int, int), ji_st2: (int, int), sim_mode: bool = False):
        '''sim_mode: count references, no actual iisubmats returned.
        '''

        assert ji_st1 <= ji_st2, f'{ji_st1=} should precede {ji_st2=}'
        ji_dist = SysMatA.iisubmat_dist(ji_st1, ji_st2)
        assert ji_dist is not None, f'distance between InStamps {ji_st1} and {ji_st2} is out of range'

        if sim_mode: self.iisubmats_ref[ji_dist] += 1
        if (ji_st1, ji_st2) not in self.iisubmats:
            self.compute_iisubmats(ji_st1, ji_st2, sim_mode)
        if sim_mode: return

        self.iisubmats_ref[ji_dist] -= 1
        if self.iisubmats_ref[ji_dist] > 0:
            return self.iisubmats[(ji_st1, ji_st2)]
        else:
            arr = self.iisubmats[(ji_st1, ji_st2)]
            del self.iisubmats[(ji_st1, ji_st2)]
            return arr


class SysMatB:
    '''System matrix B attached to a Block instance.

    INPORTANT: The -2 coefficient in Rowe+ 2011 Equation (18)
    is NOT included in this program.
    '''

    def __init__(self, blk):
        self.blk = blk

        self.iopsfovls = {}  # io stands for input-output
        self.iopsfovls_ref = np.zeros((blk.cfg.n1//2+1, blk.cfg.n1//2+1), dtype=np.uint8)

    @staticmethod
    def iosubmat_dist(ji_st_in: (int, int), ji_st_out: (int, int)) -> int:
        '''Calculate the "distance" between an input postage stamp
        and the output postage stamp of interest to store the reference counts
        in the 3-d array self.iosubmats_ref using indices (ji_st_out[0], ji_st_out[1], dist),
        where dist is returned by this static method.

        The "distance" is defined as follows:
        +---+---+---+
        | 6 | 7 | 8 |
        +---+---+---+
        | 3 | 4*| 5 |
        +---+---+---+
        | 0 | 1 | 2 |
        +---+---+---+
        where "*" denotes ji_st_out.

        Updated on 9/17/2023: Now the dist is only used for checking purposes.
        '''

        dj_st = ji_st_in[0] - ji_st_out[0]
        if abs(dj_st) > 1: return None
        di_st = ji_st_in[1] - ji_st_out[1]
        if abs(di_st) > 1: return None

        return (dj_st+1) * 3 + (di_st+1)

    def get_iosubmat(self, ji_st_in: (int, int), ji_st_out: (int, int), sim_mode: bool = False):
        '''sim_mode: count references, no actual iosubmats returned.
        '''

        dist = SysMatB.iosubmat_dist(ji_st_in, ji_st_out)
        assert dist is not None, f'distance between InStamp {ji_st_in} and OutStamp {ji_st_out} is out of range'

        ji_st_inpsf = SysMatA.ji_st2psf(ji_st_in)
        inpsf_key = tuple(ji//2 for ji in ji_st_inpsf)

        if sim_mode: self.iopsfovls_ref[inpsf_key] += 1
        if inpsf_key not in self.iopsfovls:
            inpsfgrp = self.blk.instamps[ji_st_inpsf[0]][ji_st_inpsf[1]].get_inpsfgrp(sim_mode)
            self.iopsfovls[inpsf_key] = PSFOvl(inpsfgrp, self.blk.outpsfgrp) if not sim_mode else None
        if sim_mode: return

        self.iopsfovls_ref[inpsf_key] -= 1
        iosubmat = self.iopsfovls[inpsf_key](self.blk.instamps [ji_st_in [0]][ji_st_in [1]],
                                             self.blk.outstamps[ji_st_out[0]][ji_st_out[1]])
        if self.iopsfovls_ref[inpsf_key] == 0:
            self.iopsfovls[inpsf_key].clear(); del self.iopsfovls[inpsf_key]
        return iosubmat
