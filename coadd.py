'''
Driver to coadd a block (2D array of postage stamps).

Classes
-------
InImage : Input image attached to a Block instance.
InStamp : Data structure for input pixel positions and signals.
OutStamp : Driver for postage stamp coaddition.
Block : Driver for block coaddition.

'''

from os.path import exists
from itertools import combinations, product
import gc
import sys

from pathlib import Path
import datetime
import pytz
from filelock import Timeout, FileLock

from astropy import units as u
from astropy.table import Table
import numpy as np
from scipy.special import legendre

from astropy.io import fits
from astropy import wcs
import fitsio
import matplotlib.pyplot as plt

import importlib

from .config import Timer, Settings as Stn, Config, format_axis
from .layer import check_if_idsca_exists, get_all_data, Mask
from .psfutil import PSFGrp, PSFOvl, SysMatA, SysMatB
from .lakernel import EigenKernel, CholKernel, IterKernel, EmpirKernel


class InImage:
    '''
    Input image attached to a Block instance.

    Methods
    -------
    __init__ : Constructor.
    generate_idx_grid (staticmethod) : Generate a grid of indices.
    _inpix2world2outpix : Composition of pix2world and world2pix.
    outpix2world2inpix : Inverse function of _inpix2world2outpix.

    partition_pixels : Partition input pixels into postage stamps.
    extract_layers : Extract input layers.
    clear : Free up memory space.

    smooth_and_pad (staticmethod) : Utility to smear a PSF with a tophat and a Gaussian.
    LPolyArr (staticmethod) : Utility to generate an array of Legendre polynomials.
    get_psf_pos : Get input PSF array at given position.

    '''

    def __init__(self, blk: 'Block', idsca: (int, int)) -> None:
        '''
        Constructor.

        Parameters
        ----------
        blk : Block
            The Block instance to which this InImage instance is attached.
        idsca : (int, int)
            ID of observation and SCA used.

        Returns
        -------
        None.

        '''

        self.blk = blk
        self.idsca = idsca

        self.exists_, self.infile = check_if_idsca_exists(blk.cfg, blk.obsdata, idsca)
        if self.exists_:
            with fits.open(self.infile) as f:
                self.inwcs = wcs.WCS(f[Stn.hdu_with_wcs].header)
                # print('read WCS:', self.infile, 'HDU', Stn.hdu_with_wcs)

    @staticmethod
    def generate_idx_grid(xs: np.array, ys: np.array) -> np.array:
        '''
        Generate a grid of indices.

        Parameters
        ----------
        xs : np.array, shape : (nx,)
        ys : np.array, shape : (ny,)
            x and y indices of the grid.

        Returns
        -------
        np.array, shape : (nx * ny, 2)
            All combinations of xs elements and ys elements.

        '''

        return np.moveaxis(np.array(np.meshgrid(xs, ys)), 0, -1).reshape(-1, 2)

    def _inpix2world2outpix(self, inxys: np.array) -> np.array:
        '''
        Composition of pix2world and world2pix.

        Parameters
        ----------
        inxys : np.array, shape : (npix, 2)
            x and y positions in the input image coordinates.

        Returns
        -------
        np.array, shape : (npix, 2)
            x and y positions in the output block coordinates.

        '''

        return self.blk.outwcs.all_world2pix(
                    self.inwcs.all_pix2world(inxys, 0), 0)

    def outpix2world2inpix(self, outxys: np.array) -> np.array:
        '''
        Inverse function of _inpix2world2outpix.

        Parameters
        ----------
        outxys : np.array, shape : (npix, 2)
            x and y positions in the output block coordinates.

        Returns
        -------
        np.array, shape : (npix, 2)
            x and y positions in the input image coordinates.

        '''

        return self.inwcs.all_world2pix(
                    self.blk.outwcs.all_pix2world(outxys, 0), 0)

    def partition_pixels(self, sp_res: int = 90, relax_coef: float = 1.05,
                         verbose: bool = False, visualize: bool = False) -> None:
        '''
        Partition input pixels into postage stamps.

        Parameters
        ----------
        sp_res : int, optional
            Resolution of the sparse grid. The default is 90.
        relax_coef : float, optional
            Coefficient to create enough space for input pixels. The default is 1.05.
        visualize : bool, optional
            Whether to visualize the partition process and results. The default is False.

        Returns
        -------
        None.

        '''

        if verbose:
            print(f'> partitioning pixels from InImage {self.idsca}', '@', self.blk.timer(), 's')

        # create a sparse grid of pixels to locate regions of interest
        sp_arr = np.linspace(0, Stn.sca_nside, sp_res+1, dtype=np.uint16)
        sp_inxys = InImage.generate_idx_grid(sp_arr, sp_arr)
        sp_outxys = self._inpix2world2outpix(sp_inxys).T.reshape(2, sp_res+1, sp_res+1)
        del sp_inxys

        # limits for input pixel positions in output pixel coordinates
        pix_lower =                     - self.blk.cfg.n2 - 0.5
        pix_upper = self.blk.cfg.NsideP + self.blk.cfg.n2 - 0.5

        self.is_relevant = False  # whether the input image is relevant to the output block
        relevant_matrix = np.zeros((sp_res, sp_res), dtype=bool)
        for j in range(1, sp_res):
            for i in range(1, sp_res):
                if not (pix_lower < sp_outxys[0, j, i] < pix_upper\
                    and pix_lower < sp_outxys[1, j, i] < pix_upper): continue
                i_st = int((sp_outxys[0, j, i] - pix_lower) // self.blk.cfg.n2)  # st stands for stamp
                j_st = int((sp_outxys[1, j, i] - pix_lower) // self.blk.cfg.n2)
                # assert i_st >= 0 and j_st >= 0, 'i_st < 0 or j_st < 0'

                if np.any(self.blk.use_instamps[max(j_st-2, 0):min(j_st+3, self.blk.cfg.n1P+2),
                                                max(i_st-2, 0):min(i_st+3, self.blk.cfg.n1P+2)]):
                    # at least some of the input pixels are relevant
                    self.is_relevant = True
                    # we will study all the adjacent input pixels
                    relevant_matrix[max(j-2, 0):min(j+3, sp_res),
                                    max(i-2, 0):min(i+3, sp_res)] = True

        if visualize and self.is_relevant:
            fig, axs = plt.subplots(2, 2, figsize=(10.8, 9.6))

            for i in range(2):
                ax = axs[0, i]
                im = ax.imshow(sp_outxys[i] / self.blk.cfg.n2, origin='lower')
                plt.colorbar(im, ax=ax)
                ax.contour(sp_outxys[i], levels=[pix_lower, pix_upper], colors='r')

            im = axs[1, 0].imshow(relevant_matrix, origin='lower', cmap='YlGn')
            plt.colorbar(im, ax=axs[1, 0])

            for ax, title in zip([*axs[0], axs[1, 0]],
                                 ['stamp index $i$', 'stamp index $j$', 'relevant matrix']):
                ax.set_xlabel('sparse grid $i$')
                ax.set_ylabel('sparse grid $j$')
                ax.set_title(title)
                format_axis(ax, False)

        del sp_outxys

        if not self.is_relevant:
            del sp_arr, relevant_matrix
            return
        print('input image', self.idsca, flush=True)

        # maximum number of input pixels per postage stamp (from this InImage)
        # relax_coef: the actual maximum may be larger due to distortions
        npixmax = int(((self.blk.cfg.n2 * self.blk.cfg.dtheta * u.degree.to('arcsec')) /\
                       (Stn.pixscale_native / Stn.arcsec) + 1) ** 2 * relax_coef)  # default: about 160

        # arrays for indices (in the input image grid),
        self.y_idx = np.zeros((self.blk.cfg.n1P+2, self.blk.cfg.n1P+2, npixmax), dtype=np.uint16)
        self.x_idx = np.zeros((self.blk.cfg.n1P+2, self.blk.cfg.n1P+2, npixmax), dtype=np.uint16)
        # positions (in the output block coordinates),
        self.y_val = np.zeros((self.blk.cfg.n1P+2, self.blk.cfg.n1P+2, npixmax), dtype=np.float64)
        self.x_val = np.zeros((self.blk.cfg.n1P+2, self.blk.cfg.n1P+2, npixmax), dtype=np.float64)
        # and number of pixels in each postage stamp (from this InImage)
        self.pix_count = np.zeros((self.blk.cfg.n1P+2, self.blk.cfg.n1P+2), dtype=np.uint32)

        # load masks here
        if self.blk.pmask is not None:
            mask = self.blk.pmask[self.idsca[1]-1]
        else:
            mask = np.ones((Stn.sca_nside, Stn.sca_nside), dtype=bool)

        get_all_data(self)  # shape : (n_inframe, Stn.sca_nside, Stn.sca_nside)

        cr_mask = Mask.load_cr_mask(self)
        if cr_mask is not None:
            mask = np.logical_and(mask, cr_mask)
        del cr_mask

        # if there's already a mask, *REPLACE* it here
        # This is important in the iterative PSF approach because we may need to change the input
        # layers at various stages of the calculation.
        if bool(self.blk.cfg.inlayercache):
            inlayer_mask_filepath = self.blk.cfg.inlayercache + '_{:08d}_{:02d}_mask.fits'.format(self.idsca[0],self.idsca[1])
            inlayer_mask_lockpath = inlayer_mask_filepath + '.lock'
            lock = FileLock(inlayer_mask_lockpath)
            if exists(inlayer_mask_filepath):
                try:
                    with lock.acquire(timeout=300):
                        print('loading input mask <<', inlayer_mask_filepath)
                        with fits.open(inlayer_mask_filepath) as f:
                            mask = f[0].data>0
                except Timeout:
                    raise Exception('timeout while waiting for file:', inlayer_mask_filepath)
            else:
                try:
                    with lock.acquire(timeout=300):
                        print('saving input mask >>', inlayer_mask_filepath)
                        fits.PrimaryHDU(np.where(mask,1,0).astype(np.uint8)).writeto(inlayer_mask_filepath)
                except Timeout:
                    raise Exception('timeout while waiting for file:', inlayer_mask_filepath)
        sys.stdout.flush()

        # now loop over the regions set by the sparse grid
        for j_sp in range(sp_res):
            for i_sp in range(sp_res):
                if not relevant_matrix[j_sp, i_sp]: continue
                left, right = sp_arr[i_sp:i_sp+2]
                bottom, top = sp_arr[j_sp:j_sp+2]
                inxys = InImage.generate_idx_grid(np.arange(left, right), np.arange(bottom, top))
                outxys = self._inpix2world2outpix(inxys).T.reshape(2, top-bottom, right-left)

                for j in range(top-bottom):
                    for i in range(right-left):
                        my_x, my_y = outxys[:, j, i]
                        if not (pix_lower < my_x < pix_upper and
                                pix_lower < my_y < pix_upper): continue
                        if not mask[bottom+j, left+i]: continue

                        i_st = int((my_x - pix_lower) // self.blk.cfg.n2)  # st stands for stamp
                        j_st = int((my_y - pix_lower) // self.blk.cfg.n2)
                        if not self.blk.use_instamps[j_st, i_st]: continue

                        my_idx = self.pix_count[j_st, i_st]
                        self.y_idx[j_st, i_st, my_idx] = bottom + j
                        self.x_idx[j_st, i_st, my_idx] = left + i
                        self.y_val[j_st, i_st, my_idx] = my_y
                        self.x_val[j_st, i_st, my_idx] = my_x
                        self.pix_count[j_st, i_st] += 1

                del inxys, outxys

        if visualize and self.is_relevant:
            ax = axs[1, 1]
            im = ax.imshow(self.pix_count, origin='lower', cmap='plasma')
            plt.colorbar(im, ax=ax)

            ax.set_xlabel('stamp index $i$')
            ax.set_ylabel('stamp index $j$')
            ax.set_title('pixel count')
            format_axis(ax, False)

            plt.show()

        if verbose:
            print('-->', np.sum(self.pix_count), 'pixels selected from idsca', self.idsca, end='; ')
        self.max_count = np.max(self.pix_count)
        if verbose:
            print('the most populous stamp has', self.max_count, 'pixels')
        del sp_arr, relevant_matrix, mask

    def extract_layers(self, verbose: bool = False) -> None:
        '''
        Extract input layers.

        Returns
        -------
        None.

        '''

        assert self.exists_, 'Error: input image and/or input psf do(es) not exist'

        self.data = np.zeros((self.blk.cfg.n_inframe, self.blk.cfg.n1P+2,
                              self.blk.cfg.n1P+2, self.max_count), dtype=np.float32)

        for j_st in range(self.blk.cfg.n1P+2):
            for i_st in range(self.blk.cfg.n1P+2):
                n_pix = self.pix_count[j_st, i_st]
                self.data[:, j_st, i_st, :n_pix] =\
                self.indata[:, self.y_idx[j_st, i_st, :n_pix], self.x_idx[j_st, i_st, :n_pix]]

        del self.indata, self.y_idx, self.x_idx
        if verbose:
            print(f'--> finished extracting layers for InImage {self.idsca}', '@', self.blk.timer(), 's')

    def clear(self) -> None:
        '''
        Free up memory space.

        Returns
        -------
        None.

        '''

        if self.is_relevant:
            del self.y_val, self.x_val, self.pix_count, self.data

        if hasattr(self, 'inpsf_arr' ): del self.inpsf_arr
        if hasattr(self, 'inpsf_cube'): del self.inpsf_cube

    @staticmethod
    def smooth_and_pad(inArray: np.array, tophatwidth: float = 0.0,
                       gaussiansigma: float = 0.0) -> np.array:
        '''
        Utility to smear a PSF with a tophat and a Gaussian.

        Parameters
        ----------
        inArray : np.array, shape : (ny, nx)
            Input PSF array to be smeared.
        tophatwidth : float, optional
        gaussiansigma : float, optional
            Both in units of the pixels given (not native pixel). The default is 0.0.

        Returns
        -------
        outArray : np.array, shape : (ny+npad*2, nx+npad*2)
            Smeared input PSF array.

        '''

        npad = int(np.ceil(tophatwidth + 6*gaussiansigma + 1))
        npad += (4-npad) % 4  # make a multiple of 4
        (ny, nx) = np.shape(inArray)
        nyy = ny+npad*2
        nxx = nx+npad*2
        outArray = np.zeros((nyy, nxx))
        outArray[npad:-npad, npad:-npad] = inArray
        outArrayFT = np.fft.fft2(outArray)

        # convolution
        uy = np.linspace(0, nyy-1, nyy)/nyy
        uy = np.where(uy > .5, uy-1, uy)
        ux = np.linspace(0, nxx-1, nxx)/nxx
        ux = np.where(ux > .5, ux-1, ux)
        outArrayFT *= np.sinc(ux[None, :]*tophatwidth)*np.sinc(uy[:, None]*tophatwidth) * \
            np.exp(-2.*np.pi**2*gaussiansigma ** 2*(ux[None, :]**2+uy[:, None]**2))

        outArray = np.real(np.fft.ifft2(outArrayFT))
        return outArray

    @staticmethod
    def LPolyArr(PORDER,u_,v_):
        '''
        Generates a length (PORDER+1)**2 array of the Legendre polynomials
        for n=0..PORDER { for m=0..PORDER { coef P_m(u_) P_n(v_) }}

        Parameters
        ----------
        PORDER : int
          >=0, order in each axis
        u_ : float
          x-position on chip scaled to -1..+1
        v_ : float
          y-position on chip scaled to -1..+1

        Returns
        -------
        arr: np.array, shape : ((PORDER+1)**2)
          the array of Legendre polynomial products
          constant (1) is first, then increasing x-order, then increasing y-order
        '''

        ua = np.ones(PORDER+1)
        va = np.ones(PORDER+1)
        for m in range(1,PORDER+1):
            L = legendre(m)
            ua[m] = L(u_)
            va[m] = L(v_)
        arr = np.outer(va,ua).flatten()
        return arr

    def get_psf_pos(self, psf_compute_point: np.array, use_shortrange: bool = False) -> np.array:
        '''
        Get input PSF array at given position.

        This is an interface for layer.get_all_data and psfutil.PSFGrp._build_inpsfgrp.

        Parameters
        ----------
        psf_compute_point : np.array, shape : (2,)
            Point to compute PSF in RA and Dec.
        use_shortrange : bool, optional
            If True and PSFSPLIT is set, then pulls only the short-range PSF G^(S).
            The default is False.

        Returns
        -------
        tuple : np.array, shape : see smooth_and_pad
            Input PSF array at given position. 

        '''

        # The tophat width: in use_shortrange, the psfsplit module has already included this,
        # so we set it to 0 so as to not double-count this contribution.
        tophatwidth_use = self.blk.cfg.inpsf_oversamp
        if use_shortrange: tophatwidth_use = 0

        # get the pixel location on the input image
        # (moved this up since some PSF models need it)
        pixloc = self.inwcs.all_world2pix(np.array([[*psf_compute_point]]).astype(np.float64), 0)[0]

        if self.blk.cfg.inpsf_format == 'dc2_imsim':
            if not hasattr(self, 'inpsf_arr'):
                fname = self.blk.cfg.inpsf_path + '/psf_{:d}.fits'.format(self.idsca[0])
                assert exists(fname), 'Error: input psf does not exist'
                with fitsio.FITS(fname) as fileh:
                    self.inpsf_arr = InImage.smooth_and_pad(
                        fileh[self.idsca[1]][:, :], tophatwidth=tophatwidth_use)

            this_psf = self.inpsf_arr

        elif self.blk.cfg.inpsf_format == 'anlsim':
            if not hasattr(self, 'inpsf_cube'):
                fname = self.blk.cfg.inpsf_path + '/psf_polyfit_{:d}.fits'.format(self.idsca[0])
                sskip = 0
                readskip = False
                if use_shortrange and self.blk.cfg.psfsplit:
                    fname = self.blk.cfg.inlayercache + '.psf/psf_{:d}.fits'.format(self.idsca[0])
                    readskip = True
                assert exists(fname), 'Error: input psf does not exist'
                with fits.open(fname) as f:
                    if readskip: sskip = int(f[0].header['GSSKIP'])
                    self.inpsf_cube = f[self.idsca[1]+sskip].data[:, :, :]
                    print(' <<', fname, sskip)

            # order = 1
            lpoly = InImage.LPolyArr(1, (pixloc[0]-2043.5)/2044., (pixloc[1]-2043.5)/2044.)
            # pixels are in C/Python convention since pixloc was set this way
            this_psf = InImage.smooth_and_pad(
                np.einsum('a,aij->ij', lpoly, self.inpsf_cube), tophatwidth=tophatwidth_use)/64
            # divide by 64=8**2 since anlsim files are in fractional intensity per s_in**2 instead of per (s_in/8)**2

        else:
            raise RuntimeError('Error: input psf does not exist')

        return this_psf

        # when distort_matrice is not required
        # if dWdp_out is None: return this_psf

        # get the distortion matrices d[(X,Y)perfect]/d[(X,Y)native]
        # Note that rotations and magnifications are included in the distortion matrix, as well as shear
        # Also the distortion is relative to the output grid, not to the tangent plane to the celestial sphere
        # (although we really don't want the difference to be large ...)
        # distort_matrice = np.linalg.inv(dWdp_out) \
        #     @ wcs.utils.local_partial_pixel_derivatives(self.inwcs, pixloc[0], pixloc[1]) \
        #     * self.blk.cfg.dtheta*Stn.degree/Stn.pixscale_native

        # print(pixloc, self.blk.cfg.inpsf_oversamp, np.shape(this_psf), np.sum(this_psf))
        # return this_psf, distort_matrice


class InStamp:
    '''
    Data structure for input pixel positions and signals.

    Methods
    -------
    __init__ : Constructor.
    make_selection : Return the indices of selected input pixels.
    get_inpsfgrp : Get the input PSFGrp attached to this InStamp.
    clear : Free up memory space.

    '''

    def __init__(self, blk: 'Block', j_st: int, i_st: int) -> None:
        '''
        Constructor.

        Parameters
        ----------
        blk : Block
            The Block instance to which this InStamp instance is attached.
        j_st, i_st : int, int
            InStamp index.

        Returns
        -------
        None.

        '''

        self.blk = blk
        self.j_st = j_st
        self.i_st = i_st

        # numbers of input pixels from input images and the cumulative sum
        self.pix_count = np.array([inimage.pix_count[j_st, i_st]
                                   for inimage in blk.inimages], dtype=np.uint32)
        self.pix_cumsum = np.cumsum([0] + list(self.pix_count), dtype=np.uint32)

        # input pixel positions and signals
        self.y_val = np.empty((self.pix_cumsum[-1],), dtype=np.float64)
        self.x_val = np.empty((self.pix_cumsum[-1],), dtype=np.float64)
        self.data = np.empty((blk.cfg.n_inframe, self.pix_cumsum[-1]), dtype=np.float32)

        for i_im, inimage in enumerate(blk.inimages):
            self.y_val[self.pix_cumsum[i_im]:self.pix_cumsum[i_im+1]] =\
                inimage.y_val[j_st, i_st, :self.pix_count[i_im]]
            self.x_val[self.pix_cumsum[i_im]:self.pix_cumsum[i_im+1]] =\
                inimage.x_val[j_st, i_st, :self.pix_count[i_im]]
            self.data[:, self.pix_cumsum[i_im]:self.pix_cumsum[i_im+1]] =\
                inimage.data[:, j_st, i_st, :self.pix_count[i_im]]

        if j_st % 2 == 0 and i_st % 2 == 0:
            # get where to compute the PSF and the camera distortion matrix
            # (the center of the 2x2 group of postage stamps)
            self.psf_compute_point_pix = [i_st * blk.cfg.n2 - 0.5,
                                          j_st * blk.cfg.n2 - 0.5]
            self.inpsfgrp = None
            self.inpsfgrp_ref = 0

    def make_selection(self, pivot: (float, float) = (None, None),
                       radius: float = None) -> np.array:
        '''
        Return the indices of selected input pixels.

        This is an interface for OutStamp._process_input_stamps.

        Parameters
        ----------
        pivot : (float, float), optional
            Pivot position in the output block coordinates.
            If None in one direction, select input pixels according to the other;
            if None in both directions, select all input pixels.
            The default is (None, None).
        radius : float, optional
            Select input pixels within this radius. The default is None.

        Returns
        -------
        np.array, shape : (npix,)
            Indices of selected input pixels.
            None if selecting all input pixels.

        '''

        if pivot == (None, None) or radius is None:
            return None  # select all pixels

        dist_sq = np.zeros((self.pix_cumsum[-1],))
        if pivot[0] is not None:
            dist_sq += np.square(self.x_val - pivot[0])
        if pivot[1] is not None:
            dist_sq += np.square(self.y_val - pivot[1])

        selection = np.array(np.where(dist_sq < radius**2)[0], dtype=np.uint32)
        return selection if (selection.shape[0] < self.pix_cumsum[-1]) else None

    def get_inpsfgrp(self, sim_mode: bool = False) -> None:
        '''
        Get the input PSFGrp attached to this InStamp.

        This is an interface for psfutil.SysMatA._compute_iisubmats
        and psfutil.SysMatB.get_iosubmat.

        Parameters
        ----------
        sim_mode : bool, optional
            Whether to count references without actually making inpsfgrp.
            See the docstring of psfutil.SysMatA._compute_iisubmats.
            The default is False.

        Returns
        -------
        None.

        '''

        if sim_mode:  # count references, no actual inpsfgrp involved
            self.inpsfgrp_ref += 1
            return

        if self.inpsfgrp is None:
            self.inpsfgrp = PSFGrp(in_or_out=True, inst=self)

        self.inpsfgrp_ref -= 1
        if self.inpsfgrp_ref > 0:
            return self.inpsfgrp
        else:
            inpsfgrp = self.inpsfgrp
            del self.inpsfgrp; self.inpsfgrp = None
            return inpsfgrp

    def clear(self) -> None:
        '''
        Free up memory space.

        Returns
        -------
        None.

        '''

        del self.pix_count, self.pix_cumsum
        del self.y_val, self.x_val, self.data


class OutStamp:
    '''
    Driver for postage stamp coaddition.

    Methods
    -------
    __init__ : Constructor.
    _process_input_stamps : Fetch and process input postage stamps.
    __call__ : Build system matrices and perform coaddition.
    _build_system_matrices : Build system matrices and coaddition matrices.
    _visualize_system_matrices : Visualize system matrices.
    _visualize_coadd_matrices : Visualize coaddition matrices.

    trapezoid (staticmethod) : Apply a trapezoid filter to transition pixels.
    _perform_coaddition : Perform the actual multiplication.
    _visualize_weight_computations : Display weight computations.
    _show_in_and_out_images : Display input and output images.
    _study_individual_pixels : Study individual input and output pixels.
    clear : Free up memory space.

    '''

    LAKERNEL = {
        'Eigen'    : EigenKernel,
        'Cholesky' : CholKernel,
        'Iterative': IterKernel,
        'Empirical': EmpirKernel,
    }

    def __init__(self, blk: 'Block', j_st: int, i_st: int) -> None:
        '''
        Constructor.

        Parameters
        ----------
        blk : Block
            The Block instance to which this InStamp instance is attached.
        j_st, i_st : int, int
            OutStamp index.

        Returns
        -------
        None.

        '''

        self.blk = blk
        self.j_st = j_st
        self.i_st = i_st

        # list of indices of overlapping and adjacent input postage stamps
        # the final _s indicates plural
        self.ji_st_in_s = [(j_st+dj, i_st+di) for dj in range(-1, 2) for di in range(-1, 2)]

        # no-quality control option of the empirical kernel
        self.no_qlt_ctrl = False
        if blk.cfg.linear_algebra == 'Empirical':
            self.no_qlt_ctrl = blk.cfg.no_qlt_ctrl

        # count references to PSF overlaps and system submatrices
        if not self.no_qlt_ctrl:
            for ji_st_in in self.ji_st_in_s:
                blk.sysmata.get_iisubmat(ji_st_in, ji_st_in, sim_mode=True)
                blk.sysmatb.get_iosubmat(ji_st_in, (j_st, i_st), sim_mode=True)

            for ji_st_pair in combinations(self.ji_st_in_s, 2):
                blk.sysmata.get_iisubmat(*ji_st_pair, sim_mode=True)

        # limit y and x positions of this output postage stamp, all integers
        # not including the transition pixels (of which the number
        # of columns or rows on each side is set by fade_kernel)
        self.bottom = (j_st - 1)  * blk.cfg.n2
        self.top    = self.bottom + blk.cfg.n2-1
        self.left   = (i_st - 1)  * blk.cfg.n2
        self.right  = self.left   + blk.cfg.n2-1

        fade_kernel = blk.cfg.fade_kernel  # shortcut
        # output pixel positions, all integers
        self.yx_val = np.mgrid[self.bottom-fade_kernel:self.top+fade_kernel+1,
                               self.left-fade_kernel:self.right+fade_kernel+1]

        self._process_input_stamps()

    def _process_input_stamps(self, visualize: bool = False) -> None:
        '''
        Fetch and process input postage stamps.

        This method selects input pixels to form a region like this:
        +-----+-----+-----+
        |   **|*****|**   |
        | ****|*****|**** |
        +-----+-----+-----+
        |*****|*****|*****|
        |*****|*****|*****|
        +-----+-----+-----+
        | ****|*****|**** |
        |   **|*****|**   |
        +-----+-----+-----+
        where the central postage stamp is the OutStamp we are coadding.

        Parameters
        ----------
        visualize : bool, optional
            Whether to visualize the process. The default is False.

        Returns
        -------
        None.

        '''

        # fetch instamps and select input pixels
        self.instamps   = [None for _ in range(9)]
        self.selections = [None for _ in range(9)]
        self.inpix_count = np.zeros((9,), dtype=np.uint32)

        # acceptance radius in units of output pixels
        rpix_search = (self.blk.cfg.instamp_pad / Stn.arcsec) \
        / (self.blk.cfg.dtheta * u.degree.to('arcsec'))

        # now select input pixels
        for idx, ji_st_in in enumerate(self.ji_st_in_s):
            self.instamps[idx] = self.blk.instamps[ji_st_in[0]][ji_st_in[1]]

            x_pivot = [self.left-0.5, None, self.right+0.5][ji_st_in[1]-self.i_st+1]
            y_pivot = [self.bottom-0.5, None, self.top+0.5][ji_st_in[0]-self.j_st+1]
            self.selections[idx] = self.instamps[idx].make_selection((x_pivot, y_pivot), rpix_search)
            if self.selections[idx] is None:
                self.inpix_count[idx] = self.instamps[idx].pix_cumsum[-1]
            else:
                self.inpix_count[idx] = self.selections[idx].shape[0]

        self.inpix_cumsum = np.cumsum([0] + list(self.inpix_count), dtype=np.uint32)

        if visualize:
            fig, ax = plt.subplots(figsize=(4.8, 4.8))

            for idx, instamp, selection in zip(range(9), self.instamps, self.selections):
                if selection is None: plt.scatter(instamp.x_val, instamp.y_val, s=2, c=f'C{idx}')
                ax.scatter(instamp.x_val[selection], instamp.y_val[selection], s=2, c=f'C{idx}')
            ax.axis('equal')

            ax.set_xlabel('output grid $i$')
            ax.set_ylabel('output grid $j$')
            format_axis(ax)
            plt.show()

        # read input pixel positions and signals
        iny_val = []
        inx_val = []
        indata  = []

        for inst, selection in zip(self.instamps, self.selections):
            if selection is None:
                iny_val.append(inst.y_val)
                inx_val.append(inst.x_val)
                indata .append(inst.data)
            else:
                iny_val.append(inst.y_val[selection])
                inx_val.append(inst.x_val[selection])
                indata .append(inst.data [:, selection])

        self.iny_val = np.hstack(iny_val); iny_val.clear(); del iny_val
        self.inx_val = np.hstack(inx_val); inx_val.clear(); del inx_val
        self.indata  = np.hstack(indata);  indata.clear();  del indata

    def __call__(self, visualize: bool = False,
                 save_abc: bool = False, save_t: bool = False) -> None:
        '''
        Build system matrices and perform coaddition.

        Parameters
        ----------
        visualize : bool, optional
            Whether to visualize the process. The default is False.
        save_abc : bool, optional
            Whether to save system matrices. The default is False.
        save_t : bool, optional
            Whether to save coaddition matrices. The default is False.

        Returns
        -------
        None.

        '''

        self._build_system_matrices(visualize, save_abc)
        self._perform_coaddition(visualize, save_t)
        print()

    def _build_system_matrices(self, visualize: bool = False, save_abc: bool = False) -> None:
        '''
        Build system matrices and coaddition matrices.

        Parameters
        ----------
        visualize : bool, optional
            Whether to visualize the process. The default is False.
        save_abc : bool, optional
            Whether to save system matrices. The default is False.

        Returns
        -------
        None.

        '''

        # no-quality control option of the empirical kernel
        if self.no_qlt_ctrl:
            lakernel = OutStamp.LAKERNEL[self.blk.cfg.linear_algebra](self)
            lakernel(); del lakernel
            # this produces: self.T, (n_out, n_outpix, n_inpix)
            return

        # the A-matrix first
        self.sysmata = np.zeros((self.inpix_cumsum[-1], self.inpix_cumsum[-1]))  # dtype=np.float64
        use_virmem = bool(self.blk.cfg.tempfile)

        for idx, ji_st_in, selection in zip(range(9), self.ji_st_in_s, self.selections):
            iisubmat = self.blk.sysmata.get_iisubmat(ji_st_in, ji_st_in,
                ji_st_out=(self.j_st, self.i_st) if use_virmem else None)
            if selection is not None: iisubmat = iisubmat[np.ix_(selection, selection)]

            self.sysmata[self.inpix_cumsum[idx]:self.inpix_cumsum[idx+1],
                         self.inpix_cumsum[idx]:self.inpix_cumsum[idx+1]] = iisubmat

        for idx_s, ji_st_pair, selections_ in zip(combinations(range(9), 2),
                                                  combinations(self.ji_st_in_s, 2),
                                                  combinations(self.selections, 2)):
            iisubmat = self.blk.sysmata.get_iisubmat(*ji_st_pair,
                ji_st_out=(self.j_st, self.i_st) if use_virmem else None)
            if selections_[0] is not None:
                if selections_[1] is not None:
                    iisubmat = iisubmat[np.ix_(selections_[0], selections_[1])]
                else:
                    iisubmat = iisubmat[selections_[0], :]
            else:
                if selections_[1] is not None:
                    iisubmat = iisubmat[:, selections_[1]]

            self.sysmata[self.inpix_cumsum[idx_s[0]]:self.inpix_cumsum[idx_s[0]+1],
                         self.inpix_cumsum[idx_s[1]]:self.inpix_cumsum[idx_s[1]+1]] = iisubmat
            self.sysmata[self.inpix_cumsum[idx_s[1]]:self.inpix_cumsum[idx_s[1]+1],
                         self.inpix_cumsum[idx_s[0]]:self.inpix_cumsum[idx_s[0]+1]] = iisubmat.T
            del iisubmat

        # force exact symmetry
        # A = (A+A.T)/2.

        # now the mBhalf matrix
        self.mhalfb = np.zeros((self.blk.outpsfgrp.n_psf, self.blk.cfg.n2f**2,
                                self.inpix_cumsum[-1]))  # dtype=np.float64

        for idx, ji_st_in in zip(range(9), self.ji_st_in_s):
            self.mhalfb[:, :, self.inpix_cumsum[idx]:self.inpix_cumsum[idx+1]] =\
            self.blk.sysmatb.get_iosubmat(ji_st_in, (self.j_st, self.i_st))

        # and C
        self.outovlc = self.blk.outpsfovl.outovlc

        if visualize:
            print()
            self._visualize_system_matrices()

        lakernel = OutStamp.LAKERNEL[self.blk.cfg.linear_algebra](self)
        lakernel(); del lakernel
        # this produces: self.T, self.UC, self.Sigma, self.kappa
        # T: (n_out, n_outpix, n_inpix); others: (n_out, n2f, n2f)
        if not save_abc: del self.sysmata, self.mhalfb, self.outovlc

        if self.blk.cfg.linear_algebra == 'Iterative':
            # these could be negative as the iterative kernel is not exact
            self.UC    = np.maximum(self.UC,    1e-32)
            self.Sigma = np.maximum(self.Sigma, 1e-32)

        print('  n input pix =', self.T.shape[-1])
        sumstats = '  sqUC,sqSig %iles |'
        for i in [50, 90, 98, 99]:
            sumstats += ' {:2d}% {:8.2E} {:8.2E} |'.format(
                i, np.percentile(np.sqrt(self.UC), i),
                   np.percentile(np.sqrt(self.Sigma), i))
        print(sumstats, flush=True)

        if visualize:
            print()
            self._visualize_coadd_matrices()

        if self.blk.cfg.fade_kernel > 0:
            fade_kernel = self.blk.cfg.fade_kernel  # shortcut
            OutStamp.trapezoid(self.kappa, fade_kernel)
            OutStamp.trapezoid(self.Sigma, fade_kernel)
            OutStamp.trapezoid(self.UC,    fade_kernel)

    def _visualize_system_matrices(self) -> None:
        '''
        Visualize system matrices.

        Returns
        -------
        None.

        '''

        print('OutStamp._visualize_system_matrices')

        # the A-matrix first
        print(f'{self.sysmata.shape=}')  # (n_inpix, n_inpix)
        print(f'{np.all(self.sysmata == self.sysmata.T)=}')

        fig, ax = plt.subplots(figsize=(12.8, 9.6))
        vmin = self.sysmata.max() / (self.mhalfb.max() / self.mhalfb.min()) ** 2
        im = ax.imshow(np.log10(np.clip(self.sysmata, a_min=vmin, a_max=None)), vmin=np.log10(vmin))
        plt.colorbar(im, ax=ax)

        for xy in self.inpix_cumsum[1:-1]:
            ax.axvline(xy, c='r', ls='--', lw=1.5)
            ax.axhline(xy, c='r', ls='--', lw=1.5)

        ax.set_xlabel('input pixel $i$')
        ax.set_ylabel('input pixel $j$')
        ax.set_title(r'$A$ matrix: $\log_{10} (A_{ij})$')
        format_axis(ax, False)
        plt.show()

        # now the mBhalf matrix
        print(f'{self.mhalfb.shape=}')  # (n_out, n_outpix, n_inpix)
        n_out, n_outpix, n_inpix = self.mhalfb.shape
        height = 9.6 / n_inpix * n_outpix

        for mhalfb_ in self.mhalfb:
            fig, ax = plt.subplots(figsize=(12.8, height))
            im = ax.imshow(np.log10(mhalfb_))
            plt.colorbar(im, ax=ax)

            for x in self.inpix_cumsum[1:-1]:
                ax.axvline(x, c='r', ls='--', lw=1.5)

            ax.set_xlabel('input pixel $i$')
            ax.set_ylabel(r'output pixel $\alpha$')
            ax.set_title(r'$B$ matrix: $\log_{10} (-B_{\alpha i}/2)$')
            format_axis(ax, False)
            plt.show()

        # and C
        print(f'{self.outovlc.shape=}')  # (n_out,)

    def _visualize_coadd_matrices(self) -> None:
        '''
        Visualize coaddition matrices.

        Returns
        -------
        None.

        '''

        print('OutStamp._visualize_coadd_matrices')
        fk = self.blk.cfg.fade_kernel  # shortcut

        for j_out, T_ in enumerate(self.T):
            print(f'output PSF: {j_out}')

            fig, axs = plt.subplots(1, 3, figsize=(14.4, 3.6))
            for ax, map_, title in zip(axs, [self.UC, self.Sigma, self.kappa],
                                       [r'PSF leakage: $\log_{10} (U/C)$',
                                        r'Noise amplification: $\log_{10} \Sigma$',
                                        r'Lagrange multiplier: $\log_{10} \kappa$']):
                im = ax.imshow(np.log10(map_[j_out]), origin='lower',
                               extent=[self.left-fk, self.right+fk, self.bottom-fk, self.top+fk])
                plt.colorbar(im, ax=ax)

                ax.set_title(title)
                ax.set_xlabel('output grid $i$')
                ax.set_ylabel('output grid $j$')
                format_axis(ax, False)
            plt.show()

            vmin, vmax = np.percentile(T_.ravel(), [1, 99])
            n_out, n_outpix, n_inpix = self.mhalfb.shape
            height = 9.6 / n_inpix * n_outpix

            fig, ax = plt.subplots(figsize=(12.8, height))
            im = ax.imshow(T_, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)

            for x in self.inpix_cumsum[1:-1]:
                ax.axvline(x, c='r', ls='--', lw=1.5)

            ax.set_xlabel('input pixel $i$')
            ax.set_ylabel(r'output pixel $\alpha$')
            ax.set_title(r'$T$ matrix: $T_{\alpha i}$')
            format_axis(ax, False)
            plt.show()

    @staticmethod
    def trapezoid(arr: np.array, fade_kernel: int, recover_mode: bool = False,
                  pad_widths: (int, int, int, int) = (0, 0, 0, 0),
                  do_sides: str = 'BTLR', use_trunc_sinc: bool = True) -> None:
        '''
        Apply a trapezoid filter of width 2*fade_kernel on each side.

        Parameters
        ----------
        arr : np.array, shape : (..., ny, nx)
            The array to apply the trapezoid filter.
        fade_kernel : int
            Half the width of the trapezoid filter.
        recover_mode : bool, optional
            Whether to recover faded boundaries. The default is False.
        pad_widths : (int, int, int, int), optional
            Padding width on each side (order: bottom, top, left, right).
            The default is (0, 0, 0, 0).
        do_sides : str, optional
            Which sides to apply the trapezoid filter. The default is 'BTLR'.
        use_trunc_sinc : bool, optional
            Whether to use the truncated sinc function. The default is True.

        Returns
        -------
        None.

        '''

        fk2 = fade_kernel * 2
        if not fk2 > 0: return

        ny, nx = arr.shape[-2:]
        # assert ny > fk2 and nx > fk2, 'Fatal error in OutStamp.trapezoid: ' \
        #     f'insufficient patch size, {ny= } {nx= } {fade_kernel= }'

        pb, pt, pl, pr = pad_widths
        # assert ny > pb and ny > pt, 'Fatal error in OutStamp.trapezoid: ' \
        #     f'insufficient patch size, {ny= } {pb= } {pt= }'
        # assert nx > pl and nx > pr, 'Fatal error in OutStamp.trapezoid: ' \
        #     f'insufficient patch size, {nx= } {pl= } {pr= }'
        it, ir = ny-pt-1, nx-pr-1  # starting indices on these two sides

        s = np.arange(1, fk2+1, dtype=np.float64) / (fk2+1)
        if use_trunc_sinc:
            s -= np.sin(2*np.pi*s)/(2*np.pi)
        sT = s[None, :].T

        if not recover_mode:
            if 'B' in do_sides: arr[..., pb:pb+fk2   , :] *= sT
            if 'T' in do_sides: arr[..., it:it-fk2:-1, :] *= sT
            if 'L' in do_sides: arr[..., :, pl:pl+fk2   ] *= s
            if 'R' in do_sides: arr[..., :, ir:ir-fk2:-1] *= s

        else:  # recover block boundaries
            if 'B' in do_sides: arr[..., pb:pb+fk2   , :] /= sT
            if 'T' in do_sides: arr[..., it:it-fk2:-1, :] /= sT
            if 'L' in do_sides: arr[..., :, pl:pl+fk2   ] /= s
            if 'R' in do_sides: arr[..., :, ir:ir-fk2:-1] /= s

    def _perform_coaddition(self, visualize: bool = False,
                            save_t: bool = False, use_trunc_sinc: bool = True) -> None:
        '''
        Perform the actual multiplication.

        Parameters
        ----------
        visualize : bool, optional
            Whether to visualize the process. The default is False.
        save_t : bool, optional
            Whether to save coaddition matrices. The default is False.
        use_trunc_sinc : bool, optional
            Argument for coadd_utils.trapezoid. The default is True.

        Returns
        -------
        None.

        '''

        # shortcuts
        n_out = self.blk.outpsfgrp.n_psf
        n2f = self.blk.cfg.n2f
        fade_kernel = self.blk.cfg.fade_kernel

        # multiplication by the trapezoid filter with transition width fade_kernel
        if fade_kernel > 0:
            # self.T: (n_out, n_outpix, n_inpix)
            T_view = np.moveaxis(self.T, 1, -1).reshape(
                (n_out, self.inpix_cumsum[-1], n2f, n2f))
            OutStamp.trapezoid(T_view, fade_kernel)

        # weight computations
        Tsum_image = np.zeros(self.T.shape[:2] + (self.blk.n_inimage,))  # (n_out, n_outpix, n_inimage)

        for j_st, inst, selection in zip(range(9), self.instamps, self.selections):
            if selection is None:
                my_cumsum = inst.pix_cumsum.copy()
            else:
                my_cumsum = np.searchsorted(selection, inst.pix_cumsum)
            my_cumsum += self.inpix_cumsum[j_st]

            for i_im in range(self.blk.n_inimage):
                Tsum_image[:, :, i_im] += np.sum(self.T[:, :, my_cumsum[i_im]:my_cumsum[i_im+1]], axis=2)

        self.Tsum_stamp = np.sum(Tsum_image, axis=1) / self.blk.cfg.n2 ** 2  # (n_out, n_inimage)
        self.Tsum_inpix = np.sum(Tsum_image, axis=2).reshape((n_out, n2f, n2f))
        Tsum_norm = Tsum_image / np.abs(Tsum_image).sum(axis=2)[:, :, None]
        self.Neff = 1.0 / np.sum(np.square(Tsum_norm), axis=2).reshape((n_out, n2f, n2f))
        if fade_kernel > 0: OutStamp.trapezoid(self.Neff, fade_kernel)

        del Tsum_image, Tsum_norm
        if visualize:
            print()
            self._visualize_weight_computations()

        # the actual multiplication
        self.outimage = np.einsum('oaj,ij->oia', self.T, self.indata).reshape(
            (n_out, self.blk.cfg.n_inframe, n2f, n2f))

        if visualize:
            print()
            self._show_in_and_out_images()
            print()
            self._study_individual_pixels()
        del self.iny_val, self.inx_val, self.indata
        if not save_t: del self.T

    def _visualize_weight_computations(self) -> None:
        
        '''
        Display weight computations.

        Returns
        -------
        None.

        '''

        print('OutStamp._visualize_weight_computations')
        fk = self.blk.cfg.fade_kernel  # shortcut

        for j_out in range(self.blk.outpsfgrp.n_psf):
            print(f'output PSF: {j_out}')

            fig, axs = plt.subplots(1, 3, figsize=(14.4, 3.6))

            ax = axs[0]
            ax.barh([f'${inimage.idsca}$' for inimage in self.blk.inimages], self.Tsum_stamp[0],
                     color=[f'C{i}' for i in range(self.blk.n_inimage)])

            ax.set_title('Total contribution')
            ax.set_xlabel(r'$\sum {}_\alpha \sum {}_{i \in \bar{i}} T_{\alpha i}$')
            ax.set_ylabel('input image')
            format_axis(ax, False)

            for ax, map_, title in zip(axs[1:], [self.Tsum_inpix, self.Neff],
                                       [r'Total weight: $T_{\rm tot}$',
                                        r'Effective coverage: $\bar{n}_{\rm eff}$']):
                im = ax.imshow(map_[j_out], origin='lower',
                               extent=[self.left-fk, self.right+fk, self.bottom-fk, self.top+fk])
                plt.colorbar(im, ax=ax)

                ax.set_title(title)
                ax.set_xlabel('output grid $i$')
                ax.set_ylabel('output grid $j$')
                format_axis(ax, False)
            plt.show()

    def _show_in_and_out_images(self) -> None:
        '''
        Display input and output images.

        Returns
        -------
        None.

        '''

        print('OutStamp._show_in_and_out_images')
        fk = self.blk.cfg.fade_kernel  # shortcut

        for j_in in range(self.blk.cfg.n_inframe):
            n_out = self.blk.outpsfgrp.n_psf
            fig, axs = plt.subplots(1, 1+n_out, figsize=(4.8*(1+n_out), 3.6))

            ax = axs[0]
            im = ax.scatter(self.inx_val, self.iny_val, c=self.indata[j_in], cmap='viridis', s=5)
            plt.colorbar(im, ax=ax)
            for x in [self.left-0.5, self.right+0.5]: ax.axvline(x, c='r', ls='--')
            for y in [self.bottom-0.5, self.top+0.5]: ax.axhline(y, c='r', ls='--')

            ax.axis('equal')
            ax.set_xlabel('output grid $i$')
            ax.set_ylabel('output grid $j$')
            ax.set_title('layer: ' + ('SCI' if j_in == 0 else self.blk.cfg.extrainput[j_in]))
            format_axis(ax)

            for j_out in range(n_out):
                ax = axs[1+j_out]
                im = ax.imshow(self.outimage[j_out, j_in], origin='lower',
                               extent=[self.left-fk, self.right+fk, self.bottom-fk, self.top+fk])
                plt.colorbar(im, ax=ax)

                ax.set_xlabel('output grid $i$')
                ax.set_ylabel('output grid $j$')
                ax.set_title(f'output PSF: {j_out}')
                format_axis(ax, False)

            plt.show()

    def _study_individual_pixels(self) -> None:
        '''
        Study individual input and output pixels.

        Returns
        -------
        None.

        '''

        print('OutStamp._study_individual_pixels')

        accrad = np.arange(15, 140, 5)  # acceptance radius in units of output pixels
        closest_inpix = []  # indices of input pixels closest to the corners and the center

        fk = self.blk.cfg.fade_kernel
        fk2 = fk * 2
        n2f = self.blk.cfg.n2f  # shortcuts
        for j_out, i_out in [(fk2, fk2), (fk2, n2f-1-fk2), (n2f-1-fk2, fk2),
                             (n2f-1-fk2, n2f-1-fk2), ((n2f-1)//2, (n2f-1)//2)]:
            out_idx = j_out * self.blk.cfg.n2f + i_out
            T_elems = self.T[0, out_idx, :]

            plt.figure(figsize=(10.8, 3.6))
            ax0 = plt.subplot(1, 2, 1)
            im = ax0.scatter(self.inx_val, self.iny_val, c=T_elems, cmap='viridis', s=5)
            plt.colorbar(im, ax=ax0)

            ax0.axis('equal')
            ax0.set_xlabel('output grid $i$')
            ax0.set_ylabel('output grid $j$')
            ax0.set_title(r'$T_{\alpha i}$')
            format_axis(ax0)

            # print(j_out-fk + self.bottom, i_out-fk + self.left)
            dist = np.sqrt(np.square(j_out-fk + self.bottom - self.iny_val) +\
                           np.square(i_out-fk + self.left   - self.inx_val))
            closest_inpix.append(np.argmin(dist))

            ax1 = plt.subplot(2, 2, 2)
            ax1.scatter(dist, T_elems, c=T_elems, cmap='viridis', s=5)
            ax1.axhline(              0, c='b', ls='--')
            ax1.axvline(self.blk.cfg.n2, c='r', ls='--')

            ax1.xaxis.set_ticklabels([])
            ax1.set_ylabel(r'$T_{\alpha i}$')
            format_axis(ax1)

            signal = np.empty_like(accrad, dtype=np.float64)
            for i in range(accrad.shape[0]):
                T_arr = np.where(dist <= accrad[i], T_elems, 0.0)
                signal[i] = T_arr @ self.indata[0]

            ax2 = plt.subplot(2, 2, 4)
            ax2.plot(accrad, signal, 'go-')
            ax2.axvline(self.blk.cfg.n2, c='r', ls='--')
            ax2.axhline(self.outimage[0, 0].ravel()[out_idx], c='b', ls='--')

            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xlabel('acceptance radius')
            ax2.set_ylabel('signal')
            format_axis(ax2)

            plt.show()

        for in_idx in closest_inpix:
            print(f'{(self.inx_val[in_idx], self.iny_val[in_idx])=}')
            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            im = ax.imshow(self.T[0, :, in_idx].reshape(n2f, n2f), origin='lower',
                           extent=[self.left-fk, self.right+fk, self.bottom-fk, self.top+fk])
            plt.colorbar(im, ax=ax)
            ax.scatter(self.inx_val[in_idx] - self.left,
                       self.iny_val[in_idx] - self.bottom, c='r', s=2)

            ax.set_xlabel('output grid $i$')
            ax.set_ylabel('output grid $j$')
            ax.set_title(r'$T_{\alpha i}$')
            format_axis(ax, False)

            plt.show()

    def clear(self) -> None:
        '''
        Free up memory space.

        Returns
        -------
        None.

        '''

        del self.inpix_count, self.inpix_cumsum
        self.selections.clear(); del self.selections

        if hasattr(self, 'sysmata'): del self.sysmata, self.mhalfb, self.outovlc
        if hasattr(self, 'T'):       del self.T

        del self.kappa, self.Sigma, self.UC
        del self.Tsum_stamp, self.Tsum_inpix, self.Neff
        del self.yx_val, self.outimage


class Block:
    '''
    Driver for block coaddition.

    Methods
    -------
    __init__ : Constructor.
    __call__ : Coadd this block.
    parse_config : Parse the configuration.
    _get_obs_cover: Get observations relevant to this Block.

    _build_use_instamps : Build the Boolean array use_instamps.
    _handle_postage_pad : Handle the postage padding.
    process_input_images : Process input images.
    build_input_stamps : Build input stamps from input images.

    _output_stamp_wrapper : Wrapper for output stamp coaddition.
    coadd_output_stamps : Coadd output stamps using input stamps.
    compress_map : Compress float32 map to (u)int16 to save storage.
    build_output_file : Build the output FITS file.
    clear_all : Free up all memory spaces.

    '''

    def __init__(self, cfg: Config = None, this_sub: int = 0,
                 run_coadd: bool = True) -> None:
        '''
        Constructor.

        Parameters
        ----------
        cfg : Config, optional
            Configuration for this Block. The default is None.
        this_sub : int, optional
            Number determining the location of this Block in the mosaic. The default is 0.
        run_coadd : bool, optional
            Whether to coadd this block. The default is True.
            Turn this off if you want to perform the procedure manually.

        Returns
        -------
        None.

        '''

        self.timer = Timer()
        cfg(); self.cfg = cfg
        if cfg is None: self.cfg = Config()  # use the default config

        PSFGrp.setup(npixpsf=cfg.npixpsf, oversamp=cfg.inpsf_oversamp, dtheta=cfg.dtheta, psfsplit=bool(cfg.psfsplit))
        PSFOvl.setup(flat_penalty=cfg.flat_penalty)
        self.this_sub = this_sub
        if run_coadd: self()

    def __call__(self) -> None:
        '''
        Coadd this block.

        Returns
        -------
        None.

        '''

        self.parse_config()
        # this produces: obsdata, outwcs, outpsfgrp, outpsfovl
        self.process_input_images()
        # this produces: obslist, inimages (1D list), n_inimage
        self.build_input_stamps()
        # this produces: instamps (2D list)

        self.coadd_output_stamps(sim_mode=True)
        # this produces: sysmata (object), sysmatb (object), outstamps (2D list)
        self.coadd_output_stamps(sim_mode=False)
        # this produces: out_map, T_weightmap, and additional output maps

        self.build_output_file(is_final=True)
        self.clear_all()
        print(f'finished at t = {self.timer():.2f} s')

    def parse_config(self) -> None:
        '''
        Parse the configuration.

        Returns
        -------
        None.

        '''

        print('General input information:')
        print('number of input frames = ', self.cfg.n_inframe, 'type =', self.cfg.extrainput)
        # search radius for input pixels
        rpix_search_ = self.cfg.instamp_pad / Stn.arcsec
        dtheta_ = self.cfg.dtheta * u.degree.to('arcsec')
        print(f'acceptance radius --> {rpix_search_:.6f} arcsec '
              f'or {rpix_search_/dtheta_:.6f} output pixels')
        print()

        # Get observation table
        assert self.cfg.obsfile is not None, 'Error: no obsfile found'
        print('Getting observations from {:s}'.format(self.cfg.obsfile))
        with fits.open(self.cfg.obsfile) as myf:
            # if the filter column is a string, replace with the number
            if myf[1].columns['filter'].format[-1]=='A':
                print('converting filter from names to integers:', myf[1].data['filter'])
                n_obs_tot = len(myf[1].data.field(0))
                mytable = Table(myf[1].data)
                fdata = np.zeros(n_obs_tot, dtype=np.uint16)
                for j in range(len(Stn.RomanFilters)):
                    s = Stn.RomanFilters[j]
                    for i in range(n_obs_tot):
                        if myf[1].data['filter'][i] == s:
                            fdata[i]=j
                mytable.replace_column('filter', fdata)
                myf[1] = fits.BinTableHDU(mytable)

            self.obsdata = myf[1].data
            obscols = myf[1].columns
            n_obs_tot = len(self.obsdata.field(0))
            print('Retrieved columns:', obscols.names, ' {:d} rows'.format(n_obs_tot))

        # display output information
        print('Output information: ctr at RA={:10.6f},DEC={:10.6f} LONPOLE={:10.6f}'.format(self.cfg.ra, self.cfg.dec, self.cfg.lonpole))
        print('pixel scale={:8.6f} arcsec or {:11.5E} degree'.format(
            self.cfg.dtheta * u.degree.to('arcsec'), self.cfg.dtheta))
        print('output array size = {:d} ({:d} postage stamps of {:d})'.format(
            self.cfg.NsideP, self.cfg.n1P, self.cfg.n2))
        print()

        # block information
        ibx, iby = divmod(self.this_sub, self.cfg.nblock)
        print('sub-block {:4d} <{:2d},{:2d}> of {:2d}x{:2d}={:2d}'.format(self.this_sub,
              ibx, iby, self.cfg.nblock, self.cfg.nblock, self.cfg.nblock**2))
        self.outstem = self.cfg.outstem + '_{:02d}_{:02d}'.format(ibx, iby)
        print('outputs directed to -->', self.outstem)
        if self.cfg.tempfile is not None:
            self.cache_dir = Path(self.cfg.tempfile + '_{:04d}_{:s}_cache'.format(self.this_sub,
                datetime.datetime.now(pytz.timezone('UTC')).strftime("%Y%m%d%H%M%S%f")))
            self.cache_dir.mkdir(exist_ok=True)
            print('temporary storage directed to -->', self.cache_dir)

        # make the WCS
        self.outwcs = wcs.WCS(naxis=2)
        self.outwcs.wcs.crpix = [(self.cfg.NsideP+1)/2. - self.cfg.Nside*(ibx-(self.cfg.nblock-1)/2.),
                                 (self.cfg.NsideP+1)/2. - self.cfg.Nside*(iby-(self.cfg.nblock-1)/2.)]
        self.outwcs.wcs.cdelt = [-self.cfg.dtheta, self.cfg.dtheta]
        self.outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        self.outwcs.wcs.crval = [self.cfg.ra, self.cfg.dec]
        self.outwcs.wcs.lonpole = self.cfg.lonpole

        # print the corners of the square and the center, ordering:
        #   2   3
        #     4
        #   0   1
        cornerx = [-.5, self.cfg.NsideP-.5, -.5, self.cfg.NsideP-.5, (self.cfg.NsideP-1)/2.]
        cornery = [-.5, -.5, self.cfg.NsideP-.5, self.cfg.NsideP-.5, (self.cfg.NsideP-1)/2.]
        for i in range(5):
            print(i, self.outwcs.all_pix2world(np.array([[cornerx[i], cornery[i]]]), 0))
        self.centerpos = self.outwcs.all_pix2world(np.array(
            [[cornerx[-1], cornery[-1]]]), 0)[0]  # [ra,dec] array in degrees

        print('kappa/C array', self.cfg.kappaC_arr)

        # and the output PSFs
        self.outpsfgrp = PSFGrp(in_or_out=False, blk=self)
        self.outpsfovl = PSFOvl(self.outpsfgrp, None)
        print('computed overlap, C=', self.outpsfovl.outovlc)
        print()

    def _get_obs_cover(self, radius: float) -> None:
        '''
        Get observations relevant to this Block.

        Previous comments:
          select observation/SCA pairs from a observation table

        Inputs:
          obsdata (the observation table)
          ra (in degrees)
          dec (in degrees)
          radius (in degrees)
          filter (integer, 0..10)

        Outputs:
          list of (observation, sca) that overlaps this target
          [note SCA numbers are 1..18 for ease of comparison with the Project notation]

        Parameters
        ----------
        radius : float
            Search for input images within this radius.

        Returns
        -------
        None.

        '''

        self.obslist = []
        n_obs_tot = len(self.obsdata.field(0))

        # rotate this observation to the (X,Y) of the local FoV for each observation
        # first rotate the RA direction
        x1 = np.cos(self.centerpos[1]*Stn.degree) \
           * np.cos((self.centerpos[0]-self.obsdata['ra'])*Stn.degree)
        y1 = np.cos(self.centerpos[1]*Stn.degree) \
           * np.sin((self.centerpos[0]-self.obsdata['ra'])*Stn.degree)
        z1 = np.sin(self.centerpos[1]*Stn.degree)*np.ones((n_obs_tot,))
        # then rotate the Dec direction
        x2 = np.sin(self.obsdata['dec']*Stn.degree)*x1 \
           - np.cos(self.obsdata['dec']*Stn.degree)*z1
        y2 = y1
        z2 = np.cos(self.obsdata['dec']*Stn.degree)*x1 \
           + np.sin(self.obsdata['dec']*Stn.degree)*z1
        # and finally the PA direction
        X = (-np.sin(self.obsdata['pa']*Stn.degree)*x2 \
          -   np.cos(self.obsdata['pa']*Stn.degree)*y2) / Stn.degree
        Y = (-np.cos(self.obsdata['pa']*Stn.degree)*x2 \
          +   np.sin(self.obsdata['pa']*Stn.degree)*y2) / Stn.degree
        #
        # throw away points in wrong hemisphere -- important since in orthographic projection, can have (X,Y)=0 for antipodal point
        X = np.where(z2 > 0, X, 1e49)

        for isca in range(18):
            obsgood = np.where(np.logical_and(
                np.sqrt((X-Stn.SCAFov[isca][0])**2 + (Y-Stn.SCAFov[isca][1])**2) < radius,
                self.obsdata['filter'] == self.cfg.use_filter))
            for k in range(len(obsgood[0])):
                self.obslist.append((obsgood[0][k], isca+1))

        self.obslist.sort()

    def _build_use_instamps(self) -> None:
        '''
        Build use_instamps, Boolean array indicating
        whether to use each input postage stamp.

        Returns
        -------
        None.

        '''

        self.use_instamps = np.zeros((self.cfg.n1P+2, self.cfg.n1P+2), dtype=bool)

        n_coadded = 0  # number of output postage stamps to be coadded
        for j_st in range(self.j_st_min, self.j_st_max+1, 2):
            for i_st in range(self.i_st_min, self.i_st_max+1, 2):
                for dj, di in product(range(2), range(2)):
                    self.use_instamps[j_st+dj-1:j_st+dj+2, i_st+di-1:i_st+di+2] = True

                    n_coadded += 1
                    if n_coadded == self.nrun: return

    def _handle_postage_pad(self) -> None:
        '''
        Handle the postage padding.

        Returns
        -------
        None.

        '''

        postage_pad = self.cfg.postage_pad  # shortcut
        self.j_st_min = self.i_st_min = postage_pad + 1  # 3 by default
        self.j_st_max = self.i_st_max = self.j_st_min + self.cfg.n1 - 1  # 50 by default
        self.pad_sides = ''  # will also be used in build_output_file

        # adjust these based on which side(s) to pad on
        if self.cfg.pad_sides == 'all':  # pad on all sides
            self.pad_sides = 'BTLR'

        elif self.cfg.pad_sides == 'auto':  # pad on mosaic boundaries only
            nblock = self.cfg.nblock  # shortcut
            ibx, iby = divmod(self.this_sub, self.cfg.nblock)
            if iby == 0: self.pad_sides += 'B'
            elif iby == nblock - 1: self.pad_sides += 'T'
            if ibx == 0: self.pad_sides += 'L'
            elif ibx == nblock - 1: self.pad_sides += 'R'

        elif self.cfg.pad_sides != 'none':  # pad on sides specified by the user
            self.pad_sides = self.cfg.pad_sides

        if 'B' in self.pad_sides: self.j_st_min -= postage_pad
        if 'T' in self.pad_sides: self.j_st_max += postage_pad
        if 'L' in self.pad_sides: self.i_st_min -= postage_pad
        if 'R' in self.pad_sides: self.i_st_max += postage_pad

        self.nrun = (self.j_st_max - self.j_st_min + 1) \
                  * (self.i_st_max - self.i_st_min + 1)
        if self.cfg.stoptile: self.nrun = self.cfg.stoptile
        self._build_use_instamps()

    def process_input_images(self, visualize: bool = False) -> None:
        '''
        Process input images.

        Returns
        -------
        None.

        '''

        ### Now figure out which observations we need ###

        search_radius = Stn.sca_sidelength / np.sqrt(2.) / Stn.degree \
                      + self.cfg.NsideP * self.cfg.dtheta / np.sqrt(2.)
        self._get_obs_cover(search_radius)
        print(len(self.obslist), 'observations within range ({:7.5f} deg)'.format(search_radius),
              'filter =', self.cfg.use_filter, '({:s})'.format(Stn.RomanFilters[self.cfg.use_filter]))

        self.inimages = [InImage(self, idsca) for idsca in self.obslist]
        any_exists = False
        print('The observations -->')
        print('  OBSID SCA  RAWFI    DECWFI   PA     RASCA   DECSCA       FILE (x=missing)')
        for idsca, inimage in zip(self.obslist, self.inimages):
            cpos = '                 '
            if inimage.exists_:
                any_exists = True
                cpos_coord = inimage.inwcs.all_pix2world([[Stn.sca_ctrpix, Stn.sca_ctrpix]], 0)[0]
                cpos = '{:8.4f} {:8.4f}'.format(cpos_coord[0], cpos_coord[1])
            print('{:7d} {:2d} {:8.4f} {:8.4f} {:6.2f} {:s} {:s} {:s}'.format(
                idsca[0], idsca[1], self.obsdata['ra'][idsca[0]], self.obsdata['dec'][idsca[0]],
                self.obsdata['pa'][idsca[0]], cpos, ' ' if inimage.exists_ else 'x', inimage.infile))
        print()
        assert any_exists, 'No candidate observations found to stack. Exiting now.'

        print('Reading input data ... ')
        self.pmask = Mask.load_permanent_mask(self)
        print()

        self._handle_postage_pad()
        for inimage in self.inimages:
            if not inimage.exists_:
                inimage.is_relevant = False
                continue
            inimage.partition_pixels(visualize=visualize)
            # visualize = False  # For now, visualize at most one partitioning process.
            if not inimage.is_relevant: continue
            inimage.extract_layers()
            print()
        del self.pmask

        # remove irrelevant input images
        self.obslist = [self.obslist[i] for i, inimage in enumerate(self.inimages) if inimage.is_relevant]
        self.inimages = [inimage for inimage in self.inimages if inimage.is_relevant]
        self.n_inimage = len(self.inimages)

    def build_input_stamps(self) -> None:
        '''
        Build input stamps from input images.

        Returns
        -------
        None.

        '''

        n1P = self.cfg.n1P  # shortcuts
        pad = self.cfg.postage_pad

        # current version only works when acceptance radius <= postage stamp size
        self.instamps = [[None for i_st in range(n1P+2)]
                               for j_st in range(n1P+2)]  # st stands for stamp

        n_inpix_out = 0  # number of input pixels in output region (not including padding)
        n_inpix_pad = 0  # number of input pixels in padding region

        for j_st in range(n1P+2):
            for i_st in range(n1P+2):
                if self.use_instamps[j_st, i_st]:
                    self.instamps[j_st][i_st] = InStamp(self, j_st, i_st)

                    if (pad < j_st <= n1P-pad) and (pad < i_st <= n1P-pad):
                        n_inpix_out += self.instamps[j_st][i_st].pix_cumsum[-1]
                    elif (0 < j_st <= n1P) and (0 < i_st <= n1P):
                        n_inpix_pad += self.instamps[j_st][i_st].pix_cumsum[-1]

        print(f'number of input pixels in output region: {n_inpix_out = }')
        print(f'number of input pixels in padding region: {n_inpix_pad = }')
        print()

        del self.use_instamps
        for inimage in self.inimages:
            inimage.clear()

    def _output_stamp_wrapper(self, i_st, j_st, n_coadded, sim_mode: bool = False):
        '''
        Wrapper for output stamp coaddition.

        Parameters
        ----------
        j_st, i_st : int, int
            OutStamp index.
        sim_mode : bool, optional
            Whether to count references without actually making inpsfgrp.
            See the docstring of psfutil.SysMatA._compute_iisubmats.
            The default is False.

        Returns
        -------
        None.

        '''

        assert 1 <= i_st <= self.cfg.n1P and 1 <= j_st <= self.cfg.n1P, 'outstamp out of boundary'

        if sim_mode:  # count references to PSF overlaps and system submatrices
            self.outstamps[j_st][i_st] = OutStamp(self, j_st, i_st)
        else:
            print('postage stamp {:2d},{:2d}  {:6.3f}% t= {:9.2f} s'.format(
                i_st, j_st, 100*n_coadded/self.nrun, self.timer()), flush=True)
            outst = self.outstamps[j_st][i_st]; outst()

            bottom = (j_st-1) * self.cfg.n2
            top    =  j_st    * self.cfg.n2 + self.cfg.fade_kernel*2
            left   = (i_st-1) * self.cfg.n2
            right  =  i_st    * self.cfg.n2 + self.cfg.fade_kernel*2

            self.out_map[:, :, bottom:top, left:right] += outst.outimage
            self.T_weightmap[:, :, j_st-1, i_st-1] = outst.Tsum_stamp  # weight computations

            outmaps = self.cfg.outmaps  # shortcut
            if 'U' in outmaps: self.UC_map   [:, bottom:top, left:right] += outst.UC          # fidelity map
            if 'S' in outmaps: self.Sigma_map[:, bottom:top, left:right] += outst.Sigma       # noise map
            if 'K' in outmaps: self.kappa_map[:, bottom:top, left:right] += outst.kappa       # kappa map
            if 'T' in outmaps: self.Tsum_map [:, bottom:top, left:right] += outst.Tsum_inpix  # total weight
            if 'N' in outmaps: self.Neff_map [:, bottom:top, left:right] += outst.Neff        # effective coverage

            outst.clear(); self.outstamps[j_st][i_st] = None
            # coadding the OutStamp above is the last show of the InStamp below
            inst = self.instamps[j_st-1][i_st-1]
            inst.clear(); self.instamps[j_st-1][i_st-1] = None

    def coadd_output_stamps(self, sim_mode: bool = False) -> None:
        '''
        Coadd output stamps using input stamps.

        Parameters
        ----------
        sim_mode : bool, optional
            Whether to count references without actually making inpsfgrp.
            See the docstring of psfutil.SysMatA._compute_iisubmats.
            The default is False.

        Returns
        -------
        None.

        '''

        if sim_mode:  # count references to PSF overlaps and system submatrices
            self.sysmata = SysMatA(self)
            self.sysmatb = SysMatB(self)
            self.outstamps = [[None for i_st in range(self.cfg.n1P+2)]
                                    for j_st in range(self.cfg.n1P+2)]

        else:
            n_out   = self.outpsfgrp.n_psf
            NsidePf = self.cfg.NsideP + self.cfg.fade_kernel * 2

            # make basic output array (including transition pixels on the boundaries)
            self.out_map = np.zeros((n_out, self.cfg.n_inframe, NsidePf, NsidePf), dtype=np.float32)
            # allocate ancillary arrays
            self.T_weightmap = np.zeros((n_out, self.n_inimage,
                                         self.cfg.n1P, self.cfg.n1P), dtype=np.float32)

            # additional information (will convert to integer)
            outmaps = self.cfg.outmaps  # shortcut
            shape = (n_out, NsidePf, NsidePf)
            if 'U' in outmaps: self.UC_map    = np.zeros(shape, dtype=np.float32)  # fidelity map
            if 'S' in outmaps: self.Sigma_map = np.zeros(shape, dtype=np.float32)  # noise map
            if 'K' in outmaps: self.kappa_map = np.zeros(shape, dtype=np.float32)  # kappa map
            if 'T' in outmaps: self.Tsum_map  = np.zeros(shape, dtype=np.float32)  # total weight
            if 'N' in outmaps: self.Neff_map  = np.zeros(shape, dtype=np.float32)  # effective coverage

        ### Begin loop over all the postage stamps we want to create ###

        n_coadded = 0  # number of coadded postage stamps
        for j_st in range(self.j_st_min, self.j_st_max+1, 2):
            for i_st in range(self.i_st_min, self.i_st_max+1, 2):
                for dj, di in product(range(2), range(2)):
                    self._output_stamp_wrapper(i_st+di, j_st+dj, n_coadded, sim_mode)
                    n_coadded += 1

                    if n_coadded == self.nrun:
                        if sim_mode:
                            self.sysmata.iisubmats.clear()
                            self.sysmatb.iopsfovls.clear()
                        else:
                            assert len(self.sysmata.iisubmats) == 0, 'self.sysmata.iisubmats is not empty'
                            assert len(self.sysmatb.iopsfovls) == 0, 'self.sysmatb.iopsfovls is not empty'
                        return

                if not sim_mode:
                    gc.collect()  # force a garbage collection

            if not sim_mode:
                for i_st in range(self.i_st_min, self.i_st_max+1, 2):
                    for dj in range(-1, 1):
                        inst = self.instamps[j_st+dj][i_st]
                        if inst is not None:
                            inst.clear()
                            self.instamps[j_st+dj][i_st] = None
                gc.collect()  # force a garbage collection

                print('  --> intermediate output -->\n')
                self.build_output_file(is_final=False)

    @staticmethod
    def compress_map(map_: np.array, coef: int, dtype: type, header: fits.Header = None,
                     EXTNAME: str = None, UNIT: (str, str) = None) -> fits.ImageHDU:
        '''
        Compress float32 map to (u)int16 to save storage.

        Parameters
        ----------
        map_ : np.array, shape : usually (NsideP, NsideP)
            Map to be compressed.
        coef : int
            Coefficient for log10 values.
        dtype : type
            Data type of the compressed map, np.(u)int16.
        header : fits.Header
            Template header of the HDU.
            If None, return the compressed map instead of an HDU.
        EXTNAME : str
            EXTNAME keyword of the header.
            If None, return the compressed map instead of an HDU.
        UNIT : (str, str)
            UNIT keyword of the header.
            If None, return the compressed map instead of an HDU.

        Returns
        -------
        fits.ImageHDU
            HDU containing the compressed array.

        '''

        if dtype == np.uint16:
            a_min, a_max = 0, 65535
        elif dtype == np.int16:
            a_min, a_max = -32768, 32767

        my_map = np.clip(np.floor(coef*np.log10(np.clip(
            map_, 1e-32, None)) + 0.5), a_min, a_max).astype(dtype)
        if header is None or EXTNAME is None or UNIT is None:
            return my_map

        my_hdu = fits.ImageHDU(my_map, header=header); del my_map
        my_hdu.header['EXTNAME'] = EXTNAME
        my_hdu.header['UNIT'] = UNIT
        return my_hdu

    def build_output_file(self, is_final: bool = False) -> None:
        '''
        Build the output FITS file.

        The kappa maps have been unified and merged into the main FITS file.

        Parameters
        ----------
        is_final : bool, optional
            Whether this is the final (i.e., not intermediate) output.
            If so, recover the faded block boundaries. The default is False.

        Returns
        -------
        None.

        '''

        # shortcuts
        fk = self.cfg.fade_kernel
        NsidePf = self.cfg.NsideP + fk * 2
        outmaps = self.cfg.outmaps

        if is_final:  # recover block boundaries
            OutStamp.trapezoid(self.out_map, fk, recover_mode=True)
            width = self.cfg.postage_pad * self.cfg.n2  # width of padding region
            pad_widths = (width * ('B' not in self.pad_sides), width * ('T' not in self.pad_sides),
                          width * ('L' not in self.pad_sides), width * ('R' not in self.pad_sides))
            if 'U' in outmaps: OutStamp.trapezoid(self.UC_map,    fk, True, pad_widths)
            if 'S' in outmaps: OutStamp.trapezoid(self.Sigma_map, fk, True, pad_widths)
            if 'K' in outmaps: OutStamp.trapezoid(self.kappa_map, fk, True, pad_widths)
            if 'T' in outmaps: OutStamp.trapezoid(self.Tsum_map,  fk, True, pad_widths)
            if 'N' in outmaps: OutStamp.trapezoid(self.Neff_map,  fk, True, pad_widths)

        my_header = self.outwcs.to_header()
        maphdu = fits.PrimaryHDU(self.out_map[:, :, fk:NsidePf-fk, fk:NsidePf-fk], header=my_header)
        config_hdu = fits.TableHDU.from_columns(
            [fits.Column(name='text', array=self.cfg.to_file(None).splitlines(), format='A512', ascii=True)])
        config_hdu.header['EXTNAME'] = 'CONFIG'
        if is_final:
            for package in ['numpy', 'scipy', 'astropy', 'fitsio']:
                keyword = 'V' + package.upper()[:7]
                try:
                    with importlib.import_module(package) as m:
                        config_hdu.header[keyword] = (str(m.__version), f'Current version of {package}')
                except:
                    config_hdu.header[keyword] = ('N/A', f'{package} had no version number')
        inlist_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name='obsid', array=np.array([obs[0] for obs in self.obslist]), format='J'),
            fits.Column(name='sca',   array=np.array([obs[1] for obs in self.obslist]), format='I'),
            fits.Column(name='ra',    array=np.array([self.obsdata['ra' ][obs[0]] for obs in self.obslist]), format='D', unit='degree'),
            fits.Column(name='dec',   array=np.array([self.obsdata['dec'][obs[0]] for obs in self.obslist]), format='D', unit='degree'),
            fits.Column(name='pa',    array=np.array([self.obsdata['pa' ][obs[0]] for obs in self.obslist]), format='D', unit='degree'),
            fits.Column(name='valid', array=np.array([inimage.exists_ for inimage in self.inimages]), format='L')
        ])
        inlist_hdu.header['EXTNAME'] = 'INDATA'
        T_hdu = fits.ImageHDU(self.T_weightmap)
        T_hdu.header['EXTNAME'] = 'INWEIGHT'
        T_hdu2 = fits.ImageHDU(np.transpose(self.T_weightmap, axes=(0, 2, 1, 3)).\
                               reshape((self.outpsfgrp.n_psf*self.cfg.n1P, self.n_inimage*self.cfg.n1P)))
        T_hdu2.header['EXTNAME'] = 'INWTFLAT'

        hdulist = [maphdu, config_hdu, inlist_hdu, T_hdu, T_hdu2]

        if 'U' in outmaps:
            hdulist.append(Block.compress_map(
                self.UC_map[:, fk:NsidePf-fk, fk:NsidePf-fk], -5000, np.uint16,
                my_header, 'FIDELITY', ('0.2mB', '-5000*log10(U/C)')))

        if 'S' in outmaps:
            hdulist.append(Block.compress_map(
                self.Sigma_map[:, fk:NsidePf-fk, fk:NsidePf-fk], -10000, np.int16,
                my_header, 'SIGMA', ('0.1mB', '-10000*log10(Sigma)')))

        if 'K' in outmaps:
            hdulist.append(Block.compress_map(
                self.kappa_map[:, fk:NsidePf-fk, fk:NsidePf-fk], -5000, np.uint16,
                my_header, 'KAPPA', ('0.2mB', '-5000*log10(kappa)')))

        if 'T' in outmaps:
            hdulist.append(Block.compress_map(
                self.Tsum_map[:, fk:NsidePf-fk, fk:NsidePf-fk], 200000, np.int16,
                my_header, 'INWTSUM', ('5uB', '200000*log10(Tsum)')))

        if 'N' in outmaps:
            hdulist.append(Block.compress_map(
                self.Neff_map[:, fk:NsidePf-fk, fk:NsidePf-fk], 50000, np.uint16,
                my_header, 'EFFCOVER', ('20uB', '50000*log10(Neff)')))

        hdu_list = fits.HDUList(hdulist)
        hdu_list.writeto(self.outstem+'.fits', overwrite=True)

    def clear_all(self) -> None:
        '''
        Free up all memory spaces.

        Returns
        -------
        None.

        '''

        if self.cfg.tempfile is not None: self.cache_dir.rmdir()

        del self.obsdata, self.obslist, self.outwcs
        del self.outpsfgrp, self.outpsfovl
        self.sysmata.clear(); del self.sysmata
        self.sysmatb.clear(); del self.sysmatb
        del self.out_map, self.T_weightmap

        outmaps = self.cfg.outmaps  # shortcut
        if 'U' in outmaps: del self.UC_map
        if 'S' in outmaps: del self.Sigma_map
        if 'K' in outmaps: del self.kappa_map
        if 'T' in outmaps: del self.Tsum_map
        if 'N' in outmaps: del self.Neff_map

        for j_st in range(self.cfg.n1P+2):
            for i_st in range(self.cfg.n1P+2):
                inst = self.instamps[j_st][i_st]
                if inst is not None:
                    inst.clear()
                    self.instamps[j_st][i_st] = None
