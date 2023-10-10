from os.path import exists
import shutil
from itertools import combinations, product
import gc

from astropy import units as u
import numpy as np

from astropy.io import fits
from astropy import wcs
import fitsio
import matplotlib.pyplot as plt

from .config import Timer, Settings as Stn, Config
from .psfutil import PSFGrp, PSFOvl, SysMatA, SysMatB
from .lakernel import LAKernel


class InImage:
    '''
    Input image attached to a Block instance.

    Methods
    -------
    __init__ : Constructor.
    generate_idx_grid (staticmethod) : Generate a grid of indices.
    _inpix2world2outpix : Composition of pix2world and world2pix.
    partition_pixels : Partition input pixels into postage stamps.
    extract_layers : Make and extract input layers.
    clear : Free up memory space.

    smooth_and_pad (staticmethod): Utility to smear a PSF with a tophat and a Gaussian.
    get_psf_and_distort_mat: Get input PSF array and the corresponding distortion matrix.

    '''

    def __init__(self, blk: 'Block', idsca: (int, int)) -> None:
        '''
        Constructor.

        Parameters
        ----------
        blk : Block
            The Block instance to which this InImage instance is attached to.
        idsca : (int, int)
            ID of observation and SCA used.

        Returns
        -------
        None.

        '''

        self.blk = blk
        self.idsca = idsca

        self.infile = blk.cfg.inpath+'/simple/dc2_{:s}_{:d}_{:d}.fits'.format(
            Stn.RomanFilters[blk.obsdata['filter'][idsca[0]]], idsca[0], idsca[1])
        self.exists = exists(self.infile) and exists(blk.cfg.inpsf_path+'/dc2_psf_{:d}.fits'.format(idsca[0]))
        if self.exists:
            with fits.open(self.infile) as f:
                self.inwcs = wcs.WCS(f[Stn.hdu_with_wcs].header)

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

    def partition_pixels(self, sp_res: int = 90, relax_coef: float = 1.05,
                         visualize: bool = False) -> None:
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

        print(f'> partitioning pixels from InImage {self.idsca}', '@', self.blk.timer(), 's')

        # create a sparse grid of pixels to locate regions of interest
        sp_arr = np.linspace(0, Stn.sca_nside, sp_res+1, dtype=np.uint16)
        sp_inxys = InImage.generate_idx_grid(sp_arr, sp_arr)
        sp_outxys = self._inpix2world2outpix(sp_inxys).T.reshape(2, sp_res+1, sp_res+1)
        del sp_inxys

        if visualize:
            for i in range(2):
                plt.imshow(sp_outxys[i], origin='lower')
                plt.colorbar()
                plt.contour(sp_outxys[i], levels=[50, 2550], colors='r')
                plt.show()

        # limits for stamp indices
        pix_lower = (                 self.blk.cfg.postage_pad-1) * self.blk.cfg.n2 - 0.5  #  1 x n2 by default
        pix_upper = (self.blk.cfg.n1P-self.blk.cfg.postage_pad+1) * self.blk.cfg.n2 - 0.5  # 51 x n2 by default

        self.is_relevant = False  # whether the input image is relevant to the output block
        relevant_matrix = np.zeros((sp_res, sp_res), dtype=bool)
        for j in range(1, sp_res):
            for i in range(1, sp_res):
                if pix_lower < sp_outxys[0, j, i] < pix_upper\
                and pix_lower < sp_outxys[1, j, i] < pix_upper:
                    # at least some of the input pixels are relevant
                    self.is_relevant = True
                    # we will study all the adjacent input pixels
                    relevant_matrix[max(j-2, 0):min(j+2, sp_res-1),
                                    max(i-2, 0):min(i+2, sp_res-1)] = True
        del sp_outxys

        if visualize:
            plt.imshow(relevant_matrix, origin='lower')
            plt.colorbar()
            plt.show()

        # if self.idsca not in [(95060, 12), (132154, 12)]:
        #     self.is_relevant = False  # for testing purposes
        if not self.is_relevant:
            del self.inwcs, sp_arr, relevant_matrix
            return

        # maximum number of input pixels per postage stamp (from this InImage)
        # relax_coef: the actual maximum may be larger due to distortions
        npixmax = int(((self.blk.cfg.n2 * self.blk.cfg.dtheta * u.degree.to('arcsec')) /\
                       (Stn.pixscale_native / Stn.arcsec) + 1) ** 2 * relax_coef)  # default: about 160

        # arrays for indices (in the input image grid),
        self.y_idx = np.zeros((self.blk.cfg.n1+2, self.blk.cfg.n1+2, npixmax), dtype=np.uint16)
        self.x_idx = np.zeros((self.blk.cfg.n1+2, self.blk.cfg.n1+2, npixmax), dtype=np.uint16)
        # positions (in the output block coordinates),
        self.y_val = np.zeros((self.blk.cfg.n1+2, self.blk.cfg.n1+2, npixmax), dtype=np.float32)
        self.x_val = np.zeros((self.blk.cfg.n1+2, self.blk.cfg.n1+2, npixmax), dtype=np.float32)
        # and number of pixels in each postage stamp (from this InImage)
        self.pix_count = np.zeros((self.blk.cfg.n1+2, self.blk.cfg.n1+2), dtype=np.uint16)

        # load masks here
        mask = np.ones((Stn.sca_nside, Stn.sca_nside), dtype=bool)

        # now loop over the regions set by the sparce grid
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
                        if not (pix_lower < my_x < pix_upper and pix_lower < my_y < pix_upper): continue
                        if not mask[j+bottom, i+left]: continue

                        i_st = int((my_x-pix_lower) // self.blk.cfg.n2)  # st stands for stamp
                        j_st = int((my_y-pix_lower) // self.blk.cfg.n2)

                        my_idx = self.pix_count[j_st, i_st]
                        self.y_idx[j_st, i_st, my_idx] = j+bottom
                        self.x_idx[j_st, i_st, my_idx] = i+left
                        self.y_val[j_st, i_st, my_idx] = my_y
                        self.x_val[j_st, i_st, my_idx] = my_x
                        self.pix_count[j_st, i_st] += 1

                del inxys, outxys

        if visualize:
            plt.imshow(self.pix_count, origin='lower')
            plt.colorbar()
            plt.show()

        print('-->', np.sum(self.pix_count), 'pixels selected from idsca', self.idsca, end='; ')
        self.max_count = np.max(self.pix_count)
        print('the most populous stamp has', self.max_count, 'pixels')
        del sp_arr, relevant_matrix, mask

    def extract_layers(self) -> None:
        '''
        Make and extract input layers.

        Returns
        -------
        None.

        '''

        assert self.exists, 'Error: input image and/or input psf do(es) not exist'

        self.data = np.zeros((self.blk.cfg.n_inframe, self.blk.cfg.n1+2,
                              self.blk.cfg.n1+2, self.max_count), dtype=np.float32)

        with fits.open(self.infile) as f:
            indata = f['SCI'].data - float(f['SCI'].header['SKY_MEAN'])

        for j_st in range(self.blk.cfg.n1+2):
            for i_st in range(self.blk.cfg.n1+2):
                n_pix = self.pix_count[j_st, i_st]
                self.data[0, j_st, i_st, :n_pix] =\
                indata[self.y_idx[j_st, i_st, :n_pix], self.x_idx[j_st, i_st, :n_pix]]

        # advanced layers to be made and extracted here

        del indata, self.y_idx, self.x_idx
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
            For both: In units of the pixels given (not native pixel). The default is 0.0.

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

    def get_psf_and_distort_mat(self, j_st: int, i_st: int) -> tuple:
        '''
        Get input PSF array and the corresponding distortion matrix.

        This is an interface for psfutil.PSFGrp._build_inpsfgrp.

        Parameters
        ----------
        j_st, i_st : int, int
            InStamp index.

        Returns
        -------
        tuple : (this_psf, distort_matrice)
            Smeared input PSF array : np.array, see smooth_and_pad
            and distortion matrix : np.array, shape : (2, 2)

        '''

        fname = self.blk.cfg.inpsf_path+'/dc2_psf_{:d}.fits'.format(self.idsca[0])
        assert exists(fname), 'Error: input psf does not exist'

        fileh = fitsio.FITS(fname)
        this_psf = InImage.smooth_and_pad(fileh[self.idsca[1]][:, :], tophatwidth=self.blk.cfg.inpsf_oversamp)
        fileh.close()

        # compute the psf at the center of the 2x2 group of postage stamps
        psf_compute_point_pix = [(self.blk.cfg.postage_pad+i_st) * self.blk.cfg.n2,
                                 (self.blk.cfg.postage_pad+j_st) * self.blk.cfg.n2]
        psf_compute_point = self.blk.outwcs.all_pix2world(np.array([psf_compute_point_pix]), 0)[0]
        dWdp_out = wcs.utils.local_partial_pixel_derivatives(self.blk.outwcs, *psf_compute_point_pix)

        pixloc = self.inwcs.all_world2pix(np.array([psf_compute_point.tolist()]), 0)[0]
        distort_matrice = np.linalg.inv(dWdp_out) \
            @ wcs.utils.local_partial_pixel_derivatives(self.inwcs, pixloc[0], pixloc[1]) \
            * self.blk.cfg.dtheta*Stn.degree/Stn.pixscale_native

        return this_psf, distort_matrice


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
            The Block instance to which this InStamp instance is attached to.
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
                                   for inimage in blk.inimages], dtype=np.uint16)
        self.pix_cumsum = np.cumsum([0] + list(self.pix_count), dtype=np.uint16)

        # input pixel positions and signals
        self.y_val = np.empty((self.pix_cumsum[-1],), dtype=np.float32)
        self.x_val = np.empty((self.pix_cumsum[-1],), dtype=np.float32)
        self.data = np.empty((blk.cfg.n_inframe, self.pix_cumsum[-1]), dtype=np.float32)

        for i_im, inimage in enumerate(blk.inimages):
            self.y_val[self.pix_cumsum[i_im]:self.pix_cumsum[i_im+1]] =\
                inimage.y_val[j_st, i_st, :self.pix_count[i_im]]
            self.x_val[self.pix_cumsum[i_im]:self.pix_cumsum[i_im+1]] =\
                inimage.x_val[j_st, i_st, :self.pix_count[i_im]]
            self.data[:, self.pix_cumsum[i_im]:self.pix_cumsum[i_im+1]] =\
                inimage.data[:, j_st, i_st, :self.pix_count[i_im]]

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
            if None in both direction, select all input pixels.
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

        selection = np.array(np.where(dist_sq < radius**2)[0], dtype=np.uint16)
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

        if sim_mode:  # count references, no actual inpsfgrp invovled
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
    Postage stamp coaddition.

    Methods
    -------
    __init__ : Constructor.
    _process_input_stamps : Fetch and process input postage stamps.

    __call__ : Build system matrices and perform coaddition.
    _build_system_matrices : Build system matrices and coaddition matrices.
    _visualize_system_matrices : Visualize system matrices.
    _visualize_coadd_matrices : Visualize coaddition matrices.
    _perform_coaddition : Perform the actual multiplication.
    _show_in_and_out_images : Display input and output images.
    _study_individual_pixels : Study individual input and output pixels.
    clear : Free up memory space.

    '''

    uctarget = [1.0e-6]
    targetleak = np.array(uctarget)
    imcom_smax = 0.5

    def __init__(self, blk: 'Block', j_st: int, i_st: int) -> None:
        '''
        Constructor.

        Parameters
        ----------
        blk : Block
            The Block instance to which this InStamp instance is attached to.
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

        # count references to PSF overlaps and system matrices
        for ji_st_in in self.ji_st_in_s:
            blk.sysmata.get_iisubmat(ji_st_in, ji_st_in, sim_mode=True)
            blk.sysmatb.get_iosubmat(ji_st_in, (j_st, i_st), sim_mode=True)

        for ji_st_pair in combinations(self.ji_st_in_s, 2):
            blk.sysmata.get_iisubmat(*ji_st_pair, sim_mode=True)

        # limit y and x positions of this output postage stamp, all integers
        # not including the transition pixels (of which the number is set by fade_kernel)
        self.bottom = (blk.cfg.postage_pad-1+j_st) * blk.cfg.n2
        self.top    = self.bottom + blk.cfg.n2-1
        self.left   = (blk.cfg.postage_pad-1+i_st) * blk.cfg.n2
        self.right  = self.left + blk.cfg.n2-1

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
        self.inpix_count = np.zeros((9,), dtype=np.uint16)

        # acceptance radius in units of output pixels
        radius = (self.blk.cfg.instamp_pad / Stn.arcsec) \
        / (self.blk.cfg.dtheta * u.degree.to('arcsec')) \
        + self.blk.cfg.fade_kernel * np.sqrt(2.0)

        # now select input pixels
        for idx, ji_st_in in enumerate(self.ji_st_in_s):
            self.instamps[idx] = self.blk.instamps[ji_st_in[0]][ji_st_in[1]]

            x_pivot = [self.left-0.5, None, self.right+0.5][ji_st_in[1]-self.i_st+1]
            y_pivot = [self.bottom-0.5, None, self.top+0.5][ji_st_in[0]-self.j_st+1]
            self.selections[idx] = self.instamps[idx].make_selection((x_pivot, y_pivot), radius)
            if self.selections[idx] is None:
                self.inpix_count[idx] = self.instamps[idx].pix_cumsum[-1]
            else:
                self.inpix_count[idx] = self.selections[idx].shape[0]

        self.inpix_cumsum = np.cumsum([0] + list(self.inpix_count), dtype=np.uint16)

        if visualize:
            for instamp, selection in zip(self.instamps, self):
                if selection is None: plt.scatter(instamp.x_val, instamp.y_val, s=1)
                plt.scatter(instamp.x_val[selection], instamp.y_val[selection], s=1)
            plt.axis('equal')
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

        print(f'> coadding OutStamp {(self.j_st, self.i_st)}', '@', self.blk.timer(), flush=True)
        self._build_system_matrices(visualize, save_abc)
        self._perform_coaddition(visualize, save_t)
        print()

    def _build_system_matrices(self, visualize: bool = False, save_abc: bool = False) -> None:
        '''
        Build system matrices and coaddition matrices.

        This method probably needs to be split.

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

        # the A-matrix first
        self.sysmata = np.zeros((self.inpix_cumsum[-1], self.inpix_cumsum[-1]))  # dtype=np.float64

        for idx, ji_st_in, selection in zip(range(9), self.ji_st_in_s, self.selections):
            iisubmat = self.blk.sysmata.get_iisubmat(ji_st_in, ji_st_in)
            if selection is not None: iisubmat = iisubmat[np.ix_(selection, selection)]

            self.sysmata[self.inpix_cumsum[idx]:self.inpix_cumsum[idx+1],
                         self.inpix_cumsum[idx]:self.inpix_cumsum[idx+1]] = iisubmat

        for idx_s, ji_st_pair, selections_ in zip(combinations(range(9), 2),
                                                  combinations(self.ji_st_in_s, 2),
                                                  combinations(self.selections, 2)):
            iisubmat = self.blk.sysmata.get_iisubmat(*ji_st_pair)
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

        print(f'--> getting coadd matrix for OutStamp {(self.j_st, self.i_st)}', '@', self.blk.timer(), 's', flush=True)
        # kappa_, Sigma_, UC_: (n_out, n_outpix); self.T: (n_out, n_outpix, n_inpix)
        kappa_, Sigma_, UC_, self.T = LAKernel.get_coadd_matrix_discrete(
            self.sysmata, self.mhalfb, self.outovlc,
            self.blk.cfg.kappa_arr, np.array(OutStamp.targetleak), smax=OutStamp.imcom_smax)
        if not save_abc: del self.sysmata, self.mhalfb, self.outovlc

        # post processing
        shape = (self.blk.outpsfgrp.n_psf, self.blk.cfg.n2f, self.blk.cfg.n2f)
        self.kappa = kappa_.reshape(shape); del kappa_
        self.Sigma = Sigma_.reshape(shape); del Sigma_
        self.UC = UC_.reshape(shape); del UC_

        if visualize:
            print()
            self._visualize_coadd_matrices()

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

        plt.figure(figsize=(12.8, 9.6))
        plt.imshow(np.log10(self.sysmata))
        plt.colorbar()

        for xy in self.inpix_cumsum[1:-1]:
            plt.axvline(xy, c='r', ls='--', lw=0.5)
            plt.axhline(xy, c='r', ls='--', lw=0.5)
        plt.show()

        # now the mBhalf matrix
        print(f'{self.mhalfb.shape=}')  # (n_out, n_outpix, n_inpix)

        for mhalfb_ in self.mhalfb:
            plt.figure(figsize=(12.8, 4.8))
            plt.imshow(np.log10(mhalfb_))
            plt.colorbar()

            for x in self.inpix_cumsum[1:-1]:
                plt.axvline(x, c='r', ls='--', lw=1.0)
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

        for j_out, T_ in enumerate(self.T):
            # print(f'{j_out=}')

            fig, axs = plt.subplots(1, 3, figsize=(12.8, 4.8))
            for i, map_ in enumerate([self.kappa, self.Sigma, self.UC]):
                im = axs[i].imshow(np.log10(map_[j_out]), origin='lower')
                plt.colorbar(im)
            plt.show()

            vmin, vmax = np.percentile(T_.ravel(), [1, 99])
            plt.hist(T_.ravel(), bins=np.linspace(vmin, vmax, 31))
            plt.show()

            plt.figure(figsize=(12.8, 4.8))
            plt.imshow(T_, vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.show()

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

        # coadd_utils.trapezoid
        fk2 = 2*self.blk.cfg.fade_kernel  # shortcut
        if fk2 > 0:
            s = np.arange(1, fk2+1, dtype=float) / (fk2+1)
            if use_trunc_sinc:
                s -= np.sin(2*np.pi*s)/(2*np.pi)

            T_view = np.moveaxis(self.T, 1, -1).reshape(
                (self.blk.outpsfgrp.n_psf, self.inpix_cumsum[-1],
                 self.blk.cfg.n2f, self.blk.cfg.n2f))

            T_view[..., : fk2,      :] *= s[None, :].T
            T_view[..., :-fk2-1:-1, :] *= s[None, :].T
            T_view[..., :, : fk2     ] *= s
            T_view[..., :, :-fk2-1:-1] *= s

        self.outimage = np.einsum('oaj,ij->oia', self.T, self.indata).\
        reshape((self.blk.outpsfgrp.n_psf, self.blk.cfg.n_inframe, self.blk.cfg.n2f, self.blk.cfg.n2f))

        if visualize:
            print()
            self._show_in_and_out_images()
            print()
            self._study_individual_pixels()
        del self.iny_val, self.inx_val, self.indata

        Tsum_partial = np.sum(self.T, axis=1)
        if not save_t: del self.T
        self.Tsum = np.zeros((self.blk.outpsfgrp.n_psf, self.blk.n_inimage))

        for j_st, inst, selection in zip(range(9), self.instamps, self.selections):
            if selection is None:
                my_cumsum = inst.pix_cumsum.copy()
            else:
                my_cumsum = np.searchsorted(selection, inst.pix_cumsum)
            my_cumsum += self.inpix_cumsum[j_st]
            # print(j_st, my_cumsum)

            for i_im in range(self.blk.n_inimage):
                # print(j_st, i_im, np.sum(Tsum_partial[:, my_cumsum[i_im]:my_cumsum[i_im+1]], axis=1))
                self.Tsum[:, i_im] += np.sum(Tsum_partial[:, my_cumsum[i_im]:my_cumsum[i_im+1]], axis=1)

        del Tsum_partial
        self.Tsum /= self.Tsum.sum(axis=1)[:, None]

    def _show_in_and_out_images(self) -> None:
        '''
        Display input and output images.

        Returns
        -------
        None.

        '''

        print('OutStamp._show_in_and_out_images')

        for j_in in range(self.blk.cfg.n_inframe):
            plt.scatter(self.inx_val, self.iny_val, c=self.indata[j_in], cmap='viridis', s=5)
            plt.colorbar()
            for x in [self.left, self.right]: plt.axvline(x, c='r', ls='--')
            for y in [self.bottom, self.top]: plt.axhline(y, c='r', ls='--')
            plt.show()

            # to be upgraded for the n_out > 1 case
            plt.imshow(self.outimage[0, j_in], origin='lower')
            plt.colorbar()
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

        fk2 = self.blk.cfg.fade_kernel * 2
        n2f = self.blk.cfg.n2f  # shortcuts
        for j_out, i_out in [(fk2, fk2), (fk2, n2f-1-fk2), (n2f-1-fk2, fk2),
                             (n2f-1-fk2, n2f-1-fk2), ((n2f-1)//2, (n2f-1)//2)]:
            out_idx = j_out * self.blk.cfg.n2f + i_out
            T_elems = self.T[0, out_idx, :]

            plt.figure(figsize=(12.8, 4.8))
            ax0 = plt.subplot(1, 2, 1)
            im = ax0.scatter(self.inx_val, self.iny_val, c=T_elems, cmap='viridis', s=5)
            plt.colorbar(im)

            dist = np.sqrt(np.square(j_out+(self.bottom-fk2//2) - self.iny_val) +\
                           np.square(i_out+(self.left-fk2//2)   - self.inx_val))
            closest_inpix.append(np.argmin(dist))
            ax1 = plt.subplot(2, 2, 2)
            ax1.scatter(dist, T_elems, c=T_elems, cmap='viridis', s=5)
            ax1.axhline(              0, c='b', ls='--')
            ax1.axvline(self.blk.cfg.n2, c='r', ls='--')

            signal = np.empty_like(accrad, dtype=float)
            for i in range(accrad.shape[0]):
                T_arr = np.where(dist <= accrad[i], T_elems, 0.0)
                signal[i] = T_arr @ self.indata[0]
            ax2 = plt.subplot(2, 2, 4)
            ax2.plot(accrad, signal, 'o-')
            ax2.axvline(self.blk.cfg.n2, c='r', ls='--')
            ax2.axhline(self.outimage[0, 0].ravel()[out_idx], c='b', ls='--')
            ax2.set_xlim(ax1.get_xlim())

            plt.show()

        for in_idx in closest_inpix:
            print(f'{(self.inx_val[in_idx], self.iny_val[in_idx])=}')
            plt.imshow(self.T[0, :, in_idx].reshape(56, 56), origin='lower')
            plt.colorbar()
            plt.scatter(self.inx_val[in_idx]-(self.left-fk2//2),
                        self.iny_val[in_idx]-(self.bottom-fk2//2), c='r', s=2)
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
        del self.kappa, self.Sigma, self.UC, self.Tsum
        del self.yx_val, self.outimage


class Block:
    '''
    Output block coaddition.

    Methods
    -------
    __init__ : Constructor.
    __call__ : Coadd this block.
    parse_config : Parse the configuration.
    _get_obs_cover: Get observations relevant to this Block.
    process_input_images : Process input images.
    build_input_stamps : Build input stamps from input images.
    _output_stamp_wrapper : Wrapper for output stamp coaddition.
    coadd_output_stamps : Coadd output stamps using input stamps.
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
        self.cfg = cfg
        if cfg is None: self.cfg = Config()  # use the default config

        PSFGrp.setup(npixpsf=cfg.inpsf_npix, oversamp=cfg.inpsf_oversamp)
        self.this_sub = this_sub
        if run_coadd: self()

    def __call__(self) -> None:
        '''
        Coadd this block.

        Returns
        -------
        None.

        '''

        print('> Block.parse_config', '@', self.timer(), 's')
        self.parse_config()
        # this produces: obsdata, outwcs, outpsfgrp, outpsfovl

        print('> Block.process_input_images', '@', self.timer(), 's')
        self.process_input_images()
        # this produces: obslist, inimages (1-d list), n_inimage

        print('> Block.build_input_stamps', '@', self.timer(), 's')
        self.build_input_stamps()
        # this produces: instamps (2-d list)

        print('> Block.coadd_output_stamps(sim_mode=True)', '@', self.timer(), 's')
        self.coadd_output_stamps(sim_mode=True)
        # this produces: sysmata (object), sysmatb (object), outstamps (2-d list)
        print('> Block.coadd_output_stamps(sim_mode=False)', '@', self.timer(), 's', end='\n\n')
        self.coadd_output_stamps(sim_mode=False)
        # this produces: out_map, fidelity_map, kappamin, kappamax, T_weightmap

        print('> Block.build_output_file', '@', self.timer(), 's')
        self.build_output_file()
        print('> Block.clear_all', '@', self.timer(), 's')
        self.clear_all()

    def parse_config(self) -> None:
        '''
        Parse the configuration.

        Returns
        -------
        None.

        '''

        # Get observation table
        assert self.cfg.obsfile is not None, 'Error: no obsfile found'
        print('Getting observations from {:s}'.format(self.cfg.obsfile))
        with fits.open(self.cfg.obsfile) as myf:
            self.obsdata = myf[1].data
            obscols = myf[1].columns
            n_obs_tot = len(self.obsdata.field(0))
            print('Retrieved columns:', obscols.names, ' {:d} rows'.format(n_obs_tot))

        # display output information
        print('Output information: ctr at RA={:10.6f},DEC={:10.6f}'.format(self.cfg.ra, self.cfg.dec))
        print('pixel scale={:8.6f} arcsec or {:11.5E} degree'.format(
            self.cfg.dtheta * u.degree.to('arcsec'), self.cfg.dtheta))
        print('output array size = {:d} ({:d} blocks of {:d})'.format(
            self.cfg.Nside, self.cfg.n1, self.cfg.n2))
        print('')

        # block information
        # prime number to not do all the blocks next to each other first
        p = 1567
        if self.cfg.nblock % p == 0: p = 281
        j = (self.this_sub*p) % (self.cfg.nblock**2)
        ibx, iby = divmod(j, self.cfg.nblock)
        print('sub-block {:4d} <{:2d},{:2d}> of {:2d}x{:2d}={:2d}'.format(self.this_sub,
              ibx, iby, self.cfg.nblock, self.cfg.nblock, self.cfg.nblock**2))
        self.cfg.outstem += '_{:02d}_{:02d}'.format(ibx, iby)
        print('outputs directed to -->', self.cfg.outstem)

        # make the WCS
        self.outwcs = wcs.WCS(naxis=2)
        self.outwcs.wcs.crpix = [(self.cfg.NsideP+1)/2. - self.cfg.Nside*(ibx-(self.cfg.nblock-1)/2.),
                                 (self.cfg.NsideP+1)/2. - self.cfg.Nside*(iby-(self.cfg.nblock-1)/2.)]
        self.outwcs.wcs.cdelt = [-self.cfg.dtheta, self.cfg.dtheta]
        self.outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        self.outwcs.wcs.crval = [self.cfg.ra, self.cfg.dec]

        # print the corners of the square and the center, ordering:
        #   2   3
        #     4
        #   0   1
        cornerx = [-.5, self.cfg.NsideP-.5, -.5, self.cfg.NsideP-.5, (self.cfg.NsideP-1)/2.]
        cornery = [-.5, -.5, self.cfg.NsideP-.5, self.cfg.NsideP-.5, (self.cfg.NsideP-1)/2.]
        for i in range(5):
            print(i, self.outwcs.all_pix2world(np.array([[cornerx[i], cornery[i]]]), 0))
        self.cfg.centerpos = self.outwcs.all_pix2world(np.array(
            [[cornerx[-1], cornery[-1]]]), 0)[0]  # [ra,dec] array in degrees

        # and the output PSFs and target leakages
        # (this will have to be modified for multiple outputs)
        self.outpsfgrp = PSFGrp(in_or_out=False, blk=self)
        self.outpsfovl = PSFOvl(self.outpsfgrp, None)

    def _get_obs_cover(self, radius: float) -> None:
        '''
        Get observations relevant to this Block.

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
        x1 = np.cos(self.cfg.centerpos[1]*Stn.degree) * \
            np.cos((self.cfg.centerpos[0]-self.obsdata['ra'])*Stn.degree)
        y1 = np.cos(self.cfg.centerpos[1]*Stn.degree) * \
            np.sin((self.cfg.centerpos[0]-self.obsdata['ra'])*Stn.degree)
        z1 = np.sin(self.cfg.centerpos[1]*Stn.degree)*np.ones((n_obs_tot,))
        # then rotate the Dec direction
        x2 = np.sin(self.obsdata['dec']*Stn.degree)*x1 - \
            np.cos(self.obsdata['dec']*Stn.degree)*z1
        y2 = y1
        z2 = np.cos(self.obsdata['dec']*Stn.degree)*x1 + \
            np.sin(self.obsdata['dec']*Stn.degree)*z1
        # and finally the PA direction
        X = (-np.sin(self.obsdata['pa']*Stn.degree)*x2 - \
             np.cos(self.obsdata['pa']*Stn.degree)*y2) / Stn.degree
        Y = (-np.cos(self.obsdata['pa']*Stn.degree)*x2 + \
             np.sin(self.obsdata['pa']*Stn.degree)*y2) / Stn.degree
        #
        # throw away points in wrong hemisphere -- important since in orthographic projection, can have (X,Y)=0 for antipodal point
        X = np.where(z2 > 0, X, 1e49)

        for isca in range(18):
            obsgood = np.where(np.logical_and(np.sqrt(
                (X-Stn.SCAFov[isca][0])**2 + (Y-Stn.SCAFov[isca][1])**2) < radius, self.obsdata['filter'] == self.cfg.use_filter))
            for k in range(len(obsgood[0])):
                self.obslist.append((obsgood[0][k], isca+1))

        self.obslist.sort()

    def process_input_images(self) -> None:
        '''
        Process input images.

        Returns
        -------
        None.

        '''

        ### Now figure out which observations we need ###

        search_radius = Stn.sca_sidelength / np.sqrt(2.) / Stn.degree +\
                        self.cfg.NsideP * self.cfg.dtheta / np.sqrt(2.)
        self._get_obs_cover(search_radius)
        print(len(self.obslist), 'observations within range ({:7.5f} deg)'.format(search_radius),
              'filter =', self.cfg.use_filter, '({:s})'.format(Stn.RomanFilters[self.cfg.use_filter]))

        self.inimages = [InImage(self, idsca) for idsca in self.obslist]
        any_exists = False
        print('The observations -->')
        print('  OBSID SCA  RAWFI    DECWFI   PA     RASCA   DECSCA       FILE (x=missing)')
        for idsca, inimage in zip(self.obslist, self.inimages):
            cpos = '                 '
            if inimage.exists:
                any_exists = True
                cpos_coord = inimage.inwcs.all_pix2world([[Stn.sca_ctrpix, Stn.sca_ctrpix]], 0)[0]
                cpos = '{:8.4f} {:8.4f}'.format(cpos_coord[0], cpos_coord[1])
            print('{:7d} {:2d} {:8.4f} {:8.4f} {:6.2f} {:s} {:s} {:s}'.format(
                idsca[0], idsca[1], self.obsdata['ra'][idsca[0]], self.obsdata['dec'][idsca[0]],
                self.obsdata['pa'][idsca[0]], cpos, ' ' if inimage.exists else 'x', inimage.infile))
        print('')
        assert any_exists, 'No candidate observations found to stack. Exiting now.'

        for inimage in self.inimages:
            inimage.partition_pixels()
            if not inimage.is_relevant: continue
            inimage.extract_layers()
        print()

        # remove irrelevant input images
        self.inimages = [inimage for inimage in self.inimages if inimage.is_relevant]
        self.n_inimage = len(self.inimages)

    def build_input_stamps(self) -> None:
        '''
        Build input stamps from input images.

        Returns
        -------
        None.

        '''

        # current version only works when acceptance radius <= postage stamp size
        self.instamps = [[None for i_st in range(self.cfg.n1+2)]
                               for j_st in range(self.cfg.n1+2)]  # st stands for stamp

        for j_st in range(self.cfg.n1+2):
            for i_st in range(self.cfg.n1+2):
                self.instamps[j_st][i_st] = InStamp(self, j_st, i_st)

        for inimage in self.inimages:
            inimage.clear()

    def _output_stamp_wrapper(self, i_st, j_st, sim_mode: bool = False):
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

        assert 0 < i_st < self.cfg.n1+1 and 0 < j_st < self.cfg.n1+1, 'outstamp out of boundary'

        if sim_mode:  # count references to PSF overlaps and system matrices
            self.outstamps[j_st][i_st] = OutStamp(self, j_st, i_st)
        else:
            outst = self.outstamps[j_st][i_st]; outst()

            bottom = (j_st+self.cfg.postage_pad-1) * self.cfg.n2 - self.cfg.fade_kernel
            top    = (j_st+self.cfg.postage_pad)   * self.cfg.n2 + self.cfg.fade_kernel
            left   = (i_st+self.cfg.postage_pad-1) * self.cfg.n2 - self.cfg.fade_kernel
            right  = (i_st+self.cfg.postage_pad)   * self.cfg.n2 + self.cfg.fade_kernel

            self.out_map[:, :, bottom:top, left:right] += outst.outimage

            # weight computations
            self.T_weightmap[:, :, j_st+self.cfg.postage_pad-1, i_st+self.cfg.postage_pad-1] = outst.Tsum

            # fidelity map
            self.fidelity_map[:, bottom:top, left:right] = np.minimum(
                self.fidelity_map[:, bottom:top, left:right],
                np.floor(-10*np.log10(np.clip(outst.UC, 10**(-25.5999), .99999))).astype(np.uint8))

            # kappa map
            self.kappamin[bottom:top, left:right] = np.minimum(self.kappamin[bottom:top, left:right], outst.kappa)
            self.kappamax[bottom:top, left:right] = np.maximum(self.kappamax[bottom:top, left:right], outst.kappa)

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

        if sim_mode:  # count references to PSF overlaps and system matrices
            self.sysmata = SysMatA(self)
            self.sysmatb = SysMatB(self)
            self.outstamps = [[None for i_st in range(self.cfg.n1+2)]
                                    for j_st in range(self.cfg.n1+2)]
        else:
            n_out  = self.outpsfgrp.n_psf
            NsideP = self.cfg.NsideP

            # make basic output array (NOT including transition pixels on the boundaries)
            self.out_map = np.zeros((n_out, self.cfg.n_inframe, NsideP, NsideP), dtype=np.float32)
            # and the fidelity information (store as integer, will be in dB; initialize at full fidelity)
            self.fidelity_map = np.full((n_out, NsideP, NsideP), fill_value=255, dtype=np.uint8)
            # and kappa (2 layers, min and max; start min layer at a large value))
            self.kappamin = np.full ((NsideP, NsideP), fill_value=1e32, dtype=np.float32)
            self.kappamax = np.zeros((NsideP, NsideP), dtype=np.float32)

            # allocate ancillary arrays
            self.T_weightmap = np.zeros((self.outpsfgrp.n_psf, self.n_inimage,
                                         self.cfg.n1P, self.cfg.n1P), dtype=np.float32)

        remaining = self.cfg.n1 ** 2
        if self.cfg.stoptile is not None:
            remaining = self.cfg.stoptile

        for j_st, i_st in product(range(1, 49, 2), range(1, 49, 2)):
            for dj, di in product(range(2), range(2)):
                # if i_st+di > 4: continue  # for testing purposes
                self._output_stamp_wrapper(i_st+di, j_st+dj, sim_mode)
                remaining -= 1

                if remaining == 0:
                    if sim_mode:
                        # print(self.sysmata.iisubmats_ref)
                        self.sysmata.iisubmats.clear()
                        # print(self.sysmatb.iopsfovls_ref)
                        self.sysmatb.iopsfovls.clear()
                    else:
                        assert len(self.sysmata.iisubmats) == 0, 'self.sysmata.iisubmats is not empty'
                        assert len(self.sysmatb.iopsfovls) == 0, 'self.sysmatb.iopsfovls is not empty'
                    return

            if not sim_mode:
                gc.collect()  # force a garbage collection
                # Write an intermediate map
                print('  --> intermediate output -->\n')
                maphdu = fits.PrimaryHDU(self.out_map, header=self.outwcs.to_header())
                hdu_list = fits.HDUList([maphdu])
                hdu_list.writeto(self.cfg.outstem+'_map.fits', overwrite=True)

    def build_output_file(self) -> None:
        '''
        Build the output FITS file.

        Currently the kappa maps are in a separate file.
        Those will be merged into the main FITS file.

        Returns
        -------
        None.

        '''

        maphdu = fits.PrimaryHDU(self.out_map, header=self.outwcs.to_header())
        config_hdu = fits.TableHDU.from_columns(
            [fits.Column(name='text', array=self.cfg.to_file(None).splitlines(), format='A512', ascii=True)])
        config_hdu.header['EXTNAME'] = 'CONFIG'
        inlist_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name='obsid', array=np.array([obs[0] for obs in self.obslist]), format='J'),
            fits.Column(name='sca',   array=np.array([obs[1] for obs in self.obslist]), format='I'),
            fits.Column(name='ra',    array=np.array([self.obsdata['ra' ][obs[0]] for obs in self.obslist]), format='D', unit='degree'),
            fits.Column(name='dec',   array=np.array([self.obsdata['dec'][obs[0]] for obs in self.obslist]), format='D', unit='degree'),
            fits.Column(name='pa',    array=np.array([self.obsdata['pa' ][obs[0]] for obs in self.obslist]), format='D', unit='degree'),
            fits.Column(name='valid', array=np.array([inimage.exists for inimage in self.inimages]), format='L')
        ])
        inlist_hdu.header['EXTNAME'] = 'INDATA'
        T_hdu = fits.ImageHDU(self.T_weightmap)
        T_hdu.header['EXTNAME'] = 'INWEIGHT'
        T_hdu2 = fits.ImageHDU(np.transpose(self.T_weightmap, axes=(0, 2, 1, 3)).\
                               reshape((self.outpsfgrp.n_psf*self.cfg.n1P, self.n_inimage*self.cfg.n1P)))
        T_hdu2.header['EXTNAME'] = 'INWTFLAT'
        fidelity_hdu = fits.ImageHDU(self.fidelity_map, header=self.outwcs.to_header())
        fidelity_hdu.header['EXTNAME'] = 'FIDELITY'
        fidelity_hdu.header['UNIT'] = ('dB', '-10*log10(U/C)')
        hdu_list = fits.HDUList([maphdu, config_hdu, inlist_hdu, T_hdu, T_hdu2, fidelity_hdu])
        hdu_list.writeto(self.cfg.outstem+'_map.fits', overwrite=True)

        # ... and the kappa map
        fits.PrimaryHDU(np.stack((self.kappamin, self.kappamax))).writeto(self.cfg.outstem+'_kappa.fits', overwrite=True)

        # if self.cfg.fname is not None:
        #     shutil.copy(str(self.cfg.fname), self.cfg.outstem+'_config.json')

    def clear_all(self) -> None:
        '''
        Free up all memory spaces.

        Returns
        -------
        None.

        '''

        del self.obsdata, self.obslist, self.outwcs
        del self.outpsfgrp, self.outpsfovl, self.sysmata, self.sysmatb
        del self.out_map, self.fidelity_map, self.kappamin, self.kappamax, self.T_weightmap

        for j_st in range(self.cfg.n1+2):
            for i_st in range(self.cfg.n1+2):
                self.instamps[j_st][i_st].clear()

        for j_st in range(self.cfg.n1+2):
            for i_st in range(self.cfg.n1+2):
                outst = self.outstamps[j_st][i_st]
                if outst is None: continue
                outst.clear()
