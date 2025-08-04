"""
Program to remove correlated noise stripes from RST images.
"""

import os
import glob
import time
import csv
import cProfile
import pstats
import io
import numpy as np
from astropy.io import fits
from astropy import wcs
# from scipy.signal import convolve2d, residue
from memory_profiler import profile, memory_usage
from utils import compareutils
from config import Settings as Stn, Config
import re
import sys
import copy
import pyimcom_croutines
from concurrent.futures import ProcessPoolExecutor, as_completed
from filelock import Timeout, FileLock
from scipy.ndimage import binary_dilation


TIME = False
testing=True
use_cg_float=np.float64
use_output_float=np.float32

global outfile
global outpath

filters = ['Y106', 'J129', 'H158', 'F184', 'K213']
areas = [7006, 7111, 7340, 4840, 4654]  # cm^2
model_params = {'constant': 1, 'linear': 2}
CG_models = {'FR', 'PR', 'HS', 'DY'}

s_in = 0.11  # arcsec^2
t_exp = 154  # sec

# Import config file
CFG = Config(cfg_file='configs/imdestripe_configs/config_destripe_med.json')
filter_ = filters[CFG.use_filter]
A_eff = areas[CFG.use_filter]
obsfile = CFG.ds_obsfile #location and stem of input images. overwritten by temp input dir
outpath =  CFG.ds_outpath #path to put outputs, /fs/scratch/PCON0003/klaliotis/imdestripe/
tempdir = os.getenv('TMPDIR') + '/'
labnoise_prefix = CFG.ds_indata #path to lab noise frames,  /fs/scratch/PCON0003/cond0007/anl-run-in-prod/labnoise/slope_
use_model = CFG.ds_model
permanent_mask = CFG.permanent_mask
cg_model = CFG.cg_model
cg_maxiter = CFG.cg_maxiter
cg_tol = CFG.cg_tol
cost_model = CFG.cost_model
resid_model = CFG.resid_model


if use_model not in model_params.keys():
    raise ValueError(f"Model {use_model} not in model_params dictionary.")
if CFG.cost_prior != 0:
    cost_prior = CFG.cost_prior
if cg_model not in CG_models:
    raise ValueError(f"CG model {cg_model} not in CG_models dictionary.")
outfile = outpath + filter_ + CFG.ds_outstem # the file that the output prints etc are written to

CFG()
#CFG.to_file(outpath+'ds.cfg')

t0 = time.time()

def write_to_file(text, filename=outfile):
    """
    Function to write some text to an output file
    :param text: Str, what to print
    :param filename: Str, an alternative filename if not going into the outfile
    :return: nothing
    """

    if not os.path.exists(filename):
        with open(filename, "w+") as f:
            f.write(text + '\n')
    else:
        with open(filename, "a") as f:
            f.write(text + '\n')
    print(text)


import os
import uuid
import time
import random
from astropy.io import fits
from filelock import FileLock, Timeout

@profile
def save_fits(image, filename, dir=outpath, overwrite=True, s=False, header=None, retries=3):
    """
    Save a 2D image to a FITS file with locking, retries, and atomic rename.
    Parameters
    ----------
    image : np.ndarray, 2D array to write.
    filename : str, Output filename without extension.
    dir : str, Directory to save into.
    overwrite : bool, Whether to overwrite the final target file.
    s : bool,  Whether to print status messages.
    header : fits.Header or None, Optional FITS header.
    retries : int, Number of write retry attempts if write fails.
    """
    filepath = os.path.join(dir, filename + '.fits')
    lockpath = filepath + '.lock'
    lock = FileLock(lockpath)

    for attempt in range(retries):
        try:
            with lock.acquire(timeout=30):
                tmp_filepath = filepath + f".{uuid.uuid4().hex}.tmp"
                if header is not None:
                    hdu = fits.PrimaryHDU(image, header=header)
                else:
                    hdu = fits.PrimaryHDU(image)

                hdu.writeto(tmp_filepath, overwrite=True)
                os.replace(tmp_filepath, filepath)  # Atomic move to final path

                if s:
                    write_to_file(f"Array {filename} written out to {filepath}")
                return  # Success

        except Timeout:
            write_to_file(f"Failed to write {filename}; lock acquire timeout")
            return

        except OSError as e:
            if attempt < retries - 1:
                wait_time = 1 + random.random()
                print(f"Write failed for {filepath} (attempt {attempt + 1}): {e}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to write {filepath} after {retries} attempts. Last error: {e}")



# C.H. wanted to define this before any use of sca_img so moved it up.
@profile
def apply_object_mask(image, mask=None, threshold_factor=2.5, inplace=False):
    """
    Apply a bright object mask to an image.

    :param image: 2D numpy array, the image to be masked.
    :param mask: optional 2D boolean array, the pre-existing object mask.
    :param threshold_factor: float, threshold for masking a pixel
    :param: factor to multiply with the median for thresholding.
    :param inplace: whether to modify the input image directly.
    :return image_out: the masked image.
    :return neighbor_mask: the mask applied.
    """
    if mask is not None and isinstance(mask, np.ndarray):
        neighbor_mask = mask
    else:
        median_val = np.median(image)
        high_value_mask = image >= threshold_factor * median_val
        neighbor_mask = binary_dilation(high_value_mask, structure=np.ones((5, 5), dtype=bool))

    if inplace:
        image[neighbor_mask] = 0
        return image, neighbor_mask
    else:
        image_out = np.where(neighbor_mask, 0, image)
        return image_out, neighbor_mask


def quadratic(x):
    return x ** 2


def absolute(x):
    return np.abs(x)


def huber_loss(x, d):
    return np.where(np.abs(x) <= d, quadratic(x), d**2+2*d*(np.abs(x)-d))


# Derivatives
def quad_prime(x):
    return 2 * x


def abs_prime(x):
    return np.sign(x)


def huber_prime(x, d):
    return np.where(np.abs(x) <= d, quad_prime(x), 2*d*np.sign(x))

class Cost_models:
    """
    Class holding the cost function models. This is a dictionary of functions
    """

    def __init__(self, model):

        models = {"quadratic": (quadratic, quad_prime), "absolute": (absolute, abs_prime),
                  "huber_loss": (huber_loss, huber_prime)}

        self.model = model

        if model=='huber_loss':
            self.thresh = CFG.hub_thresh
            write_to_file(f"Cost model is Huber Loss with threshold: {self.thresh}")
        else:
            self.thresh=None

        self.f, self.f_prime = models[model]


class Sca_img:
    """
    Class defining an SCA image object.
    Arguments:
        scaid: Str, the SCA id
        obsid: Str, the observation id
        interpolated: Bool; True if you want the interpolated version of this SCA and not the original. Default:False
        add_noise: Bool; True if you want to add read noise to the SCA image
        add_objmask: Bool; True if you want to apply the permanent pixel mask and a bright object mask
    Attributes:
        image: 2D np array, the SCA image (4088x4088)
        shape: Tuple, the shape of the image
        w: WCS object, the astropy.wcs object associated with this SCA
        obsid: Str, observation ID of this SCA image
        scaid: Str, SCA ID (position on focal plane) of this SCA image
        mask: 2D np array, the full pixel mask that is used on this image. Is correct only after calling apply_permanent_mask
        g_eff : 2D np array, effective gain in each pixel of the image
        params_subtracted: Bool, True if parameters have been subtracted from this image.
    Methods:
        apply_noise: apply the appropriate lab noise frame to the SCA image
        apply_permanent_mask: apply the SCA permanent pixel mask to the image
        apply_all_mask: apply the full SCA mask to the image
        subtract_parameters: Subtract a given set of parameters from self.image; updates self.image, self.params_subtracted
    """

    def __init__(self, obsid, scaid, interpolated=False, add_noise=True, add_objmask=True):

        if interpolated:
            file = fits.open(outpath + 'interpolations/' + obsid + '_' + scaid + '_interp.fits', memmap=True)
            image_hdu = 'PRIMARY'
        else:
            file = fits.open(obsfile + filter_ + '_' + obsid + '_' + scaid + '.fits', memmap=True)
            image_hdu = 'SCI'
        self.image = np.copy(file[image_hdu].data).astype(use_cg_float)

        self.shape = np.shape(self.image)
        self.w = wcs.WCS(file[image_hdu].header)
        self.header = file[image_hdu].header
        file.close()

        self.obsid = obsid
        self.scaid = scaid
        self.mask = np.ones(self.shape, dtype=bool)
        self.params_subtracted = False

        # Calculate effecive gain
        if not os.path.isfile(tempdir + obsid + '_' + scaid + '_geff.dat'):
            g0 = time.time()
            g_eff = np.memmap(tempdir + obsid + '_' + scaid + '_geff.dat', dtype='float64', mode='w+',
                              shape=self.shape)
            ra, dec = self.get_coordinates(pad=2.)
            ra = ra.reshape((4090, 4090))
            dec = dec.reshape((4090, 4090))
            derivs = np.array(((ra[1:-1, 2:] - ra[1:-1, :-2]) / 2, (ra[2:, 1:-1] - ra[:-2, 1:-1]) / 2,
                               (dec[1:-1, 2:] - dec[1:-1, :-2]) / 2, (dec[2:, 1:-1] - dec[:-2, 1:-1]) / 2))
            derivs_px = np.reshape(np.transpose(derivs), (4088 ** 2, 2, 2))
            det_mat = np.reshape(np.linalg.det(derivs_px), (4088, 4088))
            g_eff[:, :] = 1 / (np.abs(det_mat) * np.cos(np.deg2rad(dec[1:4089, 1:4089])) * t_exp * A_eff)
            g_eff.flush()
            write_to_file(f'G_eff calc duration: {time.time() - g0}')
            del g_eff

        self.g_eff = np.memmap(tempdir + obsid + '_' + scaid + '_geff.dat', dtype='float64', mode='r',
                               shape=self.shape)

        # Add a noise frame, if requested
        if add_noise: self.apply_noise()

        if add_objmask:
            _, object_mask = apply_object_mask(self.image)
            self.apply_permanent_mask()
            self.mask *= np.logical_not(
                object_mask)  # self.mask = True for good pixels, so set object_mask'ed pixels to False
            if not os.path.exists(outpath + self.obsid + '_' + self.scaid + '_mask.fits'):
                mask_img= self.mask.astype('uint8')
                save_fits(mask_img, self.obsid + '_' + self.scaid + '_mask', dir=outpath+'masks/', overwrite=True)

    @profile
    def apply_noise(self):
        """
        Add detector noise to self.image
        :param save_fig: Default None. If passed as "obsid_scaid", write out a fits file of SCA image+noise
        :return None
        """
        noiseframe = np.copy(fits.open(labnoise_prefix + self.obsid + '_' + self.scaid + '.fits')[
                                 'PRIMARY'].data) * 1.458 * 50  # times gain and N_frames
        self.image += noiseframe[4:4092, 4:4092]
        filename = self.obsid + '_' + self.scaid + '_noise'

        if not os.path.exists(test_image_dir + filename + '.fits'):
            save_fits(self.image, filename, dir=test_image_dir, overwrite=True)

    @profile
    def apply_permanent_mask(self):
        """
        Apply permanent pixel mask. Updates self.image and self.mask
        :return:
        """
        pm = fits.open(permanent_mask)[0].data[int(self.scaid) - 1].astype(bool)
        self.image *= pm
        self.mask *= pm

    @profile
    def get_permanent_mask(self):
        """
        Apply permanent pixel mask. Updates self.image and self.mask
        :return:
        """
        pm = fits.open(permanent_mask)[0].data[int(self.scaid) - 1]
        pm_array = np.copy(pm)
        return pm_array

    @profile
    def apply_all_mask(self):
        """
        Apply permanent pixel mask. Updates self.image in-place
        :return:
        """
        self.image *= self.mask

    @profile
    def subtract_parameters(self, p, j):
        """
        Subtract a set of parameters from the SCA image. Updates self.image and self.params_subtracted
        :param p: a parameters object, with current params
        :param j: int, the index of the SCA image into all_scas list
        :return: None
        """
        if self.params_subtracted:
            write_to_file('WARNING: PARAMS HAVE ALREADY BEEN SUBTRACTED. ABORTING NOW')
            sys.exit()

        params_image = p.forward_par(j)  # Make destriping params into an image
        self.image = self.image - params_image  # Update I_A.image to have the params image subtracted off
        self.params_subtracted = True

    def get_coordinates(self, pad=0.):
        """
        Create an array of ra, dec coords for the image
        :param pad: Float64, add padding to the array. default is zero.
        :return: ra, dec; 1D np arrays of length (height*width)
        """
        wcs = self.w
        h = self.shape[0] + pad
        w = self.shape[1] + pad
        x_i, y_i = np.meshgrid(np.arange(h), np.arange(w), indexing='xy')
        x_i -= pad / 2.
        y_i -= pad / 2.
        x_flat = x_i.flatten()
        y_flat = y_i.flatten()
        ra, dec = wcs.all_pix2world(x_flat, y_flat, 0)  # 0 is for the first frame (1-indexed)
        return ra, dec

    @profile
    def make_interpolated(self, ind, params=None, N_eff_min=0.5):
        """
        Construct a version of this SCA interpolated from other, overlapping ones.
        Writes the interpolated image out to the disk, to be read/used later
        The N_eff_min parameter requires some minimum effective coverage, otherwise masks that pixel.
        :param ind: int; index of this SCA in all_scas list
        :param params: parameters object; parameters to be subtracted from contributing SCAs; default Nnoe
        :param N_eff_min: float; effective coverage needed for a pixel to contribute to the interpolation
        :return: None
        """
        this_interp = np.zeros(self.shape)

        if not os.path.isfile(tempdir + self.obsid + '_' + self.scaid + '_Neff.dat'):
            N_eff = np.memmap(tempdir + self.obsid + '_' + self.scaid + '_Neff.dat', dtype='float32', mode='w+',
                              shape=self.shape)
            make_Neff = True
        else:
            N_eff = np.memmap(tempdir + self.obsid + '_' + self.scaid + '_Neff.dat', dtype='float32', mode='r',
                              shape=self.shape)
            make_Neff = False

        t_a_start = time.time()
        write_to_file(f'Starting interpolation for SCA {self.obsid}_{self.scaid}')
        sys.stdout.flush()

        N_BinA = 0

        for k, sca_b in enumerate(all_scas):
            obsid_B, scaid_B = get_ids(sca_b)

            if obsid_B != self.obsid and ov_mat[ind, k] != 0:  # Check if this sca_b overlaps sca_a
                N_BinA += 1
                I_B = Sca_img(obsid_B, scaid_B)  # Initialize image B

                if params:
                    I_B.subtract_parameters(params, k)

                I_B.apply_all_mask()  # now I_B is masked
                B_interp = np.zeros_like(self.image)
                interpolate_image_bilinear(I_B, self, B_interp)

                if make_Neff:
                    B_mask_interp = np.zeros_like(self.image)
                    interpolate_image_bilinear(I_B, self, B_mask_interp,
                                               mask=I_B.mask)  # interpolate B pixel mask onto A grid

                if obsid_B == '670' and scaid_B == '10' and make_Neff:  # only do this once
                    save_fits(B_interp, '670_10_B' + self.obsid + '_' + self.scaid + '_interp', dir=test_image_dir)

                if self.obsid == '670' and self.scaid == '10' and make_Neff:
                    save_fits(B_interp, '670_10_A' + obsid_B + '_' + scaid_B + '_interp', dir=test_image_dir)

                this_interp += B_interp
                if make_Neff:
                    N_eff += B_mask_interp

        write_to_file(f'Interpolation of {self.obsid}_{self.scaid} done. Number of contributing SCAs: {N_BinA}')
        new_mask = N_eff > N_eff_min
        this_interp = np.where(new_mask, this_interp / np.where(new_mask, N_eff, N_eff_min),
                               0)

        header = self.w.to_header(relax=True)
        this_interp = np.divide(this_interp, self.g_eff)

        save_fits(this_interp, self.obsid + '_' + self.scaid + '_interp', outpath + 'interpolations/', header=header)
        t_elapsed_a = time.time() - t_a_start

        if make_Neff: N_eff.flush()
        del N_eff
        return this_interp, new_mask


class Parameters:
    """
    Class holding the parameters for a given mosaic. This can be the destriping parameters, or a slew of other
    parameters that need to be the same shape and have the same methods...
    Attributes:
        model: Str, which destriping model to use, which specifies the number of parameters per row.
                Must be a key of the model_params dict
        n_rows: Int, number of rows in the image
        params_per_row: Int, number of parameters per row, set by model_params[model]
        params: 2D np array, the actual array of parameters.
        current_shape: Str, the current shape (1D or 2D) of SCA params
    Methods:
        params_2_images: reshape params into a 2D array; one row per SCA
        # flatten_params: reshape params into 1D vector
        forward_par: reshape one row of params array (one SCA) into a 2D array by projection along rows
    To do:
        add option for additional parameters
    """

    def __init__(self, model, n_rows):
        self.model = model
        self.n_rows = n_rows
        self.params_per_row = model_params[model]
        self.params = np.zeros((len(all_scas), self.n_rows * self.params_per_row)) #default here is float64
        self.current_shape = '2D'

    def params_2_images(self):
        """
        Reshape flattened parameters into 2D array with 1 row per sca and n_rows (in image) * params_per_row entries
        :return: None
        """
        self.params = np.reshape(self.params, (len(all_scas), self.n_rows * self.params_per_row))
        self.current_shape = '2D'

    # def flatten_params(self):
    #     """
    #     Reshape 2D params array into flat
    #     :return: None
    #     """
    #     self.params = np.ravel(self.params)
    #     self.current_shape = '1D'

    @profile
    def forward_par(self, sca_i):
        """
        Takes one SCA row (n_rows) from the params and casts it into 2D (n_rows x n_rows)
        :param sca_i: int, index of which SCA to recast into 2D
        :return: 2D np array, the image of SCA_i's parameters
        """
        if not self.current_shape == '2D':
            self.params_2_images()
        return np.array(self.params[sca_i, :])[:, np.newaxis] * np.ones((self.n_rows, self.n_rows))

@profile
def get_scas(filter, obsfile):
    """
    Function to get a list of all SCA images and their WCSs for this mosaic
    :param filter: Str, which filter to use for this run. Options: Y106, J129, H158, F184, K213
    :param prefix: Str, prefix / name of the SCA images
    :return all_scas: list(strings), list of all the SCAs in this mosiac
    :return all_wcs: list(wcs objects), the WCS object for each SCA in all_scas (same order)
    """
    n_scas = 0
    all_scas = []
    all_wcs = []
    for f in glob.glob(obsfile+ filter + '_*'):
        n_scas += 1
        m = re.search(r'(\w\d+)_(\d+)_(\d+)', f)
        if m:
            this_obsfile = str(m.group(0))
            all_scas.append(this_obsfile)
            this_file = fits.open(f, memmap=True)
            this_wcs = wcs.WCS(this_file['SCI'].header)
            all_wcs.append(this_wcs)
            this_file.close()
    write_to_file(f'N SCA images in this mosaic: {str(n_scas)}')
    write_to_file('SCA List:', 'SCA_list.txt')
    for i, s in enumerate(all_scas):
        write_to_file(f"SCA {i}: {s}", "SCA_list.txt")
    return all_scas, all_wcs

@profile
def interpolate_image_bilinear(image_B, image_A, interpolated_image, mask=None):
    """
    Interpolate values from a "reference" SCA image onto a "target" SCA coordinate grid
    Uses pyimcom_croutines.bilinear_interpolation(float* image, float* g_eff, int rows, int cols, float* coords,
                                                    int num_coords, float* interpolated_image)
    :param image_B : SCA object, the image to be interpolated
    :param image_A : SCA object, the image whose grid you are interpolating B onto
    :param interpolated_image : 2D np array, all zeros with shape of Image A.
    :return: None
    interpolated_image_B is updated in-place.
    """

    x_target, y_target, is_in_ref = compareutils.map_sca2sca(image_A.w, image_B.w, pad=0)
    coords = np.column_stack((y_target.ravel(), x_target.ravel()))

    # Verify data just before C call
    rows = int(image_B.shape[0])
    cols = int(image_B.shape[1])
    num_coords = coords.shape[0]

    sys.stdout.flush()
    sys.stderr.flush()
    if mask is not None and isinstance(mask, np.ndarray):
        mask_geff = np.ones_like(image_A.image)
        pyimcom_croutines.bilinear_interpolation(mask,
                                                 mask_geff,
                                                 rows, cols,
                                                 coords,
                                                 num_coords,
                                                 interpolated_image)
    else:
        pyimcom_croutines.bilinear_interpolation(image_B.image,
                                                 image_B.g_eff,
                                                 rows, cols,
                                                 coords,
                                                 num_coords,
                                                 interpolated_image)

    sys.stdout.flush()
    sys.stderr.flush()

@profile
def transpose_interpolate(image_A, wcs_A, image_B, original_image):
    """
     Interpolate backwards from image_A to image_B space.
     :param image_A : 2D np array, the already-interpolated gradient image
     :param wcs_A : a wcs.WCS object, image A's WCS object
     :param image_B : SCA object, the image we're interpolating the gradient back onto
     :param original_image: 2D np array, the gradient image re-interpolated into image B space
     note: bilinear_transpose(float* image, int rows, int cols, float* coords, int num_coords, float* original_image)
     :return: None
     original_image is updated in-place
     """
    x_target, y_target, is_in_ref = compareutils.map_sca2sca(wcs_A, image_B.w, pad=0)
    coords = np.column_stack((y_target.ravel(), x_target.ravel()))

    rows = int(image_B.shape[0])
    cols = int(image_B.shape[1])
    num_coords = coords.shape[0]

    pyimcom_croutines.bilinear_transpose(image_A,
                                         rows, cols,
                                         coords,
                                         num_coords,
                                         original_image)


def transpose_par(I):
    """
    Sum up the values of an image across rows
    :param I: 2D np array input array
    :return: 1D vector, the sum across each row of I
    """
    return np.sum(I, axis=1)

@profile
def get_effective_gain(sca):
    """
    retrieve the effective gain and n_eff of the image. valid only for already-interpolated images
    :param sca: Str, like "<prefix>_<obsid>_<scaid>" describing which SCA
    :return: g_eff: memmap 2D np array of the effective gain in each pixel
    :return: N_eff: memmap 2D np array of how many image "B"s contributed to that interpolated image
    """
    m = re.search(r'_(\d+)_(\d+)', sca)
    obsid = m.group(1)
    scaid = m.group(2)
    g_eff = np.memmap(tempdir + obsid + '_' + scaid + '_geff.dat', dtype='float64', mode='r', shape=(4088, 4088))
    N_eff = np.memmap(tempdir + obsid + '_' + scaid + '_Neff.dat', dtype='float32', mode='r', shape=(4088, 4088))
    return g_eff, N_eff


def get_ids(sca):
    """
    Take an SCA label and parse it out to get the Obsid and SCA id strings.
    :param sca: Str, the sca name from all_scas list
    :return obsid: Str, the observation ID
    :return scaid: Str, the SCA ID (position in focal plane)
    """
    m = re.search(r'_(\d+)_(\d+)', sca)
    obsid = m.group(1)
    scaid = m.group(2)
    return obsid, scaid


############################ Main Sequence ############################

all_scas, all_wcs = get_scas(filter_, obsfile)
write_to_file(f"{len(all_scas)} SCAs in this mosaic")

if testing:
    if os.path.isfile(outpath + 'ovmat.npy'):
        ov_mat = np.load(outpath + 'ovmat.npy')
    else:
        ovmat_t0 = time.time()
        write_to_file('Overlap matrix computing start')
        ov_mat = compareutils.get_overlap_matrix(all_wcs, verbose=True)
        np.save(outpath + 'ovmat.npy', ov_mat)
        write_to_file(f"Overlap matrix complete. Duration: {(time.time() - ovmat_t0) / 60} Minutes")
        write_to_file(f"Overlap matrix saved to: {outpath}ovmat.npy")
else:
    ovmat_t0 = time.time()
    write_to_file('Overlap matrix computing start')
    ov_mat = compareutils.get_overlap_matrix(all_wcs,
                                             verbose=True)  # an N_wcs x N_wcs matrix containing fractional overlap
    write_to_file(f"Overlap matrix complete. Duration: {(time.time() - ovmat_t0) / 60} Minutes")

def residual_function_single(k, sca_a, psi, f_prime, thresh=None):
    # Go and get the WCS object for image A
    obsid_A, scaid_A = get_ids(sca_a)
    filepath = outpath + f'interpolations/{obsid_A}_{scaid_A}_interp.fits'
    lockfile = f"{filepath}.lock"
    timeout=30
    try:
        with FileLock(lockfile, timeout=timeout):
            with fits.open(filepath, memmap=True) as file:
                wcs_A = wcs.WCS(file[0].header)
                file.close()
    except Timeout:
        print(f"RF Exception Timeout: Could not acquire lock on {filepath} within {timeout} seconds.")
    except FileNotFoundError:
        print(f"RF Exception File not found: {filepath}")
    except Exception as e:
        print(f"RF Exception Other error: {e}")

    # Calculate and then transpose the gradient of I_A-J_A
    if TIME: T = time.time()
    gradient_interpolated = f_prime(psi[k, :, :], thresh) if thresh is not None else f_prime(psi[k,:,:])

    term_1 = transpose_par(gradient_interpolated)

    # Retrieve the effective gain and N_eff to normalize the gradient before transposing back
    g_eff_A, n_eff_A = get_effective_gain(sca_a)

    # Avoid dividing by zero
    valid_mask = n_eff_A != 0
    gradient_interpolated[valid_mask] = gradient_interpolated[valid_mask] / (
            g_eff_A[valid_mask] * n_eff_A[valid_mask])
    gradient_interpolated[~valid_mask] = 0
    if TIME: write_to_file(f"Time re-normalizing SCA A for transpose interp: {time.time() - T} seconds")

    term_2_list = []
    for j, sca_b in enumerate(all_scas):
        obsid_B, scaid_B = get_ids(sca_b)

        if obsid_B != obsid_A and ov_mat[k, j] != 0:
            I_B = Sca_img(obsid_B, scaid_B)
            gradient_original = np.zeros(I_B.shape)

            if TIME: T = time.time()
            transpose_interpolate(gradient_interpolated, wcs_A, I_B, gradient_original)
            if TIME: write_to_file(f"Time in transpose interpolation: {time.time() - T} seconds")
            gradient_original *= I_B.g_eff

            term_2 = transpose_par(gradient_original)
            term_2_list.append((j, term_2))

            # if obsid_A == '670' and scaid_A == '10':
            #     write_to_file('670_10 sample stats:')
            #     write_to_file(f'Terms 1 and 2 means: {np.mean(term_1)}, {np.mean(term_2)}')
            #     write_to_file(f'G_eff_a, G_eff_b means: {np.mean(g_eff_A)}, {np.mean(I_B.g_eff)}')

    return k, term_1, term_2_list

def cost_function_single(j, sca_a, p, f, thresh=None):
    m = re.search(r'_(\d+)_(\d+)', sca_a)
    obsid_A, scaid_A = m.group(1), m.group(2)

    I_A = Sca_img(obsid_A, scaid_A)
    I_A.subtract_parameters(p, j)
    I_A.apply_all_mask()

    if obsid_A == '670' and scaid_A == '10':
        hdu = fits.PrimaryHDU(I_A.image)
        hdu.writeto(test_image_dir + '670_10_I_A_sub_masked.fits', overwrite=True)

    if TIME: t = time.time()
    J_A_image, J_A_mask = I_A.make_interpolated(j, params=p)
    if TIME: write_to_file(f"Time making {sca_a} J_A: {time.time() - t} seconds")

    J_A_mask *= I_A.mask

    psi = np.where(J_A_mask, I_A.image - J_A_image, 0)
    result = f(psi, thresh) if thresh is not None else f(psi)
    local_epsilon = np.sum(result)

    if obsid_A == '670' and scaid_A == '10':
        hdu = fits.PrimaryHDU(J_A_image * J_A_mask)
        hdu.writeto(test_image_dir + '670_10_J_A_masked.fits', overwrite=True)

        hdu = fits.PrimaryHDU(psi)
        hdu.writeto(test_image_dir + '670_10_Psi.fits', overwrite=True)

        write_to_file('Sample stats for SCA 670_10:')
        write_to_file(f'Image A mean: {np.mean(I_A.image)}')
        write_to_file(f'Image B mean: {np.mean(J_A_image)}')
        write_to_file(f'Psi mean: {np.mean(psi)}')
        write_to_file(f'f(Psi) mean: {np.mean(result)}')
        write_to_file(f"Local epsilon for SCA {j}: {local_epsilon}")

    return j, psi, local_epsilon

# Optimization Functions

def main():

    workers = os.cpu_count() // int(os.environ['OMP_NUM_THREADS']) if 'OMP_NUM_THREADS' in os.environ else 12
    write_to_file(f"## Using {workers} workers for parallel processing.")

    def cost_function(p, f, thresh=None):
        """
        Calculate the cost function with the current de-striping parameters.
        :param p: parameters object, the current parameters for de-striping
        :param f: str, keyword for function dictionary options; should also set an f_prime
        :return epsilon: int, the total cost function summed over all images
        :return psi: 3D np array, the difference images I_A-J_A
        """
        write_to_file('Initializing cost function')
        t0_cost = time.time()
        psi = np.zeros((len(all_scas), 4088, 4088))
        epsilon = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(cost_function_single, j, sca_a, p, f, thresh) for j, sca_a in enumerate(all_scas)]

        for future in as_completed(futures):
            j, psi_j, local_eps = future.result()
            psi[j, :, :] = psi_j
            epsilon += local_eps

        write_to_file(f'Ending cost function. Time elapsed: {(time.time() - t0_cost) / 60} minutes')
        write_to_file(f'Average time per cost function iteration: {(time.time() - t0_cost) / len(all_scas)} seconds')
        return epsilon, psi

    def residual_function(psi, f_prime, thresh=None, extrareturn=False):
        """
        Calculate the residual image, = grad(epsilon)
        :param psi: 3D np array, the image difference array (I_A - J_A) (N_SCA, 4088, 4088)
        :param f_prime: function, the derivative of the cost function f
                in the future this should be set by default based on what you pass for f
        :param extrareturn: Bool (default False); if True, return residual terms 1 and 2 separately
                in addition to full residuals. returns resids, resids1, resids2
        :return resids: 2D np array, with one row per SCA and one col per parameter
        """
        resids = Parameters(use_model, 4088).params
        if extrareturn:
            resids1 = np.zeros_like(resids)
            resids2 = np.zeros_like(resids)
        write_to_file('Residual calculation started')
        t_r_0 = time.time()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(residual_function_single, k, sca_a, psi, f_prime, thresh) for k, sca_a in
                       enumerate(all_scas)]

        for future in as_completed(futures):
            k, term_1, term_2_list = future.result()
            resids[k, :] -= term_1
            if extrareturn:
                resids1[k, :] -= term_1

            # Process term_2 contributions
            for j, term_2 in term_2_list:
                resids[j, :] += term_2
                if extrareturn:
                    resids2[j, :] += term_2

        write_to_file(f'Residuals calculation finished in {(time.time() - t_r_0) / 60} minutes.')
        write_to_file(f"Average time making resids per sca: {(time.time() - t_r_0) / len(all_scas)} seconds")
        if extrareturn: return resids, resids1, resids2
        return resids

    
    def linear_search(p, direction, f, f_prime, grad_current, thresh=None, n_iter=100, tol=10 ** -4):
        """
        Linear search via combination bisection and secant methods for parameters that minimize the function
         d_epsilon/d_alpha in the given direction . Note alpha = depth of step in direction
        :param p: params object, the current de-striping parameters
        :param direction: 2D np array, direction of conjugate gradient search
        :param f: function, cost function form
        :param f_prime: function, derivative of cost function form
        :param grad_current: 2D np array, current gradient
        :param n_iter: int, number of iterations at which to stop searching
        :param tol: float, absolute value of d_cost at which to converge
        :return best_p: parameters object, containing the best parameters found via search
        :return best_psi: 3D numpy array, the difference images made from images with the best_p params subtracted off
        """
        best_epsilon, best_psi = cost_function(p, f, thresh)
        best_p = copy.deepcopy(p)

        # Simple linear search
        working_p = copy.deepcopy(p)
        max_p = copy.deepcopy(p)
        min_p = copy.deepcopy(p)

        convergence_crit = 99.
        method = 'bisection'

        eta = 0.1
        d_cost_init = np.sum(grad_current * direction)
        d_cost_tol = np.abs(d_cost_init * 1*10**-3)

        if cost_model=='quadratic':
            alpha_test = -eta * (np.sum(grad_current*direction))/(np.sum(direction*direction)+1e-12)
            if alpha_test <= 0:
                # Not a descent direction — fallback
                alpha_min = -0.9
                alpha_max = 1.0
            else:
                # Curvature-based search window
                alpha_min = alpha_test * 1e-4
                alpha_max = alpha_test * 10

        elif cost_model=='huber_loss':
            alpha_test = 1.
            alpha_min = 1e-4
            alpha_max = 10

        # Calculate f(alpha_max) and f(alpha_min), which need to be defined for secant update
        write_to_file('### Calculating min and max epsilon and cost')
        max_params = p.params + alpha_max * direction
        max_p.params = max_params
        max_epsilon, max_psi = cost_function(max_p, f, thresh)
        max_resids = residual_function(max_psi, f_prime, thresh)
        d_cost_max = np.sum(max_resids * direction)

        min_params = p.params + alpha_min * direction
        min_p.params = min_params
        min_epsilon, min_psi = cost_function(min_p, f, thresh)
        min_resids = residual_function(min_psi, f_prime, thresh)
        d_cost_min = np.sum(min_resids * direction)

        conv_params = []

        for k in range(1, n_iter):
            t0_ls_iter = time.time()

            if k == 1:
                write_to_file('### Beginning linear search')
                write_to_file(f"LS Direction: {direction}")
                hdu = fits.PrimaryHDU(direction)
                hdu.writeto(test_image_dir + 'LSdirection.fits', overwrite=True)
                write_to_file(f"Initial params: {p.params}")
                write_to_file(f"Initial epsilon: {best_epsilon}")
                write_to_file(f"Initial d_cost: {d_cost_init}, d_cost tol: {d_cost_tol}")
                write_to_file(f"Initial alpha range (min, test, max): ({alpha_min}, {alpha_test}, {alpha_max})")

            if k == n_iter - 1:
                write_to_file(
                    'WARNING: Linear search did not converge!! This is going to break because best_p is not assigned.')

            if k!=1:
                alpha_test = alpha_min - (
                            d_cost_min * (alpha_max - alpha_min) / (d_cost_max - d_cost_min))  # secant update
                write_to_file(f"Secant update: alpha_test={alpha_test}")
                method = 'secant'
                if np.isnan(alpha_test):
                    write_to_file('Secant update fail-- bisecting instead')
                    alpha_test = .5 * (alpha_min + alpha_max)  # bisection update
                    write_to_file(f"Bisection update: alpha_test={alpha_test}")
                    method = 'bisection'
            elif k==1:
                alpha_test = .5 * (alpha_min + alpha_max)  # bisection update
                write_to_file(f"Bisection update: alpha_test={alpha_test}")

            working_params = p.params + alpha_test * direction
            working_p.params = working_params

            working_epsilon, working_psi = cost_function(working_p, f, thresh)
            working_resids = residual_function(working_psi, f_prime, thresh)
            d_cost = np.sum(working_resids * direction)
            convergence_crit = (alpha_max - alpha_min)
            conv_params.append([working_epsilon, alpha_test, d_cost])

            if k % 10 == 0:
                hdu = fits.PrimaryHDU(working_resids)
                hdu.writeto(test_image_dir + 'LS_Residuals_' + str(k) + '.fits', overwrite=True)

            write_to_file(f"Ending LS iteration {k}")
            write_to_file(f"Current d_cost = {d_cost}, epsilon = {working_epsilon}")
            write_to_file(f"Working resids: {working_resids}")
            write_to_file(f"Working params: {working_p.params}")
            write_to_file(f"Current alpha range (min, test, max): {alpha_min, alpha_test, alpha_max}")
            write_to_file(f"Current delta alpha: {convergence_crit}")
            if TIME:  write_to_file(f"Time spent in this LS iteration: {(time.time() - t0_ls_iter) / 60} minutes.")

            # Convergence and update criteria and checks
            if ((working_epsilon < best_epsilon + tol * alpha_test * d_cost) and (np.abs(alpha_test)>=1e-6)):
                best_epsilon = working_epsilon
                best_p = copy.deepcopy(working_p)
                best_psi = working_psi
                best_resids = working_resids
                write_to_file(f"Linear search convergence in {k} iterations")
                save_fits(best_p.params, 'best_p', dir=test_image_dir, overwrite=True)
                save_fits(np.array(conv_params), 'conv_params', dir=test_image_dir, overwrite=True)
                return best_p, best_psi ,best_resids

            # if np.abs(d_cost) < tol:
            #     write_to_file(f"Linear search convergence via |d_cost|< {tol} in {k} iterations")
            #     write_to_file("I think this is bad because nothing has actually been updated if I exit this way... \n"
            #                   "so I'm going to let this break the program bc best_resids isn't defined")
            #     hdu = fits.PrimaryHDU(best_p.params)
            #     hdu.writeto(test_image_dir + 'best_p.fits', overwrite=True)
            #     hdu = fits.PrimaryHDU(np.array(conv_params))
            #     hdu.writeto(test_image_dir + 'conv_params.fits', overwrite=True)
            #     return best_p, best_psi, best_resids
            #
            # if convergence_crit < (0.01 / current_norm):
            #     write_to_file(f"Linear search convergence via crit<{0.01 / current_norm} in {k} iterations")
            #     write_to_file("I think this is bad because nothing has actually been updated if I exit this way... \n"
            #                   "so I'm going to let this break the program bc best_resids isn't defined")
            #     hdu = fits.PrimaryHDU(best_p.params)
            #     hdu.writeto(test_image_dir + 'best_p.fits', overwrite=True)
            #     hdu = fits.PrimaryHDU(np.array(conv_params))
            #     hdu.writeto(test_image_dir + 'conv_params.fits', overwrite=True)
            #     return best_p, best_psi, best_resids

            # Updates for next iteration, if convergence isn't yet reached
            if d_cost > tol and method == 'bisection':
                alpha_max = alpha_test
                d_cost_max = d_cost
            elif d_cost < -tol and method == 'bisection':
                alpha_min = alpha_test
                d_cost_min = d_cost
            elif d_cost * d_cost_min < 0 and method == 'secant':
                alpha_max = alpha_test
                d_cost_max = d_cost
            elif d_cost * d_cost_max < 0 and method == 'secant':
                alpha_min = alpha_test
                d_cost_min = d_cost

        return best_p, best_psi

    
    def conjugate_gradient(p, f, f_prime, method='FR', tol=1e-5, max_iter=100, thresh=None):
        """
        Algorithm to use conjugate gradient descent to optimize the parameters for destriping.
        Direction is updated using Fletcher-Reeves method
        :param p: parameters object, containing initial parameters guess
        :param f: function, functional form to use for cost function
        :param f_prime: function, the derivative of f. KL: eventually f should dictate f prime
        :param method: str, the method to use for CG direction update. Default: 'FR'
                Current Options: 'FR', 'PR', 'HS', 'DY' (Fletcher-Reeves, Polak-Ribiere, Hestenes-Stiefel, Dai-Yuan)
        :param tol: float, the value of the norm at which we say CG has converged
        :param max_iter: int, number of iterations at which to force CG to stop
        :return p: params object, the best fit parameters for destriping the SCA images
        """
        write_to_file('### Starting conjugate gradient optimization')
        print(f'HL Threshold (None if other cost fn): {thresh}')

        # Initialize variables
        grad_prev = None  # No previous gradient initially
        direction = None  # No initial direction
        log_file = os.path.join(outpath, 'cg_log.csv')

        with open(log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Iteration', 'Current Norm', 'Convergence Rate', 'Step Size', 'Gradient Magnitude',
                              'Final d_cost', 'Final Epsilon', 'Time (min)', 'LS time (min)',
                             'MSE', 'Parameter Change'])

        write_to_file('### Starting initial cost function')
        global test_image_dir
        test_image_dir = outpath + '/test_images/' + str(0) + '/'
        psi = cost_function(p, f, thresh)[1]
        sys.stdout.flush()

        for i in range(max_iter):
            write_to_file(f"### CG Iteration: {i + 1}")
            test_image_dir = outpath + '/test_images/' + str(i+1) + '/'
            os.makedirs(test_image_dir, exist_ok=True)
            t_start_CG_iter = time.time()

            # Compute the gradient
            if i==0:
                grad, gr_term1, gr_term2 = residual_function(psi, f_prime, thresh, extrareturn=True)
                del gr_term1, gr_term2
                write_to_file(f"Minutes spent in initial residual function: {(time.time() - t_start_CG_iter) / 60}")
                sys.stdout.flush()

            # Compute the norm of the gradient
            global current_norm
            current_norm = np.linalg.norm(grad)

            if i == 0:
                write_to_file(f'Initial gradient: {grad}')
                norm_0 = np.linalg.norm(grad)
                write_to_file(f'Initial norm: {norm_0}')
                tol = tol * norm_0
                direction = -grad

            elif i%10 == 0 :
                beta = 0
                write_to_file(f"Current Beta: {beta} (using method: {method})")
                direction = -grad + beta * direction_prev

            else:
                # Calculate beta (direction scaling) depending on method
                if method=='FR': beta = np.sum(np.square(grad)) / np.sum(np.square(grad_prev))
                elif method=='PR': beta = max(0,np.sum(grad * (grad - grad_prev)) / (np.sum(np.square(grad_prev))))
                elif method== 'HS': beta = (np.sum(grad * (grad - grad_prev)) /
                                            np.sum(-direction_prev * (grad - grad_prev)))
                elif method== 'DY': beta = np.sum(np.square(grad)) / np.sum(-direction_prev * (grad - grad_prev))
                else: raise ValueError(f"Unknown method for CG direction update: {method}"
                                       f" (Options are: {CG_models})")

                write_to_file(f"Current Beta: {beta} (using method: {method})")

                direction = -grad + beta * direction_prev

            if current_norm < tol:
                write_to_file(f"Convergence reached at iteration: {i + 1} via norm {current_norm} < tol {tol}")
                break

            # Perform linear search
            t_start_LS = time.time()
            write_to_file(f"Initiating linear search in direction: {direction}")
            p_new, psi_new, grad_new = linear_search(p, direction, f, f_prime, grad, thresh)
            ls_time = (time.time() - t_start_LS) / 60
            write_to_file(f'Total time spent in linear search: {ls_time}')
            write_to_file(
                f'Current norm: {current_norm}, Tol * Norm_0: {tol}, Difference (CN-TOL): {current_norm - tol}')
            write_to_file(f'Current best params: {p_new.params}')

            # Calculate additional metrics
            convergence_rate = (current_norm - np.linalg.norm(grad_new)) / current_norm
            step_size = np.linalg.norm(p_new.params - p.params)
            gradient_magnitude = np.linalg.norm(grad_new)
            mse = np.mean(psi_new ** 2)
            parameter_change = np.linalg.norm(p_new.params - p.params)

            with open(log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([i + 1, current_norm, convergence_rate, step_size, gradient_magnitude,
                                  np.sum(grad * direction), np.sum(psi),
                                 (time.time() - t_start_CG_iter)/60, ls_time, mse, parameter_change])

            # Update to current values
            p = p_new
            psi = psi_new
            grad_prev = grad
            grad = grad_new
            direction_prev = direction

            write_to_file(f'Total time spent in this CG iteration: {(time.time() - t_start_CG_iter) / 60} minutes.')
            sys.stdout.flush()

            if i == max_iter - 1:
                write_to_file(f'CG reached MAX ITERATIONS {max_iter} and DID NOT converge!!!!')

        write_to_file(f'Conjugate gradient complete. Finished in {i + 1} / {max_iter} iterations')
        write_to_file(f'Final parameters: {p.params}')
        write_to_file(f'Final norm: {current_norm}')
        return p

    # Initialize parameters
    p0 = Parameters(use_model, 4088)

    # Do it
    p = conjugate_gradient(p0, Cost_models(cost_model).f, Cost_models(cost_model).f_prime,
                           cg_model, cg_tol, cg_maxiter, Cost_models(cost_model).thresh)
    hdu = fits.PrimaryHDU(p.params)
    hdu.writeto(outpath + 'final_params.fits', overwrite=True)
    print(outpath + 'final_params.fits created \n')

    for i, sca in enumerate(all_scas):
        obsid, scaid = get_ids(sca)
        this_sca = Sca_img(obsid, scaid, add_objmask=False)
        this_param_set = p.forward_par(i)
        ds_image = this_sca.image - this_param_set
        pm = this_sca.get_permanent_mask()

        hdu = fits.PrimaryHDU(ds_image, header=this_sca.header)
        hdu.header['TYPE'] = 'DESTRIPED_IMAGE'
        hdu2 = fits.ImageHDU(this_sca.image, header=this_sca.header)
        hdu2.header['TYPE'] = 'SCA_IMAGE'
        hdu3 = fits.ImageHDU(pm, header=this_sca.header)
        hdu3.header['TYPE'] = 'PERMANENT_MASk'
        hdu4 = fits.ImageHDU(this_param_set, header=this_sca.header)
        hdu4.header['TYPE'] = 'PARAMS_IMAGE'
        hdulist = fits.HDUList([hdu, hdu2, hdu3, hdu4])
        hdulist.writeto(outpath + filter_ + '_DS_' + obsid + scaid + '.fits', overwrite=True)

    write_to_file(f'Destriped images saved to {outpath + filter_} _DS_*.fits')
    write_to_file(f'Total hours elapsed: {(time.time() - t0) / 3600}')


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    mem_usage = None
    try:
        mem_usage = memory_usage(main, interval=1800, retval=False)
    finally:
        profiler.disable()
        stream=io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats()
        with open(outpath+'profile_results.txt', 'w') as f:
            f.write(stream.getvalue())
        if mem_usage is not None:
            with open(outpath + 'memory_profile_results.txt', 'w') as f:
                for i, mem in enumerate(mem_usage):
                    f.write(f"{i}\t{mem:.2f} MiB\n")