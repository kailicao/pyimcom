"""
Program to remove correlated noise stripes from RST images.
"""

import os
import glob
import time
import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy.signal import convolve2d, residue
from utils import compareutils
from .config import Settings as Stn, Config
import re
import sys
import copy
import furryparakeet.pyimcom_croutines
from concurrent.futures import ProcessPoolExecutor, as_completed
from filelock import Timeout, FileLock

tempfile_Katherine_dir = True
TIME = True

filters = ['Y106', 'J129', 'H158', 'F184', 'K213']
areas = [7006, 7111, 7340, 4840, 4654]  # cm^2
model_params = {'constant': 1, 'linear': 2}
CG_models = {'FR', 'PR', 'HS', 'DY'}

s_in = 0.11  # arcsec^2
t_exp = 154  # sec

# Import config file
CFG = Config(cfg_file='configs/config_destripe-H.json')
filter_ = filters[CFG.use_filter]
A_eff = areas[CFG.use_filter]
obsfile = CFG.obsfile
labnoise_prefix = CFG.inpath
use_model = CFG.ds_model
permanent_mask = CFG.permanent_mask
cg_model = CFG.cg_model
cost_model = CFG.cost_model
resid_model = CFG.resid_model

if use_model not in [model_params.keys()]:
    raise ValueError(f"Model {use_model} not in model_params dictionary.")
if tempfile_Katherine_dir:
    obsfile = '/fs/scratch/PCON0003/klaliotis/destripe/inputs/Roman_WAS_simple_model_'
    tempfile = '/fs/scratch/PCON0003/klaliotis/destripe/test_out/'
if CFG.cost_prior != 0:
    cost_prior = CFG.cost_prior
if cg_model not in CG_models:
    raise ValueError(f"CG model {cg_model} not in CG_models dictionary.")
global outfile
outfile = CFG.ds_outpath + filter_ + CFG.ds_outstem

CFG()
CFG.to_file(outfile+'ds.cfg')

t0 = time.time()

def write_to_file(text, filename=None):
    """
    Function to write some text to an output file
    :param text: Str, what to print
    :param filename: Str, an alternative filename if not going into the outfile
    :return: nothing
    """
    if filename is None:
        filename = outfile
    with open(filename, "a") as f:
        f.write(text + '\n')
    print(text)


def save_fits(image, filename, dir=tempfile, overwrite=True, s=False):
    """
    Function to save an image to .fits.
    :param image: 2D np array; the image
    :param filename: str; the filename
    :return: None
    """
    filepath = dir + filename + '.fits'
    lockpath = filepath + '.lock'
    lock = FileLock(lockpath)

    try:
        with lock.acquire(timeout=30):
            hdu = fits.PrimaryHDU(image)
            hdu.writeto(filepath, overwrite=overwrite)
            if s: write_to_file(f"Array {filename} written out to {dir + filename + '.fits'}")
    except Timeout:
        write_to_file(f" Failed to write {filename}; lock acquire timeout")


# C.H. wanted to define this before any use of sca_img so moved it up.
def apply_object_mask(image, mask=None):
    """
    Apply a bright object mask to an image.
    :param image: 2D numpy array, the image to be masked.
    :param mask: optional: 2D numpy array, the pre-existing object mask you wish to use
    :return: the image with bright objects (flux>2*median; could modify later) masked out
    """
    if mask is not None and isinstance(mask, np.ndarray):
        neighbor_mask = mask
    else:
        # Create a binary mask for high-value pixels (KL: could modify later)
        high_value_mask = image >= 2 * np.median(image)

        # Convolve the binary mask with a 5x5 kernel to include neighbors
        kernel = np.ones((5, 5), dtype=int)
        neighbor_mask = convolve2d(high_value_mask, kernel, mode='same') > 0

    # Set the target pixels and their neighbors to zero
    image = np.where(neighbor_mask, 0, image)
    return image, neighbor_mask


class Cost_models:
    """
    Class holding the cost function models. This is a dictionary of functions
    """

    def __init__(self, model):

        self.model = model

        def quadratic(x):
            return x ** 2

        def absolute(x):
            return np.abs(x)

        def huber_loss(x, x0, d):
            if (x - x0) <= d:
                return quadratic(x - x0)
            else:
                return absolute(x - x0)

        # Derivatives
        def quad_prime(x):
            return 2 * x

        def abs_prime(x):
            return np.sign(x)

        def huber_prime(x, x0, b):
            if (x - x0) <= b:
                return quad_prime(x - x0)
            else:
                return abs_prime(x - x0)

        models = {"quadratic": (quadratic, quad_prime), "absolute": (absolute, abs_prime),
                  "huber_loss": (huber_loss, huber_prime)}
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
            file = fits.open(tempfile + 'interpolations/' + obsid + '_' + scaid + '_interp.fits')
            image_hdu = 'PRIMARY'
        else:
            file = fits.open(obsfile + filter_ + '_' + obsid + '_' + scaid + '.fits')
            image_hdu = 'SCI'
        self.image = np.copy(file[image_hdu].data).astype(np.float32)
        self.shape = np.shape(self.image)
        self.w = wcs.WCS(file[image_hdu].header)
        file.close()

        self.obsid = obsid
        self.scaid = scaid
        self.mask = np.ones(self.shape, dtype=bool)
        self.params_subtracted = False

        # Calculate effecive gain
        if not os.path.isfile(tempfile + obsid + '_' + scaid + '_geff.dat'):
            g0 = time.time()
            g_eff = np.memmap(tempfile + obsid + '_' + scaid + '_geff.dat', dtype='float32', mode='w+',
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

        self.g_eff = np.memmap(tempfile + obsid + '_' + scaid + '_geff.dat', dtype='float32', mode='r',
                               shape=self.shape)

        # Add a noise frame, if requested
        if add_noise: self.apply_noise()

        if add_objmask:
            _, object_mask = apply_object_mask(self.image)
            self.apply_permanent_mask()
            self.mask *= np.logical_not(
                object_mask)  # self.mask = True for good pixels, so set object_mask'ed pixels to False

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

    def apply_permanent_mask(self):
        """
        Apply permanent pixel mask. Updates self.image and self.mask
        :return:
        """
        pm = fits.open(permanent_mask)[0].data[int(self.scaid) - 1].astype(bool)
        self.image *= pm
        self.mask *= pm

    def apply_all_mask(self):
        """
        Apply permanent pixel mask. Updates self.image and self.mask
        :return:
        """
        self.image *= self.mask

    def subtract_parameters(self, p, j):
        """
        Subtract a set of parameters from the SCA image. Updates self.image and self.params_subtracted
        :param p: a parameters object, with current params
        :param j: int, the index of the SCA image into all_scas list
        :return: None
        """
        if self.params_subtracted == True:
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

        if not os.path.isfile(tempfile + self.obsid + '_' + self.scaid + '_Neff.dat'):
            N_eff = np.memmap(tempfile + self.obsid + '_' + self.scaid + '_Neff.dat', dtype='float32', mode='w+',
                              shape=self.shape)
            make_Neff = True
        else:
            N_eff = np.memmap(tempfile + self.obsid + '_' + self.scaid + '_Neff.dat', dtype='float32', mode='r',
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
                # I_B.apply_noise() <-- redundant

                if self.obsid == '670' and self.scaid == '10':
                    print('Image B index:' + str(k))
                    print('\nI_B: ', obsid_B, scaid_B, 'Pre-Param-Subtraction mean:', np.mean(I_B.image))

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

        write_to_file(f'Interpolation done. Number of contributing SCAs: {N_BinA}')
        new_mask = N_eff > N_eff_min
        this_interp = np.where(new_mask, this_interp / np.where(new_mask, N_eff, N_eff_min),
                               0)  # only do the division where N_eff nonzero
        header = self.w.to_header(relax=True)
        this_interp = np.divide(this_interp, self.g_eff)
        hdu = fits.PrimaryHDU(this_interp, header=header)
        hdu.writeto(tempfile + 'interpolations/' + self.obsid + '_' + self.scaid + '_interp.fits', overwrite=True)
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
        self.params = np.zeros((len(all_scas), self.n_rows * self.params_per_row))
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

    def forward_par(self, sca_i):
        """
        Takes one SCA row (n_rows) from the params and casts it into 2D (n_rows x n_rows)
        :param sca_i: int, index of which SCA to recast into 2D
        :return: 2D np array, the image of SCA_i's parameters
        """
        if not self.current_shape == '2D':
            self.params_2_images()
        return np.array(self.params[sca_i, :])[:, np.newaxis] * np.ones((self.n_rows, self.n_rows))


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
            this_file = fits.open(f)
            this_wcs = wcs.WCS(this_file['SCI'].header)
            all_wcs.append(this_wcs)
            this_file.close()
    write_to_file(f'N SCA images in this mosaic: {str(n_scas)}')
    write_to_file('SCA List:', 'SCA_list.txt')
    for i, s in enumerate(all_scas):
        write_to_file(f"SCA {i}: {s}", "SCA_list.txt")
    return all_scas, all_wcs


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
    g_eff = np.memmap(tempfile + obsid + '_' + scaid + '_geff.dat', dtype='float32', mode='r', shape=(4088, 4088))
    N_eff = np.memmap(tempfile + obsid + '_' + scaid + '_Neff.dat', dtype='float32', mode='r', shape=(4088, 4088))
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

if tempfile_Katherine_dir:
    if os.path.isfile(tempfile + 'ovmat.npy'):
        ov_mat = np.load(tempfile + 'ovmat.npy')
    else:
        ovmat_t0 = time.time()
        write_to_file('Overlap matrix computing start')
        ov_mat = compareutils.get_overlap_matrix(all_wcs, verbose=True)
        np.save(tempfile + 'ovmat.npy', ov_mat)
        write_to_file(f"Overlap matrix complete. Duration: {(time.time() - ovmat_t0) / 60} Minutes")
        write_to_file(f"Overlap matrix saved to: {tempfile}ovmat.npy")
else:
    ovmat_t0 = time.time()
    write_to_file('Overlap matrix computing start')
    ov_mat = compareutils.get_overlap_matrix(all_wcs,
                                             verbose=True)  # an N_wcs x N_wcs matrix containing fractional overlap
    write_to_file(f"Overlap matrix complete. Duration: {(time.time() - ovmat_t0) / 60} Minutes")


def residual_function_single(k, sca_a, psi, f_prime):
    # Go and get the WCS object for image A
    obsid_A, scaid_A = get_ids(sca_a)
    file = fits.open(tempfile + 'interpolations/' + obsid_A + '_' + scaid_A + '_interp.fits')
    wcs_A = wcs.WCS(file[0].header)
    file.close()

    # Calculate and then transpose the gradient of I_A-J_A
    if TIME: T = time.time()
    gradient_interpolated = f_prime(psi[k, :, :])

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

            if obsid_A == '670' and scaid_A == '10':
                write_to_file('670_10 sample stats:')
                write_to_file(f'Terms 1 and 2 means: {np.mean(term_1)}, {np.mean(term_2)}')
                write_to_file(f'G_eff_a, G_eff_b means: {np.mean(g_eff_A)}, {np.mean(I_B.g_eff)}')

    return k, term_1, term_2_list


def cost_function_single(j, sca_a, p, f):
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
    local_epsilon = np.sum(f(psi))

    if obsid_A == '670' and scaid_A == '10':
        hdu = fits.PrimaryHDU(J_A_image * J_A_mask)
        hdu.writeto(test_image_dir + '670_10_J_A_masked.fits', overwrite=True)

        hdu = fits.PrimaryHDU(psi)
        hdu.writeto(test_image_dir + '670_10_Psi.fits', overwrite=True)

        write_to_file('Sample stats for SCA 670_10:')
        write_to_file(f'Image A mean, std: {np.mean(I_A.image)}, {np.std(I_A.image)}')
        write_to_file(f'Image B mean, std: {np.mean(J_A_image)}, {np.std(J_A_image)}')
        write_to_file(f'Psi mean, std: {np.mean(psi)}, {np.std(psi)}')
        write_to_file(f'f(Psi) mean, std: {np.mean(f(psi))}, {np.std(f(psi))}')
        write_to_file(f"Local epsilon for SCA {j}: {local_epsilon}")

    return j, psi, local_epsilon

# Optimization Functions

def main():
    def cost_function(p, f):
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

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(cost_function_single, j, sca_a, p, f) for j, sca_a in enumerate(all_scas)]

        for future in as_completed(futures):
            j, psi_j, local_eps = future.result()
            psi[j, :, :] = psi_j
            epsilon += local_eps

        write_to_file(f'Ending cost function. Time elapsed: {(time.time() - t0_cost) / 60} minutes')
        write_to_file(f'Average time per cost function iteration: {(time.time() - t0_cost) / len(all_scas)} seconds')
        return epsilon, psi

    def residual_function(psi, f_prime, extrareturn=False):
        """
        Calculate the residual image, = grad(epsilon)
        :param psi: 3D np array, the image difference array (I_A - J_A) (N_SCA, 4088, 4088)
        :param f_prime: function, the derivative of the cost function f
                in the future this should be set by default based on what you pass for f
        :param extrareturn: Bool (default False); if True, return residual terms 1 and 2 separately
                in addition to full residuals. returns resids, resids1, resids2
        :return resids: 2D np array, with one row per SCA and one col per parameter
        """
        resids = (Parameters(use_model, 4088).params)
        if extrareturn:
            resids1 = np.zeros_like(resids)
            resids2 = np.zeros_like(resids)
        write_to_file('Residual calculation started')
        t_r_0 = time.time()

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(residual_function_single, k, sca_a, psi, f_prime) for k, sca_a in
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
        write_to_file((f"Average time making resids per sca: {(time.time() - t_r_0) / len(all_scas)} seconds"))
        if extrareturn: return resids, resids1, resids2
        return resids

    def linear_search(p, direction, f, f_prime, n_iter=100, tol=10 ** -3):
        """
        Linear search via combination bisection and secant methods for parameters that minimize the function
         d_epsilon/d_alpha in the given direction . Note alpha = depth of step in direction
        :param p: params object, the current de-striping parameters
        :param direction: 2D np array, direction of conjugate gradient search
        :param f: function, cost function form
        :param f_prime: function, derivative of cost function form
        :param n_iter: int, number of iterations at which to stop searching
        :param tol: float, absolute value of d_cost at which to converge
        :return best_p: parameters object, containing the best parameters found via search
        :return best_psi: 3D numpy array, the difference images made from images with the best_p params subtracted off
        """
        best_epsilon, best_psi = cost_function(p, f)
        best_p = copy.deepcopy(p)

        # Simple linear search
        working_p = copy.deepcopy(p)

        convergence_crit = 99.
        method = 'bisection'

        if not np.any(p.params):
            alpha_max = 1
        else:
            alpha_max = 1 / np.max(p.params)

        alpha_min = -alpha_max
        conv_params = []

        for k in range(1, n_iter):
            t0_ls_iter = time.time()

            if k == 1:
                write_to_file('Beginning linear search')
                write_to_file(f"LS Direction: {direction}")
                hdu = fits.PrimaryHDU(direction)
                hdu.writeto(test_image_dir + 'LSdirection.fits', overwrite=True)
                write_to_file(f"Initial params: {p.params}")
                write_to_file(f"Initial epsilon: {best_epsilon}")

            if k == n_iter - 1:
                write_to_file(
                    'WARNING: Linear search did not converge!! This is going to break because best_p is not assigned.')

            if convergence_crit < 0.001:
                alpha_test = alpha_min - (
                            d_cost_min * (alpha_max - alpha_min) / (d_cost_max - d_cost_min))  # secant update
                write_to_file(f"Secant update: alpha_test={alpha_test}")
                method = 'secant'
                if np.isnan(alpha_test):
                    write_to_file('Secant update fail-- bisecting instead')
                    alpha_test = .5 * (alpha_min + alpha_max)  # bisection update
                    write_to_file(f"Bisection update: alpha_test={alpha_test}")
                    method = 'bisection'
            else:
                alpha_test = .5 * (alpha_min + alpha_max)  # bisection update
                write_to_file(f"Bisection update: alpha_test={alpha_test}")

            working_params = p.params + alpha_test * direction
            working_p.params = working_params

            working_epsilon, working_psi = cost_function(working_p, f)
            working_resids = residual_function(working_psi, f_prime)
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
            write_to_file(f"Time spent in this LS iteration: {(time.time() - t0_ls_iter) / 60} minutes.")

            if working_epsilon < best_epsilon:
                best_epsilon = working_epsilon
                best_p = copy.deepcopy(working_p)
                best_psi = working_psi

            if np.abs(d_cost) < tol:
                write_to_file(f"Linear search convergence via |d_cost|< {tol} in {k} iterations")
                hdu = fits.PrimaryHDU(best_p.params)
                hdu.writeto(test_image_dir + 'best_p.fits', overwrite=True)
                hdu = fits.PrimaryHDU(np.array(conv_params))
                hdu.writeto(test_image_dir + 'conv_params.fits', overwrite=True)
                return best_p, best_psi

            if convergence_crit < (0.01 / current_norm):
                write_to_file(f"Linear search convergence via crit<{0.01 / current_norm} in {k} iterations")
                hdu = fits.PrimaryHDU(best_p.params)
                hdu.writeto(test_image_dir + 'best_p.fits', overwrite=True)
                hdu = fits.PrimaryHDU(np.array(conv_params))
                hdu.writeto(test_image_dir + 'conv_params.fits', overwrite=True)
                return best_p, best_psi

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

    def conjugate_gradient(p, f, f_prime, method, tol=1e-5, max_iter=100):
        """
        Algorithm to use conjugate gradient descent to optimize the parameters for destriping.
        Direction is updated using Fletcher-Reeves method
        :param p: parameters object, containing initial parameters guess
        :param f: function, functional form to use for cost function
        :param f_prime: function, the derivative of f. KL: eventually f should dictate f prime
        :param method: str, the method to use for CG direction update.
                Current Options: 'FR', 'PR', 'HS', 'DY' (Fletcher-Reeves, Polak-Ribiere, Hestenes-Stiefel, Dai-Yuan)
        :param tol: float, the value of the norm at which we say CG has converged
        :param max_iter: int, number of iterations at which to force CG to stop
        :return p: params object, the best fit parameters for destriping the SCA images
        """
        write_to_file('Starting conjugate gradient optimization')

        # Initialize variables
        grad_prev = None  # No previous gradient initially
        direction = None  # No initial direction

        write_to_file('Starting initial cost function')
        global test_image_dir
        test_image_dir = 'LS_test_images/' + str(0) + '/'
        psi = cost_function(p, f)[1]
        sys.stdout.flush()

        for i in range(max_iter):
            write_to_file(f"CG Iteration: {i + 1}")
            if not os.path.exists('test_images/' + str(i + 1)):
                os.makedirs('test_images/' + str(i + 1))
            test_image_dir = 'test_images/' + str(i + 1) + '/'
            t_start_CG_iter = time.time()

            # Compute the gradient
            grad, gr_term1, gr_term2 = residual_function(psi, f_prime, extrareturn=True)
            # if i==0:
            #    hdu_ = fits.PrimaryHDU(np.stack((grad,gr_term1,gr_term2)))
            #    hdu_.writeto('grterms.fits', overwrite=True)
            #    del hdu_
            del gr_term1, gr_term2
            write_to_file(f"Minutes spent in residual function: {(time.time() - t_start_CG_iter) / 60}")
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
            else:
                # Calculate beta (direction scaling) depending on method
                if method=='FR': beta = np.sum(np.square(grad)) / np.sum(np.square(grad_prev))
                elif method=='PR': beta = max(0,np.sum(grad * (grad - grad_prev)) / (np.sum(np.square(grad_prev))))
                elif method== 'HS': beta = (np.sum(grad * (grad - grad_prev)) /
                                            np.sum(-direction_prev * (grad - grad_prev)))
                elif method== 'DY': beta = np.sum(np.square(grad)) / np.sum(-direction_prev * (grad - grad_prev))
                else: raise ValueError(f"Unknown method: {method}")

                write_to_file(f"Current Beta: {beta}")

                direction = -grad + beta * direction_prev

            if current_norm < tol:
                write_to_file(f"Convergence reached at iteration: {i + 1} via norm {current_norm} < tol {tol}")
                break

            # Perform linear search
            t_start_LS = time.time()
            write_to_file(f"Initiating linear search in direction: {direction}")
            p_new, psi_new = linear_search(p, direction, f, f_prime)
            write_to_file(f'Total time spent in linear search: {(time.time() - t_start_LS) / 60}')
            write_to_file(
                f'Current norm: {current_norm}, Tol * Norm_0: {tol}, Difference (CN-TOL): {current_norm - tol}')
            write_to_file(f'Current best params: {p_new.params}')

            # Update to current values
            p = p_new
            psi = psi_new
            grad_prev = grad
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
    p = conjugate_gradient(p0, Cost_models(cost_model).f, Cost_models(cost_model).f_prime)
    hdu = fits.PrimaryHDU(p.params)
    hdu.writeto(tempfile + 'final_params.fits', overwrite=True)
    print(tempfile + 'final_params.fits created \n')

    for i, sca in enumerate(all_scas):
        obsid, scaid = get_ids(sca)
        this_sca = Sca_img(obsid, scaid)
        this_param_set = p.forward_par(i)
        ds_image = this_sca.image - this_param_set

        header = this_sca.w
        hdu = fits.PrimaryHDU(ds_image, header=header)
        hdu.writeto(tempfile + filter_ + '_DS_' + obsid + scaid + '.fits', overwrite=True)

    write_to_file(f'Destriped images saved to {tempfile + filter_} _DS_*.fits')
    write_to_file(f'Total hours elapsed: {(time.time() - t0) / 3600}')


if __name__ == '__main__':
    main()
