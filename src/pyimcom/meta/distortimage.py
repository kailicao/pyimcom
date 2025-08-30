"""
Metadetection mosaic class and associated functions.

Classes
-------
MetaMosaic
    A (sub-)mosaic that can be sheared.

Functions
---------
shearimage_to_fits
    Writes a sheared image to a FITS file.

"""

import numpy as np
from astropy import wcs
from astropy.io import fits

from ..compress.compressutils import ReadFile
from ..config import Config, Settings
from ..diagnostics.outimage_utils.helper import HDU_to_bels
from . import ginterp


class MetaMosaic:
    """
    Contains mosaic information for use in meta operations.

    Parameters
    ----------
    fname : str
        File name of block to build (sub-mosaic will be a 3x3 block region centered on this).
    bbox : list of int, optional
        Boundaries [xmin, xmax, ymin, ymax]: accept only blocks with xmin<=x<xmax, ymin<=y<ymax.
        Default is to read the whole mosaic.
        This feature is useful if you haven't run the whole mosaic or pulled it to your disk.
    extpix : int, optional
        Number of pixels to extend beyond the block boundary. Uses full 3x3 submosaic if not given.
    verbose : bool, optional
        Whether to print diagnostics.

    Attributes
    ----------
    cfg : pyimcom.config.Config
        PyIMCOM configuration used in this file.
    Nside : int
        Side length of moasic image.
    in_image : np.array
        3D array; the input image cube.
    in_fidelity : np.array of float
        2D fidelity map
    in_noise : np.array of float
        2D noise map
    in_mask : np.array of bool
        2D mask
    wcs : astropy.wcs.WCS
        The WCS for the mosaic.

    Methods
    -------
    __init__
        Constructor.
    maskpix
        Masks an additional set of pixels.
    mask_fidelity_cut
        Implements a mask based on PyIMCOM fidelity.
    mask_noise_cut
        Implements a mask based on PyIMCOM noise.
    mask_caps
        Implements a mask based on a catalog of circles.
    in_mask
        Boolean mask for input data (False = OK, True = masked).
    to_file
        Writes the mosaic object to a FITS file.
    shearimage
        Generate a sheared image.
    origimage
        Same format as `shearimage`, but doesn't do shear/reconvolution (just extracts subarrays).

    """

    def __init__(self, fname, bbox=None, extpix=None, verbose=False):
        with ReadFile(fname) as f:
            c = f["CONFIG"].data["text"]
            n = len(c)
            cf = ""
            for j in range(n):
                cf += c[j] + "\n"
            self.nlayer = np.shape(f["PRIMARY"].data)[-3]
            self.im_dtype = f["PRIMARY"].data.dtype
        self.cfg = Config(cf)

        # set bounding box for which blocks we will use
        if bbox is None:
            xmin_ = ymin_ = 0
            xmax_ = ymax_ = self.cfg.nblock
        else:
            xmin_ = bbox[0]
            xmax_ = bbox[1]
            ymin_ = bbox[2]
            ymax_ = bbox[3]

        # get the file coordinates
        self.LegacyName = False
        self.cprfitsgz = False
        self.stem = fname[:-11]
        tail = fname[-11:]
        if fname[-9:] == "_map.fits":
            self.LegacyName = True
            self.stem = fname[:-15]
            tail = fname[-15:]
        if fname[-12:] == ".cpr.fits.gz":
            self.cprfitsgz = True
            self.stem = fname[:-18]
            tail = fname[-18:]
        self.ix = int(tail[1:3])
        self.iy = int(tail[4:6])

        # how many pixels to truncate around all the edges of the 3x3 region?
        if extpix is None:
            self.trunc = 0
        else:
            self.trunc = max(self.cfg.n1 * self.cfg.n2 - extpix, 0)

        # build maps
        self.Nside = 3 * self.cfg.n1 * self.cfg.n2 - 2 * self.trunc
        self.in_image = np.zeros((self.nlayer, self.Nside, self.Nside), dtype=self.im_dtype)
        self.in_fidelity = np.zeros((self.Nside, self.Nside), dtype=np.float32)
        self.in_noise = np.zeros((self.Nside, self.Nside), dtype=np.float32)
        self.in_mask = np.zeros((self.Nside, self.Nside), dtype=bool)

        # Load the data. A lot of logic in here to pull out which portions we need
        # (including handling of boundary effects)
        xpad = [self.ix == 0, self.ix == self.cfg.nblock - 1]
        ypad = [self.iy == 0, self.iy == self.cfg.nblock - 1]
        for dx in range(-1, 2):
            # lower-left corner in block is (cx,cy) in the mosaic
            cx = self.cfg.n1 * self.cfg.n2 * (1 + dx) - self.cfg.postage_pad * self.cfg.n2 - self.trunc
            sxmin = self.cfg.postage_pad * self.cfg.n2
            sxmax = sxmin + self.cfg.n1 * self.cfg.n2
            if xpad[0]:
                sxmin -= self.cfg.postage_pad * self.cfg.n2
            if xpad[1]:
                sxmax += self.cfg.postage_pad * self.cfg.n2
            if cx + sxmin < 0:
                sxmin = -cx
            if cx + sxmax > self.Nside:
                sxmax = self.Nside - cx
            for dy in range(-1, 2):
                cy = self.cfg.n1 * self.cfg.n2 * (1 + dy) - self.cfg.postage_pad * self.cfg.n2 - self.trunc
                symin = self.cfg.postage_pad * self.cfg.n2
                symax = symin + self.cfg.n1 * self.cfg.n2
                if ypad[0]:
                    symin -= self.cfg.postage_pad * self.cfg.n2
                if ypad[1]:
                    symax += self.cfg.postage_pad * self.cfg.n2
                if cy + symin < 0:
                    symin = -cy
                if cy + symax > self.Nside:
                    symax = self.Nside - cy

                # Now get this input image if it is within the mosaic
                in_x = self.ix + dx
                in_y = self.iy + dy
                if in_x >= xmin_ and in_x < xmax_ and in_y >= ymin_ and in_y < ymax_:
                    in_fname = self.stem + f"_{in_x:02d}_{in_y:02d}"
                    if self.LegacyName:
                        in_fname += "_map"
                    if self.cprfitsgz:
                        in_fname += ".cpr"
                    in_fname += ".fits"
                    if self.cprfitsgz:
                        in_fname += ".gz"
                    if verbose:
                        print(
                            f"IN {in_x:2d},{in_y:2d} [{symin:4d}:{symax:4d},{sxmin:4d}:{sxmax:4d}] "
                            f"offset x={cx:5d} y={cy:5d}"
                        )
                        print("  <<", in_fname)

                    with ReadFile(in_fname) as f:
                        # the map
                        self.in_image[:, symin + cy : symax + cy, sxmin + cx : sxmax + cx] = f[
                            "PRIMARY"
                        ].data[0, :, symin:symax, sxmin:sxmax]
                        # fidelity, converted to dB
                        self.in_fidelity[symin + cy : symax + cy, sxmin + cx : sxmax + cx] = (
                            f["FIDELITY"].data[0, symin:symax, sxmin:sxmax].astype(np.float32)
                            * HDU_to_bels(f["FIDELITY"])
                            / 0.1
                        )
                        # noise, converted to dB
                        self.in_noise[symin + cy : symax + cy, sxmin + cx : sxmax + cx] = (
                            f["SIGMA"].data[0, symin:symax, sxmin:sxmax].astype(np.float32)
                            * HDU_to_bels(f["SIGMA"])
                            / 0.1
                        )

        # mask pixels that weren't covered at all
        self.in_mask |= self.in_fidelity == 0

        # generate the WCS
        self.wcs = wcs.WCS(naxis=2)
        self.wcs.wcs.crpix = [
            0.5 - self.cfg.Nside * (self.ix - 1 - self.cfg.nblock // 2) - self.trunc,
            0.5 - self.cfg.Nside * (self.iy - 1 - self.cfg.nblock // 2) - self.trunc,
        ]
        self.wcs.wcs.cdelt = [-self.cfg.dtheta, self.cfg.dtheta]
        self.wcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        self.wcs.wcs.crval = [self.cfg.ra, self.cfg.dec]
        self.wcs.wcs.lonpole = self.cfg.lonpole

    def maskpix(self, extramask):
        """
        Provide additional masking beyond the default.

        Pixels that are True in `extramask` are masked out.

        Parameters
        ----------
        extramask : np.array of bool
            2D array of pixels to mask out.

        Returns
        -------
        None

        """

        self.in_mask = np.logical_or(extramask, self.in_mask)

    def mask_fidelity_cut(self, fidelitymin):
        """
        Masks pixels with fidelity below the cut.

        For example, ``mosaic.mask_fidelity_cut(40)`` will cut pixels with leakage worse than 40 dB
        (i.e. U/C>1e-4).

        Parameters
        ----------
        fidelitymin : float
            The fidelity cut in dB.

        Returns
        -------
        None

        """

        self.in_mask = np.logical_or(self.in_fidelity < fidelitymin, self.in_mask)

    def mask_noise_cut(self, noisemax):
        """
        Masks pixels with noise metric worse than the cut.

        For example, ``mosaic.mask_noise_cut(3)`` will cut pixels whose noise suppression is less than 3 dB
        relative to input (i.e., Sigma>10**(-0.3)).

        Parameters
        ----------
        noisemax : float
            The noise cut in dB.

        Returns
        -------
        None

        """

        self.in_mask = np.logical_or(self.in_noise < noisemax, self.in_mask)

    def mask_caps(self, ra, dec, radius):
        """
        Masks circular regions around an input catalog.

        Each of `ra`, `dec`, and `radius` should be an array-like object of the same length N
        (`radius` is permitted to be a scalar).

        Parameters
        ----------
        ra : np.array of float
            The right ascensions of the objects to mask (in degrees).
        dec : np.array of float
            The declinations of the objects to mask (in degrees).
        radius : float or np.array of float
            The radius to mask around each object (in degrees). If a scalar value is given,
            uses the same radius for all objects.

        Returns
        -------
        None

        """

        degree = np.pi / 180.0  # unit

        # convert inputs
        ra = np.array(ra).ravel().astype(np.float64)
        dec = np.array(dec).ravel().astype(np.float64)
        radius = np.array(radius).ravel().astype(np.float64)
        if len(radius) == 1:
            radius = np.zeros_like(ra) + radius[0]

        # select inputs near block center
        ns = np.shape(self.in_mask)[-1]
        ra_ctr, dec_ctr = self.wcs.all_pix2world((ns - 1) / 2, (ns - 1) / 2, 0)
        dx = np.cos(dec * degree) * np.cos((ra - ra_ctr) * degree) - np.cos(dec_ctr * degree)
        dy = np.cos(dec * degree) * np.sin((ra - ra_ctr) * degree)
        dz = np.sin(dec * degree) - np.sin(dec_ctr * degree)
        sep = (
            np.sqrt(dx**2 + dy**2 + dz**2) / degree
        )  # in degrees --- only accurate in mosaic, but that's where we care
        del dx, dy, dz
        searchrad = 0.75 * ns * self.cfg.dtheta + radius
        consider_stars = np.nonzero(sep <= searchrad)

        # shorten the arrays to only consider stars close enough to be masked
        ra = ra[consider_stars]
        dec = dec[consider_stars]
        radius = radius[consider_stars]
        N = len(ra)

        # get positions on the plane in 0-index coordinates
        pixcrd2 = self.wcs.all_world2pix(np.vstack((ra, dec)).T, 0)
        x_ = pixcrd2[:, 0]
        y_ = pixcrd2[:, 1]
        r_ = radius / self.cfg.dtheta

        # now loop over objects. make a rectangular cutout and mask the circle within it
        for j in range(N):
            xmin = max(int(np.floor(x_[j] - r_[j])), 0)
            xmax = min(int(np.ceil(x_[j] + r_[j])) + 1, ns)
            ymin = max(int(np.floor(y_[j] - r_[j])), 0)
            ymax = min(int(np.ceil(y_[j] + r_[j])) + 1, ns)
            # print((x_[j], y_[j]), r_[j], [xmin,xmax,ymin,ymax])
            if xmax <= xmin or ymax <= ymin:
                continue
            erx, ery = np.meshgrid(
                np.linspace(xmin, xmax - 1, xmax - xmin) - x_[j],
                np.linspace(ymin, ymax - 1, ymax - ymin) - y_[j],
            )
            self.in_mask[ymin:ymax, xmin:xmax] |= np.hypot(erx, ery) < r_[j]

    def to_file(self, fname):
        """
        Writes the input mosaic images to a FITS file.

        Parameters
        ----------
        fname : str
            File name for the output FITS file.

        Returns
        -------
        None

        """

        # generate the WCS
        outwcs = wcs.WCS(naxis=2)
        outwcs.wcs.crpix = [
            0.5 - self.cfg.Nside * (self.ix - 1 - self.cfg.nblock // 2),
            0.5 - self.cfg.Nside * (self.iy - 1 - self.cfg.nblock // 2),
        ]
        outwcs.wcs.cdelt = [-self.cfg.dtheta, self.cfg.dtheta]
        outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        outwcs.wcs.crval = [self.cfg.ra, self.cfg.dec]
        outwcs.wcs.lonpole = self.cfg.lonpole

        # make the HDUs
        primary = fits.PrimaryHDU(self.in_image, header=self.wcs.to_header())
        fidelity = fits.ImageHDU(self.in_fidelity, header=self.wcs.to_header())
        noise = fits.ImageHDU(self.in_noise, header=self.wcs.to_header())
        mask = fits.ImageHDU(self.in_mask.astype(np.uint8), header=self.wcs.to_header())

        primary.header["SOURCE"] = "pyimcom.meta.distortimage.MetaMosaic.to_file"
        primary.header["IMTYPE"] = "3x3 block, undistorted"
        fidelity.header["UNIT"] = "dB"
        noise.header["UNIT"] = "dB"

        fits.HDUList([primary, fidelity, noise, mask]).writeto(fname, overwrite=True)

    def shearimage(
        self,
        N,
        jac=None,
        psfgrow=1.0,
        oversamp=1.0,
        fidelity_min=None,
        Rsearch=6.0,
        select_layers=None,
        verbose=False,
    ):
        """
        Generates a sheared image and its WCS.

        Parameters
        ----------
        N : int
            Size of the output image (shape will be (`N`, `N`)).
        jac : np.array or None, optional
            2x2 Jacobian for transformation (None defaults to the identity).
        psfgrow : float, optional
            Factor (in linear scale) by which to grow the PSF.
        oversamp : float, optional
            Up-sampling factor (e.g., 1 = preserve pixel scale).
        fidelity_min : float or None, optional
            Fidelity cut (in dB) for which pixels to use.
        Rsearch : float, optional
            Search radius in interpolation, in units of coadded pixels.
        select_layers : np.array of int or None, optional
            If given, only process these layers.
        stest : int, optional
            Computes test diagnostics (leakage & noise for interpolation) every this
            many output pixels. Set to 1 to do every output pixel (but this may be slower).
        verbose : bool, optional
            Print diagnostics to terminal.

        Returns
        -------
        im : dict
            Image dictionary containing 'image', 'mask', 'wcs', 'pars', 'layers', 'psf_fwhm', and 'ref'.

        Notes
        -----
        The output image contains:
        *   ``im['image']`` : np.array, image cube (3D)
        *   ``im['mask']`` : np.array, image mask (2D, Boolean, True=masked)
        *   ``im['wcs']`` : astropy.wcs.WCS, world coordinate system object (appropriate for a FITS file)
        *   ``im['pars']`` : dict, parameter dictionary (can be turned into a FITS header)
        *   ``im['layers']`` : list of str, names of layers
        *   ``im['psf_fwhm']`` : float, 1 sigma width of output PSF in arcsec
        *   ``im['ref']`` : (int, int), projection center (x,y), 0-offset convention

        The sense of `jac` is that the *output* is related to the *input* by:
        d{input coords[i]} = sum_j jac[i,j] d{output coords[j]}

        Assumes a Gaussian PSF, returns an error if something else is used.

        """

        # check PSF type
        if self.cfg.outpsf != "GAUSSIAN":
            raise ValueError("shearimage: only works on GAUSSIAN, received " + self.cfg.outpsf)

        # Figure out the geometrical mapping. First the scale:
        J = np.identity(2) if jac is None else np.asarray(jac)
        J_orig = np.copy(J)
        J = J / oversamp
        scale = self.cfg.dtheta
        # ... and now the projection center in block coordinates (Q)
        Q_orig = (
            np.asarray([self.cfg.nblock / 2 - self.ix - 0.5, self.cfg.nblock / 2 - self.iy - 0.5])
            * self.cfg.n1
            * self.cfg.n2
        )
        Q_new = np.linalg.solve(J, Q_orig)
        xref = np.round(Q_new[0] + 1e-7) + 0.5 + N / 2  # rounds to nearest half integer for even N
        yref = np.round(Q_new[1] + 1e-7) + 0.5 + N / 2

        # origin position in the input array
        opos = J @ np.asarray([1 - xref, 1 - yref])  # recall the lower-left corner is (1,1) in FITS
        opos[0] += (self.cfg.nblock / 2 - self.ix + 1) * self.cfg.n1 * self.cfg.n2 - 0.5 - self.trunc
        opos[1] += (self.cfg.nblock / 2 - self.iy + 1) * self.cfg.n1 * self.cfg.n2 - 0.5 - self.trunc
        # the "-0.5" is because the lower-left corner of the pixel is at (-.5,-.5)
        # and have the +1 since the lower-left corner of the image is in the block (ix-1,iy-1)

        # generate the WCS
        outwcs = wcs.WCS(
            {
                "CTYPE1": "RA---STG",
                "CUNIT1": "deg",
                "CRPIX1": xref,
                "NAXIS1": N,
                "CTYPE2": "DEC--STG",
                "CUNIT2": "deg",
                "CRPIX2": yref,
                "NAXIS2": N,
                "CD1_1": -J[0, 0] * scale,
                "CD1_2": -J[0, 1] * scale,
                "CD2_1": J[1, 0] * scale,
                "CD2_2": J[1, 1] * scale,
                "CRVAL1": self.cfg.ra,
                "CRVAL2": self.cfg.dec,
                "LONPOLE": self.cfg.lonpole,
            }
        )

        # input mask
        if fidelity_min is not None:
            inmask = np.logical_or(self.in_fidelity < fidelity_min, self.in_mask)
        else:
            inmask = self.in_mask

        # get smearing information
        sigma = self.cfg.sigmatarget * Settings.pixscale_native * (180.0 / np.pi) / self.cfg.dtheta
        # recall: pixscale_native is in radians, but dtheta is in degrees, hence the conversion
        dCov = sigma**2 * (psfgrow**2 * J_orig @ J_orig.T - np.identity(2))
        C = [dCov[0, 0], dCov[0, 1], dCov[1, 1]]

        if verbose:
            print("Q_orig", Q_orig)
            print("Q_new", Q_new)
            print("opos", opos)
            print(
                "sigmatarget",
                self.cfg.sigmatarget,
                "dtheta",
                self.cfg.dtheta,
                "pixscale_native",
                Settings.pixscale_native,
            )
            print("sigma", sigma)
            print("C", C)

        # layer selection
        ul = np.arange(np.shape(self.in_image)[0]).astype(np.int64)
        if select_layers is not None:
            ul = np.array(select_layers).astype(np.int64)
        layerlist = []
        for i in range(len(ul)):
            layerlist.append(self.cfg.extrainput[ul[i]])

        image, mask, Umax, Smax = ginterp.MultiInterp(
            self.in_image[ul, :, :], inmask, (N, N), opos, J, Rsearch, sigma * np.sqrt(8 * np.log(2)), C
        )
        # could add kappa, deweight

        # SVD of the Jacobian
        z = J_orig[0, 0] + J_orig[1, 1] + 1j * (J_orig[1, 0] - J_orig[0, 1])
        cpd = np.abs(z)
        apx = np.angle(z)
        z = J_orig[0, 0] - J_orig[1, 1] + 1j * (J_orig[1, 0] + J_orig[0, 1])
        cmd = np.abs(z)
        amx = np.angle(z)
        Eig1 = (cpd + cmd) / 2.0
        Eig2 = (cpd - cmd) / 2.0
        alpha = (apx + amx) / 2.0
        mu = 1.0 / (Eig1 * Eig2)
        eta = -np.log(Eig1 / Eig2)
        eta1 = eta * np.cos(2 * alpha)
        eta2 = eta * np.sin(2 * alpha)
        g1 = np.tanh(eta / 2.0) * np.cos(2 * alpha)
        g2 = np.tanh(eta / 2.0) * np.sin(2 * alpha)
        conv = 1.0 - (Eig1 + Eig2) / 2.0

        pardict = {
            "STEM": (self.stem, "stem for file name"),
            "BLOCKX": (self.ix, "x block index"),
            "BLOCKY": (self.iy, "y block index"),
            "UMAX": (Umax, "interp - max leakage (square norm)"),
            "SMAX": (Smax, "interp - max noise (square norm)"),
            "JXX": (J_orig[0, 0], "Jacobian x_in, x_out"),
            "JXY": (J_orig[0, 1], "Jacobian x_in, y_out"),
            "JYX": (J_orig[1, 0], "Jacobian y_in, x_out"),
            "JYY": (J_orig[1, 1], "Jacobian y_in, y_out"),
            "COVXX": (C[0], "smoothing covariance xx"),
            "COVXY": (C[1], "smoothing covariance xy"),
            "COVYY": (C[2], "smoothing covariance yy"),
            "SIGMAOUT": (
                self.cfg.sigmatarget * Settings.pixscale_native * (180.0 / np.pi) * 3600 * psfgrow,
                "arcsec",
            ),
            "PIXSCALE": (self.cfg.dtheta * 3600 / oversamp, "arcsec"),
            "OVERSAMP": (oversamp, "oversampling implemented in shearimage"),
            "MU": (mu, "amplification applied"),
            "ETA1": (eta1, "shear component 1"),
            "ETA2": (eta2, "shear component 2"),
            "JROTATE": (apx, "rotation angle, CCW in-->out, radians"),
            "G1": (g1, "reduced shear component 1"),
            "G2": (g2, "reduced shear component 2"),
            "CONV": (conv, "convergence kappa"),
        }

        return {
            "image": image,
            "mask": mask,
            "wcs": outwcs,
            "pars": pardict,
            "layers": layerlist,
            "psf_fwhm": np.sqrt(8.0 * np.log(2)) * pardict["SIGMAOUT"][0],
            "ref": (xref - 1, yref - 1),
        }

    def origimage(self, N, select_layers=None):
        """
        Like shearimage, but without applying the deconvolution/shear/reconvolution (so is faster).

        Parameters
        ----------
        N : int
            Size of the output image (shape will be (`N`, `N`)).
        select_layers : np.array of int or None, optional
            If given, only process these layers.

        Returns
        -------
        im : dict
            Image dictionary

        See Also
        --------
        shearimage : Carries out a shear, same output format.

        """

        # check PSF type
        if self.cfg.outpsf != "GAUSSIAN":
            raise ValueError("shearimage: only works on GAUSSIAN, received " + self.cfg.outpsf)

        # parity check:
        # dshift = 0 --> image is *centered* on the block
        # dshift = 1 --> image is centered (-0.5,-0.5) pixels to the lower-left of the block center
        dshift = (self.cfg.n1 * self.cfg.n2 + N) % 2

        # Figure out the geometrical mapping.
        scale = self.cfg.dtheta
        xref = (self.cfg.nblock / 2 - self.ix - 0.5) * self.cfg.n1 * self.cfg.n2 + (N + dshift + 1) / 2
        yref = (self.cfg.nblock / 2 - self.iy - 0.5) * self.cfg.n1 * self.cfg.n2 + (N + dshift + 1) / 2

        # offset of output array from imput array coordinates
        opos_ = (3 * self.cfg.n1 * self.cfg.n2 - N - dshift) // 2 - self.trunc

        # generate the WCS
        outwcs = wcs.WCS(
            {
                "CTYPE1": "RA---STG",
                "CUNIT1": "deg",
                "CRPIX1": xref,
                "NAXIS1": N,
                "CTYPE2": "DEC--STG",
                "CUNIT2": "deg",
                "CRPIX2": yref,
                "NAXIS2": N,
                "CD1_1": -scale,
                "CD1_2": 0.0,
                "CD2_1": 0.0,
                "CD2_2": scale,
                "CRVAL1": self.cfg.ra,
                "CRVAL2": self.cfg.dec,
                "LONPOLE": self.cfg.lonpole,
            }
        )

        # layer selection
        ul = np.arange(np.shape(self.in_image)[0]).astype(np.int64)
        if select_layers is not None:
            ul = np.array(select_layers).astype(np.int64)
        layerlist = []
        for i in range(len(ul)):
            layerlist.append(self.cfg.extrainput[ul[i]])

        # this one is just a subarray
        if opos_ < 0 or opos_ + N > 3 * self.cfg.n1 * self.cfg.n2:
            raise ValueError("Requested image size too large (overfills 3 blocks)")
        image = self.in_image[ul, opos_ : opos_ + N, opos_ : opos_ + N]
        mask = np.copy(self.in_mask[opos_ : opos_ + N, opos_ : opos_ + N])

        pardict = {
            "STEM": (self.stem, "stem for file name"),
            "BLOCKX": (self.ix, "x block index"),
            "BLOCKY": (self.iy, "y block index"),
            "UMAX": (0.0, "interp - max leakage (square norm)"),
            "SMAX": (1.0, "interp - max noise (square norm)"),
            "JXX": (1.0, "Jacobian x_in, x_out"),
            "JXY": (0.0, "Jacobian x_in, y_out"),
            "JYX": (0.0, "Jacobian y_in, x_out"),
            "JYY": (1.0, "Jacobian y_in, y_out"),
            "COVXX": (0.0, "smoothing covariance xx"),
            "COVXY": (0.0, "smoothing covariance xy"),
            "COVYY": (0.0, "smoothing covariance yy"),
            "SIGMAOUT": (self.cfg.sigmatarget * Settings.pixscale_native * (180.0 / np.pi) * 3600, "arcsec"),
            "PIXSCALE": (self.cfg.dtheta * 3600, "arcsec"),
            "OVERSAMP": (1.0, "oversampling implemented in shearimage"),
            "MU": (1.0, "amplification applied"),
            "ETA1": (0.0, "shear component 1"),
            "ETA2": (0.0, "shear component 2"),
            "JROTATE": (0.0, "rotation angle, CCW in-->out, radians"),
            "G1": (0.0, "reduced shear component 1"),
            "G2": (0.0, "reduced shear component 2"),
            "CONV": (0.0, "convergence kappa"),
        }

        return {
            "image": image,
            "mask": mask,
            "wcs": outwcs,
            "pars": pardict,
            "layers": layerlist,
            "psf_fwhm": np.sqrt(8.0 * np.log(2)) * pardict["SIGMAOUT"][0],
            "ref": (xref - 1, yref - 1),
        }


def shearimage_to_fits(im, fname, layers=None, overwrite=False):
    """
    Utility to save a shearimage dictionary a FITS file.

    Parameters
    ----------
    im : dict
        Image dictionary from MetaMosaic.shearimage or MetaMosaic.origimage.
    fname : str
        File name for output.
    layers : np.array of int, optional
        Which layers to include.
    overwrite : bool, optional
        Whether to overwrite an existing file.

    Returns
    -------
    None

    """

    # which layers to use?
    nlayer = np.shape(im["image"])[-3]
    use_layers = layers
    if layers is None:
        use_layers = range(nlayer)
    use_layers = np.asarray(use_layers).astype(np.int16)

    H1 = fits.PrimaryHDU(im["image"][use_layers, :, :], header=im["wcs"].to_header())
    H1.header["SOURCE"] = "pyimcom.meta.distortimage.shearimage_to_fits"
    for p in im["pars"]:
        H1.header[p] = im["pars"][p]
    for q in range(len(use_layers)):
        qst = f"LAYER{q + 1:03d}"
        st = im["layers"][q]
        if st is None:
            st = "__SCI__"
        H1.header[qst] = (st, f"layer {use_layers[q] + 1:3d} here, was {q + 1:3d} in original")
    H2 = fits.ImageHDU(im["mask"].astype(np.uint8))
    fits.HDUList([H1, H2]).writeto(fname, overwrite=overwrite)
