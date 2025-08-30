"""
This file contains the PyIMCOM_WCS class, which allows PyIMCOM to take in a WCS either as a
FITS header or a GWCS from an ASDF file.

PyIMCOM_WCS supports the all_pix2world and all_world2pix methods, which are like their astropy
counterparts. A separate local_partial_pixel_derivatives2 function was written so that we don't
need to rely on astropy's partial derivative function (which has singular behavior near the poles).

Classes
-------
ABasis
    Base class for basis polynomials.
SimpleBaiss
    Simple polynomials.
LocWCS
    Internal version of gwcs plus approximations.
PyIMCOM_WCS
    Main class for PyIMCOM's WCS, intended to support functions in PyIMCOM that were
    originally written with astropy.wcs.WCS but now also have to accept gwcs inputs.

Functions
---------
local_partial_pixel_derivatives2
    Jacobian for WCS (2-sided derivative, designed to also work near poles).
_stand_alone_test
    Unit test function.

"""

import sys
from copy import deepcopy

import asdf
import astropy.wcs
import gwcs
import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

from .config import Settings

### === UTILITIES FOR APPROXIMATING A GWCS ===


class ABasis:
    """Base class for 2D basis polynomials. These are defined in the range -1<u<1 and -1<v<1.

    Want to use inherited classes that replace the coef_setup method.

    Parameters
    ----------
    p_order : int
        Max order of the polynomial.
    N : int
        Side length of grid the polynomial is evaluated on.

    Attributes
    ----------
    N_basis : int
        Number of basis modes.
    coefs : np.array
        Coefficient matrix: coefs[i,j,k] is the coefficient of u^i v^j in the kth polynomial.

    Methods
    -------
    eval
        evaluation of polynomials
    """

    def __init__(self, p_order, N):
        self.p_order = p_order
        self.N = N
        self.N_basis = ((p_order + 1) * (p_order + 2)) // 2
        self.coefs = np.zeros((p_order + 1, p_order + 1, self.N_basis))
        self.coef_setup()

    def coef_setup(self):
        """Build the coefficients.

        Shouldn't get here, always use the methods from the inherited classes."""
        pass  # shouldn't get here, always use the methods from the inherited classes

    def eval(self, u, v):
        """Takes in an array of positions, and returns an array of the basis polynomials evaluated at those
        points.

        Parameters
        ----------
        u : np.array
            Array of horizontal positions. Shape (npts,).
        v : np.array
            Array of vertical positions. Shape (npts,).

        Returns
        -------
        np.array
            A 2D array: out[k,l] = kth poly evaluated at lth point. Shape (N_basis, npts).

        Notes
        -----
        Could be replaced with a more stable version for specific types of polynomials if provided by that
        basis.

        """

        out = np.zeros((self.N_basis, np.size(u)))
        for i in range(self.p_order + 1):
            for j in range(self.p_order + 1 - i):
                out[:, :] += np.outer(self.coefs[i, j, :], u**i * v**j)

        return out


class SimpleBasis(ABasis):
    """Simple polynomials (each coefficient is 1). Base class is pyimcom.wcsutil.ABasis."""

    def coef_setup(self):
        """Build the coefficients."""

        self.basis = "simple"
        k = 0
        for i in range(self.p_order + 1):
            for j in range(self.p_order + 1 - i):
                self.coefs[i, j, k] = 1.0
                k += 1


### WCS classes ###


class LocWCS:
    """WCS built from a gwcs that can be reported in various formats.

    Parameters
    ----------
    gwcs : gwcs.WCS
        The generalized WCS.
    N : int, optional
        Side length of the array.

    Attributes
    ----------
    gwcs : gwcs.WCS
        The generalized WCS. (Same as `gwcs`.)
    N : int
        Side length of the array. (Same as `N`.)
    ra_ctr : float
        Right ascension of projection center.
    dec_ctr : float
        Declination of projection center.
    uEast : np.array
        3-component unit vector pointing East.
    uNorth : np.array
        3-component unit vector pointing North.
    J : np.array
        2x2 Jacobian from detector to tangent plane coordinates in radians.
    approx_wcs : astropy.wcs.WCS
        The best-fit TAN-SIP approximation (if set).
    max_wcs_err : float
        The worst error (in pixels) from the approx_wcs (if set).
    errmap : np.array
        The error map (in pixels) from the approx_wcs (if set). Shape (2, `N`, `N`).

    Methods
    -------
    __init__
       Constructor.
    wcs_approx_sip
       Build approximate TAN-SIP WCS.
    _make_errmap
       Build the error map.
    err_interp
        Makes error map interpolator for the TAN-SIP approximation.

    """

    def __init__(self, gwcs, N=4088):
        self.gwcs = gwcs
        self.N = N

        # get the centroid
        # positions: center, left, right, bottom, top
        ra, dec = gwcs(
            ((N - 1) / 2.0, 0, N - 1, (N - 1) / 2.0, (N - 1) / 2.0),
            ((N - 1) / 2.0, (N - 1) / 2.0, (N - 1) / 2.0, 0, N - 1),
        )
        degree = np.pi / 180.0
        x = np.zeros((5, 3))
        x[:, 0] = np.cos(dec * degree) * np.cos(ra * degree)
        x[:, 1] = np.cos(dec * degree) * np.sin(ra * degree)
        x[:, 2] = np.sin(dec * degree)
        self.ra_ctr = ra[0]
        self.dec_ctr = dec[0]
        self.uEast = np.array([-np.sin(ra[0] * degree), np.cos(ra[0] * degree), 0.0])
        self.uNorth = np.array(
            [
                -np.sin(dec[0] * degree) * np.cos(ra[0] * degree),
                -np.sin(dec[0] * degree) * np.sin(ra[0] * degree),
                np.cos(dec[0] * degree),
            ]
        )
        # Jacobian based on the above
        self.J = np.zeros((2, 2))
        self.J[0, 0] = np.dot(self.uEast, x[2, :] - x[1, :]) / (N - 1)
        self.J[0, 1] = np.dot(self.uEast, x[4, :] - x[3, :]) / (N - 1)
        self.J[1, 0] = np.dot(self.uNorth, x[2, :] - x[1, :]) / (N - 1)
        self.J[1, 1] = np.dot(self.uNorth, x[4, :] - x[3, :]) / (N - 1)

        # print(self.ra_ctr, self.dec_ctr)
        # print(self.uEast)
        # print(self.uNorth)
        # print(self.J/degree*3600/.11)

    def wcs_approx_sip(self, p_order=3, nq=100, basis="simple", verbose=False):
        """Generate approximate TAN-SIP polynomial wcs.

        Parameters
        ----------
        p_order : int, optional
            Order of polynomial to fit.
        nq : int, optional
            Grid size for fitting WCS (nq x nq).
        basis : str, optional
            Type of basis to use in fitting the WCS.
        verbose : bool
            Print lots of diagnostics to the terminal.

        Notes
        -----
        The following basis sets for the linear algerbra are available:

        * 'simple' : simple polynomials (could be unstable for high order)

        """

        N = self.N
        xx, yy = np.meshgrid(np.array(range(N)), np.array(range(N)))
        ra, dec = self.gwcs(xx.flatten(), yy.flatten())
        degree = np.pi / 180.0

        # make generic FITS header
        hdu = fits.PrimaryHDU(np.zeros((N, N), dtype=np.int8))
        header = hdu.header
        header["CTYPE1"] = "RA---TAN-SIP"
        header["CTYPE2"] = "DEC--TAN-SIP"
        header["CRVAL1"] = self.ra_ctr
        header["CRVAL2"] = self.dec_ctr
        header["CD1_1"] = self.J[0, 0] / degree
        header["CD1_2"] = self.J[0, 1] / degree
        header["CD2_1"] = self.J[1, 0] / degree
        header["CD2_2"] = self.J[1, 1] / degree
        header["CRPIX1"] = (N + 1) / 2.0
        header["CRPIX2"] = (N + 1) / 2.0
        header["A_ORDER"] = p_order
        header["B_ORDER"] = p_order

        # now we want to get the distortion
        # U = u + f(u,v)
        # V = v + g(u,v)
        # where (U,V) would map to the ideal location with the TAN projection
        # we define dU = U-u and dV = V-v

        # first get all the coordinates
        q = np.linspace(0, N - 1, nq)
        xx, yy = np.meshgrid(q, q)
        ra, dec = self.gwcs(xx.flatten(), yy.flatten())
        u_ = xx.flatten() - (N - 1) / 2.0
        v_ = yy.flatten() - (N - 1) / 2.0
        del xx, yy
        x = np.zeros((nq**2, 3))
        x[:, 0] = np.cos(dec * degree) * np.cos(ra * degree)
        x[:, 1] = np.cos(dec * degree) * np.sin(ra * degree)
        x[:, 2] = np.sin(dec * degree)
        del ra, dec
        dec_ctr = self.dec_ctr
        ra_ctr = self.ra_ctr
        pc = np.array(
            [
                np.cos(dec_ctr * degree) * np.cos(ra_ctr * degree),
                np.cos(dec_ctr * degree) * np.sin(ra_ctr * degree),
                np.sin(dec_ctr * degree),
            ]
        )
        tan_x = (x @ self.uEast) / (x @ pc)
        tan_y = (x @ self.uNorth) / (x @ pc)
        del x
        detJ = self.J[0, 0] * self.J[1, 1] - self.J[0, 1] * self.J[1, 0]
        dU = (self.J[1, 1] * tan_x - self.J[0, 1] * tan_y) / detJ - u_
        dV = (-self.J[1, 0] * tan_x + self.J[0, 0] * tan_y) / detJ - v_

        if verbose:
            print("means and standard deviations of U and V offsets:")
            print(np.mean(dU), np.std(dU))
            print(np.mean(dV), np.std(dV))

        # basis function choices
        if basis.lower() == "simple":
            MyBasis = SimpleBasis(p_order, N)

        # test
        if verbose:
            print("basis coefficients:")
            print(MyBasis.coefs)

        M = MyBasis.eval(u_, v_)  # warning: huge if nq is big! 2 GB for 4088**2 pixels, and 4th order
        af = np.zeros((MyBasis.N_basis,))
        ag = np.zeros((MyBasis.N_basis,))

        # theoretically, this converges in 1 step;
        # but at finite numerical precision, a few steps is better.
        for _ in range(4):
            if verbose:
                print(M @ (dU - M.T @ af))
            af[:] += np.linalg.solve(M @ M.T, M @ (dU - M.T @ af))
            ag[:] += np.linalg.solve(M @ M.T, M @ (dV - M.T @ ag))

        # insert fitting
        for i in range(p_order + 1):
            for j in range(p_order + 1 - i):
                header[f"A_{i:1d}_{j:1d}"] = np.dot(MyBasis.coefs[i, j, :], af)
                header[f"B_{i:1d}_{j:1d}"] = np.dot(MyBasis.coefs[i, j, :], ag)
        self.approx_wcs = astropy.wcs.WCS(header)

        # get the worst error in pixels
        err = np.hypot(dU - M.T @ af, dV - M.T @ ag)
        self.wcs_max_err = np.amax(err)
        # and save the error map
        self._make_errmap()

    def _make_errmap(self):
        """Builds the error map.

        Notes
        -----
        The shape of the error map is (2, N, N).

        The format is:

        * errmap[0,j,i] is the x-offset of pixel (i,j)
        * errmap[1,j,i] is the y-offset.

        The offset is in the sense of if there is a barred coordinate system that is the TAN-SIP approximation
        and unbarred is the true system, then::

          # xbar == x + errmap[0,y,x]
          # ybar == y + errmap[1,y,x]

        """

        if not hasattr(self, "approx_wcs"):
            raise TypeError("Missing approximated WCS")

        N = self.N
        x, y = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, N - 1, N))
        ra, dec = self.gwcs(x.ravel(), y.ravel())
        xbar, ybar = self.approx_wcs.all_world2pix(ra, dec, 0)
        del ra, dec
        d = np.zeros((2, N, N), dtype=np.float32)
        d[0, :, :] = (xbar.reshape((N, N)) - x).astype(np.float32)
        d[1, :, :] = (ybar.reshape((N, N)) - y).astype(np.float32)
        self.errmap = d

    def err_interp(self, a=4, n_pad=2048):
        """Makes interpolators for the delta x = xbar-x and delta y = ybar-y directions.

        The functions returned are of the form dX(arr), where arr is an array of shape (K,2)
        indicating K points. arr[:,0] are the y-values, and arr[:,1] are the x-values.

        Parameters
        ----------
        a : int
            Number of pixels from edge to use for linear extrapolation..
        n_pad : int
            Distance to extrapolate the error map.

        Returns
        -------
        function
            An interpolator function for the delta x error.
        function
            An interpolator function for the delta y error.

        """

        # spacings
        N = self.N
        coords = np.pad(np.linspace(0, N - 1, N), 1)
        d_ = np.pad(self.errmap, ((0, 0), (1, 1), (1, 1)))

        # fill in padding values at n_pad values from the edge of the array
        coords[0] = -n_pad
        coords[-1] = N + n_pad - 1
        d_[:, :, 0] = d_[:, :, 1] + n_pad / a * (d_[:, :, 1] - d_[:, :, 1 + a])
        d_[:, :, -1] = d_[:, :, -2] + n_pad / a * (d_[:, :, -2] - d_[:, :, -2 - a])
        d_[:, 0, :] = d_[:, 1, :] + n_pad / a * (d_[:, 1, :] - d_[:, 1 + a, :])
        d_[:, -1, :] = d_[:, -2, :] + n_pad / a * (d_[:, -2, :] - d_[:, -2 - a, :])

        # and make the interpolator
        return (
            RegularGridInterpolator(
                (coords, coords), d_[0, :, :], method="linear", bounds_error=False, fill_value=None
            ),
            RegularGridInterpolator(
                (coords, coords), d_[1, :, :], method="linear", bounds_error=False, fill_value=None
            ),
        )


### === END UTILITIES ===


class PyIMCOM_WCS:
    """
    Class that has the key methods we depend on from astropy.wcs,
    but can be constructed from other types of WCS information (including gwcs).

    Parameters
    ----------
    inwcs : fits.Header or astropy.wcs.WCS or gwcs.wcs.WCS
        The input WCS.
    noconvert : bool, optional
        Do not internally convert WCS type.

    Methods
    -------
    __init__
       Constructor.
    all_pix2world
       pixel -> world coordinates (astropy-like)
    all_world2pix
       world -> pixel coordinates (astropy-like)

    Attributes
    ----------
    constructortype : str
        What type of input was used to make this object.
    type : str
        What type of method to use in computation (currently ASTROPY or GWCS).
    obj : variable
        The WCS object being wrapped.
    err : (function,function), optional
        For 'ASTROPY+', contains an interpolator for the error. Not used for other types.

    """

    def __init__(self, inwcs, noconvert=False):
        self.array_shape = (Settings.sca_nside, Settings.sca_nside)

        if isinstance(inwcs, fits.Header):
            self.constructortype = "FITSHEADER"
            self.type = "ASTROPY"
            self.obj = astropy.wcs.WCS(inwcs)
            return

        if isinstance(inwcs, astropy.wcs.WCS):
            self.constructortype = "ASTROPY"
            self.type = "ASTROPY"
            self.obj = inwcs
            return

        if isinstance(inwcs, gwcs.wcs.WCS):
            self.constructortype = "GWCS"
            if noconvert:
                self.type = "GWCS"
                self.obj = deepcopy(inwcs)
                self.obj.bounding_box = None  # remove this since for derivatives
                # we need to go off the edge
                return
            # GWCS input, but can convert
            self.type = "ASTROPY+"  # '+' indicates with correction
            w = LocWCS(inwcs, N=Settings.sca_nside)
            w.wcs_approx_sip(p_order=2)
            self.obj = w.approx_wcs
            self.err = w.err_interp(a=8, n_pad=Settings.sca_nside // 2)  # dX,dY
            return

        # get here if nothing above works
        raise TypeError("Unrecognized WCS type.")

    def _all_pix2world(self, pos, origin):
        """
        An astropy-like function to go from pixel to world coordinates.

        Parameters
        ----------
        pos : np.array
            Pixel coordinates, shape (N,2).
        origin: int
            Offset of lower-left pixel, should be 0 or 1.

        Returns
        -------
        np.array
            World coordinates. Shape (N,2).

        """

        if self.type == "ASTROPY":
            return self.obj.all_pix2world(pos, origin)

        if self.type == "ASTROPY+":
            dp = np.vstack((self.err[0](pos[::-1, :]), self.err[1](pos[::-1, :]))).T.astype(np.float64)
            return self.obj.all_pix2world(pos + dp, origin)

        if self.type == "GWCS":
            pos = np.array(pos)
            ra, dec = self.obj.pixel_to_world_values(pos[:, 0] - origin, pos[:, 1] - origin)
            return np.vstack((ra, dec)).T

    def all_pix2world(self, *args):
        """
        An astropy-like function to go from pixel to world coordinates.

        This has both a 2-argument ``(pos, origin)`` or a 3-argument ``(pos, pos2, origin)`` format.

        In 2-argument format, `pos` is a shape (N, 2) array and the return is also a shape (N, 2) array.

        In 3-argument format, `pos` is a shape (N,) array of pixel x, `pos2` is a shape (N,) array of
        pixel y, and the return valus is ra, dec, both shape (N,) arrays. For N=1, you may use scalars.

        In both cases, `origin` is an integer indicating the index of the lower-left pixel (0 or 1).

        Parameters
        ----------
        *args : variable
            See description.

        Returns
        -------
        np.array or np.array, np.array
            World coordinates.

        See Also
        --------
        _all_pix2world : 2-argument format.

        """

        if len(args) == 2:
            return self._all_pix2world(np.array(args[0]), args[1])
        o = self._all_pix2world(np.vstack((args[0], args[1])).T, args[2])
        if isinstance(args[0], np.ndarray):
            return o[:, 0], o[:, 1]
        else:
            return o[0, 0], o[0, 1]

    def _all_world2pix(self, pos, origin):
        """
        An astropy-like function to go from world to pixel coordinates.

        Parameters
        ----------
        pos : np.array
            World coordinates, shape (N,2).
        origin: int
            Offset of lower-left pixel, should be 0 or 1.

        Returns
        -------
        np.array
            Pixel coordinates. Shape (N,2).

        """

        if self.type == "ASTROPY":
            return self.obj.all_world2pix(pos, origin)

        if self.type == "ASTROPY+":
            pos2 = self.obj.all_world2pix(pos, origin)
            pos1 = np.copy(pos2)
            # 3 iterations is overkill but we want to be sure.
            # also we want to avoid slight discontinuities
            for _ in range(3):
                dp = np.vstack((self.err[0](pos1[::-1, :]), self.err[1](pos1[::-1, :]))).T.astype(np.float64)
                pos1 = pos2 - dp
            return pos1

        if self.type == "GWCS":
            pos = np.array(pos)
            x, y = self.obj.world_to_pixel_values(pos[:, 0], pos[:, 1])
            return np.vstack((x, y)).T + origin

    def all_world2pix(self, *args):
        """
        An astropy-like function to go from pixel to world coordinates.

        This has both a 2-argument ``(pos,origin)`` or a 3-argument ``(pos,pos2,origin)`` format.

        In 2-argument format, `pos` is a shape (N, 2) array and the return is also a shape (N, 2) array.

        In 3-argument format, `pos` is a shape (N,) array of ra, `pos2` is a shape (N,) array of
        dec, and the return valus is x, y, both shape (N,) arrays. For N=1, you may use scalars.

        In both cases, `origin` is an integer indicating the index of the lower-left pixel (0 or 1).

        Parameters
        ----------
        *args : variable
            See description.

        Returns
        -------
        np.array or np.array, np.array
            Pixel coordinates.

        See Also
        --------
        _all_world2pix : 2-argument format.

        """

        if len(args) == 2:
            return self._all_world2pix(np.array(args[0]), args[1])
        o = self._all_world2pix(np.vstack((args[0], args[1])).T, args[2])
        if isinstance(args[0], np.ndarray):
            return o[:, 0], o[:, 1]
        else:
            return o[0, 0], o[0, 1]


def local_partial_pixel_derivatives2(inwcs, x, y):
    """Alternative form of the local partial derivatives function
    that is well-behaved near the poles and uses 2-sided derivatives.

    Parameters
    ----------
    inwcs : pyimcom.wcsutil.PyIMCOM_WCS
        The WCS that we are using.
    x : float
        x position in pixels (0 offset)
    y : float
        y position in pixels (0 offset)

    Returns
    -------
    jac : np.array
        2x2 Jacobian matrix, with output 0->West and 1->North

    Notes
    -----
    This is relative to unit vectors, so jac[0,:] is -cos(declination) * d(ra)/d(pix x or y).
    So note this is different from astropy local_partial_pixel_derivatives, which doesn't
    have the factor of -cos(declination).

    """

    # choose grid of positions for the numerical derivative
    x_ = x + np.array([0, 1, -1, 3, -3, 0, 0, 0, 0])
    y_ = y + np.array([0, 0, 0, 0, 0, 1, -1, 3, -3])

    # now get the RA and Dec
    degree = np.pi / 180.0
    pos_world = inwcs.all_pix2world(np.vstack((x_, y_)).T, 0)
    ra_ = pos_world[:, 0] * degree
    dec_ = pos_world[:, 1] * degree

    # convert to "East" and "North" directions
    p = np.zeros((2, np.size(x_)))
    p[0, :] = np.cos(dec_) * np.sin(ra_[0] - ra_)
    p[1, :] = np.sin(dec_) * np.cos(dec_[0]) - np.cos(dec_) * np.sin(dec_[0]) * np.cos(ra_[0] - ra_)

    # now get the Jacobian
    jac = np.zeros((2, 2))
    for j in [0, 1]:
        # uses 4-point derivative formula
        subvec = p[:, 1 + 4 * j : 5 + 4 * j]  # offsets: 1,-1,3,-3
        jac[:, j] = (27 * (subvec[:, 0] - subvec[:, 1]) - (subvec[:, 2] - subvec[:, 3])) / 48.0

    return jac / degree  # output in degrees, not radians for consistency with astropy function


def _stand_alone_test(infile):
    """
    Simple tests of the above routines.

    Parameters
    ----------
    infile : str
        File name. Can be either an L2 ASDF file or FITS with the
        WCS in the primary HDU.

    Returns
    -------
    None

    """

    if infile[-5:] == ".asdf":
        with asdf.open(infile) as f:
            wcsobj = PyIMCOM_WCS(f["roman"]["meta"]["wcs"])

    if infile[-5:] == ".fits":
        with fits.open(infile) as f:
            wcsobj = PyIMCOM_WCS(f[0].header)

    inpos = np.zeros((9, 2))
    inpos[3:6, 1] = 2043.5
    inpos[6:, 1] = 4087
    inpos[1::3, 0] = 2043.5
    inpos[2::3, 0] = 4087
    print(inpos)

    skycoord = wcsobj.all_pix2world(inpos, 0)
    print(skycoord)

    recovered = wcsobj.all_world2pix(skycoord, 0)
    print(recovered)

    jac = local_partial_pixel_derivatives2(wcsobj, 0.0, 0.0)
    print(jac * 3600)
    print(np.linalg.det(jac * 3600))


if __name__ == "__main__":
    """Command-line test, with input file as an argument."""

    _stand_alone_test(sys.argv[1])
