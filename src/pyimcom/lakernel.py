"""
Linear algebra kernels to solve linear systems.

Classes
-------
_LAKernel
    Abstract base class of linear algebra kernels.
EigenKernel
    LA kernel using eigendecomposition.
CholKernel
    LA kernel using Cholesky decomposition.
IterKernel
    LA kernel using iterative method.
EmpirKernel
    Fake LA kernel using empirical relation.

Functions
---------
conjugate_gradient
    Simplified version of scipy.sparse.linalg.cg.
_extract_submatrix
    Extract a submatrix from a symmetric matrix.
_extract_subvector
    Extract a subvector from a vector.
_assign_subvector
    Assign values to a subvector of a vector.

"""

import sys
import time
from warnings import warn

import numpy as np
from astropy import units as u
from numba import njit
from scipy.linalg import LinAlgError, cho_solve, cholesky

from .config import Settings as Stn

try:
    from furry_parakeet.pyimcom_croutines import build_reduced_T_wrap, lakernel1
except ImportError:
    try:
        from pyimcom_croutines import build_reduced_T_wrap, lakernel1
    except ImportError:
        from .routine import build_reduced_T_wrap, lakernel1


class _LAKernel:
    """
    Abstract base class of linear algebra kernels.

    Parameters
    ----------
    outst : coadd.OutStamp
        Output postage stamp to which this kernel instance is attached.

    Methods
    -------
    __init__
        Constructor.
    __call__
        Solve linear systems.

    """

    def __init__(self, outst):
        self.outst = outst
        cfg = outst.blk.cfg  # shortcut

        # get parameters
        self.n_out = cfg.n_out
        self.m = cfg.n2f**2
        self.n = self.outst.inpix_cumsum[-1]  # number of input pixels
        self.n2f = cfg.n2f

        self.kappaC_arr = cfg.kappaC_arr  # eigenvalue nodes, vector, length nv, ascending order
        self.nv = np.size(self.kappaC_arr)
        self.ucmin = cfg.uctarget
        # allowable leakage of target PSF(n_out,) / minimum U/C (after this focus on noise)
        self.smax = cfg.sigmamax  # maximum noise to allow / maximum allowed Sigma

    def __call__(self) -> None:
        """
        Solve linear systems.

        Returns
        -------
        None

        Notes
        -----
        This produces the following arrays for self.outst:

        * T = coaddition matrix, shape = (n_out, m, n)

        * UC = fractional squared error in PSF, shape = (n_out, m)

        * Sigma = output noise amplification, shape = (n_out, m)

        * kappa = Lagrange multiplier per output pixel, shape = (n_out, m)

        """

        # shape of self.outst.UC, self.outst.Sigma, and self.outst.kappa
        shape = (self.n_out, self.n2f, self.n2f)

        # special handling for n=0 (no input pixels in this postage stamp)
        if self.n == 0:
            self.outst.T = np.zeros((self.n_out, self.m, 0), dtype=np.float32)
            self.outst.UC = np.ones(
                shape, dtype=np.float32
            )  # leakage metric U=C since the 'true' output PSF is zero
            self.outst.Sigma = np.zeros(shape, dtype=np.float32)  # all zeros, no noise
            self.outst.kappa = np.ones(
                shape, dtype=np.float32
            )  # not relevant but will fill with 1's to avoid log errors
            return

        # outputs
        self.outst.T = np.zeros((self.n_out, self.m, self.n), dtype=np.float32)
        self.UC_ = np.zeros((self.n_out, self.m), dtype=np.float32)
        self.Sigma_ = np.zeros((self.n_out, self.m), dtype=np.float32)
        self.kappa_ = np.zeros((self.n_out, self.m), dtype=np.float32)

        if self.nv == 1:
            self._call_single_kappa()
        else:
            self._call_multi_kappa()

        # post processing
        self.outst.UC = self.UC_.reshape(shape)
        del self.UC_
        self.outst.Sigma = self.Sigma_.reshape(shape)
        del self.Sigma_
        self.outst.kappa = self.kappa_.reshape(shape)
        del self.kappa_


class EigenKernel(_LAKernel):
    """
    LA kernel using eigendecomposition.

    Methods
    -------
    _call_single_kappa
        Solve linear systems for single kappa node.
    _call_multi_kappa
        Solve linear systems for multiple kappa nodes.

    """

    def _call_single_kappa(self) -> None:
        """Solve linear systems for single kappa node."""

        # get parameters and arrays
        A = self.outst.sysmata  # system matrix, shape = (n, n)
        mBhalf = self.outst.mhalfb  # target overlap matrix, shape = (n_out, m, n)
        C = self.outst.outovlc  # target normalization, shape = (n_out,)

        lam, Q = np.linalg.eigh(A)  # eigensystem
        mPhalf = mBhalf @ Q  # -P/2 matrix

        for k in range(self.n_out):
            my_kappa = self.kappaC_arr[0] * C[k]
            self.kappa_[k, :] = my_kappa
            self.Sigma_[k, :] = np.sum((mPhalf[k] / (lam + my_kappa)) ** 2, axis=1)
            self.UC_[k, :] = 1 - (lam + 2 * my_kappa) / (lam + my_kappa) ** 2 @ mPhalf[k].T ** 2 / C[k]
            self.outst.T[k, :, :] = mPhalf[k] / (lam + my_kappa) @ Q.T

        del lam, Q, mPhalf

    def _call_multi_kappa(self, nbis: int = 13) -> None:
        """
        Solve linear systems for multiple kappa nodes.

        Parameters
        ----------
        nbis : int, optional
            Number of bisections.

        Returns
        -------
        None

        Notes
        -----
        Based on pyimcom_lakernel.CKernelMulti of furry-parakeet.
        This one generates multiple images. there can be n_out target PSFs.

        """

        # get parameters and arrays
        kCmin = self.kappaC_arr[0]
        kCmax = self.kappaC_arr[-1]  # range of kappa/C to test
        A = self.outst.sysmata  # system matrix, shape = (n, n)
        mBhalf = self.outst.mhalfb  # target overlap matrix, shape = (n_out, m, n)
        C = self.outst.outovlc  # target normalization, shape = (n_out,)

        lam, Q = np.linalg.eigh(A)  # eigensystem
        (n_out, m, n) = np.shape(mBhalf)  # get dimensions
        tt = np.zeros((m, n))  # output array

        for k in range(n_out):
            # using kCmin*C[k], kCmax*C[k] instead of kCmin, kCmax produces reasonable results
            lakernel1(
                lam,
                Q,
                mBhalf[k, :, :] @ Q,
                C[k],
                self.ucmin,
                kCmin * C[k],
                kCmax * C[k],
                nbis,
                self.kappa_[k, :],
                self.Sigma_[k, :],
                self.UC_[k, :],
                tt,
                self.smax,
            )
            self.kappa_[k, :] *= C[k]
            self.outst.T[k, :, :] = tt @ Q.T


class CholKernel(_LAKernel):
    """
    LA kernel using Cholesky decomposition.

    Methods
    -------
    _cholesky_wrapper
        Wrapper for cholesky (staticmethod)
    _call_single_kappa
        Solve linear systems for single kappa node.
    _call_multi_kappa
        Solve linear systems for multiple kappa nodes.

    """

    @staticmethod
    def _cholesky_wrapper(AA: np.array, di: tuple[np.array, np.array], A: np.array) -> np.array:
        """
        Wrapper for cholesky, rectifies negative eigenvalue(s) if needed.

        Parameters
        ----------
        AA : np.array
            System matrix A plus kappa times noise; shape (n,n).
        di : (np.array, np.array)
            Indices to main diagonal of AA; each has shape (n,).
        A : np.array
            Original system matrix; shape (n,n).

        Returns
        -------
        L : np.array
            Cholesky results; shape (n,n).

        """

        try:
            L = cholesky(AA, lower=True, check_finite=False)
        except LinAlgError:
            # if matrix is not quite positive definite, we can rectify it
            w, v = np.linalg.eigh(A)
            # AA[di] += kappa_arr[j] + np.abs(w[0])
            AA[di] += np.abs(w[0]) + 1e-16  # KC: this seems right
            print("repair w", w[:8])
            sys.stdout.flush()
            del v
            warn(
                "Warning: pyimcom_lakernel Cholesky decomposition failed; "
                f"fixed negative eigenvalue {w[0]:19.12e}"
            )
            L = cholesky(AA, lower=True, check_finite=False)
            AA[di] -= np.abs(w[0]) + 1e-16  # KC: let's recover AA

        return L

    def _call_single_kappa(self) -> None:
        """Solve linear systems for single kappa node."""

        # get parameters and arrays
        n = self.n
        A = self.outst.sysmata  # in-in overlap matrix, shape (n, n)
        mBhalf = self.outst.mhalfb  # in-out overlap matrix, shape (n_out, m, n)
        C = self.outst.outovlc  # out-out overlap, vector, length n_out

        # loop over output PSFs
        for j_out in range(self.n_out):
            t0 = time.time()

            # Cholesky decomposition for the only eigenvalue node
            AA = np.copy(A)
            di = np.diag_indices(n)
            my_kappa = self.kappaC_arr[0] * C[j_out]

            if my_kappa:
                AA[di] += my_kappa
            t1 = time.time()
            L = self._cholesky_wrapper(AA, di, A)
            t2 = time.time()
            Ti = cho_solve((L, True), mBhalf[j_out, :, :].T, check_finite=False).T  # (m, n)
            t3 = time.time()
            del AA, di, L

            # build values at node
            D = np.einsum("ai,ai->a", mBhalf[j_out, :, :], Ti)
            N = np.einsum("ai,ai->a", Ti, Ti)
            t4 = time.time()

            # make outputs
            self.kappa_[j_out, :] = my_kappa
            self.Sigma_[j_out, :] = N
            self.UC_[j_out, :] = 1.0 - (my_kappa * N + D) / C[j_out]
            self.outst.T[j_out, :, :] = Ti
            del D, N, Ti
            t5 = time.time()
            print(
                "Cholesky timing -> "
                f"{t1 - t0:5.2f} {t2 - t0:5.2f} {t3 - t0:5.2f} {t4 - t0:5.2f} {t5 - t0:5.2f}"
            )

    def _call_multi_kappa(self) -> None:
        """
        Solve linear systems for multiple kappa nodes.

        Returns
        -------
        None

        Notes
        -----
        Based on pyimcom_lakernel.get_coadd_matrix_discrete of furry-parakeet:
        alternative to CKernelMulti, almost same functionality but has a range of kappa.

        """

        # get parameters and arrays
        nv, m, n = self.nv, self.m, self.n
        A = self.outst.sysmata  # in-in overlap matrix, shape (n, n)
        mBhalf = self.outst.mhalfb  # in-out overlap matrix, shape (n_out, m, n)
        C = self.outst.outovlc  # out-out overlap, vector, length n_out

        Tpi = np.zeros((nv, m, n))
        #
        # loop over output PSFs
        for j_out in range(self.n_out):
            # Cholesky decomposition for each eigenvalue node
            AA = np.copy(A)
            di = np.diag_indices(n)
            kappa_arr = self.kappaC_arr * C[j_out]

            for j in range(nv):
                AA[di] += kappa_arr[j] - (kappa_arr[j - 1] if j > 0 else 0)
                L = CholKernel._cholesky_wrapper(AA, di, A)
                Tpi[j, :, :] = cho_solve((L, True), mBhalf[j_out, :, :].T, check_finite=False).T
            del AA, di, L

            # build values at nodes
            Dp = np.einsum("ai,pai->ap", mBhalf[j_out, :, :], Tpi)
            Npq = np.einsum("pai,qai->apq", Tpi, Tpi)
            Epq = np.zeros((m, nv, nv))
            for p in range(nv):
                for q in range(p):
                    Epq[:, q, p] = Epq[:, p, q] = Dp[:, q] - kappa_arr[p] * Npq[:, p, q]
                Epq[:, p, p] = Dp[:, p] - kappa_arr[p] * Npq[:, p, p]

            # now make outputs and call C function
            out_kappa = np.zeros((m,))
            out_Sigma = np.zeros((m,))
            out_UC = np.zeros((m,))
            out_w = np.zeros((m * nv,))
            build_reduced_T_wrap(
                Npq.flatten(),
                Dp.flatten() / C[j_out],
                Epq.flatten() / C[j_out],
                self.kappaC_arr,
                self.ucmin,
                self.smax,
                out_kappa,
                out_Sigma,
                out_UC,
                out_w,
            )
            del Dp, Npq, Epq

            # make outputs
            self.kappa_[j_out, :] = out_kappa * C[j_out]
            self.Sigma_[j_out, :] = out_Sigma
            self.UC_[j_out, :] = out_UC
            self.outst.T[j_out, :, :] = np.einsum("pai,ap->ai", Tpi, out_w.reshape((m, nv)))
            del out_kappa, out_Sigma, out_UC, out_w


@njit
def conjugate_gradient(A: np.array, b: np.array, rtol: float = 1.5e-3, maxiter: int = 30) -> np.array:
    """
    Simplified version of scipy.sparse.linalg.cg.

    Parameters
    ----------
    A : np.array
        System matrix A, shape (n,n)
    b : np.array
        Column vector b, shape (n,).
    rtol : float, optional
        Relative tolerance.
    maxiter : int, optional
        Maximum number of iterations.

    Returns
    -------
    x : np.array
        Solution vector b, shape (n,).

    """

    atol = np.linalg.norm(b) * rtol

    x = np.zeros_like(b)
    r = b.copy()

    rho_prev = 0.0
    p = r.copy()  # First spin

    for iteration in range(maxiter):
        rho_cur = np.dot(r, r)
        if rho_cur**0.5 < atol:  # Are we done?
            break

        if iteration > 0:
            p *= rho_cur / rho_prev  # beta
            p += r

        q = A @ p
        alpha = rho_cur / np.dot(p, q)
        x += alpha * p
        r -= alpha * q
        rho_prev = rho_cur

    return x


@njit
def _extract_submatrix(mat_orig: np.array, selection: np.array) -> np.array:
    """
    Extract a submatrix from a symmetric matrix.

    Parameters
    ----------
    mat_orig : np.array
        Symmetric matrix to be extracted, shape (n,n).
    selection : np.array
        Integer array of indices of rows and columns to extract, shape (n,).

    Returns
    -------
    mat_copy : np.array
        Extracted submatrix. Shape (n_,n_), where n_ is number of selected rows or columns.

    """

    n_ = selection.size
    mat_copy = np.empty((n_, n_))

    for j_, j in enumerate(selection):
        for i_, i in enumerate(selection):
            mat_copy[j_, i_] = mat_orig[j, i]
            if i_ > j_:
                break

    for j_ in range(n_):
        for i_ in range(j_ + 1, n_):
            mat_copy[j_, i_] = mat_copy[i_, j_]

    return mat_copy


@njit
def _extract_subvector(vec_orig: np.array, selection: np.array) -> np.array:
    """
    Extract a subvector from a vector.

    Parameters
    ----------
    vec_orig : np.array
        Vector to be extracted, shape (n,).
    selection : np.array
        Integer array of indices of elements to extract, shape (n,).

    Returns
    -------
    vec_copy : np.array
        Extracted subvector. Shape (n_,) where n_ is number of selected elements.

    """

    n_ = selection.size
    vec_copy = np.empty((n_,))

    for i_, i in enumerate(selection):
        vec_copy[i_] = vec_orig[i]

    return vec_copy


@njit
def _assign_subvector(vec_left: np.array, vec_right: np.array, selection: np.array) -> None:
    """
    Assign values to a subvector of a vector.

    Parameters
    ----------
    vec_left : np.array
        Vector to be assigned to.
    vec_right : np.array
        Subvector of values to assign.
    selection : np.array
        Integer array of indices of elements to assign to; same length as `vec_left`.

    Returns
    -------
    None

    """

    for i_, i in enumerate(selection):
        vec_left[i] = vec_right[i_]


class IterKernel(_LAKernel):
    """
    LA kernel using iterative method.

    Methods
    -------
    _iterative_wrapper
        Wrapper for iterative method (staticmethod).
    _call_single_kappa
        Solve linear systems for single kappa node.
    _call_multi_kappa
        Solve linear systems for multiple kappa nodes.

    """

    @staticmethod
    @njit
    def _iterative_wrapper(
        AA: np.array, mBhalf: np.array, relevant_matrix: np.array, rtol: float = 1.5e-3, maxiter: int = 30
    ) -> np.array:
        """
        Wrapper for conjugate gradient method.

        Parameters
        ----------
        AA : np.array
            System matrix A plus kappa times noise. Shape (n,n).
        mBhalf : np.array
            System matrix -B/2 (for a single target PSF). Shape (n,n).
        relevant_matrix : np.array
            Boolean array indicating whether to use an input pixel for an output pixel. Shape (m,n).
        rtol : float, optional
            Relative tolerance.
        maxiter : int, optional
            Maximum number of iteration.

        Returns
        -------
        np.array
            Output T matrix (for a single target PSF). Shape (m,n).

        """

        m, n = np.shape(mBhalf)
        Ti = np.zeros((m, n), dtype=np.float32)

        # loop over output pixels
        for a in range(m):
            selection = np.nonzero(relevant_matrix[a])[0]
            _assign_subvector(
                Ti[a],
                conjugate_gradient(
                    _extract_submatrix(AA, selection), _extract_subvector(mBhalf[a], selection), rtol, maxiter
                ),
                selection,
            )

        return Ti

    def _call_single_kappa(self, exact_UC: bool = False) -> None:
        """
        Solve linear systems for single kappa node.

        Parameters
        ----------
        exact_UC : bool, optional
            Whether to use exact expression for U/C.
            The default is False, as this is slow and the gain is very small.

        Returns
        -------
        None

        """

        # get parameters and arrays
        n = self.n
        A = self.outst.sysmata  # in-in overlap matrix, shape (n, n)
        mBhalf = self.outst.mhalfb  # in-out overlap matrix, shape (n_out, m, n)
        C = self.outst.outovlc  # out-out overlap, vector, length n_out

        # see if an input pixel is within an acceptance radius of an output pixel
        mddy = self.outst.yx_val[0].ravel()[:, None] - self.outst.iny_val[None, :]
        mddx = self.outst.yx_val[1].ravel()[:, None] - self.outst.inx_val[None, :]
        cfg = self.outst.blk.cfg  # shortcut
        rho_acc = (cfg.instamp_pad / Stn.arcsec) / (cfg.dtheta * u.degree.to("arcsec"))
        relevant_matrix = np.hypot(mddy, mddx) < rho_acc
        del mddy, mddx

        # loop over output PSFs
        for j_out in range(self.n_out):
            # iterative method for the only eigenvalue node
            AA = np.copy(A)
            di = np.diag_indices(n)
            my_kappa = self.kappaC_arr[0] * C[j_out]

            if my_kappa:
                AA[di] += my_kappa
            Ti = IterKernel._iterative_wrapper(
                AA, mBhalf[j_out], relevant_matrix, cfg.iter_rtol, cfg.iter_max
            )

            # build values at node
            D = np.einsum("ai,ai->a", mBhalf[j_out, :, :], Ti)
            N = np.einsum("ai,ai->a", Ti, Ti)
            if exact_UC:
                E = np.einsum("ij,ai,aj->a", A, Ti, Ti)

            # make outputs
            self.kappa_[j_out, :] = my_kappa
            self.Sigma_[j_out, :] = N
            if exact_UC:
                self.UC_[j_out, :] = 1.0 + (E - 2 * D) / C[j_out]
            else:
                self.UC_[j_out, :] = 1.0 - (my_kappa * N + D) / C[j_out]
            self.outst.T[j_out, :, :] = Ti

            del D, N, Ti
            if exact_UC:
                del E

        del relevant_matrix

    def _call_multi_kappa(self, exact_UC: bool = True) -> None:
        """
        Solve linear systems for multiple kappa nodes.

        Parameters
        ----------
        exact_UC : bool, optional
            Whether to use exact expression for U/C.
            The default is True, as the approximation does not work.
            KC: Please avoid this whenever possible as this is SUPER slow.

        Returns
        -------
        None

        """

        # get parameters and arrays
        nv, m, n = self.nv, self.m, self.n
        A = self.outst.sysmata  # in-in overlap matrix, shape (n, n)
        mBhalf = self.outst.mhalfb  # in-out overlap matrix, shape (n_out, m, n)
        C = self.outst.outovlc  # out-out overlap, vector, length n_out

        # see if an input pixel is within an acceptance radius of an output pixel
        mddy = self.outst.yx_val[0].ravel()[:, None] - self.outst.iny_val[None, :]
        mddx = self.outst.yx_val[1].ravel()[:, None] - self.outst.inx_val[None, :]
        cfg = self.outst.blk.cfg  # shortcut
        rho_acc = (cfg.instamp_pad / Stn.arcsec) / (cfg.dtheta * u.degree.to("arcsec"))
        relevant_matrix = np.hypot(mddy, mddx) < rho_acc
        del mddy, mddx

        Tpi = np.zeros((nv, m, n))
        #
        # loop over output PSFs
        for j_out in range(self.n_out):
            # iterative method for each eigenvalue node
            AA = np.copy(A)
            di = np.diag_indices(n)
            kappa_arr = self.kappaC_arr * C[j_out]

            for j in range(nv):
                AA[di] += kappa_arr[j] - (kappa_arr[j - 1] if j > 0 else 0)
                Tpi[j] = IterKernel._iterative_wrapper(
                    AA, mBhalf[j_out], relevant_matrix, cfg.iter_rtol, cfg.iter_max
                )
            del AA, di

            # build values at nodes
            Dp = np.einsum("ai,pai->ap", mBhalf[j_out, :, :], Tpi)
            Npq = np.einsum("pai,qai->apq", Tpi, Tpi)
            Epq = np.zeros((m, nv, nv))
            for p in range(nv):
                for q in range(p):
                    if exact_UC:
                        Epq[:, q, p] = Epq[:, p, q] = np.einsum("ij,ai,aj->a", A, Tpi[p], Tpi[q])
                    else:
                        Epq[:, q, p] = Epq[:, p, q] = Dp[:, q] - kappa_arr[p] * Npq[:, p, q]
                if exact_UC:
                    Epq[:, p, p] = np.einsum("ij,ai,aj->a", A, Tpi[p], Tpi[p])
                else:
                    Epq[:, p, p] = Dp[:, p] - kappa_arr[p] * Npq[:, p, p]

            # now make outputs and call C function
            out_kappa = np.zeros((m,))
            out_Sigma = np.zeros((m,))
            out_UC = np.zeros((m,))
            out_w = np.zeros((m * nv,))
            build_reduced_T_wrap(
                Npq.flatten(),
                Dp.flatten() / C[j_out],
                Epq.flatten() / C[j_out],
                self.kappaC_arr,
                self.ucmin,
                self.smax,
                out_kappa,
                out_Sigma,
                out_UC,
                out_w,
            )
            del Dp, Npq, Epq

            # make outputs
            self.kappa_[j_out, :] = out_kappa * C[j_out]
            self.Sigma_[j_out, :] = out_Sigma
            self.UC_[j_out, :] = out_UC
            self.outst.T[j_out, :, :] = np.einsum("pai,ap->ai", Tpi, out_w.reshape((m, nv)))
            del out_kappa, out_Sigma, out_UC, out_w

        del relevant_matrix


class EmpirKernel(_LAKernel):
    """
    Fake LA kernel using empirical relation.

    Methods
    -------
    _call_single_kappa
        Produce the T matrix without solving linear systems.
    _call_multi_kappa
        Pathway to `_call_single_kappa`.

    """

    def _call_single_kappa(self) -> None:
        """Produce the T matrix without solving linear systems."""

        mddy = self.outst.yx_val[0].ravel()[:, None] - self.outst.iny_val[None, :]
        mddx = self.outst.yx_val[1].ravel()[:, None] - self.outst.inx_val[None, :]
        cfg = self.outst.blk.cfg  # shortcut
        rho_acc = (cfg.instamp_pad / Stn.arcsec) / (cfg.dtheta * u.degree.to("arcsec"))

        Ti = np.maximum(rho_acc - np.hypot(mddy, mddx), 0)
        del mddy, mddx
        Ti_view = np.moveaxis(Ti, -1, 0)
        Ti_view /= np.sum(Ti, axis=-1)[None]

        # no-quality control option
        if self.outst.no_qlt_ctrl:
            self.outst.T[:, :, :] = Ti
            del Ti
            return

        # get arrays
        A = self.outst.sysmata  # in-in overlap matrix, shape (n, n)
        mBhalf = self.outst.mhalfb  # in-out overlap matrix, shape (n_out, m, n)
        C = self.outst.outovlc  # out-out overlap, vector, length n_out

        # loop over output PSFs
        for j_out in range(self.n_out):
            my_kappa = self.kappaC_arr[0] * C[j_out]

            # build values at node
            D = np.einsum("ai,ai->a", mBhalf[j_out, :, :], Ti)
            N = np.einsum("ai,ai->a", Ti, Ti)
            E = np.einsum("ij,ai,aj->a", A, Ti, Ti)

            # make outputs
            self.kappa_[j_out, :] = my_kappa
            self.Sigma_[j_out, :] = N
            self.UC_[j_out, :] = 1.0 + (E - 2 * D) / C[j_out]
            self.outst.T[j_out, :, :] = Ti
            del D, N

        del Ti

    def _call_multi_kappa(self) -> None:
        """Pathway to _call_single_kappa, since the empirical kernel doesn't use kappa."""

        self._call_single_kappa()
