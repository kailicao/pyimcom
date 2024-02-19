'''
Linear algebra kernels to solve linear systems.

Classes
-------
_LAKernel : Abstract base class of linear algebra kernels.
EigenKernel : LA kernel using eigendecomposition.
ChoKernel : LA kernel using Cholesky decomposition.
IterKernel : LA kernel using iterative method.
EmpirKernel : Fake LA kernel using empirical relation.

'''

from warnings import warn

import numpy as np
from scipy.linalg import cholesky, cho_solve
from astropy import units as u
from scipy.sparse.linalg import cg

from pyimcom_croutines import lakernel1, build_reduced_T_wrap
from .config import Settings as Stn


class _LAKernel:
    '''
    Abstract base class of linear algebra kernels.

    Methods
    -------
    __init__ : Constructor.
    __call__ : Solve linear systems.

    '''

    def __init__(self, outst: 'coadd.OutStamp') -> None:
        '''
        Constructor.

        Parameters
        ----------
        outst : coadd.OutStamp
            Output postage stamp to which this kernel instance is attached.

        Returns
        -------
        None.

        '''

        self.outst = outst
        cfg = outst.blk.cfg  # shortcut

        # get parameters
        self.n_out, self.m, self.n = np.shape(outst.mhalfb)
        # self.n_out = cfg.n_out; self.m = cfg.n2f ** 2
        # self.n = self.outst.inpix_cumsum[-1]  # number of input pixels
        self.n2f = cfg.n2f

        self.kappaC_arr = cfg.kappaC_arr  # eigenvalue nodes, vector, length nv, ascending order
        self.nv = np.size(self.kappaC_arr)
        self.ucmin = np.array([cfg.uctarget] * self.n_out)
        # allowable leakage of target PSF(n_out,) / minimum U/C (after this focus on noise)
        self.smax = cfg.sigmamax  # maximum noise to allow / maximum allowed Sigma

    def __call__(self) -> None:
        '''
        Solve linear systems.

        This produces the following arrays for self.outst:
        T = coaddition matrix, shape = (n_out, m, n)
        UC = fractional squared error in PSF, shape = (n_out, m)
        Sigma = output noise amplification, shape = (n_out, m)
        kappa = Lagrange multiplier per output pixel, shape = (n_out, m)

        Returns
        -------
        None.

        '''

        # outputs
        self.outst.T = np.zeros((self.n_out, self.m, self.n), dtype=np.float32)
        self.UC_     = np.zeros((self.n_out, self.m), dtype=np.float32)
        self.Sigma_  = np.zeros((self.n_out, self.m), dtype=np.float32)
        self.kappa_  = np.zeros((self.n_out, self.m), dtype=np.float32)

        if self.nv == 1:
            self._call_single_kappa()
        else:
            self._call_multi_kappa()

        # post processing
        shape = (self.n_out, self.n2f, self.n2f)
        self.outst.UC    = self.UC_.   reshape(shape); del self.UC_
        self.outst.Sigma = self.Sigma_.reshape(shape); del self.Sigma_
        self.outst.kappa = self.kappa_.reshape(shape); del self.kappa_


class EigenKernel(_LAKernel):
    '''
    LA kernel using eigendecomposition.

    Methods
    -------
    _call_single_kappa : Solve linear systems for single kappa node.
    _call_multi_kappa : Solve linear systems for multiple kappa nodes.

    '''

    def _call_single_kappa(self) -> None:
        '''
        Solve linear systems for single kappa node.

        Returns
        -------
        None.

        '''

        # get parameters and arrays
        A = self.outst.sysmata  # system matrix, shape = (n, n)
        mBhalf = self.outst.mhalfb  # target overlap matrix, shape = (n_out, m, n)
        C = self.outst.outovlc  # target normalization, shape = (n_out,)

        lam, Q = np.linalg.eigh(A)  # eigensystem
        mPhalf = mBhalf @ Q  # -P/2 matrix

        for k in range(self.n_out):
            my_kappa = self.kappaC_arr[0] * C[k]
            self.kappa_ [k, :] = my_kappa
            self.Sigma_ [k, :] = np.sum((mPhalf[k] / (lam+my_kappa))**2, axis=1)
            self.UC_    [k, :] = 1 - (lam+2*my_kappa) / (lam+my_kappa)**2 @ mPhalf[k].T**2 / C[k]
            self.outst.T[k, :, :] = mPhalf[k] / (lam+my_kappa) @ Q.T

        del lam, Q, mPhalf

    def _call_multi_kappa(self, nbis: int = 13) -> None:
        '''
        Solve linear systems for multiple kappa nodes.

        Based on pyimcom_lakernel.CKernelMulti of furry-parakeet:
        This one generates multiple images. there can be n_out target PSFs.

        Parameters
        ----------
        nbis : int, optional
            number of bisections. The default is 13.

        Returns
        -------
        None.

        '''

        # get parameters and arrays
        kCmin = self.kappaC_arr[0]; kCmax = self.kappaC_arr[-1]  # range of kappa/C to test
        A = self.outst.sysmata  # system matrix, shape = (n, n)
        mBhalf = self.outst.mhalfb  # target overlap matrix, shape = (n_out, m, n)
        C = self.outst.outovlc  # target normalization, shape = (n_out,)

        lam, Q = np.linalg.eigh(A)  # eigensystem
        (n_out, m, n) = np.shape(mBhalf)  # get dimensions
        tt = np.zeros((m, n))  # output array

        for k in range(n_out):
            # using kCmin*C[k], kCmax*C[k] instead of kCmin, kCmax produces reasonable results
            lakernel1(lam, Q, mBhalf[k, :, :] @ Q, C[k], self.ucmin[k], kCmin*C[k], kCmax*C[k], nbis,
                      self.kappa_[k, :], self.Sigma_[k, :], self.UC_[k, :], tt, self.smax)
            self.kappa_ [k, :] *= C[k]
            self.outst.T[k, :, :] = tt @ Q.T


class ChoKernel(_LAKernel):
    '''
    LA kernel using Cholesky decomposition.

    Methods
    -------
    _cholesky_wrapper (staticmethod) : Wrapper for cholesky.
    _call_single_kappa : Solve linear systems for single kappa node.
    _call_multi_kappa : Solve linear systems for multiple kappa nodes.

    '''

    @staticmethod
    def _cholesky_wrapper(AA: np.array, di: (np.array, np.array),
                          A: np.array) -> np.array:
        '''
        Wrapper for cholesky, rectify negative eigenvalue(s) if needed.

        Parameters
        ----------
        AA : np.array, shape : (n, n)
            System matrix plus kappa times noise.
        di : (np.array, np.array), shapes : ((n,), (n,))
            Indices to main diagonal of AA.
        A : np.array, shape : (n, n)
            Original system matrix.

        Returns
        -------
        L : np.array, shape : (n, n)
            cholesky results.

        '''

        try:
            L = cholesky(AA, lower=True, check_finite=False)
        except:
            # if matrix is not quite positive definite, we can rectify it
            w, v = np.linalg.eigh(A)
            # AA[di] += kappa_arr[j] + np.abs(w[0])
            AA[di] += np.abs(w[0]) + 1e-32  # KC: this seems right
            del v
            warn('Warning: pyimcom_lakernel Cholesky decomposition failed; '
                 'fixed negative eigenvalue {:19.12e}'.format(w[0]))
            L = cholesky(AA, lower=True, check_finite=False)
            AA[di] -= np.abs(w[0]) + 1e-32  # KC: let's recover AA

        return L

    def _call_single_kappa(self) -> None:
        '''
        Solve linear systems for single kappa node.

        Returns
        -------
        None.

        '''

        # get parameters and arrays
        n = self.n
        A = self.outst.sysmata  # in-in overlap matrix, shape (n, n)
        mBhalf = self.outst.mhalfb  # in-out overlap matrix, shape (n_out, m, n)
        C = self.outst.outovlc  # out-out overlap, vector, length n_out

        # loop over output PSFs
        for j_out in range(self.n_out):

            # Cholesky decomposition for the only eigenvalue node
            AA = np.copy(A); di = np.diag_indices(n)
            my_kappa = self.kappaC_arr[0] * C[j_out]

            AA[di] += my_kappa
            L = ChoKernel._cholesky_wrapper(AA, di, A)
            Ti = cho_solve((L, True), mBhalf[j_out, :, :].T, check_finite=False).T  # (m, n)
            del AA, di, L

            # build values at node
            D = np.einsum('ai,ai->a', mBhalf[j_out, :, :], Ti)
            N = np.einsum('ai,ai->a', Ti, Ti)

            # make outputs
            self.kappa_ [j_out, :] = my_kappa
            self.Sigma_ [j_out, :] = N
            self.UC_    [j_out, :] = 1.0 - (my_kappa*N + D)/C[j_out]
            self.outst.T[j_out, :, :] = Ti
            del D, N, Ti

    def _call_multi_kappa(self) -> None:
        '''
        Solve linear systems for multiple kappa nodes.

        Based on pyimcom_lakernel.get_coadd_matrix_discrete of furry-parakeet:
        alternative to CKernelMulti, almost same functionality but has a range of kappa

        Returns
        -------
        None.

        '''

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
            AA = np.copy(A); di = np.diag_indices(n)
            kappa_arr = self.kappaC_arr * C[j_out]

            for j in range(nv):
                AA[di] += kappa_arr[j] - (kappa_arr[j-1] if j > 0 else 0)
                L = ChoKernel._cholesky_wrapper(AA, di, A)
                Tpi[j, :, :] = cho_solve((L, True), mBhalf[j_out, :, :].T, check_finite=False).T
            del AA, di, L

            # build values at nodes
            Dp = np.einsum('ai,pai->ap', mBhalf[j_out, :, :], Tpi)
            Npq = np.einsum('pai,qai->apq', Tpi, Tpi)
            # Ep = np.einsum('qai,ai->aq', Tpi, mBhalf[j_out, :, :])
            Epq = np.zeros((m, nv, nv))
            for p in range(nv):
                for q in range(p):
                    Epq[:, q, p] = Epq[:, p, q] = Dp[:, q] - kappa_arr[p]*Npq[:, p, q]
                Epq[:, p, p] = Dp[:, p] - kappa_arr[p]*Npq[:, p, p]

            # now make outputs and call C function
            out_kappa = np.zeros((m,))
            out_Sigma = np.zeros((m,))
            out_UC    = np.zeros((m,))
            out_w     = np.zeros((m*nv,))
            build_reduced_T_wrap(
                Npq.flatten(), Dp.flatten()/C[j_out], Epq.flatten()/C[j_out],
                self.kappaC_arr, self.ucmin, self.smax, out_kappa, out_Sigma, out_UC, out_w)
            del Dp, Npq, Epq

            # make outputs
            self.kappa_ [j_out, :] = out_kappa * C[j_out]
            self.Sigma_ [j_out, :] = out_Sigma
            self.UC_    [j_out, :] = out_UC
            self.outst.T[j_out, :, :] = np.einsum('pai,ap->ai', Tpi, out_w.reshape((m, nv)))
            del out_kappa, out_Sigma, out_UC, out_w


class IterKernel(_LAKernel):
    '''
    LA kernel using iterative method.

    Methods
    -------
    _call_single_kappa : Solve linear systems for single kappa node.
    _call_multi_kappa : Solve linear systems for multiple kappa nodes.

    '''

    def _call_single_kappa(self) -> None:
        '''
        Solve linear systems for single kappa node.

        Returns
        -------
        None.

        '''

        # get parameters and arrays
        m, n = self.m, self.n
        A = self.outst.sysmata  # in-in overlap matrix, shape (n, n)
        mBhalf = self.outst.mhalfb  # in-out overlap matrix, shape (n_out, m, n)
        C = self.outst.outovlc  # out-out overlap, vector, length n_out

        # see if an input pixel is within an acceptance radius of an output pixel
        ddy = self.outst.iny_val[:, None] - self.outst.yx_val[0].ravel()[None, :]
        ddx = self.outst.inx_val[:, None] - self.outst.yx_val[1].ravel()[None, :]
        cfg = self.outst.blk.cfg  # shortcut
        rho_acc = (cfg.instamp_pad / Stn.arcsec) / (cfg.dtheta * u.degree.to('arcsec'))
        relevant_matrix = (np.hypot(ddy, ddx) < rho_acc).T; del ddy, ddx

        # loop over output PSFs
        for j_out in range(self.n_out):

            # iterative method for the only eigenvalue node
            AA = np.copy(A); di = np.diag_indices(n)
            my_kappa = self.kappaC_arr[0] * C[j_out]

            AA[di] += my_kappa
            Ti = np.zeros((m, n), dtype=np.float32)

            # loop over output pixels
            for a in range(m):
                slctn = np.where(relevant_matrix[a])[0]
                A_a = AA[np.ix_(slctn, slctn)]
                b_a = mBhalf[j_out, a, :][slctn]
                T_a, exit_code = cg(A_a, b_a, atol=1e-5)
                if exit_code != 0: print('exit_code != 0', f'{a = }')

                n_in = np.size(slctn)
                for i_in in range(n_in):
                    Ti[a, slctn[i_in]] = T_a[i_in]
                del slctn, A_a, b_a, T_a

            # build values at node
            D = np.einsum('ai,ai->a', mBhalf[j_out, :, :], Ti)
            N = np.einsum('ai,ai->a', Ti, Ti)

            # make outputs
            self.kappa_ [j_out, :] = my_kappa
            self.Sigma_ [j_out, :] = N
            self.UC_    [j_out, :] = 1.0 - (my_kappa*N + D)/C[j_out]
            self.outst.T[j_out, :, :] = Ti
            del D, N, Ti

        del relevant_matrix

    def _call_multi_kappa(self) -> None:
        '''
        Solve linear systems for multiple kappa nodes.

        Returns
        -------
        None.

        '''

        # get parameters and arrays
        nv, m, n = self.nv, self.m, self.n
        A = self.outst.sysmata  # in-in overlap matrix, shape (n, n)
        mBhalf = self.outst.mhalfb  # in-out overlap matrix, shape (n_out, m, n)
        C = self.outst.outovlc  # out-out overlap, vector, length n_out

        # see if an input pixel is within an acceptance radius of an output pixel
        ddy = self.outst.iny_val[:, None] - self.outst.yx_val[0].ravel()[None, :]
        ddx = self.outst.inx_val[:, None] - self.outst.yx_val[1].ravel()[None, :]
        cfg = self.outst.blk.cfg  # shortcut
        rho_acc = (cfg.instamp_pad / Stn.arcsec) / (cfg.dtheta * u.degree.to('arcsec'))
        relevant_matrix = (np.hypot(ddy, ddx) < rho_acc).T; del ddy, ddx

        Tpi = np.zeros((nv, m, n))
        #
        # loop over output PSFs
        for j_out in range(self.n_out):

            # Cholesky decomposition for each eigenvalue node
            AA = np.copy(A); di = np.diag_indices(n)
            kappa_arr = self.kappaC_arr * C[j_out]

            for j in range(nv):
                AA[di] += kappa_arr[j] - (kappa_arr[j-1] if j > 0 else 0)

                # loop over output pixels
                for a in range(m):
                    slctn = np.where(relevant_matrix[a])[0]
                    A_a = AA[np.ix_(slctn, slctn)]
                    b_a = mBhalf[j_out, a, :][slctn]
                    T_a, exit_code = cg(A_a, b_a, atol=1e-5)
                    if exit_code != 0: print('exit_code != 0', f'{a = }')

                    n_in = np.size(slctn)
                    for i_in in range(n_in):
                        Tpi[j, a, slctn[i_in]] = T_a[i_in]
                    del slctn, A_a, b_a, T_a
            del AA, di

            # build values at nodes
            Dp = np.einsum('ai,pai->ap', mBhalf[j_out, :, :], Tpi)
            Npq = np.einsum('pai,qai->apq', Tpi, Tpi)
            # Ep = np.einsum('qai,ai->aq', Tpi, mBhalf[j_out, :, :])
            Epq = np.zeros((m, nv, nv))
            for p in range(nv):
                for q in range(p):
                    Epq[:, q, p] = Epq[:, p, q] = Dp[:, q] - kappa_arr[p]*Npq[:, p, q]
                Epq[:, p, p] = Dp[:, p] - kappa_arr[p]*Npq[:, p, p]

            # now make outputs and call C function
            out_kappa = np.zeros((m,))
            out_Sigma = np.zeros((m,))
            out_UC    = np.zeros((m,))
            out_w     = np.zeros((m*nv,))
            build_reduced_T_wrap(
                Npq.flatten(), Dp.flatten()/C[j_out], Epq.flatten()/C[j_out],
                self.kappaC_arr, self.ucmin, self.smax, out_kappa, out_Sigma, out_UC, out_w)
            del Dp, Npq, Epq

            # make outputs
            self.kappa_ [j_out, :] = out_kappa * C[j_out]
            self.Sigma_ [j_out, :] = out_Sigma
            self.UC_    [j_out, :] = out_UC
            self.outst.T[j_out, :, :] = np.einsum('pai,ap->ai', Tpi, out_w.reshape((m, nv)))
            del out_kappa, out_Sigma, out_UC, out_w

        del relevant_matrix


class EmpirKernel(_LAKernel):
    '''
    Fake LA kernel using empirical relation.

    Methods
    -------
    _call_single_kappa : Produce the T matrix without solving linear systems.
    _call_multi_kappa : Pathway to _call_single_kappa.

    '''

    def _call_single_kappa(self) -> None:
        '''
        Produce the T matrix without solving linear systems.

        Returns
        -------
        None.

        '''

        # get arrays
        # A = self.outst.sysmata  # in-in overlap matrix, shape (n, n)
        mBhalf = self.outst.mhalfb  # in-out overlap matrix, shape (n_out, m, n)
        C = self.outst.outovlc  # out-out overlap, vector, length n_out

        ddy = self.outst.iny_val[:, None] - self.outst.yx_val[0].ravel()[None, :]
        ddx = self.outst.inx_val[:, None] - self.outst.yx_val[1].ravel()[None, :]
        cfg = self.outst.blk.cfg  # shortcut
        rho_acc = (cfg.instamp_pad / Stn.arcsec) / (cfg.dtheta * u.degree.to('arcsec'))

        Ti = np.maximum(rho_acc - np.hypot(ddy, ddx), 0).T
        Ti_view = np.moveaxis(Ti, -1, 0)
        Ti_view /= np.sum(Ti, axis=-1)[None]
        del ddy, ddx

        # loop over output PSFs
        for j_out in range(self.n_out):
            my_kappa = self.kappaC_arr[0] * C[j_out]

            # build values at node
            D = np.einsum('ai,ai->a', mBhalf[j_out, :, :], Ti)
            N = np.einsum('ai,ai->a', Ti, Ti)

            # make outputs
            self.kappa_ [j_out, :] = my_kappa
            self.Sigma_ [j_out, :] = N
            self.UC_    [j_out, :] = 1.0 - (my_kappa*N + D)/C[j_out]
            self.outst.T[j_out, :, :] = Ti
            del D, N, Ti

    def _call_multi_kappa(self) -> None:
        '''
        Pathway to _call_single_kappa.

        Returns
        -------
        None.

        '''

        self._call_single_kappa()
