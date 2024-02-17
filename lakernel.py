'''
Linear algebra kernels to solve linear systems.

Classes (current)
-------
LAKernel : Static methods to solve linear systems.

Classes (planned)
-------
LAKernel : Abstract base class of linear algebra kernels.
EigenKernel : LA kernel using eigendecomposition.
ChoKernel : LA kernel using Cholesky decomposition.
IterKernel : LA kernel using iterative method.
EmpirKernel : Fake LA kernel using empirical relation.

'''

from warnings import warn

import numpy as np
from scipy.linalg import cholesky, cho_solve

from .config import Timer
from pyimcom_croutines import lakernel1, build_reduced_T_wrap


class LAKernel:
    '''
    Linear algebra kernels, static methods to solve linear systems.

    Methods (current)
    -------
    CKernelMulti (staticmethod) : To be EigenKernel.
    get_coadd_matrix_discrete (staticmethod) : To be ChoKernel.

    '''

    @staticmethod
    def CKernelMulti(A, mBhalf, C, targetleak, kCmin=1e-16, kCmax=1e16, nbis=53, smax=1e8):
        '''
        This one generates multiple images. there can be nt target PSFs.
        if 2D arrays are input then assumes nt=1

        Inputs:
          A = system matrix, shape = (n, n)
          mBhalf = -B/2 = target overlap matrix, shape = (nt, m, n)
          C = target normalization, shape = (nt,)
          targetleak = allowable leakage of target PSF(nt,)
          kCmin, kCmax, nbis = range of kappa/C to test, number of bisections
          smax = maximum allowed Sigma

        Outputs:
          kappa = Lagrange multiplier per output pixel, shape = (nt, m)
          Sigma = output noise amplification, shape = (nt, m)
          UC = fractional squared error in PSF, shape = (nt, m)
          T = coaddition matrix, shape =(nt,m,n)

        '''

        # eigensystem
        lam, Q = np.linalg.eigh(A)

        # get dimensions and mPhalf matrix
        if mBhalf.ndim == 2:
            nt = 1
            (m, n) = np.shape(mBhalf)
            mBhalf_image = mBhalf.reshape((1, m, n))
            C_s = np.array([C])
            targetleak_s = np.array([targetleak])
        else:
            (nt, m, n) = np.shape(mBhalf)
            mBhalf_image = mBhalf
            C_s = C
            targetleak_s = targetleak

        # output arrays
        kappa = np.zeros((nt, m))
        Sigma = np.zeros((nt, m))
        UC = np.zeros((nt, m))
        T = np.zeros((nt, m, n))
        tt = np.zeros((m, n))

        for k in range(nt):
            lakernel1(lam, Q, mBhalf_image[k, :, :]@Q, C_s[k], targetleak_s[k],
                      kCmin, kCmax, nbis, kappa[k, :], Sigma[k, :], UC[k, :], tt, smax)
            T[k, :, :] = tt@Q.T
        return (kappa, Sigma, UC, T)

    @staticmethod
    def get_coadd_matrix_discrete(A, mBhalf, C, kappaC_arr, ucmin, smax=0.5):
        '''
        alternative to CKernelMulti, almost same functionality but has a range of kappa

        Inputs:
          A = in-in overlap matrix, shape (n, n)
          mBhalf = in-out overlap matrix, shape (n_out, m, n)
          C = out-out overlap, vector, length n_out
          kappaC_arr = eigenvalue nodes, vector, length nv, ascending order
          ucmin = minimum U/C (after this focus on noise)

        Named inputs:
          smax = maximum noise to allow

        Outputs:
          kappa, shape (n_out, m)
          S, shape (n_out, m)
          UC, shape (n_out, m)
          T, shape (n_out, m, n)

        '''

        # get parameters
        (n_out, m, n) = np.shape(mBhalf)
        nv = np.size(kappaC_arr)

        # outputs
        T_ = np.zeros((n_out, m, n), dtype=np.float32)
        UC_ = np.zeros((n_out, m), dtype=np.float32)
        S_ = np.zeros((n_out, m), dtype=np.float32)
        k_ = np.zeros((n_out, m), dtype=np.float32)

        Tpi = np.zeros((nv, m, n))
        #
        # loop over output PSFs
        for j_out in range(n_out):

            # Cholesky decompositions for each eigenvalue node
            L = np.zeros((nv, n, n))
            di = np.diag_indices(n)
            AA = np.copy(A)

            kappa_arr = kappaC_arr * C[j_out]
            # KC: Now that we specify kappa/C instead of kappa, decomposition
            # results can no longer be shared between output PSFs (2/16/2024)

            for j in range(nv):
                if j > 0:
                    AA[di] += kappa_arr[j] - kappa_arr[j-1]
                else:
                    AA[di] += kappa_arr[0]
                try:
                    L[j, :, :] = cholesky(AA, lower=True, check_finite=False)
                except:
                    # if matrix is not quite positive definite, we can rectify it
                    w, v = np.linalg.eigh(A)
                    # AA[di] += kappa_arr[j] + np.abs(w[0])
                    AA[di] += np.abs(w[0]) + 1e-32  # KC: this seems right
                    del v
                    warn('Warning: pyimcom_lakernel Cholesky decomposition failed; '
                         'fixed negative eigenvalue {:19.12e}'.format(w[0]))
                    L[j, :, :] = cholesky(AA, lower=True, check_finite=False)
                    AA[di] -= np.abs(w[0]) + 1e-32  # KC: let's recover AA
            del AA
            del di

            # build values at nodes
            for p in range(nv):
                Tpi[p, :, :] = cho_solve(
                    (L[p, :, :], True), mBhalf[j_out, :, :].T, check_finite=False).T
            Dp = np.einsum('ai,pai->ap', mBhalf[j_out, :, :], Tpi)
            Npq = np.einsum('pai,qai->apq', Tpi, Tpi)
            Ep = np.einsum('qai,ai->aq', Tpi, mBhalf[j_out, :, :])
            Epq = np.zeros((m, nv, nv))
            for p in range(nv):
                for q in range(p):
                    Epq[:, q, p] = Epq[:, p, q] = Ep[:, q] - \
                        kappa_arr[p]*Npq[:, p, q]
                Epq[:, p, p] = Ep[:, p] - kappa_arr[p]*Npq[:, p, p]

            # now make outputs and call C function
            out_kappa = np.zeros((m,))
            out_Sigma = np.zeros((m,))
            out_UC = np.zeros((m,))
            out_w = np.zeros((m*nv,))
            build_reduced_T_wrap(Npq.flatten(), Dp.flatten()/C[j_out], Epq.flatten()/C[j_out], kappaC_arr,
                                 ucmin, smax, out_kappa, out_Sigma, out_UC, out_w)

            # make outputs
            k_[j_out, :] = out_kappa*C[j_out]
            S_[j_out, :] = out_Sigma
            UC_[j_out, :] = out_UC
            T_[j_out, :, :] = np.einsum('pai,ap->ai', Tpi, out_w.reshape((m, nv)))

        return (k_, S_, UC_, T_)
