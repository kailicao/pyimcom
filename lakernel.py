from warnings import warn

import numpy as np
from scipy.linalg import cholesky, cho_solve

from .config import Timer
from pyimcom_croutines import build_reduced_T_wrap


class LAKernel:
    '''Linear algebra kernel, static methods to solve linear systems.
    '''

    @staticmethod
    def get_coadd_matrix_discrete(A, mBhalf, C, kappa_array, ucmin, smax=0.5,
                                  timing: bool = False):

        if timing: timer = Timer()

        # get parameters
        (n_out, m, n) = np.shape(mBhalf)
        nv = np.size(kappa_array)

        # Cholesky decompositions for each eigenvalue node
        L = np.zeros((nv, n, n))
        di = np.diag_indices(n)
        AA = np.copy(A)
        for j in range(nv):
            if j > 0:
                AA[di] += kappa_array[j] - kappa_array[j-1]
            else:
                AA[di] += kappa_array[0]
            try:
                L[j, :, :] = cholesky(
                    AA, lower=True, check_finite=False)
            except:
                # if matrix is not quite positive definite, we can rectify it
                w, v = np.linalg.eigh(A)
                AA[di] += kappa_array[j] + np.abs(w[0])
                del v
                warn('Warning: pyimcom_lakernel Cholesky decomposition failed; '
                     'fixed negative eigenvalue {:19.12e}'.format(w[0]))
                L[j, :, :] = cholesky(
                    AA, lower=True, check_finite=False)
        del AA
        del di

        if timing: print('cholesky took', timer(), 's')

        # outputs
        T_ = np.zeros((n_out, m, n), dtype=np.float32)
        UC_ = np.zeros((n_out, m), dtype=np.float32)
        S_ = np.zeros((n_out, m), dtype=np.float32)
        k_ = np.zeros((n_out, m), dtype=np.float32)

        Tpi = np.zeros((nv, m, n))
        #
        # loop over output PSFs
        for j_out in range(n_out):

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
                        kappa_array[p]*Npq[:, p, q]
                Epq[:, p, p] = Ep[:, p] - kappa_array[p]*Npq[:, p, p]

            if timing: print('cho_solve took', timer(), 's')

            # now make outputs and call C function
            out_kappa = np.zeros((m,))
            out_Sigma = np.zeros((m,))
            out_UC = np.zeros((m,))
            out_w = np.zeros((m*nv,))
            build_reduced_T_wrap(Npq.flatten(), Dp.flatten()/C[j_out], Epq.flatten()/C[j_out], kappa_array/C[j_out],
                                 ucmin, smax, out_kappa, out_Sigma, out_UC, out_w)

            # make outputs
            k_[j_out, :] = out_kappa*C[j_out]
            S_[j_out, :] = out_Sigma
            UC_[j_out, :] = out_UC
            T_[j_out, :, :] = np.einsum('pai,ap->ai', Tpi, out_w.reshape((m, nv)))

            if timing: print('C function took', timer(), 's')

        return (k_, S_, UC_, T_)
