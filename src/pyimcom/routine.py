"""
Numba version of pyimcom_croutines.c.

Slightly slower than C, used when furry-parakeet is not installed.

Functions
---------
iD5512C_getw
    Interpolation code written by Python.
iD5512C
    2D, 10x10 kernel interpolation for high accuracy.
iD5512C_sym
    iD5512C for symmetrical output.
gridD5512C
    iD5512C for rectangular grid.
lakernel1
    PyIMCOM linear algebra kernel (eigendecomposition).
lsolve_sps
    Routine to solve Ax=b.
build_reduced_T_wrap
    Makes coaddition matrix T from a reduced space.

"""

import numpy as np
from numba import njit


@njit
def iD5512C_getw(w: np.array, fh: float) -> None:
    """
    Interpolation code written by Python.

    Parameters
    ----------
    w : np.array
        Array, shape (10,); 1D interpolation weights will be written here.
    fh : float
        'xfh' and 'yfh' with 1/2 subtracted.

    Returns
    -------
    None

    """

    fh2 = fh * fh
    e_ =  (((+1.651881673372979740E-05*fh2 - 3.145538007199505447E-04)*fh2 +
          1.793518183780194427E-03)*fh2 - 2.904014557029917318E-03)*fh2 + 6.187591260980151433E-04
    o_ = ((((-3.486978652054735998E-06*fh2 + 6.753750285320532433E-05)*fh2 -
          3.871378836550175566E-04)*fh2 + 6.279918076641771273E-04)*fh2 - 1.338434614116611838E-04)*fh
    w[0] = e_ + o_
    w[9] = e_ - o_
    e_ =  (((-1.146756217210629335E-04*fh2 + 2.883845374976550142E-03)*fh2 -
          1.857047531896089884E-02)*fh2 + 3.147734488597204311E-02)*fh2 - 6.753293626461192439E-03
    o_ = ((((+3.121412120355294799E-05*fh2 - 8.040343683015897672E-04)*fh2 +
          5.209574765466357636E-03)*fh2 - 8.847326408846412429E-03)*fh2 + 1.898674086370833597E-03)*fh
    w[1] = e_ + o_
    w[8] = e_ - o_
    e_ =  (((+3.256838096371517067E-04*fh2 - 9.702063770653997568E-03)*fh2 +
          8.678848026470635524E-02)*fh2 - 1.659182651092198924E-01)*fh2 + 3.620560878249733799E-02
    o_ = ((((-1.243658986204533102E-04*fh2 + 3.804930695189636097E-03)*fh2 -
          3.434861846914529643E-02)*fh2 + 6.581033749134083954E-02)*fh2 - 1.436476114189205733E-02)*fh
    w[2] = e_ + o_
    w[7] = e_ - o_
    e_ =  (((-4.541830837949564726E-04*fh2 + 1.494862093737218955E-02)*fh2 -
          1.668775957435094937E-01)*fh2 + 5.879306056792649171E-01)*fh2 - 1.367845996704077915E-01
    o_ = ((((+2.894406669584551734E-04*fh2 - 9.794291009695265532E-03)*fh2 +
          1.104231510875857830E-01)*fh2 - 3.906954914039130755E-01)*fh2 + 9.092432925988773451E-02)*fh
    w[3] = e_ + o_
    w[6] = e_ - o_
    e_ =  (((+2.266560930061513573E-04*fh2 - 7.815848920941316502E-03)*fh2 +
          9.686607348538181506E-02)*fh2 - 4.505856722239036105E-01)*fh2 + 6.067135256905490381E-01
    o_ = ((((-4.336085507644610966E-04*fh2 + 1.537862263741893339E-02)*fh2 -
          1.925091434770601628E-01)*fh2 + 8.993141455798455697E-01)*fh2 - 1.213035309579723942E+00)*fh
    w[4] = e_ + o_
    w[5] = e_ - o_


@njit
def iD5512C(infunc: np.array, xpos: np.array, ypos: np.array,
            fhatout: np.array) -> None:
    """
    2D, 10x10 kernel interpolation for high accuracy

    Can interpolate multiple functions at a time from the same grid
    so we don't have to keep recomputing weights.

    Parameters
    ----------
    infunc : np.array
        Input function on some grid. Shape (nlayer, ngy, ngx).
    xpos : np.array
        Input x values. Shape (nout,).
    ypos : np.array
        Input y values. Shape (nout,).
    fhatout : np.array
        Location to put the output values. Shape : (nlayer, nout).

    Returns
    -------
    None

    """

    # extract dimensions
    nlayer, ngy, ngx = infunc.shape
    nout = xpos.size

    wx = np.zeros((10,))
    wy = np.zeros((10,))

    # loop over points to interpolate
    for ipos in range(nout):
        # frac and integer parts of abscissae
        x = xpos[ipos]
        y = ypos[ipos]
        xi = np.int32(x)
        yi = np.int32(y)

        # point off the grid, don't interpolate
        if xi < 4 or xi >= ngx-5 or yi < 4 or yi >= ngy-5: continue

        # note 'xfh' and 'yfh' have 1/2 subtracted
        iD5512C_getw(wx, x-xi-.5)
        iD5512C_getw(wy, y-yi-.5)

        # and the outputs; Numba does not support np.einsum
        for ilayer in range(nlayer):
            out = 0.0
            for i in range(10):
                interp_vstrip = 0.0
                for j in range(10):
                    interp_vstrip += wx[j] * infunc[ilayer, yi-4+i, xi-4+j]
                out += interp_vstrip * wy[i]
            fhatout[ilayer, ipos] = out


@njit
def iD5512C_sym(infunc: np.array, xpos: np.array, ypos: np.array,
                fhatout: np.array) -> None:
    """
    2D, 10x10 kernel interpolation for high accuracy

    Can interpolate multiple functions at a time from the same grid
    so we don't have to keep recomputing weights.

    This version assumes the output is symmetrical as a sqrt nout x sqrt nout matrix

    Parameters
    ----------
    infunc : np.array
        Input function on some grid. Shape (nlayer, ngy, ngx).
    xpos : np.array
        Input x values. Shape (nout,).
    ypos : np.array
        Input y values. Shape (nout,).
    fhatout : np.array
        Location to put the output values. Shape (nlayer, nout).

    Returns
    -------
    None

    """

    # extract dimensions
    nlayer, ngy, ngx = infunc.shape
    nout = xpos.size
    sqnout = np.int32(np.sqrt(nout+1))

    wx = np.zeros((10,))
    wy = np.zeros((10,))

    # loop over points to interpolate, but in this function only the upper half triangle
    for ipos1 in range(sqnout):
        for ipos2 in range(ipos1, sqnout):
            ipos = ipos1 * sqnout + ipos2

            # frac and integer parts of abscissae
            x = xpos[ipos]
            y = ypos[ipos]
            xi = np.int32(x)
            yi = np.int32(y)

            # point off the grid, don't interpolate
            if xi < 4 or xi >= ngx-5 or yi < 4 or yi >= ngy-5: continue

            # note 'xfh' and 'yfh' have 1/2 subtracted
            iD5512C_getw(wx, x-xi-.5)
            iD5512C_getw(wy, y-yi-.5)

            # and the outputs; Numba does not support np.einsum
            for ilayer in range(nlayer):
                out = 0.0
                for i in range(10):
                    interp_vstrip = 0.0
                    for j in range(10):
                        interp_vstrip += wx[j] * infunc[ilayer, yi-4+i, xi-4+j]
                    out += interp_vstrip * wy[i]
                fhatout[ilayer, ipos] = out

    # ... and now fill in the lower half triangle
    for ipos1 in range(1, sqnout):
        for ipos2 in range(ipos1):
            ipos = ipos1 * sqnout + ipos2
            ipos_sym = ipos2 * sqnout + ipos1
            fhatout[:, ipos] = fhatout[:, ipos_sym]


@njit
def gridD5512C(infunc: np.array, xpos: np.array, ypos: np.array,
               fhatout: np.array) -> None:
    """
    2D, 10x10 kernel interpolation for high accuracy

    this version works with output points on a rectangular grid so that the same
    weights in x and y can be used for many output points

    Parameters
    ----------
    infunc : np.array
        Input function on some grid. Shape (ngy, ngx).
    xpos : np.array
        Input x values. Shape (npi, nxo).
    ypos : np.array
        Input y values. Shape (npi, nxo).
    fhatout : np.array
        Location to put the output values. Shape (npi, nyo*nxo).

    Returns
    -------
    None

    Notes
    -----
    There are npi*nyo*nxo interpolations to be done in total,
    but for each input pixel npi, there is an nyo x nxo grid of output points.
    If you want a 3D array (npi, nyo, nxo), you can reshape `fhatout` after it
    is filled.

    """

    # extract dimensions
    ngy, ngx = infunc.shape
    npi, nxo = xpos.shape[:2]
    nyo = ypos.shape[1]

    wx_ar = np.zeros((nxo, 10))
    wy_ar = np.zeros((nyo, 10))
    xi = np.zeros((nxo,), dtype=np.int32)
    yi = np.zeros((nyo,), dtype=np.int32)

    # loop over points to interpolate
    for i_in in range(npi):
        # get the interpolation weights -- first in x, then in y.
        # do all the output points simultaneously to save time
        for ix in range(nxo):
            x = xpos[i_in, ix]
            xi[ix] = np.int32(x)

            # point off the grid, don't interpolate
            if xi[ix] < 4 or xi[ix] >= ngx-5:
                xi[ix] = 4
                wx_ar[ix] = 0.0
                continue
    
            iD5512C_getw(wx_ar[ix], x-xi[ix]-.5)

        # ... and now in y
        for iy in range(nyo):
            y = ypos[i_in, iy]
            yi[iy] = np.int32(y)

            # point off the grid, don't interpolate
            if yi[iy] < 4 or yi[iy] >= ngy-5:
                yi[iy] = 4
                wy_ar[iy] = 0.0
                continue
    
            iD5512C_getw(wy_ar[iy], y-yi[iy]-.5)

        # ... and now we can do the interpolation
        ipos = 0
        for iy in range(nyo):  # output pixel row
            for ix in range(nxo):  # output pixel column
                out = 0.0
                for i in range(10):
                    interp_vstrip = 0.0
                    for j in range(10):
                        interp_vstrip += wx_ar[ix, j] * infunc[yi[iy]-4+i, xi[ix]-4+j]
                    out += interp_vstrip * wy_ar[iy, i]
                fhatout[i_in, ipos] = out
                ipos += 1


@njit
def lakernel1(lam: np.array, Q: np.array, mPhalf: np.array,
              C: float, targetleak: float, kCmin: float, kCmax: float, nbis: int,
              kappa: np.array, Sigma: np.array, UC: np.array, T: np.array, smax: float) -> None:
    """
    PyIMCOM linear algebra kernel (eigendecomposition).

    Performs all the steps with the for loops (i.e., except the matrix diagonalization).
    In the parameter descriptions, "n" refers to the number of input pixels and "m"
    to the number of output pixels.

    Parameters
    ----------
    lam : np.array
        System matrix eigenvalues. Shape (n,).
    Q : np.array
        System matrix eigenvectors. Shape (n, n).
        (No longer used in this version of the algorithm.)
    mPhalf : np.array
        -P/2 = premultiplied target overlap matrix. Shape (m, n).
    C : float
        Target normalization.
    targetleak : float
        Allowable leakage of target PSF.
    kCmin, kCmax : float, float
        Range of kappa/C to test.
    nbis : int
        Number of bisections.
    kappa : np.array
        Lagrange multiplier per output pixel. Shape (m,).
        Writes to this array.
    Sigma : np.array
        Output noise amplification. Shape (m,).
        Writes to this array.
    UC : np.array
        Fractional squared error in PSF. Shape (m,).
        Writes to this array.
    T : np.array
        Coaddition matrix, needs to be multiplied by Q^T after the return. Shape (m,n).
        Writes to this array.
    smax : float
        Maximum allowed noise amplification Sigma.

    Returns
    -------
    None

    """

    # dimensions
    m, n = mPhalf.shape

    # now loop over pixels
    for a in range(m):
        factor = np.sqrt(kCmax/kCmin)
        kap = np.sqrt(kCmax*kCmin)

        for ib in range(nbis):
            sum_ = sum2 = 0.0
            for i in range(n):
                var = mPhalf[a, i] / (lam[i]+kap)
                sum2 += var*var
                sum_ += (lam[i]+2.0*kap) * var*var

            udc = 1.0 - sum_/C
            factor = np.sqrt(factor)
            kap *= 1.0/factor if (udc > targetleak and sum2 < smax) else factor

        # report final results
        sum_ = sum2 = 0.0
        for i in range(n):
            T[a, i] = var = mPhalf[a, i] / (lam[i]+kap)
            sum2 += var*var
            sum_ += (lam[i]+2.0*kap) * var*var

        Sigma[a] = sum2
        kappa[a] = kap
        UC[a] = 1.0 - sum_/C


@njit
def lsolve_sps(N: int, A: np.array, x: np.array, b: np.array) -> None:
    """
    Routine to solve Ax=b.

    Only the lower triangle of A is ever used (the rest need not even be allocated).
    The matrix A is destroyed.
 
    Parameters
    ----------
    N : int
        Number of 'node' eigenvalues.
    A : np.array, shape : (N, N)
        Positive definite matrix. Shape (`N`, `N`).
    x : np.array
        Vector, output. Shape (`N`,).
    b : np.array
        Vector. Shape (`N`,).

    Returns
    -------
    None

    """

    # Replace A with its Cholesky decomposition
    for i in range(N):
        for j in range(i):
            sum_ = 0.0
            for k in range(j):
                sum_ += A[i, k]*A[j, k]
            A[i, j] = (A[i, j]-sum_) / A[j, j]
        sum_ = 0.0
        for k in range(i):
            sum_ += A[i, k]*A[i, k]
        A[i, i] = np.sqrt(A[i, i]-sum_)
    # ... now the lower part of A is the Cholesky decomposition L: A = LL^T

    # now get p1 = LT-1 b
    p1 = np.empty((N,))
    for i in range(N):
        sum_ = 0.0
        for j in range(i):
            sum_ += A[i, j]*p1[j]
        p1[i] = (b[i]-sum_) / A[i, i]

    # ... and x = L^-1 p1
    for i in range(N-1, -1, -1):
        sum_ = 0.0
        for j in range(i+1, N):
            sum_ += A[j, i]*x[j]
        x[i] = (p1[i]-sum_) / A[i, i]


@njit
def build_reduced_T_wrap(Nflat: np.array, Dflat: np.array, Eflat: np.array, kappa: np.array,
                         ucmin: float, smax: float, out_kappa: np.array,
                         out_Sigma: np.array, out_UC: np.array, out_w: np.array) -> None:
    """
    Intermediate quantities to build coaddition matrix T from a reduced space.

    In the parameter descriptions, "m" refers to the number of output pixels and
    "nv" to the number of kappa nodes.

    Parameters
    ----------
    Nflat : np.array
        Input noise array. Shape (m*nv*nv,). Flattened version of (m,nv,nv).
    Dflat : np.array
        Input 1st order signal D/C. Shape (m*nv,). Flattened version of (m,nv).
    Eflat : np.array
        Input 2nd order signal E/C. Shape (m*nv*nv,). Flattened version of (m,nv,nv). 
    kappa : np.array
        List of eigenvalues, must be sorted ascending. Shape (nv,).
    ucmin : float
        Min U/C.
    smax : float
        Max Sigma (noise).
    out_kappa : np.array
        Output "kappa" parameter. Shape (m,).
    out_Sigma : np.array
        Output "Sigma". Shape (m,).
    out_UC : np.array
        Output "U/C". Shape (m,).
    out_w : np.array
        Output weights for each eigenvalue and each output pixel.
        Shape (m*nv,). Flattened version of (m,nv). 

    Returns
    -------
    None

    """

    # dimensions
    nv = kappa.size  # number of 'node' eigenvalues (must be >=2)
    m = out_kappa.size  # number of output pixels
    nv2 = nv*nv

    # allocate memory
    M2d = np.empty((nv, nv))
    w = np.empty((nv,))

    # loop over output pixels
    for a in range(m):
        # first figure out the range of kappa
        iv = nv-1
        UC = ucmin*10; S = smax/10  # do {...}
        while iv > 0 and UC > ucmin and S < smax:
            iv -= 1
            S = Nflat[a*nv2 + iv*(nv+1)]  # diagonal noises
            UC = 1.0 - 2.0*Dflat[a*nv + iv] + Eflat[a*nv2 + iv*(nv+1)]  # diagonal U/C

        # kappa should be in the range kappa[iv] .. kappa[iv+1]
        kappamid = np.sqrt(kappa[iv]*kappa[iv+1])
        factor = np.power(kappa[iv+1]/kappa[iv], 0.25)

        # iterative loop to find 'best' kappa
        for ik in range(12):
            # build matrix for this kappa
            for iv in range(nv):
                for jv in range(iv+1):
                    M2d[iv, jv] = Eflat[a*nv2 + iv+nv*jv] + kappamid*Nflat[a*nv2 + iv+nv*jv]
            # ... and get weights
            lsolve_sps(nv, M2d, w, Dflat[a*nv:(a+1)*nv])
            out_w[a*nv:(a+1)*nv] = w  # pointer to weights for this pixel

            # now get the UC and the S
            S = 0.0
            for iv in range(nv):
                sum_ = 0.0
                for jv in range(nv):
                    sum_ += Nflat[a*nv2 + iv+nv*jv] * w[jv]
                S += sum_ * w[iv]
            UC = 1.0 - kappamid*S
            for iv in range(nv):
                UC -= Dflat[a*nv + iv] * w[iv]

            # updates to kappa
            kappamid *= 1.0/factor if UC > ucmin and S < smax else factor
            factor = np.sqrt(factor)

        # output other information
        out_kappa[a] = kappamid
        out_Sigma[a] = S
        out_UC[a] = UC
