"""
Deconvolution-shear-reconvolution-resampling tools.

Functions
---------
InterpMatrix
    Constructs an interpolation matrix.
MultiInterp
    Performs the interpolation.
test
    A test function for InterpMatrix.

"""

import time

import numpy as np
import scipy
from astropy.io import fits


def InterpMatrix(Rsearch, samp, x_out, y_out, Cov, epsilon=1.0e-7, stest=1, verbose=False):
    """
    Constructs a reconvolution + interpolation matrix.

    Parameters
    ----------
    Rsearch : float
        Search radius (from corners), in gridded pixels.
    samp : float
        Sampling rate of input image (samples per FWHM).
    x_out : np.array of float
        Fractional pixel positions in x (0 to 1, inclusive), shape (Npts,).
    y_out : np.array of float
        Fractional pixel positions in y (0 to 1, inclusive), shape (Npts,).
    Cov : np.array of float
        Covariance matrix of extra smoothing. length 3, array-like [Cxx, Cxy, Cyy].
    epsilon : float, optional
        Regularization parameter to prevent singular correlations.
    stest : int, optional
        Computes diagnostics for every stest-th point instead of every point (default is every point).
        Saves time.
    verbose : bool, optional
        Print timing information?

    Returns
    -------
    posx : np.array of int
        x positions of input pixels, shape (NN,)
    posy : np.array of int
        y positions of input pixels, shape (NN,)
    T : np.array of float
        Interpolation/smoothing matrix, shape (Npts, NN).
    U : np.array of float
        Fractional squared leakage, shape (Npts,).
    Sigma : np.array of float
        Noise amplification = sum_i T_{ai}^2, shape (Npts,).

    Notes
    -----
    This function actually has the same algorithm as IMCOM embedded in it. But the "system
    matrix" A is the same in all cases so it is much faster than IMCOM.

    """

    t0 = time.time()

    # extract parameters
    R = np.sqrt(
        np.ceil(Rsearch**2) + 0.01
    )  # the 0.01 guarantees no 'edge cases' where a search depends on <= vs <
    N = int(np.ceil(R) + 1) * 2
    sigma = samp / np.sqrt(8 * np.log(2))
    # Npts = np.size(x_out)
    Cxx = float(Cov[0])
    Cxy = float(Cov[1])
    Cyy = float(Cov[2])

    # build mesh of input points
    posx1D = np.linspace(-(N // 2) + 1, N // 2, N)
    posy1D = np.linspace(-(N // 2) + 1, N // 2, N)
    posx, posy = np.meshgrid(posx1D, posy1D)
    posx = posx.flatten()
    posy = posy.flatten()
    g = np.nonzero((np.abs(posx - 0.5) - 0.5) ** 2 + (np.abs(posy - 0.5) - 0.5) ** 2 <= R**2)[
        0
    ]  # select points in search radius
    posx = posx[g]
    posy = posy[g]
    NN = np.size(posx)

    # system matrix
    sige = np.sqrt(1.0 / 2.0)
    A = np.identity(NN)
    for i in range(1, NN):
        for j in range(i):
            A[j, i] = A[i, j] = np.exp(-((posx[i] - posx[j]) ** 2) / 4.0 / sigma**2) * np.exp(
                -((posy[i] - posy[j]) ** 2) / 4.0 / sigma**2
            )
    # additional terms for regularization
    Ad = A + epsilon * np.identity(NN)
    for i in range(1, NN):
        for j in range(i):
            Ad[j, i] = Ad[i, j] = A[i, j] + epsilon * np.exp(
                -((posx[i] - posx[j]) ** 2) / 4.0 / sige**2
            ) * np.exp(-((posy[i] - posy[j]) ** 2) / 4.0 / sige**2)

    # get overlap matrices
    detCT = (2 * sigma**2 + Cxx) * (2 * sigma**2 + Cyy) - Cxy**2
    ratio_sqrtdet = np.sqrt((sigma**2 + Cxx) * (sigma**2 + Cyy) - Cxy**2) / sigma**2
    iCTxx = (2 * sigma**2 + Cyy) / detCT
    iCTxy = -Cxy / detCT
    iCTyy = (2 * sigma**2 + Cxx) / detCT

    # this is the simple algorithm, but we can do better
    # dx = posx[:,None]-x_out[None,:]
    # dy = posy[:,None]-y_out[None,:]
    # b = np.exp(-.5*(iCTxx*dx**2 + 2*iCTxy*dx*dy + iCTyy*dy**2)) * 2*sigma**2/np.sqrt(detCT)
    #
    # instead, we note that
    # .5*(iCTxx*dx**2 + 2*iCTxy*dx*dy + iCTyy*dy**2) = du**2 + dv**2
    # where we complete the square:
    # dv**2 = iCTyy/2.*(dy+iCTxy/iCTyy*dx)**2
    # du**2 = (iCTxx-iCTxy**2/iCTyy)/2.*dx
    a_ = np.sqrt((iCTxx - iCTxy**2 / iCTyy) / 2.0)
    c_ = np.sqrt(iCTyy / 2.0)
    m_ = iCTxy / iCTyy
    posu = a_ * posx
    u_out = a_ * x_out
    posv = c_ * (posy + m_ * posx)
    v_out = c_ * (y_out + m_ * x_out)
    du = posu[:, None] - u_out[None, :]
    dv = posv[:, None] - v_out[None, :]
    b = (2 * sigma**2 / np.sqrt(detCT)) * np.exp(-(du**2 + dv**2))

    # regularization
    bp = np.copy(b)
    detCT_ = (2 * sige**2 + Cxx) * (2 * sige**2 + Cyy) - Cxy**2
    iCTxx_ = (2 * sige**2 + Cyy) / detCT_
    iCTxy_ = -Cxy / detCT_
    iCTyy_ = (2 * sige**2 + Cxx) / detCT_
    a_ = np.sqrt((iCTxx_ - iCTxy_**2 / iCTyy_) / 2.0)
    c_ = np.sqrt(iCTyy_ / 2.0)
    m_ = iCTxy_ / iCTyy_
    posu = a_ * posx
    u_out = a_ * x_out
    posv = c_ * (posy + m_ * posx)
    v_out = c_ * (y_out + m_ * x_out)
    du = posu[:, None] - u_out[None, :]
    dv = posv[:, None] - v_out[None, :]
    bp += (epsilon * 2 * sige**2 / np.sqrt(detCT_)) * np.exp(-(du**2 + dv**2))

    # the interpolation matrix is built from each of the corners, and then interpolated.
    # this ensures continuity of the weights across cell boundaries
    TT = np.zeros_like(b)
    xcorner = [0.0, 1.0, 0.0, 1.0]
    ycorner = [0.0, 0.0, 1.0, 1.0]
    weights = [
        (1 - x_out) * (1 - y_out),
        x_out * (1 - y_out),
        (1 - x_out) * y_out,
        x_out * y_out,
    ]  # list of arrays
    for icorner in range(4):
        # get list of pixels needed for each corner
        g = np.nonzero((posx - xcorner[icorner]) ** 2 + (posy - ycorner[icorner]) ** 2 <= R**2)[0]
        # Cholesky decomposition
        if icorner == 0:
            Ad_ = Ad[g, :][:, g]
            cs = scipy.linalg.cho_factor(Ad_)
        #    # don't need to do this again, since it's always the same matrix
        # get T-transpose, since this is faster to generate
        # note overwrite_b is safe since b[g,:] is advanced indexing and always returns a copy
        TT[g, :] += scipy.linalg.cho_solve(cs, bp[g, :], check_finite=False) * weights[icorner][None, :]
        # TT[g,:] += np.linalg.solve(Ad[g,:][:,g],b[g,:]) * weights[icorner][None,:]
        # <-- this method was slower

    # normalize, transpose back
    T = TT.T / np.sum(TT, axis=0)[:, None]

    # want U[i] = 1./ratio_sqrtdet + np.dot(A@T[i,:]-2*b[:,i],T[i,:])
    U = 1.0 / ratio_sqrtdet + np.sum((T[::stest, :] @ A - 2 * b[:, ::stest].T) * T[::stest, :], axis=1)
    # Notes on this expression:
    # (T@A-2b.T)[i,j] = T[i,k]A[k,j]-2b[j,i]
    # then the sum is sum_j (T@A-2b.T)[i,j] * T[i,j] = sum_j T[i,k]A[k,j]T[i,j] -2 sum_j b[j,i]T[i,j]

    if verbose:
        print("InterpMatrix time =", time.time() - t0)
    return (
        np.round(posx).astype(np.int16),
        np.round(posy).astype(np.int16),
        T,
        U,
        np.sum(T[::stest, :] ** 2, axis=1),
    )


def MultiInterp(
    in_array,
    in_mask,
    out_size,
    out_origin,
    out_transform,
    Rsearch,
    samp,
    Cov,
    epsilon=1.0e-7,
    stest=1,
    blocksize=393216,
):
    """
    Interpolates from an input array to a regularly spaced output array, including some additional smoothing..

    Parameters
    ----------
    in_array : np.array of float
        Array to interpolate from (may be 3D, with multiple layers).
    in_mask : np.array of bool
        Boolean mask for input array (True = masked; False = good).
    out_size : (int, int)
        Output array size, format: (ny,nx).
    out_origin : np.array
        Length 2 vector of origin for mapping input-->output coordinates.
    out_transform : np.array
        Shape (2,2) matrix of Jacobian for mapping input-->output coordinates.
    Rsearch : float
        Search radius (from corners) in pixels in `in_array`.
    samp : float
        Sampling rate of input image (samples per FWHM).
    Cov : np.array
        Covariance matrix of extra smoothing. length 3, flattened array-like [Cxx, Cxy, Cyy].
    epsilon : float, optional
        Regularization parameter to prevent singular correlations.
    stest : int, optional
        Computes diagnostics for every stest-th point instead of every point (default is every point).
        Saves time.
    blocksize : int, optional
        Number of points to compute at once (larger is slightly faster but uses more memory).

    Returns
    -------
    out_array : np.array of float
        2D or 3D. Same number of layers as `in_array`.
    out_mask : np.array of bool
        Boolean mask for output array (True = masked; False = good).
    Umax : float
        Maximum leakage from the interpolation step.
    Smax : float
        Maximum noise metric from the interpolation step.

    Notes
    -----
    The mapping between input and output coordinates is, using `out_origin` and `out_transform`, ::

      x_in = out_transform[0][0]*x_out + out_transform[0][1]*y_out + out_origin[0]
      y_in = out_transform[1][0]*x_out + out_transform[1][1]*y_out + out_origin[1]

    Both are 0-offset, C/Python style.

    """

    # get dimensions
    nlayer = 1
    is3D = False
    if len(np.shape(in_array)) == 3:
        nlayer = np.shape(in_array)[0]
        is3D = True
    ny_in = np.shape(in_array)[-2]
    nx_in = np.shape(in_array)[-1]
    # outputs
    ny = out_size[0]
    nx = out_size[1]

    # build output array
    out_array = np.zeros((nlayer, ny * nx), dtype=in_array.dtype)
    out_mask = np.ones((ny * nx,), dtype=bool)  # default to masked
    Umax = 0.0
    Smax = 0.0

    # build output map in units of the block size
    istart = 0
    while istart < ny * nx:
        ngroup = blocksize
        if ngroup + istart > ny * nx:
            ngroup = ny * nx - istart

        # get the pixel positions
        pixnum = np.arange(istart, istart + ngroup, dtype=np.int32)
        y_out = (pixnum // nx).astype(np.float64)
        x_out = (pixnum % nx).astype(np.float64)
        x_in = out_transform[0][0] * x_out + out_transform[0][1] * y_out + out_origin[0]
        y_in = out_transform[1][0] * x_out + out_transform[1][1] * y_out + out_origin[1]
        del pixnum
        del x_out
        del y_out

        # get fractional parts
        x_in_int = np.floor(x_in).astype(np.int32)
        y_in_int = np.floor(y_in).astype(np.int32)
        x_in_frac = x_in - x_in_int
        y_in_frac = y_in - y_in_int

        # get interpolation weights (T_) and the associated offsets in x_in_offset and y_in_offset
        x_in_offset, y_in_offset, T_, U_, S_ = InterpMatrix(
            Rsearch, samp, x_in_frac, y_in_frac, Cov, stest=stest
        )
        bb = max(
            -np.amin(x_in_offset), np.amax(x_in_offset - 1), -np.amin(y_in_offset), np.amax(y_in_offset - 1)
        )
        if 2 * bb >= min(nx_in, ny_in):
            break  # this will return all zeros and mask everything
        Umax = max(Umax, np.amax(U_))
        Smax = max(Smax, np.amax(S_))
        del U_, S_

        # two layers to output mask.
        # (1) are the pixels we need in the input array?
        # (2) are they valid?
        # we'll answer question (1) here
        out_mask_sub = np.logical_or.reduce(
            [x_in_int < bb, x_in_int + 1 + bb >= nx_in, y_in_int < bb, y_in_int + 1 + bb >= ny_in]
        )
        # move these pixels to avoid reading off the edge of the array --- they are masked so this is OK
        x_in_int[out_mask_sub] = bb
        y_in_int[out_mask_sub] = bb

        for k in range(np.size(x_in_offset)):
            out_mask_sub |= in_mask[y_in_int + y_in_offset[k], x_in_int + x_in_offset[k]]
            if is3D:
                for j in range(nlayer):
                    out_array[j, istart : istart + ngroup] += (
                        T_[:, k] * in_array[j, y_in_int + y_in_offset[k], x_in_int + x_in_offset[k]]
                    )
            else:
                out_array[0, istart : istart + ngroup] += (
                    T_[:, k] * in_array[y_in_int + y_in_offset[k], x_in_int + x_in_offset[k]]
                )
        out_mask[istart : istart + ngroup] = out_mask_sub

        istart += blocksize  # move to next block

    # set masked values to zero
    for j in range(nlayer):
        out_array[j][out_mask] = 0.0

    if is3D:  # I think it is clearer this way. # noqa: SIM108
        out_array = out_array.reshape((nlayer, ny, nx))
    else:
        out_array = out_array.reshape((ny, nx))
    out_mask = out_mask.reshape((ny, nx))

    return out_array, out_mask, Umax, Smax


# This is a stand-alone test routine
def test():
    """Test for InterpMatrix."""

    ### Test for InterpMatrix ###
    ng = 17
    delta = np.linspace(0, 1.0, ng)
    x_out, y_out = np.meshgrid(delta, delta)
    x_out = x_out.flatten()
    y_out = y_out.flatten()
    x_in, y_in, T_, U_, S_ = InterpMatrix(6, 5.0, x_out, y_out, [0.05, 0, 0.025])
    print("# U:", np.amin(U_), np.amax(U_))
    print(np.vstack((x_in, y_in)))
    fits.PrimaryHDU(T_).writeto("T.fits", overwrite=True)
    print("Asymmetries:")
    NN = len(x_in)
    hflip = np.zeros(NN, dtype=np.int16)
    for i in range(NN):
        for j in range(NN):
            if x_in[i] == 1 - x_in[j] and y_in[i] == y_in[j]:
                hflip[i] = j
    for j in range(ng // 2):
        print(np.amax(np.abs(T_[j, :] - T_[-1 - j, ::-1])), np.amax(np.abs(T_[j, :] - T_[ng - 1 - j, hflip])))

    ### Test for MultiInterp ###
    samp = 4.5
    sigma = samp / np.sqrt(8 * np.log(2))
    n = 1024
    x, y = np.meshgrid(np.linspace(0, n - 1, n), np.linspace(0, n - 1, n))
    nf = 4
    nf2 = 6
    u0 = 0.243
    v0 = 0.128
    InArr = np.zeros((nf2, n, n), dtype=np.float32)
    for j in range(nf):
        InArr[j, :, :] = 1.0 + 0.1 * np.cos(2 * np.pi * (u0 * x + v0 * y) / 2.0**j)
    InArr[-2, :, :] = np.sum(InArr[:-2, :, :], axis=0) - 3.6
    for k in range(128):
        xc = 500 + 400 * np.cos(k / 64 * np.pi)
        yc = 500 + 400 * np.sin(k / 64 * np.pi)
        InArr[-1, :, :] += np.exp(-0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / sigma**2)
    InMask = np.zeros((n, n), dtype=bool)
    mat = np.array([[0.52, 0.005], [-0.015, 0.51]])
    sc = 0.5
    nout = 1950
    eC = (mat @ mat.T / sc**2 - np.identity(2)) * sigma**2
    C = [eC[0, 0], eC[0, 1], eC[1, 1]]
    pos_offset = [6.0, 3.0]
    OutArr, OutMask, Umax, Smax = MultiInterp(
        InArr, InMask, (nout, nout), pos_offset, mat, 6.0, samp, C, stest=100
    )
    print("Umax =", Umax, "Smax = ", Smax)
    fits.PrimaryHDU(InArr).writeto("InArr.fits", overwrite=True)
    fits.PrimaryHDU(OutArr).writeto("OutArr.fits", overwrite=True)

    TargetArr = np.zeros((nf2, nout, nout))
    xo, yo = np.meshgrid(np.linspace(0, nout - 1, nout), np.linspace(0, nout - 1, nout))
    W = np.exp(-2 * np.pi**2 * (u0**2 * C[0] + 2 * u0 * v0 * C[1] + v0**2 * C[2]))
    tf0 = np.exp(-2 * np.pi**2 * (u0**2 + v0**2) * sigma**2)
    for j in range(nf):
        print(j, tf0 ** (0.25**j), W ** (0.25**j))
        TargetArr[j, :, :] = 1.0 + 0.1 * np.cos(
            2
            * np.pi
            * (
                (mat[0][0] * xo + mat[0][1] * yo + pos_offset[0]) * u0
                + (mat[1][0] * xo + mat[1][1] * yo + pos_offset[1]) * v0
            )
            / 2.0**j
        ) * W ** (0.25**j)
    TargetArr[-2, :, :] = np.sum(TargetArr[:-2, :, :], axis=0) - 3.6
    for k in range(128):
        xc = 500 + 400 * np.cos(k / 64 * np.pi)
        yc = 500 + 400 * np.sin(k / 64 * np.pi)
        v = np.array([xc, yc]) - pos_offset
        tt = np.linalg.solve(mat, v)
        xt = tt[0]
        yt = tt[1]
        TargetArr[-1, :, :] += np.exp(
            -0.5 * ((xo - xt) ** 2 + (yo - yt) ** 2) / (sigma / sc) ** 2
        ) / np.linalg.det(mat / sc)
    fits.PrimaryHDU(np.where(OutMask, 0.0, OutArr - TargetArr).astype(np.float32)).writeto(
        "DiffArr.fits", overwrite=True
    )
    return


if __name__ == "__main__":
    test()
