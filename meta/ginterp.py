import numpy as np
import sys
from astropy.io import fits

def InterpMatrix(Rsearch, samp, x_out, y_out, Cov, kappa=1.e-10, deweight=1.e24):
    """Constructs an interpolation matrix.

    Required Inputs:
    Rsearch = search radius (from corners)
    samp = sampling rate of input image (samples per FWHM)
    x_out = fractional pixel positions in x (0 to 1, inclusive), length Npts
    y_out = fractional pixel positions in y (0 to 1, inclusive), length Npts
    Cov = covariance matrix of extra smoothing. length 3, array-like [Cxx, Cxy, Cyy]

    Optional Inputs:
    kappa = regularization parameter to prevent singular matrices
    deweight = parameter to de-weight points outside the search radius

    Returns:
    posx = x positions of input pixels (int), length NN
    posy = y positions of input pixels (int), length NN
    T = interpolation/smoothing matrix (matrix, form of numpy 2D array), length Npts x NN
    U = fractional squared leakage
    Sigma = sum_i T_{ai}^2 (noise squared metric)

    Comments:
    This function actually has the same algorithm as IMCOM embedded in it. But the "system
    matrix" A is the same in all cases so it is much faster than IMCOM.
    """

    # extract parameters
    R = np.sqrt(np.ceil(Rsearch**2)+.01)
    N = int(np.ceil(R)+1)*2
    sigma = samp/np.sqrt(8*np.log(2))
    Npts = np.size(x_out)
    Cxx = float(Cov[0])
    Cxy = float(Cov[1])
    Cyy = float(Cov[2])

    # build mesh of input points
    posx1D = np.linspace(-(N//2)+1,N//2,N)
    posy1D = np.linspace(-(N//2)+1,N//2,N)
    posx,posy = np.meshgrid(posx1D,posy1D)
    posx = posx.flatten()
    posy = posy.flatten()
    g = np.where((posx-.5)**2+(posy-.5)**2<=(R+.5**2)**2)
    posx = posx[g]
    posy = posy[g]
    NN = np.size(posx)

    # system matrix
    A = np.identity(NN)
    for i in range(1,NN):
        for j in range(i):
            A[j,i] = A[i,j] = np.exp(-(posx[i]-posx[j])**2/4./sigma**2) * np.exp(-(posy[i]-posy[j])**2/4./sigma**2)

    # get overlap matrices
    detCT = (2*sigma**2+Cxx) * (2*sigma**2+Cyy) - Cxy**2
    iCTxx = (2*sigma**2+Cyy)/detCT
    iCTxy =            -Cxy /detCT
    iCTyy = (2*sigma**2+Cxx)/detCT
    dx = posx[:,None]-x_out[None,:]
    dy = posy[:,None]-y_out[None,:]
    ratio_sqrtdet = np.sqrt((sigma**2+Cxx)*(sigma**2+Cyy)-Cxy**2)/sigma**2
    b = np.exp(-.5*(iCTxx*dx**2 + 2*iCTxy*dx*dy + iCTyy*dy**2)) * 2*sigma**2/np.sqrt(detCT)

    # the interpolation matrix is built from each of the corners, and then interpolated.
    # this ensures continuity of the weights across cell boundaries
    T = np.zeros_like(b.T)
    xcorner = [0.,1.,0.,1.]
    ycorner = [0.,0.,1.,1.]
    weights = [ (1-x_out)*(1-y_out), x_out*(1-y_out), (1-x_out)*y_out, x_out*y_out] # list of arrays
    for icorner in range(4):
        wvec = np.zeros(NN)
        wvec[:]  = deweight
        g = np.where((posx-xcorner[icorner])**2+(posy-ycorner[icorner])**2<=R**2)
        wvec[g] = kappa
        Aw = A + np.diag(wvec)
        Tw = np.linalg.solve(Aw,b).T
        T[:,:] += Tw[:,:]*weights[icorner][:,None]
    T[:,:] /= np.sum(T,axis=1)[:,None]

    # want U[i] = 1./ratio_sqrtdet + np.dot(A@T[i,:]-2*b[:,i],T[i,:])
    U = 1./ratio_sqrtdet + np.sum( (T@A - 2*b.T)*T, axis=1)
    # Notes on this expression:
    # (T@A-2b.T)[i,j] = T[i,k]A[k,j]-2b[j,i]
    # then the sum is sum_j (T@A-2b.T)[i,j] * T[i,j] = sum_j T[i,k]A[k,j]T[i,j] -2 sum_j b[j,i]T[i,j]

    return np.round(posx).astype(np.int16), np.round(posy).astype(np.int16), T, U, np.sum(T**2,axis=1)

def MultiInterp(in_array, in_mask, out_size, out_origin, out_transform, Rsearch, samp, Cov, kappa=1.e-10, deweight=1.e24):
    """Interpolates from an input array to a regularly spaced output array.

    Required Inputs:
    in_array = array to interpolate from (may be 3D, with multiple layers)
    in_mask = Boolean mask for input array (True = masked; False = good)
    out_size = output array size, format: (ny,nx)
    out_origin, out_transform = length 2 vector and 2x2 matrix for mapping from input-->output coordinates
    Rsearch = search radius (from corners)
    samp = sampling rate of input image (samples per FWHM)
    Cov = covariance matrix of extra smoothing. length 3, array-like [Cxx, Cxy, Cyy]

    Optional Inputs:
    kappa = regularization parameter to prevent singular matrices
    deweight = parameter to de-weight points outside the search radius

    Returns:
    out_array (same number of layers as in_array)
    out_mask = Boolean mask for output array (True = masked; False = good)
    Umax = maximum leakage from the interpolation step
    Smax = maximum noise metric from the interpolation step

    Comments:
    The mapping between input and output coordinates is

    x_in = out_transform[0][0]*x_out + out_transform[0][1]*y_out + out_origin[0]
    y_in = out_transform[1][0]*x_out + out_transform[1][1]*y_out + out_origin[1]

    (both are 0-offset, C/Python style)
    """

    # get dimensions
    nlayer = 1
    is3D = False
    if len(np.shape(in_array))==3:
        nlayer = np.shape(in_array)[0]
        is3D = True
    ny_in = np.shape(in_array)[-2]
    nx_in = np.shape(in_array)[-1]
    # outputs
    ny = out_size[0]
    nx = out_size[1]

    # build output array
    out_array = np.zeros((nlayer,ny*nx), dtype=in_array.dtype)
    out_mask = np.ones((ny*nx,), dtype=bool) # default to masked
    Umax = 0.
    Smax = 0.

    # build output map in units of the block size
    blocksize = 2**19
    istart = 0
    while istart<ny*nx:
        ngroup = blocksize
        if ngroup+istart>ny*nx: ngroup=ny*nx-istart

        # get the pixel positions
        pixnum = np.arange(istart, istart+ngroup, dtype=np.int32)
        y_out = (pixnum//nx).astype(np.float64)
        x_out = (pixnum%nx).astype(np.float64)
        x_in = out_transform[0][0]*x_out + out_transform[0][1]*y_out + out_origin[0]
        y_in = out_transform[1][0]*x_out + out_transform[1][1]*y_out + out_origin[1]
        del pixnum; del x_out; del y_out

        # get fractional parts
        x_in_int = np.floor(x_in).astype(np.int32)
        y_in_int = np.floor(y_in).astype(np.int32)
        x_in_frac = x_in - x_in_int
        y_in_frac = y_in - y_in_int

        # get interpolation weights (T_) and the associated offsets in x_in_offset and y_in_offset
        x_in_offset, y_in_offset, T_, U_, S_ = InterpMatrix(Rsearch, samp, x_in_frac, y_in_frac, Cov, kappa=kappa, deweight=deweight)
        bb = max(-np.amin(x_in_offset), np.amax(x_in_offset-1), -np.amin(y_in_offset), np.amax(y_in_offset-1))
        if 2*bb>=min(nx_in,ny_in):
            break # this will return all zeros and mask everything
        Umax = max(Umax, np.amax(U_))
        Smax = max(Smax, np.amax(S_))

        # two layers to output mask.
        # (1) are the pixels we need in the input array?
        # (2) are they valid?
        # we'll answer question (1) here
        out_mask_sub = np.logical_or.reduce([x_in_int<bb, x_in_int+1+bb>=nx_in, y_in_int<bb, y_in_int+1+bb>=ny_in])
        # move these pixels to avoid reading off the edge of the array --- they are masked so this is OK
        x_in_int[out_mask_sub] = bb
        y_in_int[out_mask_sub] = bb

        for k in range(np.size(x_in_offset)):
            out_mask_sub |= in_mask[y_in_int+y_in_offset[k],x_in_int+x_in_offset[k]]
            if is3D:
                for j in range(nlayer):
                    out_array[j,istart:istart+ngroup] += T_[:,k]*in_array[j,y_in_int+y_in_offset[k],x_in_int+x_in_offset[k]]
            else:
                out_array[0,istart:istart+ngroup] += T_[:,k]*in_array[y_in_int+y_in_offset[k],x_in_int+x_in_offset[k]]
        out_mask[istart:istart+ngroup] = out_mask_sub

        istart += blocksize # move to next block

    # set masked values to zero
    for j in range(nlayer):
        out_array[j][out_mask] = 0.

    if is3D:
        out_array = out_array.reshape((nlayer,ny,nx))
    else:
        out_array = out_array.reshape((ny,nx))
    out_mask = out_mask.reshape((ny,nx))

    return out_array, out_mask, Umax, Smax

# This is a stand-alone test routine
def test():

    ### Test for InterpMatrix ###    
    ng = 17
    delta = np.linspace(0, 1., ng)
    x_out, y_out = np.meshgrid(delta,delta)
    x_out = x_out.flatten()
    y_out = y_out.flatten()
    x_in, y_in, T_, U_, S_ = InterpMatrix(6, 5., x_out, y_out, [.05,0,.025])
    print('# U:', np.amin(U_), np.amax(U_))
    print(np.vstack((x_in,y_in)))
    fits.PrimaryHDU(T_).writeto('T.fits', overwrite=True)

    ### Test for MultiInterp ###
    samp = 5.
    sigma = samp/np.sqrt(8*np.log(2))
    n = 1024
    x, y = np.meshgrid(np.linspace(0,n-1,n),np.linspace(0,n-1,n))
    nf = 4
    nf2 = 5
    u0 = .243; v0 = .128
    InArr = np.zeros((nf2,n,n), dtype=np.float32)
    for j in range(nf): InArr[j,:,:] = 1. + .1*np.cos(2*np.pi*(u0*x+v0*y)/2.**j)
    for k in range(128):
        xc = 500 + 400*np.cos(k/64*np.pi)
        yc = 500 + 400*np.sin(k/64*np.pi)
        InArr[-1,:,:] += np.exp(-.5*((x-xc)**2+(y-yc)**2)/sigma**2)
    InMask = np.zeros((n,n), dtype=bool)
    mat = [[.475,.005],[-.01,.45]]
    nout = 2048
    C = [.25,0,0]
    pos_offset = [6.,3.]
    OutArr, OutMask, Umax, Smax = MultiInterp(InArr, InMask, (nout,nout), pos_offset, mat, 6., samp, C)
    print('Umax =', Umax, 'Smax = ', Smax)
    fits.PrimaryHDU(OutArr).writeto('OutArr.fits', overwrite=True)

    TargetArr = np.zeros((nf,nout,nout))
    xo, yo = np.meshgrid(np.linspace(0,nout-1,nout),np.linspace(0,nout-1,nout))
    W = np.exp(-2*np.pi**2*(u0**2*C[0]+2*u0*v0*C[1]+v0**2*C[2]))
    tf0 = np.exp(-2*np.pi**2*(u0**2+v0**2)*sigma**2)
    for j in range(nf):
        print(j, tf0**(.25**j), W**(.25**j))
        TargetArr[j,:,:] = 1. + .1*np.cos(2*np.pi*((mat[0][0]*xo+mat[0][1]*yo+pos_offset[0])*u0+(mat[1][0]*xo+mat[1][1]*yo+pos_offset[1])*v0)/2.**j) * W**(.25**j)
    fits.PrimaryHDU(np.where(OutMask,0.,OutArr[:nf,:,:]-TargetArr).astype(np.float32)).writeto('DiffArr.fits', overwrite=True)
    return

if __name__ == "__main__":
    test()
