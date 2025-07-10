"""This file contains the PyIMCOM_WCS class, which allows PyIMCOM to take in a WCS either as a
FITS header or a GWCS from an ASDF file.

PyIMCOM_WCS supports the all_pix2world and all_world2pix methods, which are like their astropy
counterparts. A separate local_partial_pixel_derivatives2 function was written so that we don't
need to rely on astropy's partial derivative function (which has singular behavior near the poles).
"""

import numpy as np
import sys
from copy import deepcopy
from astropy.io import fits
import asdf
import astropy.wcs
from scipy.interpolate import RegularGridInterpolator
import gwcs

from .config import Settings

### === UTILITIES FOR APPROXIMATING A GWCS ===

class ABasis:
    """Base class for basis polynomials.

    Want to use inherited classes that replace the coef_setup method.

    Attributes:
    p_order : max order of the polynomial
    N : side length of grid the polynomial is evaluated on
    N_basis : number of basis modes
    coefs : coefficient matrix: coefs[i,j,k] is the coefficient of u^i v^j in the kth polynomial
    eval : evaluation of polynomials
    """

    def __init__(self,p_order,N):
        self.p_order = p_order
        self.N = N
        self.N_basis = ((p_order+1)*(p_order+2))//2
        self.coefs = np.zeros((p_order+1, p_order+1, self.N_basis))
        self.coef_setup()

    def coef_setup(self):
        pass # shouldn't get here, always use the methods from the inherited classes

    def eval(self,u,v):
        """Takes in an array of u and v, and returns a 2D array out[k,l] = kth poly evaluated at lth point.

        Could be replaced with a more stable version for specific types of polynomials if provided by that basis.
        """

        out = np.zeros((self.N_basis, np.size(u)))
        for i in range(self.p_order+1):
            for j in range(self.p_order+1-i):
                out[:,:] += np.outer(self.coefs[i,j,:], u**i*v**j)

        return out

class SimpleBasis(ABasis):
    """Simple polynomials (each coefficient is 1)"""

    def coef_setup(self):
        self.basis = 'simple'
        k = 0
        for i in range(self.p_order+1):
            for j in range(self.p_order+1-i):
                self.coefs[i,j,k] = 1.
                k += 1

### WCS classes ###

class LocWCS:

    """WCS that can be reported in various formats.

    Methods:
    __init__ : constructor from gwcs
    wcs_approx_sip : build approximate TAN-SIP WCS
    _make_errmap : report the error map
    err_interp : makes error map interpolator for the TAN-SIP approximation

    Attributes:
    gwcs : the generalized WCS
    N : side length of the array (probably 4088)
    ra_ctr,dec_ctr : coordinates of center
    uEast, uNorth : 3-component unit vectors East and North
    J : 2x2 Jacobian from detector to tangent plane coordinates in *radians*
    approx_wcs : the best-fit astropy wcs (if set)
    max_wcs_err : the worst error (in pixels) from the approx_wcs (if set)
    errmap : error map (in pixels) from the approx_wcs (if set)
    """

    def __init__(self,gwcs,N=4088):
        self.gwcs = gwcs
        self.N = N

        # get the centroid
        # positions: center, left, right, bottom, top
        ra,dec = gwcs(( (N-1)/2., 0, N-1, (N-1)/2., (N-1)/2.), ( (N-1)/2., (N-1)/2., (N-1)/2., 0, N-1))
        degree = np.pi/180.
        x  = np.zeros((5,3))
        x[:,0] = np.cos(dec*degree)*np.cos(ra*degree)
        x[:,1] = np.cos(dec*degree)*np.sin(ra*degree)
        x[:,2] = np.sin(dec*degree)
        self.ra_ctr = ra[0]
        self.dec_ctr = dec[0]
        self.uEast = np.array( [ -np.sin(ra[0]*degree), np.cos(ra[0]*degree), 0. ] )
        self.uNorth = np.array( [ -np.sin(dec[0]*degree)*np.cos(ra[0]*degree), -np.sin(dec[0]*degree)*np.sin(ra[0]*degree), np.cos(dec[0]*degree) ])
        # Jacobian based on the above
        self.J = np.zeros((2,2))
        self.J[0,0] = np.dot(self.uEast,  x[2,:]-x[1,:])/(N-1)
        self.J[0,1] = np.dot(self.uEast,  x[4,:]-x[3,:])/(N-1)
        self.J[1,0] = np.dot(self.uNorth, x[2,:]-x[1,:])/(N-1)
        self.J[1,1] = np.dot(self.uNorth, x[4,:]-x[3,:])/(N-1)

        #print(self.ra_ctr, self.dec_ctr)
        #print(self.uEast)
        #print(self.uNorth)
        #print(self.J/degree*3600/.11)

    def wcs_approx_sip(self, p_order=3, nq=100, basis='simple', verbose=False):
        """Generate approximate TAN-SIP polynomial wcs, and store as self.wcs.

        Parameters:
        p_order : order of polynomial to fit
        nq : grid size for fitting WCS (nq x nq)
        basis : type of basis to use in fitting the WCS
        verbose : talk a lot? (bool)

        The following basis sets for the linear algerbra are available:
        simple : simple polynomials (could be unstable for high order)
        """

        N = self.N
        xx,yy = np.meshgrid(np.array(range(N)), np.array(range(N)))
        ra, dec = self.gwcs(xx.flatten(), yy.flatten())
        degree = np.pi/180.

        # make generic FITS header
        hdu = fits.PrimaryHDU(np.zeros((N,N),dtype=np.int8))
        header = hdu.header
        header['CTYPE1']  = 'RA---TAN-SIP'
        header['CTYPE2']  = 'DEC--TAN-SIP'
        header['CRVAL1']  = self.ra_ctr
        header['CRVAL2']  = self.dec_ctr
        header['CD1_1']   = self.J[0,0]/degree
        header['CD1_2']   = self.J[0,1]/degree
        header['CD2_1']   = self.J[1,0]/degree
        header['CD2_2']   = self.J[1,1]/degree
        header['CRPIX1']  = (N+1)/2.
        header['CRPIX2']  = (N+1)/2.
        header['A_ORDER'] = p_order
        header['B_ORDER'] = p_order

        # now we want to get the distortion
        # U = u + f(u,v)
        # V = v + g(u,v)
        # where (U,V) would map to the ideal location with the TAN projection
        # we define dU = U-u and dV = V-v

        # first get all the coordinates
        q = np.linspace(0, N-1, nq)
        xx,yy = np.meshgrid(q,q)
        ra, dec = self.gwcs(xx.flatten(), yy.flatten())
        u_ = xx.flatten() - (N-1)/2.
        v_ = yy.flatten() - (N-1)/2.
        del xx,yy
        x  = np.zeros((nq**2,3))
        x[:,0] = np.cos(dec*degree)*np.cos(ra*degree)
        x[:,1] = np.cos(dec*degree)*np.sin(ra*degree)
        x[:,2] = np.sin(dec*degree)
        del ra,dec
        dec_ctr = self.dec_ctr; ra_ctr = self.ra_ctr
        pc = np.array( [ np.cos(dec_ctr*degree)*np.cos(ra_ctr*degree), np.cos(dec_ctr*degree)*np.sin(ra_ctr*degree), np.sin(dec_ctr*degree) ] )
        tan_x = (x@self.uEast)/(x@pc)
        tan_y = (x@self.uNorth)/(x@pc)
        del x
        detJ = self.J[0,0]*self.J[1,1] - self.J[0,1]*self.J[1,0]
        dU = ( self.J[1,1]*tan_x -self.J[0,1]*tan_y)/detJ - u_
        dV = (-self.J[1,0]*tan_x +self.J[0,0]*tan_y)/detJ - v_

        if verbose:
            print('means and standard deviations of U and V offsets:')
            print(np.mean(dU), np.std(dU))
            print(np.mean(dV), np.std(dV))

        # basis function choices
        if basis.lower()=='simple':
            MyBasis = SimpleBasis(p_order,N)

        # test
        if verbose:
            print('basis coefficients:')
            print(MyBasis.coefs)

        M = MyBasis.eval(u_, v_) # warning: huge if nq is big! 2 GB for 4088**2 pixels, and 4th order
        af = np.zeros((MyBasis.N_basis,))
        ag = np.zeros((MyBasis.N_basis,))

        # theoretically, this converges in 1 step;
        # but at finite numerical precision, a few steps is better.
        for iter in range(4):
            if verbose:
                print(M@(dU-M.T@af))
            af[:] += np.linalg.solve(M@M.T, M@(dU-M.T@af) )
            ag[:] += np.linalg.solve(M@M.T, M@(dV-M.T@ag) )

        # insert fitting
        for i in range(p_order+1):
            for j in range(p_order+1-i):
                header['A_{:1d}_{:1d}'.format(i,j)] = np.dot(MyBasis.coefs[i,j,:],af)
                header['B_{:1d}_{:1d}'.format(i,j)] = np.dot(MyBasis.coefs[i,j,:],ag)
        self.approx_wcs = astropy.wcs.WCS(header)

        # get the worst error in pixels
        err = np.hypot(dU-M.T@af, dV-M.T@ag)
        self.wcs_max_err = np.amax(err)
        # and save the error map
        self._make_errmap()

    def _make_errmap(self):
        """Makes a (2,N,N)-shaped error map.

        Format is: errmap[0,j,i] is the x-offset of pixel (i,j); errmap[1,j,i] is the y-offset.
        offset is in the sense of if there is a barred coordinate system that is the TAN-SIP approximation
        and unbarred is the true system, then
        xbar = x + errmap[0,y,x]
        ybar = y + errmap[1,y,x]

        """

        if not hasattr(self, 'approx_wcs'):
            raise TypeError("Missing approximated WCS")

        N = self.N
        x,y = np.meshgrid(np.linspace(0,N-1,N),np.linspace(0,N-1,N))
        ra, dec = self.gwcs(x.ravel(), y.ravel())
        xbar,ybar = self.approx_wcs.all_world2pix(ra,dec,0)
        del ra,dec
        d = np.zeros((2,N,N), dtype=np.float32)
        d[0,:,:] = (xbar.reshape((N,N))-x).astype(np.float32)
        d[1,:,:] = (ybar.reshape((N,N))-y).astype(np.float32)
        self.errmap = d

    def err_interp(self, a=4, n_pad = 2048):
        """Makes interpolators for the delta x = xbar-x and delta y = ybar-y directions.

        The functions returned are of the form dX(arr), where arr is an array of shape (K,2)
        indicating K points. arr[:,0] is the y-values, and arr[:,1] is the x-values.

        The linear extrapolation is done from a pixels from each edge.
        """

        # spacings
        N = self.N
        coords = np.pad(np.linspace(0,N-1,N),1)
        d_ = np.pad(self.errmap, ( (0,0), (1,1), (1,1) ) )

        # fill in padding values at n_pad values from the edge of the array
        coords[0] = -n_pad
        coords[-1] = N+n_pad-1
        d_[:,:,0] = d_[:,:,1] + n_pad/a*(d_[:,:,1] - d_[:,:,1+a])
        d_[:,:,-1] = d_[:,:,-2] + n_pad/a*(d_[:,:,-2] - d_[:,:,-2-a])
        d_[:,0,:] = d_[:,1,:] + n_pad/a*(d_[:,1,:] - d_[:,1+a,:])
        d_[:,-1,:] = d_[:,-2,:] + n_pad/a*(d_[:,-2,:] - d_[:,-2-a,:])

        # and make the interpolator
        return (
            RegularGridInterpolator((coords,coords), d_[0,:,:], method='linear', bounds_error=False, fill_value=None),
            RegularGridInterpolator((coords,coords), d_[1,:,:], method='linear', bounds_error=False, fill_value=None)
        )

### === END UTILITIES ===

class PyIMCOM_WCS:
    """Class that has the key methods we depend on from astropy.wcs,
    but can be constructed from other types of WCS information.

    Methods:
    ----------
    __init__ : constructor
    all_pix2world : pixel -> world coordinates (astropy-like)
    all_world2pix : world -> pixel coordinates (astropy-like)

    Attributes:
    -------------
    constructortype : what type of input was used to make this object
    type : what type of method to use in computation (currently ASTROPY or GWCS)
    obj : the WCS object being wrapped
    err : for 'ASTROPY+', contains an interpolator for the error
    """

    def __init__(self, inwcs, noconvert=False):
        """Constructor. Selects from the possible types of input WCS objects.

        Possible input methods, tested in order:
        astropy.fits Header
        astropy.wcs object

        If noconvert is True, then does not try to convert a GWCS object into internal formats.
        """

        if isinstance(inwcs, fits.Header):
            self.constructortype = 'FITSHEADER'
            self.type = 'ASTROPY'
            self.obj = astropy.wcs.WCS(inwcs)
            return

        if isinstance(inwcs, astropy.wcs.WCS):
            self.constructortype = 'ASTROPY'
            self.type = 'ASTROPY'
            self.obj = inwcs
            return

        if isinstance(inwcs, gwcs.wcs.WCS):
            self.constructortype = 'GWCS'
            if noconvert:
                self.type = 'GWCS'
                self.obj = deepcopy(inwcs)
                self.obj.bounding_box = None # remove this since for derivatives
                                             # we need to go off the edge
                return
            # GWCS input, but can convert
            self.type = 'ASTROPY+' # '+' indicates with correction
            w = LocWCS(inwcs, N=Settings.sca_nside)
            w.wcs_approx_sip(p_order=2)
            self.obj = w.approx_wcs
            self.err = w.err_interp(a=8, n_pad = Settings.sca_nside//2) # dX,dY
            return

        # get here if nothing above works
        raise TypeError('Unrecognized WCS type.')

    def _all_pix2world(self, pos, origin):
        """Following astropy convention:
        arguments are (N,2) array, origin
        """

        if self.type=='ASTROPY':
            return self.obj.all_pix2world(pos,origin)

        if self.type=='ASTROPY+':
            dp = np.vstack(( self.err[0](pos[::-1,:]), self.err[1](pos[::-1,:]) )).T.astype(np.float64)
            return self.obj.all_pix2world(pos+dp,origin)

        if self.type=='GWCS':
            pos = np.array(pos)
            ra, dec = self.obj.pixel_to_world_values(pos[:,0]-origin, pos[:,1]-origin)
            return np.vstack((ra,dec)).T

    def all_pix2world(self, *args):
        """This version also allows 3-argument format with 1D arrays or scalars"""
        if len(args)==2: return self._all_pix2world(np.array(args[0]),args[1])
        o = self._all_pix2world(np.vstack((args[0],args[1])).T, args[2])
        if isinstance(args[0],np.ndarray):
            return o[:,0], o[:,1]
        else:
            return o[0,0], o[0,1]

    def _all_world2pix(self, pos, origin):
        """Following astropy convention:
        arguments are (N,2) array, origin
        """

        if self.type=='ASTROPY':
            return self.obj.all_world2pix(pos,origin)

        if self.type=='ASTROPY+':
            pos2 = self.obj.all_world2pix(pos,origin)
            pos1 = np.copy(pos2)
            # 3 iterations is overkill but we want to be sure.
            # also we want to avoid slight discontinuities
            for k in range(3):
                dp = np.vstack(( self.err[0](pos1[::-1,:]), self.err[1](pos1[::-1,:]) )).T.astype(np.float64)
                pos1 = pos2-dp
            return pos1

        if self.type=='GWCS':
            pos = np.array(pos)
            x,y = self.obj.world_to_pixel_values(pos[:,0], pos[:,1])
            return np.vstack((x,y)).T + origin

    def all_world2pix(self, *args):
        """This version also allows 3-argument format with 1D arrays or scalars"""
        if len(args)==2: return self._all_world2pix(np.array(args[0]),args[1])
        o = self._all_world2pix(np.vstack((args[0],args[1])).T, args[2])
        if isinstance(args[0],np.ndarray):
            return o[:,0], o[:,1]
        else:
            return o[0,0], o[0,1]

def local_partial_pixel_derivatives2(inwcs,x,y):
    """Alternative form of the local partial derivatives function
    that is well-behaved near the poles and uses 2-sided derivatives.

    Arguments:
    ----------
    inwcs : input wcs (astropy or with compatible all_pix2world method)
    x : x position in pixels (0 offset)
    y : y position in pixels (0 offset)

    Returns:
    --------
    jac : 2x2 Jacobian matrix, with output 0->West and 1->North

    Comments:
    ---------
    This is relative to unit vectors, so jac[0,:] is -cos(declination) * d(ra)/d(pix x or y).
    So note this is different from astroy local_partial_pixel_derivatives2, which doesn't
    have the factor of -cos(declination).
    """

    # choose grid of positions for the numerical derivative
    x_ = x + np.array([0,1,-1,3,-3,0,0,0,0])
    y_ = y + np.array([0,0,0,0,0,1,-1,3,-3])

    # now get the RA and Dec
    degree = np.pi/180.
    pos_world = inwcs.all_pix2world(np.vstack((x_,y_)).T, 0)
    ra_ = pos_world[:,0]*degree
    dec_ = pos_world[:,1]*degree

    # convert to "East" and "North" directions
    p = np.zeros((2,np.size(x_)))
    p[0,:] = np.cos(dec_)*np.sin(ra_[0]-ra_)
    p[1,:] = np.sin(dec_)*np.cos(dec_[0]) - np.cos(dec_)*np.sin(dec_[0])*np.cos(ra_[0]-ra_)

    # now get the Jacobian
    jac = np.zeros((2,2))
    for j in [0,1]:
        # uses 4-point derivative formula
        subvec = p[:,1+4*j:5+4*j] # offsets: 1,-1,3,-3
        jac[:,j] = (27*(subvec[:,0]-subvec[:,1])-(subvec[:,2]-subvec[:,3]))/48.

    return(jac/degree) # output in degrees, not radians for consistency with astropy function

def _stand_alone_test(infile):
    """Simple tests of the above routines. Can be either an L2 ASDF file or FITS with the
    WCS in the primary HDU.
    """

    if infile[-5:]=='.asdf':
        with asdf.open(infile) as f:
            wcsobj = PyIMCOM_WCS(f['roman']['meta']['wcs'])

    if infile[-5:]=='.fits':
        with fits.open(infile) as f:
            wcsobj = PyIMCOM_WCS(f[0].header)

    inpos = np.zeros((9,2))
    inpos[3:6,1] = 2043.5
    inpos[6:,1] = 4087
    inpos[1::3,0] = 2043.5
    inpos[2::3,0] = 4087
    print(inpos)

    skycoord = wcsobj.all_pix2world(inpos,0)
    print(skycoord)

    recovered = wcsobj.all_world2pix(skycoord,0)
    print(recovered)

    jac = local_partial_pixel_derivatives2(wcsobj,0.,0.)
    print(jac*3600)
    print(np.linalg.det(jac*3600))

if __name__ == "__main__":
    _stand_alone_test(sys.argv[1])
