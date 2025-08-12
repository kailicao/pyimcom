"""
Utilities to generate additional layers and handle masks.

Classes
-------
GalSimInject
    Utilities to inject objects using GalSim.
GridInject
    Utilities to inject stars using furry-parakeet C routine.
CplxNoise
    Utilities to generate 1/f noise.
Mask
    Utilities for permanent and cosmic ray masks.

Functions
---------
_get_sca_imagefile
    Returns path to required SCA image file.
check_if_idsca_exists
    Determines whether an observation (id,sca) pair exists.
get_all_data
    Makes a 3D array of the image data.

"""

from os.path import exists
import re
import sys
import time
import warnings
from filelock import Timeout, FileLock

import numpy as np
import scipy.linalg
from scipy.ndimage import convolve

import asdf
from astropy.io import fits
from astropy import wcs

import galsim
import healpy

from .config import Settings as Stn, fpaCoords
try:
    from pyimcom_croutines import iD5512C
except:
    from .routine import iD5512C

from .wcsutil import local_partial_pixel_derivatives2, PyIMCOM_WCS

class GalSimInject:
    """
    Utilities to inject objects on a HEALPix grid using GalSim into an SCA image.

    Methods
    -------
    galsim_star_grid
        Makes a grid of stars (staticmethod).
    subgen
        Helper function to generate a subsequence of a random number set (staticmethod).
    subgen_multirow
        Helper function to generate multiple subsequences of a random number set (staticmethod).
    genobj
        Generates parameters for random objects on a grid (staticmethod).
    galsim_extobj_grid
        Makes a grid of galaxies (staticmethod).

    Notes
    -----
    The random number tools are designed to consistently generate the same objects when called with different SCAs
    (so a different set of HEALPix pixels is chosen).

    Based on fluffy-garbanzo/inject_galsim_obj.py (fluffy-garbanzo is now deprecated).

    """

    @staticmethod
    def galsim_star_grid(res, mywcs, inpsf, idsca, obsdata, sca_nside, inpsf_oversamp, extraargs=None):
        """
        Draws a grid of stars on an SCA image.

        Parameters
        ----------
        res : int
            HEALPix resolution (nside = 2**res).
        mywcs : astropy.wcs.WCS or pyimcom.wcsutil.PyIMCOM_WCS
            The WCS object for the SCA.
        inpsf : function
            A function that takes a position and returns a PSF (normally an InImage.get_psf_pos method).
        idsca : (int,int)
            Tuple of observation ID and SCA number (SCA in 1..18)
        obsdata : astropy.io.fits.FITS_rec
            Observation table (needed for some data formats).
        sca_nside : int
            Side length of the SCA (4088 for Roman).
        extraargs : dict or None, optional
            If a dictionary, then contains more parameters to pass to some of the functions..

        Returns
        -------
        np.array
            An array of shape (nside,nside) containing the image of all the objects drawn.

        Notes
        -----
        The following parameters are allowed as keys in `extraargs`:

        * angleTransient : bool

          If True, then includes a transient source that is on or off
          depending on the roll angle (maps to time of year).
          Odd pixels are on for PA=0, even for PA=180.

        * FieldDependentModulation : float

          If provided, then changes the apparent flux of the drawn stars depending on where
          they are in the field. The flux varies from 1 at the center of the focal plane up
          to 1 + FieldDependentModulation at the corners.

        """

        # extract the extraargs
        #
        angleTransient = False
        FieldDependentModulation = False
        if isinstance(extraargs, dict):
            # angleTransient: transient depending on roll angle?
            if 'angleTransient' in extraargs.keys(): angleTransient = extraargs['angleTransient']
            if angleTransient:
                # need to know whether the image points 'up' or 'down'
                ra1,dec1 = mywcs.all_pix2world(.5*(fpaCoords.nside-1), fpaCoords.nside-1. ,0)
                ra2,dec2 = mywcs.all_pix2world(.5*(fpaCoords.nside-1), 0. ,0)
                s = 0
                if dec2>dec1: s=1
                if idsca[1]%3==0: s=1-s # top row of SCAs is flipped
                print('.. idsca', idsca, 'ddec', dec1-dec2, 'direction', s, '# 0 for PA 0, 1 for PA 180')

            # FieldDependentModulation: change intensity depending on distance from field center?
            if 'FieldDependentModulation' in extraargs.keys():
               FieldDependentModulation = True
               FieldDependentModulationAmplitude = extraargs['FieldDependentModulation']

        ra_cent, dec_cent = mywcs.all_pix2world((sca_nside-1)/2, (sca_nside-1)/2, 0)

        search_radius = (sca_nside * 0.11)/3600*(np.pi/180.)*np.sqrt(2)
        vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
        qp = healpy.query_disc(2**res, vec, search_radius, nest=True)
        ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=True, lonlat=True)

        # convert to SCA coordinates
        x_sca, y_sca = mywcs.all_world2pix(ra_hpix, dec_hpix, 0)
        d = 16
        msk_sca = ((x_sca >= -d) & (x_sca <= 4087+d) &
                   (y_sca >= -d) & (y_sca <= 4087+d))
        x_sca = x_sca[msk_sca]
        y_sca = y_sca[msk_sca]
        my_ra = ra_hpix[msk_sca]
        my_dec = dec_hpix[msk_sca]
        num_obj = len(x_sca)

        n_in_stamp = 280
        pad = n_in_stamp+2*(d+1)
        sca_image = galsim.ImageF(sca_nside+pad, sca_nside+pad, scale=0.11)

        firsttime = True
        for n in range(num_obj):

            # if angle transient mode is on, check if we really need this object
            if angleTransient:
                if (qp[n]+s)%2==1: continue

            psf = inpsf((my_ra[n], my_dec[n]))  # now with PSF variation
            psf_image = galsim.Image(psf, scale=0.11/inpsf_oversamp)
            # interp_psf = galsim.InterpolatedImage(psf_image, x_interpolant='lanczos32')
            # the first time, get the preferred stepk and maxk
            if firsttime:
                interp_psf_test = galsim.InterpolatedImage(psf_image, x_interpolant='lanczos32')
                stepk = interp_psf_test.stepk
                maxk = interp_psf_test.maxk
                del interp_psf_test
                firsttime = False
            psf_image.setCenter(0,0)
            interp_psf = galsim._InterpolatedImage(psf_image, x_interpolant=galsim.interpolant.Lanczos(32), force_stepk=stepk, force_maxk=maxk)

            xy = galsim.PositionD(x_sca[n], y_sca[n])
            xyI = xy.round()
            draw_offset = (xy - xyI) + galsim.PositionD(0.5, 0.5)
            b = galsim.BoundsI(xmin=xyI.x-n_in_stamp//2+pad//2+1,
                               ymin=xyI.y-n_in_stamp//2+pad//2+1,
                               xmax=xyI.x+n_in_stamp//2+pad//2,
                               ymax=xyI.y+n_in_stamp//2+pad//2)

            sub_image = sca_image[b]
            if FieldDependentModulation:
               xfpa, yfpa = fpaCoords.pix2fpa(idsca[1], x_sca[n], y_sca[n])
               st_model = galsim.DeltaFunction(flux=1. + FieldDependentModulationAmplitude*(xfpa**2+yfpa**2)/fpaCoords.Rfpa**2)
            else:
               st_model = galsim.DeltaFunction(flux=1.)
            source = galsim.Convolve([interp_psf, st_model])
            source.drawImage(sub_image, offset=draw_offset, add_to_image=True, method='no_pixel')

        return sca_image.array[pad//2:-pad//2, pad//2:-pad//2]

    @staticmethod
    def subgen(rngX, lenpix, subpix):
        """
        Returns a subset of the next large block of random numbers.

        This is designed to work even when the number of random numbers requested (`lenpix`)
        is too large for memory. It assumes no repeated entries in `subpix` (but these don't
        have to be sorted).

        Parameters
        ----------
        rngX : np.random.BitGenerator
            The random number generator; must support the advance method.
        lenpix : int
            Number of random numbers to draw.
        subpix : np.array
            1D array of integers indicating which entries to return.

        Returns
        -------
        np.array
            The array of ``R[subpix[0]] .. R[subpix[-1]]``, where ``R[0] .. R[lenpix-1]``
            would be the full-length sequence of random numbers.

        See Also
        --------
        subgen_multirow : Generate multiple rows.

        """

        N = np.size(subpix)
        out_temp = np.zeros(N)
        k = np.argsort(subpix)
        subpix_sort = subpix[k]
        nskip = subpix_sort-1
        nskip[1:] -= subpix_sort[:-1]
        nskip[0] += 1

        for i in range(N):
            rngX.advance(nskip[i])
            out_temp[i] = np.random.Generator(rngX).uniform()
        rngX.advance(lenpix-subpix_sort[-1]-1)
        out = np.zeros(N)
        for i in range(N):
            out[k[i]] = out_temp[i]

        return out

    @staticmethod
    def subgen_multirow(rngX, lenpix, subpix, P):
        """
        Returns multiple subsets of the next large block of random numbers.

        This is designed to work even when the number of random numbers requested in each row (`lenpix`)
        is too large for memory. It assumes no repeated entries in `subpix` (but these don't
        have to be sorted).

        Parameters
        ----------
        rngX : np.random.BitGenerator
            The random number generator; must support the advance method.
        lenpix : int
            Number of random numbers to draw.
        subpix : np.array
            1D array of integers indicating which entries to return.
        P : int
            Number of repititions to generate.

        Returns
        -------
        np.array
            Shape (`P`, len(`subpix`)). Each row is an array ``R[subpix[0]] .. R[subpix[-1]]``,
            where ``R[0] .. R[lenpix-1]`` would be a full-length sequence of random numbers.

        See Also
        --------
        subgen : Generate a single row.

        """

        out = np.zeros((P, np.size(subpix)))
        for j in range(P):
            out[j, :] = GalSimInject.subgen(rngX, lenpix, subpix)
        return out

    @staticmethod
    def genobj(lenpix, subpix, galstring, seed):
        """
        Generates parameters for a set of random galaxies to draw on specific pixels.

        Parameters
        ----------
        lenpix : int
            Number of pixels (should be 12 * 4**nside for HEALPix applications).
        subpix : np.array
            Pixel indices that we want to draw. Array of integers, shape (nobj,).
            Does not need to be sorted, but repititions are not allowed.
        galstring : str
            String specifying the type.
        seed : int
            Random number generator seed.

        Returns
        -------
        dict
            A dictionary containing a bunch of arrays of galaxy information.

        Notes
        -----
        The possible `galstring` values, and the keyword/value returned, are as follows:

        * ``'exp1'`` : exponential profile, random shear (up to 0.5), log distribution radius in 0.125 .. 0.5 arcsec.
          Keyword/values:

          * ``'sersic'`` : Sersic profile arrays.

            * ``'n'`` : Sersic index; float or array of float, shape (nobj,).

            * ``'r'`` : Effective radius in arcsec; float or array of float, shape (nobj,).

            * ``'t__r'`` : Truncation in effective radii; float or array of float, shape (nobj,).

            * ``'g'`` : Ellipticity (g-convention); array of float, shape (2,nobj).

        """

        # nobj = np.size(subpix)
        rngX = np.random.PCG64(seed=seed)

        # now consider each type of object
        if galstring == 'exp1':
            data = GalSimInject.subgen_multirow(rngX, lenpix, subpix, 3)
            g1 = .5*np.sqrt(data[1, :])*np.cos(2*np.pi*data[2, :])
            g2 = .5*np.sqrt(data[1, :])*np.sin(2*np.pi*data[2, :])
            mydict = {'sersic': {'n': 1., 'r': .5/4**data[0, :], 't__r': 8.},
                      'g': np.stack((g1, g2))}
        else:
            mydict = {}

        return mydict

    @staticmethod
    def galsim_extobj_grid(res, mywcs, inpsf, sca_nside, inpsf_oversamp, extraargs=[]):
        """
        Draws a grid of galaxies on an SCA image.

        Parameters
        ----------
        res : int
            HEALPix resolution (nside = 2**res).
        mywcs : astropy.wcs.WCS or pyimcom.wcsutil.PyIMCOM_WCS
            The WCS object for the SCA.
        inpsf : function
            A function that takes a position and returns a PSF (normally an InImage.get_psf_pos method).
        sca_nside : int
            Side length of the SCA (4088 for Roman).
        inpsf_oversamp : int or float
            PSF oversampling factor.
        extraargs : list of str, optional
            List of extra arguments to pass for drawing galaxies.
            An example would be ``extraargs=['seed=12345', 'rot=90', 'shear=0.2:0.1']``.

        Returns
        -------
        np.array
            An array of shape (nside,nside) containing the image of all the objects drawn.

        Notes
        -----
        The extra arguments that can be given include:

        * "seed=n" (n : int) : use n as the random number generator seed

        * "rot=theta" (theta : float) : rotate the galaxies by theta degrees prior to any shear

        * "shear=g1:g2" (g1, g2 : float) : shear the galaxies by (g1,g2), conserving area

        """

        # default parameters
        seed = 4096
        rot = None
        shear = None

        # unpack extraargs
        for arg in extraargs:
            m = re.search(r'^seed=(\d+)$', arg, re.IGNORECASE)
            if m: seed = int(m.group(1))
            m = re.search(r'^rot=(\S+)$', arg, re.IGNORECASE)
            if m: rot = float(m.group(1))
            m = re.search(r'^shear=([^ \:]+)\:([^ \:]+)$', arg, re.IGNORECASE)
            if m: shear = [float(m.group(1)), float(m.group(2))]

        # print('rng seed =', seed, '  transform: rot=', rot, 'shear=', shear)

        refscale = 0.11  # reference pixel size in arcsec
        ra_cent, dec_cent = mywcs.all_pix2world((sca_nside-1)/2, (sca_nside-1)/2, 0)

        search_radius = (sca_nside * 0.11)/3600*(np.pi/180.)*np.sqrt(2)
        vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
        qp = healpy.query_disc(2**res, vec, search_radius, nest=True)
        ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=True, lonlat=True)

        # convert to SCA coordinates
        x_sca, y_sca = mywcs.all_world2pix(ra_hpix, dec_hpix, 0)
        d = 128
        msk_sca = ((x_sca >= -d) & (x_sca <= 4087+d) &
                   (y_sca >= -d) & (y_sca <= 4087+d))
        x_sca = x_sca[msk_sca]; y_sca = y_sca[msk_sca]
        ipix = qp[msk_sca]  # pixel index of the objects within the SCA
        my_ra = ra_hpix[msk_sca]
        my_dec = dec_hpix[msk_sca]
        num_obj = len(x_sca)

        # generate object parameters
        galstring = 'exp1'
        galtype = GalSimInject.genobj(12*4**res, ipix, galstring, seed)

        n_in_stamp = 280
        pad = n_in_stamp+2*(d+1)
        sca_image = galsim.ImageF(sca_nside+pad, sca_nside+pad, scale=refscale)
        t0 = time.time()
        for n in range(num_obj):
            psf = inpsf((my_ra[n], my_dec[n]))  # now with PSF variation
            psf_image = galsim.Image(psf, scale=0.11/inpsf_oversamp)
            #interp_psf = galsim.InterpolatedImage(psf_image, x_interpolant='lanczos32')
            # the first time, get the preferred stepk and maxk
            if n==0:
                interp_psf_test = galsim.InterpolatedImage(psf_image, x_interpolant='lanczos32')
                stepk = interp_psf_test.stepk
                maxk = interp_psf_test.maxk
                del interp_psf_test
            psf_image.setCenter(0,0)
            interp_psf = galsim._InterpolatedImage(psf_image, x_interpolant=galsim.interpolant.Lanczos(32), force_stepk=stepk, force_maxk=maxk)

            # Jacobian
            Jac = local_partial_pixel_derivatives2(mywcs, x_sca[n], y_sca[n])
            # convert to reference pixel size
            Jac /= refscale / 3600.
            # now we have d(X,Y)|_{zweibein: X=E,Y=N} / d(X,Y)|_{pixel coords}

            xy = galsim.PositionD(x_sca[n], y_sca[n])
            xyI = xy.round()
            draw_offset = (xy - xyI) + galsim.PositionD(0.5, 0.5)
            b = galsim.BoundsI(xmin=xyI.x-n_in_stamp//2+pad//2+1,
                               ymin=xyI.y-n_in_stamp//2+pad//2+1,
                               xmax=xyI.x+n_in_stamp//2+pad//2,
                               ymax=xyI.y+n_in_stamp//2+pad//2)

            sub_image = sca_image[b]
            st_model_round = galsim.DeltaFunction(flux=1.)
            # now consider the possible non-point profiles
            #
            if 'sersic' in galtype:
                st_model_round = galsim.Sersic(galtype['sersic']['n'], half_light_radius=galtype['sersic']['r'][n], flux=1.0,
                                               trunc=galtype['sersic']['t__r']*galtype['sersic']['r'][n])
            #
            # now transform the round object if desired
            if 'g' in galtype:
                jshear = np.asarray([[1+galtype['g'][0, n], galtype['g'][1, n]], [galtype['g'][1, n], 1-galtype['g'][0, n]]])\
                    / np.sqrt(1.-galtype['g'][0, n]**2-galtype['g'][1, n]**2)
                st_model_undist = galsim.Transformation(
                    st_model_round, jac=jshear, offset=(0., 0.), flux_ratio=1)
            else:
                st_model_undist = st_model_round
            # rotate, if desired
            if rot is not None:
                theta = rot * np.pi/180.  # convert to radians
                jrot = np.asarray(
                    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                st_model_undist = galsim.Transformation(
                    st_model_undist, jac=jrot, offset=(0., 0.), flux_ratio=1)
            # applied shear, if desired
            if shear is not None:
                jsh = scipy.linalg.expm(np.asarray(
                    [[shear[0], shear[1]], [shear[1], -shear[0]]]))
                st_model_undist = galsim.Transformation(
                    st_model_undist, jac=jsh, offset=(0., 0.), flux_ratio=1)

            # and convert to image coordinates
            st_model = galsim.Transformation(st_model_undist, jac=np.linalg.inv(Jac),
                                             offset=(0., 0.), flux_ratio=np.abs(np.linalg.det(Jac)))

            source = galsim.Convolve([interp_psf, st_model])
            source.drawImage(sub_image, offset=draw_offset, add_to_image=True, method='no_pixel')

        t1 = time.time()-t0
        print(f'n={n}, tot={t1}'); sys.stdout.flush()
        return sca_image.array[pad//2:-pad//2, pad//2:-pad//2]


class GridInject:
    """
    Utilities to inject stars using furry-parakeet C routine.

    Based on fluffy-garbanzo/grid_inject.py.

    Methods
    -------
    make_sph_grid
        Get HEALPix pixels that are in a circular region (staticmethod).
    generate_star_grid
        Make a table of star parameters that are in a circular region (staticmethod).
    make_image_from_grid
        Make an image with a grid of stars from a WCS (staticmethod).

    """

    @staticmethod
    def make_sph_grid(res, ra, dec, radius):
        """
        Get Healpix pixels at resolution res that are within the given radius of (ra, dec).

        Parameters
        ----------
        res : int
            HEALPix resolution (nside == 2**res).
        ra : float
            Right ascension of patch center in radians.
        dec : float
            Declination of patch center in radians.
        radius : float
            Radius of patch in radians.

        Returns
        -------
        dict
            The output contains a dictionary of:
            * 'npix' : int, number of pixels used
            * 'ipix' : np.array of int, pixel indices found
            * 'rapix' : np.array of float, right ascension of pixels, in radians
            * 'decpix' : np.array of float, declination of pixels, in radians

        """

        # get healpix nside
        nside = 2**res

        # get bounding range of pixels
        # extended radius, overlap by 2 rings so there is no clipping later
        radext = radius + 3/nside
        dmin = max(dec-radext, -np.pi/2.)
        dmax = min(dec+radext, np.pi/2.)
        pmin = healpy.pixelfunc.ang2pix(nside, np.pi/2.-dmax, ra, nest=False, lonlat=False)
        pmax = healpy.pixelfunc.ang2pix(nside, np.pi/2.-dmin, ra, nest=False, lonlat=False)

        # and now the pixel values and their positions
        pvec = np.asarray(range(pmin, pmax+1)).astype(np.int64)
        theta, phi = healpy.pixelfunc.pix2ang(nside, pvec, nest=False, lonlat=False)
        thetac = np.pi/2.-theta

        mu = np.sin(thetac)*np.sin(dec) + np.cos(thetac) * np.cos(dec)*np.cos(ra-phi)
        good = np.where(mu >= np.cos(radius))

        ipix = pvec[good]
        rapix = phi[good]
        decpix = thetac[good]
        npix = np.size(ipix)

        return {'res': res, 'nside': nside, 'npix': npix, 'ipix': ipix, 'rapix': rapix, 'decpix': decpix}

    @staticmethod
    def generate_star_grid(res, myWCS, scapar={'nside': 4088, 'pix_arcsec': 0.11}):
        """
        Makes a grid of positions to inject simulated sources into an SCA image.

        Parameters
        ----------
        res : int
            HEALPix resolution (nside == 2**res)
        myWCS : astropy.wcs.WCS or pyimcom.wcsutil.PyIMCOM_WCS
            World coordinate system for the SCA
        scapar : dict
            Should have keyword/values:
            * 'sca_nside' : int, side length of SCA
            * 'pix_arcsec' : float, reference pixel scale in arcsec

        Returns
        -------
        np.array of int
            array of HEALPix pixel indices
        np.array of float
            x positions on the SCA
        np.array of float
            y positions on the SCA
        np.array of float
            right ascensions of the pixels drawn
        np.array of float
            declinations of the pixels drawn

        """

        # SCA side length in radians
        degree = np.pi/180
        sidelength = scapar['nside']*scapar['pix_arcsec']/3600*degree
        radius = sidelength

        # and center position
        cpos_local = (scapar['nside']-1)/2
        cpos_world = myWCS.all_pix2world([[cpos_local, cpos_local]], 0)[0]
        ra_ctr = cpos_world[0]*degree
        dec_ctr = cpos_world[1]*degree

        # stars
        stargrid = GridInject.make_sph_grid(res, ra_ctr, dec_ctr, radius)
        # and positions in the SCA image
        px, py = myWCS.all_world2pix(stargrid['rapix']/degree, stargrid['decpix']/degree, 0)

        return (stargrid['ipix'], px, py, stargrid['rapix']/degree, stargrid['decpix']/degree)

    @staticmethod
    def make_image_from_grid(res, inpsf, idsca, obsdata, mywcs, nside_sca, inpsf_oversamp):
        """
        Make an SCA image with this grid of stars with a PSF from a specified file with unit flux.

        Parameters
        ----------
        res : int
            HEALPix resolution (nside == 2**res)
        inpsf : function
            Should be an‎InImage.get_psf_pos method
        idsca : (int,int)
            Observation ID, SCA pair (SCA in 1 .. 18)
        astropy.io.fits.FITS_rec
            Observation table (needed for some data formats).
        mywcs : astropy.wcs.WCS or pyimcom.wcsutil.PyIMCOM_WCS
            World coordinate system for the SCA
        nside_sca : int
            Side length of the SCA (in pixels)
        inpsf_oversamp: float or int
            Oversampling factor of the PSF image (relative to native).

        Returns
        -------
        np.array
            The image of the SCA with the stars drawn. Shape (`nside_sca`, `nside_sca`).

        """

        thisimage = np.zeros((nside_sca, nside_sca))
        ipix, xsca, ysca, rapix, decpix = GridInject.generate_star_grid(res, mywcs)
        p = 6  # padding for interpolation (n/2+1 for nxn interpolation kernel)
        d = 64  # region to draw

        for istar in range(len(ipix)):
            thispsf = inpsf((rapix[istar], decpix[istar]))  # now with PSF variation
            this_xmax = min(nside_sca, int(xsca[istar])+d)
            this_xmin = max(0, int(xsca[istar])-d)
            this_ymax = min(nside_sca, int(ysca[istar])+d)
            this_ymin = max(0, int(ysca[istar])-d)
            pnx = this_xmax - this_xmin
            pny = this_ymax - this_ymin
            if pnx < 1 or pny < 1: continue

            # draw at this location
            inX = np.zeros((pny, pnx))
            inY = np.zeros((pny, pnx))
            inX[:, :] = (np.array(range(this_xmin, this_xmax)) -
                         xsca[istar])[None, :]
            inY[:, :] = (np.array(range(this_ymin, this_ymax)) -
                         ysca[istar])[:, None]
            interp_array = np.zeros((1, pny*pnx))
            (ny, nx) = np.shape(thispsf)
            iD5512C(np.pad(thispsf, p).reshape((1, ny+2*p, nx+2*p)),
                    inpsf_oversamp*inX.flatten()+(nx-1)/2.+p,
                    inpsf_oversamp*inY.flatten()+(ny-1)/2.+p, interp_array)
            thisimage[this_ymin:this_ymax, this_xmin:this_xmax] =\
                (thisimage[this_ymin:this_ymax, this_xmin:this_xmax]
                 + interp_array.reshape((pny, pnx)) * inpsf_oversamp**2)

        return thisimage


class CplxNoise:
    """
    Utilities to generate 1/f noise.

    Based on fluffy-garbanzo/inject_complex_noise.py.

    Methods
    -------
    noise_1f_frame
        1/f noise generator (staticmethod)

    """

    @staticmethod
    def noise_1f_frame(seed):
        """
        A simple 1/f noise generator, independent in each output channel.

        Parameters
        ----------
        seed : int
            The random number generator seed.

        Returns
        -------
        np.array
            An image of the 1/f noise.

        """

        this_array = np.zeros((4096, 4096))
        rng = np.random.default_rng(seed)
        len_ = 8192*128

        # get frequencies and amplitude ~ sqrt{power}
        freq = np.linspace(0, 1-1./len_, len_)
        freq[len_//2:] -= 1.
        amp = (1.e-99+np.abs(freq*len_))**(-.5)
        amp[0] = 0.
        for ch in range(32):
            # get array
            ftsignal = np.zeros((len_,), dtype=np.complex128)
            ftsignal[:] = rng.normal(loc=0., scale=1., size=(len_,))
            ftsignal[:] += 1j*rng.normal(loc=0., scale=1., size=(len_,))
            ftsignal *= amp
            block = np.fft.fft(ftsignal).real[:len_//2]/np.sqrt(2.)
            block -= np.mean(block)

            xmin = ch*128
            xmax = xmin+128
            # mapping into the image depends on L->R or R->L read order
            if ch % 2 == 0:
                this_array[:, xmin:xmax] = block.reshape((4096, 128))
            else:
                this_array[:, xmin:xmax] = block.reshape((4096, 128))[:, ::-1]

        return this_array[4:4092, 4:4092].astype(np.float32)


class Mask:
    """
    Utilities for permanent and cosmic ray masks.

    Methods
    -------
    randmask
        Simulated a random mask (staticmethod)
    load_permanent_mask
        Loads a permanent mask from a block file (staticmethod)
    load_mask_from_maskfile
        Loads a permanent mask from a file (staticmethod)
    load_cr_mask
        Simulate a cosmic ray mask (staticmethod)

    """

    @staticmethod
    def randmask(idsca, pcut, hitinfo=None):
        """
        Makes a psuedorandom mask that randomly removes groups of pixels (intended for CR simulation).

        Parameters
        ----------
        idsca : (int, int)
            Observation ID, SCA pair.
        pcut : float
            Probability that any given pixel is hit.
        hitinfo : dict or None, optional
            For future expansion, right now always use None.

        Returns
        -------
        np.array of bool
            An image of True for good pixels and False for CR-affected pixels.

        """

        seed = 100000000 + idsca[0]
        rng = np.random.default_rng(seed)
        pad = 10
        g = rng.uniform(size=(18, 2*pad+Stn.sca_nside, 2*pad+Stn.sca_nside))[idsca[1]-1, :, :]
        crhits = np.where(g < pcut, 1., 0.)  # hit mask

        # different ways of making a mask
        if hitinfo is None:
            return np.where(convolve(crhits, np.ones((3, 3)), mode='constant')[pad:-pad, pad:-pad] < .5, True, False)

    @staticmethod
    def load_permanent_mask(block: 'coadd.Block'):
        """
        Builds a permanent mask from a FITS file referenced in the configuration.

        Parameters
        ----------
        block : pyimcom.coadd.Block
            The block that we are coadding.

        Returns
        -------
        np.array of bool
            An image of True for good pixels and False for bad pixels.

        """

        if block.cfg.permanent_mask is None:
            print('No permanent mask')
            permanent_mask = None

        else:
            # permanent mask is 'True' if the pixel should be used.
            # 'GOODVAL' keyword in the input FITS file allows us to indicate
            # whether an unflagged pixel is all 0's (GOODVAL=0) or 1's (GOODVAL!=0 or missing)
            with fits.open(block.cfg.permanent_mask) as f:
                if f[0].header.get('GOODVAL')==0:
                    permanent_mask = np.where(f[0].data==0, True, False)
                else:
                    permanent_mask = np.where(f[0].data, True, False)
            nonzero_count = np.count_nonzero(permanent_mask)
            print('Permanent mask loaded --> ', nonzero_count,
                  'good pixels', nonzero_count/(18*4088**2)*100, '%')

        return permanent_mask

    @staticmethod
    def load_mask_from_maskfile(cfg,obsdata,idsca):
        """
        Loads a mask from a cached mask file.

        Parameters
        ----------
        cfg : pyimcom.config.Config
            The configuration file.
        obsdata : astropy.io.fits.FITS_rec
            Observation table (needed for some data formats).
        idsca : (int,int)
            Tuple of observation ID and SCA number (SCA in 1..18)

        Returns
        -------
        np.array of bool
            An image of True for good pixels and False for bad pixels.

        """

        without_maskfiles = ['dc2_sim', 'anlsim'] # old formats without input masks
        if cfg.informat not in without_maskfiles:
            filename = _get_sca_imagefile(cfg.inpath, idsca, obsdata, cfg.informat, extraargs={'type':'mask'})
            with fits.open(filename) as f:
                return f['MASK'].data==0
        # default return all good
        return np.ones((Stn.sca_nside, Stn.sca_nside), dtype=bool)

    @staticmethod
    def load_cr_mask(inimage: 'coadd.InImage'):
        """
        Generates a cosmic ray mask for an image (including lab noise CRs, if labnoise is used).

        Parameters
        ----------
        inimage : pyimcom.coadd.InImage
            An input image structure.

        Returns
        -------
        np.array of bool
            An image of True for good pixels and False for CR-affected pixels.

        """

        config = inimage.blk.cfg  # shortcut

        if config.cr_mask_rate > 0:
            cr_mask = Mask.randmask(inimage.idsca, config.cr_mask_rate)
            print('Cosmic ray mask: good pix --> ', np.count_nonzero(cr_mask), '/', 4088**2)

            try:
                idx = config.extrainput.index('labnoise')
            except:
                pass
            else:
                cr_mask = np.logical_and(cr_mask, np.abs(
                    inimage.indata[idx]) < config.labnoisethreshold)
                print('Lab noise threshold: good pix --> ', np.count_nonzero(cr_mask), '/', 4088**2)

        else:
            cr_mask = None

        return cr_mask


def _get_sca_imagefile(path, idsca, obsdata, format_, extraargs=None):
    """
    Main function to get input file names.

    Parameters
    ----------

    path : str
        Directory for the files.
    idsca : (int, int)
        Observation ID, SCA pair.
    obsdata : astropy.io.fits.FITS_rec
        Observation table (needed for some data formats).
    format_ : str
        Input file format type identifier from the configuration.
    extraargs : dict or None
        Additional arguments that can be provided for some types of files.
        Default is None.

    Returns
    -------
    str or None
        The file name (returns None if unrecognized format).

    Notes
    -----
    The valid formats are currently:
    * dc2_imsim
    * anlsim
    * L2_2506

    If `sca` is -1, then returns a format string (i.e., with '{:d}' or similar
    instead of the number itself).

    """

    # all formats currently have no leading 0 on the SCA number
    scastr = '{:d}'.format(idsca[1])
    if idsca[1]==-1: scastr = '{:d}'

    # get the filter
    if isinstance(obsdata,str):
        filter = obsdata
    else:
        filter = Stn.RomanFilters[obsdata['filter'][idsca[0]]]

    # for Level 2 summer 2025 format
    if format_ == 'L2_2506':
        out = path+'/sim_L2_{:s}_{:d}_{:s}.asdf'.format(filter, idsca[0], scastr)

        # insert sim layers here
        if extraargs is not None:
            if 'type' in extraargs:
                if extraargs['type'] == 'mask':
                    out = path+'/sim_L2_{:s}_{:d}_{:s}_mask.fits'.format(filter, idsca[0], scastr)
                if extraargs['type'] == 'labnoise':
                    out = path+'/labnoise/slope_{:d}_{:s}.fits'.format(idsca[0], scastr)
                if extraargs['type'] == 'truth':
                    out = path+'/truth/Roman_WAS_truth_{:s}_{:d}_{:s}.fits'.format(filter, idsca[0], scastr)
                if extraargs['type'] == 'noise':
                    out = path+'/sim_L2_{:s}_{:d}_{:s}_noise.asdf'.format(filter, idsca[0], scastr)

        return out

    # for the ANL sims
    if format_ == 'anlsim':
        out = path+'/simple/Roman_WAS_simple_model_{:s}_{:d}_{:s}.fits'.format(filter, idsca[0], scastr)

        # insert ANL sim layers here
        if extraargs is not None:
            if 'type' in extraargs:
                if extraargs['type'] == 'labnoise':
                    out = path+'/labnoise/slope_{:d}_{:s}.fits'.format(idsca[0], scastr)

        return out

    # right now this is the only other type defined
    if format_ != 'dc2_imsim':
        return None

    out = path+'/simple/dc2_{:s}_{:d}_{:s}.fits'.format(filter, idsca[0], scastr)

    if extraargs is not None:
        if 'type' in extraargs:
            if extraargs['type'] == 'truth':
                out = path+'/truth/dc2_{:s}_{:d}_{:s}.fits'.format(filter, idsca[0], scastr)
            elif extraargs['type'] == 'labnoise':
                out = path+'/labnoise/slope_{:d}_{:s}.fits'.format(idsca[0], scastr)
            elif extraargs['type'] == 'skyerr':
                out = path+'/simple/dc2_{:s}_{:d}_{:s}.fits'.format(filter, idsca[0], scastr)

    return out


def check_if_idsca_exists(cfg, obsdata, idsca):
    """
    Determines whether an observation (id,sca) pair exists.

    Parameters
    ----------
    cfg : pyimcom.config.Config
         Configuration information.
    obsdata : astropy.io.fits.FITS_rec
        Observation table (needed for some data formats).
    idsca : (int, int)
        Observation ID, SCA pair.

    Returns:
    exists_ : bool
        Whether the file exists.
    fname : str
        The file name.

    """

    fname = _get_sca_imagefile(cfg.inpath, idsca, obsdata, cfg.informat)
    exists_ = exists(fname)
    return exists_, fname


def get_all_data(inimage: 'coadd.InImage'):
    """
    Makes a 3D array of all the layers of an input image.

    The indexing of the return array is [input layer (e.g., 0=sci or sim), y, x].

    Parameters
    ----------
    inimage : pyimcom.coadd.InImage
        An input image structure.

    Returns
    -------
    np.array
        The input image; shape = (nlayer, nside_sca, nside_sca)

    """

    # read arguments from InImage attributes
    n_inframe = inimage.blk.cfg.n_inframe  # number of input frames
    idsca = inimage.idsca  # which observation to use (tuple (obsid, sca) (sca in 1..18))
    obsdata = inimage.blk.obsdata  # observation data table (information needed for some formats)
    path = inimage.blk.cfg.inpath  # directory for the files
    format_ = inimage.blk.cfg.informat  # string describing type of file name
    inwcs = inimage.inwcs  # input WCS of *this* observation
    inpsf = inimage.get_psf_pos  # now this is a **function**
    # input PSF information (to be passed to GalSimInject or GridInject routines if we draw objects)
    inpsf_oversamp = inimage.blk.cfg.inpsf_oversamp
    extrainput = inimage.blk.cfg.extrainput  # make multiple maps (list of types, first should be None, rest strings)

    # does the user want us to save the input layer cube?
    InLayerCache = False
    if bool(inimage.blk.cfg.inlayercache):
        InLayerCache = True
        inlayer_filepath = inimage.blk.cfg.inlayercache + '_{:08d}_{:02d}.fits'.format(idsca[0],idsca[1])
        inlayer_lockpath = inlayer_filepath + '.lock'
        lock = FileLock(inlayer_lockpath)

    # try to load the data, if not available keep going
    if InLayerCache:
        try:
            with lock.acquire(timeout=30):
                if exists(inlayer_filepath):
                    print('loading input layer <<', inlayer_filepath)
                    with fits.open(inlayer_filepath) as f:
                        inimage.indata = f[0].data.astype(np.float32)
                    sys.stdout.flush()
                    return
        except Timeout:
            pass
    # not available yet, we have to build it. start by making the input data array
    inimage.indata = np.zeros((n_inframe, Stn.sca_nside, Stn.sca_nside), dtype=np.float32)

    # now fill in each slice in *this* observation
    # (missing files are blank)
    filename = _get_sca_imagefile(path, idsca, obsdata, format_)
    if exists(filename):
        # these input formats both use the SCI data frame and have the sky embedded in the header
        if format_ in ['dc2_imsim', 'anlsim']:
            with fits.open(filename) as f:
                inimage.indata[0, :, :] = f['SCI'].data - float(f['SCI'].header['SKY_MEAN'])
        if format_ in ['L2_2506']:
            with asdf.open(filename) as f:
                inimage.indata[0, :, :] = f['roman']['data']
    #
    # now for the extra inputs
    # if n_inframe == 1: return <-- for saving purposes, don't want to exit here

    for i in range(1, n_inframe):
        sys.stdout.flush()
        # truth image (no noise)
        if extrainput[i].casefold() == 'truth'.casefold() or extrainput[i][:6].casefold() == 'truth,'.casefold():
            rescale = 1.0
            m = re.search(r'^truth,(.+)$', extrainput[i], re.IGNORECASE)
            if m: rescale = float(m.group(1))
            filename = _get_sca_imagefile(path, idsca, obsdata, format_, extraargs={'type': 'truth'})
            if exists(filename):
                if filename[-5:]=='.fits':
                    with fits.open(filename) as f:
                        inimage.indata[i, :, :] = f[0].data
                    if format_=='L2_2506':
                        # FITS truth files have to be flipped
                        if idsca[1]%3==0:
                            inimage.indata[i,:,:] = inimage.indata[i,:,::-1]
                        else:
                            inimage.indata[i,:,:] = inimage.indata[i,::-1,:]
                if filename[-5:]=='.asdf':
                    with fits.open(filename) as f:
                        inimage.indata[i, :, :] = f['roman']['data']
                # scaling
                inimage.indata[i, :, :] *= rescale

        # white noise frames (generated from RNG, not file)
        m = re.search(r'^whitenoise(\d+)$', extrainput[i], re.IGNORECASE)
        if m:
            q = int(m.group(1))
            seed = 1000000*(18*q+idsca[1]) + idsca[0]
            print('noise rng: frame_q={:d}, seed={:d}'.format(q, seed))
            rng = np.random.default_rng(seed)
            inimage.indata[i, :, :] = rng.normal(loc=0., scale=1., size=(Stn.sca_nside, Stn.sca_nside))
            del rng

        # 1/f noise frames (generated from RNG, not file)
        m = re.search(r'^1fnoise(\d+)$', extrainput[i], re.IGNORECASE)
        if m:
            q = int(m.group(1))
            seed = 1000000*(18*q+idsca[1]) + idsca[0]
            print('noise rng: frame_q={:d}, seed={:d}'.format(q, seed), '--> 1/f')
            inimage.indata[i, :, :] = CplxNoise.noise_1f_frame(seed)

        # lab noise frames (if applicable)
        if extrainput[i].casefold() == 'labnoise'.casefold():
            filename = _get_sca_imagefile(path, idsca, obsdata, format_, extraargs={'type': 'labnoise'})
            print('lab noise: searching for ' + filename)
            if exists(filename):
                with fits.open(filename) as f:
                    try:
                        inimage.indata[i, :, :] = f[0].data
                    except:
                        inimage.indata[i, :, :] = f[0].data[4:4092,4:4092]
                        print('  -> pulled out 4088x4088 subregion: 10th & 90th percentiles = {:6.3f} {:6.3f}'.format(
                            np.percentile(f[0].data[4:4092,4:4092], 10), np.percentile(f[0].data[4:4092,4:4092], 90)))
                if format_ == 'L2_2506':
                    # FITS labnoise files have to be flipped
                    if idsca[1]%3==0:
                        inimage.indata[i,:,:] = inimage.indata[i,:,::-1]
                    else:
                        inimage.indata[i,:,:] = inimage.indata[i,::-1,:]
            else:
                print('Warning: labnoise file not found, skipping ...')

        # skyerr (not supported in L2_2506)
        if extrainput[i].casefold() == 'skyerr'.casefold():
            filename = _get_sca_imagefile(path, idsca, obsdata, format_, extraargs={'type': 'skyerr'})
            if exists(filename):
                with fits.open(filename) as f:
                    inimage.indata[i, :, :] = f['ERR'].data - float(f['SCI'].header['SKY_MEAN'])

        # C routine star grid
        m = re.search(r'^cstar(\d+)$', extrainput[i], re.IGNORECASE)
        if m:
            res = int(m.group(1))
            print('making grid using C routines: ', res, idsca)
            inimage.indata[i, :, :] = GridInject.make_image_from_grid(
                res, inpsf, idsca, obsdata, inwcs, Stn.sca_nside, inpsf_oversamp)

        # noisy star grid
        # nstar<resolution>,<star_intensity>,<background>,<seed_index>
        m = re.search(r'^nstar(\d+),', extrainput[i], re.IGNORECASE)
        if m:
            res = int(m.group(1))
            extargs = extrainput[i].split(',')[1:]
            tot_int = float(extargs[0])
            bg = float(extargs[1])
            q = int(extargs[2])
            seed = 1000000*(18*q+idsca[1]) + idsca[0]
       	    print('noise rng: frame_q={:d}, seed={:d}'.format(q, seed))
       	    rng = np.random.default_rng(seed)
            print('making noisy stars: ', res, idsca, ' total brightness =', tot_int, 'background =', bg, 'seed index =', q)
            brightness = GridInject.make_image_from_grid(
                res, inpsf, idsca, obsdata, inwcs, Stn.sca_nside, inpsf_oversamp)
            print('min/max brightness =', np.amin(brightness), np.amax(brightness)); sys.stdout.flush()
            inimage.indata[i, :, :] = rng.poisson(lam=brightness*tot_int+bg)-bg
            del rng

        # galsim star grid
        m = re.search(r'^gsstar(\d+)$', extrainput[i], re.IGNORECASE)
        if m:
            res = int(m.group(1))
            print('making grid using GalSim: ', res, idsca)
            inimage.indata[i, :, :] = GalSimInject.galsim_star_grid(
                res, inwcs, inpsf, idsca, obsdata, Stn.sca_nside, inpsf_oversamp)

        # galsim angle-based transient star grid
        # (a test of what happens when a point source is present in one pass but not the other)
        m = re.search(r'^gstrstar(\d+)$', extrainput[i], re.IGNORECASE)
        if m:
            res = int(m.group(1))
            print('making grid using GalSim: ', res, idsca)
            inimage.indata[i, :, :] = GalSimInject.galsim_star_grid(
                res, inwcs, inpsf, idsca, obsdata, Stn.sca_nside, inpsf_oversamp, extraargs={'angleTransient': True})

        # galsim field-dependent amplitude star grid
        # (a test of what a source does if its SED causes it to be brighter (+) or fainter (-) as one moves away from zero field angle)
        m = re.search(r'^gsfdstar(\d+),(.+)+$', extrainput[i], re.IGNORECASE)
        if m:
            res = int(m.group(1))
            amp = float(m.group(2))
            print('making grid using GalSim: ', res, idsca)
            inimage.indata[i, :, :] = GalSimInject.galsim_star_grid(
                res, inwcs, inpsf, idsca, obsdata, Stn.sca_nside, inpsf_oversamp, extraargs={'FieldDependentModulation': amp})

        # galsim extobj grid
        m = re.search(r'^gsext(\d+)', extrainput[i], re.IGNORECASE)
        if m:
            res = int(m.group(1))
            extargs = extrainput[i].split(',')[1:]
            print('making grid using GalSim: ', res, idsca, 'extended object type:', extargs)
            inimage.indata[i, :, :] = GalSimInject.galsim_extobj_grid(
                res, inwcs, inpsf, Stn.sca_nside, inpsf_oversamp, extraargs=extargs)

        # noise realizations saved from romanimpreprocess
        # supported in L2_2506 and later
        m = re.search(r'^noise,(\S+)', extrainput[i], re.IGNORECASE)
        if m:
            noiselabel = str(m.group(1))
            filename = _get_sca_imagefile(path, idsca, obsdata, format_, extraargs={'type': 'noise'})
            print('extracting noise layer ' + noiselabel + ' from ' + filename)
            if exists(filename):
                with asdf.open(filename) as f:
                    # figure out which noise slice to use
                    noise_realizations = f['config']['NOISE']['LAYER']
                    jn_use = -1
                    for jn in range(len(noise_realizations)):
                        if noise_realizations[jn]==noiselabel:
                            if jn_use>=0:
                                warnings.warn('label ' + noiselabel + ' repeated in ' + filename + ': using first instance')
                                break
                            jn_use = jn
                    # now extract that slice
                    if jn_use>=0:
                        inimage.indata[i, :, :] = f['noise'][jn_use,:,:]
                    else:
                        warnings.warn('WARNING: cannot find slice ' + noiselabel + ' in ' + filename + ': continuing')
            else:
                warnings.warn('WARNING: cannot find noise file: ' + filename + ': continuing')

        sys.stdout.flush()

    # this is the end of building the data cube
    # save it if specified in the configuration file
    if InLayerCache:
        try:
            with lock.acquire(timeout=1):
                if not exists(inlayer_filepath):
                    print('saving input layer >>', inlayer_filepath)
                    pr = fits.PrimaryHDU(inimage.indata)
                    # the sciwcs HDU is simply here to save a 2D image with the science WCS --- it doesn't have science data in it
                    # need relax=True to write SIP coefficients
                    #
                    # if a gwcs is used, saves that in an ancillary ASDF file and writes 'GWCS' in the FITS header
                    is_astropy_wcs = True
                    inwcs = inimage.inwcs
                    if isinstance(inwcs, PyIMCOM_WCS):
                        if inwcs.constructortype == 'GWCS':
                            is_astropy_wcs = False
                            sciwcs = fits.ImageHDU()
                            sciwcs.header['WCSTYPE'] = 'GWCS'
                            fits.HDUList([pr,sciwcs]).writeto(inlayer_filepath)
                            af = asdf.AsdfFile()
                            af['wcs'] = inwcs.obj
                            af.write_to(inlayer_filepath[:-5]+'_wcs.asdf')
                        else:
                            inwcs = inwcs.obj
                    if is_astropy_wcs:
                        sciwcs = fits.ImageHDU(np.zeros((Stn.sca_nside, Stn.sca_nside), dtype=np.uint8),
                            header=inwcs.to_header(relax=True), name='SCIWCS')
                        sciwcs['WCSTYPE'] = 'FITS'
                        fits.HDUList([pr,sciwcs]).writeto(inlayer_filepath)
        except Timeout:
            pass
