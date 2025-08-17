"""
Tools to analyze coadded images.

Classes
-------
OutImage
    Wrapper for coadded images (blocks).
NoiseAnal
    Analysis of noise frames.
StarsAnal
    Analysis of point sources.
_BlkGrp
    Abstract base class for groups of blocks (mosiacs or suites).
Mosaic
    Wrapper for coadded mosaics (2D arrays of blocks).
Suite
    Wrapper for coadded suites (hashed arrays of blocks).

"""


from os.path import exists
from pathlib import Path
import re
from enum import Enum

import numpy as np
from astropy.io import fits
from astropy import constants as const, units as u, wcs
from scipy import ndimage

import healpy
import galsim
from .config import Timer, Settings as Stn, Config
from .coadd import OutStamp, Block
from .compress.compressutils import ReadFile
from .diagnostics.outimage_utils.helper import HDU_to_bels

class OutImage:
    """
    Wrapper for coadded images (blocks).

    Parameters
    ----------
    fpath : str
        Path to the output FITS file.
    cfg : Config, optional
        Configuration used for this output image.
        If provided, no consistency check is performed.
        If None, it will be extracted from FITS file.
    hdu_names : list of str, optional
        List of HDU names of this FITS file.
        If provided, no consistency check is performed.
        If None, it will be derived from `cfg`.

    Methods
    -------
    get_hdu_names
        Parse outmaps to get a list of HDU names.
    __init__
        Constructor.
    get_last_line
        Get last line of a text file.
    get_time_consump
        Parse terminal output to get time consumption.
    _load_or_save_hdu_list
        Load data from or save data to FITS file.
    get_coadded_layer
        Extract a coadded layer from the primary HDU.
    get_T_weightmap
        Extract T_weightmap from an additional HDU. 
    get_mean_coverage
        Compute mean coverage based on T_weightmap.
    get_output_map
        Extract an output map from the additional HDUs.
    _update_hdu_data
        Update data using data provided by a neighbor.

    """

    @staticmethod
    def get_hdu_names(outmaps: str) -> [str]:
        """
        Parse outmaps to get a list of HDU names.

        Parameters
        ----------
        outmaps : str
            outmaps attribute of a Config instance.

        Returns
        -------
        list of str
            A list of HDU names.

        """

        hdu_names = ['PRIMARY', 'CONFIG', 'INDATA', 'INWEIGHT', 'INWTFLAT']
        if 'U' in outmaps: hdu_names.append('FIDELITY')
        if 'S' in outmaps: hdu_names.append('SIGMA'   )
        if 'K' in outmaps: hdu_names.append('KAPPA'   )
        if 'T' in outmaps: hdu_names.append('INWTSUM' )
        if 'N' in outmaps: hdu_names.append('EFFCOVER')
        return hdu_names

    def __init__(self, fpath: str, cfg: Config = None, hdu_names: [str] = None) -> None:

        assert exists(fpath), f'{fpath} does not exist'
        self.fpath = fpath
        self.ibx, self.iby = map(int, Path(fpath).stem.split('_')[-2:])

        cfg(); self.cfg = cfg
        if cfg is None:
            with ReadFile(fpath) as hdu_list:
                self.cfg = Config(''.join(hdu_list['CONFIG'].data['text'].tolist()))

        self.hdu_names = hdu_names
        if hdu_names is None:
            self.hdu_names = OutImage.get_hdu_names(self.cfg.outmaps)

    @staticmethod
    def get_last_line(fname: str) -> str:
        """
        Get last line of a text file.

        Parameters
        ----------
        fname : str
            Path to the text file.

        Returns
        -------
        str
            Last line of the text file.

        """

        with open(fname, 'r') as f:
            for line in f:
                pass
            last_line = line
        return last_line

    def get_time_consump(self) -> None:
        """
        Parse terminal output to get time consumption.

        Returns
        -------
        None

        """

        fname = self.fpath.replace('.fits', '.out')
        last_line = OutImage.get_last_line(fname)
        m = re.match('finished at t = ([0-9.]+) s', last_line)
        return float(m.group(1))

    def _load_or_save_hdu_list(self, load_mode: bool = True, save_file: bool = False,
                               auto_to_all: bool = False) -> None:
        """
        Load data from or save data to FITS file.

        Parameters
        ----------
        load_mode : bool, optional
            If True, load data from FITS file (if not already loaded);
            if False, remove current data from memory (if data exist).
        save_file : bool, optional
            Only used when `load_mode` == False. If (`save_file` ==) True,
            save current data to FITS file (overwriting the existing file).
        auto_to_all : bool, optional
            Only used when load_mode == False and save_file == True.
            If (auto_to_all ==) True, change 'PADSIDES' from 'auto' to 'all'
            in the 'CONFIG' HDU of FITS file.

        Returns
        -------
        None

        """

        if load_mode:
            if not hasattr(self, 'hdu_list'):
                self.hdu_list = ReadFile(self.fpath)

        else:
            if save_file:
                assert hasattr(self, 'hdu_list'), 'no hdu_list to save'

                if auto_to_all:
                    my_cfg = Config(''.join(self.hdu_list['CONFIG'].data['text'].tolist()))
                    my_cfg.pad_sides = 'all'
                    config_hdu = fits.TableHDU.from_columns(
                        [fits.Column(name='text', array=my_cfg.to_file(None).splitlines(), format='A512', ascii=True)])
                    config_hdu.header['EXTNAME'] = 'CONFIG'
                    self.hdu_list['CONFIG'] = config_hdu

                self.hdu_list.writeto(self.fpath, overwrite=True)

            if hasattr(self, 'hdu_list'):
                self.hdu_list.close(); del self.hdu_list

    def get_coadded_layer(self, layer: str, j_out: int = 0) -> np.array:
        """
        Extract a coadded layer from the primary HDU.

        Parameters
        ----------
        layer : str
            Name of the layer to be extracted.
        j_out : int or None, optional
            Index of the output PSF. If None, return results based on all output PSFs.

        Returns
        -------
        data : np.array
            Requested coadded layer.
            The shape is either (NsideP, NsideP) or (n_out, NsideP, NsideP) (all output PSFs)

        """

        assert layer in ['SCI'] + self.cfg.extrainput[1:], f"Error: layer '{layer}' not found"
        idx = self.cfg.extrainput.index(layer if layer != 'SCI' else None)

        data_loaded = hasattr(self, 'hdu_list')
        if not data_loaded:
            self.hdu_list = ReadFile(self.fpath)

        if j_out is not None:
            data = (self.hdu_list['PRIMARY'].data[j_out, idx]).astype(np.float32)
        else:
            data = (self.hdu_list['PRIMARY'].data[:, idx]).astype(np.float32)

        if not data_loaded:
            self.hdu_list.close(); del self.hdu_list
        return data

    def get_T_weightmap(self, flat: bool = False, j_out: int = 0) -> np.array:
        """
        Extract T_weightmap from an additional HDU. 

        Parameters
        ----------
        flat : bool, optional
            Whether to read the flat version of T_weightmap.
        j_out : int or None, optional
            Only used when `flat` is False. Index of the output PSF.
            If None, return results based on all output PSFs.

        Returns
        -------
        data : np.array
            Requested T_weightmap.
            Shape is either (n_inimage, n1P, n1P) (1)
            or (n_out, n_inimage, n1P, n1P) (all output PSFs)
            or (n_out*n1P, n_inimage*n1P) (flat version)

        """

        data_loaded = hasattr(self, 'hdu_list')
        if not data_loaded:
            self.hdu_list = ReadFile(self.fpath)

        if not flat:  # read T_hdu
            if j_out is not None:
                data = (self.hdu_list['INWEIGHT'].data[j_out, ...]).astype(np.float32)
            else:
                data = (self.hdu_list['INWEIGHT'].data[:, ...]).astype(np.float32)

        else:  # read T_hdu2
            data = (self.hdu_list['INWTFLAT'].data).astype(np.float32)

        if not data_loaded:
            self.hdu_list.close(); del self.hdu_list
        return data

    def get_mean_coverage(self, padding: bool = False) -> float:
        """
        Compute mean coverage based on T_weightmap.

        We assume that mean coverage is the same for all output PSFs.

        Parameters
        ----------
        padding : bool, optional
            Whether to include padding postage stamps. The default is False.

        Returns
        -------
        mean_coverage : float
            Mean coverage based on T_weightmap.

        """

        T_weightmap = self.get_T_weightmap(j_out=0)  # shape: (n_inimage, n1P, n1P)
        if not padding:
            pad = self.cfg.postage_pad  # shortcut
            T_weightmap = T_weightmap[:, pad:-pad, pad:-pad]

        mean_coverage = np.mean(np.sum(T_weightmap.astype(bool), axis=0))
        del T_weightmap
        return mean_coverage

    def get_output_map(self, outmap: str, j_out: int = 0) -> np.array:
        """
        Extract an output map from the additional HDUs.

        Parameters
        ----------
        outmap : str
            Name of the output map to be extracted.
        j_out : int or None, optional
            Index of the output PSF.
            If None, return results based on all output PSFs.

        Returns
        -------
        data : np.array
            Requested output map. shape is either (NsideP, NsideP)
            or (n_out, NsideP, NsideP) (all output PSFs).

        """

        assert outmap in self.hdu_names, f"Error: map '{outmap}' not found"
        assert outmap in ['FIDELITY', 'SIGMA', 'KAPPA', 'INWTSUM', 'EFFCOVER'],\
        f"Error: map '{outmap}' not supported by get_output_map"

        data_loaded = hasattr(self, 'hdu_list')
        if not data_loaded:
            self.hdu_list = ReadFile(self.fpath)

        coef = int(self.hdu_list[outmap].header.comments['UNIT'].partition('*')[0])
        slice_ = np.s_[j_out] if j_out is not None else np.s_[:]
        data = np.power(10.0, self.hdu_list[outmap].data[slice_] / coef).astype(np.float32)

        dtype = self.hdu_list[outmap].data.dtype
        if dtype == np.dtype('uint16'):
            a_min, a_max = 0, 65535
        elif dtype == np.dtype('>i2'):
            a_min, a_max = -32768, 32767
        a_zero = a_min if coef > 0 else a_max
        data[data == np.power(10.0, a_zero / coef)] = 0.0

        if not data_loaded:
            self.hdu_list.close(); del self.hdu_list
        return data

    def _update_hdu_data(self, neighbor: 'OutImage', direction: str, add_mode: bool = True) -> None:
        """
        Update data using data provided by a neighbor.

        This method is developed for postprocessing, i.e.,
        sharing padding postage stamps between adjacent blocks.
        This method neither loads nor saves hdu_list.

        Parameters
        ----------
        neighbor : OutImage
            Neighboring output image (block) who shares data with "me."
        direction : str
            Which side to update. Must be 'left', 'right', 'bottom', or 'top'.
        add_mode : bool, optional
            If True, update "my" data by adding neighbor's to "mine;"
            if False, replace "my" data with neighbor's.

        Returns
        -------
        None

        """

        assert direction in ['left', 'right', 'bottom', 'top'],\
        f"Error: direction '{direction}' not supported by _update_hdu_data"

        # not necessarily the same as self.cfg
        my_cfg = Config(''.join(self.hdu_list['CONFIG'].data['text'].tolist()))
        assert my_cfg.pad_sides == 'auto', "Error: _update_hdu_data only supports pad_sides == 'auto'"
        del my_cfg

        # update PRIMARY
        NsideP = self.cfg.NsideP  # shortcut
        width = self.cfg.postage_pad * self.cfg.n2  # width of padding region
        fk = self.cfg.fade_kernel  # shortcut

        if direction == 'left':
            my_slice = np.s_[:, :, :,      0        :       width+fk]
            ur_slice = np.s_[:, :, :, NsideP-width*2:NsideP-width+fk]
        elif direction == 'right':
            my_slice = np.s_[:, :, :, NsideP-width-fk:NsideP        ]
            ur_slice = np.s_[:, :, :,        width-fk:       width*2]
        elif direction == 'bottom':
            my_slice = np.s_[:, :,      0        :       width+fk, :]
            ur_slice = np.s_[:, :, NsideP-width*2:NsideP-width+fk, :]
        elif direction == 'top':
            my_slice = np.s_[:, :, NsideP-width-fk:NsideP        , :]
            ur_slice = np.s_[:, :,        width-fk:       width*2, :]

        self.hdu_list['PRIMARY'].data[my_slice] =     self.hdu_list['PRIMARY'].data[my_slice] * add_mode \
                                                + neighbor.hdu_list['PRIMARY'].data[ur_slice]
        del my_slice, ur_slice

        # update INWEIGHT and INWTFLAT
        n1P = self.cfg.n1P  # shortcuts
        pad = self.cfg.postage_pad

        my_idscas = list(zip(    self.hdu_list['INDATA'].data['obsid'],
                                 self.hdu_list['INDATA'].data['sca']))
        ur_idscas = list(zip(neighbor.hdu_list['INDATA'].data['obsid'],
                             neighbor.hdu_list['INDATA'].data['sca']))

        for idsca in set(my_idscas) & set(ur_idscas):
            my_idx = my_idscas.index(idsca)
            ur_idx = ur_idscas.index(idsca)

            if direction == 'left':
                my_slice = np.s_[:, my_idx, :,   0      :    pad]
                ur_slice = np.s_[:, ur_idx, :, n1P-pad*2:n1P-pad]
            elif direction == 'right':
                my_slice = np.s_[:, my_idx, :, n1P-pad:n1P      ]
                ur_slice = np.s_[:, ur_idx, :,     pad:    pad*2]
            elif direction == 'bottom':
                my_slice = np.s_[:, my_idx,   0      :    pad, :]
                ur_slice = np.s_[:, ur_idx, n1P-pad*2:n1P-pad, :]
            elif direction == 'top':
                my_slice = np.s_[:, my_idx, n1P-pad:n1P      , :]
                ur_slice = np.s_[:, ur_idx,     pad:    pad*2, :]

            self.hdu_list['INWEIGHT'].data[my_slice] = neighbor.hdu_list['INWEIGHT'].data[ur_slice]
            del my_slice, ur_slice

        del my_idscas, ur_idscas

        n_out, n_inimage = self.hdu_list['INWEIGHT'].data.shape[:2]
        self.hdu_list['INWTFLAT'].data[:, :] =\
            np.transpose(self.hdu_list['INWEIGHT'].data,
                         axes=(0, 2, 1, 3)).reshape((n_out*n1P, n_inimage*n1P))

        # update output maps
        fk = self.cfg.fade_kernel

        for outmap in self.hdu_names[5:]:
            my_maps =     self.get_output_map(outmap, None)
            ur_maps = neighbor.get_output_map(outmap, None)

            if direction == 'left':
                if add_mode:
                    OutStamp.trapezoid(my_maps, fk, False, (0, 0, width-fk, 0), 'L')
                    OutStamp.trapezoid(ur_maps, fk, False, (0, 0, 0, width-fk), 'R')
                my_slice = np.s_[:, :,      0        :       width+fk]
                ur_slice = np.s_[:, :, NsideP-width*2:NsideP-width+fk]

            elif direction == 'right':
                if add_mode:
                    OutStamp.trapezoid(my_maps, fk, False, (0, 0, 0, width-fk), 'R')
                    OutStamp.trapezoid(ur_maps, fk, False, (0, 0, width-fk, 0), 'L')
                my_slice = np.s_[:, :, NsideP-width-fk:NsideP        ]
                ur_slice = np.s_[:, :,        width-fk:       width*2]

            elif direction == 'bottom':
                if add_mode:
                    OutStamp.trapezoid(my_maps, fk, False, (width-fk, 0, 0, 0), 'B')
                    OutStamp.trapezoid(ur_maps, fk, False, (0, width-fk, 0, 0), 'T')
                my_slice = np.s_[:,      0        :       width+fk, :]
                ur_slice = np.s_[:, NsideP-width*2:NsideP-width+fk, :]

            elif direction == 'top':
                if add_mode:
                    OutStamp.trapezoid(my_maps, fk, False, (0, width-fk, 0, 0), 'T')
                    OutStamp.trapezoid(ur_maps, fk, False, (width-fk, 0, 0, 0), 'B')
                my_slice = np.s_[:, NsideP-width-fk:NsideP        , :]
                ur_slice = np.s_[:,        width-fk:       width*2, :]

            coef = int(self.hdu_list[outmap].header.comments['UNIT'].partition('*')[0])
            dtype = self.hdu_list[outmap].data.dtype
            if dtype == np.dtype('>i2'): dtype = np.dtype('int16')
            self.hdu_list[outmap].data[my_slice] = Block.compress_map(
                my_maps[my_slice] * add_mode + ur_maps[ur_slice], coef, dtype)
            del my_maps, ur_maps, my_slice, ur_slice


class NoiseAnal:
    """
    Analysis of noise frames.

    Largely based on diagnostics/noise/noisespecs.py.

    Parameters
    ----------
    outim : OutImage
        Output image to analyze.
    layer : str
        Layer name of noise frame to analyze.

    Methods
    -------
    __init__
        Constructor.
    get_norm
        Get norm for 2D noise power spectrum (classmethod).
    azimuthal_average
        Compute radial profile of image (staticmethod)).
    _get_wavenumbers
        Calculate wavenumbers for the input image (staticmethod).
    __call__
        Analyze specified noise frame of given output image.
    clear
        Free up memory space.

    """

    # from noise/noisespecs.py
    AREA = {'Y106': 7006, 'J129': 7111, 'H158': 7340,
            'F184': 4840, 'K213': 4654, 'W146': 22085}  # cm^2

    # Set useful constants
    tfr = 3.08  # sec
    gain = 1.458  # electrons/DN
    ABstd = 3.631e-20  # erg/cm^2
    h = const.h.to('erg/Hz').value  # 6.62607015e-27
    m_ab = 23.9  # sample mag for PS
    s_in = 0.11  # arcsec

    # following make_plot_ln.py
    PS1D_COLORS = ['orange', 'darksalmon', 'palevioletred', 'mediumvioletred', 'darkviolet']
    PS1D_STYLES = ['solid', 'dotted', 'dashed', 'solid', 'dashdot']

    def __init__(self, outim: OutImage, layer: str) -> None:

        self.outim = outim
        self.layer = layer

        self.cfg = outim.cfg
        assert layer in ['SCI'] + self.cfg.extrainput[1:], f"Error: layer '{layer}' not found"

    @classmethod
    def get_norm(cls, layer: str, L: int, filtername: str, s_out: float) -> float:
        """
        Get norm for 2D noise power spectrum.

        Parameters
        ----------
        layer : str
            Layer name of noise frame to analyze.
        L : int
            Side length of noise frame in px.
        filtername : str
            Name of filter used for this output image.
        s_out : float
            Output pixel scale in arcsec.

        Returns
        -------
        float
            Norm for 2D noise power spectrum.

        Notes
        -----
        For simulated noise frames, dividing by s_in**2
        converts from units of S_in^2 to arcsec^2.

        """

        if layer.startswith('white'):
            return (L / s_out) ** 2  # (L * (cls.s_in/s_out)) ** 2
        elif layer.startswith('1f'):
            return (L / s_out) ** 2  # (L * (cls.s_in/s_out)) ** 2
        elif layer.startswith('lab'):
            return cls.tfr/cls.gain * cls.ABstd/cls.h * cls.AREA[filtername] * 10**(-0.4*cls.m_ab) * s_out**2

    @staticmethod
    def azimuthal_average(image: np.array, nradbins: int,
                          rbin: np.array = None, ridx: np.array = None) -> (np.array):

        """
        Compute radial profile of image.

        Parameters
        ----------
        image : np.array
            Input image, shape (L, L).
        nradbins : int
            Number of radial bins in profile.
        rbin: np.array, optional
            "labels" parameter for ndimage utilities.
            If provided, has shape (L, L).
            The default is None. If not provided, derive from `image`.shape.
        ridx: np.array, optional
            "index" parameter for ndimage utilities.
            If provided, has shape (`nradbins`,).
            The default is None. If not provided, derive from rbin.

        Returns
        -------
        radial_mean : np.array
            Mean intensity within each annulus. Main result. Shape is (`nradbins`,)
        radial_err : np.array
            Standard error on the mean: sigma / sqrt(N). Shape is (`nradbins`,)

        """

        if rbin is None:
            ny, nx = image.shape
            yy, xx = np.mgrid[:ny, :nx]
            r = np.hypot(xx - nx/2, yy - ny/2)
            rbin = (nradbins * r / r.max()).astype(int)
        if ridx is None:
            ridx = np.arange(1, rbin.max() + 1)

        radial_mean = ndimage.mean(image, labels=rbin, index=ridx)
        radial_stddev = ndimage.standard_deviation(image, labels=rbin, index=ridx)
        npix = ndimage.sum(np.ones_like(image), labels=rbin, index=ridx)
        radial_err = radial_stddev / np.sqrt(npix); del npix
        return radial_mean, radial_err

    @staticmethod
    def _get_wavenumbers(window_length: int, nradbins: int,
                         rbin: np.array = None, ridx: np.array = None) -> np.array:
        """
        Calculate wavenumbers for the input image.

        Parameters
        ----------
        window_length : int
            the length of one axis of the image.
        nradbins : int
            number of radial bins the image should be averaged into
        rbin: np.array, optional
            "labels" parameter for ndimage utilities.
            If provided, shape is (L,L).
            The default is None. If not provided, derive from image.shape.
        ridx: np.array, optional
            "index" parameter for ndimage utilities.
            If provided, shape is (`nradbins`,).
            The default is None. If not provided, derive from `rbin`.

        Returns
        -------
        kmean : np.array
            the wavenumbers for the image, shape (`nradbins`,)

        """

        k = np.fft.fftshift(np.fft.fftfreq(window_length))
        kx, ky = np.meshgrid(k, k)
        k = np.sqrt(np.square(kx) + np.square(ky))

        if rbin is None or ridx is None:
            kmean, kerr = NoiseAnal.azimuthal_average(k, nradbins)
        else:
            kmean = ndimage.mean(k, labels=rbin, index=ridx)
        return kmean

    def __call__(self, padding: bool = False, bin_: bool = True,
                 rbin: np.array = None, ridx: np.array = None) -> None:
        """
        Analyze specified noise frame of given output image.

        Parameters
        ----------
        padding : bool, optional
            Whether to include padding postage stamps. (to be implemented)
        bin_ : bool, optional
            Whether to bin the 2D spectrum into L/8 x L/8 image.
            Currently this is ignored, as only bin_ == True is supported.

        Returns
        -------
        None

        """

        L = self.cfg.NsideP  # side length in px
        indata = self.outim.get_coadded_layer(self.layer)
        if not padding:
            L = self.cfg.Nside
            # padding region around the edge
            bdpad = self.cfg.n2 * self.cfg.postage_pad
            indata = indata[bdpad:-bdpad, bdpad:-bdpad]

        s_out = self.cfg.dtheta * u.degree.to('arcsec')  # in arcsec
        norm = NoiseAnal.get_norm(self.layer, L, Stn.RomanFilters[self.cfg.use_filter], s_out)

        # Measure the 2D power spectrum of image.
        ps = np.empty((L, L), dtype=np.float64)
        rps = np.square(np.abs(np.fft.fftshift(np.fft.rfft2(indata), 0))) / norm
        ps[ :, L//2:] = rps[   :    ,     :-1  ]
        ps[1:, :L//2] = rps[L-1:0:-1, L//2:0:-1]
        ps[0 , :L//2] = rps[  0     , L//2:0:-1]
        self.ps2d = np.average(np.reshape(ps, (L//8, 8, L//8, 8)), axis=(1, 3))
        del rps, ps

        # Calculate the azimuthally-averaged 1D power spectrum of the image.
        nradbins = L//16  # Number of radial bins is side length div. into 8 from binning and then (floor) div. by 2.
        ps_1d, ps_image_err = NoiseAnal.azimuthal_average(self.ps2d, nradbins, rbin, ridx)
        # wavenumbers = NoiseAnal._get_wavenumbers(L, nradbins)

        self.ps1d = np.zeros((L//16, 2))
        # self.ps1d[:, 0] = wavenumbers   # powerspectrum.k
        self.ps1d[:, 0] = ps_1d         # powerspectrum.ps_image
        self.ps1d[:, 1] = ps_image_err  # powerspectrum.ps_image_err

    def clear(self) -> None:
        """Free up memory space."""

        if hasattr(self, 'ps2d'):
            del self.ps2d, self.ps1d


# diagnostics/starcube_nonoise_coldescr.txt

ColDescr = Enum('ColDescr', [
    'RA',            #  0: right ascension
    'DEC',           #  1: declination
    # 'BLOCK_IX',      #  2: block ix
    # 'BLOCK_IY',      #  3: block iy
    'X_POS',         #  4: x position in block image (float)
    'Y_POS',         #  5: y position in block image (float)
    # 'X_INT',         #  6: int part of x
    # 'Y_INT',         #  7: int part of y
    # 'X_FRAC',        #  8: frac part of x
    # 'Y_FRAC',        #  9: frac part of y
    'AMPLITUDE',     # 10: star fit -> amplitude
    'OFFSET_X',      # 11: star fit -> centroid offset (in output pixels), x
    'OFFSET_Y',      # 12: star fit -> centroid offset (in output pixels), y
    'WIDTH',         # 13: star fit -> sigma (in output pixels)
    'SHAPE_G1',      # 14: star fit -> g1 shape
    'SHAPE_G2',      # 15: star fit -> g2 shape
    'M42_REAL',      # 16: star 4th moment Re M42 (Zhang et al. 2023 MNRAS 525, 2441 convention)
    'M42_IMAG',      # 17: star 4th moment Im M42
    'FORCED_PLUS',   # 18: star moment with forced sigma=0.40 arcsec scale length [+ component] (in units of forced sigma**2)
    'FORCED_CROSS',  # 19: star moment with forced sigma=0.40 arcsec scale length [x component]
    'FIDELITY',      # 20: fidelity (mean in 0.5 arcsec box)
    'COVERAGE',      # 21: coverage at star center
    'MEAN_UC',       # new: mean PSF leakage U/C (in linear space)
    'MEAN_SIGMA',    # new: mean noise amplification Sigma
    'STD_TSUM',      # new: standard deviation of total weight
    'MEAN_NEFF',     # new: mean effective coverage
    ], start=0)


class StarsAnal:
    """
    Analysis of point sources.

    Largely based on diagnostics/starcube_nonoise.py.

    Parameters
    ----------
    outim : OutImage
        Output image to analyze.
    layer : str, optional
        Layer name of injected stars to analyze.

    Methods
    -------
    __init__
        Constructor.
    __call__
        Analyze given point source frame of given output image.
    clear
        Free up memory space.

    """

    bd = 40  # padding size
    bd2 = 8
    ncol = len(ColDescr)

    def __init__(self, outim: OutImage, layer: str = 'gsstar14') -> None:

        self.outim = outim
        self.layer = layer
        assert layer == 'gsstar14', "Error: currently only 'gsstar14' is supported"

        self.cfg = outim.cfg
        assert layer in ['SCI'] + self.cfg.extrainput[1:], f"Error: layer '{layer}' not found"

    def __call__(self, n: int = None, search_radius: float = None,
                 forced_scale: float = None, bdpad: int = None, res: int = None) -> None:
        """
        Analyze given point source frame of given output image.

        Parameters
        ----------
        n : int or None, optional
            Size of output images.
            If not provided, derive from self.cfg. Same for other parameters.
        search_radius : float or None, optional
            Search radius for injected point sources.
        forced_scale : float or None, optional
            Forced scale length for star moments.
        bdpad : int or None, optional
            Padding region around the edge.
        res : int or None, optional
            Resolution of HEALPix grid.

        Returns
        -------
        None

        """

        if None in [n, search_radius, forced_scale, bdpad, res]:
            n = self.cfg.NsideP  # size of output images
            blocksize = self.cfg.n1 * self.cfg.n2 * self.cfg.dtheta * Stn.degree  # radians
            search_radius = 1.5 * blocksize / np.sqrt(2.0)  # search radius
            forced_scale = 0.40 * u.arcsec.to('degree') / self.cfg.dtheta  # in output pixels
            bdpad = self.cfg.n2 * self.cfg.postage_pad  # padding region around the edge
            res = int(re.match(r'^gsstar(\d+)$', self.layer).group(1))
            # print(n, search_radius, forced_scale, bdpad, res)

        data_loaded = hasattr(self.outim, 'hdu_list')
        if not data_loaded:
            self.outim._load_or_save_hdu_list(True)

        f = self.outim.hdu_list  # alias
        use_slice = (['SCI'] + self.cfg.extrainput[1:]).index(self.layer)

        mywcs = wcs.WCS(f[0].header)
        map_ = f[0].data[0, use_slice, :, :]
        wt = np.sum(np.where(f['INWEIGHT'].data[0, :, :, :] > 0.01, 1, 0), axis=0)
        fmap = f['FIDELITY'].data[0, :, :].astype(np.float32) \
            * HDU_to_bels(f['FIDELITY'])/.1  # convert to dB
        fmap = np.floor(fmap).astype(np.int16)  # and round to integer
        del f

        outmaps = self.cfg.outmaps  # shortcut
        if 'U' in outmaps: UC_map    = self.outim.get_output_map('FIDELITY')
        if 'S' in outmaps: Sigma_map = self.outim.get_output_map('SIGMA'   )
        if 'T' in outmaps: Tsum_map  = self.outim.get_output_map('INWTSUM' )
        if 'N' in outmaps: Neff_map  = self.outim.get_output_map('EFFCOVER')

        if not data_loaded:
            self.outim._load_or_save_hdu_list(False)

        ra_cent, dec_cent = mywcs.all_pix2world([(n-1)/2], [(n-1)/2], [0.], [0.], 0, ra_dec_order=True)
        ra_cent = ra_cent[0]; dec_cent = dec_cent[0]
        vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
        qp = healpy.query_disc(2**res, vec, search_radius, nest=False)
        ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=False, lonlat=True)
        npix = len(ra_hpix)
        x, y, z1, z2 = mywcs.all_world2pix(ra_hpix, dec_hpix, np.zeros((npix,)), np.zeros((npix,)), 0)
        xi = np.rint(x).astype(np.int16); yi = np.rint(y).astype(np.int16)
        grp = np.where(np.logical_and(np.logical_and(xi >= bdpad, xi < n-bdpad),
                                      np.logical_and(yi >= bdpad, yi < n-bdpad)))
        ra_hpix = ra_hpix[grp]
        dec_hpix = dec_hpix[grp]
        x = x[grp]
        y = y[grp]
        npix = len(x)
        del vec, qp, z1, z2, grp

        self.sub_cat = np.zeros((npix, StarsAnal.ncol))
        xi = np.rint(x).astype(np.int16)
        yi = np.rint(y).astype(np.int16)
        # position information
        self.sub_cat[:, ColDescr.RA      .value] = ra_hpix
        self.sub_cat[:, ColDescr.DEC     .value] = dec_hpix
        # self.sub_cat[:, ColDescr.BLOCK_IX.value] = self.outim.ibx
        # self.sub_cat[:, ColDescr.BLOCK_IY.value] = self.outim.iby
        self.sub_cat[:, ColDescr.X_POS   .value] = x
        self.sub_cat[:, ColDescr.Y_POS   .value] = y
        # self.sub_cat[:, ColDescr.X_INT   .value] = xi
        # self.sub_cat[:, ColDescr.Y_INT   .value] = yi
        dx = x-xi  # self.sub_cat[:, ColDescr.X_FRAC  .value] = dx = x-xi
        dy = y-yi  # self.sub_cat[:, ColDescr.Y_FRAC  .value] = dy = y-yi
        del ra_hpix, dec_hpix

        bd = StarsAnal.bd  # shortcut
        # print(self.outim.ibx, self.outim.iby, self.outim.fpath, npix)
        print(npix, end=' ')
        for k in range(npix):
            newimage = map_[yi[k]+1-bd:yi[k]+bd, xi[k]+1-bd:xi[k]+bd]

            # PSF shape
            try:
                moms = galsim.Image(newimage).FindAdaptiveMom()
            except:
                continue

            self.sub_cat[k, ColDescr.AMPLITUDE.value] = moms.moments_amp
            self.sub_cat[k, ColDescr.OFFSET_X .value] = moms.moments_centroid.x-bd-dx[k]
            self.sub_cat[k, ColDescr.OFFSET_Y .value] = moms.moments_centroid.y-bd-dy[k]
            self.sub_cat[k, ColDescr.WIDTH    .value] = moms.moments_sigma
            self.sub_cat[k, ColDescr.SHAPE_G1 .value] = moms.observed_shape.g1
            self.sub_cat[k, ColDescr.SHAPE_G2 .value] = moms.observed_shape.g2

            # higher moments
            x_, y_ = np.meshgrid(np.arange(1, bd*2) - moms.moments_centroid.x,
                                 np.arange(1, bd*2) - moms.moments_centroid.y)
            e1 = moms.observed_shape.e1
            e2 = moms.observed_shape.e2
            Mxx = moms.moments_sigma**2 * (1+e1) / np.sqrt(1-e1**2-e2**2)
            Myy = moms.moments_sigma**2 * (1-e1) / np.sqrt(1-e1**2-e2**2)
            Mxy = moms.moments_sigma**2 * e2 / np.sqrt(1-e1**2-e2**2)
            D = Mxx*Myy-Mxy**2
            zeta = D*(Mxx+Myy+2*np.sqrt(D))
            u_ = ((Myy+np.sqrt(D))*x_ - Mxy*y_) / zeta**0.5
            v_ = ((Mxx+np.sqrt(D))*y_ - Mxy*x_) / zeta**0.5
            wti = newimage * np.exp(-0.5*(u_**2+v_**2))
            self.sub_cat[k, ColDescr.M42_REAL.value] = np.sum(wti*(u_**4-v_**4))/np.sum(wti)
            self.sub_cat[k, ColDescr.M42_IMAG.value] = 2*np.sum(wti*(u_**3*v_+u_*v_**3))/np.sum(wti)

            # moments with forced scale length
            wti2 = newimage * np.exp(-0.5*(x_**2+y_**2)/forced_scale**2)
            self.sub_cat[k, ColDescr.FORCED_PLUS .value] = np.sum(wti2*(x_**2-y_**2))/np.sum(wti2)/forced_scale**2
            self.sub_cat[k, ColDescr.FORCED_CROSS.value] = np.sum(wti2*(2*x_*y_))/np.sum(wti2)/forced_scale**2

            # fidelity and coverage
            central = np.s_[yi[k]+1-StarsAnal.bd2:yi[k]+StarsAnal.bd2,
                            xi[k]+1-StarsAnal.bd2:xi[k]+StarsAnal.bd2]  # central region of the star
            self.sub_cat[k, ColDescr.FIDELITY.value] = np.mean(fmap[central])
            self.sub_cat[k, ColDescr.COVERAGE.value] = wt[yi[k]//self.cfg.n2, xi[k]//self.cfg.n2]

            # new columns based on output maps
            self.sub_cat[k, ColDescr.MEAN_UC   .value] = np.mean(UC_map   [central]) if 'U' in outmaps else -1
            self.sub_cat[k, ColDescr.MEAN_SIGMA.value] = np.mean(Sigma_map[central]) if 'S' in outmaps else -1
            self.sub_cat[k, ColDescr.STD_TSUM  .value] = np.std (Tsum_map [central]) if 'T' in outmaps else -1
            if self.cfg.linear_algebra == 'Empirical': self.sub_cat[k, ColDescr.STD_TSUM  .value] = 0
            self.sub_cat[k, ColDescr.MEAN_NEFF .value] = np.mean(Neff_map [central]) if 'N' in outmaps else -1

            del newimage, x_, y_, u_, v_, wti, wti2

        del map_, wt, fmap
        del x, y, xi, yi, dx, dy

        if 'U' in outmaps: del UC_map
        if 'S' in outmaps: del Sigma_map
        if 'T' in outmaps: del Tsum_map
        if 'N' in outmaps: del Neff_map

    def clear(self) -> None:
        """
        Free up memory space.

        Returns
        -------
        None.

        """

        if hasattr(self, 'sub_cat'):
            del self.sub_cat


class _BlkGrp:
    """
    Abstract base class for groups of blocks (mosiacs or suites).

    Methods
    -------
    __call__
        Run all the analyses below.
    get_consump_map
        Get map of time consumption.
    get_coverage_map
        Get map of mean coverages.
    get_noise_power_spectra
        Analyze noise power spectra of this mosaic.
    get_star_catalog
        Analyze injected point sources of this mosaic.
    clear
        Free up memory space.

    """

    def __call__(self, overwrite: bool = False) -> None:
        """
        Run all the analyses below.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing results.

        Returns
        -------
        None

        """

        self.get_consump_map(overwrite=overwrite)  # Get map of time consumption.
        self.get_coverage_map(overwrite=overwrite)  # Get map of mean coverages.
        self.get_noise_power_spectra(overwrite=overwrite)  # Analyze noise power spectra of this mosaic.
        self.get_star_catalog(overwrite=overwrite)  # Analyze injected point sources of this mosaic.

    def get_consump_map(self, overwrite: bool = False) -> None:
        """
        Get map of time consumption.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing results.

        Returns
        -------
        None

        """

        fname = self.cfg.outstem + '_Consump.npy'
        if not overwrite and exists(fname):
            with open(fname, 'rb') as f:
                self.consump_map = np.load(f)
            return

        if self.ndim == 2:  # Mosaic
            nblock = self.cfg.nblock  # shortcut
            self.consump_map = np.zeros((nblock, nblock))
            for iby in range(nblock):
                for ibx in range(nblock):
                    self.consump_map[iby][ibx] = self.outimages[iby][ibx].get_time_consump()

        elif self.ndim == 1:  # Suite
            nrun = self.nrun
            self.consump_map = np.zeros((nrun,))
            for ib in range(nrun):
                self.consump_map[ib] = self.outimages[ib].get_time_consump()

        with open(fname, 'wb') as f:
            np.save(f, self.consump_map)

    def get_coverage_map(self, overwrite: bool = False) -> None:
        """
        Get map of mean coverages.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing results.

        Returns
        -------
        None

        """

        fname = self.cfg.outstem + '_Coverage.npy'
        if not overwrite and exists(fname):
            with open(fname, 'rb') as f:
                self.coverage_map = np.load(f)
            return

        if self.ndim == 2:  # Mosaic
            nblock = self.cfg.nblock  # shortcut
            self.coverage_map = np.zeros((nblock, nblock))
            for iby in range(nblock):
                for ibx in range(nblock):
                    self.coverage_map[iby][ibx] = self.outimages[iby][ibx].get_mean_coverage()

        elif self.ndim == 1:  # Suite
            nrun = self.nrun
            self.coverage_map = np.zeros((nrun,))
            for ib in range(nrun):
                self.coverage_map[ib] = self.outimages[ib].get_mean_coverage()

        with open(fname, 'wb') as f:
            np.save(f, self.coverage_map)

    def get_noise_power_spectra(self, bins: int = 5, overwrite: bool = False) -> None:
        """
        Analyze noise power spectra of this mosaic.

        Parameters
        ----------
        bins : int, optional
            Number of bins for 1D power spectra.
        overwrite : bool, optional
            Whether to overwrite existing results.

        Returns
        -------
        None

        """

        fname = self.cfg.outstem + '_NoisePS.npy'
        if not overwrite and exists(fname):
            with open(fname, 'rb') as f:
                self.ps2d_all = np.load(f)
                self.ps1d_all = np.load(f)
                self.wavenumbers = np.load(f)
            return

        timer = Timer()

        # identify noise layers
        noiseinput = [layer for layer in self.cfg.extrainput[1:] if 'noise' in layer]
        n_innoise = len(noiseinput)
        print(noiseinput)

        # mean coverage bins
        if not hasattr(self, 'coverage_map'):
            self.get_coverage_map()

        mc_max = self.coverage_map.max() + 1e-12
        mc_min = self.coverage_map.min() - 1e-12
        coverage_idx = ((self.coverage_map - mc_min) /
                        (mc_max-mc_min) * bins).astype(np.uint8)
        unique, counts = np.unique(coverage_idx, return_counts=True)

        # create storage
        if self.padding == False:  # Mosaic
            L = self.cfg.Nside
        else:  # Suite
            L = self.cfg.NsideP
        self.ps2d_all = np.zeros((n_innoise, L//8, L//8))
        self.ps1d_all = np.zeros((n_innoise, bins, L//16, 2))

        # derive rbin and ridx for NoiseAnal.azimuthal_average
        nradbins = L//16  # Number of radial bins is side length div. into 8 from binning and then (floor) div. by 2.
        yy, xx = np.mgrid[:L//8, :L//8]
        r = np.hypot(xx - L//8/2, yy - L//8/2)
        rbin = (nradbins * r / r.max()).astype(int)
        ridx = np.arange(1, rbin.max() + 1)
        del yy, xx, r

        self.wavenumbers = NoiseAnal._get_wavenumbers(L, nradbins)
        # dividing by s_out converts from units of cyc/s_out to cyc/arcsec
        self.wavenumbers /= self.cfg.dtheta * u.degree.to('arcsec')

        # loop over noise layers and output images
        if self.ndim == 2:  # Mosaic
            for iby in range(self.cfg.nblock):
                print(' > row {:2d}  t= {:9.2f} s'.format(iby, timer()))
                for inl, layer in enumerate(noiseinput):
                    for ibx in range(self.cfg.nblock):
                        noise = NoiseAnal(self.outimages[iby][ibx], layer)
                        noise(padding=self.padding, rbin=rbin, ridx=ridx)
                        self.ps2d_all[inl, :, :] += noise.ps2d
                        self.ps1d_all[inl, coverage_idx[iby][ibx], :, :] += noise.ps1d[:, :]
                        noise.clear(); del noise

        elif self.ndim == 1:  # Suite
            nrun = self.nrun
            for inl, layer in enumerate(noiseinput):
                for ib in range(nrun):
                    noise = NoiseAnal(self.outimages[ib], layer)
                    noise(padding=self.padding, rbin=rbin, ridx=ridx)
                    self.ps2d_all[inl, :, :] += noise.ps2d
                    self.ps1d_all[inl, coverage_idx[ib], :, :] += noise.ps1d[:, :]
                    noise.clear(); del noise

        del rbin, ridx

        # postprocessing
        if self.ndim == 2:  # Mosaic
            self.ps2d_all /= self.cfg.nblock ** 2
        elif self.ndim == 1:  # Suite
            self.ps2d_all /= self.nrun

        for idx, count in zip(unique, counts):
            self.ps1d_all[:, idx, :, :] /= count
        del coverage_idx, counts

        with open(fname, 'wb') as f:
            np.save(f, self.ps2d_all)
            np.save(f, self.ps1d_all)
            np.save(f, self.wavenumbers)

        print(f'finished at t = {timer():.2f} s')

    def get_star_catalog(self, layer: str = 'gsstar14', overwrite: bool = False) -> None:
        """
        Analyze injected point sources of this mosaic.

        Parameters
        ----------
        layer : str, optional
            Layer name of injected stars to analyze.
        overwrite : bool, optional
            Whether to overwrite existing results.

        Returns
        -------
        None

        """

        fname = self.cfg.outstem + '_StarCat.npy'
        if not overwrite and exists(fname):
            with open(fname, 'rb') as f:
                self.star_cat = np.load(f)
            return

        timer = Timer()
        self.star_cat = np.zeros((0, StarsAnal.ncol))

        n = self.cfg.NsideP  # size of output images
        blocksize = self.cfg.n1 * self.cfg.n2 * self.cfg.dtheta * Stn.degree  # radians
        search_radius = 1.5 * blocksize / np.sqrt(2.0)  # search radius
        forced_scale = 0.40 * u.arcsec.to('degree') / self.cfg.dtheta  # in output pixels
        bdpad = self.cfg.n2 * self.cfg.postage_pad  # padding region around the edge
        res = int(re.match(r'^gsstar(\d+)$', layer).group(1))
        # print(n, search_radius, forced_scale, bdpad, res)

        # loop over output images
        if self.ndim == 2:  # Mosaic
            for iby in range(self.cfg.nblock):
                print(' > row {:2d}  t= {:9.2f} s'.format(iby, timer()))

                print('star counts:', end=' ')
                for ibx in range(self.cfg.nblock):
                    stars = StarsAnal(self.outimages[iby][ibx], layer)
                    stars(n, search_radius, forced_scale, bdpad, res)
                    self.star_cat = np.concatenate((self.star_cat, stars.sub_cat), axis=0)
                    stars.clear(); del stars
                print()

        elif self.ndim == 1:  # Suite
            nrun = self.nrun
            print('star counts:', end=' ')
            for ib in range(nrun):
                stars = StarsAnal(self.outimages[ib], layer)
                stars(n, search_radius, forced_scale, bdpad, res)
                self.star_cat = np.concatenate((self.star_cat, stars.sub_cat), axis=0)
                stars.clear(); del stars
            print()

        with open(fname, 'wb') as f:
            np.save(f, self.star_cat)

        print(f'finished at t = {timer():.2f} s')

    def clear(self) -> None:
        """Free up memory space."""

        if self.ndim == 2:  # Mosaic
            for ibx in range(self.cfg.nblock):
                for iby in range(self.cfg.nblock):
                    self.outimages[iby][ibx] = None

        elif self.ndim == 1:  # Suite
            for ib in range(self.nrun):
                self.outimages[ib] = None

        if hasattr(self, 'consump_map'):  del self.consump_map
        if hasattr(self, 'coverage_map'): del self.coverage_map
        if hasattr(self, 'ps2d_all'):     del self.ps2d_all, self.ps1d_all
        if hasattr(self, 'star_cat'):     del self.star_cat


class Mosaic(_BlkGrp):
    """
    Wrapper for coadded mosaics (2D arrays of blocks).

    Parameters
    ----------
    cfg : Config
        Configuration used for this output mosaic.

    Methods
    -------
    __init__
        Constructor.
    share_padding_stamps
        Share padding postage stamps between adjacent blocks.

    """

    ndim = 2
    padding = False  # for get_noise_power_spectra

    def __init__(self, cfg: Config) -> None:
        """Constructor."""

        cfg(); self.cfg = cfg
        self.hdu_names = OutImage.get_hdu_names(cfg.outmaps)

        self.outimages = [[None for ibx in range(cfg.nblock)]
                                for iby in range(cfg.nblock)]
        for ibx in range(cfg.nblock):
            for iby in range(cfg.nblock):
                fpath = cfg.outstem + f'_{ibx:02d}_{iby:02d}.fits'
                self.outimages[iby][ibx] = OutImage(fpath, cfg, self.hdu_names)

    def share_padding_stamps(self) -> None:
        """
        Share padding postage stamps between adjacent blocks.

        Returns
        -------
        None

        """

        assert self.cfg.pad_sides == 'auto', "Error: share_padding_stamps only supports pad_sides == 'auto'"
        nblock = self.cfg.nblock  # shortcut
        timer = Timer()

        print(' > horizontal sharing')
        for iby in range(nblock):
            print(' > row {:2d}  t= {:9.2f} s'.format(iby, timer()))
            self.outimages[iby][0]._load_or_save_hdu_list(True)
            for ibx in range(nblock-1):
                self.outimages[iby][ibx+1]._load_or_save_hdu_list(True)
                self.outimages[iby][ibx]._update_hdu_data(self.outimages[iby][ibx+1], 'right', True)
                self.outimages[iby][ibx+1]._update_hdu_data(self.outimages[iby][ibx], 'left', False)
                self.outimages[iby][ibx]._load_or_save_hdu_list(False, save_file=True)
            self.outimages[iby][nblock-1]._load_or_save_hdu_list(False, save_file=True)
        print(flush=True)

        print(' > vertical sharing')
        for ibx in range(nblock):
            print(' > column {:2d}  t= {:9.2f} s'.format(ibx, timer()))
            self.outimages[0][ibx]._load_or_save_hdu_list(True)
            for iby in range(nblock-1):
                self.outimages[iby+1][ibx]._load_or_save_hdu_list(True)
                self.outimages[iby][ibx]._update_hdu_data(self.outimages[iby+1][ibx], 'top', True)
                self.outimages[iby+1][ibx]._update_hdu_data(self.outimages[iby][ibx], 'bottom', False)
                self.outimages[iby][ibx]._load_or_save_hdu_list(False, save_file=True, auto_to_all=True)
            self.outimages[nblock-1][ibx]._load_or_save_hdu_list(False, save_file=True, auto_to_all=True)
        print(flush=True)

        print(f'finished at t = {timer():.2f} s')


class Suite(_BlkGrp):
    """
    Wrapper for coadded suites (hashed arrays of blocks).

    Parameters
    ----------
    cfg : Config
        Configuration used for this output mosaic.
    prime : int, optional
        Prime number for hashing (Paper IV).
    nrun : int, optional
        Number of coadded blocks (Paper IV).

    Methods
    -------
    __init__
        Constructor.

    """

    ndim = 1
    padding = True  # for get_noise_power_spectra

    def __init__(self, cfg: Config, prime: int = 691, nrun: int = 16) -> None:
        """Constructor."""

        cfg(); self.cfg = cfg
        self.hdu_names = OutImage.get_hdu_names(cfg.outmaps)

        self.prime = prime
        self.nrun = nrun
        self.outimages = [None for ib in range(nrun)]
        for ib in range(nrun):
            ibx, iby = divmod(ib*691 % cfg.nblock**2, cfg.nblock)
            fpath = cfg.outstem + f'_{ibx:02d}_{iby:02d}.fits'
            self.outimages[ib] = OutImage(fpath, cfg, self.hdu_names)
