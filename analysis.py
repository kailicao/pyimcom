'''
Tools to analyze coadded images.

Classes
-------
OutImage : Wrapper for coadded images (blocks).
NoiseAnal : Analysis of noise frames.
Mosaic : Wrapper for coadded mosaics (2D arrays of blocks).

'''


from os.path import exists
from pathlib import Path
import re

import numpy as np
from astropy.io import fits
from astropy import constants as const
from astropy import units as u
from scipy import ndimage

from .config import Timer, Settings as Stn, Config
from .coadd import OutStamp, Block


class OutImage:
    '''
    Wrapper for coadded images (blocks).

    Methods
    -------
    get_hdu_names (staticmethod) : Parse outmaps to get a list of HDU names.
    __init__ : Constructor.
    get_last_line (staticmethod) : Get last line of a text file.
    get_time_consump : Parse terminal output to get time consumption.

    _load_or_save_hdu_list : Load data from or save data to FITS file.
    get_coadded_layer : Extract a coadded layer from the primary HDU.
    get_T_weightmap : Extract T_weightmap from an additional HDU. 
    get_mean_coverage : Compute mean coverage based on T_weightmap.
    get_output_map : Extract an output map from the additional HDUs.
    _update_hdu_data : Update data using data provided by a neighbor.

    '''

    @staticmethod
    def get_hdu_names(outmaps: str) -> [str]:
        '''
        Parse outmaps to get a list of HDU names.

        Parameters
        ----------
        outmaps : str
            outmaps attribute of a Config instance.

        Returns
        -------
        [str]
            A list of HDU names.

        '''

        hdu_names = ['PRIMARY', 'CONFIG', 'INDATA', 'INWEIGHT', 'INWTFLAT']
        if 'U' in outmaps: hdu_names.append('FIDELITY')
        if 'S' in outmaps: hdu_names.append('SIGMA'   )
        if 'K' in outmaps: hdu_names.append('KAPPA'   )
        if 'T' in outmaps: hdu_names.append('INWTSUM' )
        if 'N' in outmaps: hdu_names.append('EFFCOVER')
        return hdu_names

    def __init__(self, fpath: str, cfg: Config = None, hdu_names: [str] = None) -> None:
        '''
        Constructor.

        Parameters
        ----------
        fpath : str
            Path to the output FITS file.
        cfg : Config, optional
            Configuration used for this output image.
            If provided, no consistency check is performed.
            The default is None. If None, it will be extracted from FITS file.
        hdu_names : [str], optional
            List of HDU names of this FITS file.
            If provided, no consistency check is performed.
            The default is None. If None, it will be derived from cfg.

        Returns
        -------
        None.

        '''

        assert exists(fpath), f'{fpath} does not exist'
        self.fpath = fpath
        self.ibx, self.iby = map(int, Path(fpath).stem.split('_')[-2:])

        self.cfg = cfg
        if cfg is None:
            with fits.open(fpath) as hdu_list:
                self.cfg = Config(''.join(hdu_list['CONFIG'].data['text'].tolist()))

        self.hdu_names = hdu_names
        if hdu_names is None:
            self.hdu_names = OutImage.get_hdu_names(self.cfg.outmaps)

    @staticmethod
    def get_last_line(fname: str) -> str:
        '''
        Get last line of a text file.

        Parameters
        ----------
        fname : str
            Path to the text file.

        Returns
        -------
        str
            Last line of the text file.

        '''

        with open(fname, 'r') as f:
            for line in f:
                pass
            last_line = line
        return last_line

    def get_time_consump(self) -> None:
        '''
        Parse terminal output to get time consumption.

        Returns
        -------
        None.

        '''

        fname = self.fpath.replace('.fits', '.out')
        last_line = OutImage.get_last_line(fname)
        m = re.match('finished at t = ([0-9.]+) s', last_line)
        return float(m.group(1))

    def _load_or_save_hdu_list(self, load_mode: bool = True, save_file: bool = False,
                               auto_to_all: bool = False) -> None:
        '''
        Load data from or save data to FITS file.

        Parameters
        ----------
        load_mode : bool, optional
            If True, load data from FITS file (if not already loaded);
            if False, remove current data from memory (if data exist).
            The default is True.
        save_file : bool, optional
            Only used when load_mode == False. If (save_file ==) True,
            save current data to FITS file (overwriting the existing file).
            The default is False.
        auto_to_all : bool, optional
            Only used when load_mode == False and save_file == True.
            If (auto_to_all ==) True, change 'PADSIDES' from 'auto' to 'all'
            in the 'CONFIG' HDU of FITS file. The default is False.

        Returns
        -------
        None.

        '''

        if load_mode:
            if not hasattr(self, 'hdu_list'):
                self.hdu_list = fits.open(self.fpath)

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
        '''
        Extract a coadded layer from the primary HDU.

        Parameters
        ----------
        layer : str
            Name of the layer to be extracted.
        j_out : int, optional
            Index of the output PSF. The default is 0.
            If None, return results based on all output PSFs.

        Returns
        -------
        data : np.array, shape : either (NsideP, NsideP)
                                 or (n_out, NsideP, NsideP) (all output PSFs)
            Requested coadded layer.

        '''

        assert layer in ['SCI'] + self.cfg.extrainput[1:], f"Error: layer '{layer}' not found"
        idx = self.cfg.extrainput.index(layer if layer != 'SCI' else None)

        data_loaded = hasattr(self, 'hdu_list')
        if not data_loaded:
            self.hdu_list = fits.open(self.fpath)

        if j_out is not None:
            data = (self.hdu_list['PRIMARY'].data[j_out, idx]).astype(np.float32)
        else:
            data = (self.hdu_list['PRIMARY'].data[:, idx]).astype(np.float32)

        if not data_loaded:
            self.hdu_list.close(); del self.hdu_list
        return data

    def get_T_weightmap(self, flat: bool = False, j_out: int = 0) -> np.array:
        '''
        Extract T_weightmap from an additional HDU. 

        Parameters
        ----------
        flat : bool, optional
            Whether to read the flat version of T_weightmap. The default is False.
        j_out : int, optional
            Only used when flat == False. Index of the output PSF.
            The default is 0. If None, return results based on all output PSFs.

        Returns
        -------
        data : np.array, shape : either (n_inimage, n1P, n1P)
                                 or (n_out, n_inimage, n1P, n1P) (all output PSFs)
                                 or (n_out*n1P, n_inimage*n1P) (flat version)
            Requested T_weightmap.

        '''

        data_loaded = hasattr(self, 'hdu_list')
        if not data_loaded:
            self.hdu_list = fits.open(self.fpath)

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
        '''
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

        '''

        T_weightmap = self.get_T_weightmap(j_out=0)  # shape: (n_inimage, n1P, n1P)
        if not padding:
            pad = self.cfg.postage_pad  # shortcut
            T_weightmap = T_weightmap[:, pad:-pad, pad:-pad]

        mean_coverage = np.mean(np.sum(T_weightmap.astype(bool), axis=0))
        del T_weightmap
        return mean_coverage

    def get_output_map(self, outmap: str, j_out: int = 0) -> np.array:
        '''
        Extract an output map from the additional HDUs.

        Parameters
        ----------
        outmap : str
            Name of the output map to be extracted.
        j_out : int, optional
            Index of the output PSF. The default is 0.
            If None, return results based on all output PSFs.

        Returns
        -------
        data : np.array, shape : either (NsideP, NsideP)
                                 or (n_out, NsideP, NsideP) (all output PSFs)
            Requested output map.

        '''

        assert outmap in self.hdu_names, f"Error: map '{outmap}' not found"
        assert outmap in ['FIDELITY', 'SIGMA', 'KAPPA', 'INWTSUM', 'EFFCOVER'],\
        f"Error: map '{outmap}' not supported by get_output_map"

        data_loaded = hasattr(self, 'hdu_list')
        if not data_loaded:
            self.hdu_list = fits.open(self.fpath)

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
        '''
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
            if False, replace "my" data with neighbor's. The default is True.

        Returns
        -------
        None.

        '''

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
    '''
    Analysis of noise frames.

    Largely based on diagnostics/noise/noisespecs.py.

    Methods
    -------
    __init__ : Constructor.
    get_norm (classmethod) : Get norm for 2D noise power spectrum.
    azimuthal_average (staticmethod) : Compute radial profile of image.
    _get_wavenumbers (staticmethod) : Calculate wavenumbers for the input image.
    __call__ : Analyze specified noise frame of given output image.
    clear : Free up memory space.

    '''

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

    def __init__(self, outim: OutImage, layer: str) -> None:
        '''
        Constructor.

        Parameters
        ----------
        outim : OutImage
            Output image to analyze.
        layer : str
            Layer name of noise frame to analyze.

        Returns
        -------
        None.

        '''

        self.outim = outim
        self.layer = layer

        self.cfg = outim.cfg
        assert layer in ['SCI'] + self.cfg.extrainput[1:], f"Error: layer '{layer}' not found"

    @classmethod
    def get_norm(cls, layer: str, L: int, filtername: str, s_out: float) -> float:
        '''
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

        '''

        if layer.startswith('white'):
            return (L * (cls.s_in/s_out)) ** 2
        elif layer.startswith('1f'):
            return (L * (cls.s_in/s_out)) ** 2
        elif layer.startswith('lab'):
            return cls.tfr/cls.gain * cls.ABstd/cls.h * cls.AREA[filtername] * 10**(-0.4*cls.m_ab) * s_out**2

    @staticmethod
    def azimuthal_average(image: np.array, nradbins: int,
                          rbin: np.array = None, ridx: np.array = None) -> (np.array):

        """
        Compute radial profile of image.

        Parameters
        ----------
        image : np.array, shape : (L, L)
            Input image.
        nradbins : int
            Number of radial bins in profile.
        rbin: np.array, optional, shape : (L, L)
            "labels" parameter for ndimage utilities.
            The default is None. If not provided, derive from image.shape.
        ridx: np.array, optional, shape : (nradbins,)
            "index" parameter for ndimage utilities.
            The default is None. If not provided, derive from rbin.

        Returns
        -------
        radial_mean : np.array, shape : (nradbins,)
            Mean intensity within each annulus. Main result
        radial_err : np.array, shape : (nradbins,)
            Standard error on the mean: sigma / sqrt(N).

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
        rbin: np.array, optional, shape : (L, L)
            "labels" parameter for ndimage utilities.
            The default is None. If not provided, derive from image.shape.
        ridx: np.array, optional, shape : (nradbins,)
            "index" parameter for ndimage utilities.
            The default is None. If not provided, derive from rbin.

        Returns
        -------
        kmean : np.array, shape : (nradbins,)
            the wavenumbers for the image

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
        '''
        Analyze specified noise frame of given output image.

        Parameters
        ----------
        padding : bool, optional (to be implemented)
            Whether to include padding postage stamps. The default is False.
        bin_ : bool, optional
            Whether to bin the 2D spectrum.
            The default is True, binning 2D spectrum into L/8 x L/8 image.
            Currently this is ignored, as only bin_ == True is supported.

        Returns
        -------
        None.

        '''

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
        '''
        Free up memory space.

        Returns
        -------
        None.

        '''

        if hasattr(self, 'ps2d'):
            del self.ps2d, self.ps1d


class Mosaic:
    '''
    Wrapper for coadded mosaics (2D arrays of blocks).

    Methods
    -------
    __init__ : Constructor.
    share_padding_stamps : Share padding postage stamps between adjacent blocks.
    get_consump_map : Get map of time consumption.
    get_coverage_map : Get map of mean coverages.

    get_noise_power_spectra : Analyze noise power spectra of this mosaic.
    clear : Free up memory space.

    '''

    def __init__(self, cfg: Config) -> None:
        '''
        Constructor.

        Parameters
        ----------
        cfg : Config
            Configuration used for this output mosaic.

        Returns
        -------
        None.

        '''

        self.cfg = cfg
        self.hdu_names = OutImage.get_hdu_names(cfg.outmaps)

        self.outimages = [[None for ibx in range(cfg.nblock)]
                                for iby in range(cfg.nblock)]
        for ibx in range(cfg.nblock):
            for iby in range(cfg.nblock):
                fpath = cfg.outstem + f'_{ibx:02d}_{iby:02d}.fits'
                assert exists(fpath), f'{fpath} does not exist'
                self.outimages[iby][ibx] = OutImage(fpath, cfg, self.hdu_names)

    def share_padding_stamps(self) -> None:
        '''
        Share padding postage stamps between adjacent blocks.

        Returns
        -------
        None.

        '''

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

    def get_consump_map(self) -> None:
        '''
        Get map of time consumption.

        Returns
        -------
        None.

        '''

        fname = self.cfg.outstem + '_Consump.npy'
        if exists(fname):
            with open(fname, 'rb') as f:
                self.consump_map = np.load(f)
            return

        nblock = self.cfg.nblock  # shortcut
        self.consump_map = np.zeros((nblock, nblock))

        for iby in range(nblock):
            for ibx in range(nblock):
                self.consump_map[iby][ibx] = self.outimages[iby][ibx].get_time_consump()

        with open(fname, 'wb') as f:
            np.save(f, self.consump_map)

    def get_coverage_map(self) -> None:
        '''
        Get map of mean coverages.

        Returns
        -------
        None.

        '''

        nblock = self.cfg.nblock  # shortcut
        self.coverage_map = np.zeros((nblock, nblock))

        for iby in range(nblock):
            for ibx in range(nblock):
                self.coverage_map[iby][ibx] = self.outimages[iby][ibx].get_mean_coverage()

    def get_noise_power_spectra(self, bins: int = 5) -> None:
        '''
        Analyze noise power spectra of this mosaic.

        Parameters
        ----------
        bins : int, optional
            Number of bins for 1D power spectra. The default is 5.

        Returns
        -------
        None.

        '''

        fname = self.cfg.outstem.rpartition('/')[-1] + '_NoisePS.npy'
        if exists(fname):
            with open(fname, 'rb') as f:
                self.ps2d_all = np.load(f)
                self.ps1d_all = np.load(f)
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
        _, counts = np.unique(coverage_idx, return_counts=True)

        # create storage
        L = self.cfg.Nside  # force padding == False
        self.ps2d_all = np.zeros((n_innoise, L//8, L//8))
        self.ps1d_all = np.zeros((n_innoise, bins, L//16, 3))

        # derive rbin and ridx for NoiseAnal.azimuthal_average
        nradbins = L//16  # Number of radial bins is side length div. into 8 from binning and then (floor) div. by 2.
        yy, xx = np.mgrid[:L//8, :L//8]
        r = np.hypot(xx - L//8/2, yy - L//8/2)
        rbin = (nradbins * r / r.max()).astype(int)
        ridx = np.arange(1, rbin.max() + 1)
        del yy, xx, r

        self.ps1d_all[:, :, :, 0] = NoiseAnal._get_wavenumbers(L, nradbins)

        # loop over noise layers and output images
        for iby in range(self.cfg.nblock):
            print(' > row {:2d}  t= {:9.2f} s'.format(iby, timer()))

            for inl, layer in enumerate(noiseinput):
                for ibx in range(self.cfg.nblock):
                    noise = NoiseAnal(self.outimages[iby][ibx], layer)
                    noise(padding=False, rbin=rbin, ridx=ridx)
                    self.ps2d_all[inl, :, :] += noise.ps2d
                    self.ps1d_all[inl, coverage_idx[iby][ibx], :, 1:] += noise.ps1d[:, :]
                    noise.clear(); del noise

        del rbin, ridx

        # postprocessing
        self.ps2d_all /= self.cfg.nblock ** 2
        for idx in range(bins):
            self.ps1d_all[:, idx, :, 1:] /= counts[idx]
        del coverage_idx, counts

        with open(fname, 'wb') as f:
            np.save(f, self.ps2d_all)
            np.save(f, self.ps1d_all)

        print(f'finished at t = {timer():.2f} s')

    def clear(self) -> None:
        '''
        Free up memory space.

        Returns
        -------
        None.

        '''

        for ibx in range(self.cfg.nblock):
            for iby in range(self.cfg.nblock):
                self.outimages[iby][ibx] = None

        if hasattr(self, 'consump_map'):  del self.consump_map
        if hasattr(self, 'coverage_map'): del self.coverage_map
        if hasattr(self, 'ps2d_all'):     del self.ps2d_all, self.ps1d_all
