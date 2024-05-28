'''
Tools to analyze coadded images.

Classes
-------
OutImage : Wrapper for coadded images (blocks).

'''


import numpy as np
from astropy.io import fits

import matplotlib as mpl
from matplotlib import rcParams
rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                 'font.size': 12, 'text.latex.preamble': r"\usepackage{amsmath}",
                 'xtick.major.pad': 2, 'ytick.major.pad': 2, 'xtick.major.size': 6, 'ytick.major.size': 6,
                 'xtick.minor.size': 3, 'ytick.minor.size': 3, 'axes.linewidth': 2, 'axes.labelpad': 1})

from .config import Config


class OutImage:
    '''
    Wrapper for coadded images (blocks).

    Methods
    -------
    __init__ : Constructor.
    get_coadded_layer : Extract a coadded layer from the primary HDU.
    get_output_map : Extract an output map from the additional HDUs.
    format_axis (staticmethod) : Format a panel (an axis) of a figure.

    '''

    def __init__(self, fpath: str) -> None:
        '''
        Constructor.

        Parameters
        ----------
        fpath : str
            Path to the output FITS file.

        Returns
        -------
        None.

        '''

        self.fpath = fpath
        with fits.open(fpath) as f:
            self.cfg = Config(''.join(f['CONFIG'].data['text'].tolist()))
            self.hdus = [hdu.name for hdu in f]

    def get_coadded_layer(self, layer: str, j_out: int = 0) -> np.array:
        '''
        Extract a coadded layer from the primary HDU.

        Parameters
        ----------
        layer : str
            Name of the layer to be extracted.
        j_out : int, optional
            Index of the output PSF. The default is 0.

        Returns
        -------
        data : np.array, shape : (NsideP, NsideP)
            Requested coadded layer.

        '''

        assert layer in ['SCI']+self.cfg.extrainput[1:], f"Error: layer '{layer}' not found"
        idx = self.cfg.extrainput.index(layer if layer != 'SCI' else None)

        data = np.zeros((self.cfg.NsideP, self.cfg.NsideP), dtype=np.float32)
        with fits.open(self.fpath) as f:
            data = f['PRIMARY'].data[j_out, idx]
        return data

    def get_output_map(self, outmap: str, j_out: int = 0) -> np.array:
        '''
        Extract an output map from the additional HDUs.

        Parameters
        ----------
        outmap : str
            Name of the output map to be extracted.
        j_out : int, optional
            Index of the output PSF. The default is 0.

        Returns
        -------
        data : np.array, shape : (NsideP, NsideP)
            Requested output map.

        '''

        assert outmap in self.hdus, f"Error: map '{outmap}' not found"
        assert outmap in ['FIDELITY', 'SIGMA', 'KAPPA', 'INWTSUM', 'EFFCOVER'],\
        f"Error: map '{outmap}' not supported by get_output_map"

        data = np.zeros((self.cfg.NsideP, self.cfg.NsideP), dtype=np.float32)
        with fits.open(self.fpath) as f:
            coef = int(f[outmap].header.comments['UNIT'].partition('*')[0])
            data = np.power(10.0, f[outmap].data[j_out] / coef)
        return data

    @staticmethod
    def format_axis(ax: 'mpl.axes._axes.Axes', grid_on: bool = True) -> None:
        '''
        Format a panel (an axis) of a figure.

        Parameters
        ----------
        ax : mpl.axes._axes.Axes
            Panel to be formatted.
        grid_on : bool, optional
            Whether to add grid to the panel. The default is True.

        Returns
        -------
        None.

        '''

        ax.minorticks_on()
        if grid_on: ax.grid(visible=True, which='major', linestyle=':')
        ax.tick_params(axis='both', which='both', direction='out')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.patch.set_alpha(0.0)
