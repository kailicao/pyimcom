'''
Tools to analyze coadded images.

Classes
-------
OutImage : Wrapper for coadded images (blocks).

'''


from os.path import exists
import numpy as np
from astropy.io import fits

from .config import Config
from pyimcom.coadd import Block


class OutImage:
    '''
    Wrapper for coadded images (blocks).

    Methods
    -------
    __init__ : Constructor.
    get_coadded_layer : Extract a coadded layer from the primary HDU.
    get_output_map : Extract an output map from the additional HDUs.

    '''

    def __init__(self, fpath: str, cfg: Config = None, hdu_names: [str] = None) -> None:
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

        self.cfg = cfg
        if cfg is None:
            with fits.open(fpath) as f:
                self.cfg = Config(''.join(f['CONFIG'].data['text'].tolist()))

        self.hdu_names = hdu_names
        if hdu_names is None:
            self.hdu_names = OutImage.get_hdu_names(cfg.outmaps)

    @staticmethod
    def get_hdu_names(outmaps: str) -> [str]:
        hdu_names = ['PRIMARY', 'CONFIG', 'INDATA', 'INWEIGHT', 'INWTFLAT']
        if 'U' in outmaps: hdu_names.append('FIDELITY')
        if 'S' in outmaps: hdu_names.append('SIGMA'   )
        if 'K' in outmaps: hdu_names.append('KAPPA'   )
        if 'T' in outmaps: hdu_names.append('INWTSUM' )
        if 'N' in outmaps: hdu_names.append('EFFCOVER')
        return hdu_names

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
        data : np.array, shape : (NsideP, NsideP)
            Requested coadded layer.

        '''

        assert layer in ['SCI']+self.cfg.extrainput[1:], f"Error: layer '{layer}' not found"
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
        data : np.array, shape : (NsideP, NsideP)
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
            a_min = 0
        elif dtype == np.dtype('>i2'):
            a_min = -32768
        data[data == np.power(10.0, a_min / coef)] = 0.0

        if not data_loaded:
            self.hdu_list.close(); del self.hdu_list
        return data

    def _load_or_save_hdu_list(self, load_mode: bool = True) -> None:
        if load_mode:
            if not hasattr(self, 'hdu_list'):
                self.hdu_list = fits.open(self.fpath)
        else:
            self.hdu_list.writeto(self.fpath, overwrite=True)
            self.hdu_list.close(); del self.hdu_list

    def _update_hdu_data(self, neighbor: 'OutImage', direction: str, add_mode: bool = True) -> None:
        assert direction in ['left', 'right', 'bottom', 'top'],\
        f"Error: direction '{direction}' not supported by _update_hdu_data"

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
        for outmap in self.hdu_names[5:]:
            my_maps =     self.get_output_map(outmap, None)
            ur_maps = neighbor.get_output_map(outmap, None)

            if direction == 'left':
                my_slice = np.s_[:, :,      0        :       width+fk]
                ur_slice = np.s_[:, :, NsideP-width*2:NsideP-width+fk]
            elif direction == 'right':
                my_slice = np.s_[:, :, NsideP-width-fk:NsideP        ]
                ur_slice = np.s_[:, :,        width-fk:       width*2]
            elif direction == 'bottom':
                my_slice = np.s_[:,      0        :       width+fk, :]
                ur_slice = np.s_[:, NsideP-width*2:NsideP-width+fk, :]
            elif direction == 'top':
                my_slice = np.s_[:, NsideP-width-fk:NsideP        , :]
                ur_slice = np.s_[:,        width-fk:       width*2, :]

            coef = int(self.hdu_list[outmap].header.comments['UNIT'].partition('*')[0])
            dtype = self.hdu_list[outmap].data.dtype
            if dtype == np.dtype('>i2'): dtype = np.dtype('int16')
            self.hdu_list[outmap].data[my_slice] = Block.compress_map(
                my_maps[my_slice] * add_mode + ur_maps[ur_slice], coef, dtype)
            del my_maps, ur_maps, my_slice, ur_slice


class Mosaic:

    def __init__(self, cfg: Config) -> None:
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
        assert self.cfg.pad_sides == 'auto', "Error: this method only supports pad_sides == 'auto'"
        nblock = self.cfg.nblock  # shortcut

        # horizontal sharing
        for jbx in range(nblock):
            self.outimages[jbx][0]._load_or_save_hdu_list(True)
            for ibx in range(nblock-1):
                self.outimages[jbx][ibx+1]._load_or_save_hdu_list(True)
                self.outimages[jbx][ibx]._update_hdu_data(self.outimages[jbx][ibx+1], 'right', True)
                self.outimages[jbx][ibx+1]._update_hdu_data(self.outimages[jbx][ibx], 'left', False)
                self.outimages[jbx][ibx]._load_or_save_hdu_list(False)
            self.outimages[jbx][nblock-1]._load_or_save_hdu_list(True)

        # vertical sharing
        for ibx in range(nblock):
            self.outimages[0][ibx]._load_or_save_hdu_list(True)
            for jbx in range(nblock-1):
                self.outimages[jbx+1][ibx]._load_or_save_hdu_list(True)
                self.outimages[jbx][ibx]._update_hdu_data(self.outimages[jbx+1][ibx], 'top', True)
                self.outimages[jbx+1][ibx]._update_hdu_data(self.outimages[jbx][ibx], 'bottom', False)
                self.outimages[jbx][ibx]._load_or_save_hdu_list(False)
            self.outimages[nblock-1][ibx]._load_or_save_hdu_list(False)
