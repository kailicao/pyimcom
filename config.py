'''
Encapsulation of pyimcom background settings and configuration.

Classes
-------
Timer : All-purpose timer.
Settings : pyimcom background settings.
Config : pyimcom configuration, with JSON file interface.

'''

from time import perf_counter
from importlib.resources import files
import json

from astropy import units as u
import numpy as np


class Timer:
    '''
    All-purpose timer.

    Methods
    -------
    __init__ : Constructor.
    __call__ : Return the time elapsed since tstart in seconds.

    '''

    def __init__(self) -> None:
        '''
        Constructor.

        Returns
        -------
        None.

        '''

        self.tstart = perf_counter()

    def __call__(self, reset: bool = False) -> float:
        '''
        Return the time elapsed since tstart in seconds.

        Parameters
        ----------
        reset : bool, optional
            Whether to reset tstart. The default is False.

        Returns
        -------
        float
            Time elapsed since tstart in seconds.

        '''

        tnow = perf_counter()
        tstart = self.tstart
        if reset:
            self.tstart = tnow
        return tnow - tstart


class Settings:
    '''
    pyimcom background settings.

    This class contains assorted Roman WFI data needed for the coadd code.

    '''

    # which header in the input file contains the WCS information
    hdu_with_wcs = 'SCI'

    degree = u.degree.to('rad')  # == np.pi/180.0 == 0.017453292519943295
    arcmin = u.arcmin.to('rad')  # == degree/60.0 == 0.0002908882086657216
    arcsec = u.arcsec.to('rad')  # == arcmin/60.0 == 4.84813681109536e-06

    # filter list
    RomanFilters  = ['W146', 'F184', 'H158', 'J129', 'Y106',
                     'Z087', 'R062', 'PRSM', 'DARK', 'GRSM', 'K213']
    QFilterNative = [ 1.155,  1.456,  1.250,  1.021,  0.834,
                      0.689,  0.491,  1.009,  0.000,  1.159,  1.685]

    # linear obscuration of the telescope
    obsc = 0.31

    # SCA parameters
    pixscale_native = 0.11 * arcsec
    sca_nside = 4088  # excludes reference pixels
    sca_ctrpix = (sca_nside-1) / 2
    sca_sidelength = sca_nside * pixscale_native

    # SCA field of view centers
    # SCAFov[i,j] = position of SCA #[i+1] (i=0..17) in j coordinate (j=0 for X, j=1 for Y)
    # these are in 'WFI local' field angles, in degrees
    # just for checking coverage since only 3 decimal places
    SCAFov = np.asarray([
        [-0.071, -0.037], [-0.071,  0.109], [-0.070,  0.240], [-0.206, -0.064],
        [-0.206,  0.083], [-0.206,  0.213], [-0.341, -0.129], [-0.341,  0.018], [-0.342,  0.147],
        [ 0.071, -0.037], [ 0.071,  0.109], [ 0.070,  0.240], [ 0.206, -0.064],
        [ 0.206,  0.083], [ 0.206,  0.213], [ 0.341, -0.129], [ 0.341,  0.018], [ 0.342,  0.147]
    ])


class Config:
    '''
    pyimcom configuration, with JSON file interface.

    Methods
    -------
    __init__ : Constructor.
    _from_file : Build a configuration from a JSON file.
    _get_attrs_wrapper: Wrapper for getting an attribute or a set of attributes.
    _build_config : Terminal interface to build a configuration from scratch.
    to_file : Save the configuration to a JSON file.

    '''

    def __init__(self, cfg_file: str = '') -> None:
        '''
        Constructor.

        Parameters
        ----------
        cfg_file : str, optional
            File path to or text content of a JSON configuration file.
            The default is ''. This uses pyimcom default_config.json.
            Set cfg_file=None to build a configuration from scratch.

        Returns
        -------
        None.

        '''

        self.cfg_file = cfg_file
        if cfg_file is not None:
            if cfg_file == '':
                # print('> Using default_config.json', flush=True)
                self.cfg_file = files(__package__).joinpath('default_config.json')

            try:
                with open(self.cfg_file) as f:
                    cfg_dict = json.load(f)
            except:
                cfg_dict = json.loads(self.cfg_file)
            self._from_dict(cfg_dict)

        else:
            self._build_config()

        # calculate derived quantities
        self.NsideP = self.Nside + self.postage_pad * self.n2 * 2
        self.n1P = self.n1 + self.postage_pad * 2
        self.n2f = self.n2 + self.fade_kernel * 2

    def _from_dict(self, cfg_dict: dict) -> None:
        '''
        Build a configuration from a dictionary.

        Parameters
        ----------
        cfg_dict : dict
            Usually built from a JSON file.

        Returns
        -------
        None.

        '''

        # input files
        self.obsfile = cfg_dict['OBSFILE']
        self.inpath, self.informat = cfg_dict['INDATA']
        # which filter to make coadd
        self.use_filter = cfg_dict['FILTER']
        # input PSF information
        self.inpsf_path, self.inpsf_format, self.inpsf_oversamp = cfg_dict['INPSF']

        # permanent mask file
        self.permanent_mask = cfg_dict.get('PMASK', None)
        # CR hit probability for stochastic mask
        self.cr_mask_rate = cfg_dict.get('CMASK', 0.0)
        # input images to stack at once
        self.extrainput = [None] + cfg_dict.get('EXTRAINPUT', [])
        self.n_inframe = len(self.extrainput)
        # threshold for masking lab noise data
        self.labnoisethreshold = cfg_dict.get('LABNOISETHRESHOLD', 3.0)

        # tile center in degrees RA, DEC
        self.ra, self.dec = cfg_dict['CTR']
        # if we are doing a nblock x nblock array on the same projection
        self.nblock = cfg_dict['BLOCK']
        # and output size: n1 (number of IMCOM postage stamps)
        # n2 (size of single run), dtheta (arcsec)
        # output array size will be (n1 x n2 x dtheta) on a side
        # with padding, it is (n1 + 2*postage_pad) n2 x dtheta on a side
        self.n1, self.n2, self.dtheta = cfg_dict['OUTSIZE']
        assert self.n1 % 2 == 0, 'Error: n1 must be even since PSF computations are in 2x2 groups'
        self.dtheta *= u.arcsec.to('degree')
        self.Nside = self.n1 * self.n2

        # fading kernel width
        self.fade_kernel = cfg_dict.get('FADE', 3)
        # pad this many IMCOM postage stamps around the edge
        self.postage_pad = cfg_dict.get('PAD', 0)
        # according to the strategy or on the sides specified by the user
        self.pad_sides = cfg_dict.get('PADSIDES', 'auto')
        # stop bulding the tile after a certain number of postage stamps
        self.stoptile = cfg_dict.get('STOP', 0)

        # choose which outputs to report
        self.outmaps = cfg_dict.get('OUTMAPS', 'USKTN')
        # output stem
        self.outstem = cfg_dict['OUT']
        # temporary storage
        # virtual memory will be used if this is not empty str
        self.tempfile = cfg_dict.get('TEMPFILE', '')
        if not self.tempfile: self.tempfile = None

        # number of target output PSF(s)
        self.n_out = cfg_dict.get('NOUT', 1)
        # target output PSF type
        self.outpsf = cfg_dict.get('OUTPSF', 'AIRYOBSC')
        # target output PSF extra smearing
        self.sigmatarget = cfg_dict.get('EXTRASMOOTH', 1.5 / 2.355)

        if self.n_out > 1:  # more than one target output PSF
            self.outpsf_extra = []
            self.sigmatarget_extra = []
            for j_out in range(1, self.n_out):
                self.outpsf_extra.append(cfg_dict.get(f'OUTPSF{j_out+1}', 'AIRYOBSC'))
                self.sigmatarget_extra.append(cfg_dict.get(f'EXTRASMOOTH{j_out+1}', 1.5 / 2.355))

        # width of PSF sampling/overlap arrays in native pixels
        self.npixpsf = cfg_dict.get('NPIXPSF', 48)
        # experimental feature to change the weighting of Fourier modes
        self.amp_penalty = cfg_dict.get('AMPPEN', (0.0, 0.0))
        # amount by which to penalize having different contributions
        # to the output from different input images
        self.flat_penalty = cfg_dict.get('FLATPEN', 0.0)
        # input stamp size padding (aka acceptance radius)
        self.instamp_pad = cfg_dict.get('INPAD', 1.055) * Settings.arcsec

        # kernel to solve linear systems
        self.linear_algebra = cfg_dict.get('LAKERNEL', 'Cholesky')

        # Lagrange multiplier (kappa) information
        # list of kappa/C values, ascending order
        self.kappaC_arr = np.array(cfg_dict.get('KAPPAC', [1e-5, 1e-4, 1e-3]))
        if self.linear_algebra == 'Empirical' or self.kappaC_arr.size == 1:
            self.outmaps = self.outmaps.replace('K', '')
        # target (minimum) leakage
        self.uctarget = cfg_dict.get('UCMIN', 1e-6)
        # maximum allowed value of Sigma
        self.sigmamax = cfg_dict.get('SMAX', 0.5)

        cfg_dict.clear(); del cfg_dict

    def _get_attrs_wrapper(self, code: str, newline: bool = True) -> None:
        '''
        Wrapper for getting an attribute or a set of attributes.

        Parameters
        ----------
        code : str
            Code segment to execute.
        newline : bool, optional
            Whether to add a blank line when finished. The default is True.

        Returns
        -------
        None.

        '''

        status = True
        while status:
            try:
                exec(code)
            except Exception as error:
                print(error)
                print('# Invalid input, please try again.', flush=True)
            else:
                status = False
                if newline: print()

    def _build_config(self) -> None:
        '''
        Terminal interface to build a configuration from scratch.

        The prompts are based on comments in old text configuration files.
        assert statements for further validity checks to be added.

        Returns
        -------
        None.

        '''

        print('### GENERAL NOTE: INPUT NOTHING TO USE DEFAULT ###' '\n', flush=True)

        print('### SECTION I: INPUT FILES ###' '\n', flush=True)
        # input files: OBSFILE, INDATA, FILTER, INPSF

        print('# input observation list', flush=True)
        self._get_attrs_wrapper(
            "self.obsfile = input('OBSFILE (str): ')")

        print('# reference input file directory and naming convention' '\n'
              '# (including the WCS used for stacking)', flush=True)
        self._get_attrs_wrapper(
            "self.inpath, self.informat = input('INDATA (str str): ').split(' ')")

        print('# which filter', flush=True)
        self._get_attrs_wrapper(
            "self.use_filter = int(input('FILTER (int): '))")

        print('# input PSF files & format & oversamp', flush=True)
        self._get_attrs_wrapper(
            "self.inpsf_path, self.inpsf_format, OVERSAMP = input('INPSF (str str int): ').split(' ')" '\n'
            "self.inpsf_oversamp = int(OVERSAMP)")

        print('### SECTION II: MASKS AND LAYERS ###' '\n', flush=True)
        # masks and layers: PMASK, CMASK, EXTRAINPUT, LABNOISETHRESHOLD

        print('# mask options:' '\n'
              '# PMASK --> permanent mask (from file)' '\n'
              '# default: no permanent pixel mask' '\n'
              '# CMASK --> cosmic ray hit probability for stochastic mask', flush=True)
        self._get_attrs_wrapper(
            "PMASK = input('PMASK (str) [default: None]: ')" '\n'
            "self.permanent_mask = PMASK if PMASK else None", newline=False)
        self._get_attrs_wrapper(
            "CMASK = input('CMASK (float) [default: 0.0]: ')" '\n'
            "self.cr_mask_rate = float(CMASK) if CMASK else 0.0")

        print('# extra inputs (input images to stack at once)' '\n'
              '# (use names for each one, space-delimited; meaning of names must be coded into' '\n'
              '# layer.get_all_data, with the meaning based on the naming convention in INDATA)', flush=True)
        self._get_attrs_wrapper(
            "EXTRAINPUT = input('EXTRAINPUT (str str ...) [default: None]: ')" '\n'
            "self.extrainput = [None] + (EXTRAINPUT.split() if EXTRAINPUT else [])" '\n'
            "self.n_inframe = len(self.extrainput)")

        print('# mask out pixels with lab noise beyond this threshold' '\n'
              '# (ignored if labnoise is not in EXTRAINPUT or does not exist)', flush=True)
        self._get_attrs_wrapper(
            "LABNOISETHRESHOLD = input('LABNOISETHRESHOLD (float) [default: 3.0]: ')" '\n'
            "self.labnoisethreshold = float(LABNOISETHRESHOLD) if LABNOISETHRESHOLD else 3.0")

        print('### SECTION III: WHAT AREA TO COADD ###' '\n', flush=True)
        # what area to coadd: CTR, BLOCK, OUTSIZE

        print('# location of the output region to make', flush=True)
        self._get_attrs_wrapper(
            "self.ra, self.dec = map(float, input('CTR (float float): ').split(' '))", newline=False)
        self._get_attrs_wrapper(
            "self.nblock = int(input('BLOCK (int): '))", newline=False)
        self._get_attrs_wrapper(
            "self.n1, self.n2, self.dtheta = map(eval, input('OUTSIZE (int int float): ').split(' '))" '\n'
            "assert self.n1 % 2 == 0, 'Error: n1 must be even since PSF computations are in 2x2 groups'" '\n'
            "self.dtheta *= u.arcsec.to('degree')" '\n'
            "self.Nside = self.n1 * self.n2")

        print('### SECTION IV: MORE ABOUT POSTAGE STAMPS ###' '\n', flush=True)
        # more about postage stamps: FADE, PAD, PADSIDES, STOP

        print('# fading kernel width (number of rows or columns' '\n'
              '# of transition pixels on each side of a postage stamp)', flush=True)
        self._get_attrs_wrapper(
            "FADE = input('FADE (int) [default: 3]: ')" '\n'
            "self.fade_kernel = float(FADE) if FADE else 3" '\n'
            "assert self.n2 > self.fade_kernel * 2, 'insufficient patch size'")

        print('# number of IMCOM postage stamps to pad around each output region', flush=True)
        self._get_attrs_wrapper(
            "PAD = input('PAD (int) [default: 0]: ')" '\n'
            "self.postage_pad = float(PAD) if PAD else 0")
    
        print('# on which side(s) to pad IMCOM postage stamps' '\n'
              '# "all": pad on all sides;' '\n'
              '# "auto": pad on mosaic boundaries only;' '\n'
              '# "none": pad on none of the sides;' '\n'
              '# otherwise, please specify which side(s) to pad on' '\n'
              '# using CAPITAL letters (the order does not matter)' '\n'
              '# "B" (bottom), "T" (top), "L" (left), and "R" (right)', flush=True)
        self._get_attrs_wrapper(
            "PADSIDES = input('PADSIDES (str) [default: \"auto\"]: ')" '\n'
            "self.pad_sides = PADSIDES if PADSIDES else 'auto'")

        print('# stop execution after a certain number of postage stamps' '\n'
              '# (for testing so we don\'t have to wait for all the postage stamps)' '\n'
              '# good choices: 16 to see if it runs; 624 to get a portion of the image without using lots of time' '\n'
              '# default: don\'t stop until we get to the end', flush=True)
        self._get_attrs_wrapper(
            "STOP = input('STOP (int) [default: 0]: ')" '\n'
            "self.stoptile = int(STOP) if STOP else 0")

        print('### SECTION V: WHAT AND WHERE TO OUTPUT ###' '\n', flush=True)
        # what and where to output: OUTMAPS, OUT, TEMPFILE

        print('# choose which outputs to report' '\n'
              '# U = PSF leakage map (U_alpha/C)' '\n'
              '# S = noise map' '\n'
              '# K = kappa (Lagrange multiplier map)' '\n'
              '# T = total weight (sum over all input pixels)' '\n'
              '# N = effective coverage', flush=True)
        self._get_attrs_wrapper(
            "OUTMAPS = input('OUTMAPS (str) [default: \"USKTN\"]: ')" '\n'
            "self.outmaps = OUTMAPS if OUTMAPS else 'USKTN'")

        print('# output location:' '\n'
              '# set to something in your directory', flush=True)
        self._get_attrs_wrapper(
            "self.outstem = input('OUT (str): ')")

        print('# temporary storage location (prefix):' '\n'
              '# not to use virtual memory', flush=True)
        self._get_attrs_wrapper(
            "TEMPFILE = input('TEMPFILE (str) [default: '']: ')" '\n'
            "self.tempfile = TEMPFILE", newline=False)

        print('### SECTION VI: TARGET OUTPUT PSF(S) ###' '\n', flush=True)
        # target output PSF(s): NOUT, OUTPSF, EXTRASMOOTH
        # (optional: OUTPSF2, EXTRASMOOTH2, etc.)

        print('# number of target output PSF(s)', flush=True)
        self._get_attrs_wrapper(
            "NOUT = input('NOUT (int) [default: 1]: ')" '\n'
            "self.n_out = (int(NOUT) if NOUT else 1)" '\n'
            "assert self.n_out >= 1, 'NOUT should be at least 1'")

        print('# target output PSF type, options include' '\n'
              '# "GAUSSIAN": simple Gaussian' '\n'
              '# "AIRYOBSC": obscured Airy disk convolved with Gaussian' '\n'
              '# "AIRYUNOBSC": unobscured Airy disk convolved with Gaussian', flush=True)
        self._get_attrs_wrapper(
            "OUTPSF = input('OUTPSF (str) [default: \"AIRYOBSC\"]: ')" '\n'
            "assert OUTPSF in ['', 'GAUSSIAN', 'AIRYOBSC', 'AIRYUNOBSC'], 'unrecognized type'" '\n'
            "self.outpsf = OUTPSF if OUTPSF else 'AIRYOBSC'")

        print('# smoothing of output PSF (units: input pixels, 1 sigma)' '\n'
              '# default: FWHM Gaussian smoothing divided by 2.355 to be a sigma', flush=True)
        self._get_attrs_wrapper(
            "EXTRASMOOTH = input('EXTRASMOOTH (float) [default: 1.5 / 2.355]: ')" '\n'
            "self.sigmatarget = float(EXTRASMOOTH) if EXTRASMOOTH else (1.5 / 2.355)")

        if self.n_out > 1:  # more than one target output PSF
            self.outpsf_extra = []
            self.sigmatarget_extra = []
            for j_out in range(1, self.n_out):
                print(f'# now talking about target output PSF {j_out+1}:' '\n'
                      '# output PSF type', flush=True)
                self._get_attrs_wrapper(
                    f"OUTPSF{j_out+1} = input('OUTPSF{j_out+1} (str) [default: \"AIRYOBSC\"]: ')" '\n'
                    f"assert OUTPSF{j_out+1} in ['', 'GAUSSIAN', 'AIRYOBSC', 'AIRYUNOBSC'], 'unrecognized type'" '\n'
                    f"self.outpsf_extra.append(OUTPSF{j_out+1} if OUTPSF{j_out+1} else 'AIRYOBSC')")
                print('# smoothing of output PSF', flush=True)
                self._get_attrs_wrapper(
                    f"EXTRASMOOTH{j_out+1} = input('EXTRASMOOTH{j_out+1} (float) [default: 1.5 / 2.355]: ')" '\n'
                    f"self.sigmatarget_extra.append(float(EXTRASMOOTH{j_out+1}) if EXTRASMOOTH{j_out+1} else (1.5 / 2.355))")

        print('### SECTION VII: BUILDING LINEAR SYSTEMS ###' '\n', flush=True)
        # building linear systems: NPIXPSF, AMPPEN, FLATPEN, INPAD

        # print('# size of PSF postage stamp in native pixels')
        print('# width of PSF sampling/overlap arrays in native pixels' '\n'
              '# preferably a nice number for FFT purposes', flush=True)
        self._get_attrs_wrapper(
            "NPIXPSF = input('NPIXPSF (int) [default: 48]: ')" '\n'
            "self.npixpsf = (int(NPIXPSF) if NPIXPSF else 48)")

        print('# experimental feature to change the weighting of Fourier modes' '\n'
              '# format: (amp, sig), where amp is the amplitude,' '\n'
              '# sig is the width in units of input pixels', flush=True)
        self._get_attrs_wrapper(
            "AMPPEN = input('AMPPEN (float float) [default: (0.0, 0.0)]: ')" '\n'
            "self.amp_penalty = map(float, AMPPEN.split(' ')) if AMPPEN else (0.0, 0.0)")

        print('# amount by which to penalize having different contributions' '\n'
              '# to the output from different input images', flush=True)
        self._get_attrs_wrapper(
            "FLATPEN = input('FLATPEN (float) [default: 0.0]: ')" '\n'
            "self.flat_penalty = (float(FLATPEN) if FLATPEN else 0.0)")

        print('# input stamp size padding (aka acceptance radius) in arcsec', flush=True)
        self._get_attrs_wrapper(
            "INPAD = input('INPAD (float) [default: 1.055]: ')" '\n'
            "self.instamp_pad = (float(INPAD) if INPAD else 1.055) * Settings.arcsec")

        print('### SECTION VIII: SOLVING LINEAR SYSTEMS ###' '\n', flush=True)
        # solving linear systems: LAKERNEL, KAPPAC, UCMIN, SMAX

        print('# kernel to solve linear systems, options include' '\n'
              '# "Eigen": kernel using eigendecomposition' '\n'
              '# "Cholesky": kernel using Cholesky decomposition' '\n'
              '# "Iterative": kernel using iterative method' '\n'
              '# "Empirical": kernel using empirical method', flush=True)
        self._get_attrs_wrapper(
            "LAKERNEL = input('LAKERNEL (str) [default: \"Cholesky\"]: ')" '\n'
            "self.linear_algebra = LAKERNEL if LAKERNEL else 'Cholesky'" '\n'
            "assert self.linear_algebra in ['Eigen', 'Cholesky', 'Iterative', 'Empirical'], 'unrecognized kernel'")

        print('# Lagrange multiplier (kappa) information' '\n'
              '# list of kappa/C values, ascending order' '\n'
              '# if LAKERNEL == "Empirical", only the first kappa/C value is used;' '\n'
              '# otherwise if single value: use this fixed kappa/C value;' '\n'
              '# if multiple values: "Eigen" performs a bisection search' '\n'
              '# between the first and last kappa/C values, while "Cholesky"' '\n'
              '# and "Iterative" kernels use these kappa/C values as nodes', flush=True)
        self._get_attrs_wrapper(
            "KAPPAC = input('KAPPAC (float ...) [default: [1e-5, 1e-4, 1e-3]]: ')" '\n'
            "self.kappaC_arr = np.array(list(map(float, KAPPAC.split(' '))) if KAPPAC else [1e-5, 1e-4, 1e-3])" '\n'
            "assert np.all(np.diff(self.kappaC_arr) > 0.0), 'must be in ascending order'")
        if self.linear_algebra == 'Empirical' or self.kappaC_arr.size == 1:
            self.outmaps = self.outmaps.replace('K', '')

        print('# target (minimum) leakage', flush=True)
        self._get_attrs_wrapper(
            "UCMIN = input('UCMIN (float) [default: 1e-6]: ')" '\n'
            "self.uctarget = float(UCMIN) if UCMIN else 1e-6")

        print('# maximum allowed value of Sigma', flush=True)
        self._get_attrs_wrapper(
            "SMAX = input('SMAX (float) [default: 0.5]: ')" '\n'
            "self.sigmamax = float(SMAX) if SMAX else 0.5")

        print('# To save this configuration, call Config.to_file.' '\n', flush=True)

    def to_file(self, fname: str = '') -> None:
        '''
        Save the configuration to a JSON file.

        Parameters
        ----------
        fname : str, optional
            JSON configuration file name to save to.
            The default is ''. This overwrites pyimcom default_config.json.
            Set fname=None to get a text version of the configuration.

        Returns
        -------
        Either None
        or str
            Text version of the configuration.

        '''

        cfg_dict = {}

        # input files
        cfg_dict['OBSFILE'] = self.obsfile
        cfg_dict['INDATA'] = [self.inpath, self.informat]
        cfg_dict['FILTER'] = self.use_filter
        cfg_dict['INPSF'] = [self.inpsf_path, self.inpsf_format, self.inpsf_oversamp]

        # masks and layers
        cfg_dict['PMASK'] = self.permanent_mask
        cfg_dict['CMASK'] = self.cr_mask_rate
        cfg_dict['EXTRAINPUT'] = self.extrainput[1:]
        cfg_dict['LABNOISETHRESHOLD'] = self.labnoisethreshold

        # what area to coadd
        cfg_dict['CTR'] = [self.ra, self.dec]
        cfg_dict['BLOCK'] = self.nblock
        cfg_dict['OUTSIZE'] = [self.n1, self.n2, self.dtheta * u.degree.to('arcsec')]

        # more about postage stamps
        cfg_dict['FADE'] = self.fade_kernel
        cfg_dict['PAD'] = self.postage_pad
        cfg_dict['PADSIDES'] = self.pad_sides
        cfg_dict['STOP'] = self.stoptile

        # what and where to output
        cfg_dict['OUTMAPS'] = self.outmaps
        cfg_dict['OUT'] = self.outstem
        cfg_dict['TEMPFILE'] = self.tempfile if self.tempfile else ''

        # target output PSF(s)
        cfg_dict['NOUT'] = self.n_out
        cfg_dict['OUTPSF'] = self.outpsf
        cfg_dict['EXTRASMOOTH'] = self.sigmatarget
        if self.n_out > 1:  # more than one target output PSF
            for j_out in range(1, self.n_out):
                cfg_dict[f'OUTPSF{j_out+1}'] = self.outpsf_extra[j_out-1]
                cfg_dict[f'EXTRASMOOTH{j_out+1}'] = self.sigmatarget_extra[j_out-1]

        # building linear systems
        cfg_dict['NPIXPSF'] = self.npixpsf
        cfg_dict['AMPPEN'] = self.amp_penalty
        cfg_dict['FLATPEN'] = self.flat_penalty
        cfg_dict['INPAD'] = self.instamp_pad / Settings.arcsec

        # solving linear systems
        cfg_dict['LAKERNEL'] = self.linear_algebra
        cfg_dict['KAPPAC'] = list(self.kappaC_arr)
        cfg_dict['UCMIN'] = self.uctarget
        cfg_dict['SMAX'] = self.sigmamax

        if fname is not None:
            if fname == '':
                print('> Overwriting default_config.json', flush=True)
                fname = files(__package__).joinpath('default_config.json')

            with open(fname, 'w') as f:
                json.dump(cfg_dict, f, indent=4)
            cfg_dict.clear(); del cfg_dict

        else:  # return text version of the configuration
            res = json.dumps(cfg_dict, indent=4)
            cfg_dict.clear(); del cfg_dict
            return res
