from time import perf_counter
import json

from astropy import units as u
import numpy as np


class Timer:
    '''All-purpose timer.
    '''

    def __init__(self):
        self.tstart = perf_counter()

    def __call__(self, reset: bool = False):
        tnow = perf_counter()
        tstart = self.tstart
        if reset:
            self.tstart = tnow
        return tnow - tstart


class Settings:
    '''Background settings.
    '''

    fft_max_n_arr = 8
    # While np.fft.(i)rfft2 can handle multiple 2-d arrays at the same time,
    # the actual number of arrays should not exceed this due to memory limit.
    # For 2560 x 2560 arrays, in JupyterLab on OSC, this can be as large as 12.

    hdu_with_wcs = 'SCI'  # which header in the input file contains the WCS information

    ### This class contains assorted utilities and Roman WFI data needed for the coadd code. ###

    degree = u.degree.to('rad')  # np.pi/180. = 0.017453292519943295
    arcmin = u.arcmin.to('rad')  # degree/60. = 0.0002908882086657216
    arcsec = u.arcsec.to('rad')  # arcmin/60. = 4.84813681109536e-06

    # filter list
    RomanFilters = ['W146', 'F184', 'H158', 'J129', 'Y106',
                    'Z087', 'R062', 'PRSM', 'DARK', 'GRSM', 'K213']
    QFilterNative = [1.155,  1.456,  1.250,  1.021,  0.834,
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
        [-0.071, -0.037], [-0.071,  0.109], [-0.070,  0.240], [-0.206, -0.064], [-0.206,  0.083],
        [-0.206,  0.213], [-0.341, -0.129], [-0.341,  0.018], [-0.342,  0.147],
        [ 0.071, -0.037], [ 0.071,  0.109], [ 0.070,  0.240], [ 0.206, -0.064], [ 0.206,  0.083],
        [ 0.206,  0.213], [ 0.341, -0.129], [ 0.341,  0.018], [ 0.342,  0.147]
    ])


class Config:
    '''pyimcom configuration, with JSON file interface.
    '''

    # some default settings
    fade_kernel = 3  # fading kernel width

    def __init__(self, fname: str = 'default_config.json'):
        self.fname = fname
        if fname is not None:
            self.from_file()
        else:
            self.build_config()
        self.calc_deriv_qties()

    def from_file(self):
        with open(self.fname) as f: cfg = json.load(f)

        # input files
        self.obsfile = cfg['OBSFILE']
        self.inpath, self.informat = cfg['INDATA']

        # tile center in degrees RA, DEC
        self.ra, self.dec = cfg['CTR']
        # and output size: n1 (number of IMCOM blocks), n2 (size of single run), dtheta (arcsec)
        # output array size will be (n1 x n2 x dtheta) on a side
        # with padding, it is (n1 + 2*postage_pad) n2 x dtheta on a side
        self.n1, self.n2, self.dtheta = cfg['OUTSIZE']
        self.dtheta *= u.arcsec.to('degree')
        self.Nside = self.n1 * self.n2
        # if we are doing a nblock x nblock array on the same projection
        self.nblock = cfg['BLOCK']

        # which filter to make coadd
        self.use_filter = cfg['FILTER']

        # Lagrange multiplier (kappa) information
        # list of kappa values, ascending order
        # the current version always outputs kappa map
        self.kappa_arr = np.array(cfg['KAPPA'])

        # input PSF information
        self.inpsf_path, self.inpsf_format, self.inpsf_oversamp = cfg['INPSF']

        # output stem
        self.outstem = cfg['OUT']

        # stop bulding the tile after a certain number of postage stamps
        self.stoptile = cfg.get('STOP', None)

        # input stamp size padding, aka acceptance radius
        self.instamp_pad = cfg.get('INPAD', 1.055) * Settings.arcsec

        # pad this many IMCOM postage stamps around the edge
        self.postage_pad = cfg.get('PAD', 0)

        # output target PSF extra smearing
        self.sigmatarget = cfg.get('EXTRASMOOTH', 1.5 / 2.355)

        # input images to stack at once
        self.extrainput = [None] + cfg.get('EXTRAINPUT', [])
        self.n_inframe = len(self.extrainput)

        # permanent mask file
        self.permanent_mask = cfg.get('PMASK', None)
        # CR hit probability for stochastic mask
        self.cr_mask_rate = cfg.get('CMASK', 0.)
        # threshold for masking lab noise data
        self.labnoisethreshold = cfg.get('LABNOISETHRESHOLD', 1.)

        cfg.clear(); del cfg

    def calc_deriv_qties(self):
        self.NsideP = self.Nside + self.postage_pad * self.n2 * 2
        self.n1P = self.n1 + self.postage_pad * 2
        assert self.n1%2 == 0, 'Error: n1 must be even since PSF computations are in 2x2 groups'
        self.n2f = self.n2 + self.fade_kernel * 2

        print('General input information:')
        print('number of input frames = ', self.n_inframe, 'type =', self.extrainput)
        self.rpix_search_in = int(np.ceil((self.n2 * self.dtheta * Settings.degree / np.sqrt(2.0)
                                           + self.instamp_pad) / Settings.pixscale_native + 1))
        self.insize = self.rpix_search_in * 2
        print('input stamp radius -->', self.rpix_search_in,
              'native pixels   stamp={:3d}x{:3d}'.format(self.insize, self.insize))
        print()

    def get_attrs_wrapper(self, code: str, newline: bool = True):
        status = True
        while status:
            try:
                exec(code)
            except:
                print('# Invalid input, please try again.', flush=True)
            else:
                status = False
                if newline: print()

    def build_config(self):
        print('### THE PARAMETERS IN THIS SECTION ARE REQUIRED ###' '\n', flush=True)

        print('# input observation list', flush=True)
        self.get_attrs_wrapper(
            "self.obsfile = input('OBSFILE (str): ')")

        print('# reference input file directory and naming convention '
              '(including the WCS used for stacking)', flush=True)
        self.get_attrs_wrapper(
            "self.inpath, self.informat = input('INDATA (str str): ').split(' ')")

        print('# location of the output region to make', flush=True)
        self.get_attrs_wrapper(
            "self.ra, self.dec = map(float, input('CTR (float float): ').split(' '))", newline=False)
        self.get_attrs_wrapper(
            "self.n1, self.n2, self.dtheta = map(eval, input('OUTSIZE (int int float): ').split(' '))" '\n'
            "self.dtheta *= u.arcsec.to('degree')" '\n'
            "self.Nside = self.n1 * self.n2", newline=False)
        self.get_attrs_wrapper(
            "self.nblock = int(input('BLOCK (int): '))")

        print('# which filter', flush=True)
        self.get_attrs_wrapper(
            "self.use_filter = int(input('FILTER (int): '))")

        print('# kappa values', flush=True)
        self.get_attrs_wrapper(
            "self.kappa_arr = np.array([float(k) for k in input('KAPPA (float float ...): ').split(' ')])")

        print('# input PSF files & format & oversamp', flush=True)
        self.get_attrs_wrapper(
            "self.inpsf_path, self.inpsf_format, OVERSAMP = input('INPSF (str str int): ').split(' ')" '\n'
            "self.inpsf_oversamp = int(OVERSAMP)")

        print('# output location:' '\n'
              '# set to something in your directory', flush=True)
        self.get_attrs_wrapper(
            "self.outstem = input('OUT (str): ')")

        print('### END REQUIRED PARAMETERS ###' '\n', flush=True)

        print('### OPTIONAL PARAMETERS ###' '\n'
              '### INPUT NOTHING TO USE DEFAULT ###' '\n', flush=True)

        print('# stop execution after a certain number of postage stamps' '\n'
              '# (for testing so we don\'t have to wait for all the postage stamps)' '\n'
              '#' '\n'
              '# good choices: 16 to see if it runs; 624 to get a portion of the image without using lots of time' '\n'
              '# default: don\'t stop until we get to the end', flush=True)
        self.get_attrs_wrapper(
            "STOP = input('STOP (int) [default: None]: ')" '\n'
            "self.stoptile = int(STOP) if STOP else None")

        print('# input stamp size padding in arcsec', flush=True)
        self.get_attrs_wrapper(
            "INPAD = input('INPAD (float) [default: 1.055]: ')" '\n'
            "self.instamp_pad = (float(INPAD) if INPAD else 1.055) * Settings.arcsec")

        print('# number of IMCOM postage stamps to pad around each output region', flush=True)
        self.get_attrs_wrapper(
            "PAD = input('PAD (int) [default: 0]: ')" '\n'
            "self.postage_pad = float(PAD) if PAD else 0")

        print('# smoothing of output PSF (units: input pixels, 1 sigma)' '\n'
              '# default: FWHM Gaussian smoothing divided by 2.355 to be a sigma', flush=True)
        self.get_attrs_wrapper(
            "EXTRASMOOTH = input('EXTRASMOOTH (float) [default: 1.5 / 2.355]: ')" '\n'
            "self.sigmatarget = float(EXTRASMOOTH) if EXTRASMOOTH else (1.5 / 2.355)")

        print('# extra inputs' '\n'
              '# (use names for each one, space-delimited; meaning of names must be coded into' '\n'
              '# coadd_utils.py, with the meaning based on the naming convention in INDATA)', flush=True)
        self.get_attrs_wrapper(
            "EXTRAINPUT = input('EXTRAINPUT (str str ...) [default: None]: ')" '\n'
            "self.extrainput = [None] + (EXTRAINPUT.split() if EXTRAINPUT else [])" '\n'
            "self.n_inframe = len(self.extrainput)")

        print('# mask options:' '\n'
              '# PMASK --> permanent mask (from file)' '\n'
              '# default: no permanent pixel mask' '\n'
              '# CMASK --> cosmic ray mask (hit probability per pixel)', flush=True)
        self.get_attrs_wrapper(
            "PMASK = input('PMASK (str) [default: None]: ')" '\n'
            "self.permanent_mask = PMASK if PMASK else None", newline=False)
        self.get_attrs_wrapper(
            "CMASK = input('CMASK (float) [default: 0.]: ')" '\n'
            "self.cr_mask_rate = float(CMASK) if CMASK else 0.")

        print('### END OPTIONAL PARAMETERS ###' '\n', flush=True)
        print('# To save this configuration, call Config.to_file.' '\n', flush=True)

    def to_file(self, fname: str = 'sample_config.json'):
        cfg = {}

        cfg['OBSFILE'] = self.obsfile
        cfg['INDATA'] = [self.inpath, self.informat]

        cfg['CTR'] = [self.ra, self.dec]
        cfg['OUTSIZE'] = [self.n1, self.n2, self.dtheta * u.degree.to('arcsec')]
        cfg['BLOCK'] = self.nblock

        cfg['FILTER'] = self.use_filter
        cfg['KAPPA'] = list(self.kappa_arr)

        cfg['INPSF'] = [self.inpsf_path, self.inpsf_format, self.inpsf_oversamp]
        cfg['OUT'] = self.outstem

        if self.stoptile is not None:
            cfg['STOP'] = self.stoptile

        if self.instamp_pad != 1.055 * Settings.arcsec:
            cfg['INPAD'] = self.instamp_pad / Settings.arcsec

        if self.postage_pad != 0:
            cfg['PAD'] = self.postage_pad

        if self.sigmatarget != 1.5 / 2.355:
            cfg['EXTRASMOOTH'] = self.sigmatarget

        if self.n_inframe > 1:
            cfg['EXTRAINPUT']: self.extrainput[1:]

        if self.permanent_mask is not None:
            cfg['PMASK'] = self.permanent_mask
        if self.cr_mask_rate != 0.:
            cfg['CMASK'] = self.cr_mask_rate

        if fname is None:
            res = json.dumps(cfg, indent=4)
            cfg.clear(); del cfg
            return res

        with open(fname, 'w') as f:
            json.dump(cfg, f, indent=4)
        cfg.clear(); del cfg
