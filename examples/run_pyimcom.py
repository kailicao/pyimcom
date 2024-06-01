import sys
import numpy as np
from pyimcom.config import Config, Settings as Stn
from pyimcom.coadd import Block

if __name__ == '__main__':
    # Read in information
    cfg_file = ''  # default
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]

    cfg = Config(cfg_file)
    # cfg.extrainput = [None, 'whitenoise1', '1fnoise2', 'skyerr', 'cstar14']
    cfg.extrainput = [None]
    cfg.n_inframe = len(cfg.extrainput)
    cfg.stoptile = 4
    cfg.pad_sides = 'all'

    # cfg.outstem = '/users/PAS2055/kailicao/pyimcom_dev/test/kappa=1e-5'
    # cfg.linear_algebra = 'Cholesky'
    cfg.kappaC_arr = np.array([0.0])
    cfg.instamp_pad = 0.625 * Stn.arcsec

    # subregion information
    this_sub = 0  # default
    if len(sys.argv) > 2:
        this_sub = int(sys.argv[2])

    # prime number to not do all the blocks next to each other first
    p = 739
    this_sub = (this_sub*p) % (cfg.nblock**2)
    block = Block(cfg=cfg, this_sub=this_sub)
