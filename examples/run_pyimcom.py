"""PyIMCOM example."""

import sys
import numpy as np
from pyimcom.config import Config, Settings as Stn
from pyimcom.coadd import Block

if __name__ == '__main__':
    # Read in information
    config_file = sys.argv[1]
    cfg = Config(config_file)

    # cfg.extrainput = [None, 'truth', 'cstar14', 'whitenoise1', '1fnoise2']
    cfg.extrainput = [None]
    cfg.stoptile = 4
    cfg.pad_sides = 'all'

    # subregion information
    this_sub = int(sys.argv[2])
    blk = Block(cfg=cfg, this_sub=this_sub)
