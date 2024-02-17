import sys
import numpy as np
from pyimcom.config import Config
from pyimcom.coadd import Block

if __name__ == '__main__':
    # Read in information
    config_file = ''  # default
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    config = Config(config_file)
    config.extrainput = [None]
    config.n_inframe = 1
    config.stoptile = 4
    config.pad_sides = 'all'

    # subregion information
    this_sub = 0  # default
    if len(sys.argv) > 2:
        this_sub = int(sys.argv[2])

    # prime number to not do all the blocks next to each other first
    p = 739
    this_sub = (this_sub*p) % (config.nblock**2)
    block = Block(cfg=config, this_sub=this_sub)
