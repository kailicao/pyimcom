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
    # config.extrainput = [None, 'truth', 'whitenoise1', '1fnoise2', 'skyerr', 'labnoise',
    #                       'gsstar14', 'cstar14', 'gsext14,Seed=1300',
    #                       'gsext14,Seed=1300,shear=.2:0', 'gsext14,Seed=1300,shear=0.:2e-1']
    # config.n_inframe = len(config.extrainput) - 1
    config.stoptile = None
    config.kappa_arr = np.array([8.3908e-09, 8.3908e-08, 8.3908e-07])
    config.flat_penalty = 0.0
    config.pad_sides = 'all'
    config.tempfile = '/tmp/kailicao-tempy'

    # subregion information
    this_sub = 0  # default
    if len(sys.argv) > 2:
        this_sub = int(sys.argv[2])

    # prime number to not do all the blocks next to each other first
    p = 739
    this_sub = (this_sub*p) % (config.nblock**2)
    block = Block(cfg=config, this_sub=this_sub)
