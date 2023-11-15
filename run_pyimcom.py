import numpy as np
from pyimcom.config import Config
from pyimcom.coadd import Block

if __name__ == '__main__':
    config = Config()
    # config.extrainput = [None, 'truth', 'whitenoise1', '1fnoise2', 'skyerr', 'labnoise',
    #                      'gsstar14', 'cstar14', 'gsext14,Seed=1300',
    #                      'gsext14,Seed=1300,shear=.2:0', 'gsext14,Seed=1300,shear=0.:2e-1']
    # config.n_inframe = 11
    config.stoptile = None
    config.kappa_arr = np.array([8.3908e-09, 8.3908e-08, 8.3908e-07])
    block = Block(cfg=config, this_sub=0)

# Read in information
# config_file = sys.argv[1]
# with open(config_file) as myf: content = myf.read().splitlines()

# subregion information
# if len(sys.argv)>2:
#     this_sub = int(sys.argv[2])

# prime number to not do all the blocks next to each other first
# p = 1567
# if self.cfg.nblock % p == 0: p = 281
# j = (self.this_sub*p) % (self.cfg.nblock**2)

#   choose_outputs = 'CKMSTU' (default); which outputs to report:
#      A, B, C = IMCOM matrices
#      K = kappa (Lagrange multiplier map)
#      M = coaddition input pixel mask
#      S = noise map
#      T = coaddition matrix
#      U = PSF leakage map (U_alpha/C)
#     (A and B are large and I don't try to save them if we aren't going to use them)
