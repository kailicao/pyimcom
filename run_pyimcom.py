from pyimcom.config import Config
from pyimcom.coadd import Block

if __name__ == '__main__':
    config = Config()
    config.extrainput = [None, 'truth', 'whitenoise1', '1fnoise2', 'skyerr', 'labnoise',
                         'gsstar14', 'cstar14', 'gsext14,Seed=1300',
                         'gsext14,Seed=1300,shear=.2:0', 'gsext14,Seed=1300,shear=0.:2e-1']
    config.n_inframe = 11
    config.stoptile = None
    block = Block(cfg=config, this_sub=0)
