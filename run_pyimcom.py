from pyimcom.config import Config
from pyimcom.coadd import Block

if __name__ == '__main__':
    config = Config()
    config.inpsf_npix = 64
    config.stoptile = 4
    block = Block(cfg=config, this_sub=0)
