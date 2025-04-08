import numpy as np
from pyimcom.config import Config, Settings as Stn
from pyimcom.coadd import Block

## Run a single block

#cfg = Config(None)
cfg = Config(cfg_file = 'config_prod_H.json')

blk = Block(cfg=cfg, this_sub=0, run_coadd=True)