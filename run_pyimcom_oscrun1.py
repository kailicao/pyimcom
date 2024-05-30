import sys
import numpy as np
from .config import Config
from .coadd import Block

config_file = sys.argv[1]
cfg = Config(config_file)
block = Block(cfg=cfg, this_sub=int(sys.argv[2]))
