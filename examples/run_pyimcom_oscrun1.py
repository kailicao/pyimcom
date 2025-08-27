"""
This is a script to run one of the blocks.

Usage is python3 -m pyimcom.run_pyimcom_oscrun1 <config> <block> [draw_only]

If draw_only is specified, then stops after drawing the input layers.

This version is for use on OSC: the temporary path is set to the $TMPDIR variable.
(We are not supposed to use /tmp for this purpose.) For other platforms, you might
modify this line.

The "0" block also generates the truth catalog.

"""

import os
import sys

from .coadd import Block
from .config import Config
from .truthcats import gen_truthcats_from_cfg

config_file = sys.argv[1]
cfg = Config(config_file)
if len(sys.argv) > 3 and sys.argv[3] == "draw":
    cfg.stoptile = 4

cfg.tempfile = os.getenv("TMPDIR") + "/temp"
print(cfg.to_file(None))

# coadd this block
block = Block(cfg=cfg, this_sub=int(sys.argv[2]))

# generate the truth catalog (but only once)
if int(sys.argv[2]) == 0:
    gen_truthcats_from_cfg(cfg)
