"""
This is an example script to generate a report.

Command-line arguments are:
#. Input FITS file
#. Output stem

"""

import sys

from .mosaicimage import MosaicImage
from .noise_diagnostics import NoiseReport
from .report import ValidationReport
from .stars import SimulatedStar

if __name__ == "__main__":
    rpt = ValidationReport(sys.argv[1], sys.argv[2], clear_all=True)
    sectionlist = [MosaicImage, SimulatedStar, NoiseReport]
    for cls in sectionlist:
        s = cls(rpt)
        s.build()  # specify nblockmax to do just the lower corner
        rpt.addsections([s])
        del s
    rpt.compile()

    print("--> pdflatex log -->")
    print(str(rpt.compileproc.stdout))
