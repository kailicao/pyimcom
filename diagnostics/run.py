# This is an example script to generate a report


import sys
from .report import ValidationReport, ReportSection
from .stars import SimulatedStar
from .mosaicimage import MosaicImage
from .noise_diagnostics import NoiseReport

if __name__ == "__main__":
    # arguments are the input FITS file and the output stem
    rpt = ValidationReport(sys.argv[1], sys.argv[2], clear_all=True)
    sectionlist = [MosaicImage, SimulatedStar, NoiseReport]
    for cls in sectionlist:
            s=cls(rpt)
            s.build() # specify nblockmax to do just the lower corner
            rpt.addsections([s])
            del s
    rpt.compile()

    print('--> pdflatex log -->')
    print(str(rpt.compileproc.stdout))
