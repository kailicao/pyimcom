# This is an example script to generate a report


import sys
from .report import ValidationReport, ReportSection
from .stars import SimulatedStar
from .mosaicimage import MosaicImage

if __name__ == "__main__":
    # arguments are the input FITS file and the output stem
    rpt = ValidationReport(sys.argv[1], sys.argv[2], clear_all=True)
    for cls in [MosaicImage, SimulatedStar]:
        if True:
            s=cls(rpt)
            s.build() # specify nblockmax to do just the lower corner
            rpt.addsections([s])
            del s
        #except:
        #    print('Failed to build', cls)
    rpt.compile()

    print('--> pdflatex log -->')
    print(str(rpt.compileproc.stdout))
