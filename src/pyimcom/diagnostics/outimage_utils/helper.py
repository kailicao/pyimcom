# helper functions to read metadata in output coadded map diagnostics
#

# may need to add more packages here in the future
import numpy as np
import re
from astropy.io import fits

def UNIT_to_bels(unitstring):
   """Convert a UNIT string (usually a FITS header value) to units of bels
   Should have a number and optional SI prefix, 0.2uB (supports m=milli, u=micro, n=nano)
   Output is a floating point number, in this case 2.0e-7
   Returns np.nan if no match or unrecognized
   """

   s = re.match(r'([\d\.\-\+eE]+)([mun]?)B', unitstring)
   if not s: return np.nan # if fail pattern match

   x = float(s.group(1)) # number
   if s.group(2)=='m': x*= 1e-3
   if s.group(2)=='u': x*= 1e-6
   if s.group(2)=='n': x*= 1e-9

   return x

def HDU_to_bels(inhdu):
   """Reads the UNIT keyword from a FITS header and converts it to bels
   Returns np.nan if no match or unrecognized
   """

   return UNIT_to_bels(inhdu.header['UNIT'])
