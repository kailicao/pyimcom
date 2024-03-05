# This file contains some basic data on the Roman FPA coordinates.
# It is needed for some of the tests.

import numpy as np

class fpaCoords:
   '''This contains some static data on the FPA coordinate system.

   It also has some associated static methods.
   '''

   # focal plane coordinates of SCA centers, in mm
   xfpa = np.array([-22.14, -22.29, -22.44, -66.42, -66.92, -67.42,-110.70,-111.48,-112.64,
                     22.14,  22.29,  22.44,  66.42,  66.92,  67.42, 110.70, 111.48, 112.64])
   yfpa = np.array([ 12.15, -37.03, -82.06,  20.90, -28.28, -73.06,  42.20,  -6.98, -51.06,
                     12.15, -37.03, -82.06,  20.90, -28.28, -73.06,  42.20,  -6.98, -51.06])
   # radius to circumscribe FPAs
   Rfpa = 151.07129575137697


   # orientation of SCAs
   # -1 orient means SCA +x pointed along FPA -x, SCA +y pointed along FPA -y
   # +1 orient means SCA +x pointed along FPA +x, SCA +y pointed along FPA +y (SCA #3,6,9,12,15,18)
   sca_orient = np.array([-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1]).astype(np.int16)

   pixsize = 0.01 # in mm
   nside = 4088 # number of active pixels on a side

   @staticmethod
   def pix2fpa(sca, x, y):
      '''Method to convert pixel (x,y) on a given sca to focal plane coordinates.

      Inputs:
         sca (in form 1..18)
         x and y (in pixels)
         sca, x, y may be scalars or arrays

      Outputs:
         xfpa, yfpa (in mm)
      '''

      if np.amin(sca)<1 or np.amax(sca)>18:
         raise ValueError('Invalid SCA in fpadata.pix2fpa, range={:d},{:d}'.format(np.amin(sca),np.amax(sca)))

      return (fpaCoords.xfpa[sca-1] + fpaCoords.pixsize*(x-(fpaCoords.nside-1)/2.)*fpaCoords.sca_orient[sca-1],
              fpaCoords.yfpa[sca-1] + fpaCoords.pixsize*(y-(fpaCoords.nside-1)/2.)*fpaCoords.sca_orient[sca-1])
