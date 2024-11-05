import numpy as np
import sys
from astropy.wcs import WCS
from utils import compareutils
#import json
import os
import re
from astropy.io import fits
from config import Config

# get the json file
config_file = sys.argv[1]

# load the file using Config and get information
cfgdata = Config(config_file)

info = cfgdata.inlayercache
ra = cfgdata.ra
dec = cfgdata.dec

# separate the path from the inlayercache info 
m = re.search(r'^(.*)\/(.*)', info)
if m:
    path = m.group(1)
    exp = m.group(2)
print(path, exp)

# create empty list of exposures
exps = []

# find all the fits files and add them to the list
for roots, dirs, files in os.walk(path):
    for file in files:
        m2 = re.search(r'(\w*)_(\d*)_(\d*).fits', file)
        if m2:
            sca = m2.group(3)
        #print(sca)
        if file.startswith(exp) and file.endswith(sca+'.fits'):
            exps.append(file)
print(exps)
# reduce the length of list for testing purposes
#test = exps[:3]
#print(len(test))

# move to directory with observation files
os.chdir(path)

# loop over the list of observation pair files
for exp in exps:
    print(exp)
    hdulist = fits.open(exp)
    #print(hdulist.info())
    data = np.copy(hdulist[0].data)

    # get wcs information from fits file
    #print(hdulist['SCIWCS'].header) 
    mywcs = WCS(hdulist['SCIWCS'].header)
    hdulist.close()
    
    # define pad
    pad = 0
    # convert to x, y, z using wcs coords
    x, y, z, p = compareutils.getfootprint(mywcs, pad)
    v = np.array([x,y,z])

    # define transformation matrix 
    ex = np.array([np.sin(ra), -np.cos(ra), 0])
    ey = np.array([-np.cos(ra)*np.sin(dec), np.sin(dec)*np.sin(ra), np.cos(dec)])
    ez = np.array([-np.cos(dec)*np.cos(ra), -np.cos(dec)*np.sin(ra), -np.sin(dec)])
    T = np.column_stack((ex,ey,ez))
    print(T.shape)
    
    # perform transformation and define terms
    v_p = np.matmul(T,v)
    x_p = v_p[0]
    y_p = v_p[1]
    z_p = v_p[2]
    print(x_p, y_p, z_p)
