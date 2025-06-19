import numpy as np
import sys
from astropy.wcs import WCS
from utils import compareutils
import os
import re
from astropy.io import fits
from config import Config

# get the json file
config_file = sys.argv[1]

# load the file using Config and get information
cfgdata = Config(config_file)

info = cfgdata.inlayercache
ra = cfgdata.ra * (np.pi/180) # convert to radians
dec = cfgdata.dec * (np.pi/180) # convert to radians
lonpole = cfgdata.lonpole
nblock = cfgdata.nblock
n1 = cfgdata.n1
n2 = cfgdata.n2
dtheta_deg = cfgdata.dtheta  
blocksize_rad = n1*n2*dtheta_deg * (np.pi)/180 # convert to radians

# separate the path from the inlayercache info 
m = re.search(r'^(.*)\/(.*)', info)
if m:
    path = m.group(1)
    exp = m.group(2)

# create empty list of exposures
exps = []

# find all the fits files and add them to the list
for roots, dirs, files in os.walk(path):
    for file in files:
        m2 = re.search(r'(\w*)_(\d*)_(\d*).fits', file)
        if m2: sca = m2.group()
        if file.startswith(exp) and file.endswith('.fits'):
            exps.append(file)
print('list of files:', exps)

# move to the directory with the files
os.chdir(path)

# loop over the list of observation pair files (for each SCA)
for exp in exps:
    hdulist = fits.open(exp)
    data = np.copy(hdulist[0].data)
    
    # get wcs information from fits file
    mywcs = WCS(hdulist['SCIWCS'].header)
    hdulist.close()

    # define pad
    pad = 0
    # convert to x, y, z using wcs coords (center of SCA)
    x, y, z, p = compareutils.getfootprint(mywcs, pad)
    v = np.array([x,y,z])

    # convert to x', y', z'
    # define coordinates and transformation matrix
    ex = np.array([np.sin(ra), -np.cos(ra), 0])
    ey = np.array([-np.cos(ra)*np.sin(dec), np.sin(dec)*np.sin(ra), np.cos(dec)])
    ez = np.array([-np.cos(dec)*np.cos(ra), -np.cos(dec)*np.sin(ra), -np.sin(dec)])
    T = np.column_stack((ex,ey,ez))

    # perform transformation and define individual values
    v_p = np.matmul(T,v)
    x_p = v_p[0]
    y_p = v_p[1]
    z_p = v_p[2]
    
    # define the rotation matrix, coefficient, and additional vector
    rot = np.array([[-np.cos(lonpole), -np.sin(lonpole)],[np.sin(lonpole), -np.cos(lonpole)]])
    coeff = 1/(1-z_p)
    v_convert = np.array([x_p, y_p])

    # convert to eta and xi (block coordinates)
    block_coords = coeff * np.matmul(rot, v_convert)
    xi = block_coords[0]
    eta = block_coords[1]

    # find theta in original coordinates, convert to block coordinates
    theta = 2 * np.arctan(np.sqrt(p/(2-p)))
    sigma = (nblock*blocksize_rad)/np.sqrt(2)
    theta_max = theta * (1+(sigma**2)/4)
     
    # add theta to this set of coords
    block_coords = np.append(block_coords, theta_max)

    # convert the units of this coordinate system to blocks
    block_coords_blocks = block_coords / blocksize_rad

    # find the center of SCA relative to the bottom left of the mosaic
    SCA_coords = block_coords_blocks.copy()
    SCA_coords[:2] += (nblock/2)
    

