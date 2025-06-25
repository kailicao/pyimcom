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
block_path = cfgdata.outstem
ra = cfgdata.ra * (np.pi/180) # convert to radians
dec = cfgdata.dec * (np.pi/180) # convert to radians
lonpole = cfgdata.lonpole * (np.pi/180) # convert to radians
nblock = cfgdata.nblock
n1 = cfgdata.n1
n2 = cfgdata.n2
dtheta_deg = cfgdata.dtheta 
blocksize_rad = n1*n2*dtheta_deg * (np.pi)/180 # convert to radians
# print(ra, dec, lonpole, nblock, n1, n2, dtheta_deg)

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
        if file.startswith(exp) and file.endswith('.fits') and file[-6].isdigit():
            exps.append(file)
print('list of files:', exps)

# move to the directory with the files
os.chdir(path)

# loop over the list of observation pair files (for each SCA)
for exp in exps:
    # get SCA and obsid
    m2 = re.search(r'(\w*)_0*(\d*)_(\d*).fits', exp)
    if m2: 
        obsid = int(m2.group(2))
        sca = int(m2.group(3))
    print(obsid, sca)
    
    # inlayercache data
    hdul = fits.open(exp)

    # read in the input image, I
    I = np.copy(hdul[0].data) # this is I

    # get wcs information from fits file
    mywcs = WCS(hdul['SCIWCS'].header)
    hdul.close()

    # results from splitpsf
    # read in the kernel
    hdul2 = fits.open('{}.psf/psf_{:d}.fits'.format(info,obsid))
    K = np.copy(hdul2[sca+hdul2[0].header['KERSKIP']].data)
    # get the number of pixels on the axis
    axis_num = K.shape[1]
    # get the oversampling factor
    oversamp = hdul2[0].header['OVSAMP']
    hdul2.close()

    # define pad
    pad = 0
    # convert to x, y, z using wcs coords (center of SCA)
    x, y, z, p = compareutils.getfootprint(mywcs, pad)
    v = np.array([x,y,z])

    # convert to x', y', z'
    # define coordinates and transformation matrix
    ex = np.array([np.sin(ra), -np.cos(ra), 0])
    ey = np.array([-np.cos(ra)*np.sin(dec), -np.sin(dec)*np.sin(ra), np.cos(dec)])
    ez = np.array([-np.cos(dec)*np.cos(ra), -np.cos(dec)*np.sin(ra), -np.sin(dec)])
    T = np.array([ex,ey,ez])

    # perform transformation and define individual values
    v_p = np.matmul(T,v)
    x_p = v_p[0]
    y_p = v_p[1]
    z_p = v_p[2]
    
    # define the rotation matrix, coefficient, and additional vector
    rot = np.array([[-np.cos(lonpole), -np.sin(lonpole)],[np.sin(lonpole), -np.cos(lonpole)]])
    coeff = 2/(1-z_p)
    v_convert = np.array([x_p, y_p])

    # convert to eta and xi (block coordinates)
    block_coords = coeff * np.matmul(rot, v_convert)
    xi = block_coords[0]
    eta = block_coords[1]

    # find theta in original coordinates, convert to block coordinates
    s_in_rad = 0.11 * np.pi/(180*3600) # convert arcsec to radians
    ker_size = axis_num/oversamp * s_in_rad 
    theta = (2 * np.arctan(np.sqrt(p/(2-p))) + blocksize_rad/np.sqrt(2) + np.sqrt(2)*pad + ker_size/np.sqrt(2)) * coeff
    theta_block = theta / blocksize_rad
    # sigma = (nblock*blocksize_rad)/np.sqrt(2)    # I don't think I need these for the grid method
    # theta_max = theta * (1+(sigma**2)/4)
     
    # add theta to this set of coords
    block_coords = np.append(block_coords, theta)

    # convert the units of this coordinate system to blocks
    block_coords_blocks = block_coords / blocksize_rad

    # find the center of SCA relative to the bottom left of the mosaic
    SCA_coords = block_coords_blocks.copy()
    SCA_coords[:2] += (nblock/2) # take only the xi and eta coordinates

    # find the blocks the SCA covers in block units
    side = np.arange(nblock)+0.5
    xx, yy = np.meshgrid(side, side)
    distance = np.hypot(xx - SCA_coords[0], yy - SCA_coords[1])
    in_SCA = np.where(distance <= theta_block)
    block_list = np.stack((in_SCA[1], in_SCA[0]), axis = -1)
    # print(SCA_coords, block_list)
    # print('>', blocksize_rad, xi, eta, v)

    # loop over the blocks in the list
    for ix,iy in block_list:
        print(block_path+'_{:02d}_{:02d}.fits'.format(ix,iy))
        hdul3 = fits.open(block_path+'_{:02d}_{:02d}.fits'.format(ix,iy))
        block_data = np.copy(hdul3[0].data)