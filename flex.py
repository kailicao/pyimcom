# Galsim-compatible function to apply a second-order lensing effect
# on the image. This routine is based on GalFlex v0.5:
# GalFlex Copyright 2015 Justin Bird: http://physics.drexel.edu/~jbird/galflex/
# Modified for use with PyImcom by Katherine Laliotis

import galsim

def __linearInterpolate(xmax,N,xb,yb,betaImg):

    #Background flux of image
    betaMin=betaImg.min()
    thetaImg=np.zeros((N,N))

    dx = 2.0*xmax/N #Axis division length

    i1 = (N*(xb+xmax)/(2.0*xmax)-0.5).astype(int) #Closest index value of lensed coordinates in x
    j1 = (N*(yb+xmax)/(2.0*xmax)-0.5).astype(int) #Closest index value of lensed coordinates in y

    xc = -xmax+2.0*xmax/N*(i1+0.5) #Coordinate position of the closest index value in x
    yc = -xmax+2.0*xmax/N*(j1+0.5) #Coordinate position of the closest index value in y

    wx = 1.0 - (xb - xc)/dx #Difference between true coordinate position and index value position
    wy = 1.0 - (yb - yc)/dx #Weighting

    msk = (i1<0)|(j1<0)|(i1>=(N-1))|(j1>=(N-1)) #Mask to exclude index values located out of unlensed domain
    msk2 = ~msk #Inverse mask

    #Contribution to new cell from surrounding background cells
    a = j1[msk2]
    b = i1[msk2]
    beta1 = betaImg[a,b]
    beta2 = betaImg[a,b+1]
    beta3 = betaImg[a+1,b]
    beta4 = betaImg[a+1,b+1]

    #Add background cell contributions
    x = wx[msk2]
    y = wy[msk2]
    thetaImg[msk2] = x*y*beta1 + (1.-x)*y*beta2 + x*(1.-y)*beta3 + (1.-x)*(1.-y)*beta4

    #These excluded values use the minimum betaImg value
    thetaImg[msk] = betaMin

    return thetaImg

def lens(image, xmax, kap=0.0, gamma1=0.0, gamma2=0.0, F1=0.0, F2=0.0, G1=0.0, G2=0.0, cx=0.0, cy=0.0,**kwargs):
    if 'params' in kwargs:
        params = kwargs['params']
        kap = float(params['kappa'])
        gamma1 = float(params['gamma1'])
        gamma2 = float(params['gamma2'])
        F1 = float(params['F1'])
        F2 = float(params['F2'])
        G1 = float(params['G1'])
        G2 = float(params['G2'])

    N = len(image)
    betaImg = image
    xmax = xmax

    A11=1.0-kap-gamma1
    A22=1.0-kap+gamma1
    A12=-gamma2
    D111=-0.5*(G1+3*F1)
    D222=-0.5*(3*F2-G2)
    D112=-0.5*(F2+G2)
    D122=-0.5*(F1-G1)

    print('# A Matrix', np.stack((A11, A12, A12, A22)))
    print('# D Matrix Elements', np.stack((D111, D112, D122, D222)))

    coords = [-xmax+2.0*xmax/N*(i+0.5) for i in range(N)] #List of physical coordinates along the array axis
    X,Y = np.meshgrid(coords,coords) #Create mesh of 2 matrices - one for x values, one for y values of the coordinates
    
    #Tranforms lensed theta=(X,Y) to unlensed theta'=Aijthetaj+1/2Dijkthetajthetak
    X -= cx
    Y -= cy
    xb = cx + A11*X + A12*Y + 0.5*D111*X**2 + D112*X*Y + 0.5*D122*Y**2 #Create lensed coordinate x matrix
    yb = cy + A22*Y + A12*X + 0.5*D222*Y**2 + D122*X*Y + 0.5*D112*X**2 #Create lensed coordinate y matrix

    #Linear interpolation scheme
    image = __linearInterpolate(xmax,N,xb,yb,betaImg)
    return image

def BuildFlexedGalaxy(config, base, ignore, gsparams, logger):
    """Build a flexioned  galaxy object.

    Parameters:
        config:     The configuration dict of the object being built.
                    This should contain the Fparams, so it's base['Fparams']
        base:       The base configuration dict.
                    the galtype dictionary
        ignore:     A list of parameters that might be in the config dict,
                    but which may be ignored.  i.e. it is not an error for
                    these items to be present.
        gsparams:   An optional dict of items used to build a GSParams object
                    (may be None).
        logger:     An optional logger object to log progress (may be None).

    Returns:
        gsobject, safe

    The returned gsobject is the built GSObject instance, and safe is a bool
    value that indicates whether the object is safe to reuse for future stamps
    (e.g. if all the parameters used to build this object are constant and will
    not change for later stamps).
    """
    # If desired, log some output.
    if logger:
        logger.debug("Starting work on building Flexed Galaxy")

    # The gsparams are passed around using a dict so they can be easily added to.
    # At this point, we would typically convert them to a regular GSParams
    # instance to use when building the GSObject.
    if gsparams:
        gsparams = galsim.GSParams( **gsparams )

    # If you need a random number generator, this is the one to use.
    #rng = base['rng']

    # Build the GSObject
    
    # First initiate the galaxy object with simple exponential profile
    gsobject = galsim.Sersic(base['sersic']['n'], half_light_radius=base['sersic']['r'][n], flux=1.0,trunc=base['sersic']['t__r']*base['sersic']['r'][n])
    
    # Draw the galaxy image
    image = galsim.ImageD(N,N)
    #N should be image subdivisions in units of pixels. need to be able to read this from elsewhere?
    
    gsobject = gsobject.draw(image, dx=0.11, normalization='flux').array #dx is imcom px scale

    # Define the lensing/flexing params: [kappa, gamma1, gamma2, f1, f2, g1, g2, cx, cy]
    if 'Fparams' in base:
        params = base['Fparams'] #This statement is redundant by design I think
    else:
        params=config
    
    #Flex / lens the galaxy
    Flexed_gsobject = lens(gsobject, 10, params=params)

    safe = False  # typically, but set to True if this object is safe to reuse.
    return Flexed_gsobject, safe

galsim.config.RegisterObjectType('FlexedGalaxy', BuildFlexedGalaxy)

