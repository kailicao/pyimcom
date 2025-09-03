"""
Helper functions to read metadata in output coadded map diagnostics

Functions
---------
UNIT_to_bels
    Convert a unit string to units of bels.
HDU_to_bels
    Convert a FITS header to units of bels.

"""

# may need to add more packages here in the future
import re

import numpy as np


def UNIT_to_bels(unitstring):
    """
    Convert a UNIT string (usually a FITS header value) to units of bels

    Parameters
    ----------
    unitstring : str
        A number and optional SI prefix, e.g. ``0.2uB`` (supports m=milli, u=micro, n=nano).

    Returns
    -------
    float
        Floating version of the `unitstring`, e.g., ``2.0e-7``.
        Returns np.nan if no match or unrecognized.

    """

    s = re.match(r"([\d\.\-\+eE]+)([mun]?)B", unitstring)
    if not s:
        return np.nan  # if fail pattern match

    x = float(s.group(1))  # number
    if s.group(2) == "m":
        x *= 1e-3
    if s.group(2) == "u":
        x *= 1e-6
    if s.group(2) == "n":
        x *= 1e-9

    return x


def HDU_to_bels(inhdu):
    """
    Reads the UNIT keyword from a FITS header and converts it to bels.

    Parameters
    ----------
    inhdu : astropy.io.fits.ImageHDU
        The FITS HDU to read.

    Returns
    -------
    float
        Floating version of the unitstring encoded in the FITS header.
        Returns np.nan if no match or unrecognized.

    """

    return UNIT_to_bels(inhdu.header["UNIT"])
