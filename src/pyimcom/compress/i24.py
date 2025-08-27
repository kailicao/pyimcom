"""Functions to compress float arrays to 24 bit integers.

Functions
---------
lsbf_fwd
    Least significant bits moved first.
lsbf_rev
    Inverse of `lsbf_fwd`.
diff_fwd
    Replaces an image with an array of differences.
diff_rev
    Inverse of `diff_fwd`.
smallnum_fwd
    Re-maps numbers of small absolute value to be near 0 when unsigned.
smallnum_rev
    Inverse of `smallnum_fwd`.
i24compress
    Compresses an array using the 'I24' scheme.
i24decompress
    Decompresses an array using the 'I24' scheme.

Classes
-------
I24Cube
    Object that can be reformatted as floating point, integer, or compressed integer
    in the 'I24' scheme.

"""

import copy

import numpy as np
from astropy.io import fits

### Utilities for reorganizing uint8 cubes ###


def lsbf_fwd(im):
    """
    Takes a 2D uint8 image and swaps the bits so that the least significant
    bit goes first, then the next, etc. and the most significant bit goes last.

    If given a 3D image, then does each 2D slice. The intent would be to use
    this on objects with a shape of (3,ny,nx); it will work for any number of slices,
    but it will be slow if the number of slices (3 in this case) is large.

    Parameters
    ----------
    im : np.array of uint8
        2D or 3D image.

    Returns
    -------
    out_im : np.array of uint8
        2D or 3D bit-swapped image. Same shape as `im`.

    See Also
    --------
    lsbf_rev : Inverse function.

    """

    # 3D images are done one slice at a time
    if len(np.shape(im)) == 3:
        out_im = np.zeros_like(im)
        for j in range(np.shape(im)[0]):
            out_im[j, :, :] = lsbf_fwd(im[j, :, :])
        return out_im

    # now we have a 2D image
    ny, nx = np.shape(im)
    return np.packbits(
        np.transpose(np.unpackbits(im, bitorder="little").reshape((ny, nx, 8)), axes=(2, 0, 1)).reshape(
            (ny, nx, 8)
        ),
        bitorder="little",
    ).reshape((ny, nx))
    bits = np.zeros((8, ny, nx), dtype=bool)
    for j in range(8):
        bits[j, :, :] = (im >> j) % 2 == 1
    bits = bits.reshape((ny, nx, 8))
    out_im = np.zeros_like(im)
    for j in range(8):
        out_im += (2**j * bits[:, :, j]).astype(np.uint8)
    return out_im


def lsbf_rev(im):
    """Inverse of the bit-swapping function lsbf_fwd.

    Parameters
    ----------
    im : np.array of uint8
        2D or 3D image.

    Returns
    -------
    out_im : np.array of uint8
        2D or 3D bit-swapped image. Same shape as `im`.

    See Also
    --------
    lsbf_fwd : Forward function.

    """

    # 3D images are done one slice at a time
    if len(np.shape(im)) == 3:
        out_im = np.zeros_like(im)
        for j in range(np.shape(im)[0]):
            out_im[j, :, :] = lsbf_rev(im[j, :, :])
        return out_im

    # now we have a 2D image
    ny, nx = np.shape(im)
    return np.packbits(
        np.transpose(np.unpackbits(im, bitorder="little").reshape((8, ny, nx)), axes=(1, 2, 0)),
        bitorder="little",
    ).reshape((ny, nx))


### Utilities for differencing integer cubes ###


def diff_fwd(im, bitkeep):
    """
    Replace an int32 image with differences of the same shape.

    Parameters
    ----------
    im : np.array of int32
        A 2D input image.
    bitkeep : int
        Number of bits to keep. The maximum is 31.

    Returns
    -------
    np.array of int32
        The image of differences, same shape as `im`.

    See Also
    --------
    diff_rev : Inverse function.

    """

    ny, nx = np.shape(im)
    c = im.flatten()
    c[1:] = c[1:] - c[:-1]
    c = (2**bitkeep + c) % 2**bitkeep
    return c.reshape((ny, nx)).astype(np.int32)


def diff_rev(im, bitkeep):
    """Reconstruct an image from differences. Inverse function of diff_fwd.

    Parameters
    ----------
    im : np.array of int32
        A 2D input image of differences.
    bitkeep : int
        Number of bits to keep. The maximum is 31.

    Returns
    -------
    np.array of int32
        The original image, same shape as `im`.

    See Also
    --------
    diff_fwd : Forward function.

    """

    ny, nx = np.shape(im)
    c = im.astype(np.uint32).flatten()
    c = np.cumsum(c) & np.uint32(2**bitkeep - 1)
    return c.reshape((ny, nx)).astype(np.int32)


### Utilities for packaging small numbers together ###


def smallnum_fwd(im, bitkeep):
    """
    Remapping to package small differences of either sign near 0.

    Works on an int32 image with bitkeep bits used
    and negative numbers rolling over as ``-j --> 2**bitkeep-j``.

    Parameters
    ----------
    im : np.array of int32
        A 2D input image.
    bitkeep : int
        Number of bits to keep. The maximum is 31.

    Returns
    -------
    np.array of int32
        Re-mapped image, same shape as `im`.

    See Also
    --------
    smallnum_rev : Inverse function.

    """

    return np.where(im >= 2 ** (bitkeep - 1), 2 * (2**bitkeep - im) - 1, 2 * im)


def smallnum_rev(im, bitkeep):
    """
    Reconstructs images that have been re-mapped. Inverse of smallnum_fwd.

    Parameters
    ----------
    im : np.array of int32
        A 2D input image that has been re-mapped.
    bitkeep : int
        Number of bits to keep. The maximum is 31.

    Returns
    -------
    np.array of int32
        Original image, same shape as `im`.

    See Also
    --------
    smallnum_fwd : Forward function.

    """

    return np.where(im % 2, 2**bitkeep - 1 - im // 2, im // 2)


class I24Cube:
    """
    Class to compress a float cube to int24 with scaling, and reconstruct the original image.

    Parameters
    ----------
    inarray : np.array
        The input array. Options are
        * 2D float32 (original image)
        * 2D int32 (intermediate step: leading byte not used)
        * 3D uint8 (compressed form)
    pars : dict
        Parameters for the compression scheme. The possible keys are
        VMIN, VMAX, SOFTBIAS, DIFF, ALPHA, BITKEEP, and REORDER.
    overflow : astropy.io.fits.BinTableHDU or None, optional
        Overflow table (y,x,value). Needed if a compressed image is given as input.

    Attributes
    ----------
    ny : int
        Height of image
    nx : int
        Width of image
    pars : dict
        Compression parameters specified when constructed.
    data : np.array
        Current image data.
    mode : str
        Current image type. One of 'float32', 'int32', or 'uint8'.
    vmin : float
        Minimum of compression range.
    vmax : float
        Maximum of compression range.
    alpha : float
        Power law index of compression (1=linear).
    bitkeep : int
        Number of bits to keep (maximum=24).
    reorder : bool
        Implement bit reordering?
    softbias : int
        Integer in [0,2**24) to add (to avoid slight negative fluctuations being 111111).
        If -1, uses smallnum compression instead.
    diff : bool
        Use difference of successive pixels?
    overflow : astropy.io.fits.BinTableHDU or None
        Overflow table (y,x,value).
    reorder = use bit reordering? (boolean)

    Methods
    --------
    __init__
        Constructor.
    to_mode
        Change image format to a different (often compressed) storage mode.

    """

    def __init__(self, inarray, pars, overflow=None):
        # figure out what we have
        self.pars = copy.copy(pars)
        s = np.shape(inarray)
        self.ny, self.nx = s[-2:]
        self.data = copy.copy(inarray)
        d = len(s)

        # get the mode
        if d == 2 and inarray.dtype.name == "float32":
            self.mode = "float32"
        elif d == 2 and inarray.dtype.name == "int32":
            self.mode = "int32"
        elif d == 3 and inarray.dtype.name == "uint8":
            self.mode = "uint8"
        else:
            raise Exception("Can't initialize I24Cube: unrecognized data type or dimension.")

        # extract minimum and maximum (these need to be provided)
        self.vmin = float(pars["VMIN"])
        self.vmax = float(pars["VMAX"])
        if "SOFTBIAS" in pars:
            self.softbias = int(pars["SOFTBIAS"])
        else:
            self.softbias = 0
        if "DIFF" in pars:
            self.diff = bool(pars["DIFF"])
        else:
            self.diff = False
        if "ALPHA" in pars:
            self.alpha = float(pars["ALPHA"])
        else:
            self.alpha = 1.0
        if "BITKEEP" in pars:
            self.bitkeep = int(pars["BITKEEP"])
            if self.bitkeep >= 24 or self.bitkeep <= 0:
                raise Exception(f"Can't keep {self.bitkeep:d} bits")
        else:
            self.bitkeep = 24
        if "REORDER" in pars:
            self.reorder = bool(pars["REORDER"])
        else:
            self.reorder = True

        # overflow
        self.overflow = overflow

    def to_mode(self, mode):
        """
        Converts to the given mode.

        Parameters
        ----------
        mode : str
            Options are 'float32', 'int32', and 'uint8'.

        Returns
        -------
        None

        """

        if mode not in ["float32", "int32", "uint8"]:
            raise Exception(f"Unrecognized mode: {mode:s}")

        # nothing to do if there's no conversion
        if self.mode == mode:
            return

        # need to convert float32->int32
        if self.mode == "float32":
            posy, posx = np.where(np.logical_or(self.data < self.vmin, self.data > self.vmax))
            self.overflow = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="y", format="J", array=posy),
                    fits.Column(name="x", format="J", array=posx),
                    fits.Column(name="value", format="E", array=np.copy(self.data[posy, posx])),
                ]
            )
            y = (np.clip(self.data, self.vmin, self.vmax) - self.vmin) / (self.vmax - self.vmin)
            y = 2**self.bitkeep * y**self.alpha
            self.data = np.clip(np.floor(y).astype(np.int32), 0, 2**self.bitkeep - 1)
            del y
            if self.diff:
                self.data = diff_fwd(self.data, self.bitkeep)
            if self.softbias > 0:
                self.data = ((self.softbias + self.data) % 2**self.bitkeep).astype(np.int32)
            elif self.softbias == -1:
                self.data = smallnum_fwd(self.data, self.bitkeep)
            self.mode = "int32"

        # need to convert uint8->int32
        if self.mode == "uint8":
            if self.reorder:
                x = lsbf_rev(self.data).astype(np.int32)
            else:
                x = copy.copy(self.data.astype(np.int32))
            self.data = np.zeros((self.ny, self.nx), dtype=np.int32)
            for j in range(np.shape(x)[0]):
                self.data += x[j, :, :] << (8 * j)
            self.mode = "int32"

        # if we're done
        if self.mode == mode:
            return

        # if we still need to convert, we are starting from int32

        # need to convert int32->float32
        if mode == "float32":
            # note transformations are in the opposite order to float32->int32
            if self.softbias > 0:
                self.data = (2**self.bitkeep - self.softbias + self.data) % 2**self.bitkeep
            elif self.softbias == -1:
                self.data = smallnum_rev(self.data, self.bitkeep)
            if self.diff:
                self.data = diff_rev(self.data, self.bitkeep)
            y = (0.5 + self.data) / 2**self.bitkeep
            self.data = (self.vmin + (self.vmax - self.vmin) * y ** (1 / self.alpha)).astype(np.float32)
            if self.overflow is not None:
                posy = np.array(self.overflow.data["y"])
                posx = np.array(self.overflow.data["x"])
                self.data[posy, posx] = self.overflow.data["value"]
            self.mode = "float32"

        # need to convert int32->uint8
        if mode == "uint8":
            # make the first axis however big it needs to be
            newarray = np.zeros(((self.bitkeep + 7) // 8, self.ny, self.nx), dtype=np.uint8)
            newarray[0, :, :] = self.data % 256
            if self.bitkeep > 8:
                self.data >>= 8
                newarray[1, :, :] = self.data % 256
            if self.bitkeep > 16:
                self.data >>= 8
                newarray[2, :, :] = self.data % 256
            if self.reorder:
                self.data = lsbf_fwd(newarray)
            else:
                self.data = newarray
            self.mode = "uint8"


# stand-alone functions


def i24compress(im, scheme, pars):
    """
    Compresses an image.

    Parameters
    ----------
    scheme : str
        Compression scheme (right now supports 'I24A', 'I24B').
    pars : dict
        Parameters to pass to compression algorithm.

    Returns
    -------
    data : np.array
        The compressed data cube.
    overflow : astropy.io.fits.BinTableHDU
        The table of values that overflowed the quantization range.

    See Also
    --------
    pyimcom.compress.i24.I24Cube : Compression class; see for possible keys in `pars`.

    """

    cube = I24Cube(im, pars)
    if scheme == "I24A":
        cube.to_mode("int32")
    elif scheme == "I24B":
        cube.to_mode("uint8")
    else:
        # unrecognized scheme
        return im, None

    return cube.data, cube.overflow


def i24decompress(im, scheme, pars, overflow=None):
    """
    Decompresses an image.

    Parameters
    ----------
    scheme : str
        Compression scheme (right now supports 'I24A', 'I24B').
    pars : dict
        Parameters to pass to compression algorithm.
     overflow: astropy.io.fits.BinTableHDU or None, optional
         Overflow table (y,x,value). Needed if a compressed image is given as input.
    Returns
    -------
    data : np.array
        The de-compressed data cube.

    See Also
    --------
    pyimcom.compress.i24.I24Cube : Compression class; see for possible keys in `pars`.

    """

    cube = I24Cube(im, pars, overflow=overflow)
    if scheme == "I24A" or scheme == "I24B":
        cube.to_mode("float32")
    else:
        # unrecognized scheme
        return im

    return cube.data
