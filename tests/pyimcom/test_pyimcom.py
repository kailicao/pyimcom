"""
Small-scale test of PyIMCOM, designed for fast test of core functionality.

This does 2 blocks.
"""

import contextlib
import os

import asdf
import gwcs
import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy import wcs
from astropy.io import fits
from astropy.modeling import models
from astropy.table import Table
from gwcs import coordinate_frames as cf
from pyimcom.coadd import Block
from pyimcom.config import Config
from pyimcom.psfutil import OutPSF
from pyimcom.routine import gridD5512C
from pyimcom.truthcats import gen_truthcats_from_cfg
from scipy.signal import convolve

# constants
degree = np.pi / 180.0
nside = 4088

# center position
cra = 60.0504 * degree
cdec = -3.8 * degree

# width of PSF in output pixels, area scaling
sig = 0.9265328730414752 * 0.11 / 0.04
sc = (0.04 / 0.11) ** 2

# star position
sra = 60.0508 * degree
sdec = -3.8005 * degree

# format for the configuration file.
# $TMPDIR will get replaced with the temporary directory.
myCfg_format = """
{
    "OBSFILE": "$TMPDIR/obs.fits",
    "INDATA": [
        "$TMPDIR/in",
        "L2_2506"
    ],
    "CTR": [
	60.0504,
	-3.8
    ],
    "LONPOLE": 240.0,
    "OUTSIZE": [
        4,
	25,
	0.04
    ],
    "BLOCK": 2,
    "FILTER": 1,
    "LAKERNEL": "Cholesky",
    "KAPPAC": [
         5e-4
    ],
    "INPSF": [
	"$TMPDIR/psf",
        "L2_2506",
        6
    ],
    "EXTRAINPUT": [
        "gsstar14"
    ],
    "PADSIDES": "all",
    "OUTMAPS": "USTN",
    "OUT": "$TMPDIR/out/testout_F",
    "INPAD": 0.8,
    "NPIXPSF": 42,
    "FADE": 1,
    "PAD": 0,
    "NOUT": 1,
    "OUTPSF": "GAUSSIAN",
    "EXTRASMOOTH": 0.9265328730414752,
    "INLAYERCACHE": "$TMPDIR/cache/in"
}
"""

# CD and CRPIX values for linear approximations to the SCA WCSs.
wcsdata = np.array(
    [
        [
            3.08219178082192e-05,
            1.22309197651664e-07,
            0,
            -3.02103718199609e-05,
            9.31141597190574e-10,
            -254.695456590193,
            819.255060728745,
        ],
        [
            3.04549902152642e-05,
            1.22309197651664e-07,
            -1.22309197651663e-07,
            -2.97211350293542e-05,
            9.05141916965698e-10,
            -302.076620500445,
            5721.07850461111,
        ],
        [
            3.02103718199609e-05,
            2.44618395303327e-07,
            -1.22309197651663e-07,
            -2.88649706457926e-05,
            8.71991576701989e-10,
            -340.491336421342,
            10368.6800480357,
        ],
        [
            3.05772994129159e-05,
            1.22309197651664e-07,
            -2.44618395303327e-07,
            -3.02103718199609e-05,
            9.23721665434799e-10,
            -4684.7680248753,
            -19.9937811750988,
        ],
        [
            3.04549902152642e-05,
            3.6692759295499e-07,
            -3.66927592954991e-07,
            -2.97211350293542e-05,
            9.05022240647057e-10,
            -4754.73767727858,
            4920.56054745611,
        ],
        [
            3.02103718199609e-05,
            4.89236790606654e-07,
            -3.66927592954991e-07,
            -2.91095890410959e-05,
            8.79231993979803e-10,
            -4894.76339878177,
            9448.63987477456,
        ],
        [
            3.04549902152642e-05,
            2.44618395303327e-07,
            -3.66927592954991e-07,
            -3.03326810176125e-05,
            9.23691746355138e-10,
            -9119.77676286723,
            -2073.79302302983,
        ],
        [
            3.02103718199609e-05,
            4.89236790606654e-07,
            -3.66927592954991e-07,
            -2.98434442270059e-05,
            9.0140203200815e-10,
            -9255.53159851301,
            2786.07620817844,
        ],
        [
            2.99657534246575e-05,
            6.11545988258318e-07,
            -6.11545988258311e-07,
            -2.92318982387476e-05,
            8.75581866261235e-10,
            -9476.57488467452,
            7313.76934905176,
        ],
        [
            3.06996086105675e-05,
            -1.22309197651664e-07,
            1.22309197651664e-07,
            -3.02103718199609e-05,
            9.27431631312686e-10,
            4156.44544809343,
            827.807471449771,
        ],
        [
            3.04549902152642e-05,
            -2.44618395303327e-07,
            1.22309197651664e-07,
            -2.95988258317025e-05,
            9.0140203200815e-10,
            4207.94795539033,
            5735.52044609665,
        ],
        [
            3.02103718199609e-05,
            -2.44618395303327e-07,
            1.22309197651664e-07,
            -2.88649706457926e-05,
            8.71991576701989e-10,
            4262.97958483445,
            10367.9787270544,
        ],
        [
            3.05772994129158e-05,
            -2.44618395303327e-07,
            2.44618395303327e-07,
            -3.00880626223092e-05,
            9.1995186139759e-10,
            8568.20762326005,
            -30.0470924938209,
        ],
        [
            3.03326810176125e-05,
            -3.6692759295499e-07,
            2.44618395303327e-07,
            -2.97211350293542e-05,
            9.0143195108781e-10,
            8671.99004281589,
            4891.17687278038,
        ],
        [
            3.02103718199609e-05,
            -3.66927592954991e-07,
            3.66927592954991e-07,
            -2.89872798434442e-05,
            8.75581866261235e-10,
            8754.5221937468,
            9476.99395181958,
        ],
        [
            3.03326810176125e-05,
            -2.44618395303327e-07,
            2.44618395303327e-07,
            -3.03326810176125e-05,
            9.2001169955691e-10,
            13087.5824390244,
            -2119.77756097561,
        ],
        [
            3.03326810176125e-05,
            -3.66927592954991e-07,
            4.89236790606654e-07,
            -2.99657534246575e-05,
            9.08762125604605e-10,
            13130.6172384276,
            2825.69171001514,
        ],
        [
            3.00880626223092e-05,
            -6.11545988258318e-07,
            4.89236790606654e-07,
            -2.92318982387476e-05,
            8.79231993979802e-10,
            13350.5118589853,
            7261.98346207507,
        ],
    ]
)


def make_simple_wcs(ra, dec, pa, sca):
    """
    Makes a simple approximate WCS for an SCA.

    Parameters
    ----------
    ra : float
        RA of the WFI center in degrees.
    dec : float
        Dec of the WFI center in degrees.
    pa : float
        PA of the WFI center in degrees.
    sca : int
        SCA number, in 1 .. 18.

    Returns
    -------
    astropy.wcs.WCS
        Simple WCS approximation.

    """

    outwcs = wcs.WCS(naxis=2)
    outwcs.wcs.crpix = [wcsdata[sca - 1, -2], wcsdata[sca - 1, -1]]
    outwcs.wcs.cd = wcsdata[sca - 1, :4].reshape((2, 2))
    outwcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
    outwcs.wcs.crval = [ra, dec]
    outwcs.wcs.lonpole = pa - 180.0 if pa >= 180.0 else pa + 180.0

    return outwcs


def run_setup(temp_dir):
    """
    Generates sample input files for a pyimcom run.

    Parameters
    ----------
    temp_dir : str
        Directory in which to run the test.

    Returns
    -------
    None

    """

    # first, get the configuration file.
    with open(temp_dir + "/cfg.txt", "w") as f:
        f.write(myCfg_format.replace("$TMPDIR", temp_dir))

    # now make the observation file
    obs = []
    for j in range(14):
        jj = 10 - abs(j)
        date = 61541 + 0.01 * jj
        exptime = 139.8
        ra = 60.0 + 0.01 * jj
        dec = -4.0 + 0.1 * jj
        if j > 10:
            ra -= 0.03
        pa = 20.0
        filter = "F184" if j < 12 else "H158"
        obs.append((date, exptime, ra, dec, pa, filter))
    data = np.rec.array(
        obs, formats="float64,float64,float64,float64,float64,S4", names="date,exptime,ra,dec,pa,filter"
    )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(data=Table(data))]).writeto(
        temp_dir + "/obs.fits", overwrite=True
    )

    # now make the PSFs
    with contextlib.suppress(FileNotFoundError):
        os.mkdir(temp_dir + "/psf")
    ov = 6  # oversampling factor needs to be even here
    psf = []
    for i in range(len(obs)):
        psf.append(OutPSF.psf_cplx_airy(ov * 20, ov * 1.326, sigma=ov * 0.3, features=i % 8))
        psf_cube = np.zeros((4,) + np.shape(psf[i]), dtype=np.float32)
        psf_cube[0, :, :] = psf[i]
        imfits = [fits.PrimaryHDU()]
        for _ in range(18):
            imfits.append(fits.ImageHDU(psf_cube))
        fits.HDUList(imfits).writeto(temp_dir + f"/psf/psf_polyfit_{i:d}.fits", overwrite=True)
    ns_psf = np.shape(psf[-1])[0]
    ctr_psf = (ns_psf - 1) / 2.0

    # tophat kernel
    tk = np.ones(ov + 1)
    # 'wiggling' edges, see Numerical Recipes
    tk[0] -= 5.0 / 8.0
    tk[-1] -= 5.0 / 8.0
    tk[1] += 1.0 / 6.0
    tk[-2] += 1.0 / 6.0
    tk[2] -= 1.0 / 24.0
    tk[-3] -= 1.0 / 24.0

    # draw the images
    with contextlib.suppress(FileNotFoundError):
        os.mkdir(temp_dir + "/in")
    olog = ""
    for iobs in range(len(obs)):
        filt = data["filter"][iobs].decode("ascii")
        print(filt)
        if filt == "F184":
            for sca in range(1, 19):
                this_wcs = make_simple_wcs(data["ra"][iobs], data["dec"][iobs], data["pa"][iobs], sca)
                rapos, decpos = this_wcs.pixel_to_world_values(2043.5, 2043.5)
                olog += f"{iobs}, {sca}, {rapos}, {decpos}\n"

                mu = np.sin(cdec) * np.sin(decpos * degree) + np.cos(cdec) * np.cos(decpos * degree) * np.cos(
                    rapos * degree - cra
                )
                if mu > np.cos(0.08 * degree):
                    # need to draw this SCA
                    xstar, ystar = this_wcs.world_to_pixel_values(sra / degree, sdec / degree)
                    im = np.zeros((1, nside**2))
                    psfc = convolve(psf[iobs], np.outer(tk, tk), mode="same", method="direct")
                    gridD5512C(
                        psfc,
                        (ov * (np.linspace(0, nside - 1, nside) - xstar) + ctr_psf).reshape((1, nside)),
                        (ov * (np.linspace(0, nside - 1, nside) - ystar) + ctr_psf).reshape((1, nside)),
                        im,
                    )
                    mode = np.argmax(im)
                    mode_y = mode // nside
                    mode_x = mode % nside
                    im = im.reshape((nside, nside)).astype(np.float32)
                    print(np.sum(im))
                    print(
                        "**", iobs, sca, rapos, decpos, np.arccos(mu) / degree, xstar, ystar, mode_x, mode_y
                    )

                    # some tests on the images
                    assert np.sum(im) < 1.05
                    if np.sum(im) > 0.5:
                        assert np.hypot(xstar - mode_x, ystar - mode_y) < 1.0

                    # pipeline version of the WCS
                    distortion = models.AffineTransformation2D(this_wcs.wcs.cd, translation=[0, 0])
                    distortion.inverse = models.AffineTransformation2D(
                        np.linalg.inv(this_wcs.wcs.cd), translation=[0, 0]
                    )
                    celestial_rotation = models.RotateNative2Celestial(
                        this_wcs.wcs.crval[0], this_wcs.wcs.crval[1], this_wcs.wcs.lonpole
                    )
                    det2sky = (
                        (
                            models.Shift(-(this_wcs.wcs.crpix[0] - 1))
                            & models.Shift(-(this_wcs.wcs.crpix[1] - 1))
                        )
                        | distortion
                        | models.Pix2Sky_ARC()
                        | celestial_rotation
                    )
                    det2sky.name = "please_is_someone_actually_reading_this"
                    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
                    sky_frame = cf.CelestialFrame(
                        reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg)
                    )
                    sca_gwcs = gwcs.WCS([(detector_frame, det2sky), (sky_frame, None)])

                    # Test for the gwcs.
                    #
                    # This part isn't actually testing pyimcom yet, but if there's a problem
                    # with the test setup, we will have trouble understanding failures later.
                    xs1, ys1 = sca_gwcs.invert(sra / degree, sdec / degree)
                    print(xs1, xstar, ys1, ystar)
                    assert np.hypot(xs1 - xstar, ys1 - ystar) < 1e-3

                    # write to file. these are minimal fields that are needed.
                    asdf.AsdfFile({"roman": {"data": im, "meta": {"wcs": sca_gwcs}}}).write_to(
                        temp_dir + f"/in/sim_L2_{filt:s}_{iobs:d}_{sca:d}.asdf"
                    )

                    # Also can write a FITS version to make sure we can ...
                    # hope this is useful to look at if something goes wrong
                    # fits.PrimaryHDU(im, header=this_wcs.to_header()).writeto(
                    #     temp_dir + f"/in/sim_L2_{filt:s}_{iobs:d}_{sca:d}_asfits.fits", overwrite=True
                    # )

                    # now make the masks
                    mask = np.zeros((nside, nside), dtype=np.uint8)
                    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(mask, name="MASK")]).writeto(
                        temp_dir + f"/in/sim_L2_{filt:s}_{iobs:d}_{sca:d}_mask.fits", overwrite=True
                    )

            # corners -- for testing
            sca = 18
            olog += "-- lower left--\n"
            rapos, decpos = this_wcs.pixel_to_world_values(0.0, 0.0)
            olog += f"{iobs}, {sca}, {rapos}, {decpos}\n"
            olog += "-- lower right--\n"
            rapos, decpos = this_wcs.pixel_to_world_values(4087.0, 0.0)
            olog += f"{iobs}, {sca}, {rapos}, {decpos}\n"
            if iobs == 2:
                assert np.hypot(rapos - 59.89309302318237, decpos + 2.9109906089005753) < 0.01
            olog += "-- top left--\n"
            rapos, decpos = this_wcs.pixel_to_world_values(0.0, 4087.0)
            olog += f"{iobs}, {sca}, {rapos}, {decpos}\n"
            if iobs == 2:
                assert np.hypot(rapos - 59.733417024909365, decpos + 2.982181679089024) < 0.01

    with open(temp_dir + "/wcslog.txt", "w") as f:
        f.write(olog)

    # now make the cache directory
    cachedir = temp_dir + "/cache"
    with contextlib.suppress(FileNotFoundError):
        os.mkdir(cachedir)
    # clear old cache if it is there
    files = os.listdir(cachedir)
    for file in files:
        if file[-5:] == ".fits":
            print("removing", os.path.join(cachedir, file))
            os.remove(os.path.join(cachedir, file))

    # ... and the output directory
    with contextlib.suppress(FileNotFoundError):
        os.mkdir(temp_dir + "/out")


def study_outputs(temp_dir):
    """
    Examine PyIMCOM outputs.

    Parameters
    ----------
    temp_dir : str
        Directory in which to run the test.

    Returns
    -------
    None

    """

    ## Science star portion ##

    with fits.open(temp_dir + "/out/testout_F_00_01.fits") as fblock:
        # position of "science" star
        w = wcs.WCS(fblock[0].header)
        # there are 2 extra axes if you pull the WCS this way
        posout = w.wcs_world2pix(np.array([[sra / degree, sdec / degree, 0, 0]]), 0).ravel()
        xs = posout[0]
        ys = posout[1]
        print(f"science layer star at ({xs},{ys})")

        # output block size
        (ny, nx) = np.shape(fblock[0].data)[-2:]
        x, y = np.meshgrid(np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny))

        # predicted & data images
        p = np.exp(-0.5 * ((x - xs) ** 2 + (y - ys) ** 2) / sig**2) / (2 * np.pi * sig**2 * sc)
        d = fblock[0].data[0, 0, :, :]
        SL1 = np.sum(p * d) / np.sum(p**2)
        VAR = np.sum((d - SL1 * p) ** 2) / np.sum(p**2)
        print("**", SL1, VAR)
        print(np.sum(p))

        assert np.abs(SL1 - 1) < 5e-4
        assert VAR < 1e-5

    ## Injected star portion ##

    with fits.open(temp_dir + "/out/testout_F_TruthCat.fits") as f_inj:
        print(f_inj.info())
        # get the first star in the table --- in this case, it's the only one
        print(f_inj["TRUTH14"].data[0])
        ibx = f_inj["TRUTH14"].data["ibx"][0]
        iby = f_inj["TRUTH14"].data["iby"][0]
        xs = f_inj["TRUTH14"].data["x"][0]
        ys = f_inj["TRUTH14"].data["y"][0]
        print(ibx, iby, xs, ys)

    with fits.open(temp_dir + f"/out/testout_F_{ibx:02d}_{iby:02d}.fits") as fblock:
        # output block size
        (ny, nx) = np.shape(fblock[0].data)[-2:]
        x, y = np.meshgrid(np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny))

        # predicted & data images
        p = np.exp(-0.5 * ((x - xs) ** 2 + (y - ys) ** 2) / sig**2) / (2 * np.pi * sig**2 * sc)
        d = fblock[0].data[0, 1, :, :]
        SL1 = np.sum(p * d) / np.sum(p**2)
        VAR = np.sum((d - SL1 * p) ** 2) / np.sum(p**2)
        print("**", SL1, VAR)
        print(np.sum(p))

        assert np.abs(SL1 - 1) < 5e-4
        assert VAR < 1e-5


def test_PyIMCOM_run1(tmp_path):
    """
    Test function for a small pyimcom run.

    Parameters
    ----------
    tmp_path : str or pathlib.Path
        Directory in which to run the test.

    Returns
    -------
    None

    """

    temp_dir = str(tmp_path)
    print("using", temp_dir)

    # put together the input files
    run_setup(temp_dir)

    # this part runs all 4 blocks ... they're pretty small!
    cfg = Config(temp_dir + "/cfg.txt")
    print(cfg.to_file(None))
    # This has 4 blocks, but we only run 2 here to speed things up.
    for iblk in range(2):
        Block(cfg=cfg, this_sub=iblk)
    gen_truthcats_from_cfg(cfg)

    # now see if the outputs are right
    study_outputs(temp_dir)
