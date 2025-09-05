import numpy as np
from pyimcom.meta.ginterp import MultiInterp


def test_MultiInterp_is_successful():
    """Test for MultiInterp."""

    # This test has nf2 = 6 layers.
    # The first 4 are sine waves, then there is a sum of these, and then there is a circle of point sources.
    samp = 5.0
    rsearch = 4.5
    sigma = samp / np.sqrt(8 * np.log(2))
    n = 425
    x, y = np.meshgrid(np.linspace(0, n - 1, n), np.linspace(0, n - 1, n))
    nf = 4
    nf2 = 6
    u0 = 0.243
    v0 = 0.128
    InArr = np.zeros((nf2, n, n), dtype=np.float32)
    for j in range(nf):
        InArr[j, :, :] = 1.0 + 0.1 * np.cos(2 * np.pi * (u0 * x + v0 * y) / 2.0**j)
    InArr[-2, :, :] = np.sum(InArr[:-2, :, :], axis=0) - 3.6
    for k in range(64):
        xc = 200 + 150 * np.cos(k / 32 * np.pi)
        yc = 170 + 150 * np.sin(k / 32 * np.pi)
        InArr[-1, :, :] += np.exp(-0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / sigma**2)
    InMask = np.zeros((n, n), dtype=bool)
    mat = np.array([[0.52, 0.005], [-0.015, 0.51]])
    sc = 0.5
    nout = 720
    eC = (mat @ mat.T / sc**2 - np.identity(2)) * sigma**2
    C = [eC[0, 0], eC[0, 1], eC[1, 1]]
    pos_offset = [6.0, 3.0]
    OutArr, OutMask, Umax, Smax = MultiInterp(InArr, InMask, (nout, nout), pos_offset, mat, rsearch, samp, C)

    # Now generate the expected output
    TargetArr = np.zeros((nf2, nout, nout))
    xo, yo = np.meshgrid(np.linspace(0, nout - 1, nout), np.linspace(0, nout - 1, nout))
    W = np.exp(-2 * np.pi**2 * (u0**2 * C[0] + 2 * u0 * v0 * C[1] + v0**2 * C[2]))
    for j in range(nf):
        TargetArr[j, :, :] = 1.0 + 0.1 * np.cos(
            2
            * np.pi
            * (
                (mat[0][0] * xo + mat[0][1] * yo + pos_offset[0]) * u0
                + (mat[1][0] * xo + mat[1][1] * yo + pos_offset[1]) * v0
            )
            / 2.0**j
        ) * W ** (0.25**j)
    TargetArr[-2, :, :] = np.sum(TargetArr[:-2, :, :], axis=0) - 3.6
    for k in range(64):
        xc = 200 + 150 * np.cos(k / 32 * np.pi)
        yc = 170 + 150 * np.sin(k / 32 * np.pi)
        v = np.array([xc, yc]) - pos_offset
        tt = np.linalg.solve(mat, v)
        xt = tt[0]
        yt = tt[1]
        TargetArr[-1, :, :] += np.exp(
            -0.5 * ((xo - xt) ** 2 + (yo - yt) ** 2) / (sigma / sc) ** 2
        ) / np.linalg.det(mat / sc)
    diff = np.where(OutMask, 0.0, OutArr - TargetArr).astype(np.float32)
    # print(np.amax(np.abs(diff), axis=(1,2)))
    # fits.PrimaryHDU(diff).writeto('diff.fits')
    # <-- need to import astropy.io.fits if you want to look at this

    # Check that the differences are within expected tolerances for these settings.
    assert np.amax(np.abs(diff)) < 4e-5
    assert np.amax(np.abs(diff[1, :, :])) < 1e-5
    assert np.amax(np.abs(diff[-1, :, :])) < 1e-5
    return
