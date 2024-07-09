# usage: python correlation.py <band> <instem> <outstem>
# this code computes correlations of shapes etc. of star and galaxy catalogs from starcube2.py

import numpy as np
import fitsio as fio
import os
import treecorr
import sys

band = sys.argv[1]
in1 = sys.argv[2]
outstem = sys.argv[3]

if os.path.exists(in1 + 'LNCat_' + band + '_sample.fits'):
    d = fio.read(in1 + 'LNCat' + band + '_sample.fits')
else:
    d_type = [('ra', 'f8'), ('dec', 'f8'), ('bind_x', 'f8'), ('bind_y', 'f8'), ('x_out', 'f8'), ('y_out', 'f8'),
              ('xI_out', 'f8'), ('yI_out', 'f8'), ('dx', 'f8'), ('dy', 'f8'),
              ('amp_hsm_S', 'f8'), ('dx_hsm_S', 'f8'), ('dy_hsm_S', 'f8'), ('sig_hsm_S', 'f8'), ('g1_hsm_S', 'f8'),
              ('g2_hsm_S', 'f8'), ('M40-M04_S', 'f8'), ('M31+M13_S', 'f8'), ('Mxx-Myy_fs_S', 'f8'),
              ('Mxy+Myx_fs_S', 'f8'), ('amp_hsm_G', 'f8'), ('dx_hsm_G', 'f8'), ('dy_hsm_G', 'f8'), ('sig_hsm_G', 'f8'),
              ('g1_hsm_G', 'f8'), ('g2_hsm_G', 'f8'), ('M40-M04_G', 'f8'), ('M31+M13_G', 'f8'), ('Mxx-Myy_fs_G', 'f8'),
              ('Mxy+Myx_fs_G', 'f8'), ('mean_fid', 'f8'), ('coverage', 'f8'), ('g1_noise_S', 'f8'),
              ('g2_noise_S', 'f8'), ('g1_noise_G', 'f8'), ('g2_noise_G', 'f8'), ('truth_r', 'f8'), ('truth_g1', 'f8'),
              ('truth_g2', 'f8')]
    d = np.loadtxt(in1 + 'LNAllCat_' + band + '.txt', dtype=d_type,
                   usecols=np.arange(0, 39))
    fio.write(in1 + 'LNCat' + band + '_sample.fits', d)
print('# number of objects', len(d))


def _find_psi(ra, dec, ra_ctr, dec_ctr):
    ra = np.radians(ra)
    dec = np.radians(dec)
    ra_ctr = np.radians(ra_ctr)
    dec_ctr = np.radians(dec_ctr)

    zeta = np.arctan(np.cos(dec) * np.sin(ra - ra_ctr) / (
                -np.sin(dec) * np.cos(dec_ctr) + np.sin(dec_ctr) * np.cos(dec) * np.cos(ra - ra_ctr)))
    eta = np.arctan(np.cos(dec_ctr) * np.sin(ra_ctr - ra) / (
                -np.sin(dec_ctr) * np.cos(dec) + np.sin(dec) * np.cos(dec_ctr) * np.cos(ra - ra_ctr)))

    psi = eta - zeta
    psi_r = psi - np.pi * np.round(psi / np.pi)
    return psi_r


def _compute_treecorr_angle(g1, g2, psi):
    e1 = np.cos(2 * psi) * g1 + np.sin(2 * psi) * g2
    e2 = -np.sin(2 * psi) * g1 + np.cos(2 * psi) * g2

    return e1, e2


def _compute_GG_corr(ra, dec, g1, g2, out_f, xy=False):
    bin_config = dict(
        sep_units='arcmin',
        bin_slop=0.1,

        min_sep=0.5,
        max_sep=80,
        nbins=15,

        var_method='jackknife'
    )

    # shear-shear

    if os.path.exists(in1 + 'LNAllCat_' + band + '_patchcat.txt'):
        patchcat = in1 + 'LNAllCat_' + band + '_patchcat.txt'
    else:
        patchcat_make = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, npatch=15)
        patchcat_make.write_patch_centers(
            in1 + 'LNAllCat_' + band + '_patchcat.txt')
        patchcat = in1 + 'LNAllCat_' + band + '_patchcat.txt'

    gg = treecorr.GGCorrelation(bin_config)
    if xy:
        cat = treecorr.Catalog(x=xy[0], y=xy[1], x_units='arcsec', y_units='arcsec', g1=g1, g2=g2, patch_centers=f_pc)
    else:
        cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, patch_centers=patchcat)

    gg.process(cat)
    gg.write(out_f)


def _compute_NG_corr(ra, dec, g1, g2, out_f):
    bin_config = dict(
        sep_units='arcmin',
        bin_slop=0.1,

        min_sep=0.5,
        max_sep=80,
        nbins=15,

        var_method='jackknife'
    )

    # count-shear
    if os.path.exists(in1 + 'LNAllCat_' + band + '_patchcat.txt'):
        patchcat = in1 + 'LNAllCat_' + band + '_patchcat.txt'
    else:
        patchcat_make = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, npatch=15)
        patchcat_make.write_patch_centers(
            in1 + 'LNAllCat_' + band + '_patchcat.txt')
        patchcat = in1 + 'LNAllCat_' + band + '_patchcat.txt'

    ng = treecorr.NGCorrelation(bin_config)
    cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', patch_centers=patchcat)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, patch_centers=patchcat)
    ng.process(cat1, cat2)
    ng.write(out_f)


def _compute_NK_corr(ra, dec, kappa, out_f):
    bin_config = dict(
        sep_units='arcmin',
        bin_slop=0.1,

        min_sep=0.5,
        max_sep=80,
        nbins=15,

        var_method='jackknife'
    )

    # count-kappa
    if os.path.exists(in1 + 'LNAllCat_' + band + '_patchcat.txt'):
        patchcat = in1 + 'LNAllCat_' + band + '_patchcat.txt'
    else:
        patchcat_make = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, npatch=15)
        patchcat_make.write_patch_centers(
            in1 + 'LNAllCat_' + band + '_patchcat.txt')
        patchcat = in1 + 'LNAllCat_' + band + '_patchcat.txt'

    nk = treecorr.NKCorrelation(bin_config)
    cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', patch_centers=patchcat)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', k=kappa, patch_centers=patchcat)
    nk.process(cat1, cat2)
    nk.write(out_f)


sph_correction = False
if sph_correction:
    ra_ctr = 53.000000
    dec_ctr = -40.000000
    psi = _find_psi(d['ra'], d['dec'], ra_ctr, dec_ctr)
    g1, g2 = _compute_treecorr_angle(d['g1_hsm_S'], d['g2_hsm_S'], psi)

# 2PCF: STARS
_compute_GG_corr(d['ra'], d['dec'], d['g1_hsm_S'], d['g2_hsm_S'],
                 outstem+'star_' + band + '_shear-shear_sample.fits')
_compute_NG_corr(d['ra'], d['dec'], d['g1_hsm_S'], d['g2_hsm_S'],
                 outstem+'star_' + band + '_sky-shear_sample.fits')

for fid_cut in [40, 45, 50]:
    print('# fidelity cut: ', fid_cut)
    d = d[d['mean_fid'] > fid_cut]

    _compute_GG_corr(d['ra'], d['dec'], d['g1_hsm_S'], d['g2_hsm_S'],
                     outstem+'star_' + band + '_shear-shear_sample_fid' + str(
                         fid_cut) + '.fits')  # , xy=[d['x_out'], d['y_out']])
    _compute_NG_corr(d['ra'], d['dec'], d['g1_hsm_S'], d['g2_hsm_S'],
                     outstem+'star_' + band + '_sky-shear_sample_fid' + str(
                         fid_cut) + '.fits')
    _compute_NK_corr(d['ra'], d['dec'], d['sig_hsm_S'],
                     outstem+'star_' + band + '_sky-sigma_sample_fid' + str(
                         fid_cut) + '.fits')

# Noise Bias: STARS
mean_noise_g1 = np.mean(d['g1_noise_S'])
mean_noise_g2 = np.mean(d['g2_noise_S'])
err_noise_g1 = np.std(d['g1_noise_S']) / np.sqrt(len(d['g1_noise_S']))
err_noise_g2 = np.std(d['g2_noise_S']) / np.sqrt(len(d['g2_noise_S']))
print('# Lab noise bias g1: ', '{:.7f}'.format(mean_noise_g1), '+/-', '{:.7f}'.format(err_noise_g1))
print('# Lab noise bias g2: ', '{:.7f}'.format(mean_noise_g2), '+/-', '{:.7f}'.format(err_noise_g2))

_compute_GG_corr(d['ra'], d['dec'], d['g1_noise_S'], d['g2_noise_S'],
                 outstem+'star_' + band + '_noise-noise_sample_labnoise.fits')
_compute_NK_corr(d['ra'], d['dec'], d['g1_noise_S'],
                 outstem+'star_' + band + '_sky-noise_sample_g1_labnoise.fits')
_compute_NK_corr(d['ra'], d['dec'], d['g2_noise_S'],
                 outstem+'star_' + band + '_sky-noise_sample_g2_labnoise.fits')


if sph_correction:
    ra_ctr = 53.000000
    dec_ctr = -40.000000
    psi = _find_psi(d['ra'], d['dec'], ra_ctr, dec_ctr)
    g1, g2 = _compute_treecorr_angle(d['g1_hsm_G'], d['g2_hsm_G'], psi)

# 2PCF: GALAXIES
_compute_GG_corr(d['ra'], d['dec'], d['g1_hsm_G'], d['g2_hsm_G'],
                 outstem+'gal_' + band + '_shear-shear_sample.fits')
_compute_NG_corr(d['ra'], d['dec'], d['g1_hsm_G'], d['g2_hsm_G'],
                 outstem+'gal_' + band + '_sky-shear_sample.fits')

for fid_cut in [40, 45, 50]:
    print('# fidelity cut: ', fid_cut)
    d = d[d['mean_fid'] > fid_cut]

    _compute_GG_corr(d['ra'], d['dec'], d['g1_hsm_G'], d['g2_hsm_G'],
                     outstem+'gal_' + band + '_shear-shear_sample_fid' + str(
                         fid_cut) + '.fits')  # , xy=[d['x_out'], d['y_out']])
    _compute_NG_corr(d['ra'], d['dec'], d['g1_hsm_G'], d['g2_hsm_G'],
                     outstem+'gal_' + band + '_sky-shear_sample_fid' + str(
                         fid_cut) + '.fits')
    _compute_NK_corr(d['ra'], d['dec'], d['sig_hsm_G'],
                     outstem+'gal_' + band + '_sky-sigma_sample_fid' + str(
                         fid_cut) + '.fits')

# Noise Bias: GALAXIES
mean_noise_g1 = np.mean(d['g1_noise_G'])
mean_noise_g2 = np.mean(d['g2_noise_G'])
err_noise_g1 = np.std(d['g1_noise_G']) / np.sqrt(len(d['g1_noise_G']))
err_noise_g2 = np.std(d['g2_noise_G']) / np.sqrt(len(d['g2_noise_G']))
print('# Lab noise bias g1: ', '{:.7f}'.format(mean_noise_g1), '+/-', '{:.7f}'.format(err_noise_g1))
print('# Lab noise bias g2: ', '{:.7f}'.format(mean_noise_g2), '+/-', '{:.7f}'.format(err_noise_g2))

_compute_GG_corr(d['ra'], d['dec'], d['g1_noise_G'], d['g2_noise_G'],
                 outstem+'gal_' + band + '_noise-noise_sample_labnoise.fits')
_compute_GG_corr(d['ra'], d['dec'], d['g1_noise_G']-d['truth_g1'], d['g2_noise_G']-d['truth_g2'],
                 outstem+'gal_' + band + '_noise-noise_difference_labnoise.fits')
_compute_NK_corr(d['ra'], d['dec'], d['g1_noise_G'],
                 outstem+'gal_' + band + '_sky-noise_sample_g1_labnoise.fits')
_compute_NK_corr(d['ra'], d['dec'], d['g2_noise_G'],
                 outstem+'gal_' + band + '_sky-noise_sample_g2_labnoise.fits')
