import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.offsetbox import bbox_artist
from sklearn.decomposition import PCA
import pandas as pd

def load_row_profiles(directory, name_pattern):
    file_pattern = re.compile(name_pattern)
    row_profiles = []
    obsnames = []

    for filename in sorted(os.listdir(directory)):
        if m:=file_pattern.match(filename):
            obs = m.group(1)
            with fits.open(os.path.join(directory, filename)) as hdul:
                image = hdul[0].data
            row_medians = np.median(image, axis=1)
            row_profiles.append(row_medians)
            obsnames.append(obs)

    per_image_row_std = np.std(row_profiles, axis=1)
    print(f"Mean per-image row std: {np.mean(per_image_row_std):.4f}")
    print(f"Max per-image row std: {np.max(per_image_row_std):.4f}")

    return np.array(row_profiles), obsnames

import matplotlib.pyplot as plt

def plot_row_stability_summary(row_profiles):
    n_images, n_rows = row_profiles.shape

    # Compute per-row statistics
    mean_profile = np.mean(row_profiles, axis=0)
    std_profile = np.std(row_profiles, axis=0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Top: Heatmap of row median profiles
    im = ax1.imshow(row_profiles, aspect='auto', cmap='viridis', origin='lower',
                    vmin=np.percentile(row_profiles, 10), vmax=np.percentile(row_profiles, 90))
    ax1.set_ylabel("Image Index")
    ax1.set_title("Row Median Profiles Across Images")
    plt.colorbar(im, ax=ax1, label='Row Median (DN/fr))')

    # Bottom: Mean ± std profile
    x = np.arange(n_rows)
    ax2.errorbar(x, mean_profile, yerr=std_profile, fmt='o', ms=3, elinewidth=1, capsize=2, color='darkblue', ecolor='royalblue')
    ax2.set_ylabel("Median Value")
    ax2.set_xlabel("Row Index")
    ax2.set_yscale('log')
    ax2.set_title("Row-wise Mean ± Std Over Images")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"plots/row_stability_summary_{SCA}.png", bbox_inches='tight')



# --- Configuration ---
directory = sys.argv[1]  # Set this
SCA = str(sys.argv[2])
name_pattern = 'slope_(\d*)_('+SCA+').fits'
output_csv = "stripe_stability.csv"

# --- Run Analysis ---
row_profiles, filenames = load_row_profiles(directory, name_pattern)
plot_row_stability_summary( row_profiles)



