import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_row_profiles(directory, name_pattern, SCA):
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
    # print(f"Mean per-image row std: {np.mean(per_image_row_std):.4f}")
    # print(f"Max per-image row std: {np.max(per_image_row_std):.4f}")

    return np.array(row_profiles), obsnames

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_row_stability_summary(row_profiles, SCA):
    n_images, n_rows = row_profiles.shape

    # Compute per-row statistics
    mean_profile = np.mean(row_profiles, axis=0)
    std_profile = np.std(row_profiles, axis=0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Top: Heatmap of row median profiles
    im = ax1.imshow(
        row_profiles, aspect='auto', cmap='viridis', origin='lower',
        vmin=np.percentile(row_profiles, 10),
        vmax=np.percentile(row_profiles, 90)
    )
    ax1.set_ylabel("Observation Index")
    ax1.set_title(f"Row Median Profiles Across Images: SCA {SCA}")

    # Use make_axes_locatable to keep colorbar from shifting the axis
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Row Median (DN/fr)')

    # Bottom: Mean ± std profile
    x = np.arange(n_rows)

    ax2.fill_between(
        x, mean_profile - std_profile, mean_profile + std_profile,
        color='yellowgreen', label='±1 Sigma', alpha=0.6
    )
    ax2.fill_between(
        x, mean_profile - 2 * std_profile, mean_profile + 2 * std_profile,
        color='yellowgreen', label='±2 Sigma', alpha=0.2
    )

    ax2.plot(x, mean_profile, color='black', label='Mean', linewidth=1)

    ax2.set_ylim(-0.06, 0.06)
    # ax2.set_yscale('symlog')  # Optional symlog toggle
    ax2.set_ylabel("Median Value (DN/fr)")
    ax2.set_xlabel("Row Index")
    ax2.set_title("Row-wise Mean ± Std")

    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"plots/row_stability_summary_{SCA}.png", bbox_inches='tight')



# --- Configuration ---
directory = sys.argv[1]  # Set this
output_csv = "stripe_stability.csv"

# --- Run Analysis ---
for i in range(18):
    SCA=str(i+1)
    name_pattern = 'slope_(\d*)_(' + SCA + ').fits'
    row_profiles, filenames = load_row_profiles(directory, name_pattern, SCA=SCA)
    plot_row_stability_summary(row_profiles, SCA=SCA)



