import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.decomposition import PCA
import pandas as pd

def load_row_profiles(directory, name_pattern):
    file_pattern = re.compile(name_pattern)
    row_profiles = []
    filenames = []

    for filename in sorted(os.listdir(directory)):
        if m:=file_pattern.match(filename):
            obs = m.group(1)
            with fits.open(os.path.join(directory, filename)) as hdul:
                image = hdul[0].data
            row_medians = np.median(image, axis=1)
            row_profiles.append(row_medians)
            filenames.append(filename)

    per_image_row_std = np.std(row_profiles, axis=1)
    print(f"Mean per-image row std: {np.mean(per_image_row_std):.4f}")
    print(f"Max per-image row std: {np.max(per_image_row_std):.4f}")

    return np.array(row_profiles), filenames

def analyze_stripe_stability(row_profiles, filenames, output_csv, save_csv=False):
    n_images, n_rows = row_profiles.shape

    # Mean and std across images for each row
    row_mean = np.mean(row_profiles, axis=0)
    row_std = np.std(row_profiles, axis=0)
    frac_std = row_std / np.maximum(np.abs(row_mean), 1e-6)

    # PCA
    pca = PCA()
    pca.fit(row_profiles)
    pc1 = pca.components_[0]
    coeffs = pca.transform(row_profiles)[:, 0]
    explained_var = pca.explained_variance_ratio_[0]
    coeff_std = np.std(coeffs)

    # Pairwise correlation matrix
    corr_matrix = np.corrcoef(row_profiles)
    mean_corrs = [np.mean(np.delete(corr_matrix[i], i)) for i in range(n_images)]

    # Total power of each profile
    profile_power = np.sum((row_profiles - row_profiles.mean(axis=1, keepdims=True))**2, axis=1)

    # CSV output
    if save_csv:
        df = pd.DataFrame({
            'filename': filenames,
            'pc1_coeff': coeffs,
            'mean_corr': mean_corrs,
            'profile_power': profile_power
        })
        df.to_csv(output_csv, index=False)

        print(f"Saved analysis to {output_csv}")
    print(f"Fraction of variance explained by PC1: {explained_var:.3f}")
    print(f"Std of PC1 coefficients: {coeff_std:.3e}")

    return pc1, coeffs, explained_var, row_profiles

def plot_diagnostics(pc1, row_profiles, coeffs, filenames):
    plt.figure(figsize=(10, 4))
    plt.imshow(row_profiles, aspect='auto', cmap='viridis', origin='lower',
               vmin=np.percentile(row_profiles, 10), vmax=np.percentile(row_profiles, 90))
    plt.colorbar(label='Row Median')
    plt.title("Row Median Profiles Across Images")
    plt.xlabel("Row Index")
    plt.ylabel("Image Index")
    plt.tight_layout()
    plt.savefig(f'row_medians_{SCA}.png', bbox_inches='tight')

    plt.figure(figsize=(8, 4))
    plt.plot(pc1)
    plt.title("First Principal Component (Stripe Pattern)")
    plt.xlabel("Row Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f'pc1_{SCA}.png', bbox_inches='tight')

    plt.figure(figsize=(10, 4))
    plt.plot(coeffs, marker='o')
    plt.xticks(ticks=np.arange(len(filenames)), labels=filenames, rotation=90, fontsize=6)
    plt.title("PC1 Coefficients (Stripe Pattern Expression per Image)")
    plt.ylabel("PC1 Coefficient")
    plt.tight_layout()
    plt.savefig(f'pc1_coefs_{SCA}.png', bbox_inches='tight')

# --- Configuration ---
directory = sys.argv[1]  # Set this
SCA = str(sys.argv[2])
name_pattern = 'slope_(\d*)_('+SCA+').fits'
output_csv = "stripe_stability.csv"

# --- Run Analysis ---
row_profiles, filenames = load_row_profiles(directory, name_pattern)
pc1, coeffs, explained_var, row_profiles = analyze_stripe_stability(row_profiles, filenames, output_csv)
plot_diagnostics(pc1, row_profiles, coeffs, filenames)



