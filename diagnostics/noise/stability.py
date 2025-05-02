import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def process_fits_files(directory, name_pattern):
    """
    Check the stability of the noise in the images by calculating the mean and standard deviation of the row median
    values for a single SCA.
    Parameters
    ----------
    directory
    name_pattern

    Returns
    -------

    """
    file_pattern = re.compile(name_pattern)
    x_range=None
    all_y_vals=[]

    means = []
    std_devs = []

    plt.figure(figsize=(9,9))

    for filename in sorted(os.listdir(directory)):
        if m:=file_pattern.match(filename):
            file_path = os.path.join(directory, filename)
            obs=m.group(1)

            with fits.open(file_path) as hdul:
                noise_data = hdul["PRIMARY"].data

            row_medians = np.median(noise_data, axis=1)
            row_median_mean = np.mean(row_medians)
            row_median_std = np.std(row_medians)

            means.append(row_median_mean)
            std_devs.append(row_median_std)

            # Plot Gaussian using mean and standard deviation
            if x_range is None:
                x_range = np.linspace(-.2, .2, 100)
            y = (1 / (row_median_std * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x_range - row_median_mean) / row_median_std) ** 2)
            all_y_vals.append(y)

            plt.plot(x_range, y, label=f'Obs {obs} (mean={row_median_mean:.5f}, std={row_median_std:.5f})',
                     )

    # Show the plot
    all_y_values = np.array(all_y_vals)
    y_max = np.max(all_y_values, axis=0)
    y_min = np.min(all_y_values, axis=0)
    y_diff = y_max - y_min
    max_difference = np.max(y_diff)

    # Plot the envelope of the curves
    # plt.fill_between(x_range, y_min, y_max, color='gray', alpha=0.3, label='Envelope (max-min)')
    plt.text(0.02, 0.95, f'Max y-difference: {max_difference:.3e}', transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Move legend below the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8)
    plt.xlabel('Median Value')
    plt.ylabel('Probability Density')
    plt.title(f'Gaussian Distribution of Row Medians for Each Observation: SCA {SCA}')
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at the bottom for legend
    plt.savefig(f'./plots/stability_{SCA}.png', bbox_inches='tight')


# Example usage
directory=sys.argv[1]
SCA = str(sys.argv[2])
name_pattern = 'slope_(\d*)_('+SCA+').fits'
if not os.path.exists('./plots'):
    os.makedirs('./plots')

process_fits_files(directory, name_pattern)
