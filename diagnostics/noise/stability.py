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

    means = []
    std_devs = []

    for filename in os.listdir(directory):
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
            x = np.linspace(row_median_mean - 4 * row_median_std, row_median_mean + 4 * row_median_std, 100)
            y = (1 / (row_median_std * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x - row_median_mean) / row_median_std) ** 2)

            plt.plot(x, y, label=f'Obs {obs} (mean={row_median_mean:.2f}, std={row_median_std:.2f})',
                     )

    # Show the plot
    plt.legend()
    plt.xlabel('Median Value')
    plt.ylabel('Probability Density')
    plt.title(f'Gaussian Distribution of Row Medians for Each Observation: SCA {SCA}')
    plt.savefig(f'./plots/stability_{SCA}.png', bbox_inches='tight')


# Example usage
directory=sys.argv[1]
SCA = str(sys.argv[2])
name_pattern = 'slope_(\d*)_('+SCA+').fits'
if not os.path.exists('./plots'):
    os.makedirs('./plots')

process_fits_files(directory, name_pattern)
