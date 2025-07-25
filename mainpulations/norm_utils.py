import numpy as np
from scipy.special import expit  # sigmoid


def linear_norm(image):
    norm_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):  # loop over bands
        band = image[i]
        band_min = band.min()
        band_max = band.max()
        print(i, band_min, band_max)

        # Avoid divide-by-zero
        if band_max > band_min:
            norm_image[i] = (band - band_min) / (band_max - band_min)
        else:
            norm_image[i] = 0.0  # or np.nan if you'd prefer

    return norm_image


def linear_norm_percentile(image, lower_percentile=2, upper_percentile=98):
    """
    Normalize a single band to [0, 1] using percentile clipping.

    Parameters:
        band (np.ndarray): 2D array (single band)
        lower_percentile (float): e.g., 2 for 2nd percentile
        upper_percentile (float): e.g., 98 for 98th percentile

    Returns:
        np.ndarray: normalized band in [0, 1]
    """

    norm_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):  # loop over bands
        band = image[i]

        # Compute percentiles
        band_min = np.percentile(band, lower_percentile)
        band_max = np.percentile(band, upper_percentile)

        # Clip outliers
        band_clipped = np.clip(band, band_min, band_max)

        # Normalize to [0, 1]
        norm_image[i] = (band_clipped - band_min) / (band_max - band_min)

    return norm_image


def log_sigmoid_normalization(band, p_low=30, p_high=70):
    # Step 1: Log-transform
    band = np.log1p(band.astype(np.float32))  # log(1 + x)

    # Step 2: Percentiles on log-values
    low = np.percentile(band, p_low)
    high = np.percentile(band, p_high)

    # Step 3: Rescale to center between percentiles using sigmoid
    scale = 10.0 / (high - low)  # scale so sigmoid transitions around mid-range
    norm = expit((band - low) * scale)  # sigmoid(x)

    return norm

def linear_sigmoid_normalization(band, p_low=10, p_high=95):
    band = band.astype(np.float32)

    low = np.percentile(band, p_low)
    high = np.percentile(band, p_high)

    scale = 10.0 / (high - low)
    norm = expit((band - low) * scale)

    return norm