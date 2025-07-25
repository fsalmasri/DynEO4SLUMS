import numpy as np
import matplotlib.pyplot as plt

def plot_band_means_dual_axis(original, normalized):
    """
    Plot per-band mean values on two y-axes:
    one for original DN, one for normalized [0–1].
    """
    num_bands = original.shape[0]
    band_ids = np.arange(1, num_bands + 1)

    # Compute means
    original_means = [original[i].mean() for i in range(num_bands)]
    norm_means = [normalized[i].mean() for i in range(num_bands)]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    bar_width = 0.35

    # Plot original means (left axis)
    ax1.bar(band_ids - bar_width/2, original_means,
            width=bar_width, label='Original DN', color='skyblue')
    ax1.set_ylabel("Original Mean (DN)", color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # Right axis: normalized
    ax2 = ax1.twinx()
    ax2.bar(band_ids + bar_width/2, norm_means,
            width=bar_width, label='Normalized', color='orange')
    ax2.set_ylabel("Normalized Mean (0–1)", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title("Mean Values per Band (Original vs. Normalized)")
    plt.xticks(band_ids)
    fig.tight_layout()
    plt.show()
