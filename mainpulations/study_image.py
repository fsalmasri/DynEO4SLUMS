import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from norm_utils import (
    linear_norm,
    linear_norm_percentile,
    linear_sigmoid_normalization,
    log_sigmoid_normalization
)



bands_dic = {
    'CoastalBlue': 0,
    'Blue': 1,
    'Green': 2,
    'Red': 3,
    'RedEdge': 4,
    'NIR1': 5,
    'NIR2': 6,
    'NIR3': 7,

}

rgb_indices = [bands_dic[b] for b in ['Red', 'Green', 'Blue']]


croppedimage = np.load('cropped_img.npy')
image = np.load('image.npy')

# reflectance = croppedimage * 0.01
normalized = np.zeros_like(croppedimage, dtype=np.float32)
for i in range(croppedimage.shape[0]):
    band = croppedimage[i].astype(np.float32)
    normalized[i] = linear_sigmoid_normalization(band)

rgb_norm = normalized[rgb_indices].transpose((1, 2, 0))

save_img = Image.fromarray((rgb_norm * 255).astype(np.uint8)).save('linear_sig_norm.gif')

exit()
# plt.figure()
# plt.imshow(rgb_norm)
# plt.show()

import seaborn as sns

plt.figure()
colors = ['red', 'green', 'blue']
for i in range(3):
    band_data = normalized[:, :, i]
    flat = band_data.flatten()
    # plt.hist(flat, bins=256, color=colors[i], alpha=0.3, label=colors[i])
    sns.histplot(data=flat, bins=256, alpha=.5, kde=True)

plt.grid(False)
plt.tight_layout()
plt.show()

exit()

num_bands = image.shape[0]
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs = axs.ravel()
for i in range(num_bands):
    band_data = image[i, :, :]  # (height, width)
    flat = band_data.flatten()  # 1D array of all pixel values
    flat = flat[flat>0]
    axs[i].hist(flat, bins=256, color='gray', alpha=0.7)
    axs[i].set_title(f'Band {i+1}')
    axs[i].set_xlabel('DN value')
    axs[i].set_ylabel('Frequency')

plt.show()

