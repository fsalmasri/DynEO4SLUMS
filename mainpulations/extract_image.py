import os.path

import rasterio
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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

# Path to your TIFF file

ds = r'C:\Users\fsalm\Desktop\DynEO4SLUMS\nairobi\planetscope\nairobi_psb_sd_20230129_surfrefl_8bands_psscene_analytic_8b_sr_udm2'
tif_path = os.path.join(ds, "composite.tif")

# Open and read the image
with rasterio.open(tif_path) as src:
    image = src.read()  # shape: (bands, height, width)
    profile = src.profile  # contains metadata


rgb_img = image[rgb_indices].transpose((1, 2, 0))
print(f'full image shape: {image.shape}')
print(f'rgb image shpe: {rgb_img.shape}')
cropped_img = image[:, 4000:5000,2000:3000]

np.save('cropped_img.npy', cropped_img)
np.save('image.npy', image)

