import matplotlib.pyplot as plt
from sahi.slicing import slice_image
from PIL import Image
import numpy as np
import re
from collections import defaultdict
import os

tile_size = 512
overlap_threshold = tile_size

img = np.load('image.npy').transpose((1, 2, 0))

ds = r'C:\Users\fsalm\Desktop\DynEO4SLUMS\nairobi\planetscope\nairobi_psb_sd_20230129_surfrefl_8bands_psscene_analytic_8b_sr_udm2\ds'

os.makedirs(os.path.join(ds, "train"), exist_ok=True)
os.makedirs(os.path.join(ds, "valid"), exist_ok=True)

slice_image_result = slice_image(
    image=img,
    output_file_name='slice',
    # output_dir=ds,
    slice_height=tile_size,
    slice_width=tile_size,
    overlap_height_ratio=0.5,
    overlap_width_ratio=0.5,
)

sliced_images = slice_image_result.sliced_image_list
filenames = slice_image_result.filenames

def extract_coords_from_filename(fname):
    parts = fname.split('_')
    x = int(parts[2])
    y = int(parts[3])
    return x, y


tiles = []
for idx, fname in enumerate(filenames):
    x, y = extract_coords_from_filename(fname)
    tiles.append({'index': idx, 'x': x, 'y': y, 'fname': fname})

import random
# Randomly select 20% of tiles for validation
num_val = int(0.05 * len(tiles))
val_tiles = random.sample(tiles, num_val)

# Step 3: Mark overlapping tiles to exclude from train
val_indices = set(tile['index'] for tile in val_tiles)
excluded_from_train = set(val_indices)  # Start with val tiles themselves
# Build quick lookup for all tiles by index
tile_by_index = {tile['index']: tile for tile in tiles}



for val_tile in val_tiles:
    vx, vy = val_tile['x'], val_tile['y']
    for other_tile in tiles:
        if other_tile['index'] in excluded_from_train:
            continue  # already excluded
        ox, oy = other_tile['x'], other_tile['y']
        if abs(vx - ox) < overlap_threshold and abs(vy - oy) < overlap_threshold:
            excluded_from_train.add(other_tile['index'])

# Step 4: Assign the rest to train
train_indices = [tile['index'] for tile in tiles if tile['index'] not in excluded_from_train]

# Step 5: Collect and save
train_images = [sliced_images[i] for i in train_indices]
train_filenames = [filenames[i] for i in train_indices]

valid_images = [sliced_images[i] for i in val_indices]
valid_filenames = [filenames[i] for i in val_indices]

def test_no_overlap(train_filenames, valid_filenames, tile_size=512):
    """
    Select one file from training filenames, check that no file in validation overlaps with it.
    """
    test_fname = random.choice(train_filenames)
    tx, ty = extract_coords_from_filename(test_fname)

    for val_fname in valid_filenames:
        vx, vy = extract_coords_from_filename(val_fname)
        if abs(tx - vx) < tile_size and abs(ty - vy) < tile_size:
            print("❌ Overlap found!")
            print(f"Train tile: {test_fname}")
            print(f"Validation tile: {val_fname}")
            return False

    print("✅ No overlap found between the selected training tile and validation set.")
    return True

for i in range(10):
    test_no_overlap(train_filenames, valid_filenames)


print(f' original len: {len(filenames)}, training len: {len(train_filenames)}, validation len: {len(valid_filenames)}')

for img, fname in zip(slice_image_result.sliced_image_list, slice_image_result.filenames):

    if fname in train_filenames:
        out_path = os.path.join(ds, "train", fname)
    elif fname in valid_filenames:
        out_path = os.path.join(ds, "valid", fname)
    else:
        continue  # skip tiles not in either set
        print(f'Name not found {fname}')

    img = img.image.transpose((2,0,1))

    # Save based on type
    if isinstance(img, np.ndarray):
        np.save(out_path, img)
    else:  # Assume PIL.Image
        if not fname.endswith(".png"):
            fname = fname.replace(".npy", ".png")
        out_path = out_path.replace(".npy", ".png")
        img.save(out_path)