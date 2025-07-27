import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import cv2


class PlanetScope(torch.utils.data.Dataset):
    def __init__(self, opt, train=True, aug=False):

        self.opt = opt
        self.augment = aug
        self.train = train

        if train:
            self.DS_path = os.path.join(opt.DS_path, 'train')
        else:
            self.DS_path = os.path.join(opt.DS_path, 'valid')

        self.imgs_lst = sorted(os.listdir(self.DS_path))


    def __len__(self):
        return len(self.imgs_lst)

    def __getitem__(self, idx):

        scale_factor = 2
        img_name = os.path.basename(self.imgs_lst[idx])
        img = load_planetscope_channels(os.path.join(self.DS_path, self.imgs_lst[idx]))['rgb']
        img, min_norm, max_norm = normalize_tile(img)

        lr = resize_opencv_tensor(img, scale_factor)


        img = torch.from_numpy(img).float()
        lr = torch.from_numpy(lr).float()
        min_norm = torch.from_numpy(min_norm).float()
        max_norm = torch.from_numpy(max_norm).float()


        data = {
            'lr': lr,
            'hr': img,
            'min_norm': min_norm,
            'max_norm': max_norm,
            'fname': img_name
        }
        return data



def resize_opencv_tensor(img, scale_factor):
    """
    Resize an image in (C, H, W) format using OpenCV.
    Returns resized image in (C, H', W') format.
    """
    img_np = img.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
    h, w = img_np.shape[:2]
    resized_np = cv2.resize(img_np, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_LINEAR)
    resized = resized_np.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
    return resized



def normalize_tile(tile):
    """
    Vectorized per-channel normalization for (C, H, W) tiles.
    Falls back to 1–99 percentile if min == max for any channel.
    """

    c, h, w = tile.shape
    tile_flat = tile.reshape(c, -1)

    min_vals = tile_flat.min(axis=1, keepdims=True)  # (C, 1)
    max_vals = tile_flat.max(axis=1, keepdims=True)  # (C, 1)

    same_vals = (max_vals - min_vals) < 1e-6  # flat channels

    # fallback to percentiles where needed
    if np.any(same_vals):
        p1 = np.percentile(tile_flat, 1, axis=1, keepdims=True)
        p99 = np.percentile(tile_flat, 99, axis=1, keepdims=True)

        min_vals = np.where(same_vals, p1, min_vals)
        max_vals = np.where(same_vals, p99, max_vals)

    # Normalize
    tile_norm = (tile_flat - min_vals) / (max_vals - min_vals + 1e-6)
    tile_norm = np.clip(tile_norm, 0, 1)

    return tile_norm.reshape(c, h, w), min_vals.squeeze(), max_vals.squeeze()

def load_planetscope_channels(path):
    """
    Load PlanetScope 8-band image and return channels by type.
    Optionally returns RGB composite.

    Args:
        path (str): path to .npy file
        return_rgb (bool): if True, returns RGB image
        rgb_clip (tuple): min/max values to clip for RGB visualization

    Returns:
        dict: with keys 'visible', 'vegetation', 'all', and optionally 'rgb'
    """

    img = np.load(path).astype(np.float32)  # shape: (H, W, 8)
    # Band groupings based on SuperDove spec
    visible = img[[1, 2, 3, 4, 5]]  # Blue, Green I, Green, Yellow, Red
    vegetation = img[[6, 7]]  # RedEdge, NIR

    # RGB composite (standard RGB = Red, Green, Blue)
    r = img[5]  # Red
    g = img[3]  # Green
    b = img[1]  # Blue

    rgb = np.stack([r, g, b], axis=0)  # (3, H, W)

    return {
        "visible": visible,
        "vegetation": vegetation,
        "rgb": rgb,
        "all": img
    }



