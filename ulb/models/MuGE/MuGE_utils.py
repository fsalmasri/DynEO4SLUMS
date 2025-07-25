import matplotlib.pyplot as plt
import torch
import numpy as np
from math import sqrt





# Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
def crop(data1, h, w, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert (h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
    return data

def create_frequency_mask(w, h, radius=0.5):
    """
    Creates a circular mask to extract low frequencies.
    Args:
        w (int): Width of the 2D tensor.
        h (int): Height of the 2D tensor.
        radius (float): The radius of the low-frequency region, relative to image size.

    Returns:
        torch.Tensor: A mask with low frequencies kept (1) and high frequencies (0).
    """

    center = (w // 2, h // 2)

    # Calculate the radius based on the ratio
    radius = radius * min(w, h) / 2

    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    # Calculate the distance from the center
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Create the mask
    mask = distance <= radius

    return mask


def apply_frequency_mask(tensor, mask, invert=False):
    """
    Apply a frequency mask (low or high) to the 2D Fourier transform of a tensor.
    Args:
        tensor (torch.Tensor): The input tensor of shape (b, c, w, h).
        mask (torch.Tensor): The frequency mask of shape (w, h).
        invert (bool): If True, apply the inverse mask (for high frequencies).

    Returns:
        torch.Tensor: The masked Fourier transform.
    """
    b, c, w, h = tensor.shape

    # Compute the 2D Fourier transform
    fft_result = torch.fft.fft2(tensor)

    # Shift zero-frequency components to the center
    fft_result_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))

    # Apply the mask (either low or high frequencies)
    mask = mask.to(tensor.device)  # Ensure mask is on the same device as tensor
    if invert:
        mask = torch.logical_not(mask)  # Invert mask for high frequencies

    # Repeat the mask to match the tensor size (b, c, w, h)
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    mask = mask.repeat(b, c, 1, 1)  # Repeat the mask along batch and channel dimensions

    # Apply mask to the shifted FFT result
    fft_result_shifted = fft_result_shifted * mask

    return fft_result_shifted



def spectrum_noise(img_fft, alpha):
    ratio = 0.5
    batch_size, h, w, c = img_fft.shape
    img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
    img_abs = torch.fft.fftshift(img_abs, dim=(1))  # Shift zero-frequency component to center of spectrum

    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = 0
    img_abs_ = img_abs.clone()
    masks = torch.ones_like(img_abs)
    for i_m in range(alpha.shape[0]):
        masks[i_m] = masks[i_m] * (alpha[i_m])
    masks[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = 1

    print(h_start, h_crop, w)
    plt.imshow(masks[0,:,:,0].data.numpy())
    plt.show()
    print(img_fft.shape, img_abs.shape, masks.shape)
    exit()
    # img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = \
    #                 img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :]*(1-torch.exp(alpha)).view(-1,1,1,1)
    img_abs = img_abs_ * masks
    img_abs = torch.fft.ifftshift(img_abs, dim=(1))  # recover
    img_mix = img_abs * (np.e ** (1j * img_pha))
    return img_mix
