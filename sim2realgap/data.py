import os
import numpy as np
from PIL import Image


def load_video(path, n_frames=None, frames=None):
    """
    Load a video from a directory of .tif files.

    Parameters
    ----------
    path : str
        Path to directory containing .tif frames.
    n_frames : int, optional
        Number of frames to load from the start.
    frames : list, optional
        Specific frame indices to load.

    Returns
    -------
    np.ndarray of shape (n_frames, H, W), float32, normalized to 0-1.
    """
    all_frames = [name for name in os.listdir(path) if name.endswith('.tif')]
    all_frames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    if frames is None:
        frames = list(range(len(all_frames))) if n_frames is None else list(range(n_frames))

    ims = []
    for frame in frames:
        filename = all_frames[frame]
        im = Image.open(f"{path}/{filename}")
        ims.append(np.array(im, dtype=np.float32))

    ims = np.stack(ims, axis=0)
    ims = ims / 255.0
    return ims


def analyze_mask(mask, image):
    """
    Extract per-cell features from a segmentation mask and image.

    Parameters
    ----------
    mask : np.ndarray (H, W)
        Segmentation mask where each unique non-zero value is a cell.
    image : np.ndarray (H, W)
        Corresponding image frame.

    Returns
    -------
    num_cells : int
    areas : list of float
    mean_intensities : list of float
    locs : list of np.ndarray
    """
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]

    areas = []
    mean_intensities = []
    locs = []

    for cell_id in unique_ids:
        cell_pixels = np.where(mask == cell_id)
        area = len(cell_pixels[0]) / (mask.shape[0] * mask.shape[1])
        mean_intensity = float(image[cell_pixels].mean())
        areas.append(area)
        mean_intensities.append(mean_intensity)
        locs.append(np.stack(cell_pixels).mean(axis=1))

    return len(unique_ids), areas, mean_intensities, locs


def get_masked_pixels(image, mask):
    """
    Get pixel values of all cells, excluding background.

    Parameters
    ----------
    image : np.ndarray (H, W)
    mask : np.ndarray (H, W)

    Returns
    -------
    np.ndarray of pixel values inside cells.
    """
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image[mask > 0]