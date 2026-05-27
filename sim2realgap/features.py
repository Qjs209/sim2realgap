import numpy as np
from .data import analyze_mask, get_masked_pixels


def collect_features(images, masks, frames):
    """
    Collect cell areas and mean intensities across frames.

    Parameters
    ----------
    images : np.ndarray (n_frames, H, W)
    masks : np.ndarray (n_frames, H, W)
    frames : list of int

    Returns
    -------
    areas : np.ndarray
    intensities : np.ndarray
    """
    all_areas = []
    all_intensities = []

    for f in frames:
        _, areas, intensities, _ = analyze_mask(masks[f], images[f])
        all_areas.extend(areas)
        all_intensities.extend(intensities)

    return np.array(all_areas), np.array(all_intensities)


def collect_pixels(images, masks, frames):
    """
    Collect all cell pixel values across frames.

    Parameters
    ----------
    images : np.ndarray (n_frames, H, W)
    masks : np.ndarray (n_frames, H, W)
    frames : list of int

    Returns
    -------
    np.ndarray of pixel values.
    """
    return np.concatenate([get_masked_pixels(images[f], masks[f]) for f in frames])


def collect_full_pixels(images, frames):
    """
    Collect all pixel values across frames with no masking.

    Parameters
    ----------
    images : np.ndarray (n_frames, H, W)
    frames : list of int

    Returns
    -------
    np.ndarray of pixel values.
    """
    raw = np.concatenate([np.array(images[f]).flatten() for f in frames])
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)


def collect_cell_std(images, masks, frames):
    """
    Collect per-cell pixel standard deviation (texture) across frames.

    Parameters
    ----------
    images : np.ndarray (n_frames, H, W)
    masks : np.ndarray (n_frames, H, W)
    frames : list of int

    Returns
    -------
    np.ndarray of per-cell std values.
    """
    stds = []
    for f in frames:
        image = np.array(images[f], dtype=float)
        mask  = np.array(masks[f])
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids != 0]
        for cell_id in unique_ids:
            cell_pixels = image[mask == cell_id]
            stds.append(cell_pixels.std())
    return np.array(stds)