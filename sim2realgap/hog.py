import numpy as np
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize as sk_resize

HOG_SIZE = (128, 128)


def collect_hog_features(images, frames, hog_size=HOG_SIZE):
    """
    Collect flattened HOG feature vectors across frames.

    Parameters
    ----------
    images : np.ndarray (n_frames, H, W)
    frames : list of int
    hog_size : tuple

    Returns
    -------
    np.ndarray of flattened HOG features.
    """
    all_hog = []
    for f in frames:
        image = np.array(images[f], dtype=float)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = sk_resize(image, hog_size)
        fd = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
        all_hog.append(fd)
    return np.concatenate(all_hog)


def visualize_hog(image, hog_size=HOG_SIZE):
    """
    Return a rescaled HOG visualization image.

    Parameters
    ----------
    image : np.ndarray (H, W)
    hog_size : tuple

    Returns
    -------
    image_resized : np.ndarray
    hog_image_rescaled : np.ndarray
    """
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = sk_resize(image, hog_size)
    _, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(hog_image.min(), hog_image.max()))
    return image, hog_image_rescaled