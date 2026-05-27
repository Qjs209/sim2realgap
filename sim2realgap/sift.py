import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import SIFT
from skimage.transform import resize as sk_resize

HOG_SIZE = (128, 128)


def extract_sift(image, hog_size=HOG_SIZE):
    """
    Extract SIFT keypoints and descriptors from an image.

    Parameters
    ----------
    image : np.ndarray (H, W)
    hog_size : tuple

    Returns
    -------
    keypoints : np.ndarray
    descriptors : np.ndarray
    """
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = sk_resize(image, hog_size)
    detector = SIFT()
    detector.detect_and_extract(image)
    return detector.keypoints, detector.descriptors


def collect_sift_descriptors(images, frames, hog_size=HOG_SIZE):
    """
    Collect flattened SIFT descriptors across frames.

    Parameters
    ----------
    images : np.ndarray (n_frames, H, W)
    frames : list of int
    hog_size : tuple

    Returns
    -------
    np.ndarray of flattened descriptors.
    """
    all_desc = []
    for f in frames:
        image = np.array(images[f], dtype=float)
        try:
            _, descriptors = extract_sift(image, hog_size)
            all_desc.append(descriptors)
        except RuntimeError:
            pass
    return np.concatenate(all_desc).flatten() if all_desc else np.array([])


def visualize_sift_keypoints(images_dict, hog_size=HOG_SIZE, title="SIFT Keypoints"):
    """
    Visualize SIFT keypoints for multiple images side by side.

    Parameters
    ----------
    images_dict : dict of {label: np.ndarray (H, W)}
        e.g. {"Real": real_img, "Default": default_img, "Inferred": inferred_img}
    hog_size : tuple
    title : str
    """
    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, image) in zip(axes, images_dict.items()):
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = sk_resize(image, hog_size)
        try:
            keypoints, _ = extract_sift(image, hog_size)
            ax.scatter(keypoints[:, 1], keypoints[:, 0],
                       s=10, c='red', alpha=0.6)
        except RuntimeError:
            pass
        ax.imshow(image, cmap='gray')
        ax.set_title(label)
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()