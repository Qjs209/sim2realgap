# Sim2RealGap

A Python package for measuring the simulation-to-real gap in microscopy image simulations.

Sim2RealGap provides tools to quantitatively and visually compare real microscopy images against simulated ones, using a range of feature extraction and statistical comparison methods.

## Installation

```bash
pip install sim2realgap
```

Or install from source:

```bash
git clone https://github.com/Qjs209/Sim2RealGap
cd Sim2RealGap
pip install -e .
```

## Features

- Per-cell feature extraction (area, mean intensity, texture/std)
- Pixel-level distribution comparison
- GMM fitting and KL divergence (2D, 1D)
- HOG feature extraction and visualization
- SIFT keypoint extraction and visualization
- Histogram and KL matrix plotting

## Quick Start

```python
import sim2realgap as s2r
import numpy as np

# Load real data
real_images = s2r.load_video("path/to/real/frames")
real_masks  = s2r.load_video("path/to/real/masks")

# Load simulated data (any numpy array of shape (n_frames, H, W))
sim_images = np.load("path/to/sim_images.npy")
sim_masks  = np.load("path/to/sim_masks.npy")

frames = [0, 1, 2]

# Extract features
areas_real, intensities_real = s2r.collect_features(real_images, real_masks, frames)
areas_sim,  intensities_sim  = s2r.collect_features(sim_images,  sim_masks,  frames)

# Fit GMMs and compute KL divergence
gmm_real = s2r.fit_gmm(areas_real, intensities_real)
gmm_sim  = s2r.fit_gmm(areas_sim,  intensities_sim)

kl = s2r.kl_gmm(gmm_real, gmm_sim)
print(f"KL(real || sim) = {kl:.2f}")

# Plot
s2r.plot_histograms(
    {"Real": areas_real, "Simulated": areas_sim},
    title="Cell area distribution",
    xlabel="Area"
)
```

## API Reference

### Data loading and analysis
| Function | Description |
|---|---|
| `load_video(path, n_frames, frames)` | Load tiff stack from directory |
| `analyze_mask(mask, image)` | Extract per-cell features from mask |
| `get_masked_pixels(image, mask)` | Get cell pixel values excluding background |

### Feature collection
| Function | Description |
|---|---|
| `collect_features(images, masks, frames)` | Collect areas and intensities across frames |
| `collect_pixels(images, masks, frames)` | Collect masked cell pixel values |
| `collect_full_pixels(images, frames)` | Collect all pixel values without masking |
| `collect_cell_std(images, masks, frames)` | Collect per-cell texture (std) |

### GMM and KL divergence
| Function | Description |
|---|---|
| `fit_gmm(areas, intensities)` | Fit 2D GMM on log(area) and log(intensity) |
| `fit_gmm_1d(values)` | Fit 1D GMM on log(values) |
| `fit_gmm_pixels(values)` | Fit 1D GMM on pixel values |
| `kl_gmm(gmm_p, gmm_q)` | KL divergence KL(P \|\| Q) between two GMMs |
| `kl_gmm_1d(gmm_p, gmm_q)` | KL divergence between two 1D GMMs |

### HOG
| Function | Description |
|---|---|
| `collect_hog_features(images, frames)` | Collect HOG feature vectors across frames |
| `visualize_hog(image)` | Return HOG visualization image |

### SIFT
| Function | Description |
|---|---|
| `extract_sift(image)` | Extract SIFT keypoints and descriptors |
| `collect_sift_descriptors(images, frames)` | Collect SIFT descriptors across frames |
| `visualize_sift_keypoints(images_dict, hog_size, title)` | Visualize SIFT keypoints for multiple images side by side |

### Plotting
| Function | Description |
|---|---|
| `plot_kl_matrix(KL, names)` | Plot KL divergence matrix heatmap |
| `plot_histograms(data_dict, title, xlabel)` | Plot overlapping histograms |

## Requirements

- numpy
- matplotlib
- scikit-learn
- scikit-image
- Pillow
