import numpy as np
from sklearn.mixture import GaussianMixture


def make_feature_matrix(areas, intensities):
    """2D log-transformed feature matrix from areas and intensities."""
    return np.column_stack([np.log(np.array(areas)), np.log(np.array(intensities))])


def make_feature_matrix_1d(values):
    """1D log-transformed feature matrix."""
    return np.log(np.array(values)).reshape(-1, 1)


def make_feature_matrix_pixels(values):
    """1D feature matrix for pixel values (no log transform)."""
    return np.array(values).reshape(-1, 1)


def fit_gmm(areas, intensities, n_components=3):
    """Fit a 2D GMM on log(area) and log(intensity)."""
    X = make_feature_matrix(areas, intensities)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(X)
    return gmm


def fit_gmm_1d(values, n_components=3):
    """Fit a 1D GMM on log(values)."""
    X = make_feature_matrix_1d(values)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(X)
    return gmm


def fit_gmm_pixels(values, n_components=3):
    """Fit a 1D GMM on pixel values (no log transform)."""
    X = make_feature_matrix_pixels(values)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(X)
    return gmm


def kl_gmm(gmm_p, gmm_q, n_samples=5000):
    """
    Monte Carlo estimate of KL divergence KL(P || Q) between two GMMs.

    Parameters
    ----------
    gmm_p : GaussianMixture — the reference distribution P
    gmm_q : GaussianMixture — the approximating distribution Q
    n_samples : int

    Returns
    -------
    float — KL divergence estimate
    """
    X, _ = gmm_p.sample(n_samples)
    log_p = gmm_p.score_samples(X)
    log_q = gmm_q.score_samples(X)
    return float(np.mean(log_p - log_q))


def kl_gmm_1d(gmm_p, gmm_q, n_samples=5000):
    """KL divergence between two 1D GMMs."""
    return kl_gmm(gmm_p, gmm_q, n_samples)