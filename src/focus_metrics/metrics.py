"""Focus (sharpness) metrics for 2D microscopy images.

Every metric takes a 2D float image and returns a scalar that should be
largest at best focus. Images are not normalised inside the metrics unless
the metric itself is defined that way (e.g. normalised variance).
"""
import numpy as np
from scipy import ndimage as ndi


def variance(img):
    return float(img.var())


def normalized_variance(img):
    m = img.mean()
    return float(img.var() / m) if m > 0 else 0.0


def variance_of_laplacian(img):
    return float(ndi.laplace(img).var())


def sum_modified_laplacian(img):
    dxx = np.abs(2 * img[:, 1:-1] - img[:, :-2] - img[:, 2:])
    dyy = np.abs(2 * img[1:-1, :] - img[:-2, :] - img[2:, :])
    return float(dxx[1:-1, :].mean() + dyy[:, 1:-1].mean())


def brenner(img):
    d = img[:, 2:] - img[:, :-2]
    return float((d ** 2).mean())


def tenengrad(img):
    gx = ndi.sobel(img, axis=1)
    gy = ndi.sobel(img, axis=0)
    return float((gx ** 2 + gy ** 2).mean())


def energy_of_gradient(img):
    gx = np.diff(img, axis=1)[:-1, :]
    gy = np.diff(img, axis=0)[:, :-1]
    return float((gx ** 2 + gy ** 2).mean())


def vollath_f4(img):
    a = (img[:, :-1] * img[:, 1:]).mean()
    b = (img[:, :-2] * img[:, 2:]).mean()
    return float(a - b)


def high_freq_ratio(img, cutoff=0.15):
    """Fraction of spectral power above `cutoff` x Nyquist (mean removed)."""
    f = np.fft.rfft2(img - img.mean())
    p = np.abs(f) ** 2
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.rfftfreq(img.shape[1])[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2) / 0.5
    tot = p.sum()
    return float(p[r > cutoff].sum() / tot) if tot > 0 else 0.0


METRICS = {
    "Variance": variance,
    "Normalised variance": normalized_variance,
    "Variance of Laplacian": variance_of_laplacian,
    "Sum-modified Laplacian": sum_modified_laplacian,
    "Brenner gradient": brenner,
    "Tenengrad (Sobel)": tenengrad,
    "Energy of gradient": energy_of_gradient,
    "Vollath F4": vollath_f4,
    "High-frequency power ratio": high_freq_ratio,
}
