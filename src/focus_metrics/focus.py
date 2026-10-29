"""Find the best-focus plane of a z-stack with any metric from `METRICS`."""
import numpy as np

from .metrics import METRICS


def focus_curve(stack, metric="Variance of Laplacian"):
    """Metric value for each plane of a (n_z, H, W) stack."""
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"expected a (n_z, H, W) stack, got shape {stack.shape}")
    fn = METRICS[metric] if isinstance(metric, str) else metric
    return np.array([fn(im) for im in stack])


def parabolic_peak(z, values):
    """Sub-plane peak position: parabola through the maximum and its two neighbours.

    Falls back to the sampled maximum when the peak is on the edge of the
    stack or the three points are not concave.
    """
    z = np.asarray(z, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if z.shape != v.shape or z.ndim != 1 or len(z) == 0:
        raise ValueError("z and values must be 1D arrays of the same length")
    i = int(np.argmax(v))
    if i == 0 or i == len(v) - 1:
        return float(z[i])
    x0, x1, x2 = z[i - 1:i + 2]
    a, b, _ = np.linalg.solve(np.array([[x0**2, x0, 1], [x1**2, x1, 1], [x2**2, x2, 1]]), v[i - 1:i + 2])
    return float(np.clip(-b / (2 * a), x0, x2)) if a < 0 else float(z[i])


def best_focus(stack, z=None, metric="Variance of Laplacian"):
    """Estimated best-focus position of a stack (in units of `z`, default plane index)."""
    curve = focus_curve(stack, metric)
    z = np.arange(len(curve)) if z is None else z
    return parabolic_peak(z, curve)
