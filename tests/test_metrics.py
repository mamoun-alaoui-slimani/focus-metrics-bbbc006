import numpy as np
import pytest
from scipy import ndimage as ndi

from focus_metrics import METRICS

NAMES = list(METRICS)
# Plain intensity variance also drops with blur here, but it is not a sharpness
# metric in general (it tracks contrast), so it is held to the weaker checks only.
SHARPNESS = [n for n in NAMES if n != "Variance"]


@pytest.mark.parametrize("name", NAMES)
def test_returns_finite_float(name, cells):
    v = METRICS[name](cells)
    assert isinstance(v, float) and np.isfinite(v)


@pytest.mark.parametrize("name", NAMES)
def test_flat_image_scores_zero(name):
    assert METRICS[name](np.full((64, 64), 7.0)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", SHARPNESS)
def test_decreases_monotonically_with_blur(name, cells):
    vals = [METRICS[name](ndi.gaussian_filter(cells, s)) for s in (0.5, 1, 2, 4)]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False)), vals


@pytest.mark.parametrize("name", NAMES)
def test_peaks_at_focus(name, zstack):
    vals = [METRICS[name](im) for im in zstack]
    assert int(np.argmax(vals)) == 5


@pytest.mark.parametrize(
    "name, power",
    [("Variance", 2), ("Variance of Laplacian", 2), ("Brenner gradient", 2), ("Tenengrad (Sobel)", 2),
     ("Energy of gradient", 2), ("Vollath F4", 2), ("Sum-modified Laplacian", 1),
     ("Normalised variance", 1), ("High-frequency power ratio", 0)],
)
def test_intensity_scaling(name, power, cells):
    """Gain changes rescale each metric by a known power, so rankings across planes are gain-independent."""
    f = METRICS[name]
    assert f(3.0 * cells) == pytest.approx(3.0**power * f(cells), rel=1e-9)


# Vollath F4 is an autocorrelation difference, not a derivative: a background offset
# leaks in through the edge columns (about 6% for this field), so it is excluded.
OFFSET_INVARIANT = [n for n in NAMES if n not in ("Normalised variance", "Vollath F4")]


@pytest.mark.parametrize("name", OFFSET_INVARIANT)
def test_offset_invariance(name, cells):
    """A constant background offset must not change derivative-based metrics."""
    f = METRICS[name]
    assert f(cells + 1000) == pytest.approx(f(cells), rel=1e-9)
