import numpy as np
import pytest
from scipy import ndimage as ndi


@pytest.fixture(scope="session")
def cells():
    """Synthetic fluorescence-like field: sparse Gaussian blobs on a dim background."""
    rng = np.random.default_rng(0)
    img = np.zeros((128, 160))
    ys, xs = rng.integers(4, 124, 40), rng.integers(4, 156, 40)
    img[ys, xs] = rng.uniform(500, 1500, 40)
    return ndi.gaussian_filter(img, 1.5) * 20 + 100.0


@pytest.fixture(scope="session")
def zstack(cells):
    """11-plane stack, in focus at plane 5; defocus blur grows linearly with |z - 5|."""
    return np.stack([ndi.gaussian_filter(cells, 0.3 + 0.8 * abs(z - 5)) for z in range(11)])
