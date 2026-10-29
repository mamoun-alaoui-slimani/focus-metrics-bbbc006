import numpy as np
import pytest

from focus_metrics import METRICS, best_focus, focus_curve, parabolic_peak


def test_parabolic_peak_recovers_vertex_exactly():
    z = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
    assert parabolic_peak(z, -(z - 1.3) ** 2) == pytest.approx(1.3)


def test_parabolic_peak_uneven_spacing():
    z = np.array([-16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0])  # the BBBC006 planes used here
    assert parabolic_peak(z, -(z - 2.5) ** 2) == pytest.approx(2.5)


def test_parabolic_peak_edge_falls_back_to_sample():
    z = np.arange(5.0)
    assert parabolic_peak(z, z) == 4.0


def test_parabolic_peak_rejects_bad_input():
    with pytest.raises(ValueError):
        parabolic_peak([0, 1, 2], [1, 2])


@pytest.mark.parametrize("name", METRICS)
def test_best_focus_on_synthetic_stack(name, zstack):
    assert best_focus(zstack, metric=name) == pytest.approx(5.0, abs=0.5)


def test_best_focus_physical_units(zstack):
    z_um = (np.arange(len(zstack)) - 5) * 2.0
    assert best_focus(zstack, z=z_um) == pytest.approx(0.0, abs=1.0)


def test_focus_curve_accepts_callable(zstack):
    np.testing.assert_allclose(focus_curve(zstack, np.var), [im.var() for im in zstack])


def test_focus_curve_rejects_2d(zstack):
    with pytest.raises(ValueError):
        focus_curve(zstack[0])
