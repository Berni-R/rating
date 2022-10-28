import pytest
import numpy as np

from rating.glicko2 import Glicko2


init_test_data = [
    (1500, 350, 10),
    (1234.5, 432.1, 8.7),
    (1683.32, 62.7, 11.0),
    (931.1, 58.5, 0),
    (-53, 0, 0),
    (2100, 0, 12),
]

expectation_test_data = [
    (1200, 1200, 0.5),
    (2300, 2300, 0.5),
    (1234, 1234 - 400, (1 - 1 / 11)),
    (1234, 1234 + 400, 1 / 11),
    (1677, 1677 - 400, 1 / 1.1),
    (3245, 3245 - Glicko2.win_rate_2_rating_delta(1/3), 1/3),
    (1500, 1500 - Glicko2.win_rate_2_rating_delta(0.863), 0.863),
    (1304, 1304 - Glicko2.win_rate_2_rating_delta(0.678), 0.678),
]


@pytest.mark.parametrize("r, dev, vola", init_test_data)
def test_glicko2_init(r: float, dev: float, vola: float):
    rating = Glicko2(r=r, dev=dev, vola=vola)
    # assert repr(rating) == f"Glicko2(r={r:.1f}, dev={dev:.1f}, vola={vola:.1f})"
    assert float(rating) == r
    assert np.allclose(rating.r, r)
    assert np.allclose(rating.dev, dev)
    assert np.allclose(rating.vola, vola)


@pytest.mark.parametrize("r, dev, vola", init_test_data)
def test_glicko2_set_get(r: float, dev: float, vola: float):
    rating = Glicko2()
    rating.r = r
    rating.dev = dev
    rating.vola = vola
    assert np.allclose(rating.r, r)
    assert np.allclose(rating.dev, dev)
    assert np.allclose(rating.vola, vola)


@pytest.mark.parametrize("r, dev, vola", init_test_data)
def test_glicko2_setter_raises(r: float, dev: float, vola: float):
    if dev == 0 or vola == 0:
        return
    rating = Glicko2(r)
    with pytest.raises(ValueError):
        rating.dev = -dev
    with pytest.raises(ValueError):
        rating.vola = -vola


@pytest.mark.parametrize("r, dev, vola", init_test_data)
def test_glicko2_equal(r: float, dev: float, vola: float):
    r1 = Glicko2(r=r, dev=dev, vola=vola)
    assert r1 == Glicko2(r=r, dev=dev, vola=vola)
    assert not (r1 != Glicko2(r=r, dev=dev, vola=vola))

    assert not r1 == Glicko2(r=r - 42, dev=dev, vola=vola)
    assert r1 != Glicko2(r=r - 42, dev=dev, vola=vola)
    assert r1 != Glicko2(r=r, dev=dev + 1, vola=vola)
    assert r1 != Glicko2(r=r, dev=dev, vola=vola + 0.1)


@pytest.mark.parametrize("low, high", [(1300, 1500), (1234.5, 1234.6), (1800, 2000)])
def test_glicko2_lt_gt(low: float, high: float):
    assert low < high  # this is what we build on!

    # different dev and vola
    r_low = Glicko2(r=low, dev=100*np.random.rand(), vola=10 * np.random.rand())
    r_high = Glicko2(r=high, dev=100*np.random.rand(), vola=10 * np.random.rand())
    assert r_low < r_high
    assert r_high > r_low

    # same dev and vola
    for dev, vola in [(0, 0), (50, 0), (0, 10), (30, 5),
                      (100*np.random.rand(), 0), (0, 10*np.random.rand()),
                      (100*np.random.rand(), 10*np.random.rand())]:
        r_low = Glicko2(r=low, dev=dev, vola=vola)
        r_high = Glicko2(r=high, dev=dev, vola=vola)
        assert r_low < r_high
        assert r_high > r_low


@pytest.mark.parametrize("r, dev, vola", init_test_data)
def test_glicko2_fixed(r: float, dev: float, vola: float):
    ref = Glicko2.fixed_rating(r)
    assert ref.r == r
    assert ref.dev == 0
    assert ref.vola == 0
    assert ref.fixed

    rating = Glicko2(r, dev=1 + dev, vola=0)
    assert not rating.fixed

    rating = Glicko2(r, dev=0, vola=1 + vola)
    assert not rating.fixed

    if not dev == 0 and vola == 0:
        rating = Glicko2(r, dev, vola)
        assert not rating.fixed
        rating.fix()
        assert rating.dev == 0
        assert rating.vola == 0
        assert rating.fixed


@pytest.mark.parametrize("ra, r0, expected", expectation_test_data)
def test_glicko2_expect(ra: float, r0: float, expected: float):
    assert np.allclose(ra, r0 + Glicko2.win_rate_2_rating_delta(expected))
    assert np.allclose(Glicko2.rating_delta_2_win_rate(ra - r0), expected)

    r = Glicko2.fixed_rating(ra)
    ref = Glicko2.fixed_rating(r0)
    assert np.allclose(r.expect(ref), expected)

    ref = Glicko2(r0, dev=1 + 300 * np.random.rand())
    g = ref.g()
    if expected == 0.5:
        assert r.expect(ref, g=g) == 0.5
    else:
        assert abs(r.expect(ref, g=g) - 0.5) < abs(expected - 0.5)

    r = Glicko2(ra, dev=1 + 300 * np.random.rand())
    ref = Glicko2.fixed_rating(r0)
    assert np.allclose(r.expect(ref, g=ref.g()), expected)
