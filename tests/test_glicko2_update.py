import pytest
import numpy as np
from typing import Sequence
from numpy.typing import ArrayLike
from math import exp, sqrt, pi

from rating.glicko2 import Glicko2
from rating.glicko2 import _Q, _INIT_RATING_DEV


def random_glicko(dev: float | None = None, vol: float | None = None):
    r = np.random.normal(1500, 500)
    r = np.clip(r, -100, 2100)
    if dev is None:
        dev = 1 + 100 * np.random.rand()
    if vol is None:
        vol = 0.1 + 10 * np.random.rand()
    return Glicko2(r=float(r), dev=dev, vol=vol)


def test_update_input():
    """`Glicko2.updated` should raise Errors with some bad inputs."""
    r = random_glicko()
    # no raising
    r.updated(opponents=[r, r, r, r], results=[1.0, 0.0, 0.5, 1.0])
    r.updated(opponents=(r, r, r, r), results=(1.0, 0.0, 0.5, 1.0))
    r.updated(opponents=(r, r), results=np.array([0.0, 0.0]))

    with pytest.raises(ValueError):
        r.updated(opponents=[r, r, None], results=[1.0, 0.0, 0.5])
    with pytest.raises(ValueError):
        r.updated(opponents=[7, r], results=[1, 0])
    with pytest.raises(ValueError):
        r.updated(opponents=[r, r], results=[1, 'text'])

    with pytest.raises(ValueError):
        r.updated(opponents=[r, r], results=[1])

    with pytest.raises(ValueError):
        r.updated(opponents=r, results=[1])
    with pytest.raises(ValueError):
        r.updated(opponents=[r], results=1)
    with pytest.raises(ValueError):
        r.updated(opponents=r, results=0)


def test_update_empty():
    """If no games were played, nothing should change, or the deviation should increase with the volatility."""
    for _ in range(10):
        # differentiate between `vol > 0` and `vol = 0`
        r = random_glicko()
        r_new = r.updated([], [])
        assert r_new.r == r.r
        assert r_new.vol == r.vol
        assert r_new.dev > r.dev
        assert np.allclose(r_new.dev**2, r.dev**2 + r.vol**2)

        r = random_glicko(vol=0)
        r_new = r.updated([], [])
        assert r_new.r == r.r
        assert r_new.vol == r.vol
        assert r_new.dev == r.dev


def test_update_fixed():
    """Fixed Glicko2 ratings should never change during the update (and remain `dev == 0` and `vol == 0`)."""
    for _ in range(10):
        fixed = Glicko2.fixed_rating(r=np.random.normal(1500, 500))
        r = random_glicko()
        n = np.random.randint(1, 10)
        fixed_new = fixed.updated([r] * n, np.random.randint(3, size=n) / 2)
        assert fixed_new.r == fixed.r
        assert fixed_new.vol == 0
        assert fixed_new.dev == 0


def test_update_basics():
    """Updates should fulfil a few basic requirements:
        - winning should always increase the rating
        - loosing should always decrease the rating
        - two ratings with the same rating deviation and volatility should get the same, but opposite updates in a
          single game (i.e. one looses while the other wins, or it is a draw for both)
    """
    for _ in range(10):
        # create two different, random ratings
        r1 = random_glicko()
        r2 = random_glicko()
        if abs(float(r1) - float(r2)) < 10:
            continue

        # winning will always increase the rating - by how much is not clear, though
        r1_new = r1.updated([r2], [1])
        assert r1_new > r1
        # loosing will always decrease the rating
        r1_new = r1.updated([r2], [0])
        assert r1_new < r1

    # rating update should be symmetric, if the deviations and volatilities are the same
    for _ in range(10):
        # create two different, random ratings
        dev, vol = 100 * np.random.rand(), 10 * np.random.rand()
        r1 = random_glicko(dev=dev, vol=vol)
        r2 = random_glicko(dev=dev, vol=vol)
        if abs(float(r1) - float(r2)) < 10:
            continue

        for s in [0, 1/2, 1]:
            delta_r1 = r1.updated([r2], [s]).r - r1.r
            delta_r2 = r2.updated([r1], [1 - s]).r - r2.r
            assert np.allclose(delta_r1, -delta_r2)


def test_update_inplace():
    """The other test already implicitly assumed and tested that the `Glicko2.updated` does not change the instance
    itself. Here test, that it does when `inplace=True` is passed."""
    for _ in range(10):
        r1 = random_glicko()
        r2 = random_glicko()
        r3 = random_glicko()
        r1_init = Glicko2(r1.r, r1.dev, r1.vol)
        r1.updated([r2, r3], np.random.randint(3, size=2) / 2, inplace=True)
        assert r1 != r1_init
        assert r1.r != r1_init.r
        assert r1.dev != r1_init.dev
        assert r1.vol != r1_init.vol


def classical_update(rating: Glicko2, opponents: Sequence[Glicko2], sj: ArrayLike, c2_t: float = 0.0):
    # as described here: https://de.wikipedia.org/wiki/Glicko-System#Klassisches_Glicko-System
    def g(o: Glicko2) -> float:
        return 1.0 / sqrt(1.0 + 3.0 / pi**2 * _Q**2 * o.dev**2)

    def e(opponent: Glicko2) -> float:
        return 1.0 / (1.0 + exp(-g(opponent) * _Q * (rating.r - opponent.r)))

    # step 1
    rd_star2 = min(rating.dev**2 + c2_t, _INIT_RATING_DEV**2)

    # step 2.1
    gj = np.array([g(o) for o in opponents])
    ej = np.array([e(o) for o in opponents])
    d2_inv = _Q**2 * np.sum(gj**2 * ej * (1.0 - ej))
    s = np.sum(gj * (sj - ej))

    # step 2.2
    rd2 = 1 / (1 / rd_star2 + d2_inv)
    mu = rating.r + _Q * rd2 * s
    rd = sqrt(rd2)
    return mu, rd


def test_update_classical_correspondence():
    """In the case of zero volatility, the Glicko-2 system should correspond to a classical Glicko stystem with the
    constant c = 0, which equates to the fact, that the rating deviation does not change with time, when no games are
    played. Exactly that happens with zero volatility, too."""
    for _ in range(10):
        m = np.random.randint(1, 11)
        rating = random_glicko(vol=0)
        opps = [random_glicko(vol=0) for _ in range(m)]
        sj = np.random.choice([0, 1 / 2, 1], size=m)
        r, rd = classical_update(rating, opps, sj)
        r2 = rating.updated(opps, sj)
        assert np.allclose(r, r2.r)
        assert np.allclose(rd, r2.dev)


REF_RATING = 1500.0


def emperically_adjust(
        expected: float,
        n: int = 200, m: int = 1, r0: Glicko2 | None = None,
        draw_p: float = 0.0, reference: float = REF_RATING,
) -> tuple[Glicko2, Glicko2]:
    r = Glicko2() if r0 is None else r0
    ref = Glicko2.fixed_rating(reference)

    p = np.array([1.0 - expected, 0.0, expected], dtype=float)
    p[0] -= draw_p / 2
    p[1] += draw_p
    p[2] -= draw_p / 2
    assert np.all(p >= 0) and np.allclose(np.sum(p), 1.0)

    for k in range(n):
        s = np.random.choice([0, 1/2, 1], size=m, p=p)
        r = r.updated([ref] * m, s)
        ref = ref.updated([r] * m, 1.0 - s)

    return r, ref


@pytest.mark.parametrize("win_rate, draw", [(0.5, 0), (0.2, 0), (0.678, 0.1), (0.123, 0), (0.4, 0.05), (0.85, 0),
                                            (0.92, 0), (0.5, 0.4)])
def test_update_statistically(win_rate: float, draw: float, sigma: float = 3.0):
    """This test should be fine in the vast majority of the cases, but still can statistically fail once in a while!"""
    Glicko2.TAU = 0.2 + 1.0 * np.random.rand()
    r, ref = emperically_adjust(win_rate)
    assert ref == Glicko2.fixed_rating(REF_RATING)
    w_low, w_high = Glicko2(r.r - sigma * r.dev).expect(ref), Glicko2(r.r + sigma * r.dev).expect(ref)
    assert w_low <= win_rate <= w_high, (r, ref, f"might statstically fail the {sigma} sigma bound - try to re-run")

    # much longer empirical testing with much lower volatility to decrease the rating deviation
    # this leads to stronger confidence intervals
    r.vol = 0.1
    r, ref = emperically_adjust(win_rate, n=1000, m=10, r0=r)
    assert r.dev < 10
    w_low, w_high = Glicko2(r.r - sigma * r.dev).expect(ref), Glicko2(r.r + sigma * r.dev).expect(ref)
    assert w_low <= win_rate <= w_high, (r, ref, f"might statstically fail the {sigma} sigma bound - try to re-run")
