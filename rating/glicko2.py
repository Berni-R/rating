from typing import Sequence
from attrs import define, field, validators
from math import sqrt, exp, log, pi
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import root_scalar
from scipy.special import expit

# constant tau which constrains the volatility over time
# TODO: get a better grasp of this parameter TAU
TAU: float = 0.5
_Q: float = log(10.0) / 400.0

_INIT_RATING = 1500.0
_INIT_RATING_DEV = 350.0
_DEF_VOLATILITY = 10.0  # ~= 0.06 / _Q ~= 10.423


@define(order=True, eq=True, hash=True)
class Glicko2:
    """Implementation of the Glicko-2 rating system.

    If you set the volatility to zero, it degenerates to the classical Glicko rating system (with no increase if the
    rating deviation over time, i.e. a constant c = 0). If no matches are played, Glicko-2 rating devations do increase
    due to a non-zero volatility. In this special case, one can make the assosication c**2 * t = vola**2 as long as the
    rating deviation is smaller than 350. With a typical volatility of 0.06 / (ln(10) / 400) ~ 10.423 and t = 1, we
    derive c = vola ~ 10.423. Or vice versa, with the example from Wikipedia
    (https://de.wikipedia.org/wiki/Glicko-System) of c ~ 34.64, where the deviation reaches 350 after 100 periods, we
    would have vola ~ 34.64 ~ 0.1994 / (ln(10) / 400).

    Note that Glicko-2 ratings has the ability to adopt much quicker to rating changes, than the classical Glicko
    system. This ability increases with higher volatility. At them same time, it might over-react to result that is a
    statical outlier. The volatility should be chosen appropiately to the game / situation.

    If a rating has zero rating deviation and zero volatility, it will never change, regardless of all outcomes. This
    can be used as fixed rating reference.

    In terms of comparisons, note that two ratings are always ordered on only the rating value, but euqality requires
    all attributes (r, dev, vola) to be the same. This implies that two ratings r1 and r2 might fulfil neither of the
    following equations: r1 < r2, r1 > r2, r1 == r2.
    """
    r: float = field(default=_INIT_RATING, order=True, eq=True, converter=float)
    dev: float = field(default=_INIT_RATING_DEV, order=False, eq=True, converter=float, validator=validators.ge(0))
    vola: float = field(default=_DEF_VOLATILITY, order=False, eq=True, converter=float, validator=validators.ge(0))

    def __float__(self) -> float:
        return self.r

    @classmethod
    def fixed_rating(cls, r: float) -> 'Glicko2':
        """Create a fixed rating, i.e. one that has zeros rating devation and volatility. Hence, it does not vary."""
        return Glicko2(r, dev=0.0, vola=0.0)

    @property
    def fixed(self) -> bool:
        """Whether this is a fixed rating, i.e. non-varying, since it has zeros rating devation and volatility."""
        return self.dev == 0.0 and self.vola == 0.0

    def fix(self):
        """Fixes this rating, i.e. setting `dev = 0` and `vola = 0`."""
        self.dev = 0.0
        self.vola = 0.0

    def g(self) -> float:
        # TODO: this ominous factor seems to be closer to 1/e than 3/pi^2 - change?!
        return 1.0 / sqrt(1.0 + 3.0 / pi**2 * (_Q * self.dev)**2)

    def expect(self, opponent: 'Glicko2', g: float = 1.0) -> float:
        """Calculate the expectation value of the outcome of a game agains the given opponent, where 0 stands for a
        loss, 1/2 is a draw, and 1 is a win. This is the winning probability in the absense of draws.

        :param opponent:    The oponnent to play.
        :param g:           The g-factor of the Glicko-2 rating system.
                            It should not be used (i.e. set to one), in normal usage.
        :return: The expected outcome, which is in the interval (0, 1).
        """
        return 1.0 / (1.0 + exp(-g * _Q * (self.r - opponent.r)))

    @classmethod
    def win_rate_2_rating_delta(cls, exepect_s: float) -> float:
        """How much higher does my rating have to be (given zero rating deviations) such that I have the given expected
        outcome (~ win rate)?"""
        delta_mu = -log(1.0 / exepect_s - 1.0)
        return delta_mu / _Q

    @classmethod
    def rating_delta_2_win_rate(cls, delta: float) -> float:
        """Given my rating is `delta` higher than the opponent's (who has zero rating deviation), what is the expected
        outcome (~ win rate)?"""
        return 1.0 / (1.0 + exp(-_Q * delta))

    def updated(
            self,
            opponents: Sequence['Glicko2'],
            results: Sequence[float | int] | ArrayLike,
            inplace: bool = False,
    ) -> 'Glicko2':
        """Update this rating after one rating period, where the corresponding player played against a number of rated
        opponents with given results (0: lost, 1/2: draw, 1: won).

        :param opponents:   A sequence of opponent's ratings against which the player played in this rating period.
        :param results:     The matching sequence of results for each of the opponents. Therefore, it must be of same
                            length as `opponents`. The individual results are supposed to be 0, if this player lost
                            against the opponent; 1/2 for a draw; and 1, if they won.
        :param inplace:     Whether to update the place in-place. This is not recommended, as if one updates the players
                            of a match sequentially, later updated would incorrectly use the updated rating of this
                            player.
        :return:    The updated rating (in case of `inplace = True` a reference to self).
        """
        # as described here: https://de.wikipedia.org/wiki/Glicko-System#Glicko-2-System
        if not isinstance(opponents, Sequence):
            raise ValueError("opponents must be some sequence")
        if not all(isinstance(opponent, Glicko2) for opponent in opponents):
            raise ValueError("opponents must be some sequence of type Glicko2")
        sj = np.array(results, copy=False, dtype=float)
        if sj.ndim != 1 or len(opponents) != len(sj):
            raise ValueError()

        if len(opponents) > 0:
            gj = np.fromiter((o.g() for o in opponents), dtype=float)
            ej = np.fromiter((self.expect(oj, gj) for oj, gj in zip(opponents, gj)), dtype=float)

            v = 1.0 / np.sum(gj**2 * ej * (1.0 - ej))
            s = float(np.sum(gj * (sj - ej)))

            if self.vola == 0.0:
                vola = 0.0
            else:
                vola = self._new_vol(s, v)

            dev_star = self.dev**2 + vola**2
            dev2 = 0.0 if dev_star == 0.0 else 1.0 / (1.0 / dev_star + _Q**2 / v)
            r = self.r + dev2 * s * _Q
            dev = sqrt(dev2)
        else:
            r = self.r
            dev = sqrt(self.dev ** 2 + self.vola ** 2)
            vola = self.vola

        if inplace:
            self.r = r
            self.dev = dev
            self.vola = vola
            return self
        else:
            return Glicko2(r, dev, vola)

    def _new_vol(self, s: float, v: float, eps: float = 1e-12) -> float:
        delta2 = (v * s)**2
        h = (_Q * self.dev)**2 + v
        lns2 = 2 * log(_Q * self.vola)
        tau2 = TAU**2

        def f(x):
            ex = exp(x)
            h_ex = h + ex
            return ex * (delta2 - h_ex) / 2 / h_ex**2 - (x - lns2) / tau2

        a = lns2
        if delta2 > h:
            b = log(delta2 - h)
        else:
            k = 1
            while f(a - k * TAU) < 0:
                k = k + 1
            b = a - k * TAU

        fa = f(a)
        fb = f(b)
        while abs(b - a) > eps:
            c = a + (a - b) * fa / (fb - fa)
            fc = f(c)
            if fc * fb <= 0:
                a = b
                fa = fb
            else:
                fa /= 2.0
            b = c
            fb = fc

        return exp(a / 2) / _Q


def performance_rating(
        opponents: Sequence[Glicko2],
        results: Sequence[float | int] | ArrayLike,
        clip_range: tuple[float, float] = (0.0, 4000.0),
        tol: float = 1e-15,
) -> float:
    """Calculate the rating that would not change in an update givent the opponents and results
    (a number only, deviation and volatility can be anything; the deviation would generally change).

    :param opponents:   The opponents played. (As one would pass to `Glicko2.update()`.)
    :param results:     The results for the respective opponents. (As one would pass to `Glicko2.update()`.)
    :param clip_range:  Some extreme results, would result in extreme ratings; esp. when all games are won or lost, the
                        performance rating would technically be plus or minus infinity.
                        To avoid this (and resulting numerical difficulties), limit the range of outcomes to this.
    :param tol:         The tolerance in the determined performance rating.
    :return: The performance rating as a single floating point number.
    """
    def change_of(r: float) -> float:
        return Glicko2(r, dev=np.inf, vola=0.0).updated(opponents, results).r - r

    change_lims = change_of(clip_range[0]), change_of(clip_range[1])
    if np.prod(change_lims) >= 0:
        return clip_range[0] if change_lims[0] < clip_range[0] else clip_range[1]

    # the next 8 lines are some code duplication, but it speeds up by about 3x opposed to using `Glicko2.update()`
    sj = np.array(results, copy=False, dtype=float)
    if sj.ndim != 1 or len(opponents) != len(sj):
        raise ValueError()
    oj = np.fromiter((o.r for o in opponents), dtype=float)
    gj = np.fromiter((o.g() for o in opponents), dtype=float)

    def delta_s(r: float):
        return np.sum(gj * (sj - expit(gj * _Q * (r - oj))))

    res = root_scalar(delta_s, bracket=clip_range, xtol=tol)
    if not res.converged:
        raise RuntimeError(res.flag)
    return float(np.clip(res.root, *clip_range))
