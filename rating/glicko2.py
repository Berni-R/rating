from typing import Sequence, Any, Union
from math import sqrt, exp, log, pi
import numpy as np
from numpy.typing import ArrayLike


_INIT_RATING = 1500.0
_INIT_RATING_DEV = 350.0
_Q = log(10.0) / 400.0


class Glicko2:
    """Implementation of the Glicko-2 rating system.

    If you set the volatility to zero, it degenerates to the classical Glicko rating system (with no increase if the
    rating deviation over time, i.e. a constant c = 0). If no matches are played, Glicko-2 rating devations do increase
    due to a non-zero volatility. In this special case, one can make the assosication c**2 * t = vol**2 as long as the
    rating deviation is smaller than 350. With a typical volatility of 0.06 / (ln(10) / 400) ~ 10.423 and t = 1, we
    derive c = vol ~ 10.423. Or vice versa, with the example from Wikipedia
    (https://de.wikipedia.org/wiki/Glicko-System) of c ~ 34.64, where the deviation reaches 350 after 100 periods, we
    would have vol ~ 34.64 ~ 0.1994 / (ln(10) / 400).

    Note that Glicko-2 ratings has the ability to adopt much quicker to rating changes, than the classical Glicko
    system. This ability increases with higher volatility. At them same time, it might over-react to result that is a
    statical outlier. The volatility should be chosen appropiately to the game / situation.

    If a rating has zero rating deviation and zero volatility, it will never change, regardless of all outcomes. This
    can be used as fixed rating reference.

    In terms of comparisons, note that two ratings are always ordered on only the rating value, but euqality requires
    all attributes (r, dev, vol) to be the same. This implies that two ratings r1 and r2 might fulfil neither of the
    following equations: r1 < r2, r1 > r2, r1 == r2.
    """

    # constant tau which constrains the volatility over time
    TAU = 0.5
    DEF_VOLATILITY = 0.06 / _Q

    def __init__(self, r: float = _INIT_RATING, dev: float = _INIT_RATING_DEV, vol: float | None = None):
        """Create a classical Glicko rating or a Glicko-2 rating.

        :param r:   The (initial) rating / strength.
        :param dev: The (initial) rating deviation.
        :param vol: For a classical Glicko rating, pass zero; for a Glicko-2 rating, pass a positive (initial)
                    volatility, e.g. 0.06 / (ln(10) / 400) ~ 10.423 (the default) as done in "Example of the Glicko-2
                    system" Glickman (2022; http://www.glicko.net/glicko/glicko2.pdf).
                    Defaults to Glicko2.DEF_VOLATILITY.
        """
        if vol is None:
            vol = Glicko2.DEF_VOLATILITY
        self.r = r
        self.dev = dev
        self.vol = vol

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Glicko2) and (
            self._mu == other._mu
            and self._phi == other._phi
            and self._sigma == other._sigma
        )

    def __lt__(self, other: Union['Glicko2', float, int]) -> bool:
        return self.r < float(other)

    def __gt__(self, other: Union['Glicko2', float, int]) -> bool:
        return self.r > float(other)

    @classmethod
    def _by_unscaled_params(cls, mu: float, phi: float, sigma: float) -> 'Glicko2':
        new = cls.__new__(cls)
        new._mu = mu
        new._phi = phi
        new._sigma = sigma
        return new

    @classmethod
    def fixed_rating(cls, r: float) -> 'Glicko2':
        """Create a fixed rating, i.e. one that has zeros rating devation and volatility. Hence, it does not vary."""
        return Glicko2(r, dev=0.0, vol=0.0)

    @property
    def is_fixed(self) -> bool:
        """Whether this is a fixed rating, i.e. non-varying, since it has zeros rating devation and volatility."""
        return self._phi == 0.0 and self._sigma == 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.r:.6g}, dev={self.dev:.6g}, vol={self.vol:.6g})"

    def __float__(self) -> float:
        return self.r

    @property
    def r(self) -> float:
        """The rating value."""
        return self._mu / _Q + _INIT_RATING

    @r.setter
    def r(self, r: float):
        self._mu = _Q * (r - _INIT_RATING)

    @property
    def dev(self) -> float:
        """The rating deviation."""
        return self._phi / _Q

    @dev.setter
    def dev(self, dev: float):
        if dev < 0:
            raise ValueError("rating deviation cannot be negative")
        self._phi = _Q * dev

    @property
    def vol(self) -> float:
        """The rating volatility."""
        return self._sigma / _Q

    @vol.setter
    def vol(self, vol: float):
        if vol < 0:
            raise ValueError("volatility cannot be negative")
        self._sigma = _Q * vol

    def g(self) -> float:
        return 1.0 / sqrt(1.0 + 3.0 / pi**2 * self._phi**2)

    def expect(self, opponent: 'Glicko2') -> float:
        """Calculate the expectation value of the outcome of a game agains the given opponent, where 0 stands for a
        loss, 1/2 is a draw, and 1 is a win. This is the winning probability in the absense of draws."""
        return 1.0 / (1.0 + exp(-opponent.g() * (self._mu - opponent._mu)))

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
        opps = np.array(opponents)
        sj = np.array(results).astype(float)
        if sj.ndim != 1 or len(opps) != len(sj):
            raise ValueError()

        if len(opps) > 0:
            gj = np.array([o.g() for o in opps])
            ej = np.array([self.expect(o) for o in opps])

            v = 1.0 / np.sum(gj**2 * ej * (1.0 - ej))
            s = np.sum(gj * (sj - ej))

            if self._sigma == 0.0:
                sigma = 0.0
            else:
                delta = v * s
                sigma = self._new_sigma(delta**2, v)

            phi_star_2 = self._phi**2 + sigma**2

            phi2 = 0.0 if phi_star_2 == 0.0 else 1.0 / (1.0 / phi_star_2 + 1 / v)
            mu = self._mu + float(phi2 * s)
            phi = sqrt(phi2)
        else:
            mu = self._mu
            phi = sqrt(self._phi**2 + self._sigma**2)
            sigma = self._sigma

        if inplace:
            self._mu = mu
            self._phi = phi
            self._sigma = sigma
            return self
        else:
            return Glicko2._by_unscaled_params(mu, phi, sigma)

    def _new_sigma(self, delta2: float, v: float, eps: float = 1e-12):
        h = self._phi**2 + v
        lns2 = log(self._sigma**2)
        tau2 = self.TAU**2

        def f(x):
            ex = exp(x)
            h_ex = h + ex
            return ex * (delta2 - h_ex) / 2 / h_ex**2 - (x - lns2) / tau2

        a = lns2
        if delta2 > h:
            b = log(delta2 - h)
        else:
            k = 1
            while f(a - k * sqrt(self.TAU**2)) < 0:
                k = k + 1
            b = a - k * sqrt(self.TAU**2)

        fa = f(a)
        fb = f(b)
        while abs(b - a) > eps:
            c = a + (a - b) * fa / (fb - fa)
            fc = f(c)
            if fc * fb < 0:
                a = b
                fa = fb
            else:
                fa /= 2.0
            b = c
            fb = fc

        return exp(a / 2)
