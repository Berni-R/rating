from typing import Literal
from math import pi, sqrt
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import rv_continuous, cauchy
from scipy.stats import norm as normal_dist
from scipy.stats import t as students_t

from ..tournament import Tournament, Pair


class RatingBasedPairer:

    def __init__(
            self,
            sigma: float = 1.0,
            distribution: Literal['normal', 'cauchy', 'students-t'] | rv_continuous = 'students-t',
            nu: float = 2.0,
            random_order: bool = True,
            fix_edges: bool = True
    ):
        """A pairing generator based on the currect ratings differences, scaled by their mutual deviation. This is
        sqrt(dev_i**2 + dev_k**2) for dev_{i,k} the rating deviations of the players i and k.

        Note:
        We will generate pairings for each player that has played the minimum number of games of each player. If a for
        players i and k the pairings generated are (i, k) and (k, i) (or flipped versions, if `random_order = True`),
        both pairings will appear in the list, which is not to destroy the imposed individual, independent
        distributions (which are symmetric). This, however, will introduce some edge effects in games played for the
        total number of games played.

        :param sigma:           The width (in terms of the number of standard deviations) of the distribution used to
                                match players. If the distribution is a custom distribution (i.e. not a name), this
                                parameter is ignored.
        :param distribution:    The distribution to use when generating pairing probabilities. Naturally this is the
                                normal distribution, which should be appropiate in most situations.
                                However, if initially roughly equally strong players are clustered far apart, they might
                                get matched to seldomly for their ratings to approach quickly enough. This problem can
                                be mitigated, if one chooses a heavy-tailed distribution.
                                Another situation where heavy-tailed distributions might be helpful is when the relative
                                playing strength is not so perfectly transitive as assumed by the Glicko-2 rating
                                system.
        :param nu:              The additional parameter for the Cauchy distribution. Ignored, if the distribution is
                                a different one.
        :param random_order:    If this is True, generate the pairings in a random ordering, i.e. equally likely choose
                                between (i, k) and (k, i).
        :param fix_edges:       Only generate pairs for the players with the smallest number of games played in the
                                tournament, yet. (Which still may pair the other players.)

        :return:    A list of tuples of the indices of players that are paired.
        """
        self.random_order = random_order

        if not isinstance(distribution, rv_continuous):
            match distribution:
                case 'normal':
                    distribution = normal_dist(loc=0.0, scale=sigma)
                case 'cauchy':
                    # same mean absolute deviation (MAD) as 'normal'
                    distribution = cauchy(loc=0.0, scale=sigma * sqrt(pi / 2))
                case 'students-t':
                    distribution = students_t(loc=0.0, scale=sigma, df=nu)
                case _:
                    raise ValueError(f"unknown distribution '{distribution}'")
        self.dist: rv_continuous = distribution

        self.fix_edges = fix_edges

    def __call__(self, tournament: Tournament) -> list[Pair]:
        n_players = len(tournament.players)
        if n_players <= 1:
            return []

        # calculate the distances in rating, scaled by their mutual deviations
        r_val = np.array([r.r for r in tournament.ratings]).reshape((-1, 1))
        r_dev = np.array([r.dev for r in tournament.ratings]).reshape((-1, 1))
        rating_dists = distance_matrix(r_val, r_val)
        d_scale = np.sqrt(np.sum(r_dev[:, None, :]**2 + r_dev[None, :, :]**2, axis=-1))
        # if there is at least one fixed rating some d_scale are 0, at least those on the diagonal for this fixed rating
        finite_scale = (d_scale > 0)
        rating_dists[finite_scale] /= d_scale[finite_scale]
        rating_dists[~finite_scale] = np.inf

        n_games = tournament.games_per_player() if self.fix_edges else np.zeros(len(tournament.players))
        pairings: list[Pair] = []
        for i in np.where(np.min(n_games) == n_games)[0]:  # ensure that on average there will be n_players / 2 pairs
            p = self.dist.pdf(rating_dists[i])
            p[i] = 0.0
            p /= np.sum(p)
            k = np.random.choice(np.arange(n_players), p=p)
            i = int(i)
            k = int(k)

            n_games[i] += 1
            # do NOT `n_games[k] += 1`, since we want the distributions from above to hold, which are independent

            if self.random_order and np.random.rand() < 0.5:
                pair = (k, i)
            else:
                pair = (i, k)
            pairings.append(pair)

        return pairings
