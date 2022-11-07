import pytest
import numpy as np

from rating.glicko2 import Glicko2
from rating.tournament import Tournament, RandomPairer


class Player:

    def __init__(self, intrinsic_rating: float):
        self.intrinsic_rating = intrinsic_rating

    def __repr__(self) -> str:
        return f"Player({self.intrinsic_rating:.0f})"


def my_game(p1: Player, p2: Player) -> float:
    d = p1.intrinsic_rating - p2.intrinsic_rating
    w = Glicko2.rating_delta_2_win_rate(d)
    return np.random.choice([0, 1/2, 1], p=[1-w, 0, w])


players = [Player(r) for r in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]]


def test_tournament():
    intrinsic_ratings = [Glicko2(p.intrinsic_rating) for p in players]

    pairer = RandomPairer()

    with pytest.raises(ValueError):
        Tournament(players, my_game, pairer, initial_ratings=[])
    with pytest.raises(ValueError):
        Tournament([], my_game, pairer, initial_ratings=intrinsic_ratings)
    with pytest.raises(ValueError):
        Tournament(players[:3], my_game, pairer, initial_ratings=intrinsic_ratings[:4])

    for initital_ratings in [None, intrinsic_ratings]:
        t = Tournament(players, my_game, pairer, initial_ratings=initital_ratings)

        assert all(p1 is p2 for p1, p2 in zip(players, t.players))

        n = 100
        t.hold(n, pbar=False)

        games = t.games_per_player()

        pair_cnt = t.pair_counts(symmetric=True)
        assert np.allclose(pair_cnt, pair_cnt.T)
        assert np.all(games == pair_cnt.sum(axis=1))
        pair_cnt = t.pair_counts(symmetric=False)
        assert np.all(games == pair_cnt.sum(axis=1) + pair_cnt.T.sum(axis=1))

        if initital_ratings is not None:
            assert all(r1 == r2 for r1, r2 in zip(initital_ratings, t.initial_ratings))
        assert all(r1 != r2 for r1, r2 in zip(t.initial_ratings, t.ratings))

        rating_devs = [r.dev for r in t.ratings]
        assert np.allclose(np.array(t.ratings, dtype=float), np.array(intrinsic_ratings, dtype=float),
                           rtol=np.max(rating_devs))

        w = t.pair_scores(symmetric=True, average=True)
        assert np.allclose(w, 1 - w.T, equal_nan=True)
