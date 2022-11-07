from typing import TypeVar, TypeAlias, Callable, Sequence
from collections import OrderedDict
from copy import copy
import warnings
import gc
from tqdm.auto import trange
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..glicko2 import Glicko2, performance_rating

Player = TypeVar('Player')
Game: TypeAlias = Callable[[Player, Player], float]
Pair: TypeAlias = tuple[int, int]
Pairer: TypeAlias = Callable[['Tournament'], list[Pair]]
RoundHistory: TypeAlias = list[tuple[Pair, float]]


def gather_opponents_scores(
        ratings: Sequence[Glicko2],
        matches: RoundHistory,
) -> tuple[list[list[Glicko2]], list[list[float]]]:
    # gather opponents and scores for each player
    opponents: list[list[Glicko2]] = [[] for _ in ratings]
    scores: list[list[float]] = [[] for _ in ratings]
    for (i, k), s in matches:
        opponents[i].append(ratings[k])
        scores[i].append(s)
        opponents[k].append(ratings[i])
        scores[k].append(1.0 - s)

    return opponents, scores


def update_ratings(
        ratings: Sequence[Glicko2],
        matches: RoundHistory,
) -> tuple[Glicko2, ...]:
    opponents, scores = gather_opponents_scores(ratings, matches)
    # -> tuple[list[list[Glicko2]], list[list[float]]]:
    return tuple(r.updated(opp, s) for r, opp, s in zip(ratings, opponents, scores))


class Tournament:
    """A tournament held for a two player game of `n` players, tracking performances in the Glicko-2 rating system."""

    def __init__(
            self,
            players: Sequence[Player],
            game: Game,
            pairer: Pairer,
            initial_ratings: Sequence[Glicko2] | None = None,
            check_unique: bool = True,
    ):
        """Create a tournament to determine players' ratings.

        :param players:         The players of the tournament. These must be both, identifiers for the players and
                                instances that can be passed to the `game` function to play a game between these two
                                players.
        :param game:            A callable, that takes two players, lets them play a game and resturn the result from
                                the perspective from the first of the two players.
        :param pairer:          Some callable that takes this Tournament instance as a single argument and creates a
                                list of pairs for the next round of this tournament.
        :param initial_ratings: The initial ratings of the players before the tournament begins.
        :param check_unique:    Whether to check, if the players passed are unique.
        """
        self._players = tuple(players)
        self._game = game
        self.pairer = pairer
        if initial_ratings is None:
            self._initial_ratings = tuple(Glicko2() for _ in players)
        else:
            self._initial_ratings = tuple(initial_ratings)

        if len(self._players) != len(self._initial_ratings):
            raise ValueError(f"there must be {len(players)} initial ratings - one for each player")
        if not all(isinstance(r, Glicko2) for r in self._initial_ratings):
            raise ValueError("the initial ratings were not all Glicko2 ratings")
        if sum(int(r.fixed) for r in self._initial_ratings) > 1:
            warnings.warn("There is more than one rating fixed!")

        self._ratings = self._initial_ratings
        self._round_history: list[RoundHistory] = []

        if len(self._players) != len(self._ratings):
            raise ValueError(f"Must have exactly one rating ({len([self._ratings])}) per each player ({len(players)})!")
        if check_unique:
            for i, p_i in enumerate(self._players):
                for k, p_k in enumerate(self._players[i + 1:], start=i + 1):
                    if p_i is p_k:
                        raise ValueError(f"players must be unique - found dublicate (indices {i} and {k})")

    @property
    def players(self) -> tuple[Player, ...]:
        return self._players

    @property
    def pairer(self) -> Pairer:
        return self._pairer

    @pairer.setter
    def pairer(self, pairer: Pairer):
        if not callable(pairer):
            raise TypeError("The pairer has to be a callable.")
        self._pairer = pairer

    @property
    def ratings(self) -> tuple[Glicko2, ...]:
        """The current ratings for the players of the tournament."""
        return self._ratings

    @property
    def initial_ratings(self) -> list[Glicko2]:
        """The ratings of the players prior to the tournament."""
        return [copy(r) for r in self._initial_ratings]

    def rating_changes(self) -> NDArray[np.float_]:
        """The numerical changes in rating for the players over the tournament. (Aee also `rating_history()`.)"""
        return np.array([a.r - b.r for a, b in zip(self._ratings, self._initial_ratings)])

    def round_lengths(self) -> NDArray[np.int_]:
        return np.array([len(h) for h in self._round_history])

    @property
    def rounds_played(self) -> int:
        """Number of rounds (not games!) played in this tournament."""
        return len(self._round_history)

    def games_per_player(self) -> NDArray[np.int_]:
        """The number of games played for each player."""
        cnt = np.zeros(len(self._players), dtype=int)
        for h in self._round_history:
            for (i, k), s in h:
                cnt[i] += 1
                cnt[k] += 1
        return cnt

    def pair_counts(self, symmetric: bool = True) -> NDArray[np.int_]:
        """The counts of the player pairings done in this tournament.

        :param symmetric:   If True, simply count whether a pairing happend, but ignore the pair orders. The resulting
                            matrix will then be symmetric.
                            If False, the result matrix will count the pairing of players i and k in the index (i, k),
                            iff they played in this order, and in (k, i) if their ordering was reversed in a pairing.
        :return: A matrix of the number of pairings for each player combination.
        """
        pair_cnts = np.zeros((len(self._players),) * 2, dtype=int)
        for h in self._round_history:
            for (i, k), s in h:
                pair_cnts[i, k] += 1
        if symmetric:
            pair_cnts += pair_cnts.T
        return pair_cnts

    def pair_scores(self, symmetric: bool = True, average: bool = False) -> NDArray[np.float_]:
        """Similiar to `pair_counts`, but with scores."""
        pair_s = np.zeros((len(self._players),) * 2)

        for h in self._round_history:
            for (i, k), s in h:
                pair_s[i, k] += s
                if symmetric:
                    pair_s[k, i] += 1.0 - s

        if average:
            pair_cnts = self.pair_counts(symmetric=symmetric)

            has_games = np.where(pair_cnts != 0)
            pair_s[has_games] /= pair_cnts[has_games]
            pair_s[pair_cnts == 0] = np.nan

        return pair_s

    def scores(self) -> NDArray[np.float_]:
        """The total score for each player."""
        return self.pair_scores(symmetric=True).sum(axis=1)

    def standings(self, buchholz: bool = True) -> NDArray[np.int_]:
        """The ranks of the players."""
        scores = self.scores()
        if buchholz:
            pair_cnts = self.pair_counts(symmetric=True)
            b = np.array([(cnts[cnts > 0] * scores[cnts > 0]).sum() for cnts in pair_cnts])
            scores = pair_cnts.sum() * scores + b
        _, ranks, cnts = np.unique(-scores, return_inverse=True, return_counts=True)
        return np.array([np.sum(cnts[:r]) for r in ranks]) + 1

    def all_results(self) -> list[NDArray[np.float_]]:
        """Arrays of the results each player achieved during this tournament."""
        results: list[list[float]] = [[] for _ in self._players]
        for h in self._round_history:
            for (i, k), s in h:
                results[i].append(s)
                results[k].append(1.0 - s)
        return [np.array(a) for a in results]

    def average_results(self) -> NDArray[np.float_]:
        """The average score for each player during this tournament."""
        results = self.all_results()
        return np.array([np.mean(res) if len(res) else np.nan for res in results])

    def performance_ratings(
            self,
            clip_range: tuple[float, float] = (0.0, 4000.0),
            tol: float = 1e-15,
    ) -> NDArray[np.float_]:
        """The performance ratings of the players during the tournament. (See `glicko2.performance_rating`)"""
        flat_hist = [iks for h in self._round_history for iks in h]
        opponents, results = gather_opponents_scores(self._initial_ratings, flat_hist)
        # -> tuple[list[list[Glicko2]], list[list[float]]]:
        return np.array([
            performance_rating(opp, s, clip_range=clip_range, tol=tol)
            for opp, s in zip(opponents, results)
        ])

    def stats(self) -> pd.DataFrame:
        results = self.all_results()
        data = [
            ('place', self.standings()),
            ('score', [np.sum(res) for res in results]),
            ('wins', [np.sum(res == 1) for res in results]),
            ('draws', [np.sum(res == 1/2) for res in results]),
            ('losses', [np.sum(res == 0) for res in results]),

            ('rounds played', [len(res) for res in results]),
            ('average score', [np.mean(res) if len(res) else np.nan for res in results]),

            ('perf. rating', self.performance_ratings()),
            ('final rating', [float(r) for r in self._ratings]),
            ('final rating dev.', [r.dev for r in self._ratings]),
            ('delta rating', [float(r) - float(r0) for r0, r in zip(self._initial_ratings, self._ratings)]),
            ('initial rating', [float(r) for r in self._initial_ratings]),
            ('initial rating dev.', [r.dev for r in self._initial_ratings]),
        ]
        stats = pd.DataFrame(OrderedDict(data))
        stats.index.name = 'Player'

        return stats.sort_values('place')

    def play_round(self, pairings: list[Pair] | None = None):
        """Play a given number of rounds.

        :param pairings:    The pairings to play. If this is None, generate the pairings with the tournament's pairer.
        """
        if pairings is None:
            pairings = self._pairer(self)

        history: RoundHistory = []
        for i, k in pairings:
            score = self._game(self._players[i], self._players[k])
            history.append(((i, k), score))

        self._round_history.append(history)
        self._ratings = update_ratings(self._ratings, history)

        gc.collect()

    def hold(self, rounds: int, pbar: bool = True, **tqdm_kwargs) -> 'Tournament':
        """Hold the tournament, i.e. play a given number of rounds."""
        for _ in trange(rounds, disable=not pbar, **tqdm_kwargs):
            pairings = self._pairer(self)
            self.play_round(pairings=pairings)

        return self