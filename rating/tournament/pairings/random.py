import numpy as np

from ..tournament import Tournament, Pair


class RandomPairer:
    """A very simple random pairer. Simply choose pairs at random.

    Note: in case of an odd number `n` of players, the average number of games played per player in a tournament will
    only scatter around (n // 2) / n, this class makes no attempt in holding this scatter small. There is no guarantee
    that each player will have played at least a single game in a tournament, regardless of the number of rounds played.
    """

    def __call__(self, tournament: Tournament) -> list[Pair]:
        n = len(tournament.players)
        indices = np.arange(n)
        np.random.shuffle(indices)
        pairs = indices[: 2 * (n // 2)].reshape((n // 2, 2))
        return [(int(i), int(k)) for i, k in pairs]
