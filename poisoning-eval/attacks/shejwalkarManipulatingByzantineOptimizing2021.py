# Source https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning
# FIXME: they don't list a license -> can't really republish right?

import numpy as np

from .attack_class import Attack


# direction the attackers push against, from their own honest updates (dev_type in the original code)
def deviation(malicious_updates: np.ndarray, dev_type: str) -> np.ndarray:
    model_re = malicious_updates.mean(0)
    if dev_type == 'unit_vec':
        return model_re / np.linalg.norm(model_re)  # unit vector, dir opp to good dir
    if dev_type == 'sign':
        return np.sign(model_re)
    if dev_type == 'std':
        # the original works on gradients, these are model deltas (-lr * gradient). unit_vec / sign flip with them,
        # std doesn't, so it is negated to push the same way as the original's mean_grad - lamda * std
        return -malicious_updates.std(0)
    raise ValueError(f'unknown dev_type {dev_type!r}')

# np.linalg.norm(rows - point, axis=1) ** 2 one row at a time: the same numbers without the (n, d) temporary
def squared_distances(rows, point):
    distances = np.zeros(len(rows), np.result_type(rows, point))
    for i in range(len(rows)):
        distances[i] = np.linalg.norm((rows[i] - point)[None], axis=1)[0]
    return distances ** 2


# all pairwise squared_distances of rows, each pair computed once (rows[j] - rows[i] is exactly -(rows[i] - rows[j]))
def pairwise_squared_distances(rows):
    distances = np.zeros((len(rows), len(rows)), rows.dtype)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            distances[i, j] = np.linalg.norm((rows[j] - rows[i])[None], axis=1)[0]
            distances[j, i] = distances[i, j]
    return distances ** 2


# Min-Max (Shejwalkar & Houmansadr, NDSS'21), ported from torch to numpy.
# The attackers only use their own honest updates (partial knowledge): all of them send
# model_re - lamda * direction, with the largest lamda whose max squared distance to their updates
# stays within the max pairwise squared distance among them
class MinMaxAttack(Attack):
    name = 'MinMax'

    def __init__(self, n_attackers, threshold=50.0, dev_type='unit_vec'):
        self.n_attackers = n_attackers
        self.threshold = threshold
        self.dev_type = dev_type

    def perform_attack(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        malicious = np.arange(self.n_attackers)
        malicious_updates = updates[malicious]
        model_re = malicious_updates.mean(0)

        direction = deviation(malicious_updates, self.dev_type)

        lamda = float(self.threshold)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0.0

        max_distance = pairwise_squared_distances(malicious_updates).max()

        while abs(lamda_succ - lamda) > threshold_diff:
            mal_update = model_re - lamda * direction
            max_d = squared_distances(malicious_updates, mal_update).max()

            if max_d <= max_distance:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = model_re - lamda_succ * direction
        # logged per round by experiment.py
        self.last_info = {'lamda': lamda_succ}

        # attackers replace their own rows (the first n_attackers)
        updates = updates.copy()
        updates[malicious] = mal_update
        return updates, malicious


# Min-Sum (Shejwalkar & Houmansadr, NDSS'21), ported from torch to numpy.
# Same as Min-Max, but bounds the sum of squared distances instead of the max: the largest lamda
# whose summed squared distance to the attackers' updates stays within the smallest such sum of any of them
class MinSumAttack(Attack):
    name = 'MinSum'

    def __init__(self, n_attackers, threshold=30.0, dev_type='unit_vec'):
        self.n_attackers = n_attackers
        self.threshold = threshold
        self.dev_type = dev_type

    def perform_attack(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        malicious = np.arange(self.n_attackers)
        malicious_updates = updates[malicious]
        model_re = malicious_updates.mean(0)

        direction = deviation(malicious_updates, self.dev_type)

        lamda = float(self.threshold)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0.0

        distances = pairwise_squared_distances(malicious_updates)
        min_score = min(distances[i].sum() for i in range(len(distances)))

        while abs(lamda_succ - lamda) > threshold_diff:
            mal_update = model_re - lamda * direction
            score = squared_distances(malicious_updates, mal_update).sum()

            if score <= min_score:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = model_re - lamda_succ * direction
        # logged per round by experiment.py
        self.last_info = {'lamda': lamda_succ}

        # attackers replace their own rows (the first n_attackers)
        updates = updates.copy()
        updates[malicious] = mal_update
        return updates, malicious
