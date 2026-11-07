# Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning" (USENIX Security'20)
# partial knowledge only
# FLDetector's dir_partial_krum_lambda (https://github.com/zaixizhang/FLDetector) and
# FLPoison's FangAttack (https://github.com/vio1etus/FLPoison).
# Updates are deltas to the received global model, so the paper's w_Re is the origin here.

import numpy as np

from .attack_class import Attack


# estimated changing direction s~ (Sec. 3.2): +1 where the attackers' mean increases the parameter, else -1
def estimated_direction(own_updates):
    return np.where(own_updates.mean(0) > 0, 1.0, -1.0)


# Krum as described in the paper (Sec. 2): score = sum of Euclidean distances to the n - f - 2 nearest
# other updates; returns the selected index. Own version (byzfl squares distances, uses n - f - 1 and
# returns the vector) so that the Theorem-1-style bound below holds for the simulated Krum
def krum_index(updates, f):
    distances = np.stack([np.linalg.norm(updates - update, axis=1) for update in updates])
    return int(np.argmin(np.sort(distances, axis=1)[:, 1:len(updates) - f - 1].sum(1)))


# np.stack([np.linalg.norm(rows - row, axis=1) for row in rows]) one pair at a time: the same numbers without the
# (n, d) temporaries, and each pair computed once (rows[j] - rows[i] is exactly -(rows[i] - rows[j]))
def pairwise_distances(rows):
    distances = np.zeros((len(rows), len(rows)), rows.dtype)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            distances[i, j] = np.linalg.norm((rows[j] - rows[i])[None], axis=1)[0]
            distances[j, i] = distances[i, j]
    return distances


# Theorem 1 (Eq. 3) adapted to Eq. 4: the c own updates act as benign, num_crafted copies of w1 as compromised
# (Krum f = num_crafted). Finite only for num_crafted <= c - 2; at c - 1 any lamda is selected, so we clip the
# denominator to 1 (the paper leaves this case open)
# own_distances: (c, c) pairwise distances of the own updates, max_norm: their largest norm, d: dimension
def krum_lambda_upper_bound(own_distances, max_norm, num_crafted, d):
    num_own = len(own_distances)
    min_score = np.sort(own_distances, axis=1)[:, 1:num_own - 1].sum(1).min()
    return (min_score / max(num_own - num_crafted - 1, 1) + max_norm) / np.sqrt(d)


# krum_index on [num_crafted copies of w1, the own updates] from precomputed distances, without building the
# (num_crafted + c, d) matrix: copies are 0 apart, own_distances (c, c) and crafted_distances (c,) are given
def krum_index_crafted(own_distances, crafted_distances, num_crafted):
    n = num_crafted + len(own_distances)
    distances = np.zeros((n, n))
    distances[:num_crafted, num_crafted:] = crafted_distances
    distances[num_crafted:, :num_crafted] = crafted_distances[:, None]
    distances[num_crafted:, num_crafted:] = own_distances
    return int(np.argmin(np.sort(distances, axis=1)[:, 1:n - num_crafted - 1].sum(1)))


# Attack on Krum (Sec. 3.2, Eq. 4): w1 = -lamda * s~, lamda halved from the upper bound until Krum over
# [num_crafted copies of w1, the attackers' own honest updates] selects w1; if it never does, add one more
# copy and retry. The other attackers send random vectors within epsilon of w1 (Sec. 4.1: epsilon = 0.01)
class FangKrumAttack(Attack):
    name = 'FangKrum'

    def __init__(self, n_attackers, random_generator, epsilon=0.01, threshold=1e-5):
        assert n_attackers > 1, 'FangKrum needs at least two attackers'
        self.n_attackers = n_attackers
        self.random_generator = random_generator
        self.epsilon = epsilon
        self.threshold = threshold

    def perform_attack(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        malicious = np.arange(self.n_attackers)
        own_updates = updates[malicious]
        d = own_updates.shape[1]
        direction = estimated_direction(own_updates)

        # everything the search needs, computed once: the own updates' pairwise distances, and for
        # ||-lamda s - u||^2 = lamda^2 d + 2 lamda s.u + ||u||^2 the dot products s.u and squared norms
        own = own_updates.astype(np.float64)
        own_distances = pairwise_distances(own)
        dots = own @ direction
        squared_norms = (own ** 2).sum(1)

        for num_crafted in range(1, self.n_attackers):
            lamda = krum_lambda_upper_bound(own_distances, np.sqrt(squared_norms.max()), num_crafted, d)
            while lamda >= self.threshold:
                crafted_distances = np.sqrt(np.maximum(lamda ** 2 * d + 2 * lamda * dots + squared_norms, 0))
                if krum_index_crafted(own_distances, crafted_distances, num_crafted) < num_crafted:
                    break
                lamda /= 2
            if lamda >= self.threshold:
                break

        crafted = -lamda * direction
        # supporters uniform in the epsilon ball around w1
        noise = self.random_generator.standard_normal((self.n_attackers - 1, d))
        radii = self.epsilon * self.random_generator.random((self.n_attackers - 1, 1)) ** (1 / d)
        # logged per round by experiment.py
        self.last_info = {'lamda': float(lamda), 'num_crafted': num_crafted}

        # attackers replace their own rows (the first n_attackers)
        updates = updates.copy()
        updates[malicious] = np.vstack([crafted, crafted + noise / np.linalg.norm(noise, axis=1)[:, None] * radii])
        return updates, malicious
