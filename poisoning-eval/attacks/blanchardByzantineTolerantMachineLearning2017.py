# Gaussian noise (Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS'17)
# ByzFL's Gaussian (https://github.com/LPD-EPFL/byzfl, byzfl/attacks/attacks.py):
# random.normal(loc=mu, scale=sigma, size=d), drawn from random_generator instead of np.random so runs are seeded.
# sigma = 200 from the paper (Sec. 5: "mean zero, and isotropic covariance matrix with standard deviation 200"),
# ByzFL's default is 1.0. Each attacker draws its own vector (paper: "Byzantine processes propose vectors drawn
# from a Gaussian distribution"); ByzFL would send one draw f times. Needs no knowledge of honest updates

import numpy as np

from .attack_class import Attack


class GaussianAttack(Attack):
    name = 'Gaussian'

    def __init__(self, n_attackers, random_generator, mu=0.0, sigma=200.0):
        self.n_attackers = n_attackers
        self.random_generator = random_generator
        self.mu = mu
        self.sigma = sigma

    def perform_attack(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        malicious = np.arange(self.n_attackers)
        d = updates.shape[1]

        # attackers replace their own rows (the first n_attackers)
        updates = updates.copy()
        for i in malicious:
            updates[i] = self.random_generator.normal(loc=self.mu, scale=self.sigma, size=d)
        return updates, malicious
