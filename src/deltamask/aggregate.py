"""Beta MAP with recovered-code epsilon-smoothed binary client observations."""

import math
import numpy as np


class BayesianMask:
    def __init__(self, dimension, participation=1.0, epsilon=.01, numerical_epsilon=1e-7):
        if not 0 < participation <= 1 or dimension < 1:
            raise ValueError("Invalid aggregation configuration")
        if not 0 <= epsilon < .5 or not 0 < numerical_epsilon < .5:
            raise ValueError("Invalid smoothing or numerical clamp")
        self.epsilon, self.numerical_epsilon = epsilon, numerical_epsilon
        self.period = max(1, math.floor(1 / participation))
        self.alpha = np.ones(dimension, dtype=np.float64)
        self.beta = np.ones(dimension, dtype=np.float64)
        self.last_round = 0

    def update(self, masks, round_id):
        masks = np.asarray(masks)
        if (
            masks.ndim != 2 or len(masks) < 1
            or masks.shape[1] != len(self.alpha)
            or not np.isin(masks, [0, 1]).all()
        ):
            raise ValueError("Expected binary masks")
        return self.update_counts(masks.sum(axis=0), len(masks), round_id)

    def update_counts(self, ones, clients, round_id):
        """Accept streamed binary sufficient statistics without retaining client masks."""
        ones = np.asarray(ones)
        if (round_id != self.last_round + 1 or type(clients) != int or clients < 1
                or ones.shape != self.alpha.shape or not np.isfinite(ones).all()
                or np.any((ones < 0) | (ones > clients))
                or np.any(ones != np.floor(ones))):
            raise ValueError("Expected next round and valid binary counts")
        if round_id % self.period == 0:
            self.alpha.fill(1)
            self.beta.fill(1)
        observations = self.epsilon * clients + (1 - 2 * self.epsilon) * ones
        self.alpha += observations
        self.beta += clients - observations
        self.last_round = round_id
        return np.clip((self.alpha - 1) / (self.alpha + self.beta - 2),
                       self.numerical_epsilon, 1 - self.numerical_epsilon)
