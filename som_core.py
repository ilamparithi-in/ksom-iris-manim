import numpy as np


class SOM:
    def __init__(self, grid_x, grid_y, input_dim, alpha_init=0.5, sigma_init=5.0, seed=42):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.input_dim = input_dim
        self.alpha_init = alpha_init
        self.sigma_init = sigma_init
        self.seed = seed
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        rng = np.random.default_rng(self.seed)
        return rng.random((self.grid_x, self.grid_y, self.input_dim))

    def find_bmu(self, x):
        diff = self.weights - x
        distances = np.linalg.norm(diff, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def compute_neighborhood(self, bmu_idx, sigma):
        bmu_r, bmu_c = bmu_idx
        r = np.arange(self.grid_x)
        c = np.arange(self.grid_y)
        cc, rr = np.meshgrid(c, r)
        d2 = (rr - bmu_r) ** 2 + (cc - bmu_c) ** 2
        return np.exp(-d2 / (2.0 * sigma ** 2))

    def update_weights(self, x, bmu_idx, alpha, sigma):
        h = self.compute_neighborhood(bmu_idx, sigma)
        self.weights += alpha * h[:, :, np.newaxis] * (x - self.weights)

    def decay_parameters(self, t, total_iterations):
        tau = total_iterations
        alpha = self.alpha_init * np.exp(-t / tau)
        sigma = max(self.sigma_init * np.exp(-t / tau), 0.5)
        return alpha, sigma

    def train(self, X, epochs):
        n_samples = X.shape[0]
        total_iterations = epochs * n_samples
        rng = np.random.default_rng(self.seed)
        t = 0
        for _ in range(epochs):
            indices = rng.permutation(n_samples)
            for idx in indices:
                x = X[idx]
                alpha, sigma = self.decay_parameters(t, total_iterations)
                bmu_idx = self.find_bmu(x)
                self.update_weights(x, bmu_idx, alpha, sigma)
                t += 1

    def get_bmu_assignments(self, X):
        assignments = []
        for x in X:
            assignments.append(self.find_bmu(x))
        return np.array(assignments)
    
    def get_clusters(self, X):
        clusters = {}
        for x in X:
            bmu = self.find_bmu(x)
            if bmu not in clusters:
                clusters[bmu] = []
            clusters[bmu].append(x)
        return clusters

    def get_labeled_clusters(self, X, y):
        clusters = {}
        for x, label in zip(X, y):
            bmu = self.find_bmu(x)
            if bmu not in clusters:
                clusters[bmu] = []
            clusters[bmu].append(label)

        # convert to dominant label
        cluster_labels = {}
        for k, labels in clusters.items():
            cluster_labels[k] = max(set(labels), key=labels.count)

        return cluster_labels