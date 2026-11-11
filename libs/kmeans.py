"""K-Means de Lloyd com inicialização k-means++ e várias execuções."""

import numpy as np


def squared_distances(X, centers):
    # Evita criar um tensor de dimensão n × k × p.
    return np.column_stack([np.sum((X - center) ** 2, axis=1) for center in centers])


class KMeans:
    def __init__(self, n_clusters=3, n_init=5, max_iter=100, tol=1e-4, seed=42):
        for name, value in (("n_clusters", n_clusters), ("n_init", n_init), ("max_iter", max_iter)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} deve ser inteiro positivo.")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol deve ser finita e não negativa.")
        self.n_clusters, self.n_init = n_clusters, n_init
        self.max_iter, self.tol, self.seed = max_iter, tol, seed

    def _initial_centers(self, X, rng):
        centers = [X[rng.integers(len(X))].copy()]
        closest = squared_distances(X, centers)[:, 0]
        for _ in range(1, self.n_clusters):
            total = closest.sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError("Escala incompatível com k-means++; normaliza os dados.")
            index = rng.choice(len(X), p=closest / total)
            centers.append(X[index].copy())
            closest = np.minimum(closest, squared_distances(X, [centers[-1]])[:, 0])
        return np.asarray(centers)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X deve ser uma matriz não vazia e finita.")
        if len(np.unique(X, axis=0)) < self.n_clusters:
            raise ValueError("K não pode exceder o número de pontos distintos.")
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            centers = self._initial_centers(X, rng)
            converged = False
            for iteration in range(1, self.max_iter + 1):
                distances = squared_distances(X, centers)
                labels = distances.argmin(axis=1)
                updated = centers.copy()
                empty = []
                for cluster in range(self.n_clusters):
                    members = X[labels == cluster]
                    if len(members):
                        updated[cluster] = members.mean(axis=0)
                    else:
                        empty.append(cluster)
                # Reinicializa grupos vazios num ponto mal representado, sem repetir escolhas.
                if empty:
                    occupied = [c for c in range(self.n_clusters) if c not in empty]
                    closest = squared_distances(X, updated[occupied]).min(axis=1)
                for cluster in empty:
                    index = int(closest.argmax())
                    updated[cluster] = X[index]
                    closest = np.minimum(closest, squared_distances(X, [X[index]])[:, 0])
                shift = float(np.max(np.linalg.norm(updated - centers, axis=1)))
                centers = updated
                if not empty and shift <= self.tol:
                    converged = True
                    break
            distances = squared_distances(X, centers)
            labels = distances.argmin(axis=1)
            if len(np.unique(labels)) != self.n_clusters:
                continue
            inertia = float(distances[np.arange(len(X)), labels].sum())
            if best is None or inertia < best[0]:
                best = inertia, centers.copy(), labels.copy(), iteration, converged
        if best is None:
            raise RuntimeError("Não se obtiveram K grupos não vazios; rever K e preparação.")
        self.inertia_, self.centers_, self.labels_, self.n_iter_, self.converged_ = best
        return self

    def predict(self, X):
        if not hasattr(self, "centers_"):
            raise ValueError("Executa fit antes de predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.centers_.shape[1] or not np.isfinite(X).all():
            raise ValueError("X tem esquema incompatível ou valores não finitos.")
        return squared_distances(X, self.centers_).argmin(axis=1)
