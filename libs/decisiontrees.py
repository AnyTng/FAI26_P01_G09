"""Árvore de classificação multiclasse de raiz, com divisões pelo índice de Gini."""

import numpy as np


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=20, min_samples_leaf=10):
        for name, value, minimum in (
            ("max_depth", max_depth, 0),
            ("min_samples_split", min_samples_split, 2),
            ("min_samples_leaf", min_samples_leaf, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} deve ser inteiro >= {minimum}.")
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X deve ser uma matriz não vazia e finita.")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y deve ser um vetor alinhado com X.")
        self.classes_, codes = np.unique(y, return_inverse=True)
        self.n_features_ = X.shape[1]
        self.root_ = self._grow(X, codes, 0)
        return self

    def _grow(self, X, y, depth):
        n = len(y)
        counts = np.bincount(y, minlength=len(self.classes_))
        probabilities = counts / n
        node = {"probabilities": probabilities, "samples": n}
        if (
            depth >= self.max_depth or n < self.min_samples_split
            or n < 2 * self.min_samples_leaf or np.count_nonzero(counts) == 1
        ):
            return node
        best_impurity = 1 - float(np.sum(probabilities ** 2))
        best = None
        for feature in range(self.n_features_):
            order = np.argsort(X[:, feature], kind="stable")
            values, target = X[order, feature], y[order]
            left_n = np.arange(1, n)
            right_n = n - left_n
            # Contagens cumulativas avaliam todas as divisões sem reconstruir subconjuntos.
            left_counts = np.cumsum(np.eye(len(self.classes_))[target], axis=0)[:-1]
            right_counts = counts - left_counts
            left_gini = 1 - np.sum((left_counts / left_n[:, None]) ** 2, axis=1)
            right_gini = 1 - np.sum((right_counts / right_n[:, None]) ** 2, axis=1)
            impurity = (left_n * left_gini + right_n * right_gini) / n
            valid = (
                (values[:-1] < values[1:])
                & (left_n >= self.min_samples_leaf) & (right_n >= self.min_samples_leaf)
            )
            impurity[~valid] = np.inf
            position = int(np.argmin(impurity))
            if impurity[position] < best_impurity - 1e-12:
                best_impurity = float(impurity[position])
                # O maior valor da esquerda evita arredondamento de pontos médios.
                best = feature, float(values[position])
        if best is None:
            return node
        feature, threshold = best
        left = X[:, feature] <= threshold
        node.update(
            feature=feature, threshold=threshold,
            left=self._grow(X[left], y[left], depth + 1),
            right=self._grow(X[~left], y[~left], depth + 1),
        )
        return node

    def predict_proba(self, X):
        """Uma coluna por classe, na ordem de classes_."""
        if not hasattr(self, "root_"):
            raise ValueError("Executa fit antes de prever.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_ or not np.isfinite(X).all():
            raise ValueError("X tem esquema incompatível ou valores não finitos.")
        output = np.empty((len(X), len(self.classes_)))
        for i, row in enumerate(X):
            node = self.root_
            while "feature" in node:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            output[i] = node["probabilities"]
        return output

    def predict(self, X):
        # Em empate, a ordem estável das classes torna a previsão reproduzível.
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
